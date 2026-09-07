"""The schedule walk's enumeration contracts.

One recursive traversal enumerates the compatible subset of the classic schedule product. The
prescan asks each domain catalog once, and sampling is a reservoir over the leaf stream
(``search/pool.py``). These tests pin the traversal contract: every node and operand-use edge has a
stable schedule-site identity, the flash pair can select its paired mma choices, and structural
split choices remain separate from classic schedule choices.
"""

from __future__ import annotations

import importlib
from dataclasses import replace as dc_replace
from types import SimpleNamespace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.frontend.ir import MatmulOp, SdpaOp
from emmy.compiler.ir.pure import Fold, Lambda
from emmy.compiler.ir.schedule import Placement
from emmy.compiler.ir.schedule import classic_projection as _classic
from emmy.compiler.ir.schedule.catalog import coop_reduce_moves
from emmy.compiler.ir.schedule.views import ContractionFacts
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import OutputSpec, Reduce, TileOp
from emmy.compiler.ir.tile.ops import Sched
from emmy.compiler.pipeline.fork import iter_leaves
from emmy.compiler.pipeline.knob import family_of
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop, scan_from_loop
from emmy.compiler.pipeline.search.golden_eval import enumerate_graph
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.pipeline.search.pool import PoolSample
from tests.compiler.terms import contraction, projection, reduction, slab

_CC = (12, 0)

#: The scheduling rule, reached through ``importlib`` because its module name starts with a digit.
_SCHEDULE_RULE = importlib.import_module("emmy.compiler.pipeline.passes.lowering.tile.040_schedule")

#: The knob pins the enumeration reads off the environment. A host with one set would enumerate a
#: narrowed pool and fail the offer assertions here for a reason that has nothing to do with the
#: traversal.
_PIN_VARS = ("EMMY_KNOBS", "EMMY_PLACE", "EMMY_TILE", "EMMY_WORK", "EMMY_STAGE", "EMMY_REDUCE", "EMMY_RASTER")


def _matmul_graph(m: int, n: int, k: int, dtype: str) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Dim(m), Dim(k)), dtype=dtype), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(k), Dim(n)), dtype=dtype), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (Dim(m), Dim(n)), dtype=dtype), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _sdpa_graph(b: int = 1, h: int = 2, s: int = 64, d: int = 32) -> Graph:
    """The fused streaming cell: one primary and one derived site over its computed score edge."""
    g = Graph()
    for name in ("q", "k", "v"):
        g.add_node(InputOp(), [], Tensor(name, (Dim(b), Dim(h), Dim(s), Dim(d)), dtype="f16"), node_id=name)
    g.add_node(SdpaOp(is_causal=False), ["q", "k", "v"], Tensor("o", (Dim(b), Dim(h), Dim(s), Dim(d)), dtype="f16"), node_id="o")
    g.inputs, g.outputs = ["q", "k", "v"], ["o"]
    return g


def _code_graph(code: str) -> Graph:
    from emmy.commands.trace import graph_from_code  # noqa: PLC0415

    return graph_from_code(code)[0]


_NORM = "(lambda t: t*torch.rsqrt((t.float()*t.float()).mean(-1,keepdim=True)+1e-6).to(t.dtype))"

FIXTURES = {
    "scalar_matmul": lambda: _matmul_graph(128, 128, 128, "f32"),
    "warp_matmul": lambda: _matmul_graph(128, 128, 128, "f16"),
    "reduce_matvec": lambda: _code_graph("torch.nn.functional.linear(torch.randn(1, 4096), torch.randn(512, 4096))"),
    "fused_norm_linear": lambda: _code_graph(
        f"torch.nn.functional.linear({_NORM}(torch.randn(128, 256, dtype=torch.float16)), torch.randn(256, 256, dtype=torch.float16))"
    ),
    "flash_pair": _sdpa_graph,
}


def _rows(graph) -> list[dict]:
    """The sampled candidate rows of every schedule fork ``graph`` opens.

    The sample makes the walk exhaust its leaf stream once inside ``schedule`` (the reservoir
    retains nothing proportional to the pool), so these tests observe a complete traversal without
    flattening a live space into test memory."""
    ctx = dc_replace(Context.from_target(_CC), pool_sample=PoolSample(rows=8, seed=0))
    return enumerate_graph(graph, ctx).rows


def _pin(monkeypatch, **pins: str) -> None:
    """Pin canonical classic schedule keys without a family-wide fallback."""
    for key, value in pins.items():
        monkeypatch.setenv(f"EMMY_{key.upper()}", value)


def _pin_sdpa(monkeypatch) -> None:
    _pin(
        monkeypatch,
        **{
            "TILE@map.1/twist.1/inner": "mma_m16n8k16_f16_f32/f1x2",
            "TILE@map.1/twist": "mma_m16n8k16_f16_f32/f1x1",
            "REDUCE": "",
            "STAGE@map.1/twist.1/inner": "",
            "STAGE@map.1/twist": "d1/smem",
        },
    )


@pytest.fixture
def unpinned(monkeypatch):
    for var in _PIN_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("case", sorted(FIXTURES))
def test_the_prescan_asks_each_catalog_question_once(case, unpinned, monkeypatch) -> None:
    """Every catalog question is asked ONCE per term — the invariant the walk rests on: options are
    a function of the node and the live pins, the prescan fills ``_State.options`` once, and every
    branch expansion only READS the memo. Asking a site what it offers per branch instead
    re-resolves every stage once per member of a list those same catalogs produce — seconds per
    lowered kernel while the fork above reads a single row. The sampled enumeration exhausts every
    leaf, so a reintroduced per-branch re-ask shows up here as a repeated question, not as a slow
    test somebody eventually notices."""
    asked: list[tuple] = []
    original = _classic._options

    def spy(state, node):
        asked.append((state.tile, node))  # strong refs, so ids below cannot alias freed objects
        return original(state, node)

    monkeypatch.setattr(_classic, "_options", spy)
    assert _rows(FIXTURES[case]())
    assert asked, "the fixture built no catalog at all"
    keys = [(id(tile), id(node)) for tile, node in asked]
    repeats = len(keys) - len(set(keys))
    assert not repeats, f"_options was asked the same question {repeats} time(s) over ({len(keys)} calls)"


def test_the_prescan_reads_each_computed_a_seam_once(unpinned, monkeypatch) -> None:
    """A computed-A cone is lowered once for its stat-row seam, not once per tile plan."""
    from emmy.compiler.ir.schedule import staging, views  # noqa: PLC0415

    calls: list[tuple] = []
    original = views.cone_seam

    def spy(cone, k_name, axes=()):
        calls.append((cone, k_name))
        return original(cone, k_name, axes)

    monkeypatch.setattr(views, "cone_seam", spy)
    monkeypatch.setattr(
        staging,
        "cone_seam",
        lambda *_: (_ for _ in ()).throw(AssertionError("the fill must reuse the prescan's seam")),
    )
    rows = _rows(FIXTURES["fused_norm_linear"]())
    assert rows
    assert calls
    # ONE distinct seam, read once per TileOp that composes over the term -- the unscheduled tile
    # plus each materialized candidate re-validating itself -- and never once per candidate PLAN,
    # which is the cost this guards. The facts are cached on the kernel, so the bound is the number
    # of kernels, not the size of the search.
    assert len({(id(cone), k_name) for cone, k_name in calls}) == 1
    assert len(calls) < len(rows)


@pytest.mark.parametrize(
    "case, tile_sites, reduce_sites",
    (
        ("fused_norm_linear", 1, 2),
        pytest.param(
            "flash_pair",
            2,
            3,
            marks=pytest.mark.xfail(strict=True, reason="fused value channel on tensor cores: not on this tree yet (PR #699)"),
        ),
    ),
)
def test_computed_fold_sites_are_keyed_schedule_sites(case, tile_sites, reduce_sites, unpinned) -> None:
    """A computed cone's fold and a derived site (flash's synthesized PV) are real schedule sites:
    every row spells them with stable node identities, so nothing nested is silently undecided."""
    row = _rows(FIXTURES[case]())[0]
    assert sum(key == "TILE" or key.startswith("TILE@") for key in row) == tile_sites
    assert sum(key == "REDUCE" or key.startswith("REDUCE@") for key in row) == reduce_sites


def test_sdpa_fold_tree_offers_a_paired_mma_row(unpinned, monkeypatch) -> None:
    """The walk reaches a row where BOTH flash contractions ride the tensor core — the score's N
    tile feeding the value contraction's streamed K block through the fragment seam."""
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    _pin_sdpa(monkeypatch)
    monkeypatch.setenv("EMMY_RASTER", "")
    rows = _rows(_sdpa_graph())
    assert any(sum(key.startswith("TILE@") and "mma_" in str(value) for key, value in row.items()) == 2 for row in rows)


def test_global_classic_pins_restrict_every_applicable_site(unpinned, monkeypatch) -> None:
    """Bare families form one immutable restriction without manufacturing domain choices."""
    _pin(monkeypatch, WORK="", TILE="", REDUCE="", STAGE="", RASTER="")
    rows = _rows(_matmul_graph(64, 64, 64, "f16"))
    assert rows
    for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER"):
        assert all(all(value == "" for key, value in row.items() if family_of(key) == family) for row in rows)


def test_the_split_fork_offers_atomic_and_deferred_arms(unpinned) -> None:
    """The old product kept the cross-CTA partitions as combined rows; they are now STRUCTURAL
    siblings of the unsplit tree (``030_cut``), and a fold that admits both finalizes
    still sees the atomic and the deferred arm offered together, beside the unsplit one."""
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import iter_leaves  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415

    offered: set[str] = set()

    def decide(fp):
        if any(getattr(option, "structural", False) for option in fp.options):
            offered.update(str(v) for option in fp.options for k, v in option.knobs.items() if family_of(k) == "REDUCE")
        return next(iter_leaves(fp.options))

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target(_CC)).resolve(_matmul_graph(64, 64, 64, "f32"), decide)
    assert {"", "g2a", "g2k"} <= offered


@pytest.mark.xfail(strict=True, reason="fused value channel on tensor cores: not on this tree yet (PR #699)")
def test_the_twisted_carrier_split_offers_only_the_deferred_arm(unpinned, monkeypatch) -> None:
    """The cross-CTA split composes with the paired-mma flash cell. The offer is inspected directly
    to isolate its algebraic legality from the rest of resolution. The atomic arm is refused on the
    carrier's ARITY (``atomicAdd`` folds one additive state; the twisted
    carrier streams three), while the deferred workspace arm slices the multi-component carrier.
    And the pieces re-schedule their paired sites: under the pinned deferred split the partial's
    row still spells BOTH mma contractions."""
    from types import SimpleNamespace

    from emmy.compiler.ir.tile.ir import TileOp
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import iter_leaves  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._split import split_forks
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415

    ctx = Context.from_target(_CC)
    captured: list[TileOp] = []

    class _Captured(Exception):
        pass

    def keep(fp):
        op = fp.root_op
        if isinstance(op, TileOp) and op.op is not None and not op.place.is_mapped and not captured:
            captured.append(op)
            raise _Captured
        return next(iter_leaves(fp.options))

    with pytest.raises(_Captured):
        Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(_sdpa_graph(), keep)
    assert captured, "the flash cell must reach the tile passes as one fused kernel"
    offers = split_forks(None, SimpleNamespace(op=captured[0]))
    assert offers is not None
    spellings = {str(v) for offer in offers for v in offer.knobs.values()}
    assert "g2k" in spellings, "the deferred workspace arm must slice the twisted carrier"
    assert not any(s.startswith("g") and s.endswith("a") for s in spellings), (
        "atomicAdd folds ONE additive state; the three-component twisted carrier has no atomic arm"
    )

    monkeypatch.setenv("EMMY_WORK", "w1x1")
    _pin(
        monkeypatch,
        **{
            "TILE@map.1/twist.1/inner": "mma_m16n8k16_f16_f32/f1x2",
            "TILE@map.1/twist": "mma_m16n8k16_f16_f32/f1x1",
            "REDUCE@map": "",
            "REDUCE@map.1/twist.1/inner": "",
            "STAGE@map.1/twist.1/inner": "",
            "STAGE@map.1/twist": "d1/smem",
        },
    )
    monkeypatch.setenv("EMMY_RASTER", "")
    monkeypatch.setenv("EMMY_REDUCE@MAP.1/TWIST", "g2k")
    union_ctx = dc_replace(Context.from_target(_CC), validate_pins=False)
    out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=union_ctx).resolve(_sdpa_graph(), lambda fp: next(iter_leaves(fp.options)))
    partial = next(n.op for nid, n in out.nodes.items() if nid.endswith("__partial") and isinstance(n.op, TileOp))
    assert partial.place.is_mapped, "the partial piece must decide its own row"
    assert sum(key.startswith("TILE@") and "mma_" in str(value) for key, value in partial.knobs.items()) == 2, (
        "the sliced piece must still spell both paired mma sites"
    )


def test_an_observed_fold_offers_only_the_serial_reduce(unpinned) -> None:
    """A scan (an observed fold) preserves its stream order: no cooperative/ILP REDUCE band and
    no cross-CTA split row is ever offered — the serial fold is the whole catalog."""
    rows = _rows(_code_graph("torch.cumsum(torch.randn(8, 32), -1)"))
    assert rows, "the scan kernel must still enumerate (the serial tier realizes it)"
    for row in rows:
        offending = {k: v for k, v in row.items() if k.split("@", 1)[0] == "REDUCE" and v not in ("", None)}
        assert not offending, f"a partitioned REDUCE row reached an observed fold: {offending}"


def test_every_computed_statistic_receives_a_node_id(unpinned, monkeypatch) -> None:
    """A score contraction's computed B operand cone closes over its per-key statistic
    (``normalize_fold_tree``'s reduce-body closing), and the walk keys that relocated fold as an
    ordinary schedule site. Its identity is independent of its changing structural path."""
    _pin(
        monkeypatch,
        **{
            "TILE": "",
            "REDUCE": "",
            "STAGE": "",
        },
    )
    monkeypatch.setenv("EMMY_RASTER", "")
    graph = _code_graph(
        "torch.nn.functional.scaled_dot_product_attention("
        f"{_NORM}(torch.randn(1, 2, 8, 16, dtype=torch.float16)), "
        f"{_NORM}(torch.randn(1, 2, 8, 16, dtype=torch.float16)), "
        "torch.randn(1, 2, 8, 16, dtype=torch.float16))"
    )
    rows = _rows(graph)
    assert rows, "the fused attention kernel must still enumerate"
    reduce_keys = {key for row in rows for key in row if key.startswith("REDUCE@")}
    # Four reduce sites, each keyed by its route: the twisted carrier, the score contraction under
    # its weight cone, and the two norm statistics under the score's Q and K cones. ONE score node —
    # the cone the carrier's product multiplies by carries it, so no second binder reaches it.
    assert reduce_keys == {
        "REDUCE@map.1/twist",
        "REDUCE@map.1/twist.1/map.1/inner",
        "REDUCE@map.1/twist.1/map.1/inner.1/map.2/map.1/reduce",
        "REDUCE@map.1/twist.1/map.1/inner.2/map.2/map.1/reduce",
    }


# --- a root's member reduce domain -------------------------------------------------------------- #
#
# A root's members are its operand edges; a member that reads a provider chain closes over it as an
# operand of its own. These project the domain directly — the projection is a pure function of the
# tree and the output specs, so a hand-built root states the contract without a graph to route it
# through.

_K = Axis("k", 128)


def _provider() -> Fold:
    """The provider chain a member closes over: a workspace row read and its rsqrt, ``v25``."""
    return projection((), (Load(name="ws", input="cutbuf", index=(Var("m"),)), Assign(name="v25", op="rsqrt", args=("ws",))))


def _chain_member(acc: str, axis: str, src: str, factor: Fold):
    """One reduce fold over ``factor``, the provider it closes over."""
    scaled = Assign(name=f"{acc}__v", op="multiply", args=(f"{src}_e", factor.exposes[0]))
    return reduction(axis, (slab(f"{src}_e", src, "m", axis), factor), (scaled,), (acc,))


def _chain_root(*members, results=("acc",)):
    return projection(members, results=results)


def _tile_stub(root, output_specs=()):
    return SimpleNamespace(output_specs=output_specs, op=root, place=SimpleNamespace(free=()))


def _member_catalog() -> tuple:
    return (Reduce(), *(choice for choice in coop_reduce_moves() if not choice.coop_transposed))


def test_a_direct_chain_member_offers_the_non_transposed_catalog(unpinned) -> None:
    """A DIRECT body member of a chain-form root binds through the factorizer's chain arm — its
    sibling providers emit ahead of one shared strided loop — so it offers the whole cooperative /
    ILP catalog, priced at the offer rather than dropped at the binder."""
    red = _chain_member("acc", "k", "x", _provider())
    assert _classic._reduction_domain(_tile_stub(_chain_root(red)), red) == _member_catalog()


def test_a_transposed_band_is_not_in_a_direct_chain_members_domain(unpinned) -> None:
    """The ``coop-t`` band's σ-substitution and guarded close assume the fold is the kernel ROOT,
    so no chain member may carry one — offering it would mint one kernel from two knob spellings."""
    red = _chain_member("acc", "k", "x", _provider())
    domain = _classic._reduction_domain(_tile_stub(_chain_root(red)), red)
    assert domain, "the member still offers the serial fold and the plain bands"
    assert not any(choice.coop_transposed for choice in domain)


def test_a_fold_nested_under_a_chain_member_offers_only_the_serial_reduce(unpinned) -> None:
    """A fold nested UNDER a direct member — not itself a member of the root's own body — binds
    through that member's schedule-blind body recursion, so no partition can ride under it."""
    inner_body = Body(
        (
            Load(name="x_e", input="x", index=(Var("m"), Var("k"))),
            Accum(name="acc_inner", value="x_e", op="add", axes=("k",)),
        )
    )
    inner_loop = Loop(axis=Axis("k", 128), body=inner_body)
    outer_body = Body((inner_loop, Accum(name="acc_outer", value="acc_inner", op="add", axes=("m",))))
    outer = fold_from_loop(Loop(axis=Axis("m", 4), body=outer_body))
    inner = next(edge for edge in outer.operands if edge.axis is not None)

    root = _chain_root(outer, results=("acc_outer",))
    assert _classic._reduction_domain(_tile_stub(root), outer) == _member_catalog()
    assert _classic._reduction_domain(_tile_stub(root), inner) == (Reduce(),)


def test_a_sweep_carrying_store_keeps_a_member_serial_only_when_the_member_reads_it(unpinned) -> None:
    """A boundary store carrying an output sweep the member is evaluated over keeps it serial: the
    sweep loop must wrap the reduce loop, which only the serial fold spells. A sweep the member
    never reads wraps the projection tail alone, after the member's lane-distributed close, so the
    catalog stands — the binder realizes exactly that (``_factorize``'s sweep gate reads the root's
    free axes)."""
    red = _chain_member("acc", "k", "x", _provider())
    spec = OutputSpec(write=Write(output="o", index=(Var("m"), Var("j")), value="v"), sweep=(Axis("j", 4),))
    assert _classic._reduction_domain(_tile_stub(_chain_root(red), (spec,)), red) == _member_catalog()
    over = OutputSpec(write=Write(output="o", index=(Var("m"),), value="v"), sweep=(Axis("m", 4),))
    assert _classic._reduction_domain(_tile_stub(_chain_root(red), (over,)), red) == (Reduce(),)


def test_a_streamed_store_keeps_chain_members_serial(unpinned) -> None:
    """A boundary store that streams into a SIBLING observed member's reduce loop keeps every
    OTHER direct member serial too: the trailing splice cannot reach a loop that already sits in an
    earlier segment. Also kernel-level, and the exact gate the factorizer's chain arm applies."""
    scan_body = Body(
        (
            Load(name="y_e", input="y", index=(Var("m"), Var("j"))),
            Accum(name="scan_acc", value="y_e", op="add", axes=("j",)),
            Write(output="running", index=(Var("m"), Var("j")), value="scan_acc"),
        )
    )
    scan, _trailing = scan_from_loop(Loop(axis=Axis("j", 4), body=scan_body))
    assert scan.observe is not None
    red = _chain_member("acc", "k", "x", _provider())
    spec = OutputSpec(write=Write(output="running", index=(Var("m"), Var("j")), value=scan.observe.results[0]), sweep=())
    assert _classic._reduction_domain(_tile_stub(_chain_root(scan, red), (spec,)), red) == (Reduce(),)


def _per_cell_reductions(root, output_specs=()) -> set:
    """The reduce values ``_contraction_domain`` offers on the PER-CELL tier of ``root``'s
    contraction — asked through the contraction projection itself, not through
    ``_reduction_domain``, so that deleting the delegation between them fails this.

    The stub carries no typed inputs, so ``_warp_atoms`` refuses every tensor-core atom and the
    catalog is the scalar tiles alone; a tiled plan contracts K serially per register cell and is
    excluded here by ``is_tiled``.
    """
    con = next(edge for edge in root.operands if edge.as_contraction() is not None)
    tile = SimpleNamespace(
        output_specs=output_specs,
        op=root,
        inputs={},
        place=SimpleNamespace(free=()),
        packed_reading=lambda _node: (None, None),
    )
    domain = _classic._contraction_domain(tile, None, con, ContractionFacts(k_axis=_K))
    return {choice.reduce for choice in domain if not choice.tile.is_tiled}


def test_a_contraction_chain_member_inherits_the_member_domain(unpinned) -> None:
    """The contraction per-cell tier reads the SAME projection, so a contraction that is a direct
    chain member inherits the member catalog, the transposed exclusion, and the swept / streamed
    serial-only gates with no carve-out of its own. A contraction is a monoid with a ⊗ lift;
    nothing about the chain arm reads its algebra.

    Asked through ``_contraction_domain``, which is the only thing that makes this a test OF the
    delegation: routed through ``_reduction_domain`` directly it would stay green with the
    delegation deleted."""
    cone = projection(
        (_provider(),),
        (Load(name="a_e", input="A", index=(Var("m"), Var("k"))), Assign(name="a_scaled", op="multiply", args=("a_e", "v25"))),
    )
    con = contraction(_K, cone, (Load(name="b_e", input="B", index=(Var("k"),)), "acc"))
    root = _chain_root(con)
    assert _per_cell_reductions(root) == set(_member_catalog())

    swept = OutputSpec(write=Write(output="o", index=(Var("m"),), value="v"), sweep=(Axis("m", 4),))
    assert _per_cell_reductions(root, (swept,)) == {Reduce()}


def test_a_scoped_partition_pin_on_a_serial_only_chain_site_enumerates_nothing(unpinned) -> None:
    """The refusal direction, through the real pin path. A ``REDUCE`` value scoped to a site whose
    projected domain does not hold it empties that site's restriction, and the kernel enumerates NO
    row at all — the pin is never quietly satisfied by the serial fold it did not name. That, not
    a per-site exception, is what replaced the old walk's refusal: #691 made a pin a restriction on
    the projected domain, so a partition a reduce cannot carry simply has nothing to select: a reduce
    read PER STEP of its member (it indexes the member's axis) lowers inside that member's loop and
    is no chain member, so its site projects the serial fold alone.

    Only a SCOPED pin refuses. A graph-wide bare ``REDUCE: coop`` is applicable at a site only when
    ``coop`` is already in that site's projected values, so on a serial-only member it is silently
    inapplicable — the same adaptation that lets one ambient pin sweep a whole model.

    The positive direction rides along: the DIRECT member's own site does enumerate under the same
    pin, so a green assertion here cannot come from the kernel being unschedulable outright."""
    provider = _provider()
    scaled = Assign(name="acc_inner__v", op="multiply", args=("x_e", provider.exposes[0]))
    inner = reduction("k", (slab("x_e", "x", "m", "j", "k"), provider), (scaled,), ("acc_inner",))
    # The outer fold over ``j`` accumulates the inner's state as its per-step value — a lift whose
    # result is the bound operand param, as the total lift spells ``Accum(acc_outer, acc_inner)``.
    outer = Fold(
        operands=(inner,),
        lift=Lambda.closing(("j", "acc_inner"), Body(()), ("acc_inner",)),
        init=(0.0,),
        base=Lambda.componentwise(("add",), ("acc_outer",)),
    )
    root = _chain_root(outer, results=("acc_outer",))
    tile = TileOp(op=root, place=Placement(free=(Axis("m", 4),)), axes=(Axis("m", 4), Axis("j", 4), _K), name="k_chain_probe", knobs={})
    assert tile.op is outer, "an identity projection over one member dissolves into the member"

    member = tile.op
    nested = next(edge for edge in member.operands if edge.axis is not None)
    sched = Sched(tile, place=tile.place.on_grid())
    member_key, nested_key = sched.key("REDUCE", member), sched.key("REDUCE", nested)
    ctx = Context.from_target(_CC)

    def rows(pins: dict) -> list[dict]:
        with pinned_knobs(pins):
            return [dict(leaf.knobs) for leaf in iter_leaves(_SCHEDULE_RULE.classic_forks(tile, tile.name, {}, ctx))]

    unpinned_rows = rows({})
    # The member is the kernel's peeled root, so its rows are exactly its projected domain — the
    # transposed band included, which only a root may carry.
    assert {str(row[member_key]) for row in unpinned_rows} == {choice.spell() for choice in _classic._reduction_domain(tile, member)}
    assert any(choice.coop_transposed for choice in _classic._reduction_domain(tile, member))
    assert {str(row[nested_key]) for row in unpinned_rows} == {""}, "the nested fold is serial-only"

    assert rows({nested_key: "coop"}) == [], "a partition scoped to a serial-only site must enumerate nothing"
    assert rows({member_key: "coop"}), "the direct member's own site still enumerates under the same value"
