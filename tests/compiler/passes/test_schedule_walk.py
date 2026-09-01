"""The schedule walk's enumeration contracts — what outlived the deleted addressable product.

The walk has no product space and no ``space[i]``: one recursive traversal IS the enumeration, the
prescan is the one place a catalog question is asked, and sampling is a reservoir over the leaf
stream (``search/pool.py``). What survives of the old space's contract is pinned here: one prescan
per term, computed and derived fold sites keyed as schedule sites, the paired mma row on the flash
pair, and the structural split fork standing beside the walk with both finalize arms on offer.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp, SdpaOp
from emmy.compiler.pipeline.knob import family_of
from emmy.compiler.pipeline.passes.lowering.tile import _schedule
from emmy.compiler.pipeline.search.golden_eval import enumerate_graph
from emmy.compiler.pipeline.search.pool import PoolSample

_CC = (12, 0)

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
    original = _schedule._options

    def spy(state, node):
        asked.append((state.tile, node))  # strong refs, so ids below cannot alias freed objects
        return original(state, node)

    monkeypatch.setattr(_schedule, "_options", spy)
    assert _rows(FIXTURES[case]())
    assert asked, "the fixture built no catalog at all"
    keys = [(id(tile), id(node)) for tile, node in asked]
    repeats = len(keys) - len(set(keys))
    assert not repeats, f"_options was asked the same question {repeats} time(s) over ({len(keys)} calls)"


def test_the_prescan_reads_each_computed_a_seam_once(unpinned, monkeypatch) -> None:
    """A computed-A cone is lowered once for its stat-row seam, not once per tile plan."""
    from emmy.compiler.pipeline.passes.lowering.tile import _staging  # noqa: PLC0415

    calls: list[tuple] = []
    original = _schedule.cone_seam

    def spy(cone, k_name):
        calls.append((cone, k_name))
        return original(cone, k_name)

    monkeypatch.setattr(_schedule, "cone_seam", spy)
    monkeypatch.setattr(
        _staging,
        "cone_seam",
        lambda *_: (_ for _ in ()).throw(AssertionError("the fill must reuse the prescan's seam")),
    )
    assert _rows(FIXTURES["fused_norm_linear"]())
    assert calls
    keys = [(id(cone), k_name) for cone, k_name in calls]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("case, tile_sites, reduce_sites", (("fused_norm_linear", 1, 2), ("flash_pair", 2, 3)))
def test_computed_fold_sites_are_keyed_schedule_sites(case, tile_sites, reduce_sites, unpinned) -> None:
    """A computed cone's fold and a derived site (flash's synthesized PV) are real schedule sites:
    every row spells them, keyed by the tree-path codec, so nothing nested is silently undecided."""
    row = _rows(FIXTURES[case]())[0]
    assert sum(key == "TILE" or key.startswith("TILE@") for key in row) == tile_sites
    assert sum(key == "REDUCE" or key.startswith("REDUCE@") for key in row) == reduce_sites


def test_sdpa_fold_tree_offers_a_paired_mma_row(unpinned, monkeypatch) -> None:
    """The walk reaches a row where BOTH flash contractions ride the tensor core — the score's N
    tile feeding the value contraction's streamed K block through the fragment seam."""
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    # The score's N tile is the value contraction's streamed K block.
    monkeypatch.setenv("EMMY_TILE@A3", "mma_m16n8k16_f16_f32/f1x2")
    monkeypatch.setenv("EMMY_TILE@PJ", "mma_m16n8k16_f16_f32/f1x1")
    monkeypatch.setenv("EMMY_STAGE", "")
    monkeypatch.setenv("EMMY_RASTER", "")
    monkeypatch.setenv("EMMY_REDUCE", "")
    rows = _rows(_sdpa_graph())
    assert any(sum(key.startswith("TILE@") and "mma_" in str(value) for key, value in row.items()) == 2 for row in rows)


def test_the_split_fork_offers_atomic_and_deferred_arms(unpinned) -> None:
    """The old product kept the cross-CTA partitions as combined rows; they are now STRUCTURAL
    siblings of the unsplit tree (``035_split_reduce``), and a fold that admits both finalizes
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


def test_the_twisted_carrier_split_offers_only_the_deferred_arm(unpinned, monkeypatch) -> None:
    """The cross-CTA split composes with the paired-mma flash cell. Asserted at the offer
    function: after a cut fork decides on the same kernel, the engine's structural replay resolves
    a later structural fork inline (the count evidence is per-op, not per-rule — see
    ``_replay_structural_decision``), so the split's siblings never reach a decide on this graph —
    the offer itself is the observable. The atomic
    arm is refused on the carrier's ARITY (``atomicAdd`` folds one additive state; the twisted
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

    def keep(fp):
        op = fp.root_op
        if isinstance(op, TileOp) and op.op is not None and not op.place.is_mapped and not captured:
            captured.append(op)
        return next(iter_leaves(fp.options))

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
    monkeypatch.setenv("EMMY_TILE@A3", "mma_m16n8k16_f16_f32/f1x2")
    monkeypatch.setenv("EMMY_TILE@PJ", "mma_m16n8k16_f16_f32/f1x1")
    monkeypatch.setenv("EMMY_STAGE", "")
    monkeypatch.setenv("EMMY_RASTER", "")
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target(_CC)).resolve(
        _sdpa_graph(), lambda fp: next(iter_leaves(fp.options))
    )
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


def test_computed_b_statistic_is_a_keyed_schedule_site(unpinned, monkeypatch) -> None:
    """A score contraction's computed B operand cone closes over its per-key statistic
    (``normalize_fold_tree``'s reduce-body closing), and the walk keys that relocated fold as an
    ordinary schedule site — nothing nested inside a B edge is silently undecided."""
    monkeypatch.setenv("EMMY_TILE", "")
    monkeypatch.setenv("EMMY_STAGE", "")
    monkeypatch.setenv("EMMY_RASTER", "")
    monkeypatch.setenv("EMMY_REDUCE", "")
    graph = _code_graph(
        "torch.nn.functional.scaled_dot_product_attention("
        f"{_NORM}(torch.randn(1, 2, 8, 16, dtype=torch.float16)), "
        f"{_NORM}(torch.randn(1, 2, 8, 16, dtype=torch.float16)), "
        "torch.randn(1, 2, 8, 16, dtype=torch.float16))"
    )
    rows = _rows(graph)
    assert rows, "the fused attention kernel must still enumerate"
    keyed_under_b = [key for row in rows for key in row if key.split("@", 1)[0] == "REDUCE" and "b" in key.split("@", 1)[-1].split(".")]
    assert keyed_under_b, "no REDUCE site was keyed inside a computed B operand cone"


def test_a_sweep_reading_fold_offers_only_the_serial_reduce(unpinned) -> None:
    """A fold whose cone reads a boundary store's sweep axis must be ENCLOSED by the output sweep
    loop (the materializer binds the projection unpeeled), and a partitioned combine cannot ride
    inside the per-lane sweep — so the serial fold is the whole catalog, decided at the offer.
    Offering a band and declining it at the kernel binder instead costs one full greedy
    re-resolve per declined row (DeepSeek-V4's fused ``k_div_36_reduce`` on the live V100)."""
    from types import SimpleNamespace

    from emmy.compiler.ir.axis import Axis, AxisRole
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.stmt import Accum, Body, Load, Loop, Write
    from emmy.compiler.ir.tile import OutputSpec, ReducePlan
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop

    body = Body(
        (
            Load(name="x_e", input="x", index=(Var("m"), Var("k"), Var("j"))),
            Accum(name="acc", value="x_e", op="add", axes=("k",)),
        )
    )
    red = fold_from_loop(Loop(axis=Axis("k", 128), body=body, role=AxisRole.PLANAR))
    assert red is not None
    sweep_spec = OutputSpec(write=Write(output="o", index=(Var("m"), Var("j")), value="v"), sweep=Axis("j", 4))

    def state(specs):
        return SimpleNamespace(tile=SimpleNamespace(output_specs=specs, op=red), work_pin=None, transposed_ok=False)

    assert _schedule._reduce_moves(state((sweep_spec,)), red, None) == [ReducePlan()]
    # Without the sweep read the catalog stays whole.
    assert len(_schedule._reduce_moves(state(()), red, None)) > 1


def test_a_direct_member_of_a_chain_form_root_offers_the_catalog(unpinned) -> None:
    """A direct body member of a chain-form root binds through the chain arm — its provider chain
    emits ahead of the strided loop — so it offers the full catalog, minus the transposed band
    (whose close assumes the kernel-root fold shape). A live ``REDUCE`` pin is honored the same
    way: a legal non-transposed partition binds exactly, and a transposed one is a recorded
    refusal."""
    from types import SimpleNamespace

    from emmy.compiler.ir.axis import Axis, AxisRole
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.pure import Fold
    from emmy.compiler.ir.schedule import Workers
    from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
    from emmy.compiler.ir.tile import ReducePlan
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    body = Body(
        (
            Load(name="x_e", input="x", index=(Var("m"), Var("k"))),
            Assign(name="scaled", op="multiply", args=("x_e", "v25")),
            Accum(name="acc", value="scaled", op="add", axes=("k",)),
        )
    )
    red = fold_from_loop(Loop(axis=Axis("k", 128), body=body, role=AxisRole.PLANAR))
    assert red is not None
    chain = (Load(name="ws", input="cutbuf", index=()), Assign(name="v25", op="rsqrt", args=("ws",)))
    root = Fold.projection(body=Body((*chain, red)), results=("acc",))
    assert not root.operands, "the fed member must remain in the body for this shape to be under test"

    def state(op, transposed_ok=False):
        return SimpleNamespace(
            tile=SimpleNamespace(output_specs=(), op=op), work_pin=Workers(kind="thread", units=(1, 32)), transposed_ok=transposed_ok
        )

    moves = _schedule._reduce_moves(state(root), red, None)
    assert moves == [p for p in _schedule._reduce_catalog(state(root), 128) if not p.coop_transposed]

    # A legal non-transposed pin is honored exactly, against the kernel's WORK inventory.
    with pinned_knobs({"REDUCE": "coop"}):
        assert _schedule._reduce_moves(state(root), red, "REDUCE") == [ReducePlan.of(coop=32)]
    # A transposed pin still refuses on a chain member — its close assumes the kernel-root fold
    # shape — even with ``transposed_ok=True``, proving the refusal is the chain arm's own, not
    # the general transposed-geometry gate `_transposed_refusal` would raise at `transposed_ok=False`.
    with pinned_knobs({"REDUCE": "coop-t"}), pytest.raises(_schedule.PinRefused, match="a chain-form member cannot realize it"):
        _schedule._reduce_moves(state(root, transposed_ok=True), red, "REDUCE")


def test_a_fold_nested_under_a_chain_member_offers_only_the_serial_reduce(unpinned) -> None:
    """A fold nested UNDER a chain-form root's direct member — not itself a member of the root's
    own body — binds through that member's own body recursion, so no cooperative / ILP partition
    can ride under it either: the serial fold is the whole catalog, and a live pin naming a
    partition is a recorded refusal, never a silent drop."""
    from types import SimpleNamespace

    from emmy.compiler.ir.axis import Axis, AxisRole
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.pure import Fold
    from emmy.compiler.ir.schedule import Workers
    from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
    from emmy.compiler.ir.tile import ReducePlan
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    inner_body = Body(
        (
            Load(name="x_e", input="x", index=(Var("m"), Var("k"))),
            Accum(name="acc_inner", value="x_e", op="add", axes=("k",)),
        )
    )
    inner_loop = Loop(axis=Axis("k", 128), body=inner_body, role=AxisRole.PLANAR)
    outer_body = Body((inner_loop, Accum(name="acc_outer", value="acc_inner", op="add", axes=("m",))))
    outer = fold_from_loop(Loop(axis=Axis("m", 4), body=outer_body, role=AxisRole.PLANAR))
    inner = next(member for member in outer.lift.body if isinstance(member, Fold))

    chain = (Load(name="ws", input="cutbuf", index=()), Assign(name="v25", op="rsqrt", args=("ws",)))
    root = Fold.projection(body=Body((*chain, outer)), results=("acc_outer",))
    assert not root.operands, "the fed member must remain in the body for this shape to be under test"

    def state():
        return SimpleNamespace(
            tile=SimpleNamespace(output_specs=(), op=root), work_pin=Workers(kind="thread", units=(1, 32)), transposed_ok=False
        )

    assert _schedule._reduce_moves(state(), inner, None) == [ReducePlan()]
    with pinned_knobs({"REDUCE": "coop"}), pytest.raises(_schedule.PinRefused, match="nested under a chain-form root's member"):
        _schedule._reduce_moves(state(), inner, "REDUCE")


def test_a_sweep_carrying_store_keeps_chain_members_serial(unpinned) -> None:
    """A chain-form root's boundary store carrying an output sweep keeps every direct member
    serial too — even one the sweep axis never enters — because the sweep loop must enclose the
    whole kernel tail and a partitioned member's lane-distributed close cannot re-run per swept
    cell. A live pin naming a partition is a recorded refusal naming the sweep, not a silent
    drop."""
    from types import SimpleNamespace

    from emmy.compiler.ir.axis import Axis, AxisRole
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.pure import Fold
    from emmy.compiler.ir.schedule import Workers
    from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
    from emmy.compiler.ir.tile import OutputSpec, ReducePlan
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    body = Body(
        (
            Load(name="x_e", input="x", index=(Var("m"), Var("k"))),
            Assign(name="scaled", op="multiply", args=("x_e", "v25")),
            Accum(name="acc", value="scaled", op="add", axes=("k",)),
        )
    )
    red = fold_from_loop(Loop(axis=Axis("k", 128), body=body, role=AxisRole.PLANAR))
    assert red is not None
    chain = (Load(name="ws", input="cutbuf", index=()), Assign(name="v25", op="rsqrt", args=("ws",)))
    root = Fold.projection(body=Body((*chain, red)), results=("acc",))
    sweep_spec = OutputSpec(write=Write(output="o", index=(Var("m"), Var("j")), value="v"), sweep=Axis("j", 4))

    def state():
        return SimpleNamespace(
            tile=SimpleNamespace(output_specs=(sweep_spec,), op=root), work_pin=Workers(kind="thread", units=(1, 32)), transposed_ok=False
        )

    assert _schedule._reduce_moves(state(), red, None) == [ReducePlan()]
    with pinned_knobs({"REDUCE": "coop"}), pytest.raises(_schedule.PinRefused, match="boundary store carries an output sweep"):
        _schedule._reduce_moves(state(), red, "REDUCE")


def test_a_streamed_store_keeps_chain_members_serial(unpinned) -> None:
    """A chain-form root's boundary store that streams into a SIBLING observed member's reduce
    loop keeps every OTHER direct member serial too: the trailing append that splices a streamed
    store cannot reach a loop that already sits in an earlier segment, so the whole kernel binds
    without a peel — same contract as the swept store, decided at the offer so the binder never
    drops a stamped partition. A live pin naming a partition is a recorded refusal naming the
    streamed store, not a silent drop."""
    from types import SimpleNamespace

    from emmy.compiler.ir.axis import Axis, AxisRole
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.pure import Fold
    from emmy.compiler.ir.schedule import Workers
    from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
    from emmy.compiler.ir.tile import OutputSpec, ReducePlan
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop, scan_from_loop
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    scan_body = Body(
        (
            Load(name="y_e", input="y", index=(Var("m"), Var("j"))),
            Accum(name="scan_acc", value="y_e", op="add", axes=("j",)),
            Write(output="running", index=(Var("m"), Var("j")), value="scan_acc"),
        )
    )
    scan, _trailing = scan_from_loop(Loop(axis=Axis("j", 4), body=scan_body, role=AxisRole.PLANAR))
    assert scan.observe is not None

    body = Body(
        (
            Load(name="x_e", input="x", index=(Var("m"), Var("k"))),
            Assign(name="scaled", op="multiply", args=("x_e", "v25")),
            Accum(name="acc", value="scaled", op="add", axes=("k",)),
        )
    )
    red = fold_from_loop(Loop(axis=Axis("k", 128), body=body, role=AxisRole.PLANAR))
    assert red is not None
    chain = (Load(name="ws", input="cutbuf", index=()), Assign(name="v25", op="rsqrt", args=("ws",)))
    root = Fold.projection(body=Body((*chain, scan, red)), results=("acc",))
    streamed_spec = OutputSpec(write=Write(output="running", index=(Var("m"), Var("j")), value=scan.observe.results[0]), sweep=None)

    def state():
        return SimpleNamespace(
            tile=SimpleNamespace(output_specs=(streamed_spec,), op=root),
            work_pin=Workers(kind="thread", units=(1, 32)),
            transposed_ok=False,
        )

    assert _schedule._reduce_moves(state(), red, None) == [ReducePlan()]
    with (
        pinned_knobs({"REDUCE": "coop"}),
        pytest.raises(_schedule.PinRefused, match="streams into a sibling observed member's reduce loop"),
    ):
        _schedule._reduce_moves(state(), red, "REDUCE")
