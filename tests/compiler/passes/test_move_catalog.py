"""Structural-coverage test for the permitted-move catalog (``search/space.py``).

The scheduling emit (``010_recognize`` → ``_schedule``) enumerates the catalog into the tile fork; this
file pins the catalog's **legal product** two ways:

- the catalog function ``scalar_tile_moves()`` equals the hand-computed ``(par × reg)`` grid plus the
  per-cell tile, legality-guarded (``par_n·par_m ≤ 1024``), read through the SITE spelling each
  move stores as (its site ``TILE`` value + the ``WORK`` inventory it implies). Membership is what
  is pinned, never position — the catalogs rank nothing;
- the **leaf set** the scheduler actually emits over a matmul fixture equals that product (keyed
  ``TILE@<k_axis>``) — so a missing / extra move is caught structurally, without lowering a kernel.
"""

from __future__ import annotations

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.pure.fold import Channel, Fold
from emmy.compiler.ir.schedule import ReducePlan, Stage, TilePlan, Workers, plan_workers, resolve_site_tile
from emmy.compiler.ir.stmt import Assign, Body, Load
from emmy.compiler.ir.tile import Placement, TileOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import iter_leaves
from emmy.compiler.pipeline.knob import axis_of, family_of, family_value
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.space import MAX_BLOCK_THREADS as _MAX_BLOCK_THREADS
from emmy.compiler.pipeline.search.space import scalar_tile_moves

# The hand-computed legal product as explicit literals — the per-cell tile and the (par × reg) box
# as the pair each move STORES: its site-local ``TILE`` value (the register sub-tile; the default
# ``f1x1`` suppresses to empty and a unit ``reg_m`` drops the ``x`` half) and the ``WORK`` thread
# inventory its parallel widths imply. The two ladders and the one bound are restated by hand here —
# NOT recomputed from ``_SCALAR_TILE_SPACE`` — so a change to either dimension, to the thread budget,
# or to the enumeration order is caught explicitly.
_PARS = [(pn, pm) for pn in (16, 32, 64) for pm in (8, 16) if pn * pm <= _MAX_BLOCK_THREADS]
_REGS = [(rn, rm) for rn in (1, 2, 4) for rm in (1, 2, 4, 6, 8, 10, 12, 14, 26)]


def _reg_spelling(rn: int, rm: int) -> str:
    """How a register sub-tile spells site-locally: the ``f1x1`` default suppresses entirely and a
    unit ``reg_m`` drops the ``x`` half."""
    if (rn, rm) == (1, 1):
        return ""
    return f"f{rn}" if rm == 1 else f"f{rn}x{rm}"


_EXPECTED_MOVES = [("", "")] + [(_reg_spelling(*reg), f"t{pn}x{pm}") for pn, pm in _PARS for reg in _REGS]


def _stored(plan: TilePlan) -> tuple[str, str]:
    """The (site ``TILE`` value, ``WORK`` inventory) pair a tile move stores."""
    work = plan_workers(plan)
    return plan.spell(), (work.spell() if work is not None else "")


def test_scalar_tile_moves_equals_hand_product():
    moves = scalar_tile_moves()
    assert [_stored(p) for p in moves] == _EXPECTED_MOVES
    assert TilePlan() in moves  # the untiled per-cell tile is a member; where it sits is not a rule
    assert len(set(moves)) == len(moves)  # no duplicate candidates
    # Every move round-trips its stored spelling and stays inside the thread budget.
    for plan in moves:
        site, work = _stored(plan)
        assert resolve_site_tile(site, Workers.parse(work)) == plan
        assert plan.units_n * plan.units_m <= _MAX_BLOCK_THREADS


def _matmul_graph() -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (1, Dim(64), Dim(64))), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(64), Dim(64))), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (1, Dim(64), Dim(64))), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g


def test_schedule_leaves_key_tile_canonically():
    """Each emitted contraction leaf keys its output tile by the CANONICAL codec spelling
    (phase 3): a single-contraction kernel's shortest unique key is bare ``TILE`` — the exact
    spelling the golden/DB corpus stores, so the stamped row IS the stored row."""
    axes: set[str | None] = set()

    def decide(fp):
        leaf = next(iter_leaves(fp.options))
        for k in getattr(leaf, "knobs", {}):
            if family_of(k) == "TILE":
                axes.add(axis_of(k))
        return leaf

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(_matmul_graph(), decide)
    assert axes == {None}  # one contraction -> the bare canonical spelling, never axis-suffixed


def _fp16_matmul_graph() -> Graph:
    """A static fp16 square matmul (K=512, tile-divisible for every ``bk``) — warp (mma) moves
    are eligible, and the big 256×256 warp tiles fit unmasked."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Dim(512), Dim(512)), "f16"), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(512), Dim(512)), "f16"), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (Dim(512), Dim(512)), "f16"), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g


def test_over_budget_warp_tile_offers_only_gmem_direct(monkeypatch):
    """The known 128 KiB slot is addressed through a narrow pin and offers no staged sibling."""
    ctx = Context.from_target((8, 9))  # the issue's sm_89 cap (101376 B)
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f32/f4x8/k8")
    monkeypatch.setenv("EMMY_WORK", "w4x4")
    monkeypatch.setenv("EMMY_REDUCE", "")
    rows: list[dict] = []

    def decide(fp):
        leaves = list(iter_leaves(fp.options))
        rows.extend(dict(getattr(leaf, "knobs", {}) or {}) for leaf in leaves)
        return leaves[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(_fp16_matmul_graph(), decide)
    assert rows
    assert all(family_value(row, "STAGE") == "" for row in rows)


def _reduce_graph() -> Graph:
    """A bare full-row sum reduce (the ``reduce.2048x2048`` golden shape) — a lifted
    :class:`Fold` over a 2048-cell free grid."""
    from emmy.compiler.ir.frontend.ir import MeanOp

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2048, 2048)), node_id="x")
    g.add_node(MeanOp(), ["x"], Tensor("y", (2048, 1)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    return g


def test_bare_reduce_forks_the_coop_catalog():
    """A bare reduce must fork the full legal ``coop_reduce_moves()`` catalog beside serial,
    whatever the free grid measures. A grid-size rule once collapsed the fork to ONE serial spec
    (no options offered), so the tune benched a single variant and greedy deployed 53× behind the
    pinned ``b16``/``b32`` reduce goldens (eighth golden sweep, finding 3). The offer is a function
    of legality alone now: the reduce extent has to be able to feed the band, and nothing else
    narrows it."""
    from emmy.compiler.pipeline.search.space import coop_reduce_moves

    rows: list[dict] = []

    def decide(fp):
        from emmy.compiler.pipeline.pipeline import _is_structural_option

        leaves = list(iter_leaves(fp.options))
        if any(_is_structural_option(leaf) for leaf in leaves):
            return next(leaf for leaf in leaves if not _is_structural_option(leaf))
        for leaf in leaves:
            rows.append(dict(getattr(leaf, "knobs", {}) or {}))
        return leaves[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(_reduce_graph(), decide)
    assert rows, "no fork was offered for the bare reduce"
    # A row's REDUCE value sheds the coop width into WORK, so the catalog identity is the
    # (site value, WORK) pair — the 16- and 32-wide folds both spell "coop" but ride distinct
    # t16 / t32 inventories.
    offered = [(str(family_value(r, "REDUCE")), str(r.get("WORK", ""))) for r in rows]
    assert ("", "") in offered  # the serial fold is a member of the set; its position is not a rule

    # This bare reduce meets the transposed band's structural gate (scalar tail, no
    # shared-row stage, static K, 32-divisible free grid), so the FULL catalog is offered —
    # bt/g-composites included. Rows that fail the gate (softmax/rms shapes) drop the band;
    # that arm is covered by the schedule tests, not this catalog assertion.
    def site_of(plan: ReducePlan) -> tuple[str, str]:
        return plan.spell(), (f"t{plan.coop}" if plan.coop > 1 else "")

    assert set(offered) == {("", ""), *(site_of(p) for p in coop_reduce_moves())}, f"catalog rows missing: {offered}"


def _computed_b_term():
    """A contraction ``sum_k a[m, k] · b_k`` whose B edge is COMPUTED — an inline ``Fold`` over its
    own axis. The parent fill realizes the computed edge, so only the contraction is a schedule
    site."""
    from emmy.compiler.ir.axis import Axis, AxisRole
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.pure.fold import Channel, Fold
    from emmy.compiler.ir.stmt import Accum, Body, Load, Loop
    from emmy.compiler.ir.tile import TileOp
    from emmy.compiler.ir.tile.ir import Placement
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop

    inner = fold_from_loop(
        Loop(
            axis=Axis("j", 256),
            body=Body((Load(name="w_e", input="w", index=(Var("m"), Var("j"))), Accum(name="bacc", value="w_e", op="add", axes=("j",)))),
            role=AxisRole.PLANAR,
        )
    )
    node = Fold.contraction(
        k_axis=Axis("k", 256),
        a=Load(name="a_e", input="a", index=(Var("m"), Var("k"))),
        channels=(Channel(b=inner, acc="acc"),),
    )
    return TileOp(op=node, place=Placement(free=(Axis("m", 64), Axis("n", 64))))


def _computed_a_term() -> TileOp:
    """An f32 contraction whose A value is computed inline and shared across the N tile."""
    m, n, k = Axis("m", 8), Axis("n", 8), Axis("k", 16)
    a = Fold.projection(
        body=Body(
            (
                Load(name="score", input="scores", index=(Var("m"), Var("k"))),
                Assign(name="prob", op="exp", args=("score",)),
            )
        )
    )
    node = Fold.contraction(
        k_axis=k,
        a=a,
        channels=(Channel(b=Load(name="value", input="values", index=(Var("k"), Var("n"))), acc="acc"),),
    )
    return TileOp(
        op=node,
        place=Placement(free=(m, n)),
        inputs={"scores": Tensor("scores", (8, 16), "f32"), "values": Tensor("values", (16, 8), "f32")},
        outputs={"out": Tensor("out", (8, 8), "f32")},
    )


def test_independent_roots_only_cross_physically_compatible_tiles(monkeypatch):
    """Reversed algebraic m/n readings may share a grid only at equal physical axis widths."""
    from types import SimpleNamespace

    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.pipeline.passes.lowering.tile import _schedule as sch

    first = SimpleNamespace(keys={"TILE": "TILE@first"}, site=SimpleNamespace(node="first"))
    second = SimpleNamespace(keys={"TILE": "TILE@second"}, site=SimpleNamespace(node="second"))
    first_plan = TilePlan(regs=(1, 2))  # physical m=1, n=2
    compatible = TilePlan(regs=(2, 1))  # under (n, m): physical n=2, m=1
    incompatible = TilePlan(regs=(1, 2))  # under (n, m): physical n=1, m=2
    rows = {
        "first": [sch._Row(knobs={"TILE@first": first_plan.spell()}, plans={"TILE@first": first_plan})],
        "second": [
            sch._Row(knobs={"TILE@second": compatible.spell()}, plans={"TILE@second": compatible}),
            sch._Row(knobs={"TILE@second": incompatible.spell()}, plans={"TILE@second": incompatible}),
        ],
    }

    monkeypatch.setattr(sch, "_rows_at", lambda _term, root, _work, **_kwargs: rows[root.site.node])
    axes = (Axis("m", 8), Axis("n", 8))
    sched = SimpleNamespace(placed=lambda node, plan: plan.at(*(axes if node == "first" else tuple(reversed(axes)))))
    term = SimpleNamespace(
        tree=(first, second),
        sched=sched,
        tile_nodes={"TILE@first": "first", "TILE@second": "second"},
        fragment_edges=(),
    )

    result = sch._term_rows(term, None)
    assert len(result) == 1
    assert result[0] == sch._Row.union((rows["first"][0], rows["second"][0]))


def test_fragment_consumer_may_inline_an_untiled_producer():
    """A scalar child has no fragment interface; the tiled consumer may evaluate it into smem."""
    from types import SimpleNamespace

    from emmy.compiler.pipeline.passes.lowering.tile import _schedule as sch

    warp = resolve_site_tile("mma_m16n8k16_f16_f32/f1x1", Workers.parse("w1x1"))
    scalar = TilePlan()
    term = SimpleNamespace(fragment_edges=(("TILE@consumer", "TILE@producer"),))

    def interface(consumer, producer):
        plans = tuple(sorted({"TILE@consumer": consumer, "TILE@producer": producer}.items()))
        stages = (("TILE@consumer", Stage(transport="smem")),)
        return True, (), plans, stages

    assert sch._merge_interfaces(term, (interface(warp, scalar),)) is not None
    assert sch._merge_interfaces(term, (interface(scalar, warp),)) is None


# --- the WORK pin's one non-narrowing branch ----------------------------------------------------- #


def test_work_pin_widens_only_where_the_site_offers_no_warp_inventory(monkeypatch):
    """A pin NARROWS — except in ``_inventories``' one fallback, where a ``WORK`` pin that matches
    no candidate is offered BESIDE the catalog's own inventories instead of replacing them.

    That branch is the PIN-BLEED rule: one env pin, several kernels in the graph, and this term is
    not the one it was written for. A pin the site DOES offer narrows to exactly that inventory; a
    pin it cannot offer leads and the catalog's own stay as siblings, so the term still maps rather
    than being left unmapped over a pin that was never about it. The fixture below is a pure reduce
    — a term with no warp geometry of any kind — which is what makes the second half a statement
    about pin bleed and not about coverage: the twisted streaming site enumerates its own warp
    inventories now, so a ``w<M>x<N>`` pin narrows there like anywhere else."""
    from emmy.compiler.ir.axis import Axis, AxisRole
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.stmt import Accum, Body, Load, Loop
    from emmy.compiler.ir.tile import Placement, TileOp
    from emmy.compiler.pipeline.passes.lowering.tile import _schedule as sch
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop

    ctx = Context.from_target((12, 0))
    fold = fold_from_loop(
        Loop(
            axis=Axis("kv", 256),
            body=Body((Load(name="v_e", input="v", index=(Var("m"), Var("kv"))), Accum(name="acc", value="v_e", op="add", axes=("kv",)))),
            role=AxisRole.PLANAR,
        )
    )
    tile = TileOp(op=fold, place=Placement(free=(Axis("m", 64),)))

    def inventories() -> list[str]:
        term = sch._Term(tile, tile.place.on_grid(), ctx)
        return [w.spell() if w is not None else "" for w in sch._inventories(term)]

    monkeypatch.delenv("EMMY_WORK", raising=False)
    offered = inventories()
    assert offered and not any(w.startswith("w") for w in offered), offered  # reduce bands only

    # A pin the site DOES offer narrows to it alone.
    monkeypatch.setenv("EMMY_WORK", offered[-1])
    assert inventories() == [offered[-1]]

    # A pin it cannot offer LEADS, and the catalog's own stay as siblings rather than being emptied.
    monkeypatch.setenv("EMMY_WORK", "w4x1")
    widened = inventories()
    assert widened == ["w4x1", *offered], widened


# --- what the enumeration owes: membership, not position ----------------------------------------- #


def _rows_of(tile) -> list[dict]:
    """Every row ``tile`` enumerates."""
    from emmy.compiler.pipeline.passes.lowering.tile import _schedule as sch

    term = sch._Term(tile, tile.place.on_grid(), Context.from_target((12, 0)))
    rows, _keys, _total = sch._enumerate(term)
    assert rows, "the term enumerated nothing"
    return rows


def test_f32_computed_a_contraction_offers_a_tiled_scalar_row():
    """Without an f32 MMA atom, computed A must still ride a scalar output tile so one A value can
    be reused across N instead of the per-cell fallback recomputing its entire cone for every output."""
    rows = _rows_of(_computed_a_term())

    def tile_of(row) -> TilePlan:
        work = Workers.parse(str(row.get("WORK", "")))
        reduce = ReducePlan.parse(str(family_value(row, "REDUCE") or ""), work)
        return resolve_site_tile(str(family_value(row, "TILE") or ""), work, reduce.coop)

    plans = [tile_of(row) for row in rows]
    assert any(plan.is_tiled and not plan.is_warp for plan in plans), "computed A lost every tiled scalar schedule"


def _reduce_term():
    """A bare 4096-wide row reduce over a 64-cell grid, as an unmapped ``TileOp``."""
    from emmy.compiler.ir.axis import Axis, AxisRole
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.stmt import Accum, Body, Load, Loop
    from emmy.compiler.ir.tile import Placement, TileOp
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop

    fold = fold_from_loop(
        Loop(
            axis=Axis("k", 4096),
            body=Body((Load(name="x_e", input="x", index=(Var("m"), Var("k"))), Accum(name="acc", value="x_e", op="add", axes=("k",)))),
            role=AxisRole.PLANAR,
        )
    )
    return TileOp(op=fold, place=Placement(free=(Axis("m", 64),)))


def test_the_all_off_row_is_always_offered(monkeypatch):
    """The untiled / serial / gmem-direct schedule — every family at its declared OFF — is legal on
    any term the walk can schedule, so it is always a MEMBER of the enumerated set.

    This is the enumeration fact that replaced the old "option-0 is each family's conservative
    default" obligation. Position is no longer anything: nothing may lead a list to steer a compile
    with no evidence, and such a compile taking an arbitrary row is accepted. What must still hold
    is that the all-OFF row exists to be picked, by evidence or by a pin — a term that could not
    spell it would have a hole in its space, not a slow default."""
    from emmy.compiler.pipeline.knob import is_off_value, stamp_schedule_families

    monkeypatch.delenv("EMMY_WORK", raising=False)
    for label, tile in {"bare reduce": _reduce_term(), "computed-B contraction": _computed_b_term()}.items():
        stamped = [stamp_schedule_families({k: v for k, v in row.items() if not k.startswith("S_")}) for row in _rows_of(tile)]
        assert any(all(is_off_value(family_of(fam), value) for fam, value in row.items()) for row in stamped), (
            f"{label}: no row spells every family's OFF value"
        )


def test_a_cooperative_row_spells_its_own_inventory(monkeypatch):
    """Since step 7 a cooperative band's WIDTH lives in ``WORK``, not in the ``REDUCE`` value — so a
    row that partitions a fold cooperatively is only well-formed beside the thread inventory that
    carries the width. Checked over the WHOLE set rather than one leading row: two rows spelling
    one wire format while naming different kernels is the defect, and it has nothing to do with
    which of them the walk emitted first."""
    monkeypatch.delenv("EMMY_WORK", raising=False)
    for label, tile in {"bare reduce": _reduce_term(), "computed-B contraction": _computed_b_term()}.items():
        for row in _rows_of(tile):
            work = str(row.get("WORK", ""))
            coop = [v for k, v in row.items() if family_of(k) == "REDUCE" and isinstance(v, str) and "coop" in v]
            if not coop:
                continue
            parsed = Workers.parse(work or None)
            assert parsed is not None and parsed.kind == "thread", f"{label}: {coop} rides WORK={work!r}, not a thread band"
            assert ReducePlan.parse(coop[0], parsed).coop == parsed.units[0], f"{label}: {coop} disagrees with WORK={work!r}"
