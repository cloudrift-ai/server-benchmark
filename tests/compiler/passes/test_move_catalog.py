"""Structural coverage for the classic schedule move catalog.

The tile schedule enumerates the catalog into the tile fork; this
file pins the catalog's **legal set** three ways:

- the catalog function ``scalar_tile_moves()`` equals the hand-computed pure-register,
  one-dimensional thread, and ``(par × reg)`` grids plus the per-cell tile, legality-guarded
  (``par_n·par_m ≤ 1024``), read through the SITE spelling each move stores as (its site ``TILE``
  value + the ``WORK`` inventory it implies). Membership is what is pinned, never position — the
  catalogs rank nothing;
- the **leaf set** the walk actually emits over an f32 matmul / bare-reduce fixture equals that
  catalog, so a missing / extra move is caught structurally, without lowering a kernel;
- the complete leaf set the scheduler emits, including multi-site worker agreement.
"""

from __future__ import annotations

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.schedule import Reduce, Tile, Work, derive_workers, resolve_site_tile
from emmy.compiler.ir.schedule.catalog import MAX_BLOCK_THREADS as _MAX_BLOCK_THREADS
from emmy.compiler.ir.schedule.catalog import coop_reduce_moves, scalar_tile_moves
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.tile import Placement, TileOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import iter_leaves
from emmy.compiler.pipeline.knob import axis_of, complete_kernel_row, family_of, family_value, is_off_value
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from emmy.compiler.pipeline.pipeline import Run
from tests.compiler.terms import contraction, projection

# The hand-computed legal products as explicit literals — the per-cell tile, one-dimensional thread
# ladder, pure-register box, and (par × reg) box
# as the pair each move STORES: its site-local ``TILE`` value (the register sub-tile; the default
# ``f1x1`` spells ``f1`` on a parallel tile and a unit ``reg_m`` drops the ``x`` half) and the ``WORK`` thread
# inventory its parallel widths imply. The two ladders and the one bound are restated by hand here —
# NOT recomputed from the implementation spaces — so a change to any dimension, to the thread
# budget, or to the enumeration order is caught explicitly.
_PARS = [(pn, pm) for pn in (16, 32, 64) for pm in (8, 16) if pn * pm <= _MAX_BLOCK_THREADS]
_PAR_1D = (32, 64, 128, 256, 512)
_PURE_REGS = [(rn, rm) for rn in (1, 2, 3, 4) for rm in (1, 2, 4) if (rn, rm) != (1, 1)]
_PARALLEL_REGS = [(rn, rm) for rn in (1, 2, 4, 26) for rm in (1, 2, 4, 6, 8, 10, 12, 14, 26)]


def _reg_spelling(rn: int, rm: int) -> str:
    """How a register sub-tile spells site-locally: parallel ``f1x1`` spells ``f1`` and a
    unit ``reg_m`` drops the ``x`` half."""
    if (rn, rm) == (1, 1):
        return "f1"
    return f"f{rn}" if rm == 1 else f"f{rn}x{rm}"


_EXPECTED_MOVES = [
    ("", ""),
    *(("f1", f"t{pn}") for pn in _PAR_1D),
    *((_reg_spelling(*reg), "") for reg in _PURE_REGS),
    *((_reg_spelling(*reg), f"t{pn}x{pm}") for pn, pm in _PARS for reg in _PARALLEL_REGS),
]


def _stored(plan: Tile) -> tuple[str, str]:
    """The (site ``TILE`` value, ``WORK`` inventory) pair a tile move stores."""
    work = derive_workers((plan,))
    return plan.spell(), (work.spell() if work is not None else "")


def test_scalar_tile_moves_equals_hand_product():
    moves = scalar_tile_moves()
    assert [_stored(p) for p in moves] == _EXPECTED_MOVES
    assert Tile() in moves  # the untiled per-cell tile is a member; where it sits is not a rule
    assert len(set(moves)) == len(moves)  # no duplicate candidates
    # Every move round-trips its stored spelling and stays inside the thread budget.
    for plan in moves:
        site, work = _stored(plan)
        assert resolve_site_tile(site, Work.parse(work)) == plan
        assert plan.units_n * plan.units_m <= _MAX_BLOCK_THREADS


def test_coop_reduce_moves_equals_hand_product():
    """The normal cooperative and ILP stages form one fixed product; parameters do not add rows."""
    expected = {
        *(Reduce.of(coop=coop, reg=reg) for coop in (1, 4, 8, 16, 32, 64, 128, 256, 512) for reg in (1, 2, 4) if coop > 1 or reg > 1),
        *(Reduce.of(coop=coop, coop_transposed=True) for coop in (32, 64, 128, 256)),
    }
    assert set(coop_reduce_moves()) == expected
    assert len(coop_reduce_moves()) == len(expected)


def _matmul_graph() -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (1, Dim(64), Dim(64))), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(64), Dim(64))), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (1, Dim(64), Dim(64))), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g


def test_schedule_leaves_key_tile_canonically():
    """Each emitted contraction leaf keys its output tile by the CANONICAL codec spelling
    (phase 3): a single-contraction kernel's shortest unique key is bare ``TILE``."""
    axes: set[str | None] = set()

    def decide(fp):
        leaf = next(iter_leaves(fp.options))
        for k in getattr(leaf, "knobs", {}):
            if family_of(k) == "TILE":
                axes.add(axis_of(k))
        return leaf

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(_matmul_graph(), decide)
    assert axes == {None}


def _fp16_matmul_graph() -> Graph:
    """A static fp16 square matmul (K=512, tile-divisible for every ``bk``) — warp (mma) moves
    are eligible, and the big 256×256 warp tiles fit unmasked."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Dim(512), Dim(512)), "f16"), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(512), Dim(512)), "f16"), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (Dim(512), Dim(512)), "f16"), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g


def _tile_scheduled(op) -> bool:
    """Whether ``op`` is a kernel the tile schedule decided — it carries a ``TILE`` family key."""
    return bool(getattr(op, "knobs", None)) and any(family_of(k) == "TILE" for k in op.knobs)


def test_tile_pin_forces_the_named_warp_row(monkeypatch):
    """A TILE pin at one inventory forces the walk to exactly the row it names.

    The pin names one tile at one inventory, so the walk is FORCED and there is no fork to read the
    offer off: the rows it offered are the leaves of whatever fork survived, plus the kernel it
    actually realized — a forced walk offers exactly the row it built. The STAGE assertion below is
    vacuously true until the staging cluster restores the family; it then becomes the smem-budget
    claim this fixture was built for (the known over-budget 128 KiB slot on sm_89 must offer no
    staged sibling)."""
    ctx = Context.from_target((8, 9))  # the issue's sm_89 cap (101376 B)
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f32/f4x8/k8")
    monkeypatch.setenv("EMMY_WORK", "w4x4")
    monkeypatch.setenv("EMMY_REDUCE", "")
    rows: list[dict] = []

    def decide(fp):
        leaves = list(iter_leaves(fp.options))
        if "schedule" in fp.match.rule.name:  # the walk's own fork — not the placement / split offers
            rows.extend(dict(getattr(leaf, "knobs", {}) or {}) for leaf in leaves)
        return leaves[0]

    resolved, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(_fp16_matmul_graph(), decide)
    rows.extend(dict(node.op.knobs) for node in resolved.nodes.values() if _tile_scheduled(node.op))
    assert rows
    assert all(family_value(row, "TILE") == "mma_m16n8k16_f16_f32/f4x8/k8" for row in rows)
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
    def site_of(plan: Reduce) -> tuple[str, str]:
        return plan.spell(), (f"t{plan.coop}" if plan.coop > 1 else "")

    assert set(offered) == {("", ""), *(site_of(p) for p in coop_reduce_moves())}, f"catalog rows missing: {offered}"


def _computed_b_term() -> TileOp:
    """A contraction ``sum_k a[m, k] · b_k`` whose B edge is COMPUTED — an inline ``Fold`` over its
    own axis (``b_k = Σ_j w[k, j]``, so the edge varies with ``k`` and the pair reads as bilinear).
    The parent fill realizes the computed edge, so only the contraction is a schedule site."""
    m, n, k, j = Axis("m", 64), Axis("n", 64), Axis("k", 256), Axis("j", 256)
    inner = fold_from_loop(
        Loop(
            axis=j,
            body=Body((Load(name="w_e", input="w", index=(Var("k"), Var("j"))), Accum(name="bacc", value="w_e", op="add", axes=("j",)))),
        )
    )
    node = contraction(k, Load(name="a_e", input="a", index=(Var("m"), Var("k"))), (inner, "acc"))
    return TileOp(op=node, place=Placement(free=(m, n)), axes=(m, n, k, j))


def _computed_a_term() -> TileOp:
    """An f32 contraction whose A value is computed inline and shared across the N tile."""
    m, n, k = Axis("m", 8), Axis("n", 8), Axis("k", 16)
    a = projection((), (Load(name="score", input="scores", index=(Var("m"), Var("k"))), Assign(name="prob", op="exp", args=("score",))))
    node = contraction(k, a, (Load(name="value", input="values", index=(Var("k"), Var("n"))), "acc"))
    return TileOp(
        op=node,
        place=Placement(free=(m, n)),
        axes=(m, n, k),
        inputs={"scores": Tensor("scores", (8, 16), "f32"), "values": Tensor("values", (16, 8), "f32")},
        outputs={"out": Tensor("out", (8, 8), "f32")},
    )


# --- WORK pin narrowing -------------------------------------------------------------------------- #


def _rows_of(tile, ctx=None) -> list[dict]:
    """Every row ``tile`` enumerates — the leaves of the walk's own fork (a fully forced walk is
    still a one-leaf fork, so the engine records its row as a decision)."""
    from importlib import import_module

    classic_forks = import_module("emmy.compiler.pipeline.passes.lowering.tile.040_schedule").classic_forks
    out = classic_forks(tile, "k", {}, ctx or Context.from_target((12, 0)))
    return [dict(leaf.knobs) for leaf in iter_leaves(out)]


def test_work_pin_never_widens_a_site_catalog(monkeypatch):
    """A matching pin narrows to one inventory; an unmatched pin offers no schedule row."""
    ctx = Context.from_target((12, 0))
    tile = _row_reduce(Axis("m", 64), Axis("kv", 256), "v")

    def inventories() -> set[str]:
        """The inventories the term OFFERS — read off the rows themselves, which is now the only
        place they exist: a row spells the inventory its own options claimed, so what is offered
        and what is spellable cannot drift apart."""
        return {str(row.get("WORK", "")) for row in _rows_of(tile, ctx)}

    monkeypatch.delenv("EMMY_WORK", raising=False)
    offered = inventories()
    assert offered and not any(w.startswith("w") for w in offered), offered  # reduce bands only

    # A pin the site DOES offer narrows to it alone.
    def width(w: str) -> int:
        parsed = Work.parse(w or None)
        return parsed.count if parsed is not None else 0

    widest = max(offered, key=width)
    monkeypatch.setenv("EMMY_WORK", widest)
    assert inventories() == {widest}

    # A pin it cannot offer never manufactures an inventory or restores unpinned siblings.
    monkeypatch.setenv("EMMY_WORK", "w4x1")
    assert inventories() == set()


# --- what the enumeration owes: membership, not position ----------------------------------------- #


def _plain_matmul_term() -> TileOp:
    """A static f32 matmul as an unmapped ``TileOp`` — no warp atoms (f32 has no tensor-core cell),
    so the scalar catalog is the whole ``TILE`` offer."""
    m, n, k = Axis("m", 64), Axis("n", 64), Axis("k", 64)
    node = contraction(
        k, Load(name="a_e", input="a", index=(Var("m"), Var("k"))), (Load(name="b_e", input="b", index=(Var("k"), Var("n"))), "acc")
    )
    return TileOp(
        op=node,
        place=Placement(free=(m, n)),
        axes=(m, n, k),
        inputs={"a": Tensor("a", (64, 64), "f32"), "b": Tensor("b", (64, 64), "f32")},
        outputs={"out": Tensor("out", (64, 64), "f32")},
    )


def test_matmul_leaf_set_equals_the_scalar_catalog(monkeypatch):
    """The leaf set the walk emits over a plain f32 matmul equals ``scalar_tile_moves()`` read
    through the stored (site ``TILE`` value, ``WORK``) pair — a missing or extra move is caught
    structurally, without lowering a kernel. Membership, never position. Restricted to the
    serial-fold rows: the per-cell tier also offers its cooperative / ILP K partitions, whose
    ``WORK`` is the reduce band's thread inventory, not a tile move."""
    for var in ("EMMY_TILE", "EMMY_WORK", "EMMY_REDUCE"):
        monkeypatch.delenv(var, raising=False)
    rows = _rows_of(_plain_matmul_term())
    assert rows, "the term enumerated nothing"
    serial = [r for r in rows if not str(family_value(r, "REDUCE") or "")]
    offered = {(str(family_value(r, "TILE") or ""), str(r.get("WORK", ""))) for r in serial}
    assert offered == {_stored(p) for p in scalar_tile_moves()}


def test_f32_computed_a_contraction_offers_a_tiled_scalar_row():
    """Without an f32 MMA atom, computed A must still ride a scalar output tile so one A value can
    be reused across N instead of the per-cell fallback recomputing its entire cone for every output."""
    rows = _rows_of(_computed_a_term())
    assert rows, "the term enumerated nothing"

    def tile_of(row) -> Tile:
        work = Work.parse(str(row.get("WORK", "")))
        reduce = Reduce.parse(str(family_value(row, "REDUCE") or ""), work)
        return resolve_site_tile(str(family_value(row, "TILE") or ""), work, reduce.coop)

    plans = [tile_of(row) for row in rows]
    assert any(plan.is_tiled and not plan.is_warp for plan in plans), "computed A lost every tiled scalar schedule"


def _row_reduce(m: Axis, k: Axis, buffer: str) -> TileOp:
    """A bare row reduce ``acc[m] = Σ_k buffer[m, k]`` as an unmapped ``TileOp``."""
    load = Load(name=f"{buffer}_e", input=buffer, index=(Var(m.name), Var(k.name)))
    fold = fold_from_loop(Loop(axis=k, body=Body((load, Accum(name="acc", value=load.name, op="add", axes=(k.name,))))))
    return TileOp(op=fold, place=Placement(free=(m,)), axes=(m, k))


def _reduce_term() -> TileOp:
    """A bare 4096-wide row reduce over a 64-cell grid."""
    return _row_reduce(Axis("m", 64), Axis("k", 4096), "x")


def test_the_all_off_row_is_always_offered(monkeypatch):
    """The untiled / serial / gmem-direct schedule — every family at its declared OFF — is legal on
    any term the walk can schedule, so it is always a MEMBER of the enumerated set.

    This is the enumeration fact that replaced the old "option-0 is each family's conservative
    default" obligation. Position is no longer anything: nothing may lead a list to steer a compile
    with no evidence, and such a compile taking an arbitrary row is accepted. What must still hold
    is that the all-OFF row exists to be picked, by evidence or by a pin — a term that could not
    spell it would have a hole in its space, not a slow default."""
    monkeypatch.delenv("EMMY_WORK", raising=False)
    for label, tile in {"bare reduce": _reduce_term(), "computed-B contraction": _computed_b_term()}.items():
        rows = _rows_of(tile)
        assert rows, f"{label}: the term enumerated nothing"
        stamped = [complete_kernel_row({k: v for k, v in row.items() if not k.startswith("S_")}) for row in rows]
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
        rows = _rows_of(tile)
        assert rows, f"{label}: the term enumerated nothing"
        for row in rows:
            work = str(row.get("WORK", ""))
            coop = [v for k, v in row.items() if family_of(k) == "REDUCE" and isinstance(v, str) and "coop" in v]
            if not coop:
                continue
            parsed = Work.parse(work or None)
            assert parsed is not None and parsed.kind == "thread", f"{label}: {coop} rides WORK={work!r}, not a thread band"
            assert Reduce.parse(coop[0], parsed).coop == parsed.units[0], f"{label}: {coop} disagrees with WORK={work!r}"
