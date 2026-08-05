"""Structural-coverage test for the permitted-move catalog (``search/space.py``).

The scheduling emit (``010_recognize`` → ``_schedule``) enumerates the catalog into the tile fork; this
file pins the catalog's **legal product** two ways:

- the catalog function ``scalar_tile_moves()`` equals the hand-computed ``(par × reg)`` grid + per-cell,
  option-0-first and legality-guarded (``par_n·par_m ≤ 1024``), read through the SITE spelling each
  move stores as (its site ``TILE`` value + the ``WORK`` inventory it implies);
- the **leaf set** the scheduler actually emits over a matmul fixture equals that product (keyed
  ``TILE@<k_axis>``) — so a missing / extra move is caught structurally, without lowering a kernel.
"""

from __future__ import annotations

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.schedule import ReducePlan, TilePlan, Workers, plan_workers, resolve_site_tile
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import flatten_leaves
from emmy.compiler.pipeline.knob import axis_of, family_of, family_value
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.space import MAX_BLOCK_THREADS as _MAX_BLOCK_THREADS
from emmy.compiler.pipeline.search.space import scalar_tile_moves

# The hand-computed legal product as explicit literals — per-cell option-0, then the (par × reg) box
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
    assert moves[0] == TilePlan()  # conservative per-cell option-0 leads (cold greedy stays stable)
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


def test_schedule_leaf_set_equals_catalog():
    """The tile scheduler's emitted leaf set over a static f32 matmul equals the catalog's legal
    product (keyed ``FAMILY@<k_axis>``) — the enumeration IS the tile × stage × reduce move product:

    - distinct ``TILE`` values = exactly ``scalar_tile_moves()`` (f32 → no warp moves);
    - the per-cell tile rides serial + the coop/ILP moves (non-output-tiled tier only), no split-K
      (one thread per cell already saturates the 64×64 grid — the occupancy gate);
    - every clean tiled tile rides the RESOLVED stage spellings (gmem-direct + the resolver-deduped
      cp.async / TMA ring depths) × (serial + the divisor-guarded split-K widths); a masked-N tile
      (tile_n overhangs N) declines staging and rides gmem-direct only — staging composes with
      split-K (the split-K option re-resolves against the sliced inner node and ``030_split_reduce`` threads
      the stage onto the partial).
    """
    from emmy.compiler.pipeline.search.space import coop_reduce_moves, raster_moves, splitk_moves

    rows: list[dict] = []

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        for leaf in leaves:
            row = dict(getattr(leaf, "knobs", {}) or {})
            if any("TILE" in family_of(k) for k in row):
                rows.append(row)
        return leaves[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(_matmul_graph(), decide)
    assert rows, "no TILE fork was emitted for the matmul"

    # A row spells its TILE/REDUCE values site-locally and its worker widths once in WORK, so the
    # catalog identity is recovered by resolving each value against the row's inventory — the same
    # rule the scheduler's own ``_materialize`` applies — back into the typed slice the catalog
    # hands out.
    def full_tile(r: dict) -> TilePlan:
        work = Workers.parse(str(r.get("WORK", "")))
        return resolve_site_tile(str(family_value(r, "TILE") or ""), work, str(family_value(r, "REDUCE") or ""))

    def full_reduce(r: dict) -> ReducePlan:
        return ReducePlan.parse(str(family_value(r, "REDUCE") or ""), Workers.parse(str(r.get("WORK", ""))))

    tiles = {full_tile(r) for r in rows}
    assert tiles == set(scalar_tile_moves())
    by_tile: dict[TilePlan, list[dict]] = {}
    for r in rows:
        by_tile.setdefault(full_tile(r), []).append(r)
    percell = by_tile[TilePlan()]
    # The per-cell tier offers serial + every coop/ILP move whose fold fits the reduce extent
    # (the reduce spec's ``coop <= extent and reg <= extent`` gate) — so the wide ``b64``–``b512``
    # folds drop out on this K=64 matmul, exactly as they would on any reduce narrower than the fold.
    # Layout gates the bands too (WS5): this fixture's B is canonical ``B[k, n]``, so the plain
    # coop moves (lanes interleaved along K — uncoalesced on k-major B) are refused and only the
    # transposed band (lanes sweep N) plus the reg-only ILP moves survive.
    legal_coop = {p for p in coop_reduce_moves() if p.coop <= 64 and p.reg <= 64 and (p.coop_transposed or p.coop == 1)}
    assert {full_reduce(r) for r in percell} == {ReducePlan(), *legal_coop}
    assert all(family_value(r, "STAGE") == "" for r in percell), "per-cell has no operand slab to stage (decided-empty)"
    # Every tiled tile is the full (resolved stages) × (serial + split widths) product, the split
    # rows carrying the SAME stage spellings as the unsplit rows (staging composes with split-K).
    n_reduces = 1 + len(splitk_moves())
    for plan, tiled in by_tile.items():
        if not plan.is_tiled:
            continue
        where = plan.spell() or "<per-thread cell>"
        stages = {str(family_value(r, "STAGE")) for r in tiled if not full_reduce(r).stages}
        # A masked-N tile (tile_n overhangs the fixture's N=64) declines cp.async / TMA staging — the
        # B-slab fill would fault a row-crossing copy — so it rides gmem-direct only, mirroring the
        # warp tier's refusal. A clean tile carries the full resolved stage spellings.
        if plan.units_n * plan.reg_n > 64:
            assert stages == {""}, f"{where}: masked-N must decline staging: {stages}"
        else:
            assert {"", "d1/cp"} <= stages, f"{where}: missing the base resolved stages: {stages}"
        splits = {full_reduce(r) for r in tiled if full_reduce(r).stages}
        assert splits == set(splitk_moves()), f"{where}: {splits}"
        split_stages = {str(family_value(r, "STAGE")) for r in tiled if full_reduce(r).stages}
        assert split_stages == stages, f"{where}: split rows must carry the same stage spellings"
        # Every contraction row also spells the launch-order codec (``RASTER``, bare/root-global) —
        # the flat ``""`` and the grouped ``gm8`` siblings, crossing the whole (stage × reduce)
        # product; the fifth factor of the catalog.
        assert {r.get("RASTER") for r in tiled} == set(raster_moves()), f"{where}: RASTER family incomplete"
        assert len(tiled) == len(stages) * n_reduces * len(raster_moves()), f"{where}: {len(tiled)} rows"
    # This matmul is BATCHED (a leading literal batch dim in A's gmem index). The leading dim is
    # tile/K-invariant, so TMA boxes it as an extent-1 dim with the operand's own origin expr
    # (the TMA operand box-rank rule — the flash K/V convention extended to the matmul tiers; the gemma
    # ``[1, seq, K]`` unit-batch views were the motivating decline) and resolves wherever the box
    # fits the 1..256 hardware range — an oversized register tile still declines per-tile.
    all_stages = {str(family_value(r, "STAGE")) for r in rows}
    assert any("tma" in s for s in all_stages), f"batched tile/K-invariant operands must offer TMA somewhere: {all_stages}"


def test_schedule_leaves_key_tile_canonically():
    """Each emitted contraction leaf keys its output tile by the CANONICAL codec spelling
    (phase 3): a single-contraction kernel's shortest unique key is bare ``TILE`` — the exact
    spelling the golden/DB corpus stores, so the stamped row IS the stored row."""
    axes: set[str | None] = set()

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        for leaf in leaves:
            for k in getattr(leaf, "knobs", {}):
                if family_of(k) == "TILE":
                    axes.add(axis_of(k))
        return leaves[0]

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


def test_warp_staged_rows_fit_the_smem_budget():
    """Every enumerated warp row with a non-empty ``STAGE`` fits its depth-1 operand slot in the
    ctx smem budget, and an over-budget tile still rides gmem-direct. The 256×256 ``w4x4/f4x8``
    tile at ``k8`` needs a 128 KiB slot — over any current cap — and the warp stage resolver used
    to floor the depth clamp at 1 instead of declining, so the row sailed through the fork and
    died at materialize (`validate(ctx)`), leaving an un-lowered ``TileOp`` in the tune's terminal
    (issue #327)."""
    ctx = Context.from_target((8, 9))  # the issue's sm_89 cap (101376 B)
    rows: list[dict] = []

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        for leaf in leaves:
            row = dict(getattr(leaf, "knobs", {}) or {})
            if any("TILE" in family_of(k) for k in row):
                rows.append(row)
        return leaves[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(_fp16_matmul_graph(), decide)
    # F1 site grammar: the warp tier is discriminated by the WORK entry; the site TILE value
    # resolves its units against it.
    from emmy.compiler.ir.schedule import Workers

    staged = [r for r in rows if str(r.get("WORK", "")).startswith("w") and family_value(r, "STAGE")]
    assert staged, "no staged warp rows were enumerated"
    for r in staged:
        plan = TilePlan.parse(str(family_value(r, "TILE")), Workers.parse(str(r["WORK"])))
        slot = (plan.tile_m + plan.tile_n) * plan.bk * plan.atom.atom_k * plan.atom.operand_dtype("a").nbytes
        assert slot <= ctx.max_dynamic_smem, f"{family_value(r, 'TILE')} / {family_value(r, 'STAGE')}: slot {slot} over budget"
    # The over-budget tile itself stays enumerable — gmem-direct only (its every staged sibling
    # resolver-declines), split-K rows included.
    big = [r for r in rows if family_value(r, "TILE") == "mma_m16n8k16_f16_f32/f4x8/k8" and r.get("WORK") == "w4x4"]
    assert big, "the 256x256 warp tile dropped out of the enumeration"
    assert all(family_value(r, "STAGE") == "" for r in big)
    assert any(family_value(r, "REDUCE") for r in big), "split-K must still ride the gmem-direct rows"


def _reduce_graph() -> Graph:
    """A bare full-row sum reduce (the ``reduce.2048x2048`` golden shape) — a lifted
    :class:`Fold` with a 2048-cell free grid, well past the coop heuristic's free cap."""
    from emmy.compiler.ir.frontend.ir import MeanOp

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2048, 2048)), node_id="x")
    g.add_node(MeanOp(), ["x"], Tensor("y", (2048, 1)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    return g


def test_bare_reduce_forks_the_coop_catalog():
    """A bare reduce must fork the full legal ``coop_reduce_moves()`` catalog beside serial, even
    when the free grid exceeds the conservative heuristic's cap — the heuristic previously
    collapsed the fork to ONE serial spec (no options offered), so the tune benched a single
    variant and greedy deployed 53× behind the pinned ``b16``/``b32`` reduce goldens (eighth
    golden sweep, finding 3). Option-0 stays the heuristic's conservative pick (cold-greedy
    deploys are unchanged); the catalog rows are what keep the reduce goldens reachable."""
    from emmy.compiler.pipeline.search.space import coop_reduce_moves

    rows: list[dict] = []

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        for leaf in leaves:
            rows.append(dict(getattr(leaf, "knobs", {}) or {}))
        return leaves[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(_reduce_graph(), decide)
    assert rows, "no fork was offered for the bare reduce"
    # A row's REDUCE value sheds the coop width into WORK, so the catalog identity is the
    # (site value, WORK) pair — the 16- and 32-wide folds both spell "coop" but ride distinct
    # t16 / t32 inventories.
    offered = [(str(family_value(r, "REDUCE")), str(r.get("WORK", ""))) for r in rows]
    assert offered[0] == ("", "")  # 2048 free cells over the free-grid cap: the conservative heuristic pick leads

    # This bare reduce meets the transposed band's structural gate (scalar tail, no
    # shared-row stage, static K, 32-divisible free grid), so the FULL catalog is offered —
    # bt/g-composites included. Rows that fail the gate (softmax/rms shapes) drop the band;
    # that arm is covered by the schedule tests, not this catalog assertion.
    def site_of(plan: ReducePlan) -> tuple[str, str]:
        return plan.spell(), (f"t{plan.coop}" if plan.coop > 1 else "")

    assert set(offered) == {("", ""), *(site_of(p) for p in coop_reduce_moves())}, f"catalog rows missing: {offered}"


# --- the TWO-SITE product: what a flat builder structurally could not enumerate ------------------ #
#
# Every term in the live corpus has ONE scheduling site, so the recursion's merge — the key spelling
# at a NON-primary site, the shared inventory both sites resolve against, the per-site row-count
# equation — would otherwise land untested and only be found wrong when the fused cone and the flash
# streaming pair arrive. This fixture is the cheapest term with two sites: a contraction whose B
# channel is a COMPUTED edge (an inline fold), so the parent enumerates tile × stage × reduce and
# the edge enumerates its own reduce partition, under one WORK.


def _two_site_term():
    """A contraction ``sum_k a[m, k] · b_k`` whose B edge is COMPUTED — an inline ``Fold`` over its
    own axis. Two scheduling sites: the contraction (``TILE`` / ``STAGE`` / ``REDUCE``) and the
    edge (``REDUCE`` / ``STAGE``), the edge keyed by its own axis."""
    from emmy.compiler.ir.axis import Axis, AxisRole
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.stmt import Accum, Body, Load, Loop
    from emmy.compiler.ir.tile import Channel, Fold, TileOp
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


def test_two_site_term_merges_both_sites_under_one_inventory():
    """The recursion's merge, asserted on the smallest two-site term:

    - every row spells BOTH sites — the contraction's families at the bare (primary) keys and the
      computed edge's at its own axis-suffixed ones, so no leaf is missing a key another leaf has.
      An addressed key NO row decides is not spelled at all (``_decided``): the edge's ``STAGE@j``
      is absent because a nested fold has no transport of its own to choose, and spelling it empty
      everywhere would fabricate a second node group for the featurizer to pool;
    - both sites resolve against the SAME ``WORK`` entry (one kernel, one inventory), which is what
      the work-first order buys — a coop edge and a tiled parent are simply never combined;
    - the row count is the PER-SITE product, not a flat one.
    """
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.passes.lowering.tile import _schedule as sch

    tile = _two_site_term()
    term = sch._Term(tile, tile.place.on_grid(), Context.from_target((12, 0)))
    rows, keys, owner = sch._enumerate([term])
    assert rows, "the two-site term enumerated nothing"
    assert set(owner.values()) == {term}, "a single-reading term has one owner"

    # Two sites, and the deeper one is keyed by its own axis — the primary keeps the bare spelling
    # the stored corpus uses.
    assert keys == ["TILE", "STAGE", "REDUCE", "REDUCE@j"], keys
    assert all(set(r) >= set(keys) for r in rows), "a leaf is missing a key its siblings spell"

    # One inventory: every row's edge REDUCE resolves against the row's own WORK, and a cooperative
    # edge is only co-representable with a per-cell parent tile.
    from emmy.compiler.ir.schedule import ReducePlan, Workers

    for r in rows:
        work = Workers.parse(r["WORK"] or None)
        edge = ReducePlan.parse(r["REDUCE@j"], work)
        if edge.coop > 1:
            assert work is not None and work.kind == "thread" and work.count == edge.coop, r
            assert r["TILE"] == "", r  # a tiled parent claims the same threads — never both

    # The row count is the product ACROSS sites, per inventory — the equation a flat builder over one
    # node's families cannot state — pruned by the ONE inventory validation, which is a JOINT fact
    # over the pair (a coop edge and a tiled parent claim the same threads) and so cannot factor per
    # site.
    def claim(*values):
        """The pair's joint inventory claim, or ``None`` when the two sites want different
        cooperative widths — ``_merge``'s rule, since a ``REDUCE`` value spells no width and the
        kernel has exactly one ``WORK`` entry to carry it."""
        plans = {i: v["TILE"] for i, v in enumerate(values) if v.get("TILE") is not None}
        coop = 1
        for v in values:
            red = v.get("REDUCE")
            if red is None or red.coop <= 1:
                continue
            if coop > 1 and red.coop != coop:
                return None
            coop = red.coop
        return sch._Row(knobs={}, plans=plans, coop=coop)

    by_work: dict[str, int] = {}
    for r in rows:
        by_work[r["WORK"]] = by_work.get(r["WORK"], 0) + 1
    for work_spell, n in by_work.items():
        work = Workers.parse(work_spell or None)
        parent = sch._contraction_values(term, term.tree[0].site.node, work)
        edge = sch._site_values(term, term.tree[0].children[0].site, work, term.tree[0])
        pairs = [(p, e) for p in parent for e in edge if (c := claim(p, e)) is not None and sch._work_holds(c, work)]
        assert pairs and len(pairs) <= len(parent) * len(edge), f"{work_spell!r}: {len(pairs)} pairs"
        assert n == len(pairs) * len(sch._raster_values(term)), f"{work_spell!r}: {n} != {len(pairs)} pairs x rasters"


def test_two_site_rows_are_distinct_and_materialize_both_sites():
    """The two halves of "enumeration recurses, materialization must too", on the same fixture.

    A row is its SPELLED knob dict, so two candidate combinations that spell identically are one
    row, not two. Since step 7 a ``REDUCE`` value carries no coop width — it lives once in ``WORK``
    — so folding the sites' claims with ``max`` admitted four child widths under one ``t32`` parent
    as four byte-identical rows (180 of them here). The claim is a consistency instead.

    And every site the walk DECIDES must reach the op: the enumeration keys ``REDUCE@j``, so the
    materialized ``TileOp`` must carry a slice there. Stamping the root alone made a nested key a
    knob no kernel realized — the row said ``r2`` and ``op.schedule`` came back empty."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.knob import canonical_row_key
    from emmy.compiler.pipeline.passes.lowering.tile import _schedule as sch

    tile = _two_site_term()
    term = sch._Term(tile, tile.place.on_grid(), Context.from_target((12, 0)))
    rows, _keys, owner = sch._enumerate([term])

    seen = [canonical_row_key(r) for r in rows]
    assert len(seen) == len(set(seen)), f"{len(seen) - len(set(seen))} rows spell identically"

    decided = [r for r in rows if r.get("REDUCE@j")]
    assert decided, "the fixture must decide its nested site on some row"
    for row in decided:
        op = sch._materialize(owner[canonical_row_key(row)], row, "k", {})
        assert "REDUCE@j" in op.schedule, f"row spells REDUCE@j={row['REDUCE@j']!r} but the op carries {sorted(op.schedule)}"
        assert op.schedule["REDUCE@j"].spell() == row["REDUCE@j"], (op.schedule["REDUCE@j"].spell(), row["REDUCE@j"])


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
        return [w.spell() if w is not None else "" for w in sch._inventories([term])]

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


# --- the ordering obligation: each family's FIRST value is its conservative default -------------- #


def _first_row(tile) -> dict:
    """The row a prior-free path deploys: the enumeration's leading row for ``tile``."""
    from emmy.compiler.pipeline.passes.lowering.tile import _schedule as sch

    term = sch._Term(tile, tile.place.on_grid(), Context.from_target((12, 0)))
    rows, _keys, _owner = sch._enumerate([term])
    assert rows, "the term enumerated nothing"
    return rows[0]


def test_option_zero_is_conservative_per_family(monkeypatch):
    """Enumeration produces a SET, so ranking is the prior's job — but POSITION still deploys a
    kernel on three prior-free paths (no prior and no golden, every leaf blocklisted, and the
    validate-retry budget exhausted). The obligation those paths rest on is that each family's
    FIRST value is its conservative default, and it is PER-FAMILY rather than global: the reduce
    tier deliberately leads with its cooperative heuristic pick, not a serial one, because that is
    the historical cold deploy.

    Mechanized here rather than trusted: every family of the leading row stamps its declared OFF
    value, with ``REDUCE`` the one encoded exception — and since step 7 that exception has TWO
    spellings, because the band's WIDTH moved into ``WORK``. So the pair is checked together: an
    option-0 that leads with a cooperative band must spell exactly that band's inventory, never a
    wider one and never a tile."""
    from emmy.compiler.ir.axis import Axis, AxisRole
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.stmt import Accum, Body, Load, Loop
    from emmy.compiler.ir.tile import Placement, TileOp
    from emmy.compiler.pipeline.knob import is_off_value, stamp_schedule_families
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop

    monkeypatch.delenv("EMMY_WORK", raising=False)
    reduce_fold = fold_from_loop(
        Loop(
            axis=Axis("k", 4096),
            body=Body((Load(name="x_e", input="x", index=(Var("m"), Var("k"))), Accum(name="acc", value="x_e", op="add", axes=("k",)))),
            role=AxisRole.PLANAR,
        )
    )
    terms = {
        "bare reduce": TileOp(op=reduce_fold, place=Placement(free=(Axis("m", 64),))),
        "two-site contraction": _two_site_term(),
    }
    for label, tile in terms.items():
        row = _first_row(tile)
        stamped = stamp_schedule_families({k: v for k, v in row.items() if not k.startswith("S_")})
        work = Workers.parse(stamped.get("WORK") or None)
        coop = max((ReducePlan.parse(v, work).coop for k, v in stamped.items() if family_of(k) == "REDUCE" and v), default=1)
        for fam, value in stamped.items():
            if family_of(fam) == "REDUCE":
                continue  # the encoded exception — the reduce tier leads with its cooperative pick
            if family_of(fam) == "WORK" and coop > 1:
                # …and the inventory that carries its width is the SAME exception, not a second one.
                assert value == f"t{coop}", f"{label}: option-0 leads with coop={coop} but spells WORK={value!r}"
                continue
            assert is_off_value(family_of(fam), value), f"{label}: option-0 stamps {fam}={value!r}, not the family's OFF"
