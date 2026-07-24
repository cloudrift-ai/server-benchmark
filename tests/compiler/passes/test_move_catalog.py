"""Structural-coverage test for the permitted-move catalog (``search/space.py``).

The scheduling emit (``010_recognize`` → ``_schedule``) enumerates the catalog into the tile fork; this
file pins the catalog's **legal product** two ways:

- the catalog function ``scalar_tile_moves()`` equals the hand-computed ``(par × reg)`` grid + per-cell,
  option-0-first and legality-guarded (``par_n·par_m ≤ 1024``);
- the **leaf set** the scheduler actually emits over a matmul fixture equals that product (keyed
  ``TILE@<k_axis>``) — so a missing / extra move is caught structurally, without lowering a kernel.
"""

from __future__ import annotations

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.schedule import ReducePlan, TilePlan
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import flatten_leaves
from emmy.compiler.pipeline.knob import axis_of, family_of, family_value
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.space import MAX_BLOCK_THREADS as _MAX_BLOCK_THREADS
from emmy.compiler.pipeline.search.space import scalar_tile_moves

# The hand-computed legal product as explicit literals — per-cell option-0, then the (par × reg) grid
# spelled through the codec (the default ``f1x1`` register sub-tile suppresses, so ``reg=(1,1)`` is the
# bare ``n<par_n>x<par_m>``). Written out (not recomputed from ``_SCALAR_*``) so a change to the grid,
# the ordering, or the legality filter is caught here explicitly. The per-par register list is the
# square/skewed core plus the golden-informed deep-FM points (f2x6..f2x14, f4x6..f4x26).
_REG_SUFFIXES = [
    "", "/f2x2", "/f4x4", "/f2x4", "/f4x2",
    "/f2x6", "/f2x8", "/f2x14", "/f4x6", "/f4x8", "/f4x10", "/f4x12", "/f4x14", "/f4x26",
]  # fmt: skip
_EXPECTED_MOVES = [""] + [f"n{pn}x{pm}{reg}" for pn, pm in ((16, 8), (16, 16), (32, 8), (32, 16), (64, 16)) for reg in _REG_SUFFIXES]


def test_scalar_tile_moves_equals_hand_product():
    moves = scalar_tile_moves()
    assert moves == _EXPECTED_MOVES
    assert moves[0] == ""  # conservative per-cell option-0 leads (cold greedy stays stable)
    assert len(set(moves)) == len(moves)  # no duplicate candidates
    # Every non-empty move round-trips the codec grammar and stays inside the thread budget.
    for spec in moves[1:]:
        plan = TilePlan.parse(spec)
        assert plan.spell() == spec
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
      split-K (``_splitk_option`` re-resolves against the sliced inner node and ``030_split_reduce`` threads
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
    tiles = {str(family_value(r, "TILE")) for r in rows}
    assert tiles == set(scalar_tile_moves())
    by_tile: dict[str, list[dict]] = {}
    for r in rows:
        by_tile.setdefault(str(family_value(r, "TILE")), []).append(r)
    percell = by_tile[""]
    # The per-cell tier offers serial + every coop/ILP move whose fold fits the reduce extent
    # (``_reduce_specs``'s ``coop <= extent and reg <= extent`` gate) — so the wide ``b64``–``b512``
    # folds drop out on this K=64 matmul, exactly as they would on any reduce narrower than the fold.
    legal_coop = {m for m in coop_reduce_moves() if (p := ReducePlan.parse(m)).coop <= 64 and p.reg <= 64}
    assert {str(family_value(r, "REDUCE")) for r in percell} == {"", *legal_coop}
    assert all(family_value(r, "STAGE") == "" for r in percell), "per-cell has no operand slab to stage (decided-empty)"
    # Every tiled tile is the full (resolved stages) × (serial + split widths) product, the split
    # rows carrying the SAME stage spellings as the unsplit rows (staging composes with split-K).
    # Each DEFERRED-KERNEL split (``g<w>k`` — a partial + finalize pair) additionally mirrors a
    # ``PLACE@fin=fuse`` sibling (the finalize-inline offer, evidence-only), so those widths count
    # twice.
    n_reduces = 1 + len(splitk_moves(warp=False))
    n_fin_mirrors = sum(1 for m in splitk_moves(warp=False) if ReducePlan.parse(m).finalize == "kernel")
    for tile_spec, tiled in by_tile.items():
        if not tile_spec:
            continue
        stages = {str(family_value(r, "STAGE")) for r in tiled if not family_value(r, "REDUCE")}
        # A masked-N tile (tile_n overhangs the fixture's N=64) declines cp.async / TMA staging — the
        # B-slab fill would fault a row-crossing copy — so it rides gmem-direct only, mirroring the
        # warp tier's refusal. A clean tile carries the full resolved stage spellings.
        plan = TilePlan.parse(tile_spec)
        if plan.units_n * plan.reg_n > 64:
            assert stages == {""}, f"{tile_spec}: masked-N must decline staging: {stages}"
        else:
            assert {"", "d1/cp"} <= stages, f"{tile_spec}: missing the base resolved stages: {stages}"
        splits = {str(family_value(r, "REDUCE")) for r in tiled if family_value(r, "REDUCE")}
        assert splits == set(splitk_moves(warp=False)), f"{tile_spec}: {splits}"
        split_stages = {str(family_value(r, "STAGE")) for r in tiled if family_value(r, "REDUCE")}
        assert split_stages == stages, f"{tile_spec}: split rows must carry the same stage spellings"
        # Every contraction row also spells the launch-order codec (``RASTER``, bare/root-global) —
        # the flat ``""`` and the grouped ``gm8`` siblings, crossing the whole (stage × reduce)
        # product; the fifth factor of the catalog.
        assert {r.get("RASTER") for r in tiled} == set(raster_moves()), f"{tile_spec}: RASTER family incomplete"
        assert len(tiled) == len(stages) * (n_reduces + n_fin_mirrors) * len(raster_moves()), f"{tile_spec}: {len(tiled)} rows"
    # This matmul is BATCHED (a leading literal batch dim in A's gmem index). The leading dim is
    # tile/K-invariant, so TMA boxes it as an extent-1 dim with the operand's own origin expr
    # (``_tma_operand_rank_ok`` — the flash K/V convention extended to the matmul tiers; the gemma
    # ``[1, seq, K]`` unit-batch views were the motivating decline) and resolves wherever the box
    # fits the 1..256 hardware range — an oversized register tile still declines per-tile.
    all_stages = {str(family_value(r, "STAGE")) for r in rows}
    assert any("tma" in s for s in all_stages), f"batched tile/K-invariant operands must offer TMA somewhere: {all_stages}"


def test_schedule_leaves_key_tile_by_contraction_axis():
    """Each emitted contraction leaf keys its output tile ``TILE@<k_axis>`` (a single eligible axis),
    so the bare catalog spelling canonicalizes onto the node's contraction axis."""
    axes: set[str | None] = set()

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        for leaf in leaves:
            for k in getattr(leaf, "knobs", {}):
                if family_of(k) == "TILE":
                    axes.add(axis_of(k))
        return leaves[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(_matmul_graph(), decide)
    assert axes and None not in axes  # every TILE key is axis-named (TILE@<k>), never bare
    assert len(axes) == 1  # one contraction → one eligible k-axis


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
    tile at ``k8`` needs a 128 KiB slot — over any current cap — and ``_resolve_warp_stage`` used
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
    staged = [r for r in rows if str(family_value(r, "TILE")).startswith("a:") and family_value(r, "STAGE")]
    assert staged, "no staged warp rows were enumerated"
    for r in staged:
        plan = TilePlan.parse(str(family_value(r, "TILE")))
        slot = (plan.tile_m + plan.tile_n) * plan.bk * plan.atom.atom_k * plan.atom.operand_dtype("a").nbytes
        assert slot <= ctx.max_dynamic_smem, f"{family_value(r, 'TILE')} / {family_value(r, 'STAGE')}: slot {slot} over budget"
    # The over-budget tile itself stays enumerable — gmem-direct only (its every staged sibling
    # resolver-declines), split-K rows included.
    big = [r for r in rows if family_value(r, "TILE") == "a:mma_m16n8k16_f16_f32/w4x4/f4x8/k8"]
    assert big, "the 256x256 warp tile dropped out of the enumeration"
    assert all(family_value(r, "STAGE") == "" for r in big)
    assert any(family_value(r, "REDUCE") for r in big), "split-K must still ride the gmem-direct rows"


def _reduce_graph() -> Graph:
    """A bare full-row sum reduce (the ``reduce.2048x2048`` golden shape) — a lifted
    :class:`Reduction` with a 2048-cell free grid, well past the coop heuristic's free cap."""
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
    reduces = [str(family_value(r, "REDUCE")) for r in rows]
    assert reduces[0] == ""  # 2048 free cells > _FREE_CAP: the conservative heuristic pick leads
    # This bare reduce meets the transposed band's structural gate (scalar tail, no
    # shared-row stage, static K, 32-divisible free grid), so the FULL catalog is offered —
    # bt/g-composites included. Rows that fail the gate (softmax/rms shapes) drop the band;
    # that arm is covered by the schedule tests, not this catalog assertion.
    assert set(reduces) == {"", *coop_reduce_moves()}, f"catalog rows missing: {reduces}"


def test_factor_k_refuses_non_dividing_pin_width():
    """A pinned split width that does not divide K must raise, not silently truncate: the
    enumeration path filters ``k % cta`` upstream, so a non-divisor only arrives via an
    ``EMMY_REDUCE`` pin, and ``_factor_k``'s ``K // w`` kslice would drop the remainder
    columns of the contraction (the scalar tier has no mma-step check to catch it)."""
    import pytest

    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.pipeline.passes.lowering.tile._schedule import _factor_k

    with pytest.raises(ValueError, match="does not divide"):
        _factor_k(Axis(name="k", extent=Dim(1024)), 3)
