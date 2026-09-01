"""Enumerate and rank candidates for program-backed golden records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.features import tile_signature
from emmy.compiler.pipeline.search.metrics import dual_rank
from emmy.compiler.pipeline.search.pool import Candidates


@dataclass(frozen=True)
class Ranked:
    """Where one record's recorded config landed in its own candidate enumeration.

    ``rank`` / ``rank_optimistic`` are ``None`` when the recorded knobs are not in the enumeration
    at all — a pin or dtype mismatch, which is a real defect class and must stay distinguishable
    from "ranked last". ``pool`` is the size of the enumeration the rank is against, and travels
    with it because a rank alone says nothing."""

    best: dict
    rank: int | None
    pool: int
    rank_optimistic: int | None


def enumerate_graph(graph, ctx: Context, *, family: str = "") -> Candidates:
    """The planner's candidate enumeration for any ``graph`` — the SAME rows the scheduler's fork
    tree offers a live compile, captured by resolving the graph through ``TILE_PASSES`` with a
    decide that flattens each fork's leaves. Every leaf encodes one accepted classic ``Schedule``
    with bare kernel keys and exact ``@n`` / ``@n.e`` sites, which is exactly what
    ``tile_signature`` joins a golden against. ``family`` keeps only rows carrying that knob
    family (``"TILE"`` for a contraction pool); ``""`` keeps every row with a per-node schedule
    knob (a reduce's ``REDUCE`` fork). The one live-fork capture the matmul
    offline fitter and record evaluator share.

    Returns :class:`~.pool.Candidates` — the rows beside the size of the pools they came from.
    Under ``ctx.pool_sample`` the rows are a DRAW and ``total`` is the exact size, and BOTH count
    the same population: distinct schedule-space stamps. Equal problems produce the same stamp,
    so their identical draw and total are collected once — a rank against ``total`` is a rank
    within the space the fit actually ranks. With no sample the rows are every kernel's fork rows
    and ``total`` is ``len(rows)``, so a caller that reports both prints today's numbers unchanged."""
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import iter_leaves, leaf_knobs  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import WORK  # noqa: PLC0415

    rows: list[dict] = []
    wanted = (family,) if family else ("TILE", "REDUCE", "STAGE")
    # The sample's size sink is PER CALL: the enumerator writes each pool's exact size there as it
    # goes (a fork tree has no channel to return it through), so it is cleared before the walk and
    # read straight after.
    sample = ctx.pool_sample if ctx is not None else None
    if sample is not None:
        sample.totals.clear()
    seen_pools: set[str] = set()

    def decide(fp):
        if sample is not None:
            # One contribution per schedule-space stamp, matching the totals sink's keyed dedupe:
            # an equal problem overwrites the same total, and appending its identical draw again
            # would make ``rows`` and ``total`` count different populations.
            opened = set(sample.totals) - seen_pools
            seen_pools.update(opened)
            if not opened:
                return _first(fp.options)
        for leaf in iter_leaves(fp.options):
            row = leaf_knobs(leaf)
            # A schedule row always spells the kernel-global ``WORK``; a structural arm's knob
            # delta (a cut, the cross-CTA split's g-half or the unsplit receipt) never does — the
            # stated row-identity marker (the classic scheduler's leaf boundary).
            if WORK.name not in row:
                continue
            if any(family_of(k) in wanted for k in row):
                rows.append(row)
        return _first(fp.options)

    def _first(options):
        # A pin may empty an early lazy branch while leaving a later sibling live. Walk to the
        # first complete leaf across the whole sibling set, matching the resolver's own traversal;
        # only an entirely empty fork means the schedule is not offered.
        option = next(iter_leaves(options), None)
        if option is None:
            from emmy.compiler.pipeline.pipeline import NO_OPTION  # noqa: PLC0415

            return NO_OPTION
        return option

    terminal, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(graph, decide)
    if sample is None:
        # A fully pinned classic problem can collapse without opening a policy-visible fork. Its
        # complete row still belongs to the enumeration: read it from the realized kernel set so
        # multi-kernel structural targets expose every independently scheduled problem.
        from emmy.compiler.ir.tile.ir import TileOp  # noqa: PLC0415
        from emmy.compiler.pipeline.knob import SCHEDULE_FAMILIES  # noqa: PLC0415

        for node in terminal.nodes.values():
            if not isinstance(node.op, TileOp) or WORK.name not in node.op.knobs:
                continue
            row = {key: value for key, value in node.op.knobs.items() if family_of(key) in SCHEDULE_FAMILIES}
            if any(family_of(key) in wanted for key in row) and row not in rows:
                rows.append(row)
    return Candidates(rows, sum(sample.totals.values()) if sample is not None else len(rows))


def evaluate_record(record, ctx: Context, scorer: Callable[[dict], float] | None = None) -> Ranked:
    """Rank a generic program-backed record in its current candidate enumeration."""
    from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior import OfflinePrior  # noqa: PLC0415

    with pinned_knobs(record.pin_map):
        candidates = enumerate_graph(record.target_program.copy(), ctx)
    rows = candidates.rows
    if not rows:
        return Ranked({}, None, 0, None)
    if scorer is None:
        prior = OfflinePrior()
        base = {**ctx.features(), **record.structural_features}

        def scorer(row):
            return -prior.mean_score({**base, **row})

    want = tile_signature(record.knobs) if record.knobs else None
    golden_index = next((i for i, row in enumerate(rows) if tile_signature(row) == want), None) if want else None
    scores = [scorer(row) for row in rows]
    best = max(range(len(rows)), key=scores.__getitem__)
    rank, rank_opt = dual_rank(scores, golden_index) if golden_index is not None else (None, None)
    return Ranked(rows[best], rank, candidates.total, rank_opt)
