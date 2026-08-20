"""Enumerate and rank candidates for program-backed golden records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.features import tile_signature
from emmy.compiler.pipeline.search.pool import Candidates
from emmy.compiler.pipeline.search.prior.fit.rank import dual_rank


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
    decide that flattens each fork's leaves (each leaf's knob row keyed by the CANONICAL codec
    spelling — bare on a single-primary tree, ``TILE@dd``-style on flash — exactly
    what ``tile_signature`` joins a golden against). ``family`` keeps only rows carrying that knob
    family (``"TILE"`` for a contraction pool); ``""`` keeps every row with a per-node schedule
    knob (a reduce's ``REDUCE`` fork). The one live-fork capture the matmul
    offline fitter and record evaluator share.

    Returns :class:`~.pool.Candidates` — the rows beside the size of the pools they came from.
    Under ``ctx.pool_sample`` the rows are a DRAW and ``total`` is the full size; with no sample
    the two agree, so a caller that reports both prints today's numbers unchanged."""
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import Fork, flatten_leaves  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415

    rows: list[dict] = []
    wanted = (family,) if family else ("TILE", "REDUCE", "STAGE")
    # The sample's size sink is PER CALL: the enumerator writes each pool's exact size there as it
    # goes (a fork tree has no channel to return it through), so it is cleared before the walk and
    # read straight after.
    sample = ctx.pool_sample if ctx is not None else None
    if sample is not None:
        sample.totals.clear()

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        for leaf in leaves:
            row = dict(getattr(leaf, "knobs", None) or {})
            if any(family_of(k) in wanted for k in row):
                rows.append(row)
        option = fp.options[0]
        while isinstance(option, Fork) and not option.is_leaf:
            option = option.expand()[0]
        return option

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(graph, decide)
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
