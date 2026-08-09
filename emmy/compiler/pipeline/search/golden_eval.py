"""Enumerate and rank candidates for program-backed golden records."""

from __future__ import annotations

from collections.abc import Callable

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.features import tile_signature
from emmy.compiler.pipeline.search.prior.fit.rank import dual_rank


def enumerate_graph(graph, ctx: Context, *, family: str = "") -> list[dict]:
    """The planner's candidate enumeration for any ``graph`` — the SAME rows the scheduler's fork
    tree offers a live compile, captured by resolving the graph through ``TILE_PASSES`` with a
    decide that flattens each fork's leaves (each leaf's knob row keyed by the CANONICAL codec
    spelling — bare on a single-primary tree, ``TILE@dd``-style on flash — exactly
    what ``tile_signature`` joins a golden against). ``family`` keeps only rows carrying that knob
    family (``"TILE"`` for a contraction pool); ``""`` keeps every row with a per-node schedule
    knob (a reduce's ``REDUCE`` fork). The one live-fork capture the matmul
    offline fitter and record evaluator share."""
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import Fork, flatten_leaves  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415

    rows: list[dict] = []
    wanted = (family,) if family else ("TILE", "REDUCE", "STAGE")

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
    return rows


def evaluate_record(record, ctx: Context, scorer: Callable[[dict], float] | None = None) -> tuple[dict, int | None, int, int | None]:
    """Rank a generic program-backed record in its current candidate enumeration."""
    from contextlib import nullcontext  # noqa: PLC0415

    from emmy.compiler.pipeline.search.golden import fast_math_knobs  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior import OfflinePrior  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC  # noqa: PLC0415

    gate = F16_MMA_F32_ACC.pinned("1") if fast_math_knobs(record.knobs) else nullcontext()
    with gate:
        rows = enumerate_graph(record.target_program.copy(), ctx)
    if not rows:
        return {}, None, 0, None
    if scorer is None:
        prior = OfflinePrior()
        base = {**ctx.features(), **record.structural_features}

        def scorer(row):
            return -prior.score({**base, **row})

    want = tile_signature(record.knobs) if record.knobs else None
    golden_index = next((i for i, row in enumerate(rows) if tile_signature(row) == want), None) if want else None
    scores = [scorer(row) for row in rows]
    best = max(range(len(rows)), key=scores.__getitem__)
    rank, rank_opt = dual_rank(scores, golden_index) if golden_index is not None else (None, None)
    return rows[best], rank, len(rows), rank_opt
