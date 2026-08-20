"""Model-agnostic golden-rank metrics — how a scored candidate pool turns into ranks.

These are shared by every consumer of a score vector (the linear trainer's objective, the CV harness, the
golden evals) and deliberately know nothing about any model class: they take scores, they return ranks. The
tie conventions are the load-bearing part — see :func:`rank_of_golden` vs :func:`dual_rank`.

Each single-index function has a SET counterpart (:func:`best_rank`, :func:`best_dual_rank`) for a pool that
carries several positives. Both take the best over the set, and at one positive both are the single-index
function itself, bit for bit — which is what keeps a one-positive fit byte-identical to the one before the set
existed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def rank_of_golden(scores: np.ndarray, gidx: int) -> int:
    """0-based rank of the golden by descending score. Ties count AGAINST the golden
    (``>=``): greedy deploy breaks score ties by enumeration order, and option-0 is the
    per-cell / gmem-direct row — a tie IS a miss at deploy time. (The old ``>`` convention,
    which counted ties in the golden's favor, let a fit with zero ``D_stage_*`` weights
    report top-1 golden ranks while the deploy pick
    landed on the per-cell row — the 2026-07-02 sweep's 5-15x regressions.)

    This is the FIT OBJECTIVE's rank (all ties count, emission order ignored) — the
    reported metric rank is :func:`dual_rank`, whose pessimistic flavor mirrors the
    deploy tiebreak exactly (only ties *earlier* in emission order count)."""
    return int((scores >= scores[gidx]).sum()) - 1


def dual_rank(scores, gidx: int) -> tuple[int, int]:
    """The golden's 0-based rank by descending score, both tie semantics, from one pass:
    ``(rank, rank_optimistic)``. ``rank`` is tie-PESSIMISTIC — strictly-greater rows plus
    score-ties earlier in emission order, i.e. exactly the rows a greedy argmin deploys
    ahead of the golden. ``rank_optimistic`` counts strictly-greater rows only — fair when
    tied rows are genuine equivalents. The difference is the tie-plateau width at the
    golden's score: a large gap flags a saturated/undecided scorer (the 2026-07 bug where
    the scoring exponential saturated scored "top-1" on the optimistic count while cold
    deploys shipped emission-order picks 12-29x off)."""
    s = np.asarray(scores, dtype=float)
    g = s[gidx]
    optimistic = int((s > g).sum())
    return optimistic + int((s[:gidx] == g).sum()), optimistic


def best_rank(scores: np.ndarray, pinned: Sequence[int]) -> int:
    """The best :func:`rank_of_golden` over a pool's positive rows — the FIT OBJECTIVE's per-group term
    when a pool has several verified-optimum configs.

    Minimum, not mean: deploy ships one config, so any acceptable one ranked first is a win, while a mean
    would spend weights pushing up the fifth-best config, which changes nothing that ships."""
    return min(rank_of_golden(scores, i) for i in pinned)


def best_dual_rank(scores, pinned: Sequence[int]) -> tuple[int, int]:
    """The reported ``(rank, rank_optimistic)`` for a pool with several positives: :func:`dual_rank` of
    whichever positive attains :func:`best_rank`, so the pessimistic count still uses the deploy tiebreak
    against THAT positive's own emission position. Positives arrive ascending, so a tie on ``best_rank``
    resolves to the earliest-emitted one — the row greedy would actually deploy."""
    return dual_rank(scores, min(pinned, key=lambda i: rank_of_golden(scores, i)))


def topk_table(ranks: list[int], ks=(1, 5, 10, 25, 50, 100)) -> str:
    n = len(ranks)
    parts = [f"top{k}={sum(r < k for r in ranks)}/{n}" for k in ks]
    med = sorted(ranks)[n // 2]
    return "  ".join(parts) + f"   median={med}  mean_log2={np.mean([math.log2(r + 1) for r in ranks]):.2f}"
