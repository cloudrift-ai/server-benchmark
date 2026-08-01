"""Model-agnostic golden-rank metrics — how a scored candidate pool turns into ranks.

These are shared by every consumer of a score vector (the linear trainer's objective, the CV harness, the
golden evals) and deliberately know nothing about any model class: they take scores, they return ranks. The
tie conventions are the load-bearing part — see :func:`rank_of_golden` vs :func:`dual_rank`.
"""

from __future__ import annotations

import math

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


def topk_table(ranks: list[int], ks=(1, 5, 10, 25, 50, 100)) -> str:
    n = len(ranks)
    parts = [f"top{k}={sum(r < k for r in ranks)}/{n}" for k in ks]
    med = sorted(ranks)[n // 2]
    return "  ".join(parts) + f"   median={med}  mean_log2={np.mean([math.log2(r + 1) for r in ranks]):.2f}"
