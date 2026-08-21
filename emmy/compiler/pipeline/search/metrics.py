"""How a scored candidate pool becomes a number — ranks, rank correlation, and regret.

Pure functions over a score vector and, where the question needs it, the measured latencies beside it.
Nothing here knows a model class, reads a file, or returns a string: a caller decides what to score and
how to print it, and this module decides only what the metric MEANS. Everything that resolves here has one
definition — the rank family, the three calibration paths that each used to call ``spearmanr`` themselves,
and the pick-vs-measured-best ratio that had a copy in every diagnostic that quoted it.

Two vocabularies meet here and they point opposite ways, so every function names which one it takes:

- **rank** questions (:func:`rank_of_golden`, :func:`dual_rank`, and their ``best_`` set forms) take a
  ranking QUALITY vector, higher = predicted faster, and ask where a known-good row landed in it. They
  need no measurements — a golden is a row someone already verified.
- **regret** and **correlation** questions (:func:`topk_regret`, :func:`spearman`) take a ``predicted_cost``
  vector, lower = predicted faster, alongside the measured latencies, and ask what the prediction costs.
  They are only answerable where every candidate was actually benched.

Reading one as the other silently reverses the answer, which is why the two families are spelled apart
rather than sharing one ``scores`` parameter.

``predicted_cost`` is deliberately not called µs. **Only its ORDER is ever read** — the regret family sorts
by it and takes every value it reports from the measurements, and Spearman is a rank correlation — so any
monotone-decreasing-in-goodness quantity is a valid input. That is what makes these metrics answerable for
the offline prior, which is fitted on RANKS and whose output is an ordinal quality with no latency meaning
at all, exactly as it is for the online prior, which regresses µs.

The offline prior's two available spellings — the raw quality (negated) and the deployed
``exp(-scale · quality)`` proxy — therefore order a pool identically, and measurement says the gap between
them is not merely small but unreachable: over all 1377 recorded goldens the shipped artifact's quality
spans 0 to 277, an exp argument of at most 27.7 against the ±700 float-overflow bound, 25x of headroom.
Pass whichever the caller already holds.

Both families charge the model for a tie, but they charge it differently, and the difference is not a
detail. :func:`rank_of_golden` models EMISSION ORDER — the row greedy actually deploys out of a plateau —
so a golden emitted first still wins its plateau. :func:`topk_pick` refuses to model emission order at all
and takes the WORST tied candidate, because it is answering what the model guarantees rather than what one
enumeration happened to produce. On the same tied vector the two can disagree completely: a perfect rank
0 beside the worst possible regret. That is honest, not contradictory — they are different questions.
"""

from __future__ import annotations

import math
import warnings
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
    against THAT positive's own emission position. Positives arrive ascending, so a tie on :func:`best_rank`
    resolves to the earliest-emitted one — the row greedy would actually deploy."""
    return dual_rank(scores, min(pinned, key=lambda i: rank_of_golden(scores, i)))


def topk_pick(predicted_cost: Sequence[float], measured_us: Sequence[float], k: int) -> int | None:
    """The index a search keeps after benching the model's top ``k`` predictions and taking the measured
    best of them — the ONE selection rule this module has.

    At ``k = 1`` there is nothing to bench: the single prediction ships, and this is the plain argmin with
    ties resolved to the WORST measured candidate. That tie rule is the point. The model said the tied
    candidates were equal, so which one ships is settled by emission order — something it neither knows nor
    controls — and crediting it with the lucky outcome would report skill it does not have. The worst tied
    candidate is the guarantee the model actually offers.

    Above ``k = 1`` the same reasoning applies to the frontier boundary: among equal predictions the
    adversary hands you the slowest, so ties sort worse-measured-first and the ``k`` slots fill with them.

    ``None`` when the pool has ``k`` candidates or fewer, and this guard is the whole reason the parameter
    is explicit. A group of 5 asked for its top 10 answers with its entire self — a perfect result for a
    reason that has nothing to do with the model, and indistinguishable in an aggregate from a model that
    genuinely found the optimum. A report must count the excluded groups rather than average them in; at the
    measurement freeze's median group size of 5, ``k = 10`` excludes most of the corpus."""
    n = len(predicted_cost)
    if n <= k:
        return None
    order = sorted(range(n), key=lambda i: (predicted_cost[i], -measured_us[i]))
    return min(order[:k], key=lambda i: measured_us[i])


def topk_regret(predicted_cost: Sequence[float], measured_us: Sequence[float], k: int) -> float | None:
    """What :func:`topk_pick` costs, as a ratio to the pool's measured best: ``1.0`` = the model's frontier
    contains the optimum, ``1.4`` = 40% slower than was achievable.

    A view of :func:`topk_pick`, not a second computation — at ``k = 1`` it is the deploy question (the pick
    ships, so its latency IS the cost), and above it the tuning question (bench the top ``k``, keep the best).

    ``None`` whenever :func:`topk_pick` is, and also when the pool's best measurement is non-positive, which
    is a label that is not a latency rather than a fast one. Note that a FAILED bench is not that case: it
    carries :data:`~.bench_record.FAIL_SENTINEL_US`, a large POSITIVE number, and would divide perfectly
    well into a meaningless ratio. Excluding failures is the caller's status filter, not this function's."""
    i = topk_pick(predicted_cost, measured_us, k)
    if i is None:
        return None
    best = min(measured_us)
    return None if best <= 0 else measured_us[i] / best


def spearman(predicted_cost: Sequence[float], measured_us: Sequence[float]) -> float | None:
    """Rank correlation between the predicted and the measured ordering over one pool. BOTH arguments
    point the same way — lower = faster — so ``+1`` means the model orders the candidates exactly as the
    hardware does and ``0`` means it carries no ordering information at all. Ranks only, so the predicted
    side needs no units (see the module docstring).

    The whole-pool counterpart of :func:`topk_regret`: regret asks what the model's best guesses cost,
    correlation asks whether it understands the pool at all. A model can score well on one and badly on
    the other — a model that nails the top few and shuffles the tail keeps its regret and loses its ρ,
    which is exactly the difference worth seeing.

    ``None`` when ρ is undefined: fewer than two candidates, a constant vector on either side (every
    prediction equal — the feature/label vocabulary collapse the calibration gate exists to catch), or no
    scipy. That last one matters because the deploy gate reads this: a missing optional dependency must
    leave the model UNMEASURED (``None`` passes the gate) rather than crash a tune run. A caller decides
    its own minimum group size; this only reports what is computable."""
    try:
        from scipy.stats import spearmanr  # noqa: PLC0415
    except ImportError:
        return None

    if len(predicted_cost) < 2:
        return None
    with warnings.catch_warnings():
        # A constant vector is a HANDLED case here (it returns ``None`` below), and a report walks hundreds
        # of groups — scipy's warning would be one line of noise per collapsed group, not a signal.
        # Unqualified: naming scipy's ``ConstantInputWarning`` would mean importing it, and it only exists
        # from scipy 1.9 — an older scipy would then take the ImportError path and report every pool as
        # unmeasurable, which reads exactly like scipy being absent and would quietly void the deploy gate.
        warnings.simplefilter("ignore")
        rho = spearmanr(list(predicted_cost), list(measured_us)).statistic
    return None if math.isnan(rho) else float(rho)
