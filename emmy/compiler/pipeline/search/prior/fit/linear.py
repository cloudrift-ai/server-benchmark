"""Offline learning-to-rank fit of the :class:`OfflinePrior` linear weights.

The fit core behind ``scripts/golden_knob_heuristics.py`` — the script keeps the
golden *case building* (reconstructing each golden's candidate enumeration needs
the command layer's snippet tracer, which ``pipeline/`` must not import) and
delegates the model fitting, rank evaluation, and artifact assembly here, so the
trainer is importable library code with the one featurization and one artifact
format shared by every caller.

A **case** is the tuple ``(name, tier, golden_idx, feats)``: one golden's candidate
pool, already featurized (``feats`` is the per-row ``D_*`` (+ ``MMA_tier``) feature
dict list) with the golden's row pinned at ``golden_idx``. ``tier`` is
``"thread"`` / ``"warp"`` / ``"dyn"`` / ``"reduce"`` / ``"pointwise"`` — used only
to balance the per-tier case weights in the rank objective.

The fit itself (:func:`fit_weights`) is random search + coordinate descent over one
linear weight vector, minimizing the tier-weighted mean ``log2(rank+1)`` of the
golden across all cases — deterministic for a given seeded ``rng``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION

logger = logging.getLogger(__name__)

# Declared per-feature |raw weight| bounds — the decision-8 "bound damage" pattern for the fit.
# The rank objective has no counterpressure on a feature whose golden-pool variance is tiny: descent
# inflates it until every off-feature candidate in every pool is suppressed, and the surplus magnitude
# is invisible to the objective but catastrophic at fork scoring, where a partial prefix that has not
# decided the knob scores the feature as 0.0 — a subtree that decided it then beats every undecided
# sibling by exp(w·scale) unconditionally. Measured on D_pow2_threads (2026-07-29, TinyLlama twin DB +
# the 436-case golden set): the golden objective saturates by 112 (static top1 57/436, identical to the
# unbounded 686 on every metric) while cold-shape fork sibling regret poisons above it (8.3x at ≤112,
# 110x at ≥224 — the 10 ms/layer serial fused-norm misdeploy). Bounds are RAW-space |weight| caps,
# enforced on the seed (so a poisoned incumbent heals on the next refit) and on every search step.
WEIGHT_BOUNDS: dict[str, float] = {
    "D_pow2_threads": 112.0,
}


def feature_matrix(feats: list[dict[str, float]], names: list[str]) -> np.ndarray:
    return np.array([[f.get(n, 0.0) for n in names] for f in feats], dtype=float)


def rank_of_golden(scores: np.ndarray, gidx: int) -> int:
    """0-based rank of the golden by descending score. Ties count AGAINST the golden
    (``>=``): greedy deploy breaks score ties by enumeration order, and option-0 is the
    per-cell / gmem-direct row — a tie IS a miss at deploy time. (The old ``>`` tie-optimism
    let a fit with zero ``D_stage_*`` weights report top-1 golden ranks while the deploy pick
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
    golden's score: a large gap flags a saturated/undecided scorer (the 2026-07 exp-squash
    bug scored "top-1" on the optimistic count while cold deploys shipped emission-order
    picks 12-29x off)."""
    s = np.asarray(scores, dtype=float)
    g = s[gidx]
    optimistic = int((s > g).sum())
    return optimistic + int((s[:gidx] == g).sum()), optimistic


def topk_table(ranks: list[int], ks=(1, 5, 10, 25, 50, 100)) -> str:
    n = len(ranks)
    parts = [f"top{k}={sum(r < k for r in ranks)}/{n}" for k in ks]
    med = sorted(ranks)[n // 2]
    return "  ".join(parts) + f"   median={med}  mean_log2={np.mean([math.log2(r + 1) for r in ranks]):.2f}"


def eval_weights(mats: list[np.ndarray], gidx: list[int], w: np.ndarray) -> list[int]:
    return [rank_of_golden(m @ w, gi) for m, gi in zip(mats, gidx, strict=True)]


def objective(ranks: list[int], weights: list[float]) -> float:
    # Weighted mean log2(rank+1): rewards pushing every golden up, dominated by the
    # worst offenders. Per-case ``weights`` balance the tiers (fp16 warp is only
    # ~7/32 cases, so it'd be drowned out unweighted). Lower is better.
    vals = [w * math.log2(r + 1) for r, w in zip(ranks, weights, strict=True)]
    return float(sum(vals) / sum(weights))


def fit_weights(cases, names, sd_ref, *, seed_w, rng, samples):
    """Random-search + coordinate-descent one weight vector over ``cases``.
    Each fit z-scores over its own candidate pool; ``seed_w`` arrives scaled by
    ``sd_ref`` (``ones`` for a raw-weight seed, the previous fit's ``sd`` to
    chain fits) and is re-scaled into this pool's z-space. Returns
    ``(best_w, best_ranks, mu, sd)`` in this pool's z-space."""
    mats = [feature_matrix(feats, names) for _, _, _, feats in cases]
    gidx = [gi for _, _, gi, _ in cases]
    tier_n = {t: sum(1 for _, ct, _, _ in cases if ct == t) for _, t, _, _ in cases}
    cw = [1.0 / tier_n[t] for _, t, _, _ in cases]

    # Z-score over this fit's candidate pool so weights are comparable across features.
    allf = np.concatenate(mats, axis=0)
    mu, sd = allf.mean(0), allf.std(0)
    sd[sd == 0] = 1.0
    matsz = [(m - mu) / sd for m in mats]

    # WEIGHT_BOUNDS are raw-space caps; raw = w_z / sd (see raw_weights), so bound_z = bound_raw · sd.
    bz = np.array([WEIGHT_BOUNDS.get(n, np.inf) for n in names]) * sd

    best_w = np.clip(seed_w * sd / sd_ref, -bz, bz)  # re-scale the seed into this pool's z-space
    best_ranks = eval_weights(matsz, gidx, best_w)
    best_obj = objective(best_ranks, cw)
    logger.info("  seed: %s", topk_table(best_ranks))

    for _ in range(samples):
        w = np.clip(rng.standard_normal(len(names)), -bz, bz)
        ranks = eval_weights(matsz, gidx, w)
        ob = objective(ranks, cw)
        if ob < best_obj:
            best_obj, best_w, best_ranks = ob, w, ranks

    # Coordinate-descent refine around the best.
    step = 1.0
    for _ in range(8):
        improved = False
        for j in range(len(names)):
            for delta in (step, -step):
                w = best_w.copy()
                w[j] = min(max(w[j] + delta, -bz[j]), bz[j])
                if w[j] == best_w[j]:
                    continue
                ranks = eval_weights(matsz, gidx, w)
                ob = objective(ranks, cw)
                if ob < best_obj:
                    best_obj, best_w, best_ranks, improved = ob, w, ranks, True
        if not improved:
            step /= 2

    logger.info("  best: %s", topk_table(best_ranks))
    for (name, tier, _, _), r in sorted(zip(cases, best_ranks, strict=True), key=lambda t: -t[1]):
        logger.info("    %-32s [%-6s] rank=%5d", name, tier, r)
    return best_w, best_ranks, mu, sd


def raw_weights(names, best_w, sd) -> dict[str, float]:
    # Fold the z-score into the weights so they apply to RAW features directly:
    # score = ((raw-mu)/sd)·w = raw·(w/sd) - const; the const drops out of ranking.
    return {name: float(best_w[i] / sd[i]) for i, name in enumerate(names) if abs(best_w[i] / sd[i]) > 1e-4}


@dataclass
class TwoStageFit:
    """One trainer invocation's fitted weight sets, in raw-feature space
    (:func:`raw_weights` — the artifact spelling). ``dyn_raw`` / ``dyn_ranks`` are
    ``None`` when the input had no dynamic cases: the CALLER decides the fallback
    (the script carries the incumbent's dynamic set forward; a CV fold treats the
    set as unfittable rather than silently substituting a stale vector)."""

    static_raw: dict[str, float]
    static_ranks: list[int]
    dyn_raw: dict[str, float] | None
    dyn_ranks: list[int] | None


def fit_two_stage(cases, names, *, seed_weights: dict[str, float], rng, samples: int) -> TwoStageFit:
    """The incumbent trainer's full chaining as one call: a static fit over the
    non-``dyn`` cases seeded from the ``seed_weights`` raw dict (zeros where a name is
    absent — an empty dict seeds from zero), then the dynamic fit over the ``dyn``
    cases seeded from the static result in its z-space (``sd_ref`` chaining). ``rng``
    is consumed sequentially by both stages, matching the script's draw order. The
    static case list must be non-empty (the dynamic stage seeds from it) — callers
    guard, this function does not."""
    static_cases = [c for c in cases if c[1] != "dyn"]
    dyn_cases = [c for c in cases if c[1] == "dyn"]
    seed_raw = np.array([seed_weights.get(n, 0.0) for n in names])
    logger.info("== static fit (%d cases) ==", len(static_cases))
    static_w, static_ranks, _, static_sd = fit_weights(static_cases, names, np.ones(len(names)), seed_w=seed_raw, rng=rng, samples=samples)
    static_raw = raw_weights(names, static_w, static_sd)
    if not dyn_cases:
        return TwoStageFit(static_raw, static_ranks, None, None)
    logger.info("== dynamic fit (%d cases) ==", len(dyn_cases))
    dyn_w, dyn_ranks, _, dyn_sd = fit_weights(dyn_cases, names, static_sd, seed_w=static_w, rng=rng, samples=samples)
    return TwoStageFit(static_raw, static_ranks, raw_weights(names, dyn_w, dyn_sd), dyn_ranks)


def build_artifact(*, weights: dict[str, float], weights_dynamic: dict[str, float], params: dict, provenance: dict) -> dict:
    """Assemble the ``OfflinePrior`` weights artifact dict in its checked-in shape.
    ``params`` carries through from the incumbent unchanged (the fit touches only the
    linear weights); ``provenance`` is caller-supplied whole (fitted date, script,
    args, case counts, notes) so the assembly stays a pure, deterministic function."""
    return {
        "feat_ver": FEATURIZER_VERSION,
        "kind": "linear",
        "weights": weights,
        "weights_dynamic": weights_dynamic,
        "params": params,
        "provenance": provenance,
    }
