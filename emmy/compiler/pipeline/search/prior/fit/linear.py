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

import numpy as np

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION

logger = logging.getLogger(__name__)


def feature_matrix(feats: list[dict[str, float]], names: list[str]) -> np.ndarray:
    return np.array([[f.get(n, 0.0) for n in names] for f in feats], dtype=float)


def rank_of_golden(scores: np.ndarray, gidx: int) -> int:
    """0-based rank of the golden by descending score. Ties count AGAINST the golden
    (``>=``): greedy deploy breaks score ties by enumeration order, and option-0 is the
    per-cell / gmem-direct row — a tie IS a miss at deploy time. (The old ``>`` tie-optimism
    let a fit with zero ``D_stage_*`` weights report top-1 golden ranks while the deploy pick
    landed on the per-cell row — the 2026-07-02 sweep's 5-15x regressions.)"""
    return int((scores >= scores[gidx]).sum()) - 1


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

    best_w = seed_w * sd / sd_ref  # re-scale the seed into this pool's z-space
    best_ranks = eval_weights(matsz, gidx, best_w)
    best_obj = objective(best_ranks, cw)
    logger.info("  seed: %s", topk_table(best_ranks))

    for _ in range(samples):
        w = rng.standard_normal(len(names))
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
                w[j] += delta
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
