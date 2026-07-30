"""The linear trainer and its fitted model — the offline learning-to-rank fit of the
:class:`OfflinePrior` weights.

This module owns everything specific to the linear model class: the loss (:func:`objective` — tier-weighted
golden rank), the optimizer (:func:`fit_weights` — random search + coordinate descent), the static/dyn
two-stage chaining (:func:`fit_two_stage`), and the fitted model (:class:`TwoStageFit`, whose
:meth:`~TwoStageFit.score_rows` is the single fit-side home of the static-vs-dynamic weight-set split — the
CV harness and any other consumer score through it and never touch weight dicts). The dataset representation
lives in :mod:`.group`, the model-agnostic rank metrics in :mod:`.rank`, and the fold/metrics harness in
:mod:`.cv`; the trainer is importable library code with the one featurization and one artifact format shared
by every caller (``emmy fit`` and the legacy ``scripts/golden_knob_heuristics.py`` wrapper).

The fit consumes :class:`~.group.Group` lists directly. Each fit z-scores over its own candidate pool;
``seed_w`` arrives scaled by ``sd_ref`` (``ones`` for a raw-weight seed, the previous fit's ``sd`` to chain
fits). Deterministic for a given seeded ``rng``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION
from emmy.compiler.pipeline.search.prior.fit.group import Group
from emmy.compiler.pipeline.search.prior.fit.rank import rank_of_golden, topk_table

logger = logging.getLogger(__name__)


def eval_weights(mats: list[np.ndarray], gidx: list[int], w: np.ndarray) -> list[int]:
    return [rank_of_golden(m @ w, gi) for m, gi in zip(mats, gidx, strict=True)]


def objective(ranks: list[int], weights: list[float]) -> float:
    # Weighted mean log2(rank+1): rewards pushing every golden up, dominated by the
    # worst offenders. Per-case ``weights`` balance the tiers (fp16 warp is only
    # ~7/32 cases, so it'd be drowned out unweighted). Lower is better.
    vals = [w * math.log2(r + 1) for r, w in zip(ranks, weights, strict=True)]
    return float(sum(vals) / sum(weights))


def fit_weights(groups: list[Group], names, sd_ref, *, seed_w, rng, samples):
    """Random-search + coordinate-descent one weight vector over ``groups``.
    Each fit z-scores over its own candidate pool; ``seed_w`` arrives scaled by
    ``sd_ref`` (``ones`` for a raw-weight seed, the previous fit's ``sd`` to
    chain fits) and is re-scaled into this pool's z-space. Returns
    ``(best_w, best_ranks, mu, sd)`` in this pool's z-space."""
    mats = [g.matrix(names) for g in groups]
    gidx = [g.pinned_idx for g in groups]
    tier_n = {t: sum(1 for g in groups if g.tier == t) for t in {g.tier for g in groups}}
    cw = [1.0 / tier_n[g.tier] for g in groups]

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
    for g, r in sorted(zip(groups, best_ranks, strict=True), key=lambda t: -t[1]):
        logger.info("    %-32s [%-6s] rank=%5d", g.name, g.tier, r)
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

    def score_rows(self, group: Group) -> np.ndarray | None:
        """The group's per-row linear scores (higher = predicted faster) under the raw
        weight sets — the artifact-spelling scoring, exactly what the shipped prior
        ranks with (away from the interaction gates). The static-vs-dynamic weight-set
        selection lives HERE and nowhere else on the fit side. ``None`` when the group
        needs the dynamic set and this fit has none (an unfittable fold — callers
        exclude it up front)."""
        w = self.dyn_raw if group.tier == "dyn" else self.static_raw
        if w is None:
            return None
        names = sorted(w)
        return group.matrix(names) @ np.array([w[n] for n in names])

    def to_artifact(self, *, params: dict, provenance: dict) -> dict:
        """This fit as the ``OfflinePrior`` weights artifact dict. Both weight sets must
        be present — a caller shipping a fit with no dynamic set substitutes its
        fallback (e.g. the incumbent's) into ``dyn_raw`` first."""
        assert self.dyn_raw is not None, "no dynamic weight set — substitute a fallback before assembling the artifact"
        return build_artifact(weights=self.static_raw, weights_dynamic=self.dyn_raw, params=params, provenance=provenance)


def fit_two_stage(groups: list[Group], names, *, seed_weights: dict[str, float], rng, samples: int) -> TwoStageFit:
    """The incumbent trainer's full chaining as one call: a static fit over the
    non-``dyn`` groups seeded from the ``seed_weights`` raw dict (zeros where a name is
    absent — an empty dict seeds from zero), then the dynamic fit over the ``dyn``
    groups seeded from the static result in its z-space (``sd_ref`` chaining). ``rng``
    is consumed sequentially by both stages, matching the script's draw order. The
    static group list must be non-empty (the dynamic stage seeds from it) — callers
    guard, this function does not."""
    static_groups = [g for g in groups if g.tier != "dyn"]
    dyn_groups = [g for g in groups if g.tier == "dyn"]
    seed_raw = np.array([seed_weights.get(n, 0.0) for n in names])
    logger.info("== static fit (%d cases) ==", len(static_groups))
    static_w, static_ranks, _, static_sd = fit_weights(static_groups, names, np.ones(len(names)), seed_w=seed_raw, rng=rng, samples=samples)
    static_raw = raw_weights(names, static_w, static_sd)
    if not dyn_groups:
        return TwoStageFit(static_raw, static_ranks, None, None)
    logger.info("== dynamic fit (%d cases) ==", len(dyn_groups))
    dyn_w, dyn_ranks, _, dyn_sd = fit_weights(dyn_groups, names, static_sd, seed_w=static_w, rng=rng, samples=samples)
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
