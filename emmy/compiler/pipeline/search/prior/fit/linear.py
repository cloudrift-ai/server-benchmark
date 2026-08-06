"""The linear trainer and its fitted model — the offline learning-to-rank fit of the
:class:`OfflinePrior` weights.

This module owns everything specific to the linear model class: the loss (:func:`objective` — mean golden
rank — plus :func:`l2_penalty`, the raw-space L2 regularizer), the optimizer (:func:`fit_weights` —
random search + coordinate descent), the static/dyn
two-stage chaining (:func:`fit_two_stage`), and the fitted model (:class:`TwoStageFit`, whose
:meth:`~TwoStageFit.score_rows` is the single fit-side home of the static-vs-dynamic weight-set split — the
CV harness and any other consumer score through it and never touch weight dicts). The dataset representation
lives in :mod:`.group`, the model-agnostic rank metrics in :mod:`.rank`, and the fold/metrics harness in
:mod:`.cv`; the trainer is importable library code with the one featurization and one artifact format shared
by every caller of ``emmy fit``.

What is optimized is the **deployed** scoring function, not a linear proxy for it: :func:`quality_rows` is the
vector form of :meth:`~..offline.OfflinePrior.quality`, sharing the one
:func:`~..offline.atomic_free_term` definition, and the interaction's ``(weight, threshold)`` pair are descent
coordinates alongside the feature weights (:data:`PARAM_NAMES`). Because the optimizer is derivative-free,
fitting a threshold costs nothing extra — and a scoring constant the fit cannot see is a constant the fit
optimizes around, which is what the hand-set gates were doing until 2026-08-05.

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
from emmy.compiler.pipeline.search.prior.offline import atomic_free_term

logger = logging.getLogger(__name__)


# Default strength of the raw-space L2 penalty in the fit loss. The rank objective has no
# counterpressure on a feature whose golden-pool variance is tiny: any raw magnitude above the
# rank-saturation point scores identically, so the fit's pick there is arbitrary — invisible to
# golden rank, catastrophic at fork scoring, where a prefix that has not decided the knob scores
# the feature 0.0 and an inflated weight hands every decided subtree an unconditional win (the
# D_pow2_threads 686 cold-deploy incident: a TinyLlama serve deployed ~150x off the DRAM floor).
# The penalty must be RAW-space (w_z / sd): the poisoned 686 raw weight was an ordinary O(1)
# z-space weight — plain ridge on w_z cannot see it. Scale: the data-fit term is a mean
# log2(rank+1), O(1); the shipped artifacts' raw sum-of-squares is ~1.5e4 excluding the poisoned
# weight, so 1e-6 prices legit magnitudes at ~0.015 — far below any genuine rank improvement —
# while giving every rank-flat direction a strict descent toward zero (a tie-breaker, not a
# trade-off; ``emmy fit --l2`` overrides).
DEFAULT_L2 = 1e-6


# The scalar scoring params the descent fits alongside the weights, in coordinate order. These are the
# offline prior's non-linear term (:func:`~..offline.atomic_free_term`) — everything else in the deployed
# quality IS a linear weight, so it lives in the weight vector and needs no separate coordinate.
PARAM_NAMES = ("atomic_free_weight", "atomic_free_split_threshold")


def gate_columns(mat: np.ndarray, names) -> tuple[np.ndarray, np.ndarray]:
    """The two feature columns :func:`~..offline.atomic_free_term` reads, defaulted exactly as the
    deployed ``feats.get`` calls do when the featurization omits them (finalize 0.0, split count 1.0)."""
    idx = {n: j for j, n in enumerate(names)}
    n_rows = len(mat)
    fin = mat[:, idx["D_finalize_kernel"]] if "D_finalize_kernel" in idx else np.zeros(n_rows)
    spl = mat[:, idx["D_splitk"]] if "D_splitk" in idx else np.ones(n_rows)
    return fin, spl


def quality_rows(mat: np.ndarray, w: np.ndarray, gates: tuple[np.ndarray, np.ndarray], params: np.ndarray) -> np.ndarray:
    """A pool's per-row deployed quality (higher = predicted faster) — the linear part plus the
    atomic-free interaction, scored through the SAME function ``OfflinePrior`` deploys. The fit's
    objective is this quantity's golden rank, so the fitter optimizes the deployed ranking itself
    rather than a proxy that omits the term."""
    return mat @ w + atomic_free_term(gates[0], gates[1], weight=params[0], threshold=params[1])


def eval_weights(mats, gidx: list[int], gates, w: np.ndarray, params: np.ndarray) -> list[int]:
    return [rank_of_golden(quality_rows(m, w, g, params), gi) for m, g, gi in zip(mats, gates, gidx, strict=True)]


def l2_penalty(w: np.ndarray, sd: np.ndarray) -> float:
    """Sum of squared RAW-space weights for a z-space weight vector (raw = w_z / sd, the
    format the artifact stores — see :func:`raw_weights`). Unweighted by any λ: the loss composes
    it as ``objective + l2 * l2_penalty``."""
    return float(np.sum((w / sd) ** 2))


def objective(ranks: list[int]) -> float:
    # Mean log2(rank+1) over the cases, unweighted: rewards pushing every golden up, dominated
    # by the worst offenders. Lower is better. Cases count ONE each — the retired ``1/count(tier)``
    # weighting gave every tier an equal share of the loss regardless of size, so with 396 warp
    # against 13 thread cases a single fp32 golden outweighed 30 fp16 ones (and on a single-tier
    # slice one case could carry half the loss). Its comment justified that by "fp16 warp is only
    # ~7/32 cases" — a ratio that inverted long ago.
    return float(sum(math.log2(r + 1) for r in ranks) / len(ranks))


def fit_weights(groups: list[Group], names, sd_ref, *, seed_w, seed_params, rng, samples, l2=DEFAULT_L2, fit_params=True):
    """Random-search + coordinate-descent the deployed scoring function over ``groups``, minimizing
    ``objective + l2 * l2_penalty`` (the golden rank plus the raw-space L2 — see :data:`DEFAULT_L2`
    for why the loss carries the penalty).

    The optimized quantity is :func:`quality_rows` — the linear weights AND the atomic-free
    interaction's ``(weight, threshold)`` pair, which ride the descent as :data:`PARAM_NAMES`
    coordinates past the feature block. Fitting them is only possible because the optimizer is
    derivative-free: a threshold has no useful gradient, but a coordinate step over it is ordinary.
    ``fit_params=False`` freezes the pair at ``seed_params`` (the dynamic stage, which inherits the
    static fit's pair — the artifact carries ONE params block for both weight sets, so a second
    independent fit of it would be discarded and its weights left tuned against a value that never
    deploys).

    Each fit z-scores over its own candidate pool; ``seed_w`` arrives scaled by ``sd_ref`` (``ones``
    for a raw-weight seed, the previous fit's ``sd`` to chain fits) and is re-scaled into this pool's
    z-space. The params are NOT z-scored — they are raw quality units, and ``matz @ w_z`` equals the
    raw quality up to a per-pool constant that ranking drops, so the two add consistently. Returns
    ``(best_w, best_params, best_ranks, mu, sd)`` with ``best_w`` in this pool's z-space."""
    mats = [g.matrix(names) for g in groups]
    gidx = [g.pinned_idx for g in groups]

    # Z-score over this fit's candidate pool so weights are comparable across features.
    allf = np.concatenate(mats, axis=0)
    mu, sd = allf.mean(0), allf.std(0)
    sd[sd == 0] = 1.0
    matsz = [(m - mu) / sd for m in mats]
    gates = [gate_columns(m, names) for m in mats]  # raw columns: the gate reads split counts, not z-scores

    def loss(w: np.ndarray, ranks: list[int]) -> float:
        return objective(ranks) + l2 * l2_penalty(w, sd)

    best_w = seed_w * sd / sd_ref  # re-scale the seed into this pool's z-space
    best_p = np.asarray(seed_params, dtype=float)
    best_ranks = eval_weights(matsz, gidx, gates, best_w, best_p)
    best_obj = loss(best_w, best_ranks)
    logger.info("  seed: %s", topk_table(best_ranks))

    for _ in range(samples):
        w = rng.standard_normal(len(names))
        ranks = eval_weights(matsz, gidx, gates, w, best_p)
        ob = loss(w, ranks)
        if ob < best_obj:
            best_obj, best_w, best_ranks = ob, w, ranks

    # Coordinate-descent refine around the best, over the feature weights and then the scalar
    # params. On a rank-flat direction the penalty is the only gradient — the descent walks the
    # plateau toward zero magnitude, which is what heals a poisoned incumbent seed on the next refit.
    n_coord = len(names) + (len(PARAM_NAMES) if fit_params else 0)
    step = 1.0
    for _ in range(8):
        improved = False
        for j in range(n_coord):
            for delta in (step, -step):
                w, p = best_w, best_p
                if j < len(names):
                    w = best_w.copy()
                    w[j] += delta
                else:
                    p = best_p.copy()
                    p[j - len(names)] += delta
                ranks = eval_weights(matsz, gidx, gates, w, p)
                ob = loss(w, ranks)
                if ob < best_obj:
                    best_obj, best_w, best_p, best_ranks, improved = ob, w, p, ranks, True
        if not improved:
            step /= 2

    logger.info("  best: %s", topk_table(best_ranks))
    logger.info("  params: %s", ", ".join(f"{n}={v:g}" for n, v in zip(PARAM_NAMES, best_p, strict=True)))
    for g, r in sorted(zip(groups, best_ranks, strict=True), key=lambda t: -t[1]):
        logger.info("    %-32s [%-6s] rank=%5d", g.name, g.tier, r)
    return best_w, best_p, best_ranks, mu, sd


def raw_weights(names, best_w, sd) -> dict[str, float]:
    # Fold the z-score into the weights so they apply to RAW features directly:
    # score = ((raw-mu)/sd)·w = raw·(w/sd) - const; the const drops out of ranking.
    return {name: float(best_w[i] / sd[i]) for i, name in enumerate(names) if abs(best_w[i] / sd[i]) > 1e-4}


@dataclass
class TwoStageFit:
    """One trainer invocation's fitted weight sets, in raw-feature space
    (:func:`raw_weights` — the format the artifact stores). ``dyn_raw`` / ``dyn_ranks`` are
    ``None`` when the input had no dynamic cases: the CALLER decides the fallback
    (the script carries the incumbent's dynamic set forward; a CV fold treats the
    set as unfittable rather than silently substituting a stale vector)."""

    static_raw: dict[str, float]
    static_ranks: list[int]
    dyn_raw: dict[str, float] | None
    dyn_ranks: list[int] | None
    params: dict[str, float]

    def score_rows(self, group: Group) -> np.ndarray | None:
        """The group's per-row quality (higher = predicted faster) under the raw weight sets
        and this fit's params — scored exactly as the shipped prior ranks, interaction term
        included. The static-vs-dynamic weight-set selection lives HERE and nowhere else on the
        fit side. ``None`` when the group needs the dynamic set and this fit has none (an
        unfittable fold — callers exclude it up front)."""
        w = self.dyn_raw if group.tier == "dyn" else self.static_raw
        if w is None:
            return None
        # The gate columns must be present whether or not the weight dict names them (a pruned
        # zero weight drops the key), so score over the union.
        names = sorted(set(w) | {"D_finalize_kernel", "D_splitk"})
        mat = group.matrix(names)
        vec = np.array([w.get(n, 0.0) for n in names])
        return quality_rows(mat, vec, gate_columns(mat, names), np.array([self.params[n] for n in PARAM_NAMES]))

    def to_artifact(self, *, params: dict, provenance: dict) -> dict:
        """This fit as the ``OfflinePrior`` weights artifact dict. ``params`` supplies the
        NON-fitted scoring params (``scale``); this fit's own :attr:`params` override the
        fitted ones. Both weight sets must be present — a caller shipping a fit with no
        dynamic set substitutes its fallback (e.g. the incumbent's) into ``dyn_raw`` first."""
        assert self.dyn_raw is not None, "no dynamic weight set — substitute a fallback before assembling the artifact"
        return build_artifact(
            weights=self.static_raw, weights_dynamic=self.dyn_raw, params={**params, **self.params}, provenance=provenance
        )


def fit_two_stage(
    groups: list[Group], names, *, seed_weights: dict[str, float], seed_params: dict[str, float], rng, samples: int, l2: float = DEFAULT_L2
) -> TwoStageFit:
    """The incumbent trainer's full chaining as one call: a static fit over the
    non-``dyn`` groups seeded from the ``seed_weights`` raw dict (zeros where a name is
    absent — an empty dict seeds from zero), then the dynamic fit over the ``dyn``
    groups seeded from the static result in its z-space (``sd_ref`` chaining). ``rng``
    is consumed sequentially by both stages, matching the script's draw order. The
    static group list must be non-empty (the dynamic stage seeds from it) — callers
    guard, this function does not.

    The scalar params (:data:`PARAM_NAMES`, seeded from ``seed_params``) are fitted by the STATIC
    stage and frozen for the dynamic one: the artifact carries a single params block that both
    weight sets score under, so the dynamic weights must be fit against the pair that will deploy."""
    static_groups = [g for g in groups if g.tier != "dyn"]
    dyn_groups = [g for g in groups if g.tier == "dyn"]
    seed_raw = np.array([seed_weights.get(n, 0.0) for n in names])
    seed_p = np.array([seed_params[n] for n in PARAM_NAMES])
    logger.info("== static fit (%d cases) ==", len(static_groups))
    static_w, params, static_ranks, _, static_sd = fit_weights(
        static_groups, names, np.ones(len(names)), seed_w=seed_raw, seed_params=seed_p, rng=rng, samples=samples, l2=l2
    )
    fitted = {n: float(v) for n, v in zip(PARAM_NAMES, params, strict=True)}
    static_raw = raw_weights(names, static_w, static_sd)
    if not dyn_groups:
        return TwoStageFit(static_raw, static_ranks, None, None, fitted)
    logger.info("== dynamic fit (%d cases) ==", len(dyn_groups))
    dyn_w, _, dyn_ranks, _, dyn_sd = fit_weights(
        dyn_groups, names, static_sd, seed_w=static_w, seed_params=params, rng=rng, samples=samples, l2=l2, fit_params=False
    )
    return TwoStageFit(static_raw, static_ranks, raw_weights(names, dyn_w, dyn_sd), dyn_ranks, fitted)


def build_artifact(*, weights: dict[str, float], weights_dynamic: dict[str, float], params: dict, provenance: dict) -> dict:
    """Assemble the ``OfflinePrior`` weights artifact dict in its checked-in shape.
    ``params`` is the complete scoring-param block — the caller merges the fit's own fitted
    params over whatever it carries forward (``scale``); ``provenance`` is caller-supplied whole
    (fitted date, script, args, case counts, notes) so the assembly stays a pure, deterministic
    function."""
    return {
        "feat_ver": FEATURIZER_VERSION,
        "kind": "linear",
        "weights": weights,
        "weights_dynamic": weights_dynamic,
        "params": params,
        "provenance": provenance,
    }
