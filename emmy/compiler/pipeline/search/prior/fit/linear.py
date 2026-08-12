"""The linear trainer — the offline learning-to-rank fit of the :class:`OfflinePrior` weights.

:class:`LinearTrainer` holds the hyperparameters and :meth:`~LinearTrainer.fit` turns a
:class:`~.group.Group` list into a :class:`LinearFit`. The trainer is immutable and its ``fit`` is pure, so a
fit is a function of ``(groups, hyperparameters)`` alone: the same inputs give a byte-identical artifact, one
instance serves every cross-validation fold without copying, and an A/B between two fits measures the fit
inputs rather than run-to-run noise.

This module owns what is specific to the linear model class: the default loss (:func:`mean_log_rank`) and the
raw-space L2 regularizer (:func:`l2_penalty`), the optimizer (:func:`fit_weights` — random search plus
coordinate descent), and the static→dynamic chaining. Anything a different model class would also need lives
elsewhere: the scoring function itself in :mod:`..linear_model`, the dataset representation in :mod:`.group`,
the model-agnostic rank metrics in :mod:`.rank`, and the fold/metrics harness in :mod:`.cv`.

What is optimized is the **deployed** scoring function, not a proxy for it. The descent scores through
:func:`~..linear_model.quality_columns`, the same arithmetic ``OfflinePrior`` ranks with, and the atomic-free
interaction's ``(weight, threshold)`` pair ride the descent as coordinates alongside the feature weights
(:data:`PARAM_NAMES`). Because the optimizer is derivative-free, fitting a threshold costs nothing extra — and
a scoring constant the fit cannot see is a constant the fit optimizes around, which is what the hand-set gates
were doing until 2026-08-05.

Each fit z-scores over its own candidate pool; ``seed_w`` arrives scaled by ``sd_ref`` (``ones`` for a
raw-weight seed, the previous fit's ``sd`` to chain fits).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from emmy.compiler.pipeline.search.prior.fit.group import Group
from emmy.compiler.pipeline.search.prior.fit.rank import rank_of_golden, topk_table
from emmy.compiler.pipeline.search.prior.linear_model import LinearModel, gate_columns, quality_columns

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


def eval_weights(mats, gidx: list[int], gates, w: np.ndarray, params: np.ndarray) -> list[int]:
    """Each pool's golden rank under a candidate weight vector. Scores through
    :func:`~..linear_model.quality_columns` — the same arithmetic ``OfflinePrior`` deploys — so the fit's
    objective is the deployed ranking itself rather than a proxy that omits the interaction term."""
    return [
        rank_of_golden(quality_columns(m, w, g, weight=params[0], threshold=params[1]), gi)
        for m, g, gi in zip(mats, gates, gidx, strict=True)
    ]


def l2_penalty(w: np.ndarray, sd: np.ndarray) -> float:
    """Sum of squared RAW-space weights for a z-space weight vector (raw = w_z / sd, the
    format the artifact stores — see :func:`raw_weights`). Unweighted by any λ: the loss composes
    it as ``objective + l2 * l2_penalty``."""
    return float(np.sum((w / sd) ** 2))


def mean_log_rank(ranks: list[int]) -> float:
    """The default fit loss: mean ``log2(rank+1)`` over the cases, unweighted. Lower is better.

    Rewards pushing every golden up and is dominated by the worst offenders. Cases count ONE each —
    the retired ``1/count(tier)`` weighting gave every tier an equal share of the loss regardless of
    size, so with 396 warp against 13 thread cases a single fp32 golden outweighed 30 fp16 ones (and
    on a single-tier slice one case could carry half the loss). Its comment justified that by "fp16
    warp is only ~7/32 cases" — a ratio that inverted long ago.

    Any ``list[int] -> float`` may replace it via :attr:`LinearTrainer.objective`; the fit records
    which one it ran under, since two fits are only comparable under the same loss."""
    return float(sum(math.log2(r + 1) for r in ranks) / len(ranks))


def fit_weights(
    groups: list[Group], names, sd_ref, *, seed_w, seed_params, rng, samples, l2=DEFAULT_L2, fit_params=True, objective=mean_log_rank
):
    """Random-search + coordinate-descent the deployed scoring function over ``groups``, minimizing
    ``objective + l2 * l2_penalty`` (the ranking loss plus the raw-space L2 — see :data:`DEFAULT_L2`
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
    # Two streaming passes (mean, then squared deviation) and an IN-PLACE scaling, rather than
    # one concatenated copy plus a fresh list: ``Group.matrix`` already returned a private array,
    # and the pools are large enough that holding several copies of the dataset at once decides
    # whether the fit runs at all — after the tile-scheduler rebuild one fp16 golden enumerates
    # ~78k rows and the golden corpus is ~18 GB, so the concatenate + comprehension spelling
    # peaked near 70 GB and died. Same values — the second pass is the population variance
    # ``allf.std(0)`` computed, and identical rows stay identical under an affine map, so the
    # rank ties this objective counts (``>=``) are exactly the ties it counted before.
    n = sum(len(m) for m in mats)
    mu = sum(m.sum(0) for m in mats) / n
    sd = np.sqrt(sum(((m - mu) ** 2).sum(0) for m in mats) / n)
    sd[sd == 0] = 1.0
    # BEFORE the scaling, and as copies: the interaction compares a raw split COUNT against a raw
    # threshold, and the in-place pass below would otherwise both standardize those values and
    # write through the column views.
    gates = [gate_columns(m, names) for m in mats]
    for m in mats:
        m -= mu
        m /= sd
    matsz = mats

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


def _fit_stages(groups, names, *, seed_weights, seed_params, rng, samples, l2, objective):
    """The static→dynamic chaining: a static fit over the non-dynamic groups seeded from the
    ``seed_weights`` raw dict (zeros where a name is absent — an empty dict seeds from zero), then
    the dynamic fit over the dynamic groups seeded from the static result in its z-space (``sd_ref``
    chaining). ``rng`` is consumed sequentially by both stages. The static group list must be
    non-empty (the dynamic stage seeds from it) — callers guard, this does not.

    The scalar params (:data:`PARAM_NAMES`, seeded from ``seed_params``) are fitted by the STATIC
    stage and frozen for the dynamic one: the artifact carries a single params block that both
    weight sets score under, so the dynamic weights must be fit against the pair that will deploy.

    Returns ``(static_raw, dyn_raw, fitted_params, static_ranks, dyn_ranks)``, with the dynamic
    entries ``None`` when the input had no dynamic groups."""
    static_groups = [g for g in groups if not g.dynamic]
    dyn_groups = [g for g in groups if g.dynamic]
    seed_raw = np.array([seed_weights.get(n, 0.0) for n in names])
    seed_p = np.array([seed_params[n] for n in PARAM_NAMES])
    logger.info("== static fit (%d cases) ==", len(static_groups))
    static_w, params, static_ranks, _, static_sd = fit_weights(
        static_groups, names, np.ones(len(names)), seed_w=seed_raw, seed_params=seed_p, rng=rng, samples=samples, l2=l2, objective=objective
    )
    fitted = {n: float(v) for n, v in zip(PARAM_NAMES, params, strict=True)}
    static_raw = raw_weights(names, static_w, static_sd)
    if not dyn_groups:
        return static_raw, None, fitted, static_ranks, None
    logger.info("== dynamic fit (%d cases) ==", len(dyn_groups))
    dyn_w, _, dyn_ranks, _, dyn_sd = fit_weights(
        dyn_groups,
        names,
        static_sd,
        seed_w=static_w,
        seed_params=params,
        rng=rng,
        samples=samples,
        l2=l2,
        fit_params=False,
        objective=objective,
    )
    return static_raw, raw_weights(names, dyn_w, dyn_sd), fitted, static_ranks, dyn_ranks


@dataclass(frozen=True)
class LinearTrainer:
    """The offline-prior trainer: hyperparameters in, a fitted :class:`LinearFit` out.

    Immutable, and :meth:`fit` never touches ``self`` — so ONE instance serves every
    cross-validation fold with no copying, and a fit is a pure function of
    ``(groups, hyperparameters)``. That is what makes a refit reproducible and an A/B between two
    fits a measurement of the fit inputs rather than of run-to-run noise.

    ``init`` is the incumbent model this fit chains from. It always seeds the scalar params (two
    numbers a fit re-derives, not a per-golden memory) and, when ``warm_start``, the feature weights
    too. Fold models set ``warm_start=False``: the incumbent's weights were themselves fit on every
    golden, so seeding a fold from them would leak each held-out golden into the model that is
    supposed to have never seen it. ``init.scale`` carries into the fitted model unchanged — it is
    rank-neutral, so the fit has no opinion on it."""

    feature_names: tuple[str, ...]
    init: LinearModel
    samples: int = 0
    l2: float = DEFAULT_L2
    random_state: int = 0
    warm_start: bool = True
    objective: Callable[[list[int]], float] = mean_log_rank

    def fit(self, groups: list[Group]) -> LinearFit:
        """Fit both weight sets over ``groups``. The RNG is built here from
        :attr:`random_state`, so a fold's fit never depends on how many folds ran before it."""
        names = list(self.feature_names)
        static_raw, dyn_raw, fitted, static_ranks, dyn_ranks = _fit_stages(
            groups,
            names,
            seed_weights=self.init.weights if self.warm_start else {},
            seed_params={
                "atomic_free_weight": self.init.atomic_free_weight,
                "atomic_free_split_threshold": self.init.atomic_free_split_threshold,
            },
            rng=np.random.default_rng(self.random_state),
            samples=self.samples,
            l2=self.l2,
            objective=self.objective,
        )
        model = LinearModel(
            weights=static_raw,
            weights_dynamic=dyn_raw,
            scale=self.init.scale,
            atomic_free_weight=fitted["atomic_free_weight"],
            atomic_free_split_threshold=fitted["atomic_free_split_threshold"],
        )
        return LinearFit(model, static_ranks, dyn_ranks)


@dataclass(frozen=True)
class LinearFit:
    """One :meth:`LinearTrainer.fit` result: the fitted model plus the golden ranks it reached.

    ``model.weights_dynamic`` / :attr:`dyn_ranks` are ``None`` when the input had no dynamic groups.
    The CALLER decides what to do about that — the shipping path substitutes the incumbent's dynamic
    set, a CV fold treats the set as unfittable rather than silently scoring under a stale vector."""

    model: LinearModel
    static_ranks: list[int]
    dyn_ranks: list[int] | None

    def score_rows(self, group: Group) -> np.ndarray | None:
        """The group's per-row quality (higher = predicted faster), scored exactly as the shipped
        prior ranks. ``None`` when the group needs the dynamic set and this fit has none."""
        w = self.model.weights_dynamic if group.dynamic else self.model.weights
        if w is None:
            return None
        # The gate columns must be present whether or not the weight dict names them (a pruned
        # zero weight drops the key), so score over the union.
        names = sorted(set(w) | {"D_finalize_kernel", "D_splitk"})
        return self.model.quality_rows(group.matrix(names), names, dynamic=group.dynamic)

    @property
    def notes(self) -> str:
        """The one-line provenance summary the artifact records — rank tables per weight set plus
        the fitted scalar params."""
        dyn = f"dynamic {topk_table(self.dyn_ranks)}" if self.dyn_ranks is not None else "no dynamic cases"
        params = ", ".join(f"{n}={getattr(self.model, n):g}" for n in sorted(PARAM_NAMES))
        return f"static {topk_table(self.static_ranks)}; {dyn}; params {params}"
