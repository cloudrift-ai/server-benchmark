"""The fitted linear scoring function as a value object — the ONE definition the fitter optimizes and the
deploy path ranks by.

Two access shapes over one arithmetic: :meth:`LinearModel.quality` takes a feature dict (how ``OfflinePrior``
scores a live candidate) and :meth:`LinearModel.quality_rows` takes a packed pool matrix (what ``emmy fit``
descends on — one fp16 golden enumerates ~78k rows, so the per-dict path is not an option there). Before this
module the two were separate transcriptions of the same formula kept in step by a parity test; drift between them
would mean the fitter optimizing something other than what deploys.

The public scoring methods borrow ``Prior``'s own featurized surface — ``mean_score_features`` and
``explain_features``. That is deliberate: ``FallbackPrior`` and the attribution diagnostics already compose priors
on exactly those, so a holder can delegate to any model through them without a second vocabulary. (``Prior`` also
declares a batched ``mean_scores_features``; there is no model-side version because this model has no vectorized
per-dict path — the base's element-wise default over :meth:`~LinearModel.mean_score_features` IS the
implementation. A model that gains one adds the method then.) :meth:`~LinearModel.quality` /
:meth:`~LinearModel.quality_rows` are linear-only — the pre-transform ranking quantity a derivative-free descent
walks; a tree model has no equivalent.

Weight-set routing is ONE fact read from ONE place: the ``S_ext_n_symbolic_axis`` stamp
(:data:`ROUTING_FEATURES`). A symbolic-axis (masked-tile) kernel prices differently from its static counterpart —
boundary-guard tax on small tiles, staged prologues locked out, occupancy over a free-dim product that excludes
the symbolic axis — so it ranks under ``weights_dynamic``. That stamp must never ALSO be a fitted weight: it is
constant across a candidate pool, so a linear term on it adds the same constant to every candidate and cancels
exactly out of the within-pool ranking. The rank objective cannot see it, and whatever value a descent lands on
there is decided by the regularizer and noise. It routes; it is never a coordinate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION

# Features that SELECT a weight set rather than contribute a term. Owned here because routing is the model's
# business; ``fit/group.py`` imports this to hold the columns out of the fitted matrix entirely, so a routing
# feature cannot become a descent coordinate by accident.
ROUTING_FEATURES = ("S_ext_n_symbolic_axis",)

# The artifact's ``params`` block: the complete key set, in the order it is written. ``storage.write_json``
# preserves insertion order. This is also the set ``offline._load_artifact`` requires an artifact to carry, so the
# writer and the deploy-time completeness check cannot drift — add a scalar param here and both follow.
#
# The order matches what the fitter has emitted since the scalar params became fitted coordinates; it is NOT the
# order in the currently shipped ``offline_weights.json``, which predates that, so the next refit rewrites those
# two lines once. Reordering to match the shipped file would instead change what a refit emits.
PARAM_ORDER = ("scale", "atomic_free_weight", "atomic_free_split_threshold")


def atomic_free_term(finalize_kernel, splitk, *, weight: float, threshold: float):
    """The split-K finalize interaction, the one part of the quality that is NOT a linear weight: above
    ``threshold`` splits REWARD the deferred combine kernel, below it PENALIZE so a narrow split keeps the cheap
    atomicAdd fast path. The atomic finalize scores zero either way (``finalize_kernel`` is 0), keeping its
    geometry-driven rank.

    Written elementwise so ONE definition serves both access shapes: the dict path passes python floats, the
    matrix path whole feature columns. That shared definition is what makes the fit's objective the deployed
    score rather than a proxy — a hand-set constant here would be a constant the fit optimizes around."""
    return weight * finalize_kernel * ((splitk >= threshold) * 2.0 - 1.0)


def gate_columns(mat: np.ndarray, names) -> tuple[np.ndarray, np.ndarray]:
    """The two feature columns :func:`atomic_free_term` reads, defaulted exactly as the dict path's
    ``feats.get`` calls do when the featurization omits them (finalize 0.0, split count 1.0).

    Copies, never column views: the fitter z-scores its pools IN PLACE, and these values must stay in raw units —
    the interaction compares a split COUNT against its threshold."""
    idx = {n: j for j, n in enumerate(names)}
    n_rows = len(mat)
    fin = mat[:, idx["D_finalize_kernel"]].copy() if "D_finalize_kernel" in idx else np.zeros(n_rows)
    spl = mat[:, idx["D_splitk"]].copy() if "D_splitk" in idx else np.ones(n_rows)
    return fin, spl


def quality_columns(mat: np.ndarray, w: np.ndarray, gates: tuple[np.ndarray, np.ndarray], *, weight: float, threshold: float):
    """A pool's per-row quality (higher = predicted faster) from already-prepared columns — the linear part plus
    the atomic-free interaction.

    Takes the weight vector and gate columns rather than a :class:`LinearModel` because the fitter calls it
    mid-descent, where the weights are a raw array in the pool's z-space and no model exists yet.
    :meth:`LinearModel.quality_rows` is the same computation entered from a fitted model."""
    return mat @ w + atomic_free_term(gates[0], gates[1], weight=weight, threshold=threshold)


@dataclass(frozen=True)
class LinearModel:
    """A fitted linear ranker over ``features.knob_features`` — two weight sets, the scalar scoring params, and
    nothing else. Immutable and comparable, so a fit result is a value the caller can diff, swap and serialize.

    ``weights_dynamic`` is ``None`` only inside an incomplete fit (a training slice with no symbolic-axis cases).
    A deploy artifact always carries both sets — :meth:`to_artifact` refuses to write a model that does not, and
    the caller substitutes its fallback set first."""

    weights: dict[str, float]
    weights_dynamic: dict[str, float] | None
    # exp() argument scale — keeps the deployed proxy in a finite, sane range. Rank-neutral (a monotone
    # transform of the quality), so it is carried between fits rather than fitted.
    scale: float
    # The atomic-free interaction's fitted pair (see :func:`atomic_free_term`). Both ARE fitted — they are the
    # one term the fit cannot express as a linear weight, so ``emmy fit`` searches them as descent coordinates
    # alongside the weights. A constant the fit cannot see is a constant the fit optimizes around.
    atomic_free_weight: float
    atomic_free_split_threshold: float

    # --- the shared model surface (Prior's own names) ---------------------------------------------------

    def mean_score_features(self, feats: dict) -> float:
        """Latency proxy (``exp(-scale · quality)``), lower is better. A row the weights have no opinion on
        scores the neutral ``1.0``, so ties fall to enumeration order.

        The clip is on the exp ARGUMENT, never the quality: a former ±80 quality clip sat inside the live range
        (at scale 0.1 thousands of good warp tiles score past 80), so the whole good region collapsed onto one
        ``exp(-8)`` plateau and greedy's argmin fell through to emission order. Within ±700 the proxy stays
        strictly ordered; beyond it ``exp`` would overflow or underflow anyway. Consumers needing a BOUNDED value
        (the ``FallbackPrior`` tilt multiplier) clamp on their side."""
        return math.exp(max(min(-self.scale * self.quality(feats), 700.0), -700.0))

    def explain_features(self, feats: dict) -> dict[str, float]:
        """EXACT per-term decomposition of :meth:`quality` (higher = predicted faster): each nonzero linear term
        by its feature name, plus the atomic-free interaction as a ``gate:*`` pseudo-term. Invariant
        (unit-tested): the terms sum to the quality :meth:`mean_score_features` exponentiates, so a two-row term
        diff is the model's exact preference gap. (The float-safety clip is ignored here — it exists for
        finiteness, never inside the live range.)"""
        w_set = self.weight_set(self.is_dynamic_row(feats))
        terms = {k: w * feats[k] for k, w in w_set.items() if feats.get(k, 0.0)}
        if feats.get("D_finalize_kernel", 0.0):
            terms["gate:atomic_free"] = atomic_free_term(
                feats["D_finalize_kernel"],
                feats.get("D_splitk", 1.0),
                weight=self.atomic_free_weight,
                threshold=self.atomic_free_split_threshold,
            )
        return terms

    # --- linear-only: the quantity the fit optimizes -----------------------------------------------------

    def quality(self, feats: dict) -> float:
        """The ranking quantity itself (higher = predicted faster), before the monotone ``exp(-scale·)`` wrapper:
        the linear weights over ``feats`` plus the atomic-free interaction. Routes itself on the row's stamp —
        a live candidate always carries the full featurization."""
        w_set = self.weight_set(self.is_dynamic_row(feats))
        quality = sum(w * feats.get(k, 0.0) for k, w in w_set.items())
        return quality + atomic_free_term(
            feats.get("D_finalize_kernel", 0.0),
            feats.get("D_splitk", 1.0),  # the split-K count (REDUCE@<k>.cta)
            weight=self.atomic_free_weight,
            threshold=self.atomic_free_split_threshold,
        )

    def quality_rows(self, mat: np.ndarray, names, *, dynamic: bool) -> np.ndarray:
        """:meth:`quality` over a whole packed pool — column ``j`` of ``mat`` is feature ``names[j]``.

        The CALLER routes here rather than the row doing it itself: a feature view may legitimately omit the
        stamp column (it is constant within the pool and carries no weight), so the matrix cannot be relied on to
        answer the question. The dataset carries the answer instead."""
        w_set = self.weight_set(dynamic)
        vec = np.array([w_set.get(n, 0.0) for n in names])
        return quality_columns(
            mat, vec, gate_columns(mat, names), weight=self.atomic_free_weight, threshold=self.atomic_free_split_threshold
        )

    @staticmethod
    def is_dynamic_row(feats: dict) -> bool:
        """Whether a featurized row takes the dynamic weight set — the ONE reading of the routing stamp."""
        return any(feats.get(name, 0.0) > 0 for name in ROUTING_FEATURES)

    def weight_set(self, dynamic: bool) -> dict[str, float]:
        """The weight dict a row of this kind scores under — the ONE place the two sets are chosen between.
        Raises when the dynamic set is asked for and this model has none; callers that must tolerate that
        (an unfittable CV fold) check :attr:`weights_dynamic` first."""
        w_set = self.weights_dynamic if dynamic else self.weights
        if w_set is None:
            raise ValueError("this model has no dynamic weight set — it came from a fit with no symbolic-axis cases")
        return w_set

    # --- artifact round-trip -----------------------------------------------------------------------------

    @classmethod
    def from_artifact(cls, obj: dict) -> LinearModel:
        """Construct from an artifact dict. Deliberately does NOT version-gate: the strict ``feat_ver`` check is
        deploy policy and lives in ``offline._load_artifact``, while the fitter's incumbent-seed read must
        tolerate a mismatch — a refit after a featurizer change is exactly the case a refit exists to fix, and a
        stale key simply seeds ``0.0``."""
        params = obj.get("params", {})
        dyn = obj.get("weights_dynamic")
        return cls(
            weights=dict(obj.get("weights", {})),
            weights_dynamic=None if dyn is None else dict(dyn),
            scale=float(params["scale"]),
            atomic_free_weight=float(params.get("atomic_free_weight", 0.0)),
            atomic_free_split_threshold=float(params.get("atomic_free_split_threshold", 0.0)),
        )

    def to_artifact(self, *, provenance: dict) -> dict:
        """This model as the weights artifact dict, in its checked-in shape. ``provenance`` is caller-supplied
        whole (fitted date, script, args, case counts, notes) so the assembly stays pure and deterministic."""
        if self.weights_dynamic is None:
            raise ValueError("no dynamic weight set — substitute a fallback set before assembling the artifact")
        return {
            "feat_ver": FEATURIZER_VERSION,
            "kind": "linear",
            "weights": self.weights,
            "weights_dynamic": self.weights_dynamic,
            "params": {name: float(getattr(self, name)) for name in PARAM_ORDER},
            "provenance": provenance,
        }
