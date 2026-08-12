"""Offline prior — a stateless, fit-offline linear :class:`Prior` over
``features.knob_features``.

This is the *untrained* prior: the cold-start ranking the search uses before any
tuning data exists. It replaces the old hand-coded matmul heuristic
(``score_matmul_thread`` + the ``_priority_matmul_*`` enumeration sort) — same
features, now expressed as a fixed linear model over the one shared feature dict
``features.knob_features`` produces, so there is a SINGLE ranking path: a config is
scored by a ``Prior`` (this one cold, ``OnlinePrior`` once trained), composed
behind :class:`~emmy.compiler.pipeline.search.prior.fallback.FallbackPrior`.

``score`` returns a positive latency *proxy* (``exp(-scale · wᵀfeatures)``),
**lower is better** — matching ``OnlinePrior``'s polarity. The proxy is not
calibrated µs; only its ordering (greedy argmin / PUCT relative ``P``) matters.
The scoring itself lives in :class:`~emmy.compiler.pipeline.search.prior.linear_model.LinearModel` —
this class is the adapter that turns a knob dict into features and satisfies the
``Prior`` contract around it. The proxy's magnitude may span ``e**±700``, so a
consumer needing a bounded multiplier (the ``FallbackPrior`` tilt) bounds it itself.

The weights live in the repo-checked artifact ``offline_weights.json`` next to
this module (override with ``EMMY_OFFLINE_FILE`` / ``emmy eval … --offline-file``
to A/B a candidate fit), written by ``emmy fit --artifact`` jointly
over EVERY kernel regime — fp32-scalar / fp16-warp matmul, cooperative reduce, and
pointwise goldens — so one model over the shared ``D_*`` features (plus ``MMA_tier`` and
``MMA_acc_bits``) ranks them all. Everything in the score is fitted: the weights, and the one
non-linear term's ``(weight, threshold)`` pair, which ``emmy fit`` searches as ordinary
descent coordinates. Two weight sets, selected at score time on the stamped
``S_ext_n_symbolic_axis`` flag (see ``linear_model``, which owns that routing and why the
stamp must never also carry a weight), and the TMA-conditioned ``D_tma_*`` terms are where
the one model prices Hopper/Blackwell tiles separately. Per-refit history rides the artifact's
``provenance`` block and the findings reports, not this docstring. The artifact is
version-gated on ``feat_ver`` — the weight keys are that featurizer version's
feature names, so a cross-version file is meaningless and loading it is a hard error
(refit, don't guess); a *retired* key inside a same-version file is merely a dead
term (``feats.get(k, 0.0)``).
"""

from __future__ import annotations

import functools
from pathlib import Path

from emmy import config, storage
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION, knob_features
from emmy.compiler.pipeline.search.prior.base import Prior
from emmy.compiler.pipeline.search.prior.linear_model import LinearModel

_DEFAULT_FILE = Path(__file__).parent / "offline_weights.json"

# The artifact's scalar scoring params, named identically in the JSON ``params`` block and the
# ``OfflinePrior.__init__`` kwargs. ``scale`` is rank-neutral (a monotone transform of quality) and
# fixed; the two ``atomic_free_*`` params ARE fitted — they are the one term the fit cannot express as
# a linear weight, so ``emmy fit`` searches them as descent coordinates alongside the weights.
_PARAM_KEYS = (
    "scale",
    "atomic_free_split_threshold",
    "atomic_free_weight",
)


@functools.lru_cache(maxsize=8)
def _load_artifact(path_str: str) -> dict:
    """Load and validate a weights artifact — hard error on a missing/corrupt file
    or a ``feat_ver`` mismatch. No silent fallback: an A/B that quietly reverts to
    other weights measures nothing, and the shipped default is schema-tested."""
    obj = storage.read_json(Path(path_str))
    if not isinstance(obj, dict):
        raise RuntimeError(
            f"offline prior weights artifact missing or unreadable: {path_str} "
            f"(set EMMY_OFFLINE_FILE to a fitted artifact or regenerate the default "
            f"with 'emmy fit --artifact')"
        )
    found = obj.get("feat_ver")
    if not isinstance(found, int) or found != FEATURIZER_VERSION:
        raise RuntimeError(
            f"offline prior weights artifact {path_str} has feat_ver={found!r}, "
            f"expected {FEATURIZER_VERSION} — its weight keys are spelled in a different "
            f"featurizer vocabulary. Refit it: emmy fit --artifact"
        )
    missing = [k for k in ("weights", "weights_dynamic", "params") if k not in obj]
    missing += [f"params.{k}" for k in _PARAM_KEYS if k not in obj.get("params", {})]
    if missing:
        raise RuntimeError(f"offline prior weights artifact {path_str} lacks {missing}")
    return obj


class OfflinePrior(Prior):
    """Fixed linear ranker over ``knob_features`` — the cold-start prior.

    An adapter, not a model: the scoring is a :class:`LinearModel` (the one definition the fitter
    also optimizes), and this class adds what ``Prior`` needs around it — knob-dict featurization
    plus the training surface (``fit`` / ``add_rows`` / ``maybe_refit`` / ``to_json``), which are
    no-ops here (it has nothing to learn) so it composes cleanly under :class:`FallbackPrior`.
    Pass a ready ``model``, or let the weights and scalar scoring params resolve from the weights
    artifact (``config.offline_path()`` override → the repo-checked default); explicit per-field
    kwargs win over the file, field by field."""

    def __init__(
        self,
        *,
        model: LinearModel | None = None,
        weights: dict[str, float] | None = None,
        weights_dynamic: dict[str, float] | None = None,
        scale: float | None = None,
        atomic_free_split_threshold: float | None = None,
        atomic_free_weight: float | None = None,
    ) -> None:
        super().__init__()
        if model is None:
            given = (weights, weights_dynamic, scale, atomic_free_split_threshold, atomic_free_weight)
            art = None if all(v is not None for v in given) else _load_artifact(str(config.offline_path() or _DEFAULT_FILE))
            params = art["params"] if art is not None else {}
            model = LinearModel(
                weights=weights if weights is not None else art["weights"],
                weights_dynamic=weights_dynamic if weights_dynamic is not None else art["weights_dynamic"],
                scale=scale if scale is not None else params["scale"],
                atomic_free_weight=atomic_free_weight if atomic_free_weight is not None else params["atomic_free_weight"],
                atomic_free_split_threshold=(
                    atomic_free_split_threshold if atomic_free_split_threshold is not None else params["atomic_free_split_threshold"]
                ),
            )
        self._model = model

    @property
    def model(self) -> LinearModel:
        """The scoring function this prior ranks with."""
        return self._model

    @property
    def fitted(self) -> bool:
        return True

    def fit(self) -> None:  # nothing to learn
        return None

    def add_rows(self, rows) -> None:  # noqa: ARG002 — stateless, ignores observations
        return None

    def maybe_refit(self, *, force: bool = False) -> bool:  # noqa: ARG002
        return False

    def to_json(self) -> dict | None:  # not persisted
        return None

    def score(self, knobs: dict) -> float:
        """Latency proxy (``exp(-scale · quality)``), lower is better. A config the
        weights have no opinion on (no ``D_*`` features — e.g. a non-tiled kernel)
        scores the neutral ``1.0``, so ties fall to enumeration order. Symbolic-axis
        (masked-tile) kernels rank under the dynamic weight set."""
        return self.mean_score_features(knob_features(knobs))

    def mean_score(self, knobs: dict) -> float:
        return self.score(knobs)

    def quality(self, feats: dict) -> float:
        """The ranking quantity itself (higher = predicted faster), before the monotone
        ``exp(-scale··)`` wrapper. This is what ``emmy fit`` minimizes golden rank over — the
        fitter scores through the SAME :class:`LinearModel`, so the fitted objective IS the
        deployed ranking and not a proxy for it."""
        return self._model.quality(feats)

    def mean_score_features(self, feats: dict) -> float:
        """:meth:`score` from an already-featurized row — the entry point the
        attribution diagnostics use to mask individual features (a deleted key scores
        as its ``0.0`` no-opinion default, which for a linear model is exact term removal)."""
        return self._model.mean_score_features(feats)

    def explain_features(self, feats: dict) -> dict[str, float]:
        """EXACT per-term decomposition of the quality score — see
        :meth:`LinearModel.explain_features` for the summation invariant."""
        return self._model.explain_features(feats)
