"""Offline prior — a stateless, fit-offline :class:`Prior` over ``features.knob_features``.

This is the *untrained* prior: the cold-start ranking the search uses before any tuning data exists. There is a
SINGLE ranking path — a config is scored by a ``Prior`` (this one cold, ``OnlinePrior`` once trained), composed
behind :class:`~emmy.compiler.pipeline.search.prior.fallback.FallbackPrior`.

``mean_score`` returns a positive latency *proxy* (``exp(-scale · quality)``), **lower is better** — matching
``OnlinePrior``'s polarity. The proxy is not calibrated µs; only its ordering matters (greedy argmin, and the
sibling-relative ``Prior.policy`` PUCT consumes). Its magnitude may span ``e**±700``, so a consumer needing a
bounded quantity derives one — which is why ``policy`` normalizes within a sibling set rather than using the raw
value.

**Two model classes, one adapter.** The scoring itself lives in a model value object: :mod:`.linear_model` (fixed
weights over the ``D_*`` geometry/occupancy features, plus one fitted non-linear interaction and a second weight
set selected on the ``S_ext_n_symbolic_axis`` stamp) or :mod:`.catboost_model` (a CatBoost ranker, which needs
neither — a tree splits on the routing stamp and forms the interaction from its two columns). This class is the
adapter that turns a knob dict into features and satisfies the ``Prior`` contract around whichever one it holds.

The model lives in the repo-checked artifact ``offline_weights.json`` next to this module (override with
``EMMY_OFFLINE_FILE`` / ``emmy eval … --offline-file`` to A/B a candidate fit), written by ``emmy fit
--artifact`` jointly over EVERY kernel regime — fp32-scalar / fp16-warp matmul, cooperative reduce, and pointwise
goldens — so one model ranks them all. Its ``kind`` field names the model class and :func:`_load_artifact`
dispatches on it; per-refit history rides the artifact's ``provenance`` block and the findings reports, not this
docstring. The artifact is version-gated on ``feat_ver`` — its feature names are that featurizer version's, so a
cross-version file is meaningless and loading it is a hard error (refit, don't guess); a *retired* feature inside
a same-version file is merely a dead term.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from pathlib import Path

from emmy import config, storage
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION, knob_features
from emmy.compiler.pipeline.search.prior import catboost_model, linear_model
from emmy.compiler.pipeline.search.prior.base import Prior
from emmy.compiler.pipeline.search.prior.catboost_model import CatBoostModel
from emmy.compiler.pipeline.search.prior.linear_model import LinearModel

_DEFAULT_FILE = Path(__file__).parent / "offline_weights.json"

# The model classes an artifact's ``kind`` field can name, each with the top-level keys its
# ``from_artifact`` reads and the scalar ``params`` keys its ``to_artifact`` emits. Both key sets are
# derived from the writer rather than re-spelled here: what a fit emits and what a deploy load demands
# must not be able to drift, or a new scalar param ships unvalidated and silently defaults.
#
# ``kind`` has always been written; this is where it is finally read. A file naming an unknown kind is a
# hard error, not a fallback to linear — see :func:`_load_artifact`.
_KINDS = {
    "linear": (LinearModel, ("weights", "weights_dynamic", "params"), linear_model.PARAM_ORDER),
    "catboost": (CatBoostModel, ("cols", "model", "params"), catboost_model.PARAM_ORDER),
}

# The linear model's scalar scoring params, named identically in the JSON ``params`` block and the
# ``OfflinePrior.__init__`` kwargs: ``scale`` is rank-neutral (a monotone transform of quality) and carried;
# the two ``atomic_free_*`` params ARE fitted, being the one term the fit cannot express as a linear weight.
_PARAM_KEYS = linear_model.PARAM_ORDER


@functools.lru_cache(maxsize=8)
def _load_artifact(path_str: str) -> dict:
    """Load and validate a weights artifact — hard error on a missing/corrupt file, a ``feat_ver``
    mismatch, an unknown ``kind``, or keys that kind's model class needs. No silent fallback: an A/B
    that quietly reverts to other weights measures nothing, and both shipped artifacts are
    schema-tested."""
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
            f"expected {FEATURIZER_VERSION} — its features are spelled in a different "
            f"featurizer vocabulary. Refit it: emmy fit --artifact"
        )
    kind = obj.get("kind")
    if kind not in _KINDS:
        raise RuntimeError(
            f"offline prior weights artifact {path_str} has kind={kind!r}, expected one of {sorted(_KINDS)} — "
            f"refit it with 'emmy fit --trainer <kind> --artifact'"
        )
    _, top_keys, param_keys = _KINDS[kind]
    missing = [k for k in top_keys if k not in obj]
    missing += [f"params.{k}" for k in param_keys if k not in obj.get("params", {})]
    if missing:
        raise RuntimeError(f"offline prior weights artifact {path_str} (kind={kind}) lacks {missing}")
    return obj


class OfflinePrior(Prior):
    """Fixed ranker over ``knob_features`` — the cold-start prior.

    An adapter, not a model: the scoring is a :class:`LinearModel` or a
    :class:`~emmy.compiler.pipeline.search.prior.catboost_model.CatBoostModel` (whichever the artifact's ``kind``
    names — the same definition the fitter optimizes), and this class adds what ``Prior`` needs around it —
    knob-dict featurization plus the training surface (``fit`` / ``add_rows`` / ``maybe_refit`` / ``to_json``),
    which are no-ops here (it has nothing to learn) so it composes cleanly under :class:`FallbackPrior`.
    Two ways to construct, and they do not mix: pass a ready ``model``, or let it resolve from the weights
    artifact (``config.offline_path()`` override → the repo-checked default). The per-field kwargs are
    LINEAR-ONLY and win over the file field by field; passing one against a non-linear artifact raises, exactly
    as combining them with ``model=`` does — an A/B that quietly measured the unmodified model would measure
    nothing."""

    def __init__(
        self,
        *,
        model=None,
        weights: dict[str, float] | None = None,
        weights_dynamic: dict[str, float] | None = None,
        scale: float | None = None,
        atomic_free_split_threshold: float | None = None,
        atomic_free_weight: float | None = None,
    ) -> None:
        super().__init__()
        # Keyed by LinearModel's own field names, which is what lets the merge below be a ``replace``.
        overrides = {
            "weights": weights,
            "weights_dynamic": weights_dynamic,
            "scale": scale,
            "atomic_free_weight": atomic_free_weight,
            "atomic_free_split_threshold": atomic_free_split_threshold,
        }
        if model is not None and any(v is not None for v in overrides.values()):
            raise ValueError("OfflinePrior takes either a ready model= or per-field overrides, not both")
        self._model = model if model is not None else self._resolve(overrides)

    @staticmethod
    def _resolve(overrides: dict):
        """The model the artifact names, with the linear per-field overrides layered over it. A fully-specified
        override set skips the file read entirely (tests construct a model that way); anything less reads the
        artifact and dispatches on its ``kind``."""
        if all(v is not None for v in overrides.values()):
            return LinearModel(**overrides)
        path = str(config.offline_path() or _DEFAULT_FILE)
        art = _load_artifact(path)
        cls = _KINDS[art["kind"]][0]
        if cls is not LinearModel:
            if any(v is not None for v in overrides.values()):
                raise ValueError(
                    f"the offline weights artifact {path} is kind={art['kind']!r}, which has no per-field "
                    f"overrides — pass a ready model= instead"
                )
            return cls.from_artifact(art)
        # ``from_artifact`` is lenient about the params block; ``_load_artifact`` has already proved every key
        # this kind needs is present, so the two agree here and the field-by-field merge is a plain replace.
        return replace(LinearModel.from_artifact(art), **{k: v for k, v in overrides.items() if v is not None})

    @property
    def model(self):
        """The scoring function this prior ranks with — a :class:`LinearModel` or a ``CatBoostModel``."""
        return self._model

    @property
    def masking_exact(self) -> bool:
        """Whether deleting a feature from a row is exact term removal — the model's own answer (true for the
        linear model, false for a tree), read by the ablation diagnostics to caveat their Δ."""
        return self._model.masking_exact

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

    def mean_score(self, knobs: dict) -> float:
        """Latency proxy (``exp(-scale · quality)``), lower is better. Under the linear model a config the
        weights have no opinion on (no ``D_*`` features — e.g. a non-tiled kernel) scores the neutral ``1.0``,
        so ties fall to enumeration order, and symbolic-axis (masked-tile) kernels rank under the dynamic weight
        set; under the tree model both facts are ordinary splits."""
        return self.mean_score_features(knob_features(knobs))

    def mean_scores(self, knobs_list: list[dict]) -> list[float]:
        """Batched :meth:`mean_score` — featurize the whole candidate set, then ONE scoring pass. The tree model
        has a vectorized predict whose per-call overhead would otherwise be paid once per candidate."""
        return self.mean_scores_features([knob_features(k) for k in knobs_list])

    def quality(self, feats: dict) -> float:
        """The linear model's ranking quantity (higher = predicted faster), before the monotone
        ``exp(-scale··)`` wrapper. This is what ``emmy fit``'s linear cell minimizes golden rank over — the
        fitter scores through the SAME :class:`LinearModel`, so the fitted objective IS the deployed ranking and
        not a proxy for it. Linear-only: a tree's pre-transform score has no additive reading, and its
        equivalent entry point is ``CatBoostModel.quality_rows``."""
        return self._model.quality(feats)

    def mean_score_features(self, feats: dict) -> float:
        """:meth:`mean_score` from an already-featurized row — the entry point the attribution diagnostics use
        to mask individual features. A deleted key carries the model's own absent semantics: the linear model's
        ``0.0`` no-opinion default (exact term removal), the tree's ``NaN`` missing bucket (a re-route, hence
        :attr:`masking_exact`)."""
        return self._model.mean_score_features(feats)

    def mean_scores_features(self, feats_list: list[dict]) -> list[float]:
        """Batched :meth:`mean_score_features` — one scoring pass over the whole set."""
        return self._model.mean_scores_features(feats_list)

    def explain_features(self, feats: dict) -> dict[str, float]:
        """Per-term decomposition of the quality score, summing to it exactly — the linear model's per-weight
        terms or the tree's TreeSHAP values. See each model's ``explain_features`` for its invariant."""
        return self._model.explain_features(feats)
