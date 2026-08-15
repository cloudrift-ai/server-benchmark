"""The fitted CatBoost ranker as a value object — the tree-model counterpart of :mod:`.linear_model`, and the
second model class :class:`~emmy.compiler.pipeline.search.prior.offline.OfflinePrior` can rank with.

Same contract as the linear model and the same two access shapes over one arithmetic: a feature-dict path (how a
live candidate is scored) and a packed-matrix path (what ``emmy fit`` trains and evaluates on). What differs is
everything the linear model needed *because* it is additive:

- **No weight sets.** A linear model prices a symbolic-axis (masked-tile) kernel with a second weight vector
  because it cannot form a product; a tree splits on ``S_ext_n_symbolic_axis`` and prices both regimes inside one
  model. The routing feature is therefore an ordinary COLUMN here (:data:`~.linear_model.ROUTING_FEATURES` is held
  out of :class:`~.fit.group.Group`'s matrix, so the trainer re-adds it from ``Group.dynamic``), and a live
  candidate already carries it in its own featurization.
- **No fitted scalar interaction.** The atomic-free split-K gate is a hand-built term in the linear model because
  the deployed quality cannot express it as a weight. A tree forms ``D_finalize_kernel × D_splitk`` from the two
  columns, so nothing here corresponds to it.
- **No exact decomposition by construction.** :meth:`CatBoostModel.explain_features` returns TreeSHAP values,
  which ARE exactly additive to the raw prediction, so the ``Σ(term(a) − term(b))`` preference-gap contract the
  blame diagnostics rest on still holds. Masking a feature is nonetheless no longer exact term removal — see
  :attr:`CatBoostModel.masking_exact`.

**Absent features are ``NaN``**, CatBoost's own missing-value bucket (``nan_mode="Min"``), matching the online
prior. "This knob is not decided / not stamped on this row" is a different fact from a knob that is present and
legitimately zero (``WM=0``, ``STAGE="00"`` → popcount 0.0), and only a tree can act on the difference. It matters
most on the MCTS selection path, which scores *partial* fork prefixes: filling 0.0 there would fabricate a decided
value. The training rows carry the same semantics — see ``Group.matrix``'s ``fill``.

``quality`` (the raw ranker output, HIGHER = predicted faster) is the polarity :meth:`LinearModel.quality` uses, so
both model classes feed the shared score and policy wrappers unchanged.
"""

from __future__ import annotations

import base64
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION

# The artifact's ``params`` block for this model class, in written order — the twin of the linear model's
# ``PARAM_ORDER``. ``offline._load_artifact`` demands exactly the keys the writer emits, so a new scalar param is
# added once, here. ``scale`` is rank-neutral for the greedy argmin (a monotone transform of the quality) but NOT
# for PUCT: it sets how sharply ``Prior.policy`` separates one fork's siblings.
PARAM_ORDER = ("scale",)

# Raw QuerySoftMax outputs sit around O(1), so the exp wrapper needs no shrinking to stay in a sane range. (The
# linear model's ~0.1 exists because its quality is a sum over ~60 unbounded weighted features.)
DEFAULT_SCALE = 1.0

# Absent-feature fill. Spelled once: the deploy-side dict path and the fit-side matrix path must agree, or the
# model is asked with semantics it was never trained under.
ABSENT = np.nan


def to_bytes(model) -> bytes:
    """A fitted CatBoost model as native ``cbm`` bytes. CatBoost has no in-memory to-bytes API, so the format
    round-trips through a tempfile. ONE spelling of that round-trip, so no two callers can drift on it."""
    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
        tmp = f.name
    try:
        model.save_model(tmp, format="cbm")
        with open(tmp, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(tmp)


def from_bytes(blob: bytes, model):
    """Load native ``cbm`` bytes into ``model`` — a fresh estimator of the right class, which the caller supplies
    because the blob does not carry it (the online prior loads a regressor, this one a ranker)."""
    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
        f.write(blob)
        tmp = f.name
    try:
        model.load_model(tmp, format="cbm")
    finally:
        os.unlink(tmp)
    return model


def to_b64(model) -> str:
    """:func:`to_bytes` base64'd — the spelling the ONLINE prior's checkpoint needs, being a single
    self-contained JSON file with nowhere to put a sidecar. The offline artifact stores its bytes BESIDE the
    JSON instead; see :meth:`CatBoostModel.to_artifact`."""
    return base64.b64encode(to_bytes(model)).decode("ascii")


def from_b64(blob: str, model):
    """The read half of :func:`to_b64`."""
    return from_bytes(base64.b64decode(blob), model)


def new_ranker(**params):
    """A ``CatBoostRanker`` carrying the settings every fit and load here shares. Imported lazily so a ``compile``
    that never loads a tree artifact does not pay the catboost import."""
    from catboost import CatBoostRanker  # noqa: PLC0415

    return CatBoostRanker(verbose=False, allow_writing_files=False, **params)


@dataclass(frozen=True)
class CatBoostModel:
    """A fitted CatBoost ranker over ``features.knob_features``, plus the column order it reads and the scalar
    scoring params. Immutable, so a fit result is a value the caller can swap and serialize.

    ``cols`` is the model's feature vocabulary IN ORDER — the booster indexes by position, so this tuple is as
    load-bearing as the weights dict is for the linear model, and it ships in the artifact beside the blob."""

    booster: Any
    cols: tuple[str, ...]
    scale: float = DEFAULT_SCALE

    # Masking a feature is NOT exact term removal for a tree — it re-routes every split that reads the column, and
    # a masked row is out of distribution for a model trained without feature dropout. The ablation diagnostics
    # read this to caveat their Δ. The linear model's twin is True.
    masking_exact = False

    # --- the shared model surface (Prior's own names) ---------------------------------------------------

    def mean_score_features(self, feats: dict) -> float:
        """Latency proxy (``exp(-scale · quality)``), lower is better — the linear model's polarity and wrapper,
        so both model classes plug into the same greedy argmin and the same policy normalization."""
        return self.mean_scores_features([feats])[0]

    def mean_scores_features(self, feats_list: list[dict]) -> list[float]:
        """Batched :meth:`mean_score_features` — ONE ``predict`` over the whole ``N × cols`` matrix.

        This is the path that matters, not an optimization of the single-row one: a greedy deploy flattens a
        kernel's entire candidate set into one scoring call, and per-row prediction over a ~78k-row pool would pay
        CatBoost's per-call overhead 78k times."""
        if not feats_list:
            return []
        # Clip the exp ARGUMENT, never the quality: within ±700 the proxy stays strictly ordered, and beyond it
        # exp would overflow or underflow anyway (the linear model's identical float-safety bound).
        return [math.exp(max(min(-self.scale * q, 700.0), -700.0)) for q in self.quality_rows(self.matrix(feats_list))]

    def explain_features(self, feats: dict) -> dict[str, float]:
        """Per-term decomposition of one row's :meth:`quality_rows`, as TreeSHAP values.

        Satisfies the same exactness contract the linear decomposition does — SHAP values for a tree ensemble sum
        to the prediction, the trailing element CatBoost returns being the base value (carried as
        ``base:expected``) — so a two-row term diff is still the model's exact preference gap. Zero contributions
        are dropped, matching the linear spelling: they are neutral in the total and would only pad the report."""
        from catboost import Pool  # noqa: PLC0415

        shap = self.booster.get_feature_importance(Pool(self.matrix([feats])), type="ShapValues")[0]
        terms = {name: float(v) for name, v in zip(self.cols, shap[:-1], strict=True) if v}
        terms["base:expected"] = float(shap[-1])
        return terms

    # --- the packed-pool path ---------------------------------------------------------------------------

    def matrix(self, feats_list: list[dict]) -> np.ndarray:
        """Feature dicts packed into this model's own column order, absent key = :data:`ABSENT`."""
        return np.array([[f.get(c, ABSENT) for c in self.cols] for f in feats_list], dtype=float)

    def quality_rows(self, mat: np.ndarray) -> np.ndarray:
        """Per-row ranking quality (higher = predicted faster) over a matrix already in :attr:`cols` order — the
        raw ranker output, before the monotone ``exp(-scale·)`` wrapper. This is the quantity ``emmy fit`` ranks
        goldens by and the quantity the deployed score is a transform of, so the fitted objective IS the deployed
        ranking rather than a proxy for it."""
        return np.asarray(self.booster.predict(mat), dtype=float)

    # --- artifact round-trip -----------------------------------------------------------------------------

    @property
    def blob(self) -> bytes:
        """The fitted booster as ``cbm`` bytes — what the caller writes to the sidecar named by
        :meth:`to_artifact`. A property rather than part of the artifact dict so the dict stays pure JSON."""
        return to_bytes(self.booster)

    @classmethod
    def from_artifact(cls, obj: dict, *, base_dir=None) -> CatBoostModel:
        """Construct from an artifact dict: column order, params, and the booster read from the ``model_file``
        sidecar resolved RELATIVE TO ``base_dir`` (the directory the JSON was loaded from, so an artifact and its
        weights move together).

        Deliberately does NOT version-gate: the strict ``feat_ver`` check is deploy policy and lives in
        ``offline._load_artifact``. An inline base64 ``model`` is still accepted — that was the pre-sidecar
        spelling, and run artifacts under ``_tune/`` predate the split."""
        name = obj.get("model_file")
        if name is not None:
            path = Path(name)
            if not path.is_absolute():
                if base_dir is None:
                    raise ValueError(f"artifact names a relative model_file {name!r} but no base_dir was given to resolve it against")
                path = Path(base_dir) / path
            booster = from_bytes(path.read_bytes(), new_ranker())
        else:
            booster = from_b64(obj["model"], new_ranker())
        return cls(booster=booster, cols=tuple(obj["cols"]), scale=float(obj.get("params", {}).get("scale", DEFAULT_SCALE)))

    def to_artifact(self, *, provenance: dict, model_file: str = "weights.cbm") -> dict:
        """This model as a weights artifact dict. Same envelope as the linear model's (``feat_ver`` / ``kind`` /
        ``params`` / ``provenance``, with ``provenance`` caller-supplied whole so the assembly stays pure);
        ``kind`` is what tells the loader which class to rebuild.

        The booster does NOT ride in the JSON. It goes to ``model_file`` beside it — a RELATIVE name, so the pair
        can be moved or copied to a host together — and the caller writes :attr:`blob` there. Base64 inside the
        JSON cost 33% over the raw bytes and turned a reviewable text artifact into one unreadable line; a tree
        has no line-level diff to offer either way, but the JSON's own fields (cols, params, provenance) stay
        legible. Keep the name relative to the artifact: an absolute path would break the moment the pair is
        rsynced to a box with a different scratch directory."""
        return {
            "feat_ver": FEATURIZER_VERSION,
            "kind": "catboost",
            "cols": list(self.cols),
            "model_file": model_file,
            "params": {name: float(getattr(self, name)) for name in PARAM_ORDER},
            "provenance": provenance,
        }
