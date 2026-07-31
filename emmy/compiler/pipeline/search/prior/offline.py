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
Because ordering is the whole contract, the squash must never saturate over live
qualities: the exp argument is clipped only at the float-safety bound (±700), so
equal qualities — and only equal qualities — score equal. (A former ±80 *quality*
clip sat inside the live range and collapsed every good tile onto one plateau;
greedy then deployed by emission order.) The proxy's magnitude may therefore span
``e**±700`` — a consumer needing a bounded multiplier (the ``FallbackPrior`` tilt)
bounds it itself.

The weights live in the repo-checked artifact ``offline_weights.json`` next to
this module (override with ``EMMY_OFFLINE_FILE`` / ``emmy eval … --offline-file``
to A/B a candidate fit), written by ``emmy fit --artifact`` jointly
over EVERY kernel regime — fp32-scalar / fp16-warp matmul, cooperative reduce, and
pointwise goldens — so one un-gated linear model over the shared ``D_*`` features
(plus ``MMA_tier``) ranks them all. Two weight sets, selected at score time on the
stamped ``S_ext_n_symbolic_axis`` flag: ``weights`` for static shapes,
``weights_dynamic`` for symbolic-axis (masked-tile) kernels — a masked tile prices
differently from its static twin (boundary-guard tax on small tiles, staged
prologues locked out, occupancy over a free-dim product that excludes the symbolic
axis), and the TMA-conditioned ``D_tma_*`` terms are where the one model prices
Hopper/Blackwell tiles separately. Per-refit history rides the artifact's
``provenance`` block and the findings reports, not this docstring. The artifact is
version-gated on ``feat_ver`` — the weight keys are spelled in the featurizer
vocabulary, so a cross-version file is meaningless and loading it is a hard error
(refit, don't guess); a *retired* key inside a same-version file is merely a dead
term (``feats.get(k, 0.0)``).
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

from emmy import config, storage
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION, knob_features
from emmy.compiler.pipeline.search.prior.base import Prior

_DEFAULT_FILE = Path(__file__).parent / "offline_weights.json"

# The artifact's five scalar scoring params, in the spelling shared by the JSON
# ``params`` block and ``OfflinePrior.__init__`` kwargs.
_PARAM_KEYS = (
    "scale",
    "atomic_free_split_threshold",
    "atomic_free_weight",
    "scalar_on_warp_weight",
    "splitk_roundtrip_weight",
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

    Stateless: ``fitted`` is always ``True`` (it has nothing to learn), and the
    training surface (``fit`` / ``add_rows`` / ``maybe_refit`` / ``to_json``) are
    no-ops so it composes cleanly under :class:`FallbackPrior`. Weights and the
    scalar scoring params resolve from the weights artifact
    (``config.offline_path()`` override → the repo-checked default) unless passed
    explicitly; explicit kwargs win per field."""

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        weights_dynamic: dict[str, float] | None = None,
        scale: float | None = None,
        atomic_free_split_threshold: float | None = None,
        atomic_free_weight: float | None = None,
        scalar_on_warp_weight: float | None = None,
        splitk_roundtrip_weight: float | None = None,
    ) -> None:
        super().__init__()
        given = (
            weights,
            weights_dynamic,
            scale,
            atomic_free_split_threshold,
            atomic_free_weight,
            scalar_on_warp_weight,
            splitk_roundtrip_weight,
        )
        art = None if all(v is not None for v in given) else _load_artifact(str(config.offline_path() or _DEFAULT_FILE))
        params = art["params"] if art is not None else {}
        self._w = weights if weights is not None else art["weights"]
        self._w_dyn = weights_dynamic if weights_dynamic is not None else art["weights_dynamic"]
        # exp() argument scale — keeps the proxy in a finite, sane range; does not
        # affect ranking (monotone), only the proxy's magnitude.
        self._scale = scale if scale is not None else params["scale"]
        # Atomic-free split-K preference.
        # A gate, NOT a linear weight (a plain linear weight can't express the
        # "good when split wide, bad when split narrow" interaction). The online
        # OnlinePrior takes over once real atomic-vs-free ``H_opt=3`` rows exist.
        # The shipped weight is 0.0 (term OFF): activating it when ``D_finalize_kernel``
        # came alive (2026-07-07) regressed golden top-50 coverage 13→10 (the 5090's
        # ``g2k`` split-2 golden contradicts the narrow-split penalty's sign). Re-enable
        # only with refit params that pass the golden-rank gate.
        self._atomic_free_split_threshold = (
            atomic_free_split_threshold if atomic_free_split_threshold is not None else params["atomic_free_split_threshold"]
        )
        self._atomic_free_weight = atomic_free_weight if atomic_free_weight is not None else params["atomic_free_weight"]
        # Scalar-on-warp-eligible penalty + split-K workspace round-trip price.
        # Gates like the atomic-free term — no training rows carry the stamps yet,
        # and a plain linear weight can't express "only bad when the alternative exists".
        # ``scalar_on_warp_weight`` must outweigh the scalar tile's accumulated geometry
        # bonuses under BOTH weight sets (the dyn set hands scalar rows ~+30 quality via
        # ``D_bn_ge_bm`` / band features a warp row structurally cannot earn — the qwen3-emb
        # projection deploys landed scalar at 5-20× the -O3 cost of their enumerated mma
        # siblings). ``splitk_roundtrip_weight`` is a mild price (~5 quality at free_prod
        # ≈ 512·1024): the deferred finalize IS the right shape for wide mma splits.
        self._scalar_on_warp_weight = scalar_on_warp_weight if scalar_on_warp_weight is not None else params["scalar_on_warp_weight"]
        self._splitk_roundtrip_weight = (
            splitk_roundtrip_weight if splitk_roundtrip_weight is not None else params["splitk_roundtrip_weight"]
        )

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

    def mean_score_features(self, feats: dict) -> float:
        """:meth:`score` from an already-featurized row — the seam the attribution
        diagnostics mask individual features through (a deleted key scores as its
        ``0.0`` no-opinion default, which for a linear model is exact term removal)."""
        w_set = self._w_dyn if feats.get("S_ext_n_symbolic_axis", 0.0) > 0 else self._w
        quality = sum(w * feats.get(k, 0.0) for k, w in w_set.items())
        # Deferred-kernel split-K finalize gate (local term — see __init__). The
        # cross-CTA finalize is the REDUCE codec's ``c`` letter, featurized as
        # ``D_finalize_kernel`` (1 when the deferred ``c<cta>k`` combine kernel is on).
        # The ``af_on · (±1)`` product is the interaction a plain weight can't express:
        # above the split threshold REWARD the deferred fold (higher quality → lower
        # latency proxy), below it PENALIZE so a narrow split keeps the cheap atomicAdd
        # fast-path. The atomic finalize scores zero either way (af_on = 0), so it keeps
        # its geometry-driven rank.
        af_on = feats.get("D_finalize_kernel", 0.0)
        if af_on:
            splitk = feats.get("D_splitk", 1.0)  # the split-K count (REDUCE@<k>.cta)
            many_splits = splitk >= self._atomic_free_split_threshold
            quality += self._atomic_free_weight * af_on * (1.0 if many_splits else -1.0)
        # Tensor-core preference gates (see __init__): a scalar tile on a warp-eligible
        # contraction eats the roofline penalty; a deferred split-K finalize pays its
        # workspace round-trip. Both features are 0 wherever the stamps don't apply.
        quality -= self._scalar_on_warp_weight * feats.get("D_scalar_on_warp_eligible", 0.0)
        quality -= self._splitk_roundtrip_weight * feats.get("D_splitk_roundtrip", 0.0)
        # Clip the exp ARGUMENT at the float-safety bound, never the quality: the retired
        # ±80 quality clip sat inside the live range (at scale 0.1 thousands of good warp
        # tiles score past 80), so the whole good region collapsed onto one exp(-8)
        # plateau — greedy's argmin then fell through to emission order (the w1x1 gemma
        # misdeploys) while every tied golden "ranked 0". Within ±700 the proxy stays
        # strictly ordered; beyond it exp() would overflow / underflow anyway. Consumers
        # that need a BOUNDED value (the FallbackPrior tilt multiplier) clamp on their side.
        return math.exp(max(min(-self._scale * quality, 700.0), -700.0))

    def explain_features(self, feats: dict) -> dict[str, float]:
        """EXACT per-term decomposition of the quality score (higher = predicted
        faster): each nonzero linear term by its feature name, plus the three
        hardcoded interaction gates as ``gate:*`` pseudo-terms — a blame table that
        omitted the ±40 scalar-on-warp gate would misattribute exactly the misses it
        dominates. Invariant (unit-tested): the terms sum to the same quality
        :meth:`mean_score_features` exponentiates, so a two-row term diff is the
        model's exact preference gap. (The float-safety clip on the exp argument is
        ignored here — it exists for finiteness, never inside the live range.)"""
        w_set = self._w_dyn if feats.get("S_ext_n_symbolic_axis", 0.0) > 0 else self._w
        terms = {k: w * feats[k] for k, w in w_set.items() if feats.get(k, 0.0)}
        af_on = feats.get("D_finalize_kernel", 0.0)
        if af_on:
            many_splits = feats.get("D_splitk", 1.0) >= self._atomic_free_split_threshold
            terms["gate:atomic_free"] = self._atomic_free_weight * af_on * (1.0 if many_splits else -1.0)
        if feats.get("D_scalar_on_warp_eligible", 0.0):
            terms["gate:scalar_on_warp"] = -self._scalar_on_warp_weight * feats["D_scalar_on_warp_eligible"]
        if feats.get("D_splitk_roundtrip", 0.0):
            terms["gate:splitk_roundtrip"] = -self._splitk_roundtrip_weight * feats["D_splitk_roundtrip"]
        return terms
