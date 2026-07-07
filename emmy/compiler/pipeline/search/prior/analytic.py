"""Analytic prior — a stateless, hand-weighted :class:`Prior` over
``features.knob_features``.

This is the *untrained* prior: the cold-start ranking the search uses before any
tuning data exists. It replaces the old hand-coded matmul heuristic
(``score_matmul_thread`` + the ``_priority_matmul_*`` enumeration sort) — same
features, now expressed as a fixed linear model over the one shared feature dict
``features.knob_features`` produces, so there is a SINGLE ranking path: a config is
scored by a ``Prior`` (this one cold, ``CatBoostPrior`` once trained), composed
behind :class:`~emmy.compiler.pipeline.search.prior.fallback.FallbackPrior`.

``score`` returns a positive latency *proxy* (``exp(-scale · wᵀfeatures)``),
**lower is better** — matching ``CatBoostPrior``'s polarity. The proxy is not
calibrated µs; only its ordering (greedy argmin / PUCT relative ``P``) matters.
The weights :data:`_W_A` are fit offline by ``scripts/golden_knob_heuristics.py``
jointly over EVERY kernel regime — fp32-scalar / fp16-warp matmul, cooperative
reduce, and pointwise goldens — so one un-gated linear model over the shared
``D_*`` features (plus ``MMA_tier``) ranks them all (the warp tier rides tier-aware
targets in ``_geom_feats`` plus a positive ``MMA_tier`` weight — fp16/bf16 prefer
the tensor-core tile over the scalar one, the warp-first default that used to live
in enumeration order; the reduce signal rides thread-count / occupancy as
cooperative ``BR`` raises the thread count). It replaces the old per-mode
``_priority_*`` enumeration sorts (matmul / reduce / pointwise), which were the
cold ranking before.
"""

from __future__ import annotations

import math

from emmy.compiler.pipeline.search.features import knob_features
from emmy.compiler.pipeline.search.prior.base import Prior

# Linear weights over ``features.knob_features`` (``D_*`` geometry keys + ``MMA_tier``),
# fit offline by ``scripts/golden_knob_heuristics.py`` jointly over ALL kernel
# regimes — fp32-scalar + fp16/bf16-warp matmul, cooperative reduce, and pointwise
# goldens — tier-balanced (each regime weighted equally so the sparse
# reduce/pointwise tiers aren't drowned by the matmul shapes), minimizing the
# goldens' tier-weighted mean ``log2(rank+1)``. Refit 2026-07-02 over the deep-FM-widened
# ``_SCALAR_REG`` grid (the reachability fix): live-5090 golden median rank 4 (top10 22/42),
# all-cards static median 37 — the wider pool inflates raw ranks, so compare medians only
# within one grid. (The reduce/pointwise cases currently enumerate zero rows — the rebuilt
# reduce fork emits nothing through the live-fork capture — so this fit is matmul-only in
# practice; the reduce tier's cold rank signal is an open gap.) Dominant terms: occupancy
# (``D_ctas_ge_sm``/``D_near_waves`` — keep #CTAs ≈ 2 waves over the SMs), the
# ``D_bm_band`` thread-tile band, the tier-split warp BK target (``D_w_near_bk`` —
# BK≈2 on the TMA tile), the positive ``MMA_tier`` (fp16/bf16 prefer the warp
# tensor-core tile — the fp16 cases enumerate BOTH tiers, so the fit must rank the
# warp golden over the scalar candidates), and the reduce signal rides
# ``D_threads``/occupancy (cooperative ``BR`` raises thread count toward the
# target). One un-gated linear model serves every regime, so some band weights
# compromise across regimes (e.g. ``D_bn_band`` is mildly negative — matmul wants
# the band but reduce wants BN=1; the fit trades it for occupancy).
_W_A: dict[str, float] = {
    "D_pow2_threads": 56.06460471525204,
    "D_splitk_le2": 18.435656861012102,
    "D_ctas_ge_sm": -9.973442761046039,
    "D_wspec_warps": -9.057401908053343,
    "D_bm_band": 7.893496726828649,
    "D_bn_band": -7.823384305652918,
    "MMA_tier": 7.172491368387892,
    "D_stage_async": -6.214925214371601,
    "D_bn_ge_bm": 6.175249346262891,
    "D_stage_ring": 5.193701747343408,
    "D_splitk_excess": -4.918184039295852,
    "D_near_waves": 4.779627471036479,
    "D_w_l2_bk": 3.8318001959254397,
    "D_l2_bm": -3.4096523889221046,
    "D_tilen_clean": 3.304726391558603,
    "D_near_kchunks": -2.4803958401849546,
    "D_square": 2.4682813427691133,
    "D_bk_ge32": -2.250612760379462,
    "D_stage_tma": 2.038360258227803,
    "D_near_tilen": -1.872229368479013,
    "D_near_area": 1.661659507348094,
    "D_near_threads": 1.6102345635910087,
    "D_l2_reuse": 1.279785081847164,
    "D_l2_bn": -1.1315248840332646,
    "D_log2_area": -1.038544031822043,
    "D_l2_bk": 0.9757523358723075,
    "D_l2_threads": -0.9542014237345692,
    "D_splitk": -0.947829756669734,
    "D_neg_masked_k": -0.8646029028646691,
    "D_neg_masked_m": -0.8646029028646691,
    "D_neg_masked_n": -0.8646029028646691,
    "D_l2_cells_occ": 0.6315974578405096,
    "D_stage_reg_depth": 0.5176041698192284,
    "D_w_near_bk": -0.31416170403853766,
    "D_near_intensity": -0.31230584089622293,
    "D_log2_waves": -0.3096053721652958,
    "D_aspect": -0.26074058469642597,
    "D_log2_ctas": -0.19597136958382297,
    "D_near_cells": -0.17771621148992126,
    "D_cells_cap": -0.14767192811087732,
    "D_cells": -0.09744937254034974,
    "D_tile_n": -0.014204261399731513,
    "D_stage_depth": 0.00521186749996049,
    "D_reuse": -0.004804011945045964,
    "D_tile_m": 0.003385545472191824,
    "D_threads": -0.0009447333009831508,
}


# Masked-tier (symbolic-axis) weights — fit by the same script over the dynamic
# (``.dynM``) goldens only. A masked-tile kernel prices differently from its
# static twin: the boundary guard taxes small tiles, the staged prologues the
# static weights reward are locked out on symbolic rows, and the occupancy terms
# see a free-dim product that EXCLUDES the symbolic axis (the 992 stamp), so the
# static weights systematically under-size masked tiles (``BM 8/16``,
# ``SPLITK 1/2`` — the dynM seed report's finding 4). Selected at score time on
# the stamped ``S_ext_n_symbolic_axis`` flag.
_W_A_DYN: dict[str, float] = {
    "D_pow2_threads": 42.714971542355855,
    "D_ctas_ge_sm": -28.043989776686107,
    "D_wspec_warps": -4.700049554908328,
    "D_stage_tma": 4.310486931929542,
    "D_w_near_bk": -3.9853062126276715,
    "D_stage_ring": 3.8465338086843315,
    "D_tilen_clean": 3.4047764061634527,
    "D_l2_reuse": 3.14113185926559,
    "D_bm_band": 3.0497898271941617,
    "D_bn_band": -2.92853447908977,
    "D_log2_ctas": -2.410572010507699,
    "D_splitk_le2": 2.192469464704354,
    "D_bn_ge_bm": 2.1731098324488607,
    "D_l2_bm": -1.9983136234006313,
    "D_splitk_excess": 1.6870439608853967,
    "D_stage_reg_depth": -1.4904329902507725,
    "D_l2_bn": -1.4404137279304239,
    "D_neg_masked_n": 1.2915904848884612,
    "D_bk_ge32": -0.8965483150597096,
    "D_l2_threads": -0.8402838211441096,
    "D_finalize_kernel": 0.829960031801419,
    "D_w_l2_bk": -0.8189885344311161,
    "D_l2_cells_occ": 0.8105008956036689,
    "D_stage_async": 0.6859062649995931,
    "D_near_intensity": -0.5882351447562264,
    "D_neg_masked_m": 0.5582125267295052,
    "D_neg_masked_k": -0.4296618098730851,
    "D_near_area": 0.3686371039197252,
    "D_near_threads": -0.3185998731002451,
    "D_square": 0.31788924224270787,
    "D_aspect": -0.26974344055814103,
    "D_near_waves": -0.26508450064247524,
    "D_near_kchunks": 0.23820227696760266,
    "D_near_tilen": 0.22443802238715357,
    "D_cells": -0.16199478709740808,
    "D_cells_cap": -0.15015342859180322,
    "D_splitk": 0.14913734988550814,
    "D_stage_depth": -0.11947358017861708,
    "MMA_tier": -0.10340824749158893,
    "D_log2_area": -0.05594271212432451,
    "D_l2_bk": 0.03966053994125637,
    "D_reuse": 0.02983300600443265,
    "D_tile_n": -0.02354169053544002,
    "D_near_cells": 0.013724606743399325,
    "D_log2_waves": -0.0032583000160762575,
    "D_threads": -0.003179968274213939,
    "D_tile_m": -0.00010695420800540532,
}


class AnalyticPrior(Prior):
    """Fixed linear ranker over ``knob_features`` — the cold-start prior.

    Stateless: ``fitted`` is always ``True`` (it has nothing to learn), and the
    training surface (``fit`` / ``add_rows`` / ``maybe_refit`` / ``to_json``) are
    no-ops so it composes cleanly under :class:`FallbackPrior`. Two weight sets:
    ``weights`` for static shapes, ``weights_dynamic`` for symbolic-axis
    (masked-tile) kernels — picked per score on ``S_ext_n_symbolic_axis``."""

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        weights_dynamic: dict[str, float] | None = None,
        scale: float = 0.1,
        atomic_free_split_threshold: float = 4.0,
        atomic_free_weight: float = 5.0,
        scalar_on_warp_weight: float = 40.0,
        splitk_roundtrip_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self._w = weights if weights is not None else _W_A
        self._w_dyn = weights_dynamic if weights_dynamic is not None else _W_A_DYN
        # exp() argument scale — keeps the proxy in a finite, sane range; does not
        # affect ranking (monotone), only the proxy's magnitude.
        self._scale = scale
        # Atomic-free split-K preference.
        # Hardcoded — NOT fit into ``_W_A`` (a plain linear weight can't express the
        # "good when split wide, bad when split narrow" interaction). The learned
        # CatBoostPrior takes over once real atomic-vs-free ``H_opt=3`` rows exist.
        self._atomic_free_split_threshold = atomic_free_split_threshold
        self._atomic_free_weight = atomic_free_weight
        # Scalar-on-warp-eligible penalty + split-K workspace round-trip price.
        # Hardcoded like the atomic-free term — no training rows carry the new stamps yet,
        # and a plain linear weight can't express "only bad when the alternative exists".
        # ``scalar_on_warp_weight`` must outweigh the scalar tile's accumulated geometry
        # bonuses under BOTH weight sets (the dyn set hands scalar rows ~+30 quality via
        # ``D_bn_ge_bm`` / band features a warp row structurally cannot earn — the qwen3-emb
        # projection deploys landed scalar at 5-20× the -O3 cost of their enumerated mma
        # siblings). ``splitk_roundtrip_weight`` is a mild price (~5 quality at free_prod
        # ≈ 512·1024): the deferred finalize IS the right shape for wide mma splits.
        self._scalar_on_warp_weight = scalar_on_warp_weight
        self._splitk_roundtrip_weight = splitk_roundtrip_weight

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
        feats = knob_features(knobs)
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
        return math.exp(-self._scale * max(min(quality, 80.0), -80.0))

    def mean_score(self, knobs: dict) -> float:
        return self.score(knobs)
