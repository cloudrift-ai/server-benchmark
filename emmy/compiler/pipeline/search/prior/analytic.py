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
    "D_pow2_threads": 56.064604715252045,
    "D_wspec_warps": -24.257928498394435,
    "D_bn_band": -18.84269694844684,
    "D_stage_tma": 18.717239275069474,
    "MMA_tier": 18.171894758846168,
    "D_stage_async": 17.632064515181334,
    "D_splitk_le2": 14.3272884557364,
    "D_w_near_bk": -13.513445772588467,
    "D_splitk_excess": -11.380927300585197,
    "D_stage_reg_depth": -10.992053852671837,
    "D_ctas_ge_sm": -9.973442761046039,
    "D_stage_ring": 7.266411378449085,
    "D_bn_ge_bm": 6.175249346262891,
    "D_near_waves": 4.094219045750717,
    "D_w_l2_bk": 3.8318001959254397,
    "D_bm_band": -3.125815915965274,
    "D_l2_bm": -2.813537060079752,
    "D_near_kchunks": -2.4803958401849546,
    "D_square": 2.4682813427691133,
    "D_bk_ge32": -2.250612760379462,
    "D_near_tilen": -1.872229368479013,
    "D_near_area": 1.661659507348094,
    "D_l2_reuse": 1.2797850818471643,
    "D_l2_bn": -1.1315248840332646,
    "D_log2_area": -1.038544031822043,
    "D_l2_bk": 0.9757523358723075,
    "D_l2_threads": -0.9542014237345692,
    "D_neg_masked_k": -0.8646029028646691,
    "D_neg_masked_m": -0.8646029028646691,
    "D_neg_masked_n": -0.8646029028646691,
    "D_tilen_clean": 0.8575036370703841,
    "D_stage_depth": -0.7706835774426998,
    "D_splitk": -0.562782237146292,
    "D_near_intensity": -0.31230584089622293,
    "D_log2_waves": -0.3096053721652958,
    "D_near_threads": 0.28478634495364613,
    "D_aspect": -0.26074058469642597,
    "D_log2_ctas": -0.19597136958382297,
    "D_near_cells": -0.17771621148992126,
    "D_cells": -0.15033533823472636,
    "D_cells_cap": -0.14767192811087732,
    "D_l2_cells_occ": 0.10074311937306302,
    "D_tile_m": 0.017878083455336286,
    "D_tile_n": -0.014204261399731513,
    "D_reuse": -0.004804011945045963,
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
    "D_ctas_ge_sm": -28.049416306987936,
    "D_stage_tma": 10.311663430135269,
    "D_wspec_warps": -9.600205920261855,
    "D_stage_ring": 8.031202386703683,
    "D_pow2_threads": 5.536773836554719,
    "D_bn_ge_bm": 4.415479530948023,
    "D_splitk_le2": 4.282947897470383,
    "D_w_near_bk": -3.9840434564414764,
    "D_log2_ctas": -3.1515515658852546,
    "D_bm_band": 3.0501575746742238,
    "D_bn_band": -2.9288876054479727,
    "D_tilen_clean": 2.1695047468026094,
    "D_l2_bm": -1.9985579520331742,
    "D_splitk_excess": 1.6870439608853967,
    "D_near_tilen": 1.5992628367976942,
    "D_stage_reg_depth": -1.4890000034419477,
    "D_neg_masked_n": 1.2915904848884612,
    "D_near_threads": -0.990004931320281,
    "D_bk_ge32": -0.8965483150597096,
    "D_finalize_kernel": 0.829960031801419,
    "D_w_l2_bk": -0.8189885344311161,
    "D_l2_cells_occ": 0.8105008956036689,
    "D_stage_async": 0.6848803259217643,
    "D_stage_depth": 0.6722793037663716,
    "D_neg_masked_m": 0.5582125267295052,
    "D_l2_bn": -0.5404307987580594,
    "D_splitk": 0.5404191129714471,
    "D_aspect": 0.5107250561585116,
    "D_log2_waves": 0.4818492104433922,
    "D_neg_masked_k": -0.4296618098730851,
    "D_near_area": 0.36859820129924153,
    "D_near_intensity": 0.3369825435390131,
    "D_square": 0.3178616056231372,
    "D_near_kchunks": 0.23771245195723872,
    "D_l2_reuse": -0.12756603126978133,
    "MMA_tier": -0.10337548227926727,
    "D_cells_cap": -0.09509366379321203,
    "D_near_cells": 0.0871595265752497,
    "D_l2_threads": -0.06307751814009688,
    "D_log2_area": -0.05544529821916362,
    "D_l2_bk": 0.03966053994125637,
    "D_reuse": 0.02982421937026558,
    "D_cells": -0.02433851397356188,
    "D_near_waves": -0.02229187068901151,
    "D_tile_n": -0.00492814227331677,
    "D_threads": -0.0011827346975242477,
    "D_tile_m": -0.00010694165001567948,
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
