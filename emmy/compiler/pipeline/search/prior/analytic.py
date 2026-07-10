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
# goldens' tier-weighted mean ``log2(rank+1)``. Refit 2026-07-07: the FIRST fit whose objective
# actually includes the reduce / pointwise cases (the #318 fork widening makes them enumerate — 12
# reduce-fork rows, 3 pointwise rows — and the TILE-less reduce featurization gives the fit
# distinguishing columns; the reduce/pointwise goldens land at rank 0-2 in their pools). 49 static
# + 12 dynamic cases: static median 30 → 27 (top1 5 → 8/49), dynamic median 1279 → 179. rms_norm /
# softmax / attention goldens still enumerate zero rows through the live-fork capture — the
# remaining cold-rank gap. The previously-dead ``D_finalize_kernel`` / ``D_reduce_ilp`` columns now
# carry fit weights (the atomic-free hand-term in ``score`` stays 0 — the linear weight covers it,
# fit against real pools). Dominant terms: occupancy
# (``D_ctas_ge_sm``/``D_near_waves`` — keep #CTAs ≈ 2 waves over the SMs), the
# ``D_bm_band`` thread-tile band, the tier-split warp BK target (``D_w_near_bk`` —
# BK≈2 on the TMA tile), the positive ``MMA_tier`` (fp16/bf16 prefer the warp
# tensor-core tile — the fp16 cases enumerate BOTH tiers, so the fit must rank the
# warp golden over the scalar candidates), and the reduce signal rides
# ``D_threads``/occupancy (cooperative ``BR`` raises thread count toward the
# target). One un-gated linear model serves every regime, so some band weights
# compromise across regimes (e.g. ``D_bn_band`` is mildly negative — matmul wants
# the band but reduce wants BN=1; the fit trades it for occupancy).
# Refit 2026-07-09 (arch-differentiation features, from the 4090/5090 golden-sweep findings): the
# featurizer gained the warp-grid arrangement (``D_w_grid_m/n/aspect`` — wide-vs-narrow warp grids
# over one pool were previously indistinguishable, the 5090's 0/17 TILE-match axis) and the
# TMA-conditioned geometry interactions (``D_tma_*`` — TMA only enumerates on Hopper/Blackwell, so
# these terms are where ONE weight set prices those cards' tiles separately; the fit offsets the
# global ``D_w_grid_m`` penalty on TMA rows via ``D_tma_grid_m/n``). Fit with coordinate descent
# from the incumbent seed ONLY (``--samples 0``): the random-restart stage reached a worse dyn
# objective while trashing node-store calibration (4090 leaf Spearman +0.46 → +0.18), so the
# incumbents move only where the golden objective demands. 58 static + 11 dynamic cases (the
# sweeps' newly-recorded goldens included). ``eval analytic`` (the deployed scoring path, gates
# included): top1 31 → 40/53, top50 41 → 47/53, every 2026-07-09-sweep ``.dynM`` miss to rank 0
# (mlp_down 2358 → 0, square.512 1198 → 0, qkv 780 → 0, o_proj 663 → 0); 4090 node-store TILE fork
# regret 3.62x → 2.31x, calibration +0.46 → +0.53 (the 5090 node table stays unadjudicable until
# the degenerate-bench roofline floor lands — its regressed rows sit on physically-impossible
# baselines). The masked tier diverges through the TMA terms: ``D_tma_grid_m/n`` +4.75 vs the
# static +2.49 and ``D_tma_l2_splitk`` +3.15 vs −0.68 — the split-K-under-TMA credit the 5090's
# recorded wins (``g4k``/``k4``) demanded, and the warp-aspect-under-masking divergence the 4090
# report predicted, both riding the dyn weight set (the regime split IS the masking condition).
_W_A: dict[str, float] = {
    "D_pow2_threads": 56.064604715252045,
    "D_wspec_warps": -37.20725494610845,
    "D_stage_tma": 34.72463241182819,
    "D_stage_async": 26.39183708293144,
    "D_bn_band": -18.84269694844684,
    "MMA_tier": 18.17189475884617,
    "D_stage_reg_depth": -16.58897291874536,
    "D_splitk_le2": 14.3272884557364,
    "D_w_near_bk": -13.513445772588467,
    "D_ctas_ge_sm": -11.19771419806356,
    "D_bn_ge_bm": 8.392492143187166,
    "D_stage_ring": 7.265811802177642,
    "D_splitk_excess": -4.378479881652623,
    "D_near_waves": 4.094219045750717,
    "D_w_l2_bk": 3.8318001959254397,
    "D_w_grid_m": -3.188169346618735,
    "D_bm_band": -3.125815915965274,
    "D_tma_grid_m": 2.488719829570663,
    "D_near_kchunks": -2.4803958401849546,
    "D_tma_grid_n": 2.469026988987517,
    "D_square": 2.468281342769113,
    "D_l2_bn": -2.4621734431664835,
    "D_bk_ge32": -2.250612760379462,
    "D_near_tilen": -1.872229368479013,
    "D_near_area": 1.661659507348094,
    "D_finalize_kernel": 1.595613572829753,
    "D_log2_area": -1.5817687429343905,
    "D_stage_depth": -1.5439384014596318,
    "D_w_grid_n": -1.424978258907764,
    "D_l2_bm": -1.016664591735981,
    "D_l2_bk": 0.9757523358723075,
    "D_neg_masked_k": -0.8646029028646691,
    "D_neg_masked_m": -0.8646029028646691,
    "D_neg_masked_n": -0.8646029028646691,
    "D_l2_reuse": 0.7423708659279166,
    "D_tma_l2_splitk": -0.6799448986880451,
    "D_l2_cells_occ": 0.6332227060795635,
    "D_tilen_clean": -0.35717182254632074,
    "D_tma_log2_area": 0.35016470113957704,
    "D_near_intensity": -0.31230584089622293,
    "D_log2_waves": -0.3096053721652958,
    "D_near_threads": 0.28478634495364613,
    "D_log2_ctas": -0.19597136958382297,
    "D_l2_threads": -0.17842037213830914,
    "D_near_cells": -0.17771621148992126,
    "D_cells_cap": -0.14767192811087732,
    "D_aspect": 0.12744471745958774,
    "D_cells": -0.07129999491127952,
    "D_splitk": 0.01461566432371748,
    "D_tile_n": -0.014204261399731511,
    "D_tile_m": 0.010710054894838165,
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
    "D_pow2_threads": 56.064604715252045,
    "D_wspec_warps": -37.20725494610845,
    "D_stage_tma": 34.724632411828196,
    "D_ctas_ge_sm": -31.017907401162894,
    "D_stage_async": 29.390713167920737,
    "D_bn_band": -18.842696948446843,
    "MMA_tier": 18.17189475884617,
    "D_stage_reg_depth": -16.58897291874536,
    "D_splitk_le2": 16.412270371195632,
    "D_w_near_bk": -13.513445772588465,
    "D_stage_ring": 13.487222636619627,
    "D_tilen_clean": 9.385844213879363,
    "D_bn_ge_bm": 8.392492143187166,
    "D_l2_bn": -5.627769315796922,
    "D_tma_grid_m": 4.751858339655151,
    "D_tma_grid_n": 4.748122089727162,
    "D_l2_bm": -4.6824967615689,
    "D_splitk_excess": -4.378479881652623,
    "D_w_l2_bk": 3.8318001959254397,
    "D_log2_ctas": -3.7112598684821614,
    "D_square": 3.6235224203379897,
    "D_near_intensity": 3.369498664707686,
    "D_splitk_deficit": 3.323039986615204,
    "D_tma_l2_splitk": 3.14951925777414,
    "D_bm_band": -3.125815915965274,
    "D_near_waves": 3.116920162866393,
    "D_near_kchunks": -3.0464157545056683,
    "D_bk_ge32": -2.250612760379462,
    "D_l2_reuse": 1.8258927050396851,
    "D_aspect": 1.6793979227725437,
    "D_near_area": 1.6616595073480942,
    "D_near_threads": 1.5969662238049749,
    "D_finalize_kernel": 1.5956135728297531,
    "D_tma_aspect": 1.2728226103550313,
    "D_splitk": 1.182281907271687,
    "D_log2_area": -1.0321013225539433,
    "D_l2_bk": 0.9757523358723075,
    "D_neg_masked_k": -0.8646029028646691,
    "D_neg_masked_m": -0.8646029028646691,
    "D_neg_masked_n": -0.8646029028646691,
    "D_near_tilen": 0.8524900232796672,
    "D_stage_depth": -0.7652756190400337,
    "D_l2_cells_occ": 0.6332227060795635,
    "D_w_grid_n": 0.4825271540008625,
    "D_l2_threads": -0.17842037213830916,
    "D_log2_waves": 0.16967068674229394,
    "D_tma_log2_area": -0.14978999992751255,
    "D_reuse": 0.11101082334795016,
    "D_near_cells": 0.10671736661060958,
    "D_cells_cap": -0.09373458023720967,
    "D_cells": 0.0365747008360558,
    "D_w_grid_m": -0.033970305176734605,
    "D_tile_n": 0.02203094231212109,
    "D_tile_m": 0.010710054894838169,
    "D_threads": -0.008917737840240542,
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
        atomic_free_weight: float = 0.0,
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
        # Weight defaults to 0.0 (term OFF): its input ``D_finalize_kernel`` was dead
        # (never forwarded by ``features._reduce_decomp``) when the ±5.0 params were
        # written, so they were never validated — and activating them when the feature
        # came alive (2026-07-07) regressed golden top-50 coverage 13→10 (the 5090's
        # ``g2k`` split-2 golden contradicts the narrow-split penalty's sign). Re-enable
        # only with refit params that pass the golden-rank gate. Same reason the fit
        # noise weight on the then-constant ``D_finalize_kernel`` column was dropped
        # from ``_W_A_DYN``: both changes keep the cold ranking byte-identical to the
        # dead-feature era while the learned prior consumes the now-live signal.
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
        # NOTE: this term was DEAD until 2026-07-07 (``D_finalize_kernel`` never forwarded —
        # see ``features._reduce_decomp``); activation measured ~neutral on the golden gate
        # (median 0 / top1 21 / top10 26 / top50 31 unchanged, top25 28→27; the g<n>k goldens
        # sink within their pools, their non-split competitors rise).
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
        return math.exp(-self._scale * max(min(quality, 80.0), -80.0))

    def explain_features(self, feats: dict) -> dict[str, float]:
        """EXACT per-term decomposition of the quality score (higher = predicted
        faster): each nonzero linear term by its feature name, plus the three
        hardcoded interaction gates as ``gate:*`` pseudo-terms — a blame table that
        omitted the ±40 scalar-on-warp gate would misattribute exactly the misses it
        dominates. Invariant (unit-tested): the terms sum to the same quality
        :meth:`mean_score_features` exponentiates, so a two-row term diff is the
        model's exact preference gap. (The ±80 clip inside the squash is ignored
        here — it changes the proxy's magnitude at the extremes, never the terms.)"""
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
