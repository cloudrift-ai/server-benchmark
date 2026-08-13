"""Reduce-partition featurization on TILE-less rows.

The load-bearing regression here: the ``REDUCE`` codec used to featurize ONLY inside
``_tile_features`` / ``_warp_tile_features``, so every TILE-less row — a pure reduce kernel's
leaves and the partial rows at a contraction's REDUCE fork — produced byte-identical feature
vectors however the reduce was partitioned (759 µs serial vs 17 µs coop siblings), and no prior
could rank them (the 2026-07-07 cold-baseline 30–68x REDUCE-regret finding).
"""

from __future__ import annotations

from emmy.compiler.pipeline.search.features import knob_features

_CTX = {"S_ext_free_prod": 512.0, "S_ext_reduce_prod": 2048.0, "S_reduce_add": 1.0, "H_sm_count": 128.0}


def _feats(reduce_codec: str, work: str = "", **extra):
    """A tileless reduce row: the site-local ``REDUCE`` value plus the ``WORK`` inventory its
    cooperative width lives in."""
    return knob_features({**_CTX, "REDUCE@a1": reduce_codec, "WORK": work, **extra})


def test_tileless_reduce_children_featurize_distinctly():
    """The 8-sibling fork class from the baseline: serial / coop folds / cross-CTA splits must all
    produce distinct vectors — identical vectors are unrankable by ANY model."""
    vectors = [_feats(c, w) for c, w in (("", ""), ("coop", "t4"), ("coop", "t8"), ("coop", "t64"), ("g2k/coop", "t64"), ("r4", ""))]
    seen = {tuple(sorted(v.items())) for v in vectors}
    assert len(seen) == len(vectors)


def test_coop_fold_rides_thread_features():
    """The cooperative thread fold must move the thread-count features."""
    serial, coop = _feats(""), _feats("coop", "t64")
    assert serial["D_threads"] == 1.0
    assert coop["D_threads"] == 64.0
    assert coop["D_l2_threads"] > serial["D_l2_threads"]


def test_cross_cta_split_rides_splitk_and_finalize():
    """``g<n>`` moves split-K; the finalize letter separates the deferred-kernel combine from the
    atomic fast-path (``g2k`` vs ``g2a``)."""
    atomic, kernel = _feats("g2a/coop", "t8"), _feats("g2k/coop", "t8")
    assert atomic["D_splitk"] == kernel["D_splitk"] == 2.0
    assert atomic["D_finalize_kernel"] == 0.0
    assert kernel["D_finalize_kernel"] == 1.0


def test_ilp_fold_rides_its_own_feature():
    """``r<n>`` changes neither threads nor split-K — only ``D_reduce_ilp`` separates it from
    serial."""
    serial, ilp = _feats(""), _feats("r4")
    assert serial["D_threads"] == ilp["D_threads"] and serial["D_splitk"] == ilp["D_splitk"]
    assert serial["D_reduce_ilp"] == 0.0
    assert ilp["D_reduce_ilp"] == 2.0


def test_transposed_coop_band_featurizes_distinctly():
    """``coop-t`` (the k-major transposed matvec band) is an entirely different kernel from its
    interleaved ``coop`` twin at the same width — the letter must reach the features and the
    signature (transposed goldens are recorded in the per-GPU YAMLs and must join unambiguously)."""
    from emmy.compiler.pipeline.search.features import tile_signature

    inter, transp = _feats("coop", "t256"), _feats("coop-t", "t256")
    assert inter["D_threads"] == transp["D_threads"] == 256.0
    assert inter["D_reduce_transposed"] == 0.0
    assert transp["D_reduce_transposed"] == 1.0
    assert tile_signature({"REDUCE@a1": "coop", "WORK": "t256"}) != tile_signature({"REDUCE@a1": "coop-t", "WORK": "t256"})


def test_pointwise_rows_stay_feature_free():
    """A row with no REDUCE family key gets no ``D_*`` block — pointwise behavior unchanged."""
    feats = knob_features(dict(_CTX))
    assert not any(k.startswith("D_") for k in feats)


def test_tiled_rows_do_not_take_the_reduce_block():
    """The surgical gate: a row with a decided TILE codec featurizes through ``_tile_features``
    exactly as before — the reduce block must NOT fire (its ``D_reduce_ilp`` marker is absent), so
    every previously-featurized row is byte-identical and golden ranks cannot move."""
    tiled = knob_features({**_CTX, "TILE@a1": "f2x4", "WORK": "t16x8", "REDUCE@a1": "coop"})
    assert "D_reduce_ilp" not in tiled
    assert tiled["D_threads"] == 16 * 8 * 128  # BN·BM·coop (the coop width IS the t16x8 inventory) — the scalar-tile path


def test_tileless_rows_keep_the_scalar_on_warp_guard():
    """A per-cell contraction leaf (TILE decided-OFF, coop REDUCE) on a warp-ELIGIBLE kernel is a
    scalar row competing against tensor cores — it must carry ``D_scalar_on_warp_eligible`` so the
    offline guard weight can bury it. Granting such rows the thread/occupancy bonuses WITHOUT the
    guard deployed a 1157 µs per-cell kernel over the 3.5 µs mma golden (square.512.fp16, the
    2026-07-07 5090 first-gate A/B)."""
    eligible = knob_features({**_CTX, "S_warp_eligible": 1.0, "REDUCE@a1": "coop", "WORK": "t256"})
    assert eligible["D_scalar_on_warp_eligible"] == 1.0
    plain = knob_features({**_CTX, "REDUCE@a1": "coop", "WORK": "t256"})  # no warp offer -> guard stays 0
    assert plain["D_scalar_on_warp_eligible"] == 0.0


# ---------------------------------------------------------------------------
# Warp-grid + TMA-conditioned features (the 2026-07-09 arch-differentiation additions)
# ---------------------------------------------------------------------------


def _warp_feats(pin: tuple[str, str], **extra):
    tile_codec, work = pin
    return knob_features({**_CTX, "TILE@a1": tile_codec, "WORK": work, "REDUCE@a1": "", **extra})


def test_warp_grid_separates_same_tile_different_grids():
    """The load-bearing case from the 4090/5090 golden sweeps: two warp variants realizing the
    SAME CTA tile (same threads, cells, tile_m x tile_n — every pre-existing geometry feature
    ties) via a different warp-grid arrangement were byte-identical to every prior — only the
    grid features separate them."""
    a = _warp_feats(("mma_m16n8k16_f16/f2x2", "w2x2"))  # 64x32 tile as a 2x2 warp grid
    b = _warp_feats(("mma_m16n8k16_f16/f1x4", "w4x1"))  # the same 64x32 tile as a 4x1 grid
    for key in ("D_threads", "D_cells", "D_log2_area", "D_aspect"):
        assert a[key] == b[key], key
    assert (a["D_w_grid_m"], a["D_w_grid_n"]) != (b["D_w_grid_m"], b["D_w_grid_n"])


def test_warp_grid_absent_on_scalar_rows():
    """Grid features are warp-tier only — a scalar thread tile must not fabricate them (the
    skip-if-missing 0.0 default is the tier-split convention, like ``D_w_*_bk``)."""
    scalar = knob_features({**_CTX, "TILE@a1": "f2x4", "WORK": "t16x8", "REDUCE@a1": ""})
    assert not any(k.startswith("D_w_grid") for k in scalar)


def test_warp_bk_siblings_featurize_distinctly():
    """The slab K-chunk (the warp ``TILE`` codec's ``k<n>``, ``TilePlan.bk``) must reach the
    ``D_w_*_bk`` features — it used to read the never-set reduce ``serial`` (always 1), so every
    ``k<n>`` sibling featurized byte-identically and no prior could rank them."""
    k1, k4 = _warp_feats(("mma_m16n8k16_f16/f2x2", "w4x2")), _warp_feats(("mma_m16n8k16_f16/f2x2/k4", "w4x2"))
    assert k1["D_w_l2_bk"] == 0.0
    assert k4["D_w_l2_bk"] == 2.0
    k2 = _warp_feats(("mma_m16n8k16_f16/f2x2/k2", "w4x2"))
    assert k2["D_w_near_bk"] == 0.0  # the shallow bk≈2 target
    assert k1["D_w_near_bk"] == -1.0


def test_warp_rows_carry_the_finalize_letter():
    """A warp ``g<n>k`` row must featurize its deferred-combine finalize — the letter was dropped
    on the warp tier (the ``_geom_feats`` "atomic" default applied), so a warp ``g2k`` row was
    indistinguishable from its ``g2a`` twin exactly where wide split-K matters most."""
    kernel = knob_features({**_CTX, "TILE@a1": "mma_m16n8k16_f16/f2x2", "WORK": "w4x2", "REDUCE@a1": "g2k"})
    atomic = knob_features({**_CTX, "TILE@a1": "mma_m16n8k16_f16/f2x2", "WORK": "w4x2", "REDUCE@a1": "g2a"})
    assert kernel["D_finalize_kernel"] == 1.0
    assert atomic["D_finalize_kernel"] == 0.0


def test_warp_grid_orientation_is_physical():
    """The warp codec's m/n orientation is physical (``atom_m = 16 ≠ atom_n = 8``) and both
    orders enumerate — transposed siblings must featurize distinctly, with ``tile_m``/``tile_n``
    on the TRUE axes (the old wide-is-n slot sort collapsed ``w4x2``/``w2x4`` and emitted a
    fictitious square 64×64 for both)."""
    a = _warp_feats(("mma_m16n8k16_f16/f2x2", "w4x2"))  # 4·2·16 = 128 rows × 2·2·8 = 32 cols
    b = _warp_feats(("mma_m16n8k16_f16/f2x2", "w2x4"))  # the transpose: 64 × 64
    assert (a["D_tile_m"], a["D_tile_n"]) == (128.0, 32.0)
    assert (b["D_tile_m"], b["D_tile_n"]) == (64.0, 64.0)
    assert a["D_w_grid_m"] != b["D_w_grid_m"]


def test_stage_split_flag_featurizes():
    """The flash stream enumerates the per-edge transport split (``d1/tma/split``) as a sibling of
    the plain stage — the ``split`` flag must reach the features and the signature."""
    from emmy.compiler.pipeline.search.features import tile_signature

    plain = {**_CTX, "TILE@a1": "mma_m16n8k16_f16/f2x2", "WORK": "w4x2", "REDUCE@a1": "", "STAGE@a1": "d1/tma"}
    split = {**plain, "STAGE@a1": "d1/tma/split"}
    assert knob_features(plain)["D_stage_split"] == 0.0
    assert knob_features(split)["D_stage_split"] == 1.0
    assert tile_signature(plain) != tile_signature(split)


def test_enumerated_warp_pool_featurizes_injectively():
    """The propagation property behind all of the above: every tile in the enumerated warp pool
    (``_WARP_UNITS × _WARP_REGS × _WARP_BK`` for one atom) must produce a distinct feature vector
    AND a distinct ``tile_signature`` — a collision class is a set of siblings no prior can rank
    and no golden can join unambiguously (the pre-fix pool collapsed 468 tiles → 83 classes)."""
    from emmy.compiler.ir.schedule import plan_workers
    from emmy.compiler.pipeline.search.features import tile_signature
    from emmy.compiler.pipeline.search.space import warp_tile_moves

    moves = [(p.spell(), plan_workers(p).spell()) for p in warp_tile_moves(("mma_m16n8k16_f16_f32",))]
    vectors = {tuple(sorted(knob_features({**_CTX, "TILE@a1": m, "WORK": w, "REDUCE@a1": ""}).items())) for m, w in moves}
    signatures = {tile_signature({"TILE@a1": m, "WORK": w, "REDUCE@a1": ""}) for m, w in moves}
    assert len(vectors) == len(moves)
    assert len(signatures) == len(moves)


def test_warp_row_full_vector_matches_hand_computed_encoding():
    """The FEATURIZER_VERSION=3 encoding anchor: the complete feature vector of one maximal warp
    row (atom expansion + true-axis geometry + occupancy + stage/TMA + work/raster), with every
    value hand-derived from the codec definitions — ``w4x2/f2x2`` on ``m16n8k16`` is a 128×32
    tile (``WM·FM·atom_m × WN·FN·atom_n``), 8 warps = 256 threads, ``k2`` = 2 atom_k units. Any
    key or value drift here is an encoding change and must ride a version bump."""
    import math

    free, k_ext, sm = 2097152.0, 4096.0, 170.0  # M=512·N=4096, K=4096
    ctx = {"S_ext_free_prod": free, "S_ext_reduce_prod": k_ext, "S_warp_eligible": 1.0, "H_sm_count": sm}
    got = knob_features(
        {
            **ctx,
            "TILE@a1": "mma_m16n8k16_f16_f16/f2x2/k2",
            "REDUCE@a1": "g2k",
            "STAGE@a1": "d3/tma/p2",
            "WORK": "w4x2+p2",
            "RASTER": "gm8",
        }
    )
    waves = math.log2((free / 4096 * 2) / sm)  # 1024 CTAs over 170 SMs
    needed = k_ext / math.sqrt(free)  # the K-heavy split floor 2√2 (occupancy already saturated)
    reuse = 4096.0 / 160.0
    expect = {
        **ctx,
        "MMA_tier": 1.0, "MMA_atom_m": 16.0, "MMA_atom_n": 8.0, "MMA_atom_k": 16.0,
        "MMA_a_bits": 16.0, "MMA_acc_bits": 16.0,
        "D_threads": 256.0, "D_cells": 4.0, "D_tile_m": 128.0, "D_tile_n": 32.0,
        "D_log2_area": 12.0, "D_reuse": reuse, "D_aspect": 2.0,
        "D_l2_threads": 8.0, "D_near_threads": -1.0, "D_pow2_threads": 1.0,
        "D_cells_cap": 4.0, "D_near_cells": -12.0, "D_near_area": 0.0, "D_square": -2.0,
        "D_l2_reuse": math.log2(reuse), "D_near_intensity": -abs(math.log2(reuse) - 5.0),
        "D_near_kchunks": -abs(math.log2(k_ext / 2) - 5.0),
        "D_neg_masked_m": 0.0, "D_neg_masked_n": 0.0, "D_neg_masked_k": 0.0,
        "D_l2_bn": 0.0, "D_l2_bm": 0.0, "D_bn_ge_bm": 0.0, "D_bn_band": 0.0, "D_bm_band": 0.0,
        "D_l2_bk": 0.0, "D_bk_ge32": 0.0, "D_w_l2_bk": 1.0, "D_w_near_bk": 0.0,
        "D_splitk": 2.0, "D_splitk_le2": 1.0, "D_finalize_kernel": 1.0,
        "D_tilen_clean": 1.0, "D_near_tilen": -1.0, "D_scalar_on_warp_eligible": 0.0,
        "D_log2_ctas": 10.0, "D_log2_waves": waves, "D_near_waves": -abs(waves - 1.0),
        "D_ctas_ge_sm": 1.0, "D_splitk_excess": 0.0, "D_splitk_deficit": math.log2(needed / 2.0),
        "D_splitk_roundtrip": 21.0, "D_l2_cells_occ": 2.0,
        "D_w_grid_m": 2.0, "D_w_grid_n": 1.0, "D_w_grid_aspect": 1.0,
        "D_stage_depth": 3.0, "D_stage_prefetch": 1.0, "D_stage_async": 1.0, "D_stage_tma": 1.0,
        "D_stage_reg_depth": 2.0, "D_stage_split": 0.0,
        "D_tma_aspect": 2.0, "D_tma_log2_area": 12.0, "D_tma_grid_m": 2.0, "D_tma_grid_n": 1.0,
        "D_tma_l2_splitk": 1.0,
        "D_wspec_warps": 2.0, "D_raster_group": 8.0, "D_raster_gn": 0.0,
    }  # fmt: skip
    assert set(got) == set(expect), sorted(set(got) ^ set(expect))
    for key, val in expect.items():
        assert math.isclose(got[key], val, rel_tol=0, abs_tol=1e-9), (key, got[key], val)


def test_tma_interactions_fire_only_on_tma_stage():
    """The TMA-conditioned geometry terms are the one-weight-set stand-in for a per-arch split:
    they must mirror the geometry on a TMA-staged row and stay absent under cp.async, so
    pre-Hopper pools rank exactly as before."""
    tma = _warp_feats(("mma_m16n8k16_f16/f2x2", "w2x4"), **{"STAGE@a1": "d2/tma"})
    cp = _warp_feats(("mma_m16n8k16_f16/f2x2", "w2x4"), **{"STAGE@a1": "d2/cp"})
    assert tma["D_tma_grid_m"] == tma["D_w_grid_m"]
    assert tma["D_tma_grid_n"] == tma["D_w_grid_n"]
    assert tma["D_tma_aspect"] == tma["D_aspect"]
    assert tma["D_tma_log2_area"] == tma["D_log2_area"]
    assert not any(k.startswith("D_tma_") for k in cp)
