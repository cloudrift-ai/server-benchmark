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


def _feats(reduce_codec: str, **extra):
    return knob_features({**_CTX, "REDUCE@a1": reduce_codec, **extra})


def test_tileless_reduce_children_featurize_distinctly():
    """The 8-sibling fork class from the baseline: serial / coop folds / cross-CTA splits must all
    produce distinct vectors — identical vectors are unrankable by ANY model."""
    vectors = [_feats(c) for c in ("", "b4", "b8", "b64", "g2k/b64", "r4")]
    seen = {tuple(sorted(v.items())) for v in vectors}
    assert len(seen) == len(vectors)


def test_coop_fold_rides_thread_features():
    """``b<n>`` is a cooperative thread fold — it must move the thread-count features."""
    serial, coop = _feats(""), _feats("b64")
    assert serial["D_threads"] == 1.0
    assert coop["D_threads"] == 64.0
    assert coop["D_l2_threads"] > serial["D_l2_threads"]


def test_cross_cta_split_rides_splitk_and_finalize():
    """``g<n>`` moves split-K; the finalize letter separates the deferred-kernel combine from the
    atomic fast-path (``g2k`` vs ``g2a``)."""
    atomic, kernel = _feats("g2a/b8"), _feats("g2k/b8")
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


def test_pointwise_rows_stay_feature_free():
    """A row with no REDUCE family key gets no ``D_*`` block — pointwise behavior unchanged."""
    feats = knob_features(dict(_CTX))
    assert not any(k.startswith("D_") for k in feats)


def test_tiled_rows_do_not_take_the_reduce_block():
    """The surgical gate: a row with a decided TILE codec featurizes through ``_tile_features``
    exactly as before — the reduce block must NOT fire (its ``D_reduce_ilp`` marker is absent), so
    every previously-featurized row is byte-identical and golden ranks cannot move."""
    tiled = knob_features({**_CTX, "TILE@a1": "n16x8/f2x4", "REDUCE@a1": "b8"})
    assert "D_reduce_ilp" not in tiled
    assert tiled["D_threads"] == 16 * 8 * 8  # BN·BM·coop — the pre-fix scalar-tile path


def test_tileless_rows_keep_the_scalar_on_warp_guard():
    """A per-cell contraction leaf (TILE decided-OFF, coop REDUCE) on a warp-ELIGIBLE kernel is a
    scalar row competing against tensor cores — it must carry ``D_scalar_on_warp_eligible`` so the
    offline guard weight can bury it. Granting such rows the thread/occupancy bonuses WITHOUT the
    guard deployed a 1157 µs per-cell kernel over the 3.5 µs mma golden (square.512.fp16, the
    2026-07-07 5090 first-gate A/B)."""
    eligible = knob_features({**_CTX, "S_warp_eligible": 1.0, "REDUCE@a1": "b256"})
    assert eligible["D_scalar_on_warp_eligible"] == 1.0
    plain = knob_features({**_CTX, "REDUCE@a1": "b256"})  # no warp offer -> guard stays 0
    assert plain["D_scalar_on_warp_eligible"] == 0.0


# ---------------------------------------------------------------------------
# Warp-grid + TMA-conditioned features (the 2026-07-09 arch-differentiation additions)
# ---------------------------------------------------------------------------


def _warp_feats(tile_codec: str, **extra):
    return knob_features({**_CTX, "TILE@a1": tile_codec, "REDUCE@a1": "", **extra})


def test_warp_grid_separates_same_tile_different_grids():
    """The load-bearing case from the 4090/5090 golden sweeps: two warp variants realizing the
    SAME CTA tile (same threads, cells, tile_m x tile_n — every pre-existing geometry feature
    ties) via a different warp-grid arrangement were byte-identical to every prior — only the
    grid features separate them."""
    a = _warp_feats("a:mma_m16n8k16_f16/w2x2/f2x2")  # 64x32 tile as a 2x2 warp grid
    b = _warp_feats("a:mma_m16n8k16_f16/w4x1/f1x4")  # the same 64x32 tile as a 4x1 grid
    for key in ("D_threads", "D_cells", "D_log2_area", "D_aspect"):
        assert a[key] == b[key], key
    assert (a["D_w_grid_m"], a["D_w_grid_n"]) != (b["D_w_grid_m"], b["D_w_grid_n"])


def test_warp_grid_absent_on_scalar_rows():
    """Grid features are warp-tier only — a scalar thread tile must not fabricate them (the
    skip-if-missing 0.0 default is the tier-split convention, like ``D_w_*_bk``)."""
    scalar = knob_features({**_CTX, "TILE@a1": "n16x8/f2x4", "REDUCE@a1": "b8"})
    assert not any(k.startswith("D_w_grid") for k in scalar)


def test_tma_interactions_fire_only_on_tma_stage():
    """The TMA-conditioned geometry terms are the one-weight-set stand-in for a per-arch split:
    they must mirror the geometry on a TMA-staged row and stay absent under cp.async, so
    pre-Hopper pools rank exactly as before."""
    tma = _warp_feats("a:mma_m16n8k16_f16/w2x4/f2x2", **{"STAGE@a1": "d2/tma/ring"})
    cp = _warp_feats("a:mma_m16n8k16_f16/w2x4/f2x2", **{"STAGE@a1": "d2/cp/ring"})
    assert tma["D_tma_grid_m"] == tma["D_w_grid_m"]
    assert tma["D_tma_grid_n"] == tma["D_w_grid_n"]
    assert tma["D_tma_aspect"] == tma["D_aspect"]
    assert tma["D_tma_log2_area"] == tma["D_log2_area"]
    assert not any(k.startswith("D_tma_") for k in cp)
