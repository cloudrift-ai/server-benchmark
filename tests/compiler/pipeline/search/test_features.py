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
    analytic guard weight can bury it. Granting such rows the thread/occupancy bonuses WITHOUT the
    guard deployed a 1157 µs per-cell kernel over the 3.5 µs mma golden (square.512.fp16, the
    2026-07-07 5090 first-gate A/B)."""
    eligible = knob_features({**_CTX, "S_warp_eligible": 1.0, "REDUCE@a1": "b256"})
    assert eligible["D_scalar_on_warp_eligible"] == 1.0
    plain = knob_features({**_CTX, "REDUCE@a1": "b256"})  # no warp offer -> guard stays 0
    assert plain["D_scalar_on_warp_eligible"] == 0.0
