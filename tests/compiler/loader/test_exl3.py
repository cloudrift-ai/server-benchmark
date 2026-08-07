"""EXL3 trellis-coded weight decode (``loader.exl3``): bit-window extraction against a bitwise
reference, byte-exact pack/unpack roundtrip, the 3INST computed codebook against a scalar
reference, mma-fragment tile placement, the Hadamard/sign fold, and — when the pinned
GLM-4.5-Air-exl3 checkpoint is in the local HF cache — internal invariants on real tensors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from emmy.compiler.loader.exl3 import (
    CODEBOOK_SCALE,
    codebook_values,
    decode_exl3_linear,
    decode_trellis,
    fold_hadamard,
    pack_trellis,
    trellis_windows,
)

rng = np.random.default_rng(7)


def _random_trellis(kt, nt, K):
    """Any random int16 words form a valid circular code stream — the format has no invalid bytes."""
    return rng.integers(-(2**15), 2**15, (kt, nt, 16 * K)).astype(np.int16)


# ===================================================================
# Scalar references, straight from the format (independent of the module's vectorization)
# ===================================================================


def _ref_windows(tile_u16, K):
    """Bitwise window reference: expand the stream to individual bits, read each window MSB-first."""
    u32 = [int(tile_u16[2 * i]) | (int(tile_u16[2 * i + 1]) << 16) for i in range(8 * K)]
    nbits = 256 * K
    bits = [(u32[j // 32] >> (31 - j % 32)) & 1 for j in range(nbits)]
    wins = []
    for t in range(256):
        w = 0
        for b in range(16):
            w = (w << 1) | bits[((t + 1) * K - 16 + b) % nbits]
        wins.append(w)
    return np.array(wins, dtype=np.uint16)


def _f16_bits(u):
    return np.array(u, dtype=np.uint16).view(np.float16)


def _ref_codebook(w, cb):
    """Scalar 3INST reference: python ints mod 2^32, numpy fp16 scalar arithmetic."""
    m = (1 << 32) - 1
    x = int(w)
    if cb == 0:
        x = (x * 89226354 + 64248484) & m
    elif cb == 1:
        x = (x * 0xCBAC1FED) & m
    elif cb == 2:
        x = (x * 0x83DCD12D) & m
        s = ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24) + 0x6400) & 0xFFFF
        return np.float16(np.float64(_f16_bits(s)) * np.float64(_f16_bits(0x1EEE)) + np.float64(_f16_bits(0xC931)))
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    return _f16_bits(x & 0xFFFF) + _f16_bits(x >> 16)


def _ref_decode_tile(tile_i16, K, cb):
    """Scalar tile decode: windows → codebook → the mma B-fragment (row, col) placement."""
    wins = _ref_windows(tile_i16.view(np.uint16), K)
    tile = np.zeros((16, 16), dtype=np.float16)
    for lane in range(32):
        for j in range(8):
            r = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)
            c = lane // 4 + 8 * (j >> 2)
            tile[r, c] = _ref_codebook(wins[8 * lane + j], cb)
    return tile


# ===================================================================
# Bit windows and the packed stream
# ===================================================================


@pytest.mark.parametrize("K", range(1, 9))
def test_windows_match_bitwise_reference(K):
    tr = _random_trellis(2, 3, K)
    wins = trellis_windows(tr)
    for i in range(2):
        for j in range(3):
            np.testing.assert_array_equal(wins[i, j], _ref_windows(tr[i, j].view(np.uint16), K))


@pytest.mark.parametrize("K", range(1, 9))
def test_windows_overlap_invariant(K):
    """Tail-biting: window(t) >> K == window(t-1) mod 2^(16-K), circularly (t=0 wraps to t=255)."""
    wins = trellis_windows(_random_trellis(3, 2, K)).astype(np.uint32)
    prev = np.roll(wins, 1, axis=-1)
    np.testing.assert_array_equal(wins >> K, prev & ((1 << (16 - K)) - 1))


@pytest.mark.parametrize("K", range(1, 9))
def test_pack_roundtrip_byte_exact(K):
    """unpack → repack reproduces the stored int16 words exactly — pins every bit's placement."""
    tr = _random_trellis(2, 3, K)
    np.testing.assert_array_equal(pack_trellis(trellis_windows(tr), K), tr)


def test_trellis_validation_errors():
    with pytest.raises(ValueError, match="3-D"):
        trellis_windows(np.zeros((4, 32), dtype=np.int16))
    with pytest.raises(ValueError, match="int16"):
        trellis_windows(np.zeros((1, 1, 32), dtype=np.int32))
    with pytest.raises(ValueError, match="integer K"):
        trellis_windows(np.zeros((1, 1, 24), dtype=np.int16))  # 24 = 16*1.5
    with pytest.raises(ValueError, match="integer K"):
        trellis_windows(np.zeros((1, 1, 16 * 9), dtype=np.int16))  # K = 9
    with pytest.raises(ValueError, match="last dim must be 256"):
        pack_trellis(np.zeros((2, 2, 128), dtype=np.uint16), 2)
    with pytest.raises(ValueError, match=r"K must be in \[1, 8\]"):
        pack_trellis(np.zeros((2, 2, 256), dtype=np.uint16), 9)


# ===================================================================
# 3INST computed codebook
# ===================================================================


@pytest.mark.parametrize("cb", [0, 1, 2])
def test_codebook_matches_scalar_reference(cb):
    wins = np.concatenate([np.array([0, 1, 0x7FFF, 0x8000, 0xFFFF]), rng.integers(0, 65536, 512)]).astype(np.uint16)
    ref = np.array([_ref_codebook(w, cb) for w in wins], dtype=np.float16)
    got = codebook_values(wins, cb)
    np.testing.assert_array_equal(got.view(np.uint16), ref.view(np.uint16))  # bit-exact, not tolerance


def test_codebook_full_table_pins_the_scale():
    """Full-table stats over all 65536 windows: mean ~0, std == the encoder's CODEBOOK_SCALE
    constant (1.24371088) — the anchor for the rms sanity checks on real decoded tiles."""
    vals = codebook_values(np.arange(65536, dtype=np.uint16)).astype(np.float32)
    assert abs(float(vals.std()) - CODEBOOK_SCALE) < 1e-4
    assert abs(float(vals.mean())) < 1e-2
    assert not np.isnan(vals).any() and not np.isinf(vals).any()


def test_codebook_rejects_unknown_id():
    with pytest.raises(ValueError, match="unknown codebook id"):
        codebook_values(np.zeros(4, dtype=np.uint16), 3)


# ===================================================================
# Tile placement and whole-tensor decode
# ===================================================================


@pytest.mark.parametrize(("K", "cb"), [(2, 0), (6, 0), (2, 1), (3, 2)])
def test_decode_trellis_matches_scalar_reference(K, cb):
    """Vectorized decode == the scalar per-tile reference (windows, codebook, placement), exactly.
    K=2 and K=6 are the two rungs the pinned GLM-4.5-Air 2.0bpw checkpoint uses (body / lm_head)."""
    tr = _random_trellis(2, 2, K)
    got = decode_trellis(tr, cb)
    assert got.shape == (32, 32) and got.dtype == np.float16
    for i in range(2):
        for j in range(2):
            ref = _ref_decode_tile(tr[i, j], K, cb)
            np.testing.assert_array_equal(got[16 * i : 16 * i + 16, 16 * j : 16 * j + 16].view(np.uint16), ref.view(np.uint16))


# ===================================================================
# Hadamard / sign fold
# ===================================================================


def _ref_sylvester(n):
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def test_fold_hadamard_basis_element():
    """A single 1 at (0, 0) spreads to a constant 1/128 over its own 128x128 block and exactly
    zero outside — 1/128 is exact in fp16, so this is an exact expectation."""
    w_hat = np.zeros((256, 256), dtype=np.float16)
    w_hat[0, 0] = 1.0
    out = fold_hadamard(w_hat, np.ones(256, dtype=np.float16), np.ones(256, dtype=np.float16))
    np.testing.assert_array_equal(out[:128, :128], np.full((128, 128), 1 / 128, dtype=np.float16))
    np.testing.assert_array_equal(out[128:, :], 0)
    np.testing.assert_array_equal(out[:, 128:], 0)


def test_fold_hadamard_matches_dense_reference():
    """Blocked fold == the dense block-diagonal H128 sandwich computed independently in float64.
    The comparison tolerance is the fold's own contract — fp16 rounding of the fold is
    implementation-defined (exllamav3's fused-vs-reference test tolerates 2e-3 relative)."""
    w_hat = (rng.standard_normal((256, 384)) * CODEBOOK_SCALE).astype(np.float16)
    suh = (rng.choice([-1.0, 1.0], 256) * 0.01).astype(np.float16)
    svh = rng.choice([-1.0, 1.0], 384).astype(np.float16)
    h = _ref_sylvester(128) / np.sqrt(128)
    hk = np.kron(np.eye(2), h)  # block-diagonal over the 256 rows
    hn = np.kron(np.eye(3), h)  # block-diagonal over the 384 cols
    ref = np.diag(suh.astype(np.float64)) @ hk @ w_hat.astype(np.float64) @ hn @ np.diag(svh.astype(np.float64))
    out = fold_hadamard(w_hat, suh, svh)
    np.testing.assert_allclose(out.astype(np.float32), ref.astype(np.float32), rtol=2e-3, atol=1e-6)


def test_fold_hadamard_validation_errors():
    with pytest.raises(ValueError, match="not a multiple of 128"):
        fold_hadamard(np.zeros((64, 128), dtype=np.float16), np.ones(64, np.float16), np.ones(128, np.float16))
    with pytest.raises(ValueError, match="do not match"):
        fold_hadamard(np.zeros((128, 128), dtype=np.float16), np.ones(64, np.float16), np.ones(128, np.float16))


# ===================================================================
# One-linear decode (the sibling-tensor entry point)
# ===================================================================


def test_decode_exl3_linear_composes_decode_and_fold():
    tr = _random_trellis(8, 8, 2)
    suh = (rng.choice([-1.0, 1.0], 128) * 0.01).astype(np.float16)
    svh = rng.choice([-1.0, 1.0], 128).astype(np.float16)
    out = decode_exl3_linear(tr, suh, svh)
    ref = fold_hadamard(decode_trellis(tr, 0), suh, svh)
    np.testing.assert_array_equal(out.view(np.uint16), ref.view(np.uint16))


def test_decode_exl3_linear_marker_selects_codebook():
    """Codebook selection is by marker PRESENCE; the stored marker value is never read."""
    tr = _random_trellis(8, 8, 2)
    suh, svh = np.ones(128, dtype=np.float16), np.ones(128, dtype=np.float16)
    via_mcg = decode_exl3_linear(tr, suh, svh, mcg=np.array(0xCBAC1FED, dtype=np.uint32).view(np.int32))
    np.testing.assert_array_equal(via_mcg.view(np.uint16), fold_hadamard(decode_trellis(tr, 1), suh, svh).view(np.uint16))
    via_mul1 = decode_exl3_linear(tr, suh, svh, mul1=np.array(0, dtype=np.int32))  # value ignored
    np.testing.assert_array_equal(via_mul1.view(np.uint16), fold_hadamard(decode_trellis(tr, 2), suh, svh).view(np.uint16))


def test_decode_exl3_linear_rejects_both_markers():
    tr = _random_trellis(8, 8, 2)
    ones = np.ones(128, dtype=np.float16)
    with pytest.raises(ValueError, match="mutually exclusive"):
        decode_exl3_linear(tr, ones, ones, mcg=np.array(1, np.int32), mul1=np.array(1, np.int32))


# ===================================================================
# Real checkpoint invariants (skips cleanly when the pinned snapshot is not cached)
# ===================================================================

_REPO = "turboderp/GLM-4.5-Air-exl3"
_REVISION = "a1adde54568f29a04c4c369180be2c17286dbec6"  # the 2.0bpw rung, pinned


def _cached_snapshot() -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    p = try_to_load_from_cache(_REPO, "model.safetensors.index.json", revision=_REVISION)
    return Path(p).parent if isinstance(p, str) else None


_SNAPSHOT = _cached_snapshot()

requires_glm_exl3 = pytest.mark.skipif(_SNAPSHOT is None, reason="pinned GLM-4.5-Air-exl3 2.0bpw snapshot not in the HF cache")


def _load_real(name: str, tile_slice=None):
    """Read one tensor (or a leading-axes tile slice of it) from the cached snapshot."""
    from safetensors import safe_open

    index = json.loads((_SNAPSHOT / "model.safetensors.index.json").read_text())
    with safe_open(str(_SNAPSHOT / index["weight_map"][name]), framework="numpy") as f:
        if tile_slice is None:
            return f.get_tensor(name)
        return f.get_slice(name)[tile_slice]


# Dense-layer, MoE-expert, attention, and head tensors; lm_head covers the K=6 rung.
_REAL_NAMES = [
    "model.layers.0.mlp.up_proj",
    "model.layers.5.mlp.experts.0.down_proj",
    "model.layers.5.self_attn.q_proj",
    "lm_head",
]


@requires_glm_exl3
@pytest.mark.parametrize("name", _REAL_NAMES)
def test_real_tensor_windows_and_repack(name):
    """On real checkpoint tiles: the tail-biting overlap invariant holds, and repacking the
    extracted windows reproduces the stored bytes exactly (pins alignment and endianness)."""
    tr = _load_real(name + ".trellis", np.s_[:8, :8])
    K = tr.shape[-1] // 16
    wins = trellis_windows(tr)
    w32 = wins.astype(np.uint32)
    np.testing.assert_array_equal(w32 >> K, np.roll(w32, 1, axis=-1) & ((1 << (16 - K)) - 1))
    np.testing.assert_array_equal(pack_trellis(wins, K), tr)


@requires_glm_exl3
def test_real_tensor_decode_statistics():
    """Decoded hat-basis values have rms near CODEBOOK_SCALE; the folded weight has a plausible
    fp16 weight scale, with suh carrying the magnitude and svh nearly pure signs."""
    name = "model.layers.0.mlp.up_proj"
    tr = _load_real(name + ".trellis", np.s_[:8, :8])
    suh = _load_real(name + ".suh", np.s_[:128])
    svh = _load_real(name + ".svh", np.s_[:128])
    w_hat = decode_trellis(tr)

    def rms(a):
        return float(np.sqrt(np.mean(np.square(a.astype(np.float32)))))

    assert 1.0 < rms(w_hat) < 1.5  # ~ CODEBOOK_SCALE
    w = fold_hadamard(w_hat, suh, svh)
    assert 0.001 < rms(w) < 0.1
    assert rms(suh) < 0.1  # magnitude side
    assert 0.5 < rms(svh) < 2.0  # ~ pure signs
