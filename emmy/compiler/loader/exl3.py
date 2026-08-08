"""EXL3 (QTIP-class trellis-coded) checkpoint decode: pure numpy weight reconstruction.

An EXL3 linear stores its logical ``(in_features = k, out_features = n)`` weight — orientation
``(in, out)``, ``y = x @ W`` — as sibling tensors:

- ``trellis`` int16 ``(k/16, n/16, 16*K)``: the packed codes, one self-contained tail-biting trellis
  walk per 16x16 weight tile. ``K = trellis.shape[-1] // 16`` is the integer bits/weight (1..8) and is
  the ONLY place K is recorded.
- ``suh`` fp16 ``(k,)`` / ``svh`` fp16 ``(n,)``: input/output-side per-channel multipliers (random
  signs with every scale folded in); together with the fixed 128-block Hadamard transform they restore
  the original basis.
- optional ``mcg`` / ``mul1`` int32 scalar markers: PRESENCE selects codebook 1 / 2 (the stored values
  are never read); neither present selects codebook 0.
- optional ``bias`` fp16 ``(n,)``: a plain tensor, nothing to decode.

Decode is exact reconstruction, not approximation: bit-window extraction from each tile's circular
code stream, the 3INST computed codebook (exact uint32 + fp16 arithmetic, bit-exact against the CUDA
kernels), the mma-fragment tile ordering, then the fold
``W = diag(suh) . H128 . W_hat . H128 . diag(svh)`` with ``1/sqrt(128)`` per side.

Format reference: exllamav3 v1.4.0 — ``exl3_dq.cuh`` (bit windows), ``pack.cu`` (stream layout),
``codebook.cuh`` (3INST), ``exl3_lib/quantize.py`` (``tensor_core_perm``, ``codebook_scale``),
``reconstruct.cu`` (tile grid, Hadamard fold). Loader-band module: downstream of the frontend a
decoded weight is just a constant.
"""

from __future__ import annotations

import numpy as np

# The full-codebook standard deviation of codebook 0 (exllamav3 quantize.py ``codebook_scale``):
# decoded code values are approximately N(0, CODEBOOK_SCALE^2), so a decoded tile's rms sits near it.
CODEBOOK_SCALE = 1.24371088
# Hadamard block size on both sides; both weight dims are padded to multiples of 128 at encode time.
HAD_BLOCK = 128

_MASK = np.uint32(0x8FFF_8FFF)  # keeps sign + low mantissa bits of the two pseudo-random fp16 halves
_XOR = np.uint32(0x3B60_3B60)  # pins both halves' exponents
_TILE_BATCH = 32768  # tiles per vectorized batch — bounds the uint64 window temp to ~64 MB


def _tile_perm() -> np.ndarray:
    """Trellis step index ``e`` → row-major position in the 16x16 tile (``e = 8*lane + j``).

    The 256 steps follow the mma m16n8k16 B-fragment layout: lane ``l``'s eight consecutive steps
    ARE its B-fragment registers, which is why a GPU decode needs no shuffle at all
    (``tensor_core_perm`` in exllamav3's quantize.py; rows are in-features, cols out-features).
    """
    e = np.arange(256)
    lane, j = e // 8, e % 8
    row = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)
    col = lane // 4 + 8 * (j >> 2)
    return row * 16 + col


_PERM = _tile_perm()  # step → row-major position
_PERM_INV = np.argsort(_PERM)  # row-major position → step


def _check_trellis(trellis: np.ndarray) -> int:
    """Validate a packed trellis tensor and return K (integer bits/weight, 1..8)."""
    if trellis.ndim != 3:
        raise ValueError(f"trellis must be 3-D (k/16, n/16, 16*K), got shape {trellis.shape}")
    if trellis.dtype not in (np.int16, np.uint16):
        raise ValueError(f"trellis must be int16 words, got {trellis.dtype}")
    K, rem = divmod(trellis.shape[-1], 16)
    if rem or not 1 <= K <= 8:
        raise ValueError(f"trellis last dim {trellis.shape[-1]} does not encode an integer K in [1, 8]")
    return K


def _batch_windows(u16: np.ndarray, K: int) -> np.ndarray:
    """``(T, 16*K)`` uint16 packed tiles → ``(T, 256)`` uint16 windows.

    Stream convention (``pack.cu`` / ``exl3_dq.cuh``): consecutive int16 pairs join into
    little-endian uint32s — the odd word of each pair holds the EARLIER bits, an artifact of the
    SWAP16 store — and bits are MSB-first within each uint32. Window ``t`` is stream bits
    ``[(t+1)*K - 16, (t+1)*K) mod 256*K`` (tail-biting: step 0 borrows the stream's last ``16-K``
    bits). A 16-bit window spans at most two uint32 words, so it is one shift out of the uint64
    join of the circularly adjacent word pair.
    """
    u32 = u16[:, 0::2].astype(np.uint64) | (u16[:, 1::2].astype(np.uint64) << np.uint64(16))
    nbits, nwords = 256 * K, 8 * K
    start = (np.arange(1, 257) * K - 16) % nbits  # first stream-bit index of each window
    word = start // 32
    joined = (u32[:, word] << np.uint64(32)) | u32[:, (word + 1) % nwords]
    shift = (np.uint64(48) - (start % 32)).astype(np.uint64)
    return ((joined >> shift) & np.uint64(0xFFFF)).astype(np.uint16)


def trellis_windows(trellis: np.ndarray) -> np.ndarray:
    """Extract every tile's trellis windows: int16 ``(kt, nt, 16*K)`` → uint16 ``(kt, nt, 256)``.

    ``window(t)`` is the 16-bit trellis state whose codebook value is weight step ``t``. The
    tail-biting overlap invariant ``window(t) >> K == window(t-1) mod 2^(16-K)`` holds circularly
    on every valid stream (a useful self-check against real checkpoint data).
    """
    K = _check_trellis(trellis)
    kt, nt = trellis.shape[:2]
    tiles = np.ascontiguousarray(trellis).view(np.uint16).reshape(kt * nt, 16 * K)
    out = np.empty((kt * nt, 256), dtype=np.uint16)
    for lo in range(0, len(tiles), _TILE_BATCH):
        out[lo : lo + _TILE_BATCH] = _batch_windows(tiles[lo : lo + _TILE_BATCH], K)
    return out.reshape(kt, nt, 256)


def pack_trellis(windows: np.ndarray, K: int) -> np.ndarray:
    """Inverse of :func:`trellis_windows` — repack windows into the stored int16 stream.

    Keeps the K fresh (lowest) bits of each window, concatenates them MSB-first into the
    ``256*K``-bit circular stream, and stores each uint32-aligned word pair half-swapped (the
    SWAP16 layout of ``pack_trellis_kernel``). Validation / fixture support: the byte-exact
    roundtrip ``pack_trellis(trellis_windows(t), K) == t`` pins the placement of every bit.
    """
    if windows.shape[-1] != 256:
        raise ValueError(f"windows last dim must be 256, got {windows.shape}")
    if not 1 <= K <= 8:
        raise ValueError(f"K must be in [1, 8], got {K}")
    lead = windows.shape[:-1]
    fresh = windows.reshape(-1, 256).astype(np.uint32) & np.uint32((1 << K) - 1)
    bits = (fresh[:, :, None] >> (K - 1 - np.arange(K))[None, None, :]) & 1
    u32 = (bits.reshape(-1, 8 * K, 32) << (31 - np.arange(32))[None, None, :]).sum(axis=2).astype(np.uint32)
    out = np.empty((u32.shape[0], 16 * K), dtype=np.uint16)
    out[:, 0::2] = (u32 & np.uint32(0xFFFF)).astype(np.uint16)
    out[:, 1::2] = (u32 >> np.uint32(16)).astype(np.uint16)
    return out.view(np.int16).reshape(*lead, 16 * K)


def codebook_values(windows: np.ndarray, cb: int = 0) -> np.ndarray:
    """The 3INST computed codebook: 16-bit windows → fp16 values (``decode_3inst<cb>``).

    No table is stored anywhere — the codebook is exact integer + fp16 arithmetic (all integer
    steps mod 2^32), so CPU reproduction is bit-exact:

    - ``cb=0`` (default): ``x = x*89226354 + 64248484``; mask/xor pins two fp16 halves; value is
      their fp16 sum (round-to-nearest-even, == ``__hadd``).
    - ``cb=1`` (``mcg`` marker): ``x = x*0xCBAC1FED``, then the same mask/xor/half-sum.
    - ``cb=2`` (``mul1`` marker): ``x = x*0x83DCD12D``; ``s = bytesum(x) + 0x6400``; value is
      ``fma_fp16(fp16_bits(s), 0x1EEE, 0xC931)`` — computed in float64, where both product and sum
      are exact, so the single fp16 cast reproduces the fused single rounding of ``__hfma``.
    """
    x = windows.astype(np.uint32)
    if cb == 2:
        x = x * np.uint32(0x83DCD12D)
        s = ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24) + np.uint32(0x6400)).astype(np.uint16)
        h = s.view(np.float16).astype(np.float64)
        k_mul = float(np.array(0x1EEE, dtype=np.uint16).view(np.float16))  # ~1/147.7
        k_add = float(np.array(0xC931, dtype=np.uint16).view(np.float16))  # ~-10.39
        return (h * k_mul + k_add).astype(np.float16)
    if cb == 0:
        x = x * np.uint32(89226354) + np.uint32(64248484)
    elif cb == 1:
        x = x * np.uint32(0xCBAC1FED)
    else:
        raise ValueError(f"unknown codebook id {cb}")
    x = (x & _MASK) ^ _XOR
    lo = (x & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    hi = (x >> np.uint32(16)).astype(np.uint16).view(np.float16)
    return lo + hi  # one fp16 add, round-to-nearest-even == __hadd


def decode_trellis(trellis: np.ndarray, cb: int = 0) -> np.ndarray:
    """Decode packed codes to the hat-basis weights: int16 ``(k/16, n/16, 16*K)`` → fp16 ``(k, n)``.

    Rows are in-features, cols out-features (``reconstruct_kernel``'s output indexing). The result
    is ``W_hat`` — what the fused kernels contract against, and the surface that must match
    exllamav3's own reconstruction bit-exactly; :func:`fold_hadamard` restores the original basis.
    Tiles are independent walks and decode fully vectorized, in batches that bound peak memory.
    """
    K = _check_trellis(trellis)
    kt, nt = trellis.shape[:2]
    tiles = np.ascontiguousarray(trellis).view(np.uint16).reshape(kt * nt, 16 * K)
    out = np.empty((kt * nt, 256), dtype=np.float16)
    for lo in range(0, len(tiles), _TILE_BATCH):
        vals = codebook_values(_batch_windows(tiles[lo : lo + _TILE_BATCH], K), cb)
        out[lo : lo + _TILE_BATCH] = vals[:, _PERM_INV]  # step order → row-major tile order
    return out.reshape(kt, nt, 16, 16).transpose(0, 2, 1, 3).reshape(16 * kt, 16 * nt)


def _sylvester_had(n: int) -> np.ndarray:
    """The natural-order Sylvester Hadamard matrix of size ``n`` (a power of two), float64."""
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def fold_hadamard(w_hat: np.ndarray, suh: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """``W = diag(suh) . H128 . W_hat . H128 . diag(svh)`` — the original-basis fp16 weights.

    ``H128`` is the block-diagonal natural-order Sylvester Hadamard applied per 128-block along
    each dim, scaled ``1/sqrt(128)`` per side; both dims are 128-divisible by construction.
    Computed in float64 with ONE rounding into fp16. The fold is mathematically exact but its
    fp16 rounding is implementation-defined — exllamav3's fused GPU fold rounds per step in fp16
    and its own fused-vs-reference test tolerates 2e-3 relative; the hat-basis decode
    (:func:`decode_trellis`) is the bit-exact surface.
    """
    k, n = w_hat.shape
    if k % HAD_BLOCK or n % HAD_BLOCK:
        raise ValueError(f"weight shape {w_hat.shape} is not a multiple of {HAD_BLOCK} on both dims")
    suh, svh = np.asarray(suh), np.asarray(svh)
    if suh.shape != (k,) or svh.shape != (n,):
        raise ValueError(f"suh/svh shapes {suh.shape}/{svh.shape} do not match weight shape {w_hat.shape}")
    h = _sylvester_had(HAD_BLOCK) / np.sqrt(HAD_BLOCK)
    w = w_hat.astype(np.float64)
    w = (h @ w.reshape(k // HAD_BLOCK, HAD_BLOCK, n)).reshape(k, n)  # H128 . W_hat, per 128 rows
    w = (w.reshape(k, n // HAD_BLOCK, HAD_BLOCK) @ h).reshape(k, n)  # … . H128, per 128 cols
    w *= suh.astype(np.float64)[:, None]
    w *= svh.astype(np.float64)[None, :]
    return w.astype(np.float16)


def decode_exl3_linear(
    trellis: np.ndarray,
    suh: np.ndarray,
    svh: np.ndarray,
    *,
    mcg: np.ndarray | None = None,
    mul1: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct one EXL3 linear's fp16 weight ``(in, out)`` from its sibling tensors.

    ``mcg`` / ``mul1`` are the optional marker siblings: PRESENCE selects codebook 1 / 2 and the
    stored values are never read — pass whatever the checkpoint holds (or ``None``). K comes from
    the trellis shape. The optional ``bias`` sibling is a plain fp16 tensor with nothing to
    decode, so it is not an argument here; the module computes ``y = x @ W (+ bias)``.
    """
    if mcg is not None and mul1 is not None:
        raise ValueError("mcg and mul1 codebook markers are mutually exclusive")
    cb = 1 if mcg is not None else 2 if mul1 is not None else 0
    return fold_hadamard(decode_trellis(trellis, cb), suh, svh)
