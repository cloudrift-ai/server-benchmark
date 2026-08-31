"""``SdpaOp.forward`` masked fills at reduced precision.

The masked positions fill with -inf through ``np.where``. The earlier spelling subtracted
``mask * 1e9``, and under NumPy 2 promotion the python-float ``1e9`` casts to the SCORES dtype
first — inf in f16 — so ``0 * inf`` poisoned every VISIBLE position with NaN and the whole
attention output was NaN for any MASKED f16 graph (causal or banded; an unmasked SDPA never
touches the fill). The numpy backend is the compiler's parity oracle, so its reference must
stay finite at every dtype a trace promises.
"""

from __future__ import annotations

import numpy as np

from emmy.compiler.ir.frontend.ir import SdpaOp

rng = np.random.default_rng(3)


def _qkv(dtype):
    return tuple((rng.standard_normal((1, 2, 8, 16)) * 0.3).astype(dtype) for _ in range(3))


def test_sdpa_forward_f16_causal_stays_finite_and_matches_f32():
    q, k, v = _qkv(np.float16)
    out = SdpaOp(is_causal=True).forward(q, k, v)
    assert not np.isnan(out).any(), "the causal fill poisoned visible positions"
    ref = SdpaOp(is_causal=True).forward(*(t.astype(np.float32) for t in (q, k, v)))
    np.testing.assert_allclose(out.astype(np.float32), ref, atol=2e-3)


def test_sdpa_forward_f16_sliding_window_stays_finite():
    q, k, v = _qkv(np.float16)
    out = SdpaOp(is_causal=True, sliding_window=2).forward(q, k, v)
    assert not np.isnan(out).any()


def test_sdpa_forward_f32_reference_agrees_with_the_subtractive_fill():
    """At f32 the old ``scores - mask * 1e9`` spelling and the -inf fill agree to f32 rounding:
    both drive the masked exponentials to zero, and no visible score sits near 1e9. Pinned so
    the fill change reads as an f16 repair, not an f32 reference move."""
    q, k, v = _qkv(np.float32)
    out = SdpaOp(is_causal=True).forward(q, k, v)
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = (q @ np.swapaxes(k, -2, -1) * scale).astype(np.float32)
    scores = scores - np.triu(np.ones(scores.shape[-2:], dtype=np.float32), k=1) * 1e9
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    ref = (e / e.sum(axis=-1, keepdims=True)) @ v
    np.testing.assert_allclose(out, ref, rtol=2e-5, atol=1e-7)
