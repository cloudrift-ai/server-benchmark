"""Synthetic checkpoint builders shared across subsystem tests."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.loader.exl3 import decode_trellis, fold_hadamard, pack_trellis

_rng = np.random.default_rng(7)


def exl3_linear_tensors(base: str, n: int, k: int, K: int = 2, cb: int = 0):
    """Build one EXL3-coded linear and its logical ``(n, k)`` fp16 reference weight."""
    torch = pytest.importorskip("torch")

    n_pad, k_pad = -(-n // 128) * 128, -(-k // 128) * 128
    windows = _rng.integers(0, 1 << 16, (k_pad // 16, n_pad // 16, 256)).astype(np.uint16)
    trellis = pack_trellis(windows, K)
    suh = (_rng.standard_normal(k_pad) * 0.01).astype(np.float16)
    svh = _rng.choice([-1.0, 1.0], n_pad).astype(np.float16)
    tensors = {
        f"{base}.trellis": torch.from_numpy(trellis),
        f"{base}.suh": torch.from_numpy(suh),
        f"{base}.svh": torch.from_numpy(svh),
    }
    if cb == 1:
        tensors[f"{base}.mcg"] = torch.tensor(0x7BAC1FED, dtype=torch.int32)
    elif cb == 2:
        tensors[f"{base}.mul1"] = torch.tensor(0, dtype=torch.int32)
    reference = fold_hadamard(decode_trellis(trellis, cb), suh, svh).T[:n, :k]
    return tensors, reference
