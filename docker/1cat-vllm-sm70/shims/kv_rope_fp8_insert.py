# SPDX-License-Identifier: Apache-2.0

"""Public KV-only seam for pinned 1Cat DeepSeek V4 SM70 serving.

This file is installed into 1Cat's ``deepseek_v4.sm70`` package by the Emmy
image layer. Keeping the private Triton launch inside that package lets Emmy
depend only on this explicit six-argument boundary.
"""

import torch
from vllm.models.deepseek_v4.common.ops.cache_utils import quantize_and_insert_k_cache
from vllm.models.deepseek_v4.sm70.qnorm_rope_kv_fp8_insert import (
    _HALF_ROPE,
    _HEAD_DIM,
    _NOPE_DIM,
    _ROPE_DIM,
    _sm70_qnorm_rope_kernel,
)


def sm70_kv_rope_fp8_insert(
    kv: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    block_size: int,
) -> None:
    """Apply GPT-J RoPE to KV, then quantize and insert the paged cache."""
    assert kv.dtype == torch.float16
    assert kv.ndim == 2 and kv.shape[-1] == _HEAD_DIM and kv.is_contiguous()
    assert swa_kv_cache.dtype == torch.uint8 and swa_kv_cache.ndim >= 2 and swa_kv_cache.is_contiguous()
    assert slot_mapping.dtype == torch.int64 and slot_mapping.ndim == 1 and slot_mapping.is_contiguous()
    assert positions.dtype == torch.int64 and positions.shape == (kv.shape[0],) and positions.is_contiguous()
    assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.ndim == 2 and cos_sin_cache.shape[1] == _ROPE_DIM
    assert kv.device == swa_kv_cache.device == slot_mapping.device == positions.device == cos_sin_cache.device
    assert slot_mapping.shape[0] <= kv.shape[0] and block_size > 0

    num_tokens = kv.shape[0]
    kv_roped = torch.empty_like(kv)
    _sm70_qnorm_rope_kernel[(num_tokens, 1)](
        kv,
        kv,
        kv_roped,
        positions,
        cos_sin_cache,
        0.0,
        num_tokens,
        num_heads=0,
        HEAD_DIM=_HEAD_DIM,
        ROPE_DIM=_ROPE_DIM,
        NOPE_DIM=_NOPE_DIM,
        HALF_ROPE=_HALF_ROPE,
        num_warps=4,
    )
    quantize_and_insert_k_cache(
        kv_roped,
        swa_kv_cache.view(swa_kv_cache.shape[0], -1),
        slot_mapping,
        block_size=block_size,
    )
