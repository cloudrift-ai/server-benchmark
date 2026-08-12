"""Pinned EXL3 CUDA source used by the native serving adapter."""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

_ROOT = Path(__file__).parent / "upstream"
_QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
_SYMBOL_SUFFIX = "EvPK6__halfPS0_S3_S3_S3_PfPPKtPS2_S8_S7_S8_S8_S7_S8_S8_PKlSA_S2_iiiiiifiiiiPi"


def _flatten(path: Path, seen: set[Path]) -> list[str]:
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)
    lines: list[str] = [f"// BEGIN pinned ExLlamaV3 source: {path.relative_to(_ROOT)}"]
    for line in path.read_text().splitlines():
        match = _QUOTED_INCLUDE.match(line)
        if match is not None:
            child = (path.parent / match.group(1)).resolve()
            if child.is_file() and child.is_relative_to(_ROOT):
                lines.extend(_flatten(child, seen))
                continue
        lines.append(line)
    lines.append(f"// END pinned ExLlamaV3 source: {path.relative_to(_ROOT)}")
    return lines


@cache
def source(bits: int, tile_n: int, codebook: int) -> str:
    """Return one self-contained CUDA translation unit for the pinned fused MoE kernel."""
    if bits not in range(1, 9) or tile_n not in (128, 256) or codebook not in (1, 2):
        raise ValueError(f"unsupported EXL3 fused kernel: K={bits}, tile_n={tile_n}, codebook={codebook}")
    lines = _flatten(_ROOT / "quant" / "exl3_moe_kernel.cuh", set())
    lines.extend(
        (
            "typedef void (*emmy_exl3_moe_fn)(EXL3_MOE_KERNEL_ARGS);",
            f"emmy_exl3_moe_fn emmy_keep_exl3_moe() {{ return exl3_moe_kernel<{bits}, {tile_n}, {codebook}>; }}",
            _ROUTE_SOURCE,
        )
    )
    return "\n".join(lines) + "\n"


def symbol(bits: int, tile_n: int, codebook: int) -> str:
    """Itanium ABI name emitted by nvcc for the selected template instantiation."""
    return f"_Z15exl3_moe_kernelILi{bits}ELi{tile_n}ELi{codebook}E{_SYMBOL_SUFFIX}"


_ROUTE_SOURCE = r"""
extern "C" __global__ void emmy_exl3_moe_route_m1(
    const int64_t* indices,
    const half* scores,
    int64_t* expert_count,
    int64_t* token_sorted,
    half* weight_sorted,
    float* output,
    int num_experts,
    int top_k,
    int hidden_dim)
{
    for (int i = threadIdx.x; i < num_experts + 1; i += blockDim.x) expert_count[i] = 0;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) output[i] = 0.0f;
    __syncthreads();
    if (threadIdx.x != 0) return;

    int64_t sorted_indices[16];
    half sorted_scores[16];
    for (int i = 0; i < top_k; ++i)
    {
        sorted_indices[i] = indices[i];
        sorted_scores[i] = scores[i];
    }
    for (int i = 1; i < top_k; ++i)
    {
        int64_t idx = sorted_indices[i];
        half score = sorted_scores[i];
        int j = i - 1;
        while (j >= 0 && sorted_indices[j] > idx)
        {
            sorted_indices[j + 1] = sorted_indices[j];
            sorted_scores[j + 1] = sorted_scores[j];
            --j;
        }
        sorted_indices[j + 1] = idx;
        sorted_scores[j + 1] = score;
    }
    for (int i = 0; i < top_k; ++i)
    {
        int64_t idx = sorted_indices[i];
        if (idx >= 0 && idx < num_experts) ++expert_count[idx];
        token_sorted[i] = 0;
        weight_sorted[i] = sorted_scores[i];
    }
}
"""
