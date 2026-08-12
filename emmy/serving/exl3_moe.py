"""Native single-token EXL3 sparse expert execution.

The graph compiler remains the general fallback. This module recognizes the exact compressed
F.linear/SILU expert ABI and launches the pinned fused CUDA implementation without decoding or
staging expert weights.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_POINTER_INPUTS = tuple(f"w_{projection}{leaf}" for projection in ("gate", "up", "down") for leaf in ("", "_suh", "_svh"))


@dataclass(frozen=True)
class Exl3MoeSpec:
    bits: int
    codebook: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    tile_n: int


def fused_m1_spec(inputs, codebooks, *, hidden_size: int, intermediate_size: int, top_k: int, activation) -> Exl3MoeSpec | None:
    """Return the native ABI description, or None when the graph-compiled fallback must run."""
    if top_k < 1 or top_k > 16 or hidden_size % 128 or intermediate_size % 128:
        return None
    if activation.__class__.__name__ not in {"SiLU", "SiLUActivation"}:
        return None
    if any(name not in inputs for name in _POINTER_INPUTS):
        return None
    num_experts = inputs["w_gate"].shape[0]
    if any(t.shape[0] != num_experts for t in inputs.values()):
        return None

    logical = {
        "w_gate": hidden_size * intermediate_size,
        "w_up": hidden_size * intermediate_size,
        "w_down": hidden_size * intermediate_size,
    }
    rates = []
    for name, elements in logical.items():
        codes = inputs[name]
        stored_bits = math.prod(codes.shape[1:]) * codes.element_size() * 8
        rate, remainder = divmod(stored_bits, elements)
        if remainder or rate not in range(1, 9):
            return None
        rates.append(rate)
    if len(set(rates)) != 1:
        return None
    bits = rates[0]
    expected = {
        "w_gate": (hidden_size // 16, intermediate_size // 16, bits * 16),
        "w_up": (hidden_size // 16, intermediate_size // 16, bits * 16),
        "w_down": (intermediate_size // 16, hidden_size // 16, bits * 16),
    }
    for name, shape in expected.items():
        if str(inputs[name].dtype) != "torch.int16" or not inputs[name].is_contiguous() or tuple(inputs[name].shape[1:]) != shape:
            return None
        suh, svh = inputs[f"{name}_suh"], inputs[f"{name}_svh"]
        if str(suh.dtype) != "torch.float16" or not suh.is_contiguous() or tuple(suh.shape[1:]) != (shape[0] * 16,):
            return None
        if str(svh.dtype) != "torch.float16" or not svh.is_contiguous() or tuple(svh.shape[1:]) != (shape[1] * 16,):
            return None

    cbs = {int(codebooks.get(name, 0)) for name in ("w_gate", "w_up", "w_down")}
    if len(cbs) != 1 or (codebook := cbs.pop()) not in (1, 2):
        return None
    tile_n = 256 if hidden_size % 256 == 0 and intermediate_size % 256 == 0 else 128
    return Exl3MoeSpec(bits, codebook, hidden_size, intermediate_size, num_experts, top_k, tile_n)


@dataclass
class Exl3MoePointers:
    """Device pointer tables plus their owning torch tensors."""

    source_tensors: tuple
    torch_tables: tuple
    cupy_tables: tuple


class Exl3MoeM1:
    """One shape group's fused kernel and reusable single-token workspace."""

    _SMEM = 90 * 1024
    _LOCK_INTS = 1024 * 1024 + 2 * 1024 + 2 + 64
    _SMS_PER_EXPERT = 8
    _MAX_SMS_PER_EXPERT = 32

    def __init__(self, spec: Exl3MoeSpec):
        import cupy as cp
        import torch

        from emmy.compiler.backend.cuda import nvcc
        from emmy.serving.native.exl3 import source, symbol

        self.spec = spec
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if props.major < 8:
            raise RuntimeError("the pinned native EXL3 MoE kernel is qualified only on SM80 and newer GPUs")
        cuda_source = source(spec.bits, spec.tile_n, spec.codebook)
        arch = f"sm_{cp.cuda.Device().compute_capability}"
        cubin = nvcc.compile_to_cubin(cuda_source, symbol(spec.bits, spec.tile_n, spec.codebook), arch=arch)
        self._module = cp.RawModule(path=str(cubin))
        self._kernel = self._module.get_function(symbol(spec.bits, spec.tile_n, spec.codebook))
        self._kernel.max_dynamic_shared_size_bytes = self._SMEM
        self._route = self._module.get_function("emmy_exl3_moe_route_m1")

        sms = torch.cuda.get_device_properties().multi_processor_count
        self._concurrency = sms // self._SMS_PER_EXPERT
        self._num_groups = min(self._concurrency, spec.top_k)
        self._group_size = min(sms // self._num_groups, self._MAX_SMS_PER_EXPERT)
        self._block_size = 256 * 32 // 16

        self._expert_count = torch.empty(spec.num_experts + 1, dtype=torch.int64, device="cuda")
        self._token_sorted = torch.empty(spec.top_k, dtype=torch.int64, device="cuda")
        self._weight_sorted = torch.empty(spec.top_k, dtype=torch.float16, device="cuda")
        self.output = torch.empty(1, spec.hidden_size, dtype=torch.float32, device="cuda")
        self._temp_gate = torch.empty(self._concurrency, 1, spec.hidden_size, dtype=torch.float16, device="cuda")
        self._temp_up = torch.empty_like(self._temp_gate)
        self._inter_gate = torch.empty(self._concurrency, 1, spec.intermediate_size, dtype=torch.float16, device="cuda")
        self._inter_up = torch.empty_like(self._inter_gate)
        self._locks = torch.zeros(self._LOCK_INTS, dtype=torch.int32, device="cuda")

        self._expert_count_cp = cp.from_dlpack(self._expert_count)
        self._token_sorted_cp = cp.from_dlpack(self._token_sorted)
        self._weight_sorted_cp = cp.from_dlpack(self._weight_sorted)
        self._output_cp = cp.from_dlpack(self.output)
        self._temp_gate_cp = cp.from_dlpack(self._temp_gate)
        self._temp_up_cp = cp.from_dlpack(self._temp_up)
        self._inter_gate_cp = cp.from_dlpack(self._inter_gate)
        self._inter_up_cp = cp.from_dlpack(self._inter_up)
        self._locks_cp = cp.from_dlpack(self._locks)
        self._reported = False

    def pointer_tables(self, inputs) -> Exl3MoePointers:
        import cupy as cp
        import torch

        sources = tuple(inputs[name] for name in _POINTER_INPUTS)
        owners = tuple(
            torch.tensor(
                [tensor.data_ptr() + expert * tensor.stride(0) * tensor.element_size() for expert in range(self.spec.num_experts)],
                dtype=torch.int64,
                device=tensor.device,
            )
            for name in _POINTER_INPUTS
            for tensor in (inputs[name],)
        )
        return Exl3MoePointers(sources, owners, tuple(cp.from_dlpack(table) for table in owners))

    def __call__(self, x, scores, indices, pointers: Exl3MoePointers):
        import cupy as cp
        import numpy as np

        if x.shape != (1, self.spec.hidden_size) or str(x.dtype) != "torch.float16":
            raise ValueError(f"native EXL3 MoE needs one fp16 row of width {self.spec.hidden_size}")
        if scores.shape != indices.shape or scores.numel() != self.spec.top_k or str(scores.dtype) != "torch.float16":
            raise ValueError("native EXL3 MoE needs fixed fp16 top-k scores and matching indices")
        if not self._reported:
            logger.info(
                "native EXL3 MoE decode active (E=%d, top-k=%d, K%d, codebook %d)",
                self.spec.num_experts,
                self.spec.top_k,
                self.spec.bits,
                self.spec.codebook,
            )
            self._reported = True

        x_cp = cp.from_dlpack(x.detach().contiguous())
        scores_cp = cp.from_dlpack(scores.detach().contiguous())
        indices_cp = cp.from_dlpack(indices.detach().contiguous())
        self._route(
            (1,),
            (256,),
            (
                indices_cp,
                scores_cp,
                self._expert_count_cp,
                self._token_sorted_cp,
                self._weight_sorted_cp,
                self._output_cp,
                np.int32(self.spec.num_experts),
                np.int32(self.spec.top_k),
                np.int32(self.spec.hidden_size),
            ),
        )
        self._kernel(
            (self._group_size, 1, self._num_groups),
            (self._block_size,),
            (
                x_cp,
                self._temp_gate_cp,
                self._temp_up_cp,
                self._inter_gate_cp,
                self._inter_up_cp,
                self._output_cp,
                *pointers.cupy_tables,
                self._expert_count_cp,
                self._token_sorted_cp,
                self._weight_sorted_cp,
                np.int32(self.spec.hidden_size),
                np.int32(self.spec.intermediate_size),
                np.int32(self.spec.num_experts),
                np.int32(self.spec.top_k),
                np.int32(1),
                np.int32(self._num_groups),
                np.float32(0.0),
                np.int32(0),
                np.int32(self.spec.bits),
                np.int32(self.spec.bits),
                np.int32(self.spec.bits),
                self._locks_cp,
            ),
            shared_mem=self._SMEM,
        )
        return self.output
