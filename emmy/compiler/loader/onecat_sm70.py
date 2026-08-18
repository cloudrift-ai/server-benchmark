"""Birth-time integration for 1Cat's retained SM70 projection carriers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

PROFILE_ROWS = (1, 2, 4, 8, 16, 128, 1024, 4096)
PROFILE_CAPACITY = 4096
PROFILE_PINS = {
    "LOOPIFY": "0",
    "RASTER": "",
    "REDUCE": "",
    "STAGE": "d1/smem",
    "TILE": "mma_m8n8k4_f16_f32/f4x4/k8",
    "WORK": "w1x1",
}


@dataclass(frozen=True)
class ProjectionSpec:
    name: str
    n: int
    k: int
    interleave_halves: bool = False
    grouped: bool = False


PROJECTION_SPECS = (
    ProjectionSpec("fused_wqa_wkv", 1536, 4096),
    ProjectionSpec("attention_wq_b_wo_b", 4096, 1024),
    ProjectionSpec("grouped_wo_a", 1024, 4096, grouped=True),
    ProjectionSpec("indexer_wq_b", 8192, 1024),
    ProjectionSpec("shared_gate_up", 512, 4096, interleave_halves=True),
    ProjectionSpec("shared_down", 4096, 256),
)
_SPEC_BY_CONTRACT = {(spec.n, spec.k, spec.interleave_halves, spec.grouped): spec for spec in PROJECTION_SPECS}


@dataclass(frozen=True)
class ProjectionProfile:
    spec: ProjectionSpec
    rows: int
    symbolic: bool = False


@dataclass(frozen=True)
class ProjectionBinding:
    """Format-free serving boundary returned after loader-side validation."""

    profile: ProjectionProfile
    x: Any
    carriers: tuple[tuple[str, Any], ...]
    output_shape: tuple[int, ...]


def expected_inputs() -> tuple[str, ...]:
    return ("x", "weight", "weight_scale")


def projection_graph(profile: ProjectionProfile):
    """Dissolve the retained physical layout into generic graph algebra."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.dtype import F16
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.frontend.ir import LinearOp
    from emmy.compiler.loader.physical import spell_physical_inputs
    from emmy.compiler.loader.sm70_fp8 import expected_sm70_fp8_metadata, retained_sm70_fp8_storage

    spec = profile.spec
    rows = Dim("num_tokens", hint=PROFILE_CAPACITY) if profile.symbolic else profile.rows
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (rows, spec.k), dtype=F16), node_id="x")
    graph.add_node(InputOp(), [], Tensor("weight", (spec.n, spec.k), dtype=F16), node_id="weight")
    graph.add_node(LinearOp(), ["x", "weight"], Tensor("output", (rows, spec.n), dtype=F16), node_id="output")
    graph.inputs = ["x", "weight"]
    graph.outputs = ["output"]
    weight_shape = (1, spec.k, spec.n) if spec.grouped else (spec.k, spec.n)
    scale_shape = (1, spec.k // 128, spec.n) if spec.grouped else (spec.k // 128, spec.n)
    storage = retained_sm70_fp8_storage(
        (spec.n, spec.k),
        weight_shape=weight_shape,
        scale_shape=scale_shape,
        metadata=expected_sm70_fp8_metadata((spec.n, spec.k)),
        interleave_halves=spec.interleave_halves,
        group_index=0 if spec.grouped else None,
    )
    spell_physical_inputs(graph, {"weight": storage})
    graph.validate()
    return graph


def linear_method_class(module: ModuleType | None = None) -> type | None:
    """Return the exact 1Cat method class owned by this birth integration."""
    if module is None:
        module = importlib.import_module("vllm.model_executor.layers.quantization.fp8")
    return getattr(module, "Fp8LinearMethod", None)


def bind_projection(layer: Any, x: Any, bias: Any, platform_supported) -> ProjectionBinding | None:
    """Validate one live carrier export and return a generic named binding."""
    import torch

    try:
        if bias is not None or not bool(layer.sm70_fp8_turbomind):
            return None
        grouped = bool(getattr(layer, "sm70_fp8_bmm", False))
        interleaved = bool(getattr(layer, "sm70_fp8_gated_silu_primary", False))
        if grouped:
            n = int(layer.sm70_fp8_bmm_output_size)
            if int(layer.sm70_fp8_bmm_groups) != 1 or x.ndim != 3 or int(x.shape[-2]) != 1:
                return None
        else:
            n = int(layer.output_size_per_partition)
            if x.ndim != 2:
                return None
        k = int(x.shape[-1])
        spec = _SPEC_BY_CONTRACT.get((n, k, interleaved, grouped))
        if spec is None:
            return None
        rows = int(x.numel() // k)
        profile = ProjectionProfile(spec, rows if rows in PROFILE_ROWS else PROFILE_CAPACITY, symbolic=rows not in PROFILE_ROWS)
        weight_shape = (1, k, n) if grouped else (k, n)
        scale_shape = (1, k // 128, n) if grouped else (k // 128, n)
        weight = layer.weight
        weight_scale = layer.weight_scale_inv
        metadata = layer.sm70_fp8_meta
        supported = (
            0 < rows <= PROFILE_CAPACITY
            and platform_supported(x, weight, weight_scale)
            and x.dtype == torch.float16
            and weight.dtype == torch.float8_e4m3fn
            and weight_scale.dtype == torch.float16
            and tuple(weight.shape) == weight_shape
            and tuple(weight_scale.shape) == scale_shape
            and x.is_contiguous()
            and weight.is_contiguous()
            and weight_scale.is_contiguous()
            and metadata.is_cuda
            and metadata.device == x.device
            and metadata.dtype == torch.int64
            and tuple(metadata.shape) == (2,)
            and metadata.is_contiguous()
            and int(layer.sm70_fp8_k_ld) == 32 * k
            and int(layer.sm70_fp8_q_ld) == n
        )
        if not supported:
            return None
        x_2d = x.reshape(rows, k)
        output_shape = (*x.shape[:-2], 1, n) if grouped else (*x.shape[:-1], n)
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError, RuntimeError):
        return None
    return ProjectionBinding(profile, x_2d, (("weight", weight), ("weight_scale", weight_scale)), output_shape)


__all__ = [
    "PROFILE_CAPACITY",
    "PROFILE_PINS",
    "PROFILE_ROWS",
    "PROJECTION_SPECS",
    "ProjectionBinding",
    "ProjectionProfile",
    "ProjectionSpec",
    "bind_projection",
    "expected_inputs",
    "linear_method_class",
    "projection_graph",
]
