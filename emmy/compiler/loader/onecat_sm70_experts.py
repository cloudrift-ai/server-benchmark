"""Birth-time integration for 1Cat's retained SM70 expert carriers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

EXPERTS = 256
HIDDEN = 4096
INTERMEDIATE = 256
TOP_K = 6
SWIGLU_LIMIT = 10.0


@dataclass(frozen=True)
class ExpertBinding:
    """Format-free serving boundary returned after loader-side validation."""

    rows: int
    x: Any
    weights: Any
    ids: Any
    carriers: tuple[tuple[str, Any], ...]


def spell_expert_inputs(graph) -> None:
    """Dissolve both retained expert stages into generic graph algebra."""
    from emmy.compiler.loader.quant import mxfp4_sm70_mma884_storage, spell_mxfp4_inputs

    spell_mxfp4_inputs(
        graph,
        {
            "w13": mxfp4_sm70_mma884_storage((EXPERTS, 2 * INTERMEDIATE, HIDDEN)),
            "w2": mxfp4_sm70_mma884_storage((EXPERTS, HIDDEN, INTERMEDIATE)),
        },
    )


def expert_method_class(module: ModuleType | None = None) -> type | None:
    """Return the exact 1Cat method class owned by this birth integration."""
    if module is None:
        module = importlib.import_module("vllm.model_executor.layers.quantization.mxfp4_sm70_moe")
    return getattr(module, "Mxfp4SM70MoEMethod", None)


def bind_experts(layer: Any, x: Any, weights: Any, ids: Any, platform_supported) -> ExpertBinding | None:
    """Validate one live retained carrier export and return named generic inputs."""
    import torch

    try:
        rows = int(x.shape[0]) if x.ndim == 2 else -1
        retained = (
            ("w13", layer.w13_tm_weight),
            ("w13_scale", layer.w13_tm_scales),
            ("w2", layer.w2_tm_weight),
            ("w2_scale", layer.w2_tm_scales),
        )
        expected = (
            ((EXPERTS, HIDDEN, 2 * INTERMEDIATE // 8), torch.int32),
            ((EXPERTS, HIDDEN // 32, 2 * INTERMEDIATE), torch.uint8),
            ((EXPERTS, INTERMEDIATE, HIDDEN // 8), torch.int32),
            ((EXPERTS, INTERMEDIATE // 32, HIDDEN), torch.uint8),
        )
        tensors = tuple(tensor for _name, tensor in retained)
        retained_ok = all(
            tuple(tensor.shape) == shape and tensor.dtype == dtype and tensor.device == x.device and tensor.is_contiguous()
            for tensor, (shape, dtype) in zip(tensors, expected, strict=True)
        )
        valid = (
            0 < rows <= 4096
            and platform_supported(x, weights, ids, *tensors)
            and x.dtype == torch.float16
            and weights.dtype == torch.float32
            and ids.dtype == torch.int32
            and tuple(x.shape) == (rows, HIDDEN)
            and tuple(weights.shape) == tuple(ids.shape) == (rows, TOP_K)
            and weights.device == ids.device == x.device
            and x.is_contiguous()
            and weights.is_contiguous()
            and ids.is_contiguous()
            and retained_ok
            and getattr(layer, "sm70_mxfp4_num_experts", None) == EXPERTS
            and getattr(layer, "local_num_experts", None) == EXPERTS
            and getattr(layer, "global_num_experts", None) == EXPERTS
            and getattr(layer, "expert_map", None) is None
            and getattr(layer, "apply_router_weight_on_input", None) is False
            and float(getattr(layer, "swiglu_limit", 0.0)) == SWIGLU_LIMIT
        )
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError, RuntimeError):
        return None
    return ExpertBinding(rows, x, weights, ids, retained) if valid else None


__all__ = ["ExpertBinding", "bind_experts", "expert_method_class", "spell_expert_inputs"]
