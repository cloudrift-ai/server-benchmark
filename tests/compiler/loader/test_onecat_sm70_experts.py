"""Loader-birth contract for 1Cat's retained SM70 expert carriers."""

from dataclasses import dataclass

import torch

from emmy.compiler.loader.onecat_sm70_experts import bind_experts, expert_method_class
from emmy.serving.deepseek_experts import trace_deepseek_experts

_CUDA0 = torch.device("cuda:0")


@dataclass
class _Tensor:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device = torch.device("cuda:0")

    @property
    def ndim(self):
        return len(self.shape)

    @staticmethod
    def is_contiguous():
        return True


def _layer(device=_CUDA0):
    from types import SimpleNamespace

    return SimpleNamespace(
        w13_tm_weight=_Tensor((256, 4096, 64), torch.int32, device),
        w13_tm_scales=_Tensor((256, 128, 512), torch.uint8, device),
        w2_tm_weight=_Tensor((256, 256, 512), torch.int32, device),
        w2_tm_scales=_Tensor((256, 8, 4096), torch.uint8, device),
        sm70_mxfp4_num_experts=256,
        local_num_experts=256,
        global_num_experts=256,
        expert_map=None,
        apply_router_weight_on_input=False,
        swiglu_limit=10.0,
    )


def _inputs(rows, device=_CUDA0):
    return (
        _Tensor((rows, 4096), torch.float16, device),
        _Tensor((rows, 6), torch.float32, device),
        _Tensor((rows, 6), torch.int32, device),
    )


def test_bind_experts_accepts_every_bounded_width_and_returns_named_carriers():
    for rows in (1, 17, 4096):
        binding = bind_experts(_layer(), *_inputs(rows), lambda *_tensors: True)
        assert binding is not None and binding.rows == rows
        assert tuple(name for name, _tensor in binding.carriers) == ("w13", "w13_scale", "w2", "w2_scale")


def test_bind_experts_rejects_shape_and_device_drift():
    layer = _layer()
    layer.w13_tm_weight.shape = (256, 4096, 63)
    assert bind_experts(layer, *_inputs(3), lambda *_tensors: True) is None

    layer = _layer(torch.device("cuda:1"))
    assert bind_experts(layer, *_inputs(3), lambda *_tensors: True) is None


def test_expert_method_class_is_resolved_at_loader_birth():
    from types import SimpleNamespace

    class Expert:
        pass

    assert expert_method_class(SimpleNamespace(Mxfp4SM70MoEMethod=Expert)) is Expert
    assert expert_method_class(SimpleNamespace()) is None


def test_spelling_exposes_only_compact_carriers_at_the_external_abi():
    graph = trace_deepseek_experts(rows=1)

    assert graph.inputs == ["x", "route_weights", "route_ids", "w13", "w2", "w13_scale", "w2_scale"]
    assert not any("mxfp4" in type(node.op).__name__.lower() for node in graph.nodes.values())
