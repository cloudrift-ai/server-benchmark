import math
from types import SimpleNamespace

import pytest

from emmy.compiler.loader.onecat_sm70 import PROJECTION_SPECS, ProjectionProfile, bind_projection, projection_graph


class _Tensor:
    def __init__(self, shape, dtype, *, device="cuda:0", is_cuda=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.is_cuda = is_cuda
        self.ndim = len(self.shape)

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return True

    def reshape(self, *shape):
        return _Tensor(shape, self.dtype, device=self.device, is_cuda=self.is_cuda)


def _layer(torch, spec):
    weight_shape = (1, spec.k, spec.n) if spec.grouped else (spec.k, spec.n)
    scale_shape = (1, spec.k // 128, spec.n) if spec.grouped else (spec.k // 128, spec.n)
    return SimpleNamespace(
        sm70_fp8_turbomind=True,
        sm70_fp8_bmm=spec.grouped,
        sm70_fp8_bmm_groups=1,
        sm70_fp8_bmm_output_size=spec.n,
        sm70_fp8_gated_silu_primary=spec.interleave_halves,
        output_size_per_partition=spec.n,
        sm70_fp8_k_ld=32 * spec.k,
        sm70_fp8_q_ld=spec.n,
        sm70_fp8_meta=_Tensor((2,), torch.int64),
        weight=_Tensor(weight_shape, torch.float8_e4m3fn),
        weight_scale_inv=_Tensor(scale_shape, torch.float16),
    )


def test_all_six_retained_profiles_have_one_generic_physical_graph_contract() -> None:
    assert tuple(spec.name for spec in PROJECTION_SPECS) == (
        "fused_wqa_wkv",
        "attention_wq_b_wo_b",
        "grouped_wo_a",
        "indexer_wq_b",
        "shared_gate_up",
        "shared_down",
    )
    for spec in PROJECTION_SPECS:
        graph = projection_graph(ProjectionProfile(spec, 8))
        assert tuple(graph.inputs) == ("x", "weight", "weight_scale")
        assert graph.nodes["weight"].output.shape == ((1, spec.k, spec.n) if spec.grouped else (spec.k, spec.n))
        assert graph.nodes["weight_scale"].output.shape == ((1, spec.k // 128, spec.n) if spec.grouped else (spec.k // 128, spec.n))


def test_bind_projection_accepts_exact_carriers_and_grouped_shape() -> None:
    torch = pytest.importorskip("torch")
    for spec in PROJECTION_SPECS:
        x_shape = (8, 1, spec.k) if spec.grouped else (8, spec.k)
        x = _Tensor(x_shape, torch.float16)
        layer = _layer(torch, spec)
        binding = bind_projection(layer, x, None, lambda *_args: True)
        assert binding is not None
        assert binding.profile == ProjectionProfile(spec, 8)
        assert binding.x.shape == (8, spec.k)
        assert binding.output_shape == ((8, 1, spec.n) if spec.grouped else (8, spec.n))

        layer.sm70_fp8_q_ld += 1
        assert bind_projection(layer, x, None, lambda *_args: True) is None


def test_bind_projection_requires_cached_metadata_tensor_without_reading_it() -> None:
    torch = pytest.importorskip("torch")
    spec = PROJECTION_SPECS[0]
    x = _Tensor((8, spec.k), torch.float16)
    layer = _layer(torch, spec)

    layer.sm70_fp8_meta = _Tensor((2,), torch.int32)
    assert bind_projection(layer, x, None, lambda *_args: True) is None
    layer.sm70_fp8_meta = _Tensor((2, 1), torch.int64)
    assert bind_projection(layer, x, None, lambda *_args: True) is None


def test_non_manifest_row_uses_one_bounded_symbolic_profile() -> None:
    torch = pytest.importorskip("torch")
    spec = PROJECTION_SPECS[0]
    binding = bind_projection(
        _layer(torch, spec),
        _Tensor((33, spec.k), torch.float16),
        None,
        lambda *_args: True,
    )

    assert binding is not None
    assert binding.profile == ProjectionProfile(spec, 4096, symbolic=True)
