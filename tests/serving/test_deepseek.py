from math import prod

import torch

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.backend.plan import plan_from_graph
from emmy.compiler.target import set_target
from emmy.serving.deepseek import (
    FusedQKvRmsNormModule,
    InverseRopeModule,
    trace_fused_q_kv_rmsnorm,
    trace_inverse_rope,
)


def test_fused_q_kv_rmsnorm_matches_separate_strided_views():
    torch.manual_seed(731)
    fused = torch.randn((4, 24), dtype=torch.float16)
    q_weight = torch.randn((16,), dtype=torch.float16)
    kv_weight = torch.randn((8,), dtype=torch.float16)
    actual_q, actual_kv = FusedQKvRmsNormModule(16).module()(fused, q_weight, kv_weight)

    expected = []
    for value, weight in ((fused[:, :16], q_weight), (fused[:, 16:], kv_weight)):
        value_fp32 = value.float()
        expected.append((value_fp32 * torch.rsqrt((value_fp32 * value_fp32).mean(-1, keepdim=True) + 1e-6) * weight.float()).half())

    torch.testing.assert_close(actual_q, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual_kv, expected[1], rtol=0, atol=0)
    assert actual_q.is_contiguous()
    assert actual_kv.is_contiguous()


def test_inverse_rope_matches_exact_interleaved_reference():
    torch.manual_seed(731)
    x = torch.randn((3, 2, 12), dtype=torch.float16)
    positions = torch.tensor([1, 4, 7], dtype=torch.int64)
    cache = torch.randn((8, 4), dtype=torch.float32)
    actual = InverseRopeModule(4).module()(x, positions, cache)

    expected = x.clone()
    pairs = x[..., -4:].float().reshape(3, 2, 2, 2)
    cos, sin = cache[positions, :2, None], cache[positions, 2:, None]
    expected[..., -4::2] = (pairs[..., 0] * cos.transpose(1, 2) + pairs[..., 1] * sin.transpose(1, 2)).half()
    expected[..., -3::2] = (pairs[..., 1] * cos.transpose(1, 2) - pairs[..., 0] * sin.transpose(1, 2)).half()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_deepseek_dense_boundaries_lower_on_sm70_without_capacity_scratch():
    graphs = (
        trace_fused_q_kv_rmsnorm(rows=2, q_size=16, kv_size=8),
        trace_inverse_rope(rows=2, heads=2, head_dim=12, rope_dim=4, context=8),
    )
    try:
        set_target((7, 0))
        plans = tuple(plan_from_graph(CudaBackend(tune_db=None).compile(graph)) for graph in graphs)
    finally:
        set_target(None)

    q_kv_plan, inverse_plan = plans
    assert q_kv_plan.launches and inverse_plan.launches
    assert len(q_kv_plan.outputs) == 2
    # The current multi-stat lowering may materialize one smaller half between
    # launches; it must never reproduce the full fused projection input.
    scratch = tuple(buffer for buffer in q_kv_plan.buffers if buffer.role == "scratch")
    assert len(scratch) <= 1
    assert all(prod(dim.as_static() for dim in buffer.shape) * buffer.dtype.np.itemsize <= 2 * 8 * 512 for buffer in scratch)
    assert tuple(buffer.name for buffer in inverse_plan.buffers if buffer.role == "scratch") == ()


def test_deepseek_dense_boundaries_share_symbolic_token_extent():
    graphs = (
        trace_fused_q_kv_rmsnorm(rows=8, q_size=16, kv_size=8, dynamic=True),
        trace_inverse_rope(rows=8, heads=2, head_dim=12, rope_dim=4, context=8, dynamic=True),
    )
    for graph in graphs:
        for name in graph.inputs:
            tensor = graph.nodes[name].output
            if name in {"fused_q_kv", "x", "positions"}:
                assert not tensor.shape[0].is_static
                assert tensor.shape[0].as_atom_name() == "num_tokens"
        for name in graph.outputs:
            token_dim = graph.nodes[name].output.shape[0]
            assert not token_dim.is_static
            assert token_dim.as_atom_name() == "num_tokens"
