import numpy as np
import pytest
import torch

from emmy.serving.mhc import (
    HcHeadModule,
    MhcBroadcastModule,
    MhcFusedModule,
    MhcPostModule,
    MhcPreModule,
    fixed_sinkhorn,
    trace_hc_head,
    trace_mhc_broadcast,
    trace_mhc_fused,
    trace_mhc_post,
    trace_mhc_pre,
)


@pytest.fixture
def mhc_inputs():
    torch.manual_seed(731)
    tokens, streams, hidden = 2, 4, 16
    return {
        "x": torch.randn((tokens, hidden), dtype=torch.float16),
        "residual": torch.randn((tokens, streams, hidden), dtype=torch.float16),
        "post": torch.randn((tokens, streams, 1), dtype=torch.float32),
        "comb": torch.randn((tokens, streams, streams), dtype=torch.float32),
        "fn": torch.randn((24, streams * hidden), dtype=torch.float32),
        "fn_broadcast": torch.randn((24, hidden), dtype=torch.float32),
        "scale": torch.randn((3,), dtype=torch.float32),
        "base": torch.randn((24,), dtype=torch.float32),
        "norm_weight": torch.randn((hidden,), dtype=torch.float16),
        "head_fn": torch.randn((streams, streams * hidden), dtype=torch.float32),
        "head_scale": torch.randn((1,), dtype=torch.float32),
        "head_base": torch.randn((streams,), dtype=torch.float32),
    }


def _reference_mix(residual, fn, scale, base, *, prenorm_residual=None):
    flat = (prenorm_residual if prenorm_residual is not None else residual).float().flatten(1)
    flat = flat * torch.rsqrt((flat * flat).mean(dim=-1, keepdim=True) + 1e-6)
    logits = torch.nn.functional.linear(flat, fn)
    return _reference_mix_from_logits(residual, logits, scale, base)


def _reference_mix_from_logits(residual, logits, scale, base):
    tokens, streams, _ = residual.shape
    pre_logits, post_logits, comb_logits = logits.split((streams, streams, streams * streams), dim=-1)
    pre_base, post_base, comb_base = base.split((streams, streams, streams * streams), dim=-1)
    pre = torch.sigmoid(pre_logits * scale[0] + pre_base) + 1e-6
    post = 2 * torch.sigmoid(post_logits * scale[1] + post_base)
    comb = torch.softmax(comb_logits.view(tokens, streams, streams) * scale[2] + comb_base.view(streams, streams), -1) + 1e-6
    comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
    collapsed = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).half()
    return post.unsqueeze(-1), comb, collapsed


def _reference_weighted_rms(x, weight):
    x_float = x.float()
    return (x_float * torch.rsqrt((x_float * x_float).mean(dim=-1, keepdim=True) + 1e-6)).half() * weight


def _reference_fixed_sinkhorn(logits, eps=1e-6, iterations=20):
    values = torch.softmax(logits, dim=-1) + eps
    values = values / (values.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        values = values / (values.sum(dim=-1, keepdim=True) + eps)
        values = values / (values.sum(dim=-2, keepdim=True) + eps)
    return values


def test_fixed_sinkhorn_matches_exact_eager_order_and_rejects_unbounded_inputs():
    logits = torch.randn((2, 4, 4), dtype=torch.float32)
    torch.testing.assert_close(fixed_sinkhorn(logits), _reference_fixed_sinkhorn(logits), rtol=0, atol=0)

    with pytest.raises(TypeError, match="float32"):
        fixed_sinkhorn(logits.half())
    with pytest.raises(ValueError, match="square"):
        fixed_sinkhorn(torch.empty((1, 3, 4), dtype=torch.float32))
    with pytest.raises(ValueError, match=r"\[1,8\]"):
        fixed_sinkhorn(torch.empty((1, 9, 9), dtype=torch.float32))
    with pytest.raises(ValueError, match=r"\[1,32\]"):
        fixed_sinkhorn(logits, iterations=33)


def test_fixed_sinkhorn_trace_and_tensor_ir_preserve_the_fp32_contract():
    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.ir.tensor.ir import FixedSinkhornOp
    from emmy.compiler.trace.torch import trace_module

    class Module(torch.nn.Module):
        def forward(self, logits):
            return fixed_sinkhorn(logits, eps=2e-6, iterations=4)

    graph = trace_module(Module(), (torch.empty((2, 3, 3), dtype=torch.float32, device="meta"),))
    [node] = [node for node in graph.nodes.values() if isinstance(node.op, FixedSinkhornOp)]
    assert node.op == FixedSinkhornOp(eps=2e-6, iterations=4)
    assert tuple(dim.as_static() for dim in node.output.shape) == (2, 3, 3)
    assert node.output.dtype.name == "f32"

    rng = np.random.default_rng(731)
    logits = rng.normal(size=(2, 3, 3)).astype(np.float32)
    actual, _ = NumpyBackend().run(graph, input_data={"logits": logits})
    expected = _reference_fixed_sinkhorn(torch.from_numpy(logits), eps=2e-6, iterations=4).numpy()
    np.testing.assert_allclose(actual.outputs[graph.outputs[0]], expected, rtol=2e-6, atol=2e-7)


def test_fixed_sinkhorn_lowers_to_one_straight_line_sm70_kernel():
    from emmy.compiler.context import Context
    from emmy.compiler.ir.cuda import CudaOp
    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
    from emmy.compiler.trace.torch import trace_module

    class Module(torch.nn.Module):
        def forward(self, logits):
            return fixed_sinkhorn(logits, iterations=2)

    graph = trace_module(Module(), (torch.empty((2, 2, 2), dtype=torch.float32, device="meta"),))
    loop_graph = Pipeline.build(LOOP_PASSES).run(graph.copy(), ctx=Context.from_target((7, 0)))
    [loop] = [node.op for node in loop_graph.nodes.values() if isinstance(node.op, LoopOp)]
    assert len(loop.writes) == 4

    cuda_graph = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((7, 0)))
    [cuda] = [node.op for node in cuda_graph.nodes.values() if isinstance(node.op, CudaOp)]
    source = cuda.kernel_source
    assert "fmaxf" in source and "expf" in source
    assert "if (_gid < 2)" in source
    assert "for (int" not in source
    assert "__syncthreads" not in source


def test_mhc_pre_matches_exact_fp32_state_algebra(mhc_inputs):
    values = mhc_inputs
    actual = MhcPreModule()(values["residual"], values["fn"], values["scale"], values["base"], values["norm_weight"])
    post, comb, collapsed = _reference_mix(values["residual"], values["fn"], values["scale"], values["base"])

    torch.testing.assert_close(actual[0], post, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], comb, rtol=0, atol=0)
    torch.testing.assert_close(actual[2], _reference_weighted_rms(collapsed, values["norm_weight"]), rtol=0, atol=0)
    assert actual[0].dtype == actual[1].dtype == torch.float32
    assert actual[2].dtype == torch.float16


def test_mhc_fused_and_post_share_the_same_stream_update(mhc_inputs):
    values = mhc_inputs
    residual_float = values["x"].float().unsqueeze(1) * values["post"] + torch.bmm(
        values["comb"].transpose(1, 2), values["residual"].float()
    )
    residual = residual_float.half()
    torch.testing.assert_close(
        MhcPostModule()(values["x"], values["residual"], values["post"], values["comb"]),
        residual,
        rtol=0,
        atol=0,
    )
    actual = MhcFusedModule()(
        values["x"],
        values["residual"],
        values["post"],
        values["comb"],
        values["fn"],
        values["scale"],
        values["base"],
        values["norm_weight"],
    )
    post, comb, collapsed = _reference_mix(
        residual,
        values["fn"],
        values["scale"],
        values["base"],
        prenorm_residual=residual_float,
    )

    torch.testing.assert_close(actual[0], residual, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], post, rtol=0, atol=0)
    torch.testing.assert_close(actual[2], comb, rtol=0, atol=0)
    torch.testing.assert_close(actual[3], _reference_weighted_rms(collapsed, values["norm_weight"]), rtol=0, atol=0)

    rounded_post, rounded_comb, _ = _reference_mix(residual, values["fn"], values["scale"], values["base"])
    assert not torch.equal(actual[1], rounded_post)
    assert not torch.equal(actual[2], rounded_comb)

    prefill = MhcFusedModule(fp32_stage=False)(
        values["x"],
        values["residual"],
        values["post"],
        values["comb"],
        values["fn"],
        values["scale"],
        values["base"],
        values["norm_weight"],
    )
    torch.testing.assert_close(prefill[1], rounded_post, rtol=0, atol=0)
    torch.testing.assert_close(prefill[2], rounded_comb, rtol=0, atol=0)


def test_mhc_broadcast_and_head_preserve_live_dtypes(mhc_inputs):
    values = mhc_inputs
    residual, post, comb, normalized = MhcBroadcastModule()(
        values["x"], values["fn_broadcast"], values["scale"], values["base"], values["norm_weight"]
    )
    expected_residual = values["x"].unsqueeze(1).expand(-1, 4, -1).contiguous()
    x_float = values["x"].float()
    normalized_x = x_float * torch.rsqrt((x_float * x_float).mean(dim=-1, keepdim=True) + 1e-6)
    expected_post, expected_comb, collapsed = _reference_mix_from_logits(
        expected_residual,
        torch.nn.functional.linear(normalized_x, values["fn_broadcast"]),
        values["scale"],
        values["base"],
    )
    torch.testing.assert_close(residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(post, expected_post, rtol=0, atol=0)
    torch.testing.assert_close(comb, expected_comb, rtol=0, atol=0)
    torch.testing.assert_close(normalized, _reference_weighted_rms(collapsed, values["norm_weight"]), rtol=0, atol=0)

    head = HcHeadModule()(residual, values["head_fn"], values["head_scale"], values["head_base"])
    flat = residual.float().flatten(1)
    normalized_flat = flat * torch.rsqrt((flat * flat).mean(dim=-1, keepdim=True) + 1e-6)
    mix = torch.sigmoid(torch.nn.functional.linear(normalized_flat, values["head_fn"]) * values["head_scale"] + values["head_base"]) + 1e-6
    expected_head = (mix.unsqueeze(-1) * residual.float()).sum(dim=1).half()
    torch.testing.assert_close(head, expected_head, rtol=0, atol=0)
    assert (residual.dtype, post.dtype, comb.dtype, normalized.dtype, head.dtype) == (
        torch.float16,
        torch.float32,
        torch.float32,
        torch.float16,
        torch.float16,
    )


def test_live_mhc_traces_preserve_fp32_parameter_contracts():
    broadcast = trace_mhc_broadcast(rows=1)
    pre = trace_mhc_pre(rows=1)
    fused = trace_mhc_fused(rows=1)
    post = trace_mhc_post(rows=1)
    head = trace_hc_head(rows=1)

    assert broadcast.buffer("fn_broadcast").dtype.name == "f32"
    assert pre.buffer("fn").dtype.name == "f32"
    assert fused.buffer("fn").dtype.name == "f32"
    assert fused.buffer("residual").dtype.name == "f16"
    assert fused.buffer("post").dtype.name == "f32"
    assert post.buffer("comb").dtype.name == "f32"
    assert head.buffer("fn").dtype.name == "f32"


def test_mhc_prefill_traces_share_symbolic_token_extent_across_live_state():
    graphs = (
        trace_mhc_broadcast(rows=128, hidden=16, dynamic=True),
        trace_mhc_pre(rows=128, hidden=16, dynamic=True),
        trace_mhc_fused(rows=128, hidden=16, dynamic=True),
        trace_mhc_post(rows=128, hidden=16, dynamic=True),
        trace_hc_head(rows=128, hidden=16, dynamic=True),
    )
    for graph in graphs:
        for name in graph.outputs:
            token_dim = graph.nodes[name].output.shape[0]
            assert not token_dim.is_static
            assert token_dim.as_atom_name() == "num_tokens"


def test_wide_mhc_traces_spell_fixed_stream_sums_as_pointwise_work():
    from emmy.compiler.ir.frontend.ir import MatmulOp
    from emmy.compiler.ir.tensor.ir import ReduceOp

    def fixed_stream_reductions(graph):
        def is_fixed_stream_reduction(node):
            return isinstance(node.op, MatmulOp) or (isinstance(node.op, ReduceOp) and node.op.axis == 1)

        return [node for node in graph.nodes.values() if is_fixed_stream_reduction(node)]

    assert fixed_stream_reductions(trace_mhc_post(rows=1, hidden=16))
    assert not fixed_stream_reductions(trace_mhc_post(rows=128, hidden=16))
    assert fixed_stream_reductions(trace_mhc_fused(rows=1, hidden=16))
    assert not fixed_stream_reductions(trace_mhc_fused(rows=128, hidden=16))


def test_wide_mhc_pointwise_stream_algebra_matches_the_reference_modules(mhc_inputs):
    values = mhc_inputs
    cases = (
        (
            MhcPreModule(),
            MhcPreModule(pointwise_streams=True),
            (values["residual"], values["fn"], values["scale"], values["base"], values["norm_weight"]),
        ),
        (
            MhcBroadcastModule(),
            MhcBroadcastModule(pointwise_streams=True),
            (values["x"], values["fn_broadcast"], values["scale"], values["base"], values["norm_weight"]),
        ),
        (
            MhcPostModule(),
            MhcPostModule(pointwise_streams=True),
            (values["x"], values["residual"], values["post"], values["comb"]),
        ),
        (
            MhcFusedModule(),
            MhcFusedModule(pointwise_streams=True),
            (
                values["x"],
                values["residual"],
                values["post"],
                values["comb"],
                values["fn"],
                values["scale"],
                values["base"],
                values["norm_weight"],
            ),
        ),
        (
            HcHeadModule(),
            HcHeadModule(pointwise_streams=True),
            (values["residual"], values["head_fn"], values["head_scale"], values["head_base"]),
        ),
    )
    for reference, pointwise, inputs in cases:
        expected = reference(*inputs)
        actual = pointwise(*inputs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        actual = actual if isinstance(actual, tuple) else (actual,)
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=2e-6, atol=2e-7)
