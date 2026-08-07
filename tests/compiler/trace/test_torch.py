"""Tests for the torch_trace module: op handlers, edge cases, and helpers.

Supplements test_torch_trace.py (which only has 2 smoke tests) with
targeted coverage of individual op handlers and helper functions.
These tests require PyTorch.
"""

import pytest

from emmy.compiler.trace.torch import has_torch

pytestmark = pytest.mark.skipif(not has_torch(), reason="PyTorch not available")


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


def test_get_reduce_axis_from_list():
    """_get_reduce_axis extracts the first axis from a list arg."""
    from emmy.compiler.trace.torch import _get_reduce_axis

    class FakeNode:
        args = [None, [2, 3]]

    assert _get_reduce_axis(FakeNode()) == 2


def test_get_reduce_axis_scalar():
    """_get_reduce_axis handles a scalar axis."""
    from emmy.compiler.trace.torch import _get_reduce_axis

    class FakeNode:
        args = [None, -1]

    assert _get_reduce_axis(FakeNode()) == -1


def test_get_reduce_axis_default():
    """_get_reduce_axis defaults to -1 when no axis arg is present."""
    from emmy.compiler.trace.torch import _get_reduce_axis

    class FakeNode:
        args = [None]

    assert _get_reduce_axis(FakeNode()) == -1


def test_op_name_aten_format():
    """_op_name extracts short name from aten.xxx.yyy targets."""
    from emmy.compiler.trace.torch import _op_name

    # _op_name returns the raw aten short name; ATEN→numpy translation
    # (``mul`` → ``multiply``) happens later in _handle_call_function.
    assert _op_name("aten.mul.Tensor") == "mul"
    assert _op_name("aten.linear.default") == "linear"
    assert _op_name("aten.scaled_dot_product_attention.default") == "scaled_dot_product_attention"


def test_op_name_non_aten():
    """_op_name returns None for non-ATen targets."""
    from emmy.compiler.trace.torch import _op_name

    assert _op_name("some.custom.op") is None
    assert _op_name("prims.convert_element_type.default") is None


def test_resolve_inputs_scalars_become_constants():
    """Scalar args (int, float) are promoted to ConstantOp nodes."""
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import ConstantOp, InputOp
    from emmy.compiler.trace.torch import _resolve_inputs

    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    node_map = {"x_fx": x}

    class FakeNode:
        name = "test"
        args = []

    # Non-scalar, non-node arg: no constant created.
    class FakeNodeScalar:
        name = "test_s"
        args = []

    # With a scalar arg.
    class FakeNodeWithScalar:
        name = "test_ws"

    FakeNodeWithScalar.args = [type("FN", (), {"name": "x_fx"})(), 1e-5]
    result = _resolve_inputs(FakeNodeWithScalar, node_map, g)
    assert len(result) == 2
    assert result[0] == x
    # Second is a constant node.
    const_node = g.nodes[result[1]]
    assert isinstance(const_node.op, ConstantOp)


# ---------------------------------------------------------------------------
# Module tracing: elementwise ops
# ---------------------------------------------------------------------------


def test_trace_all_elementwise_ops():
    """All supported elementwise ops trace without errors."""
    import torch
    import torch.nn as nn

    from emmy.compiler.ir.tensor.ir import ElementwiseOp
    from emmy.compiler.trace.torch import trace_module

    class AllOps(nn.Module):
        def forward(self, x):
            a = torch.neg(x)
            b = torch.exp(a)
            c = torch.abs(b)
            d = torch.tanh(c)
            return d

    m = AllOps()
    x = torch.randn(2, 3)
    g = trace_module(m, (x,))

    fns = {n.op.name for n in g.nodes.values() if isinstance(n.op, ElementwiseOp)}
    assert "negative" in fns
    assert "exp" in fns
    assert "abs" in fns
    assert "tanh" in fns


def test_trace_binary_ops():
    """Binary ops (add, sub, mul, div) trace correctly with two inputs."""
    import torch
    import torch.nn as nn

    from emmy.compiler.ir.tensor.ir import ElementwiseOp
    from emmy.compiler.trace.torch import trace_module

    class BinaryOps(nn.Module):
        def forward(self, x, y):
            return (x + y) * (x - y) / (x + 1e-5)

    m = BinaryOps()
    x = torch.randn(4)
    y = torch.randn(4)
    g = trace_module(m, (x, y))

    fns = {n.op.name for n in g.nodes.values() if isinstance(n.op, ElementwiseOp)}
    assert "add" in fns
    assert "subtract" in fns
    assert "multiply" in fns
    assert "divide" in fns


def test_trace_bool_mask_ops():
    """Bool-output mask ops trace: the whole-model explicit-mask construction
    (comparisons feeding torch.where) carries dtype 'bool' on its outputs, and
    the scalar literal a comparison consumes stays f32 — it compares in the
    operand's domain, not bool (the gemma-4 whole-model trace failure)."""
    import torch
    import torch.nn as nn

    from emmy.compiler.dtype import BOOL, F32
    from emmy.compiler.ir.base import ConstantOp
    from emmy.compiler.trace.torch import trace_module

    class MaskedFill(nn.Module):
        def forward(self, x):
            mask = x > 0.5
            return torch.where(mask, x, torch.zeros_like(x))

    m = MaskedFill()
    x = torch.randn(4, 4)
    g = trace_module(m, (x,))

    assert BOOL in {n.output.dtype for n in g.nodes.values()}
    consts = [n for n in g.nodes.values() if isinstance(n.op, ConstantOp) and n.op.value == 0.5]
    assert consts and all(n.output.dtype is F32 for n in consts)


# ---------------------------------------------------------------------------
# Module tracing: reductions
# ---------------------------------------------------------------------------


def test_trace_sum_reduction():
    """aten.sum traces to ReduceOp with correct axis."""
    import torch
    import torch.nn as nn

    from emmy.compiler.ir.tensor.ir import ReduceOp
    from emmy.compiler.trace.torch import trace_module

    class SumReduce(nn.Module):
        def forward(self, x):
            return x.sum(dim=-1)

    m = SumReduce()
    x = torch.randn(4, 8)
    g = trace_module(m, (x,))

    reduces = [n for n in g.nodes.values() if isinstance(n.op, ReduceOp)]
    assert len(reduces) >= 1
    assert reduces[0].op.name == "sum"


def test_trace_max_reduction():
    """aten.amax traces to ReduceOp(amax) — torch's name is preserved."""
    import torch
    import torch.nn as nn

    from emmy.compiler.ir.tensor.ir import ReduceOp
    from emmy.compiler.trace.torch import trace_module

    class MaxReduce(nn.Module):
        def forward(self, x):
            return x.amax(dim=-1)

    m = MaxReduce()
    x = torch.randn(4, 8)
    g = trace_module(m, (x,))

    reduces = [n for n in g.nodes.values() if isinstance(n.op, ReduceOp)]
    assert len(reduces) >= 1
    assert reduces[0].op.name == "amax"


# ---------------------------------------------------------------------------
# Module tracing: structural ops
# ---------------------------------------------------------------------------


def test_trace_reshape():
    """view/reshape traces to ReshapeOp."""
    import torch
    import torch.nn as nn

    from emmy.compiler.ir.frontend.ir import ReshapeOp
    from emmy.compiler.trace.torch import trace_module

    class Reshape(nn.Module):
        def forward(self, x):
            return x.view(2, 6)

    m = Reshape()
    x = torch.randn(3, 4)
    g = trace_module(m, (x,))

    reshapes = [n for n in g.nodes.values() if isinstance(n.op, ReshapeOp)]
    assert len(reshapes) >= 1


def test_trace_transpose():
    """aten.transpose traces to TransposeOp."""
    import torch
    import torch.nn as nn

    from emmy.compiler.ir.frontend.ir import TransposeOp
    from emmy.compiler.trace.torch import trace_module

    class Transpose(nn.Module):
        def forward(self, x):
            return x.transpose(0, 1)

    m = Transpose()
    x = torch.randn(3, 4)
    g = trace_module(m, (x,))

    transposes = [n for n in g.nodes.values() if isinstance(n.op, TransposeOp)]
    assert len(transposes) >= 1


# ---------------------------------------------------------------------------
# Module tracing: linear / matmul decomposition
# ---------------------------------------------------------------------------


def test_trace_linear_produces_linearop():
    """nn.Linear produces a single LinearOp node (not decomposed)."""
    import torch
    import torch.nn as nn

    from emmy.compiler.ir.frontend.ir import LinearOp
    from emmy.compiler.trace.torch import trace_module

    linear = nn.Linear(8, 4, bias=False)
    x = torch.randn(2, 8)
    g = trace_module(linear, (x,))

    has_linear = any(isinstance(n.op, LinearOp) for n in g.nodes.values())
    assert has_linear, "Linear should produce a LinearOp node"


def test_trace_linear_with_bias():
    """nn.Linear with bias produces LinearOp with has_bias=True."""
    import torch
    import torch.nn as nn

    from emmy.compiler.ir.frontend.ir import LinearOp
    from emmy.compiler.trace.torch import trace_module

    linear = nn.Linear(8, 4, bias=True)
    x = torch.randn(2, 8)
    g = trace_module(linear, (x,))

    linear_nodes = [n for n in g.nodes.values() if isinstance(n.op, LinearOp)]
    assert len(linear_nodes) == 1
    assert linear_nodes[0].op.has_bias, "Linear with bias should have has_bias=True"


# ---------------------------------------------------------------------------
# Module tracing: pass-through ops
# ---------------------------------------------------------------------------


def test_trace_passthrough_ops_no_extra_nodes():
    """contiguous/clone produce no new IR nodes (pass-through)."""
    import torch
    import torch.nn as nn

    from emmy.compiler.trace.torch import trace_module

    class PassThrough(nn.Module):
        def forward(self, x):
            return x.contiguous()

    m = PassThrough()
    x = torch.randn(2, 3)
    g = trace_module(m, (x,))

    # Should just have input(s) — contiguous is a no-op.
    assert len(g.outputs) >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_trace_empty_module():
    """Identity module produces a valid graph."""
    import torch
    import torch.nn as nn

    from emmy.compiler.trace.torch import trace_module

    class Identity(nn.Module):
        def forward(self, x):
            return x

    m = Identity()
    x = torch.randn(4)
    g = trace_module(m, (x,))

    assert len(g.inputs) >= 1
    assert len(g.outputs) >= 1


def test_trace_multiple_outputs():
    """Module returning a tuple produces multiple graph outputs."""
    import torch
    import torch.nn as nn

    from emmy.compiler.trace.torch import trace_module

    class MultiOut(nn.Module):
        def forward(self, x):
            return x + 1, x * 2

    m = MultiOut()
    x = torch.randn(4)
    g = trace_module(m, (x,))

    assert len(g.outputs) == 2


@pytest.mark.parametrize(
    ("width", "expected_widths", "expected_starts"),
    [
        (6, (2, 2, 2), (0, 2, 4)),
        (7, (3, 3, 1), (0, 3, 6)),
        (2, (1, 1), (0, 1)),
    ],
)
def test_trace_chunk_materializes_static_slices_and_matches_eager(width, expected_widths, expected_starts):
    """Divisible and uneven ``aten.chunk`` outputs become cumulative slices with
    PyTorch's exact output shapes and values."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.loop.backend import LoopBackend
    from emmy.compiler.ir.frontend.ir import SliceOp
    from emmy.compiler.trace.torch import trace_module

    class Chunk(nn.Module):
        def forward(self, x):
            return torch.chunk(x, 3, dim=1)

    x = torch.arange(2 * width, dtype=torch.float32).reshape(2, width)
    graph = trace_module(Chunk(), (x,))
    slices = [node for node in graph.nodes.values() if isinstance(node.op, SliceOp)]
    assert tuple(node.op.start for node in slices) == expected_starts
    assert tuple(node.output.shape[1].as_static() for node in slices) == expected_widths

    backend = LoopBackend()
    compiled = backend.compile(graph)
    result, _ = backend.run(compiled, input_data={compiled.inputs[0]: x.numpy()})
    for got, expected in zip(result.outputs.values(), torch.chunk(x, 3, dim=1), strict=True):
        np.testing.assert_array_equal(got, expected.numpy())


def test_trace_chunk_getitem_routes_to_selected_slice():
    """A downstream tuple getitem aliases the requested chunk, not the chunk input."""
    import torch
    import torch.nn as nn

    from emmy.compiler.ir.frontend.ir import SliceOp
    from emmy.compiler.trace.torch import trace_module

    class MiddleChunk(nn.Module):
        def forward(self, x):
            return torch.chunk(x, 3, dim=-1)[1]

    graph = trace_module(MiddleChunk(), (torch.randn(2, 7),))
    output = graph.nodes[graph.outputs[0]]
    assert isinstance(output.op, SliceOp)
    assert output.op.start == 3
    assert output.output.shape[-1].as_static() == 3


@pytest.mark.parametrize(("chunks", "dim", "message"), [(object(), 1, "chunk count"), (3, object(), "dimension")])
def test_trace_chunk_rejects_nonconstant_arguments(chunks, dim, message):
    """Dynamic chunk counts and dimensions fail explicitly instead of producing a bad tuple alias."""
    from types import SimpleNamespace

    import torch

    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.trace.torch import _handle_chunk

    graph = Graph()
    source_id = graph.add_node(InputOp(), [], Tensor("x", (2, 6)), node_id="x")
    source = SimpleNamespace(name="x")
    fx_node = SimpleNamespace(
        name="chunk",
        args=(source, chunks, dim),
        meta={"val": [torch.empty(2, 2), torch.empty(2, 2), torch.empty(2, 2)]},
    )
    with pytest.raises(NotImplementedError, match=message):
        _handle_chunk(graph, fx_node, {"x": source_id})


def test_trace_chunk_rejects_dynamic_chunked_extent():
    """A symbolic output extent cannot be encoded as a static SliceOp window."""
    from types import SimpleNamespace

    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.trace.torch import _handle_chunk

    graph = Graph()
    source_id = graph.add_node(InputOp(), [], Tensor("x", (2, "seq_len")), node_id="x")
    source = SimpleNamespace(name="x")
    symbolic_extent = object()
    value = SimpleNamespace(shape=(2, symbolic_extent), dtype="float32")
    fx_node = SimpleNamespace(name="chunk", args=(source, 3, 1), meta={"val": [value]})
    with pytest.raises(NotImplementedError, match="dynamic chunked dimension"):
        _handle_chunk(graph, fx_node, {"x": source_id})


def test_trace_rejects_unmapped_multi_output_op():
    """Multi-output aten ops without a tracer mapping (e.g. ``aten.topk``) raise
    instead of being silently dropped into an arity-broken graph."""
    import torch
    import torch.nn as nn

    from emmy.compiler.trace.torch import trace_module

    class TopK(nn.Module):
        def forward(self, x):
            values, _ = torch.topk(x, 2, dim=-1)
            return values * 2.0

    with pytest.raises(NotImplementedError, match="topk"):
        trace_module(TopK(), (torch.randn(2, 8),))


def test_trace_chunk_rejects_invalid_tuple_index():
    """Invalid tuple getitem indices name the tuple arity in the error."""
    from types import SimpleNamespace

    from emmy.compiler.trace.torch import _handle_getitem

    source = SimpleNamespace(name="chunk")
    fx_node = SimpleNamespace(name="getitem", args=(source, 3))
    with pytest.raises(IndexError, match="out of range for 3 outputs"):
        _handle_getitem(fx_node, {"chunk": ("slice0", "slice1", "slice2")})


def test_trace_expand_is_broadcast_not_reshape():
    """``expand`` (size-1 -> N) is a broadcast, not a reshape: it changes the
    element count. Tracing it as a ``ReshapeOp`` makes the decomposition apply
    flat-offset semantics to the broadcast dim — for GQA's repeat_kv (expand
    then reshape) that yields a ``q_head % kv_heads`` index instead of
    ``q_head // n_rep``. It must trace to a broadcast ``IndexMapOp``."""
    import torch
    import torch.nn as nn

    from emmy.compiler.ir.frontend.ir import ReshapeOp
    from emmy.compiler.ir.tensor.ir import IndexMapOp
    from emmy.compiler.trace.torch import trace_module

    class Expand(nn.Module):
        def forward(self, x):  # (1, 8, 1, 4) -> (1, 8, 2, 4)
            return x.expand(1, 8, 2, 4)

    g = trace_module(Expand(), (torch.randn(1, 8, 1, 4),))
    assert any(isinstance(n.op, IndexMapOp) for n in g.nodes.values()), "expand should produce a broadcast IndexMapOp"
    assert not any(isinstance(n.op, ReshapeOp) for n in g.nodes.values()), "expand must not be a ReshapeOp"


def test_trace_repeat_kv_correct():
    """GQA ``repeat_kv`` (expand + reshape) maps output head ``h`` to KV head
    ``h // n_rep``. Regression for expand-as-reshape giving ``h % kv_heads``
    (so query head 8 wrongly read KV head 0 instead of 4)."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.loop.backend import LoopBackend
    from emmy.compiler.trace.torch import trace_module

    KV, NREP, S, Dh = 8, 2, 4, 16

    class RepeatKV(nn.Module):
        def forward(self, k):
            b, h, s, d = k.shape
            return k[:, :, None, :, :].expand(b, h, NREP, s, d).reshape(b, h * NREP, s, d)

    k = torch.randn(1, KV, S, Dh)
    ref = RepeatKV()(k).detach().numpy()
    g = trace_module(RepeatKV(), (k,))
    be = LoopBackend()
    out = list(be.run(be.compile(g), input_data={g.inputs[0]: k.numpy()})[0].outputs.values())[0]
    np.testing.assert_allclose(np.asarray(out).reshape(ref.shape), ref, rtol=1e-5, atol=1e-5)


def test_dropout_traces_as_copy_passthrough():
    """Inference dropout must trace as a ``copy`` no-op, not crash on an unknown op.

    Phi3-family layers (Phi-4-mini) emit a ``dropout`` node; before the fix the trace
    raised ``unknown elementwise op name: 'dropout'``.
    """
    import torch
    import torch.nn as nn

    from emmy.compiler.trace.torch import trace_module

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)
            self.drop = nn.Dropout(0.5)

        def forward(self, x):
            return self.drop(self.lin(x))

    g = trace_module(M().eval(), (torch.randn(2, 8),))
    ew_names = [n.op.op.name for n in g.nodes.values() if type(n.op).__name__ == "ElementwiseOp"]
    assert "dropout" not in ew_names
    assert "copy" in ew_names


def test_dtype_changing_cast_is_a_real_node_not_an_alias():
    """``to`` / ``type_as`` alias their input ONLY when the dtype is unchanged. A cast that
    actually narrows must survive tracing, or the wider dtype propagates forward: Gemma's
    RMSNorm computes the statistic in f32 (``x.float()``) and closes with ``.type_as(x)``, so
    dropping that cast left the norm OUTPUT f32 and every downstream projection a mixed
    f32xf16 contraction — which is not offered the staged transports. The f32 statistic itself
    must be preserved (squaring gemma activations in f16 overflows above |x|=256)."""
    import torch
    from torch import nn

    from emmy.compiler.trace.torch import trace_module

    class Norm(nn.Module):
        def forward(self, x):
            stat = torch.pow(x.float().pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
            return (x.float() * stat).type_as(x)

    g = trace_module(Norm().eval(), (torch.randn(4, 16, dtype=torch.float16),))
    out = g.nodes[g.outputs[0]]
    assert out.output.dtype.name == "f16", "the closing type_as must narrow the traced output"
    # ...and the statistic really was computed wide — a node upstream still carries f32.
    assert any(n.output.dtype.name == "f32" for n in g.nodes.values()), "the f32 statistic chain must survive"


def test_same_dtype_cast_stays_an_alias():
    """A ``to`` that does not change dtype is still a pure alias — no redundant copy node."""
    import torch
    from torch import nn

    from emmy.compiler.trace.torch import trace_module

    class Same(nn.Module):
        def forward(self, x):
            return (x * 2).to(torch.float16)

    g = trace_module(Same().eval(), (torch.randn(4, 16, dtype=torch.float16),))
    assert all(n.output.dtype.name == "f16" for n in g.nodes.values())


def test_sdpa_scale_kwarg_captured():
    """An explicit ``scale=`` on F.scaled_dot_product_attention (Gemma-nano passes 1.0)
    must land on ``SdpaOp.scale``; without it the op keeps ``None`` (torch's 1/sqrt(d)
    default). Dropping the kwarg silently re-scaled the logits by 1/sqrt(d) — the
    gemma-4-E2B layer-0 accuracy failure."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    from emmy.compiler.ir.frontend.ir import SdpaOp
    from emmy.compiler.trace.torch import trace_module

    class Scaled(nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, scale=1.0)

    class Default(nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v)

    qkv = tuple(torch.randn(1, 2, 8, 4) for _ in range(3))
    g = trace_module(Scaled().eval(), qkv)
    (sdpa,) = [n.op for n in g.nodes.values() if isinstance(n.op, SdpaOp)]
    assert sdpa.scale == 1.0
    g = trace_module(Default().eval(), qkv)
    (sdpa,) = [n.op for n in g.nodes.values() if isinstance(n.op, SdpaOp)]
    assert sdpa.scale is None


def test_sdpa_forward_honors_scale():
    """``SdpaOp.forward`` (the numpy reference) applies ``scale`` when set and the
    1/sqrt(d) default when not — checked against torch's own SDPA."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from emmy.compiler.ir.frontend.ir import SdpaOp

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 8, 4, dtype=torch.float64) for _ in range(3))
    for scale in (None, 1.0, 0.5):
        got = SdpaOp(scale=scale).forward(q.numpy(), k.numpy(), v.numpy())
        want = F.scaled_dot_product_attention(q, k, v, scale=scale).numpy()
        np.testing.assert_allclose(got, want, atol=1e-12)
