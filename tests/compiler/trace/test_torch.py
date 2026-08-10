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


def test_trace_flatten_is_reshape():
    """aten.flatten changes rank and must not enter the elementwise fallback."""
    import torch
    import torch.nn as nn

    from emmy.compiler.ir.frontend.ir import ReshapeOp
    from emmy.compiler.trace.torch import trace_module

    class FlattenModule(nn.Module):
        def forward(self, x):
            return x.flatten(2)

    g = trace_module(FlattenModule(), (torch.randn(1, 8, 4, 16),))
    reshapes = [n for n in g.nodes.values() if isinstance(n.op, ReshapeOp)]
    assert len(reshapes) == 1
    assert tuple(reshapes[0].output.shape) == (1, 8, 64)


def test_trace_new_zeros_constructs_declared_shape_and_matches_eager():
    """``Tensor.new_zeros`` constructs a tensor; its source is not an elementwise operand."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.ir.base import ConstantOp
    from emmy.compiler.ir.tensor.ir import IndexMapOp
    from emmy.compiler.trace.torch import trace_module

    class NewZeros(nn.Module):
        def forward(self, x):
            return x.new_zeros((x.shape[0], x.shape[1], 8, 16))

    x = torch.randn(1, 3, 4, 32)
    graph = trace_module(NewZeros(), (x,))
    assert tuple(graph.nodes[graph.outputs[0]].output.shape) == (1, 3, 8, 16)
    assert any(isinstance(node.op, ConstantOp) and node.op.value == 0.0 for node in graph.nodes.values())
    assert isinstance(graph.nodes[graph.outputs[0]].op, IndexMapOp)

    backend = NumpyBackend()
    result, _ = backend.run(backend.compile(graph), input_data={graph.inputs[0]: x.numpy()})
    got = next(iter(result.outputs.values()))
    np.testing.assert_array_equal(got, NewZeros()(x).numpy())


def test_trace_new_full_constructs_declared_shape_and_matches_eager():
    """``Tensor.new_full`` uses a static fill scalar, not receiver values."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.ir.base import ConstantOp
    from emmy.compiler.trace.torch import trace_module

    class NewFull(nn.Module):
        def forward(self, x):
            return x.new_full((x.shape[0], x.shape[1], 8, 16), float("-inf"))

    x = torch.randn(1, 3, 4, 32)
    graph = trace_module(NewFull(), (x,))
    fill_nodes = [node for node in graph.nodes.values() if isinstance(node.op, ConstantOp) and node.op.value == float("-inf")]
    assert len(fill_nodes) == 1
    assert tuple(graph.nodes[graph.outputs[0]].output.shape) == (1, 3, 8, 16)

    backend = NumpyBackend()
    result, _ = backend.run(backend.compile(graph), input_data={graph.inputs[0]: x.numpy()})
    got = next(iter(result.outputs.values()))
    np.testing.assert_array_equal(got, NewFull()(x).numpy())


def test_trace_copy_uses_broadcast_source_values_and_matches_eager():
    """``copy_(dest, src)`` produces broadcast source values; destination values are discarded."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.ir.tensor.ir import IndexMapOp
    from emmy.compiler.trace.torch import trace_module

    class Copy(nn.Module):
        def forward(self, template, source):
            return template.new_zeros((1, 3, 8, 16)).copy_(source)

    template = torch.randn(1, 3, 4, 32)
    source = torch.randn(1, 1, 8, 16)
    graph = trace_module(Copy(), (template, source))
    output = graph.nodes[graph.outputs[0]]
    assert isinstance(output.op, IndexMapOp)
    assert tuple(output.output.shape) == (1, 3, 8, 16)

    backend = NumpyBackend()
    input_data = {name: value.numpy() for name, value in zip(graph.inputs, (template, source), strict=True)}
    result, _ = backend.run(backend.compile(graph), input_data=input_data)
    got = next(iter(result.outputs.values()))
    np.testing.assert_array_equal(got, Copy()(template, source).numpy())


def test_trace_rejects_copy_mutation_observed_through_original_base():
    """A functional copy result cannot stand in for a later read through its mutated base."""
    import torch
    import torch.nn as nn

    from emmy.compiler.trace.torch import trace_module

    class CopyThenReadBase(nn.Module):
        def forward(self, template, source):
            base = template.new_zeros((1, 3, 8, 16))
            base[:, 1:].copy_(source)
            return base + 1.0

    template = torch.randn(1, 3, 4, 32)
    source = torch.randn(1, 2, 8, 16)
    with pytest.raises(NotImplementedError, match="observable alias mutation.*original destination"):
        trace_module(CopyThenReadBase(), (template, source))


def test_trace_exports_in_inference_grad_mode_without_higher_order_wrapper():
    """A ``no_grad`` helper must export its tensor ops directly, not as a tuple-valued HOP."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.trace.torch import trace_module

    class NoGradPair(nn.Module):
        @torch.no_grad()
        def trig(self, x):
            return x.cos(), x.sin()

        def forward(self, x):
            return self.trig(x)

    x = torch.randn(2, 3)
    graph = trace_module(NoGradPair(), (x,))
    assert len(graph.outputs) == 2

    backend = NumpyBackend()
    result, _ = backend.run(backend.compile(graph), input_data={graph.inputs[0]: x.numpy()})
    for got, expected in zip(result.outputs.values(), NoGradPair()(x), strict=True):
        np.testing.assert_allclose(got, expected.numpy(), rtol=1e-6, atol=1e-6)


def test_trace_masked_fill_lowers_to_ternary_where_and_matches_eager():
    """Scalar masked fill preserves ``-inf`` exactly; arithmetic selection would create NaNs."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.loop.backend import LoopBackend
    from emmy.compiler.ir.tensor.ir import ElementwiseOp
    from emmy.compiler.trace.torch import trace_module

    class MaskedFill(nn.Module):
        def forward(self, x, mask):
            return x.masked_fill(mask, float("-inf"))

    x = torch.randn(1, 3, 4)
    mask = torch.tensor([[[False, True, False, True]]])
    graph = trace_module(MaskedFill(), (x, mask))
    where_nodes = [node for node in graph.nodes.values() if isinstance(node.op, ElementwiseOp) and node.op.name == "where"]
    assert len(where_nodes) == 1 and where_nodes[0].op.arity == 3

    backend = LoopBackend()
    compiled = backend.compile(graph)
    input_data = {name: value.numpy() for name, value in zip(compiled.inputs, (x, mask), strict=True)}
    result, _ = backend.run(compiled, input_data=input_data)
    got = next(iter(result.outputs.values()))
    np.testing.assert_array_equal(got, MaskedFill()(x, mask).numpy())


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


def test_trace_bmm_produces_matmul_and_matches_eager():
    """aten.bmm is the fixed-rank spelling of the existing batched MatmulOp."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.loop.backend import LoopBackend
    from emmy.compiler.ir.frontend.ir import MatmulOp
    from emmy.compiler.trace.torch import trace_module

    class BatchMatmul(nn.Module):
        def forward(self, a, b):
            return torch.bmm(a, b)

    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 4, 5)
    graph = trace_module(BatchMatmul(), (a, b))
    assert sum(isinstance(node.op, MatmulOp) for node in graph.nodes.values()) == 1

    backend = LoopBackend()
    compiled = backend.compile(graph)
    result, _ = backend.run(
        compiled,
        input_data={compiled.inputs[0]: a.numpy(), compiled.inputs[1]: b.numpy()},
    )
    got = next(iter(result.outputs.values()))
    np.testing.assert_allclose(got, torch.bmm(a, b).numpy(), rtol=1e-5, atol=1e-5)


def test_trace_default_softplus_is_stable_frontend_op():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from emmy.compiler.ir.tensor.ir import ElementwiseOp
    from emmy.compiler.trace.torch import trace_module

    class Softplus(nn.Module):
        def forward(self, x):
            return F.softplus(x)

    graph = trace_module(Softplus(), (torch.randn(2, 8),))
    ops = [node.op.name for node in graph.nodes.values() if isinstance(node.op, ElementwiseOp)]
    assert ops == ["softplus"]


def test_trace_simple_tensor_index_is_gather_and_matches_eager():
    """``table[index]`` is the axis-zero embedding form of GatherOp."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.loop.backend import LoopBackend
    from emmy.compiler.ir.tensor.ir import GatherOp
    from emmy.compiler.trace.torch import trace_module

    class Index(nn.Module):
        def forward(self, table, indices):
            return table[indices]

    table = torch.randn(10, 6)
    indices = torch.tensor([3, 1, 7, 3], dtype=torch.long)
    graph = trace_module(Index(), (table, indices))
    gathers = [node for node in graph.nodes.values() if isinstance(node.op, GatherOp)]
    assert len(gathers) == 1 and gathers[0].op.axis == 0

    backend = LoopBackend()
    compiled = backend.compile(graph)
    result, _ = backend.run(
        compiled,
        input_data={compiled.inputs[0]: table.numpy(), compiled.inputs[1]: indices.numpy()},
    )
    got = next(iter(result.outputs.values()))
    np.testing.assert_array_equal(got, table[indices].numpy())


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


def test_trace_split_with_sizes_materializes_static_slices_and_matches_eager():
    """Non-uniform static splits use the same cumulative-slice representation as chunk."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.loop.backend import LoopBackend
    from emmy.compiler.ir.frontend.ir import SliceOp
    from emmy.compiler.trace.torch import trace_module

    class Split(nn.Module):
        def forward(self, x):
            return torch.split(x, (2, 3, 1), dim=-1)

    x = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    graph = trace_module(Split(), (x,))
    slices = [node for node in graph.nodes.values() if isinstance(node.op, SliceOp)]
    assert tuple(node.op.start for node in slices) == (0, 2, 5)
    assert tuple(node.output.shape[-1].as_static() for node in slices) == (2, 3, 1)

    backend = LoopBackend()
    compiled = backend.compile(graph)
    result, _ = backend.run(compiled, input_data={compiled.inputs[0]: x.numpy()})
    for got, expected in zip(result.outputs.values(), torch.split(x, (2, 3, 1), dim=-1), strict=True):
        np.testing.assert_array_equal(got, expected.numpy())


def test_trace_unbind_materializes_axis_index_maps_and_matches_eager():
    """Unbind results fix the removed input coordinate to each tuple index."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.loop.backend import LoopBackend
    from emmy.compiler.ir.tensor.ir import IndexMapOp
    from emmy.compiler.trace.torch import trace_module

    class Unbind(nn.Module):
        def forward(self, x):
            return torch.unbind(x, dim=1)

    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    graph = trace_module(Unbind(), (x,))
    maps = [node for node in graph.nodes.values() if isinstance(node.op, IndexMapOp)]
    assert len(maps) == 3
    assert all(tuple(node.output.shape) == (2, 4) for node in maps)

    backend = LoopBackend()
    compiled = backend.compile(graph)
    result, _ = backend.run(compiled, input_data={compiled.inputs[0]: x.numpy()})
    for got, expected in zip(result.outputs.values(), torch.unbind(x, dim=1), strict=True):
        np.testing.assert_array_equal(got, expected.numpy())


def test_trace_repeat_interleave_materializes_index_map_and_matches_eager():
    """A scalar repeat maps each output coordinate back by integer division."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.loop.backend import LoopBackend
    from emmy.compiler.ir.tensor.ir import IndexMapOp
    from emmy.compiler.trace.torch import trace_module

    class Repeat(nn.Module):
        def forward(self, x):
            return torch.repeat_interleave(x, 2, dim=-1)

    x = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    graph = trace_module(Repeat(), (x,))
    maps = [node for node in graph.nodes.values() if isinstance(node.op, IndexMapOp)]
    assert len(maps) == 1
    assert tuple(maps[0].output.shape) == (2, 2, 6)

    backend = LoopBackend()
    compiled = backend.compile(graph)
    result, _ = backend.run(compiled, input_data={compiled.inputs[0]: x.numpy()})
    got = next(iter(result.outputs.values()))
    np.testing.assert_array_equal(got, torch.repeat_interleave(x, 2, dim=-1).numpy())


def test_trace_stack_materializes_multi_source_index_map_and_matches_eager():
    """Stack selects one source by the inserted output-axis coordinate."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.loop.backend import LoopBackend
    from emmy.compiler.ir.tensor.ir import IndexMapOp
    from emmy.compiler.trace.torch import trace_module

    class Stack(nn.Module):
        def forward(self, x, y, z):
            return torch.stack((x, y, z), dim=1)

    inputs = tuple(torch.arange(8, dtype=torch.float32).reshape(2, 4) + offset for offset in (0, 10, 20))
    graph = trace_module(Stack(), inputs)
    maps = [node for node in graph.nodes.values() if isinstance(node.op, IndexMapOp)]
    assert len(maps) == 1
    assert tuple(maps[0].output.shape) == (2, 3, 4)

    backend = LoopBackend()
    compiled = backend.compile(graph)
    input_data = {name: value.numpy() for name, value in zip(compiled.inputs, inputs, strict=True)}
    result, _ = backend.run(compiled, input_data=input_data)
    got = next(iter(result.outputs.values()))
    np.testing.assert_array_equal(got, torch.stack(inputs, dim=1).numpy())


def test_trace_max_dim_values_lowers_reduction_and_matches_eager():
    """The values item of max.dim is a normal maximum reduction."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.loop.backend import LoopBackend
    from emmy.compiler.ir.tensor.ir import ReduceOp
    from emmy.compiler.trace.torch import trace_module

    class MaxValues(nn.Module):
        def forward(self, x):
            return x.max(dim=-1, keepdim=True).values

    x = torch.randn(2, 3, 8)
    graph = trace_module(MaxValues(), (x,))
    reduces = [node for node in graph.nodes.values() if isinstance(node.op, ReduceOp)]
    assert len(reduces) == 1

    backend = LoopBackend()
    compiled = backend.compile(graph)
    result, _ = backend.run(compiled, input_data={compiled.inputs[0]: x.numpy()})
    got = next(iter(result.outputs.values()))
    np.testing.assert_allclose(got, x.max(dim=-1, keepdim=True).values.numpy(), rtol=1e-6, atol=1e-6)


def test_trace_max_dim_indices_rejected():
    """Argmax indices need their own stable IR semantics; never alias them to values."""
    import torch
    import torch.nn as nn

    from emmy.compiler.trace.torch import trace_module

    class MaxIndices(nn.Module):
        def forward(self, x):
            return x.max(dim=-1).indices

    with pytest.raises(NotImplementedError, match="argmax indices"):
        trace_module(MaxIndices(), (torch.randn(2, 8),))


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


def test_trace_prunes_output_dead_local_mutation_before_unsupported_op_mapping():
    """An unobserved local scatter branch is not part of the exported function value.

    FX retains the mutating branch as impure, but Emmy must walk only output-live nodes. A live
    topk remains covered by ``test_trace_rejects_unmapped_multi_output_op`` above.
    """
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.trace.torch import trace_module

    class DeadTopKScatter(nn.Module):
        def forward(self, x):
            _values, indices = torch.topk(x, x.shape[-1], dim=-1)
            local = x.new_full((x.shape[0], x.shape[1] + 1), float("-inf"))
            local.scatter_(-1, indices, 0.0)
            local[..., : x.shape[-1]]  # noqa: B018 — deliberately dead descendant
            return x * 2.0

    module = DeadTopKScatter()
    x = torch.randn(2, 8)
    exported = torch.export.export(module, (x,))
    assert any("aten.topk" in str(node.target) for node in exported.graph_module.graph.nodes)

    graph = trace_module(module, (x,))
    assert all("topk" not in node.id for node in graph.nodes.values())
    backend = NumpyBackend()
    result, _ = backend.run(backend.compile(graph), input_data={graph.inputs[0]: x.numpy()})
    np.testing.assert_allclose(next(iter(result.outputs.values())), (x * 2.0).numpy(), rtol=1e-6, atol=1e-6)


def test_trace_keeps_a_mutation_through_a_view_of_the_output():
    """A no-user write remains observable when its receiver aliases the returned tensor."""
    import torch
    import torch.nn as nn

    from emmy.compiler.trace.torch import _output_live_fx_nodes

    class ReturnedAliasMutation(nn.Module):
        def forward(self, x):
            shifted = torch.roll(x, -1, 1)
            shifted[:, -1].fill_(0)
            return shifted

    exported = torch.export.export(ReturnedAliasMutation(), (torch.arange(9).reshape(1, 9),))
    nodes = list(exported.graph_module.graph.nodes)
    live = _output_live_fx_nodes(nodes)
    assert {node.name for node in live} == {"x", "roll", "select", "fill_", "output"}


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


def test_trace_elementwise_maximum_not_reduce():
    """``torch.maximum`` (binary elementwise) maps to ElementwiseOp('maximum'), never to a
    ReduceOp — the reduce spelling of max is ``amax`` and stays a ReduceOp (see
    test_trace_max_reduction)."""
    import torch
    import torch.nn as nn

    from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
    from emmy.compiler.trace.torch import trace_module

    class MinMax(nn.Module):
        def forward(self, x, y):
            return torch.maximum(x, y) + torch.minimum(x, y)

    x, y = torch.randn(4, 8), torch.randn(4, 8)
    g = trace_module(MinMax(), (x, y))
    assert not [n for n in g.nodes.values() if isinstance(n.op, ReduceOp)]
    fns = {n.op.name for n in g.nodes.values() if isinstance(n.op, ElementwiseOp)}
    assert {"maximum", "minimum"} <= fns
    maxi = next(n for n in g.nodes.values() if isinstance(n.op, ElementwiseOp) and n.op.name == "maximum")
    assert len(maxi.inputs) == 2, "binary maximum must keep both operands"


def test_trace_clamp_decomposes_to_min_max_chain():
    """``clamp(min, max)`` decomposes to maximum-then-minimum with the bound constants;
    one-sided ``clamp(max=)`` / ``clamp_min`` / ``clamp_max`` skip the absent side."""
    import torch
    import torch.nn as nn

    from emmy.compiler.ir.tensor.ir import ElementwiseOp
    from emmy.compiler.trace.torch import trace_module

    def _fns(module, n_in=1):
        g = trace_module(module, tuple(torch.randn(3, 4) for _ in range(n_in)))
        return g, [n.op.name for n in g.nodes.values() if isinstance(n.op, ElementwiseOp)]

    class Both(nn.Module):
        def forward(self, x):
            return x.clamp(-7.0, 7.0)

    class MaxOnly(nn.Module):
        def forward(self, x):
            return x.clamp(max=7.0)

    class MinOnly(nn.Module):
        def forward(self, x):
            return x.clamp_min(-1.5)

    g, fns = _fns(Both())
    assert fns == ["maximum", "minimum"]
    _, fns = _fns(MaxOnly())
    assert fns == ["minimum"]
    _, fns = _fns(MinOnly())
    assert fns == ["maximum"]


def test_trace_clamp_matches_torch_eager():
    """End-to-end: the traced clamp chain (the gpt-oss clamped-SwiGLU shape) interprets to
    torch's own values through the numpy reference backend."""
    import numpy as np
    import torch
    import torch.nn as nn

    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.trace.torch import trace_module

    class ClampedSwiglu(nn.Module):
        def forward(self, gate, up):
            gate = gate.clamp(max=7.0)
            up = up.clamp(-7.0, 7.0)
            glu = gate * torch.sigmoid(gate * 1.702)
            return (up + 1) * glu

    torch.manual_seed(0)
    gate, up = torch.randn(4, 16) * 5, torch.randn(4, 16) * 5
    m = ClampedSwiglu()
    g = trace_module(m, (gate, up))
    backend = NumpyBackend()
    outs = backend.run(backend.compile(g), input_data={"gate": gate.numpy(), "up": up.numpy()})[0].outputs
    with torch.no_grad():
        want = m(gate, up).numpy()
    np.testing.assert_allclose(next(iter(outs.values())), want, rtol=1e-5, atol=1e-6)
