"""Tests for the broadcast_to helper and its integration into decomposition.

``broadcast_to`` replaces the old ``002_insert_broadcast_indexmap`` optimization
pass: decomposition rules call the helper directly to wrap smaller-shape
inputs in IndexMapOps, so every ElementwiseOp post-decomposition has all
inputs at the output shape — the rank-preserving Tensor IR invariant.
"""

from dataclasses import replace

import numpy as np

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.expr import Var, placeholder
from emmy.compiler.ir.loop import Accum, Assign, Axis, Load, Loop, LoopOp, Write
from emmy.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp, IndexSource
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

rng = np.random.default_rng(42)
_backend = NumpyBackend()


def _run(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return _backend.run(_backend.compile(graph), input_data=inputs)[0].outputs


# ===================================================================
# broadcast_to — unit tests
# ===================================================================


def test_broadcast_to_is_identity_when_shapes_match():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    out = broadcast_to(g, "x", (4, 8))
    assert out.id == "x", "matching shapes must short-circuit"
    assert len(g.nodes) == 1


def test_broadcast_to_adds_indexmap_for_rank_mismatch():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("y", (8,)), node_id="y")
    out = broadcast_to(g, "y", (4, 8))
    assert out.id != "y"
    assert out.output.shape == (4, 8)
    assert isinstance(out.op, IndexMapOp)


def test_broadcast_to_scalar_to_tensor():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("s", (1,)), node_id="s")
    out = broadcast_to(g, "s", (2, 4, 8))
    assert out.output.shape == (2, 4, 8)
    assert isinstance(out.op, IndexMapOp)


def test_broadcast_to_rejects_non_size_1_mismatch():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("y", (4, 3)), node_id="y")
    try:
        broadcast_to(g, "y", (4, 8))
    except ValueError as e:
        assert "broadcast" in str(e).lower()
    else:
        raise AssertionError("should have raised on non-size-1 dim mismatch")


# ===================================================================
# broadcast_to correctness via numpy backend
# ===================================================================


def test_broadcast_to_preserves_semantics_rank_mismatch():
    """(4, 8) + broadcast_to((8,)) == numpy's (4, 8) + (8,)."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (8,)), node_id="y")
    y_bc = broadcast_to(g, "y", (4, 8))
    g.add_node(ElementwiseOp("add"), ["x", y_bc], Tensor("z", (4, 8)), node_id="z")
    g.inputs, g.outputs = ["x", "y"], ["z"]
    x = rng.standard_normal((4, 8)).astype(np.float32)
    y = rng.standard_normal((8,)).astype(np.float32)
    result = _run(g, {"x": x, "y": y})
    np.testing.assert_allclose(list(result.values())[0], x + y, rtol=1e-6)


def test_broadcast_to_scalar_mul_correctness():
    """Scalar (1,) broadcasts and multiplies element-wise."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2, 4, 8)), node_id="x")
    g.add_node(ConstantOp(name="s", value=2.0), [], Tensor("s", (1,)), node_id="s")
    s_bc = broadcast_to(g, "s", (2, 4, 8))
    g.add_node(ElementwiseOp("multiply"), ["x", s_bc], Tensor("z", (2, 4, 8)), node_id="z")
    g.inputs, g.outputs = ["x"], ["z"]
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    result = _run(g, {"x": x})
    np.testing.assert_allclose(list(result.values())[0], x * 2.0, rtol=1e-6)


def test_broadcast_to_3d_correctness():
    """(2, 4, 8) + broadcast_to((4, 1)) covers the per-row scalar bias shape."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2, 4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (4, 1)), node_id="y")
    y_bc = broadcast_to(g, "y", (2, 4, 8))
    g.add_node(ElementwiseOp("add"), ["x", y_bc], Tensor("z", (2, 4, 8)), node_id="z")
    g.inputs, g.outputs = ["x", "y"], ["z"]
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    y = rng.standard_normal((4, 1)).astype(np.float32)
    result = _run(g, {"x": x, "y": y})
    np.testing.assert_allclose(list(result.values())[0], x + y, rtol=1e-6)


# ===================================================================
# Integration: decomposition emits broadcast-explicit IR
# ===================================================================


def test_tracer_emits_broadcast_explicit_elementwise():
    """After tracing `a + b` with broadcast-requiring shapes, every
    ElementwiseOp in the graph has matched-shape inputs (the tracer wraps
    smaller inputs in IndexMapOps via broadcast_to)."""
    import torch

    from emmy.compiler.ir.tensor.ir import ElementwiseOp
    from emmy.compiler.trace.torch import trace_module

    class BroadcastAdd(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    graph = trace_module(BroadcastAdd(), (torch.randn(4, 8), torch.randn(8)))
    for n in graph.nodes.values():
        if not isinstance(n.op, ElementwiseOp):
            continue
        out_shape = tuple(n.output.shape)
        for inp_id in n.inputs:
            inp_shape = tuple(graph.nodes[inp_id].output.shape)
            assert inp_shape == out_shape, (
                f"ElementwiseOp {n.id} input shape {inp_shape} != output {out_shape} — tracer must insert broadcast IndexMapOp"
            )


# ===================================================================
# Index-map composition — source-program storage boundaries
# ===================================================================


def _public_indexmap_boundary_graph() -> Graph:
    row, inner = Axis("row", 4), Axis("inner", 8)
    reduction = LoopOp(
        body=(
            Loop(
                axis=row,
                body=(
                    Loop(
                        axis=inner,
                        body=(
                            Load(name="value", input="x", index=(Var("row"), Var("inner"))),
                            Accum(name="total", value="value", op="add"),
                        ),
                    ),
                    Write(output="stage", index=(Var("row"),), value="total"),
                ),
            ),
        ),
    )

    def identity() -> IndexMapOp:
        return IndexMapOp(out_shape=(4,), sources=(IndexSource(0, (placeholder(0),)),))

    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (4, 8), F16), node_id="x")
    graph.add_node(reduction, ["x"], Tensor("stage", (4,), F16, transient=True), node_id="stage")
    graph.add_node(identity(), ["stage"], Tensor("rounded", (4,), F16), node_id="rounded")
    graph.add_node(identity(), ["rounded"], Tensor("view", (4,), F16), node_id="view")
    graph.inputs, graph.outputs = ["x"], ["view"]
    return graph


def test_indexmap_composition_preserves_a_public_rounding_boundary() -> None:
    graph = _public_indexmap_boundary_graph()
    values = {"x": rng.standard_normal((4, 8)).astype(np.float16)}
    before = _run(graph, values)

    optimized = Pipeline.build(["frontend/optimization"], select={"compose_indexmaps"}).run(graph)
    assert "rounded" in optimized.nodes
    np.testing.assert_allclose(_run(optimized, values)["view"], before["view"], rtol=0, atol=0)

    lifted = Pipeline.build(["loop/lifting"]).run(optimized)
    rounded = lifted.nodes["rounded"].op
    copies = [stmt.dtype for stmt in rounded.body.iter() if isinstance(stmt, Assign) and stmt.op.name == "copy"]
    assert copies == [F16]


def test_indexmap_composition_still_elides_a_transient_shape_boundary() -> None:
    graph = _public_indexmap_boundary_graph()
    graph.nodes["rounded"].outputs = (replace(graph.nodes["rounded"].output, transient=True),)

    optimized = Pipeline.build(["frontend/optimization"], select={"compose_indexmaps"}).run(graph)

    assert "rounded" not in optimized.nodes
