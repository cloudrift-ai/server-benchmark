"""Generic constant-subgraph folding: computation folds, storage/layout policy survives."""

from __future__ import annotations

import numpy as np

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.frontend.ir import ReshapeOp, TransposeOp
from emmy.compiler.ir.tensor.ir import CastOp, ElementwiseOp, RangeOp
from emmy.compiler.loader.binder import evaluate_source_graph
from emmy.compiler.pipeline import Pipeline

_RULE = "032_fold_constant_subgraphs"


def _apply(graph: Graph) -> Graph:
    return Pipeline.build(["frontend/decomposition"], select=[_RULE]).run(graph)


def _compute_graph() -> Graph:
    graph = Graph()
    a = graph.add_node(
        op=ConstantOp(name="a", source_path="m.a", source_shape=(4, 4), source_dtype="f32"),
        inputs=[],
        output=Tensor("a", (4, 4), "f32"),
    )
    b = graph.add_node(
        op=ConstantOp(name="b", source_path="m.b", source_shape=(4, 4), source_dtype="f32"),
        inputs=[],
        output=Tensor("b", (4, 4), "f32"),
    )
    graph.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[a, b],
        output=Tensor("out", (4, 4), "f32"),
        node_id="out",
    )
    graph.outputs = ["out"]
    return graph


def _storage_decode_graph() -> Graph:
    graph = Graph()
    bits = graph.add_node(
        op=ConstantOp(name="w", source_path="m.w", source_shape=(4, 4), source_dtype="f8e4m3"),
        inputs=[],
        output=Tensor("w_bits", (4, 4), "f8e4m3"),
    )
    graph.add_node(
        op=ElementwiseOp(op="from_f8e4m3"),
        inputs=[bits],
        output=Tensor("out", (4, 4), "f32"),
        node_id="out",
    )
    graph.outputs = ["out"]
    return graph


def test_compute_cone_folds_to_one_record_constant():
    folded = _apply(_compute_graph())
    assert set(folded.nodes) == {"out"}
    op = folded.nodes["out"].op
    assert isinstance(op, ConstantOp) and op.source_graph is not None
    value = evaluate_source_graph(
        op.source_graph,
        {"m.a": np.full((4, 4), 2.0, np.float32), "m.b": np.full((4, 4), 3.0, np.float32)},
    )
    np.testing.assert_array_equal(value, np.full((4, 4), 6.0, np.float32))


def test_scalar_literals_participate_in_constant_computation():
    graph = Graph()
    source = graph.add_node(
        op=ConstantOp(name="a", source_path="m.a", source_shape=(4,), source_dtype="f32"),
        inputs=[],
        output=Tensor("a", (4,), "f32"),
    )
    scalar = graph.add_node(op=ConstantOp(name="two", value=2), inputs=[], output=Tensor("two", (1,), "f32"))
    scalar_bc = graph.add_node(
        op=ReshapeOp(shape=(1, 1)),
        inputs=[scalar],
        output=Tensor("two_2d", (1, 1), "f32"),
    )
    scalar_flat = graph.add_node(
        op=ReshapeOp(shape=(1,)),
        inputs=[scalar_bc],
        output=Tensor("two_flat", (1,), "f32"),
    )
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

    expanded = broadcast_to(graph, scalar_flat, (4,))
    graph.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[source, expanded],
        output=Tensor("out", (4,), "f32"),
        node_id="out",
    )
    graph.outputs = ["out"]
    folded = _apply(graph)
    value = evaluate_source_graph(folded.nodes["out"].op.source_graph, {"m.a": np.arange(4, dtype=np.float32)})
    np.testing.assert_array_equal(value, np.arange(4, dtype=np.float32) * 2)


def test_storage_decode_cone_stays_expanded():
    graph = _storage_decode_graph()
    before = set(graph.nodes)
    assert set(_apply(graph).nodes) == before


def test_storage_expanding_generic_cast_stays_expanded():
    graph = Graph()
    source = graph.add_node(
        op=ConstantOp(name="words", source_path="m.words", source_shape=(16,), source_dtype="i16"),
        inputs=[],
        output=Tensor("words", (16,), "i16"),
    )
    graph.add_node(op=CastOp(dtype="i64"), inputs=[source], output=Tensor("out", (16,), "i64"), node_id="out")
    graph.outputs = ["out"]
    assert set(_apply(graph).nodes) == {source, "out"}


def test_source_free_range_algebra_still_folds():
    graph = Graph()
    axis = graph.add_node(op=RangeOp(stop=128, dtype="i32"), inputs=[], output=Tensor("axis", (128,), "i32"))
    graph.add_node(op=CastOp(dtype="f64"), inputs=[axis], output=Tensor("factor", (128,), "f64"), node_id="factor")
    graph.outputs = ["factor"]
    folded = _apply(graph)
    assert set(folded.nodes) == {"factor"}
    assert folded.nodes["factor"].op.source_graph is not None


def test_source_free_factor_folds_inside_a_preserved_storage_expansion():
    graph = Graph()
    words = graph.add_node(
        op=ConstantOp(name="words", source_path="m.words", source_shape=(16,), source_dtype="i16"),
        inputs=[],
        output=Tensor("words", (16,), "i16"),
    )
    decoded = graph.add_node(op=CastOp(dtype="f64"), inputs=[words], output=Tensor("decoded", (16,), "f64"))
    axis = graph.add_node(op=RangeOp(stop=16, dtype="i32"), inputs=[], output=Tensor("axis", (16,), "i32"))
    factor = graph.add_node(op=CastOp(dtype="f64"), inputs=[axis], output=Tensor("factor", (16,), "f64"), node_id="factor")
    graph.add_node(op=ElementwiseOp(op="multiply"), inputs=[decoded, factor], output=Tensor("out", (16,), "f64"), node_id="out")
    graph.outputs = ["out"]

    folded = _apply(graph)
    assert folded.nodes["factor"].op.source_graph is not None
    assert folded.nodes["factor"].op.source_path is None
    assert isinstance(folded.nodes["out"].op, ElementwiseOp)
    assert any(op.source_path == "m.words" for _nid, op in folded.loadable_constants())


def test_shared_source_free_branch_folds_once():
    graph = Graph()
    axis = graph.add_node(op=RangeOp(stop=16, dtype="i32"), inputs=[], output=Tensor("axis", (16,), "i32"), node_id="axis")
    one = graph.add_node(op=ConstantOp(name="one", value=1), inputs=[], output=Tensor("one", (1,), "i32"))
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

    one = broadcast_to(graph, one, (16,))
    branch = graph.add_node(
        op=ElementwiseOp(op="add"),
        inputs=[axis, one],
        output=Tensor("branch", (16,), "i32"),
        node_id="branch",
    )
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("x", (16,), "i32"), node_id="x")
    graph.add_node(op=ElementwiseOp(op="add"), inputs=[branch, "x"], output=Tensor("left", (16,), "i32"), node_id="left")
    graph.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[branch, "x"],
        output=Tensor("right", (16,), "i32"),
        node_id="right",
    )
    graph.inputs = ["x"]
    graph.outputs = ["left", "right"]

    folded = _apply(graph)
    assert isinstance(folded.nodes["branch"].op, ConstantOp)
    assert folded.nodes["branch"].op.source_graph is not None
    np.testing.assert_array_equal(evaluate_source_graph(folded.nodes["branch"].op.source_graph, {}), np.arange(16) + 1)


def test_layout_only_cone_stays_for_target_layout_policy():
    graph = Graph()
    source = graph.add_node(
        op=ConstantOp(name="a", source_path="m.a", source_shape=(2, 4), source_dtype="f32"),
        inputs=[],
        output=Tensor("a", (2, 4), "f32"),
    )
    graph.add_node(
        op=TransposeOp(axes=(1, 0)),
        inputs=[source],
        output=Tensor("out", (4, 2), "f32"),
        node_id="out",
    )
    graph.outputs = ["out"]
    assert set(_apply(graph).nodes) == set(graph.nodes)


def test_activation_fed_computation_never_folds():
    graph = Graph()
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,), "f32"), node_id="x")
    scale = graph.add_node(op=ConstantOp(name="s", value=2), inputs=[], output=Tensor("s", (1,), "f32"))
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

    scale = broadcast_to(graph, scale, (4,))
    graph.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=["x", scale],
        output=Tensor("out", (4,), "f32"),
        node_id="out",
    )
    graph.inputs, graph.outputs = ["x"], ["out"]
    assert set(_apply(graph).nodes) == set(graph.nodes)


def test_externally_consumed_interior_declines_fold():
    graph = _compute_graph()
    graph.outputs = ["out", next(nid for nid, node in graph.nodes.items() if node.output.name == "a")]
    assert set(_apply(graph).nodes) == set(graph.nodes)
