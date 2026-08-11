from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import ReshapeOp
from emmy.compiler.specialize import specialize_program


def test_specialize_program_binds_tensor_and_semantic_dimensions():
    rows = Dim("num_tokens", hint=32)
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (rows, 8)), node_id="x")
    graph.add_node(ReshapeOp(shape=(rows * 2, 4)), ["x"], Tensor("y", (rows * 2, 4)), node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y"]

    specialized = specialize_program(graph, {"num_tokens": 16})

    assert specialized.nodes["x"].outputs[0].shape == (16, 8)
    assert specialized.nodes["y"].op.shape == (32, 4)
    assert specialized.nodes["y"].outputs[0].shape == (32, 4)
    assert graph.nodes["x"].outputs[0].shape == (Dim("num_tokens"), 8)


def test_specialize_program_rejects_invalid_binding_values():
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (Dim("num_tokens"),)), node_id="x")
    graph.inputs = graph.outputs = ["x"]

    for invalid in (0, -1, True):
        try:
            specialize_program(graph, {"num_tokens": invalid})
        except ValueError as exc:
            assert "positive integers" in str(exc)
        else:
            raise AssertionError(f"accepted invalid binding {invalid!r}")
