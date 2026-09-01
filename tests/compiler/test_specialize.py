import numpy as np

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.expr import Literal, placeholder
from emmy.compiler.ir.frontend.ir import ReshapeOp, SliceOp
from emmy.compiler.ir.tensor.ir import IndexMapOp, IndexSource
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


def test_specialize_program_binds_named_frontend_shape_fields_only():
    rows = Dim("num_tokens", hint=32)
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (rows, 4)), node_id="x")
    graph.add_node(ReshapeOp(shape=("num_tokens", 4)), ["x"], Tensor("y", (rows, 4)), node_id="y")
    graph.add_node(
        SliceOp(shape=("num_tokens", 2), dim=1, start=0),
        ["y"],
        Tensor("z", (rows, 2)),
        node_id="z",
    )
    graph.add_node(
        ConstantOp(name="num_tokens", value=1),
        [],
        Tensor("unrelated", (1,), "f32"),
        node_id="unrelated",
    )
    graph.inputs, graph.outputs = ["x"], ["z"]

    specialized = specialize_program(graph, {"num_tokens": 16})
    missing = specialize_program(graph, {"other": 8})

    assert specialized.nodes["y"].op.shape == (16, 4)
    assert specialized.nodes["z"].op.shape == (16, 2)
    assert specialized.nodes["unrelated"].op.name == "num_tokens"
    assert missing.nodes["y"].op.shape == ("num_tokens", 4)
    assert missing.nodes["z"].op.shape == ("num_tokens", 2)


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


def test_specialize_program_keeps_a_coordinate_predicate_out_of_the_extent_range():
    """An unbound symbolic dimension is a tensor EXTENT and is therefore at least 1; a
    ``select`` is a predicate over the output's own coordinates and carries no such floor.
    Reading a coordinate as an extent folds ``out_coord_1 < 1`` to false, which drops the
    first source of every two-source IndexMap — the NVFP4 pair table (``_f4_pair_table``)
    then decodes each byte's high nibble into both halves, and the folded literal is an
    integer where the select is read as a mask.

    Pinned on the values, not on the expression shape: the specialized op must still put
    source 0 in column 0."""
    rows = Dim("num_tokens", hint=32)
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (rows, 4)), node_id="x")
    graph.add_node(InputOp(), [], Tensor("low", (4,)), node_id="low")
    graph.add_node(InputOp(), [], Tensor("high", (4,)), node_id="high")
    graph.add_node(
        IndexMapOp(
            out_shape=(4, 2),
            sources=(
                IndexSource(input_idx=0, coord_map=(placeholder(0),), select=placeholder(1).lt(Literal(1, "int"))),
                IndexSource(input_idx=1, coord_map=(placeholder(0),)),
            ),
        ),
        ["low", "high"],
        Tensor("pairs", (4, 2)),
        node_id="pairs",
    )
    graph.inputs, graph.outputs = ["x", "low", "high"], ["pairs"]

    specialized = specialize_program(graph, {"num_tokens": 16})

    low = np.arange(4, dtype=np.float32)
    high = np.arange(4, dtype=np.float32) + 10
    pairs = specialized.nodes["pairs"].op.forward(low, high)
    assert np.array_equal(pairs, np.stack([low, high], axis=-1))
