import numpy as np
import pytest

from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import placeholder
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.loader.physical import (
    PhysicalCarrier,
    PhysicalInputStorage,
    TypedAlgebra,
    spell_physical_inputs,
)
from emmy.compiler.tensor import Tensor


def _graph():
    graph = Graph()
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("x", (2, 4), "f16"), node_id="x")
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("weight", (2, 4), "f16"), node_id="weight")
    graph.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=["x", "weight"],
        output=Tensor("out", (2, 4), "f16"),
        node_id="out",
    )
    graph.inputs = ["x", "weight"]
    graph.outputs = ["out"]
    return graph


def _storage():
    row, col = placeholder(0), placeholder(1)
    return PhysicalInputStorage(
        logical_shape=(2, 4),
        carriers=(
            PhysicalCarrier("values", "", (4, 2), "f8e4m3", (col, row)),
            PhysicalCarrier("multiplier", "_factor", (1, 2), "f16", (row * 0, row)),
        ),
        algebra=(
            TypedAlgebra("decoded", "from_f8e4m3", ("values",), "f16"),
            TypedAlgebra("scaled", "multiply", ("decoded", "multiplier"), "f16"),
        ),
        output="scaled",
    )


def test_spell_physical_input_dissolves_descriptor_and_matches_values():
    from emmy.compiler.backend.numpy.backend import NumpyBackend
    from emmy.compiler.dtype import decode_f8

    graph = _graph()
    assert spell_physical_inputs(graph, {"weight": _storage()}) == {"weight": ("weight", "weight_factor")}
    graph.validate()
    assert graph.inputs == ["x", "weight", "weight_factor"]
    assert graph.nodes["weight"].output.dtype.name == "f8e4m3"
    assert tuple(dim.as_static() for dim in graph.nodes["weight"].output.shape) == (4, 2)

    bits = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.uint8)
    factor = np.array([[2.0, 3.0]], dtype=np.float16)
    x = np.arange(8, dtype=np.float16).reshape(2, 4)
    compiled = NumpyBackend().compile(graph)
    result, _ = NumpyBackend().run(compiled, input_data={"x": x, "weight": bits, "weight_factor": factor})
    expected_weight = decode_f8(bits, "f8e4m3").T.astype(np.float16) * factor.T
    np.testing.assert_array_equal(result.outputs[compiled.outputs[0]], x * expected_weight)


@pytest.mark.parametrize(
    ("storage", "message"),
    [
        (PhysicalInputStorage((3, 4), _storage().carriers, _storage().algebra, "scaled"), "does not reproduce"),
        (PhysicalInputStorage((2, 4), (), (), "missing"), "at least one carrier"),
        (PhysicalInputStorage((2, 4), _storage().carriers, (), "missing"), "output must name"),
        (
            PhysicalInputStorage(
                (2, 4),
                _storage().carriers,
                (TypedAlgebra("bad", "multiply", ("unknown",), "f16"),),
                "bad",
            ),
            "must name earlier",
        ),
    ],
)
def test_spell_physical_input_rejects_invalid_descriptors(storage, message):
    with pytest.raises(ValueError, match=message):
        spell_physical_inputs(_graph(), {"weight": storage})
