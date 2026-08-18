"""Numerical contract for bounded FixedSinkhornOp tensor-to-loop lifting."""

import numpy as np
import pytest

from emmy.compiler.backend.loop import LoopBackend
from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tensor.ir import FixedSinkhornOp


def _graph(*, batch: int, size: int, iterations: int) -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("logits", (batch, size, size), "f32"), node_id="logits")
    graph.add_node(
        FixedSinkhornOp(eps=1e-6, iterations=iterations),
        ["logits"],
        Tensor("normalized", (batch, size, size), "f32"),
        node_id="normalized",
    )
    graph.inputs = ["logits"]
    graph.outputs = ["normalized"]
    return graph


@pytest.mark.parametrize(("size", "iterations"), [(2, 2), (3, 4)])
def test_fixed_sinkhorn_lifting_preserves_numpy_semantics(size, iterations):
    graph = _graph(batch=2, size=size, iterations=iterations)
    logits = np.random.default_rng(731).normal(size=(2, size, size)).astype(np.float32)
    expected, _ = NumpyBackend().run(graph.copy(), input_data={"logits": logits})

    backend = LoopBackend()
    lowered = backend.compile(graph)
    [loop] = [node.op for node in lowered.nodes.values() if isinstance(node.op, LoopOp)]
    assert len(loop.writes) == size * size
    actual, _ = backend.run(lowered, input_data={"logits": logits})

    np.testing.assert_allclose(actual.outputs["normalized"], expected.outputs["normalized"], rtol=2e-6, atol=2e-7)


def test_fixed_sinkhorn_accepts_symbolic_batch_but_not_symbolic_matrix():
    op = FixedSinkhornOp()
    assert op.matrix_size((Dim("num_tokens", hint=8), 4, 4)) == 4
    with pytest.raises(ValueError, match="static matrix dimensions"):
        op.matrix_size((Dim("num_tokens", hint=8), Dim("rows", hint=4), 4))
