"""Maximal loop fusion is one schedule-blind fixpoint."""

import numpy as np

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.ir.loop import Loop, LoopOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline


def _nests_reduce(loop_op: LoopOp) -> bool:
    reduce_names = loop_op.reduce_axis_names

    def walk(body, inside_reduce: bool) -> bool:
        for stmt in body:
            if isinstance(stmt, Loop):
                is_reduce = stmt.axis.name in reduce_names
                if is_reduce and inside_reduce:
                    return True
                if walk(stmt.body, inside_reduce or is_reduce):
                    return True
        return False

    return walk(loop_op.body, False)


def _chained_matmuls(m=8, k0=4, k1=6, n=5) -> Graph:
    """``(x @ w0) @ w1`` — the smallest graph with nested contractions after fusion."""
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (m, k0)), node_id="x")
    graph.add_node(InputOp(), [], Tensor("w0", (k1, k0)), node_id="w0")
    graph.add_node(InputOp(), [], Tensor("w1", (n, k1)), node_id="w1")
    graph.add_node(LinearOp(), ["x", "w0"], Tensor("h", (m, k1)), node_id="h")
    graph.add_node(LinearOp(), ["h", "w1"], Tensor("y", (m, n)), node_id="y")
    graph.inputs, graph.outputs = ["x", "w0", "w1"], ["y"]
    return graph


def test_chained_matmuls_fuse_into_one_nested_reduction() -> None:
    result = Pipeline.build(LOOP_PASSES).run(_chained_matmuls())
    kernels = [node for node in result.nodes.values() if isinstance(node.op, LoopOp)]
    assert [node.id for node in kernels] == ["y"]
    assert _nests_reduce(kernels[0].op)


def test_nested_reduction_fusion_preserves_numerics() -> None:
    from emmy.compiler.backend.numpy import NumpyBackend

    rng = np.random.default_rng(0)
    inputs = {
        "x": rng.standard_normal((8, 4)).astype(np.float32),
        "w0": rng.standard_normal((6, 4)).astype(np.float32),
        "w1": rng.standard_normal((5, 6)).astype(np.float32),
    }
    backend = NumpyBackend()
    fused = Pipeline.build(LOOP_PASSES).run(_chained_matmuls())
    got = next(iter(backend.run(backend.compile(fused), input_data=inputs)[0].outputs.values()))
    want = (inputs["x"] @ inputs["w0"].T) @ inputs["w1"].T
    np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-5, atol=1e-5)
