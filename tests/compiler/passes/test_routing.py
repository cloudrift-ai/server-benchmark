"""Reference and direct-CUDA contracts for deterministic routing operations."""

import numpy as np

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.tensor.ir import IndexedTopKOp, StableTopKOp


def _stable_graph(rows=2):
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("ranking", (rows, 8), "f32"), node_id="ranking")
    graph.add_node(InputOp(), [], Tensor("payload", (rows, 8), "f32"), node_id="payload")
    graph.add_node(
        StableTopKOp(k=3, scale=1.5),
        ["ranking", "payload"],
        outputs=[Tensor("weights", (rows, 3), "f32"), Tensor("ids", (rows, 3), "i32")],
        node_id="weights",
    )
    graph.inputs, graph.outputs = ["ranking", "payload"], ["weights", "ids"]
    return graph


def _indexed_graph(rows=2):
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("payload", (rows, 16), "f32"), node_id="payload")
    graph.add_node(InputOp(), [], Tensor("table", (4, 3), "i32"), node_id="table")
    graph.add_node(InputOp(), [], Tensor("row_indices", (rows,), "i64"), node_id="row_indices")
    graph.add_node(
        IndexedTopKOp(k=3, scale=1.5, reduction_lanes=4, lane_chunk=2),
        ["payload", "table", "row_indices"],
        outputs=[Tensor("weights", (rows, 3), "f32"), Tensor("ids", (rows, 3), "i32")],
        node_id="weights",
    )
    graph.inputs, graph.outputs = ["payload", "table", "row_indices"], ["weights", "ids"]
    return graph


def test_stable_topk_keeps_lower_id_ties_and_separate_payload_rounding():
    ranking = np.array([[1, 5, 5, 2, 5, 0, 3, 4], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    payload = np.arange(1, 17, dtype=np.float32).reshape(2, 8)
    result, _ = NumpyBackend().run(_stable_graph(), input_data={"ranking": ranking, "payload": payload})

    np.testing.assert_array_equal(result.outputs["ids"], [[1, 2, 4], [0, 1, 2]])
    selected = np.array([[2, 3, 5], [9, 10, 11]], dtype=np.float32)
    expected = selected * (np.float32(1.5) / selected.sum(axis=1, keepdims=True, dtype=np.float32))
    np.testing.assert_array_equal(result.outputs["weights"], expected)


def test_indexed_topk_uses_declared_lane_chunk_reduction_order():
    payload = np.zeros((2, 16), dtype=np.float32)
    payload[0, [1, 6, 9]] = [1e20, 3.0, -1e20]
    payload[1, [2, 3, 15]] = [1.0, 2.0, 5.0]
    table = np.array([[1, 6, 9], [2, 3, 15], [0, 1, 2], [4, 5, 6]], dtype=np.int32)
    result, _ = NumpyBackend().run(
        _indexed_graph(),
        input_data={"payload": payload, "table": table, "row_indices": np.array([0, 1], dtype=np.int64)},
    )

    np.testing.assert_array_equal(result.outputs["ids"], [[1, 6, 9], [2, 3, 15]])
    # Candidate 1 and 9 share lane zero, so their cancellation precedes the
    # XOR-tree addition of candidate 6. A sequential sum would instead be zero.
    np.testing.assert_array_equal(result.outputs["weights"][0], np.array([5e19, 1.5, -5e19], dtype=np.float32))
    np.testing.assert_allclose(result.outputs["weights"][1], [0.1875, 0.375, 0.9375], rtol=0, atol=0)


def test_direct_routing_cuda_lowerings_preserve_ties_guards_and_symbolic_rows():
    stable = CudaBackend(tune_db=None).compile(_stable_graph(Dim("num_tokens", hint=8)))
    [stable_op] = [node.op for node in stable.nodes.values() if isinstance(node.op, CudaOp)]
    assert stable_op.runtime_args == ("num_tokens",)
    assert "value > best_value" in stable_op.kernel_source
    assert "value >= best_value" not in stable_op.kernel_source
    assert stable_op.block == ((1,), (1,), (1,))

    indexed = CudaBackend(tune_db=None).compile(_indexed_graph(Dim("num_tokens", hint=8)))
    [indexed_op] = [node.op for node in indexed.nodes.values() if isinstance(node.op, CudaOp)]
    assert indexed_op.runtime_args == ("num_tokens",)
    assert "table_row < 0 || table_row >= 4" in indexed_op.kernel_source
    assert "candidate >= 0 && candidate < 16" in indexed_op.kernel_source
    assert "__shfl_xor_sync" in indexed_op.kernel_source
    assert indexed_op.block == ((4,), (1,), (1,))
