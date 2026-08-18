"""Reference and direct-CUDA contracts for deterministic routing operations."""

import numpy as np

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.tensor.ir import ExpertBucketOp, IndexedTopKOp, RouteUnbucketOp, StableTopKOp, WeightedRouteSumOp


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


def _bucket_graph():
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("ids", (3, 2), "i32"), node_id="ids")
    graph.add_node(
        ExpertBucketOp(experts=4, routes=2, rows_per_group=2),
        ["ids"],
        outputs=[
            Tensor("grouped_routes", (5, 2), "i32"),
            Tensor("group_experts", (5,), "i32"),
            Tensor("inverse", (3, 2), "i32"),
        ],
        node_id="grouped_routes",
    )
    graph.inputs, graph.outputs = ["ids"], ["grouped_routes", "group_experts", "inverse"]
    return graph


def _unbucket_graph():
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("base", (6, 2), "f16"), node_id="base")
    graph.add_node(InputOp(), [], Tensor("grouped", (2, 2, 2), "f16"), node_id="grouped")
    graph.add_node(InputOp(), [], Tensor("inverse", (3, 2), "i32"), node_id="inverse")
    graph.add_node(
        RouteUnbucketOp(rows_per_group=2, shard_index=1),
        ["base", "grouped", "inverse"],
        Tensor("output", (6, 2), "f16"),
        node_id="output",
    )
    graph.inputs, graph.outputs = ["base", "grouped", "inverse"], ["output"]
    return graph


def _weighted_sum_graph(rows=2):
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("partials", (rows, 3, 4), "f16"), node_id="partials")
    graph.add_node(InputOp(), [], Tensor("weights", (rows, 3), "f32"), node_id="weights")
    graph.add_node(
        WeightedRouteSumOp(routes=3),
        ["partials", "weights"],
        Tensor("output", (rows, 4), "f16"),
        node_id="output",
    )
    graph.inputs, graph.outputs = ["partials", "weights"], ["output"]
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


def test_expert_bucket_reference_groups_routes_and_returns_inverse_rows():
    ids = np.array([[2, 1], [2, 3], [1, 2]], dtype=np.int32)
    result, _ = NumpyBackend().run(_bucket_graph(), input_data={"ids": ids})

    np.testing.assert_array_equal(result.outputs["grouped_routes"], [[1, 4], [0, 2], [5, -1], [3, -1], [-1, -1]])
    np.testing.assert_array_equal(result.outputs["group_experts"], [1, 2, 2, 3, 0])
    np.testing.assert_array_equal(result.outputs["inverse"], [[2, 0], [3, 6], [1, 4]])


def test_route_unbucket_functionally_updates_only_one_shard():
    base = np.arange(12, dtype=np.float16).reshape(6, 2)
    grouped = (100 + np.arange(8, dtype=np.float16)).reshape(2, 2, 2)
    inverse = np.array([[0, 4], [5, 7], [2, 6]], dtype=np.int32)
    result, _ = NumpyBackend().run(
        _unbucket_graph(),
        input_data={"base": base, "grouped": grouped, "inverse": inverse},
    )
    expected = base.copy()
    expected[[1, 2, 3, 5]] = grouped.reshape(4, 2)[[0, 1, 3, 2]]
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_weighted_route_sum_uses_slot_order_and_one_fp16_narrowing():
    partials = np.arange(24, dtype=np.float16).reshape(2, 3, 4) / np.float16(8)
    weights = np.array([[0.25, 0.5, 0.75], [0.75, 0.5, 0.25]], dtype=np.float32)
    result, _ = NumpyBackend().run(
        _weighted_sum_graph(),
        input_data={"partials": partials, "weights": weights},
    )
    expected = np.zeros((2, 4), dtype=np.float32)
    for slot in range(3):
        expected = np.asarray(expected + partials[:, slot].astype(np.float32) * weights[:, slot, None], dtype=np.float32)
    np.testing.assert_array_equal(result.outputs["output"], expected.astype(np.float16))


def test_direct_expert_layout_cuda_lowerings_have_bounded_launch_geometry():
    bucket = CudaBackend(tune_db=None).compile(_bucket_graph())
    [bucket_op] = [node.op for node in bucket.nodes.values() if isinstance(node.op, CudaOp)]
    assert bucket_op.grid == ((1,), (1,), (1,))
    assert bucket_op.block == ((256,), (1,), (1,))
    assert "atomicAdd(&counts[expert], 1)" in bucket_op.kernel_source
    assert "inverse[i] = -1" in bucket_op.kernel_source
    assert "grouped_routes[grouped_row] = route" in bucket_op.kernel_source

    unbucket = CudaBackend(tune_db=None).compile(_unbucket_graph())
    [unbucket_op] = [node.op for node in unbucket.nodes.values() if isinstance(node.op, CudaOp)]
    assert unbucket_op.block == ((256,), (1,), (1,))
    assert unbucket_op.kernel_source.startswith("#include <cuda_fp16.h>")
    assert "inverse[route] - 4" in unbucket_op.kernel_source

    weighted = CudaBackend(tune_db=None).compile(_weighted_sum_graph(Dim("num_tokens", hint=8)))
    [weighted_op] = [node.op for node in weighted.nodes.values() if isinstance(node.op, CudaOp)]
    assert weighted_op.runtime_args == ("num_tokens",)
    assert weighted_op.block == ((256,), (1,), (1,))
    assert weighted_op.kernel_source.startswith("#include <cuda_fp16.h>")
    assert "for (int slot = 0; slot < 3; ++slot)" in weighted_op.kernel_source
