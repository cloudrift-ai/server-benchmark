import numpy as np
import torch

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.backend.plan import plan_from_graph
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.target import set_target
from emmy.serving.mxfp4 import (
    CompactGroupedRowsModule,
    GroupedRowsModule,
    trace_compact_grouped_rows,
    trace_grouped_mxfp4_stage,
    trace_grouped_rows,
    trace_routed_mxfp4_stage,
)


def test_routed_mxfp4_stage_is_one_direct_compact_sm70_program():
    graph = trace_routed_mxfp4_stage(rows=4, experts=3, out_features=16, in_features=32)
    try:
        set_target((7, 0))
        plan = plan_from_graph(CudaBackend(tune_db=None).compile(graph))
    finally:
        set_target(None)

    assert plan.inputs == ["x", "weight", "expert_ids", "weight_scale"]
    assert len(plan.outputs) == 1
    assert len(plan.launches) == 1
    assert tuple(buffer.name for buffer in plan.buffers if buffer.role == "scratch") == ()


def test_routed_mxfp4_stage_preserves_unvalidated_ue8m0_special_codes():
    graph = trace_routed_mxfp4_stage(rows=1, experts=1, out_features=2, in_features=32)
    equal_ops = [node for node in graph.nodes.values() if isinstance(node.op, ElementwiseOp) and node.op.op.name == "equal"]
    assert len(equal_ops) == 2
    feed = {
        "x": np.ones((1, 32), dtype=np.float16),
        "weight": np.full((1, 2, 16), 0x22, dtype=np.uint8),
        "expert_ids": np.zeros((1,), dtype=np.int32),
        "weight_scale": np.array([[[0], [255]]], dtype=np.uint8),
    }

    result, _ = NumpyBackend().run(graph, input_data=feed)
    actual = result.outputs[graph.outputs[0]]
    np.testing.assert_array_equal(actual[:, :1], np.zeros((1, 1), dtype=np.float16))
    assert np.isnan(actual[:, 1:]).all()


def test_grouped_row_pack_and_compact_are_inverse_on_live_prefix():
    x = torch.arange(8 * 4, dtype=torch.float16).view(8, 4)
    offsets = torch.tensor([0, 2, 2, 5, 8], dtype=torch.int32)
    grouped = GroupedRowsModule(rows_per_group=4).module()(x, offsets)
    actual = CompactGroupedRowsModule(rows=8, rows_per_group=4).module()(grouped, offsets)

    torch.testing.assert_close(actual, x, rtol=0, atol=0)
    assert grouped.shape == (4, 4, 4)
    assert torch.count_nonzero(grouped[1]) == 0


def test_grouped_mxfp4_stage_exposes_volta_mma_without_decoded_storage():
    graph = trace_grouped_mxfp4_stage(groups=4, rows_per_group=16, experts=3, out_features=16, in_features=32)
    try:
        set_target((7, 0))
        with pinned_knobs({"TILE": "mma_m8n8k4_f16_f32/f1x1", "WORK": "w1x1", "STAGE": "d1/smem"}):
            plan = plan_from_graph(CudaBackend(tune_db=None).compile(graph))
    finally:
        set_target(None)

    assert len(plan.launches) == 1
    assert tuple(buffer.name for buffer in plan.buffers if buffer.role == "scratch") == ()
    source = plan.kernels[plan.launches[0].kernel_name].source
    assert source is not None and "mma.sync" in source


def test_grouped_row_pack_and_compact_lower_without_scratch_storage():
    graphs = (
        trace_grouped_rows(rows=8, groups=4, rows_per_group=4, features=16),
        trace_compact_grouped_rows(rows=8, groups=4, rows_per_group=4, features=16),
    )
    try:
        set_target((7, 0))
        plans = tuple(plan_from_graph(CudaBackend(tune_db=None).compile(graph)) for graph in graphs)
    finally:
        set_target(None)

    for plan in plans:
        assert len(plan.launches) == 1
        assert tuple(buffer.name for buffer in plan.buffers if buffer.role == "scratch") == ()
