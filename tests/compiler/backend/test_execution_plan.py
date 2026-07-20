"""Execution-plan projection + JSON round-trip (``backend/plan.py``) — CPU-only.

The plan is the runtime projection of a lowered ``Graph[CudaOp]``; these tests pin
(1) the projection (buffers/constants/launches/kernels/weights/symbols pulled off a
hand-built graph), (2) the JSON round-trip through the pack expression grammar —
including the composite ceil-div grid factors that the Graph-JSON path can't
round-trip — and (3) the pack-side weight load-op vocabulary matching the binder.
"""

import json

import numpy as np
import pytest

from emmy.compiler.backend.plan import (
    PLAN_FORMAT_VERSION,
    WeightSpec,
    _encode_load_ops,
    apply_weight_loads,
    plan_from_dict,
    plan_from_graph,
    plan_to_dict,
)
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.cuda import CudaOp, TmaDescMeta
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.frontend.ir import ReshapeOp, TransposeOp
from emmy.compiler.loader.binder import apply_load_ops


def _sample_graph() -> Graph:
    """One symbolic-seq CudaOp launch over an input, a weight (with a load chain), a static
    scalar, and a runtime (context_value) constant — every plan field populated."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", ("seq_len", 64)), node_id="x")
    g.add_node(op=ConstantOp(name="eps", value=1e-6), inputs=[], output=Tensor("eps", (1,)), node_id="eps")
    g.add_node(
        op=ConstantOp(name="w", source_path="model.w", load_ops=(TransposeOp(axes=(1, 0)),), source_shape=(64, 64)),
        inputs=[],
        output=Tensor("w", (64, 64)),
        node_id="w",
    )
    g.add_node(op=ConstantOp(name="div", context_value=Var("seq_len")), inputs=[], output=Tensor("div", (1,)), node_id="div")
    ceil_div = BinaryExpr("//", BinaryExpr("+", Var("seq_len"), Literal(15, "int")), Literal(16, "int"))
    g.add_node(
        op=CudaOp(
            kernel_source="__global__ void k_test() {}",
            kernel_name="k_test",
            arg_order=("x", "w", "eps", "div", "y"),
            grid=((ceil_div,), ("seq_len", 2), (1,)),
            block=((128,), (1,), (1,)),
            smem_bytes=1024,
            zero_outputs=("y",),
            tma_descriptors=(TmaDescMeta(name="x_desc", src_buf="x", box_extents=(64, 32), swizzle="B128"),),
            runtime_args=("seq_len",),
        ),
        inputs=["x", "w", "eps", "div"],
        output=Tensor("y", ("seq_len", 64)),
        node_id="y",
    )
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def test_plan_projection():
    plan = plan_from_graph(_sample_graph())
    assert plan.backend == "cuda"
    assert plan.inputs == ["x"] and plan.outputs == ["y"]
    roles = {b.name: b.role for b in plan.buffers}
    assert roles == {"x": "input", "eps": "constant", "w": "constant", "div": "constant", "y": "output"}
    assert plan.constants == {"eps": 1e-6}
    assert plan.runtime_constants == {"div": Var("seq_len")}
    assert plan.weights == {"w": WeightSpec(source_path="model.w", load_ops=(("transpose", (1, 0)),))}
    assert set(plan.kernels) == {"k_test"}
    assert plan.kernels["k_test"].uses_tma and plan.kernels["k_test"].source.startswith("__global__")
    assert plan.symbolic_bindings == {"seq_len": ("x", 0)}
    (launch,) = plan.launches
    assert launch.kernel_name == "k_test" and launch.runtime_args == ("seq_len",)


def test_plan_json_round_trip():
    plan = plan_from_graph(_sample_graph())
    wire = json.loads(json.dumps(plan_to_dict(plan)))
    assert plan_from_dict(wire) == plan


def test_plan_round_trip_preserves_binary_key():
    plan = plan_from_graph(_sample_graph())
    spec = plan.kernels["k_test"]
    spec.source = None
    spec.binary_key = "deadbeef" * 5
    restored = plan_from_dict(json.loads(json.dumps(plan_to_dict(plan))))
    assert restored.kernels["k_test"].binary_key == spec.binary_key
    assert restored.kernels["k_test"].source is None
    assert restored == plan


def test_plan_format_version_gate():
    d = plan_to_dict(plan_from_graph(_sample_graph()))
    d["format"] = PLAN_FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="format"):
        plan_from_dict(d)


def test_grid_expr_survives_round_trip():
    plan = plan_from_graph(_sample_graph())
    restored = plan_from_dict(json.loads(json.dumps(plan_to_dict(plan))))
    (launch,) = restored.launches
    # The composite ceil-div factor must come back as a real Expr (evaluable), not a string.
    (factor,) = launch.grid[0]
    assert factor.eval({"seq_len": 33}) == 3
    assert launch.grid[1] == ("seq_len", 2)


@pytest.mark.parametrize(
    "load_ops",
    [
        (TransposeOp(axes=(1, 0)),),
        (TransposeOp(axes=(0, 2)),),  # 2-tuple = swap, aten.transpose semantics
        (TransposeOp(axes=(2, 0, 1)),),
        (ReshapeOp(shape=(6, -1)),),
        (TransposeOp(axes=(1, 0, 2)), ReshapeOp(shape=(4, 6))),
    ],
)
def test_apply_weight_loads_matches_binder(load_ops):
    src = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)[:, :, :, 0] + 0.5  # (2, 3, 4) non-contiguous
    encoded = _encode_load_ops(load_ops)
    assert encoded is not None
    np.testing.assert_array_equal(apply_weight_loads(src, encoded), apply_load_ops(src, load_ops))


def test_unsupported_load_op_encodes_as_none():
    # A symbolic reshape can't ride the pack vocabulary — the weight must be marked unbindable.
    assert _encode_load_ops((ReshapeOp(shape=("seq_len", 64)),)) is None
