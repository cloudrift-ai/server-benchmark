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
    PLAN_FORMAT_INDIRECT,
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
    d["format"] = PLAN_FORMAT_INDIRECT + 1  # past every format the runtime speaks
    with pytest.raises(ValueError, match="format"):
        plan_from_dict(d)


def _indirect_graph() -> Graph:
    """One launch whose weight input is an indirect operand (table + selector + slot)."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1, 64)), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("w", (128, 64)), node_id="w")
    g.add_node(
        op=CudaOp(
            kernel_source="__global__ void k_ind() {}",
            kernel_name="k_ind",
            arg_order=("x", "w", "y"),
            grid=((1,), (1,), (1,)),
            block=((128,), (1,), (1,)),
            indirect_args=(("w", "w__table", "w__sel", 0),),
        ),
        inputs=["x", "w"],
        output=Tensor("y", (1, 128)),
        node_id="y",
    )
    g.inputs, g.outputs = ["x", "w"], ["y"]
    return g


def test_plan_indirect_operand_projection_and_round_trip():
    """An indirect operand projects off the CudaOp, serializes the plan as
    ``PLAN_FORMAT_INDIRECT`` (a runtime ignoring the field would pass the wrong arg pack —
    old readers must reject and fall back to the full compile), and survives the JSON round
    trip; a plan without the field keeps format ``PLAN_FORMAT_VERSION`` byte-compatibly."""
    plan = plan_from_graph(_indirect_graph())
    (launch,) = plan.launches
    assert launch.indirect_args == (("w", "w__table", "w__sel", 0),)
    d = plan_to_dict(plan)
    assert d["format"] == PLAN_FORMAT_INDIRECT
    restored = plan_from_dict(json.loads(json.dumps(d)))
    assert restored == plan
    # A plan with no indirect operands stays on the original format — existing packs and
    # readers are untouched.
    assert plan_to_dict(plan_from_graph(_sample_graph()))["format"] == PLAN_FORMAT_VERSION


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


def test_weight_spec_source_parts_projection_and_round_trip():
    """A ``source_parts`` (merged sibling weight) constant projects into the plan and survives
    the JSON round-trip; ``assemble_source`` binds it as the axis-0 concat on both spec kinds."""
    from emmy.compiler.loader.binder import assemble_source

    g = Graph()
    g.add_node(
        op=ConstantOp(
            name="w_cat",
            source_parts=(("m.wq", (3, 4)), ("m.wk", (2, 4))),
            source_shape=(5, 4),
            load_ops=(TransposeOp(axes=(1, 0)),),
        ),
        inputs=[],
        output=Tensor("w_cat", (4, 5)),
        node_id="w_cat",
    )
    g.add_node(
        op=CudaOp(
            kernel_source="__global__ void k_t() {}",
            kernel_name="k_t",
            arg_order=("w_cat", "y"),
            grid=((1,), (1,), (1,)),
            block=((32,), (1,), (1,)),
            smem_bytes=0,
        ),
        inputs=["w_cat"],
        output=Tensor("y", (4, 5)),
        node_id="y",
    )
    g.outputs = ["y"]

    plan = plan_from_graph(g)
    spec = plan.weights["w_cat"]
    assert spec.source_path is None
    assert spec.source_parts == (("m.wq", (3, 4)), ("m.wk", (2, 4)))
    restored = plan_from_dict(json.loads(json.dumps(plan_to_dict(plan))))
    assert restored == plan

    wq = np.arange(12, dtype=np.float32).reshape(3, 4)
    wk = np.arange(8, dtype=np.float32).reshape(2, 4) + 9.0
    sources = {"m.wq": wq, "m.wk": wk}
    expected = np.concatenate([wq, wk], axis=0)
    np.testing.assert_array_equal(assemble_source(restored.weights["w_cat"], sources), expected)
    np.testing.assert_array_equal(assemble_source(g.nodes["w_cat"].op, sources), expected)


def _one_weight_plan(op, nid="w"):
    """A minimal plan over one loadable constant ``op``, so a weight spec can be inspected."""
    g = Graph()
    g.add_node(op=op, inputs=[], output=Tensor(nid, (4, 4)), node_id=nid)
    g.add_node(
        op=CudaOp(
            kernel_source="__global__ void k_t() {}",
            kernel_name="k_t",
            arg_order=(nid, "y"),
            grid=((1,), (1,), (1,)),
            block=((32,), (1,), (1,)),
            smem_bytes=0,
        ),
        inputs=[nid],
        output=Tensor("y", (4, 4)),
        node_id="y",
    )
    g.outputs = ["y"]
    return plan_from_graph(g)


def _slice_index_map(out_shape, spans):
    """A single-source ``IndexMapOp`` reading ``spans`` (per-axis ``(start, step)``)."""
    from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
    from emmy.compiler.ir.tensor.ir import IndexMapOp, IndexSource

    coords = tuple(
        BinaryExpr("+", BinaryExpr("*", Var(f"out_coord_{i}"), Literal(step, "int")), Literal(start, "int"))
        for i, (start, step) in enumerate(spans)
    )
    return IndexMapOp(out_shape=out_shape, sources=(IndexSource(input_idx=0, coord_map=coords),))


@pytest.mark.parametrize(("shape", "spans"), [((3,), ((0, 1),)), ((2, 3), ((1, 1), (0, 2))), ((4,), ((2, 1),))])
def test_slice_index_map_encodes_and_matches_the_binder(shape, spans):
    """A folded ``SliceOp`` reaches the plan as an ``IndexMapOp``; the pack vocabulary encodes
    the affine per-axis form as a plain numpy slice (the trellis speller's ``svh[:n]`` N-pad
    trim, which used to disable pack saving for the WHOLE program)."""
    op = _slice_index_map(shape, spans)
    encoded = _encode_load_ops((op,))
    assert encoded is not None and encoded[0][0] == "slice"
    src = np.arange(8 * 9, dtype=np.float32).reshape(8, 9)
    src = src if len(shape) == 2 else src[0]
    np.testing.assert_array_equal(apply_weight_loads(src, encoded), apply_load_ops(src, (op,)))


def test_non_affine_index_map_stays_outside_the_vocabulary():
    """A transposing index map is NOT a slice — it must encode as ``None`` rather than as a
    silently wrong per-axis read."""
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.tensor.ir import IndexMapOp, IndexSource

    swap = IndexMapOp(out_shape=(3, 2), sources=(IndexSource(input_idx=0, coord_map=(Var("out_coord_1"), Var("out_coord_0"))),))
    assert _encode_load_ops((swap,)) is None


def test_computed_constant_projects_as_a_source_op_and_binds():
    """A zero-leaf ``source_graph`` bind record (the trellis basis-restore Hadamard) has NO
    checkpoint key: without a plan-side form it projects to ``source_path=None`` and vanishes
    from the bound feed. It projects as ``source_op`` instead, round-trips, and rebuilds."""
    from emmy.compiler.ir.frontend.ir import HadamardOp
    from emmy.compiler.loader.binder import assemble_source
    from emmy.compiler.loader.exl3 import sylvester_hadamard

    record = Graph()
    record.outputs = [record.add_node(op=HadamardOp(size=4), inputs=[], output=Tensor("h", (4, 4)))]
    plan = _one_weight_plan(ConstantOp(name="h", source_graph=record, source_shape=(4, 4)))
    assert plan.weights["w"].source_op == ("hadamard", (4,))
    assert plan.weights["w"].load_ops == ()  # bindable: the pack save gate must not trip
    restored = plan_from_dict(json.loads(json.dumps(plan_to_dict(plan))))
    assert restored == plan
    np.testing.assert_array_equal(assemble_source(restored.weights["w"], {}), sylvester_hadamard(4))


def test_unreproducible_bind_record_marks_the_weight_unbindable(caplog):
    """A record the plan cannot reproduce (leaves it would have to read from the checkpoint)
    must mark the weight ``load_ops=None`` — the pack save then refuses, rather than writing a
    pack whose boot silently drops the weight."""
    import logging

    record = Graph()
    record.add_node(op=InputOp(), inputs=[], output=Tensor("leaf", (4, 4)), node_id="leaf")
    record.outputs = ["leaf"]
    with caplog.at_level(logging.WARNING, logger="emmy.compiler.backend.plan"):
        plan = _one_weight_plan(ConstantOp(name="h", source_graph=record, source_shape=(4, 4)))
    assert plan.weights["w"].source_op is None
    assert plan.weights["w"].load_ops is None
    assert "bind record" in caplog.text


def test_plan_mimo_node_mints_per_buffer_specs_and_writes():
    """A multi-output CudaOp node yields one BufferSpec per BUFFER (aux buffer
    scratch-roled) and a launch whose ``writes`` lists every produced buffer."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (32, 64)), node_id="x")
    g.add_node(
        op=CudaOp(
            kernel_source="__global__ void k_mo() {}",
            kernel_name="k_mo",
            arg_order=("x", "y", "y__sq"),
            grid=((1,), (1,), (1,)),
            block=((128,), (1,), (1,)),
            zero_outputs=("y__sq",),
        ),
        inputs=["x"],
        outputs=[Tensor("y", (32, 64)), Tensor("y__sq", (32,))],
        node_id="y",
    )
    g.add_node(
        op=CudaOp(
            kernel_source="__global__ void k_c() {}",
            kernel_name="k_c",
            arg_order=("y", "y__sq", "z"),
            grid=((1,), (1,), (1,)),
            block=((128,), (1,), (1,)),
        ),
        inputs=["y", "y__sq"],
        output=Tensor("z", (32, 64)),
        node_id="z",
    )
    g.inputs, g.outputs = ["x"], ["z"]
    plan = plan_from_graph(g)
    roles = {b.name: b.role for b in plan.buffers}
    assert roles == {"x": "input", "y": "scratch", "y__sq": "scratch", "z": "output"}
    launch_mo = next(lc for lc in plan.launches if lc.kernel_name == "k_mo")
    assert launch_mo.writes == ("y", "y__sq")
    restored = plan_from_dict(json.loads(json.dumps(plan_to_dict(plan))))
    assert restored == plan
    assert next(lc for lc in restored.launches if lc.kernel_name == "k_mo").writes == ("y", "y__sq")


def test_base_backend_runs_tuple_forward_per_buffer():
    """The default Backend.run stores per-buffer values for a multi-output
    node whose ``forward`` returns a tuple matched to ``node.outputs``."""
    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.ir.base import Op
    from emmy.compiler.ir.tensor.ir import ElementwiseOp

    class _SquareAndRowsum(Op):
        def forward(self, x):
            return x * x, (x * x).sum(axis=1)

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    g.add_node(op=_SquareAndRowsum(), inputs=["x"], outputs=[Tensor("y", (4, 8)), Tensor("y__sq", (4,))], node_id="y")
    g.add_node(op=ElementwiseOp(op="exp"), inputs=["y__sq"], output=Tensor("z", (4,)), node_id="z")
    g.inputs, g.outputs = ["x"], ["z"]

    x = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    result, _ = NumpyBackend().run(g, input_data={"x": x})
    np.testing.assert_allclose(result.outputs["z"], np.exp((x * x).sum(axis=1)), rtol=1e-5)
