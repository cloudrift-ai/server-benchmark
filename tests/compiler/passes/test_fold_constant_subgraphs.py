"""``032_fold_constant_subgraphs`` — the generic constant-subgraph fold that dissolves the
birth-time dequant algebra: matching rule (maximal constant-only cone with a storage-decode op),
the digest-safety scope (nothing without a decode op folds), the ``EMMY_FP8_EXPAND`` skip, exact
bind-time numerics vs the dequant reference, and the ordering proof that the ``050``/``060``
layout folds compose their transposes onto the folded constant."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.dtype import decode_f8
from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.frontend.ir import LinearOp, ReshapeOp, TransposeOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.loader.binder import bind_constants, evaluate_source_graph
from emmy.compiler.loader.quant import dequantize
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to
from emmy.compiler.tensor import Tensor

rng = np.random.default_rng(7)

_RULE = "032_fold_constant_subgraphs"


def _apply(graph: Graph) -> Graph:
    return Pipeline.build(["frontend/decomposition"], select=[_RULE]).run(graph)


def _finite_bits(shape):
    bits = rng.integers(0, 256, shape).astype(np.uint8)
    bits[bits == 0x7F] = 0x00
    bits[bits == 0xFF] = 0x80
    return bits


def _cone_graph(scale_shape=(8, 1), *, dtype="f32", op_name="multiply"):
    """The degenerate-block dequant cone the birth-time speller emits: bits constant → decode
    cast → broadcast scale ⊗."""
    g = Graph()
    w = g.add_node(
        op=ConstantOp(name="p_w", source_path="layer.weight", source_shape=(8, 16), source_dtype="f8e4m3"),
        inputs=[],
        output=Tensor("p_w_bits", (8, 16), "f8e4m3"),
    )
    scale = g.add_node(
        op=ConstantOp(name="p_w_scale", source_path="layer.weight_scale", source_shape=scale_shape, source_dtype="f32"),
        inputs=[],
        output=Tensor("p_w_scale", scale_shape, "f32"),
    )
    cast = g.add_node(op=ElementwiseOp(op="from_f8e4m3"), inputs=[w], output=Tensor("p_w_dq", (8, 16), dtype))
    s_bc = broadcast_to(g, scale, (8, 16))
    g.add_node(op=ElementwiseOp(op=op_name), inputs=[cast, s_bc], output=Tensor("p_w", (8, 16), dtype), node_id="p_w")
    g.inputs, g.outputs = [], ["p_w"]
    return g


def _sources(bits, scale):
    return {"layer.weight": bits, "layer.weight_scale": scale}


# ===================================================================
# The fold: one record constant, exact bind-time numerics
# ===================================================================


def test_fold_collapses_cone_to_one_record_constant():
    folded = _apply(_cone_graph())
    assert set(folded.nodes) == {"p_w"}
    op = folded.nodes["p_w"].op
    assert isinstance(op, ConstantOp) and op.source_graph is not None
    assert op.source_path is None and not op.source_parts
    assert op.source_shape == (8, 16) and op.source_dtype == "f32"
    # The record is the cone verbatim: its leaves name the checkpoint sources.
    leaf_paths = {lop.source_path for _nid, lop in op.source_graph.loadable_constants()}
    assert leaf_paths == {"layer.weight", "layer.weight_scale"}


@pytest.mark.parametrize(("op_name", "dtype"), [("multiply", "f32"), ("divide", "f32"), ("multiply", "f16")])
def test_fold_record_evaluates_bit_identical_to_bind_time_dequant(op_name, dtype):
    """The record's evaluation IS the bind-time dequant: same LUT decode, same f32 multiply
    (numpy promotes the f16 tensor against the f32 scale), same single rounding into the
    compute dtype — ``assert_array_equal``, not allclose."""
    folded = _apply(_cone_graph(dtype=dtype, op_name=op_name))
    bits = _finite_bits((8, 16))
    scale = (np.abs(rng.standard_normal((8, 1))) + 0.5).astype(np.float32)
    val = evaluate_source_graph(folded.nodes["p_w"].op.source_graph, _sources(bits, scale))
    ref = dequantize(decode_f8(bits, "f8e4m3"), scale, inverse=op_name == "divide")
    if dtype == "f16":
        ref = ref.astype(np.float16)
    np.testing.assert_array_equal(val, ref)
    assert val.dtype == ref.dtype


def test_fold_is_maximal_block_form_folds_as_one():
    """The 2-D-block cone ends in a reshape-back; the fold must fire once at that maximal root
    — never leave a residual multiply/reshape behind a cast-only fold."""
    g = Graph()
    w = g.add_node(
        op=ConstantOp(name="p_w", source_path="layer.weight", source_shape=(8, 16), source_dtype="f8e4m3"),
        inputs=[],
        output=Tensor("p_w_bits", (8, 16), "f8e4m3"),
    )
    scale = g.add_node(
        op=ConstantOp(name="p_w_scale", source_path="layer.weight_scale", source_shape=(2, 1, 4, 1), source_dtype="f32"),
        inputs=[],
        output=Tensor("p_w_scale", (2, 1, 4, 1), "f32"),
    )
    cast = g.add_node(op=ElementwiseOp(op="from_f8e4m3"), inputs=[w], output=Tensor("p_w_dq", (8, 16), "f32"))
    blk = g.add_node(op=ReshapeOp(shape=(2, 4, 4, 4)), inputs=[cast], output=Tensor("p_w_blk", (2, 4, 4, 4), "f32"))
    s_bc = broadcast_to(g, scale, (2, 4, 4, 4))
    mul = g.add_node(op=ElementwiseOp(op="multiply"), inputs=[blk, s_bc], output=Tensor("p_w_sblk", (2, 4, 4, 4), "f32"))
    g.add_node(op=ReshapeOp(shape=(8, 16)), inputs=[mul], output=Tensor("p_w", (8, 16), "f32"), node_id="p_w")
    g.inputs, g.outputs = [], ["p_w"]

    folded = _apply(g)
    assert set(folded.nodes) == {"p_w"}
    bits = _finite_bits((8, 16))
    scale_np = (np.abs(rng.standard_normal((2, 4))) + 0.5).astype(np.float32)
    val = evaluate_source_graph(folded.nodes["p_w"].op.source_graph, _sources(bits, scale_np.reshape(2, 1, 4, 1)))
    np.testing.assert_array_equal(val, dequantize(decode_f8(bits, "f8e4m3"), scale_np))


def test_fold_idempotent():
    once = _apply(_cone_graph())
    twice = _apply(once)
    assert set(twice.nodes) == {"p_w"} and twice.nodes["p_w"].op.source_graph is not None


# ===================================================================
# Scope + gate: fold NOTHING without a decode op; EMMY_FP8_EXPAND skips
# ===================================================================


def test_flag_on_keeps_the_cone_in_graph(monkeypatch):
    monkeypatch.setenv("EMMY_FP8_EXPAND", "1")
    g = _cone_graph()
    before = set(g.nodes)
    assert set(_apply(g).nodes) == before


def test_decode_free_constant_cone_is_out_of_scope():
    """The digest-safety scope: a constant-only cone WITHOUT a storage-decode op (an existing
    model's constant mask-math style cone) must not fold — widening is gated on digest
    evidence."""
    g = Graph()
    a = g.add_node(
        op=ConstantOp(name="a", source_path="m.a", source_shape=(4, 4), source_dtype="f32"),
        inputs=[],
        output=Tensor("a", (4, 4), "f32"),
    )
    b = g.add_node(
        op=ConstantOp(name="b", source_path="m.b", source_shape=(4, 4), source_dtype="f32"),
        inputs=[],
        output=Tensor("b", (4, 4), "f32"),
    )
    g.add_node(op=ElementwiseOp(op="multiply"), inputs=[a, b], output=Tensor("c", (4, 4), "f32"), node_id="c")
    g.inputs, g.outputs = [], ["c"]
    before = set(g.nodes)
    assert set(_apply(g).nodes) == before


def test_activation_fed_cone_never_folds():
    """A decode op over a graph INPUT is not a constant cone."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (8, 16), "f8e4m3"), node_id="x")
    g.add_node(op=ElementwiseOp(op="from_f8e4m3"), inputs=[x], output=Tensor("y", (8, 16), "f32"), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    before = set(g.nodes)
    assert set(_apply(g).nodes) == before


def test_scalar_constant_leaf_bounds_the_fold():
    """A ``value`` scalar constant is not a source-backed leaf — the loader never visits it —
    so the cone containing it cannot fold as a whole. The decode SUB-cone is still a maximal
    constant cone of its own and folds; the scalar multiply stays as runtime compute."""
    g = Graph()
    w = g.add_node(
        op=ConstantOp(name="p_w", source_path="layer.weight", source_shape=(8, 16), source_dtype="f8e4m3"),
        inputs=[],
        output=Tensor("p_w_bits", (8, 16), "f8e4m3"),
    )
    s = g.add_node(op=ConstantOp(name="s", value=0.5), inputs=[], output=Tensor("s", (1,), "f32"))
    cast = g.add_node(op=ElementwiseOp(op="from_f8e4m3"), inputs=[w], output=Tensor("p_w_dq", (8, 16), "f32"))
    s_bc = broadcast_to(g, s, (8, 16))
    g.add_node(op=ElementwiseOp(op="multiply"), inputs=[cast, s_bc], output=Tensor("p_w", (8, 16), "f32"), node_id="p_w")
    g.inputs, g.outputs = [], ["p_w"]
    folded = _apply(g)
    assert "p_w" in folded.nodes and isinstance(folded.nodes["p_w"].op, ElementwiseOp)  # the multiply survives
    record_node = folded.nodes["p_w_dq"]
    assert isinstance(record_node.op, ConstantOp) and record_node.op.source_graph is not None
    bits = _finite_bits((8, 16))
    val = evaluate_source_graph(record_node.op.source_graph, {"layer.weight": bits})
    np.testing.assert_array_equal(val, decode_f8(bits, "f8e4m3"))


def test_externally_consumed_interior_declines_the_fold():
    """A mid-cone value read outside the cone (here: the decode cast is also a graph output)
    must survive — the fold declines rather than orphan the reader."""
    g = _cone_graph()
    cast_id = next(nid for nid, n in g.nodes.items() if isinstance(n.op, ElementwiseOp) and n.op.op.decodes is not None)
    g.outputs = ["p_w", cast_id]
    before = set(g.nodes)
    assert set(_apply(g).nodes) == before


# ===================================================================
# Ordering proof: fold at 032 → 040 emits the weight transpose → 050 folds
# it onto the SAME folded constant's load_ops (sm_90+ band)
# ===================================================================


def test_later_layout_folds_compose_onto_the_folded_constant():
    from emmy.compiler import target as target_mod

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1, 4, 16), "f32"), node_id="x")
    w = g.add_node(
        op=ConstantOp(name="p_w", source_path="layer.weight", source_shape=(8, 16), source_dtype="f8e4m3"),
        inputs=[],
        output=Tensor("p_w_bits", (8, 16), "f8e4m3"),
    )
    scale = g.add_node(
        op=ConstantOp(name="p_w_scale", source_path="layer.weight_scale", source_shape=(8, 1), source_dtype="f32"),
        inputs=[],
        output=Tensor("p_w_scale", (8, 1), "f32"),
    )
    cast = g.add_node(op=ElementwiseOp(op="from_f8e4m3"), inputs=[w], output=Tensor("p_w_dq", (8, 16), "f32"))
    s_bc = broadcast_to(g, scale, (8, 16))
    mul = g.add_node(op=ElementwiseOp(op="multiply"), inputs=[cast, s_bc], output=Tensor("p_w", (8, 16), "f32"), node_id="p_w")
    g.add_node(op=LinearOp(), inputs=["x", mul], output=Tensor("y", (1, 4, 8), "f32"), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]

    target_mod.set_target((12, 0))
    try:
        lowered = Pipeline.build(["frontend/decomposition"]).run(g)
    finally:
        target_mod.set_target(None)

    records = {nid: n for nid, n in lowered.nodes.items() if isinstance(n.op, ConstantOp) and n.op.source_graph is not None}
    assert len(records) == 1
    nid, node = next(iter(records.items()))
    assert [type(lop).__name__ for lop in node.op.load_ops] == ["TransposeOp"]
    assert tuple(d.as_static() for d in node.output.shape) == (16, 8)

    bits = _finite_bits((8, 16))
    scale_np = (np.abs(rng.standard_normal((8, 1))) + 0.5).astype(np.float32)
    bound = bind_constants(lowered, _sources(bits, scale_np))
    np.testing.assert_array_equal(bound[nid], dequantize(decode_f8(bits, "f8e4m3"), scale_np).T)


def test_transpose_op_node_alone_does_not_fold():
    """A bare transpose over a plain constant is 050/060's business (a ``load_ops`` append),
    not this rule's — no decode op, no fold, byte-identical layout-fold behavior."""
    g = Graph()
    w = g.add_node(
        op=ConstantOp(name="w", source_path="m.w", source_shape=(8, 16), source_dtype="f32"),
        inputs=[],
        output=Tensor("w", (8, 16), "f32"),
    )
    g.add_node(op=TransposeOp(axes=(1, 0)), inputs=[w], output=Tensor("wt", (16, 8), "f32"), node_id="wt")
    g.inputs, g.outputs = [], ["wt"]
    before = set(g.nodes)
    assert set(_apply(g).nodes) == before
