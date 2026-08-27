"""Binder evaluation of persisted ``source_graph`` bind records via the NumPy backend."""

from __future__ import annotations

import numpy as np

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp, IndexSource
from emmy.compiler.loader.binder import evaluate_source_graph


def _arange_record(n: int = 128) -> Graph:
    """The persisted spelling of a folded ``aten.arange(0, 4 * n, 4)``: scalar stop
    constant → broadcast to ``(n,)`` → elementwise ``arange`` → ``* step``. Mirrors the
    ``add_53`` bind record of the sm70 deepseek golden rows, where the elementwise
    ``arange`` over a multi-element operand crashed the interpreter (``np.arange`` is
    not elementwise — its scalar coercion raises on a multi-element array)."""
    graph = Graph()
    stop = graph.add_node(
        op=ConstantOp(name="stop", value=float(n)),
        inputs=[],
        output=Tensor("stop", (1,), "i64"),
    )
    stop_bc = graph.add_node(
        op=IndexMapOp(out_shape=(n,), sources=(IndexSource(input_idx=0, coord_map=(Literal(0, "int"),)),)),
        inputs=[stop],
        output=Tensor("stop_bc", (n,), "i64"),
    )
    ramp = graph.add_node(
        op=ElementwiseOp(op="arange"),
        inputs=[stop_bc],
        output=Tensor("ramp", (n,), "i64"),
    )
    step = graph.add_node(
        op=ConstantOp(name="step", value=4.0),
        inputs=[],
        output=Tensor("step", (1,), "i64"),
    )
    step_bc = graph.add_node(
        op=IndexMapOp(out_shape=(n,), sources=(IndexSource(input_idx=0, coord_map=(Literal(0, "int"),)),)),
        inputs=[step],
        output=Tensor("step_bc", (n,), "i64"),
    )
    graph.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[ramp, step_bc],
        output=Tensor("out", (n,), "i64"),
        node_id="out",
    )
    graph.outputs = ["out"]
    return graph


def test_elementwise_arange_is_an_index_ramp():
    """Each element's value is its own index, in the operand's shape and dtype."""
    impl = ElementwiseImpl("arange")
    operand = np.full((128,), 128, dtype=np.int64)
    np.testing.assert_array_equal(impl(operand), np.arange(128, dtype=np.int64))
    assert impl(operand).dtype == np.int64


def test_source_free_arange_record_evaluates():
    value = evaluate_source_graph(_arange_record(), {})
    np.testing.assert_array_equal(value, np.arange(128, dtype=np.int64) * 4)
