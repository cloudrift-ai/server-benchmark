"""A dtype-changing ``copy`` prints as what it does to the source's contents.

From a bits-carrier dtype (a float-family name stored on an integer carrier: fp8
bits, packed fp4 pairs) the identity moves stored bits, so it prints
``emmy::to_bits``. From an ordinary dtype it converts the value, so it stays
``emmy::cast``.
"""

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp


def _copy_graph(src_dtype: str, dst_dtype: str) -> Graph:
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8), src_dtype), node_id="x")
    y = g.add_node(op=ElementwiseOp(op="copy"), inputs=[x], output=Tensor("y", (4, 8), dst_dtype))
    g.inputs = [x]
    g.outputs = [y]
    return g


def test_copy_from_packed_fp4_prints_to_bits():
    assert "emmy::to_bits(inputs.x)" in _copy_graph("f4e2m1x2", "i32").pretty_print()


def test_copy_from_fp8_bits_prints_to_bits():
    assert "emmy::to_bits(inputs.x)" in _copy_graph("f8e4m3", "i32").pretty_print()


def test_value_converting_copy_still_prints_cast():
    assert "emmy::cast(inputs.x)" in _copy_graph("f32", "f16").pretty_print()


def test_same_dtype_copy_prints_bare():
    out = _copy_graph("f16", "f16").pretty_print()
    assert "emmy::cast" not in out and "emmy::to_bits" not in out
