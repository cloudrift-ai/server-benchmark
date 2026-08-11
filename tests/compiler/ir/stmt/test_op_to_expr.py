"""Unit tests for ``op_to_expr`` — elementwise op-name → Expr translation."""

from emmy.compiler.dtype import F32
from emmy.compiler.ir.expr import BinaryExpr, FuncCallExpr, Literal, TernaryExpr
from emmy.compiler.ir.stmt import Assign, Body
from emmy.compiler.ir.stmt.base import dtype_promote, op_to_expr
from emmy.compiler.ir.stmt.normalize import eliminate_copy_aliases


def test_square_renders_to_self_multiply():
    """``square`` lowers to ``x * x``.

    ReLU²/squared-ReLU MLPs (e.g. AFM-4.5B) emit a ``square`` elementwise op; the CUDA
    renderer previously had no case for it and raised ``NotImplementedError``.
    """
    x = Literal(3.0, "float")
    e = op_to_expr("square", [x])
    assert isinstance(e, BinaryExpr)
    assert e.op == "*"
    assert e.left is x and e.right is x


def test_copy_is_identity_passthrough():
    """``copy`` returns its input unchanged (dropout lowers through this path)."""
    x = Literal(1.0, "float")
    assert op_to_expr("copy", [x]) is x


def test_typed_copy_is_a_cast_not_an_alias():
    cast = Assign(name="wide", op="copy", args=("narrow",), dtype=F32)
    assert eliminate_copy_aliases(Body((cast,))) == Body((cast,))


def test_sin_cos_render_as_intrinsics():
    """DiT timestep embedding keeps its runtime trigonometric ops."""
    x = Literal(1.0, "float")
    for name in ("sin", "cos"):
        expr = op_to_expr(name, [x])
        assert isinstance(expr, FuncCallExpr)
        assert expr.name == name


def test_mask_ops_render():
    """The explicit-mask subgraph's ops (comparisons, bool combines, where)
    translate — the gemma-4 whole-model tune previously died rendering
    ``equal``. ``bitwise_*`` carry bool-mask semantics and spell as logical
    ops (a float ``|`` would not compile on the f32-promoted operands)."""
    a, b = Literal(1.0, "float"), Literal(2.0, "float")
    for name, c_op in [
        ("equal", "=="),
        ("not_equal", "!="),
        ("greater", ">"),
        ("less", "<"),
        ("greater_equal", ">="),
        ("less_equal", "<="),
        ("bitwise_or", "||"),
        ("bitwise_and", "&&"),
    ]:
        e = op_to_expr(name, [a, b])
        assert isinstance(e, BinaryExpr) and e.op == c_op

    w = op_to_expr("where", [a, b, Literal(3.0, "float")])
    assert isinstance(w, TernaryExpr) and w.cond is a


def test_boolean_bitwise_not_is_logical_zero_test_and_integer_fails_closed():
    a = Literal(1.0, "float")
    for dtype in (None, "f32", "bool"):
        expr = op_to_expr("bitwise_not", [a], dtype=dtype)
        assert isinstance(expr, BinaryExpr) and expr.op == "=="

    import pytest

    with pytest.raises(NotImplementedError, match="boolean masks only"):
        op_to_expr("bitwise_not", [a], dtype="u32")


def test_integer_ops_keep_integer_spelling_without_changing_mask_semantics():
    """Integer bit manipulation uses C integer operators; f32 masks stay logical."""
    a, b = Literal(7, "int"), Literal(3, "int")
    for name, c_op in [
        ("floor_divide", "//"),
        ("remainder", "%"),
        ("left_shift", "<<"),
        ("right_shift", ">>"),
        ("bitwise_or", "|"),
        ("bitwise_and", "&"),
        ("bitwise_xor", "^"),
    ]:
        expr = op_to_expr(name, [a, b], dtype="u32")
        assert isinstance(expr, BinaryExpr) and expr.op == c_op

    assert op_to_expr("bitwise_and", [a, b]).op == "&&"
    assert op_to_expr("bitwise_or", [a, b]).op == "||"


def test_integer_elementwise_promotion_is_operation_specific():
    for op in ("add", "multiply", "floor_divide", "remainder", "left_shift", "bitwise_xor", "bitwise_count"):
        assert dtype_promote(op, ["u32", "u32"]) == "u32"
    assert dtype_promote("bitwise_or", ["u64", "u64"]) == "u64"
    assert dtype_promote("left_shift", ["u16", "u32"]) == "u32"
    assert dtype_promote("right_shift", ["u64", "i32"]) == "u64"
    assert dtype_promote("bitwise_and", ["i64", "u32"]) == "i64"
    assert dtype_promote("bitwise_and", ["f32", "f32"]) == "f32"
    assert dtype_promote("equal", ["i32", "i32"]) == "f32"


def test_integer_bitwise_count_stays_an_abstract_intrinsic():
    value = Literal(7, "int")
    expr = op_to_expr("bitwise_count", [value], dtype="u32")
    assert isinstance(expr, FuncCallExpr) and expr.name == "bitwise_count" and expr.args == (value,)
