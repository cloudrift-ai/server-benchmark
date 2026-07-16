"""Unit tests for ``op_to_expr`` — elementwise op-name → Expr translation."""

from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr
from emmy.compiler.ir.stmt.base import op_to_expr


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
