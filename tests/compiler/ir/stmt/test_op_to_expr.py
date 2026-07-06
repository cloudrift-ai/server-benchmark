"""Unit tests for ``op_to_expr`` — elementwise op-name → Expr translation."""

from emmy.compiler.ir.expr import BinaryExpr, Literal
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
