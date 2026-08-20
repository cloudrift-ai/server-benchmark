"""The split-index recomposition fold in ``BinaryExpr.simplify``: ``(x/c)*c + x%c → x``.

The ``loop/canonicalize`` axis re-fusion spells a split store's coordinates as ``f/Q`` and
``f%Q`` in separate buffer dims; the row-major address flatten then produces exactly
``(f/Q)*(Q·s) + (f%Q)·s`` chains, and this fold collapses them back to the affine ``f·s`` so
the emitted address is byte-identical to the unsplit spelling. The identity holds for every
int under C truncated division AND Python floor division, so the fold is unconditional."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, Var


def _simplify(e):
    return e.simplify(SimplifyCtx.empty())


def _lit(v: int) -> Literal:
    return Literal(v, "int")


def test_bare_pair_recomposes():
    f = Var("f")
    e = BinaryExpr("+", BinaryExpr("*", BinaryExpr("//", f, _lit(512)), _lit(512)), BinaryExpr("%", f, _lit(512)))
    assert _simplify(e) == f


@pytest.mark.parametrize("c,k", [(512, 1), (64, 2), (24, 512), (8, 8)])
def test_scaled_pair_preserves_value(c: int, k: int):
    """``(f/c)*(c·k) + (f%c)·k`` folds to ``f·k`` — numerically pinned over a range."""
    f = Var("f")
    quot = BinaryExpr("*", BinaryExpr("//", f, _lit(c)), _lit(c * k))
    rem = BinaryExpr("%", f, _lit(c)) if k == 1 else BinaryExpr("*", BinaryExpr("%", f, _lit(c)), _lit(k))
    e = BinaryExpr("+", quot, rem)
    s = _simplify(e)
    assert "%" not in s.pretty() and "//" not in s.pretty() and "/" not in s.pretty()
    for v in range(0, 4 * c + 3):
        assert e.eval({"f": v}) == s.eval({"f": v}), f"mismatch at f={v}: {s.pretty()}"


def test_pair_inside_longer_chain():
    """Other addends survive around the recomposed pair — the flattened store address shape
    (``row·N + (f/Q)·Q + f%Q``)."""
    f, row = Var("f"), Var("row")
    e = BinaryExpr(
        "+",
        BinaryExpr("+", BinaryExpr("*", row, _lit(12288)), BinaryExpr("*", BinaryExpr("//", f, _lit(512)), _lit(512))),
        BinaryExpr("%", f, _lit(512)),
    )
    s = _simplify(e)
    assert "%" not in s.pretty()
    for fv, rv in ((0, 0), (511, 1), (513, 3), (12287, 7)):
        assert e.eval({"f": fv, "row": rv}) == s.eval({"f": fv, "row": rv})


def test_mismatched_divisors_do_not_fold():
    f = Var("f")
    e = BinaryExpr("+", BinaryExpr("*", BinaryExpr("//", f, _lit(512)), _lit(512)), BinaryExpr("%", f, _lit(256)))
    s = _simplify(e)
    for v in (0, 255, 256, 511, 513):
        assert e.eval({"f": v}) == s.eval({"f": v})
    assert "%" in s.pretty(), "different divisors must not recompose"


def test_mismatched_scale_does_not_fold():
    """``(f/c)*a + f%c`` with ``a`` not a multiple of ``c`` stays as-is."""
    f = Var("f")
    e = BinaryExpr("+", BinaryExpr("*", BinaryExpr("//", f, _lit(512)), _lit(500)), BinaryExpr("%", f, _lit(512)))
    s = _simplify(e)
    for v in (0, 511, 512, 1025):
        assert e.eval({"f": v}) == s.eval({"f": v})
    assert "%" in s.pretty()


def test_different_dividends_do_not_fold():
    f, g = Var("f"), Var("g")
    e = BinaryExpr("+", BinaryExpr("*", BinaryExpr("//", f, _lit(64)), _lit(64)), BinaryExpr("%", g, _lit(64)))
    s = _simplify(e)
    assert "%" in s.pretty()
