"""The value reading — ``Stmt.as_expr`` / ``Body.as_expr``.

One cone, read into an ``Expr`` the caller may decide over, with the two scope rules that make the
answer honest: a nested body's temp is not one value at this level, and a name read before it is
bound already means something else here.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import FuncCallExpr, Var
from emmy.compiler.ir.stmt import Accum, Assign, Const, Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.value import RING, exp, ring, values


def _asn(name: str, op: str, *args: str) -> Assign:
    return Assign(name=name, op=op, args=args)


def _eval(body: Body, name: str, env: dict) -> float:
    """The cone evaluated statement by statement — the reading's oracle."""
    env = dict(env)
    for stmt in body:
        env[stmt.name] = stmt.value if isinstance(stmt, Const) else stmt.op(*(env[arg] for arg in stmt.args))
    return env[name]


def _summand() -> Body:
    """Attention's streamed weight: ``exp(s − m)·v``."""
    return Body([_asn("d", "subtract", "s", "m"), _asn("w", "exp", "d"), _asn("p", "multiply", "w", "v")])


def test_a_cone_reads_to_the_call_tree_its_ops_spell() -> None:
    read = _summand().as_expr("p", lambda op: True)
    assert read == FuncCallExpr("multiply", (FuncCallExpr("exp", (FuncCallExpr("subtract", (Var("s"), Var("m"))),)), Var("v")))


def test_an_unlicensed_op_stays_an_atom() -> None:
    """``through`` is legality: the ring reading stops at ``exp``, so the weight is one opaque name.
    That is the point — seeing through it would factor ``exp(−m)`` out of ``exp(s − m)``."""
    read = _summand().as_expr("p", ring)
    assert read == FuncCallExpr("multiply", (Var("w"), Var("v")))
    assert read.free_vars() == {"w", "v"}
    assert "exp" not in RING


def test_a_value_bound_in_a_nested_body_stays_an_atom() -> None:
    """The scope rule. ``t`` is one value per iteration, not one value here, so the reading must not
    resolve through it even though ``Body.definitions`` (recursive) can find it."""
    inner = Body([_asn("t", "multiply", "x", "y"), Accum(name="acc", value="t")])
    body = Body([Loop(axis=Axis("k", 8), body=inner), _asn("r", "add", "acc", "t")])
    assert "t" in body.definitions
    assert body.as_expr("r", lambda op: True) == FuncCallExpr("add", (Var("acc"), Var("t")))


def test_a_name_read_before_it_is_bound_never_aliases_its_own_binding() -> None:
    """A monoid's combine rebinds its accumulator: ``acc = add(sa, sb)`` whose ``sa`` reads the
    INCOMING ``acc``. A name-keyed reading would loop or answer with a value from the future; a
    shadowing binding's opaque value therefore carries its position."""
    body = Body([_asn("sa", "multiply", "alpha", "acc"), _asn("acc", "copy", "sa")])
    read = body.as_expr("acc", ring)
    assert read == Var("acc@1")
    assert body.as_expr("sa", ring) == FuncCallExpr("multiply", (Var("alpha"), Var("acc")))


def test_an_ordinary_temp_keeps_its_own_spelling() -> None:
    """The counterpart: a name only ever bound before it is read is not shadowing anything, so it
    reports under its own name — which is what lets a consumer act on the atoms a split names."""
    assert _summand().as_expr("p", ring).free_vars() == {"w", "v"}


@pytest.mark.parametrize("point", [(0.5, -1.25, 2.0), (-3.0, 0.75, 1.5)])
def test_a_read_cone_evaluates_to_what_the_statements_compute(point: tuple[float, ...]) -> None:
    """``Expr.eval`` resolves a ``FuncCallExpr`` through the op registry, so the reading needs no
    evaluator of its own — and agreeing with the statement-by-statement oracle is what says the
    reading preserved the cone rather than rearranged it."""
    env = dict(zip(("s", "m", "v"), point, strict=True))
    body = _summand()
    np.testing.assert_allclose(body.as_expr("p", lambda op: True).eval(env), _eval(body, "p", env))


def test_the_authoring_language_builds_the_same_tree_the_reading_returns() -> None:
    """``Value`` is how a definition and a test SPELL a value — op-name calls, never ``BinaryExpr``,
    so a value can never meet the integer index simplifier's rules by accident."""
    s, m, v = values("s m v")
    assert (exp(s - m) * v).expr == _summand().as_expr("p", lambda op: True)
