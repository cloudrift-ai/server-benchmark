"""The value algebra — how a real-valued cone is SPELLED, beside :mod:`~emmy.compiler.ir.expr`'s
integer index algebra.

Two languages over one node type. ``expr.py``'s ``_ExprOps`` builds ``BinaryExpr`` for COORDINATES,
where ``apply_binop`` floor-divides for ``/`` and the simplifier applies integer ``div`` / ``mod``
rules. A VALUE is never that: it is a ``FuncCallExpr`` keyed by the ``ElementwiseImpl`` name, so the
index simplifier treats it as opaque and no value can meet an integer rule by accident.

:class:`Value` is the authoring surface — the language a definition is written in, and the language
a test states an expected reading in. It builds IR and knows nothing about sympy; deciding what a
cone MEANS is :mod:`~emmy.compiler.ir.symbolic`'s job.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.ir.expr import Expr, FuncCallExpr, Literal, Var

#: The default ``through`` for a value reading — the ring a reader may reassociate without
#: licensing anything else. ``exp`` is deliberately absent: seeing through it factors ``exp(−m)``
#: out of ``exp(s − m)``, the overflow a twisted carrier exists to avoid. A consumer that owns that
#: risk (the twist, which replaces the tree's spelling wholesale) names its own wider predicate.
RING = frozenset({"add", "subtract", "multiply", "divide", "negative"})


def ring(op) -> bool:  # noqa: ANN001 — any ElementwiseImpl
    """The default legality predicate: see through the ring, treat everything else as an atom."""
    return op.name in RING


@dataclass(frozen=True)
class Value:
    """One real-valued expression under Python operators — every operator an op-name call."""

    expr: Expr

    def __add__(self, other) -> Value:  # noqa: ANN001
        return _call("add", self, other)

    def __radd__(self, other) -> Value:  # noqa: ANN001
        return _call("add", other, self)

    def __sub__(self, other) -> Value:  # noqa: ANN001
        return _call("subtract", self, other)

    def __rsub__(self, other) -> Value:  # noqa: ANN001
        return _call("subtract", other, self)

    def __mul__(self, other) -> Value:  # noqa: ANN001
        return _call("multiply", self, other)

    def __rmul__(self, other) -> Value:  # noqa: ANN001
        return _call("multiply", other, self)

    def __truediv__(self, other) -> Value:  # noqa: ANN001
        return _call("divide", self, other)

    def __rtruediv__(self, other) -> Value:  # noqa: ANN001
        return _call("divide", other, self)

    def __neg__(self) -> Value:
        return _call("negative", self)


def value(name: str) -> Value:
    """One free value by name."""
    return Value(Var(name))


def values(names: str) -> tuple[Value, ...]:
    """Several free values, whitespace-separated — ``s, m, v = values("s m v")``."""
    return tuple(value(name) for name in names.split())


def _coerce(x) -> Value:  # noqa: ANN001
    return x if isinstance(x, Value) else Value(Literal(float(x), "float"))


def _call(name: str, *args) -> Value:  # noqa: ANN001
    return Value(FuncCallExpr(name, tuple(_coerce(a).expr for a in args)))


def exp(x) -> Value:  # noqa: ANN001
    return _call("exp", x)


def sqrt(x) -> Value:  # noqa: ANN001
    return _call("sqrt", x)


def maximum(a, b) -> Value:  # noqa: ANN001
    return _call("maximum", a, b)


__all__ = ["RING", "Value", "exp", "maximum", "ring", "sqrt", "value", "values"]
