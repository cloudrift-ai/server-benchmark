"""The one closed map from :class:`~emmy.compiler.ir.expr.Expr` to sympy, and the decisions taken
over it.

``Expr`` is SYNTAX: it preserves association and temps exactly, which is what a kernel is spelled
from. sympy is SEMANTICS: it decides equality modulo ring identities, so a consumer asks what a cone
MEANS instead of matching a stored spelling. Two expression types, two jobs.

The direction is ONE-WAY. Nothing here builds IR from a sympy expression — sympy normalizes at
construction (``Mul(Mul(a, b), c)`` flattens, ``x − x`` vanishes, argument order is canonical),
which is right for deciding and wrong for spelling. A consumer that needs IR renames and splices the
statements it already has; the readings below tell it WHICH ones.

Only algorithmic sympy is used — ``expand``, ``cancel``, ``Poly``, ``as_powers_dict``. Never
``simplify``: a heuristic rewrite whose result depends on the version is not a compiler decision.

The three readings share a shape: a ``Body``, the name whose cone is read, a ``varies`` predicate
naming which atoms stream along the fold axis, and the ``through`` legality predicate the reading
sees ops with (:mod:`~emmy.compiler.ir.value` supplies the default ring). What each returns is a
decision plus the NAMES and STATEMENTS a consumer may act on — never an expression to emit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import sympy

from emmy.compiler.ir.expr import BinaryExpr, Builtin, CastExpr, Expr, FuncCallExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.stmt import Stmt
from emmy.compiler.ir.stmt.body import Body

# Value ops with a sympy meaning. Anything else — ``exp`` under a ring-only reading, ``rsqrt``, a
# storage decode — becomes an UNINTERPRETED function of its arguments, which compares structurally
# and so collapses the decision to alpha-equality exactly where nothing is interpretable.
_CALLS: dict[str, Callable] = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "negative": lambda a: -a,
    "exp": sympy.exp,
}

# Index arithmetic. ``/`` floor-divides because that is what ``apply_binop`` does: an ``Expr``
# reached through a coordinate is an integer algebra, and reading it as exact division would state
# an equality the emitted kernel does not honour.
_BINOPS: dict[str, Callable] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: sympy.floor(a / b),
    "//": lambda a, b: sympy.floor(a / b),
    "%": sympy.Mod,
    "<": sympy.Lt,
    "<=": sympy.Le,
    ">": sympy.Gt,
    ">=": sympy.Ge,
    "==": sympy.Eq,
}


def symbolic(expr: Expr):
    """``expr`` as a sympy expression. Total: an unknown call, a cast or a GPU builtin is an
    uninterpreted atom rather than an error, so a reading never fails on a node kind."""
    if isinstance(expr, Var | Builtin):
        return sympy.Symbol(expr.name)
    if isinstance(expr, Literal):
        value = expr.value
        return sympy.Integer(int(value)) if float(value).is_integer() else sympy.Float(value)
    if isinstance(expr, BinaryExpr):
        left, right = symbolic(expr.left), symbolic(expr.right)
        op = _BINOPS.get(expr.op)
        return op(left, right) if op is not None else sympy.Function(f"op{expr.op}")(left, right)
    if isinstance(expr, FuncCallExpr):
        args = tuple(symbolic(a) for a in expr.args)
        call = _CALLS.get(expr.name)
        return call(*args) if call is not None else sympy.Function(expr.name)(*args)
    if isinstance(expr, TernaryExpr):
        return sympy.Piecewise((symbolic(expr.if_true), symbolic(expr.cond)), (symbolic(expr.if_false), True))
    if isinstance(expr, CastExpr):
        return sympy.Function(f"cast_{expr.dtype}")(symbolic(expr.expr))
    raise NotImplementedError(f"symbolic: unhandled Expr node {type(expr).__name__}")


def equal(a: Expr, b: Expr) -> bool:
    """Whether two value cones denote the same value, modulo ring identities."""
    return sympy.cancel(sympy.expand(symbolic(a) - symbolic(b))) == 0


@dataclass(frozen=True)
class Factored:
    """One product split by axis variance.

    ``invariant`` and ``varying`` are ATOM NAMES with multiplicity — a square is listed twice, so a
    consumer sees ``x·x`` for what it is rather than as one factor. Each invariant carries whether
    it divides, since ``Σ x/c`` equals ``(Σ x)/c`` only for an invariant ``c``. ``spine`` is the
    statements the reading saw through: the cone of the factored name minus the cones of its atoms,
    which is exactly what a consumer removes when it rebuilds the lift around the split.
    """

    invariant: tuple[tuple[str, bool], ...]
    varying: tuple[str, ...]
    spine: tuple[Stmt, ...]


def _spine(body: Body, name: str, expr: Expr) -> tuple[Stmt, ...]:
    """The statements the reading consumed — this scope's cone of ``name`` minus the CONES of the
    atoms that survived into ``expr``. Subtracting the atoms themselves is not enough: an atom is
    opaque, so everything that built it (``exp``'s ``s − m``) belongs to the atom, not the spine."""
    atoms = expr.free_vars()
    kept = {id(stmt) for atom in atoms for stmt in body.backward_cone((atom,)).members}
    return tuple(stmt for stmt in body.backward_cone((name,)).members if id(stmt) not in kept)


def factor(body: Body, name: str, varies: Callable[[str], bool], through) -> Factored | None:  # noqa: ANN001
    """The value cone of ``name`` split into invariant and varying factors, or ``None`` when it is
    not one product.

    Refusals are STRUCTURAL — the reading is a sum, or a denominator is not a plain atom, so there
    is no product to split and ``Σ`` cannot commute past it. Whether the split is USEFUL (does it
    have both an invariant and a varying side?) is the calling rule's condition, not this one's.
    """
    expr = body.as_expr(name, through)
    powers = sympy.expand(symbolic(expr)).as_powers_dict()
    invariant: list[tuple[str, bool]] = []
    varying: list[str] = []
    for base, power in powers.items():
        if base.is_number:
            continue
        if not isinstance(base, sympy.Symbol) or not power.is_Integer:
            return None  # a compound base (a sum, an uninterpreted call over several atoms) is not a factor
        count, divides = abs(int(power)), int(power) < 0
        if varies(base.name):
            if divides:
                return None  # nothing licenses moving a fold into a varying denominator
            varying.extend([base.name] * count)
        else:
            invariant.extend([(base.name, divides)] * count)
    return Factored(tuple(invariant), tuple(varying), _spine(body, name, expr))


Monomials = tuple[tuple[tuple[tuple[str, int], ...], object], ...]


def linearize(body: Body, name: str, varies: Callable[[str], bool], through) -> Monomials | None:  # noqa: ANN001
    """The value cone of ``name`` as monomials in the STREAMED atoms, each with its coefficient.

    ``((("s", 2),), 1)`` reads "the ``s²`` term has coefficient 1"; the empty monomial is the term
    free of streamed atoms. Every coefficient is an expression over the invariants by construction,
    which is what makes this the reading for "is this fold's summand linear in what it streams?".

    ``None`` when the cone is not polynomial in the streamed atoms (a stream in a denominator, an
    uninterpreted call over one) or when nothing streams at all.

    Coefficients come back as sympy expressions: a consumer DECIDES with them — is this coefficient
    the bare atom ``alpha``? is it free of the other states? — and then acts on the names it already
    holds. Building IR from one is out of bounds (see the module docstring).
    """
    expr = body.as_expr(name, through)
    streamed = sorted(atom for atom in expr.free_vars() if varies(atom))
    if not streamed:
        return None
    try:
        poly = sympy.Poly(sympy.expand(symbolic(expr)), *(sympy.Symbol(atom) for atom in streamed))
    except sympy.PolynomialError:
        return None
    return tuple(
        (tuple((atom, power) for atom, power in zip(streamed, monomial, strict=True) if power), coeff) for monomial, coeff in poly.terms()
    )


__all__ = ["Factored", "equal", "factor", "linearize", "symbolic"]
