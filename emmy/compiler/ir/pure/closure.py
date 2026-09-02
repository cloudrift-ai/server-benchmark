"""``Closure`` — a pure :class:`Lambda` paired with the enclosing iteration axes it may read.

The one scoped-lambda concept shared by the Tile-level canonical forms and the lowering passes.
Unlike a conventional closure, the environment is an INDEX SPACE: ``axes`` names enclosing
iteration variables, never data values — a normalized term's values arrive through operand
edges, and the only names its lift may capture are axes bound by its ancestors. That restriction
is what keeps equivalence tractable (:meth:`Closure.canonical` renames the environment
positionally) and what makes a tree position an instantiation: alpha-equal closures under EQUAL
captures denote one value, while the same form under different captures stays distinct values of
one function.

A ``Closure`` is CLOSED BY CONSTRUCTION: :meth:`__post_init__` refuses one whose lambda reads a
value it did not bind. The two things that must work on an OPEN lambda are free functions beside
it — :func:`value_captures` (the question "is this closed?", which has to be askable about a term
that is not) and :func:`canonical_under` (the sharing unification takes the alpha-quotient of
whole terms that still capture). Keeping the invariant optional on the type is what let captures
accumulate: nothing refused them, so every consumer grew a way to cope instead.

Equivalence is a comparison-time VIEW, never a stored normal form — canonicalizing in place
would clobber meaningful axis names in the tree. And it is a property, not a sharing mechanism:
a rewrite may merge equivalent closures only where it supplies the binder that unifies their
names (the contraction's single A slot in the semiring canonicalization) or where the captures
already coincide (``tile/normalize._share_common_cones``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import cached_property

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.fold import Fold, operand_name
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt.body import _member_reads


def _lambda_members(members):
    """Walk every binding inside a lambda, including Fold operand edges and algebra bodies.

    Takes a plain iterable: it also walks TERM sequences (a Fold's operand edges), which a ``Body``
    may not hold — a body is statements, and ``Body.__new__`` refuses anything else.
    """
    for stmt in members:
        yield stmt
        if isinstance(stmt, Fold):
            for edge in stmt.operands:
                if isinstance(edge, Fold):
                    yield from _lambda_members((edge,))
                else:
                    yield edge
            yield from _lambda_members(stmt.lift.body)
        else:
            for nested in stmt.nested():
                yield from _lambda_members(nested)


def canonical_under(fn: Lambda, axes: tuple[str, ...]) -> tuple:
    """A whole lambda's alpha-canonical key — :func:`canonical_key` over its parts."""
    return canonical_key(fn.body, fn.params, fn.results, axes)


def canonical_key(members, params: tuple[str, ...], results: tuple, axes: tuple[str, ...]) -> tuple:
    """The alpha-canonical form of a member sequence, with the enclosing ``axes`` renamed
    positionally — a comparison KEY, never a stored form.

    Takes the members directly rather than a ``Lambda`` so a bare TERM can be quotiented without
    being wrapped in a body it may not legally sit in: a ``Fold`` is not a ``Stmt``. Callers that
    do hold a lambda pass its parts (:func:`canonical_under`).
    """
    members_in = tuple(members)
    members = tuple(_lambda_members(members_in))
    reads = {name for stmt in members for name in _member_reads(stmt)}
    bound_axes = tuple(name for stmt in members for name in stmt.binds_axes())
    axis_order = tuple(dict.fromkeys((*axes, *bound_axes)))
    active_axes = tuple(name for name in axis_order if name in reads or name in params or name in bound_axes)
    names = {name: f"_a{i}" for i, name in enumerate(active_axes)}

    p = 0
    for name in params:
        if name not in names:
            names[name] = f"_p{p}"
            p += 1
    v = 0
    for stmt in members:
        for name in stmt.defines():
            if name not in names:
                names[name] = f"_v{v}"
                v += 1

    def rename(name: str) -> str:
        return names.get(name, name)

    sigma = Sigma({name: Var(names[name]) for name in active_axes})

    def rename_axis(axis: Axis) -> Axis:
        name = names.get(axis.name)
        return replace(axis, name=name) if name is not None else axis

    renamed = tuple(stmt.rewrite(rename, sigma, rename_axis) for stmt in members_in)
    # A hashable TUPLE, not a Lambda. The members may include a term, which no Lambda body may
    # hold; and a key only has to compare, not be a well-formed binder. Members are keyed by their
    # canonical rendering, which is defined for statements and terms alike.
    return (
        tuple(rename(name) for name in params),
        tuple(repr(stmt) for stmt in renamed),  # defined for statements and terms alike; equal after renaming iff equal
        tuple(rename(result) if isinstance(result, str) else result for result in results),
    )


@dataclass(frozen=True, eq=False)
class Closure:
    """A pure lambda scoped by the enclosing iteration axes it may capture, in binding order.

    Equality and hash are ALPHA-INVARIANT — two closures are equal when their canonical forms
    coincide — while the stored spelling is preserved: correspondence uses (the seam clustering's
    sibling axis pairing, a drain's real capture names) read the original ``fn`` and ``axes``.
    """

    fn: Lambda
    axes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.axes, tuple):
            object.__setattr__(self, "axes", tuple(self.axes))
        stray = [axis for axis in self.axes if not isinstance(axis, str)]
        if stray:
            raise ValueError(f"Closure axes must be iteration-axis names, got {stray}")
        # CLOSED BY CONSTRUCTION. ``Lambda`` states this rule and delegates it ("free names ⊆
        # params ∪ enclosing iteration vars — the consuming Fold's check") but cannot decide it:
        # an axis reference and a value reference are the same ``Var``, so ``λ(k) → x[m, k]`` is
        # legal where ``m`` is an ancestor's axis and illegal where a sibling defines it. This is
        # the type that carries both halves, so it is the type that enforces it — as a formation
        # gate, not an optional method. Leaving it optional is how captures accumulated silently.
        # No capture check: ``Lambda`` itself refuses a body that reads what it does not bind, so
        # every ``fn`` reaching here is already closed. ``axes`` names WHICH of its params are the
        # enclosing environment — what :meth:`canonical` renames positionally — not a permission.
        stray_axes = [axis for axis in self.axes if axis not in self.fn.params]
        if stray_axes:
            raise ValueError(f"Closure axes {stray_axes} are not params of the lambda they scope")

    def canonical(self) -> tuple:
        """The alpha-canonical form, the enclosing iteration axes included.

        :meth:`Lambda.canonical` handles names bound by the lambda itself. A Fold tree also needs
        captured axes canonicalized so equivalent lifts at different tree positions compare
        equal. Unused enclosing axes do not affect the result.
        """
        return self._canonical

    @cached_property
    def _canonical(self) -> tuple:
        return canonical_under(self.fn, self.axes)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Closure) and self.canonical() == other.canonical()

    def __hash__(self) -> int:
        return hash(self.canonical())


def equivalent_clusters(keys: Iterable[tuple]) -> tuple[tuple[int, ...], ...]:
    """Partition alpha-canonical KEYS into equivalent clusters, in input order.

    Takes keys (:func:`edge_key`, :func:`term_key`) rather than ``Closure`` objects, so a term can
    be compared without being wrapped in a body it may not sit in. The returned indices let a
    caller keep its own Fold or graph metadata beside this general equivalence analysis.
    """
    clusters: dict[tuple, list[int]] = {}
    for index, key in enumerate(keys):
        clusters.setdefault(key, []).append(index)
    return tuple(tuple(cluster) for cluster in clusters.values())


def edge_key(operand, axes) -> tuple:
    """The alpha-quotient of one operand EDGE, as a comparison key.

    Replaces wrapping the edge in a ``Closure`` — that put a term inside a statement ``Body``, which
    a body may not hold. The key binds exactly the axes the edge references, so it doubles as the
    positional capture correspondence the seam clustering pairs siblings by.
    """
    declared = (
        (set(operand.lift.params) - operand.binds_axes())
        if isinstance(operand, Fold)
        else {name for expr in operand.exprs() for name in expr.free_vars()}
    )
    params = tuple(axis for axis in axes if axis in declared)
    return canonical_key((operand,), params, (operand_name(operand),), params)


def term_key(term) -> tuple:
    """The alpha-quotient of a whole TERM under an empty environment — the sharing unification's
    key. Same reason as :func:`edge_key`: no body wrapper, because a term is not a statement."""
    return canonical_key((term,), (), tuple(term.defines()), ())


__all__ = ["canonical_key", "canonical_under", "Closure", "edge_key", "equivalent_clusters", "term_key"]
