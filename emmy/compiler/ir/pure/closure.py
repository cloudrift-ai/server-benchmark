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
from emmy.compiler.ir.pure.fold import Fold, edge_free_axes, operand_name
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.stmt.body import _member_reads


def _lambda_members(body: Body):
    """Walk every binding inside a lambda, including Fold operand edges and algebra bodies."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, Fold):
            for edge in stmt.operands:
                if isinstance(edge, Fold):
                    yield from _lambda_members(Body((edge,)))
                else:
                    yield edge
            yield from _lambda_members(stmt.lift.body)
        else:
            for nested in stmt.nested():
                yield from _lambda_members(nested)


def value_captures(fn: Lambda, axes: Iterable[str]) -> frozenset[str]:
    """Free names of ``fn`` that are NOT environment axes — data read from sibling definitions.

    The QUESTION, asked of a lambda and a scope. It is a free function and not a
    :class:`Closure` method because a Closure is closed by construction: asking whether something
    is closed must be possible about a term that is not.
    """
    return fn.free_names() - frozenset(axes)


def canonical_under(fn: Lambda, axes: tuple[str, ...]) -> Lambda:
    """``fn``'s alpha-canonical form with the enclosing iteration ``axes`` renamed positionally.

    Also a free function, for the same reason: the sharing unification takes the alpha-quotient of
    whole terms that still capture (``tile/normalize._share_common_cones``), so canonicalization
    cannot require the closed invariant.
    """
    members = tuple(_lambda_members(fn.body))
    reads = {name for stmt in members for name in _member_reads(stmt)}
    bound_axes = tuple(name for stmt in members for name in stmt.binds_axes())
    axis_order = tuple(dict.fromkeys((*axes, *bound_axes)))
    active_axes = tuple(name for name in axis_order if name in reads or name in fn.params or name in bound_axes)
    names = {name: f"_a{i}" for i, name in enumerate(active_axes)}

    p = 0
    for name in fn.params:
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

    renamed = Body(stmt.rewrite(rename, sigma, rename_axis) for stmt in fn.body)
    return Lambda(
        params=tuple(rename(name) for name in fn.params),
        body=renamed,
        results=tuple(rename(result) if isinstance(result, str) else result for result in fn.results),
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
        captures = value_captures(self.fn, self.axes)
        if captures:
            raise ValueError(
                f"Closure captures {sorted(captures)} from an enclosing scope: a term's values arrive through "
                f"operand edges bound positionally to lift params; only the axes {list(self.axes)} may be free."
            )

    @classmethod
    def over_edge(cls, operand, axes: Iterable[str]) -> Closure:
        """Wrap one operand edge as a closure over the axes it references, kept in ``axes`` order.

        The wrapping lambda binds exactly the referenced axes, so :attr:`axes` doubles as the
        edge's positional capture correspondence (what the seam clustering pairs siblings by).
        """
        free = edge_free_axes(operand)
        params = tuple(axis for axis in axes if axis in free)
        return cls(Lambda(params=params, body=Body((operand,)), results=(operand_name(operand),)), params)

    def canonical(self) -> Lambda:
        """The alpha-canonical form, the enclosing iteration axes included.

        :meth:`Lambda.canonical` handles names bound by the lambda itself. A Fold tree also needs
        captured axes canonicalized so equivalent lifts at different tree positions compare
        equal. Unused enclosing axes do not affect the result.
        """
        return self._canonical

    @cached_property
    def _canonical(self) -> Lambda:
        return canonical_under(self.fn, self.axes)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Closure) and self.canonical() == other.canonical()

    def __hash__(self) -> int:
        return hash(self.canonical())


def equivalent_clusters(closures: Iterable[Closure]) -> tuple[tuple[int, ...], ...]:
    """Partition closures into alpha-equivalent clusters, in input order.

    The returned indices let a later pass keep its own Fold or graph metadata beside this general
    equivalence analysis.
    """
    clusters: dict[Closure, list[int]] = {}
    for index, closure in enumerate(closures):
        clusters.setdefault(closure, []).append(index)
    return tuple(tuple(cluster) for cluster in clusters.values())


__all__ = ["canonical_under", "Closure", "equivalent_clusters", "value_captures"]
