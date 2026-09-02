"""``Closure`` — a pure :class:`Lambda` paired with the enclosing iteration axes it may read.

The one scoped-lambda concept shared by the Tile-level canonical forms and the lowering passes.
Unlike a conventional closure, the environment is an INDEX SPACE: ``axes`` names enclosing
iteration variables, never data values — a normalized term's values arrive through operand
edges, and the only names its lift may capture are axes bound by its ancestors. That restriction
is what keeps equivalence tractable (:meth:`Closure.canonical` renames the environment
positionally) and what makes a tree position an instantiation: alpha-equal closures under EQUAL
captures denote one value, while the same form under different captures stays distinct values of
one function.

A ``Closure`` is CLOSED BY CONSTRUCTION: :meth:`__post_init__` refuses one whose axes are not
params of the lambda they scope. Keeping the invariant optional on the type is what let captures
accumulate: nothing refused them, so every consumer grew a way to cope instead.

Scoping is for LAMBDAS. A term is not one — a ``Fold`` composes through operand edges and no
``Body`` may hold it — so a term's and an operand edge's alpha-quotient is
:func:`~emmy.compiler.ir.pure.fold.alpha_canonical`, which returns the same kind back. Both are
built from the same :func:`~emmy.compiler.ir.pure.fold.alpha_rename`.

Equivalence is a comparison-time VIEW, never a stored normal form — canonicalizing in place
would clobber meaningful axis names in the tree. And it is a property, not a sharing mechanism:
a rewrite may merge equivalent closures only where it supplies the binder that unifies their
names (the contraction's single A slot in the semiring canonicalization) or where the captures
already coincide (``tile/normalize._share_common_cones``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from emmy.compiler.ir.pure.fold import alpha_rename
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.stmt.body import Body


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
        # No capture check: ``Lambda`` itself refuses a body that reads what it does not bind, so
        # every ``fn`` reaching here is already closed. ``axes`` names WHICH of its params are the
        # enclosing environment — what :meth:`canonical` renames positionally — not a permission.
        stray_axes = [axis for axis in self.axes if axis not in self.fn.params]
        if stray_axes:
            raise ValueError(f"Closure axes {stray_axes} are not params of the lambda they scope")

    def canonical(self) -> Lambda:
        """The alpha-canonical form, the enclosing iteration axes included — a ``Lambda``.

        :meth:`Lambda.canonical` handles names bound by the lambda itself; this also renumbers the
        captured axes positionally, so equivalent lifts at different tree positions compare equal.
        Every axis is a param (:meth:`__post_init__`), so which params are the environment is
        itself part of the form: one lambda scoped by different axes has different canonical forms.
        """
        return self._canonical

    @cached_property
    def _canonical(self) -> Lambda:
        # A change of NAMES, not of kind: the result denotes the same function and satisfies every
        # invariant ``Lambda`` states, so ``Lambda.__post_init__`` re-checks it on the way out.
        renamed, rename = alpha_rename(self.fn.body, self.fn.params, self.axes)
        return Lambda(
            params=tuple(rename(name) for name in self.fn.params),
            body=Body(renamed),
            results=tuple(rename(result) if isinstance(result, str) else result for result in self.fn.results),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Closure) and self.canonical() == other.canonical()

    def __hash__(self) -> int:
        return hash(self.canonical())


__all__ = ["Closure"]
