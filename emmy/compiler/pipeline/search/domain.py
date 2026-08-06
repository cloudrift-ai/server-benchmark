"""The candidate domain as a CONSTRAINED INTEGER SET — declare the dimensions, declare the
multiplicative bounds that couple them, enumerate the legal points.

This is the machinery for generating a family's candidate values from its stated constraints
instead of curating them by hand (:mod:`~emmy.compiler.pipeline.search.space`'s move grids). The
constraints that bound a schedule family are products of the unknowns — ``wm·wn·32 ≤ 1024`` (the
CTA thread budget), ``fm·fn ≤ 32`` (the C-fragment budget) — so the feasible set is not convex and
there is no coordinate change that makes both the products and the bounds affine at once: prime
exponents linearize the products but turn ``≤`` into divisibility (a partial order), and real logs
linearize both but leave the feasible points off any lattice. What survives is the honest thing:
keep integer coordinates, keep the products multiplicative, and ENUMERATE.

Brute force, but not blind. :meth:`Space.__iter__` walks the dimensions in declaration order and
drops a prefix the moment a bound's running product can no longer be satisfied — every value is
``≥ 1``, so a final product is a multiple of any partial one and never smaller. A budget like
``wm·wn·32 ≤ 1024`` therefore kills its subtree at the first factor that overruns it rather than at
the leaf.

Worked example — a warp tile's free geometry, generated rather than listed::

    Space(
        dims=(
            Dimension("wm", (1, 2, 4, 8, 16)),
            Dimension("wn", (1, 2, 4, 8, 16)),
            Dimension("fn", tuple(range(1, 33))),
        ),
        bounds=(Bound(("wm", "wn"), limit=1024, coeff=32),),  # the CTA thread budget
    )

A bound states ONE comparison, ``coeff · Πdims ≤ limit``, because that is the only one any live
domain needs. Equality and divisibility were spelled here too, for constraints the curated grids
still carry (a flash tile covering the head dimension exactly; a K-step dividing a static extent).
They had no caller, and a comparison nothing states is a comparison nobody has had to get right —
so they went. Both are a few lines to restore, and the pruning contract each would need is written
on :meth:`Bound.holds`: a partial product prunes only because the final one is a multiple of it.

Scope: this module knows integers and products only. Categorical legality (an operand dtype, a
transport's eligibility, a repack rule) and anything that reads the term is the scheduler's, exactly
as it is for the curated grids.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    """One integer dimension: a name and its finite candidate values.

    Values must be ``≥ 1``. That is not a formality — the prefix pruning relies on a product being
    monotone in every factor, and a zero or negative value breaks it.
    """

    name: str
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"dimension {self.name!r} declares no values")
        bad = [v for v in self.values if v < 1]
        if bad:
            raise ValueError(f"dimension {self.name!r} declares non-positive value(s) {bad}; products must stay monotone to prune")


@dataclass(frozen=True)
class Bound:
    """A multiplicative budget over the dimensions: ``coeff · ∏ dims ≤ limit`` — a thread,
    fragment or box budget, the only comparison a live domain states (see the module docstring for
    the two that were spelled here and had no caller).

    A dimension may repeat in ``dims`` and then contributes its value once per occurrence.
    ``coeff`` folds in the constants the bound multiplies by (an atom's ``atom_n``, a warp's 32
    lanes), so the dimensions stay the only unknowns.
    """

    dims: tuple[str, ...]
    limit: int
    coeff: int = 1

    def __post_init__(self) -> None:
        if not self.dims:
            raise ValueError(f"bound <= {self.limit} names no dimension")
        if self.limit < 1 or self.coeff < 1:
            raise ValueError(f"bound {self.spell()} needs a positive limit and coeff")

    def spell(self) -> str:
        """The bound as text — for error messages, not a stored codec."""
        lhs = "*".join(self.dims) if self.coeff == 1 else f"{self.coeff}*{'*'.join(self.dims)}"
        return f"{lhs} <= {self.limit}"

    def holds(self, product: int) -> bool:
        """Whether ``product`` — ``coeff`` times the dims bound SO FAR — can still satisfy this
        bound. A budget needs to know nothing else: over-budget stays over-budget, since every
        value is ``≥ 1`` and the final product is a multiple of any partial one. That is the whole
        pruning contract, and a comparison that does NOT have it (an equality, testable at a
        partial product only through divisibility) would have to say so here."""
        return product <= self.limit


@dataclass(frozen=True)
class Space:
    """A bounded integer set: the cartesian product of ``dims``, narrowed by ``bounds``.

    Iterating yields each legal point as ``{dimension name: value}`` in declaration order — the
    dimensions vary in declaration order too, last one fastest, so the first declared dimension's
    first value leads. Per-family option-0 ordering is therefore a property of how the caller
    declares the space, the same contract the curated grids carry.
    """

    dims: tuple[Dimension, ...]
    bounds: tuple[Bound, ...] = ()

    def __post_init__(self) -> None:
        names = [d.name for d in self.dims]
        if not names:
            raise ValueError("a space declares no dimensions")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate dimension name among {names}")
        for b in self.bounds:
            unknown = sorted({d for d in b.dims if d not in set(names)})
            if unknown:
                raise ValueError(f"bound {b.spell()} names undeclared dimension(s) {unknown}")

    def __iter__(self) -> Iterator[dict[str, int]]:
        # Per bound: how many times each dimension occurs in it.
        reps = [tuple(b.dims.count(d.name) for d in self.dims) for b in self.bounds]

        def walk(i: int, point: dict[str, int], products: tuple[int, ...]) -> Iterator[dict[str, int]]:
            if i == len(self.dims):
                yield dict(point)
                return
            dim = self.dims[i]
            for v in dim.values:
                running = list(products)
                for bi, bound in enumerate(self.bounds):
                    if not reps[bi][i]:
                        continue
                    running[bi] *= v ** reps[bi][i]
                    if not bound.holds(running[bi]):
                        break
                else:
                    yield from walk(i + 1, {**point, dim.name: v}, tuple(running))

        yield from walk(0, {}, tuple(b.coeff for b in self.bounds))


__all__ = ["Bound", "Dimension", "Space"]
