"""The candidate domain as a CONSTRAINED INTEGER SET — declare the dimensions, declare the
multiplicative bounds that couple them, enumerate the legal points.

This is the machinery for generating a family's candidate values from its stated constraints
instead of curating them by hand (:mod:`~emmy.compiler.pipeline.search.space`'s move grids). The
constraints that bound a schedule family are products of the unknowns — ``wm·wn·32 ≤ 1024`` (the
CTA thread budget), ``fm·fn ≤ 32`` (the C-fragment budget), ``wn·fn·atom_n = d`` (a tile that must
cover the head dimension), ``bk·atom_k`` divides a static K — so the feasible set is not convex and
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
        bounds=(
            Bound(("wm", "wn"), limit=1024, coeff=32),          # the CTA thread budget
            Bound(("wn", "fn"), limit=64, op="==", coeff=8),    # cover d=64 at atom_n=8
        ),
    )

:meth:`Space.contains` is the other half: the membership check a RECORDED configuration is tested
against, so a generated domain can be gated on "every golden stays reachable" the way the curated
catalog is today.

Scope: this module knows integers and products only. Categorical legality (an operand dtype, a
transport's eligibility, a repack rule) and anything that reads the term is the scheduler's, exactly
as it is for the curated grids.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import prod

# The comparisons a bound may state against its product.
_OPS = ("<=", "==", "divides")


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
    """A multiplicative constraint over the dimensions: ``coeff · ∏ dims  <op>  limit``.

    ``op`` is one of:

    - ``"<="`` — the product is at most ``limit`` (a thread, fragment or box budget).
    - ``"=="`` — the product covers ``limit`` exactly (a tile that must tile an extent).
    - ``"divides"`` — the product divides ``limit`` (a K-step that must tile a static extent).

    A dimension may repeat in ``dims`` and then contributes its value once per occurrence.
    ``coeff`` folds in the constants the bound multiplies by (an atom's ``atom_n``, a warp's 32
    lanes), so the dimensions stay the only unknowns.
    """

    dims: tuple[str, ...]
    limit: int
    op: str = "<="
    coeff: int = 1

    def __post_init__(self) -> None:
        if not self.dims:
            raise ValueError(f"bound {self.op} {self.limit} names no dimension")
        if self.op not in _OPS:
            raise ValueError(f"bound op {self.op!r} is not one of {_OPS}")
        if self.limit < 1 or self.coeff < 1:
            raise ValueError(f"bound {self.spell()} needs a positive limit and coeff")

    def spell(self) -> str:
        """The bound as text — for error messages, not a stored codec."""
        lhs = "*".join(self.dims) if self.coeff == 1 else f"{self.coeff}*{'*'.join(self.dims)}"
        return f"{lhs} {self.op} {self.limit}"

    def holds(self, product: int, *, complete: bool) -> bool:
        """Whether ``product`` — ``coeff`` times the dims bound SO FAR — can still satisfy this
        bound. ``complete`` marks it final (every dim this bound names is bound).

        A partial product prunes because the final one is a multiple of it: over-budget stays
        over-budget, and a product that does not divide ``limit`` never will.
        """
        if self.op == "<=":
            return product <= self.limit
        if self.op == "==":
            return product == self.limit if complete else self.limit % product == 0
        return self.limit % product == 0


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
        # Per bound: the depth at which it becomes COMPLETE (the walk has bound the last dimension
        # it names) and how many times each dimension occurs in it.
        last = [max(i for i, d in enumerate(self.dims) if d.name in b.dims) for b in self.bounds]
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
                    if not bound.holds(running[bi], complete=i == last[bi]):
                        break
                else:
                    yield from walk(i + 1, {**point, dim.name: v}, tuple(running))

        yield from walk(0, {}, tuple(b.coeff for b in self.bounds))

    def points(self) -> list[dict[str, int]]:
        """Every legal point, enumerated."""
        return list(self)

    def contains(self, point: dict[str, int]) -> bool:
        """Whether ``point`` is a member: it binds exactly the declared dimensions, each to one of
        that dimension's values, and satisfies every bound. This is the check a recorded golden is
        tested against — "the search can still reach it" — without enumerating the space."""
        if set(point) != {d.name for d in self.dims}:
            return False
        if any(point[d.name] not in d.values for d in self.dims):
            return False
        return all(b.holds(b.coeff * prod(point[n] for n in b.dims), complete=True) for b in self.bounds)


__all__ = ["Bound", "Dimension", "Space"]
