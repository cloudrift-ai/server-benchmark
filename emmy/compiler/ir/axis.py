"""Axis primitives shared across all IR layers.

``Axis`` is the iteration-variable identity (name + extent) used by Loop
IR, Tile IR, and Kernel IR.

Lives at ``ir/axis.py`` rather than inside any one IR package because the
concept spans every layer. Loop IR re-exports ``Axis`` for convenience
(lifting passes use ``from ir.loop import ...`` to grab the full Loop-body
vocabulary in one import); Tile and Kernel IR import ``Axis`` directly
from here.

The pre-refactor ``BoundAxis`` / ``BIND_BLOCK`` / ``BIND_THREAD`` triple
that packaged "axis plus its launch-coord binding" has been deleted: the
typed tile flavors (``GridTile`` / ``ThreadTile`` / ``RegisterTile``)
carry bare ``Axis`` tuples and encode the binding in the flavor's type.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from emmy.compiler.dim import Dim, to_dim
from emmy.compiler.ir.expr import Expr, Interval, Literal, SimplifyCtx


class AxisRole(enum.Enum):
    """The scheduling **role** of one iteration axis — read structurally off the loop body,
    not a stored algebra kind (the removed ``classify_algebra`` tagged the whole *kernel*; this
    tags each *axis*, and stays re-derivable from the body).

    Detection (``lowering/tile``) stamps each loop with its role so scheduling dispatches on the
    axis's job, never on a node *type*:

    - ``FREE`` — a parallel / output-grid axis (no fold; one independent cell per value).
    - ``PLANAR`` — a plain reduce axis (``sum`` / ``max`` / ``mean`` — the degenerate ``id``
      twist). Several stacked ``PLANAR`` axes form a reduction *plane*.
    - ``CONTRACTION`` — a reduce axis whose body distributes a ``⊗`` lift over ≥ 2 operands (the
      matmul K axis).
    - ``TWISTED`` — a ψ-conjugated reduce carrying an ``exp``-family :class:`Twist` (online
      softmax / flash).

    Every role but ``FREE`` is a reduce axis (the carrier rides the loop)."""

    FREE = "free"
    PLANAR = "planar"
    CONTRACTION = "contraction"
    TWISTED = "twisted"

    @property
    def is_reduce(self) -> bool:
        """True for every fold role (everything but :attr:`FREE`)."""
        return self is not AxisRole.FREE


# Sentinel upper bound for a symbolic loop axis ``[0, hi]``. Only its ``lo = 0``
# matters (gives the non-negativity the ``(i*c + …)//c → i`` div fold needs);
# the finite ``hi`` never enables a wrong fold (``_div_mod_decompose``'s
# ``rng.hi < n`` check just fails for it). Matches ``dim._simplify``'s sentinel.
_SYMBOLIC_AXIS_HI = 1 << 30


@dataclass(frozen=True)
class Window:
    """The slice of a PARENT axis an axis walks — the ONE windowing vocabulary.

    ``parent`` is the pre-split axis this one was carved out of: top-level axes (the ones the
    frontend traces) carry no window at all, and every sub-axis the partition planner produces
    (``M_b`` / ``M_t`` / ``M_r`` from ``M``) points at the original. Downstream passes (MMA
    factorization, the scope-walk origin derivation) group surrounding axes by that identity
    instead of by name-suffix convention.

    ``base`` / ``bound`` are the slice's ABSOLUTE start / end in the parent's coordinates, set
    when a cross-CTA split hands each CTA its own window of a reduce stream (``030_split_reduce``
    shrinks the axis to the slice length): the fold walks its local ``[0, extent)`` and any
    consumer needing the absolute coordinate — gmem / TMA operand bases, the causal mask's key
    columns — adds ``base``. ``bound`` additionally stops a SYMBOLIC slice whose end falls
    mid-tensor (``min((s+1)·B, S)``): the realizer runs ``bound − base`` local steps and masks
    against it, since a mid-tensor slice end reads VALID keys belonging to the next slice, which
    the extent-only tail machinery would not exclude. ``None`` on both = the whole parent.
    """

    parent: Axis | None = None
    base: Expr | None = None
    bound: Expr | None = None


@dataclass(frozen=True)
class Axis:
    """One named iteration variable.

    Referenced from ``Expr`` subtrees by ``Var(name)``. ``extent`` is a
    :class:`Dim` — static (``Dim(32)``) in most cases today, symbolic
    (``Dim("seq_len")``) for dynamic dims. Construction coerces a bare
    ``int`` / ``str`` to ``Dim`` for ergonomics, so ``Axis("m", 32)``
    keeps working.

    ``window`` is the slice of a parent axis this one walks (:class:`Window` — its ``parent``
    provenance and, for a cross-CTA slice, the absolute ``base`` / ``bound``). ONE windowing
    concept, read by the realizer and the mask machinery alike. Excluded from equality / hashing so
    Var-rename invariance is preserved — two Axes with the same name and extent are the same axis
    regardless of which window of what they walk.
    """

    name: str
    extent: Dim
    window: Window | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.extent, Dim):
            object.__setattr__(self, "extent", to_dim(self.extent))

    @property
    def source_axis(self) -> Axis | None:
        """The pre-split axis this one was carved out of — the window's ``parent``."""
        return self.window.parent if self.window is not None else None

    def split(self, factor: int) -> tuple[Axis, Axis]:
        """Split this axis into ``(outer, inner)`` for tile-style decomposition.

        Outer extent is ``self.extent // factor``, inner extent is ``factor``.
        Names follow the ``f"{self.name}_o"`` / ``f"{self.name}_i"`` convention
        so tiled IR remains readable. v1 requires divisibility — non-divisible
        extents need a residue-tail story that no current rule wants. Symbolic
        extents refuse to split (M3 of the dynamic-shapes plan).

        Children inherit the window's ``parent = self.source_axis or self`` —
        top-level axes become their own parent on first split; further splits
        chain to the same original.
        """
        ext = self.extent.as_static()
        if ext % factor != 0:
            raise ValueError(f"Axis.split: {self.name} extent {ext} not divisible by {factor}")
        w = Window(parent=self.source_axis or self)
        return (Axis(f"{self.name}_o", ext // factor, window=w), Axis(f"{self.name}_i", factor, window=w))

    def extent_expr(self) -> Expr:
        """This axis's extent as an ``Expr`` — a literal int (static) or the symbolic ``Dim``
        expr (dynamic ``seq_len``)."""
        return Literal(self.extent.as_static(), "int") if self.extent.is_static else self.extent.expr


def extend_simplify_ctx(ctx: SimplifyCtx, axis: Axis) -> SimplifyCtx:
    """Push an iteration axis ``[0, extent)`` into a ``SimplifyCtx`` so index
    expressions over it fold.

    A **static** extent gets a precise ``Interval`` (as before) plus its size as
    a literal exclusive bound. A **symbolic** extent — previously dropped, which
    left symbolic-seq indices unsimplified — gets ``[0, sentinel]`` (the
    non-negativity the ``(i*c + …)//c → i`` div fold needs) plus its extent
    ``Expr`` as an exclusive upper bound, so ``i % extent → i`` folds. Together
    these collapse the delinearized seq coordinate (``(i*stride + feat)/stride %
    seq_len``) back to ``i`` exactly as a static seq does, removing the runtime
    integer div/mod from masked-tile repack kernels."""
    ext = axis.extent
    if ext.is_static:
        n = ext.as_static()
        return ctx.extend(axis.name, Interval(0, n - 1), bound=Literal(n, "int"))
    return ctx.extend(axis.name, Interval(0, _SYMBOLIC_AXIS_HI), bound=ext.expr)


__all__ = ["Axis", "AxisRole", "Window", "extend_simplify_ctx"]
