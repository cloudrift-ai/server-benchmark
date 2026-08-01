"""The PLACED reading of a stored :class:`~emmy.compiler.ir.tile.ir.Contraction` — the caller's
placement + schedule facts bound to the node, as a lowering-side VIEW rather than node fields.

The stored node is pure algebra (`k_axis` + the `a` edge + the `Channel`s). Everything a tier needs
on top of that — which output axes the `(m, n)` cell tiles, which leading grid axes ride untiled,
the resolved `TilePlan` / `Stage` — belongs to the CALLER: the placement comes off `TileOp.place`
and the schedule slices out of `TileOp.schedule` (through `ops.Sched`, their one home since 1r).
Binding them here keeps the term free of schedule, so a node's identity (`ops.term_key`, `==`,
`hash`) is its algebra and nothing else — two kernels differing only in tile no longer key apart,
and no emission path can leak a scheduled node into a stored term.

Same shape as `_reduction.Reduction`: the retired stored fields survive as a lowering-side reading,
constructed at the point of use and never embedded. The derived geometry the tiers consume (`mn` /
`m` / `n` / `atom` / `block_threads`) lives on the view because it is a function of `tile` + `axes`;
every ALGEBRA read (`a`, `b`, `channels`, `k_axis`, `out`, `a_computed`, `b_trans`, `loop`, …)
proxies straight through to the node, so a tier holding a `Placed` reads it exactly as it read the
stamped copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from emmy.compiler.ir.schedule import Side, Stage, TilePlan
from emmy.compiler.ir.tile.ir import Contraction

if TYPE_CHECKING:
    from emmy.compiler.ir.atom import Atom
    from emmy.compiler.ir.axis import Axis


@dataclass(frozen=True)
class Placed:
    """A contraction node bound to its placement + schedule. Build it at the point of use — the
    materializer (`_factor.factorize`), the flash realizer (`_twist`) and the option builders
    (`_schedule`) — never store it: `node` is the term, `self` is the reading."""

    node: Contraction
    axes: tuple[Axis, Axis]  # placement: the tiled output (m_axis, n_axis)
    tile: TilePlan = field(default_factory=TilePlan)  # schedule: leaf atom + unit/register widths + K-chunk
    # The one MATERIALIZER-only member: a schedule-side probe (``tile/_view.contraction_view``)
    # leaves it None by design — the schedule's gates read the tiled cell alone — and ``stage`` is
    # anyway a schedule RESULT, living in ``TileOp.schedule`` and threaded back in at materialize.
    stage: Stage | None = None  # schedule: the resolved operand smem pipeline (None = gmem-direct)

    def __post_init__(self) -> None:
        """Bind the view's placement onto the ``tile`` slice, so the plan alone derives the ``(m,
        n)`` geometry. Every re-tiling of a view goes through ``dataclasses.replace(v, tile=plan)``
        with a plan straight out of the move catalog (axis-free), so binding here is what keeps
        those copies placed — the caller never has to remember ``.at``."""
        if self.tile.axes is None:
            object.__setattr__(self, "tile", self.tile.at(*self.axes))

    def __getattr__(self, name: str):
        """Every ALGEBRA read proxies to the stored node — `a` / `b` / `channels` / `k_axis` /
        `out` / `acc` / `a_computed` / `b_trans` / `loop` / `as_fold()` and the rest. Only the
        placement- and schedule-derived members below are the view's own, so a tier that used to
        hold a stamped `Contraction` copy reads a `Placed` unchanged."""
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "node"), name)

    def replace_node(self, **changes) -> Placed:
        """A view over an EDITED node — the algebra changes (split-K's σ-reindexed ``kslice``
        operands), the placement / schedule binding rides along unchanged. The successor of
        ``replace(placed_contraction, k_axis=…)`` back when the two were one object."""
        return replace(self, node=replace(self.node, **changes))

    @property
    def m_axis(self) -> Axis:
        return self.axes[0]

    @property
    def n_axis(self) -> Axis:
        return self.axes[1]

    @property
    def atom(self) -> Atom:
        return self.tile.atom

    # The ``(m, n)`` geometry is the PLAN's — the placement rides on it (``TilePlan.axes``), so the
    # Side pair derives there and this view only forwards. Nothing here re-derives a width.
    @property
    def mn(self) -> tuple[Side, Side]:
        """The ``(m, n)`` output sides — forwarded to :attr:`TilePlan.mn`."""
        return self.tile.mn

    @property
    def m(self) -> Side:
        return self.tile.m

    @property
    def n(self) -> Side:
        return self.tile.n

    @property
    def block_threads(self) -> int | None:
        return self.tile.launch_threads


def place(node: Contraction, m_axis: Axis, n_axis: Axis, tile: TilePlan | None = None, stage: Stage | None = None) -> Placed:
    """Bind ``node`` to its caller placement + schedule facts. The output axes are the CALLER's
    placement (the trailing grid axes for a root kernel; a flash consumer supplies its own) — they
    are bound onto the ``tile`` slice itself (:meth:`TilePlan.at`), so the geometry the tiers read
    is a function of the slice alone.

    The view is the TILED CELL's reading — the ``(m, n)`` pair and nothing outside it. A kernel's
    LEADING (batch / ksplit) grid axes are not bound here: the emission that needs them
    (``kernel/_atom``'s per-cell rename) takes them from the caller that owns the grid, so the
    view never carries a placement fact it cannot decide."""
    return Placed(node=node, axes=(m_axis, n_axis), tile=(tile if tile is not None else TilePlan()).at(m_axis, n_axis), stage=stage)
