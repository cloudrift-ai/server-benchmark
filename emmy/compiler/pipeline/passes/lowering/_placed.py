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

from emmy.compiler.ir.schedule import Stage, TilePlan
from emmy.compiler.ir.tile.ir import Contraction, Side

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
    lead_axes: tuple[Axis, ...] = ()  # placement: the leading (batch / ksplit) grid axes
    tile: TilePlan = field(default_factory=TilePlan)  # schedule: leaf atom + unit/register widths + K-chunk
    stage: Stage | None = None  # schedule: the resolved operand smem pipeline (None = gmem-direct)

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

    # ---- the (m, n) output sides: each axis paired with its derived per-CTA tile geometry (width /
    # unit / register counts, read off the ``tile``) + the bound block/unit var names (the original
    # m/n names live in the operand indices, so the bound axes take a fresh ``_b`` / ``_u`` suffix).
    # ``_factor`` threads these as the ``(m, n)`` pair, mirroring the schedule's tuples. ---------- #
    @property
    def mn(self) -> tuple[Side, Side]:
        """The ``(m, n)`` output sides — each output axis paired with its derived per-CTA tile
        geometry (width / unit / register counts, read off the ``tile``). Built as the pair;
        :attr:`m` / :attr:`n` index it."""
        t = self.tile
        dims = ((t.tile_m, t.units_m, t.reg_m), (t.tile_n, t.units_n, t.reg_n))
        return tuple(
            Side(axis=ax, tile=tile, units=units, reg=reg, block=ax.name + "_b", unit=ax.name + "_u")
            for ax, (tile, units, reg) in zip(self.axes, dims, strict=True)
        )

    @property
    def m(self) -> Side:
        return self.mn[0]

    @property
    def n(self) -> Side:
        return self.mn[1]

    @property
    def block_threads(self) -> int | None:
        bt = self.tile.block_threads
        return bt if bt > 1 else None  # None ⇒ the scalar default block size


def place(
    node: Contraction,
    m_axis: Axis,
    n_axis: Axis,
    lead_axes: tuple = (),
    tile: TilePlan | None = None,
    stage: Stage | None = None,
) -> Placed:
    """Bind ``node`` to its caller placement + schedule facts — the successor of the retired
    ``Contraction.placed`` field stamp. The output axes are the CALLER's placement (the trailing
    grid axes for a root kernel; a flash consumer supplies its own), the ``tile`` / ``stage`` the
    CALLER's schedule slices read from ``TileOp.schedule``."""
    return Placed(node=node, axes=(m_axis, n_axis), lead_axes=tuple(lead_axes), tile=tile if tile is not None else TilePlan(), stage=stage)
