"""The generic tiling layer — where a schedule's PLAN becomes actual :class:`Axis` objects.

``020_schedule`` decides (a ``TilePlan`` / ``ReducePlan`` / ``Stage``, and which axis plays m, n or
k); nothing it produces is an axis a kernel loops over. This module is the other half: the four
levels a contraction's output cell is tiled through — GRID block / UNIT / REGISTER / ATOM —
realized as the bound ``Tile`` axes plus the per-cell coordinate arithmetic that indexes them.

    atomize → register_tile → unit_tile → grid_tile

Each level zips the per-axis :class:`AxisOffset` pair (``Tiling.offset``) with the ``(m, n)``
:class:`Side` pair, so the two axes never split into ``*_m`` / ``*_n`` locals, and each accumulates
one term of :meth:`AxisOffset.base` — the register cell's real coordinate, ``block·(units·reg·atom)
+ unit·(reg·atom) + r·atom``. The UNIT is the atom's parallel thread footprint: a warp for mma, a
single thread for scalar, so the tensor-core warp tile and the scalar parallel thread-tile are the
SAME level, differing only in the atom's ``lanes``. :func:`grid_tile` is the finalizer and the ONE
seal every kernel binds through, whatever tier built it.

This layer is **algebra-free**: it knows a `Side` pair, integer counts and three callables, and
nothing about node kinds, contractions, reduce plans, operands or the ambient ``Ctx``. What varies
per tier arrives as those callables (``state_decls`` / ``reduce_region`` / ``store`` — from
``_atom.reduce_codegen`` + a sink for a contraction, from the reduce tier's own fold otherwise);
the splice is shared. Keeping it apart from ``_factor`` (which walks nodes and dispatches tiers)
is what makes that separation checkable rather than merely intended.

Leading ``_`` so the pass loader skips this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expr import BinaryExpr, Builtin, Expr, Literal, Var
from emmy.compiler.ir.kernel import Tile
from emmy.compiler.ir.stmt import Body, Cond, Stmt
from emmy.compiler.ir.tile.ir import Side


def shrink_axis(axis: Axis, reg: int) -> Axis:
    """The grid (cell) axis for a register-tiled free axis: ``ceil(E / reg)`` cells, each a
    per-thread ``reg``-wide register sub-tile. ``Dim.ceil_div`` keeps a symbolic extent
    symbolic (``(seq_len+reg-1)//reg``) so the launch grid sizes from the runtime extent."""
    if reg <= 1:
        return axis
    return Axis(name=axis.name, extent=axis.extent.ceil_div(reg), window=Window(parent=axis.source_axis or axis))


@dataclass(frozen=True)
class AxisOffset:
    """One output axis's per-register-cell coordinate, accumulated across the tiling levels (atom →
    register → unit → grid). :meth:`base` reproduces ``block·(units·reg·atom) + unit·(reg·atom) +
    r·atom`` once the UNIT level is present (the mma warp tile AND the scalar thread tile both go
    through :func:`unit_tile`), else the bare ``Var(block)·reg + r``."""

    atom_dim: int  # the atom step along this axis
    reg: int = 1  # register sub-cells per unit
    block_var: str | None = None  # the grid-block axis var (set at grid_tile)
    unit_var: str | None = None  # the UNIT-level var — a warp for mma, a thread for scalar
    unit_count: int = 1

    def base(self, r: int) -> Expr:
        """The offset of register cell index ``r`` along this axis."""
        reg_term = Literal(r * self.atom_dim, "int")
        if self.unit_var is not None:  # block·(units·reg·atom) + unit·(reg·atom) + r·atom
            tile = self.unit_count * self.reg * self.atom_dim
            e = BinaryExpr("*", Var(self.block_var), Literal(tile, "int"))
            e = BinaryExpr("+", e, BinaryExpr("*", Var(self.unit_var), Literal(self.reg * self.atom_dim, "int")))
            return BinaryExpr("+", e, reg_term)
        return BinaryExpr("+", BinaryExpr("*", Var(self.block_var), Literal(self.reg, "int")), reg_term)  # no unit level


@dataclass(frozen=True)
class Tiling:
    """The accumulating tiling state threaded through ``atomize → register_tile → unit_tile →
    grid_tile`` — the per-axis ``(m, n)`` :class:`AxisOffset` tuple ``offset`` + the bound ``Tile``
    axes (unit → grid) + ``block_threads``. Each level ``zip``\\ s ``offset`` with the ``(m, n)``
    :class:`Side` pair, so the two axes never split into ``*_m`` / ``*_n`` locals."""

    offset: tuple[AxisOffset, AxisOffset]
    axes: tuple[Axis, ...] = ()
    block_threads: int | None = None


def atomize(atoms: tuple[int, int]) -> Tiling:
    """The leaf: a single ``(atom_m, atom_n)`` atom (1×1 for a scalar cell). Seeds the per-axis
    offset with the atom step; the atom-lane offset stays OUT of σ (added at render)."""
    return Tiling(offset=tuple(AxisOffset(atom_dim=a) for a in atoms))


def register_tile(t: Tiling, mn: tuple[Side, Side]) -> Tiling:
    """The REGISTER level: ``m.reg × n.reg`` atoms per thread/warp. Records the cell counts; the
    per-cell ``r·atom_dim`` term is applied at :meth:`AxisOffset.base`."""
    return replace(t, offset=tuple(replace(o, reg=s.reg) for o, s in zip(t.offset, mn, strict=True)))


def unit_tile(t: Tiling, mn: tuple[Side, Side]) -> Tiling:
    """The UNIT level: ``m.units × n.units`` parallel units per CTA, where a *unit* is the atom's
    thread footprint — a warp (32 lanes) for an mma atom, a single thread for a scalar atom (so the
    tensor-core warp tile and the scalar parallel thread-tile are the same level, differing only in
    the atom's ``lanes``). Adds the unit term ``unit·(reg·atom)`` to each axis offset + the per-axis
    unit axes."""
    offset = tuple(replace(o, unit_var=s.unit, unit_count=s.units) for o, s in zip(t.offset, mn, strict=True))
    axes = (*t.axes, *(Axis(name=s.unit, extent=s.units) for s in mn))
    return replace(t, offset=offset, axes=axes)


def grid_tile(
    t: Tiling,
    *,
    mn: tuple[Side | None, Side | None],
    lead_axes: tuple[Axis, ...] = (),
    block_threads: int | None,
    lanes: int = 1,
    state_decls: Callable[[list[tuple[int, int]]], list[Stmt]],
    reduce_region: Callable[..., tuple[list[Stmt], list[Stmt]]],
    store: Callable[..., list[Stmt]],
    workers: object = None,
    raster: object = None,
) -> Tile:
    """The GRID level + finalize — the ONE seal every kernel binds through: bind the block axes (the
    shrunk grid), set the per-axis grid term ``block·tile``, append any leading (untiled) grid axes
    verbatim and — when the atom is warp-cooperative (``lanes > 1``) — the atom ``_lane`` axis, then
    splice the codegen callables' state + reduce-region + per-cell stores into the ``Tile``. The
    three callables (atom-specific for a contraction, from :func:`~...kernel._atom.reduce_codegen` +
    the ``store`` sink; the reduce tier's fill / partitioned fold / projection close) are the only
    variation; the splice is shared. They take the per-cell ``offset`` (the ``(m, n)``
    :class:`AxisOffset` tuple) + the ``mn`` :class:`Side` pair.

    ``mn[0] is None`` is a 1-D output grid (only ``n`` tiled) — no ``m`` block axis is bound.
    ``mn == (None, None)`` is the fully-untiled output (the reduce tier / degenerate fold): one cell
    per thread, no block axis at all — the whole grid rides ``lead_axes``, and a tiled REDUCE axis
    contributes its lane through ``t.axes``. ``lanes == 1`` (scalar) emits no ``_lane`` axis.

    ``workers`` (a resolved :class:`WarpSpec`) appends its producer band as ``Tile.aux_threads`` and
    guards the stores to the compute band — an aux thread's wrapped decode aliases a compute cell,
    so an unguarded ``store`` would double-write it."""
    offset = tuple(replace(o, block_var=s.block) if s is not None else o for o, s in zip(t.offset, mn, strict=True))
    block_axes = tuple(
        shrink_axis(Axis(name=s.block, extent=s.axis.extent, window=Window(parent=s.axis)), s.tile) for s in mn if s is not None
    )
    lane_axes = (Axis(name="_lane", extent=lanes),) if lanes > 1 else ()
    axes = (*lead_axes, *block_axes, *t.axes, *lane_axes)

    cells = [(i, j) for i in range(offset[0].reg) for j in range(offset[1].reg)]
    state = state_decls(cells)
    top_decls, kstmts = reduce_region(cells, offset, mn)
    stores = [s for (i, j) in cells for s in store(i, j, offset, mn)]
    aux_threads = 0
    if workers is not None:
        aux_threads = 32 * workers.aux_warps
        stores = [Cond(cond=BinaryExpr("<", Builtin("thread_idx.x"), Literal(block_threads, "int")), body=tuple(stores))]
    # A 2-D-tiled output (both mn Sides bound) marks its (m, n) block axes as
    # rasterization-eligible — the structural fact only this seal knows; whether (and how) the
    # CTA order actually groups them is the resolved ``RASTER`` codec threaded down off the
    # TileOp's knobs (``None`` / ineligible ⇒ the flat N-fastest order, byte-identical codegen).
    raster_axes = (mn[0].block, mn[1].block) if (mn[0] is not None and mn[1] is not None) else None
    return Tile(
        axes=axes,
        body=Body((*state, *top_decls, *kstmts, *stores)),
        block_threads=block_threads,
        aux_threads=aux_threads,
        raster_axes=raster_axes,
        raster_group=(raster.group if raster is not None and raster_axes is not None else None),
        raster_orient=(raster.orient if raster is not None else "m"),
    )
