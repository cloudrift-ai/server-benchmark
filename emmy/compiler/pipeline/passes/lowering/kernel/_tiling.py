"""Seal already-blockified axes around a kernel body.

Symbolic blockification chooses the logical axes and schedule materialization binds their widths.
This module does not split axes or compute coordinates. It only places the bound GRID and UNIT
axes around the state, reduction, and store regions supplied by the lowering tier.
"""

from __future__ import annotations

from collections.abc import Callable

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Builtin, Literal
from emmy.compiler.ir.kernel import Tile
from emmy.compiler.ir.schedule import Side
from emmy.compiler.ir.stmt import Body, Cond, Stmt


def grid_tile(
    *,
    mn: tuple[Side, Side] | tuple[None, None],
    lead_axes: tuple = (),
    inner_axes: tuple = (),
    block_threads: int | None,
    lanes: int = 1,
    state_decls: Callable[[list[tuple[int, int]]], list[Stmt]],
    reduce_region: Callable[..., tuple[list[Stmt], list[Stmt]]],
    store: Callable[..., list[Stmt]],
    workers: object = None,
    raster: object = None,
) -> Tile:
    """Place one resolved block domain around the three lowering regions."""
    tiled = mn[0] is not None and mn[1] is not None
    if tiled:
        sides = mn
        block_axes = tuple(side.axes[0] for side in sides)
        unit_axes = tuple(side.axes[1] for side in sides)
        cells = [(i, j) for i in range(sides[0].reg) for j in range(sides[1].reg)]
        offset = sides
    else:
        block_axes = unit_axes = ()
        cells = [(0, 0)]
        offset = (None, None)
    lane_axes = (Axis("_lane", lanes),) if lanes > 1 else ()
    axes = (*lead_axes, *block_axes, *unit_axes, *inner_axes, *lane_axes)

    state = state_decls(cells)
    top_decls, reduction = reduce_region(cells, offset, mn)
    stores = [stmt for i, j in cells for stmt in store(i, j, offset, mn)]
    aux_threads = 0
    if workers is not None:
        aux_threads = 32 * workers.producer_warps
        stores = [Cond(cond=BinaryExpr("<", Builtin("thread_idx.x"), Literal(block_threads, "int")), body=tuple(stores))]
    raster_axes = (mn[0].block, mn[1].block) if tiled else None
    return Tile(
        axes=axes,
        body=Body((*state, *top_decls, *reduction, *stores)),
        block_threads=block_threads,
        aux_threads=aux_threads,
        raster_axes=raster_axes,
        raster_group=(raster.group if raster is not None and not raster.is_direct and raster_axes is not None else None),
        raster_orient=(raster.orient if raster is not None and not raster.is_direct else "m"),
    )
