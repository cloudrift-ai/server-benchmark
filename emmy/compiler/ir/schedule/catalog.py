"""Finite move catalogs used by the classic schedule model."""

from __future__ import annotations

from emmy.compiler.ir.atom import ATOM_REGISTRY

from . import Reduce, Stage, Tile
from .domain import Bound, Dimension, Space

MAX_BLOCK_THREADS = 1024
WARP_LANES = 32
MAX_FRAGMENT_CELLS = 32
MAX_FRAGMENT_REGISTERS = 128
MAX_REGISTERS_PER_THREAD = 255
MAX_REGISTERS_PER_CTA = 64 * 1024

_SCALAR_REGISTER_SPACE = Space(
    dims=(
        Dimension("reg_n", (1, 2, 3, 4)),
        Dimension("reg_m", (1, 2, 4)),
    )
)

_SCALAR_1D_TILE_SPACE = Space(dims=(Dimension("par_n", (32, 64, 128, 256, 512)),))

_SCALAR_PARALLEL_TILE_SPACE = Space(
    dims=(
        Dimension("par_n", (16, 32, 64)),
        Dimension("par_m", (8, 16)),
        Dimension("reg_n", (1, 2, 4, 26)),
        Dimension("reg_m", (1, 2, 4, 6, 8, 10, 12, 14, 26)),
    ),
    bounds=(Bound(("par_n", "par_m"), limit=MAX_BLOCK_THREADS),),
)


def scalar_tile_moves() -> list[Tile]:
    """Return the finite scalar-contraction tile domain."""
    moves = [Tile()]
    moves.extend(Tile(units=(1, point["par_n"])) for point in _SCALAR_1D_TILE_SPACE)
    moves.extend(
        Tile(regs=(point["reg_m"], point["reg_n"])) for point in _SCALAR_REGISTER_SPACE if (point["reg_m"], point["reg_n"]) != (1, 1)
    )
    moves.extend(
        Tile(
            units=(point["par_m"], point["par_n"]),
            regs=(point["reg_m"], point["reg_n"]),
        )
        for point in _SCALAR_PARALLEL_TILE_SPACE
    )
    return moves


_WARP_TILE_SPACE = Space(
    dims=(
        Dimension("wm", (1, 2, 4, 8, 16)),
        Dimension("wn", (1, 2, 4, 8, 16)),
        Dimension("fm", (1, 2, 4, 8)),
        Dimension("fn", (1, 2, 4, 8)),
        Dimension("bk", (1, 2, 4, 8)),
    ),
    bounds=(
        Bound(("wm", "wn"), limit=MAX_BLOCK_THREADS, coeff=WARP_LANES),
        Bound(("fm", "fn"), limit=MAX_FRAGMENT_CELLS),
    ),
)


def warp_tile_moves(atom_names: tuple[str, ...]) -> list[Tile]:
    """Return the finite warp tile domain for the supplied atom families."""
    moves = []
    for name in atom_names:
        atom = ATOM_REGISTRY[name]
        moves.extend(
            Tile(
                atom=atom,
                units=(point["wm"], point["wn"]),
                regs=(point["fm"], point["fn"]),
                bk=point["bk"],
            )
            for point in _WARP_TILE_SPACE
            if point["fm"] * point["fn"] * atom.accumulator_registers_per_lane <= MAX_FRAGMENT_REGISTERS
        )
    return moves


def stage_moves(*, warp: bool, ctx=None) -> list[Stage]:
    """Return the finite staging domain, filtered to transports available on ``ctx``."""
    depths = [
        Stage.parse(spelling)
        for spelling in (
            "d1/smem-async",
            "d2/smem-async",
            "d3/smem-async",
            "d4/smem-async",
            "d1/smem-tma",
            "d2/smem-tma",
            "d3/smem-tma",
            "d4/smem-tma",
        )
    ]
    if warp:
        smem = [Stage.parse(spelling) for spelling in ("d1/smem", "d2/smem", "d3/smem", "d4/smem", "d1/smem/p2", "d2/smem/p2")]
        depths = [*smem, *depths, Stage.parse("d2/smem-async/p2"), Stage.parse("d2/smem-tma/p2")]
    return depths if ctx is None else [move for move in depths if move.available_on(ctx)]


def raster_moves() -> tuple[str, ...]:
    """Return the finite kernel raster domain."""
    return "", "gm8", "gn4", "gn8"


def producer_band_moves() -> tuple[int, ...]:
    """Return the finite producer-band domain, including uniform execution."""
    return 0, 1, 2


SPLITK_WIDTHS: tuple[int, ...] = (2, 4, 8, 16, 32)


def splitk_moves() -> list[Reduce]:
    """Return cross-CTA split choices for both supported finalization modes."""
    return [Reduce.of(cta=width, finalize=finalize) for width in SPLITK_WIDTHS for finalize in ("kernel", "atomic")]


def coop_reduce_moves() -> list[Reduce]:
    """Return the finite cooperative and register reduction domain."""
    return [
        *(Reduce.of(coop=coop, reg=reg) for coop in (1, 4, 8, 16, 32, 64, 128, 256, 512) for reg in (1, 2, 4) if coop > 1 or reg > 1),
        *(Reduce.of(coop=width, coop_transposed=True) for width in (32, 64, 128, 256)),
    ]


__all__ = [
    "MAX_BLOCK_THREADS",
    "MAX_FRAGMENT_CELLS",
    "MAX_FRAGMENT_REGISTERS",
    "MAX_REGISTERS_PER_CTA",
    "MAX_REGISTERS_PER_THREAD",
    "SPLITK_WIDTHS",
    "WARP_LANES",
    "coop_reduce_moves",
    "producer_band_moves",
    "raster_moves",
    "scalar_tile_moves",
    "splitk_moves",
    "stage_moves",
    "warp_tile_moves",
]
