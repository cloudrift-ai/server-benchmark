"""Tile IR — a map/reduce kernel with its schedule made explicit.

See :mod:`.ir` and :mod:`.schedule`. The layer between Loop IR and Kernel IR: a
:class:`TileOp` holds the structural-IR root ``op`` (a :class:`~.ir.Fold` / :class:`~.ir.Fold` /
:class:`~.ir.Contraction`, with computed operands stored inline on their edges) plus
``place`` / ``work`` / ``knobs`` and the tree-path-keyed ``schedule`` dict, so the *schedule*
(free axes, reduce partition, grid binding) stays separate from the term; dispatch reads
``ops.axis_role``, no per-kind type.
"""

from emmy.compiler.ir.atom import AtomKind
from emmy.compiler.ir.schedule import (
    FoldMove,
    Level,
    Placement,
    ReducePlan,
    ReduceStage,
    Stage,
    TilePlan,
    WarpSpec,
)
from emmy.compiler.ir.tile.ir import (
    Channel,
    Contraction,
    Fold,
    Store,
    TileOp,
    deep_defines,
    deep_reads,
    effect_tail,
    refs_axis,
    split_effects,
    stmt_axis_names,
)

__all__ = [
    "AtomKind",
    "Channel",
    "Contraction",
    "FoldMove",
    "Level",
    "Placement",
    "ReducePlan",
    "ReduceStage",
    "Fold",
    "Stage",
    "Store",
    "TileOp",
    "TilePlan",
    "WarpSpec",
    "deep_defines",
    "deep_reads",
    "effect_tail",
    "refs_axis",
    "split_effects",
    "stmt_axis_names",
]
