"""Tile IR — a map/reduce kernel with its schedule made explicit.

See :mod:`.ir` and :mod:`.schedule`. The layer between Loop IR and Kernel IR: a
:class:`TileOp` holds the structural-IR root ``op`` (a :class:`~.ir.Map` / :class:`~.ir.Fold` /
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
    Map,
    Store,
    TileOp,
    composed_contraction,
    demote_operands,
    effect_tail,
    split_effects,
)

__all__ = [
    "AtomKind",
    "Channel",
    "Contraction",
    "FoldMove",
    "Level",
    "Map",
    "Placement",
    "ReducePlan",
    "ReduceStage",
    "Fold",
    "Stage",
    "Store",
    "TileOp",
    "TilePlan",
    "WarpSpec",
    "composed_contraction",
    "demote_operands",
    "effect_tail",
    "split_effects",
]
