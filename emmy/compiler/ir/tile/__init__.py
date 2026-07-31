"""Tile IR — a map/reduce kernel with its schedule made explicit.

See :mod:`.ir` and :mod:`.schedule`. The layer between Loop IR and Kernel IR: a
:class:`TileOp` holds the structural-IR root ``op`` (a :class:`~.ir.Map` / :class:`~.ir.Fold` /
:class:`~.ir.ContractionView`, with computed operands stored inline on their edges) plus
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
    ContractionView,
    Fold,
    Map,
    Store,
    TileOp,
    composed_contraction,
    contraction_view,
    demote_operands,
    effect_tail,
    is_contraction_fold,
    shared_operand,
    split_effects,
    stored_contraction,
)

__all__ = [
    "AtomKind",
    "Channel",
    "ContractionView",
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
    "contraction_view",
    "demote_operands",
    "effect_tail",
    "is_contraction_fold",
    "shared_operand",
    "split_effects",
    "stored_contraction",
]
