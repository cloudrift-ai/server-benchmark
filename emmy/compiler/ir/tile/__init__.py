"""Tile IR — a map/reduce kernel with its schedule made explicit.

See :mod:`.ir` and :mod:`.schedule`. The layer between Loop IR and Kernel IR: a
:class:`TileOp` holds the structural-IR root ``op`` (a :class:`~.ir.Fold` / :class:`~.ir.Fold` /
a contraction, with computed operands stored inline on their edges) plus
``place`` / ``work`` / ``knobs`` and the tree-path-keyed ``schedule`` dict, so the *schedule*
(free axes, reduce partition, grid binding) stays separate from the term; dispatch reads the
node's derived ``Fold.role``, no per-kind type.
"""

from emmy.compiler.ir.schedule import FoldMove, Level, Placement, ReducePlan, ReduceStage
from emmy.compiler.ir.tile.ir import Channel, Fold, Store, TileOp, effect_tail, split_effects

__all__ = [
    "Channel",
    "Fold",
    "FoldMove",
    "Level",
    "Placement",
    "ReducePlan",
    "ReduceStage",
    "Store",
    "TileOp",
    "effect_tail",
    "split_effects",
]
