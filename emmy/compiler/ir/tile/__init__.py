"""Tile IR — a map/reduce kernel with its schedule made explicit.

See :mod:`.ir` and :mod:`.schedule`. The layer between Loop IR and Kernel IR: a
:class:`TileOp` holds one structural-IR root ``op`` (a :class:`Fold`, with contractions derived
from its algebra and computed operands stored inline on their edges) plus
``place`` / ``work`` / ``knobs`` and the tree-path-keyed ``schedule`` dict, so the *schedule*
(free axes, reduce partition, grid binding) stays separate from the term; dispatch reads the
node's derived ``Fold.role``, no per-kind type.
"""

from emmy.compiler.ir.pure.fold import Channel, Fold
from emmy.compiler.ir.schedule import FoldMove, Level, Placement, ReducePlan, ReduceStage
from emmy.compiler.ir.tile.ir import (
    OutputSpec,
    ProjectionRegion,
    TileOp,
    apply_output_specs,
    extract_output_specs,
    lower_with_output_specs,
)
from emmy.compiler.ir.tile.normalize import lambda_equivalent_clusters, normalize_fold_tree

__all__ = [
    "Channel",
    "Fold",
    "FoldMove",
    "Level",
    "Placement",
    "ReducePlan",
    "ReduceStage",
    "OutputSpec",
    "ProjectionRegion",
    "TileOp",
    "apply_output_specs",
    "lambda_equivalent_clusters",
    "lower_with_output_specs",
    "normalize_fold_tree",
    "extract_output_specs",
]
