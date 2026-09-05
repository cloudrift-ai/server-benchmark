"""Tile IR — a map/reduce kernel with its schedule made explicit.

See :mod:`.ir` and :mod:`.schedule`. The layer between Loop IR and Kernel IR: a
:class:`TileOp` holds one structural-IR root ``op`` (a :class:`Fold`, with contractions derived
from its algebra and computed operands stored inline on their edges), structural placement, one
accepted :class:`Schedule`, separate materialization facts, and search knobs. The schedule stays
separate from the term; dispatch reads the node's derived classification, not a per-kind type.
"""

from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule import FoldMove, Level, Placement, Reduce, ReduceStage
from emmy.compiler.ir.tile.block import (
    BlockAxis,
    BlockClaim,
    BoundBlockAxis,
    BoundBlockLoop,
    BoundSiteBlocks,
    BoundTransposedLoop,
    SiteBlocks,
    bind_site,
    blockify,
)
from emmy.compiler.ir.tile.ir import (
    OutputSpec,
    TileOp,
    apply_output_specs,
    extract_output_specs,
    observed_result_names,
)
from emmy.compiler.ir.tile.normalize import normalize_fold_tree

__all__ = [
    "BlockAxis",
    "BlockClaim",
    "BoundBlockAxis",
    "BoundBlockLoop",
    "BoundSiteBlocks",
    "BoundTransposedLoop",
    "Fold",
    "FoldMove",
    "Level",
    "Placement",
    "Reduce",
    "ReduceStage",
    "SiteBlocks",
    "OutputSpec",
    "TileOp",
    "apply_output_specs",
    "bind_site",
    "blockify",
    "observed_result_names",
    "normalize_fold_tree",
    "extract_output_specs",
]
