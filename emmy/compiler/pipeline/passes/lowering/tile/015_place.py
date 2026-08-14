"""Resolve structural placement on recognized, unmapped tile algebra.

Recognition always builds the maximal algebraic region. This rule then offers the generic
``PLACE`` fork before scheduling: keep that region or materialize one closed child seam. Cut pieces
remain recognized ``TileOp`` trees, so placement cannot erase a contraction or other reading and
each piece reaches the same schedule enumerator independently.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_prologue_contraction
from emmy.compiler.pipeline.passes.lowering.tile._cut import placement_options

PATTERN = [Pattern("root", TileOp)]

_RESOLVED = "placement_resolved"


def rewrite(match: Match, root: Node, ctx=None):
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped or tile.meta.get(_RESOLVED):
        raise RuleSkipped("placement already resolved / nothing to place")

    tree = tile.op
    free = tuple(tile.place.free)
    stores = tuple(tile.stores)
    prologue = bind_prologue_contraction(tree, free)
    if prologue is not None:
        tree, n_axis, stores = prologue
        free = (*free, n_axis)

    fused = replace(tile, meta={**tile.meta, _RESOLVED: True})
    return placement_options(ctx, dict(tile.knobs or {}), match, root, tree, free, stores, fused)
