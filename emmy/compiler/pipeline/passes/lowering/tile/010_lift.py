"""Lift a ``LoopOp`` completely into one unmapped Fold-tree ``TileOp``."""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.lowering.tile._lift import lift_tile

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node, ctx=None) -> TileOp:
    del match, ctx
    loop: LoopOp = root.op
    tile = lift_tile(loop, name=loop.name)
    tile.outputs = {root.output.name: root.output}
    return tile
