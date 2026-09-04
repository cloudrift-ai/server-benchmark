"""Give a twisted carrier's channels a semiring to live in, before scheduling sees the tree."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.tile._block import block_tree

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node, ctx=None) -> TileOp:
    del match, ctx
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped or tile.schedule is not None:
        raise RuleSkipped("TileOp already scheduled / nothing to rewrite")
    blocked = block_tree(tile.op, tile.axes)
    if blocked is None:
        raise RuleSkipped("no blockable twisted carrier")
    op, axes = blocked
    return replace(tile, op=op, axes=axes)
