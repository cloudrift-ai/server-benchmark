"""Rewrite the general exp-family Fold cluster before Tile scheduling."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.tile._twist import rewrite_twisted

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node, ctx=None) -> TileOp:
    del match, ctx
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped or tile.schedule is not None:
        raise RuleSkipped("TileOp already scheduled / nothing to rewrite")
    op = rewrite_twisted(tile.op)
    if op == tile.op:
        raise RuleSkipped("no exp-family Fold cluster")
    return replace(tile, op=op)
