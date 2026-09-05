"""Install the symbolic axis blocks that every later schedule choice binds."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.tile import TileOp, blockify
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node, ctx=None) -> TileOp:
    del match, ctx
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped or tile.schedule is not None:
        raise RuleSkipped("TileOp already scheduled / nothing to block")
    blocks = blockify(tile)
    if blocks == tile.blocks:
        raise RuleSkipped("symbolic blocks already installed")
    return replace(tile, blocks=blocks)
