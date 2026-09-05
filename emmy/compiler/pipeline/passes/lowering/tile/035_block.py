"""Block a twisted carrier — split its stream into blocks so a channel reads as a contraction.

A NORMALIZATION, not a decision: the rewrite is parameter-free (the width is the form's, read off
the stream's own extent) and idempotent (every axis it installs carries the receipt), so it opens
no fork, adds no schedule family, and leaves one kernel identity per term. ``ir/tile/block.py``
owns the rewrite and states why only a twisted carrier is blocked.

It is a PASS rather than ``TileOp.__post_init__`` for one reason: the kernel-set cut runs at
``030_cut``, and a piece it mints is a different carrier from the one it was cut out of. Attention
cut at its value channel leaves a statistics piece whose channels are sums of the weight — nothing
there becomes bilinear, so nothing there should be blocked, and blocking ahead of the cut would
have handed that piece a block it pays a second pass for and gets nothing from. Running here, each
branch of the cut fork blocks what its own term deserves.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.block import block_tree
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node, ctx=None) -> TileOp:
    del match, ctx
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped or tile.schedule is not None:
        raise RuleSkipped("TileOp already scheduled / nothing to block")
    blocked = block_tree(tile.op, tile.axes)
    if blocked is None:
        raise RuleSkipped("no stream a block gives a contraction to")
    op, axes = blocked
    return replace(tile, op=op, axes=axes)
