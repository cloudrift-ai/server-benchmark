"""Block a twisted carrier — split its stream into blocks so a channel reads as a contraction.

A NORMALIZATION, not a decision: the rewrite is parameter-free (the width is the form's, read off
the stream's own extent) and idempotent (every axis it installs carries the receipt), so it opens
no fork, adds no schedule family, and leaves one kernel identity per term. ``ir/tile/block.py``
owns the rewrite and states why only a twisted carrier is blocked.

It is a PASS rather than ``TileOp.__post_init__``, and it runs AFTER ``030_cut``, for one reason: a
piece the cut mints is a different carrier from the one it came out of. Attention cut at its value
channel leaves a statistics piece whose channels are sums of the weight — nothing there becomes
bilinear, so nothing there should be blocked, and blocking ahead of the fork handed that piece a
block it paid a second pass over the stream for (an 18.8 ms piece on a 512-key head where the
fused kernel is 128 µs). Running here, each branch of the fork blocks what its own term deserves.
A block is still a working set inside ONE kernel, so no seam evaluated over a block coordinate is
cuttable either (``_cut.cuttable_seams``).
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
