"""Fuse the exp-family reduce pair into its twisted carrier — offered beside the two-pass tree.

The carrier is what ``Fold.twist`` builds; the two-pass tree is what the lift reconstructs from that
carrier's own online loop (``_twist.relift`` through ``_untwist``), so both offers are the lift's
output. ``TWIST`` is an input pin like ``FAST_MATH``: pinned, it decides here; unpinned, the pass
forks, the carrier first (the cold pick keeps the single-pass kernel) and the two-pass tree second
as a structural arm — a different kernel, whose value channel is a contraction node with a ``TILE``
site of its own, at the price of one more pass over the axis.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.knob import family_pins
from emmy.compiler.pipeline.passes.lowering.tile._twist import relift, rewrite_twisted

PATTERN = [Pattern("root", TileOp)]

TWISTED, TWO_PASS = "twisted", "two-pass"


def rewrite(match: Match, root: Node, ctx=None):
    del ctx
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped or tile.schedule is not None:
        raise RuleSkipped("TileOp already scheduled / nothing to rewrite")
    op = rewrite_twisted(tile.op, tile.axes)
    if op == tile.op:
        raise RuleSkipped("no exp-family Fold cluster")
    twisted = replace(tile, op=op)
    pin = dict(family_pins("TWIST")).get("TWIST")
    if pin == TWISTED:
        return twisted
    if pin not in {None, TWO_PASS}:
        raise ValueError(f"bad TWIST value {pin!r}; expected {TWISTED!r} or {TWO_PASS!r}")
    two_pass = relift(twisted, match.graph)
    if pin == TWO_PASS:
        if two_pass is None:
            raise ValueError(f"TWIST={TWO_PASS}: the lift does not reconstruct the two-pass tree of {tile.name}")
        return two_pass
    options = [DeferredFork(lambda: twisted, {"TWIST": TWISTED})]
    if two_pass is not None:
        options.append(DeferredFork(lambda: two_pass, {"TWIST": TWO_PASS}, structural=True))
    return options
