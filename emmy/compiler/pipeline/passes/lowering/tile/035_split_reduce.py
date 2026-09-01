"""Enumerate cross-CTA reduction cuts before schedule assignments."""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.schedule import CutScheduleContext, schedule
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.tile._split import split_forks

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node, ctx=None):
    del ctx
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped or tile.schedule is not None or tile.split_consumed:
        raise RuleSkipped("TileOp already split-consumed / scheduled")
    choices = split_forks(match, root)
    if choices is None:
        raise RuleSkipped("no reduce fold to split, or the kernel is itself a piece of a realized split")
    options = tuple(assignment.kernel for assignment in schedule(CutScheduleContext(tuple(choices))))
    return list(options) if len(options) > 1 else options[0]
