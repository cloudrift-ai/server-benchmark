"""Offer the cross-CTA reduce split beside the unsplit tree — a STRUCTURAL fork like ``030_cut``.

A split produces fresh kernels whose cost is a SUM (``policy/greedy._resolved_price``), exactly
like a cut — it is a kernel-set decision, never a per-row schedule knob — so it stands beside the
cut fork and BEFORE ``040_schedule``: the rewrite consumes only the stored Fold algebra, and each
piece re-enters the pass scan as a fresh unmapped ``TileOp`` that decides its own row. The offer,
the pin reading and the realization all live in the ``_split`` helper; a ``REDUCE`` pin's
``g<n>[a|k]`` half is consumed HERE (the pieces' sliced axes are the receipt), and the rest of the
pin's value reaches the pieces' own schedule walks.
"""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped

# NOTE: no ``Knob`` objects may be imported here — ``Pass.load`` scans rule modules for ``Knob``
# attrs and OFF-fills any it finds bare onto every variant of the pass. Pin reads live in the
# ``_split`` helper.
from emmy.compiler.pipeline.passes.lowering.tile._split import split_forks

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node, ctx=None):
    del ctx
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped or tile.schedule or tile.split_decided:
        raise RuleSkipped("TileOp already split-decided / scheduled")
    options = split_forks(match, root)
    if options is None:
        raise RuleSkipped("no reduce fold to split, or the kernel is itself a piece of a realized split")
    # A one-option offer stays a FORK (the lone unsplit arm included), exactly like ``030_cut``'s
    # pinned-fuse path and a forced schedule walk: the engine records a one-option fork as a
    # decision, which is what keys the choice into the trace.
    return options if len(options) > 1 else options[0]
