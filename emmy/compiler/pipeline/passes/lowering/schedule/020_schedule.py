"""Schedule a recognized (UNMAPPED) ``TileOp``: map its free axes onto the grid and offer the
scheduling fork — the second half of the Loop-IR → Tile-IR boundary.

``010_recognize`` is purely structural: it reads the algebra off a ``LoopOp`` and emits an UNMAPPED
:class:`~emmy.compiler.ir.tile.ir.TileOp` (its ``op`` set, ``place`` carrying just the free axes).
THIS rule picks that up and decides the schedule — the free-axis → grid mapping plus the per-node
``TILE`` / ``REDUCE`` / ``STAGE`` / ``WORK`` / ``RASTER`` families — through the ``_schedule``
helper's ONE recursive row enumerator over the term's own site tree.

Splitting the two halves is what makes the fork ONE thing: a kernel reaches scheduling by three
routes — the ordinary lift, flash's graph rewrite, and a placement cut's recognized pieces — and
all three converge here. The engine restarts its rule scan after every functional rewrite, so a
``TileOp`` this pass's ``010`` just emitted is matched here on the next sweep, exactly as
``030_split_reduce`` already matches the ``TileOp``\\ s this rule produces.

**Empty enumeration is a skip, not a failure.** A term the enumerator cannot schedule leaves the
``TileOp`` unmapped and the materializer lowers it on its per-cell path. That is the guardrail
contract: the enumeration returns ``[]``, never raises, and this rule never guesses a schedule it
could not enumerate.
"""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.fork import Fork

# NOTE: no ``Knob`` objects (``TILE`` / ``REDUCE`` / ``STAGE``) may be imported here — ``Pass.load``
# scans rule modules for ``Knob`` attrs and OFF-fills any it finds bare onto every variant of the
# pass. Pin reads / knob-key spelling ride the enumerator's helpers instead.
from emmy.compiler.pipeline.passes.lowering.tile._schedule import schedule

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node, ctx=None) -> Fork | list[TileOp] | TileOp:
    del match  # the scheduled op replaces the matched node in place — no graph surgery here
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped:
        raise RuleSkipped("TileOp already scheduled / nothing to map")
    rows = schedule(tile, tile.name, tile.knobs, ctx)
    options = rows if isinstance(rows, list) else [rows]
    if not options:
        raise RuleSkipped("no enumerable schedule row for this term — leave it unmapped")
    return options if len(options) > 1 else options[0]
