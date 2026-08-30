"""Schedule a lifted (UNMAPPED) ``TileOp``: map its free axes onto the grid and offer the
scheduling fork — the second half of the Loop-IR → Tile-IR boundary.

``010_lift`` is purely structural: it reads the algebra off a ``LoopOp`` and emits an UNMAPPED
:class:`~emmy.compiler.ir.tile.ir.TileOp` (its ``op`` set, ``place`` carrying just the free axes).
THIS rule picks that up and decides the schedule — the free-axis → grid mapping plus the per-node
``TILE`` / ``REDUCE`` / ``STAGE`` / ``WORK`` / ``RASTER`` families through ``_classic``.

Direct projection, plain-reduction, scalar-contraction, precision-gated tensor-core,
materialized-operand copy staging, smem compute-fill staging, and kernel-global raster schedules are
live while the remaining classic domains are being reconstructed behind an explicit unavailable
boundary. The fixed candidate-space contract is Algorithm 1(c, p, t): the schedule restriction
``c`` is one immutable context evaluated only on complete assignments from the independently
projected kernel, node, and edge domains. A traversal may prune ``p``/``t`` incompatibility, but it
never unpacks ``c`` and cannot make traversal order semantic.

Splitting the two halves is what makes the fork ONE thing: a kernel reaches scheduling by
several routes — the ordinary lift and a cross-CTA split's partial and finalize — and all converge here. The engine restarts its
rule scan after every functional rewrite, so a ``TileOp`` this pass's ``010`` just emitted is
matched here on the next sweep, and so is every unmapped ``TileOp`` a structural rewrite minted.
That is exactly why none of them needs a special case: each arrives as a kernel with no schedule,
like any other, and this rule cannot tell them apart.

Once reconstruction reaches this rule, empty enumeration remains a skip rather than a guessed
schedule. During reconstruction, ``ClassicScheduleUnavailable`` is intentionally loud and is the
only failure accepted by the exact test registry.
"""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.fork import Fork

# NOTE: no ``Knob`` objects (``TILE`` / ``REDUCE`` / ``STAGE``) may be imported here — ``Pass.load``
# scans rule modules for ``Knob`` attrs and OFF-fills any it finds bare onto every variant of the
# pass. Pin reads / knob-key spelling ride the enumerator's helpers instead; the family NAMES below
# are plain strings and a function, which that scan does not see.
from emmy.compiler.pipeline.knob import STRUCT_PREFIX
from emmy.compiler.pipeline.passes.lowering.tile._classic import schedule
from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node, ctx=None) -> Fork | list[TileOp] | TileOp:
    del match  # the scheduled op replaces the matched node in place — no graph surgery here
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped:
        raise RuleSkipped("TileOp already scheduled / nothing to map")
    # This pass DECIDES, so it requires the kernel's identity. Every row it enumerates carries the
    # ``S_*`` stamp forward, and that is what the prior ranks on, what a recorded golden matches by,
    # and what the measurement is later filed under — decide without it and the fork's pick is made
    # against an empty signature that matches every kernel and identifies none. the ``IdentityStrategy`` stamps at birth
    # ahead of this rule for exactly that reason, so an unstamped kernel here is a pass-order
    # break, not a case to handle.
    assert any(k.startswith(STRUCT_PREFIX) for k in tile.knobs), (
        f"{tile.name!r}: scheduling a kernel with no structural identity — the IdentityStrategy stamps at birth"
    )
    # A structural apply can mint a fresh kernel after ``030_cut`` has run in the current sweep.
    # Do not probe the schedule space and defer only after a refusal: the cut domain is the outer
    # enumeration, so every remaining cut must be explored before this inner enumeration starts.
    if not tile.placement_decided and cuttable_seams(tile):
        raise RuleSkipped("cut enumeration precedes classic schedule enumeration")
    options = schedule(tile, tile.name, tile.knobs, ctx)
    if not options:
        raise RuleSkipped("no enumerable schedule row for this term — leave it unmapped")
    return options if len(options) > 1 else options[0]
