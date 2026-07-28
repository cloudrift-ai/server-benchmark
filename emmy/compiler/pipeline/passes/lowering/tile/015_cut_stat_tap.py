"""Realize ``PLACE@stat=fuse`` — cut a row-statistic TAP back out of its settled host.

The inverted resting state of the retired ``025_sink_row_reduce``: loop fusion taps the norm's
statistic into its producer unconditionally (``loop/fission/010_tap_row_stat``), and THIS rule is
option-0's realizer — always legal, so the fused (tapped) state is safe to be canonical. Like
``020_cut_edge`` it is a pure REALIZER: the decision lives in the schedule (the ``PLACE@stat``
stamp the scheduler threads onto every tapped row), the graph carries the kernel shape.

The rewrite (``_tap.cut_out_taps``, shared with ``034_attach_taps``'s ineligible-host degrade):
the producer re-emits MAPPED — its picked schedule was chosen for the untapped host (the
recognition peel classified the stripped body), so dropping the decoration + the aux output slot
changes nothing about the kernel — and each sweep consumer re-welds its statistic (the tap's term
chain + a reconstructed reduce ``Loop`` in place of the ``T__sq`` ``Load``) and re-enters ``010``
un-mapped, landing on the local-stat (coop) norm form and its own ``rms_norm``-kind goldens —
today's deployment, structurally identical to the never-fissioned graph (the round-trip contract).

Ordering: this rule sorts BEFORE ``020_cut_edge`` and ``030_split_reduce``, so a fuse-stamped tap
is gone before the cone cut or a grid split restructure the host. A ``sink``-stamped host passes
through — UNLESS its row also carries ``PLACE@cone=cut``: the cone cut re-enters both halves
through ``010`` and the tap decoration cannot ride that restructure, so the co-stamped combination
degrades here to the always-legal cut-out (sink and cone-cut are both evidence-only rows; the
combination is never offered as one row, only reachable via pins)."""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.tile._tap import cut_out_taps
from emmy.compiler.pipeline.search.space import place_decision

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    tile: TileOp = root.op
    if not tile.taps:
        raise RuleSkipped("untapped kernel — nothing to cut out")
    stamped = tile.knobs.get("PLACE@stat")
    decision = stamped if stamped in ("fuse", "sink") else place_decision("stat")
    if decision == "sink" and tile.knobs.get("PLACE@cone") != "cut":
        raise RuleSkipped("PLACE@stat is sink — the tap stays (030 relocates across a split; 034 attaches)")
    return cut_out_taps(match, root)
