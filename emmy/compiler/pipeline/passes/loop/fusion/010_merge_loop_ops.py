"""Merge adjacent ``LoopOp``s via graph splicing.

Every match is represented as a producer-to-sink region: one consumer is
the degenerate two-node case, while a fan-out uses its nearest owned
reconvergence. The complete region goes to the N-way splicer. Shared
internal definitions remain SSA and are emitted once per equal
scope/coordinate demand; no frontend treeification is needed. The splicer also
handles multiple consumer loads and shared external inputs uniformly
(first-seen slot assignment + splice-edge routing).

Fusion is greedy-maximal and algebra-only: every legal merge is taken. Which form of a region is
worth deploying — fused, or split into smaller kernels — is decided by the deploy evidence
hierarchy (tuned goldens, measurements, the prior); this rule never weighs shapes, hardware, or
downstream pattern knowledge. Its refusals are semantic — structural region ownership, a real
splicer rejection, the fence around an already-realized ``__cut_`` workspace (which keeps a
placement decision from being re-fused), and two readable-seam refusals judged on the MERGED
form: a merge must not nest a reduce ``Loop`` inside another reduce ``Loop``, and must not
entangle a multi-statistic compound (the online-softmax pair) beyond its readable tails — the
flat same-extent normalize sweep, or a free sweep of flat same-extent additive folds (the
expectation channels of a fused softmax·V region, which the online-softmax pairing joins into
one streaming loop) — and must not chain a per-step statistic into a fold inside a free sweep (a
reduce another reduce in the same sweep reads: attention's k-norm replayed per query row ahead
of ``Q·Kᵀ``) — other shapes fall to the raw-loop escape downstream (no schedule tier, no
``PLACE`` seam), so evidence could never price the split back — plus one boundedness bound:
``_total_work`` sums the
enclosing free×reduce iteration count of every compute leaf, and a merge that grows it by more
than ``_BLOWUP_FACTOR`` is refused. That bound is what keeps the downstream problem finite, not a
performance preference: unbounded splicing folds a whole transformer layer into ONE loop nest
(measured: 57 nested loops with the row statistic replayed inside a ~10¹³-iteration nest) — a
form no schedule can run and recognition cannot certify. Within the bound, every fused/split
tradeoff belongs to measured evidence.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._helpers import closed_loop_consumer_region as _closed_loop_consumer_region
from emmy.compiler.pipeline.passes.loop.fusion._merge import merge_region as _merge_region

PATTERN = [Pattern("producer", LoopOp)]
# Region discovery is dynamic. Watching immediate consumers preserves overlap
# invalidation when matches are enumerated in batches.
WATCH_CONSUMERS = True


def rewrite(match: Match, producer: Node) -> Graph | None:
    graph = match.graph
    if not isinstance(producer.op, LoopOp):
        raise RuleSkipped("producer is no longer a LoopOp")
    users = graph.users(producer.id)
    found = _closed_loop_consumer_region(graph, producer)
    if found is None:
        if len(users) > 1:
            raise RuleSkipped("producer fan-out has no closed reconvergent Loop region")
        raise RuleSkipped("producer has no Loop consumer region")
    region, sink = found
    return _merge_region(match, region, sink)
