"""Drain every NARROWING merge before general fusion offers a widening one.

Same region discovery, same splicer, same refusals and same work bound as ``loop/fusion`` — this
rule differs from it in exactly one line, the predicate below, and both call
:func:`_merge.merge_region` so the merge semantics have one home.

A merge is DIRECTIONAL: it makes the SINK the region's output, so whatever width the sink holds
must now be written. Ordering on that fact is what this pass is for. A merge whose sink is no
wider than its producer can only shrink what gets written — a product folding into its own reduce,
an index map dissolving into its consumer's index. Taking all of those first means each contraction
has CLOSED before anything is offered a chance to splice into its open product.

Without that ordering the outcome depends on which match the enumeration reached first, and one
order is catastrophic: splice a compute producer into a still-open product and the product becomes
the region's output, so the free-axes × contraction-axis outer product is written to gmem — and the
fold that would have collapsed it can then only arrive as a reduce nested in a reduce, which
``loop/fusion`` refuses as an unreadable seam. Measured on a 1-layer Qwen3-0.6B trunk at seq 512:
a 6.006 GiB scratch slab, against 0.026 GiB with this pass in front.

**This pass refuses nothing.** A widening merge is DEFERRED, not suppressed: ``loop/fusion`` offers
every one of them afterwards, and what happens then is the existing refusals' decision, not a new
one. That is why the fix is an ordering and not a gate — no legal form leaves the enumeration.

It has to be a PASS rather than another rule in ``loop/fusion``. The cursor advances rule-by-rule
within a pass and re-enumerates, so two rules' batches INTERLEAVE — measured: as a `009_` rule the
compute producer still reached the open product first, and the trunk stayed at 6.006 GiB. A pass is
left only once it is quiescent, so this reaches fixpoint globally before the next pass starts.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._helpers import closed_loop_consumer_region as _closed_loop_consumer_region
from emmy.compiler.pipeline.passes.loop.fusion._merge import merge_region as _merge_region

PATTERN = [Pattern("producer", LoopOp)]
# Region discovery is dynamic, exactly as in ``loop/fusion``: watching immediate consumers
# preserves overlap invalidation when matches are enumerated in batches.
WATCH_CONSUMERS = True

#: Stand-in extent for a symbolic axis. Both sides of the comparison below share their symbolic
#: axes, so holding every one at the same value keeps the comparison relative-stable.
_SYMBOLIC_EXTENT = 128


def _output_numel(node: Node) -> int:
    """Element count of ``node``'s output buffer (symbolic extents at ``_SYMBOLIC_EXTENT``)."""
    numel = 1
    for dim in node.output.shape:
        numel *= dim.as_static() if dim.is_static else _SYMBOLIC_EXTENT
    return numel


def rewrite(match: Match, producer: Node) -> Graph | None:
    graph = match.graph
    if not isinstance(producer.op, LoopOp):
        raise RuleSkipped("producer is no longer a LoopOp")
    found = _closed_loop_consumer_region(graph, producer)
    if found is None:
        raise RuleSkipped("producer has no closed Loop consumer region")
    region, sink = found
    if _output_numel(sink) > _output_numel(producer):
        raise RuleSkipped("widening merge — deferred to loop/fusion, after every narrowing one has run")
    return _merge_region(match, region, sink)
