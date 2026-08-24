"""Greedily fuse closed ``LoopOp`` regions to a fixed point.

Each match is one producer-to-consumer edge or a closed fan-out DAG. The
single rule repeats until no legal region remains.

Fusion has only correctness boundaries and the fence around an already-realized
``__cut_`` workspace. Recognition and scheduling never gate fusion.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._helpers import build_merged_region as _build_merged_region
from emmy.compiler.pipeline.passes.loop.fusion._helpers import closed_loop_consumer_region as _closed_loop_consumer_region
from emmy.compiler.pipeline.passes.loop.fusion._helpers import wrap_merge_fragment as _wrap_merge_fragment

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
    if any("__cut_" in node_id for node_id in region - {sink.id}):
        raise RuleSkipped("region crosses a decided placement cut")

    merged = _build_merged_region(graph, region, sink)
    if merged is None:
        raise RuleSkipped("N-way Loop splicer rejected the region")

    match.consumed = region
    match.output = sink.id
    return _wrap_merge_fragment(graph, merged, sink)
