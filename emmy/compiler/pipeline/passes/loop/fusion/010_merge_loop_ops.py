"""Merge adjacent ``LoopOp``s via graph splicing.

Every match is represented as a producer-to-sink region: one consumer is
the degenerate two-node case, while a fan-out uses its nearest owned
reconvergence. The complete region goes to the N-way splicer. Shared
internal definitions remain SSA and are emitted once per equal
scope/coordinate demand; no frontend treeification is needed. The splicer also
handles multiple consumer loads and shared external inputs uniformly
(first-seen slot assignment + splice-edge routing).

The materialization-policy guards are duplicated contraction work and aggregate compute growth:
a contraction that feeds separate statistic/value paths stays materialized, because splicing would
re-run the whole contraction. Then
``_total_work`` sums the enclosing free×reduce iteration count of every compute
leaf, and a merge is refused when that grows by more than
``_BLOWUP_FACTOR``. Structural region ownership, a real splicer rejection,
and the fence around an already-realized ``__cut_`` workspace remain
invariants rather than candidate-selection gates.

The factor was picked empirically by sweeping 2…1024 on a TinyLlama block at
sequence length 32. Values 2–16 tied at roughly 4.18 ms / 18 launches; 32–512
shifted to roughly 4.7 ms / 17 launches; 1024 unlocked an approximately 1000×
up-projection-to-down-projection nesting and took 433 ms. Eight is the middle
of the best plateau while retaining useful epilogue fusion.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import Accum, Assign, Load, Loop, LoopOp
from emmy.compiler.ir.stmt import Body
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._helpers import build_merged_region as _build_merged_region
from emmy.compiler.pipeline.passes.loop.fusion._helpers import closed_loop_consumer_region as _closed_loop_consumer_region
from emmy.compiler.pipeline.passes.loop.fusion._helpers import wrap_merge_fragment as _wrap_merge_fragment

_BLOWUP_FACTOR = 8


def _walk_leaf_costs(loop_op: LoopOp):
    """Yield ``(stmt, enclosing_free_prod × enclosing_reduce_prod)`` per body leaf."""
    reduce_names = loop_op.reduce_axis_names

    def walk(stmts: Body, free_prod: int, reduce_prod: int):
        for stmt in stmts:
            if isinstance(stmt, Loop):
                # Fusion variants share symbolic axes, so the same placeholder
                # keeps their relative cost stable.
                extent = stmt.axis.extent.as_static() if stmt.axis.extent.is_static else 128
                if stmt.axis.name in reduce_names:
                    yield from walk(stmt.body, free_prod, reduce_prod * extent)
                else:
                    yield from walk(stmt.body, free_prod * extent, reduce_prod)
            else:
                yield stmt, free_prod * reduce_prod

    yield from walk(loop_op.body, 1, 1)


def _total_work(loop_op: LoopOp) -> int:
    """Sum enclosing-loop iterations over arithmetic leaves.

    Splicing a producer body at two demand sites doubles this metric, unlike
    the previous maximum-nest approximation.
    """
    return sum(cost for stmt, cost in _walk_leaf_costs(loop_op) if isinstance(stmt, (Assign, Accum))) or 1


def _duplicated_contraction(graph: Graph, region: set[str]) -> str | None:
    """Return the first contraction duplicated across statistic/value paths.

    A contraction producer is a materialized common subexpression. The splicer shares repeated
    equal-coordinate demands, but a statistic/value fan-out duplicates the reduction. The
    aggregate work cap deliberately permits a 2× increase; a duplicated weight reduction is
    already catastrophic, so keep that producer as the shared buffer instead.
    """

    def source_buffers(producer_id: str, seen_nodes: frozenset[str] = frozenset()) -> frozenset[str]:
        if producer_id in seen_nodes:
            return frozenset()
        producer = graph.nodes.get(producer_id)
        if producer is None or not isinstance(producer.op, LoopOp):
            return frozenset({producer_id})
        next_seen = seen_nodes | {producer_id}
        return frozenset().union(*(source_buffers(input_id, next_seen) for input_id in producer.inputs))

    def contraction_sources(producer_id: str, seen_nodes: frozenset[str] = frozenset()) -> frozenset[str]:
        """Distinct load buffers that feed an internal additive reduction.

        Lifting initially separates the pointwise product and ``ReduceOp`` into
        neighboring LoopOps, so recognize the contraction across that internal
        producer edge as well as after it has been spliced into one body.
        """

        producer = graph.nodes[producer_id]
        op = producer.op
        if not isinstance(op, LoopOp):
            return frozenset()
        defs = op.analyze().defs

        def load_buffers(name: str, seen_names: frozenset[str] = frozenset()) -> frozenset[str]:
            if name in seen_names or (stmt := defs.get(name)) is None:
                return frozenset()
            if isinstance(stmt, Load):
                return source_buffers(stmt.input, seen_nodes | {producer_id})
            next_seen = seen_names | {name}
            return frozenset().union(*(load_buffers(dep, next_seen) for dep in stmt.deps()))

        sources: set[str] = set()
        for accum in (stmt for stmt in op.body.iter() if isinstance(stmt, Accum) and stmt.op.name == "add"):
            sources.update(load_buffers(accum.value))
        if sources:
            return frozenset(sources)
        next_seen = seen_nodes | {producer_id}
        return frozenset().union(*(contraction_sources(input_id, next_seen) for input_id in producer.inputs))

    region_stmts = tuple(stmt for node_id in region for stmt in graph.nodes[node_id].op.body.iter())
    online_softmax_region = (
        any(isinstance(stmt, Accum) and stmt.op.name == "maximum" for stmt in region_stmts)
        and any(isinstance(stmt, Assign) and stmt.op.name == "exp" for stmt in region_stmts)
        and any(isinstance(stmt, Accum) and stmt.op.name == "add" for stmt in region_stmts)
    )

    def branch_reaches_reduce(start: str) -> bool:
        pending = [start]
        seen: set[str] = set()
        while pending:
            node_id = pending.pop()
            if node_id in seen or node_id not in region:
                continue
            seen.add(node_id)
            op = graph.nodes[node_id].op
            if any(isinstance(stmt, Accum) for stmt in op.body.iter()):
                return True
            pending.extend(set(graph.users(node_id)) & region)
        return False

    def consumer_has_statistic_and_value(consumer: LoopOp, producer_id: str) -> bool:
        loads = tuple(load for load in consumer.loads if load.input == producer_id)
        if len(loads) < 2:
            return False
        closures = consumer.body.deps_closure
        statistic_loads = {
            load.name for load in loads if any(load.name in closures.get(accum.value, frozenset()) for accum in consumer.body.accums)
        }
        return bool(statistic_loads) and any(load.name not in statistic_loads for load in loads)

    for producer_id in region:
        producer = graph.nodes[producer_id]
        if not isinstance(producer.op, LoopOp) or len(contraction_sources(producer_id)) < 2:
            continue
        branches = set(graph.users(producer_id)) & region
        split_statistic_value = (
            len(branches) > 1
            and any(branch_reaches_reduce(branch) for branch in branches)
            and any(not branch_reaches_reduce(branch) for branch in branches)
        )
        joined_statistic_value = any(
            consumer_has_statistic_and_value(graph.nodes[consumer_id].op, producer_id) for consumer_id in region - {producer_id}
        )
        if (split_statistic_value or joined_statistic_value) and not online_softmax_region:
            return producer_id
    return None


PATTERN = [Pattern("producer", LoopOp)]
# Region discovery is dynamic. Watching immediate consumers preserves overlap
# invalidation when matches are enumerated in batches.
WATCH_CONSUMERS = True


def _merge_region(match: Match, region: set[str], sink: Node) -> Graph:
    """Splice an owned one- or multi-consumer region, capped only by compute growth."""
    graph = match.graph
    if any("__cut_" in node_id for node_id in region - {sink.id}):
        raise RuleSkipped("region crosses a decided placement cut")

    duplicated = _duplicated_contraction(graph, region)
    if duplicated is not None:
        raise RuleSkipped(f"contraction {duplicated!r} would be duplicated across statistic/value paths")

    merged = _build_merged_region(graph, region, sink)
    if merged is None:
        raise RuleSkipped("N-way Loop splicer rejected the region")
    pre_work = sum(_total_work(graph.nodes[node_id].op) for node_id in region)
    post_work = _total_work(merged)
    if post_work > _BLOWUP_FACTOR * pre_work:
        raise RuleSkipped(f"work blowup: post={post_work} > {_BLOWUP_FACTOR}× pre={pre_work}")

    match.consumed = region
    match.output = sink.id
    return _wrap_merge_fragment(graph, merged, sink)


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
