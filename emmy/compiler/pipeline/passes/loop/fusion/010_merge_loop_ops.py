"""Greedily fuse maximal downstream ``LoopOp`` regions to a fixed point.

Separate consumers become output ports of one kernel. All roots enter the
same worklist, so shared upstream statements remain one SSA definition.

Fusion has only correctness boundaries. Tile lifting and scheduling never gate fusion.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import LoopOp, UnfusableStmt, splice_graph
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("producer", LoopOp)]


def _loop_consumer_region(graph: Graph, producer: Node) -> tuple[set[str], tuple[str, ...]] | None:
    """Return the maximal downstream ``LoopOp`` region and its live buffers."""
    region: set[str] = set()
    pending = [producer.id]
    while pending:
        nid = pending.pop()
        if nid in region:
            continue
        region.add(nid)
        pending.extend(user for user in graph.users(nid) if isinstance(graph.nodes[user].op, LoopOp))
    if len(region) < 2:
        return None
    live = _live_outputs(graph, region)
    return (region, live) if live else None


def _live_outputs(graph: Graph, region: set[str]) -> tuple[str, ...]:
    """The region's buffers read outside it or exported by the graph, in topological order."""
    graph_outputs = set(graph.outputs)
    live: list[str] = []
    for nid in graph.topological_order():
        if nid not in region:
            continue
        for buf in graph.nodes[nid].buffer_names():
            if buf in graph_outputs or graph.buffer_users(buf) - region:
                live.append(buf)
    return tuple(live)


def _build_merged_region(graph: Graph, region: set[str], live_outputs: tuple[str, ...]) -> LoopOp | None:
    """Splice one maximal region, sharing equal upstream demands across roots."""
    order = [nid for nid in graph.topological_order() if nid in region]
    sub = Graph()
    external: list[str] = []
    for nid in order:
        for inp in graph.nodes[nid].inputs:
            producer = graph.producer(inp)
            assert producer is not None
            if producer.id not in region and inp not in external:
                external.append(inp)
    for ext_id in external:
        ext_t = graph.buffer(ext_id)
        assert ext_t is not None
        sub.add_node(InputOp(), [], ext_t, node_id=ext_id)
    for nid in order:
        node = graph.nodes[nid]
        sub.add_node(node.op, list(node.inputs), outputs=node.outputs, node_id=nid)
    sub.outputs = list(live_outputs)

    result = splice_graph(sub, surface_unfusable=True)
    return result[0] if result is not None else None


def _wrap_multi_output_fragment(
    graph: Graph,
    merged: LoopOp,
    live_outputs: tuple[str, ...],
) -> tuple[Graph, dict[str, str]]:
    """Wrap one merged LoopOp and map every old live buffer to its new port."""
    owner = graph.producer(live_outputs[0])
    assert owner is not None
    node_id = f"merged_{owner.id}"
    new_buffers = (node_id, *(f"{node_id}__out{i}" for i in range(1, len(live_outputs))))
    rename = dict(zip(live_outputs, new_buffers, strict=True))

    tensors: list[Tensor] = []
    for i, (old, new) in enumerate(zip(live_outputs, new_buffers, strict=True)):
        tensor = graph.buffer(old)
        assert tensor is not None
        tensors.append(Tensor(tensor.name if i == 0 else new, tensor.shape, tensor.dtype))
    merged = merged.rename_buffers(rename)
    # Root insertion may reorder sibling loop nests. Kernel ABI order follows
    # graph liveness, not incidental body order.
    merged = replace(merged, outputs=dict(zip(new_buffers, tensors, strict=True)))
    frag = Graph()
    for inp_id in merged.inputs:
        ext_t = graph.buffer(inp_id)
        assert ext_t is not None
        frag.add_node(InputOp(), [], ext_t, node_id=inp_id)
    frag.add_node(merged, list(merged.inputs), outputs=tensors, node_id=node_id)
    frag.outputs = list(new_buffers)
    return frag, rename


def rewrite(match: Match, producer: Node) -> Graph | None:
    graph = match.graph
    if not isinstance(producer.op, LoopOp):
        raise RuleSkipped("producer is no longer a LoopOp")
    found = _loop_consumer_region(graph, producer)
    if found is None:
        raise RuleSkipped("producer has no Loop consumer region")
    region, live_outputs = found

    # The maximal region may contain a recurrence-shaped chain no budget can construct (the
    # splicer's per-stmt binding cap names the offending loop). Dropping ONLY that node and
    # retrying keeps every other merge in the region — abandoning the whole region on one doomed
    # chain is what shattered DeepSeek-V4's post block into 433 kernels where the pre-maximal
    # fusion produced 92. Each retry costs the cap's refusal (milliseconds); the chain's length
    # bounds the retries.
    merged = None
    while merged is None:
        try:
            merged = _build_merged_region(graph, region, live_outputs)
        except UnfusableStmt as doom:
            if doom.origin == producer.id or doom.origin not in region or len(region) <= 2:
                raise RuleSkipped(f"region is dominated by an unfusable chain: {doom}") from doom
            # Dropping ONLY the doomed node would leave region nodes on both sides of it — the
            # merged op would then feed it and consume it, a cycle. Its downstream closure within
            # the region leaves with it; those nodes re-enter fusion through their own matches.
            dropped = {doom.origin}
            pending = [doom.origin]
            while pending:
                for user in graph.users(pending.pop()):
                    if user in region and user not in dropped:
                        dropped.add(user)
                        pending.append(user)
            region = region - dropped
            if producer.id not in region or len(region) < 2:
                raise RuleSkipped(f"region shrank away from its producer: {doom}") from doom
            live_outputs = _live_outputs(graph, region)
            if not live_outputs:
                raise RuleSkipped(f"nothing fusable remains beside the chain: {doom}") from doom
            continue
        if merged is None:
            raise RuleSkipped("N-way Loop splicer rejected the region")

    # Liftability is a correctness boundary: a merge the tile lift cannot spell (an output whose
    # value is captured from an enclosing scope, an interior write no OutputSpec split
    # round-trips) would otherwise hard-fail the compile at 010_lift, where nothing can decline
    # it. The preflight is the exact lift the lowering runs — with the round-trip gate comparing
    # both sides under construction normalization, a pure reorder no longer fails it, so the
    # predicate matches what the pipeline truly accepts. Declining leaves the region to smaller
    # merges; every pre-fusion LoopOp lifts, so a compilable graph always remains.
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import lift_loop_op  # noqa: PLC0415

    try:
        lift_loop_op(merged)
    except ValueError as exc:
        raise RuleSkipped(f"merged region does not lift: {exc}") from exc

    fragment, output_map = _wrap_multi_output_fragment(graph, merged, live_outputs)
    match.consumed = region
    match.output = live_outputs[0] if len(live_outputs) == 1 else output_map
    return fragment
