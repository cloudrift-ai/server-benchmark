"""Shared helpers for the ``loop/fusion`` rules.

Lives in a ``_``-prefixed module so the pass loader skips it (only
``NNN_<name>.py`` files are loaded as rules — see ``Pass.load``). Both
``010_merge_loop_ops`` and ``005_split_shared_indexmap`` import from here,
so the pure-indexmap predicate and the Write-output renamer stay defined
once. The splice/fragment plumbing for a one- or multi-consumer region
also lives here, keeping graph assembly separate from fusion policy.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import Accum, Assign, LoopOp, Write, splice_graph


def is_pure_indexmap(loop_op: LoopOp) -> bool:
    """Body contains only Loops / Loads / Writes / Selects — no compute
    (``Assign``) or ``Accum``.

    Such a kernel is an ``IndexMapOp`` lifted into Loop IR: broadcast,
    transpose, reshape, slice, cat. Its content is pure coord rewriting +
    copying. Fusing a non-indexmap producer (one with real compute)
    *into* such a consumer forces the producer's body to land inside
    the indexmap's iteration space — materializing any broadcast the
    indexmap was expressing lazily.
    """
    for s in loop_op.body.iter():
        if isinstance(s, (Assign, Accum)):
            return False
    return True


def is_castfree_indexmap(graph: Graph, producer) -> bool:
    """A pure indexmap that also PRESERVES dtype end-to-end — the only kind fusion/splitting may
    treat as free plumbing. A dtype-changing copy (a traced cast, materialized source-shaped by
    ``005_split_cast_from_indexmap``) lifts to the same Assign-free load→write body as a re-index,
    but dissolving it erases the dtype boundary the operand-staging gates key on — and a consumer
    reading the wide buffer through the narrow declared dtype decodes garbage (the fan-out cast on
    a merged sibling-linear projection: fp32 bytes read as fp16 denormals). Shared by
    ``010_merge_loop_ops`` and ``005_split_shared_indexmap``."""
    if not is_pure_indexmap(producer.op):
        return False
    out_dt = producer.output.dtype.name
    for i in producer.inputs:
        src_t = graph.buffer(i)
        if src_t is not None and src_t.dtype.name != out_dt:
            return False
    return True


def rename_write_output(op: LoopOp, *, old: str, new: str) -> LoopOp:
    """Return ``op`` with every ``Write`` whose ``output == old`` rewritten
    to ``output=new`` (recursively descends into nested Loops). Used by
    fusion to align a spliced/duplicated root's Writes with its new graph
    node id (buf names == node ids)."""

    def fn(s):
        if isinstance(s, Write) and s.output == old:
            return Write(output=new, index=s.index, value=s.value)
        return s

    return LoopOp(body=op.body.map(fn))


def closed_loop_consumer_region(graph: Graph, producer: Node) -> tuple[set[str], Node] | None:
    """Return the nearest closed producer-to-sink ``LoopOp`` region.

    One consumer is the degenerate two-node region. A fan-out must have a
    common Loop descendant; policy checks (escapes, reductions, recognition
    boundaries, and work growth) remain the merge rule's responsibility.
    """

    def reachable(start: str) -> set[str]:
        found: set[str] = set()
        pending = [start]
        while pending:
            nid = pending.pop()
            if nid in found:
                continue
            node = graph.nodes.get(nid)
            if node is None or not isinstance(node.op, LoopOp):
                continue
            found.add(nid)
            pending.extend(graph.users(nid))
        return found

    branches = sorted(graph.users(producer.id))
    if not branches or any(not isinstance(graph.nodes[nid].op, LoopOp) for nid in branches):
        return None
    common: set[str] | None = None
    for branch in branches:
        branch_reachable = reachable(branch)
        common = branch_reachable if common is None else common & branch_reachable
    if not common:
        return None
    sink_id = next((nid for nid in graph.topological_order() if nid in common), None)
    if sink_id is None:
        return None

    forward = reachable(producer.id)
    region: set[str] = set()
    pending = [sink_id]
    while pending:
        nid = pending.pop()
        if nid in region or nid not in forward:
            continue
        region.add(nid)
        for inp in graph.nodes[nid].inputs:
            parent = graph.producer(inp)
            if parent is not None:
                pending.append(parent.id)
    if producer.id not in region or not set(branches) <= region:
        return None
    interior = region - {sink_id}
    if any(nid in graph.outputs or graph.users(nid) - region for nid in interior):
        return None
    # Internal splice edges currently identify a Loop's primary buffer by its
    # node id. Leave MIMO/secondary-buffer regions on their established path.
    for nid in region:
        node = graph.nodes[nid]
        if len(node.outputs) != 1:
            return None
        for inp in node.inputs:
            parent = graph.producer(inp)
            if parent is not None and parent.id in region and inp != parent.id:
                return None
    return region, graph.nodes[sink_id]


def build_merged_region(graph: Graph, region: set[str], sink: Node) -> LoopOp | None:
    """Splice an owned DAG of ``LoopOp`` nodes into ``sink``.

    ``region`` may be the ordinary two-node case or fan out and reconverge.
    ``splice_graph`` preserves that structure: a producer statement
    demanded by several internal consumers is keyed by origin, scope and
    coordinate substitution, so equal demands share one SSA definition while
    genuinely different coordinates are emitted separately.
    """
    if sink.id not in region or any(not isinstance(graph.nodes[nid].op, LoopOp) for nid in region):
        return None

    order: list[str] = []
    seen: set[str] = set()

    def visit(nid: str) -> None:
        if nid in seen:
            return
        seen.add(nid)
        for inp in graph.nodes[nid].inputs:
            parent = graph.producer(inp)
            if parent is not None and parent.id in region:
                visit(parent.id)
        order.append(nid)

    visit(sink.id)
    if seen != region:
        return None

    sub = Graph()
    external: list[str] = []
    for nid in order:
        for inp in graph.nodes[nid].inputs:
            producer = graph.producer(inp)
            producer_id = producer.id if producer is not None else inp
            if producer_id not in region and inp not in external:
                external.append(inp)
    for ext_id in external:
        ext_t = graph.buffer(ext_id)
        shape = ext_t.shape if ext_t is not None else ()
        dtype = ext_t.dtype if ext_t is not None else "f32"
        sub.add_node(InputOp(), [], Tensor(ext_id, shape, dtype), node_id=ext_id)
    for nid in order:
        node = graph.nodes[nid]
        sub.add_node(node.op, list(node.inputs), outputs=node.outputs, node_id=nid)
    sub.outputs = [sink.id]

    result = splice_graph(sub)
    if result is None:
        return None
    merged, _ = result
    return rename_write_output(merged, old=sink.id, new=f"merged_{sink.id}")


def wrap_merge_fragment(graph: Graph, merged: LoopOp, consumer: Node) -> Graph:
    """Wrap a merged ``LoopOp`` in the single-node output fragment the rule
    returns. The fragment node's ``inputs`` list must be in the SAME order
    as ``merged``'s body Loads seed them (first-use order) so the
    interpreter — which positionally zips ``node.inputs`` against
    ``input_bufs`` — keys arrays by the right buf name."""
    merged_inputs = list(merged.inputs)
    frag = Graph()
    for inp_id in merged_inputs:
        ext_t = graph.buffer(inp_id)
        shape = ext_t.shape if ext_t is not None else ()
        dtype = ext_t.dtype if ext_t is not None else "f32"
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)
    out_id = frag.add_node(
        merged,
        merged_inputs,
        Tensor(
            consumer.output.name,
            consumer.output.shape,
            consumer.output.dtype,
        ),
        node_id=f"merged_{consumer.id}",
    )
    frag.outputs = [out_id]
    return frag
