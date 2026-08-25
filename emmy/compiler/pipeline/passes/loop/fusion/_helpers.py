"""Shared graph assembly for the ``loop/fusion`` rules.

Lives in a ``_``-prefixed module so the pass loader skips it. Maximal fusion
uses the region/splice helpers; the output-reshape fold shares the small
single-output fragment wrapper and pure-indexmap predicate.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import Accum, Assign, LoopOp, Write, splice_graph


def is_pure_indexmap(loop_op: LoopOp) -> bool:
    """Whether the body is layout-only: Loops / Loads / Writes / Selects."""
    for s in loop_op.body.iter():
        if isinstance(s, (Assign, Accum)):
            return False
    return True


def loop_consumer_region(graph: Graph, producer: Node) -> tuple[set[str], tuple[str, ...]] | None:
    """Return the maximal downstream ``LoopOp`` region and its live buffers.

    Separate fan-out branches stay in one region. A buffer is live when it is
    a graph output or has a user outside the region. A realized placement
    workspace (``__cut_``) may absorb its producers but is never crossed.
    """
    if not isinstance(producer.op, LoopOp) or "__cut_" in producer.id:
        return None

    region: set[str] = set()
    pending = [producer.id]
    while pending:
        nid = pending.pop()
        if nid in region:
            continue
        node = graph.nodes.get(nid)
        if node is None or not isinstance(node.op, LoopOp):
            continue
        region.add(nid)
        if "__cut_" in nid:
            continue
        for user in graph.users(nid):
            if isinstance(graph.nodes[user].op, LoopOp):
                pending.append(user)
    if len(region) < 2:
        return None

    graph_outputs = set(graph.outputs)
    live: list[str] = []
    for nid in graph.topological_order():
        if nid not in region:
            continue
        for buf in graph.nodes[nid].buffer_names():
            if buf in graph_outputs or graph.buffer_users(buf) - region:
                live.append(buf)
    return (region, tuple(live)) if live else None


def build_merged_region(graph: Graph, region: set[str], live_outputs: tuple[str, ...]) -> LoopOp | None:
    """Splice one maximal region, sharing equal upstream demands across roots."""
    if any(not isinstance(graph.nodes[nid].op, LoopOp) for nid in region):
        return None
    order = [nid for nid in graph.topological_order() if nid in region]
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
    sub.outputs = list(live_outputs)

    result = splice_graph(sub)
    if result is None:
        return None
    merged, _ = result
    return merged


def wrap_multi_output_fragment(
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

    def retarget(stmt):
        if isinstance(stmt, Write) and stmt.output in rename:
            return Write(
                output=rename[stmt.output],
                index=stmt.index,
                values=stmt.values,
                value_dtype=stmt.value_dtype,
                atomic=stmt.atomic,
                swizzle=stmt.swizzle,
            )
        return stmt

    tensors: list[Tensor] = []
    for i, (old, new) in enumerate(zip(live_outputs, new_buffers, strict=True)):
        tensor = graph.buffer(old)
        assert tensor is not None
        tensors.append(Tensor(tensor.name if i == 0 else new, tensor.shape, tensor.dtype))
    merged = LoopOp(body=merged.body.map(retarget), name=merged.name)
    # Root insertion may reorder sibling loop nests. Kernel ABI order follows
    # graph liveness, not incidental body order.
    merged.outputs = dict(zip(new_buffers, tensors, strict=True))
    frag = Graph()
    for inp_id in merged.inputs:
        ext_t = graph.buffer(inp_id)
        shape = ext_t.shape if ext_t is not None else ()
        dtype = ext_t.dtype if ext_t is not None else "f32"
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)
    frag.add_node(merged, list(merged.inputs), outputs=tensors, node_id=node_id)
    frag.outputs = list(new_buffers)
    return frag, rename


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
