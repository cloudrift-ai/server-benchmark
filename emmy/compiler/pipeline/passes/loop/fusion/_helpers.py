"""Shared helpers for the ``loop/fusion`` rules.

Lives in a ``_``-prefixed module so the pass loader skips it (only
``NNN_<name>.py`` files are loaded as rules — see ``Pass.load``). Both
``010_merge_loop_ops`` and ``005_split_shared_indexmap`` import from here,
so the pure-indexmap predicate and the Write-output renamer stay defined
once. The splice/fragment plumbing of a producer→consumer merge
(``build_merged_op`` / ``wrap_merge_fragment``) also lives here so the
post-split re-fusion rule (``lowering/tile/006_merge_split_glue``) reuses
it under its own guard set instead of duplicating the assembly.
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


def build_merged_op(graph: Graph, producer: Node, consumer: Node) -> LoopOp | None:
    """Splice ``producer`` into ``consumer`` and return the merged ``LoopOp``
    (Writes renamed to the fragment node id ``merged_<consumer.id>``), or
    ``None`` when ``splice_graph`` rejects the pattern (σ-solve failure,
    missing axis in consumer scope, splicer-internal validity issues — the
    pair stays separate).

    Builds the two-node subgraph ``splice_graph`` expects: producer,
    consumer, and their non-producer external inputs as ``InputOp`` nodes,
    so the splicer can classify each Load via the graph edges (LoopOp→LoopOp
    is a splice edge; LoopOp→InputOp is external)."""
    sub = Graph()
    for ext_id in list(producer.inputs) + list(consumer.inputs):
        if ext_id == producer.id or ext_id in sub.nodes:
            continue
        ext_t = graph.buffer(ext_id)
        shape = ext_t.shape if ext_t is not None else ()
        dtype = ext_t.dtype if ext_t is not None else "f32"
        sub.add_node(InputOp(), [], Tensor(ext_id, shape, dtype), node_id=ext_id)
    sub.add_node(producer.op, list(producer.inputs), producer.output, node_id=producer.id)
    sub.add_node(consumer.op, list(consumer.inputs), consumer.output, node_id=consumer.id)
    sub.outputs = [consumer.id]

    result = splice_graph(sub)
    if result is None:
        return None
    merged, _ = result
    return rename_write_output(merged, old=consumer.id, new=f"merged_{consumer.id}")


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
