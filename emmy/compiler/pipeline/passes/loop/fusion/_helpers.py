"""Shared helpers for the ``loop/fusion`` rules.

Lives in a ``_``-prefixed module so the pass loader skips it (only
``NNN_<name>.py`` files are loaded as rules — see ``Pass.load``). Both
``010_merge_loop_ops`` and ``005_split_shared_indexmap`` import from here,
so the pure-indexmap / axis-folding predicates and the flash-boundary
protections (offer-site shape, pending contraction half, the predictive
exp-contraction scan) stay defined once. The splice/fragment plumbing of a
producer→consumer merge (``build_merged_op`` / ``wrap_merge_fragment``) also
lives here so the post-split re-fusion rule (``lowering/tile/006_merge_split_glue``)
reuses it under its own guard set instead of duplicating the assembly.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import Accum, Assign, Load, LoopOp, Write, splice_graph


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


def folds_axes(loop_op: LoopOp) -> bool:
    """Some ``Load`` index slot reads ≥2 loop vars — an axis-merging/splitting indexmap
    (a reshape fold like ``h·D + d``), as opposed to a permutation / broadcast / slice
    whose slots each read at most one var (including GQA's ``head // g``, which keeps its
    own slot). Fusing such an indexmap bakes the multi-var affine slot into the consumer's
    operand ``Load`` — the operand is no longer plainly indexed per slot, which de-certifies
    consumers whose recognition requires plain slots (the flash offer site's
    ``_extract_qk`` / ``_extract_v_layout``)."""
    for s in loop_op.body.iter():
        if isinstance(s, Load) and any(len(e.free_vars()) > 1 for e in s.index):
            return True
    return False


def through_indexmap_users(graph: Graph, node: Node, hops: int = 8) -> Node:
    """Resolve a single-user chain of pure-indexmap LoopOps DOWNSTREAM to the first compute
    consumer — the node a fusion into ``node`` is effectively feeding. A folding indexmap
    reaches the flash offer site through the matmul's unsqueeze/broadcast scaffolding
    indexmap, so the flash guards must judge the compute consumer at the chain's end, not
    the scaffolding in between."""
    n = node
    while hops and isinstance(n.op, LoopOp) and is_pure_indexmap(n.op):
        users = graph.users(n.id)
        if len(users) != 1:
            break
        nxt = graph.nodes.get(next(iter(users)))
        if nxt is None or not isinstance(nxt.op, LoopOp):
            break
        n = nxt
        hops -= 1
    return n


def pending_contraction_half(graph: Graph, consumer: Node) -> bool:
    """``consumer`` is the bare-product half of a decomposed matmul (``matmul_decompose``'s
    ``*_ew`` node: an accum-free elementwise body) whose sum-reduce partner has not merged in
    yet — its single user is a ``LoopOp`` that sum-``Accum``\\ s a ``Load`` of this buffer. The
    halves must merge FIRST: a compute-bearing producer spliced into the bare product beforehand
    rides under the partner's reduce loop on the later merge, so that merge trips the work-blowup
    guard and the contraction stays MATERIALIZED — sdpa's P@V wrote its full ``[b,h,m,n,d]``
    weight×V outer product to gmem and the flash form never certified (the Gemma finding-2
    chain: the V-norm cone fused into the P@V product first)."""
    op = consumer.op
    if any(isinstance(s, Accum) for s in op.body.iter()):
        return False  # has its own reduce — a full kernel, not a bare product half
    users = graph.users(consumer.id)
    if len(users) != 1:
        return False
    partner = graph.nodes.get(next(iter(users)))
    if partner is None or not isinstance(partner.op, LoopOp):
        return False
    reads_product = any(ld.input == consumer.id for ld in partner.op.body.loads)
    sums = any(isinstance(s, Accum) and s.op.reduce_canon == "add" for s in partner.op.body.iter())
    return reads_product and sums


def is_softmax_shaped(op) -> bool:
    """The op carries softmax's structural signature — an ``exp`` Assign AND a ``maximum``
    Accum (the rowmax). Distinguishes a softmax half from other exp-bearing kernels (a
    tanh-approx gelu has ``exp`` but no rowmax), so the flash boundary guards never
    tax the FFN's fused norm→linear/gelu edges."""
    if not isinstance(op, LoopOp):
        return False
    has_exp = any(isinstance(s, Assign) and s.op.name == "exp" for s in op.body.iter())
    has_rowmax = any(isinstance(s, Accum) and s.op.reduce_canon == "maximum" for s in op.body.iter())
    return has_exp and has_rowmax


def is_flash_offer_shaped(op) -> bool:
    """``op`` already contains the whole softmax-then-P@V composite — softmax-shaped
    (:func:`is_softmax_shaped`) AND sum-contracting. Structural on purpose (NOT
    ``is_fold_offer_site``): the recognizer's verdict flips false while an operand cone is
    fused in, which would disarm this guard exactly when it is needed — the circularity that
    let Gemma's V-norm scale-mul re-fuse after the softmax merged in."""
    return (
        isinstance(op, LoopOp)
        and is_softmax_shaped(op)
        and any(isinstance(s, Accum) and s.op.reduce_canon == "add" for s in op.body.iter())
    )


def sum_contracts_exp_producer(graph: Graph, consumer: Node, producer: Node) -> bool:
    """``consumer`` is a sum-contraction — or the bare product half of one still awaiting its
    reduce partner (:func:`pending_contraction_half`) — one of whose OTHER operands is (or is
    one producer hop away from) an exp-bearing ``LoopOp``: the P@V kernel before its softmax
    fuses in, at any point of the softmax's own assembly and the contraction's own. Once the
    softmax does, the kernel is the flash offer site; a producer fused into the V side in the
    meantime breaks the certification just as surely, so it is deferred the same way. The
    softmax-side operand itself (``producer``) is exempt — that IS the softmax fusing in. A
    gelu chain feeding a contraction stays fuseable: the gelu IS the producer there, so the
    other-input scan never sees its exp. The scan resolves THROUGH pure-indexmap wrappers
    (the model-graph P operand reaches the product half via its unsqueeze/broadcast indexmap
    until that fuses; an indexmap in the chain must not hide the exp)."""
    op = consumer.op
    sums = any(isinstance(s, Accum) and s.op.reduce_canon == "add" for s in op.body.iter())
    if not sums and not pending_contraction_half(graph, consumer):
        return False

    def resolve(nid, hops: int = 8) -> Node | None:
        """Follow a single-input chain of pure-indexmap LoopOps to its compute source."""
        n = graph.nodes.get(nid)
        while hops and n is not None and isinstance(n.op, LoopOp) and is_pure_indexmap(n.op) and len(n.inputs) == 1:
            n = graph.nodes.get(n.inputs[0])
            hops -= 1
        return n

    def has_exp(node) -> bool:
        return (
            node is not None
            and isinstance(node.op, LoopOp)
            and any(isinstance(s, Assign) and s.op.name == "exp" for s in node.op.body.iter())
        )

    for inp in consumer.inputs:
        if inp == producer.id:
            continue
        n = resolve(inp)
        if n is None or not isinstance(n.op, LoopOp):
            continue
        # The softmax may still be mid-assembly: the div piece feeding the P@V carries no exp of
        # its own until the exp piece merges in, so scan one producer hop deeper too.
        if has_exp(n) or any(has_exp(resolve(i)) for i in n.inputs if i in graph.nodes):
            return True
    return False


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
        ext = graph.nodes.get(ext_id)
        shape = ext.output.shape if ext is not None else ()
        dtype = ext.output.dtype if ext is not None else "f32"
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
        ext = graph.nodes.get(inp_id)
        shape = ext.output.shape if ext is not None else ()
        dtype = ext.output.dtype if ext is not None else "f32"
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
