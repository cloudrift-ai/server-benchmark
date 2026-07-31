"""Merge two adjacent ``LoopOp``s via graph splicing.

Matches a ``LoopOp`` whose sole consumer is another ``LoopOp`` and fuses
them by handing a two-node subgraph to ``splice_graph``. The splicer
handles multiple consumer Loads of the producer and shared external
inputs uniformly (first-seen slot assignment + splice-edge routing).
Splicing refuses patterns it doesn't handle yet (non-trivial σ writer
forms, etc.); those boundaries stay as separate kernels.

Blowup guards: two metrics, both summed over body leaves (max-per-leaf
wasn't enough — a fusion that introduces a second large leaf alongside
an existing one looks free to a max, but the actual runtime work
doubles).

- ``_total_work``: sum over compute leaves (``Assign`` / ``Accum``) of
  ``enclosing_free × enclosing_reduce`` — proxy for arithmetic.
- ``_total_expensive_work``: the same count restricted to transcendental
  ``Assign`` operations. These cannot hide behind a large contraction's
  cheap-FMA count: duplicating GELU's tanh across the output-column loop
  is much slower than materializing its input once.
- ``_total_reads``: sum over ``Load`` stmts of the same product — proxy
  for memory traffic. Global reads dominate cost on small-M matmuls
  where arithmetic is bandwidth-bound, so a fusion that grows reads
  without growing work is still a regression.

A fusion is refused if **either** metric grows by more than
``_BLOWUP_FACTOR`` over the producer+consumer sum. In addition, a
``multi-load-of-reducer`` guard refuses fusions where the consumer reads
the producer from multiple ``Load`` stmts **and** the producer contains
any reduce axis — inlining a reduce twice recomputes it, which
``_total_*`` catches *in ratio* but only when producer is big enough
relative to consumer; the guard catches it structurally.

Factor picked empirically — swept 2…1024 on TinyLlama block (seq=32):
2–16 ties at ~4.18ms/18 launches (best), 32–512 shifts to ~4.7ms/17
launches (one harmful silu→down_proj fusion lands), 1024 unlocks the
up_proj→down_proj nesting (~1000×) and the block takes 433ms. 8 is the
middle of the best plateau and still lets the epilogue-fusion cases
through.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import Accum, Assign, Load, Loop, LoopOp
from emmy.compiler.ir.stmt import Body, Select
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._helpers import build_merged_op as _build_merged_op
from emmy.compiler.pipeline.passes.loop.fusion._helpers import is_castfree_indexmap as _is_castfree_indexmap_shared
from emmy.compiler.pipeline.passes.loop.fusion._helpers import is_pure_indexmap as _is_pure_indexmap
from emmy.compiler.pipeline.passes.loop.fusion._helpers import wrap_merge_fragment as _wrap_merge_fragment

_BLOWUP_FACTOR = 8
# Expensive scalar functions are costed separately from the aggregate FMA
# proxy. A producer-side tanh/exp executed once per (M, K) value must not
# move under a contraction's N loop merely because the contraction's own
# multiply/accumulate count dominates ``_total_work``.
_EXPENSIVE_OPS = frozenset(
    {
        "cos",
        "erf",
        "exp",
        "exp_fast",
        "log",
        "log10",
        "log2",
        "rsqrt",
        "sin",
        "sqrt",
        "tan",
        "tanh",
    }
)


def _walk_leaf_costs(loop_op: LoopOp):
    """Yield ``(stmt, enclosing_free_prod × enclosing_reduce_prod)`` per body leaf.

    Leaf = any non-``Loop`` stmt. Products accumulate along the actually
    enclosing ``Loop`` chain; sibling free axes don't pile onto a reduce
    leaf's cost (and vice versa), matching the original max-nest semantics.
    """
    reduce_names = loop_op.reduce_axis_names

    def walk(stmts: Body, free_prod: int, reduce_prod: int):
        for s in stmts:
            if isinstance(s, Loop):
                # Heuristic: substitute a placeholder for symbolic extents — fusion
                # ranking compares variants of the same graph so the same symbolic
                # axis appears identically on both sides.
                extent = s.axis.extent.as_static() if s.axis.extent.is_static else 128
                if s.axis.name in reduce_names:
                    yield from walk(s.body, free_prod, reduce_prod * extent)
                else:
                    yield from walk(s.body, free_prod * extent, reduce_prod)
            else:
                yield s, free_prod * reduce_prod

    yield from walk(loop_op.body, 1, 1)


def _total_work(loop_op: LoopOp) -> int:
    """Sum of enclosing-loop iterations over compute leaves (Assign + Accum).

    Counts how many times each arithmetic stmt executes across the full
    iteration space. A fusion that splices a producer's body in twice
    doubles this number — the old max-nest metric couldn't see that.
    """
    return sum(cost for stmt, cost in _walk_leaf_costs(loop_op) if isinstance(stmt, (Assign, Accum))) or 1


def _total_expensive_work(loop_op: LoopOp) -> int:
    """Count executions of transcendental elementwise operations.

    Unlike :func:`_total_work`, zero is meaningful here: most kernels have
    no transcendental work at all.
    """
    return sum(cost for stmt, cost in _walk_leaf_costs(loop_op) if isinstance(stmt, Assign) and stmt.op.name in _EXPENSIVE_OPS)


def _single_source_activation(graph: Graph, producer: Node) -> bool:
    """Whether ``producer``'s non-scalar data dependencies share one root.

    A decomposed unary activation can retain both an intermediate and its
    original input as external buffers (GELU reads ``scaled(x)`` and ``x``),
    so counting direct inputs is insufficient. Collapse inputs reachable
    from another input to their independent roots. Multi-source gated
    activations (GeGLU/SwiGLU) deliberately stay outside the generic
    materialization policy: changing those serving kernels requires
    measured placement evidence and matching GPU goldens.
    """

    def non_scalar(node_id: str) -> bool:
        node = graph.nodes.get(node_id)
        if node is None:
            return False
        numel = 1
        for extent in node.output.shape:
            if not extent.is_static:
                return True
            numel *= extent.as_static()
        return numel > 1

    data_inputs = {node_id for node_id in producer.inputs if non_scalar(node_id)}

    def has_data_input_ancestor(node_id: str) -> bool:
        pending = list(graph.nodes[node_id].inputs)
        seen: set[str] = set()
        while pending:
            ancestor = pending.pop()
            if ancestor in seen:
                continue
            seen.add(ancestor)
            if ancestor in data_inputs:
                return True
            node = graph.nodes.get(ancestor)
            if node is not None:
                pending.extend(node.inputs)
        return False

    roots = {node_id for node_id in data_inputs if not has_data_input_ancestor(node_id)}
    return len(roots) == 1


def _has_peer_activation_input(graph: Graph, producer: Node, consumer: Node) -> bool:
    """Whether the consumer combines the activation with a peer value.

    A merged sibling-linear may expose gate and up as two slices of one
    packed buffer, so the peer can have a different declared shape while
    still being an ancestor shared by the activation and its consumer.
    """
    shape = producer.output.shape
    ancestors: set[str] = set()
    pending = list(producer.inputs)
    while pending:
        ancestor = pending.pop()
        if ancestor in ancestors:
            continue
        ancestors.add(ancestor)
        node = graph.nodes.get(ancestor)
        if node is not None:
            pending.extend(node.inputs)

    return any(
        node_id != producer.id and (node := graph.nodes.get(node_id)) is not None and (node.output.shape == shape or node_id in ancestors)
        for node_id in consumer.inputs
    )


def _total_reads(loop_op: LoopOp) -> int:
    """Sum of enclosing-loop iterations over ``Load`` stmts.

    Proxy for global-memory traffic (no cache modeling — all Loads
    count). A fusion that multiplies reads by a seq factor shows up as
    a ratio blowup here even when arithmetic stays flat.
    """
    return sum(cost for stmt, cost in _walk_leaf_costs(loop_op) if isinstance(stmt, Load)) or 1


# Producers with more than a handful of ops per output element are "reduce-heavy":
# their output at position p requires non-trivial compute (typically a reduce whose
# body depends on p). Duplicating such a producer's body (multi-load fusion) then
# re-executes the reduce per load site — what softmax-over-matmul (scaled_qk) does
# at scale. Pure-elementwise chains sit at ~1–3 ops/output; softmax's
# (max + exp) sits at ~3; a matmul sits at reduce_extent (head_dim=64). Threshold
# 4 separates the two regimes cleanly.
_REDUCE_HEAVY_WORK_PER_OUTPUT = 4


def _reduce_heavy(op: LoopOp) -> bool:
    # A softmax assembly (rowmax-bearing body) discounts its ``add`` Assigns: they are the score's
    # mask applications (causal / banded coord Selects, the explicit additive bias) — one
    # predicate-add per mask, not real duplicated compute. Without the discount a multi-mask
    # score (a stamped sliding-window SDPA carries up to three) pushes the cheap (max + exp)
    # reducer past the threshold and the softmax cone never assembles onto its P@V offer site.
    work = _total_work(op)
    if _has_rowmax(op):
        work -= sum(cost for stmt, cost in _walk_leaf_costs(op) if isinstance(stmt, Assign) and stmt.op.name == "add")
    return work > _REDUCE_HEAVY_WORK_PER_OUTPUT * _output_numel(op)


def _output_numel(loop_op: LoopOp) -> int:
    reduce_names = loop_op.reduce_axis_names
    n = 1
    for a in loop_op.axes:
        if a.name not in reduce_names:
            n *= a.extent.as_static() if a.extent.is_static else 128
    return n


def _count_loads_from(consumer_op: LoopOp, producer_buf: str) -> int:
    """Number of ``Load`` stmts in the consumer body reading producer's output buffer."""
    return sum(1 for ld in consumer_op.body.loads if ld.input == producer_buf)


def _pending_contraction_half(graph: Graph, consumer: Node) -> bool:
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


def _is_castfree_indexmap(graph: Graph, producer: Node) -> bool:
    """A pure indexmap that also PRESERVES dtype end-to-end — the only kind the
    materialization guards below may exempt as plumbing. A dtype-changing copy (a traced
    cast — ``005_split_cast_from_indexmap`` materializes them as source-shaped copy nodes)
    lifts to the same Assign-free load→write body as a re-index, but splicing it into a
    contraction / flash offer site erases the dtype boundary the operand-staging gates key
    on (``kv_stage_ok`` reads the operand BUFFER dtype — the gemma V-norm's f32→f16 cast
    fused into P@V left the flash V reading the f32 buffer, gmem-direct forever). So a
    dtype-changing copy takes the guards like any compute-bearing producer and stays
    materialized at the traced cast boundary. Shared logic in ``_helpers`` — the fan-out
    splitter needs the same gate."""
    return _is_castfree_indexmap_shared(graph, producer)


def _is_softmax_shaped(op) -> bool:
    """The op carries softmax's structural signature — an ``exp`` Assign AND a ``maximum``
    Accum (the rowmax). Distinguishes a softmax half from other exp-bearing kernels (a
    tanh-approx gelu has ``exp`` but no rowmax), so the flash boundary guards below never
    tax the FFN's fused norm→linear/gelu edges."""
    if not isinstance(op, LoopOp):
        return False
    has_exp = any(isinstance(s, Assign) and s.op.name == "exp" for s in op.body.iter())
    has_rowmax = any(isinstance(s, Accum) and s.op.reduce_canon == "maximum" for s in op.body.iter())
    return has_exp and has_rowmax


def _is_flash_offer_shaped(op) -> bool:
    """``op`` already contains the whole softmax-then-P@V composite — softmax-shaped
    (:func:`_is_softmax_shaped`) AND sum-contracting. Structural on purpose (NOT
    ``is_fold_offer_site``): the recognizer's verdict flips false while an operand cone is
    fused in, which would disarm this guard exactly when it is needed — the circularity that
    let Gemma's V-norm scale-mul re-fuse after the softmax merged in."""
    return (
        isinstance(op, LoopOp)
        and _is_softmax_shaped(op)
        and any(isinstance(s, Accum) and s.op.reduce_canon == "add" for s in op.body.iter())
    )


def _sum_contracts_exp_producer(graph: Graph, consumer: Node, producer: Node) -> bool:
    """``consumer`` is a sum-contraction one of whose OTHER operands is (or is one producer hop
    away from) an exp-bearing ``LoopOp`` — the P@V kernel before its softmax fuses in, at any
    point of the softmax's own assembly. Once it does, the kernel is the flash offer site; a
    compute producer fused into the V side in the meantime breaks the certification just as
    surely, so it is deferred the same way. The softmax-side operand itself (``producer``) is
    exempt — that IS the softmax fusing in. A gelu chain feeding a contraction stays fuseable:
    the gelu IS the producer there, so the other-input scan never sees its exp."""
    op = consumer.op
    if not any(isinstance(s, Accum) and s.op.reduce_canon == "add" for s in op.body.iter()):
        return False

    def has_exp(node) -> bool:
        return isinstance(node.op, LoopOp) and any(isinstance(s, Assign) and s.op.name == "exp" for s in node.op.body.iter())

    for inp in consumer.inputs:
        if inp == producer.id:
            continue
        n = graph.producer(inp)
        if n is None or not isinstance(n.op, LoopOp):
            continue
        # The softmax may still be mid-assembly: the div piece feeding the P@V carries no exp of
        # its own until the exp piece merges in, so scan one producer hop deeper too.
        if has_exp(n) or any(has_exp(p) for i in n.inputs if (p := graph.producer(i)) is not None):
            return True
    return False


def _is_reduce_partner_merge(producer: Node, consumer: Node) -> bool:
    """This merge IS the contraction-halves merge: some add-``Accum`` in ``consumer`` sums the
    producer's buffer DIRECTLY — its accumulated value is the name a ``Load`` of ``producer``
    binds, with nothing in between (``acc += load(product)``, ``matmul_decompose``'s reduce
    partner, even after scale/mask epilogues fused into it). An operand cone feeding a merged
    contraction never matches: its load is multiplied before the accumulate, so the ``Accum``
    value is an ``Assign``'s name, not the Load's."""
    op = consumer.op
    product_names = {nm for ld in op.body.loads if ld.input == producer.id for nm in ld.names}
    if not product_names:
        return False
    return any(isinstance(s, Accum) and s.op.reduce_canon == "add" and s.value in product_names for s in op.body.iter())


def _has_rowmax(op) -> bool:
    """A ``maximum`` Accum — softmax's rowmax signature, present from the decomposition's very
    first kernels (before the exp piece merges in), and absent from every non-softmax consumer
    a matmul feeds in practice (gelu has ``exp`` but no rowmax)."""
    return isinstance(op, LoopOp) and any(isinstance(s, Accum) and s.op.reduce_canon == "maximum" for s in op.body.iter())


def _mask_epilogue(op) -> bool:
    """A pure score-mask application: accum-free, every compute ``Assign`` an ``add``, and any
    ``Select`` a COORDINATE mask (both branch values scalar-constant Loads — the causal / banded
    keep-vs-``-inf`` pick; the additive-bias add has no Select at all). This is the piece of the
    attention score that belongs WITH the softmax consumer — flash classification reads the mask
    chain off the rowmax feed, and re-synthesis drops anything left behind on the score producer.
    RoPE / norm cones never match: their Selects read tensor data, their Assigns carry mul/trig."""
    if not isinstance(op, LoopOp) or any(isinstance(s, Accum) for s in op.body.iter()):
        return False
    stmts = list(op.body.iter())
    assigns = [s for s in stmts if isinstance(s, Assign)]
    if not assigns or not all(s.op.name == "add" for s in assigns):
        return False
    scalar_names = {nm for s in stmts if isinstance(s, Load) and not any(e.free_vars() for e in s.index) for nm in s.names}
    return all(all(br.value in scalar_names for br in s.branches) for s in stmts if isinstance(s, Select))


def _feeds_softmax(graph: Graph, consumer: Node) -> bool:
    """``consumer``'s output reaches a softmax rowmax within two user hops — the attention
    SCORE producer (QK^T, possibly through the additive-mask elementwise between it and the
    softmax), at any point of the softmax's own assembly. Flash recovers Q/K from the score
    producer as plain ``Load``\\ s (``_extract_qk``), so a compute producer (RoPE / q-k-norm)
    fused into it de-certifies the whole unit; those cones stay materialized instead (the
    B-track — computing them inside the flash nest — replaces this materialization later)."""
    for uid in graph.users(consumer.id):
        u = graph.nodes.get(uid)
        if u is None or not isinstance(u.op, LoopOp):
            continue
        if _has_rowmax(u.op):
            return True
        for uid2 in graph.users(uid):
            u2 = graph.nodes.get(uid2)
            if u2 is not None and _has_rowmax(getattr(u2, "op", None)):
                return True
    return False


PATTERN = [
    Pattern("producer", LoopOp),
    Pattern("consumer", LoopOp),
]


def rewrite(match: Match, producer: Node, consumer: Node) -> Graph | None:
    graph = match.graph
    if not isinstance(producer.op, LoopOp) or not isinstance(consumer.op, LoopOp):
        raise RuleSkipped("producer or consumer is no longer a LoopOp")
    if producer.id not in consumer.inputs:
        raise RuleSkipped(f"producer {producer.id!r} is not an input of consumer {consumer.id!r}")
    # A producer that is itself a GRAPH OUTPUT must stay materialized: the splice consumes the
    # node wholesale, so fusing it silently DROPS that output from the compiled graph (the
    # escaping-residual ``(rms_norm(x + r), x + r)`` shape returned only the norm). Same guard
    # ``005_split_shared_indexmap`` applies.
    if producer.id in graph.outputs:
        raise RuleSkipped(f"producer {producer.id!r} is a graph output — it must stay materialized")
    # A decided placement-cut workspace is not fusion's to undo: the cut realizer names its seam
    # buffers ``…__cut_…``, and a sliced re-compile (tune mode) re-enters loop fusion with the
    # pieces as ordinary producer→consumer pairs — a stat-BEARING piece is braked incidentally
    # (reduce-heavy), but a stat-free value piece is pure pointwise with one consumer, this
    # rule's core move, and re-inlining it silently reverts the decided placement (then the
    # re-cut collides on the existing workspace node).
    if "__cut_" in producer.id:
        raise RuleSkipped(f"{producer.id!r} is a decided placement-cut workspace — the cut is not fusion's to undo")

    # Multi-load-of-reduce-heavy-producer guard: if the consumer references
    # the producer's output via more than one Load stmt AND the producer does
    # more than a few ops per output element (i.e., has a real reduce whose
    # body can't be shared across the consumer's load positions), fusion
    # would duplicate the reduce per load site. Catches SDPA's softmax over
    # matmul — scaled_qk (head_dim reduce) feeds both row-max and exp, and
    # fusing would re-run the matmul head_dim reduce at every output element.
    # Pure-elementwise producers and "cheap" reducers like softmax's
    # (max + exp) — where the reduce collapses to a row scalar the splicer
    # can hoist — stay fuseable.
    if _reduce_heavy(producer.op) and _count_loads_from(consumer.op, producer.id) > 1:
        raise RuleSkipped("reduce-heavy producer feeds consumer through >1 Load — fusion would duplicate the reduce")

    # ContractionView-half protection: only the product's own indexmap plumbing (unsqueeze /
    # broadcast scaffolding) may fuse into a bare matmul product before its sum-reduce partner
    # does — see :func:`_pending_contraction_half`. The blocked producer isn't lost: once the
    # halves merge, it retries against the full contraction kernel on a later fixpoint sweep.
    # A dtype-changing copy (a traced cast, split source-shaped by ``005_split_cast_from_indexmap``)
    # never merges INTO pure-indexmap plumbing: the combined kernel would carry the cast on the
    # plumbing's output — an unsqueeze/broadcast-shaped buffer — instead of the source-shaped
    # boundary. The plumbing stays lazy and folds into the downstream real consumer, which then
    # reads the cast's materialized (narrow-dtype) buffer through the composed index. The reverse
    # direction (castfree plumbing merging into the cast) keeps the cast's own shape and stays legal.
    if _is_pure_indexmap(producer.op) and not _is_castfree_indexmap(graph, producer) and _is_pure_indexmap(consumer.op):
        raise RuleSkipped("cast copy feeding indexmap plumbing — the cast stays at the traced dtype boundary")

    if not _is_castfree_indexmap(graph, producer) and _pending_contraction_half(graph, consumer):
        raise RuleSkipped("consumer is an unfused contraction half — the product merges with its reduce first")

    # Flash-consumer protection: a softmax-then-P@V kernel (the flash recognizer's offer site) is
    # owned by ``try_flash`` — a compute-bearing producer fused into it lands extra Loads in the
    # P@V accum loop and the V-operand extraction fails (``others == 1``), so the flash form
    # silently never certifies (a computed V — Gemma's V-norm — was the finding-2 chain's second
    # link). The producer stays materialized; flash streams it as the plain-buffer V. The test is
    # deliberately PREDICTIVE as well (``_sum_contracts_exp_producer``), and the formed-composite test is structural
    # (``_is_flash_offer_shaped``) rather than the recognizer's verdict: fusion order is free, so
    # the V-side producer can arrive while the kernel is still a bare P@V contraction whose
    # softmax half hasn't fused in yet — the offer site only forms afterwards, too late to
    # protect. The score producer's mirror-image deferral lives in tile recognition
    # (``is_flash_score_producer``).
    if not _is_castfree_indexmap(graph, producer) and (
        _is_flash_offer_shaped(consumer.op) or _sum_contracts_exp_producer(graph, consumer, producer)
    ):
        raise RuleSkipped("consumer is a (future) flash softmax-then-P@V offer site — its operands stay materialized")

    # Flash score-producer protection, the Q/K mirror of the above: the score matmul feeding a
    # softmax must keep its Q/K as plain ``Load``\\ s for ``_extract_qk`` — an inlined RoPE /
    # q-k-norm cone de-certifies the flash unit ("score producer's Q/K are not plain loads").
    # Only accum-free elementwise producers defer (the operand-cone shape: RoPE's mul/trig
    # chain, the norm's final scale-mul) — reduce-bearing producers are the score's own
    # dataflow assembling itself (the QK contraction fusing into its scale epilogue) and
    # pass through, as does the bare-product half merging with its reduce partner. A MASK
    # epilogue (the causal / banded / additive-bias adds) is exempt: it belongs WITH the
    # softmax consumer — flash classification reads the mask chain off the rowmax feed, so a
    # chain of mask adds must stay free to assemble onto the softmax past one another.
    if (
        not _is_castfree_indexmap(graph, producer)
        and not any(isinstance(s, Accum) for s in producer.op.body.iter())
        and not _is_reduce_partner_merge(producer, consumer)
        and not _mask_epilogue(producer.op)
        and _feeds_softmax(graph, consumer)
    ):
        raise RuleSkipped("consumer feeds a softmax (attention score producer) — Q/K cones stay materialized")

    # …and the contraction must not chase the masks the other way: the QK reduce fusing INTO a
    # mask epilogue strands the mask on the score-producer side of the flash boundary, where
    # re-synthesis silently DROPS it (the fused kernel then attends outside the mask). Pair
    # ordering decided this before the banded mask added a second mask node; now it is explicit.
    if (
        not _is_castfree_indexmap(graph, producer)
        and any(isinstance(s, Accum) for s in producer.op.body.iter())
        and _mask_epilogue(consumer.op)
        and _feeds_softmax(graph, consumer)
    ):
        raise RuleSkipped("score contraction stays clear of softmax mask epilogues — the masks ride the softmax consumer")

    # ``build_merged_op`` hands a two-node subgraph to ``splice_graph`` and
    # returns None on any unsupported pattern: σ-solve failure (writer/reader
    # index forms incompatible), missing axis in consumer scope, or
    # splicer-internal validity issues. The rule treats them uniformly —
    # the producer/consumer pair stays separate.
    merged = _build_merged_op(graph, producer, consumer)
    if merged is None:
        raise RuleSkipped(f"splice_graph rejected pattern: {producer.id!r} -> {consumer.id!r}")

    pre_work = _total_work(producer.op) + _total_work(consumer.op)
    pre_expensive_work = _total_expensive_work(producer.op) + _total_expensive_work(consumer.op)
    pre_reads = _total_reads(producer.op) + _total_reads(consumer.op)
    post_work = _total_work(merged)
    post_expensive_work = _total_expensive_work(merged)
    post_reads = _total_reads(merged)
    if post_work > _BLOWUP_FACTOR * pre_work:
        raise RuleSkipped(f"work blowup: post={post_work} > {_BLOWUP_FACTOR}× pre={pre_work}")
    # Flash attention deliberately streams exp(score) into P@V: no full
    # probability matrix is materialized, and the tile recognizer owns that
    # composite. Do not apply the generic activation→contraction brake to it.
    if (
        pre_expensive_work
        and post_expensive_work > _BLOWUP_FACTOR * pre_expensive_work
        and not _is_flash_offer_shaped(merged)
        and _single_source_activation(graph, producer)
        and not _has_peer_activation_input(graph, producer, consumer)
    ):
        raise RuleSkipped(f"expensive-work blowup: post={post_expensive_work} > {_BLOWUP_FACTOR}× pre={pre_expensive_work}")
    if post_reads > _BLOWUP_FACTOR * pre_reads:
        raise RuleSkipped(f"read blowup: post={post_reads} > {_BLOWUP_FACTOR}× pre={pre_reads}")

    # Broadcast-materialization guard: fusing a compute-bearing producer into
    # a pure-indexmap consumer whose output volume exceeds the producer's
    # replicates the producer's body across the extra axes (the indexmap's
    # broadcast stops being lazy). Skip — the indexmap can still fuse the
    # *other* way, into its downstream consumer.
    if _is_pure_indexmap(consumer.op) and not _is_pure_indexmap(producer.op) and _output_numel(consumer.op) > _output_numel(producer.op):
        raise RuleSkipped(
            f"broadcast materialization: pure-indexmap consumer numel {_output_numel(consumer.op)} > "
            f"compute producer numel {_output_numel(producer.op)}"
        )

    frag = _wrap_merge_fragment(graph, merged, consumer)
    match.output = consumer.id
    match.consumed = {producer.id, consumer.id}
    return frag
