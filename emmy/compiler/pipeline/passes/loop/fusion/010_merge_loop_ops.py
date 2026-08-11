"""Merge adjacent ``LoopOp``s via graph splicing.

Every match is represented as a producer-to-sink region: one consumer is
the degenerate two-node case, while a fan-out uses its nearest owned
reconvergence. The complete region goes to the N-way splicer. Shared
internal definitions remain SSA and are emitted once per equal
scope/coordinate demand; no frontend treeification is needed. The splicer also
handles multiple consumer Loads and shared external inputs uniformly
(first-seen slot assignment + splice-edge routing).
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
from emmy.compiler.pipeline.passes.loop.fusion._helpers import build_merged_region as _build_merged_region
from emmy.compiler.pipeline.passes.loop.fusion._helpers import closed_loop_consumer_region as _closed_loop_consumer_region
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
    packed buffer. Once the complete gated activation has fused into one
    LoopOp those slices are no longer separate graph inputs: they are distinct
    coordinate reads of the same packed input. Preserve that information so
    region fusion does not misclassify GeGLU/SwiGLU as a unary activation and
    materialize a serving kernel that already has measured fused goldens.

    Before the activation itself is fused, the peer can instead have a
    different declared shape while still being an ancestor shared by the
    activation and its consumer. Keep that established graph-level check too.
    """
    load_sites: dict[str, set[tuple]] = {}
    for load in producer.op.body.loads:
        tensor = graph.buffer(load.input)
        if tensor is None or not tensor.shape or all(dim.is_static and dim.as_static() == 1 for dim in tensor.shape):
            continue
        load_sites.setdefault(load.input, set()).add(load.index)
    if any(len(indices) > 1 for indices in load_sites.values()):
        return True

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


def _accum_feed_names(op: LoopOp) -> set[str]:
    """SSA names that transitively FEED an add-``Accum``'s value — the contraction's own
    dataflow. A post-reduce epilogue operand (a residual add's other input, combined with the
    finalized accumulator AFTER the reduce loop) never lands in this set."""
    sources: dict[str, tuple[str, ...]] = {}
    for s in op.body.iter():
        if isinstance(s, Assign):
            sources[s.name] = s.args
        elif isinstance(s, Select):
            sources[s.name] = tuple(b.value for b in s.branches)
    feed: set[str] = set()
    frontier = [s.value for s in op.body.iter() if isinstance(s, Accum) and s.op.reduce_canon == "add"]
    while frontier:
        nm = frontier.pop()
        if nm in feed:
            continue
        feed.add(nm)
        frontier.extend(sources.get(nm, ()))
    return feed


def _sum_contracts_exp_producer(graph: Graph, consumer: Node, producer: Node) -> bool:
    """``consumer`` is a sum-contraction one of whose OTHER contraction operands is (or is one
    producer hop away from) an exp-bearing ``LoopOp`` — the P@V kernel before its softmax fuses
    in, at any point of the softmax's own assembly. Once it does, the kernel is the flash offer
    site; a compute producer fused into the V side in the meantime breaks the certification just
    as surely, so it is deferred the same way. The softmax-side operand itself (``producer``) is
    exempt — that IS the softmax fusing in. A gelu chain feeding a contraction stays fuseable:
    the gelu IS the producer there, so the other-input scan never sees its exp.

    Only operands whose loads feed the add-``Accum`` count (:func:`_accum_feed_names`): a
    post-reduce EPILOGUE operand is not part of the contraction, so an exp upstream of it says
    nothing about a future flash site. The residual stream is the load-bearing case — every
    residual add past layer 0 sits within two producer hops of the previous layer's silu (its
    sigmoid's exp) or softmax, and treating it as a V operand kept every o_proj product
    materialized (a (b, s, k, n) gmem scratch per layer)."""
    op = consumer.op
    if not any(isinstance(s, Accum) and s.op.reduce_canon == "add" for s in op.body.iter()):
        return False

    def has_exp(node) -> bool:
        return isinstance(node.op, LoopOp) and any(isinstance(s, Assign) and s.op.name == "exp" for s in node.op.body.iter())

    feed = _accum_feed_names(op)
    for inp in consumer.inputs:
        if inp == producer.id:
            continue
        if not any(nm in feed for ld in op.body.loads if ld.input == inp for nm in ld.names):
            continue  # post-reduce epilogue operand — not a contraction operand
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


PATTERN = [Pattern("producer", LoopOp)]
# The region is discovered dynamically. Watching immediate consumers preserves
# the pair rule's overlap invalidation when matches are enumerated in batches.
WATCH_CONSUMERS = True


def _guard_region_member(graph: Graph, member: Node, sink: Node) -> None:
    """Apply the established pair boundary policy to one region member.

    For a two-node region ``member`` is the direct producer. For a fan-out,
    checking every interior member against the common sink prevents an
    indirect typed-copy, contraction-half, or attention operand cone from
    bypassing the same materialization boundary merely because it branched.
    """
    if _reduce_heavy(member.op) and _count_loads_from(sink.op, member.id) > 1:
        raise RuleSkipped("reduce-heavy producer feeds sink through >1 Load — fusion would duplicate the reduce")

    if _is_pure_indexmap(member.op) and not _is_castfree_indexmap(graph, member) and _is_pure_indexmap(sink.op):
        raise RuleSkipped("cast copy feeding indexmap plumbing — the cast stays at the traced dtype boundary")

    if not _is_castfree_indexmap(graph, member) and _pending_contraction_half(graph, sink):
        raise RuleSkipped("sink is an unfused contraction half — the product merges with its reduce first")

    # Flash P@V operands and Q/K score cones must remain plain loads at the
    # recognition boundary. These tests are predictive because fusion order is
    # free: the complete flash composite may not have formed yet.
    if not _is_castfree_indexmap(graph, member) and (_is_flash_offer_shaped(sink.op) or _sum_contracts_exp_producer(graph, sink, member)):
        raise RuleSkipped("sink is a (future) flash softmax-then-P@V offer site — its operands stay materialized")

    if (
        not _is_castfree_indexmap(graph, member)
        and not any(isinstance(stmt, Accum) for stmt in member.op.body.iter())
        and not _is_reduce_partner_merge(member, sink)
        and not _mask_epilogue(member.op)
        and _feeds_softmax(graph, sink)
    ):
        raise RuleSkipped("sink feeds a softmax (attention score producer) — Q/K cones stay materialized")

    if (
        not _is_castfree_indexmap(graph, member)
        and any(isinstance(stmt, Accum) for stmt in member.op.body.iter())
        and _mask_epilogue(sink.op)
        and _feeds_softmax(graph, sink)
    ):
        raise RuleSkipped("score contraction stays clear of softmax mask epilogues — the masks ride the softmax sink")


def _merge_region(match: Match, producer: Node, region: set[str], sink: Node) -> Graph:
    """Splice a validated one- or multi-consumer region through one path."""
    graph = match.graph
    interior = region - {sink.id}
    if any("__cut_" in nid for nid in interior):
        raise RuleSkipped("region crosses a decided placement cut")
    for nid in interior:
        member = graph.nodes[nid]
        if len(region) > 2 and any(isinstance(stmt, Accum) for stmt in member.op.body.iter()):
            raise RuleSkipped(f"interior reducer {nid!r} stays materialized")
        _guard_region_member(graph, member, sink)
    if len(region) > 2 and (_has_rowmax(sink.op) or _is_flash_offer_shaped(sink.op) or _feeds_softmax(graph, sink)):
        raise RuleSkipped("reconvergent region crosses an attention recognition boundary")

    merged = _build_merged_region(graph, region, sink)
    if merged is None:
        raise RuleSkipped("N-way Loop splicer rejected the region")
    pre_work = sum(_total_work(graph.nodes[nid].op) for nid in region)
    pre_expensive_work = sum(_total_expensive_work(graph.nodes[nid].op) for nid in region)
    pre_reads = sum(_total_reads(graph.nodes[nid].op) for nid in region)
    post_work = _total_work(merged)
    post_expensive_work = _total_expensive_work(merged)
    post_reads = _total_reads(merged)
    if post_work > _BLOWUP_FACTOR * pre_work:
        raise RuleSkipped(f"work blowup: post={post_work} > {_BLOWUP_FACTOR}× pre={pre_work}")
    if post_reads > _BLOWUP_FACTOR * pre_reads:
        raise RuleSkipped(f"read blowup: post={post_reads} > {_BLOWUP_FACTOR}× pre={pre_reads}")
    if pre_expensive_work and post_expensive_work > _BLOWUP_FACTOR * pre_expensive_work and not _is_flash_offer_shaped(merged):
        if len(region) > 2 or (_single_source_activation(graph, producer) and not _has_peer_activation_input(graph, producer, sink)):
            raise RuleSkipped(f"expensive-work blowup: post={post_expensive_work} > {_BLOWUP_FACTOR}× pre={pre_expensive_work}")

    if (
        len(region) == 2
        and _is_pure_indexmap(sink.op)
        and not _is_pure_indexmap(producer.op)
        and _output_numel(sink.op) > _output_numel(producer.op)
    ):
        raise RuleSkipped(
            f"broadcast materialization: pure-indexmap consumer numel {_output_numel(sink.op)} > "
            f"compute producer numel {_output_numel(producer.op)}"
        )

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
    return _merge_region(match, producer, region, sink)
