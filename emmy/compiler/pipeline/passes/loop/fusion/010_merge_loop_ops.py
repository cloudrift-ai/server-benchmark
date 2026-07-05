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

Pure-indexmap producers are exempt from the flash-boundary guards below
EXCEPT when the indexmap folds axes (``_helpers.folds_axes`` — a Load slot
reading ≥2 vars, e.g. a flat ``[s, h·d]`` projection's reshape): fusing one
leaves the flash offer site's operand not plainly indexed per slot, so
recognition (``_extract_qk`` / ``_extract_v_layout``) fails and the fuse
silently degrades to cut. Such indexmaps stay materialized at those sites
(a small layout copy) — the V-side counterpart of keeping RoPE cones out
of the score producer.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import Accum, Assign, Load, Loop, LoopOp
from emmy.compiler.ir.stmt import Body
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._helpers import build_merged_op as _build_merged_op
from emmy.compiler.pipeline.passes.loop.fusion._helpers import folds_axes as _folds_axes
from emmy.compiler.pipeline.passes.loop.fusion._helpers import is_flash_offer_shaped as _is_flash_offer_shaped
from emmy.compiler.pipeline.passes.loop.fusion._helpers import is_pure_indexmap as _is_pure_indexmap
from emmy.compiler.pipeline.passes.loop.fusion._helpers import pending_contraction_half as _pending_contraction_half
from emmy.compiler.pipeline.passes.loop.fusion._helpers import sum_contracts_exp_producer as _sum_contracts_exp_producer
from emmy.compiler.pipeline.passes.loop.fusion._helpers import through_indexmap_users as _through_indexmap_users
from emmy.compiler.pipeline.passes.loop.fusion._helpers import wrap_merge_fragment as _wrap_merge_fragment

_BLOWUP_FACTOR = 8


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
    return _total_work(op) > _REDUCE_HEAVY_WORK_PER_OUTPUT * _output_numel(op)


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

    # Contraction-half protection: only the product's own indexmap plumbing (unsqueeze /
    # broadcast scaffolding) may fuse into a bare matmul product before its sum-reduce partner
    # does — see :func:`_pending_contraction_half`. The blocked producer isn't lost: once the
    # halves merge, it retries against the full contraction kernel on a later fixpoint sweep.
    if not _is_pure_indexmap(producer.op) and _pending_contraction_half(graph, consumer):
        raise RuleSkipped("consumer is an unfused contraction half — the product merges with its reduce first")

    # Flash-consumer protection: a softmax-then-P@V kernel (the flash recognizer's offer site) is
    # owned by ``try_flash`` — a compute-bearing producer fused into it lands extra Loads in the
    # P@V accum loop and the V-operand extraction fails (``others == 1``), so the flash form
    # silently never certifies (a computed V — Gemma's V-norm — was the finding-2 chain's second
    # link). The producer stays materialized; flash streams it as the plain-buffer V. An
    # AXIS-FOLDING indexmap (``_folds_axes`` — a flat ``[s, h·d]`` projection's reshape, the
    # model-graph V layout) breaks the certification the same way — its fused Load slot is no
    # longer a plain var, so ``_extract_v_layout`` fails — so it loses the pure-indexmap
    # exemption here; permutation / broadcast / slice indexmaps keep fusing. A folding indexmap
    # reaches the offer site THROUGH the matmul's unsqueeze/broadcast scaffolding indexmap, so
    # its guard site is the compute consumer at the end of that chain
    # (``_through_indexmap_users``), not the scaffolding. The test is
    # deliberately PREDICTIVE as well (``_sum_contracts_exp_producer`` — which also covers the
    # bare P@V product half before its reduce partner merges), and the formed-composite test is structural
    # (``_is_flash_offer_shaped``) rather than the recognizer's verdict: fusion order is free, so
    # the V-side producer can arrive while the kernel is still a bare P@V contraction whose
    # softmax half hasn't fused in yet — the offer site only forms afterwards, too late to
    # protect. The score producer's mirror-image deferral lives in tile recognition
    # (``is_flash_score_producer``).
    folding_indexmap = _is_pure_indexmap(producer.op) and _folds_axes(producer.op)
    flash_site = _through_indexmap_users(graph, consumer) if folding_indexmap else consumer
    if (not _is_pure_indexmap(producer.op) or folding_indexmap) and (
        _is_flash_offer_shaped(flash_site.op) or _sum_contracts_exp_producer(graph, flash_site, producer)
    ):
        raise RuleSkipped("consumer is a (future) flash softmax-then-P@V offer site — its operands stay materialized")

    # Flash score-producer protection, the Q/K mirror of the above: the score matmul feeding a
    # softmax must keep its Q/K as plain ``Load``\\ s for ``_extract_qk`` — an inlined RoPE /
    # q-k-norm cone de-certifies the flash unit ("score producer's Q/K are not plain loads").
    # Only accum-free elementwise producers defer (the operand-cone shape: RoPE's mul/trig
    # chain, the norm's final scale-mul) — reduce-bearing producers are the score's own
    # dataflow assembling itself (the QK contraction fusing into its scale/mask epilogue) and
    # pass through, as does the bare-product half merging with its reduce partner. An
    # axis-folding indexmap loses the pure-indexmap exemption here too (same reason as the
    # V side: its fused Load slot de-certifies ``_extract_qk``'s plain-slot recovery).
    if (
        (not _is_pure_indexmap(producer.op) or folding_indexmap)
        and not any(isinstance(s, Accum) for s in producer.op.body.iter())
        and not _is_reduce_partner_merge(producer, consumer)
        and _feeds_softmax(graph, flash_site)
    ):
        raise RuleSkipped("consumer feeds a softmax (attention score producer) — Q/K cones stay materialized")

    # ``build_merged_op`` hands a two-node subgraph to ``splice_graph`` and
    # returns None on any unsupported pattern: σ-solve failure (writer/reader
    # index forms incompatible), missing axis in consumer scope, or
    # splicer-internal validity issues. The rule treats them uniformly —
    # the producer/consumer pair stays separate.
    merged = _build_merged_op(graph, producer, consumer)
    if merged is None:
        raise RuleSkipped(f"splice_graph rejected pattern: {producer.id!r} -> {consumer.id!r}")

    pre_work = _total_work(producer.op) + _total_work(consumer.op)
    pre_reads = _total_reads(producer.op) + _total_reads(consumer.op)
    post_work = _total_work(merged)
    post_reads = _total_reads(merged)
    if post_work > _BLOWUP_FACTOR * pre_work:
        raise RuleSkipped(f"work blowup: post={post_work} > {_BLOWUP_FACTOR}× pre={pre_work}")
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
