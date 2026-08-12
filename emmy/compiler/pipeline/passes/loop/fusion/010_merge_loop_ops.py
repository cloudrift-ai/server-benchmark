"""Merge adjacent ``LoopOp``s via graph splicing.

Every match is represented as a producer-to-sink region: one consumer is
the degenerate two-node case, while a fan-out uses its nearest owned
reconvergence. The complete region goes to the N-way splicer. Shared
internal definitions remain SSA and are emitted once per equal
scope/coordinate demand; no frontend treeification is needed. The splicer also
handles multiple consumer loads and shared external inputs uniformly
(first-seen slot assignment + splice-edge routing).

Two materialization-policy guards remain. Aggregate compute growth:
``_total_work`` sums the enclosing free×reduce iteration count of every compute
leaf, and a merge is refused when that grows by more than ``_BLOWUP_FACTOR``.
And the multi-load-of-reducer refusal: a reduce-heavy producer read through
more than one ``Load`` across the rest of the region stays materialized —
splicing it re-executes its reduce per demand site, which the work ratio only
sees when the producer is large relative to the consumer; the refusal catches
it structurally. A rowmax-bearing sink is exempt: the softmax assembly
swallowing its score producer is a fused root flash recognition accepts and
re-synthesizes without the duplication. Structural region ownership, a real splicer rejection,
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
from emmy.compiler.ir.loop import Accum, Assign, Loop, LoopOp
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


# Producers with more than a handful of operations per output element are
# "reduce-heavy": their output at position p requires non-trivial compute,
# typically a reduce whose body depends on p. Splicing such a producer at
# several demand sites re-executes that reduce per site. Pure-elementwise
# chains sit at roughly 1–3 operations per output and softmax's max + exp at
# roughly 3, while a contraction sits at its reduce extent; a threshold of 4
# separates the regimes.
_REDUCE_HEAVY_WORK_PER_OUTPUT = 4


def _output_numel(loop_op: LoopOp) -> int:
    reduce_names = loop_op.reduce_axis_names
    numel = 1
    for axis in loop_op.axes:
        if axis.name not in reduce_names:
            numel *= axis.extent.as_static() if axis.extent.is_static else 128
    return numel


def _has_rowmax(loop_op: LoopOp) -> bool:
    """A ``maximum`` Accum — softmax's rowmax signature."""
    return any(isinstance(stmt, Accum) and stmt.op.reduce_canon == "maximum" for stmt in loop_op.body.iter())


def _reduce_heavy(loop_op: LoopOp) -> bool:
    """More than ``_REDUCE_HEAVY_WORK_PER_OUTPUT`` operations per output element.

    A softmax assembly (rowmax-bearing body) discounts its ``add`` Assigns:
    they are the score's mask applications — one predicate-add per mask, not
    real duplicated compute. Without the discount a multi-mask score (a
    stamped sliding-window SDPA carries up to three) pushes the cheap
    max + exp reducer past the threshold and the softmax cone never assembles
    onto its P@V offer site.
    """
    work = _total_work(loop_op)
    if _has_rowmax(loop_op):
        work -= sum(cost for stmt, cost in _walk_leaf_costs(loop_op) if isinstance(stmt, Assign) and stmt.op.name == "add")
    return work > _REDUCE_HEAVY_WORK_PER_OUTPUT * _output_numel(loop_op)


def _region_loads_from(graph: Graph, region: set[str], producer_id: str) -> int:
    """Number of ``Load`` stmts across the other region members reading ``producer_id``'s buffer."""
    return sum(1 for node_id in region - {producer_id} for load in graph.nodes[node_id].op.body.loads if load.input == producer_id)


PATTERN = [Pattern("producer", LoopOp)]
# Region discovery is dynamic. Watching immediate consumers preserves overlap
# invalidation when matches are enumerated in batches.
WATCH_CONSUMERS = True


def _merge_region(match: Match, region: set[str], sink: Node) -> Graph:
    """Splice an owned one- or multi-consumer region; refuse reduce duplication and compute growth."""
    graph = match.graph
    interior = region - {sink.id}
    if any("__cut_" in node_id for node_id in interior):
        raise RuleSkipped("region crosses a decided placement cut")
    # A rowmax-bearing sink is exempt: a softmax assembly's repeated reads are its
    # two-pass max+exp sweep over the one score it swallows, and flash recognition
    # accepts that fused root and re-synthesizes the unit without the duplication.
    if not _has_rowmax(sink.op):
        for node_id in interior:
            if _reduce_heavy(graph.nodes[node_id].op) and _region_loads_from(graph, region, node_id) > 1:
                raise RuleSkipped("reduce-heavy producer read through >1 Load — fusion would duplicate the reduce")

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
