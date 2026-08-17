"""Merge adjacent ``LoopOp``s via graph splicing.

Every match is represented as a producer-to-sink region: one consumer is
the degenerate two-node case, while a fan-out uses its nearest owned
reconvergence. The complete region goes to the N-way splicer. Shared
internal definitions remain SSA and are emitted once per equal
scope/coordinate demand; no frontend treeification is needed. The splicer also
handles multiple consumer loads and shared external inputs uniformly
(first-seen slot assignment + splice-edge routing).

Fusion is greedy-maximal and algebra-only: every legal merge is taken. Which form of a region is
worth deploying — fused, or split into smaller kernels — is decided by the deploy evidence
hierarchy (tuned goldens, measurements, the prior); this rule never weighs shapes, hardware, or
downstream pattern knowledge. Its refusals are semantic — structural region ownership, a real
splicer rejection, the fence around an already-realized ``__cut_`` workspace (which keeps a
placement decision from being re-fused), and two readable-seam refusals judged on the MERGED
form: a merge must not nest a reduce ``Loop`` inside another reduce ``Loop``, and must not
entangle a multi-statistic compound (the online-softmax pair) beyond its readable tails — the
flat same-extent normalize sweep, or a free sweep of flat same-extent additive folds (the
expectation channels of a fused softmax·V region, which the online-softmax pairing joins into
one streaming loop) — other shapes fall to the raw-loop escape downstream (no schedule tier, no
``PLACE`` seam), so evidence could never price the split back — plus one boundedness bound:
``_total_work`` sums the
enclosing free×reduce iteration count of every compute leaf, and a merge that grows it by more
than ``_BLOWUP_FACTOR`` is refused. That bound is what keeps the downstream problem finite, not a
performance preference: unbounded splicing folds a whole transformer layer into ONE loop nest
(measured: 57 nested loops with the row statistic replayed inside a ~10¹³-iteration nest) — a
form no schedule can run and recognition cannot certify. Within the bound, every fused/split
tradeoff belongs to measured evidence.
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


def _nests_reduce(loop_op: LoopOp) -> bool:
    """Whether the body carries a reduce ``Loop`` (transitively) inside another reduce ``Loop``.
    That shape is unreadable downstream: ``_lift_cell`` keeps only the raw-loop escape for a
    cell with more than one top-level reduce or a reduce nested in a reduce, so no schedule
    tier beyond option-0 exists and no ``PLACE`` seam can split it back apart."""
    reduce_names = loop_op.reduce_axis_names

    def walk(stmts: Body, inside: bool) -> bool:
        for stmt in stmts:
            if isinstance(stmt, Loop):
                is_reduce = stmt.axis.name in reduce_names
                if is_reduce and inside:
                    return True
                if walk(stmt.body, inside or is_reduce):
                    return True
        return False

    return walk(loop_op.body, False)


def _entangled_multi_stat(loop_op: LoopOp) -> bool:
    """Whether a body holds ≥2 sibling reduce ``Loop``s (a multi-statistic compound — the
    online-softmax pair) entangled with anything beyond its readable tails. Two tail shapes are
    readable: a flat sibling loop over the same extent (the normalize sweep the online-softmax
    pairing consumes), and a free sibling sweep whose directly nested loops are all flat
    same-extent single-``Accum`` additive reduce loops with pure companions — the expectation
    channels the pairing sinks into and joins (a fused softmax·V region: one additive fold per
    output cell). A sibling loop over a FOREIGN extent outside these shapes, or any deeper
    nesting, replays or expands the statistics into a cell recognition can only keep as the
    raw-loop escape — no schedule tier, no ``PLACE`` seam."""
    reduce_names = loop_op.reduce_axis_names

    def additive_folds(lp: Loop, extent) -> bool:
        inner = [s for s in lp.body if isinstance(s, Loop)]
        if not inner:
            return False
        for t in inner:
            if t.axis.name not in reduce_names or t.axis.extent != extent:
                return False
            if any(isinstance(s, Loop) for s in t.body):
                return False
            accs = [s for s in t.body if isinstance(s, Accum)]
            if len(accs) != 1 or accs[0].op.reduce_canon != "add":
                return False
            if any(not s.pure for s in t.body if not isinstance(s, Accum)):
                return False
        return True

    def walk(stmts: Body) -> bool:
        loops = [s for s in stmts if isinstance(s, Loop)]
        reds = [lp for lp in loops if lp.axis.name in reduce_names]
        if len(reds) >= 2:
            first = reds[0].axis.extent
            if any(r.axis.extent != first for r in reds[1:]):
                return True
            for lp in loops:
                if lp.axis.extent == first and not any(isinstance(s, Loop) for s in lp.body):
                    continue  # the flat same-extent tail (a further statistic / the normalize sweep)
                if lp.axis.name not in reduce_names and additive_folds(lp, first):
                    continue  # a free sweep of same-extent additive folds (expectation channels)
                return True
            return False
        return any(walk(lp.body) for lp in loops)

    return walk(loop_op.body)


def _total_work(loop_op: LoopOp) -> int:
    """Sum enclosing-loop iterations over arithmetic leaves.

    Splicing a producer body at two demand sites doubles this metric, unlike
    the previous maximum-nest approximation.
    """
    return sum(cost for stmt, cost in _walk_leaf_costs(loop_op) if isinstance(stmt, (Assign, Accum))) or 1


PATTERN = [Pattern("producer", LoopOp)]
# Region discovery is dynamic. Watching immediate consumers preserves overlap
# invalidation when matches are enumerated in batches.
WATCH_CONSUMERS = True


def _merge_region(match: Match, region: set[str], sink: Node) -> Graph:
    """Splice an owned one- or multi-consumer region, capped only by compute growth."""
    graph = match.graph
    if any("__cut_" in node_id for node_id in region - {sink.id}):
        raise RuleSkipped("region crosses a decided placement cut")

    merged = _build_merged_region(graph, region, sink)
    if merged is None:
        raise RuleSkipped("N-way Loop splicer rejected the region")
    if _nests_reduce(merged) and not any(_nests_reduce(graph.nodes[node_id].op) for node_id in region):
        raise RuleSkipped("merge nests a reduce loop inside a reduce loop — an unreadable seam (raw-loop escape only)")
    if _entangled_multi_stat(merged) and not any(_entangled_multi_stat(graph.nodes[node_id].op) for node_id in region):
        # The single-statistic computed-A shape is readable (the fused norm→linear kind); a
        # multi-statistic compound (the online-softmax pair) stays readable only with its flat
        # same-extent normalize tail or a free sweep of same-extent additive folds (the
        # expectation channels the pairing joins). A merge that entangles the pair with any
        # other tail produces a cell recognition keeps as the raw-loop escape.
        raise RuleSkipped("merge entangles a multi-statistic compound — an unreadable seam")
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
