"""The region splice shared by the two merge passes.

Lives in a ``_``-prefixed module so the pass loader skips it (only ``NNN_<name>.py`` files load as
rules). ``loop/prefusion`` and ``loop/fusion`` run the SAME splice with the same refusals and the
same work bound; they differ only in the producer predicate that decides which regions each one
offers, so the merge semantics have exactly one home.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import Accum, Assign, Loop, LoopOp
from emmy.compiler.ir.stmt import Body
from emmy.compiler.pipeline import Match, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._helpers import build_merged_region as _build_merged_region
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

    The refusal's criterion is REVERSIBILITY, not readability alone: a merge evidence cannot price
    the split back out of is one the compiler can never undo. Recognition may discharge the
    refusal when placement proves an exact grouped inverse: two alpha-equivalent computed edges
    reconstruct one workspace producer and remain beside the fused form as a priced sibling.
    Every other nested reduce keeps this fail-closed refusal."""
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


def _recognized_reuse_pieces(loop_op: LoopOp) -> tuple[LoopOp, LoopOp] | None:
    """The exact grouped placement inverse when recognition can prove one.

    The dependency is lazy because fusion stays a Loop-IR pass; it asks the existing tile
    recognizer only for a boundedness witness, never for a performance decision. The returned
    child and parent are the same structural split option placement will expose later.
    """
    from emmy.compiler.pipeline.passes.lowering.tile._cut import reusable_cut_pieces  # noqa: PLC0415

    return reusable_cut_pieces(loop_op)


def _may_need_grouped_reuse(graph: Graph, region: set[str], sink: Node) -> bool:
    """Whether construction may expose or preserve a grouped placement inverse.

    A new grouped placement shares one additive contraction at exactly two distinct coordinate
    demands. Everything between that producer and the sink is linear: only the producer fans out,
    only the sink joins, and every intermediate has one coordinate demand. A merge that extends
    one already-recognized inverse is eligible only along a single linear path. The post-build
    recognizer remains the authoritative witness in both cases.
    """
    parents = {node_id: set() for node_id in region}
    users = {node_id: set() for node_id in region}
    demands: dict[str, list[tuple]] = {node_id: [] for node_id in region}
    for consumer_id in region:
        for load in graph.nodes[consumer_id].op.body.loads:
            if load.input in region:
                parents[consumer_id].add(load.input)
                users[load.input].add(consumer_id)
                if not any(load.index == existing for existing in demands[load.input]):
                    demands[load.input].append(load.index)

    sources = [node_id for node_id in region if not parents[node_id]]
    if len(sources) != 1:
        return False
    source = sources[0]

    linear = all(
        (not parents[node_id] and len(users[node_id]) == 1 and len(demands[node_id]) == 1)
        if node_id == source
        else (len(parents[node_id]) == 1 and not users[node_id])
        if node_id == sink.id
        else (len(parents[node_id]) == 1 and len(users[node_id]) == 1 and len(demands[node_id]) == 1)
        for node_id in region
    )
    if linear and any(_recognized_reuse_pieces(graph.nodes[node_id].op) is not None for node_id in region):
        return True

    accums = [stmt for stmt in graph.nodes[source].op.body.iter() if isinstance(stmt, Accum)]
    if len(accums) != 1 or accums[0].op.reduce_canon != "add" or _nests_reduce(graph.nodes[source].op):
        return False
    if len(demands[source]) != 2:
        return False
    for node_id in region - {source, sink.id}:
        if len(parents[node_id]) != 1 or len(users[node_id]) != 1 or len(demands[node_id]) != 1:
            return False
    return not users[sink.id]


def merge_region(match: Match, region: set[str], sink: Node) -> Graph:
    """Splice an owned one- or multi-consumer region, capped only by compute growth.

    Shared by both merge passes: ``loop/prefusion`` and ``loop/fusion`` differ only in WHICH
    regions they offer, never in how one is spliced or what is refused."""
    graph = match.graph
    if any("__cut_" in node_id for node_id in region - {sink.id}):
        raise RuleSkipped("region crosses a decided placement cut")

    pre_work = sum(_total_work(graph.nodes[node_id].op) for node_id in region)
    construction_limit = None if _may_need_grouped_reuse(graph, region, sink) else _BLOWUP_FACTOR * pre_work
    merged = _build_merged_region(graph, region, sink, max_work=construction_limit)
    if merged is None:
        raise RuleSkipped("N-way Loop splicer rejected the region")
    reuse_pieces = _recognized_reuse_pieces(merged)
    if reuse_pieces is None and any(_recognized_reuse_pieces(graph.nodes[node_id].op) is not None for node_id in region):
        raise RuleSkipped("merge destroys a recognized grouped placement inverse")
    if _nests_reduce(merged) and not any(_nests_reduce(graph.nodes[node_id].op) for node_id in region) and reuse_pieces is None:
        raise RuleSkipped("merge nests a reduce loop inside a reduce loop — an unreadable seam (raw-loop escape only)")
    if (
        _entangled_multi_stat(merged)
        and not any(_entangled_multi_stat(graph.nodes[node_id].op) for node_id in region)
        and reuse_pieces is None
    ):
        # The single-statistic computed-A shape is readable (the fused norm→linear kind); a
        # multi-statistic compound (the online-softmax pair) stays readable only with its flat
        # same-extent normalize tail or a free sweep of same-extent additive folds (the
        # expectation channels the pairing joins). A merge that entangles the pair with any
        # other tail produces a cell recognition keeps as the raw-loop escape.
        raise RuleSkipped("merge entangles a multi-statistic compound — an unreadable seam")
    post_work = _total_work(merged)
    if reuse_pieces is not None:
        # Raw Loop IR spells the equal producer once at each demand site and may nest one copy
        # below an output sweep. Recognition proves that the tile form reuses the producer, and
        # placement supplies the priced materialized sibling. Bound that recognized work, not the
        # duplicated spelling; unrelated merges keep the ordinary raw-loop count.
        post_work = min(post_work, sum(_total_work(piece) for piece in reuse_pieces))
    if post_work > _BLOWUP_FACTOR * pre_work:
        raise RuleSkipped(f"work blowup: post={post_work} > {_BLOWUP_FACTOR}× pre={pre_work}")

    match.consumed = region
    match.output = sink.id
    return _wrap_merge_fragment(graph, merged, sink)
