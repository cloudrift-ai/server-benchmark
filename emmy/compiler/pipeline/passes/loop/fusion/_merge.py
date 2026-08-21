"""The region splice shared by the two merge passes.

Lives in a ``_``-prefixed module so the pass loader skips it (only ``NNN_<name>.py`` files load as
rules). ``loop/prefusion`` and ``loop/fusion`` run the SAME splice with the same refusals and the
same work bound; they differ only in the producer predicate that decides which regions each one
offers, so the merge semantics have exactly one home.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var, _ExprOps
from emmy.compiler.ir.loop import Accum, Assign, Loop, LoopOp
from emmy.compiler.ir.pure.fold import deep_defines, deep_reads
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
    the split back out of is one the compiler can never undo. Attention is the case that makes the
    distinction sharp. Its ``Q·Kᵀ`` producer has TWO consumers in the merged cell (the streaming
    softmax statistic and the weight cone), so splicing it duplicates the producer at each demand
    site — and no ``PLACE`` cut puts that back: a cut fragment serves ONE consumer (an operand edge
    is inline, and there is no let table), so cutting both seams mints two score kernels, never the
    one shared producer the unmerged graph had. The merged cell is READABLE downstream — recognition
    lifts each spliced producer to an operand edge of the step's fold
    (``lowering/tile/_fromloop``), it binds as a computed-A cone over a computed score and it runs —
    but until that fused form is realized at the tensor-core tier it is far slower than the
    two-kernel graph this refusal preserves, and evidence has no way back."""
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


def _chains_statistic_in_sweep(loop_op: LoopOp) -> bool:
    """Whether a free sweep replays a reduce ``Loop`` whose accumulators feed a LATER reduce
    ``Loop`` in the same sweep — a per-step statistic chained into a contraction.

    One fold per sweep step is readable: the fused norm→linear sweep (the row statistic is the
    sweep's outer sibling, each step is one dot) and softmax·V's expectation channels (one
    additive fold per channel). A statistic recomputed INSIDE the sweep and consumed by a second
    fold there is not — recognition cannot hoist it to an operand (its reads are sweep-local) and
    keeps the whole step as the raw-loop escape, with no schedule tier and no ``PLACE`` seam, so
    evidence could never price the split back. Attention's k-norm → RoPE → ``Q·Kᵀ`` region is the
    case: fused greedily, the key statistic replays once per query row and the contraction runs
    scalar."""
    reduce_names = loop_op.reduce_axis_names

    def walk(stmts: Body, in_sweep: bool) -> bool:
        # Names carrying a statistic computed at THIS level: a reduce loop's accumulators and the
        # pure stmts derived from them. Names from enclosing levels are the readable outer state.
        tainted: set[str] = set()
        seen_reduce = False
        for stmt in stmts:
            if not isinstance(stmt, Loop):
                if set(stmt.deps()) & tainted:
                    tainted |= set(stmt.defines())
                continue
            is_reduce = stmt.axis.name in reduce_names
            if is_reduce:
                if in_sweep and tainted & deep_reads(list(stmt.body)):
                    return True
                tainted |= deep_defines(stmt)
            # A free loop is a sweep once a sibling reduce precedes it (or it sits inside one).
            if walk(stmt.body, in_sweep or (not is_reduce and seen_reduce)):
                return True
            seen_reduce = seen_reduce or is_reduce
        return False

    return walk(loop_op.body, False)


def _mentions_plain(expr, axis: str) -> bool:
    """Whether ``axis`` occurs in ``expr`` outside every floor-division-by-literal subtree. A
    ``%``-by-literal counts as plain: ``n % 16`` varies within the class ``n / 16`` names, so a
    read carrying both (the packed-word decode of a coded weight) covers ``n`` fully."""
    import dataclasses  # noqa: PLC0415

    if isinstance(expr, Var):
        return expr.name == axis
    if isinstance(expr, BinaryExpr) and expr.op in ("/", "//") and isinstance(expr.right, Literal):
        return False
    if not dataclasses.is_dataclass(expr):
        return False
    for f in dataclasses.fields(expr):
        v = getattr(expr, f.name)
        for child in v if isinstance(v, (tuple, list)) else (v,):
            if isinstance(child, _ExprOps) and _mentions_plain(child, axis):
                return True
    return False


def _replays_reduce_under_composite_axis(loop_op: LoopOp) -> bool:
    """Whether a reduce ``Loop`` sits under a free axis its reads depend on ONLY through a
    floor-division quotient (``flat / 256``): the reduce's value is constant across each class of that
    axis, so it is replayed — 256× for the flattened ``(head, d)`` output of softmax·V, whose
    per-head statistic then recomputes for every ``d`` — and no loop exists to hoist it out of
    (the lift hoists a loop-INVARIANT statistic ahead of a sweep; a composite-indexed one is
    neither invariant nor addressable by a ``PLACE`` seam). A reduce that does not read the axis
    at all is the hoistable norm→linear shape and stays."""
    reduce_names = loop_op.reduce_axis_names

    def walk(stmts: Body, free: list[str]) -> bool:
        for stmt in stmts:
            if not isinstance(stmt, Loop):
                continue
            if stmt.axis.name in reduce_names:
                exprs = [e for s in _deep_stmts(stmt.body) for e in s.exprs()]
                for axis in free:
                    mentioned = [e for e in exprs if axis in e.free_vars()]
                    if mentioned and not any(_mentions_plain(e, axis) for e in mentioned):
                        return True
                continue  # nested reduces read as the step's composed producers
            if walk(stmt.body, [*free, stmt.axis.name]):
                return True
        return False

    return walk(loop_op.body, [])


def _deep_stmts(body: Body):
    for s in body:
        yield s
        for b in s.nested():
            yield from _deep_stmts(b)


def _collapsing_loads(loop_op: LoopOp, producers: set[str]) -> bool:
    """Whether ``loop_op`` loads one of ``producers`` through a div/mod-by-literal of its own
    axes — a reshape that collapses (or splits) loop axes, whether the sink is the bare reshape
    or a compute that already absorbed it (the output-gate multiply over the flattened heads)."""
    from emmy.compiler.ir.stmt import Load  # noqa: PLC0415

    def composite(expr) -> bool:
        import dataclasses  # noqa: PLC0415

        if isinstance(expr, BinaryExpr) and expr.op in ("/", "//", "%") and isinstance(expr.right, Literal) and expr.left.free_vars():
            return True
        if not dataclasses.is_dataclass(expr):
            return False
        for f in dataclasses.fields(expr):
            v = getattr(expr, f.name)
            if any(isinstance(c, _ExprOps) and composite(c) for c in (v if isinstance(v, (tuple, list)) else (v,))):
                return True
        return False

    return any(composite(e) for s in _deep_stmts(loop_op.body) if isinstance(s, Load) and s.input in producers for e in s.index or ())


def _total_work(loop_op: LoopOp) -> int:
    """Sum enclosing-loop iterations over arithmetic leaves.

    Splicing a producer body at two demand sites doubles this metric, unlike
    the previous maximum-nest approximation.
    """
    return sum(cost for stmt, cost in _walk_leaf_costs(loop_op) if isinstance(stmt, (Assign, Accum))) or 1


def merge_region(match: Match, region: set[str], sink: Node) -> Graph:
    """Splice an owned one- or multi-consumer region, capped only by compute growth.

    Shared by both merge passes: ``loop/prefusion`` and ``loop/fusion`` differ only in WHICH
    regions they offer, never in how one is spliced or what is refused."""
    graph = match.graph
    if any("__cut_" in node_id for node_id in region - {sink.id}):
        raise RuleSkipped("region crosses a decided placement cut")

    producers = region - {sink.id}
    if _collapsing_loads(sink.op, producers) and any(graph.nodes[node_id].op.reduce_axis_names for node_id in producers):
        # Splicing a reduce-bearing producer at a collapsing reshape's load sites re-runs its
        # reduces per output element, indexed by the flat axis's quotient — the per-head softmax
        # statistic replayed for every ``d`` of the ``(head, d)`` flatten, or P·V spelled per flat
        # cell with composite operand indices no slab loader can address — with no loop to hoist
        # it out of and no seam to cut. ``030_fold_output_reshape`` retargets a pure reshape onto
        # the producer's writes instead, keeping its loop nest; a computing sink stays a kernel.
        raise RuleSkipped("sink reads a reduce-bearing producer through a collapsing reshape — an unreadable seam")
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
    if _chains_statistic_in_sweep(merged) and not any(_chains_statistic_in_sweep(graph.nodes[node_id].op) for node_id in region):
        raise RuleSkipped("merge chains a per-step statistic into a fold inside a free sweep — an unreadable seam")
    if _replays_reduce_under_composite_axis(merged) and not any(
        _replays_reduce_under_composite_axis(graph.nodes[node_id].op) for node_id in region
    ):
        raise RuleSkipped("merge replays a reduce under a free axis it reads only through a floor-division quotient — an unreadable seam")
    pre_work = sum(_total_work(graph.nodes[node_id].op) for node_id in region)
    post_work = _total_work(merged)
    if post_work > _BLOWUP_FACTOR * pre_work:
        raise RuleSkipped(f"work blowup: post={post_work} > {_BLOWUP_FACTOR}× pre={pre_work}")

    match.consumed = region
    match.output = sink.id
    return _wrap_merge_fragment(graph, merged, sink)
