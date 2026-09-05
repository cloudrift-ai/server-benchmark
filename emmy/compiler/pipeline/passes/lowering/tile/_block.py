"""Blockify a reduce axis — the rewrite that hands the scheduler a block width to bind.

Blocking splits one reduce axis into ``k_o × k_i`` and re-associates the fold over the two levels.
What each level runs is the only thing that differs between carriers:

- a PLANAR fold — a plain reduction, and a contraction, which is a reduction whose lift is a
  product — runs the SAME monoid at both levels. Associativity is the whole licence. The inner
  level keeps the fold's own lift, so a contraction stays bilinear and its site keeps the
  tensor-core tier; the outer level is a plain reduce over the trip count, which is what a
  cross-CTA split partitions.
- a TWISTED carrier's ⊕ is a rescaling program, never a commutative op with an identity, so
  ``as_contraction`` refuses at its first gate and NO site inside one can be bilinear. Attention's
  ``P·V`` is not missing from the tree; it is spelled as a coefficient of that ⊕. Blocking
  SEPARATES the two monoids: the twisted ⊕ stays on the outer fold and the inner level runs the
  base monoid over a per-block contribution, so the channel whose contribution is a product of two
  distinct cones reads as a contraction. That is FlashAttention-2's shape, derived rather than
  recognized — everything the rewrite needs is read out of the stored combine (``β``, the factor
  the merge puts on the incoming side), and the value it multiplies is the fold's own lift result.

Blocking does not change what the kernel computes, so it changes no kernel identity: it happens
INSIDE the schedule walk (``040_schedule.block_problems``), once per candidate width, and the
kernel the pipeline identifies and prices stays the unblocked one. Nor does it need a codec family
of its own — a blocked contraction's inner axis IS its K, so the width is exactly the ``k<bk>`` half
of that site's ``TILE`` and ``_blocked_kstep`` holds the two to each other. A plain reduction is not
blocked at all: ``REDUCE`` already spells its partition, and a cross-CTA split already factors the
axis.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim, simplify_extent
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.pure import Fold, Lambda
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body
from emmy.compiler.pipeline.passes.lowering.tile._split import _sliced_edge


def is_blocked(axes) -> bool:
    """Whether this kernel's axis table already carries a blocked stream (the re-firing receipt)."""
    return any(axis.window is not None and axis.window.block for axis in axes)


# ---- the split ------------------------------------------------------------------------------- #


def _split_axis(axis: Axis, width: int) -> tuple[Axis, Expr]:
    """``(the outer trip axis, the block width)`` for one reduce axis blocked at ``width``.

    The outer count is ``ceil(extent / width)``, composite in both unknowns, so one tree serves a
    static extent and a symbolic one.
    """
    blk = Literal(width, "int")
    extent = axis.extent.expr
    trips = Dim(simplify_extent(BinaryExpr("/", BinaryExpr("+", extent, BinaryExpr("-", blk, Literal(1, "int"))), blk)))
    outer = Axis(f"{axis.name}_o", trips, window=Window(parent=axis.source_axis or axis, block=True, trip=True))
    return outer, blk


def _block_binder(axis: Axis, outer: Axis, blk: Expr, tag: str) -> tuple[Axis, Sigma]:
    """One inner fold's OWN binder over the block, and the σ that reads the absolute coordinate.

    Each inner fold is a separate loop over the same block — the pivot's pass, then each channel's —
    so each needs a coordinate of its own. A term tree names one coordinate per name; sharing the
    binder makes ``lower`` place them as one loop, and the weight then reads a block pivot that has
    not finished accumulating.
    """
    inner = Axis(f"{axis.name}_{tag}", Dim(blk), window=Window(parent=axis.source_axis or axis, block=True))
    sigma = Sigma({axis.name: BinaryExpr("+", BinaryExpr("*", Var(outer.name), blk), Var(inner.name))})
    return inner, sigma


def _reindexed(fold: Fold, sigma: Sigma) -> tuple:
    """The fold's operand edges reading the block's ABSOLUTE coordinate.

    Only the operands move: the fold keeps its own binder over the narrowed axis, so nothing is
    captured. ``_sliced_edge`` is the cross-CTA split's reindexer, and it is the same job — σ maps
    one coordinate onto an EXPRESSION over several, so a param's image is that expression's free
    names rather than a rename.
    """
    return tuple(_sliced_edge(edge, sigma, fold.axis) for edge in fold.operands)


def _over_blocks(fold: Fold, outer: Axis, inner_folds: tuple[Fold, ...]) -> Fold:
    """The outer fold: the SAME ``(init, combine)`` over the block contributions.

    A block carrier has the same shape as a singleton one and the monoid does not care which it
    merges. The lift RETURNS the contributions and the carrier keeps the original state names — a
    copy onto the state name instead declares the state inside the step and shadows the carrier,
    so the merge would fold each block into itself.
    """
    contributed = tuple(name for edge in inner_folds for name in edge.exposes)
    return Fold(
        operands=inner_folds,
        lift=Lambda.closing((outer.name, *contributed), Body(()), contributed),
        init=fold.init,
        combine=fold.combine,
    )


# ---- the planar carrier: a reduction, and a contraction, which is one with a bilinear lift ---- #


def block_planar(fold: Fold, axis: Axis, width: int) -> tuple[Fold, tuple[Axis, ...]] | None:
    """A CONTRACTION re-derived over blocks under the same monoid, or ``None``.

    The inner level keeps the fold's own lift and its own ⊕, so the site stays bilinear and takes a
    tensor-core tile over a K of exactly one block; the outer level folds the per-block states
    through that same ⊕. The block IS the fragment depth, so the row's ``TILE`` spells the width
    and nothing else has to.

    A PLAIN reduction is deliberately left alone. Its reduce axis is already partitioned by
    ``REDUCE`` — ``g<n>`` across CTAs, ``coop``/``r<n>`` within one — and a cross-CTA split already
    factors the axis structurally, so blocking it would restate a decision that family makes and
    give the width no spelling of its own.
    """
    view = fold.as_reduction()
    if view is None or view.ops is None:
        return None  # twisted: :func:`block_twisted` owns it
    if fold.as_contraction() is None:
        return None  # a plain reduction: REDUCE already spells its partition
    outer, blk = _split_axis(axis, width)
    inner, sigma = _block_binder(axis, outer, blk, "i")
    states = tuple(f"{state}__blk" for state in view.states)
    body = Body(tuple(stmt.substitute(sigma) for stmt in fold.lift.body))
    block = Fold(
        operands=_reindexed(fold, sigma),
        lift=Lambda.closing((inner.name, *fold.lift.params[1:]), body, fold.lift.results),
        init=fold.init,
        combine=Lambda.componentwise(view.ops, states),
    )
    return _over_blocks(fold, outer, (block,)), (outer, inner)


# ---- the twisted carrier: two monoids, one per level ------------------------------------------ #


def _pivot_and_rescales(fold: Fold) -> tuple[str, Assign, dict[int, str]] | None:
    """Read the combine as ``(pivot op, the advance's cone, {channel index: its β name})``.

    A carrier is blockable when one component is a PIVOT — its update reads only its own pair,
    under an associative commutative op — and every other component merges as ``α·s + β·s__o`` with
    the coefficients drawn from the pivot's cone. That is the whole legality test, and it is a
    reading of the stored combine: no recipe, no op-name list.
    """
    combine, defs = fold.combine, fold.combine.body.definitions
    states = fold.as_reduction().states
    n = len(states)
    if n < 2 or combine.params[:n] != tuple(states):
        return None
    others = combine.params[n:]

    # The pivot: result 0 defined by a commutative associative op over its own pair (possibly
    # behind a copy, which is how the advance re-exposes the merged value).
    head = defs.get(combine.results[0])
    while isinstance(head, Assign) and head.op.name == "copy":
        head = defs.get(head.args[0])
    if not isinstance(head, Assign) or not (head.op.associative and head.op.commutative and head.op.has_identity):
        return None
    if set(head.args) != {states[0], others[0]}:
        return None

    betas: dict[int, str] = {}
    for index in range(1, n):
        stmt = defs.get(combine.results[index])
        if not isinstance(stmt, Assign) or stmt.op.reduce_canon != "add" or len(stmt.args) != 2:
            return None
        sides = [defs.get(arg) for arg in stmt.args]
        if any(not isinstance(s, Assign) or not s.op.semiring_product or len(s.args) != 2 for s in sides):
            return None
        # The side reading the INCOMING state carries β; the other carries α.
        incoming = [s for s in sides if others[index] in s.args]
        if len(incoming) != 1:
            return None
        (beta,) = (arg for arg in incoming[0].args if arg != others[index])
        betas[index] = beta
    return head.op.name, head, betas


def _beta_cone(fold: Fold, beta: str, pivot: Fold, m_blk: str, operands: tuple, bound: tuple) -> Fold:
    """The per-element weight, as one shared zero-axis term.

    ``β`` is the factor the combine puts on the INCOMING side of a merge. Evaluated with the pivot
    bound to the BLOCK's pivot and the incoming side to this element's own value, it is exactly the
    element's weight within the block — attention's ``exp(s − m_blk)``, derived from the combine
    rather than restated. Shared by every channel, so the weight tile is computed once per block.
    """
    states, others = fold.as_reduction().states, fold.combine.params[len(fold.as_reduction().states) :]
    score = fold.lift.results[0]
    weight = fold.combine.cone(beta).rename({states[0]: m_blk, others[0]: score})
    cone = fold.lift.cone(score)
    kept = tuple(param for param in bound if param in cone.params)
    edges = tuple(operands[index] for index, (param, _, _) in enumerate(fold.bindings) if param in kept)
    body = Body((*cone.body, *weight.body))
    return Fold(operands=(*edges, pivot), lift=Lambda.closing((*kept, m_blk), body, (beta,)))


def _channel(fold: Fold, index: int, weight: Fold, beta: str, operands: tuple, inner: Axis) -> Fold:
    """One channel's block contribution: ``⊕_{k_i} β · L``, where ``L`` is the fold's own lift
    result for that channel.

    When ``L`` is the multiplicative identity — a denominator counting one per element — the
    product folds away and the channel is a plain sum of weights. When it is a streamed value, the
    contribution is a product of two DISTINCT operand cones under a plain ``add``, which is a
    semiring step: the lift is the bare product, so ``as_contraction`` reads it and the site takes
    a tensor-core tile. That is the whole point of blocking.
    """
    state = fold.as_reduction().states[index]
    carried = fold.lift.results[index]
    defs = fold.lift.body.definitions
    const = defs.get(carried)
    unit = const is not None and not isinstance(const, Assign) and getattr(const, "value", None) == 1.0
    edges, params = [weight], [beta]
    if not unit:
        for index, (param, _, _) in enumerate(fold.bindings):
            if param == carried:
                edges.append(operands[index])  # the REINDEXED edge: the original still reads the un-split axis
                params.append(param)
    # Every channel exposes its OWN state name. A unit channel re-exposes the shared weight under
    # that name rather than returning ``beta`` directly: the outer combine is the stored twisted
    # program renamed onto these names, and ``beta`` is one of its internal temps — exposing it
    # would collide the state with the temp and corrupt the merge.
    # The lift's per-element result and the carried state need DIFFERENT names: they become two
    # declarations in one emitted scope, and the inner one shadows the accumulator — the merge then
    # reads its own per-element value on both sides and folds nothing.
    name = f"{state}__blk"
    element = f"{name}__e"
    op, args = ("copy", (beta,)) if unit else ("multiply", (beta, carried))
    return Fold(
        operands=tuple(edges),
        lift=Lambda.closing((inner.name, *params), Body((Assign(name=element, op=op, args=args),)), (element,)),
        init=(fold.init[index],),
        combine=Lambda.componentwise((("add"),), (name,)),
    )


def block_twisted(fold: Fold, axis: Axis, width: int) -> tuple[Fold, tuple[Axis, ...]] | None:
    """The twisted ``fold`` re-derived over blocks, or ``None`` when it is not blockable.

    Returns the blocked fold and the axes the kernel must now hold: the outer trip count, and one
    narrowed binder per inner fold. The result is one outer fold carrying the SAME
    ``(init, combine)`` over inner folds that run the base monoid.
    """
    read = _pivot_and_rescales(fold)
    if read is None:
        return None
    pivot_op, _, betas = read
    view = fold.as_reduction()
    outer, blk = _split_axis(axis, width)
    binders = [_block_binder(axis, outer, blk, tag) for tag in ("p", *(f"c{i}" for i in range(1, len(view.states))))]

    lift, states = fold.lift, view.states
    bound = tuple(param for param, _, _ in fold.bindings)
    score = lift.results[0]

    # 1. The block pivot — the fold's own per-element pivot value under the pivot ⊕, over its own
    #    binder. Only the score's cone and the edges it reads: carrying the whole lift would bind
    #    the streamed value here too, and one slab bound in two sibling folds is one name declared
    #    twice in the emitted scope.
    m_blk = f"{states[0]}__blk"
    pivot_axis, pivot_sigma = binders[0]
    pivot_edges = _reindexed(fold, pivot_sigma)
    cone = lift.cone(score)
    kept = tuple(param for param in bound if param in cone.params)
    pivot = Fold(
        operands=tuple(pivot_edges[index] for index, (param, _, _) in enumerate(fold.bindings) if param in kept),
        lift=Lambda.closing((pivot_axis.name, *kept), cone.body, (score,)).rename({score: f"{m_blk}__e"}),
        init=(fold.init[0],),
        combine=Lambda.componentwise((pivot_op,), (m_blk,)),
    )

    # 2. The weight cone: the combine's own β, evaluated at (pivot := the block pivot, other := this
    #    element's score). One zero-axis term, shared by every channel — so the P tile is computed
    #    once per block and read by both the denominator and the expectation.
    if len(set(betas.values())) != 1:
        return None  # channels rescaled by different factors: not one shared weight
    beta = next(iter(betas.values()))

    # 3. One inner fold per channel, each over its OWN binder, with its own weight instance —
    #    separate loops over the same block, so each needs a coordinate of its own.
    channels = []
    for index in range(1, len(states)):
        chan_axis, chan_sigma = binders[index]
        chan_edges = _reindexed(fold, chan_sigma)
        weight = _beta_cone(fold, beta, pivot, m_blk, chan_edges, bound)
        channels.append(_channel(fold, index, weight, beta, chan_edges, chan_axis))

    inner_folds = (pivot, *channels)
    return _over_blocks(fold, outer, inner_folds), (outer, *(binder for binder, _ in binders))


# ---- the tree walk ---------------------------------------------------------------------------- #


def _blockable(fold: Fold, axis: Axis | None) -> bool:
    """Whether one fold's reduce axis can still be blocked.

    A scan is excluded — blocking changes which prefixes its observer sees. So is an axis a
    structural rewrite has already consumed: a cross-CTA partition, and a block itself.
    """
    if axis is None or fold.combine is None or fold.observe is not None:
        return False
    window = axis.window
    if window is not None and (window.block or window.partition):
        return False
    return axis.extent.is_static and axis.extent.as_static() > 1


def block(fold: Fold, axis: Axis, width: int) -> tuple[Fold, tuple[Axis, ...]] | None:
    """``fold`` blocked at ``width``, whichever carrier it is — or ``None``."""
    return block_planar(fold, axis, width) or block_twisted(fold, axis, width)


def blockable_streams(root: Fold, axes: tuple) -> tuple[tuple[Fold, Axis], ...]:
    """Every root-most reduce stream this tree can block, with the axis each one folds."""
    table = {axis.name: axis for axis in axes}
    found: list[tuple[Fold, Axis]] = []
    seen: set[int] = set()

    def descend(term: Fold) -> None:
        if id(term) in seen:
            return
        seen.add(id(term))
        axis = table.get(term.axis) if term.axis is not None else None
        if _blockable(term, axis) and block(term, axis, axis.extent.as_static()) is not None:
            found.append((term, axis))
            return
        for edge in term.operands:
            descend(edge)

    descend(root)
    return tuple(found)


def block_tree(root: Fold, axes: tuple, widths: dict[int, int]) -> tuple[Fold, tuple[Axis, ...]]:
    """The tree with each stream in ``widths`` blocked at its width, and the axes that follow.

    ``widths`` is keyed by the ``id`` of the pre-block fold, which is what
    :func:`blockable_streams` hands back — a route would have to be re-resolved against a tree that
    is changing underneath it.
    """
    table = {axis.name: axis for axis in axes}
    installed: list[Axis] = []
    done: dict[int, Fold] = {}

    def descend(term: Fold) -> Fold:
        if id(term) in done:
            return done[id(term)]
        axis = table.get(term.axis) if term.axis is not None else None
        got = block(term, axis, widths[id(term)]) if id(term) in widths and axis is not None else None
        if got is not None:
            out, fresh = got
            installed.extend(fresh)
        else:
            edges = tuple(descend(edge) for edge in term.operands)
            out = replace(term, operands=edges) if any(a is not b for a, b in zip(edges, term.operands, strict=True)) else term
        done[id(term)] = out
        return out

    blocked = descend(root)
    fresh = {axis.name for axis in installed}
    return blocked, (*(axis for axis in axes if axis.name not in fresh), *installed)
