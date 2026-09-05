"""Block a twisted carrier — the canonical form that gives its channels a semiring to live in.

Blocking splits one reduce axis into ``k_o × k_i`` and re-associates the fold over the two levels.
Only a TWISTED carrier is blocked, because it is the only carrier a block gives anything: a
contraction's block is already spelled by ``bk`` and a plain reduction's partition by ``REDUCE``,
so splitting either term would restate another family's decision as a shape.

A twisted fold is different in kind. Its ⊕ is a rescaling program, never a commutative op with an
identity, so ``as_contraction`` refuses at its first gate and NO site inside one can ever be
bilinear. Attention's ``P·V`` is not missing from the tree; it is spelled as a coefficient of that
⊕, which no schedule can put on a tensor core. Blocking SEPARATES the two monoids: the twisted ⊕
stays on the outer fold and the inner level runs the base monoid over a per-block contribution, so
the channel whose contribution is a product of two distinct cones reads as a contraction. That is
FlashAttention-2's shape, derived rather than recognized — everything the rewrite needs is read out
of the stored combine (``β``, the factor the merge puts on the incoming side), and the value it
multiplies is the fold's own lift result. No recipe is consulted and no operation family is matched;
Welford is declined by the same reading, because its M2 is a square.

The WIDTH appears nowhere in the term. The outer axis walks the stream's own extent in strides
(``Axis.step``) and each inner binder's extent is the width symbol, so the σ that reads the absolute
coordinate is plain ``k_o + k_i``. That is what makes the rewrite parameter-free and idempotent —
hence a NORMALIZATION, run from ``TileOp.__post_init__`` — and what lets every block form of a
kernel be the same kernel. The width is bound at materialization, from the ``TILE`` at the site
blocking created: a blocked site's inner axis IS its K, so the block is exactly that tile's mma
K-step and no schedule family spells it twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from emmy.compiler.dim import Dim, simplify_extent
from emmy.compiler.ir.axis import Axis, Window, block_width_var
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.pure import Fold, Lambda
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Load


def is_blocked(axes) -> bool:
    """Whether this kernel's axis table already carries a blocked stream (the re-firing receipt)."""
    return any(axis.window is not None and axis.window.block is not None for axis in axes)


# ---- the split ------------------------------------------------------------------------------- #


def sliced_edge(edge, sigma: Sigma, k_name: str, kslice=None, ksplit: Axis | None = None):
    """An operand edge σ-reindexed to absolute k for a split partition — the SAME rule on either
    edge. A MATERIALIZED edge rewrites its gmem index; a COMPUTED cone rewrites its per-cell BODY
    and every K-VARYING producer edge it composes (attention's per-cell score contraction — the
    slice's own k coordinate reaches gmem through that node, so leaving it unreindexed makes every
    partition recompute partition 0's scores). The cone's row-invariant prologue (the per-row
    statistic the K seam reads off the node boundary) spans the whole row and stays FULL-ROW in
    every partition, each recomputing it — the REDUNDANT-STATISTIC split. That redundancy is what
    the split trades for parallelism; whether it pays on a given shape is evidence's decision."""
    if isinstance(edge, Load):
        return replace(edge, index=tuple(sigma.apply(e) for e in edge.index))

    def images(name: str) -> tuple[str, ...]:
        # A coordinate a term takes as a value rides a trailing param, so a σ-reindex re-spells it
        # with the body. σ is not a rename: a split maps one coordinate onto an EXPRESSION over two
        # (slice and partition), so a param's image is the free names of that expression, in order.
        mapped = sigma.get(name)
        return (name,) if mapped is None else tuple(dict.fromkeys(mapped.free_vars()))

    ops = tuple(sliced_edge(e, sigma, k_name, kslice, ksplit) if k_name in e.free_axes else e for e in edge.operands)
    body = Body(tuple(s.substitute(sigma) for s in edge.lift.body))
    params = tuple(dict.fromkeys(name for param in edge.lift.params for name in images(param)))
    return replace(edge, operands=ops, lift=replace(edge.lift, params=params, body=body))


def _split_axis(axis: Axis) -> tuple[Axis, Expr]:
    """``(the outer block-walking axis, the symbolic block width)`` for one blocked reduce axis.

    The outer axis walks the SAME extent the stream always had, in strides of the width — so the
    trip count is ``ceil(extent / blk)`` without anyone computing it, the absolute coordinate is
    plain ``k_o + k_i``, and the width appears nowhere in the term. It lives on the two axes: as
    this one's ``step`` and as the binder's extent. That is what makes blocking parameter-free, and
    therefore a normalization rather than a decision.
    """
    blk = Var(block_width_var(axis.name))
    outer = Axis(f"{axis.name}_o", axis.extent, window=Window(parent=axis.source_axis or axis, block=blk.name, trip=True), step=blk)
    return outer, blk


def _block_binder(axis: Axis, outer: Axis, blk: Expr, tag: str) -> tuple[Axis, Sigma]:
    """One inner fold's OWN binder over the block, and the σ that reads the absolute coordinate.

    Each inner fold is a separate loop over the same block — the pivot's pass, then each channel's —
    so each needs a coordinate of its own. A term tree names one coordinate per name; sharing the
    binder makes ``lower`` place them as one loop, and the weight then reads a block pivot that has
    not finished accumulating.
    """
    inner = Axis(f"{axis.name}_{tag}", Dim(blk), window=Window(parent=axis.source_axis or axis, block=blk.name))
    sigma = Sigma({axis.name: BinaryExpr("+", Var(outer.name), Var(inner.name))})
    return inner, sigma


def _reindexed(fold: Fold, sigma: Sigma) -> tuple:
    """The fold's operand edges reading the block's ABSOLUTE coordinate.

    Only the operands move: the fold keeps its own binder over the narrowed axis, so nothing is
    captured. ``_sliced_edge`` is the cross-CTA split's reindexer, and it is the same job — σ maps
    one coordinate onto an EXPRESSION over several, so a param's image is that expression's free
    names rather than a rename.
    """
    return tuple(sliced_edge(edge, sigma, fold.axis) for edge in fold.operands)


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


def block_twisted(fold: Fold, axis: Axis) -> tuple[Fold, tuple[Axis, ...]] | None:
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
    outer, blk = _split_axis(axis)
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
    if window is not None and (window.block is not None or window.partition):
        return False
    return axis.extent.is_static and axis.extent.as_static() > 1


def block(fold: Fold, axis: Axis) -> tuple[Fold, tuple[Axis, ...]] | None:
    """``fold`` blocked, or ``None`` when blocking would give it nothing.

    Only a TWISTED carrier is blocked. A contraction's block is already spelled — ``bk`` says how
    many atom K-steps one inner step consumes and the materializer chunks K by it — and a plain
    reduction's partition is ``REDUCE``'s, with the cross-CTA split already factoring the axis.
    Splitting either term would restate another family's decision as a shape.
    """
    return block_twisted(fold, axis)


def block_tree(root: Fold, axes: tuple) -> tuple[Fold, tuple[Axis, ...]] | None:
    """Every root-most blockable stream of ``root``, blocked — or ``None`` when none is.

    Parameter-free and idempotent: the width never enters the term, and the ``Window`` every
    installed axis carries is the receipt that stops a second split. That is what makes this a
    NORMALIZATION rather than a decision, and why it can run from ``TileOp.__post_init__``.
    """
    table = {axis.name: axis for axis in axes}
    installed: list[Axis] = []
    done: dict[int, Fold] = {}

    def descend(term: Fold) -> Fold:
        if id(term) in done:
            return done[id(term)]
        axis = table.get(term.axis) if term.axis is not None else None
        got = block(term, axis) if _blockable(term, axis) else None
        if got is not None:
            out, fresh = got
            installed.extend(fresh)
        else:
            edges = tuple(descend(edge) for edge in term.operands)
            out = replace(term, operands=edges) if any(a is not b for a, b in zip(edges, term.operands, strict=True)) else term
        done[id(term)] = out
        return out

    blocked = descend(root)
    if not installed:
        return None
    fresh = {axis.name for axis in installed}
    return blocked, (*(axis for axis in axes if axis.name not in fresh), *installed)


def bind_widths(axes: tuple, widths: Mapping[str, int]) -> tuple[Axis, ...]:
    """The axis table with each block width bound — the ONE place a block stops being a symbol.

    Only the table moves, because only the table ever named the width: the outer axis's ``step``
    and the binder's extent. The term is untouched, which is why every block form of a kernel is
    the same kernel.
    """
    values = {name: Literal(width, "int") for name, width in widths.items()}

    def bound(axis: Axis) -> Axis:
        step = axis.step.substitute(values) if axis.step is not None else None
        return replace(axis, extent=Dim(simplify_extent(axis.extent.expr.substitute(values))), step=step)

    return tuple(bound(axis) for axis in axes)
