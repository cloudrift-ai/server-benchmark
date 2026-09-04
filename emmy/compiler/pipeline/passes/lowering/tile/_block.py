"""Blockify a twisted carrier — the rewrite that gives its channels a semiring to live in.

A twisted fold's ⊕ is a rescaling program, never a commutative op with an identity, so
``as_contraction`` refuses at its first gate and NO site inside one can ever be bilinear. Attention's
``P·V`` is not missing from the tree; it is spelled as a coefficient of that ⊕, which no schedule can
put on a tensor core.

Blocking separates the two monoids. The reduce axis splits into ``k_o × k_i``; the twisted ⊕ stays
on the outer fold, and the inner level runs the BASE monoid — plain ``maximum`` / ``add`` — over a
per-block contribution. The channel whose contribution is a product of two distinct cones then reads
as a contraction at the inner site, which is FlashAttention-2's shape derived rather than recognized.

Everything the rewrite needs is stored on the fold. The per-step rescale coefficient is read out of
the combine (``β``, the factor the merge puts on the incoming side), and the value it multiplies is
the fold's own lift result for that channel. No recipe is consulted, no operation family is matched:
a carrier whose combine does not read as pivot-plus-linear-channels is simply not blocked.
"""

from __future__ import annotations

import os
from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.pure import Fold, Lambda
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body
from emmy.compiler.pipeline.passes.lowering.tile._split import _sliced_edge

#: The block width's name. A plain ``Var`` for now — the schedule resolves it BY NAME, and the
#: kernel's axis table is where its value is declared, since that table is already the one registry
#: of "what extent does this coordinate have". Hand-set through ``EMMY_BLOCK`` while the agreement
#: that would derive it (the score tile's N against the value fold's K) does not exist yet.
BLK = "blk"


def _width() -> int:
    return int(os.environ.get("EMMY_BLOCK", "64"))


def _monoid(state: str, op: str) -> Lambda:
    """The componentwise ⊕ over one carried state — what an inner fold folds through."""
    other = f"{state}__o"
    return Lambda(params=(state, other), body=Body((Assign(name=state, op=op, args=(state, other)),)), results=(state,))


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


def _split_axis(axis: Axis) -> tuple[Axis, Expr]:
    """``(the outer trip axis, the block width)`` for one reduce axis blocked by :data:`BLK`.

    The outer count is ``ceil(extent / blk)``, composite in both unknowns, so one tree serves a
    static extent and a symbolic one.
    """
    blk = Literal(_width(), "int") if os.environ.get("EMMY_BLOCK_VAR") != "1" else Var(BLK)
    extent = axis.extent.expr
    trips = BinaryExpr("/", BinaryExpr("+", extent, BinaryExpr("-", blk, Literal(1, "int"))), blk)
    outer = Axis(f"{axis.name}_o", Dim(trips), window=Window(parent=axis.source_axis or axis))
    return outer, blk


def _block_binder(axis: Axis, outer: Axis, blk: Expr, tag: str) -> tuple[Axis, Sigma]:
    """One inner fold's OWN binder over the block, and the σ that reads the absolute coordinate.

    Each inner fold is a separate loop over the same block — the pivot's pass, then each channel's —
    so each needs a coordinate of its own. A term tree names one coordinate per name; sharing the
    binder makes ``lower`` place them as one loop, and the weight then reads a block pivot that has
    not finished accumulating.
    """
    inner = Axis(f"{axis.name}_{tag}", Dim(blk), window=Window(parent=axis.source_axis or axis))
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
        combine=_monoid(name, "add"),
    )


def block(fold: Fold, axes: dict) -> tuple[Fold, tuple[Axis, ...]] | None:
    """The twisted ``fold`` re-derived over blocks, or ``None`` when it is not blockable.

    Returns the blocked fold and the axes the kernel must now hold: the outer trip count, and the
    fold's own axis narrowed to one block. The result is one outer fold carrying the SAME
    ``(init, combine)`` — a block carrier has the same shape as a singleton one, and the monoid does
    not care which it merges — over inner folds that run the base monoid. Associativity is what
    makes the two equal.
    """
    view = fold.as_reduction()
    if view is None or view.ops is not None or fold.observe is not None or fold.axis is None:
        return None  # componentwise (already plain), or a scan whose prefixes blocking would change
    read = _pivot_and_rescales(fold)
    if read is None:
        return None
    pivot_op, advance, betas = read
    axis = axes[fold.axis]
    if not axis.extent.is_static:
        return None  # prototype: static extents only
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
        combine=_monoid(m_blk, pivot_op),
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

    # 4. The outer fold: same init, same combine, block contributions as its operands.
    inner_folds = (pivot, *channels)
    # The lift RETURNS the block contributions and the carrier keeps the original state names — the
    # same shape the unblocked fold has (results ``(v1, in7, acc3__one)`` against states
    # ``(acc1, acc5__sum, acc3)``). A copy onto the state name instead declares the state inside the
    # step and shadows the carrier, so the merge folds each block into itself.
    contributed = tuple(edge.exposes[0] for edge in inner_folds)
    blocked = Fold(
        operands=inner_folds,
        lift=Lambda.closing((outer.name, *contributed), Body(()), contributed),
        init=fold.init,
        combine=fold.combine,
    )
    return blocked, (outer, pivot_axis, *(axis for axis, _ in binders[1:]))


def block_tree(root: Fold, axes: tuple) -> tuple[Fold, tuple[Axis, ...]] | None:
    """Every blockable twisted carrier in the tree, blocked — with the axes the kernel must hold.

    ``None`` when nothing blocks, so the pass declines rather than rebuilding an equal tree. The
    receipt against re-firing is the IR itself: a blocked outer axis's extent READS the block width,
    and no other axis does.
    """
    table = {axis.name: axis for axis in axes}
    installed: list[Axis] = []
    done: dict[int, Fold] = {}

    def descend(term: Fold) -> Fold:
        if id(term) in done:
            return done[id(term)]
        out = term
        blockable = term.axis is not None and BLK not in table[term.axis].extent.expr.free_vars()
        got = block(term, table) if blockable else None
        if got is not None:
            out, fresh = got
            installed.extend(fresh)
        else:
            edges = tuple(descend(edge) for edge in term.operands)
            if any(a is not b for a, b in zip(edges, term.operands, strict=True)):
                out = replace(term, operands=edges)
        done[id(term)] = out
        return out

    blocked = descend(root)
    if not installed:
        return None
    installed.append(Axis(BLK, Dim(_width())))
    fresh = {axis.name for axis in installed}
    return blocked, (*(axis for axis in axes if axis.name not in fresh), *installed)
