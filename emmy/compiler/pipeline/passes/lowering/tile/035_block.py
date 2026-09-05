"""Block a twisted carrier — split its stream into blocks so a channel reads as a contraction.

A NORMALIZATION, not a decision: the rewrite is parameter-free (the width is the form's, read off
the stream's own extent) and idempotent (every axis it installs carries the receipt), so it opens
no fork, adds no schedule family, and leaves one kernel identity per term. Only a twisted carrier is
blocked, and the module docstring below the pass entry states why.

It is a PASS rather than ``TileOp.__post_init__``, and it runs AFTER ``030_cut``, for one reason: a
piece the cut mints is a different carrier from the one it came out of. Attention cut at its value
channel leaves a statistics piece whose channels are sums of the weight — nothing there becomes
bilinear, so nothing there should be blocked, and blocking ahead of the fork handed that piece a
block it paid a second pass over the stream for (an 18.8 ms piece on a 512-key head where the
fused kernel is 128 µs). Running here, each branch of the fork blocks what its own term deserves.
A block is still a working set inside ONE kernel, so no seam evaluated over a block coordinate is
cuttable either (``_cut.cuttable_seams``).
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Node
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.pure import Fold, Lambda
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body
from emmy.compiler.ir.tile import TileOp, blockify
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.tile._split import sliced_edge

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node, ctx=None) -> TileOp:
    del match, ctx
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped or tile.schedule is not None:
        raise RuleSkipped("TileOp already scheduled / nothing to block")
    blocked = block_tree(tile.op, tile.axes)
    if blocked is None:
        blocks = blockify(tile)
        if blocks == tile.blocks:
            raise RuleSkipped("symbolic blocks already installed")
        return replace(tile, blocks=blocks)
    op, axes = blocked
    # Structural blocking creates new Fold sites; declare their symbolic blocks on the new tree.
    result = replace(tile, op=op, axes=axes, blocks=())
    return replace(result, blocks=blockify(result))


MAX_BLOCK = 64


def block_width(extent: int) -> int | None:
    """The block ``extent`` is cut into — the largest power-of-two fraction of it within
    :data:`MAX_BLOCK` — or ``None`` when it is not cut at all.

    A stream no wider than one block is NOT cut: one block is the whole stream, which is the
    unblocked kernel, and reaching it by leaving the term alone is better than by a second spelling
    of it. Neither is a stream a whole number of blocks does not fit — the alternative is a masked
    tail on a stream whose channels each carry their own pass over it, and the tail's identity
    would have to be threaded through the pivot and every channel separately.

    The width is the FORM's, not a decision: the row that schedules a blocked site chunks the block
    with its own ``bk`` exactly as it chunks any other K, so nothing here is a fork and nothing
    downstream sizes itself against a symbol.
    """
    width = extent
    while width > MAX_BLOCK:
        if width % 2:
            return None
        width //= 2
    return width if width < extent else None


def _split_axis(axis: Axis, width: int) -> Axis:
    """The outer block-walking axis of one blocked reduce axis.

    It walks the SAME extent the stream always had, in strides of ``width`` — so the trip count is
    ``extent / width`` without anyone computing it, the absolute coordinate is plain ``k_o + k_i``,
    and no width ever enters the term's index arithmetic. ``Fold.lower`` renders a strided axis as
    the ``StridedLoop`` that already says exactly this.
    """
    parent = axis.source_axis or axis
    return Axis(f"{axis.name}_o", axis.extent, window=Window(parent=parent, block=True), step=Literal(width, "int"))


def _block_binder(axis: Axis, outer: Axis, width: int, tag: str) -> tuple[Axis, Sigma]:
    """One inner fold's OWN binder over the block, and the σ that reads the absolute coordinate.

    The pivot's pass and the channels' are separate loops over the same block, so each level needs
    a coordinate of its own: sharing one binder with the pivot makes ``lower`` place them as one
    loop, and the weight then reads a block pivot that has not finished accumulating.
    """
    inner = Axis(f"{axis.name}_{tag}", Dim(width), window=Window(parent=axis.source_axis or axis, block=True))
    sigma = Sigma({axis.name: BinaryExpr("+", Var(outer.name), Var(inner.name))})
    return inner, sigma


def _reads(fold: Fold, operands: tuple, reads) -> tuple[tuple[str, ...], tuple]:
    """The params and edges of every operand ``reads`` touches, CLOSED over the operand.

    ``Fold.bindings`` binds one param per RESULT COMPONENT, so an edge exposing several binds
    several params and the positional binding only lines up when a kept edge keeps all of them.
    ``operands`` is the σ-reindexed parallel of ``fold.operands``.
    """
    bound = tuple(param for param, _, _ in fold.bindings)
    at = tuple(index for index, edge in enumerate(fold.operands) for _ in edge.exposes)
    keep = dict.fromkeys(index for param, index in zip(bound, at, strict=True) if param in reads)
    return tuple(param for param, index in zip(bound, at, strict=True) if index in keep), tuple(operands[i] for i in keep)


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


def _beta_cone(fold: Fold, beta: str, pivot: Fold, m_blk: str, operands: tuple) -> Fold:
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
    kept, edges = _reads(fold, operands, cone.params)
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
    const = fold.lift.body.definitions.get(carried)
    unit = const is not None and not isinstance(const, Assign) and getattr(const, "value", None) == 1.0
    # The channel's own value cone, over the REINDEXED edges it reads: the fold's originals still
    # read the un-split axis. A unit channel has no value to read at all.
    value = Lambda.closing((), Body(()), ()) if unit else fold.lift.cone(carried)
    kept, value_edges = _reads(fold, operands, value.params)
    edges, params = (weight, *value_edges), (beta, *kept)
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
        operands=edges,
        lift=Lambda.closing((inner.name, *params), Body((*value.body, Assign(name=element, op=op, args=args))), (element,)),
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
    width = block_width(axis.extent.as_static())
    if width is None:
        return None
    view = fold.as_reduction()
    outer = _split_axis(axis, width)
    pivot_axis, pivot_sigma = _block_binder(axis, outer, width, "p")
    chan_axis, chan_sigma = _block_binder(axis, outer, width, "c")

    lift, states = fold.lift, view.states
    score = lift.results[0]

    # 1. The block pivot — the fold's own per-element pivot value under the pivot ⊕, over its own
    #    binder. Only the score's cone and the edges it reads: carrying the whole lift would bind
    #    the streamed value here too, and one slab bound in two sibling folds is one name declared
    #    twice in the emitted scope.
    m_blk = f"{states[0]}__blk"
    pivot_edges = _reindexed(fold, pivot_sigma)
    cone = lift.cone(score)
    kept, pivot_operands = _reads(fold, pivot_edges, cone.params)
    pivot = Fold(
        operands=pivot_operands,
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

    # 3. The channels: every one over the SAME binder, reading ONE weight instance. They are
    #    independent accumulations over the same block, so they belong in one loop and the weight
    #    is computed once for all of them — two passes over a block, which is the fewest a pivot
    #    the weights read can be reached in. Only the PIVOT needs a binder of its own.
    chan_edges = _reindexed(fold, chan_sigma)
    weight = _beta_cone(fold, beta, pivot, m_blk, chan_edges)
    channels = tuple(_channel(fold, index, weight, beta, chan_edges, chan_axis) for index in range(1, len(states)))

    # A block costs a second pass over the stream, and pays for it only by making a channel
    # BILINEAR. When none becomes a contraction — a plain online softmax, whose channels are sums
    # of the weight itself — there is nothing for a tensor core to take and the carrier keeps the
    # partitions it already had. That is the same reading as the refusals above, applied to the
    # result instead of the algebra.
    if not any(channel.as_contraction() is not None for channel in channels):
        return None
    return _over_blocks(fold, outer, (pivot, *channels)), (outer, pivot_axis, chan_axis)


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
        got = block_twisted(term, axis) if _blockable(term, axis) else None
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
