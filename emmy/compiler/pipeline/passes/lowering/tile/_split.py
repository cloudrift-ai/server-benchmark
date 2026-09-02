"""Offer and realize the cross-CTA reduce split as a STRUCTURAL graph rewrite.

The split changes the kernel SET, exactly like a ``PLACE`` cut: a kernel that splits does not run,
so its cost is the Σ over the pieces it produces (``policy/greedy._resolved_price``), which is why
the split is the ``030_cut`` fork and not a schedule row. It runs BEFORE scheduling — the
rewrite consumes only the stored :class:`Fold` algebra, never a schedule decision — and each piece
is a fresh unmapped :class:`TileOp` that re-enters the pass scan and decides its own row at
``040_schedule`` like any newly lifted tree.

Each partial evaluates the same ``Fold(init, combine)`` over a contiguous axis slice and writes its
complete state tuple. The deferred finalize identity-lifts those tuples through the same ``init``
and ``combine``, then applies the original projection. This is the common path for additive and
exp-family monoids; the split does not recognize carrier families. A CONTRACTION slices through
``Fold.contraction`` over the σ-reindexed operand edges — the cone's row-invariant prologue stays
FULL-ROW in every partition (the redundant-statistic split); any other fold slices through the
generic ``Fold.rewrite``.

The atomic arm is the generic exception: it is legal only for a single additive state component
whose projection distributes over addition. Otherwise the deferred f32 workspace preserves the
full state until the finalize combines and projects it.

Every piece is a fresh unmapped :class:`TileOp`. A graph splice restarts the lowering pass scan:
scheduling offers each piece its own row. An axis :class:`Window` records that the partition has
already been consumed and prevents recursive splitting — the receipt is the IR itself, no flag.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BF16, F16, F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.algebra import component_ops
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule import Reduce, Work
from emmy.compiler.ir.schedule.catalog import splitk_moves
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Load, Write
from emmy.compiler.ir.stmt.passes import projection_distributes
from emmy.compiler.ir.tile import (
    OutputSpec,
    Placement,
    TileOp,
    extract_output_specs,
    lower_with_output_specs,
)
from emmy.compiler.ir.tile.ir import apply_output_specs
from emmy.compiler.ir.tile.ops import Sched, carries_partition, head, projection_regions, projection_tail
from emmy.compiler.pipeline import Match
from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.knob import axis_of, consume_kernel_row
from emmy.compiler.pipeline.search.space import REDUCE, WORK

logger = logging.getLogger(__name__)

_SPLIT = "_ksplit"  # the cross-CTA split grid axis


# ---- what a kernel / an axis can carry -------------------------------------------------------- #


def splitk_width(k_axis: Axis, width: int) -> str | None:
    """A cross-CTA split needs a STATIC reduce axis its width divides evenly — the σ-reindex
    reconstructs an absolute k from ``ksplit·(K/w) + kslice``, which is a bijection only when the
    extent is known and ``w`` divides it. Total over a symbolic axis rather than raising out of
    ``as_static``, so the catalog drops the width and a pin reports the reason."""
    if not k_axis.extent.is_static:
        return f"cross-CTA split of the symbolic reduce axis {k_axis.name!r} is not built"
    big_k = k_axis.extent.as_static()
    if big_k % width == 0:
        return None
    return f"split-K width {width} does not divide K={big_k}; pick a dividing split width."


def _direct_atomic_output(outputs) -> str | None:
    """Whether direct cross-CTA partials avoid another low-precision rounding step.

    F16/BF16 destinations round once per CTA; the deferred finalize instead combines f32
    carrier state and rounds once at the output boundary."""
    lowp = sorted({str(t.dtype) for t in outputs.values() if t.dtype in (F16, BF16)})
    if not lowp:
        return None
    return (
        f"direct atomic REDUCE writes each partial into {'/'.join(lowp)} output storage; "
        "use the deferred f32 workspace finalize (REDUCE=g<n>k) so the output rounds once"
    )


def atomic_finalize(node: Fold, tail, outputs) -> str | None:
    """Whether the cross-CTA split may take its DIRECT ``atomicAdd`` arm — every condition, once,
    beside the move it filters. The deferred workspace finalize (``REDUCE=g<n>k``) carries any
    carrier and any projection, so each refusal names it as the alternative. Four things must
    hold, and they are stated together because the offer, the pin, and the realization all have to
    agree — a pin that reached the rewrite past a refusal the offer applied would crash the
    emitter instead of refusing:

    - ONE state component. ``atomicAdd`` folds a scalar; a twisted carrier's
      ``(maximum, denominator, …)`` tuple has no atomic instruction at all.
    - An ADDITIVE ⊕. The emitted instruction IS ``atomicAdd`` — a ``max`` carrier's partials
      would be SUMMED, silently wrong (the deferred finalize folds any monoid).
    - An output the partials can round into once per CTA (:func:`_direct_atomic_output`).
    - A projection that DISTRIBUTES over the add. The atomic arm applies the epilogue per
      partition, before the combine, so anything but a linear-homogeneous map mis-scales each
      CTA's contribution.

    Storage is asked BEFORE the projection: a narrowing store spells its rounding as a conversion
    in that same epilogue (``loop/lifting/090_spell_store_rounding``), which no linear-homogeneous
    reading admits — so both refuse a low-precision output and the storage reason is the one that
    names why.

    ``tail`` differs between the two askers: the OFFER passes the kernel's whole projection tail,
    the realization the MIMO-selected region's. The offer's read is the superset, so the only
    possible divergence is OVER-refusal at the offer (a sibling region's non-distributive stmt
    refusing an atomic the owned region could carry) — safe, since a withheld atomic row leaves
    the deferred finalize, never a crash past an offer the realizer cannot honor."""
    states = tuple(node.combine.results)
    if len(states) != 1:
        return (
            f"atomic REDUCE folds ONE additive state component; this carrier has {len(states)} "
            f"({', '.join(states)}) — use the deferred f32 workspace finalize (REDUCE=g<n>k)"
        )
    ops = component_ops(node.combine)
    if ops is None or ops[0].name != "add":
        plus = "a twisted combine" if ops is None else f"⊕ = {ops[0].name}"
        return f"atomic REDUCE emits atomicAdd, which folds only an ADDITIVE carrier; this one has {plus} — use REDUCE=g<n>k"
    storage = _direct_atomic_output(outputs)
    if storage is not None:
        return storage
    if tail and not projection_distributes(tuple(tail), states):
        return (
            "atomic REDUCE applies the projection epilogue per partition, so it must distribute "
            "over the add; this one does not (a fused bias / activation) — use the deferred "
            "workspace finalize (REDUCE=g<n>k), which projects once after the combine"
        )
    return None


def _enforce(reason: str | None) -> None:
    """Raise a refusal a PIN ran into — the offer's catalog arm drops instead."""
    if reason is not None:
        raise ValueError(reason)


def _projection_refusal(tile: TileOp, node) -> str | None:
    """Why the kernel's projection cannot survive a split of ``node`` (``None`` when it can) — the
    MIMO decomposition the realizer performs, asked at the OFFER so an unrealizable split is never
    offered: an independent-projection kernel must partition into output-owning regions and the
    split fold must own one of them (a projection reading SEVERAL roots into one output has no
    piece to hand the epilogue to). The residence guard leads: a head fold the realization cannot
    STRIP from the projection — neither the kernel's own node, an operand edge, nor a top-level
    projection member (``head``'s sweep case: a fold reading the boundary store's sweep axis lands
    inside the sweep ``Loop`` ``apply_output_specs`` wraps) — would leave the epilogue re-running
    the whole reduction and shadowing the workspace states, so the split declines there."""
    op = tile.op
    if (
        op is not node
        and all(edge is not node for edge in getattr(op, "operands", ()))
        and not any(stmt is node for stmt in projection_tail(tile))
    ):
        return "the head fold is nested inside the projection's sweep loop; the split cannot strip it"
    if not isinstance(op, Fold) or op.axis is not None or len(op.operands) < 2:
        return None
    try:
        regions = projection_regions(op, tile.output_specs)
    except ValueError as e:
        return str(e)
    if not any(fold is node for fold, _, _ in regions):
        return "the split fold does not own an independent projection region"
    return None


# ---- the offer: the unsplit tree beside every split the head fold admits ---------------------- #


def split_pending(tile: TileOp) -> bool:
    """Whether this kernel still has a cross-CTA split decision for ``030_cut`` to consume."""
    node = head(tile.op)
    return (
        node is not None
        and node.axis is not None
        and node.combine is not None
        and not node.observed
        and not carries_partition(tile.op)
        and not tile.split_consumed
    )


def split_forks(match: Match, root: Node, *, unsplit_tile: TileOp | None = None) -> list[DeferredFork] | None:
    """The split fork for ``root``'s kernel — the unsplit tree first, then one STRUCTURAL option
    per :func:`splitk_moves` member the head fold admits — or ``None`` when there is nothing to
    decide (no reduce fold, or the kernel is itself a piece of a realized split: the sliced axis's
    partition ``Window`` is the receipt, so the pieces re-entering the cut fixpoint and skip here;
    an ambient pin can never split twice). ``040_schedule`` then consumes the same receipt when it
    strips the pin's ``g`` half before composing each piece's own assignment.

    A ``REDUCE`` pin is authoritative over its cross-CTA ``g<n>[a|k]`` half and ONLY that half:
    the rest of the value (``coop`` / ``r<n>``) is the pieces' own schedule, which the walk reads
    off the same pin minus the consumed stage. A pin naming a split the head fold cannot carry
    raises the recorded refusal (``REDUCE`` has no choice of tier, so there is no drop layer);
    a pin with no ``g`` half decides UNSPLIT, exactly as a spelled row with no ``g`` half does."""
    tile: TileOp = root.op
    if not split_pending(tile):
        return None
    node = head(tile.op)
    assert node is not None and node.axis is not None
    key = Sched(tile).key("REDUCE", node) or "REDUCE"
    unsplit = DeferredFork(lambda: replace(unsplit_tile or tile, split_consumed=True), {key: ""})
    element = axis_of(key)
    pin = REDUCE.narrow_at(element) if element else REDUCE.raw()
    tail = projection_tail(tile)
    if pin is not None:
        plan = Reduce.parse(pin, Work.parse(WORK.raw()))
        if not plan.needs_split:
            return [unsplit]
        _enforce(splitk_width(node.axis, plan.cta))
        _enforce(_projection_refusal(tile, node))
        if plan.finalize == "atomic":
            _enforce(atomic_finalize(node, tail, tile.outputs))
        return [_split_fork(match, root, key, plan.cta, plan.finalize)]
    if (why := _projection_refusal(tile, node)) is not None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("no split offered: %s", why)
        return [unsplit]
    options: list[DeferredFork] = [unsplit]
    atomic_why = atomic_finalize(node, tail, tile.outputs)
    for plan in splitk_moves():
        why = splitk_width(node.axis, plan.cta) or (atomic_why if plan.finalize == "atomic" else None)
        if why is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("split g%d%s not offered: %s", plan.cta, plan.finalize[0], why)
            continue
        options.append(_split_fork(match, root, key, plan.cta, plan.finalize))
    return options


def _split_fork(match: Match, root: Node, key: str, cta: int, finalize: str) -> DeferredFork:
    spelling = Reduce.of(cta=cta, finalize=finalize).spell()
    return DeferredFork(lambda: realize_split(match, root, cta, finalize), {key: spelling}, structural=True)


# ---- slicing the head fold -------------------------------------------------------------------- #


def _slice_fold(fold: Fold, b: int) -> Fold:
    """The same monoid Fold over one CTA's absolute contiguous slice — the generic
    (non-contraction) slicer: the whole fold rides ``Fold.rewrite`` under the σ-offset."""
    axis = fold.axis
    assert axis is not None
    offset = BinaryExpr("+", Var(axis.name), BinaryExpr("*", Var(_SPLIT), Literal(b, "int")))
    sigma = Sigma({axis.name: offset})
    sliced_axis = replace(axis, extent=Dim(b), window=Window(parent=axis.source_axis or axis, partition=True))
    # RE-DERIVED over a narrower axis, not renamed and not substituted-through: the fold keeps its
    # own binder (same name, sliced extent) while its operands' coordinates take the σ-offset. A
    # blanket σ would be refused as capture — this fold BINDS the name σ maps — and rightly so;
    # what changes here is the axis itself, which only the caller can say.
    operands = tuple(_sliced_edge(edge, sigma, axis.name, sliced_axis) for edge in fold.operands)
    body = Body(tuple(stmt.substitute(sigma) for stmt in fold.lift.body))
    return replace(fold, axes=(sliced_axis,), operands=operands, lift=replace(fold.lift, body=body))


def _factor_k(k_axis: Axis, w: int) -> tuple[Axis, Axis, Sigma]:
    """Factor a STATIC contraction axis into ``ksplit × kslice``. ``ksplit`` (extent ``w``, name
    ``_<k>_ks``) becomes the partial's lead grid axis, parallelized across CTAs and combined in
    the finalize; ``kslice`` (extent ``K/w``, the ORIGINAL name) is the sliced contraction's. The
    ``sigma`` maps the original ``k`` to ``ksplit·(K/w) + kslice`` so the operand loads
    reconstruct the absolute index; distinct names are what avoid a double-reduce. The slice
    carries its parentage: a cross-CTA split is CONSUMED by the rewrite that realizes it, and an
    axis that is already a partition window is one nothing may partition again."""
    b = k_axis.extent.as_static() // w
    ksplit = Axis(name=f"_{k_axis.name}_ks", extent=Dim(w))
    kslice = replace(k_axis, extent=Dim(b), window=Window(parent=k_axis.source_axis or k_axis, partition=True))
    sigma = Sigma({k_axis.name: BinaryExpr("+", BinaryExpr("*", Var(ksplit.name), Literal(b, "int")), Var(k_axis.name))})
    return ksplit, kslice, sigma


def _sliced_edge(edge, sigma: Sigma, k_name: str, kslice=None):
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
        # A term BINDS its coordinates — as its own axes and, positionally, as the lift's param
        # prefix — so a σ-reindex renames all three in lockstep. σ is not a rename: a split maps
        # one coordinate onto an EXPRESSION over two (slice and partition), so a param's image is
        # the free names of that expression, in order.
        mapped = sigma.get(name)
        return (name,) if mapped is None else tuple(dict.fromkeys(mapped.free_vars()))

    ops = tuple(e.substitute(sigma) if k_name in e.index_space else e for e in edge.operands)
    body = Body(tuple(s.substitute(sigma) for s in edge.lift.body))
    params = tuple(dict.fromkeys(name for param in edge.lift.params for name in images(param)))
    axes = tuple(dict.fromkeys((kslice if axis.name == k_name else axis) for axis in edge.axes if axis.name != k_name or kslice))
    axes = tuple(axis for axis in axes if axis.name in params)
    return replace(edge, axes=axes, operands=ops, lift=replace(edge.lift, params=params, body=body))


def _sliced_contraction(node: Fold, w: int) -> tuple[Axis, Fold]:
    """``(ksplit, sliced)`` for a contraction head: the SAME bilinear node a non-split matmul
    builds, over ``kslice`` with operands σ-reindexed to absolute k, threading the node's OWN
    semiring (the reassociation ``fold_k = fold_{ksplit} ∘ fold_{kslice}`` is licensed by that
    ⊕-monoid's associativity). ``Fold.contraction`` regenerates the componentwise ⊕ over the same
    accumulator names, so the finalize folds the workspace states through the same monoid."""
    ksplit, kslice, sigma = _factor_k(node.axis, w)
    # Rebuilt DIRECTLY over the σ-reindexed operands, in stored order: the slice is the same term
    # with a narrower axis, so its lift, monoid and seeds are the node's own — there is nothing for
    # a former to re-derive, and no role to re-name.
    operands = tuple(_sliced_edge(edge, sigma, node.axis.name, kslice) for edge in node.operands)
    sliced = replace(node, axes=(kslice,), operands=operands)
    return ksplit, sliced


# ---- the piece / fragment builders ------------------------------------------------------------ #


def _boundary(stmts) -> tuple[tuple, tuple]:
    """Split a projection into a pure body and output specifications."""
    split = extract_output_specs(tuple(stmts))
    if split is None:
        return tuple(stmts), ()
    return split


def _cell_index(stmts, free) -> tuple:
    """The output-cell index the original kernel writes (the projection ``Write``'s index,
    or — for a bare carrier whose grid-cell store is glue — the free-axis vars)."""
    for s in stmts:
        if isinstance(s, Write):
            return s.index
    return tuple(Var(ax.name) for ax in free)


def _frag(match: Match, root: Node) -> Graph:
    """A fragment seeded with the split node's inputs — the graph a piece is stamped against (its
    structural features fold in its operands' dtypes, which need the buffers)."""
    frag = Graph()
    for inp in root.inputs:
        frag.add_node(op=InputOp(), inputs=[], output=match.graph.buffer(inp), node_id=inp)
    return frag


def _piece_inputs(root: Node, body, *first: str) -> list[str]:
    """Return fragment buffers followed by external inputs actually read by a piece."""
    if isinstance(body, TileOp):
        body = lower_with_output_specs(body.op, body.output_specs)
    elif isinstance(body, Fold):
        body = body.lower()
    reads = {load.input for load in Body.coerce(body).loads}
    return [*first, *(inp for inp in root.inputs if inp in reads)]


def _add_output_piece(match: Match, frag: Graph, root: Node, piece: TileOp, inputs: list[str]) -> Graph:
    """Add a fresh piece with its owned output ports and arrange their splice identities."""
    buffers = root.buffer_names()
    renamed = {name: f"{name}__split" for name in buffers}
    piece = replace(
        piece,
        output_specs=tuple(
            replace(spec, write=replace(spec.write, output=renamed.get(spec.write.output, spec.write.output)))
            for spec in piece.output_specs
        ),
    )
    tensors = (
        replace(root.outputs[0], name=buffers[0]),
        *(replace(tensor, name=renamed[name]) for name, tensor in zip(buffers[1:], root.outputs[1:], strict=True)),
    )
    frag.add_node(op=piece, inputs=inputs, outputs=tensors, node_id=renamed[buffers[0]])
    frag.outputs.extend(renamed.values())
    output = dict(match.output) if isinstance(match.output, dict) else {}
    output.update(renamed)
    match.output = output
    return frag


def _one(match: Match, frag: Graph, root: Node, piece: TileOp) -> Graph:
    """The ATOMIC arm's one-kernel fragment. It replaces the split kernel with ONE kernel, but it
    is a SPLICE, never an op rebind: a rebind is how the engine says "the same kernel, decided
    further", so it merges the replaced op's knobs forward and does not restart the pass scan. The
    atomic partial is a different kernel — its own placement, its own body — and it has to reach
    scheduling carrying nothing of the kernel it replaced."""
    return _add_output_piece(match, frag, root, piece, list(root.inputs))


def _wrap(body: Body, operands: tuple) -> Fold:
    """A zero-axis term over ``body``, exposing its last definition — what a projection returns."""
    bound = tuple(name for edge in operands for name in edge.exposes)
    results = next((stmt.defines()[-1:] for stmt in reversed(tuple(body)) if stmt.defines()), ())
    return Fold(operands=operands, lift=Lambda.closing(bound, body, results))


def _piece(body, free, *, output_specs: tuple = ()) -> TileOp:
    """One fresh unscheduled Tile kernel preserving its Fold algebra verbatim."""
    op = body if isinstance(body, Fold) else _wrap(Body.coerce(body), ())
    piece = TileOp(op=op, place=Placement(free=tuple(free)), output_specs=output_specs)
    # A split CONSUMES the kernel it replaces: the piece drops its schedule row and its structural
    # identity. Built fresh here, so this states the contract rather than doing work — and the rule
    # that mints a kernel is where that has to be said.
    return replace(piece, knobs=consume_kernel_row(piece.knobs))


def _state_fold(axis: Axis, algebra: Fold, loads: tuple[Load, ...], scope: tuple[Axis, ...] = ()) -> Fold:
    """Fold already-reduced state tuples through ``algebra``'s unchanged monoid."""
    values = tuple(name for load in loads for name in load.defines())
    return Fold(
        axes=(axis,),
        # The workspace reads are slabs like any other gmem read: they declare the split axis and
        # the output coordinates the enclosing placement binds.
        operands=tuple(Fold.slab(load, (axis, *scope)) for load in loads),
        lift=Lambda(params=(axis.name, *values), body=Body(), results=values),
        init=algebra.init,
        combine=algebra.combine,
    )


def _project(fold: Fold, body) -> Fold:
    """Attach a pure projection body to one Fold, dropping the empty wrapper."""
    body = Body.coerce(body)
    return _wrap(body, (fold,)) if body else fold


def _output_root(root: Node, outputs: set[str]) -> Node:
    """A graph-node view containing only the output ports owned by one projection Fold."""
    by_name = dict(zip(root.buffer_names(), root.outputs, strict=True))
    ordered = tuple(name for name in root.buffer_names() if name in outputs)
    if outputs != set(ordered):
        raise ValueError(f"projection stores target unknown output buffers: {sorted(outputs - set(ordered))}")
    tensors = tuple(replace(by_name[name], name=name) for name in ordered)
    return replace(root, id=ordered[0], outputs=tensors)


def _split_projection(tile: TileOp, root: Node, selected: Fold):
    """Separate an independent MIMO projection into output-owning Fold pieces."""
    op = tile.op
    if not isinstance(op, Fold) or op.axis is not None or len(op.operands) < 2:
        return root, tuple(projection_tail(tile)), ()

    pieces = []
    chosen = None
    for fold, body, stores in projection_regions(op, tile.output_specs):
        node = _output_root(root, {store.write.output for store in stores})
        entry = (node, fold, body, stores)
        if fold is selected:
            chosen = entry
        else:
            pieces.append(entry)
    if chosen is None:
        raise ValueError("the split Fold does not own an independent projection region")
    node, _, body, stores = chosen
    return node, tuple(apply_output_specs(body, stores)), tuple(pieces)


def _add_projection_pieces(match: Match, frag: Graph, pieces: tuple, free: tuple) -> Graph:
    """Add the unsplit independent projection Folds as fresh schedulable kernels. Each is a piece
    of the REALIZED split — the kernel-set decision was consumed by the kernel it addressed, and
    one pinned split means one split — so it carries the consumed-split receipt
    (``split_consumed``): a ``REDUCE`` pin's ``g`` half strips on it instead of splitting the
    sibling region again (or raising)."""
    for root, fold, body, stores in pieces:
        tile = replace(_piece(_project(fold, body), free, output_specs=stores), split_consumed=True)
        _add_output_piece(match, frag, root, tile, _piece_inputs(root, tile))
    return frag


def _captured_prologue(partial_fold: Fold, pre: tuple, split: Axis, free: tuple) -> tuple:
    """The projection-prologue stmts the sliced fold still CAPTURES — their backward cone, carried
    into the partial. The chain form keeps the head fold as a BODY member of its projection
    wrapper (:func:`~emmy.compiler.ir.tile.ops.head`), so a prologue stmt evaluated once per cell
    (a scalar scale load) can define a name the fold's lift reads; slicing the fold alone would
    leave that capture dangling in the partial."""
    # Read off the DECLARATION rather than lowering the partial and scanning it for free names:
    # a term states what it needs, and `edge_axes` reads that declaration. The axis
    # subtraction stays — a coordinate is supplied by the enclosing loop, not by the prologue —
    # but the axes a nested term BINDS are already excluded by the reading itself.
    axes = {split.name, *(a.name for a in free)}
    captures = partial_fold.index_space - frozenset(axes)
    if not captures:
        return ()
    return Body(pre).backward_cone(captures).members


# ---- the realization -------------------------------------------------------------------------- #


def realize_split(match: Match, root: Node, cta: int, finalize: str) -> Graph:
    """Build the split fragment: the partial + deferred finalize pair, or the atomic arm's one
    kernel. Always a ``Graph``, never a ``TileOp`` — this rewrite's whole job is to change the
    kernel SET, and a 1:1 op rebind is how the engine says the OPPOSITE (same kernel, decided
    further — knobs merged forward, no pass-scan restart). The one-kernel atomic arm splices too,
    via :func:`_one`."""
    tile: TileOp = root.op
    # The fold NODE carries the algebra — every algebra read below (state names, identities, the
    # cross-partition combine) is off the node, never a loop annotation. The projection (when the
    # kernel carries one) rides the zero-axis ``Fold`` wrapper — its ONE home; peel it here, with
    # its output specifications reconstituted, and retarget its root ``Write`` below.
    node = head(tile.op)
    assert node is not None, "the split offer fires on node-form kernels only"
    root, projection, projection_pieces = _split_projection(tile, root, node)
    free = tuple(tile.place.free)
    if node.as_contraction() is not None:
        split, partial_fold = _sliced_contraction(node, cta)
    else:
        _enforce(splitk_width(node.axis, cta))
        split = Axis(name=_SPLIT, extent=Dim(cta))
        partial_fold = _slice_fold(node, node.axis.extent.as_static() // cta)
    states = tuple(partial_fold.combine.results)
    n_comp = len(states)
    out = root.output
    cell = _cell_index(projection, free)
    # The chain form keeps the head fold as a BODY member of its projection wrapper (``head``'s
    # sweep case), so ``projection`` still contains it. Strip it — the epilogue's states come from
    # the workspace combine (or the atomic partial), and keeping the fold would re-run the whole
    # reduction per cell AND shadow those states — and carry the prologue cone the sliced fold
    # still captures into the partial (:func:`_captured_prologue`).
    fold_at = next((i for i, stmt in enumerate(projection) if stmt is node), None)
    prologue: tuple = ()
    if fold_at is not None:
        prologue = _captured_prologue(partial_fold, projection[:fold_at], split, free)
        projection = (*projection[:fold_at], *projection[fold_at + 1 :])
    frag = _frag(match, root)

    if finalize == "atomic":
        # Direct atomic finalize: ONE kernel — each CTA atomicAdds its slice's state into the
        # output (zero-init'd per launch), the GRID stage consumed into the grid. ``projection``
        # is the kernel's epilogue (``mean``'s ``×1/N``, …); a bare carrier has just the output
        # ``Write``. Whether the arm can carry it is :func:`atomic_finalize`'s one answer — the
        # same predicate the offer and the pin applied, re-asked here over the SELECTED region's
        # projection (the offer read the whole tail, a conservative superset — see its docstring).
        _enforce(atomic_finalize(partial_fold, projection, tile.outputs))
        if projection:
            atomic_proj = tuple(replace(s, atomic=True) if isinstance(s, Write) else s for s in projection)
        else:
            atomic_proj = (Write(output=out.name, index=cell, values=states, atomic=True),)
        p_body, p_stores = _boundary(atomic_proj)
        if fold_at is not None:
            # Body-resident form: keep the wrapper's own shape — prologue, the sliced fold in
            # place, then the per-partition epilogue (operand edges evaluate before the body, so
            # ``_project`` would put a captured prologue AFTER the fold that reads it).
            # ``fold_at`` indexes the PRE-strip ``projection``, and it stays valid against
            # ``p_body`` because the strip removed exactly the stmt AT that index and
            # ``_boundary`` only extracts trailing ``Write``s, which sit after the fold.
            piece = _piece((*p_body[:fold_at], partial_fold, *p_body[fold_at:]), (split, *free), output_specs=p_stores)
        else:
            piece = _piece(_project(partial_fold, p_body), (split, *free), output_specs=p_stores)
        result = _one(match, frag, root, piece)
        return _add_projection_pieces(match, result, projection_pieces, free)

    # Deferred finalize: write every raw component to ``ws[(comp,) ksplit, *cell]``. The workspace
    # shape MUST match the rank of the index the writes/loads use — or ``render_index``'s
    # rank-mismatch fallback silently flattens without strides (colliding partials). ``ws_cell``
    # is the FREE-axis vars (the partial has no original ``Write`` to copy), so size the workspace
    # by the free extents, not ``out.shape`` (whose extent-1 batch dims the grid never carries). A
    # multi-component carrier packs its per-component states into a leading ``comp`` axis; the
    # single-component workspace stays ``ws[ksplit, *cell]``. The workspace is **f32**: it holds
    # raw pre-projection accumulator states (summed + ⊗-combined by the finalize), and the
    # pre-projection state must not round-trip through the output dtype (an fp16 round-trip can
    # saturate outlier partials to ±inf before the combine and costs the mantissa of every
    # partition sum).
    ws_name = f"{out.name}__partial"
    ws_shape = (Dim(n_comp), Dim(cta), *(a.extent for a in free)) if n_comp > 1 else (Dim(cta), *(a.extent for a in free))
    ws_cell = tuple(Var(ax.name) for ax in free)

    def ws_index(i: int) -> tuple:
        lead = (Literal(i, "int"), Var(split.name)) if n_comp > 1 else (Var(split.name),)
        return (*lead, *ws_cell)

    # --- partial kernel: reduce a CTA's slice, write its carrier state to the workspace. The
    # split axis joins as a lead grid axis via the partial tile's OWN placement — the view derives
    # lead axes from the placement, so nothing is restamped on the node.
    ws_stores = tuple(OutputSpec(write=Write(output=ws_name, index=ws_index(i), value=states[i])) for i in range(n_comp))
    partial_body = (*prologue, partial_fold) if prologue else partial_fold
    partial_tile = _piece(partial_body, (split, *free), output_specs=ws_stores)

    # --- finalize kernel: identity-lift each workspace state tuple through the SAME monoid.
    # The merge axis carries the SAME consumed-split receipt the partial's slice does: the
    # finalize enumerates the partitions of a split that already happened, so the receipt must
    # read as a kernel that already realized one. Without it an ambient ``REDUCE`` pin splits the
    # finalize too and its workspace collides with the partial's (``<out>__partial`` exists).
    fin_axis = replace(split, window=Window(parent=split, partition=True))
    other = tuple(f"{nm}__p" for nm in states)
    loads = tuple(Load(name=other[i], input=ws_name, index=ws_index(i)) for i in range(n_comp))
    fin_fold = _state_fold(fin_axis, partial_fold, loads, free)
    fin_proj, fin_stores = _boundary(projection)
    if not fin_stores and not Body(tuple(fin_proj)).writes:
        # The projected value is the LAST defining stmt's name — the epilogue tail may end with
        # non-defining stmts, so scan backward instead of indexing [-1].
        out_val = next((s.defines()[-1] for s in reversed(fin_proj) if s.defines()), states[0])
        fin_stores = (OutputSpec(write=Write(output=out.name, index=cell, value=out_val)),)
    # The finalize is stamped AFTER the workspace joins the fragment: it reads that buffer, and a
    # kernel's structural features fold in its operands' dtypes, which only resolve once the
    # buffer is a graph node.
    frag.add_node(op=partial_tile, inputs=list(root.inputs), output=Tensor(ws_name, ws_shape, F32), node_id=ws_name)
    fin_tile = _piece(_project(fin_fold, fin_proj), free, output_specs=fin_stores)
    result = _add_output_piece(match, frag, root, fin_tile, _piece_inputs(root, fin_tile, ws_name))
    return _add_projection_pieces(match, result, projection_pieces, free)


__all__ = ["atomic_finalize", "realize_split", "split_forks", "splitk_width"]
