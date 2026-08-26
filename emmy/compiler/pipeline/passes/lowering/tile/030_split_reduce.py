"""Realize a cross-CTA ``GRID`` reduce partition as a graph rewrite.

The rewrite consumes only the stored :class:`Fold` algebra. Each partial evaluates the same
``Fold(init, combine)`` over a contiguous axis slice and writes its complete state tuple. The
deferred finalize identity-lifts those tuples through the same ``init`` and ``combine``, then
applies the original projection. This is the common path for additive and exp-family monoids;
split-reduce does not recognize carrier families.

The atomic arm is the generic exception: it is legal only for a single additive state component
whose projection distributes over addition. Otherwise the deferred f32 workspace preserves the
full state until the finalize combines and projects it.

Every piece is a fresh unmapped :class:`TileOp`. A graph splice restarts the lowering pass scan:
total lift and the exp-family rewrite observe an already canonical Fold tree, while scheduling
offers each piece its own row. An axis :class:`Window` records that the partition has already been
consumed and prevents recursive splitting.

Structural split-K has one additional realization: when scheduling has already factored an outer
partition Fold around a bilinear Fold, the partial keeps the inner bilinear Fold bare so the normal
contraction binder can schedule it. Both paths use the same state-tuple finalize.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule import FoldMove, Level
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Load, Write
from emmy.compiler.ir.tile import (
    Placement,
    ReducePlan,
    Store,
    TileOp,
    split_effects,
)
from emmy.compiler.ir.tile.ir import effect_tail
from emmy.compiler.ir.tile.ops import head, projection_regions, projection_tail, reduce_plan
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.knob import consume_kernel_row
from emmy.compiler.pipeline.passes.lowering.tile import _legality as legal

PATTERN = [Pattern("root", TileOp)]

_SPLIT = "_ksplit"  # the cross-CTA split grid axis


def _slice_fold(fold: Fold, b: int) -> Fold:
    """The same monoid Fold over one CTA's absolute contiguous slice."""
    axis = fold.axis
    assert axis is not None
    offset = BinaryExpr("+", Var(axis.name), BinaryExpr("*", Var(_SPLIT), Literal(b, "int")))
    sigma = Sigma({axis.name: offset})
    sliced_axis = replace(axis, extent=Dim(b), window=Window(parent=axis.source_axis or axis, partition=True))
    return fold.rewrite(
        lambda name: name,
        sigma,
        lambda candidate: sliced_axis if candidate.name == axis.name else candidate,
    )


def _boundary(stmts) -> tuple[tuple, tuple]:
    """Split a projection into a pure body and boundary stores."""
    split = split_effects(tuple(stmts))
    if split is None:
        return tuple(stmts), ()
    return split


def _cell_index(stmts, grid) -> tuple:
    """The output-cell index the original kernel writes (the projection ``Write``'s index,
    or — for a bare carrier whose grid-cell store is glue — the grid-axis vars). Read off the
    kernel's lowered body (``Fold.lower`` — the annotated loop nest, zero-axis and iterating alike)."""
    for s in stmts:
        if isinstance(s, Write):
            return s.index
    return tuple(Var(ax.name) for ax in grid)


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
        body = effect_tail(body.op.lower(), body.stores)
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
        stores=tuple(
            replace(store, write=replace(store.write, output=renamed.get(store.write.output, store.write.output))) for store in piece.stores
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
    scheduling carrying nothing of the row it replaced."""
    return _add_output_piece(match, frag, root, piece, list(root.inputs))


def _piece(body, free, *, stores: tuple = ()) -> TileOp:
    """One fresh unscheduled Tile kernel preserving its Fold algebra verbatim."""
    op = body if isinstance(body, Fold) else Fold.projection(body=Body.coerce(body))
    piece = TileOp(op=op, place=Placement(free=tuple(free)), stores=stores)
    # A split CONSUMES the kernel it replaces: the piece drops its schedule row and its structural
    # identity. Built fresh here, so this states the contract rather than doing work — and the rule
    # that mints a kernel is where that has to be said.
    piece.knobs = consume_kernel_row(piece.knobs)
    return piece


def _state_fold(axis: Axis, algebra: Fold, loads: tuple[Load, ...]) -> Fold:
    """Fold already-reduced state tuples through ``algebra``'s unchanged monoid."""
    values = tuple(name for load in loads for name in load.defines())
    return Fold(
        axis=axis,
        operands=loads,
        lift=Lambda(params=(axis.name, *values), body=Body(), results=values),
        init=algebra.init,
        combine=algebra.combine,
    )


def _project(fold: Fold, body) -> Fold:
    """Attach a pure projection body to one Fold, dropping the empty wrapper."""
    body = Body.coerce(body)
    return Fold.projection(operands=(fold,), body=body) if body else fold


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
    for fold, body, stores in projection_regions(op, tile.stores):
        node = _output_root(root, {store.write.output for store in stores})
        entry = (node, fold, body, stores)
        if fold is selected:
            chosen = entry
        else:
            pieces.append(entry)
    if chosen is None:
        raise ValueError("the scheduled Fold does not own an independent projection region")
    node, _, body, stores = chosen
    return node, tuple(effect_tail(body, stores)), tuple(pieces)


def _add_projection_pieces(match: Match, frag: Graph, pieces: tuple, free: tuple) -> Graph:
    """Add the unsplit independent projection Folds as fresh schedulable kernels."""
    for root, fold, body, stores in pieces:
        tile = _piece(_project(fold, body), free, stores=stores)
        _add_output_piece(match, frag, root, tile, _piece_inputs(root, tile))
    return frag


def _split_contraction(
    match: Match,
    root: Node,
    tile: TileOp,
    node,
    outer: Fold,
    plan: ReducePlan,
    split: Axis,
    projection=(),
    projection_pieces: tuple = (),
):
    """Realize a structural split-K whose inner bilinear Fold is already factored."""
    out = root.output
    grid = tile.place.grid
    cell = tuple(Var(a.name) for a in grid)
    frag = _frag(match, root)
    states = tuple(outer.combine.results)
    n_comp = len(states)  # 1 = plain matmul; N = the multi-channel (gate/up) node's per-channel accs
    acc = states[0]
    epilogue = list(projection)  # the fused projection off the zero-axis ``Fold`` wrapper (empty for a bare matmul)

    def _partial(body):
        """The partial kernel's node: the stored fold verbatim, its retargeted stores riding the
        zero-axis ``Fold`` wrapper (the one home for a projection). ``ksplit`` joins as a lead grid axis via
        the partial tile's OWN grid — the view derives lead axes from the placement, so nothing is
        restamped on the node."""
        return _project(node, body)

    # The GRID stage decides the cross-CTA move; this rewrite only realizes it. The arm's
    # conditions are ``_legality.atomic_finalize``'s — the same call the enumeration and the pin
    # made — so a divergence between what was chosen and what is realizable reports that predicate's
    # reason rather than a second wording of it.
    (cross_move,) = next(st for st in plan.stages if st.level is Level.GRID).combine(warp_size=32)
    if cross_move is FoldMove.ATOMIC:
        legal.enforce(legal.atomic_finalize(states, epilogue, tile.outputs), pinned=True)
        if epilogue:
            atomic_epi = tuple(replace(s, atomic=True) if isinstance(s, Write) else s for s in epilogue)
        else:
            atomic_epi = (Write(output=out.name, index=cell, value=acc, atomic=True),)
        p_body, p_stores = _boundary(atomic_epi)
        result = _one(match, frag, root, _piece(_partial(p_body), (split, *grid), stores=p_stores))
        return _add_projection_pieces(match, result, projection_pieces, tuple(grid))

    # Deferred finalize: write every raw component to ``ws[(comp,) ksplit, *cell]``.
    # The workspace shape MUST match the rank of the index the writes/loads use — or
    # ``render_index``'s rank-mismatch fallback silently flattens without strides (colliding
    # partials; the misaligned-vector-store crash). ``cell`` is the GRID vars (the structural
    # partial has no original Write to copy), so size the workspace by the grid extents, not
    # ``out.shape`` (whose extent-1 batch dims the grid never carries). A multi-channel node
    # packs its per-channel accs into a leading ``comp`` axis (the residual path's convention);
    # the single-component workspace stays ``ws[ksplit, *cell]``. The workspace is **f32**: it
    # holds raw pre-projection accumulator states (summed + ⊗-combined by the finalize), and the
    # pre-projection state must not round-trip through the output dtype (the flash split-KV /
    # An fp16 round-trip can saturate outlier partials to ±inf before
    # the combine and costs the mantissa of every partition sum).
    ws_name = f"{out.name}__partial"
    ws_shape = (Dim(plan.cta), *(a.extent for a in grid)) if n_comp == 1 else (Dim(n_comp), Dim(plan.cta), *(a.extent for a in grid))

    def ws_index(i: int) -> tuple:
        lead_ix = (Var(split.name),) if n_comp == 1 else (Literal(i, "int"), Var(split.name))
        return (*lead_ix, *cell)

    ws_stores = tuple(Store(write=Write(output=ws_name, index=ws_index(i), value=states[i])) for i in range(n_comp))
    partial_tile = _piece(_partial(()), (split, *grid), stores=ws_stores)

    # --- finalize kernel: identity-lift each workspace state tuple through the SAME monoid.
    other = tuple(f"{nm}__p" for nm in states)
    loads = tuple(Load(name=other[i], input=ws_name, index=ws_index(i)) for i in range(n_comp))
    # The merge axis carries the SAME consumed-split receipt the partial's slice does: the finalize
    # enumerates the partitions of a split that already happened, so ``_splittable_axis`` must read it
    # as a kernel that already realized one. Without the receipt an ambient ``REDUCE`` pin splits the
    # finalize too and its workspace collides with the partial's (``<out>__partial`` already exists).
    fin_axis = replace(split, window=Window(parent=split, partition=True))
    fin_fold = _state_fold(fin_axis, outer, loads)
    fin_proj, fin_stores = _boundary(epilogue)
    if not fin_stores and not any(isinstance(s, Write) for s in fin_proj):
        # The projected value is the LAST defining stmt's name — the epilogue tail may end with
        # non-defining stmts, so scan backward instead of indexing [-1].
        out_val = next((s.defines()[-1] for s in reversed(fin_proj) if s.defines()), acc)
        fin_stores = (Store(write=Write(output=out.name, index=cell, value=out_val)),)
    # Stamped AFTER the workspace is in the fragment: the finalize reads it, and its structural
    # features fold in its operands' dtypes, which only resolve once the buffer is a graph node.
    frag.add_node(op=partial_tile, inputs=list(root.inputs), output=Tensor(ws_name, ws_shape, F32), node_id=ws_name)
    fin_tile = _piece(_project(fin_fold, fin_proj), grid, stores=fin_stores)
    result = _add_output_piece(match, frag, root, fin_tile, _piece_inputs(root, fin_tile, ws_name))
    return _add_projection_pieces(match, result, projection_pieces, tuple(grid))


def rewrite(match: Match, root: Node) -> Graph:
    # Always a ``Graph``, never a ``TileOp``: this rule's whole job is to change the kernel SET, and
    # a 1:1 op rebind is how the engine says the OPPOSITE (same kernel, decided further — knobs
    # merged forward, no pass-scan restart). The one-kernel atomic arm splices too, via ``_one``.
    # An arm with nothing to do raises ``RuleSkipped``; none returns ``None``.
    tile: TileOp = root.op
    # The reduce partition lives on the Fold node (off the schedule) — ``reduce_plan`` reads
    # it there, falling back to the ``TileOp``'s residual ``reduce`` field for a non-tiled
    # contraction's split-K (still a zero-axis fold).
    plan = reduce_plan(tile) if tile.op is not None else None
    if plan is None or not plan.needs_split:
        raise RuleSkipped("no cross-CTA split stage — nothing to split")

    op = tile.op
    # The fold NODE carries the algebra (``reduce_plan`` guaranteed a ``Fold`` head) — every
    # algebra read below (state names, identities, the cross-partition combine) is off the node,
    # never a loop annotation.
    fold_node = head(op)
    assert fold_node is not None, "split-reduce fires on node-form kernels only (reduce_plan gates on a node head)"
    cta = plan.cta
    rax = fold_node.axis
    # Structural split-K: the axis and inner bilinear Fold were already factored by scheduling,
    # so the partial keeps that Fold bare for the normal contraction binder.
    # The projection (when the split node carries one) rides the zero-axis ``Fold`` wrapper over the split
    # ``Fold`` — its ONE home; peel it here and hand it to the realizer.
    # The projection with the kernel-boundary stores reconstituted (1q) — the split realizers
    # retarget the root ``Write`` exactly as when it rode the zero-axis ``Fold`` body. A projection whose
    # only stmt was the root ``Write`` leaves a BARE fold behind (the zero-axis ``Fold`` wrapper dropped
    # with its last in-body stmt), so the reconstitution keys off the TileOp, not the node shape.
    root, projection, projection_pieces = _split_projection(tile, root, fold_node)
    # The split node's inner contraction — multi-channel included — rides the outer reduce's
    # identity-lift operand composition (the one composition rule; ``Fold.composed``).
    inner = fold_node.composed
    if inner is not None:
        return _split_contraction(match, root, tile, inner, fold_node, plan, rax, projection, projection_pieces)
    # Static-and-divisible is ``_legality.splitk_width``'s question, asked once for both the
    # structurally factored split (through ``_factor_k``) and this direct one.
    legal.enforce(legal.splitk_width(rax, cta), pinned=True)
    b = rax.extent.as_static() // cta
    states = tuple(fold_node.combine.results)
    n_comp = len(states)

    out = root.output
    grid = tile.place.grid
    after = list(projection)
    cell = _cell_index(after, grid)
    split = Axis(name=_SPLIT, extent=Dim(cta))
    partial_fold = _slice_fold(fold_node, b)

    # --- atomic finalize: ONE kernel — each CTA atomicAdds its slice's state into the output
    # (zero-init'd per launch). Additive (single-component) carriers only; the GRID stage is
    # consumed into the grid (the split becomes a grid axis), no second node. The move itself
    # derives from the one placement-keyed selector (ReduceStage.combine).
    (cross_move,) = next(st for st in plan.stages if st.level is Level.GRID).combine(warp_size=32)
    if cross_move is FoldMove.ATOMIC:
        # ``after`` is the kernel's projection epilogue (``mean``'s ``×1/N``, a fused
        # bias/activation, …); a bare carrier (``sum`` / a contraction matmul) has just the output
        # ``Write``. Whether the atomic arm can carry it — along with the carrier's arity and the
        # output's storage width — is ``_legality.atomic_finalize``'s one answer.
        legal.enforce(legal.atomic_finalize(states, after, tile.outputs), pinned=True)
        if after:
            atomic_proj = tuple(replace(s, atomic=True) if isinstance(s, Write) else s for s in after)
        else:
            # A bare carrier (``sum`` / a contraction matmul) — its grid-cell store is glue; synthesize
            # the atomic ``Write`` of the carrier state directly.
            atomic_proj = (Write(output=out.name, index=cell, values=states, atomic=True),)
        proj_body, proj_stores = _boundary(atomic_proj)
        frag = _frag(match, root)
        piece = _piece(_project(partial_fold, proj_body), (split, *grid), stores=proj_stores)
        result = _one(match, frag, root, piece)
        return _add_projection_pieces(match, result, projection_pieces, tuple(grid))

    # The ``__partial`` workspace packs every carrier-state component: ``ws[comp, cta, *free]``
    # (the ``comp`` leading axis is dropped for a single-component additive carrier, so the
    # additive workspace stays ``ws[cta, *free]``). Any multi-component carrier writes its
    # complete state tuple to the leading ``comp`` slices.
    # The workspace shape MUST match the rank of ``ws_index`` — sized by the GRID extents, never
    # ``out.shape`` (whose extent-1 batch dims the grid never carries): a rank mismatch makes
    # ``render_index``'s fallback flatten WITHOUT strides, colliding the partitions' states (the
    # statistic-with-projection split wrote ``ws[ksplit + cell]``). And it is **f32**: it holds
    # raw pre-projection accumulator states — the same rule as the contraction arm above (the
    # workspace rule).
    ws_name = f"{out.name}__partial"
    ws_shape = (Dim(n_comp), Dim(cta), *(a.extent for a in grid)) if n_comp > 1 else (Dim(cta), *(a.extent for a in grid))
    ws_cell = tuple(Var(ax.name) for ax in grid)  # grid-rank by construction; ``cell`` (the output
    # Write's index, possibly full-rank with batch literals) stays the OUTPUT store's index only.

    def ws_index(i: int) -> tuple:
        lead = (Literal(i, "int"), Var(_SPLIT)) if n_comp > 1 else (Var(_SPLIT),)
        return (*lead, *ws_cell)

    # --- partial kernel: reduce a CTA's slice, write its carrier state to the workspace -----
    ws_stores = tuple(Store(write=Write(output=ws_name, index=ws_index(i), value=states[i])) for i in range(n_comp))
    frag = _frag(match, root)
    partial_tile = _piece(partial_fold, (split, *grid), stores=ws_stores)

    # --- finalize kernel: identity-lift each workspace state tuple through the SAME monoid.
    # The merge axis carries the SAME consumed-split receipt the partial's slice does: the finalize
    # enumerates the partitions of a split that already happened, so ``_splittable_axis`` must read it
    # as a kernel that already realized one. Without the receipt an ambient ``REDUCE`` pin splits the
    # finalize too and its workspace collides with the partial's (``<out>__partial`` already exists).
    fin_axis = replace(split, window=Window(parent=split, partition=True))
    other = tuple(f"{nm}__p" for nm in states)
    loads = tuple(Load(name=other[i], input=ws_name, index=ws_index(i)) for i in range(n_comp))
    fin_fold = _state_fold(fin_axis, fold_node, loads)
    fin_proj, fin_stores = _boundary(after)
    if not fin_stores and not Body(tuple(fin_proj)).writes:
        # Backward scan — the epilogue tail may end with non-defining stmts (see the deferred
        # kernel arm above).
        out_val = next((s.defines()[-1] for s in reversed(fin_proj) if s.defines()), states[0])
        fin_stores = (Store(write=Write(output=out.name, index=cell, value=out_val)),)
    # --- splice the two-kernel fragment in place of the single split TileOp ----------------
    # The finalize is stamped AFTER the workspace joins the fragment: it reads that buffer, and a
    # kernel's structural features fold in its operands' dtypes.
    frag.add_node(op=partial_tile, inputs=list(root.inputs), output=Tensor(ws_name, ws_shape, F32), node_id=ws_name)
    fin_tile = _piece(_project(fin_fold, fin_proj), grid, stores=fin_stores)
    result = _add_output_piece(match, frag, root, fin_tile, _piece_inputs(root, fin_tile, ws_name))
    return _add_projection_pieces(match, result, projection_pieces, tuple(grid))
