"""Cross-CTA split-reduce (the ``cta`` tier) — consume a ``GRID`` ``ReduceStage``.

A reduce partition with a ``GRID`` stage (``ReducePlan.needs_split``) splits the reduce
axis across CTAs. This pass realizes that split as a **graph rewrite** — the schedule
carries the partition, the graph carries the kernel count. It reads the reduce STRUCTURE off
the kernel's annotated reduce ``Loop`` (``loop.axis`` / position) and the ALGEBRA off the
``Fold`` node through the lowering-side :class:`Reduction` view:

- **partial kernel** — the ``cta`` stage becomes an extra grid axis (``_ksplit``); each CTA
  reduces its **contiguous slice** ``[s·B, (s+1)·B)`` of the reduce axis (``B =
  extent / cta``) and contributes its carrier *state* (not the projected output).
- **finalize** — two arms, picked by the ``GRID`` stage's finalize letter
  (``ReducePlan.finalize``):
  - ``"kernel"`` — the partial writes its state to a ``ws[cta, *free]`` ``__partial``
    workspace; a sibling **finalize kernel** seeds the carrier state then folds the
    workspace over the split axis via ``Reduction.state_merge`` (the cross-partition
    combine, a renderable :class:`StateMerge`) and projects the output. **2 nodes.** The only
    legal arm for a twisted carrier (flash's ``e^{Δm}`` rescale can't be an atomic).
  - ``"atomic"`` — the partial ``atomicAdd``\\ s its (additive) state into the output (applying
    the kernel's projection epilogue per-partition first, when that epilogue *distributes* over
    the add — ``mean``'s ``×1/N``; a non-distributive one like ``l2``'s ``sqrt`` is refused, use
    ``"kernel"``); the output is zero-init'd per launch. **1 node.** Additive carriers only.

**Every piece is a BRAND-NEW kernel.** Each is minted UNMAPPED, carrying no knob, no placement
and no schedule slice of the kernel it replaces, its structural features re-derived from its OWN
body (:func:`_piece`). ``020_schedule`` picks each up on the pass-scan restart and offers it a
fork, exactly as it does a freshly recognized term: no pass can tell a piece from a fresh kernel,
and none tries. The pre-split row decided THAT the split happens; it decides nothing about how
either piece runs, and the search is told nothing about why a slice has several kernels — a split
node simply has no latency of its own and prices as the Σ over the kernels it produced.

Even the one-kernel atomic arm splices a ``Graph`` (:func:`_one`). A 1:1 op rebind is how the
engine says "the same kernel, decided further", so it merges the replaced op's knobs forward and
does not restart the pass scan — the piece would inherit the row it was minted to shed and would
never reach a fork of its own.

So the GRID stage is **consumed** by the kernel that realizes it: the pieces come back through the
enumeration as ordinary kernels, and what stops the partial re-splitting its own slice (K=512 →
256 → … → 1) is the sliced axis itself, built as a ``Window`` of its parent — no provenance flag,
just the axis's shape (``_schedule._splittable_axis``). ``lowering/kernel`` therefore only ever
sees single-launch kernels (``assert not needs_split``).

This cut handles **additive** carriers — a degenerate ``PLANAR`` reduce (``sum``) and a
``CONTRACTION`` contraction (split-K matmul), one carrier-state component each — and, through the
residual path's carrier-generic deferred-kernel finalize, a multi-component ``TWISTED`` carrier
(the streaming softmax's ``(m, d)`` state packs into the workspace's leading ``comp`` axis and the
finalize folds partitions through the exp-family combine). The WARP-tiled fragment-residence
split-KV arm was deleted with the flash pattern compiler; it returns with the TWISTED computed-A
recognizer arm. Pays where the un-split grid starves the SMs; pin ``REDUCE=g<n>k``.

**Two shapes of contraction split-K.** A structural ``Fold(axis=ksplit,
source=Fold.contraction(k_axis=kslice))`` (built schedule-side) has its K axis already
factored + operands offset, so :func:`_split_contraction` makes the partial the **bare bilinear fold**
— it factorizes to **mma** (or scalar) through ``_factor.factorize``, ``ksplit`` prefixed as a lead
grid axis, no ``_slice_loop``. The residual path below (a plain-sum ``sum`` split, or a coop/ILP
contraction still on a zero-axis ``Fold``) keeps the loop-slicing rewrite.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis, AxisRole, Window
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.schedule import FoldMove, Level
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Body, Init, Load, Loop, Write
from emmy.compiler.ir.stmt.passes import projection_distributes as _projection_distributes
from emmy.compiler.ir.tile import (
    Fold,
    Placement,
    ReducePlan,
    Store,
    TileOp,
    split_effects,
)
from emmy.compiler.ir.tile.ir import effect_tail
from emmy.compiler.ir.tile.ops import head, projection_tail, reduce_loop, reduce_plan
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering._reduction import Reduction
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import nodify_reduce

PATTERN = [Pattern("root", TileOp)]

_SPLIT = "_ksplit"  # the cross-CTA split grid axis


def _slice_loop(rloop: Loop, b: int) -> Loop:
    """Slice the reduce ``Loop`` to a CTA's contiguous block: offset every reduce-axis load by
    ``_ksplit · B`` (σ on the loop body) and shrink the axis extent to ``B`` (so the loop walks
    ``[0, B)`` while reading ``[s·B, (s+1)·B)``). Only the operand load indices move — the
    body's fold ``Accum``\\ s (the loop-level algebra spelling) ride through unchanged."""
    rax = rloop.axis
    offset = BinaryExpr("+", Var(rax.name), BinaryExpr("*", Var(_SPLIT), Literal(b, "int")))
    sigma = Sigma({rax.name: offset})
    ident = lambda n: n  # noqa: E731

    def _keep_axes(orig, new):
        # The generic σ arm expands an ``Accum``'s reduce ``axes`` to the substitution's free
        # vars (``a1`` → ``(_ksplit, a1)``) — correct for a general axis rename, but the slice's
        # ``_ksplit`` is a GRID axis of the partial, not a reduce axis of its loop: the accum
        # still folds only its own slice. Keep the original axes so the sliced loop stays the
        # λ-representable canonical shape (the derived serial step stamps ``(axis,)``).
        if isinstance(new, Accum):
            return replace(new, axes=orig.axes)
        return new

    new_body = tuple(_keep_axes(s, s.rewrite(ident, sigma)) for s in rloop.body)
    # The slice records its parentage — see ``_schedule._factor_k``: the partition is consumed
    # here, and an axis that is already a window of a parent cannot be partitioned again.
    new_ax = replace(rax, extent=Dim(b), window=Window(parent=rax.source_axis or rax))
    return Loop(axis=new_ax, body=Body(new_body), unroll=rloop.unroll, role=rloop.role)


def _boundary(stmts, plain_only: bool = False) -> tuple[tuple, tuple]:
    """Split a retargeted projection into ``(pure body, boundary Store decorations)`` — the split
    kernels carry their stores on ``TileOp.stores`` like every other kernel (1q). The raw spelling
    stands whole (empty stores) when ``split_effects``' round-trip gate declines, or — under
    ``plain_only`` (the FLAT finalize / atomic-partial kernels, whose materialization reattaches
    top-level ``Write``\\ s only) — when the split took an output-sweep store."""
    split = split_effects(tuple(stmts))
    if split is None or (plain_only and any(st.sweep is not None for st in split[1])):
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


def _one(frag: Graph, root: Node, piece: TileOp) -> Graph:
    """The ATOMIC arm's one-kernel fragment. It replaces the split kernel with ONE kernel, but it
    is a SPLICE, never an op rebind: a rebind is how the engine says "the same kernel, decided
    further", so it merges the replaced op's knobs forward and does not restart the pass scan. The
    atomic partial is a different kernel — its own placement, its own body — and it has to reach
    ``020_schedule`` carrying nothing of the one it replaced."""
    frag.add_node(op=piece, inputs=list(root.inputs), output=root.output, node_id=root.output.name)
    frag.outputs = [root.output.name]
    return frag


def _residual(map_op: Fold, like: Fold) -> Fold:
    """The split partial's op: nodify the flat zero-axis fold (:func:`nodify_reduce` — the sliced
    annotated ``Loop`` leaves the flat body for a ``Fold`` source, so the partial's fn stays
    strict; ``like`` is the pre-slice fold whose algebra a TWISTED loop extracts against). A
    partial with a prologue ahead of its reduce loop cannot nodify (``nodify_reduce`` asserts a
    head-position loop) — it keeps the raw flat spelling.

    Nothing is keyed into a schedule dict here: the piece decides its own reduce partition at its
    own fork, and the pre-split kernel's has no bearing on it."""
    body = list(map_op.body)
    if not body or not (isinstance(body[0], Loop) and body[0].role.is_reduce):
        return map_op
    op2, _ = nodify_reduce(map_op, like)  # not λ-representable ⇒ ``op2 is map_op``, the flat spelling
    return op2


def _piece(op, free, *, graph: Graph, name: str, stores: tuple = ()) -> TileOp:
    """One split piece as a BRAND-NEW kernel: an UNMAPPED ``TileOp`` carrying no knob, no
    placement and no schedule slice of the kernel it replaces, its structural features re-derived
    from its OWN body. ``020_schedule`` picks it up on the pass-scan restart and offers it a fork,
    exactly as it does a freshly recognized term — nothing downstream can tell the two apart.

    The pre-split kernel's row decided that the split happens; it decides nothing about how either
    piece runs. Threading it onto the partial (as this rule once did, because the graph splice
    drops knobs the way an op rebind does not) is what left the partial wearing 21 ``S_*`` features
    describing a body it no longer had, and the finalize already-placed with no knobs at all — no
    fork, no identity, untunable, and a whole-variant knob row that belonged to neither kernel."""
    return TileOp(op=op, name=name, place=Placement(free=tuple(free)), stores=tuple(stores))


def _split_contraction(match: Match, root: Node, tile: TileOp, node, outer: Fold, plan: ReducePlan, split: Axis, projection=()):
    """Realize a **structural** split-K ``Fold(axis=ksplit, step=[Fold])`` — the K axis is
    already factored (``split`` == ``ksplit``, extent == ``cta``) and the operands offset, so the
    partial is the **bare bilinear fold** with ``ksplit`` prefixed as a lead grid axis (each CTA a fixed
    partition) and its projection retargeted to the workspace / an atomic output. Because the partial
    is contraction, materialize expands it through ``_factor.factorize`` — **mma** for a warp
    atom, scalar otherwise. No ``_slice_loop`` (unlike the residual plain-sum path).

    Finalize matches the additive-carrier finalize: ``atomic`` (``g<w>a``) atomicAdds the partition's
    ``acc`` into the zero-init'd output — ONE kernel, both tiers (an mma partial's C fragment rides
    ``RegStore.atomic``: the packed f16x2/bf16x2 ``atomicAdd`` pair, per-element for f32) — at the
    cost of one output-dtype rounding per partition (the deferred arm's f32 workspace rounds once);
    ``kernel`` (``g<w>k``) writes each partition's ``acc`` to a ``ws[ksplit, *cell]`` workspace and a
    sibling finalize kernel sums it + runs the projection epilogue."""
    out = root.output
    grid = tile.place.grid
    cell = tuple(Var(a.name) for a in grid)
    frag = _frag(match, root)
    alg = Reduction(outer)
    states = alg.names
    n_comp = len(states)  # 1 = plain matmul; N = the multi-channel (gate/up) node's per-channel accs
    acc = states[0]
    epilogue = list(projection)  # the fused projection off the zero-axis ``Fold`` wrapper (empty for a bare matmul)

    def _partial(body):
        """The partial kernel's node: the stored fold verbatim, its retargeted stores riding the
        zero-axis ``Fold`` wrapper (the one home for a projection). ``ksplit`` joins as a lead grid axis via
        the partial tile's OWN grid — the view derives lead axes from the placement, so nothing is
        restamped on the node."""
        return Fold.projection(body=Body(body), operands=(node,))

    # The cross-CTA MOVE derives from the one placement-keyed selector (ReduceStage.combine over
    # the GRID stage) — this rewrite only realizes it; the carrier / projection legality raises
    # below stay here (they need the graph context the selector doesn't hold).
    (cross_move,) = next(st for st in plan.stages if st.level is Level.GRID).combine(warp_size=32)
    if cross_move is FoldMove.ATOMIC:
        if n_comp != 1:
            raise NotImplementedError("atomic finalize needs an additive (1-component) carrier; pin the deferred g<w>k finalize")
        if epilogue:
            # Apply the projection per-partition before the atomicAdd — legal only if it distributes.
            if not _projection_distributes(epilogue, states):
                raise NotImplementedError(
                    "atomic finalize can't carry a non-distributive projection on a split-K matmul; "
                    "pin the deferred-kernel finalize (REDUCE=g<w>k)"
                )
            atomic_epi = tuple(replace(s, atomic=True) if isinstance(s, Write) else s for s in epilogue)
        else:
            atomic_epi = (Write(output=out.name, index=cell, value=acc, atomic=True),)
        p_body, p_stores = _boundary(atomic_epi)
        return _one(frag, root, _piece(_partial(p_body), (split, *grid), name=tile.name, stores=p_stores, graph=frag))

    # --- deferred kernel finalize: partial writes each raw state to ``ws[(comp,) ksplit, *cell]``.
    # The workspace shape MUST match the rank of the index the writes/loads use — or
    # ``render_index``'s rank-mismatch fallback silently flattens without strides (colliding
    # partials; the misaligned-vector-store crash). ``cell`` is the GRID vars (the structural
    # partial has no original Write to copy), so size the workspace by the grid extents, not
    # ``out.shape`` (whose extent-1 batch dims the grid never carries). A multi-channel node
    # packs its per-channel accs into a leading ``comp`` axis (the residual path's convention);
    # the single-component workspace stays ``ws[ksplit, *cell]``. The workspace is **f32**: it
    # holds raw pre-projection accumulator states (summed + ⊗-combined by the finalize), and the
    # pre-projection state must not round-trip through the output dtype (the flash split-KV /
    # 020 channel-workspace rule — an fp16 round-trip saturates outlier partials to ±inf before
    # the combine and costs the mantissa of every partition sum).
    ws_name = f"{out.name}__partial"
    ws_shape = (Dim(plan.cta), *(a.extent for a in grid)) if n_comp == 1 else (Dim(n_comp), Dim(plan.cta), *(a.extent for a in grid))

    def ws_index(i: int) -> tuple:
        lead_ix = (Var(split.name),) if n_comp == 1 else (Literal(i, "int"), Var(split.name))
        return (*lead_ix, *cell)

    ws_stores = tuple(Store(write=Write(output=ws_name, index=ws_index(i), value=states[i])) for i in range(n_comp))
    partial_tile = _piece(_partial(()), (split, *grid), name=f"{tile.name or out.name}__partial", stores=ws_stores, graph=frag)

    # --- finalize kernel: seed each state, fold ``ws`` over ``ksplit`` (``as_state_merge`` — the
    # 1-component additive fold, or the N-component per-channel sums), then the original
    # projection epilogue (the multi-channel ⊗-combine applies HERE, once, after the sums) or a
    # bare store — the same finalize shape the residual path uses. The seeds + merge loop are
    # raw loop IR (the finalize is not a recognized term); its root store rides ``TileOp.stores``.
    other = tuple(f"{nm}__p" for nm in states)
    combine = alg.state_merge(other)
    ids = alg.identities()
    seeds = tuple(Init(name=nm, identity=ids[nm], dtype=F32) for nm in states)
    loads = tuple(Load(name=other[i], input=ws_name, index=ws_index(i)) for i in range(n_comp))
    # PLANAR, stated rather than inferred: ``Loop.is_reduce``'s structural fallback looks for an
    # ``Accum`` / ``Mma`` carrier and this fold's is a ``StateMerge``, so an unannotated cross-
    # partition loop reads as a FREE axis — the finalize then featurizes as a 3-deep parallel nest
    # with no reduction. ``010_recognize`` annotates every reduce loop it lifts; a kernel minted
    # here has to arrive the same way, or it cannot featurize like the kernel it is.
    fin_loop = Loop(axis=split, body=Body((*loads, combine)), role=AxisRole.PLANAR)
    fin_proj, fin_stores = _boundary(epilogue)
    if not fin_stores and not any(isinstance(s, Write) for s in fin_proj):
        # The projected value is the LAST defining stmt's name — the epilogue tail may end with
        # non-defining stmts, so scan backward instead of indexing [-1].
        out_val = next((s.defines()[-1] for s in reversed(fin_proj) if s.defines()), acc)
        fin_stores = (Store(write=Write(output=out.name, index=cell, value=out_val)),)
    fin_op = Fold.projection(body=Body((*seeds, fin_loop, *fin_proj)))
    # Stamped AFTER the workspace is in the fragment: the finalize reads it, and its structural
    # features fold in its operands' dtypes, which only resolve once the buffer is a graph node.
    frag.add_node(op=partial_tile, inputs=list(root.inputs), output=Tensor(ws_name, ws_shape, F32), node_id=ws_name)
    fin_tile = _piece(fin_op, grid, name=tile.name or out.name, stores=fin_stores, graph=frag)
    frag.add_node(op=fin_tile, inputs=[ws_name], output=Tensor(out.name, out.shape, out.dtype), node_id=out.name)
    frag.outputs = [out.name]
    return frag


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
    rloop = reduce_loop(op)
    # The fold NODE carries the algebra (``reduce_plan`` guaranteed a ``Fold`` head) — every
    # algebra read below (state names, identities, the cross-partition combine) is off the node,
    # never a loop annotation.
    fold_node = head(op)
    assert fold_node is not None, "split-reduce fires on node-form kernels only (reduce_plan gates on a node head)"
    cta = plan.cta
    rax = rloop.axis
    # Structural split-K: ``op`` is ``Fold(axis=ksplit, step=[Fold.contraction(k_axis=kslice)])`` —
    # the axis is already factored + operands offset (built schedule-side), so the partial
    # is the **bare bilinear fold** (→ ``factorize`` → mma / scalar), no ``_slice_loop``.
    # The projection (when the split node carries one) rides the zero-axis ``Fold`` wrapper over the split
    # ``Fold`` — its ONE home; peel it here and hand it to the realizer.
    # The projection with the kernel-boundary stores reconstituted (1q) — the split realizers
    # retarget the root ``Write`` exactly as when it rode the zero-axis ``Fold`` body. A projection whose
    # only stmt was the root ``Write`` leaves a BARE fold behind (the zero-axis ``Fold`` wrapper dropped
    # with its last in-body stmt), so the reconstitution keys off the TileOp, not the node shape.
    projection = tuple(projection_tail(tile))
    # The split node's inner contraction — multi-channel included — rides the outer reduce's
    # identity-lift operand composition (the one composition rule; ``Fold.composed``).
    inner = fold_node.composed
    if inner is not None:
        return _split_contraction(match, root, tile, inner, fold_node, plan, rax, projection)
    if not rax.extent.is_static:
        raise NotImplementedError("cross-CTA split of a symbolic reduce axis is not built yet")
    extent = rax.extent.as_static()
    if extent % cta != 0:
        raise NotImplementedError(f"cross-CTA split needs a divisible reduce axis (extent {extent} % cta {cta})")
    b = extent // cta
    alg = Reduction(fold_node)
    states = alg.names
    n_comp = len(states)

    out = root.output
    grid = tile.place.grid
    # The lowered loop nest (zero-axis and iterating alike) — find the annotated reduce loop
    # in it by position (``reduce_loop`` returns a fresh synthesized loop for a ``Fold``, so key
    # off the lowered list, not object identity).
    stmts = effect_tail(op.lower(), tile.stores)  # boundary stores reconstituted — ``after`` keeps its Write
    cell = _cell_index(stmts, grid)
    split = Axis(name=_SPLIT, extent=Dim(cta))
    idx = next(i for i, s in enumerate(stmts) if isinstance(s, Loop) and s.role.is_reduce)
    sliced_loop = _slice_loop(stmts[idx], b)
    # The stmts before / after the reduce loop: ``before`` is the (typically empty) prologue, ``after``
    # the projection epilogue (its own loads + computes + the output ``Write``).
    before = tuple(stmts[:idx])
    after = list(stmts[idx + 1 :])

    # --- atomic finalize: ONE kernel — each CTA atomicAdds its slice's state into the output
    # (zero-init'd per launch). Additive (single-component) carriers only; the GRID stage is
    # consumed into the grid (the split becomes a grid axis), no second node. The move itself
    # derives from the one placement-keyed selector (ReduceStage.combine).
    (cross_move,) = next(st for st in plan.stages if st.level is Level.GRID).combine(warp_size=32)
    if cross_move is FoldMove.ATOMIC:
        if n_comp != 1:
            raise NotImplementedError("atomic finalize needs an additive (1-component) carrier; the twisted carrier is kernel-only")
        # The kernel's projection epilogue (``mean``'s ``×1/N``, a fused bias/activation, …) rides
        # on ``after``; a bare carrier (``sum`` / a contraction matmul) has just the output ``Write``.
        # Atomic finalize applies the projection PER-PARTITION before the ``atomicAdd``, so it must
        # distribute over the add — else each CTA's contribution is mis-scaled. When it doesn't
        # distribute, refuse loudly: the deferred-kernel finalize (``g<n>k``) projects once after the
        # combine and is always correct.
        if after:
            if not _projection_distributes(after, states):
                raise NotImplementedError(
                    "atomic finalize can't carry a non-distributive projection epilogue "
                    "(e.g. a fused bias / activation on a split reduce); pin the deferred-kernel "
                    "finalize instead (REDUCE=…g<n>k)"
                )
            atomic_proj = tuple(replace(s, atomic=True) if isinstance(s, Write) else s for s in after)
        else:
            # A bare carrier (``sum`` / a contraction matmul) — its grid-cell store is glue; synthesize
            # the atomic ``Write`` of the carrier state directly.
            atomic_proj = (Write(output=out.name, index=cell, values=states, atomic=True),)
        proj_body, proj_stores = _boundary(atomic_proj, plain_only=True)
        atomic_op = Fold.projection(body=Body((*before, sliced_loop, *proj_body)))
        frag = _frag(match, root)
        piece = _piece(_residual(atomic_op, fold_node), (split, *grid), name=tile.name, stores=proj_stores, graph=frag)
        return _one(frag, root, piece)

    # The ``__partial`` workspace packs every carrier-state component: ``ws[comp, cta, *free]``
    # (the ``comp`` leading axis dropped for a single-component additive carrier, so the
    # additive workspace stays ``ws[cta, *free]``). A multi-component (twisted flash) carrier
    # writes its ``(m, l, O)`` state to the three ``comp`` slices, no multi-output kernel.
    # The workspace shape MUST match the rank of ``ws_index`` — sized by the GRID extents, never
    # ``out.shape`` (whose extent-1 batch dims the grid never carries): a rank mismatch makes
    # ``render_index``'s fallback flatten WITHOUT strides, colliding the partitions' states (the
    # statistic-with-projection split wrote ``ws[ksplit + cell]``). And it is **f32**: it holds
    # raw pre-projection accumulator states — the same rule as the contraction arm above (the
    # flash split-KV / 020 channel-workspace rule).
    ws_name = f"{out.name}__partial"
    ws_shape = (Dim(n_comp), Dim(cta), *(a.extent for a in grid)) if n_comp > 1 else (Dim(cta), *(a.extent for a in grid))
    ws_cell = tuple(Var(ax.name) for ax in grid)  # grid-rank by construction; ``cell`` (the output
    # Write's index, possibly full-rank with batch literals) stays the OUTPUT store's index only.

    def ws_index(i: int) -> tuple:
        lead = (Literal(i, "int"), Var(_SPLIT)) if n_comp > 1 else (Var(_SPLIT),)
        return (*lead, *ws_cell)

    # --- partial kernel: reduce a CTA's slice, write its carrier state to the workspace -----
    ws_stores = tuple(Store(write=Write(output=ws_name, index=ws_index(i), value=states[i])) for i in range(n_comp))
    partial_op = Fold.projection(body=Body((*before, sliced_loop)))
    frag = _frag(match, root)
    partial_tile = _piece(
        _residual(partial_op, fold_node), (split, *grid), name=f"{tile.name or out.name}__partial", stores=ws_stores, graph=frag
    )

    # --- finalize kernel: seed the carrier state, then fold each partition's state from the
    # workspace over the split axis via the fold's cross-partition combine (``Reduction.state_merge`` —
    # a renderable :class:`StateMerge`, the same combine the cooperative tier uses). A flat zero-axis fold
    # of loop-IR: ``Init`` seeds, the split ``Loop`` (loads + the combine), then the original
    # projection + store.
    other = tuple(f"{nm}__p" for nm in states)
    combine = alg.state_merge(other)
    ids = alg.identities()
    seeds = tuple(Init(name=states[i], identity=ids[states[i]], dtype=F32) for i in range(n_comp))
    loads = tuple(Load(name=other[i], input=ws_name, index=ws_index(i)) for i in range(n_comp))
    # PLANAR for the same reason as the deferred-kernel arm above.
    fin_loop = Loop(axis=split, body=Body((*loads, combine)), role=AxisRole.PLANAR)
    fin_proj, fin_stores = _boundary(after, plain_only=True)
    if not fin_stores and not Body(tuple(fin_proj)).writes:
        # Backward scan — the epilogue tail may end with non-defining stmts (see the deferred
        # kernel arm above).
        out_val = next((s.defines()[-1] for s in reversed(fin_proj) if s.defines()), states[0])
        fin_stores = (Store(write=Write(output=out.name, index=cell, value=out_val)),)
    fin_op = Fold.projection(body=Body((*seeds, fin_loop, *fin_proj)))

    # --- splice the two-kernel fragment in place of the single split TileOp ----------------
    # The finalize is stamped AFTER the workspace joins the fragment: it reads that buffer, and a
    # kernel's structural features fold in its operands' dtypes.
    frag.add_node(op=partial_tile, inputs=list(root.inputs), output=Tensor(ws_name, ws_shape, F32), node_id=ws_name)
    fin_tile = _piece(fin_op, grid, name=tile.name or out.name, stores=fin_stores, graph=frag)
    frag.add_node(op=fin_tile, inputs=[ws_name], output=Tensor(out.name, out.shape, out.dtype), node_id=out.name)
    frag.outputs = [out.name]
    return frag
