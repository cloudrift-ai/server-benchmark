"""Cross-CTA split-reduce (the ``cta`` tier) — consume a ``GRID`` ``ReduceStage``.

A reduce partition with a ``GRID`` stage (``ReducePlan.needs_split``) splits the reduce
axis across CTAs. This pass realizes that split as a **graph rewrite** — the schedule
carries the partition, the graph carries the kernel count. It reads the reduce structure
off the kernel's annotated reduce ``Loop`` (``loop.carrier`` / ``loop.axis``), never an
op-tree node:

- **partial kernel** — the ``cta`` stage becomes an extra grid axis (``_ksplit``); each CTA
  reduces its **contiguous slice** ``[s·B, (s+1)·B)`` of the reduce axis (``B =
  extent / cta``) and contributes its carrier *state* (not the projected output).
- **finalize** — two arms, picked by the ``GRID`` stage's finalize letter
  (``ReducePlan.finalize``):
  - ``"kernel"`` — the partial writes its state to a ``ws[cta, *free]`` ``__partial``
    workspace; a sibling **finalize kernel** seeds the carrier state then folds the
    workspace over the split axis via ``carrier.as_state_merge`` (the cross-partition
    combine, a renderable :class:`StateMerge`) and projects the output. **2 nodes.** The only
    legal arm for a twisted carrier (flash's ``e^{Δm}`` rescale can't be an atomic).
  - ``"atomic"`` — the partial ``atomicAdd``\\ s its (additive) state into the output (applying
    the kernel's projection epilogue per-partition first, when that epilogue *distributes* over
    the add — ``mean``'s ``×1/N``; a non-distributive one like ``l2``'s ``sqrt`` is refused, use
    ``"kernel"``); the output is zero-init'd per launch. **1 node.** Additive carriers only.

So the schedule's GRID stage is **consumed** here (the partial's plan is stripped of it,
the finalize is a fresh ``ReducePlan``); ``lowering/kernel`` only ever sees single-launch
kernels (``assert not needs_split``).

This cut handles **additive** carriers — a degenerate ``PLANAR`` reduce (``sum``) and a
``CONTRACTION`` contraction (split-K matmul), one carrier-state component each — and the twisted
multi-component carrier: **flash split-KV** (:func:`_split_twisted_warp`), where a WARP-tiled
``TWISTED`` streaming tree keeps its fragment residence in the partial (the ``Fold`` axis
shrinks to the slice and its absolute base rides that axis's ``Window``), stores the raw
``(m, l, O)`` state to an f32 workspace, and the finalize folds the partitions through the
exp-family LSE combine before projecting. Pays where the un-split grid starves the SMs (few
heads / short query axis); pin ``REDUCE=g<n>k``.

**Two shapes of contraction split-K.** A structural ``Fold(axis=ksplit,
source=ContractionView(k_axis=kslice))`` (built by ``_schedule._splitk_option``) has its K axis already
factored + operands offset, so :func:`_split_contraction` makes the partial the **bare ContractionView**
— it factorizes to **mma** (or scalar) through ``_factor.factorize``, ``ksplit`` prefixed as a lead
grid axis, no ``_slice_loop``. The residual path below (a plain-sum ``sum`` split, or a coop/ILP
contraction still on a ``Map``) keeps the loop-slicing rewrite.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis, AxisRole, Window
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.schedule import FoldMove, Level
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Body, Init, Load, Loop, Write
from emmy.compiler.ir.tile import (
    Fold,
    Map,
    Placement,
    ReducePlan,
    Store,
    TileOp,
    split_effects,
)
from emmy.compiler.ir.tile.ir import composed_contraction, effect_tail
from emmy.compiler.ir.tile.ops import Sched, lower, nodify_reduce, projection_tail, reduce_loop, reduce_plan, sched_of, seal_workers
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.tile._carrier import projection_distributes as _projection_distributes

PATTERN = [Pattern("root", TileOp)]

_SPLIT = "_ksplit"  # the cross-CTA split grid axis


def _slice_loop(rloop: Loop, b: int) -> Loop:
    """Slice the reduce ``Loop`` to a CTA's contiguous block: offset every reduce-axis load by
    ``_ksplit · B`` (σ on the loop body) and shrink the axis extent to ``B`` (so the loop walks
    ``[0, B)`` while reading ``[s·B, (s+1)·B)``). The carrier (state algebra) rides through
    unchanged — only the operand load indices move."""
    rax = rloop.axis
    offset = BinaryExpr("+", Var(rax.name), BinaryExpr("*", Var(_SPLIT), Literal(b, "int")))
    sigma = Sigma({rax.name: offset})
    ident = lambda n: n  # noqa: E731
    new_body = tuple(s.rewrite(ident, sigma) for s in rloop.body)
    new_ax = replace(rax, extent=Dim(b))
    return Loop(axis=new_ax, body=Body(new_body), unroll=rloop.unroll, role=rloop.role, carrier=rloop.carrier)


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
    kernel's lowered body (``ops.lower`` — the annotated loop nest, ``Map`` / ``Fold`` alike)."""
    for s in stmts:
        if isinstance(s, Write):
            return s.index
    return tuple(Var(ax.name) for ax in grid)


def _strip_grid(plan: ReducePlan) -> ReducePlan:
    """``plan`` with the ``GRID`` stage removed — the partial is a single-launch reduce over
    its slice (the cross-CTA combine is realized by the finalize, not in the partial)."""
    return ReducePlan(tuple(s for s in plan.stages if s.level is not Level.GRID))


def _residual(map_op: Map, plan: ReducePlan) -> tuple[Map | Fold, dict]:
    """The split partial's op + its schedule slices: nodify the flat ``Map``
    (:func:`nodify_reduce` — 1q: the sliced annotated ``Loop`` leaves the flat body for a
    ``Fold`` source, so the partial's fn stays strict) and, when the ``GRID`` strip leaves a
    coop / ILP partition, key the residual plan on the fold in the partial's OWN schedule dict.
    A partial with a prologue ahead of its reduce loop cannot nodify (``nodify_reduce`` asserts
    a head-position loop) — it keeps the raw flat spelling, serial only (as before 1q)."""
    stripped = _strip_grid(plan)
    body = list(map_op.body)
    prologued = not body or not (isinstance(body[0], Loop) and body[0].carrier is not None)
    if prologued and not (stripped.coop > 1 or stripped.reg > 1):
        return map_op, {}
    op2, fold = nodify_reduce(map_op)
    if not (stripped.coop > 1 or stripped.reg > 1):
        return op2, {}
    sched = Sched(op2, {})
    sched.put("REDUCE", fold, stripped)
    return op2, sched.table


def _carrier_identities(carrier) -> dict[str, float]:
    """The per-state-component neutral element (the finalize seeds the state with it before
    folding the partitions). A degenerate carrier reads it off its dissolved ``Accum``\\ s; a
    twisted carrier (flash) off the ``Accum`` folds in its streaming ``merge`` (``l``/``O`` →
    add → 0, ``m`` → maximum → −inf)."""
    accums = carrier.as_accums()
    if accums is not None:
        return {a.name: a.op.identity for a in accums}
    return {s.name: s.op.identity for s in carrier.merge if isinstance(s, Accum)}


def _mapped(op, grid, *, name: str = "", knobs: dict | None = None, free=None, schedule: dict | None = None, stores: tuple = ()) -> TileOp:
    """A **mapped** ``TileOp`` wrapping ``op`` over ``grid`` (so ``_schedule`` skips it).
    ``schedule`` is the kernel's slice dict (1r — the scheduler-resolved operand :class:`Stage`
    and any residual reduce partition, RE-KEYED against the partial's own tree). ``knobs`` carries
    the split node's decided knob row (schedule codecs + stamped ``S_*``) onto the **partial** —
    the engine merges knobs forward on 1:1 rebinds only, so without this the graph splice drops
    them and the deployed split kernels record no schedule identity (the A/B table / ``--json``
    then can't say what greedy deployed, and a golden's ``ShapeKey`` matches no kernel)."""
    place = Placement(free=tuple(free) if free is not None else tuple(grid), grid=tuple(grid))
    out = TileOp(op=op, name=name, place=place, knobs=dict(knobs or {}), schedule=dict(schedule or {}), stores=tuple(stores))
    seal_workers(out)
    return out


def _split_contraction(match: Match, root: Node, tile: TileOp, node, carrier, plan: ReducePlan, split: Axis, projection=()):
    """Realize a **structural** split-K ``Fold(axis=ksplit, step=[ContractionView])`` — the K axis is
    already factored (``split`` == ``ksplit``, extent == ``cta``) and the operands offset, so the
    partial is the **bare ContractionView** with ``ksplit`` prefixed as a lead grid axis (each CTA a fixed
    partition) and its projection retargeted to the workspace / an atomic output. Because the partial
    is a ``ContractionView``, materialize expands it through ``_factor.factorize`` — **mma** for a warp
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
    src_sched = sched_of(tile)
    inner_tile, inner_stage = src_sched.tile_of(node), src_sched.stage_of(node)

    def _partial_sched(partial_op) -> dict:
        """The partial's schedule dict — the sliced contraction's tile / resolved pipeline,
        RE-KEYED against the partial's own tree (the codec keys are tree-relative)."""
        sched = Sched(partial_op, {})
        sched.put("TILE", node, inner_tile)
        sched.put("STAGE", node, inner_stage)
        return sched.table

    states = carrier.state.names
    n_comp = len(states)  # 1 = plain matmul; N = the multi-channel (gate/up) node's per-channel accs
    acc = states[0]
    epilogue = list(projection)  # the fused projection off the ``Map`` wrapper (empty for a bare matmul)

    def _partial(body):
        """The partial kernel's node: the stored fold verbatim, its retargeted stores riding the
        ``Map`` wrapper (the one home for a projection). ``ksplit`` joins as a lead grid axis via
        the partial tile's OWN grid — the view derives lead axes from the placement, so nothing is
        restamped on the node."""
        return Map(body=Body(body), sources=(node,))

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
        p_op = _partial(p_body)
        return _mapped(p_op, (split, *grid), name=tile.name, knobs=tile.knobs, schedule=_partial_sched(p_op), stores=p_stores)

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
    p_op = _partial(())
    partial_tile = _mapped(
        p_op, (split, *grid), name=f"{tile.name}__partial", knobs=tile.knobs, schedule=_partial_sched(p_op), stores=ws_stores
    )

    # --- finalize kernel: seed each state, fold ``ws`` over ``ksplit`` (``as_state_merge`` — the
    # 1-component additive fold, or the N-component per-channel sums), then the original
    # projection epilogue (the multi-channel ⊗-combine applies HERE, once, after the sums) or a
    # bare store — the same finalize shape the residual path uses. The seeds + merge loop are
    # raw loop IR (the finalize is not a recognized term); its root store rides ``TileOp.stores``.
    other = tuple(f"{nm}__p" for nm in states)
    combine = carrier.as_state_merge(other)
    ids = _carrier_identities(carrier)
    seeds = tuple(Init(name=nm, identity=ids[nm], dtype=F32) for nm in states)
    loads = tuple(Load(name=other[i], input=ws_name, index=ws_index(i)) for i in range(n_comp))
    fin_loop = Loop(axis=split, body=Body((*loads, combine)))
    fin_proj, fin_stores = _boundary(epilogue)
    if not fin_stores and not any(isinstance(s, Write) for s in fin_proj):
        # The projected value is the LAST defining stmt's name — the epilogue tail may end with
        # non-defining stmts (e.g. a sunk RowAccum), so scan backward instead of indexing [-1].
        out_val = next((s.defines()[-1] for s in reversed(fin_proj) if s.defines()), acc)
        fin_stores = (Store(write=Write(output=out.name, index=cell, value=out_val)),)
    fin_op = Map(body=Body((*seeds, fin_loop, *fin_proj)))
    fin_tile = _mapped(fin_op, grid, name=tile.name, stores=fin_stores)

    frag = Graph()
    for inp in root.inputs:
        frag.add_node(op=InputOp(), inputs=[], output=match.graph.buffer(inp), node_id=inp)
    frag.add_node(op=partial_tile, inputs=list(root.inputs), output=Tensor(ws_name, ws_shape, F32), node_id=ws_name)
    frag.add_node(op=fin_tile, inputs=[ws_name], output=Tensor(out.name, out.shape, out.dtype), node_id=out.name)
    frag.outputs = [out.name]
    return frag


def _split_twisted_warp(match: Match, root: Node, tile: TileOp, op: Map, plan: ReducePlan, split: Axis):
    """Realize flash split-KV: a **warp-tiled** ``TWISTED`` streaming tree whose GRID stage slices
    the kv stream across ``cta`` CTAs. The partial keeps the WHOLE fragment-resident tree — the
    ``Fold`` gets its axis shrunk to the slice length ``B`` and its absolute base on
    ``offset`` (``_ksplit · B``; the realizer walks the local ``[0, B)`` window and re-bases the
    gmem/TMA coords + causal mask, composing with the causal tile-skip) — and stores the RAW
    ``(m, l, O)`` state: O per output cell, the d-invariant row stats once per row at the last
    slot's origin, into an **f32** workspace (the pre-projection state must not round-trip
    through the output dtype). The finalize folds the workspace over the split axis via the
    carrier's ``as_state_merge`` (the LSE cross-partition combine) and runs the ORIGINAL
    projection (``O/l`` + the layout-aware store) per output element. Deferred-kernel finalize
    only — the twisted ``e^{Δm}`` rescale can't be an atomic."""
    (red,) = op.sources
    head = red.step_stmts()[0]  # the role=CONTRACTION score fold (the hoisted operand edge) — its tile is the schedule read
    carrier = red.carrier
    cta = plan.cta
    src_sched = sched_of(tile)
    head_tile = src_sched.tile_of(head)
    atom_n = head_tile.atom.shape[1]
    bn = head_tile.regs[1] * atom_n
    ext = red.axis.extent
    if ext.is_static:
        b_dim = Dim(ext.as_static() // cta)
        if ext.as_static() % cta != 0:
            raise NotImplementedError(f"flash split-KV needs a divisible kv extent ({ext.as_static()} % cta {cta})")
        if b_dim.as_static() % bn != 0:
            raise NotImplementedError(f"flash split-KV slice ({b_dim.as_static()}) must be divisible by the streaming key block ({bn})")
        bound = None  # the shrunk static extent alone bounds the slice
    else:
        # Symbolic kv: the slice width is the bn-ALIGNED ceil split ``B = ceil(S / (cta·bn)) · bn``
        # (a composite Dim — every slice base lands block-aligned, so only each slice's own tail
        # chunk is partial), and ``bound = min((s+1)·B, S)`` is the slice's absolute end — the
        # realizer stops the stream at ``bound − offset`` and masks/clamps against it (a mid-tensor
        # slice end reads VALID next-slice keys the extent-only tail masks would keep). A slice
        # wholly past ``S`` runs zero steps and contributes the carrier identities, which the
        # finalize's LSE combine folds as exact no-ops.
        b_dim = ext.ceil_div(cta * bn) * bn
        nxt = BinaryExpr("*", BinaryExpr("+", Var(_SPLIT), Literal(1, "int")), b_dim.expr)
        bound = TernaryExpr(cond=BinaryExpr("<", nxt, ext.expr), if_true=nxt, if_false=ext.expr)
    (cross_move,) = next(st for st in plan.stages if st.level is Level.GRID).combine(warp_size=32)
    if cross_move is FoldMove.ATOMIC:
        raise NotImplementedError("the twisted (m, l, O) state can't finalize atomically; pin the deferred kernel (REDUCE=g<n>k)")

    out = root.output
    grid = tile.place.grid  # (batch…, m_blocks) — the warp placement's shrunk query axis, d folded away
    names = carrier.state.names
    channels = carrier.twist.channels
    expect = next(n for n, ch in zip(names, channels, strict=True) if ch.lift is not None)
    n_comp = len(names)
    # The query / value axes are PLACEMENT facts — the pre-split tile's trailing free axes (the
    # warp placement preserves ``free`` while its grid shrinks the query axis and folds ``d`` away).
    m_axis, d_axis = tile.place.free[-2], tile.place.free[-1]
    batch = tuple(a for a in grid[:-1])

    # The workspace is OURS (bare ``(batch…, m, d)`` order — the fused output layout only matters
    # at the finalize's store); f32 so the pre-projection state keeps the stream's precision.
    ws_name = f"{out.name}__partial"
    ws_shape = (Dim(n_comp), Dim(cta), *(a.extent for a in batch), m_axis.extent, d_axis.extent)

    def ws_index(i: int, name: str) -> tuple:
        cell = (Var(m_axis.name), Var(d_axis.name) if name == expect else Literal(0, "int"))
        return (Literal(i, "int"), Var(split.name), *(Var(a.name) for a in batch), *cell)

    off = BinaryExpr("*", Var(split.name), b_dim.expr)
    # The slice rides the AXIS's window — its absolute base / end in the un-split stream.
    sliced = replace(red, axis=replace(red.axis, extent=b_dim, window=Window(parent=red.axis, base=off, bound=bound)))
    ws_stores = tuple(Store(write=Write(output=ws_name, index=ws_index(i, names[i]), value=names[i])) for i in range(n_comp))
    partial_map = Map(body=Body(()), sources=(sliced,))
    # The partial's slices RE-KEY against its own tree: the QK/PV warp plans + the stream's
    # resolved K/V pipeline carry over; the GRID stage is consumed (any residual partition would
    # re-key here too, but the stream's plan is GRID-only by construction).
    p_sched = Sched(partial_map, {})
    sliced_head = sliced.step_stmts()[0]
    sliced_pv = next(st for st in list(sliced.step_stmts())[1:] if getattr(st, "role", None) is AxisRole.CONTRACTION)
    orig_pv = next(st for st in list(red.step_stmts())[1:] if getattr(st, "role", None) is AxisRole.CONTRACTION)
    p_sched.put("TILE", sliced_head, head_tile)
    p_sched.put("TILE", sliced_pv, src_sched.tile_of(orig_pv))
    p_sched.put("STAGE", sliced, src_sched.stage_of(red))
    stripped = _strip_grid(plan)
    if stripped.stages:
        p_sched.put("REDUCE", sliced, stripped)
    # The partial's grid gains ``_ksplit`` and keeps the shrunk query axis; its FREE axes keep the
    # true ``(m, d)`` tail so the fragment realizer re-derives the views (``Ctx.free``).
    partial_tile = _mapped(
        partial_map,
        (split, *grid),
        name=f"{tile.name}__partial",
        knobs=tile.knobs,
        free=(split, *grid[:-1], m_axis, d_axis),
        schedule=p_sched.table,
        stores=ws_stores,
    )

    # Finalize: per output element, seed the state, fold the partitions (the exp-family LSE
    # combine), then the original projection + layout-aware store (``op.body`` verbatim).
    other = tuple(f"{nm}__p" for nm in names)
    combine = carrier.as_state_merge(other)
    ids = _carrier_identities(carrier)
    seeds = tuple(Init(name=nm, identity=ids[nm], dtype=F32) for nm in names)
    loads = tuple(Load(name=other[i], input=ws_name, index=ws_index(i, names[i])) for i in range(n_comp))
    fin_loop = Loop(axis=split, body=Body((*loads, combine)))
    fin_op = Map(body=Body((*seeds, fin_loop, *op.body)))
    fin_tile = _mapped(fin_op, (*batch, m_axis, d_axis), name=tile.name, stores=tile.stores)

    frag = Graph()
    for inp in root.inputs:
        frag.add_node(op=InputOp(), inputs=[], output=match.graph.buffer(inp), node_id=inp)
    frag.add_node(op=partial_tile, inputs=list(root.inputs), output=Tensor(ws_name, ws_shape, F32), node_id=ws_name)
    frag.add_node(op=fin_tile, inputs=[ws_name], output=Tensor(out.name, out.shape, out.dtype), node_id=out.name)
    frag.outputs = [out.name]
    return frag


def rewrite(match: Match, root: Node) -> TileOp | Graph | None:
    tile: TileOp = root.op
    # The reduce partition lives on the Fold node (off the schedule) — ``reduce_plan`` reads
    # it there, falling back to the ``TileOp``'s residual ``reduce`` field for a non-tiled
    # contraction's split-K (still a Map).
    plan = reduce_plan(tile) if tile.op is not None else None
    if plan is None or not plan.needs_split:
        raise RuleSkipped("no cross-CTA split stage — nothing to split")

    op = tile.op
    rloop = reduce_loop(op)
    carrier = rloop.carrier
    cta = plan.cta
    rax = rloop.axis
    # Structural split-K: ``op`` is ``Fold(axis=ksplit, step=[ContractionView(k_axis=kslice)])`` —
    # the axis is already factored + operands offset (``_schedule._splitk_option``), so the partial
    # is the **bare ContractionView** (→ ``factorize`` → mma / scalar), no ``_slice_loop``.
    # The projection (when the split node carries one) rides the ``Map`` wrapper over the split
    # ``Fold`` — its ONE home; peel it here and hand it to the realizer.
    # The projection with the kernel-boundary stores reconstituted (1q) — the split realizers
    # retarget the root ``Write`` exactly as when it rode the ``Map`` body. A projection whose
    # only stmt was the root ``Write`` leaves a BARE fold behind (the ``Map`` wrapper dropped
    # with its last in-body stmt), so the reconstitution keys off the TileOp, not the node shape.
    split_root = op.sources[0] if isinstance(op, Map) and op.sources else op
    projection = tuple(projection_tail(tile))
    # The split node's inner contraction — multi-channel included — rides the outer reduce's
    # identity-lift operand composition (the one composition rule; ``ir.composed_contraction``).
    inner = composed_contraction(split_root)
    if inner is not None and sched_of(tile).tile_of(inner) is not None:
        return _split_contraction(match, root, tile, inner, carrier, plan, rax, projection)
    # Flash split-KV: a warp-tiled TWISTED streaming tree keeps its fragment residence in the
    # partial (the scalar residual path below would drop it to the per-cell tier).
    if isinstance(op, Map) and len(op.sources) == 1 and isinstance(op.sources[0], Fold) and op.sources[0].role is AxisRole.TWISTED:
        red_stmts = op.sources[0].step_stmts()
        head = red_stmts[0] if len(red_stmts) else None
        head_tile = sched_of(tile).tile_of(head) if isinstance(head, Fold) else None
        if head_tile is not None and head_tile.is_warp:
            # A symbolic kv splits too: ``_split_twisted_warp`` builds the bn-aligned runtime slice
            # width and the absolute ``bound`` the realizer stops/masks against.
            return _split_twisted_warp(match, root, tile, op, plan, Axis(name=_SPLIT, extent=Dim(cta)))
    if not rax.extent.is_static:
        raise NotImplementedError(
            "cross-CTA split of a symbolic reduce axis is not built yet (warp flash split-KV above is the one built symbolic arm)"
        )
    extent = rax.extent.as_static()
    if extent % cta != 0:
        raise NotImplementedError(f"cross-CTA split needs a divisible reduce axis (extent {extent} % cta {cta})")
    b = extent // cta
    states = carrier.state.names
    n_comp = len(states)

    out = root.output
    grid = tile.place.grid
    # The lowered loop nest (``Map`` / ``Fold`` alike) — find the carrier-bearing reduce loop
    # in it by position (``reduce_loop`` returns a fresh synthesized loop for a ``Fold``, so key
    # off the lowered list, not object identity).
    stmts = effect_tail(lower(op), tile.stores)  # boundary stores reconstituted — ``after`` keeps its Write
    cell = _cell_index(stmts, grid)
    split = Axis(name=_SPLIT, extent=Dim(cta))
    idx = next(i for i, s in enumerate(stmts) if isinstance(s, Loop) and s.carrier is not None)
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
        atomic_op = Map(body=Body((*before, sliced_loop, *proj_body)))
        res_op, res_sched = _residual(atomic_op, plan)
        return _mapped(res_op, (split, *grid), name=tile.name, knobs=tile.knobs, schedule=res_sched, stores=proj_stores)

    # The ``__partial`` workspace packs every carrier-state component: ``ws[comp, cta, *free]``
    # (the ``comp`` leading axis dropped for a single-component additive carrier, so the
    # additive workspace stays ``ws[cta, *free]``). A multi-component (twisted flash) carrier
    # writes its ``(m, l, O)`` state to the three ``comp`` slices, no multi-output kernel.
    ws_name = f"{out.name}__partial"
    ws_shape = (Dim(n_comp), Dim(cta), *out.shape) if n_comp > 1 else (Dim(cta), *out.shape)

    def ws_index(i: int) -> tuple:
        lead = (Literal(i, "int"), Var(_SPLIT)) if n_comp > 1 else (Var(_SPLIT),)
        return (*lead, *cell)

    # --- partial kernel: reduce a CTA's slice, write its carrier state to the workspace -----
    ws_stores = tuple(Store(write=Write(output=ws_name, index=ws_index(i), value=states[i])) for i in range(n_comp))
    partial_op = Map(body=Body((*before, sliced_loop)))
    res_op, res_sched = _residual(partial_op, plan)
    partial_tile = _mapped(res_op, (split, *grid), name=f"{tile.name}__partial", knobs=tile.knobs, schedule=res_sched, stores=ws_stores)

    # --- finalize kernel: seed the carrier state, then fold each partition's state from the
    # workspace over the split axis via the carrier's ``as_state_merge`` (a renderable
    # :class:`StateMerge`, the same combine the cooperative tier uses). A flat ``Map`` of loop-IR:
    # ``Init`` seeds, the split ``Loop`` (loads + the combine), then the original projection + store.
    other = tuple(f"{nm}__p" for nm in states)
    combine = carrier.as_state_merge(other)
    ids = _carrier_identities(carrier)
    seeds = tuple(Init(name=states[i], identity=ids[states[i]], dtype=F32) for i in range(n_comp))
    loads = tuple(Load(name=other[i], input=ws_name, index=ws_index(i)) for i in range(n_comp))
    fin_loop = Loop(axis=split, body=Body((*loads, combine)))
    fin_proj, fin_stores = _boundary(after, plain_only=True)
    if not fin_stores and not Body(tuple(fin_proj)).writes:
        # Backward scan — the epilogue tail may end with non-defining stmts (see the deferred
        # kernel arm above).
        out_val = next((s.defines()[-1] for s in reversed(fin_proj) if s.defines()), states[0])
        fin_stores = (Store(write=Write(output=out.name, index=cell, value=out_val)),)
    fin_op = Map(body=Body((*seeds, fin_loop, *fin_proj)))
    fin_tile = _mapped(fin_op, grid, name=tile.name, stores=fin_stores)

    # --- splice the two-kernel fragment in place of the single split TileOp ----------------
    frag = Graph()
    for inp in root.inputs:
        frag.add_node(op=InputOp(), inputs=[], output=match.graph.buffer(inp), node_id=inp)
    frag.add_node(op=partial_tile, inputs=list(root.inputs), output=Tensor(ws_name, ws_shape, out.dtype), node_id=ws_name)
    frag.add_node(op=fin_tile, inputs=[ws_name], output=Tensor(out.name, out.shape, out.dtype), node_id=out.name)
    frag.outputs = [out.name]
    return frag
