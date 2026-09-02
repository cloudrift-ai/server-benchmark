"""The factorizer — the recursive ``TileOp`` emitter and the ONE root-binding pipeline every
scheduled Fold seals through. Compatible independent roots each use that pipeline, then share one
physical grid. The per-atom codegen **strategies** it drives live in ``_atom.py``, and the
axis-realization layer it seals through in ``_tiling.py``.

:func:`factorize` is the entry ``010_materialize`` calls once per kernel: it builds the ambient
:class:`Ctx` and dispatches ``tile.op`` through the recursion :func:`_factorize`, which walks the
``Fold`` node node tree. A :class:`~...ir.Fold` with a ``source``
**recurses** (its projection walked into the ``tail``); each leaf binds to the grid via the single
:func:`_bind` pipeline, whose form is read off the node's SCHEDULE — which axes are tiled — never a
kernel kind: a tiled :class:`~...ir.bilinear fold` tiles its OUTPUT ``(m, n)`` axes (register / warp
cells), a cooperating :class:`~...ir.Fold` tiles its REDUCE axis (:func:`_tile_reduce_axis` —
``coop`` lanes + ``reg`` ILP chains), and everything else tiles nothing (the degenerate
one-thread-per-cell fold). All three seal through the one :func:`grid_tile` finalizer; the per-cell
body is the term's own lowering (``Fold.lower``), never a second walk.

The output tiling reads its **geometry straight off the** contraction **node** (``tile_m`` /
``mask_m`` / ``m_b`` / ``block_threads`` / …, derived there from the ``tile`` schedule + the output
axes), expands both atoms through the *same* four-level tiling pipeline (``atomize →
register_tile → unit_tile → grid_tile``, in **``_tiling.py``** — the algebra-free layer that turns
the schedule's plan into bound ``Axis`` objects), and splices in two codegen halves from
the per-atom strategies in **``_atom.py``**: :func:`~...kernel._atom.reduce_codegen` — the reusable,
**sink-agnostic** ``(state_decls, reduce_region)`` (accumulator/operand decls + the contraction
K-loop) — and a per-cell **sink** ``store(i, j, offset, mn)`` (default
:func:`~...kernel._atom.store_sink`, the matmul sink). A caller may replace the sink while reusing
the same ``reduce_codegen``.

The reduce-axis tiling (:func:`_tile_reduce_axis` + the shared-row staging apply) folds the reduce
axis ``coop`` ways across threads and ``reg`` ways across per-thread accumulators, then the
REG-tree fold, the cross-thread combine (:func:`emit_combine`), and the projection — algebra-
generic through the :class:`Reduction` view (a contraction is the degenerate algebra of its
additive fold).

The smem operand-staging pipeline lives in ``_stage.py`` (the :class:`~...kernel._stage.Transport`
strategy + the one :func:`~...kernel._stage.staged_kloop`); the ONE atom-agnostic driver
(``_atom._staged``) builds the transport, the atom strategy supplying only the slab drain leaf.
It is driven off the node's ``STAGE`` codec →
:class:`~...schedule.Stage` (``d<depth>`` gmem→smem ring · ``smem``/``smem-async``/``smem-tma`` transport ·
``p<n>`` smem→register double-buffer). The **scalar** contraction tier stays gmem-direct. The fused
norm→linear **shared-row** prologue is Stage-driven too: the schedule detects the reused input row
and stamps an ``smem`` :class:`~...schedule.Stage` whose slab list names it; :func:`_tile_reduce_axis` only
applies it (the 1-D ``sync_row_fill`` + the load rewrite). Leading ``_`` so the pass loader skips this
module."""

from __future__ import annotations

from dataclasses import dataclass, replace

from emmy.compiler.backend.cuda.dtype import cuda_name
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel import Tile
from emmy.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule import Raster
from emmy.compiler.ir.schedule.views import cone_seam
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Body, Cond, Init, Load, Loop, Select, SelectBranch, Stmt, StridedLoop, Write
from emmy.compiler.ir.tile import FoldMove, Level, Reduce, ReduceStage
from emmy.compiler.ir.tile.ir import apply_output_specs, observed_result_names
from emmy.compiler.ir.tile.ops import UnbindableProjection, projection_regions, projection_root, sched_of
from emmy.compiler.pipeline.passes.lowering._reduction import Reduction
from emmy.compiler.pipeline.passes.lowering.kernel._atom import (
    clamp_last,
    copy_cell,
    fold_store_sink,
    fold_store_tail,
    reduce_codegen,
    scheduled_fold_contraction,
    store_sink,
)
from emmy.compiler.pipeline.passes.lowering.kernel._stage import sync_row_fill
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import atomize, grid_tile, register_tile, unit_tile

# ---- the ambient cell environment and the wire a node produces ---------------------------------- #


@dataclass(frozen=True)
class Handle:
    """A produced tensor a parent wires up — "a value that needs wiring." ``name`` is the SSA name
    holding it; ``residence`` is HOW a consumer reads it (``reg`` = a scalar register value today;
    the tensor-core rebuild adds ``reg_frag`` — an mma fragment — plus a fragment descriptor
    ``(mma_role, shape, dtype)`` and the accumulator→operand recast at a node boundary)."""

    name: str
    residence: str = "reg"


@dataclass(frozen=True)
class Ctx:
    """The ambient cell environment threaded DOWN the recursion — established once for the whole
    kernel and passed unchanged so every node reads/writes at the same output cell. ``grid`` is the
    kernel's grid axes; ``inputs`` the operand buffer table (dtype/shape); ``output`` the root
    output buffer name. The operand smem pipeline is NOT here — it rides the node it decorates
    (the ``STAGE`` slice)."""

    grid: tuple
    inputs: dict | None = None
    output: str = ""
    workers: object = None  # the resolved WarpSpec worker split (None = uniform SIMT)
    raster: object = None  # the parsed RASTER codec (ir.schedule.Raster; None = flat launch order)
    # The accepted classic assignment bound to its Fold sites (read through ``ops.Sched``): all
    # per-node tile/reduce and per-edge stage reads go through here; the term stores no choices.
    sched: object = None
    # The placement's FREE axes — the un-shrunk originals. A split partial may prefix ``_ksplit``;
    # contraction views derive their output axes from the trailing pair.
    free: tuple = ()


def _wire(op: Fold) -> Handle:
    """The produced-value :class:`Handle` of a node — its primary exposed name: the carrier state's
    first component, a contraction's primary accumulator, a projection's result. A wrapper that
    exposes nothing surfaces its first operand's."""
    if op.exposes:
        return Handle(op.exposes[0])
    return _wire(op.operands[0]) if op.operands else Handle("")


def factorize(tile, root, store=None) -> Tile:
    """The entry to the recursive emitter — build the ambient :class:`Ctx` from the ``TileOp`` and its
    root graph node, then dispatch its ``op`` into a bound ``Tile`` via :func:`_factorize`. ``out_val``
    (the kernel's finalized output SSA name — the root node's produced :class:`Handle`) is resolved
    once here and threaded down for the store glue."""
    # Stored trees are already resolved — a computed operand is an inline node on its edge, so the
    # emitter below walks the tree as stored and every reader (``cone_seam``) reads the node
    # boundary straight off ``Fold.a``.
    # An empty schedule fork deliberately leaves the term unmapped. Materialization is the
    # guardrail's scalar fallback: bind one thread to every free-axis cell instead of carrying an
    # empty grid into loop lowering while the body still references those coordinates.
    if not tile.place.is_mapped:
        tile = replace(tile, place=tile.place.on_grid())

    op = tile.op
    ctx = Ctx(
        grid=tuple(tile.place.grid),
        inputs=tile.inputs,
        output=(root.output.name if root is not None else ""),
        workers=tile.workers,
        free=tuple(tile.place.free),
        sched=sched_of(tile),
        # The launch-order codec rides the TileOp's stamped knobs (kernel-scoped metadata, no
        # per-node structure) — parse it once here; ``grid_tile`` applies it where the 2-D
        # (m, n) block grid makes it meaningful.
        raster=Raster.parse((tile.knobs or {}).get("RASTER", "")),
    )
    out_val = _wire(op).name if op is not None else ""
    return _factorize(op, ctx, tail=(), out_val=out_val, store=store, output_specs=tuple(tile.output_specs))


def _factorize(op, ctx: Ctx, tail: tuple, out_val: str, store=None, output_specs: tuple = ()) -> Tile:
    """The recursive root walk — peel the projecting zero-axis ``Fold``\\ s, then bind each leaf to the grid via
    the ONE binding pipeline. A zero-axis :class:`Fold` with an operand recurses: its ``body`` (the projection /
    epilogue) is emitted as the term's own lowering, the kernel-boundary
    output specifications reconstituted into it (``apply_output_specs``), and the result prepended to ``tail``;
    everything else is a leaf, bound by :func:`_bind` — the single pipeline, whose form is read off
    the node's SCHEDULE (which axes are tiled), never a kernel kind. Nested scheduled contractions
    and their enclosing carrier factorize through this same walk."""
    if (isinstance(op, Fold) and op.axis is None) and op.operands:
        # An output-tiled root may sit under its sweep's epilogue projection (``projection_root``).
        tiled = [
            edge
            for edge in op.operands
            if (root := projection_root(edge)) is not None and root.as_contraction() is not None and ctx.sched.tile_of(root) is not None
        ]
        if len(tiled) > 1:
            return _bind_roots(op, ctx, output_specs)
        root = tiled[0] if tiled else op.operands[0]
        siblings = [stmt for edge in op.operands if edge is not root for stmt in edge.lower()]
        proj = list(dict.fromkeys([*siblings, *op.lift.body]))
        # A STREAMED store (values = an observer's results) rides the recursion down to the leaf
        # so the scalar arm can splice it into the observed fold's reduce loop — applying it here
        # would land it in the projection tail, after a loop that is not yet emitted.
        observed = observed_result_names(op)
        streamed = tuple(spec for spec in output_specs if set(spec.write.values) <= observed)
        streamed_ids = {id(spec) for spec in streamed}
        plain = tuple(spec for spec in output_specs if id(spec) not in streamed_ids)
        # An output sweep whose axis the peeled root's cone reads cannot wrap here: the peel binds
        # the root OUTSIDE the projection, so no wrap position inside ``proj`` encloses the reduce
        # and the sweep coordinate renders as an undefined identifier at nvcc (DeepSeek-V4's fused
        # ``k_div_36_reduce``, whose per-column mean read the store sweep's column). The serial
        # fold realizes the shape whole — bind the UNPEELED projection so ``apply_output_specs``
        # wraps operand and projection together. A cooperative / ILP partition (or an output-tiled
        # contraction root) has no such realization — the sweep is distributed across the lanes it
        # would have to re-run on — so the row is declined and the greedy retries the next one.
        sweeps = tuple(spec.sweep.name for spec in plain if spec.sweep is not None)
        free = frozenset(sweeps) & root.free_axes
        swept = [name for name in sweeps if name in free]
        if swept:
            # The schedule at stake is the ITERATING node's, which a chain of zero-axis projections
            # may sit above (``root`` is then a projection, carrying no ``REDUCE`` site of its
            # own). Descend to it, or the decline reads an absent plan off the wrapper and binds a
            # partitioned row as the serial fold — the emission is correct but silently drops the
            # partition the row was priced on.
            node = root
            while isinstance(node, Fold) and node.axis is None and node.operands:
                node = node.operands[0]
            plan = ctx.sched.get("REDUCE", node) if isinstance(node, Fold) and node.axis is not None else None
            if (plan is not None and (plan.coop > 1 or plan.reg > 1)) or (
                node.as_contraction() is not None and ctx.sched.tile_of(node) is not None
            ):
                raise UnbindableProjection(
                    f"the bound reduce's cone reads output sweep axis {swept[0]!r} — a cooperative / ILP "
                    f"partition cannot re-run the reduce per swept cell"
                )
            return _bind(op, ctx, tail, out_val, store, output_specs=output_specs)
        if plain:
            proj = apply_output_specs(proj, plain)
        return _factorize(root, ctx, tail=(*proj, *tail), out_val=out_val, store=store, output_specs=streamed)
    if output_specs and isinstance(op, Fold) and op.axis is None:
        # A zero-axis root with no operand edge still owns a real projection body. Reconstitute
        # its output specifications only after that body is emitted so an output sweep wraps every stmt
        # that reads the sweep coordinate.
        return _bind(op, ctx, tail, out_val, store, output_specs=output_specs)
    if output_specs:
        # A non-projection flat root can carry plain root ``Write``\\ s only. A STREAMED store
        # (its values an observer's results) stays a spec so the scalar arm splices it into the
        # observed fold's reduce loop; the rest append as the kernel tail.
        assert all(st.sweep is None for st in output_specs), "sweep stores ride a projecting zero-axis fold"
        observed = observed_result_names(op)
        streamed = tuple(st for st in output_specs if set(st.write.values) <= observed)
        streamed_ids = {id(st) for st in streamed}
        tail = (*tail, *(st.write for st in output_specs if id(st) not in streamed_ids))
        return _bind(op, ctx, tail, out_val, store, output_specs=streamed)
    return _bind(op, ctx, tail, out_val, store)


def _merge_root_tiles(tiles: tuple[Tile, ...]) -> Tile:
    """Merge independently bound regions that use one physical grid and worker inventory."""
    first = tiles[0]
    axes = {axis.name: axis for axis in first.axes}
    for tile in tiles[1:]:
        current = {axis.name: axis for axis in tile.axes}
        if current != axes:
            raise ValueError("output-tiled roots disagree on their physical grid")
        if (tile.block_threads, tile.aux_threads) != (first.block_threads, first.aux_threads):
            raise ValueError("output-tiled roots disagree on their worker inventory")

    local = {}
    body = []
    top_defs = set()
    for tile in tiles:
        for stmt in tile.body:
            declarations = stmt.local_decls()
            if declarations:
                prior = tuple(local.get(name) for name in declarations)
                if any(previous is not None and previous != stmt for previous in prior):
                    conflict = next(name for name, previous in zip(declarations, prior, strict=True) if previous not in (None, stmt))
                    raise ValueError(f"output-tiled roots require incompatible local buffer {conflict!r}")
                if all(previous is not None for previous in prior):
                    continue
                for name in declarations:
                    local[name] = stmt
            overlap = top_defs & set(stmt.defines())
            if overlap:
                raise ValueError(f"output-tiled roots reuse top-level SSA names: {sorted(overlap)}")
            top_defs.update(stmt.defines())
            body.append(stmt)
    return replace(first, body=Body(body))


def _bind_roots(op: Fold, ctx: Ctx, output_specs: tuple) -> Tile:
    """Bind compatible independent contraction roots separately, then share their physical grid.
    A root's tail is its region's projection: the epilogue term over it (its other operands
    emitted ahead of its step) and the root-body statements its outputs read."""
    tiles = []
    for index, (root, region, body, owned_specs) in enumerate(projection_regions(op, output_specs)):
        epilogue: list[Stmt] = []
        if region is not root:
            epilogue = [*(s for edge in region.operands if edge is not root for s in edge.lower()), *region.step()]
        tail = tuple(apply_output_specs([*epilogue, *body], owned_specs))
        tiles.append(_bind(root, ctx, tail, root.exposes[0], frag_ns=f"_r{index}"))
    return _merge_root_tiles(tuple(tiles))


def has_write(stmts: list[Stmt]) -> bool:
    """Any ``Write`` reachable in ``stmts`` (deep — a projection's output sweep nests its
    ``Write`` inside a per-cell ``Loop``)."""
    for s in stmts:
        if isinstance(s, Write):
            return True
        if any(has_write(list(b)) for b in s.nested()):
            return True
    return False


def with_store(stmts: list[Stmt], output: str, grid, value: str) -> list[Stmt]:
    """Append the output-store glue when the body has none — a bare reduction / contraction produces
    its finalized value as the SSA name ``value`` (the carrier state / accumulator, or a projection's
    last def) that must be written to the output buffer at the grid cell. A body that already carries
    a ``Write`` needs no glue (``value`` is left unread). The caller resolves ``value`` off the node
    (the root's primary exposed name / the recursion's produced ``Handle``) so this helper stays node-agnostic."""
    if has_write(stmts):
        return stmts
    index = tuple(Var(ax.name) for ax in grid)
    return [*stmts, Write(output=output, index=index, value=value)]


def _bind(op, ctx: Ctx, tail: tuple, out_val: str, store=None, *, output_specs: tuple = (), frag_ns: str = "") -> Tile:
    """The ONE root binder — every kernel binds through the same pipeline: read WHICH AXES the
    schedule tiles off the node, build the fold region, and seal through the one :func:`grid_tile`
    finalizer. The cases are points of one ``(output-tiling) × (reduce-folding)`` space, selected by
    the schedule — never separate emitters:

    - a contraction tiles its OUTPUT ``(m, n)`` axes — register / warp cells through
      ``atomize → register_tile → unit_tile``, the reduce (K) serial per cell from the atom's
      :func:`reduce_codegen`, and ``store`` the per-cell sink (default :func:`store_sink`). Its projection
      arrives as ``tail`` — peeled off the wrapping zero-axis fold, the ONE home for a projection; the bare
      grid-``Write`` glue is synthesized here (it needs ``ctx.output``, so it can't ride the node).
    - a :class:`Fold` whose :class:`Reduce` cooperates tiles its REDUCE axis instead
      (:func:`_tile_reduce_axis` — ``coop`` lanes at the unit level, ``reg`` ILP chains at the
      register level, the carrier merge closing the fold). The output stays one cell per thread:
      the 1×1 ``atomize`` with the whole grid riding ``lead_axes`` untiled.
    - anything else (a pure pointwise zero-axis fold, a trivial plan) tiles NOTHING — the degenerate
      one-thread-per-cell fold: the per-cell body (``op.lower()``; a serial reduce ``Loop`` sits
      inside it) + ``tail`` + the ``out_val`` store glue is the whole fold region."""
    grid = tuple(ctx.grid)
    # The OUTPUT-tiled dispatch: contraction whose schedule holds a TILE slice, over a
    # grid with an ``(m, n)`` pair to place it on. The node is pure algebra; the tiled reading comes
    # off the slice, which arrives ALREADY PLACED from ``Sched.tile_of`` (the ``(m, n)`` pair is a
    # function of the site, so the binding lives on the scheduling structure, not here) — the
    # geometry the atom reads is the slice's own, not a separate view object's. A stored node
    # WITHOUT a TILE slice takes the reduce tiers instead (the per-cell / coop-K forms), where the
    # whole grid rides untiled.
    folded = scheduled_fold_contraction(op, ctx.sched) if isinstance(op, Fold) and op.axis is not None else None
    if folded is not None:
        c, value_child, tile, stage = folded
        projection = tail
        tail = fold_store_tail(tail, op, c)
    else:
        c, value_child, projection = op, None, ()
        tile = ctx.sched.tile_of(op) if isinstance(op, Fold) and op.as_contraction() is not None else None
        stage = ctx.sched.get("STAGE", op) if tile is not None else None
    if tile is not None and tile.axes is not None and len(grid) >= 2:
        epi = list(tail)
        if not has_write(epi):
            epi = with_store(epi, ctx.output, grid, c.exposes[0])
        # The cone's K seam, read straight off the inline operand node (``None`` for a gmem-``Load``
        # A — its whole body is the per-cell fill).
        seam = cone_seam(c.operands[0], c.axis.name) if value_child is None and c.operands[0].as_slab() is None else None
        # The leading (batch / ksplit) grid axes ride untiled below the ``(m, n)`` cell — the GRID's
        # fact, not the tiled cell's, so they are threaded to the emission that needs them (the
        # per-cell rename's shared coordinates) from here, where the kernel grid is in hand.
        lead = grid[:-2]
        carried = {}
        state_decls, reduce_region = reduce_codegen(
            c,
            tile,
            stage,
            ctx.inputs,
            ctx.workers,
            seam,
            lead,
            frag_ns,
            fold=op if value_child is not None else None,
            value_child=value_child,
            sched=ctx.sched,
            projection=projection,
            carried=carried,
        )
        if store is not None:
            sink = store
        elif value_child is not None:
            sink = fold_store_sink(tile, tuple(epi), carried, frag_ns)
        else:
            sink = store_sink(c, tile, Body(tuple(epi)), lead, frag_ns)
        t = unit_tile(register_tile(atomize(tile.atom.shape[:2]), tile.mn), tile.mn)
        mn, bt, lanes = tile.mn, tile.launch_threads, tile.atom.lanes
    else:
        # The reduce partition rides the :class:`Fold` node; ``None`` for a pure pointwise /
        # scalar per-cell zero-axis ``Fold`` (no partition). Every partitioned reduction is a
        # ``Fold`` node (a projecting zero-axis
        # ``Fold`` was already peeled off by :func:`_factorize`).
        plan = (ctx.sched.get("REDUCE", op) or Reduce()) if isinstance(op, Fold) else None
        t, mn, lead, lanes = atomize((1, 1)), (None, None), grid, 1
        # The streamed-store reading — a full tree walk — derived once for the degenerate arm's
        # ``apply_output_specs``. Empty without specs, which is exactly when it is not asked.
        observed = observed_result_names(op) if output_specs else frozenset()
        if plan is None or (plan.coop <= 1 and plan.reg <= 1):
            body = list(dict.fromkeys([*op.lower(), *tail]))
            if output_specs:
                # ``observed`` streams a scan store into its reduce loop; every other spec keeps
                # its kernel-tail reconstitution.
                body = apply_output_specs(body, output_specs, observed=observed)
            state, fold, close, bt = [], with_store(body, ctx.output, grid, out_val), [], None
        elif plan.coop_transposed:
            # The ``coop-t`` k-major matvec partition: the innermost output axis splits into a
            # shrunk ``<out>_blk`` grid axis (×32) + the 32-wide ``n_lane`` thread axis (with
            # ``k_co`` between them), so B loads coalesce across lanes. The emitted body's
            # output-var references were σ-substituted to ``blk·32 + n_lane`` inside (clamped,
            # and the store guarded, when 32 does not tile the swept extent).
            state, fold, close, lanes_axes = _tile_reduce_axis_transposed(op, plan, ctx, tail, out_val)
            out_ax = next(a for a in reversed(grid) if not (a.extent.is_static and a.extent.as_static() == 1))
            blk = Axis(name=f"{out_ax.name}_blk", extent=out_ax.extent.ceil_div(32), window=Window(parent=out_ax))
            lead = tuple(blk if a.name == out_ax.name else a for a in grid)
            t = replace(t, axes=lanes_axes)
            bt = plan.coop
        else:
            state, fold, close, lane = _tile_reduce_axis(op, plan, ctx, tail, out_val)
            t = replace(t, axes=(lane,)) if lane is not None else t
            bt = plan.coop if lane is not None else None

        def state_decls(_cells):
            return state

        def reduce_region(_cells, _offset, _mn):
            return [], fold

        def sink(_i, _j, _offset, _mn):
            return close

    return grid_tile(
        t,
        mn=mn,
        lead_axes=lead,
        block_threads=bt,
        lanes=lanes,
        state_decls=state_decls,
        reduce_region=reduce_region,
        store=sink,
        # The scheduler stamps ``workers`` on scheduled contractions; every other arm arrives with
        # ``None`` — safe to thread unconditionally.
        workers=ctx.workers,
        raster=ctx.raster,
    )


# ---- the tiled REDUCE axis (cooperative / ILP) -------------------------------------------------- #
# A PLANAR / TWISTED monoid reduce (sum / max / mean / RMSNorm / softmax / the coop-KV TWISTED flash
# reduce) partitions the reduce axis ``coop`` ways across the CTA's threads (cooperation) and ``reg``
# ways across per-thread register accumulators (ILP). The serial reduce ``Loop`` becomes a
# :class:`StridedLoop` of step ``coop·reg``; for ``reg > 1`` its body is replicated ``reg`` times
# (each copy offset by ``r·coop`` and folding its own accumulator). After the loop: the REG tree
# folds the ``reg`` accumulators into one (``Reduction.merge_stmts``), then — if ``coop > 1`` — the
# cross-thread combine (:func:`emit_combine`), then the projection. The op tree + ``lower`` are
# shared with the other tiers; only the partition changes.


def _mask_streamed(body: list[Stmt], axis: str, offset: int, extent, stream_identity: tuple[str, float] | None = None) -> list[Stmt]:
    """Clamp-to-identity a masked tail copy's contribution. Two forms, selected by the CARRIER:

    **Plain (``id``) carrier** (``stream_identity is None``): mask each ``Accum``'s folded
    ``value`` — a ``Select`` of the value when ``axis + offset < extent`` else the fold's own
    identity (``op.identity`` — ``sum`` → 0, ``max`` → −inf), so an out-of-range copy folds a
    no-op. Masking the FOLD (not the load) is what makes a **prologue** correct — ``sum(x·x)``
    past the extent needs the *additive* identity 0, which masking the load to the *multiply*
    identity (1) would not give.

    **Twisted carrier** (``stream_identity`` = ``(pivot term name in this copy, pivot fold
    identity)`` — −inf for the ``exp`` family): mask the PIVOT TERM — the per-element score value
    the whole merge chain derives from — which the twisted monoid absorbs by law (``max(m, −inf)
    = m``, the rescale ``exp(m − m) = 1``, the weight ``exp(−inf − m) = 0``, a lifted value
    channel folding ``0·V``). Per-``Accum`` masking is NOT sufficient there: the merge's shared
    intermediates (``t0 = max(m, s_raw)`` feeding the ``l·exp(m − t0)`` rescale) read the raw
    wrapped value, so a duplicate read larger than the running max silently down-scales the
    denominator while the masked pivot stays put (the 2026-07-09 symbolic-softmax r4
    miscompare). The clamp lands on the TERM, not the streamed loads: a flash score is computed
    by a nested Q@K contraction whose INPUT loads must stay raw (a dot of clamped −inf inputs is
    not the fold identity).

    Either way the streamed ``Load`` index is already wrapped in-bounds (``% extent`` via the
    caller's σ), so the read itself is safe."""
    cond = BinaryExpr("<", BinaryExpr("+", Var(axis), Literal(offset, "int")), extent)
    out: list[Stmt] = []
    if stream_identity is not None:
        term, identity = stream_identity
        renames: dict[str, str] = {}
        clamped = False
        for s in body:
            if renames:
                s = s.rewrite(lambda n: renames.get(n, n))
            out.append(s)
            if not clamped:
                defs = set(s.defines()) | {nm for b in s.nested() for st in b.iter() for nm in st.defines()}
                if term in defs:
                    ident, masked = f"{term}__id", f"{term}__mv"
                    out.append(Init(name=ident, identity=identity, dtype=F32))
                    out.append(Select(name=masked, branches=(SelectBranch(term, cond), SelectBranch(ident, Literal(1, "int")))))
                    renames[term] = masked
                    clamped = True
        assert clamped, f"masked twisted ILP copy: pivot term {term!r} not defined in the reduce body"
        return out
    for s in body:
        if isinstance(s, Accum):
            ident, masked = f"{s.value}__id", f"{s.value}__m"
            out.append(Init(name=ident, identity=s.op.identity, dtype=F32))
            out.append(Select(name=masked, branches=(SelectBranch(s.value, cond), SelectBranch(ident, Literal(1, "int")))))
            out.append(replace(s, value=masked))
        else:
            out.append(s)
    return out


def _replicate(
    body: Body, r: int, coop: int, axis: Axis, masked: bool, protected: frozenset[str], stream_identity: tuple[str, float] | None = None
) -> list[Stmt]:
    """Copy ``r`` of the reduce body for the REG (ILP) fold. Copy 0 is the body verbatim.
    Copy ``r > 0`` suffixes every per-copy SSA name with ``__r{r}`` (its accumulator + temps
    are an independent chain) — EXCEPT the shared iteration coordinates in ``protected`` (the
    grid / reduce / lane axis vars, common to all copies) — and offsets its streamed reads by
    ``r·coop`` (σ on the reduce axis). A ``masked`` copy wraps the read in-bounds (``% extent``)
    and clamps the tail contribution to a no-op (:func:`_mask_streamed`; ``stream_identity``
    selects the twisted-carrier form)."""
    if r == 0:
        return list(body)
    offset = r * coop
    shifted = BinaryExpr("+", Var(axis.name), Literal(offset, "int"))
    index_expr = BinaryExpr("%", shifted, axis.extent_expr()) if masked else shifted
    sigma = Sigma({axis.name: index_expr})
    out = copy_cell(body, sigma, f"__r{r}", protected)
    if not masked:
        return out
    if stream_identity is not None:
        term, identity = stream_identity
        # The pivot term is body-defined, so this copy's spelling carries the copy suffix.
        stream_identity = (term if term in protected else f"{term}__r{r}", identity)
    return _mask_streamed(out, axis.name, offset, axis.extent_expr(), stream_identity)


def _restage_loads(stmts: list[Stmt], buf: str, smem: str, n_grid: int, grid_vars: tuple) -> list[Stmt]:
    """Rewrite every ``(grid…, k)`` scalar ``Load`` of ``buf`` to read ``smem[k]`` (the staged
    row), recursing into nested bodies. Other loads (and ``buf`` loads with a different index
    shape) pass through untouched."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Load) and s.is_scalar and s.input == buf and len(s.index) == n_grid + 1 and tuple(s.index[:n_grid]) == grid_vars:
            out.append(Load(name=s.name, input=smem, index=(s.index[-1],)))
            continue
        bodies = s.nested()
        if bodies:
            s = s.with_bodies(tuple(Body(tuple(_restage_loads(list(b), buf, smem, n_grid, grid_vars))) for b in bodies))
        out.append(s)
    return out


def emit_combine(
    red, t: str, n_threads: int, *, warp_size: int = 32, segmented: bool = False, inner: tuple[str, int] | None = None
) -> list[Stmt]:
    """Build the cross-thread combine of a cooperative reduce — the algebra read off the
    ``red`` fold NODE's stored ``combine`` — over ``n_threads`` cooperating threads,
    reassigning the carried state in place.

    The mechanism per level is derived by :meth:`ReduceStage.combine`:

    - a ``SHFL`` fold → one ``WarpShuffle`` register butterfly. The XOR butterfly never
      crosses an aligned ``width``-lane group, so a lone ``SHFL`` is also the SEGMENTED
      per-row combine for strided-cooperative rows (caller passes ``segmented=True``).
    - a ``SMEM`` fold **after** a ``SHFL`` → the hierarchical cross-warp slab: lane-0 of each
      warp stages its broadcast state to a ``smem[n_warps]`` slab per component; one ``Sync``
      + ``TreeHalve(tid_var="warp")`` collapses across warps and broadcasts.
    - a standalone ``SMEM`` → the block slab: every thread stages its partial, one ``Sync``,
      a single ``TreeHalve`` reduces + broadcasts in place.

    The stored combine's surface (its results / second-operand params / the
    :attr:`~emmy.compiler.pipeline.passes.lowering._reduction.Reduction.combine_states` program)
    drives the nodes; the
    combine renders at the accumulator dtype (fp32 for a reduction, with the fold's own dtype
    honored when set)."""
    state = red.names
    state_b = red.state_b
    prog = red.combine_states
    dtype = next((a.dtype for a in prog if a.dtype is not None), None) or F32

    smem_c = cuda_name(dtype)
    bufs = tuple(f"{st}_smem" for st in state)
    if inner is not None:
        # The transposed (``coop-t``) combine: threads sharing an output lane sit ``scale``
        # apart in tid, so a shuffle would fold DIFFERENT outputs — always the segment-indexed
        # smem tree: ``n_threads`` k-slices × ``scale`` lanes per slab, each lane's tree
        # halving its own segment (``TreeHalve.inner``).
        iv, scale = inner
        idx = BinaryExpr("+", BinaryExpr("*", Var(t), Literal(scale, "int")), Var(iv))
        out: list[Stmt] = [Smem(name=b, extents=(n_threads * scale,), dtype=smem_c) for b in bufs]
        out += [Write(output=b, index=(idx,), value=st) for b, st in zip(bufs, state, strict=True)]
        out.append(Sync())
        out.append(
            TreeHalve(bufs=bufs, state=state, state_b=state_b, combine_states=prog, length=n_threads, tid_var=t, dtype=dtype, inner=inner)
        )
        return out
    folds = ReduceStage(Level.BLOCK, n_threads).combine(warp_size=warp_size, segmented=segmented)
    out = []
    for i, fold in enumerate(folds):
        if fold is FoldMove.SHFL:
            # The lane-level butterfly: warp-wide when followed by a cross-warp SMEM stage
            # (hierarchical), else the full ``n_threads`` (one warp / segment).
            width = warp_size if (len(folds) == 2 and folds[1] is FoldMove.SMEM) else n_threads
            out.append(WarpShuffle(state=state, state_b=state_b, combine_states=prog, length=width, dtype=dtype))
        elif fold is FoldMove.SMEM:
            hierarchical = i > 0 and folds[i - 1] is FoldMove.SHFL
            width = n_threads // warp_size if hierarchical else n_threads
            tid_var = "warp" if hierarchical else t
            out += [Smem(name=b, extents=(width,), dtype=smem_c) for b in bufs]
            if hierarchical:
                # Lane-0 of each warp stages that warp's broadcast state, indexed by ``warp``.
                out.append(
                    Cond(
                        cond=BinaryExpr("==", Var("lane"), Literal(0, "int")),
                        body=tuple(Write(output=b, index=(Var("warp"),), value=st) for b, st in zip(bufs, state, strict=True)),
                    )
                )
            else:
                out += [Write(output=b, index=(Var(tid_var),), value=st) for b, st in zip(bufs, state, strict=True)]
            out.append(Sync())
            out.append(TreeHalve(bufs=bufs, state=state, state_b=state_b, combine_states=prog, length=width, tid_var=tid_var, dtype=dtype))
        else:  # FoldMove.ATOMIC / FoldMove.REG — cross-CTA / register tiers, not emitted by the intra-CTA walk.
            raise NotImplementedError(f"intra-CTA combine cannot emit {fold} (cta/reg tiers are future work)")
    return out


def combine_tail(red, *, reg: int, coop: int, lane) -> list[Stmt]:
    """The algebra-driven **partial merge** that follows a partitioned reduce loop — the one place the
    two partial-fold geometries are assembled: the REG-tree fold of the ``reg`` ILP register copies
    into copy 0 (``Reduction.merge_stmts``), then — when threads cooperate (``lane`` is a lane :class:`Axis`,
    not ``None``) — the cross-thread :func:`emit_combine`. Both reassign the carried state **in place**
    (the survivor SSA names hold the full reduction), so the post-reduce projection reads them directly.

    Algebra-generic (the ⊕ read off the ``red`` fold node's stored combine): a monoid reduce and a
    contraction's degenerate additive fold identically, so a cooperative reduce and a (future)
    cooperative-K contraction share this tail. ``merge_stmts`` keys a twisted combine's temps on
    the copy name, so each fold's internals are already unique."""
    merge: list[Stmt] = [st for r in range(1, reg) for st in red.merge_stmts(tuple(f"{n}__r{r}" for n in red.names))]
    if lane is not None:
        merge += emit_combine(red, t=lane.name, n_threads=coop)
    return merge


def _tile_reduce_axis_transposed(
    op: Fold, plan, ctx: Ctx, tail: tuple, out_val: str
) -> tuple[list[Stmt], list[Stmt], list[Stmt], tuple[Axis, ...]]:
    """The ``coop-t`` (transposed) cooperative reduce — the k-major-B matvec partition: 32
    ``n_lane`` threads (innermost) sweep the OUTPUT axis so B loads coalesce across lanes at
    every k step, and ``coop/32`` ``k_co`` slices ride the upper thread bits. The emitted body
    keeps referencing the original output axis var — one σ substitutes it with
    ``blk·32 + n_lane`` (the caller rebinds the shrunk ``ceil(E/32)`` ``<out>_blk`` grid axis; an
    overhanging last block clamp-reads and guards its store). The combine is
    the segment-indexed smem tree (``emit_combine(inner=…)`` — never a shuffle: adjacent lanes
    hold different outputs); the projection stores guard on ``k_co == 0``, each lane writing its
    own cell. Unsupported here (the enumeration must not offer ``t`` on them): shared-row
    ``smem`` shared-row staging, distributed full-row projections (a ``Loop`` in the tail)."""
    grid = ctx.grid
    coop, reg = plan.coop, plan.reg
    lanes_n = 32
    k_ways = coop // lanes_n
    assert coop % lanes_n == 0 and k_ways >= 1, f"b{coop}t needs a multiple of {lanes_n}"
    stage = ctx.sched.get("STAGE", op)
    assert not (stage is not None and stage.smem), "transposed coop cannot ride shared-row staging"
    out_ax = next(a for a in reversed(grid) if not (a.extent.is_static and a.extent.as_static() == 1))

    *hoisted, rloop = op.lower()
    alg = Reduction(op)
    axis = rloop.axis
    stride = k_ways * reg
    masked = reg > 1 and not (axis.extent.is_static and axis.extent.as_static() % stride == 0)

    n_lane = Axis(name=f"{out_ax.name}_ln", extent=lanes_n)
    k_co = Axis(name=f"{axis.name}_co", extent=k_ways) if k_ways > 1 else None
    start = Var(k_co.name) if k_co is not None else Literal(0, "int")
    blk_name = f"{out_ax.name}_blk"
    # The swept cell this lane owns. The grid is ``ceil(E / 32)`` blocks, so a swept axis 32 does
    # not tile leaves the last block's upper lanes OVERHANGING: they clamp-read the last valid
    # column (a duplicate sweep, in-bounds) and their store is discarded by the guard below — the
    # same masked-overhang contract the tiled contraction's ``clamp_last`` / ``Cond`` pair states.
    cell = BinaryExpr("+", BinaryExpr("*", Var(blk_name), Literal(lanes_n, "int")), Var(n_lane.name))
    out_ext = out_ax.extent_expr()
    overhang = not (out_ax.extent.is_static and out_ax.extent.as_static() % lanes_n == 0)
    subst = Sigma({out_ax.name: clamp_last(cell, out_ext) if overhang else cell})  # the sweep's reads
    store_subst = Sigma({out_ax.name: cell})  # the guarded projection: in range by the guard, so no clamp

    nested_axes = {lp.axis.name for lp in rloop.body.iter_of_type(Loop, StridedLoop)}
    defined = {nm for s in rloop.body.iter() for nm in s.defines()}
    expr_external = {v for s in rloop.body.iter() for e in s.exprs() for v in e.free_vars()} - defined
    # A value defined ahead of the loop and read inside it (a hoisted operand's) is one value
    # shared by every register copy — the same exclusion :func:`_strided_fold` makes.
    deps_external = {nm for s in rloop.body.iter() for nm in s.deps()} - defined
    protected = frozenset(
        {axis.name, *(ax.name for ax in grid), blk_name, n_lane.name, *axis.extent_expr().free_vars(), *nested_axes, *expr_external}
        | deps_external
        | ({k_co.name} if k_co is not None else set())
    )
    stream_identity = (str(alg.terms[0]), ElementwiseImpl("maximum").identity) if alg.twisted else None
    copies: list[Stmt] = []
    for r in range(reg):
        copies.extend(_replicate(rloop.body, r, k_ways, axis, masked, protected, stream_identity))
    strided = StridedLoop(axis=axis, start=start, step=Literal(stride, "int"), body=Body(tuple(copies)), unroll=rloop.unroll)
    strided = strided.substitute(subst)

    merge: list[Stmt] = [st for r in range(1, reg) for st in alg.merge_stmts(tuple(f"{n}__r{r}" for n in alg.names))]
    if k_co is not None:
        merge += emit_combine(alg, t=k_co.name, n_threads=k_ways, inner=(n_lane.name, lanes_n))

    tail_stmts = with_store(list(tail), ctx.output, grid, out_val)
    tail_stmts = [s.substitute(store_subst) for s in tail_stmts]
    if overhang:
        tail_stmts = [Cond(cond=BinaryExpr("<", cell, out_ext), body=tuple(tail_stmts))]
    if k_co is not None:
        tail_stmts = [Cond(cond=BinaryExpr("==", Var(k_co.name), Literal(0, "int")), body=tuple(tail_stmts))]

    lanes_axes = ((k_co,) if k_co is not None else ()) + (n_lane,)
    return [], [*(s.substitute(subst) for s in hoisted), strided, *merge], tail_stmts, lanes_axes


def _strided_fold(op: Fold, rloop, plan, ctx: Ctx, lane: Axis | None) -> list[Stmt]:
    """The partitioned reduce loop for ONE fold — ``reg`` ILP chains striding ``coop·reg`` from the
    lane's start, then the REG-tree merge and (when threads cooperate) the cross-thread combine.
    ``rloop`` is the fold's already-emitted serial reduce ``Loop``; the caller owns any prologue
    ``lower`` hoisted ahead of it and any smem row-staging rewrite."""
    coop, reg = plan.coop, plan.reg
    alg = Reduction(op)
    axis = rloop.axis
    stride = coop * reg
    masked = reg > 1 and not (axis.extent.is_static and axis.extent.as_static() % stride == 0)
    start = Var(lane.name) if lane is not None else Literal(0, "int")

    # The reduce loop: ``reg`` interleaved accumulator chains (ILP), striding the axis by
    # ``coop·reg`` from the lane's start. The dissolved fold ``Accum``\\ s seed each copy's
    # accumulator (``StridedLoop.render``).
    # The shared iteration coordinates (grid + reduce + lane axis vars) and the symbolic
    # extent's runtime arg(s) (e.g. ``seq_len``) are common to every register copy — exclude
    # them from the per-copy SSA rename. So too any nested loop-axis variable (a child contraction
    # contraction's own reduce coordinate ``dd`` / ``j``): ``copy_cell``'s ``rewrite`` renames
    # a var's USES but not a ``Loop``'s own axis DECLARATION, so suffixing the uses (``dd__r1``)
    # while the ``for`` decl stays ``dd`` emits an undefined identifier. Each copy re-declares
    # its own nested loop, so a shared name is correct (loop-scoped).
    nested_axes = {lp.axis.name for lp in rloop.body.iter_of_type(Loop, StridedLoop)}
    # ... and ANY external name the body's index/extent Exprs read without defining — a symbolic
    # dim can enter through a buffer's flattened STRIDES (a 4-D tensor's ``seq_len``) on an op
    # whose own reduce extent is static, where none of the named sets above cover it; renaming
    # such a use (``seq_len__r3``) emits an undeclared identifier (surfaced by the 2026-07-09
    # offline-weights refit steering dynamic scalar SDPA onto the ILP fold).
    defined = {nm for s in rloop.body.iter() for nm in s.defines()}
    expr_external = {v for s in rloop.body.iter() for e in s.exprs() for v in e.free_vars()} - defined
    # ... and the same for the SSA-deps channel: an ``Assign``'s args are name strings ``deps()``
    # reports and ``exprs()`` does not, so a value defined ahead of the loop (a hoisted scalar
    # load, a provider chain a cut left before the reduce) and read inside it is invisible to the
    # Expr scan above. It is one value shared by every copy — renaming its uses (``in3__r1``)
    # emits an undeclared identifier (surfaced by DeepSeek-V4 post4096's two-cut piece).
    deps_external = {nm for s in rloop.body.iter() for nm in s.deps()} - defined
    protected = frozenset(
        {axis.name, *(ax.name for ax in ctx.grid), *axis.extent_expr().free_vars(), *nested_axes, *expr_external, *deps_external}
        | ({lane.name} if lane is not None else set())
    )
    # A twisted fold's masked tail clamps the STREAMED VALUE to the pivot fold's identity
    # (the ``exp`` family's running max, −inf) — see :func:`_mask_streamed`'s twisted form.
    stream_identity = (str(alg.terms[0]), ElementwiseImpl("maximum").identity) if alg.twisted else None
    copies: list[Stmt] = []
    for r in range(reg):
        copies.extend(_replicate(rloop.body, r, coop, axis, masked, protected, stream_identity))
    strided = StridedLoop(axis=axis, start=start, step=Literal(stride, "int"), body=Body(tuple(copies)), unroll=rloop.unroll)

    # The carrier-driven partial merge: the REG-tree fold of the ``reg`` ILP copies into the survivor
    # (copy 0's names) + (when threads cooperate) the cross-thread combine, reassigning the carried
    # state in place. The one shared tail a cooperative reduce and a future cooperative-K contraction
    # both emit (``combine_tail``).
    return [strided, *combine_tail(alg, reg=reg, coop=coop, lane=lane)]


def _lane_close(tail: list[Stmt], lane: Axis | None, coop: int, ctx: Ctx, out_val: str) -> list[Stmt]:
    """The post-reduce projection close. A full-row output (softmax / RMSNorm) distributes its FREE
    sweep across the coop lanes; a scalar output is written once, guarded to lane 0. With no
    cooperation (coop == 1) the single thread runs the projection as-is. A raw REDUCE loop in the
    tail (a restored sibling fold the classifier could not consume) is NOT a sweep: lane striding
    it would leave each lane an uncombined partial, so it runs SERIALLY per lane — every lane
    computes the identical full fold, and the tail's unguarded stores stay deterministic because
    every lane writes the same value."""
    if lane is None:
        body_tail = with_store(tail, ctx.output, ctx.grid, out_val)
    elif any(isinstance(s, Loop) and not s.is_reduce for s in tail):
        body_tail = [
            StridedLoop(axis=s.axis, start=Var(lane.name), step=Literal(coop, "int"), body=s.body, unroll=s.unroll)
            if isinstance(s, Loop) and not s.is_reduce
            else s
            for s in tail
        ]
    elif any(isinstance(s, Loop) for s in tail):
        body_tail = list(tail)  # reduce-bearing scalar tail: identical per lane, stores deterministic
    else:
        stored = with_store(tail, ctx.output, ctx.grid, out_val)
        body_tail = [Cond(cond=BinaryExpr("==", Var(lane.name), Literal(0, "int")), body=tuple(stored))]
    return body_tail


def _tile_reduce_axis(op: Fold, plan, ctx: Ctx, tail: tuple, out_val: str) -> tuple[list[Stmt], list[Stmt], list[Stmt], Axis | None]:
    """Tile the REDUCE axis per the node's cooperating :class:`Reduce` — the reduce counterpart
    of the output ``unit_tile`` / ``register_tile`` levels: ``coop`` lanes across threads (the
    ``_co`` lane axis, the axis's UNIT level) and ``reg`` ILP chains across per-thread accumulators
    (its REGISTER level — cyclic, copy ``r`` offset by ``r·coop``, the loop striding ``coop·reg``).
    It takes the per-cell reduce loop from the node's own lowering and returns the pieces the
    one pipeline (:func:`_bind` → :func:`grid_tile`) seals: ``(state, fold, close, lane)`` — the
    shared-row fill decls, the strided fold loop + the carrier merge (the REG tree + the
    cross-thread combine), the distributed projection close, and the lane :class:`Axis` (``None``
    for standalone ILP — one thread per cell, lane fixed at 0)."""
    grid = ctx.grid
    coop = plan.coop

    # The per-cell reduce loop is the node's own lowering (``Fold.lower``) off the :class:`Fold`
    # **node** — the walk reaches any nested contraction as a node. The algebra
    # is read off the ``Fold`` node itself (:class:`Reduction` — a contraction's K fold and a
    # monoid's reduce fold both answer it, so the algebra-generic ``merge_stmts`` /
    # ``combine_states`` machinery folds either). An operand that does not index the fold's axis
    # is hoisted ahead of the loop and leads the region; the enclosing zero-axis ``Fold``'s
    # projection is ``tail`` (already walked).
    *hoisted, rloop = op.lower()
    axis = rloop.axis

    # The cooperative lane axis (Tile-decoded, innermost) — present only when threads
    # cooperate; standalone ILP (coop == 1) runs one thread per cell, lane fixed at 0.
    lane = Axis(name=f"{axis.name}_co", extent=coop) if coop > 1 else None
    start = Var(lane.name) if lane is not None else Literal(0, "int")

    # Shared-row staging (the fused norm→linear prologue): an input row folded by the cooperative
    # reduce AND re-read per output column of a contraction tail rides a first-class ``smem``
    # :class:`Stage` whose ``smem`` names the row — DETECTED scheduler-side (schedule-side)
    # and only APPLIED here: fill the row into smem once (cooperatively) and rewrite both readers to
    # the slab. Only the cooperative tier (coop > 1) is ever stamped; a contraction operand ``Stage``
    # (the coop-K matmul's pinned pipeline) never sets ``smem``, so it passes through untouched.
    tail_src = list(tail)
    fill_stmts: list[Stmt] = []
    op_stage = ctx.sched.get("STAGE", op)
    if lane is not None and op_stage is not None and op_stage.smem:
        (staged,) = op_stage.smem
        grid_vars = tuple(Var(a.name) for a in grid)
        smem_name = f"{staged}_smem"
        fill_stmts = sync_row_fill(
            slab=smem_name,
            src=staged,
            extent=axis.extent.as_static(),
            grid_vars=grid_vars,
            linear_tid=start,
            n_threads=coop,
            dtype=cuda_name(ctx.inputs[staged].dtype),
        )
        n_grid = len(grid)
        rloop = replace(rloop, body=Body(tuple(_restage_loads(list(rloop.body), staged, smem_name, n_grid, grid_vars))))
        tail_src = _restage_loads(tail_src, staged, smem_name, n_grid, grid_vars)

    fold = _strided_fold(op, rloop, plan, ctx, lane)
    return fill_stmts, [*hoisted, *fold], _lane_close(tail_src, lane, coop, ctx, out_val), lane
