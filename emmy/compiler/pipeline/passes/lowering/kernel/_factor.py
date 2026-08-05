"""The factorizer — the recursive ``TileOp``-root emitter and the ONE root binder every kernel
seals through. The per-atom codegen **strategies** it drives live in ``_atom.py``, and the
axis-realization layer it seals through in ``_tiling.py``.

:func:`factorize` is the entry ``010_materialize`` calls once per kernel: it builds the ambient
:class:`Ctx` and dispatches ``tile.op`` through the recursion :func:`_factorize`, which walks the
``Fold`` node node tree. A :class:`~...ir.Fold` with a ``source``
**recurses** (its projection walked into the ``tail``); the leaf binds to the grid via the single
:func:`_bind` pipeline, whose form is read off the node's SCHEDULE — which axes are tiled — never a
kernel kind: a tiled :class:`~...ir.bilinear fold` tiles its OUTPUT ``(m, n)`` axes (register / warp
cells), a cooperating :class:`~...ir.Fold` tiles its REDUCE axis (:func:`_tile_reduce_axis` —
``coop`` lanes + ``reg`` ILP chains), and everything else tiles nothing (the degenerate
one-thread-per-cell fold). All three seal through the one :func:`grid_tile` finalizer; the per-cell
body is built by the shared recursion :func:`_emit` (which walks ``source`` AND ``step``,
reaching flash's Q@K / P@V as nodes).

The output tiling reads its **geometry straight off the** bilinear ``Fold`` **node** (``tile_m`` /
``mask_m`` / ``m_b`` / ``block_threads`` / …, derived there from the ``tile`` schedule + the output
axes), expands both atoms through the *same* four-level tiling pipeline (``atomize →
register_tile → unit_tile → grid_tile``, in **``_tiling.py``** — the algebra-free layer that turns
the schedule's plan into bound ``Axis`` objects), and splices in two codegen halves from
the per-atom strategies in **``_atom.py``**: :func:`~...kernel._atom.reduce_codegen` — the reusable,
**sink-agnostic** ``(state_decls, reduce_region)`` (accumulator/operand decls + the contraction
K-loop) — and a per-cell **sink** ``store(i, j, offset, mn)`` (default
:func:`~...kernel._atom.store_sink`, the matmul sink; ``factorize(tile, root, store=…)`` swaps it —
the flash inner QK/PV pass a sink that bridges the accumulator into the streaming-softmax twist,
reusing the same ``reduce_codegen``).

The reduce-axis tiling (:func:`_tile_reduce_axis` + the shared-row staging apply) folds the reduce
axis ``coop`` ways across threads and ``reg`` ways across per-thread accumulators, then the
REG-tree fold, the cross-thread combine (:func:`emit_combine`), and the projection — algebra-
generic through the :class:`Reduction` view (a contraction is the degenerate algebra of its
additive fold).

The smem operand-staging pipeline lives in ``_stage.py`` (the :class:`~...kernel._stage.Transport`
strategy + the one :func:`~...kernel._stage.staged_kloop`); the ONE atom-agnostic driver
(``_atom._staged``) builds the transport, the atom strategy supplying only the slab drain leaf.
It is driven off the node's ``STAGE`` codec →
:class:`~...schedule.Stage` (``d<depth>`` gmem→smem ring · ``sync``/``cp``/``tma`` transport ·
``p<n>`` smem→register double-buffer). The **scalar** contraction tier stays gmem-direct. The fused
norm→linear **shared-row** prologue is Stage-driven too: the schedule detects the reused input row
and stamps a ``sync`` :class:`~...schedule.Stage` whose ``smem`` names it; :func:`_tile_reduce_axis` only
applies it (the 1-D ``sync_row_fill`` + the load rewrite). Leading ``_`` so the pass loader skips this
module."""

from __future__ import annotations

from dataclasses import dataclass, replace

from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis, AxisRole, Window
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel import Tile
from emmy.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Body, Cond, Init, Load, Loop, Select, SelectBranch, Stmt, StridedLoop, Write
from emmy.compiler.ir.tile import FoldMove, Level, ReducePlan, ReduceStage
from emmy.compiler.ir.tile.ir import Fold, effect_tail, is_contraction
from emmy.compiler.ir.tile.ops import cone_seam
from emmy.compiler.pipeline.passes.lowering._reduction import Reduction, loop_state_head
from emmy.compiler.pipeline.passes.lowering.kernel._atom import copy_cell, reduce_codegen, store_sink
from emmy.compiler.pipeline.passes.lowering.kernel._stage import sync_row_fill
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import atomize, grid_tile, register_tile, unit_tile
from emmy.compiler.pipeline.passes.lowering.kernel._twist import FLASH_WARP_AXIS, realize_warp_twist, warp_source

# ---- the recursive node walk: Ctx down, Frag up ------------------------------------------------ #
# The hierarchical emitter (the tile-IR-rebuild mandate: ONE recursion over the node tree, no
# divergent codegen path). :func:`_emit` walks a ``Fold`` node tree —
# through ``source`` AND ``step`` — threading a :class:`Ctx` down (the ambient cell environment)
# and returning a :class:`Frag` up (the per-cell loop-IR body + the produced :class:`Handle` wire +
# the reduce ``carrier`` when the node folds one). The ONE root binder (:func:`_bind`) consumes the
# recursion: the output-tiled bilinear ``Fold`` arm splices the atom's codegen through ``grid_tile``,
# and the reduce partitioner (:func:`_tile_reduce_axis`) builds its per-cell reduce loop via
# :func:`_emit`, so a nested bilinear ``Fold`` (flash's Q@K / P@V) is reached AS A NODE — scalar-nested
# at block=1, while a WARP-TILED tree realizes at fragment residence through ``_twist`` (the
# ``warp_source`` read in :func:`_bind`), the per-node warp tiles stamped by the scheduler.


@dataclass(frozen=True)
class Handle:
    """A produced tensor a parent wires up — "a value that needs wiring." ``name`` is the SSA name
    holding it; ``residence`` is HOW a consumer reads it (``reg`` = a scalar register value today;
    the tensor-core rebuild adds ``reg_frag`` — an mma fragment — plus a fragment descriptor
    ``(mma_role, shape, dtype)`` and the accumulator→operand recast at a node boundary)."""

    name: str
    residence: str = "reg"


@dataclass(frozen=True)
class Frag:
    """What a node contributes UP the recursion: the per-cell loop-IR ``body`` it emits (the reduce /
    contraction loop nest / the projection sweep), the produced :class:`Handle` ``out`` (the wire a
    parent connects to), and the reduce ``carrier`` — set iff this node folds a reduce whose
    cross-partition combine a root binder must emit (``None`` for a pure pointwise map / a scalar
    per-cell contraction)."""

    body: list[Stmt]
    out: Handle


@dataclass(frozen=True)
class Ctx:
    """The ambient cell environment threaded DOWN the recursion — established once for the whole
    kernel and passed unchanged so every node reads/writes at the same output cell. ``grid`` is the
    kernel's grid axes; ``inputs`` the operand buffer table (dtype/shape); ``output`` the root
    output buffer name. The operand smem pipeline is NOT here — it rides the node it decorates
    (the ``STAGE`` slice). (The tensor-core rebuild adds the warp
    ``bind``/``cell`` register tile — owned per-node by a bilinear ``Fold``'s ``tile`` — and the
    inbound ``wires`` handles, e.g. flash's score fragment feeding P@V's A operand.)"""

    grid: tuple
    inputs: dict | None = None
    output: str = ""
    workers: object = None  # the resolved WarpSpec worker split (None = uniform SIMT)
    raster: object = None  # the parsed RASTER codec (ir.schedule.Raster; None = flat launch order)
    # The kernel's schedule slices (``TileOp.schedule`` bound to its op tree — ``ops.Sched``): the
    # per-node ``tile`` / ``reduce`` / ``stage`` reads all go through here (1r — the term stores
    # no slices).
    sched: object = None
    # The placement's FREE axes — the un-shrunk originals (a warp-flash grid shrinks the query axis
    # and folds the value axis away; a split partial prefixes ``_ksplit``). The twist / chain
    # realizers derive their contraction views' output axes from the trailing pair.
    free: tuple = ()


def _emit(op, ctx: Ctx) -> Frag:
    """Recurse over a structural node, returning its :class:`Frag` (per-cell body + wire + carrier).
    The single node-kind dispatch every kernel's compute flows through — walking ``source`` AND
    ``step`` so flash's Q@K / P@V contractions are reached as nodes. Scalar-nested: a node's body
    is its lowered loop-IR (byte-identical to ``Fold.lower``); a WARP-TILED tree does not reach this
    walk — ``_bind`` realizes it at fragment residence through ``_twist`` instead."""
    if isinstance(op, Fold) and op.axis is None:
        src = _emit(op.operands[0], ctx) if op.operands else None
        prefix = list(src.body) if src is not None else []
        return Frag(body=[*prefix, *_emit_body(op.body, ctx)], out=_map_wire(op))
    if isinstance(op, Fold):
        stmts = _emit_body(Body(op.spliced_step()), ctx)  # operand edges splice ahead of first use
        loop = Loop(axis=op.axis, body=Body(tuple(stmts)), unroll=op.unroll, role=op.role)
        return Frag(body=[loop], out=Handle(op.out))
    if is_contraction(op):
        # The per-cell scalar contraction (no TILE slice): the node's synthesized mul-add loop —
        # operand edges already flattened in place, so the walk below only passes stmts through.
        loop = op.loop
        stmts = _emit_body(loop.body, ctx)
        return Frag(body=[Loop(axis=loop.axis, body=Body(tuple(stmts)), unroll=loop.unroll, role=loop.role)], out=Handle(op.out))
    raise TypeError(f"_emit: expected a Fold node, got {type(op).__name__}")


def _map_wire(op: Fold) -> Handle:
    """The :class:`Handle` a parent wires to for a zero-axis ``Fold`` node — mirrors ``Fold.out``'s cases but
    stays robust where ``Fold.out`` would raise. An empty body surfaces the ``source``'s wire; a
    ``Write``-terminated body is a ROOT sink (stored to gmem, never wired) so surfaces the written
    value at ``gmem`` residence; a body ending in an annotated reduce / contraction ``Loop`` surfaces
    its carried state's head (:func:`loop_state_head` — the acc / carried value, NOT the loop's
    empty ``defines``);
    otherwise the last defining stmt (a pointwise lift / projection), or ``""`` for a sink whose store
    rides inside a projection sweep ``Loop`` (a don't-care — nothing consumes it)."""
    if len(op.body) == 0:
        return _emit_wire(op.operands[0]) if op.operands else Handle("")
    last = op.body[-1]
    if isinstance(last, Write):
        return Handle(last.values[-1], residence="gmem")
    if isinstance(last, (Loop, StridedLoop)) and last.role.is_reduce:
        return Handle(loop_state_head(last))
    defs = last.defines()
    return Handle(defs[-1] if defs else "")


def _emit_wire(op) -> Handle:
    """The produced-value :class:`Handle` of any node — a ``Fold`` / bilinear ``Fold`` names its
    carrier / accumulator; a zero-axis ``Fold`` scans for its last defining stmt (:func:`_map_wire`)."""
    if isinstance(op, Fold) and op.axis is None:
        return _map_wire(op)
    return Handle(op.out)  # Fold.out — the carrier state, or a contraction's primary acc; always safe


def _emit_body(body, ctx: Ctx) -> list[Stmt]:
    """Walk a ``Body`` of loop-IR stmts, recursing into any nested structural node (a
    :class:`Fold` tree) via :func:`_emit` and passing plain
    stmts through — the codegen-layer node-walk (the dispatch seam ``ir._flatten_nodes`` cannot host,
    since a warp-tiled nested contraction lowers to mma, not a scalar loop)."""
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Fold):
            out.extend(_emit(s, ctx).body)
        else:
            out.append(s)
    return out


def factorize(tile, root, store=None) -> Tile:
    """The entry to the recursive emitter — build the ambient :class:`Ctx` from the ``TileOp`` and its
    root graph node, then dispatch its ``op`` into a bound ``Tile`` via :func:`_factorize`. ``out_val``
    (the kernel's finalized output SSA name — the root node's produced :class:`Handle`) is resolved
    once here and threaded down for the store glue."""
    from emmy.compiler.ir.schedule import Raster  # noqa: PLC0415 — keep the module torch-free at import

    # Stored trees are already resolved — a computed operand is an inline node on its edge, so the
    # emitter below walks the tree as stored and every reader (``cone_seam``) reads the node
    # boundary straight off ``Fold.a``.
    from emmy.compiler.ir.tile.ops import sched_of  # noqa: PLC0415

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
    out_val = _emit_wire(op).name if op is not None else ""
    return _factorize(op, ctx, tail=(), out_val=out_val, store=store, stores=tuple(tile.stores))


def _factorize(op, ctx: Ctx, tail: tuple, out_val: str, store=None, stores: tuple = ()) -> Tile:
    """The recursive root walk — peel the projecting zero-axis ``Fold``\\ s, then bind the leaf to the grid via
    the ONE binder. A :class:`Fold` with a ``source`` **recurses**: its ``body`` (the projection /
    epilogue) is walked (:func:`_emit_body`, reaching any nested node), the kernel-boundary
    ``stores`` reconstituted into it (``effect_tail`` — 1q: the root ``Write``\\ s / output sweep
    left the term for ``TileOp.stores``, so the tail downstream sees is byte-identical to the
    stored-``Write`` era), and the result prepended to ``tail``;
    everything else is a leaf, bound by :func:`_bind` — the single pipeline, whose form is read off
    the node's SCHEDULE (which axes are tiled), never a kernel kind. There is **no** flash /
    attention special case: flash is the two-bilinear ``Fold`` ``TWISTED`` reduce tree, so its Q@K /
    P@V contractions and its streaming reduce factorize through this one walk (scalar block=1
    today; a nested warp-tiled contraction routes through the ``_emit`` bilinear ``Fold`` seam). A
    bespoke emitter would be a divergent codegen path the mandate forbids."""
    if (isinstance(op, Fold) and op.axis is None) and op.operands:
        proj = _emit_body(op.body, ctx)
        if stores:
            proj = effect_tail(proj, stores)
        return _factorize(op.operands[0], ctx, tail=(*proj, *tail), out_val=out_val, store=store)
    if stores:
        # A flat / bare root with boundary stores — plain root ``Write``\\ s only (a sweep store
        # always rides a projecting zero-axis ``Fold``, whose peel above consumed it).
        assert all(st.sweep is None for st in stores), "sweep stores ride a projecting zero-axis fold"
        tail = (*tail, *(st.write for st in stores))
    return _bind(op, ctx, tail, out_val, store)


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
    (``Fold.out`` / the recursion's produced ``Handle``) so this helper stays node-agnostic."""
    if has_write(stmts):
        return stmts
    index = tuple(Var(ax.name) for ax in grid)
    return [*stmts, Write(output=output, index=index, value=value)]


def _bind(op, ctx: Ctx, tail: tuple, out_val: str, store=None) -> Tile:
    """The ONE root binder — every kernel binds through the same pipeline: read WHICH AXES the
    schedule tiles off the node, build the fold region, and seal through the one :func:`grid_tile`
    finalizer. The cases are points of one ``(output-tiling) × (reduce-folding)`` space, selected by
    the schedule — never separate emitters:

    - a a bilinear ``Fold`` tiles its OUTPUT ``(m, n)`` axes — register / warp cells through
      ``atomize → register_tile → unit_tile``, the reduce (K) serial per cell from the atom's
      :func:`reduce_codegen`, ``store`` the per-cell sink (default :func:`store_sink`; the flash
      inner QK/PV pass a sink that bridges the accumulator into the softmax twist). Its projection
      arrives as ``tail`` — peeled off the wrapping zero-axis fold, the ONE home for a projection; the bare
      grid-``Write`` glue is synthesized here (it needs ``ctx.output``, so it can't ride the node).
    - a :class:`Fold` whose :class:`ReducePlan` cooperates tiles its REDUCE axis instead
      (:func:`_tile_reduce_axis` — ``coop`` lanes at the unit level, ``reg`` ILP chains at the
      register level, the carrier merge closing the fold). The output stays one cell per thread:
      the 1×1 ``atomize`` with the whole grid riding ``lead_axes`` untiled.
    - anything else (a pure pointwise zero-axis fold, a trivial plan) tiles NOTHING — the degenerate
      one-thread-per-cell fold: the per-cell body (:func:`_emit`; a serial reduce ``Loop`` sits
      inside it) + ``tail`` + the ``out_val`` store glue is the whole fold region."""
    grid = tuple(ctx.grid)
    # The OUTPUT-tiled dispatch: a bilinear ``Fold`` whose schedule holds a TILE slice, over a
    # grid with an ``(m, n)`` pair to place it on. The node is pure algebra; the tiled reading comes
    # off the slice, which arrives ALREADY PLACED from ``Sched.tile_of`` (the ``(m, n)`` pair is a
    # function of the site, so the binding lives on the scheduling structure, not here) — the
    # geometry the atom reads is the slice's own, not a separate view object's. A stored node
    # WITHOUT a TILE slice takes the reduce tiers instead (the per-cell / coop-K forms), where the
    # whole grid rides untiled.
    tile = ctx.sched.tile_of(op) if is_contraction(op) else None
    if tile is not None and tile.axes is not None and len(grid) >= 2:
        c = op
        epi = list(tail)
        if not has_write(epi):
            epi = with_store(epi, ctx.output, grid, c.out)
        # The cone's K seam, read straight off the inline operand node (``None`` for a gmem-``Load``
        # A — its whole body is the per-cell fill).
        seam = cone_seam(c.a) if (not isinstance(c.a, Load)) else None
        # The leading (batch / ksplit) grid axes ride untiled below the ``(m, n)`` cell — the GRID's
        # fact, not the tiled cell's, so they are threaded to the emission that needs them (the
        # per-cell rename's shared coordinates) from here, where the kernel grid is in hand.
        lead = grid[:-2]
        state_decls, reduce_region = reduce_codegen(c, tile, ctx.sched.get("STAGE", c), ctx.inputs, ctx.workers, seam, lead)
        sink = store if store is not None else store_sink(c, tile, Body(tuple(epi)), lead)
        t = unit_tile(register_tile(atomize(tile.atom.shape[:2]), tile.mn), tile.mn)
        mn, bt, lanes = tile.mn, tile.launch_threads, tile.atom.lanes
    else:
        # The reduce partition rides the :class:`Fold` node; ``None`` for a pure pointwise /
        # scalar per-cell zero-axis ``Fold`` (no partition). Every partitioned reduce — monoid, flash, coop-K /
        # split contraction — is a ``Fold`` node after ``ops.nodify_reduce`` (a projecting
        # zero-axis ``Fold`` was already peeled off by :func:`_factorize`).
        plan = (ctx.sched.get("REDUCE", op) or ReducePlan()) if isinstance(op, Fold) else None
        t, mn, lead, lanes = atomize((1, 1)), (None, None), grid, 1
        wsrc = warp_source(op, ctx.sched)
        csrc = chain_source(op, ctx.sched) if wsrc is None else None
        if wsrc is not None:
            # A warp-tiled TWISTED tree (the schedule stamped mma TilePlans on its contractions):
            # the per-step values live in mma C-fragments, so the whole reduce realizes at fragment
            # residence (``_twist``) and the kernel is warp-collective — the same ``lanes`` seam the
            # output-tiled contraction arm uses. ``units[0] > 1`` warps per CTA each own their own
            # query-row block: the warp axis joins the Tile decode ahead of the lane axis, and the
            # block launches ``units[0]`` warps.
            state, fold, close = realize_warp_twist(op, ctx, tail)
            wtile = ctx.sched.tile_of(wsrc)
            lanes = wtile.atom.lanes
            um = wtile.units[0]
            if um > 1:
                t = replace(t, axes=(Axis(name=FLASH_WARP_AXIS, extent=um),))
            bt = lanes * um
        elif csrc is not None:
            # The chain schedule — the expect column axis rides a per-thread register vector (the
            # FA-2 shared-score form); one thread per (grid) cell, the column index a literal.
            state, fold, close = _realize_chain(op, ctx, tail, csrc)
            bt = None
        elif plan is None or (plan.coop <= 1 and plan.reg <= 1):
            state, fold, close, bt = [], with_store([*_emit(op, ctx).body, *tail], ctx.output, grid, out_val), [], None
        elif plan.coop_transposed:
            # The ``b<n>t`` k-major matvec partition: the innermost output axis splits into a
            # shrunk ``<out>_blk`` grid axis (×32) + the 32-wide ``n_lane`` thread axis (with
            # ``k_co`` between them), so B loads coalesce across lanes. The emitted body's
            # output-var references were σ-substituted to ``blk·32 + n_lane`` inside.
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
        # The scheduler stamps ``workers`` on a contraction row or a warp-flash (TWISTED) row only;
        # every other arm arrives with ``None`` — safe to thread unconditionally.
        workers=ctx.workers,
        raster=ctx.raster,
    )


# ---- the tiled REDUCE axis (cooperative / ILP) -------------------------------------------------- #
# A PLANAR / TWISTED monoid reduce (sum / max / mean / RMSNorm / softmax / the coop-KV TWISTED flash
# reduce) partitions the reduce axis ``coop`` ways across the CTA's threads (cooperation) and ``reg``
# ways across per-thread register accumulators (ILP). The serial reduce ``Loop`` becomes a
# :class:`StridedLoop` of step ``coop·reg``; for ``reg > 1`` its body is replicated ``reg`` times
# (each copy offset by ``r·coop`` and folding its own accumulator). After the loop: the REG tree
# folds the ``reg`` accumulators into one (``as_state_merge``), then — if ``coop > 1`` — the
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
    :func:`~emmy.compiler.ir.stmt.algebra.combine_states_of` program) drives the nodes; the
    combine renders at the accumulator dtype (fp32 for a reduction, with the fold's own dtype
    honored when set)."""
    state = red.names
    state_b = red.state_b
    prog = red.combine_states
    dtype = next((a.dtype for a in prog if a.dtype is not None), None) or F32

    from emmy.compiler.backend.cuda.dtype import cuda_name as _cuda_name  # noqa: PLC0415

    smem_c = _cuda_name(dtype)
    bufs = tuple(f"{st}_smem" for st in state)
    if inner is not None:
        # The transposed (``b<n>t``) combine: threads sharing an output lane sit ``scale``
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
    into copy 0 (``state_merge_of``), then — when threads cooperate (``lane`` is a lane :class:`Axis`,
    not ``None``) — the cross-thread :func:`emit_combine`. Both reassign the carried state **in place**
    (the survivor SSA names hold the full reduction), so the post-reduce projection reads them directly.

    Algebra-generic (the ⊕ read off the ``red`` fold node's stored combine): a monoid reduce and a
    contraction's degenerate additive fold identically, so a cooperative reduce and a (future)
    cooperative-K contraction share this tail. ``state_merge_of``
    keys its finalize temps on the copy name, so each fold's internals are already unique."""
    merge: list[Stmt] = [red.state_merge(tuple(f"{n}__r{r}" for n in red.names)) for r in range(1, reg)]
    if lane is not None:
        merge += emit_combine(red, t=lane.name, n_threads=coop)
    return merge


def chain_source(op, sched):
    """The expect fold of a TWISTED tree carrying a SCALAR register tile over its
    output column axis (the chain schedule — the column axis rides a per-thread register vector),
    or ``None``. The structural schedule read the one binder keys the chain realization on; the
    tile is a schedule slice (``sched``), never a node field."""
    red = (op.operands[0] if op.operands else None) if (isinstance(op, Fold) and op.axis is None) else op
    if not isinstance(red, Fold) or red.role is not AxisRole.TWISTED:
        return None
    pv = next((s for s in list(red.step_stmts())[1:] if is_contraction(s)), None)
    if pv is None:
        return None
    ptile = sched.tile_of(pv)
    if ptile is not None and not ptile.is_warp and ptile.regs != (1, 1):
        return pv
    return None


def _flat_stmts(stmts):
    for s in stmts:
        yield s
        for b in s.nested():
            yield from _flat_stmts(list(b))


def _stmt_axis_hit(s: Stmt, axis: str) -> bool:
    idx = getattr(s, "index", None)
    if idx and any(axis in e.free_vars() for e in idx):
        return True
    return isinstance(s, Select) and any(axis in br.select.free_vars() for br in s.branches)


def _stmt_reads(s: Stmt) -> set[str]:
    if isinstance(s, Accum):
        return {s.name, s.value}
    if isinstance(s, Select):
        return {br.value for br in s.branches}
    deps = getattr(s, "deps", None)
    return set(deps()) if callable(deps) else set(getattr(s, "args", ()) or ())


def _taint(stmts: list[Stmt], axis: str) -> frozenset[str]:
    """The SSA names transitively dependent on the free ``axis`` — the register-vector slice of the
    per-cell body (everything else is shared across the vector's columns)."""
    tainted: set[str] = set()
    changed = True
    while changed:
        changed = False
        for s in _flat_stmts(stmts):
            d = set(s.defines())
            if d and not (d <= tainted) and (_stmt_axis_hit(s, axis) or (_stmt_reads(s) & tainted)):
                tainted |= d
                changed = True
    return frozenset(tainted)


def _vectorize_axis(stmts: list[Stmt], axis: str, count: int, tainted: frozenset[str], protected: frozenset[str]) -> list[Stmt]:
    """Replicate every column-dependent stmt per register column (σ ``axis → j``, names suffixed
    ``_{j}``), keeping shared stmts single — the FA-2 shared-score restructuring: the score /
    softmax stats compute once per streamed key, the per-column slice fans out. Recurses into loop
    bodies (a loop stays single; a column-touched carrier fans out per :func:`_vector_carrier`)."""
    out: list[Stmt] = []
    for s in stmts:
        bodies = s.nested()
        if bodies:
            s = s.with_bodies(tuple(Body(tuple(_vectorize_axis(list(b), axis, count, tainted, protected))) for b in bodies))
            out.append(s)
            continue
        if (set(s.defines()) & tainted) or _stmt_axis_hit(s, axis) or (_stmt_reads(s) & tainted):
            for j in range(count):
                out += copy_cell([s], Sigma({axis: Literal(j, "int")}), f"_{j}", protected)
        else:
            out.append(s)
    return out


def _realize_chain(op, ctx: Ctx, tail: tuple, pv) -> tuple[list[Stmt], list[Stmt], list[Stmt]]:
    """Realize a chain-scheduled TWISTED tree — the ``(state, fold, close)`` triple: the per-cell
    body with the expect column axis register-vectorized (the score shared), and the projection +
    store replicated per column (the column index a literal — the axis left the grid). ``pv`` is the
    stored expect fold; its column axis is the placement's trailing free axis (it left the grid)."""
    axis = ctx.free[-1].name
    count = ctx.sched.tile_of(pv).reg_n  # the column (n) register vector
    (rloop,) = _emit(op, ctx).body
    # A layout-aware output ``Write`` the node carries (flash's ``_out_store_index``) is the store
    # TEMPLATE — its index already places the grid axes at the output buffer's real slots (transpose /
    # broadcast dims). Split it off the projection tail; the per-column store reuses its index with the
    # column (n) axis substituted by the register literal ``j``. Absent it, fall back to the bare
    # grid-cell store (grid vars + the column literal) — the head-major identity.
    store_tmpl = tail[-1] if tail and isinstance(tail[-1], Write) else None
    proj_tail = tuple(tail[:-1]) if store_tmpl is not None else tail
    all_stmts = [*rloop.body, *proj_tail]
    tainted = _taint(all_stmts, axis)
    protected = frozenset({nm for s in _flat_stmts(all_stmts) for nm in (*s.defines(), *_stmt_reads(s))} - tainted)
    body = _vectorize_axis(list(rloop.body), axis, count, tainted, protected)
    fold = [replace(rloop, body=Body(tuple(body)))]
    close = _vectorize_axis(list(proj_tail), axis, count, tainted, protected)
    if store_tmpl is not None:
        out_val = store_tmpl.values[-1]
        base_index = store_tmpl.index
    else:
        out_val = proj_tail[-1].defines()[-1] if proj_tail else pv.out
        base_index = (*(Var(a.name) for a in ctx.grid), Var(axis))
    for j in range(count):
        val = f"{out_val}_{j}" if out_val in tainted else out_val
        idx = tuple(Literal(j, "int") if (isinstance(e, Var) and e.name == axis) else e for e in base_index)
        close.append(Write(output=ctx.output, index=idx, value=val))
    return [], fold, close


def _tile_reduce_axis_transposed(
    op: Fold, plan, ctx: Ctx, tail: tuple, out_val: str
) -> tuple[list[Stmt], list[Stmt], list[Stmt], tuple[Axis, ...]]:
    """The ``b<n>t`` (transposed) cooperative reduce — the k-major-B matvec partition: 32
    ``n_lane`` threads (innermost) sweep the OUTPUT axis so B loads coalesce across lanes at
    every k step, and ``coop/32`` ``k_co`` slices ride the upper thread bits. The emitted body
    keeps referencing the original output axis var — one σ substitutes it with
    ``blk·32 + n_lane`` (the caller rebinds the shrunk ``<out>_blk`` grid axis). The combine is
    the segment-indexed smem tree (``emit_combine(inner=…)`` — never a shuffle: adjacent lanes
    hold different outputs); the projection stores guard on ``k_co == 0``, each lane writing its
    own cell. Unsupported here (the enumeration must not offer ``t`` on them): shared-row
    ``sync`` staging, distributed full-row projections (a ``Loop`` in the tail)."""
    grid = ctx.grid
    coop, reg = plan.coop, plan.reg
    lanes_n = 32
    k_ways = coop // lanes_n
    assert coop % lanes_n == 0 and k_ways >= 1, f"b{coop}t needs a multiple of {lanes_n}"
    stage = ctx.sched.get("STAGE", op)
    assert not (stage is not None and stage.smem), "transposed coop cannot ride shared-row staging"
    out_ax = next(a for a in reversed(grid) if not (a.extent.is_static and a.extent.as_static() == 1))

    (rloop,) = _emit(op, ctx).body
    alg = Reduction(op)
    axis = rloop.axis
    stride = k_ways * reg
    masked = reg > 1 and not (axis.extent.is_static and axis.extent.as_static() % stride == 0)

    n_lane = Axis(name=f"{out_ax.name}_ln", extent=lanes_n)
    k_co = Axis(name=f"{axis.name}_co", extent=k_ways) if k_ways > 1 else None
    start = Var(k_co.name) if k_co is not None else Literal(0, "int")
    blk_name = f"{out_ax.name}_blk"
    subst = Sigma({out_ax.name: BinaryExpr("+", BinaryExpr("*", Var(blk_name), Literal(lanes_n, "int")), Var(n_lane.name))})
    ident = lambda n: n  # noqa: E731

    nested_axes = {lp.axis.name for lp in rloop.body.iter_of_type(Loop, StridedLoop)}
    defined = {nm for s in rloop.body.iter() for nm in s.defines()}
    expr_external = {v for s in rloop.body.iter() for e in s.exprs() for v in e.free_vars()} - defined
    protected = frozenset(
        {axis.name, *(ax.name for ax in grid), blk_name, n_lane.name, *axis.extent_expr().free_vars(), *nested_axes, *expr_external}
        | ({k_co.name} if k_co is not None else set())
    )
    stream_identity = (str(alg.terms[0]), ElementwiseImpl("maximum").identity) if alg.twisted else None
    copies: list[Stmt] = []
    for r in range(reg):
        copies.extend(_replicate(rloop.body, r, k_ways, axis, masked, protected, stream_identity))
    strided = StridedLoop(axis=axis, start=start, step=Literal(stride, "int"), body=Body(tuple(copies)), unroll=rloop.unroll)
    strided = strided.rewrite(ident, subst)

    merge: list[Stmt] = [alg.state_merge(tuple(f"{n}__r{r}" for n in alg.names)) for r in range(1, reg)]
    if k_co is not None:
        merge += emit_combine(alg, t=k_co.name, n_threads=k_ways, inner=(n_lane.name, lanes_n))

    tail_stmts = with_store(list(tail), ctx.output, grid, out_val)
    tail_stmts = [s.rewrite(ident, subst) for s in tail_stmts]
    if k_co is not None:
        tail_stmts = [Cond(cond=BinaryExpr("==", Var(k_co.name), Literal(0, "int")), body=tuple(tail_stmts))]

    lanes_axes = ((k_co,) if k_co is not None else ()) + (n_lane,)
    return [], [strided, *merge], tail_stmts, lanes_axes


def _tile_reduce_axis(op: Fold, plan, ctx: Ctx, tail: tuple, out_val: str) -> tuple[list[Stmt], list[Stmt], list[Stmt], Axis | None]:
    """Tile the REDUCE axis per the node's cooperating :class:`ReducePlan` — the reduce counterpart
    of the output ``unit_tile`` / ``register_tile`` levels: ``coop`` lanes across threads (the
    ``_co`` lane axis, the axis's UNIT level) and ``reg`` ILP chains across per-thread accumulators
    (its REGISTER level — cyclic, copy ``r`` offset by ``r·coop``, the loop striding ``coop·reg``).
    It drives the recursion (:func:`_emit`) for the per-cell reduce loop and returns the pieces the
    one pipeline (:func:`_bind` → :func:`grid_tile`) seals: ``(state, fold, close, lane)`` — the
    shared-row fill decls, the strided fold loop + the carrier merge (the REG tree + the
    cross-thread combine), the distributed projection close, and the lane :class:`Axis` (``None``
    for standalone ILP — one thread per cell, lane fixed at 0)."""
    grid = ctx.grid
    coop, reg = plan.coop, plan.reg

    # Build the per-cell reduce loop via the recursion (:func:`_emit`) off the :class:`Fold`
    # **node** — the walk reaches any nested contraction (flash Q@K / P@V) as a node. The algebra
    # is read off the ``Fold`` node itself (:class:`Reduction` — a contraction's K fold and a
    # monoid's reduce fold both answer it, so the algebra-generic ``state_merge`` /
    # ``combine_states`` machinery folds either). A ``Fold`` has no prologue
    # ahead of its loop; the enclosing zero-axis ``Fold``'s projection is ``tail`` (already walked).
    (rloop,) = _emit(op, ctx).body
    alg = Reduction(op)
    axis = rloop.axis
    stride = coop * reg
    masked = reg > 1 and not (axis.extent.is_static and axis.extent.as_static() % stride == 0)

    # The cooperative lane axis (Tile-decoded, innermost) — present only when threads
    # cooperate; standalone ILP (coop == 1) runs one thread per cell, lane fixed at 0.
    lane = Axis(name=f"{axis.name}_co", extent=coop) if coop > 1 else None
    start = Var(lane.name) if lane is not None else Literal(0, "int")

    # Shared-row staging (the fused norm→linear prologue): an input row folded by the cooperative
    # reduce AND re-read per output column of a contraction tail rides a first-class ``sync``
    # :class:`Stage` whose ``smem`` names the row — DETECTED scheduler-side (schedule-side)
    # and only APPLIED here: fill the row into smem once (cooperatively) and rewrite both readers to
    # the slab. Only the cooperative tier (coop > 1) is ever stamped; a contraction operand ``Stage``
    # (the coop-K matmul's pinned pipeline) never sets ``smem``, so it passes through untouched.
    tail_src = list(tail)
    fill_stmts: list[Stmt] = []
    op_stage = ctx.sched.get("STAGE", op)
    if lane is not None and op_stage is not None and op_stage.smem:
        from emmy.compiler.backend.cuda.dtype import cuda_name  # noqa: PLC0415

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

    # The reduce loop: ``reg`` interleaved accumulator chains (ILP), striding the axis by
    # ``coop·reg`` from the lane's start. The dissolved fold ``Accum``\\ s seed each copy's
    # accumulator (``StridedLoop.render``).
    # The shared iteration coordinates (grid + reduce + lane axis vars) and the symbolic
    # extent's runtime arg(s) (e.g. ``seq_len``) are common to every register copy — exclude
    # them from the per-copy SSA rename. So too any NESTED loop-axis var (a flash Q@K / P@V
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
    protected = frozenset(
        {axis.name, *(ax.name for ax in grid), *axis.extent_expr().free_vars(), *nested_axes, *expr_external}
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
    merge = combine_tail(alg, reg=reg, coop=coop, lane=lane)

    # Post-reduce projection. A full-row output (softmax / RMSNorm) distributes its sweep
    # across the coop lanes; a scalar output is written once, guarded to lane 0. With no
    # cooperation (coop == 1) the single thread runs the projection as-is.
    tail = tail_src
    if lane is None:
        body_tail = with_store(tail, ctx.output, grid, out_val)
    elif any(isinstance(s, Loop) for s in tail):
        body_tail = [
            StridedLoop(axis=s.axis, start=Var(lane.name), step=Literal(coop, "int"), body=s.body, unroll=s.unroll)
            if isinstance(s, Loop)
            else s
            for s in tail
        ]
    else:
        stored = with_store(tail, ctx.output, grid, out_val)
        body_tail = [Cond(cond=BinaryExpr("==", Var(lane.name), Literal(0, "int")), body=tuple(stored))]

    return fill_stmts, [strided, *merge], body_tail, lane
