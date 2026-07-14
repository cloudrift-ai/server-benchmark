"""Operand-staging assembly for the kernel emitters (``_factor.factorize``).

The single home for every operand-staging *transport*: the warp tier's cooperative gmem→smem
2-D slab fills (``cp.async`` / TMA, off a :class:`~emmy.compiler.ir.schedule.Stage`) and
the reduce tier's ``sync`` 1-D shared-row fill (:func:`sync_row_fill`, the fused norm→linear
prologue) both live here, indexed off the same linear-tid / thread-count seam. Assembles the
surviving kernel-IR transport leaf nodes (``Smem`` / ``CpAsyncCopy`` / ``CpAsyncCommit`` /
``CpAsyncWait`` / ``Sync`` — and the TMA quartet ``TmaDescriptor`` / ``TmaLoad`` /
``Mbarrier*``).

The fill is written against a small :class:`CtaTile` thread-striping seam (a linear intra-CTA
thread id + the thread count), NOT a materializer's internal warp/register geometry — so one fill
helper drives any tier that stages; the per-operand gmem tile-base rides the :class:`Operand`. The
A/B operands themselves ride as an ``(a, b)`` :class:`Operand` pair, so a transport loops over the
pair instead of spelling A then B. The staged K-loop itself is ONE liveness-scheduled
skeleton, :func:`pipelined_kloop` — per operand-group a ``(transport, depth)`` pair, the fill /
wait / barrier placement DERIVED from which tagged body segments read each group's slabs (the
whole-body single-group entry is :func:`staged_kloop`; the warp-flash alternating form is the same
walk over its three tagged segments) — driven by a :class:`Transport` strategy
(:class:`CpAsyncTransport` / :class:`TmaTransport`) — the two producers put behind one
``fill``/``commit``/``wait`` seam. The
slab feeds the same staged ``LdmatrixLoad`` / scalar ``Load`` drain regardless of which producer
(cp.async / TMA) filled it. A slab feeding an mma drain is **swizzled** (:func:`pick_swizzle_atom`
per operand): TMA permutes the 16 B chunks in hardware during the box copy, a cp.async fill applies
the identical XOR in software on its destination index, and each staged ``LdmatrixLoad`` reads back
through the matching address XOR — which is what keeps the ldmatrix drain free of smem bank
conflicts (the unswizzled cp path was 4-way conflicted on 64 B rows / 8-way on 128 B ones — the
sm_89 gap to cuBLAS). The sync compute-fill applies the same XOR on its slab ``Write`` stores (one
vector store per 16 B run), and its per-cell producer-cone replication is planned by
:meth:`SyncTransport._run_plans` — run-invariant stmts hoist once, a stride-1 k-indexed gmem
``Load`` merges into one vector ``Load`` per run. Scalar-``Load``-drained slabs stay plain
row-major (NONE-swizzle, linear writes and reads).
``_atom._staged`` — the one atom-agnostic driver — builds the transport, asks the atom strategy
for the drain leaf, and calls :func:`staged_kloop`.

Leading ``_`` so the pass loader (globs ``*.py``, skips ``_``-prefixed) skips it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Builtin, Expr, Literal, SimplifyCtx, TernaryExpr, Var, affine_form
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    MbarrierArrive,
    MbarrierArriveExpectTx,
    MbarrierInit,
    MbarrierWait,
    SetMaxNReg,
    Smem,
    Sync,
    TmaDescriptor,
    TmaLoad,
)
from emmy.compiler.ir.stmt import Body, Cond, Load, Loop, Stmt, StridedLoop, Write


def _mul(a: Expr, b: Expr) -> Expr:
    return BinaryExpr("*", a, b)


def _add(a: Expr, b: Expr) -> Expr:
    return BinaryExpr("+", a, b)


def _lit(n: int) -> Expr:
    return Literal(int(n), "int")


@dataclass(frozen=True)
class CtaTile:
    """The tile-agnostic thread-striping seam a cooperative fill indexes off — a linear intra-CTA
    thread id + the CTA thread count. Built from a materializer's decoded grid vars (the warp tier's
    ``(m_w·WN + n_w)·32 + lane`` linear id), so the tier's geometry never leaks into the fill. The
    per-operand gmem tile-base rides the :class:`Operand` instead."""

    linear_tid: Expr  # intra-CTA linear thread id (0 .. n_threads-1)
    n_threads: int


def _cp_async_width(slab_cols: int, elem_bytes: int) -> int:
    """Elements per ``cp.async`` — the widest contiguous run whose byte size is a
    legal cp.async width (4 / 8 / 16) and that divides the inner (contiguous) slab
    extent (a chunk never straddles a slab row). The slab's inner dim maps stride-1
    to the gmem inner dim (canonical A[m,k] / B[k,n]), so a V-run is contiguous in
    both."""
    for nbytes in (16, 8, 4):
        v = nbytes // elem_bytes
        if v >= 1 and slab_cols % v == 0:
            return v
    return 1  # elem_bytes > 16 — never (fp16/bf16/fp32 only)


def cp_async_fill(
    *,
    slab: str,
    shape: tuple[int, int],
    src: str,
    gmem_index,
    cta: CtaTile,
    elem_bytes: int,
    name: str,
    row_offset: Expr | None = None,
    swizzle: str = "NONE",
) -> list[Stmt]:
    """Cooperatively ``cp.async``-copy a ``rows × cols`` (= ``shape``) row-major smem
    ``slab`` from gmem ``src``. ``gmem_index(row_expr, col_expr)`` returns the gmem
    index tuple for slab cell ``(row, col)``. The CTA's ``n_threads`` lanes stripe
    ``rows·cols / V``-element chunks (``V`` = :func:`_cp_async_width`); each
    lane runs ``for e = tid; e < n_chunks; e += n_threads``. Emits the fill loop only
    — the caller appends one ``CpAsyncCommit`` + ``CpAsyncWait`` + ``Sync`` after the
    A and B fills together. The loop bound (not a predicate) masks the tail, so every
    lane still reaches the shared barrier (the barrier-under-mask invariant).

    ``row_offset`` (the gmem→smem ring): when staging through a depth>1 slab, it picks
    the ring SLOT — the write row becomes ``row_offset + row`` (each slot is a contiguous
    ``rows``-row block), so the fill targets one slot while the drain reads another.

    ``swizzle`` (the cp.async SOFTWARE slab swizzle): XOR the flattened destination element
    index (``CpAsyncCopy.swizzle`` → ``emmy_swizzle_<mode>``), permuting 16-byte chunks within
    each swizzle row exactly as the TMA hardware would — the staged ``LdmatrixLoad`` drain
    reads back through the same XOR, so fill and drain agree by construction and the un-XORed
    NONE-swizzle kernel is bit-identical. This is the sm_89 bank-conflict fix: the plain
    row-major slab leaves the drain 4-way (64 B rows) / 8-way (128 B rows) conflicted."""
    slab_rows, slab_cols = shape
    v = _cp_async_width(slab_cols, elem_bytes)
    n_chunks = (slab_rows * slab_cols) // v
    fe = Axis(name=f"_f{name}", extent=n_chunks)
    base = _mul(Var(fe.name), _lit(v))  # flat element offset of this chunk
    row = BinaryExpr("/", base, _lit(slab_cols))
    col = BinaryExpr("%", base, _lit(slab_cols))
    smem_row = _add(row_offset, row) if row_offset is not None else row
    copy = CpAsyncCopy(
        smem=slab,
        smem_index=(smem_row, col),
        src=src,
        src_index=tuple(gmem_index(row, col)),
        nbytes=v * elem_bytes,
        swizzle=swizzle,
    )
    loop = StridedLoop(axis=fe, start=cta.linear_tid, step=_lit(cta.n_threads), body=Body((copy,)), unroll=False)
    return [loop]


def cp_async_commit() -> list[Stmt]:
    """Close the current cp.async batch into a commit-group (the depth-``D`` ring commits one
    group per filled slot, so ``CpAsyncWait(group=D-1)`` can drain exactly the slot it needs)."""
    return [CpAsyncCommit()]


def cp_async_wait(group: int) -> list[Stmt]:
    """Wait until at most ``group`` cp.async commit-groups remain in flight, then a CTA barrier
    so every lane sees the drained slot (the depth-``D`` ring's per-chunk handshake — the commit
    happened in the prefetch fill above)."""
    return [CpAsyncWait(group=group), Sync()]


def slab_smem(name: str, rows: int, cols: int, dtype: str, *, align: int = 0) -> Smem:
    """A row-major ``rows × cols`` operand slab; ``ldm`` (the staged-load row stride)
    is ``cols``. ``align`` stamps an explicit byte alignment — TMA destination slabs
    need 128 B (``cp.async.bulk.tensor`` requires an aligned smem base)."""
    return Smem(name=name, extents=(rows, cols), dtype=dtype, align=align)


def sync_row_fill(*, slab: str, src: str, extent: int, grid_vars: tuple, linear_tid: Expr, n_threads: int, dtype: str) -> list[Stmt]:
    """The ``sync``-transport 1-D operand fill: cooperatively copy the CTA-shared row
    ``src[grid…, 0:extent]`` into a length-``extent`` smem ``slab``, then a CTA barrier so
    every lane sees the filled row before the reader drains it. The ``n_threads`` cooperating
    lanes stripe it (``for k = linear_tid; k < extent; k += n_threads``), the same
    linear-tid / thread-count seam :func:`cp_async_fill` indexes off — so every transport's
    fill lives here. This is the scalar reduce tier's shared-row prologue (the fused
    norm→linear input row), the single-buffer ``sync`` counterpart of the async 2-D slab
    fills above."""
    fe = Axis(name=f"_{slab}_f", extent=extent)
    val = f"_{slab}_v"
    load = Load(name=val, input=src, index=(*grid_vars, Var(fe.name)))
    write = Write(output=slab, index=(Var(fe.name),), value=val)
    loop = StridedLoop(axis=fe, start=linear_tid, step=_lit(n_threads), body=Body((load, write)), unroll=False)
    return [Smem(name=slab, extents=(extent,), dtype=dtype), loop, Sync()]


def sync_stat_fill(
    *, stats: tuple[str, ...], slab_of, row_axis: Axis, row_body: list[Stmt], cta: CtaTile, dtype: str = "float"
) -> list[Stmt]:
    """The ``sync``-transport per-row STATISTIC prologue — the fused norm→linear warp edge's
    cooperative prologue, run ONCE before the staged K-loop: the CTA stripes the tile's rows **one
    row per WARP** (``for r = warp; r < rows; r += n_warps``); the warp's 32 lanes stride the row's
    stat reduce ``Loop`` (coalesced — consecutive lanes read consecutive elements) and close the
    fold with the carrier's shuffle butterfly (:func:`emit_combine` at warp width — the state
    broadcasts to every lane), each lane then runs the scalar epilogue redundantly and lane 0
    writes each bridged ``stats`` value into its length-``rows`` smem row (``slab_of(name)``); one
    CTA barrier publishes them to the A compute-fill. A ``row_body`` with no foldable reduce
    ``Loop`` (or a sub-warp CTA) falls back to the serial one-row-per-THREAD stripe."""
    decls: list[Stmt] = [Smem(name=slab_of(nm), extents=(row_axis.extent.as_static(),), dtype=dtype) for nm in stats]
    writes = tuple(Write(output=slab_of(nm), index=(Var(row_axis.name),), value=nm) for nm in stats)
    rl_i = next((i for i, s in enumerate(row_body) if isinstance(s, Loop) and s.is_reduce and s.carrier is not None), None)
    if rl_i is None or cta.n_threads % 32 or cta.n_threads < 32:
        body = (*row_body, *writes)
        loop = StridedLoop(axis=row_axis, start=cta.linear_tid, step=_lit(cta.n_threads), body=Body(body), unroll=False)
        return [*decls, loop, Sync()]
    from emmy.compiler.pipeline.passes.lowering.kernel._factor import emit_combine  # noqa: PLC0415 — avoid an import cycle

    rl = row_body[rl_i]
    lane = BinaryExpr("%", cta.linear_tid, _lit(32))
    warp = BinaryExpr("/", cta.linear_tid, _lit(32))
    fold = StridedLoop(axis=rl.axis, start=lane, step=_lit(32), body=rl.body, unroll=False)
    combine = emit_combine(rl.carrier, t="_lane", n_threads=32)
    guarded = Cond(cond=BinaryExpr("==", lane, _lit(0)), body=writes)
    body = (*row_body[:rl_i], fold, *combine, *row_body[rl_i + 1 :], guarded)
    loop = StridedLoop(axis=row_axis, start=warp, step=_lit(cta.n_threads // 32), body=Body(body), unroll=False)
    return [*decls, loop, Sync()]


# --------------------------------------------------------------------------- #
# TMA (``cp.async.bulk.tensor``) descriptor — the host-built ``CUtensorMap`` the
# box copies index off (the copy + mbarrier handshake live in :class:`TmaTransport`
# below). Shares the slab + staged-``LdmatrixLoad`` drain with the cp.async path.
# --------------------------------------------------------------------------- #

# TMA destination smem must be aligned (``cp.async.bulk.tensor`` faults otherwise);
# 128 B satisfies the NONE-swizzle box copy. A swizzled slab aligns to the full swizzle
# atom (8 rows × width) — the coordinate-only ldmatrix XOR only reproduces the hardware
# deposit when the slab base zeroes the swizzle's source address bits.
TMA_SLAB_ALIGN = 128
_SWIZZLE_SLAB_ALIGN = {"NONE": TMA_SLAB_ALIGN, "B32": 256, "B64": 512, "B128": 1024}

# TMA hardware-swizzle atom widths in bytes, widest-first. The widest atom that divides a
# slab's inner-row byte span wins (best bank-conflict spread on the ldmatrix drain).
_SWIZZLE_BY_BYTES: tuple[tuple[int, str], ...] = ((128, "B128"), (64, "B64"), (32, "B32"))


def pick_swizzle_atom(inner_elems: int, elem_bytes: int) -> tuple[int, str]:
    """The swizzle atom for a slab whose inner (contiguous) row is ``inner_elems``
    elements of ``elem_bytes`` — the ONE mode derivation the TMA descriptor (hardware
    in-copy swizzle) and the cp.async fill (software destination XOR) both consume.
    Returns ``(atom_elems, mode)``: the widest atom in
    ``{128, 64, 32}`` B that fits and divides the span wins — a 256 B inner (128 fp16
    elems) picks ``B128`` (64 elems, descriptor box split ``[2, 64]``); a 64 B inner (32
    fp16 elems) picks ``B64`` (32 elems, no split). ``(inner_elems, "NONE")`` when no
    atom fits (the descriptor keeps the unswizzled box). Shared by the descriptor build,
    the box-coordinate split, and the drain's per-slab mode derivation so all three agree
    (a disagreement makes the descriptor's swizzle claim a width its box doesn't have —
    a TMA copy deadlock)."""
    inner_bytes = inner_elems * elem_bytes
    for wb, mode in _SWIZZLE_BY_BYTES:
        we = wb // elem_bytes
        if 1 <= we <= inner_elems and inner_elems % we == 0 and inner_bytes % wb == 0:
            return we, mode
    return inner_elems, "NONE"


def tma_descriptor(name: str, src: str, box: tuple[int, int], dtype: str, *, swizzle: str = "NONE", elem_bytes: int = 4) -> TmaDescriptor:
    """A host-encoded ``CUtensorMap`` for the operand ``src`` with a ``box`` tile
    (C-order). The source globalDim is resolved from the bound array at launch, so a
    symbolic (masked-M) extent rides the runtime shape and TMA zero-fills the box
    overhang. ``swizzle="NONE"``: the plain row-major slab feeds the staged
    ``LdmatrixLoad`` drain directly. A swizzled descriptor whose inner box span exceeds
    the swizzle atom SPLITS the inner dim down to the atom (TMA rejects a swizzle
    narrower than the inner box dim; the linear smem deposit is unchanged — ``[rows,
    cols/atom, atom]`` is the same contiguous layout), the matching box-coordinate split
    living in :meth:`TmaTransport.fill`."""
    if swizzle != "NONE":
        atom, _ = pick_swizzle_atom(box[-1], elem_bytes)
        if atom < box[-1]:
            box = (*box[:-1], box[-1] // atom, atom)
    return TmaDescriptor(name=name, src_buf=src, src_shape=(), box_extents=box, swizzle=swizzle, dtype=dtype)


# --------------------------------------------------------------------------- #
# The Transport strategy — the one interface the staged K-loop drives, and the
# two producers behind it (cp.async / TMA). A :class:`Transport` owns the operand
# slab layout + the fill/commit/wait handshake; :func:`staged_kloop` owns the
# depth-parametrized control flow. The two are structurally different primitives
# (cp.async is fill → commit → wait-group; TMA is an arrive/expect-tx + box copy
# gated by an mbarrier phase) put behind ONE seam so ``depth`` becomes the sole
# buffering knob: ``depth == 1`` is the degenerate single-buffer loop, ``depth >= 2``
# a gmem→smem prefetch ring.
# --------------------------------------------------------------------------- #


def _slot_row(slot: Expr, rows_per_slot: int) -> Expr | None:
    """The row offset of ring ``slot`` — ``slot·rows_per_slot``, or ``None`` for a literal
    slot 0 (the single-buffer / ring-slot-0 case). ``None`` keeps the emitted index free of a
    dead ``+ 0·rows`` term, so single-buffer staging stays bit-identical to its gmem baseline."""
    if isinstance(slot, Literal) and slot.value == 0:
        return None
    return _mul(slot, _lit(rows_per_slot))


@dataclass(frozen=True)
class Operand:
    """One staged operand — A or B — the per-operand slab geometry both transports index off. The two
    ride as an ``(a, b)`` pair so :meth:`CpAsyncTransport.fill` / ``slab_decls`` loop over them instead
    of spelling A then B. ``shape`` is ``(rows, cols)`` of one ring slot (A ``(tile_m, bk)`` / B
    ``(bk, tile_n)``); each slab is ``ring·rows`` rows (slot ``s`` at row ``s·rows``), plain row-major /
    NONE-swizzle, feeding the same staged ``LdmatrixLoad`` (mma) / scalar ``Load`` drain. ``index`` maps
    a K-chunk offset ``k0`` to the cp.async ``(row, col) → gmem-index`` closure; ``coords`` maps ``k0`` to
    the TMA box origin. Each transport reads only the one it needs."""

    tag: str  # "a" / "b" — the cp.async fill name + the smem-slab / descriptor suffix
    buf: str  # gmem source buffer
    shape: tuple[int, int]  # (rows, cols) of one ring slot
    index: Callable[[Expr], Callable]  # k0 -> ((row, col) -> gmem index)   (cp.async)
    coords: Callable[[Expr], tuple]  # k0 -> gmem box origin                (TMA)
    # Smem swizzle mode this operand's slab is written with ("NONE"/"B32"/"B64"/"B128") — derived
    # by the mma tier (`_MmaOps.slab_swizzles`). TMA transport: the hardware swizzles in-copy; the
    # cp.async transport applies the same XOR in SOFTWARE on each fill's destination index
    # (:func:`cp_async_fill` ``swizzle``). Either way the ldmatrix drain applies the matching XOR;
    # a scalar plain-`Load` drain and the sync write path stay NONE.
    swizzle: str = "NONE"
    # Extra smem columns padding each slab row (cp.async transport only — a TMA box deposit is
    # dense). The pad breaks the same-bank stride of a power-of-two row — the flash K/V slabs'
    # bank-conflict fix (`_twist._PAD`), whose drains are not plain ldmatrix rows; the mma matmul
    # tier uses the software ``swizzle`` above instead (zero smem growth). Mutually exclusive with
    # a non-NONE ``swizzle`` (the XOR's atom derivation assumes dense rows). The fill's logical
    # (row, col) writes stride the padded decl automatically (``render_index`` flattens against
    # it), and the drain's ``ldm`` carries the padded stride. Data cells only — a chunk never
    # straddles the pad.
    pad_cols: int = 0
    # The TMA descriptor box, when it differs from the 2-D slab ``shape`` — a BATCHED operand
    # (flash K/V: gmem ``(B, H, S, D)``) boxes as rank-N with leading extent-1 dims
    # (``(1, 1, bn, head_dim)``); the ``coords`` closure supplies the matching leading origin
    # coordinates (the load's own batch/head index exprs — GQA's ``h // group`` rides through).
    # ``None`` = the 2-D ``shape`` (the matmul tier's plain operands).
    box: tuple[int, ...] | None = None
    # This operand's OWN element dtype / size, when it differs from the transport-level
    # ``slab_dtype`` / ``elem_bytes`` (a mixed-dtype scalar contraction — fp32 A × fp16 B, the
    # norm→linear split-combine shape). ``None`` inherits the transport's. Sizing the B fill with
    # A's element width issued 16 B ``cp.async`` chunks at fp32 spacing over fp16 memory —
    # misaligned addresses + overlapped data (the Gemma ``k_linear_reduce`` bench_fail cluster).
    dtype: str | None = None
    elem_bytes: int | None = None

    @property
    def slab(self) -> str:
        return f"_{self.tag}_smem"

    @property
    def desc(self) -> str:
        return f"_desc_{self.tag}"

    def slot_row(self, slot: Expr) -> Expr | None:
        """The ring-slot row offset into this operand's multi-slot slab (``None`` for slot 0)."""
        return _slot_row(slot, self.shape[0])


@dataclass(frozen=True)
class SyncOperand:
    """One ``sync``-transport slab operand — filled by plain per-thread COMPUTE / COPY, not an
    async copy. ``value(k0, row, col)`` returns the stmts producing the cell's value at slab
    coords ``(row, col)`` of the K-chunk at ``k0`` + the SSA name holding it: a gmem ``Load`` for
    a copy fill, the fused producer CONE for a compute fill (the fused-edge A operand — the
    computed tile materializes straight into the slab the ``ldmatrix`` drain reads)."""

    tag: str  # "a" / "b" — the smem-slab suffix
    shape: tuple[int, int]  # (rows, cols) of one ring slot
    value: Callable[[Expr, Expr, Expr], tuple[list[Stmt], str]]  # (k0, row, col) -> (stmts, name)
    # Smem swizzle mode this slab is written with ("NONE"/"B32"/"B64"/"B128") — the mma tier's
    # `_MmaOps.slab_swizzles`, applied by the fill's ``Write`` (the same flattened-index XOR the
    # cp.async fill uses) and read back by the ``ldmatrix`` drain. NONE outside the mma tier
    # (a plain-``Load`` drain cannot read a swizzled slab).
    swizzle: str = "NONE"

    @property
    def slab(self) -> str:
        return f"_{self.tag}_smem"

    def slot_row(self, slot: Expr) -> Expr | None:  # noqa: ARG002
        """Always ``None`` — a sync-filled slab is SINGLE-BUFFER by construction: the compute /
        copy fill runs on the drain's own threads, so a prefetch slot for it buys no overlap
        (the work is serial either way) while doubling the slab. The sync transport's ``depth``
        rings its ``async_operands`` only (:class:`SyncTransport`)."""
        return None


@dataclass(frozen=True)
class SyncTransport:
    """The ``sync`` producer — per-thread compute/copy fills closed by ONE CTA barrier. This is the
    mma tier's ``sync`` transport: the fused-edge compute-fill (a producer cone materializing the A
    tile) rides ``operands``; plain-copy operands (the fused edge's B weights) ride
    ``async_operands`` as vectorized ``cp.async`` fills issued BEFORE the compute fill, so the
    hardware copies fly underneath it — the same ``fill``/``commit``/``wait`` seam as the pure
    cp.async / TMA producers, closed by one ``CpAsyncWait`` + CTA barrier. ``depth >= 2`` is the
    **asymmetric (B-only) prefetch ring**: only the ``async_operands`` slabs ring (the cp.async
    B copies for chunk ``i+ring-1`` fly under the A fill AND the drain of chunk ``i``), while the
    sync-filled slabs stay single-buffer and fill the CURRENT chunk off the skeleton's ``k0_cur``
    handle — ringing a compute fill buys no overlap (it runs on the drain's own threads). NOTE:
    even the B-only ring measured slower on the reference gemma shape (the extra B slot crosses
    the smem occupancy quantization) — the ring stays pin-only, ``d1`` the deployable form.

    The compute fill assigns each thread a run of ``V`` **contiguous** slab cells (``V`` = the
    16-byte vector width, always dividing the slab's inner extent): the ``row``/``col`` derivation
    hoists out of the per-cell code (one div/mod per run, not per cell), the per-thread gmem reads
    and smem stores are contiguous (nvcc merges them into wide accesses), and the cone stmts are
    replicated per lane-local cell with a ``__c<j>`` SSA suffix."""

    operands: tuple[SyncOperand, ...]
    slab_dtype: str
    cta: CtaTile
    # The optional one-shot per-row statistic prologue (:func:`sync_stat_fill` — the fused
    # norm→linear cone's cooperative reduce), emitted once ahead of the K-loop.
    prologue_stmts: tuple[Stmt, ...] = ()
    # Plain-copy operands filled by vectorized ``cp.async`` instead of the per-thread compute loop.
    async_operands: tuple[Operand, ...] = ()
    elem_bytes: int = 2

    def slab_decls(self, ring: int) -> list[Stmt]:
        # Sync-filled slabs are single-buffer (see :meth:`SyncOperand.slot_row`); only the
        # async (cp.async B) slabs allocate the ring.
        return [
            *(slab_smem(op.slab, op.shape[0], op.shape[1], self.slab_dtype) for op in self.operands),
            *(slab_smem(op.slab, ring * op.shape[0], op.shape[1], self.slab_dtype) for op in self.async_operands),
        ]

    def prologue(self, ring: int) -> list[Stmt]:  # noqa: ARG002
        return list(self.prologue_stmts)

    @staticmethod
    def _affine_step(a: Expr, b: Expr, ctx: SimplifyCtx) -> int | None:
        """The literal difference ``b - a`` when both are affine over the same free vars with
        equal coefficients (``affine_form`` — the vectorizer's decomposition; a plain
        ``simplify`` cannot fold ``(k0 + 1) - k0``), else ``None``."""
        free = a.free_vars() | b.free_vars()
        fa, fb = affine_form(a, free), affine_form(b, free)
        if fa is None or fb is None or fa[1] != fb[1]:
            return None
        d = BinaryExpr("-", fb[0], fa[0]).simplify(ctx)
        return d.value if isinstance(d, Literal) and isinstance(d.value, int) else None

    @staticmethod
    def _run_plans(op: SyncOperand, v: int) -> list[str]:
        """Per-position emission plan for one operand's ``V``-cell fill run — ``"hoist"`` /
        ``"vector"`` / ``"cell"``. Classified by PROBING the value closure at synthetic coords
        (col 0 vs col 1; the probe stmts are planning-only, never emitted):

        - identical at both cols ⇒ run-INVARIANT (the stat-row loads, whose index is the run's
          row alone) — emit once, unsuffixed, but only while its SSA deps stay outside the
          per-cell defs (a dep on a replicated name demotes it back to per-cell);
        - a scalar ``Load`` whose last-dim index advances by exactly +1 per cell, leading dims
          col-free, and whose ``col=0, k0=0`` anchor simplifies to a ``V``-aligned literal (the
          cone's k-indexed operand read; K-divisibility eligibility makes the buffer's k extent
          ``V``-aligned) ⇒ the run's V scalar loads merge into ONE vector ``Load``;
        - anything else replicates per cell (the transposed-B copy's strided gather stays
          per-cell — its last-dim step is the gmem row stride, not 1)."""
        probe_row, probe_k0, zero = Var("__srow"), Var("__sk0"), Literal(0, "int")
        s0, _ = op.value(probe_k0, probe_row, zero)
        s1, _ = op.value(probe_k0, probe_row, Literal(1, "int"))
        sz, _ = op.value(zero, probe_row, zero)
        ctx = SimplifyCtx.empty()
        plans: list[str] = []
        cell_defs: set[str] = set()
        for p, (a, b) in enumerate(zip(s0, s1, strict=True)):
            if "\n".join(a.pretty()) == "\n".join(b.pretty()) and not (set(a.deps()) & cell_defs):
                plans.append("hoist")
                continue
            step = SyncTransport._affine_step(a.index[-1], b.index[-1], ctx) if isinstance(a, Load) and isinstance(b, Load) else None
            anchor = sz[p].index[-1].simplify(ctx) if isinstance(sz[p], Load) else None
            if (
                isinstance(a, Load)
                and a.is_scalar
                and b.is_scalar
                and a.input == b.input
                and len(a.index) == len(b.index)
                and all(x.pretty() == y.pretty() for x, y in zip(a.index[:-1], b.index[:-1], strict=True))
                and step == 1
                and isinstance(anchor, Literal)
                and anchor.value % v == 0
            ):
                plans.append("vector")
            else:
                plans.append("cell")
            cell_defs |= set(a.defines())
        return plans

    def fill(self, *, k0: Expr, slot: Expr, k0_cur: Expr | None = None) -> list[Stmt]:
        out: list[Stmt] = []
        # Issue the async copies FIRST — at ``depth 1`` they are in flight while the compute fill
        # below runs; at ``depth >= 2`` (the B-only ring) ``k0``/``slot`` are the skeleton's
        # PREFETCH chunk/slot, so they additionally fly under the previous chunk's drain.
        for op in self.async_operands:
            out += cp_async_fill(
                slab=op.slab,
                shape=op.shape,
                src=op.buf,
                gmem_index=op.index(k0),
                cta=self.cta,
                elem_bytes=self.elem_bytes,
                name=op.tag,
                row_offset=op.slot_row(slot),
                swizzle=op.swizzle,
            )
        # The sync (compute / copy) fills target the CURRENT chunk in their single-buffer slabs —
        # ``k0_cur`` (== ``k0`` at depth 1; the un-clamped loop base under a prefetch ring). The
        # ring-priming call passes ``k0_cur=None``: there is no current chunk yet, so only the
        # async prefetches issue.
        if k0_cur is None:
            return out
        for op in self.operands:
            rows, cols = op.shape
            v = _cp_async_width(cols, self.elem_bytes)
            fe = Axis(name=f"_f{op.tag}", extent=(rows * cols) // v)
            base = _mul(Var(fe.name), _lit(v))
            row = BinaryExpr("/", base, _lit(cols))  # constant across the run: v divides cols
            col = BinaryExpr("%", base, _lit(cols))
            smem_row = row
            plans = self._run_plans(op, v)
            body: list[Stmt] = []
            vals: list[str] = []
            cell_stmts: list[list[Stmt]] = []
            for j in range(v):
                cell_col = _add(col, _lit(j)) if j else col
                stmts, val = op.value(k0_cur, row, cell_col)
                cell_stmts.append(stmts)
                vals.append(f"{val}__c{j}")
            # Per-cell SSA defs — the names each replica suffixes. HOISTED positions (run-invariant
            # stmts: the stat-row loads, whose value is identical across the run's cells) emit once,
            # unsuffixed, and per-cell references pass through to them; VECTOR positions (a scalar
            # gmem ``Load`` whose last-dim index advances by exactly +1 per cell — the cone's
            # k-indexed operand read) merge the run's V scalar 2 B loads into ONE vector ``Load``
            # binding every cell's suffixed name (one 16 B ld like the cp.async fill, instead of V
            # scalar loads — the compute fill issued 3.6x cuBLAS's LSU instructions). Everything
            # else replicates per cell as before.
            local = {nm for p, st in enumerate(cell_stmts[0]) if plans[p] != "hoist" for nm in st.defines()}
            for p in range(len(cell_stmts[0])):
                if plans[p] == "hoist":
                    body.append(cell_stmts[0][p])
                    continue
                if plans[p] == "vector":
                    ld = cell_stmts[0][p]
                    body.append(Load(names=tuple(f"{ld.names[0]}__c{j}" for j in range(v)), input=ld.input, index=ld.index, dtype=ld.dtype))
                    continue
                for j in range(v):
                    sfx = f"__c{j}"
                    ren = lambda nm, sfx=sfx, local=local: f"{nm}{sfx}" if nm in local else nm  # noqa: E731
                    body.append(cell_stmts[j][p].rewrite(ren))
            # ONE vector Write per run (the run is ``V`` contiguous 16-byte-aligned cells, so the
            # store is a single st.128 like the cp.async fill's chunk deposit) — V scalar 2 B
            # stores at 16 B thread stride were 8-way bank-conflicted, and the downstream
            # vectorizer cannot re-merge them (the run base's ``(fe·V) % cols`` anchor defeats
            # its affine alignment proof). A swizzled slab relocates the whole aligned chunk
            # (the XOR passes intra-chunk bits through), so the vector store stays legal.
            body.append(Write(output=op.slab, index=(smem_row, col), values=tuple(vals), swizzle=op.swizzle))
            out.append(StridedLoop(axis=fe, start=self.cta.linear_tid, step=_lit(self.cta.n_threads), body=Body(tuple(body)), unroll=False))
        return out

    def commit(self) -> list[Stmt]:
        return cp_async_commit() if self.async_operands else []

    def wait(self, *, in_flight: int, slot: Expr, phase: Expr) -> list[Stmt]:  # noqa: ARG002
        return cp_async_wait(in_flight) if self.async_operands else [Sync()]


@dataclass(frozen=True)
class CpAsyncTransport:
    """The cp.async producer: cooperative gmem→smem fills committed into groups, drained by
    ``CpAsyncWait(group=in_flight)``. ``operands`` is the ``(A, B)`` :class:`Operand` pair; each
    ``operand.index(k0)`` is the per-cell ``(row, col) → gmem-index`` closure (its tier bakes in the
    masked-axis clamp)."""

    operands: tuple[Operand, Operand]
    slab_dtype: str
    elem_bytes: int
    cta: CtaTile

    def slab_decls(self, ring: int) -> list[Stmt]:
        return [slab_smem(op.slab, ring * op.shape[0], op.shape[1] + op.pad_cols, op.dtype or self.slab_dtype) for op in self.operands]

    def prologue(self, ring: int) -> list[Stmt]:
        return []

    def fill(self, *, k0: Expr, slot: Expr, k0_cur: Expr | None = None) -> list[Stmt]:  # noqa: ARG002 — k0_cur is the sync transport's current-chunk handle
        out: list[Stmt] = []
        for op in self.operands:
            # The software swizzle XOR assumes dense power-of-two rows; a padded slab already
            # breaks the same-bank stride another way — the two fixes are mutually exclusive.
            assert op.swizzle == "NONE" or op.pad_cols == 0, "a padded slab must stay NONE-swizzle"
            out += cp_async_fill(
                slab=op.slab,
                shape=op.shape,
                src=op.buf,
                gmem_index=op.index(k0),
                cta=self.cta,
                elem_bytes=op.elem_bytes or self.elem_bytes,
                name=op.tag,
                row_offset=op.slot_row(slot),
                swizzle=op.swizzle,
            )
        return out

    def commit(self) -> list[Stmt]:
        return cp_async_commit()

    def wait(self, *, in_flight: int, slot: Expr, phase: Expr) -> list[Stmt]:
        return cp_async_wait(in_flight)  # keep ``in_flight`` prefetch groups outstanding + a CTA barrier


@dataclass(frozen=True)
class TmaTransport:
    """The TMA (``cp.async.bulk.tensor``) producer: one thread issues an ``arrive.expect_tx`` + a box
    copy per :class:`Operand` onto a **per-slot mbarrier array**; every thread waits on the slot's
    parity. The multi-slot mbarrier is what makes ``depth`` a free knob for TMA — ``wait(slot, phase)``
    gates the ring slot the same way ``CpAsyncWait(in_flight)`` gates a commit group."""

    operands: tuple[Operand, Operand]
    slab_dtype: str
    elem_bytes: int
    cta: CtaTile
    mbar: str = "_mbar"

    @property
    def _tid0(self) -> Expr:
        return BinaryExpr("==", self.cta.linear_tid, _lit(0))

    @property
    def _total_bytes(self) -> int:
        return sum(math.prod(op.shape) * (op.elem_bytes or self.elem_bytes) for op in self.operands)

    def slab_decls(self, ring: int) -> list[Stmt]:
        # TMA destination smem must be 128 B-aligned (a swizzled slab to its full atom period,
        # so the drain's from-base XOR matches the hardware deposit); one mbarrier per ring slot.
        decls: list[Stmt] = [
            tma_descriptor(
                op.desc,
                op.buf,
                op.box or op.shape,
                op.dtype or self.slab_dtype,
                swizzle=op.swizzle,
                elem_bytes=op.elem_bytes or self.elem_bytes,
            )
            for op in self.operands
        ]
        decls += [
            slab_smem(op.slab, ring * op.shape[0], op.shape[1], op.dtype or self.slab_dtype, align=_SWIZZLE_SLAB_ALIGN[op.swizzle])
            for op in self.operands
        ]
        decls.append(Smem(name=self.mbar, extents=(ring,), dtype="unsigned long long"))
        return decls

    def _box_coords(self, op: Operand, k0: Expr) -> tuple:
        """The operand's TMA box-origin coordinates at K-chunk ``k0`` — split to match a
        swizzle-split descriptor box: when the box's inner dim was split down to the swizzle atom,
        the origin's inner coordinate divides by the atom width and a literal-0 atom coordinate is
        appended (the origin is always atom-aligned: the inner coordinate is a multiple of the slab
        inner span — ``bk_elems`` steps for A, ``tile_n`` blocks for B — which the atom divides)."""
        coords = op.coords(k0)
        if op.swizzle == "NONE":
            return coords
        inner = (op.box or op.shape)[-1]
        atom, _ = pick_swizzle_atom(inner, op.elem_bytes or self.elem_bytes)
        if atom >= inner:
            return coords
        return (*coords[:-1], BinaryExpr("/", coords[-1], _lit(atom)), _lit(0))

    def prologue(self, ring: int) -> list[Stmt]:
        # Init every ring slot's mbarrier (one producer ``arrive`` per phase), then a CTA barrier so
        # every consumer sees the init before its first wait.
        inits = tuple(MbarrierInit(mbar=self.mbar, count=1, slot=_lit(s)) for s in range(ring))
        return [Cond(cond=self._tid0, body=inits), Sync()]

    def fill(self, *, k0: Expr, slot: Expr, k0_cur: Expr | None = None) -> list[Stmt]:  # noqa: ARG002 — k0_cur is the sync transport's current-chunk handle
        body: list[Stmt] = [MbarrierArriveExpectTx(mbar=self.mbar, bytes_=self._total_bytes, slot=slot)]
        for op in self.operands:
            body.append(
                TmaLoad(
                    smem=op.slab,
                    smem_index=(op.slot_row(slot) or _lit(0), _lit(0)),
                    desc=op.desc,
                    coords=self._box_coords(op, k0),
                    mbar=self.mbar,
                    mbar_slot=slot,
                )
            )
        return [Cond(cond=self._tid0, body=tuple(body))]

    def commit(self) -> list[Stmt]:
        return []  # TMA has no commit-group; the arrive.expect_tx above already armed the barrier

    def wait(self, *, in_flight: int, slot: Expr, phase: Expr) -> list[Stmt]:
        return [MbarrierWait(mbar=self.mbar, phase=phase, slot=slot)]


# The warp-specialized slot-release ring — consumers arrive after draining a slot, the producer
# parity-waits before refilling it (the reverse handshake the uniform path's trailing CTA barrier
# provided). One u64 mbarrier per ring slot, arrive count = the compute-thread count.
_EMPTY_MBAR = "_mbar_empty"

# Register redistribution between the bands (``setmaxnreg``, sm_90+ — every warp-spec kernel is
# TMA-gated, so the ``sm_<nn>a`` compile arch is already in effect). The proven pre-rebuild split:
# producers drop to the 24-register floor, consumers raise to 240. ``setmaxnreg.inc`` claims from
# the SM's per-CTA pool, so emit the pair only when the raised total provably fits the 64K-register
# file — past that envelope the split still runs, just without redistribution.
_PRODUCER_REGS = 24
_CONSUMER_REGS = 240
_SM_REGFILE = 65536


def _wspec_kloop(
    *,
    transport: TmaTransport,
    drain: Callable[[Expr], list[Stmt]],
    ring: int,
    bk_elems: int,
    n_chunks: int,
    k_extent: int,
    aux_threads: int,
    block_threads: int,
    k_end: Expr | None = None,
) -> tuple[list[Stmt], list[Stmt]]:
    """The warp-SPECIALIZED staged K-loop — the same fill → wait → drain phases as the uniform
    skeleton below, split across two warp bands instead of software-pipelined in-warp. The producer
    band rides at the TAIL of the thread block (``threadIdx.x >= block_threads`` — the compute
    warps' grid decode is untouched, and the wrapped aux decode makes ``linear_tid == 0`` elect
    exactly the first producer thread for the TMA fill, so the transport is reused verbatim):

    - **producer** (``aux_threads``): its ELECTED thread (the transport's ``linear_tid == 0`` —
      the wrapped aux decode makes that exactly the band's first thread) primes slots
      ``0..ring-2``, then per chunk parity-waits the consumers' release of the target slot
      (:data:`_EMPTY_MBAR`, skipped on the first lap) and arms + box-copies the prefetch chunk.
      The non-elected producer threads run an empty loop and park.
    - **compute** (``block_threads``): per chunk parity-waits the data mbarrier, drains the slot
      (ldmatrix + mma), then a named ``bar.sync`` over the compute band (a CTA-wide
      ``__syncthreads()`` is UB on the divergent role branch) and ONE elected arrive releases the
      slot — the elected release rides the barrier's happens-before, so the empty mbarrier counts
      1, not ``block_threads`` (one smem atomic per chunk instead of one per thread; the
      pre-rebuild split used the same shape). The arrive helper carries the ``fence.proxy.async``
      that orders the band's GENERIC slab reads before the producer's next ASYNC box-copy into
      the slot.

    ``SetMaxNReg`` redistributes the register file between the branches when the raised total fits.
    """
    tid = Builtin("thread_idx.x")
    decls = transport.slab_decls(ring)
    decls.append(Smem(name=_EMPTY_MBAR, extents=(ring,), dtype="unsigned long long"))
    # Prologue (pre-split, CTA-wide): ONE raw-tid-elected thread inits both mbarrier rings — the
    # transport's wrapped ``linear_tid == 0`` election would match one compute AND one aux thread
    # here — then a CTA barrier publishes the init. The transport's own prologue is not used.
    inits = tuple(MbarrierInit(mbar=transport.mbar, count=1, slot=_lit(s)) for s in range(ring))
    inits += tuple(MbarrierInit(mbar=_EMPTY_MBAR, count=1, slot=_lit(s)) for s in range(ring))
    pre: list[Stmt] = [Cond(cond=BinaryExpr("==", tid, _lit(0)), body=inits), Sync()]

    k0, K = "_ks", k_extent
    i_expr = BinaryExpr("/", Var(k0), _lit(bk_elems))
    kaxis = Axis(name=k0, extent=K)
    setmaxnreg = _CONSUMER_REGS * block_threads + _PRODUCER_REGS * aux_threads <= _SM_REGFILE

    # Producer: prefetch chunk ``c = i + ring - 1`` into slot ``c % ring`` (k0 clamped to the last
    # chunk on the overrun tail, exactly as the uniform ring does); from the second lap on
    # (``c >= ring`` ⟺ ``i >= 1``) wait release generation ``c/ring - 1`` first. The empty-wait
    # rides INSIDE the transport's elected-thread fill ``Cond`` — only the thread that issues the
    # box copy needs the slot's release.
    pref_chunk = BinaryExpr("+", i_expr, _lit(ring - 1))
    pref_slot = BinaryExpr("%", pref_chunk, _lit(ring))
    empty_phase = BinaryExpr("%", BinaryExpr("-", BinaryExpr("/", pref_chunk, _lit(ring)), _lit(1)), _lit(2))
    if ring >= 2:
        k0_next = BinaryExpr("+", Var(k0), _lit((ring - 1) * bk_elems))
        if k_end is not None:  # clamp onto the last chunk this CTA drains (the causal early stop)
            end_var = Var(f"{k0}_end")  # the loop's hoisted for-init bound (StridedLoop.end)
            last_k0 = BinaryExpr("*", BinaryExpr("/", BinaryExpr("-", end_var, _lit(1)), _lit(bk_elems)), _lit(bk_elems))
            k0_pref = TernaryExpr(cond=BinaryExpr("<", k0_next, end_var), if_true=k0_next, if_false=last_k0)
        else:
            k0_pref = TernaryExpr(cond=BinaryExpr("<", k0_next, _lit(K)), if_true=k0_next, if_false=_lit((n_chunks - 1) * bk_elems))
    else:
        k0_pref = Var(k0)
    (fill_cond,) = transport.fill(k0=k0_pref, slot=pref_slot)
    assert isinstance(fill_cond, Cond), "TmaTransport.fill is the elected-thread Cond"
    empty_wait = Cond(cond=BinaryExpr(">=", i_expr, _lit(1)), body=(MbarrierWait(mbar=_EMPTY_MBAR, phase=empty_phase, slot=pref_slot),))
    prod_body: list[Stmt] = [Cond(cond=fill_cond.cond, body=(empty_wait, *fill_cond.body))]
    prod: list[Stmt] = [SetMaxNReg(_PRODUCER_REGS, "dec")] if setmaxnreg else []
    for s in range(ring - 1):  # prime chunks 0..ring-2 into slots 0..ring-2 (release generation 0)
        prod += transport.fill(k0=_lit(s * bk_elems), slot=_lit(s))
    prod.append(StridedLoop(axis=kaxis, start=_lit(0), step=_lit(bk_elems), body=Body(tuple(prod_body)), unroll=False, end=k_end))

    # Compute: wait the data parity, drain the slot, close the band on the named barrier, release.
    read_slot = BinaryExpr("%", i_expr, _lit(ring))
    read_phase = BinaryExpr("%", BinaryExpr("/", i_expr, _lit(ring)), _lit(2))
    cons_body: list[Stmt] = list(transport.wait(in_flight=ring - 1, slot=read_slot, phase=read_phase))
    cons_body += drain(read_slot)
    cons_body.append(Sync(barrier_id=1, count=block_threads))
    cons_body.append(
        Cond(cond=BinaryExpr("==", transport.cta.linear_tid, _lit(0)), body=(MbarrierArrive(mbar=_EMPTY_MBAR, slot=read_slot),))
    )
    cons: list[Stmt] = [SetMaxNReg(_CONSUMER_REGS, "inc")] if setmaxnreg else []
    cons.append(StridedLoop(axis=kaxis, start=_lit(0), step=_lit(bk_elems), body=Body(tuple(cons_body)), unroll=False, end=k_end))

    role = Cond(cond=BinaryExpr(">=", tid, _lit(block_threads)), body=tuple(prod), else_body=tuple(cons))
    return decls, [*pre, role]


def _staged_slabs(transport) -> frozenset[str]:
    """The smem slab names ``transport`` fills — the liveness key a scheduled operand-group's live
    range is derived against (segments tag the slab names they READ; the group's range is the
    ``[first, last]`` interval of segments whose tags intersect these names)."""
    ops = (*getattr(transport, "operands", ()), *getattr(transport, "async_operands", ()))
    return frozenset(op.slab for op in ops)


@dataclass(eq=False)
class _Group:
    """One operand-group being placed by :func:`pipelined_kloop` (internal working state — identity
    semantics, the counting pass indexes groups by object). ``kind`` is the DERIVED fill placement:
    ``"ring"`` (depth ≥ 2 — prefetch chunk ``i+ring-1`` at the top of the body), ``"current"``
    (depth 1, live across the whole body — fill chunk ``i`` at the top, wait immediately) or
    ``"kill"`` (depth 1, live in a proper sub-interval — refill chunk ``i+1`` at the kill point,
    right after the barrier past the last reader). ``lag`` is the iteration distance between a
    chunk's fill and its wait (``ring-1`` / ``0`` / ``1`` respectively) — also the number of
    chunks the prologue primes."""

    transport: object
    ring: int
    read_slot: Expr
    read_phase: Expr
    first: int = 0
    last: int = 0
    kind: str = "current"
    lag: int = 0
    n_flight: int = 0  # cp.async wait_group count (the static counting pass below)


def pipelined_kloop(
    *,
    operands: tuple[tuple[object, int], ...],
    build_segments: Callable[[tuple[Expr, ...]], list[tuple[list[Stmt], frozenset[str]]]],
    bk_elems: int,
    n_chunks: int | Dim,
    k_extent: int | Dim,
    k0: str = "_ks",
    k_end: Expr | None = None,
) -> tuple[list[Stmt], list[Stmt]]:
    """The **one** liveness-scheduled staged K-loop skeleton — every staged form (the matmul tier's
    paired prefetch ring, the warp-flash stream's alternating single-slab pipeline, the degenerate
    single buffer) is this scheduler run over the loop body's dataflow, none is its own skeleton.

    ``operands`` is the ``(transport, depth)`` pair per scheduled operand-group; ``build_segments``
    is called ONCE with each group's read-slot expr (positional) and returns the body as ordered
    ``(stmts, reads)`` segments, ``reads`` naming the smem slabs the segment drains. Two structural
    facts drive every placement decision:

    - **live range**: the ``[first, last]`` segment interval whose ``reads`` intersect the group's
      slab names (:func:`_staged_slabs`) — wait before ``first``, and a CTA barrier after ``last``
      (all readers past the slab) before any refill may overwrite it;
    - **depth**: ``ring = min(depth, n_chunks)`` slots. ``ring >= 2`` prefetches chunk ``i+ring-1``
      at the top of the body (today's paired ring). ``ring == 1`` live across the WHOLE body fills
      chunk ``i`` at the top and waits immediately (no overlap to be had — the single-buffer
      degenerate). ``ring == 1`` live in a PROPER sub-interval refills chunk ``i+1`` at its kill
      point — the refill overlaps every segment outside the live range (the flash alternating
      form: K refills under softmax + P·V, V under the next step's Q·K — not because attention
      alternates, but because that is where their live ranges end). The prologue primes exactly
      ``lag`` chunks — the fills the ``lag`` virtual iterations before step 0 would have issued.

    A loop-INVARIANT staged operand (value delta 0 per iteration — the alternating form's staged Q)
    never enters the schedule: with nothing advancing there is no wait and no refill, so its whole
    derivation is a prologue-only fill the caller emits ahead of the loop.

    Synchronization is derived, not hand-counted: TMA groups parity-wait their own mbarrier
    (``phase = (i / ring) % 2``); cp.async groups get ``wait_group(N)`` with ``N`` = the count of
    younger commits issued between this group's fill and its wait point — a static counting pass
    over the placed schedule (the alternating K,V,K,V queue's uniform ``wait_group(1)`` is this
    count; the ring's ``ring-1`` likewise). Tail refills clamp onto the last needed chunk
    (re-fetched, never waited — keeps the fills unguarded and every barrier CTA-uniform).

    ``k_end`` (CTA-uniform, ``≤ k_extent``) stops the chunk loop early — the causal flash stream's
    triangular kv bound; the clamp re-pins onto the last NEEDED chunk via the loop's hoisted
    ``<k0>_end`` for-init bound. A symbolic ``k_extent`` (a ``Dim``) allocates the full ``depth``
    ring and clamps every fill — ring prefetch and kill-point refill alike — against the runtime
    chunk count. Returns ``(slab_decls, [prologue…, outer_loop])``."""
    symbolic = isinstance(k_extent, Dim)
    i_expr = BinaryExpr("/", Var(k0), _lit(bk_elems))  # chunk index of the current step

    groups: list[_Group] = []
    for transport, depth in operands:
        ring = depth if symbolic else (min(depth, n_chunks) if n_chunks >= 2 else 1)
        if ring == 1:
            read_slot, read_phase = _lit(0), BinaryExpr("%", i_expr, _lit(2))
        else:
            read_slot = BinaryExpr("%", i_expr, _lit(ring))
            read_phase = BinaryExpr("%", BinaryExpr("/", i_expr, _lit(ring)), _lit(2))
        groups.append(_Group(transport=transport, ring=ring, read_slot=read_slot, read_phase=read_phase))

    segments = build_segments(tuple(g.read_slot for g in groups))

    for g in groups:
        slabs = _staged_slabs(g.transport)
        readers = [s for s, (_stmts, reads) in enumerate(segments) if reads & slabs]
        assert readers, f"pipelined_kloop: no segment reads the staged slabs {sorted(slabs)}"
        g.first, g.last = readers[0], readers[-1]
        if g.ring >= 2:
            g.kind, g.lag = "ring", g.ring - 1
        elif g.first == 0 and g.last == len(segments) - 1:
            g.kind, g.lag = "current", 0
        else:
            g.kind, g.lag = "kill", 1  # a symbolic extent rides the same runtime clamp as the ring prefetch

    def _fill_k0(lag: int) -> Expr:
        """The refill's chunk base — ``k0 + lag·bk``, clamped onto the last needed chunk (the
        ``k_end`` early stop reads the loop's hoisted ``<k0>_end`` for-init bound — inlining the
        raw expr per iteration measurably spills; a symbolic extent clamps at runtime)."""
        k0_next = BinaryExpr("+", Var(k0), _lit(lag * bk_elems))
        if k_end is not None:
            end_var = Var(f"{k0}_end")
            last_k0 = BinaryExpr("*", BinaryExpr("/", BinaryExpr("-", end_var, _lit(1)), _lit(bk_elems)), _lit(bk_elems))
            bound: Expr = end_var
        elif symbolic:
            last_k0 = BinaryExpr("*", BinaryExpr("-", n_chunks.expr, _lit(1)), _lit(bk_elems))
            bound = k_extent.expr
        else:
            last_k0 = _lit((n_chunks - 1) * bk_elems)
            bound = _lit(k_extent)
        return TernaryExpr(cond=BinaryExpr("<", k0_next, bound), if_true=k0_next, if_false=last_k0)

    decls: list[Stmt] = []
    pre: list[Stmt] = []
    for g in groups:
        decls += g.transport.slab_decls(g.ring)
    for g in groups:
        pre += g.transport.prologue(g.ring)
    for g in groups:  # prime exactly ``lag`` chunks per group — what iterations -lag..-1 would have filled
        for c in range(g.lag):
            pre += g.transport.fill(k0=_lit(c * bk_elems), slot=_lit(c))
            pre += g.transport.commit()

    # The wait-group counting pass: lay the committing fills out in body order (top fills first,
    # then the kill-point refills by segment), and for each group count the commits issued between
    # its chunk's fill (``lag`` iterations back, position ``f_idx``) and its wait point (before
    # segment ``first`` — after the top fills and any earlier kill fills). Uniform-code assumption:
    # ALL threads issue fills in the same body order, so the per-thread commit queues agree — holds
    # by construction (fills are cooperative loops, never warp-divergent). Non-committing groups
    # (TMA / bare sync) get the value for signature parity only; their waits ignore it.
    committing = [g for g in groups if g.transport.commit()]
    events = [g for g in committing if g.kind in ("ring", "current")]
    events += sorted((g for g in committing if g.kind == "kill"), key=lambda g: g.last)
    m = len(events)
    for g in groups:
        if g not in events:
            g.n_flight = g.ring - 1 if g.kind == "ring" else g.lag
            continue
        f_idx = events.index(g)
        w_cnt = sum(1 for e in events if e.kind in ("ring", "current") or e.last < g.first)
        g.n_flight = (w_cnt - f_idx - 1) if g.lag == 0 else (m - 1 - f_idx) + (g.lag - 1) * m + w_cnt
        assert g.n_flight >= 0, "wait-group counting derived a negative in-flight count"

    body: list[Stmt] = []
    for g in groups:  # top fills: the ring prefetch / the single-buffer current-chunk fill
        if g.kind == "ring":
            fill_slot = BinaryExpr("%", BinaryExpr("+", i_expr, _lit(g.lag)), _lit(g.ring))
            body += g.transport.fill(k0=_fill_k0(g.lag), slot=fill_slot, k0_cur=Var(k0))
            body += g.transport.commit()
        elif g.kind == "current":
            body += g.transport.fill(k0=Var(k0), slot=_lit(0), k0_cur=Var(k0))
            body += g.transport.commit()
    for s, (stmts, _reads) in enumerate(segments):
        for g in groups:
            if g.first == s:
                body += g.transport.wait(in_flight=g.n_flight, slot=g.read_slot, phase=g.read_phase)
        body += stmts
        enders = [g for g in groups if g.last == s]
        if enders:
            body.append(Sync())  # every reader past the slab(s) before a refill / later prefetch overwrites them
            for g in enders:
                if g.kind == "kill":
                    body += g.transport.fill(k0=_fill_k0(1), slot=_lit(0))
                    body += g.transport.commit()

    outer = StridedLoop(
        axis=Axis(name=k0, extent=k_extent), start=_lit(0), step=_lit(bk_elems), body=Body(tuple(body)), unroll=False, end=k_end
    )
    return decls, [*pre, outer]


def staged_kloop(
    *,
    transport,
    drain: Callable[[Expr], list[Stmt]],
    depth: int,
    bk_elems: int,
    n_chunks: int | Dim,
    k_extent: int | Dim,
    workers=None,
    block_threads: int | None = None,
    k0: str = "_ks",
    k_end: Expr | None = None,
) -> tuple[list[Stmt], list[Stmt]]:
    """The whole-body staged K-loop — ONE operand-group live across the entire ``drain``, run through
    :func:`pipelined_kloop` (the segment list is the single ``(drain(slot), slabs)`` entry, so the
    scheduler derives exactly the classic phases): ``ring = min(depth, n_chunks)`` slots (``<2``
    chunks ⇒ nothing to prefetch, ``ring == 1``); ``ring == 1`` fills chunk ``i`` into slot 0 and
    waits everything; ``ring >= 2`` primes chunks ``0..ring-2`` in the prologue, prefetches chunk
    ``i+ring-1`` (clamped to the last chunk so the commit/wait stays uniform across all CTA threads
    — the barrier-under-mask invariant) into slot ``(i+ring-1) % ring``, waits ``ring-1`` chunks in
    flight and ``drain``\\ s slot ``i % ring``. Returns ``(slab_decls, [prologue…, outer_loop])``.

    ``transport`` supplies fill/commit/wait + the slab layout; ``drain(slot)`` is the atom leaf reading
    ring ``slot`` (``ldmatrix`` fragments / scalar slab ``Load``\\ s). For TMA the wait phase toggles per
    slot generation (``chunk // ring``); cp.async ignores it (it gates on the commit group instead).

    ``workers`` (a resolved :class:`~emmy.compiler.ir.schedule.WarpSpec`) splits the same phases
    across producer / compute warp bands instead (:func:`_wspec_kloop`) — TMA transport only (the
    scheduler's legality gate), ``block_threads`` naming the compute band.

    ``k0`` names the chunk loop variable (default ``"_ks"``) — a drain whose body references the
    chunk base by name (the warp-flash stream's absolute score columns) passes its own axis name.

    ``k_end`` (an ``Expr`` over in-scope grid vars, CTA-uniform, ``≤ k_extent``) stops the chunk
    loop early — the causal flash stream's triangular kv bound. Chunks past it fold the carrier
    identity exactly, so skipping them is bit-identical; the prefetch clamp re-pins onto the last
    NEEDED chunk (``((k_end − 1) / bk) · bk``). CTA-uniformity keeps the in-loop barriers legal."""
    # A symbolic ``k_extent`` (warp-flash over a runtime seq_len) is a ``Dim``: the chunk count is a
    # runtime value, so allocate the full ``depth`` ring (the tuned hint has ≥ depth chunks) and let
    # the transport absorb any over-primed tail chunk (TMA zero-fills its box; cp.async clamp-reads
    # the last valid key rows) — the drain masks those keys to the fold identity, so it stays
    # bit-identical to gmem-direct. Static callers pass plain ``int``s.
    if workers is not None:
        symbolic = isinstance(k_extent, Dim)
        assert not symbolic, "warp-spec + symbolic-kv staging is not built (WSPEC drives static-kv TMA only)"
        assert isinstance(transport, TmaTransport), "warp specialization drives the TMA transport only (scheduler legality)"
        assert block_threads is not None, "warp specialization needs the compute-band thread count"
        return _wspec_kloop(
            transport=transport,
            drain=drain,
            ring=min(depth, n_chunks) if n_chunks >= 2 else 1,
            bk_elems=bk_elems,
            n_chunks=n_chunks,
            k_extent=k_extent,
            aux_threads=32 * workers.aux_warps,
            block_threads=block_threads,
            k_end=k_end,
        )
    slabs = _staged_slabs(transport)
    return pipelined_kloop(
        operands=((transport, depth),),
        build_segments=lambda slots: [(drain(slots[0]), slabs)],
        bk_elems=bk_elems,
        n_chunks=n_chunks,
        k_extent=k_extent,
        k0=k0,
        k_end=k_end,
    )
