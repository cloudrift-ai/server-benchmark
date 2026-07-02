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
pair instead of spelling A then B. The staged K-loop itself is ONE
skeleton, :func:`staged_kloop` (``fill → commit → wait → drain → Sync``, ``depth`` the sole
buffering knob), driven by a :class:`Transport` strategy (:class:`CpAsyncTransport` /
:class:`TmaTransport`) — the two producers put behind one ``fill``/``commit``/``wait`` seam. The
slab feeds the same staged ``LdmatrixLoad`` / scalar ``Load`` drain regardless of which producer
(cp.async / TMA) filled it. cp.async/sync slabs are plain row-major (NONE-swizzle, linear writes
and reads); a TMA slab feeding an mma drain is **swizzled** (:func:`pick_swizzle_atom` per operand
— the hardware permutes 16 B chunks during the box copy and each staged ``LdmatrixLoad`` applies
the matching address XOR), which is what keeps the ldmatrix drain free of smem bank conflicts.
``_atom._staged`` — the one atom-agnostic driver — builds the transport, asks the atom strategy
for the drain leaf, and calls :func:`staged_kloop`.

Leading ``_`` so the pass loader (globs ``*.py``, skips ``_``-prefixed) skips it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Builtin, Expr, Literal, TernaryExpr, Var
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
    *, slab: str, shape: tuple[int, int], src: str, gmem_index, cta: CtaTile, elem_bytes: int, name: str, row_offset: Expr | None = None
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
    ``rows``-row block), so the fill targets one slot while the drain reads another."""
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
    """The TMA swizzle atom for a slab whose inner (contiguous) row is ``inner_elems``
    elements of ``elem_bytes``. Returns ``(atom_elems, mode)``: the widest atom in
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
    # TMA smem swizzle mode this operand's slab is written with ("NONE"/"B32"/"B64"/"B128") —
    # derived by the mma tier (`_MmaOps.slab_swizzles`, TMA transport only: the hardware swizzles
    # in-copy and the ldmatrix drain applies the matching XOR; a scalar plain-`Load` drain and the
    # cp.async/sync write paths stay NONE).
    swizzle: str = "NONE"

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

    @property
    def slab(self) -> str:
        return f"_{self.tag}_smem"

    def slot_row(self, slot: Expr) -> Expr | None:
        """The ring-slot row offset into this operand's multi-slot slab (``None`` for slot 0)."""
        return _slot_row(slot, self.shape[0])


@dataclass(frozen=True)
class SyncTransport:
    """The ``sync`` producer — per-thread compute/copy fills closed by ONE CTA barrier. This is the
    mma tier's ``sync`` transport: the fused-edge compute-fill (a producer cone materializing the A
    tile) rides ``operands``; plain-copy operands (the fused edge's B weights) ride
    ``async_operands`` as vectorized ``cp.async`` fills issued BEFORE the compute fill, so the
    hardware copies fly underneath it — the same ``fill``/``commit``/``wait`` seam as the pure
    cp.async / TMA producers, closed by one ``CpAsyncWait`` + CTA barrier. ``depth >= 2`` rings
    every slab (the sync compute fill writes its prefetch slot like any producer; the wait keeps
    ``ring-1`` cp.async groups in flight, so the B copies overlap the drain across chunks).

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
        return [slab_smem(op.slab, ring * op.shape[0], op.shape[1], self.slab_dtype) for op in (*self.operands, *self.async_operands)]

    def prologue(self, ring: int) -> list[Stmt]:  # noqa: ARG002
        return list(self.prologue_stmts)

    def fill(self, *, k0: Expr, slot: Expr) -> list[Stmt]:
        out: list[Stmt] = []
        # Issue the async copies FIRST — they are in flight while the compute fill below runs.
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
            )
        for op in self.operands:
            rows, cols = op.shape
            v = _cp_async_width(cols, self.elem_bytes)
            fe = Axis(name=f"_f{op.tag}", extent=(rows * cols) // v)
            base = _mul(Var(fe.name), _lit(v))
            row = BinaryExpr("/", base, _lit(cols))  # constant across the run: v divides cols
            col = BinaryExpr("%", base, _lit(cols))
            off = op.slot_row(slot)
            smem_row = _add(off, row) if off is not None else row
            body: list[Stmt] = []
            for j in range(v):
                cell_col = _add(col, _lit(j)) if j else col
                stmts, val = op.value(k0, row, cell_col)
                # Suffix only the run-LOCAL defs (each cell replica binds fresh SSA); references to
                # externally-defined names — grid vars, the loop axis, ``k0`` — pass through.
                local = {nm for st in stmts for nm in st.defines()}
                sfx = f"__c{j}"
                ren = lambda nm, local=local, sfx=sfx: f"{nm}{sfx}" if nm in local else nm  # noqa: E731
                body += [st.rewrite(ren) for st in stmts]
                body.append(Write(output=op.slab, index=(smem_row, cell_col), value=f"{val}{sfx}"))
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
        return [slab_smem(op.slab, ring * op.shape[0], op.shape[1], self.slab_dtype) for op in self.operands]

    def prologue(self, ring: int) -> list[Stmt]:
        return []

    def fill(self, *, k0: Expr, slot: Expr) -> list[Stmt]:
        out: list[Stmt] = []
        for op in self.operands:
            out += cp_async_fill(
                slab=op.slab,
                shape=op.shape,
                src=op.buf,
                gmem_index=op.index(k0),
                cta=self.cta,
                elem_bytes=self.elem_bytes,
                name=op.tag,
                row_offset=op.slot_row(slot),
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
        return sum(math.prod(op.shape) for op in self.operands) * self.elem_bytes

    def slab_decls(self, ring: int) -> list[Stmt]:
        # TMA destination smem must be 128 B-aligned (a swizzled slab to its full atom period,
        # so the drain's from-base XOR matches the hardware deposit); one mbarrier per ring slot.
        decls: list[Stmt] = [
            tma_descriptor(op.desc, op.buf, op.shape, self.slab_dtype, swizzle=op.swizzle, elem_bytes=self.elem_bytes)
            for op in self.operands
        ]
        decls += [
            slab_smem(op.slab, ring * op.shape[0], op.shape[1], self.slab_dtype, align=_SWIZZLE_SLAB_ALIGN[op.swizzle])
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
        atom, _ = pick_swizzle_atom(op.shape[-1], self.elem_bytes)
        if atom >= op.shape[-1]:
            return coords
        return (*coords[:-1], BinaryExpr("/", coords[-1], _lit(atom)), _lit(0))

    def prologue(self, ring: int) -> list[Stmt]:
        # Init every ring slot's mbarrier (one producer ``arrive`` per phase), then a CTA barrier so
        # every consumer sees the init before its first wait.
        inits = tuple(MbarrierInit(mbar=self.mbar, count=1, slot=_lit(s)) for s in range(ring))
        return [Cond(cond=self._tid0, body=inits), Sync()]

    def fill(self, *, k0: Expr, slot: Expr) -> list[Stmt]:
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
    prod.append(StridedLoop(axis=kaxis, start=_lit(0), step=_lit(bk_elems), body=Body(tuple(prod_body)), unroll=False))

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
    cons.append(StridedLoop(axis=kaxis, start=_lit(0), step=_lit(bk_elems), body=Body(tuple(cons_body)), unroll=False))

    role = Cond(cond=BinaryExpr(">=", tid, _lit(block_threads)), body=tuple(prod), else_body=tuple(cons))
    return decls, [*pre, role]


def staged_kloop(
    *,
    transport,
    drain: Callable[[Expr], list[Stmt]],
    depth: int,
    bk_elems: int,
    n_chunks: int,
    k_extent: int,
    workers=None,
    block_threads: int | None = None,
) -> tuple[list[Stmt], list[Stmt]]:
    """The **one** staged K-loop skeleton — ``fill → commit → wait → drain → Sync`` over the K-chunks,
    with ``depth`` the sole buffering knob and ``transport`` the sole producer seam. Returns
    ``(slab_decls, [prologue…, outer_loop])``.

    ``ring = min(depth, n_chunks)`` slots (``<2`` chunks ⇒ nothing to prefetch, ``ring == 1``):

    - ``ring == 1`` (single buffer): fill chunk ``i`` into slot 0, wait everything, ``drain`` slot 0.
    - ``ring >= 2`` (gmem→smem prefetch ring): a prologue primes chunks ``0..ring-2`` into slots
      ``0..ring-2``; each loop step prefetches chunk ``i+ring-1`` (clamped to the last chunk so the
      commit/wait stays uniform across all CTA threads — the barrier-under-mask invariant) into slot
      ``(i+ring-1) % ring``, then waits ``ring-1`` chunks in flight and ``drain``\\ s slot ``i % ring``.

    ``transport`` supplies fill/commit/wait + the slab layout; ``drain(slot)`` is the atom leaf reading
    ring ``slot`` (``ldmatrix`` fragments / scalar slab ``Load``\\ s). For TMA the wait phase toggles per
    slot generation (``chunk // ring``); cp.async ignores it (it gates on the commit group instead).

    ``workers`` (a resolved :class:`~emmy.compiler.ir.schedule.WarpSpec`) splits the same phases
    across producer / compute warp bands instead (:func:`_wspec_kloop`) — TMA transport only (the
    scheduler's legality gate), ``block_threads`` naming the compute band."""
    ring = min(depth, n_chunks) if n_chunks >= 2 else 1
    if workers is not None:
        assert isinstance(transport, TmaTransport), "warp specialization drives the TMA transport only (scheduler legality)"
        assert block_threads is not None, "warp specialization needs the compute-band thread count"
        return _wspec_kloop(
            transport=transport,
            drain=drain,
            ring=ring,
            bk_elems=bk_elems,
            n_chunks=n_chunks,
            k_extent=k_extent,
            aux_threads=32 * workers.aux_warps,
            block_threads=block_threads,
        )
    k0, K = "_ks", k_extent
    decls = transport.slab_decls(ring)
    pre = transport.prologue(ring)
    for s in range(ring - 1):  # prime chunks 0..ring-2 into slots 0..ring-2 (phase 0)
        pre += transport.fill(k0=_lit(s * bk_elems), slot=_lit(s))
        pre += transport.commit()

    i_expr = BinaryExpr("/", Var(k0), _lit(bk_elems))  # chunk index of the current step
    body: list[Stmt] = []
    if ring == 1:
        phase = BinaryExpr("%", i_expr, _lit(2))
        body += transport.fill(k0=Var(k0), slot=_lit(0))
        body += transport.commit()
        body += transport.wait(in_flight=0, slot=_lit(0), phase=phase)
        body += drain(_lit(0))
        body.append(Sync())
    else:
        pref_chunk = BinaryExpr("+", i_expr, _lit(ring - 1))  # logical index of the prefetched chunk
        pref_slot = BinaryExpr("%", pref_chunk, _lit(ring))
        read_slot = BinaryExpr("%", i_expr, _lit(ring))
        read_phase = BinaryExpr("%", BinaryExpr("/", i_expr, _lit(ring)), _lit(2))
        last_k0 = (n_chunks - 1) * bk_elems
        k0_next = BinaryExpr("+", Var(k0), _lit((ring - 1) * bk_elems))
        k0_pref = TernaryExpr(cond=BinaryExpr("<", k0_next, _lit(K)), if_true=k0_next, if_false=_lit(last_k0))
        body += transport.fill(k0=k0_pref, slot=pref_slot)
        body += transport.commit()
        body += transport.wait(in_flight=ring - 1, slot=read_slot, phase=read_phase)
        body += drain(read_slot)
        body.append(Sync())  # done reading this slot before a later chunk prefetches into it

    outer = StridedLoop(axis=Axis(name=k0, extent=K), start=_lit(0), step=_lit(bk_elems), body=Body(tuple(body)), unroll=False)
    return decls, [*pre, outer]
