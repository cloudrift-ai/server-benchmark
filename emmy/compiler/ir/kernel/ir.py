"""Kernel IR — the fully-scheduled kernel form, just above CUDA source.

Kernel IR sits between Tile IR (schedule decisions as structural Stmts)
and CUDA source (text). Its body contains the explicit hardware
machinery: ``Tile`` (thread/block coord bindings), ``Smem``
(``__shared__`` arrays), ``Sync`` (``__syncthreads`` barriers),
``TreeHalve`` (cross-thread reduction over smem), ``StridedLoop``
(strided per-thread loop).

Pipeline shape::

    Tile IR ──materialize_tile──▶ Kernel IR
                    ──render_kernelop──▶ CUDA source

**Leaf compute reuses Loop IR directly**. ``Load`` / ``Assign`` /
``Select`` / ``Write`` / ``Accum`` / ``Cond`` / ``Loop`` come straight
from ``ir.loop`` — buf names are strings so they're directly renderable.

Kernel IR deliberately contains no scheduling decisions — those live in
Tile IR and are materialized away before reaching this layer. A
``KernelOp`` is what the CUDA backend turns into a ``RawKernel`` launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from emmy.compiler.dtype import F32, DataType
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import _REDUCE_SPELLING, ElementwiseImpl
from emmy.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    Expr,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)
from emmy.compiler.ir.stmt import (
    INDENT,
    Accum,
    Assign,
    Body,
    Cond,
    Load,
    Loop,
    RenderCtx,
    Select,
    SelectBranch,
    Stmt,
    StridedLoop,
    Write,
    _pad,
    pretty_body,
    render_body,
)
from emmy.compiler.ir.stmt.base import render_merge_program
from emmy.compiler.ir.stmt.ir import BodyOp

# The widest iteration space a 32-bit flat thread id can address — past this the
# ``Tile`` decode must widen to 64-bit (see ``Tile.render``).
_INT32_MAX = 2**31 - 1

# ---------------------------------------------------------------------------
# Hardware primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Smem(Stmt):
    """Declare a per-block ``__shared__`` array.

    Renders to ``__shared__ [__align__(N)] <dtype> <name>[<prod(extents)>];``.
    ``extents`` is the multi-dim shape used to flatten ``Load`` / ``Write``
    indices against this buffer. smem_bytes for ``CudaOp`` is computed by
    walking the KernelOp body and summing ``prod(extents) * sizeof(dtype)``
    across distinct ``Smem`` declarations.

    ``align`` is an optional explicit byte alignment. TMA destinations
    (``cp.async.bulk.tensor``) require 16-byte aligned smem; the
    materializer sets ``align=16`` on TMA-target slabs. Default 0 means
    "no explicit alignment" — falls back to the dtype's natural alignment.
    """

    name: str
    extents: tuple[int, ...]
    dtype: str = "float"
    align: int = 0

    def local_decls(self) -> tuple[str, ...]:
        return (self.name,)

    def pretty(self, indent: str = "") -> list[str]:
        ext = ", ".join(str(e) for e in self.extents) or "-"
        ali = f" align={self.align}" if self.align else ""
        return [f"{indent}Smem {self.dtype} {self.name}[{ext}]{ali}"]

    def render(self, ctx: RenderCtx) -> list[str]:
        """``__shared__ <dtype> <name>[<prod(extents)>];`` and register the
        buffer's shape so subsequent ``Load``/``Write`` flatten correctly.

        When ``ctx.smem_dynamic_offsets`` carries this buffer's offset,
        emit a pointer alias into the shared ``_smem_pool`` (extern
        dynamic smem) instead of a static ``__shared__`` array — the
        renderer falls back to the dynamic path when total per-CTA smem
        would exceed the 48 KB static cap.
        """
        from emmy.compiler.backend.cuda.dtype import canonical_from_cuda_name  # noqa: PLC0415

        total = 1
        for e in self.extents:
            total *= int(e)
        ctx.shapes[self.name] = tuple(int(e) for e in self.extents)
        # Register the smem buffer's canonical dtype so Load/Write
        # against this name pick the right local C type. Mbarrier slabs
        # (``"unsigned long long"``) map to None and stay out of the
        # dtype-aware path.
        smem_canonical = canonical_from_cuda_name(self.dtype)
        if smem_canonical is not None:
            ctx.buffer_dtypes[self.name] = smem_canonical
        if self.name in ctx.smem_dynamic_offsets:
            offset = ctx.smem_dynamic_offsets[self.name]
            return [f"{_pad(ctx.indent)}{self.dtype}* {self.name} = reinterpret_cast<{self.dtype}*>(_smem_pool + {offset});"]
        ali = f"__align__({self.align}) " if self.align else ""
        return [f"{_pad(ctx.indent)}__shared__ {ali}{self.dtype} {self.name}[{total}];"]


@dataclass(frozen=True)
class Sync(Stmt):
    """Thread-group barrier.

    ``barrier_id == 0`` (default): ``__syncthreads();`` — CTA-wide.

    ``warp=True``: ``__syncwarp();`` — warp-scope only. For ordering a warp's own
    smem writes before its own reads of a WARP-PRIVATE buffer (the flash C→A
    handoff slab): post-Volta independent thread scheduling means even
    intra-warp write→read needs the sync, but a CTA-wide barrier there convoys
    the block's warps for no ordering benefit.

    ``barrier_id > 0`` and ``count`` set: ``bar.sync <id>, <count>;`` —
    named-barrier synchronizing exactly ``count`` threads on barrier id
    ``<id>`` (one of 1..15). Used inside warp-specialized branches where
    only a subset of CTA threads execute the sync — `__syncthreads()`
    would be CUDA UB because the enclosing ``if (warp < P)`` condition
    diverges across warps. The count must equal the number of threads
    that actually arrive at this barrier in flight."""

    barrier_id: int = 0
    count: int | None = None
    warp: bool = False  # warp-scope __syncwarp() (barrier_id must stay 0)

    def pretty(self, indent: str = "") -> list[str]:
        if self.warp:
            return [f"{indent}Sync(warp)"]
        if self.barrier_id == 0:
            return [f"{indent}Sync"]
        return [f"{indent}Sync(bar={self.barrier_id}, count={self.count})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        if self.warp:
            if self.barrier_id != 0:
                raise ValueError("Sync(warp=True) cannot carry a named barrier_id")
            return [f"{pad}__syncwarp();"]
        if self.barrier_id == 0:
            return [f"{pad}__syncthreads();"]
        if self.count is None:
            raise ValueError(f"Sync(barrier_id={self.barrier_id}) requires count")
        return [f'{pad}asm volatile("bar.sync {self.barrier_id}, {self.count};\\n" ::: "memory");']


@dataclass(frozen=True)
class Tile(Stmt):
    """Tile coordinate binding — one GPU thread per output element.

    The kernel-IR realization of a :class:`~emmy.compiler.ir.tile.TileOp`'s
    thread schedule (geometry only — the combine, if any, lives in the wrapped
    ``body``): it maps the kernel's iteration space (the product of its ``axes``
    extents — the tile's ``grid_axes``) onto a 1-D linear thread grid. Each
    thread derives a global linear id
    ``_gid = blockIdx.x * blockDim.x + threadIdx.x``, guards ``_gid < N``
    (``N`` = ∏ extents), decodes its per-axis indices from ``_gid``, and
    runs the ``body`` once. Index motion (broadcast, transpose, gather) is
    already encoded in the body's ``Load`` index Exprs, so the decode only
    has to bind the axis induction variables.

    Renders to::

        int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (_gid < N) {
            int a0 = _gid / s0;          // outermost (s0 = ∏ inner extents)
            int a1 = _gid / s1 % e1;
            ...
            <body>
        }

    The cuda lowering reads ``N`` to size the launch grid
    (``ceil(N / blockDim)`` CTAs of ``blockDim`` threads). Static extents only
    for now — a symbolic axis raises (the runtime-arg / masked-tile forms land
    as the skeleton grows).

    ``block_threads`` is the per-CTA thread count for a **cooperative** tile —
    the materializer sets it to ``coop · ∏block-cells`` so the cuda lowering
    derives ``blockDim = block_threads`` and ``gridDim = N / block_threads``
    (the linear ``_gid`` decode then groups ``block_threads`` consecutive cells
    per CTA, the innermost being the cooperative lanes). ``None`` is the scalar
    tier (one thread per cell, ``blockDim = _BLOCK_SIZE``).

    ``aux_threads`` is the warp-specialized producer band appended PAST the
    cooperative cells (``blockDim = block_threads + aux_threads``, the grid
    still ``N / block_threads``). The decode becomes ``_gid = blockIdx.x ·
    block_threads + threadIdx.x % block_threads``: a compute thread decodes its
    own cell exactly as before, while an aux thread wraps onto the CTA's first
    cells — its BLOCK-axis vars (functions of ``blockIdx`` alone) are correct
    (the producer fill needs the CTA tile origin), its unit/lane vars alias a
    compute cell and are never read (the producer branch guards on the raw
    ``threadIdx.x``, and the store side is guarded to compute threads).
    """

    axes: tuple[Axis, ...]
    body: Body
    block_threads: int | None = None
    aux_threads: int = 0
    # The ``(m, n)`` BLOCK axis names of a 2-D-tiled contraction output — the structural
    # eligibility for CTA rasterization, stamped by ``grid_tile`` (only there is the m/n
    # identity known). ``None`` = not a 2-D contraction grid (reduce tier, 1-D output).
    raster_axes: tuple[str, str] | None = None
    # The resolved ``RASTER`` codec (grid_tile, off the TileOp's stamped knobs): the flat CTA
    # order iterates ``raster_group`` block-tiles of the ``raster_orient`` axis fastest within
    # each launch stripe, so consecutive CTAs share the same streamed operand slab and its
    # repeat reads hit L2 instead of DRAM (``m`` groups M-tiles — B streamed; ``n`` the
    # transpose — A streamed). ``None`` = the plain N-fastest row-major order.
    raster_group: int | None = None
    raster_orient: str = "m"

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body(self.body))
        if self.aux_threads and self.block_threads is None:
            raise ValueError("Tile.aux_threads requires a cooperative block_threads")

    def nested(self) -> tuple[Body, ...]:
        return (self.body,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return Tile(
            axes=self.axes,
            body=body,
            block_threads=self.block_threads,
            aux_threads=self.aux_threads,
            raster_axes=self.raster_axes,
            raster_group=self.raster_group,
            raster_orient=self.raster_orient,
        )

    def binds_axes(self) -> frozenset[str]:
        return frozenset(a.name for a in self.axes)

    @property
    def extents(self) -> tuple[int, ...]:
        """Static extent of each axis (raises on a symbolic extent)."""
        out: list[int] = []
        for a in self.axes:
            if not a.extent.is_static:
                raise NotImplementedError(f"Tile: symbolic axis {a.name!r} not supported yet")
            out.append(a.extent.as_static())
        return tuple(out)

    @property
    def n_elements(self) -> int:
        """Total iteration-space size — one thread per element. Static axes only (raises on a
        symbolic axis; the dynamic-grid path uses :attr:`n_dim` instead)."""
        from math import prod  # noqa: PLC0415

        return prod(self.extents) if self.axes else 1

    @property
    def cells(self) -> int:
        """The per-CTA cooperative cell count — ``block_threads`` when set, else the scalar tier's
        ``_BLOCK_SIZE``. This is ``blockDim`` minus any aux band, i.e. the divisor the cuda lowering
        uses for ``gridDim = ceil(n_elements / cells)``. So ``n_elements % cells == 0`` (and no aux
        band) means the static grid covers the iteration space exactly — no partial last block, so
        the ``_gid < N`` tail guard is provably always-true and :meth:`render` elides it."""
        if self.block_threads is not None:
            return self.block_threads
        from emmy.compiler.ir.kernel.render import _BLOCK_SIZE  # noqa: PLC0415

        return _BLOCK_SIZE

    @property
    def is_static_grid(self) -> bool:
        """True iff every grid axis has a static extent (the static launch path)."""
        return all(a.extent.is_static for a in self.axes)

    @property
    def n_dim(self):
        """Total iteration-space size as a :class:`Dim` — the ∏ of the axis extents, holding a
        symbolic factor (a dynamic ``seq_len``) when any axis is symbolic. The cuda lowering
        sizes the launch from this (``ceil(n_dim / blockDim)`` CTAs, ``seq_len`` resolved at
        launch); the renderer guards ``_gid < n_dim`` so the partial last block is masked."""
        from emmy.compiler.dim import Dim  # noqa: PLC0415

        n = Dim(1)
        for a in self.axes:
            n = n * a.extent
        return n

    def pretty(self, indent: str = "") -> list[str]:
        names = ", ".join(a.name for a in self.axes)
        # A symbolic grid (dynamic seq_len) prints its Dim product — ``n_elements`` would raise,
        # and ``pretty`` feeds ``structural_key`` / the op-inventory persist, so it must stay total.
        n = self.n_elements if self.is_static_grid else self.n_dim
        return [f"{indent}Tile[{names}] (N={n})", *pretty_body(self.body, indent + INDENT)]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        # Symbolic grid (a dynamic free axis): size the guard + decode from the runtime extents
        # via the symbolic-aware helpers (the ``Dim`` name renders to its ``int`` arg). Kept off
        # the static path so static codegen stays byte-identical.
        if not self.is_static_grid:
            from emmy.compiler.ir.stmt.blocks import _render_grid_axis_decode, _stride_c  # noqa: PLC0415

            inner = ctx.child()
            # A symbolic grid's total is unboundable at render time, so the flat id and every
            # stride divisor are ALWAYS 64-bit — a batched coop-reduce grid overflows int32 at
            # runtime-legal extents (batch·seq·N·coop = 4·512·6144·256 ≈ 3.2e9 on the demoted
            # fused gate/up kernel → cudaErrorIllegalAddress; the static branch below reaches
            # the same overflow class and widens exactly when its known total needs it). The
            # ``(long long)blockIdx.x`` lowers to one mul.wide; the decoded axis vars stay
            # ``int`` (each bounded by its extent).
            n = _stride_c(list(self.axes), inner, wide=True)
            # Same aux-band decode as the static path below: blockDim includes the producer band,
            # so the cell id must be derived from ``block_threads``, not ``blockDim``.
            sgid = (
                f"(long long)blockIdx.x * {self.block_threads} + threadIdx.x % {self.block_threads}"
                if self.aux_threads
                else "(long long)blockIdx.x * blockDim.x + threadIdx.x"
            )
            out = [
                f"{pad}long long _gid = {sgid};",
                f"{pad}if (_gid < {n}) {{",
            ]
            out.extend(_render_grid_axis_decode(self.axes, "_gid", inner, wide=True))
            out.extend(render_body(self.body, inner))
            out.append(f"{pad}}}")
            return out
        extents = self.extents
        # Inner-stride of each axis = product of the extents to its right.
        strides: list[int] = [1] * len(extents)
        acc = 1
        for i in range(len(extents) - 1, -1, -1):
            strides[i] = acc
            acc *= extents[i]
        n = self.n_elements
        # A flat id past INT32_MAX wraps negative in 32-bit arithmetic and every decoded index
        # goes wild (a per-cell coop-reduce grid reaches it: M·N·coop = 4096·3840·256 ≈ 4.0e9 on
        # the gemma-4 down-proj demote → cudaErrorIllegalAddress). Widen the id to 64-bit exactly
        # when the static iteration space needs it — ``(long long)blockIdx.x`` lowers to one
        # mul.wide, and the (slower) 64-bit axis divisions only land on kernels that would
        # otherwise fault. The symbolic branch above widens ALWAYS: its runtime extent is
        # unboundable at render time (batch·seq·N·coop crossed int32 at seq 512 on the merged
        # gate/up demote).
        wide = n > _INT32_MAX
        bidx = "(long long)blockIdx.x" if wide else "blockIdx.x"
        # A warp-specialized producer band (``aux_threads``) launches PAST the cooperative cells:
        # decode off ``blockIdx·block_threads + threadIdx % block_threads`` so an aux thread wraps
        # onto this CTA's cells (correct block coords; its unit/lane aliases are never read).
        gid = (
            f"{bidx} * {self.block_threads} + threadIdx.x % {self.block_threads}"
            if self.aux_threads
            else f"{bidx} * blockDim.x + threadIdx.x"
        )
        # ``gridDim = ceil(n / cells)`` CTAs cover the cells. When ``cells`` divides ``n`` exactly and
        # there is no warp-specialized aux band, every launched thread maps to a real cell — the tail
        # guard ``_gid < n`` is always-true (and nvcc can't prove it: ``gridDim`` is a launch arg), so
        # emit the decode + body flat. Otherwise the partial last block (or the aux threads that launch
        # past the cells) needs masking.
        guard = self.aux_threads != 0 or n % self.cells != 0
        out = [f"{pad}{'long long' if wide else 'int'} _gid = {gid};"]
        inner = ctx.child()
        ipad = _pad(inner.indent)
        if guard:
            out.append(f"{pad}if (_gid < {n}) {{")
        # Grouped CTA rasterization (the resolved ``RASTER`` codec): remap the flat ``(m, n)``
        # block sub-id so ``G`` block-tiles of the grouped axis iterate fastest within each
        # launch stripe. Consecutive CTAs then share the same streamed operand slab (L2 reuse)
        # instead of re-streaming it from DRAM once per row of the other axis (the flat
        # N-fastest order's failure mode). With the grouped axis ``a`` (extent Ea) and the
        # other axis ``b`` (extent Eb): ``a = grp·G + (sub % (G·Eb)) % gsz``,
        # ``b = (sub % (G·Eb)) / gsz`` — the standard threadblock swizzle, with
        # ``gsz = min(G, Ea − grp·G)`` handling the ragged last group.
        raster = None  # (grouped_idx, other_idx, G) when active
        if self.raster_group and self.raster_axes:
            names = [a.name for a in self.axes]
            if self.raster_axes[0] in names and self.raster_axes[1] in names:
                mi, ni = names.index(self.raster_axes[0]), names.index(self.raster_axes[1])
                if ni == mi + 1:
                    gi, oi = (mi, ni) if self.raster_orient == "m" else (ni, mi)
                    g_eff = min(self.raster_group, extents[gi])
                    if extents[gi] > 1 and g_eff > 1:
                        raster = (gi, oi, g_eff)
        if raster is not None:
            gi, oi, g = raster
            ea, eb = extents[gi], extents[oi]
            mi, ni = min(gi, oi), max(gi, oi)
            sn = strides[ni]
            sub = "_gid" if sn == 1 else f"_gid / {sn}"
            # ``% (Em·En)`` drops when (m, n) are the outermost axes (the sub-id is bounded
            # by the ``_gid < N`` guard / exact grid).
            if mi != 0:
                sub = f"({sub}) % {ea * eb}"
            decls = [f"{ipad}int _rsub = {sub};"]
            if ea % g == 0:
                a_expr = f"(_rsub / {g * eb}) * {g} + _rsub % {g}"
                b_expr = f"(_rsub % {g * eb}) / {g}"
            else:
                decls.append(f"{ipad}int _rgrp = _rsub / {g * eb};")
                decls.append(f"{ipad}int _rsz = min({g}, {ea} - _rgrp * {g});")
                a_expr = f"_rgrp * {g} + (_rsub % {g * eb}) % _rsz"
                b_expr = f"(_rsub % {g * eb}) / _rsz"
        for i, a in enumerate(self.axes):
            s, e = strides[i], extents[i]
            if raster is not None and i == mi:
                out.extend(decls)
                out.append(f"{ipad}int {a.name} = {a_expr if i == gi else b_expr};")
                continue
            if raster is not None and i == ni:
                out.append(f"{ipad}int {a.name} = {a_expr if i == gi else b_expr};")
                continue
            term = "_gid" if s == 1 else f"_gid / {s}"
            # The outermost axis needs no modulo (``_gid / s0 < e0`` since _gid < N).
            expr = term if i == 0 else f"({term}) % {e}"
            out.append(f"{ipad}int {a.name} = {expr};")
        out.extend(render_body(self.body, inner))
        if guard:
            out.append(f"{pad}}}")
        return out


@dataclass(frozen=True)
class CpAsyncCopy(Stmt):
    """Issue one ``cp.async.{ca,cg}.shared.global`` instruction.

    Replaces the per-thread ``Load(reg) + Write(smem)`` pair in cooperative
    loads on sm_80+. The hardware copies ``nbytes`` (one contiguous vector of
    ``nbytes / sizeof(dtype)`` elements) directly from global to shared without
    a register staging slot, freeing the thread register and removing the
    LDG → STS dependency.

    ``nbytes`` ∈ {4, 8, 16} (the cp.async copy sizes). 16-byte copies use the
    ``.cg`` (cache-global, bypass-L1) qualifier — the streaming form CUTLASS /
    cuBLAS use, requiring 16-byte-aligned smem + global addresses — and 4/8-byte
    copies use ``.ca``. ``smem_index`` / ``src_index`` address the *start* of
    the contiguous chunk; ``_stage_expand`` picks the widest legal ``nbytes`` so
    e.g. an fp16 tile stages 8 halves per ``cp.async``.

    Renders to inline PTX. The asm reads the smem address via
    ``cvta.to.shared.u32`` and the global pointer as a 64-bit value;
    indices flatten via ``render_index`` against the buffer's declared
    shape (same as ``Load`` / ``Write``)."""

    smem: str  # destination smem buffer name
    smem_index: tuple
    src: str  # source global buffer name
    src_index: tuple
    nbytes: int = 4  # bytes per cp.async (4/8/16); 16 ⇒ .cg, else .ca
    # Software smem swizzle mode ("NONE"/"B32"/"B64"/"B128"): XOR the flattened destination
    # element index via ``emmy_swizzle_<mode>`` — the same address-based permutation the
    # ``LdmatrixLoad`` drain applies, so a swizzled slab round-trips (``LDMATRIX_SWIZZLE_XOR``).
    swizzle: str = "NONE"

    def external_reads(self) -> tuple[str, ...]:
        return (self.src,)

    def pretty(self, indent: str = "") -> list[str]:
        smem_idx = ", ".join(e.pretty() for e in self.smem_index)
        src_idx = ", ".join(e.pretty() for e in self.src_index)
        swz = f" swz={self.swizzle}" if self.swizzle in LDMATRIX_SWIZZLE_XOR else ""
        return [f"{indent}cp.async[{self.nbytes}B] {self.smem}[{smem_idx}]{swz} <- {self.src}[{src_idx}]"]

    def render(self, ctx: RenderCtx) -> list[str]:
        from emmy.compiler.ir.stmt import render_index

        smem_flat = render_index(self.smem, self.smem_index, ctx)
        src_flat = render_index(self.src, self.src_index, ctx)
        if self.swizzle in LDMATRIX_SWIZZLE_XOR:
            smem_flat = f"emmy_swizzle_{self.swizzle.lower()}({smem_flat})"
        pad = _pad(ctx.indent)
        # ``emmy_cp_async_{cg,ca}`` (the cp.async prelude) does the ``cvta`` internally, so this is a
        # single call — no ``_smem_addr`` local and no wrapping ``{ }`` block. .cg is 16-byte-only.
        dst, src = f"&{self.smem}[{smem_flat}]", f"&{self.src}[{src_flat}]"
        call = f"emmy_cp_async_cg({dst}, {src})" if self.nbytes == 16 else f"emmy_cp_async_ca<{self.nbytes}>({dst}, {src})"
        return [f"{pad}{call};"]


@dataclass(frozen=True)
class CpAsyncCommit(Stmt):
    """``cp.async.commit_group;`` — finalize the preceding cp.async copies
    issued by this thread into a commit group. Pairs with
    ``CpAsyncWait`` to wait for that group to drain."""

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}cp.async.commit_group"]

    def render(self, ctx: RenderCtx) -> list[str]:
        return [f"{_pad(ctx.indent)}emmy_cp_async_commit();"]


@dataclass(frozen=True)
class CpAsyncWait(Stmt):
    """``cp.async.wait_group N;`` — block this thread until ≤ N cp.async
    groups remain in flight. ``group=0`` waits for everything (synchronous
    style); larger values stagger waits for software pipelining."""

    group: int = 0

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}cp.async.wait_group({self.group})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        return [f"{_pad(ctx.indent)}emmy_cp_async_wait<{self.group}>();"]


@dataclass(frozen=True)
class TmaDescriptor(Stmt):
    """Host-built ``CUtensorMap`` descriptor for a TMA box copy.

    Declarative — renders nothing inside the kernel. The CUDA backend
    walks the ``KernelOp`` body, calls ``cuTensorMapEncodeTiled`` once
    per distinct ``TmaDescriptor`` at compile time, and binds the
    resulting 128-byte descriptor as a ``__grid_constant__`` kernel
    arg. ``name`` matches the parameter name referenced by sibling
    ``TmaLoad`` stmts. ``src_buf`` names the global buffer the descriptor
    addresses (used by the kernel-arg pipeline to add ``src_buf`` to the
    parameter list).
    """

    name: str
    src_buf: str
    src_shape: tuple[int, ...]
    box_extents: tuple[int, ...]
    swizzle: str = "NONE"
    dtype: str = "float"

    def external_reads(self) -> tuple[str, ...]:
        return (self.src_buf,)

    def pretty(self, indent: str = "") -> list[str]:
        shape = ", ".join(str(e) for e in self.src_shape)
        box = ", ".join(str(e) for e in self.box_extents)
        return [f"{indent}TmaDescriptor {self.name} <- {self.src_buf}[{shape}] box=[{box}] swizzle={self.swizzle}"]

    def render(self, ctx: RenderCtx) -> list[str]:
        # Descriptor is a kernel parameter built host-side; no in-kernel render.
        return []


@dataclass(frozen=True)
class TmaLoad(Stmt):
    """``cp.async.bulk.tensor.<rank>d`` — single-thread box copy.

    Issued by exactly one thread of the CTA (the materializer wraps this
    in ``Cond(tid==0, [...])``). Source coords come from the staging
    Stmt's ``origin`` (CTA-uniform); destination is the dynamic-phase
    smem slab. Pairs with ``MbarrierArriveExpectTx`` (issued immediately
    before) and ``MbarrierWait`` (issued by every consumer thread)."""

    smem: str
    smem_index: tuple
    desc: str
    coords: tuple
    mbar: str
    mbar_slot: Expr | None = None

    def pretty(self, indent: str = "") -> list[str]:
        smem_idx = ", ".join(e.pretty() for e in self.smem_index)
        coords = ", ".join(e.pretty() for e in self.coords)
        s = "" if self.mbar_slot is None else f"[{self.mbar_slot.pretty()}]"
        return [f"{indent}TmaLoad {self.smem}[{smem_idx}] <- {self.desc}({coords}) mbar={self.mbar}{s}"]

    def render(self, ctx: RenderCtx) -> list[str]:
        from emmy.compiler.ir.stmt import render_index

        smem_flat = render_index(self.smem, self.smem_index, ctx)
        rank = len(self.coords)
        # PTX expects coords in driver order (innermost-first), which is
        # the reverse of our C-order ``coords`` (== ``stage.origin``).
        coord_args = ", ".join(c.render(ctx) for c in reversed(self.coords))
        mbar_addr = "&" + self.mbar if self.mbar_slot is None else f"&{self.mbar}[{self.mbar_slot.render(ctx)}]"
        pad = _pad(ctx.indent)
        # The ``cp_async_bulk_tensor_<rank>d`` helpers are defined in the
        # kernel prelude — single-CTA ``.shared::cta`` qualifier; cluster
        # launches would need a separate helper variant.
        return [
            f"{pad}cp_async_bulk_tensor_{rank}d(&{self.smem}[{smem_flat}], {self.desc}, {coord_args}, {mbar_addr});",
        ]


@dataclass(frozen=True)
class MbarrierInit(Stmt):
    """``mbarrier.init.shared.b64 [&mbar[slot]], count;`` — one-shot init.

    With ``slot=None``: initializes a scalar mbarrier (single-slot path).
    With ``slot=Expr``: initializes one element of a per-slot mbarrier
    array. The materializer emits one ``MbarrierInit`` per literal slot
    index in the buffer count, hoisted to the kernel prologue inside
    ``Cond(threadIdx.x == 0, [...]) + Sync()``."""

    mbar: str
    count: int = 1
    slot: Expr | None = None

    def pretty(self, indent: str = "") -> list[str]:
        s = "" if self.slot is None else f"[{self.slot.pretty()}]"
        return [f"{indent}MbarrierInit({self.mbar}{s}, count={self.count})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        addr = "&" + self.mbar if self.slot is None else f"&{self.mbar}[{self.slot.render(ctx)}]"
        return [f"{pad}mbarrier_init({addr}, {self.count});"]


@dataclass(frozen=True)
class MbarrierArriveExpectTx(Stmt):
    """``mbarrier.arrive.expect_tx.shared.b64`` — declare the expected
    transaction byte count for the in-flight TMA copy on this barrier.
    With per-slot mbarrier arrays, ``slot`` selects the ring buffer slot."""

    mbar: str
    bytes_: int
    slot: Expr | None = None

    def pretty(self, indent: str = "") -> list[str]:
        s = "" if self.slot is None else f"[{self.slot.pretty()}]"
        return [f"{indent}MbarrierArriveExpectTx({self.mbar}{s}, bytes={self.bytes_})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        addr = "&" + self.mbar if self.slot is None else f"&{self.mbar}[{self.slot.render(ctx)}]"
        return [f"{pad}mbarrier_arrive_expect_tx({addr}, {self.bytes_});"]


@dataclass(frozen=True)
class MbarrierArrive(Stmt):
    """``mbarrier.arrive.shared.b64`` — simple arrive (no transaction-byte
    count). Used by warp-specialized consumer warps to signal "slot empty"
    so the producer can refill it on the next ring iteration. The producer
    side uses ``MbarrierArriveExpectTx`` instead (which declares the
    incoming TMA byte count)."""

    mbar: str
    slot: Expr | None = None

    def pretty(self, indent: str = "") -> list[str]:
        s = "" if self.slot is None else f"[{self.slot.pretty()}]"
        return [f"{indent}MbarrierArrive({self.mbar}{s})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        addr = "&" + self.mbar if self.slot is None else f"&{self.mbar}[{self.slot.render(ctx)}]"
        return [f"{pad}mbarrier_arrive({addr});"]


@dataclass(frozen=True)
class SetMaxNReg(Stmt):
    """``setmaxnreg.{inc,dec}.sync.aligned.u32 N;`` — Hopper+ register-budget
    redistribution. ``direction="dec"`` shrinks the calling warp's max
    register count to ``count`` (returning registers to the pool);
    ``direction="inc"`` claims up to ``count`` registers. Used by warp-
    specialized kernels so producer warps drop registers and consumer
    warps claim them, decoupling occupancy from the consumer's pressure.

    Requires sm_90+. On older targets NVCC rejects the instruction at
    compile time — only a warp-specialized materialization emits this Stmt,
    and that path is already TMA-gated to sm_90+ anyway."""

    count: int
    direction: str  # "inc" or "dec"

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}SetMaxNReg.{self.direction}({self.count})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        return [f'{pad}asm volatile("setmaxnreg.{self.direction}.sync.aligned.u32 {self.count};\\n");']


@dataclass(frozen=True)
class MbarrierWait(Stmt):
    """``mbarrier.try_wait.parity.shared.b64`` — block until the barrier's
    parity flips for ``phase``. Replaces ``CpAsyncWait + Sync`` for TMA
    transports — mbarrier arrival already provides CTA-wide visibility.
    ``slot`` selects the ring buffer slot when the mbarrier is an array."""

    mbar: str
    phase: Expr
    slot: Expr | None = None

    def pretty(self, indent: str = "") -> list[str]:
        s = "" if self.slot is None else f"[{self.slot.pretty()}]"
        return [f"{indent}MbarrierWait({self.mbar}{s}, phase={self.phase.pretty()})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        addr = "&" + self.mbar if self.slot is None else f"&{self.mbar}[{self.slot.render(ctx)}]"
        phase_expr = self.phase.render(ctx)
        return [f"{pad}mbarrier_wait_parity({addr}, {phase_expr});"]


@dataclass(frozen=True)
class TreeHalve(Stmt):
    """Cross-thread combine of a (possibly multi-component) monoid state over a
    PRE-POPULATED power-of-two smem tree — the > warp_size fallback for
    :class:`WarpShuffle`.

    Each state component has its own smem slab ``bufs[i][0..length)``, populated by
    the caller (``emit_combine`` writes the per-thread partials, or the
    hierarchical path writes one per-warp partial per slot). This stmt tree-reduces
    each slab into ``bufs[i][0]`` applying the carrier's ``combine_states`` at every
    halving step (rendered at ``dtype``), then broadcast-reassigns every
    ``state[i]`` from ``bufs[i][0]`` **in place** (no ``_b`` rename — every thread
    ends holding the full reduction in the carried SSA names). ``length`` is a power
    of two ≤ ``blockDim.x``. A scalar reduce is the 1-component case
    (``state=("acc",)``, ``combine_states=(Assign("acc", op, ("acc","acc__o")),)``).

    Named-barrier support: ``barrier_id > 0`` routes the per-iter sync to
    ``bar.sync <id>, <count>`` instead of ``__syncthreads()`` — required inside a
    warp-specialized consumer branch (a __syncthreads on a warp-divergent cond is
    UB).
    """

    bufs: tuple[str, ...]
    state: tuple[str, ...]
    state_b: tuple[str, ...]
    combine_states: tuple[Assign, ...]
    length: int
    tid_var: str
    dtype: DataType = F32
    barrier_id: int = 0
    barrier_count: int | None = None
    # Segment-indexed halving (the transposed b<n>t combine): each slab holds ``length``
    # SEGMENTS of ``inner[1]`` slots, and the tree halves the segment index at a fixed inner
    # slot — ``buf[t·scale + inner_var]`` vs ``buf[(t+s)·scale + inner_var]``, broadcast from
    # ``buf[inner_var]``. ``None`` keeps the flat ``buf[t]`` layout.
    inner: tuple[str, int] | None = None

    def deps(self) -> tuple[str, ...]:
        return tuple(self.state)

    def pretty(self, indent: str = "") -> list[str]:
        bar = "" if self.barrier_id == 0 else f", bar={self.barrier_id}/{self.barrier_count}"
        return [f"{indent}TreeHalve({', '.join(self.state)}, length={self.length}, tid={self.tid_var}{bar})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        in1 = _pad(ctx.indent + 1)
        in2 = _pad(ctx.indent + 2)
        t = self.tid_var
        ty = ctx.type_name(self.dtype.name)
        if self.barrier_id == 0:
            sync_line = f"{in1}__syncthreads();"
        else:
            if self.barrier_count is None:
                raise ValueError(f"TreeHalve(barrier_id={self.barrier_id}) requires barrier_count")
            sync_line = f'{in1}asm volatile("bar.sync {self.barrier_id}, {self.barrier_count};\\n" ::: "memory");'
        half = int(self.length) // 2
        if self.inner is not None:
            iv, scale = self.inner
            slot, slot_s, root = f"{t} * {scale} + {iv}", f"({t} + s) * {scale} + {iv}", str(iv)
        else:
            slot, slot_s, root = t, f"{t} + s", "0"
        out: list[str] = [f"{pad}for (int s = {half}; s > 0; s >>= 1) {{", f"{in1}if ({t} < s) {{"]
        # Shadow temps named after the carried state so ``combine_states`` (which
        # reassigns ``state``) folds ``buf[t+s]`` into ``buf[t]`` per component.
        for buf, st, sb in zip(self.bufs, self.state, self.state_b, strict=True):
            out.append(f"{in2}{ty} {st} = {buf}[{slot}];")
            out.append(f"{in2}{ty} {sb} = {buf}[{slot_s}];")
            ctx.ssa_dtypes[st] = self.dtype.name
            ctx.ssa_dtypes[sb] = self.dtype.name
        out.extend(render_merge_program(self.combine_states, self.state, ctx, pad=in2, dtype=self.dtype))
        for buf, st in zip(self.bufs, self.state, strict=True):
            out.append(f"{in2}{buf}[{slot}] = {st};")
        out.append(f"{in1}}}")
        out.append(sync_line)
        out.append(f"{pad}}}")
        # Broadcast the reduced state back into the carried SSA names (in place) — from the
        # segment root (this lane's slot 0) when segment-indexed.
        for buf, st in zip(self.bufs, self.state, strict=True):
            out.append(f"{pad}{st} = {buf}[{root}];")
            ctx.ssa_dtypes[st] = self.dtype.name
        return out


@dataclass(frozen=True)
class WarpShuffle(Stmt):
    """Cross-thread combine of a (possibly multi-component) monoid state over
    ``length`` lanes via ``__shfl_xor_sync`` — a register-only butterfly (no smem,
    no syncthreads).

    Each lane holds a full per-thread partial ``state`` tuple (a scalar reduce is
    the 1-component case ``state=("acc",)``). Each butterfly step shuffles **every**
    component down into ``state_b`` and applies the carrier's ``combine_states``
    (the state-merges-state monoid op, rendered at ``dtype``) to fold the shuffled
    neighbor into this lane's state, reassigning ``state`` **in place**. After the
    last step every lane holds the full reduction in the SAME SSA names — so the
    post-reduce epilogue reads them with no rename (kills the old scalar ``<name>_b``
    broadcast).

    ``length`` must be a power of two ≤ ``warp_size``; the XOR butterfly never
    crosses an aligned ``length``-lane group, so this is also the SEGMENTED per-row
    combine for strided-cooperative rows. ``commutative`` (required — the butterfly
    reorders) is checked at the carrier.

    The shuffle mask is ``__activemask()``, NOT a hard-coded ``0xffffffff``: a
    whole-CTA cooperative reduce with ``BR < warp_size`` launches a **partial warp**
    (e.g. ``block=8``), and naming the absent lanes ``8..31`` in the mask is undefined
    behavior — on the e2e captured-graph path it manifested as a multi-millisecond
    stall (a per-token×head RMSNorm went 28 ms instead of µs). The combine is reached
    with the warp converged (the per-CTA reduce has no data-dependent divergence and
    the only enclosing guard is warp-uniform), so ``__activemask()`` returns exactly
    the participating lanes — correct for a partial warp, a full warp, AND a
    non-warp-multiple block's partial final warp (where no single static literal mask
    is correct per warp).
    """

    state: tuple[str, ...]
    state_b: tuple[str, ...]
    combine_states: tuple[Assign, ...]
    length: int
    dtype: DataType = F32

    def deps(self) -> tuple[str, ...]:
        return tuple(self.state)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}WarpShuffle({', '.join(self.state)}, length={self.length})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        inner = _pad(ctx.indent + 1)
        ty = ctx.type_name(self.dtype.name)
        out: list[str] = []
        s = int(self.length) // 2
        while s > 0:
            # Block-scope each step so the shuffled state_b + merge temps redeclare
            # cleanly per round (the carried state is declared by an enclosing Init).
            out.append(f"{pad}{{")
            for st, sb in zip(self.state, self.state_b, strict=True):
                out.append(f"{inner}{ty} {sb} = __shfl_xor_sync(__activemask(), {st}, {s});")
                ctx.ssa_dtypes[sb] = self.dtype.name
            out.extend(render_merge_program(self.combine_states, self.state, ctx, pad=inner, dtype=self.dtype))
            out.append(f"{pad}}}")
            s >>= 1
        return out


@dataclass(frozen=True)
class Reassign(Stmt):
    """Reassign an already-declared carried scalar — ``name = value;`` (no ``float``
    decl). The streaming-flash online-softmax stats (``m`` / ``l``) are carried across
    the KV-tile loop: an enclosing ``Init`` declares them, the per-tile recurrence
    computes fresh SSA temps, and this rebinds the carried name to the new value
    (``Assign`` always *declares*, which would shadow the carried value)."""

    name: str
    value: str

    def deps(self) -> tuple[str, ...]:
        return (self.value,)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}{self.name} := {self.value}"]

    def render(self, ctx: RenderCtx) -> list[str]:
        return [f"{_pad(ctx.indent)}{self.name} = {self.value};"]


#: The per-arg distribution kinds of a :class:`FragmentApply` operand.
FRAG = "frag"  # a C-fragment operand — indexed per element (``arg[i]``)
ROW = "row"  # a per-row scalar — broadcast by row (suffix ``0`` for rows g, ``1`` for rows g+8)
UNIFORM = "uniform"  # a cell-uniform scalar / literal — the same value for all 4 elements


@dataclass(frozen=True)
class FragLayout:
    """The **per-atom** mma C-fragment register layout — the one place the fragment-tier nodes read
    their geometry instead of hard-coding m16n8 magic numbers. Lifting it into a descriptor is the
    "geometry generality" seam: a second atom plugs in by adding a :func:`frag_layout` entry, not by
    editing each node.

    - ``n_elems`` — f32 registers per lane in the C-fragment.
    - ``elem_row`` — the row index (``0 .. rows_per_lane-1``) each element belongs to; ``rows_per_lane``
      derives from it.
    - ``reduce_group`` — the row-reduce ``__shfl_xor`` butterfly lane span.
    - ``lane_decl`` — the per-kernel lane preamble the coordinate ``Expr``s below reference.
    - ``row_off`` / ``col_off`` — the per-element coordinate **offsets as ``Expr``s** (over the
      ``lane_decl`` locals): ``row_off[r]`` the in-tile row of row-index ``r``, ``col_off[i]`` the
      in-N-atom column of element ``i``. :class:`FragmentMask` adds the tile origin and substitutes
      these into its coordinate predicate (so the mask is a generic ``Expr``, not hard-coded CUDA).

    Only m16n8 is modeled today (both the QK^T and P@V atoms); ``frag_layout`` raises for any other,
    so an unmodeled atom fails loudly rather than miscompiling."""

    n_elems: int
    elem_row: tuple[int, ...]
    reduce_group: int
    lane_decl: str
    row_off: tuple[Expr, ...]
    col_off: tuple[Expr, ...]

    @property
    def rows_per_lane(self) -> int:
        return max(self.elem_row) + 1


def _m16n8_col(b: int) -> Expr:
    """The in-N-atom column of an m16n8 C-fragment element with col-bit ``b``: ``_t * 2 + b``."""
    return BinaryExpr("+", BinaryExpr("*", Var("_t"), Literal(2, "int")), Literal(b, "int"))


#: The standard ``mma.sync.m16n8`` D/C-fragment layout — 4 f32 / lane: elements ``[0,1]`` are row
#: ``g = lane/4`` (cols ``(lane%4)*2 + {0,1}``), ``[2,3]`` are row ``g+8``; a row's 8 cols span the
#: 4 lanes differing in ``lane%4`` (the ``reduce_group = 4`` butterfly).
M16N8 = FragLayout(
    n_elems=4,
    elem_row=(0, 0, 1, 1),
    reduce_group=4,
    lane_decl="const int _g = (threadIdx.x & 31) >> 2, _t = (threadIdx.x & 31) & 3;",
    row_off=(Var("_g"), BinaryExpr("+", Var("_g"), Literal(8, "int"))),
    col_off=(_m16n8_col(0), _m16n8_col(1), _m16n8_col(0), _m16n8_col(1)),
)


def frag_layout(atom_m: int, atom_n: int) -> FragLayout:
    """The :class:`FragLayout` for an mma atom — the single per-atom geometry source. Raises for any
    atom not modeled (only m16n8 today), so the fragment realizer fails loudly on a new tier."""
    if (atom_m, atom_n) == (16, 8):
        return M16N8
    raise NotImplementedError(f"no fragment C-layout modeled for atom m{atom_m}n{atom_n}")


def _lane_preamble(ctx: RenderCtx, pad: str, decl: str) -> list[str]:
    """Emit the mma lane preamble (``_g`` / ``_t``) once per C scope: ``decl`` if ``_g`` is not yet
    declared here (tracked in ``ctx.scope_decls``), else ``[]``. Every emitter defines ``_g``/``_t``
    identically (``(threadIdx.x & 31) >> 2`` / ``& 3``), so a later stmt in the same scope reuses the
    first declaration instead of re-scoping itself — no redundant ``{ }`` around the store / mask."""
    if "_g" in ctx.scope_decls:
        return []
    ctx.scope_decls.update(("_g", "_t"))
    return [f"{pad}{decl}"]


@dataclass(frozen=True)
class FragmentApply(Stmt):
    """Generic per-element pointwise op over ``mma.sync`` ``m16n8`` C-fragments — the
    **carrier-generic** fragment-tier sibling of the scalar ``Assign``, and the ONE fragment
    pointwise node (it subsumes the former hard-coded ``FragmentExp`` / ``FragmentScale``).

    Writes ``out`` (a ``float[4]`` C-fragment) ``= op(args…)`` per element ``i`` via the same
    ``op_to_expr`` translation the scalar ``Assign`` uses — so ANY elementwise op reaches the
    tensor-core tier, not just softmax's ``exp`` / scale. Each arg is one of three
    :data:`FRAG` / :data:`ROW` / :data:`UNIFORM` ``kinds``:

    - ``FRAG`` — a C-fragment, indexed ``arg[i]``;
    - ``ROW`` — a per-row scalar, broadcast by row (suffix ``0`` for rows ``g`` = elements
      ``[0,1]``, ``1`` for rows ``g+8`` = elements ``[2,3]`` — the m16n8 2-rows/lane layout);
    - ``UNIFORM`` — a cell-uniform scalar / literal, the same value for all 4 elements.

    Realizations: ``exp(s − m)`` = a ``subtract`` (FRAG, ROW) then an ``exp`` (FRAG); ``O *= α`` =
    an in-place ``multiply`` (FRAG, ROW); ``O /= l`` = an in-place ``divide`` (FRAG, ROW); ``S *=
    scale`` = an in-place ``multiply`` (FRAG, UNIFORM). ``in_place`` reassigns ``out`` (no ``float
    out[4]`` decl — ``out`` must be the first FRAG arg).

    Each ``args`` entry is a ``str`` for a FRAG / UNIFORM operand, or a ``(row0, row1)`` pair of
    SSA names for a ROW operand (the two per-row scalars stored explicitly — so SSA rename keeps
    them consistent with their definitions; a bare name + render-time suffix would diverge under
    rename)."""

    out: str
    op: ElementwiseImpl
    args: tuple[object, ...]  # str (FRAG / UNIFORM) | per-row name tuple (ROW)
    kinds: tuple[str, ...]  # per-arg: FRAG | ROW | UNIFORM
    in_place: bool = False
    layout: FragLayout = M16N8  # the per-atom C-fragment geometry (n_elems + elem→row)
    # Extra UNARY ops composed onto the result, outermost last — so ``exp(s − m)`` is one node
    # (``op=subtract``, ``post=(exp,)``) rendering ``out[_e] = expf(s[_e] − m)`` rather than a
    # subtract stmt feeding an exp stmt. Populated only by ``100_loopify``'s pin-gated chain fusion;
    # empty by default (the base op alone).
    post: tuple[ElementwiseImpl, ...] = ()

    def deps(self) -> tuple[str, ...]:
        out: list[str] = []
        for a, k in zip(self.args, self.kinds, strict=True):
            out += list(a) if k == ROW else [a]
        return tuple(out)

    def defines(self) -> tuple[str, ...]:
        return (self.out,)

    def pretty(self, indent: str = "") -> list[str]:
        def _show(a: object, k: str) -> str:
            return f"{a}[]" if k == FRAG else (f"{list(a)}" if k == ROW else str(a))

        shown = ", ".join(_show(a, k) for a, k in zip(self.args, self.kinds, strict=True))
        chain = "".join(f" |> {p.name}" for p in self.post)
        return [f"{indent}FragmentApply({self.out} {'*=' if self.in_place else '<-'} {self.op.name}({shown}){chain})"]

    def _row_expr(self, names: tuple) -> Expr:
        """The per-row scalar for the loop element ``_e`` — a ternary over the layout's row split
        (m16n8: ``(_e < 2 ? row0 : row1)``). ``elem_row`` must be contiguous blocks (it is for every
        modeled atom), so the boundaries are ``block = n_elems / rows_per_lane`` apart."""
        lay = self.layout
        rows, block = lay.rows_per_lane, lay.n_elems // lay.rows_per_lane
        if list(lay.elem_row) != [r for r in range(rows) for _ in range(block)]:
            raise NotImplementedError(f"FragmentApply loop render needs a contiguous elem_row, got {lay.elem_row!r}")
        expr: Expr = Var(names[rows - 1])
        for r in range(rows - 2, -1, -1):
            expr = TernaryExpr(BinaryExpr("<", Var("_e"), Literal((r + 1) * block, "int")), Var(names[r]), expr)
        return expr

    def _arg_e(self, name: object, kind: str, ctx: RenderCtx) -> Expr:
        """The loop-body operand expression (indexed by the element loop var ``_e``)."""
        if kind == FRAG:
            return Var(f"{name}[_e]")
        if kind == ROW:
            return self._row_expr(name)
        # UNIFORM — the fragment algebra is f32; a narrower scalar (e.g. an ``__half`` constant
        # load) converts through the target intrinsic, like the scalar ``Assign``'s promote rule.
        nm = str(name)
        dt = ctx.ssa_dtypes.get(nm) if ctx.ssa_dtypes else None
        if dt and dt != "f32":
            from emmy.compiler.ir.expr import FuncCallExpr  # noqa: PLC0415

            converted = ctx.target.convert(nm, dt, "f32")
            paren = converted.index("(") if "(" in converted else -1
            if paren > 0 and converted.endswith(")"):
                return FuncCallExpr(converted[:paren], [Var(nm)])
        return Var(nm)

    def render(self, ctx: RenderCtx) -> list[str]:
        """One ``#pragma unroll`` loop over the fragment's ``n_elems`` — ``out[_e] = post…(op(args_e))``
        — instead of ``n_elems`` straight-line assignments. The ROW operand renders as the row-split
        ternary, so the m16n8 2-rows/lane structure stays legible (``_e < 2 ? row0 : row1``); identical
        SASS (nvcc unrolls the pragma)."""
        from emmy.compiler.ir.stmt.base import op_to_expr  # noqa: PLC0415

        pad = _pad(ctx.indent)
        n = self.layout.n_elems
        inner = ctx.child()
        ipad = _pad(inner.indent)
        argvars = [self._arg_e(a, k, inner) for a, k in zip(self.args, self.kinds, strict=True)]
        expr = op_to_expr(self.op.name, argvars)
        for post_op in self.post:  # compose the fused unary tail (outermost last)
            expr = op_to_expr(post_op.name, [expr])
        lines = [] if self.in_place else [f"{pad}float {self.out}[{n}];"]
        lines.append(f"{pad}#pragma unroll")
        lines.append(f"{pad}for (int _e = 0; _e < {n}; _e++) {{")
        lines.append(f"{ipad}{self.out}[_e] = {expr.render(inner)};")
        lines.append(f"{pad}}}")
        return lines


@dataclass(frozen=True)
class FragmentRowReduce(Stmt):
    """Per-row reduction over an ``mma.sync`` C-fragment's N (column) lanes — the
    flash fragment-softmax ``rowmax`` / ``rowsum`` (validated by the FA-2 reference
    kernel in ``tests/compiler/e2e/test_attention_coverage.py``).

    Each lane of an ``m16n8`` C-fragment owns 4 f32 elements: rows ``g`` / ``g+8``
    (``g = lane/4``), cols ``(lane%4)*2 + {0,1}``. A ``BN``-wide score tile is
    ``len(frags)`` such fragments (one per N-atom). Reducing over N (the kv columns)
    is: combine each fragment's in-lane col pair (``frag[0]∘frag[1]`` for row ``g``,
    ``frag[2]∘frag[3]`` for row ``g+8``) across all N-atoms, then a ``__shfl_xor``
    butterfly over the ``group``-lane column set (``group=4`` for ``m16n8`` — lanes
    differing in ``lane%4`` hold the 8 distinct cols of a row). After the butterfly
    every lane in a column group holds the full per-row reduction, so the two outputs
    ``top`` (rows ``g``) / ``bot`` (rows ``g+8``) are correct on every lane — the
    fragment-distributed (2 rows/lane) form the online-softmax stats need.

    Distinct from :class:`WarpShuffle` (which reduces a whole per-thread monoid state
    over a cooperative-K lane set): this reduces *within* one warp's C-fragment over
    the atom's N direction, keyed on the PTX C-layout."""

    top: str  # the per-row reduction for rows g (broadcast across the column group)
    bot: str  # the per-row reduction for rows g+8
    frags: tuple[str, ...]  # the C-fragment arrays (float[4] each), one per N-atom of the BN tile
    op: ElementwiseImpl  # the reduce op (maximum for rowmax, add for rowsum)
    group: int = 4  # column-group lane span (m16n8: 4 lanes hold a row's 8 cols)
    dtype: DataType = F32

    def deps(self) -> tuple[str, ...]:
        return self.frags

    def defines(self) -> tuple[str, ...]:
        return (self.top, self.bot)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}FragmentRowReduce({self.top}, {self.bot} <- {', '.join(self.frags)}, op={self.op.name})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        f32 = ctx.type_name("f32")

        def combine(parts: list[str]) -> str:
            e = parts[0]
            for p in parts[1:]:
                e = _binary_combine_expr(self.op, e, p, ctx.target, "f32")
            return e

        top_parts = [f"{f}[{i}]" for f in self.frags for i in (0, 1)]
        bot_parts = [f"{f}[{i}]" for f in self.frags for i in (2, 3)]
        out = [f"{pad}{f32} {self.top} = {combine(top_parts)};", f"{pad}{f32} {self.bot} = {combine(bot_parts)};"]
        ctx.ssa_dtypes[self.top] = ctx.ssa_dtypes[self.bot] = "f32"
        s = int(self.group) // 2
        while s > 0:
            for nm in (self.top, self.bot):
                shfl = f"__shfl_xor_sync(0xffffffff, {nm}, {s})"
                out.append(f"{pad}{nm} = {_binary_combine_expr(self.op, nm, shfl, ctx.target, 'f32')};")
            s >>= 1
        return out


#: The reserved coordinate Vars a :class:`FragmentMask` predicate is written over — the element's
#: ABSOLUTE query row / key column; the render substitutes each element's coords (tile origin +
#: layout offset) for these.
FRAG_ROW = "__frow"
FRAG_COL = "__fcol"


@dataclass(frozen=True)
class FragmentMask(Stmt):
    """Generic per-element **coordinate-predicated fill** over an mma C-fragment — the ONE fragment
    mask node (it subsumes the former ``FragmentCausalMask`` / ``FragmentBoundaryMask``). Writes
    ``fill`` (the carrier's fold identity — the soft ``-1e30`` so ``max(m, fill) = m`` and
    ``exp(fill − m) = 0``) to every element whose absolute coordinates satisfy ``mask_when``, a
    predicate ``Expr`` over the reserved coordinate vars :data:`FRAG_ROW` (``__frow``, absolute
    query row) / :data:`FRAG_COL` (``__fcol``, absolute key column).

    The render adds the tile origin (``row_base`` / ``col_base``) to the layout's per-element offset
    and substitutes the result for ``__frow`` / ``__fcol``, then emits a guarded write — so the
    predicate is a generic ``Expr``, not hard-coded CUDA. Causal = ``__fcol > __frow``; symbolic
    boundary = ``__fcol >= seq_len``; any coordinate predicate (windowed, banded, …) is a different
    ``mask_when`` over the same node. Applied to the scaled score before the rowmax; emitting two
    masks in sequence ANDs their keep-predicates (both write ``fill``). ``row_base`` is required iff
    ``mask_when`` references ``__frow``."""

    frag: str
    mask_when: Expr
    col_base: Expr
    row_base: Expr | None = None
    fill: float = -1e30
    layout: FragLayout = M16N8

    def deps(self) -> tuple[str, ...]:
        return (self.frag,)

    def defines(self) -> tuple[str, ...]:
        return (self.frag,)

    def exprs(self) -> tuple[Expr, ...]:
        base = (self.mask_when, self.col_base)
        return base + ((self.row_base,) if self.row_base is not None else ())

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}FragmentMask({self.frag} where {self.mask_when.pretty()})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        lay = self.layout
        fill = ctx.identity_literal(self.fill, "f32")
        lines = _lane_preamble(ctx, pad, lay.lane_decl)
        for i in range(lay.n_elems):
            sub: dict[str, Expr] = {FRAG_COL: BinaryExpr("+", self.col_base, lay.col_off[i])}
            if self.row_base is not None:
                sub[FRAG_ROW] = BinaryExpr("+", self.row_base, lay.row_off[lay.elem_row[i]])
            pred = self.mask_when.substitute(sub).render(ctx)
            lines.append(f"{pad}if ({pred}) {self.frag}[{i}] = {fill};")
        return lines


@dataclass(frozen=True)
class FragmentBiasAdd(Stmt):
    """Per-element **additive gmem bias** over an mma C-fragment — the fragment-tier realization of
    the explicit additive score mask (SDPA's ``attn_mask`` float bias: the HF precomputed causal /
    sliding-window band). For every element, load ``buf`` at the element's ABSOLUTE ``(row, col)``
    coordinates and add it into the fragment: the render adds the tile origin (``row_base`` /
    ``col_base``) to the layout's per-element offset and substitutes the result for the reserved
    :data:`FRAG_ROW` / :data:`FRAG_COL` vars in ``index`` — the load-index TEMPLATE the realizer
    built from the bias ``Load``'s own index (leading broadcast dims pre-folded to literals; a
    symbolic seq pre-clamps the coordinate exprs, and the clamped duplicates land on cells the
    boundary :class:`FragmentMask` / store guard discards). The buffer element converts to f32 on
    the add (the fragment algebra is f32); lanes ``_t = 0..3`` of a column group read 8 contiguous
    columns, so the warp's bias reads coalesce like the epilogue leaf loads."""

    frag: str
    buf: str
    index: tuple[Expr, ...]  # load-index template over FRAG_ROW / FRAG_COL
    col_base: Expr
    row_base: Expr
    layout: FragLayout = M16N8

    def deps(self) -> tuple[str, ...]:
        return (self.frag,)

    def defines(self) -> tuple[str, ...]:
        return (self.frag,)

    def external_reads(self) -> tuple[str, ...]:
        return (self.buf,)

    def exprs(self) -> tuple[Expr, ...]:
        return (*self.index, self.col_base, self.row_base)

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(e.pretty() for e in self.index)
        return [f"{indent}FragmentBiasAdd({self.frag} += {self.buf}[{idx}])"]

    def render(self, ctx: RenderCtx) -> list[str]:
        from emmy.compiler.ir.stmt import render_index  # noqa: PLC0415

        pad = _pad(ctx.indent)
        lay = self.layout
        conv = {"f16": "__half2float({})", "bf16": "__bfloat162float({})"}
        dt = ctx.buffer_dtypes.get(self.buf, "f32")
        lines = _lane_preamble(ctx, pad, lay.lane_decl)
        for i in range(lay.n_elems):
            sub: dict[str, Expr] = {
                FRAG_COL: BinaryExpr("+", self.col_base, lay.col_off[i]),
                FRAG_ROW: BinaryExpr("+", self.row_base, lay.row_off[lay.elem_row[i]]),
            }
            idx = tuple(e.substitute(sub) for e in self.index)
            flat = render_index(self.buf, idx, ctx)
            lines.append(f"{pad}{self.frag}[{i}] += {conv.get(dt, '{}').format(f'{self.buf}[{flat}]')};")
        return lines


# ---------------------------------------------------------------------------
# Warp-level MMA: ``mma.sync.aligned`` + ``ldmatrix`` (the ``s16816`` path
# cuBLAS / CUTLASS use) — the sole tensor-core Stmt family. Operands are
# *explicit per-thread register arrays* with a PTX-fixed lane→element layout,
# referenced positionally inside inline PTX (``RegFragment`` / ``LdmatrixLoad``
# / ``MmaSyncPtx`` / ``RegStore``), rendered via the ``_MMA_SYNC_PRELUDE``
# helper wrappers (pure PTX — NVRTC-clean, no ``<mma.h>``). Emitted by
# ``kernel/005_lower_atom_tile`` from the ``Mma`` op's ``Atom`` spec.
# (The opaque ``nvcuda::wmma`` node family was removed — the swizzled mma.sync
# slab beat it.)
# ---------------------------------------------------------------------------


def _mma_sync_nregs(role: str, shape: tuple[int, int, int], dtype: DataType | None = None) -> int:
    """Per-lane register count for an mma.sync operand of cell ``shape``.

    ``a`` (M×K) and ``b`` (K×N) pack two 16-bit or four 8-bit elements per 32-bit register
    (``dtype`` is the operand element type; ``None`` keeps the 16-bit default); ``c`` holds one
    float per register (f32 accumulate) or two packed halfs (the f16-accumulate atom). For
    ``m16n8k16`` f16: ``a[4]`` (256 halfs / 32 lanes / 2), ``b[2]`` (128 / 32 / 2), ``c[4]`` f32
    (128 f32 / 32) / ``c[2]`` f16; for ``m16n8k32`` fp8: ``a[4]`` (512 bytes / 32 / 4), ``b[2]``."""
    m, n, k = shape
    if role in ("a", "b"):
        per_reg = 4 if dtype is not None and dtype.nbytes == 1 else 2
        return (m * k if role == "a" else k * n) // (32 * per_reg)
    if role == "c":
        return (m * n) // 64 if dtype is not None and dtype.nbytes == 2 else (m * n) // 32
    raise ValueError(f"mma.sync: unsupported role {role!r}; expected 'a', 'b', or 'c'")


@dataclass(frozen=True)
class RegFragment(Stmt):
    """Declare an mma.sync per-thread register array.

    mma.sync multiplicands are explicit per-lane register arrays —
    ``unsigned a[4]`` / ``unsigned b[2]`` (f16, two halfs per 32-bit
    reg) — and the accumulator is ``float c[4]`` (f32) or, on the
    f16-accumulate atom, packed ``unsigned c[2]`` (two halfs per reg —
    the same element map, pair-packed). ``shape`` is the cell
    ``(M, N, K)``; the count derives from ``shape`` + ``role`` (+ the C
    dtype) via :func:`_mma_sync_nregs`. The ``c`` array is
    zero-initialised at declaration, so the mma.sync path needs no
    separate fill node.

    ``count`` (default 1) arrays a whole fragment FAMILY into one declaration —
    ``float O_i_f[count][n_regs]`` — so a run of ``count`` sibling fragments
    (``O_i_f0 … O_i_f{count-1}``) collapses to a single arrayed decl indexed
    ``O_i_f[t]``. Only ``100_loopify`` sets ``count > 1`` (the pin-gated
    re-roll); ``count == 1`` renders the flat ``float name[n_regs]`` form,
    byte-identical."""

    name: str
    role: str  # "a" / "b" / "c"
    shape: tuple[int, int, int]
    dtype: DataType
    count: int = 1  # >1 arrays the fragment family: ``<ty> name[count][n_regs]`` (loopify only)
    nregs: int | None = None  # explicit for layouts whose one instruction realizes several PTX cells

    def _nregs(self) -> int:
        return self.nregs if self.nregs is not None else _mma_sync_nregs(self.role, self.shape, self.dtype)

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def local_decls(self) -> tuple[str, ...]:
        return (self.name,)

    def pretty(self, indent: str = "") -> list[str]:
        m, n, k = self.shape
        cnt = f"{self.count}][" if self.count > 1 else ""
        dims = f"[{cnt}{self._nregs()}]"
        return [f"{indent}RegFragment {self.role}:{self.dtype.name} {self.name}{dims} ({m}x{n}x{k})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        n_regs = self._nregs()
        ctx.ssa_dtypes[self.name] = self.dtype.name
        dims = f"[{self.count}][{n_regs}]" if self.count > 1 else f"[{n_regs}]"
        if self.role == "c" and self.dtype.nbytes != 2:
            # Brace-init zeros the whole (arrayed) accumulator; ``= {0.0f, ...}`` stays the flat form.
            init = " = {}" if self.count > 1 else f" = {{{', '.join(['0.0f'] * n_regs)}}}"
            return [f"{_pad(ctx.indent)}float {self.name}{dims}{init};"]
        if self.role == "c":
            # f16-accumulate C: packed half pairs, zero bits = 0.0h pairs (rezeroed by FragmentPromote).
            init = " = {}" if self.count > 1 else f" = {{{', '.join(['0u'] * n_regs)}}}"
            return [f"{_pad(ctx.indent)}unsigned {self.name}{dims}{init};"]
        return [f"{_pad(ctx.indent)}unsigned {self.name}{dims};"]


# Per-lane fp16 element XOR matching the TMA hardware smem swizzle (the
# ldmatrix consumer side). The producer (cuTensorMapEncodeTiled with
# SWIZZLE_{128,64,32}B) permutes 16-byte (8-fp16) chunks within each swizzle
# row by the row-index-mod-2^B; the ldmatrix consumer must apply the same
# permutation to its per-lane element offset. All three modes are CUTLASS
# ``Swizzle<B,4,3>`` (M=4 → 16-byte base, S=3 → 8-row tile), differing ONLY in
# B (3/2/1 XORed bits). In bytes the swizzle is
# ``off ^= ((off >> (M+S)) & (2^B-1)) << M`` = ``((off >> 7) & mask) << 4``;
# halving to fp16 elements turns ``>> 7`` into ``>> 6`` and ``<< 4`` into
# ``<< 3`` — so the element shift is 6 for EVERY mode (a per-mode shift of
# 6/5/4 silently corrupts B64/B32 while leaving B128 correct, which is easy to
# miss since B128 dominates). Only the row mask changes: 0x7 / 0x3 / 0x1. The
# 8-row × atom period divides the slab + slot strides, so the XOR is correct
# measured from the buffer base. Maps mode → (element shift, row mask); the
# chunk delta is always ``<< 3``.
#
# The XOR is purely address-based, so it is also the SOFTWARE swizzle: a
# ``cp.async``-filled slab applies the same helper to its fill DESTINATION
# index (:class:`CpAsyncCopy` ``swizzle``) and the ldmatrix drain reads it
# back through the identical XOR — fill and drain agree by construction, no
# hardware copy engine involved. Intra-chunk bits (0..2) pass through, so
# narrower 4/8-byte fills keep their offset inside the relocated 16-byte
# chunk and ``cp.async`` address alignment is preserved.
LDMATRIX_SWIZZLE_XOR: dict[str, tuple[int, int]] = {
    "B128": (6, 0x7),
    "B64": (6, 0x3),
    "B32": (6, 0x1),
}


@dataclass(frozen=True)
class LdmatrixLoad(Stmt):
    """``ldmatrix.sync.aligned.m8n8.x{2,4}[.trans].b16`` — load one operand
    fragment from smem into a per-thread register array.

    ``frag`` is the destination :class:`RegFragment`; ``src_buffer`` /
    ``src_index`` are the cell's smem tile base (the same ``(buffer,
    base-offset)`` ``_mma_src_index`` computes); ``ldm``
    is the smem row stride in elements. Each lane derives its own row
    address from its warp lane id (``threadIdx.x & 31``): the 16×K ``a``
    tile uses ``x4`` (``row = lane%16``, K-col block ``(lane/16)*8``); the
    K×8 ``b`` tile uses ``x2.trans`` (``row = lane%16``) so a row-major
    smem slab feeds the mma's col-major B operand.

    ``staged`` (default ``True``) selects the transport. The modern fragment layout's ``ldmatrix`` is **smem
    only**, while the Volta layout uses its cooperative gather against either address space. When the
    operand was NOT staged into shared memory
    (``staged=False``, ``src_buffer`` is the gmem operand) the render emits the
    ``emmy_mma_load_{a,b}_gmem`` helper instead — a gmem-direct fragment load that
    replicates the same lane→element map without ldmatrix. Slower (no smem reuse)
    but correct; ``005_lower_atom_tile`` picks per operand based on whether an
    enclosing ``StageBundle`` staged it.

    ``gmem_guard`` (gmem-direct only) carries a masked-tile boundary as
    ``(base Expr, bound Expr)`` on the operand's lane-varying output axis
    (A's M rows, B's N cols): the render switches to the ``_mclamp`` /
    ``_nclamp`` helper, which clamps the lane coordinate to the ``bound -
    base`` in-range elements — the gmem-direct analogue of the staged
    slab-fill clamp, so an unstaged masked cell still lowers instead of
    reading past the runtime-sized buffer. Clamped lanes read duplicates;
    their stores are masked by the RegStore guards."""

    frag: str
    src_buffer: str
    src_index: tuple
    role: str  # "a" → x4; "b" → x2.trans
    ldm: int = 0
    swizzle: str = "NONE"  # TMA smem swizzle mode the slab was written with
    staged: bool = True  # False → gmem-direct fragment load (operand not in smem)
    gmem_guard: tuple[Expr, Expr] | None = None  # masked-axis (base, bound); gmem-direct only
    k_zero: tuple[Expr, Expr] | None = None  # masked-K reduce axis (base, bound); gmem-direct only
    b_trans: bool = False  # role "b" only: B stored N×K (Q@K^T native col-major) → gmem-direct trans helper / staged plain x2
    # A second B fragment filled by the SAME ldmatrix (the ``096_pair_ldmatrix_loads`` peephole):
    # the paired x4 loads two slab-adjacent B fragments in one instruction — ``frag`` gets the
    # matrices addressed by lanes 0-15, ``pair_frag`` those of lanes 16-31. Staged role-"b"
    # NONE-swizzle only; ``src_index`` stays the FIRST (low) fragment's tile base.
    pair_frag: str | None = None
    # A staged 1-BYTE (fp8) operand slab: ldmatrix is b16-only below sm_100a, so the drain is the
    # cooperative per-lane byte gather instead — the SAME lane→element map the gmem-direct
    # fragment loaders spell, pointed at the smem slab. A 16-bit destination fragment (W8A16)
    # converts per element via the loader's F template arg; an fp8 fragment (the k32 atoms)
    # repacks raw bytes via the ``_b8`` family. NONE-swizzle by construction; the padded row
    # stride rides ``ldm``. Staged only.
    byte_slab: bool = False
    fragment_layout: str = "m16n8k16"

    def deps(self) -> tuple[str, ...]:
        return (self.frag,) if self.pair_frag is None else (self.frag, self.pair_frag)

    def defines(self) -> tuple[str, ...]:
        return (self.frag,) if self.pair_frag is None else (self.frag, self.pair_frag)

    def external_reads(self) -> tuple[str, ...]:
        return (self.src_buffer,)

    def exprs(self) -> tuple[Expr, ...]:
        guard = () if self.gmem_guard is None else self.gmem_guard
        kz = () if self.k_zero is None else self.k_zero
        return (*self.src_index, *guard, *kz)

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(e.pretty() for e in self.src_index)
        if self.pair_frag is not None:
            variant = "x4 pair" if self.b_trans else "x4.trans pair"
            pair = f"({self.frag}, {self.pair_frag})"
            return [f"{indent}LdmatrixLoad {pair} <- {self.src_buffer}[{idx}] ({variant}, ldm={self.ldm or 'auto'})"]
        if self.byte_slab:
            variant = "byte gather"
        elif self.fragment_layout == "m8n8k4" and self.staged:
            variant = "shared gather"
        else:
            variant = ("x4" if self.role == "a" else ("x2" if self.b_trans else "x2.trans")) if self.staged else "gmem-direct"
        guard = "" if self.gmem_guard is None else f" guard<{self.gmem_guard[1].pretty()}"
        return [f"{indent}LdmatrixLoad {self.frag} <- {self.src_buffer}[{idx}] ({variant}{guard}, ldm={self.ldm or 'auto'})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        from emmy.compiler.ir.stmt import render_index  # noqa: PLC0415

        flat = render_index(self.src_buffer, self.src_index, ctx)
        ldm = self.ldm if self.ldm else _resolve_ldm(self.src_buffer, ctx)
        if not self.staged:
            # Operand not staged in smem — ldmatrix can't reach gmem, so read the
            # fragment straight from gmem (each lane adds its own (row,col) inside
            # the helper). No swizzle: that's a TMA-smem-layout concern only.
            # A source buffer traced at a dtype other than the fragment's (the gemma layer's
            # f32 V intermediate) converts per element: the helper's fragment element type F
            # is passed explicitly, so the pack writes true 16-bit halves instead of raw f32
            # bit patterns (matched dtypes keep the bare call — byte-identical output).
            src_dt = ctx.buffer_dtypes.get(self.src_buffer, "f16")
            frag_dt = ctx.ssa_dtypes.get(self.frag) or src_dt
            targs = "" if src_dt == frag_dt else f"<{ctx.type_name(src_dt)}, {ctx.type_name(frag_dt)}>"
            b8 = frag_dt in ("f8e4m3", "f8e5m2")
            if self.fragment_layout == "m8n8k4":
                if self.k_zero is not None:
                    kbase, kbound = self.k_zero[0].render(ctx), self.k_zero[1].render(ctx)
                    k_left = f"({kbound}) - ({kbase})"
                    if self.gmem_guard is not None:
                        base, bound = self.gmem_guard[0].render(ctx), self.gmem_guard[1].render(ctx)
                        mn_left = f"({bound}) - ({base})"
                        helper = (
                            "emmy_mma884_load_a_gmem_mclamp_kzero"
                            if self.role == "a"
                            else ("emmy_mma884_load_b_gmem_trans_nclamp_kzero" if self.b_trans else "emmy_mma884_load_b_gmem_nclamp_kzero")
                        )
                        return [f"{_pad(ctx.indent)}{helper}{targs}({self.frag}, &{self.src_buffer}[{flat}], {ldm}, {mn_left}, {k_left});"]
                    helper = (
                        "emmy_mma884_load_a_gmem_kzero"
                        if self.role == "a"
                        else ("emmy_mma884_load_b_gmem_trans_kzero" if self.b_trans else "emmy_mma884_load_b_gmem_kzero")
                    )
                    return [f"{_pad(ctx.indent)}{helper}{targs}({self.frag}, &{self.src_buffer}[{flat}], {ldm}, {k_left});"]
                if self.gmem_guard is not None:
                    base, bound = self.gmem_guard[0].render(ctx), self.gmem_guard[1].render(ctx)
                    helper = (
                        "emmy_mma884_load_a_gmem_mclamp"
                        if self.role == "a"
                        else ("emmy_mma884_load_b_gmem_trans_nclamp" if self.b_trans else "emmy_mma884_load_b_gmem_nclamp")
                    )
                    return [f"{_pad(ctx.indent)}{helper}{targs}({self.frag}, &{self.src_buffer}[{flat}], {ldm}, ({bound}) - ({base}));"]
                helper = (
                    "emmy_mma884_load_a_gmem"
                    if self.role == "a"
                    else ("emmy_mma884_load_b_gmem_trans" if self.b_trans else "emmy_mma884_load_b_gmem")
                )
                return [f"{_pad(ctx.indent)}{helper}{targs}({self.frag}, &{self.src_buffer}[{flat}], {ldm});"]
            if b8:
                # fp8 fragments gather RAW bytes with the k32 fragment map (the ``_b8`` helper
                # family) — there is no per-element convert (the mma consumes storage bits), so
                # the source buffer must already carry the fragment dtype, and no masked-K byte
                # gather exists (the schedule offers the fp8 atoms only on a static K).
                if src_dt != frag_dt:
                    raise ValueError(f"fp8 mma fragment {self.frag}: source buffer {self.src_buffer!r} is {src_dt}, not {frag_dt}")
                if self.k_zero is not None:
                    raise ValueError("fp8 mma fragments have no masked-K byte gather — the fp8 atoms are static-K only")
            sfx = "_b8" if b8 else ""
            if self.k_zero is not None:
                # Masked-K (symbolic reduce): zero-fill (not clamp) the K halves
                # past the runtime extent so the mma reduction stays correct — K is
                # summed, so a duplicate corrupts it. ``k_left`` = in-range K elements
                # from the tile base; may co-occur with an M/N clamp. Transposed-B (K
                # contiguous) has its own (n, k)-swapped zero-fill helper.
                kbase, kbound = self.k_zero[0].render(ctx), self.k_zero[1].render(ctx)
                k_left = f"({kbound}) - ({kbase})"
                if self.gmem_guard is not None:
                    base, bound = self.gmem_guard[0].render(ctx), self.gmem_guard[1].render(ctx)
                    mn_left = f"({bound}) - ({base})"
                    if self.role == "a":
                        helper = "emmy_mma_load_a_gmem_mclamp_kzero"
                    else:
                        helper = "emmy_mma_load_b_gmem_trans_nclamp_kzero" if self.b_trans else "emmy_mma_load_b_gmem_nclamp_kzero"
                    return [f"{_pad(ctx.indent)}{helper}{targs}({self.frag}, &{self.src_buffer}[{flat}], {ldm}, {mn_left}, {k_left});"]
                if self.role == "a":
                    helper = "emmy_mma_load_a_gmem_kzero"
                else:
                    helper = "emmy_mma_load_b_gmem_trans_kzero" if self.b_trans else "emmy_mma_load_b_gmem_kzero"
                return [f"{_pad(ctx.indent)}{helper}{targs}({self.frag}, &{self.src_buffer}[{flat}], {ldm}, {k_left});"]
            if self.gmem_guard is not None:
                # Masked axis: clamp the lane coordinate to the in-range
                # elements left from the tile base (>= 1 — the boundary Cond
                # admitted the tile).
                base, bound = self.gmem_guard[0].render(ctx), self.gmem_guard[1].render(ctx)
                if self.role == "a":
                    helper = "emmy_mma_load_a_gmem_mclamp"
                else:
                    helper = "emmy_mma_load_b_gmem_trans_nclamp" if self.b_trans else "emmy_mma_load_b_gmem_nclamp"
                return [f"{_pad(ctx.indent)}{helper}{sfx}{targs}({self.frag}, &{self.src_buffer}[{flat}], {ldm}, ({bound}) - ({base}));"]
            if self.role == "a":
                helper = "emmy_mma_load_a_gmem"
            else:
                helper = "emmy_mma_load_b_gmem_trans" if self.b_trans else "emmy_mma_load_b_gmem"
            return [f"{_pad(ctx.indent)}{helper}{sfx}{targs}({self.frag}, &{self.src_buffer}[{flat}], {ldm});"]
        if self.fragment_layout == "m8n8k4":
            # SM70 has no ldmatrix. The Volta fragment's cooperative lane map is the same for
            # global and shared addresses, so point its inlined gather at the staged slab; ptxas
            # resolves the generic pointer to ordinary LDS instructions. Fill-side clamps make
            # the slab exact, and the first implementation intentionally keeps it unswizzled.
            assert self.pair_frag is None and not self.byte_slab, "the Volta staged drain is an unpacked f16 gather"
            assert self.swizzle == "NONE", "the Volta shared-memory gather has no swizzled layout"
            slab_dt = ctx.buffer_dtypes.get(self.src_buffer, "f16")
            frag_dt = ctx.ssa_dtypes.get(self.frag) or slab_dt
            targs = "" if slab_dt == frag_dt else f"<{ctx.type_name(slab_dt)}, {ctx.type_name(frag_dt)}>"
            helper = (
                "emmy_mma884_load_a_smem"
                if self.role == "a"
                else ("emmy_mma884_load_b_smem_trans" if self.b_trans else "emmy_mma884_load_b_smem")
            )
            return [f"{_pad(ctx.indent)}{helper}{targs}({self.frag}, &{self.src_buffer}[{flat}], {ldm});"]
        if self.byte_slab:
            # Staged 1-byte (fp8) slab — the cooperative byte-gather drain: the gmem fragment
            # loaders' lane→element map pointed at the smem slab (generic-address loads; ptxas
            # resolves them to LDS after inlining). A 16-bit fragment (W8A16) converts per
            # element via the loader's F template arg (the same ``<__nv_fp8_e4m3, __half>``
            # spelling as the gmem-direct convert); an fp8 fragment (the k32 atoms) repacks raw
            # bytes via the ``_b8`` family. ``flat`` already strides the PADDED slab decl and
            # ``ldm`` carries the same padded row stride, so the two agree by construction.
            slab_dt = ctx.buffer_dtypes.get(self.src_buffer, "f8e4m3")
            frag_dt = ctx.ssa_dtypes.get(self.frag) or "f16"
            k_contig = self.role == "a" or self.b_trans  # each lane's K run is slab-contiguous
            if frag_dt in ("f8e4m3", "f8e5m2"):
                # k32 byte repack. Contiguous-K lanes load one u32 (the ``_smem_b8v`` family);
                # a canonical K-major B strides its 4 bytes across rows — the scalar gather.
                if k_contig:
                    trans = "_trans" if self.b_trans else ""
                    call = f"emmy_mma_load_{self.role}_smem{trans}_b8v<{ctx.type_name(slab_dt)}>"
                else:
                    call = "emmy_mma_load_b_gmem_b8"
            elif self.b_trans and frag_dt == "f16":
                # W8A16 pair convert: one fp8x2 load + one hardware cvt.rn.f16x2 per k-half.
                x2 = {"f8e4m3": "__nv_fp8x2_e4m3", "f8e5m2": "__nv_fp8x2_e5m2"}[slab_dt]
                call = f"emmy_mma_load_b_smem_trans_f8_f16<{ctx.type_name(slab_dt)}, {x2}>"
            else:
                # Canonical K-major B (K-strided bytes) or a bf16 fragment: the scalar
                # per-element convert via the gmem loader template, pointed at the slab.
                trans = "_trans" if self.b_trans else ""
                call = f"emmy_mma_load_{self.role}_gmem{trans}<{ctx.type_name(slab_dt)}, {ctx.type_name(frag_dt)}>"
            return [f"{_pad(ctx.indent)}{call}({self.frag}, &{self.src_buffer}[{flat}], {ldm});"]
        lane = "(threadIdx.x & 31)"
        if self.pair_frag is not None:
            # Paired x4 — two slab-adjacent B fragments in one ldmatrix (096_pair_ldmatrix_loads).
            # Transposed-B (N-major slab): the pair is N-adjacent — lanes 0-15 address ``frag``'s 8
            # N rows × two k8 col-halves, lanes 16-31 the same at N+8 (plain x4). Canonical-B
            # (K-major slab): the pair is N(col)-adjacent — lanes 0-15 address the 16 K rows at
            # ``frag``'s col, lanes 16-31 at col+8 (x4.trans).
            assert self.role == "b" and self.staged, "paired ldmatrix is a staged B-operand fusion"
            if self.b_trans:
                elem = f"{flat} + (({lane} % 8) + ({lane} / 16) * 8) * {ldm} + (({lane} / 8) % 2) * 8"
                helper = "emmy_ldmatrix_x4_pair"
            else:
                elem = f"{flat} + ({lane} % 16) * {ldm} + ({lane} / 16) * 8"
                helper = "emmy_ldmatrix_x4_trans_pair"
            # The helper loads both fragments' registers directly (no ``_p4`` staging temp / block).
            return [f"{_pad(ctx.indent)}{helper}({self.frag}, {self.pair_frag}, {self._swizzled_addr(elem)});"]
        if self.role == "a":
            # 16×16 A: x4 — lane addresses M-row (lane%16), K-col block (lane/16)*8.
            elem = f"{flat} + ({lane} % 16) * {ldm} + ({lane} / 16) * 8"
            addr = self._swizzled_addr(elem)
            return [f"{_pad(ctx.indent)}emmy_ldmatrix_x4({self.frag}, {addr});"]
        # Transposed-B (Q@K^T): the slab keeps the operand's native N-major layout
        # (N rows × K cols), which IS the mma's col-major B — a plain x2 (no
        # .trans) with N-row addresses: lanes 0-7 address the 8 N rows of the k16
        # half 0, lanes 8-15 the same rows at K col 8 (each 8x8 matrix's rows land
        # as the fragment's (k, k+1) pairs, cf. ``emmy_mma_load_b_gmem_trans``).
        if self.b_trans:
            elem = f"{flat} + ({lane} % 8) * {ldm} + (({lane} / 8) % 2) * 8"
            addr = self._swizzled_addr(elem)
            return [f"{_pad(ctx.indent)}emmy_ldmatrix_x2({self.frag}, {addr});"]
        # 16×8 B: x2.trans — lane addresses K-row (lane%16); .trans yields col-major.
        elem = f"{flat} + ({lane} % 16) * {ldm}"
        addr = self._swizzled_addr(elem)
        return [f"{_pad(ctx.indent)}emmy_ldmatrix_x2_trans({self.frag}, {addr});"]

    def _swizzled_addr(self, elem: str) -> str:
        if self.swizzle not in LDMATRIX_SWIZZLE_XOR:
            return f"&{self.src_buffer}[{elem}]"
        # ``emmy_swizzle_<mode>`` (preamble, built from ``LDMATRIX_SWIZZLE_XOR``) applies
        # ``e ^ (((e >> shift) & mask) << 3)`` — the helper spells the (often long) element
        # index once instead of inlining it twice around the XOR.
        return f"&{self.src_buffer}[emmy_swizzle_{self.swizzle.lower()}({elem})]"


@dataclass(frozen=True)
class MmaSyncPtx(Stmt):
    """``mma.sync.aligned.m{M}n{N}k{K}.row.col.f32.{ab}.{ab}.f32`` — one
    tensor-core MMA via inline PTX (the ``s16816`` instruction).

    ``c_frag`` is the accumulator array (both input and output —
    ``d = a·b + c``); ``a_frag`` / ``b_frag`` are the multiplicand
    arrays filled by :class:`LdmatrixLoad`. ``shape`` spells the M/N/K.
    ``ab_dtype`` (``"f16"`` / ``"bf16"`` / ``"e4m3"`` / ``"e5m2"`` — the PTX
    dtype field, :attr:`AtomKind.ab_dtype`) tags the operand element type —
    same-width dtypes share a fragment layout and differ only in the PTX
    instruction's dtype field, so the render just picks the
    matching ``emmy_mma_…`` wrapper. ``c_dtype`` is the accumulator element
    type: ``"f32"`` (the default — ``float c[4]``) or ``"f16"`` (the
    f16-accumulate atom — packed ``unsigned c[2]``, full HMMA rate on the
    consumer dies; the lowering pairs it with a periodic
    :class:`FragmentPromote` into an f32 shadow fragment)."""

    c_frag: str
    a_frag: str
    b_frag: str
    shape: tuple[int, int, int]
    ab_dtype: str = "f16"
    c_dtype: str = "f32"

    def deps(self) -> tuple[str, ...]:
        return (self.c_frag, self.a_frag, self.b_frag)

    def defines(self) -> tuple[str, ...]:
        # Accumulates into c_frag in place — a definition, like MmaSyncPtx.
        return (self.c_frag,)

    def pretty(self, indent: str = "") -> list[str]:
        m, n, k = self.shape
        acc = "" if self.c_dtype == "f32" else f" acc:{self.c_dtype}"
        return [f"{indent}MmaSyncPtx {self.c_frag} += {self.a_frag} @ {self.b_frag} (m{m}n{n}k{k} {self.ab_dtype}{acc})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        m, n, k = self.shape
        # Wrapper names follow the atom convention's <ab>_<acc> dtype pair (emmy_mma_m16n8k16_f16_f32).
        # ``c`` is passed for both the ``d`` (out) and ``c`` (in) operands.
        wrapper = f"emmy_mma_m{m}n{n}k{k}_{self.ab_dtype}_{self.c_dtype}"
        return [f"{_pad(ctx.indent)}{wrapper}({self.c_frag}, {self.a_frag}, {self.b_frag}, {self.c_frag});"]


@dataclass(frozen=True)
class FragmentPromote(Stmt):
    """Fold an f16-accumulate mma **C fragment** into its f32 **shadow fragment** and rezero it —
    the chunked-accumulation promote of the ``_f16acc`` atom family: the mma chain accumulates at
    the full f16-accumulate HMMA rate, and every K chunk this promote-adds the packed f16 partials
    into the f32 shadow (``cvt.f32.f16`` + add per element) and re-zeros the f16 fragment, so the
    accumulation error stays bounded by one chunk's length. ``dst`` is the f32 shadow the store /
    epilogue reads (``float[4]``); ``src`` the packed f16 mma accumulator (``unsigned[2]``) —
    both defined here (``src`` is rezeroed), so reorderings keep the promote pinned between the
    mma chain and the store."""

    dst: str  # f32 shadow accumulator fragment (4 × f32) — the store-side view
    src: str  # packed f16 mma accumulator fragment (2 × u32) — rezeroed after the fold

    def deps(self) -> tuple[str, ...]:
        return (self.dst, self.src)

    def defines(self) -> tuple[str, ...]:
        return (self.dst, self.src)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}FragmentPromote {self.dst} += {self.src} (f16acc chunk fold, {self.src} rezeroed)"]

    def render(self, ctx: RenderCtx) -> list[str]:
        return [f"{_pad(ctx.indent)}emmy_mma_promote_f16acc({self.dst}, {self.src});"]


@dataclass(frozen=True)
class FragmentRepack(Stmt):
    """Convert two k-adjacent mma **C fragments** (f32, m16n8) into one 16-bit **A operand
    fragment** (m16k16) IN REGISTERS — the ``AtomKind.c_to_a_repack`` lane-map compatibility:
    per lane, ``srcs[0]``'s four values are exactly the A fragment's low k-half pairs and
    ``srcs[1]``'s its high k-half, so the conversion is four ``cvt.rn.{f16,bf16}x2.f32`` packs
    with no shuffle and no smem round-trip (the flash P→A handoff; same round-to-nearest-even
    the smem path's ``RegStore`` applied, so the repack is bit-identical to it). The emitter
    gates on the atom capability at schedule time — this node assumes it."""

    frag: str  # destination A fragment (4 × u32)
    srcs: tuple[str, str]  # (low k-half, high k-half) C fragments (4 × f32 each)
    ab_dtype: str = "f16"

    def deps(self) -> tuple[str, ...]:
        return self.srcs

    def defines(self) -> tuple[str, ...]:
        return (self.frag,)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}FragmentRepack {self.frag} <- ({self.srcs[0]}, {self.srcs[1]}) ({self.ab_dtype})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        return [f"{_pad(ctx.indent)}emmy_c_to_a_{self.ab_dtype}({self.frag}, {self.srcs[0]}, {self.srcs[1]});"]


@dataclass(frozen=True)
class EpilogueLoad:
    """One leaf operand of a fused pointwise epilogue (see :class:`RegEpilogue`).

    Not a body :class:`Stmt` — a payload riding the :class:`RegStore`.
    ``index`` is the cell-base coordinate tuple (struct-shared with the
    epilogue's Write); per fragment element the render adds the element's own
    row / col motion on every dim whose ``role`` is ``"m"`` / ``"n"`` (at
    *this* buffer's dim stride — so a transposed or broadcast operand reads
    correctly), and nothing on ``"fixed"`` dims (literals, batch / grid vars —
    uniform across the cell)."""

    name: str
    buffer: str
    index: tuple
    roles: tuple[str, ...]


@dataclass(frozen=True)
class RegEpilogue:
    """A pure pointwise SSA chain folded into the fragment store.

    Captured by ``kernel/005_lower_atom_tile`` from the backward slice between
    the accumulator and the Write (the scalar Load / Assign stmts are stripped
    — the accumulator has no scalar SSA name on the fragment path). The render
    evaluates the chain per fragment element in f32 with ``acc`` substituted
    by the element and each leaf loaded at the element's own (row, col); the
    chain ops reuse the scalar renderer's ``op_to_expr`` translation, so any
    elementwise op with a CUDA spelling works. ``ops`` are ``(name, op_name,
    args)`` in topological (body) order; ``result`` is the SSA name the Write
    stored."""

    acc: str
    loads: tuple[EpilogueLoad, ...]
    # ``dtype`` is an optional semantic result boundary.  Narrow results are converted down and
    # back to f32 before the next fused op, reproducing the round that a materialized tensor store
    # and reload would have imposed while keeping the epilogue's arithmetic environment uniform.
    ops: tuple[tuple[str, str, tuple[str, ...], str | None], ...]
    result: str
    # Coord-predicated Selects (the causal attention mask), rendered before the
    # ``ops`` chain as per-element ternaries. Each is ``(name, branches)`` where
    # ``branches`` is ``((cond_expr | None, value_name), ...)`` — the predicate
    # carries ``__M__`` / ``__N__`` placeholder Vars the store substitutes with
    # the fragment element's own (row, col); the last branch is the else.
    selects: tuple[tuple[str, tuple[tuple[Expr | None, str], ...]], ...] = ()
    # Additional ``(acc_name, frag_name)`` accumulator bindings — a multi-fold contraction's
    # extra C fragments (the fused gate/up edge): each name substitutes to its fragment's
    # element alongside ``acc``, so the chain combines the folds per cell (SwiGLU et al).
    extra_accs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class RegStore(Stmt):
    """Store an mma.sync f32 accumulator array to the output buffer with a
    per-lane epilogue downconvert.

    mma.sync has no ``store_matrix_sync`` — each lane owns 4 elements of
    the M×N=16×8 C tile (rows ``g`` / ``g+8`` with ``g = lane/4``, cols
    ``(lane%4)*2 + {0,1}``) and writes them directly. ``frag`` is the f32
    ``float c[4]``; when the destination buffer is narrower (``__half*``)
    each value is converted via ``__float2half``. ``ldm`` is the output row stride
    (N) — ``0`` auto-resolves from the buffer's inner extent.

    ``epilogue`` optionally carries a fused pointwise chain
    (:class:`RegEpilogue` — residual adds, bias / scale broadcasts,
    activations): evaluated per fragment element in f32 right before the
    downconvert (the CUTLASS epilogue-visitor pattern).

    ``m_guard`` / ``n_guard`` carry a masked-tile boundary as ``(base Expr,
    bound Expr)`` — the tile's row / col coordinate of fragment element (0,0)
    and the axis's real extent (a ``Literal`` for a static overhang axis, the
    symbolic ``Var`` for a runtime-sized one). The enclosing boundary ``Cond``
    only gates on the tile base (the atom-lane offsets ``_g`` / ``_t`` are
    render-local, invisible to σ), so an atom tile *straddling* the bound
    passes the Cond while its trailing rows / cols are out of range. A guard
    predicates each fragment element's store (and its epilogue gmem reads) at
    its own coordinate: ``base + _g (+8) < bound`` per row, ``base + _t*2
    (+1) < bound`` per col. ``None`` (the default) renders the unguarded
    fast path unchanged. ``m_guard`` alone keeps the vectorized row-pair
    stores (a pair shares a row); an ``n_guard`` forces per-element scalar
    stores (the pair straddles the column bound).

    ``atomic`` renders each store as an ``atomicAdd`` accumulate instead of a
    plain assign — ``030_split_reduce``'s atomic finalize on the mma tier: every
    split partition's C fragment adds into the (per-launch zero-init'd) output.
    f16 / bf16 keep the vectorized row pair (native packed-pair ``atomicAdd``);
    an f32 destination has no packed atomic below sm_90, so it drops to
    per-element scalar ``atomicAdd``\\ s."""

    dst_buffer: str
    dst_index: tuple
    frag: str
    shape: tuple[int, int, int]
    ldm: int = 0
    epilogue: RegEpilogue | None = None
    m_guard: tuple[Expr, Expr] | None = None
    n_guard: tuple[Expr, Expr] | None = None
    atomic: bool = False
    fragment_layout: str = "m16n8k16"

    def deps(self) -> tuple[str, ...]:
        extra = () if self.epilogue is None else tuple(fr for _, fr in self.epilogue.extra_accs)
        return (self.frag, *extra)

    def external_reads(self) -> tuple[str, ...]:
        # The fused epilogue's leaf loads are gmem reads this stmt performs
        # directly (their original Load stmts were stripped by
        # 005_lower_atom_tile), so they must be declared here for the kernel
        # signature / render shapes to include the buffers.
        if self.epilogue is None:
            return ()
        return tuple(ld.buffer for ld in self.epilogue.loads)

    def external_writes(self) -> tuple[str, ...]:
        return (self.dst_buffer,)

    def exprs(self) -> tuple[Expr, ...]:
        epi = () if self.epilogue is None else tuple(e for ld in self.epilogue.loads for e in ld.index)
        guards = tuple(e for g in (self.m_guard, self.n_guard) if g is not None for e in g)
        return (*self.dst_index, *epi, *guards)

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(e.pretty() for e in self.dst_index)
        epi = ""
        if self.epilogue is not None:
            chain = ", ".join(op for _, op, _, _ in self.epilogue.ops)
            bufs = ", ".join(ld.buffer for ld in self.epilogue.loads)
            epi = f" epilogue[{chain}]({bufs or 'no loads'})"
        guards = ""
        if self.m_guard is not None:
            guards += f" m<{self.m_guard[1].pretty()}"
        if self.n_guard is not None:
            guards += f" n<{self.n_guard[1].pretty()}"
        acc = " (atomic)" if self.atomic else ""
        return [f"{indent}RegStore {self.dst_buffer}[{idx}] <- {self.frag}{epi}{guards}{acc} (ldm={self.ldm or 'auto'})"]

    def _pair_store(self, addr: str, v0: str, v1: str, vec2: str, packer: str) -> str:
        """One vectorized row-pair store line — a plain assign, or (``atomic``) the packed-pair
        ``atomicAdd`` (native f16x2 / bf16x2 red; the caller keeps f32 off this path pre-sm90)."""
        tgt = f"reinterpret_cast<{vec2}*>(&{self.dst_buffer}[{addr}])"
        if self.atomic:
            return f"atomicAdd({tgt}, {packer}({v0}, {v1}));"
        return f"*{tgt} = {packer}({v0}, {v1});"

    def _elem_store(self, addr: str, val: str, dst_dt: str) -> str:
        """One per-element store line — plain assign (the implicit f32→dst conversion), or
        (``atomic``) a scalar ``atomicAdd`` with the conversion made explicit (no float overload
        on a ``__half*`` / ``__nv_bfloat16*`` destination)."""
        if self.atomic:
            conv = {"f16": "__float2half({})", "bf16": "__float2bfloat16({})"}.get(dst_dt, "{}")
            return f"atomicAdd(&{self.dst_buffer}[{addr}], {conv.format(val)});"
        return f"{self.dst_buffer}[{addr}] = {val};"

    def _element_coords(self) -> list[tuple[str, str, Expr, Expr]]:
        """Per-lane ``(row C text, col C text, row Expr, col Expr)`` for this fragment layout."""
        from emmy.compiler.ir.expr import BinaryExpr, Var  # noqa: PLC0415

        if self.fragment_layout == "m8n8k4":
            out = []
            for i in range(8):
                ro, co = i & 2, (i & 4) + (i & 1)
                row = Var("_vr") if ro == 0 else BinaryExpr("+", Var("_vr"), Literal(ro, "int"))
                col = Var("_vc") if co == 0 else BinaryExpr("+", Var("_vc"), Literal(co, "int"))
                out.append(("_vr" if ro == 0 else f"(_vr + {ro})", "_vc" if co == 0 else f"(_vc + {co})", row, col))
            return out
        return [
            (
                "_g" if i < 2 else "(_g + 8)",
                f"(_t * 2 + {i & 1})",
                Var("_g") if i < 2 else BinaryExpr("+", Var("_g"), Literal(8, "int")),
                BinaryExpr("+", BinaryExpr("*", Var("_t"), Literal(2, "int")), Literal(i & 1, "int")),
            )
            for i in range(4)
        ]

    def _element_values(self, ctx: RenderCtx) -> tuple[list[list[str]], list[str]]:
        """``(per_element_preamble, values)``: the four per-lane store values,
        plus, per element, the leaf-load / chain-op declaration lines that
        element needs (so a guarded render can scope element ``i``'s gmem
        reads inside element ``i``'s boundary check). Without
        an epilogue the values are the bare ``frag[i]`` and the preambles are
        empty. With one, each element ``i`` (row ``_g``/``_g+8``, col
        ``2_t+{0,1}``) declares its leaf loads (converted to f32; offsets per
        the dim roles at each buffer's own stride) and the chain ops (via
        ``op_to_expr`` — the same translation the scalar ``Assign`` render
        uses), all scoped inside the store's ``{ }`` block. Leaf loads are
        scalar; lanes ``_t = 0..3`` cover 8 contiguous columns, so the warp's
        accesses coalesce regardless."""
        coords = self._element_coords()
        if self.epilogue is None:
            return [[] for _ in coords], [f"{self.frag}[{i}]" for i in range(len(coords))]
        from emmy.compiler.ir.expr import BinaryExpr, Var  # noqa: PLC0415
        from emmy.compiler.ir.stmt import render_index  # noqa: PLC0415
        from emmy.compiler.ir.stmt.base import op_to_expr  # noqa: PLC0415

        epi = self.epilogue
        conv = {"f16": "__half2float({})", "bf16": "__bfloat162float({})"}
        # Coord-predicated Selects (causal mask) need the cell-base M / N coords
        # (the last two var-bearing output dims) — the element adds its own
        # (row, col) to get the absolute coordinate the predicate compares.
        sel_m_base = sel_n_base = None
        if epi.selects:
            dvd = [e for e in self.dst_index if e.free_vars()]
            sel_m_base = dvd[-2] if len(dvd) >= 2 else None
            sel_n_base = dvd[-1] if dvd else None
        per_elem: list[list[str]] = []
        vals: list[str] = []
        for i, (row, col, row_off, col_off) in enumerate(coords):
            lines: list[str] = []
            env = {epi.acc: f"{self.frag}[{i}]", **{a: f"{fr}[{i}]" for a, fr in epi.extra_accs}}
            for ld in epi.loads:
                temp = f"{ld.name}_e{i}"
                if ld.buffer in ctx.literal_constants:
                    lines.append(f"const float {temp} = {float(ctx.literal_constants[ld.buffer])!r}f;")
                    env[ld.name] = temp
                    continue
                flat = render_index(ld.buffer, ld.index, ctx)
                parts = [flat]
                for d, role in enumerate(ld.roles):
                    if role == "fixed":
                        continue
                    stride = _dim_stride(ld.buffer, d, ctx)
                    motion = row if role == "m" else col
                    parts.append(motion if stride == 1 else f"{motion} * {stride}")
                addr = " + ".join(parts)
                dt = ctx.buffer_dtypes.get(ld.buffer, "f32")
                lines.append(f"const float {temp} = {conv.get(dt, '{}').format(f'{ld.buffer}[{addr}]')};")
                env[ld.name] = temp
            # Coord-predicated Selects (the causal mask): a per-element ternary.
            # ``__M__`` / ``__N__`` substitute to this element's absolute (row,
            # col); branches fold right with the last branch as the else.
            m_abs = BinaryExpr("+", sel_m_base, row_off) if sel_m_base is not None else row_off
            n_abs = BinaryExpr("+", sel_n_base, col_off) if sel_n_base is not None else col_off
            coord = {"__M__": m_abs, "__N__": n_abs}
            for sel_name, branches in epi.selects:
                expr = env[branches[-1][1]]
                for cond, value in reversed(branches[:-1]):
                    rc = cond.substitute(coord).render(ctx)
                    expr = f"(({rc}) ? {env[value]} : {expr})"
                lines.append(f"const float {sel_name}_e{i} = {expr};")
                env[sel_name] = f"{sel_name}_e{i}"
            for name, op_name, args, result_dtype in epi.ops:
                expr = op_to_expr(op_name, [Var(env[a]) for a in args])
                body = expr.render(ctx)
                if result_dtype is not None and result_dtype != "f32":
                    body = ctx.target.convert(ctx.target.convert(body, "f32", result_dtype), result_dtype, "f32")
                lines.append(f"const float {name}_e{i} = {body};")
                env[name] = f"{name}_e{i}"
            per_elem.append(lines)
            vals.append(env[epi.result])
        return per_elem, vals

    def render(self, ctx: RenderCtx) -> list[str]:
        from emmy.compiler.ir.stmt import render_index  # noqa: PLC0415

        flat = render_index(self.dst_buffer, self.dst_index, ctx)
        ldm = self.ldm if self.ldm else _resolve_ldm(self.dst_buffer, ctx)
        dst_dt = ctx.buffer_dtypes.get(self.dst_buffer, "f32")
        pad = _pad(ctx.indent)
        lane = "(threadIdx.x & 31)"
        pre, vals = self._element_values(ctx)
        if self.fragment_layout == "m8n8k4":
            return self._render_m8n8k4(ctx, flat=flat, ldm=ldm, dst_dt=dst_dt, pre=pre, vals=vals)
        # C is 16×8: lane owns (row g, cols 2t,2t+1) and (row g+8, cols 2t,2t+1)
        # with g = lane/4, t = lane%4. The two cols per row are CONTIGUOUS, so
        # each row's pair is one vectorized 4-byte store (``__half2`` / ``float2``)
        # rather than two scalar stores — halves the epilogue store count and
        # the address arithmetic. ldm strides over N (the output row width). The
        # base ``flat`` is tile-aligned and ``2t`` is even, so the pair is
        # 4-/8-byte aligned. The ``{ }`` block scopes _g/_t (and the per-element
        # epilogue temps) per RegStore.
        if self.m_guard is not None or self.n_guard is not None:
            return self._render_guarded(ctx, flat=flat, ldm=ldm, dst_dt=dst_dt, pre=pre, vals=vals)
        lane_stmt = f"const int _g = {lane} >> 2; const int _t = {lane} & 3;"
        if self.epilogue is None:
            # Flash output store: no per-element epilogue temps, so declare _g/_t once per scope
            # (reused by sibling stores) and emit the stores flat — no wrapping { } block, so a
            # store re-rolled into a loop renders ``for (…) { … }`` not ``for (…) { { … } }``.
            head, base, close = _lane_preamble(ctx, pad, lane_stmt), pad, ""
        else:
            # Fused epilogue: the { } scopes the per-element temps (they collide across sibling
            # register-tile cells, which reuse the same names), so it stays.
            head, base, close = [f"{pad}{{ {lane_stmt}"], f"{pad}  ", " }"
        body = [f"{base}{ln}" for group in pre for ln in group]
        vec2 = {"f16": "__half2", "bf16": "__nv_bfloat162", "f32": "float2"}.get(dst_dt)
        if vec2 is not None and not (self.atomic and dst_dt == "f32"):  # no packed f32 atomicAdd pre-sm90
            packer = {"f16": "__floats2half2_rn", "bf16": "__floats2bfloat162_rn", "f32": "make_float2"}[dst_dt]
            lo = f"{flat} + _g * {ldm} + _t * 2"
            hi = f"{flat} + (_g + 8) * {ldm} + _t * 2"
            body += [
                f"{base}{self._pair_store(lo, vals[0], vals[1], vec2, packer)}",
                f"{base}{self._pair_store(hi, vals[2], vals[3], vec2, packer)}",
            ]
        else:  # per-element scalar stores (dtypes without a 2-vector packer; atomic f32)
            body += [
                f"{base}{self._elem_store(f'{flat} + _g * {ldm} + _t * 2 + 0', vals[0], dst_dt)}",
                f"{base}{self._elem_store(f'{flat} + _g * {ldm} + _t * 2 + 1', vals[1], dst_dt)}",
                f"{base}{self._elem_store(f'{flat} + (_g + 8) * {ldm} + _t * 2 + 0', vals[2], dst_dt)}",
                f"{base}{self._elem_store(f'{flat} + (_g + 8) * {ldm} + _t * 2 + 1', vals[3], dst_dt)}",
            ]
        if close:
            body[-1] += close
        return head + body

    def _render_m8n8k4(self, ctx: RenderCtx, *, flat: str, ldm, dst_dt: str, pre: list[list[str]], vals: list[str]) -> list[str]:
        """Store the four m8n8k4 computation groups arranged as one logical 16x16 cell."""
        pad = _pad(ctx.indent)
        lane = "(threadIdx.x & 31)"
        # Each four-lane group and its +16 partner own one 8x8 computation. Place the four
        # groups as quadrants; _vr/_vc are this lane's base row/column within the 16x16 cell.
        lines = [
            f"{pad}{{ const int _vl = {lane}; const int _vq = (_vl & 15) >> 2;",
            f"{pad}  const int _vr = (_vq >> 1) * 8 + (_vl >> 4) * 4 + (_vl & 1);",
            f"{pad}  const int _vc = (_vq & 1) * 8 + (_vl & 2);",
        ]
        mbase = mbound = nbase = nbound = None
        if self.m_guard is not None:
            mbase, mbound = (e.render(ctx) for e in self.m_guard)
        if self.n_guard is not None:
            nbase, nbound = (e.render(ctx) for e in self.n_guard)
        for i, (row, col, _row_expr, _col_expr) in enumerate(self._element_coords()):
            preds = []
            if mbound is not None:
                preds.append(f"({mbase}) + {row} < ({mbound})")
            if nbound is not None:
                preds.append(f"({nbase}) + {col} < ({nbound})")
            indent = f"{pad}  "
            if preds:
                lines.append(f"{indent}if ({' && '.join(preds)}) {{")
                indent += "  "
            lines.extend(f"{indent}{ln}" for ln in pre[i])
            addr = f"{flat} + {row} * {ldm} + {col}"
            lines.append(f"{indent}{self._elem_store(addr, vals[i], dst_dt)}")
            if preds:
                lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")
        return lines

    def _render_guarded(self, ctx: RenderCtx, *, flat: str, ldm, dst_dt: str, pre: list[list[str]], vals: list[str]) -> list[str]:
        """Masked-tile store: each fragment element's store (and its epilogue
        gmem reads, which index the same possibly-out-of-range coordinates)
        runs under that element's own boundary check. With only an ``m_guard``
        the two columns of a row share the row predicate, so the vectorized
        row-pair store survives; an ``n_guard`` splits the pair (its columns
        straddle the bound) into per-element scalar stores."""
        pad = _pad(ctx.indent)
        lane = "(threadIdx.x & 31)"
        m_pred: list[str | None] = [None] * 4
        n_pred: list[str | None] = [None] * 4
        if self.m_guard is not None:
            base, bound = self.m_guard[0].render(ctx), self.m_guard[1].render(ctx)
            m_pred = [f"({base}) + _g < ({bound})"] * 2 + [f"({base}) + _g + 8 < ({bound})"] * 2
        if self.n_guard is not None:
            base, bound = self.n_guard[0].render(ctx), self.n_guard[1].render(ctx)
            n_pred = [
                f"({base}) + _t * 2 < ({bound})",
                f"({base}) + _t * 2 + 1 < ({bound})",
                f"({base}) + _t * 2 < ({bound})",
                f"({base}) + _t * 2 + 1 < ({bound})",
            ]
        lines = [f"{pad}{{ const int _g = {lane} >> 2; const int _t = {lane} & 3;"]
        vec2 = {"f16": "__half2", "bf16": "__nv_bfloat162", "f32": "float2"}.get(dst_dt)
        if self.n_guard is None and vec2 is not None and not (self.atomic and dst_dt == "f32"):
            # Row-guarded vectorized pairs: elements {0,1} share row _g,
            # {2,3} share row _g+8; each pair's preamble + store sit inside
            # the pair's row check.
            packer = {"f16": "__floats2half2_rn", "bf16": "__floats2bfloat162_rn", "f32": "make_float2"}[dst_dt]
            for pair, addr_row in ((0, "_g"), (2, "(_g + 8)")):
                addr = f"{flat} + {addr_row} * {ldm} + _t * 2"
                lines.append(f"{pad}  if ({m_pred[pair]}) {{")
                lines += [f"{pad}    {ln}" for ln in (*pre[pair], *pre[pair + 1])]
                lines.append(f"{pad}    {self._pair_store(addr, vals[pair], vals[pair + 1], vec2, packer)}")
                lines.append(f"{pad}  }}")
            lines.append(f"{pad}}}")
            return lines
        # Per-element scalar stores under the conjunction of the live guards.
        for i in range(4):
            row = "_g" if i < 2 else "(_g + 8)"
            preds = " && ".join(p for p in (m_pred[i], n_pred[i]) if p is not None)
            addr = f"{flat} + {row} * {ldm} + _t * 2 + {i & 1}"
            lines.append(f"{pad}  if ({preds}) {{")
            lines += [f"{pad}    {ln}" for ln in pre[i]]
            lines.append(f"{pad}    {self._elem_store(addr, vals[i], dst_dt)}")
            lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")
        return lines


def _ext_to_c(ext, ctx: RenderCtx) -> int | str:
    """One buffer-shape extent as a C term: static dims yield their ``int``,
    a symbolic ``Dim`` renders its Expr against the runtime kernel arg
    (parenthesized — the result is interpolated into ``* {…}`` address
    strings)."""
    if hasattr(ext, "is_static"):
        if ext.is_static:
            return ext.as_static()
        return f"({ext.expr.render(ctx)})"
    return int(ext)


def _dim_stride(buffer: str, dim: int, ctx: RenderCtx) -> int | str:
    """Row-major element stride of ``buffer``'s dim ``dim`` — the product of
    the trailing extents. Used by the fragment-epilogue render to apply a
    fragment element's row / col motion on an operand dim at that operand's
    own layout (a transposed residual's "n" dim strides by its row width,
    not by 1). A symbolic trailing extent makes the stride a C expression
    string over the runtime kernel arg instead of an int."""
    shape = ctx.shapes.get(buffer)
    if shape is None:
        raise ValueError(f"RegStore epilogue: buffer {buffer!r} not in ctx.shapes (no shape registered)")
    static = 1
    sym: list[str] = []
    for ext in shape[dim + 1 :]:
        term = _ext_to_c(ext, ctx)
        if isinstance(term, int):
            static *= term
        else:
            sym.append(term)
    if not sym:
        return static
    expr = " * ".join(sym if static == 1 else [str(static), *sym])
    return f"({expr})"


def _resolve_ldm(buffer: str, ctx: RenderCtx) -> int | str:
    """Look up the row-major leading-dimension stride (= inner extent)
    for ``buffer`` from the kernel render context. Used by
    :class:`LdmatrixLoad` / :class:`RegStore` when ``ldm == 0`` (auto).
    Accepts both raw int extents and :class:`Dim` extents (Tensor.shape) —
    a symbolic inner extent (e.g. QK^T's ``(seq, seq)`` output) resolves to
    a C expression over the runtime kernel arg (the M9 runtime-ldm path);
    every consumer interpolates ``ldm`` into an address string, so the
    int/str split is transparent."""
    shape = ctx.shapes.get(buffer)
    if shape is None:
        raise ValueError(f"_resolve_ldm: buffer {buffer!r} not in ctx.shapes (no shape registered)")
    return _ext_to_c(shape[-1], ctx)


def _binary_combine_expr(op: ElementwiseImpl, a: str, b: str, target=None, dt: str = "f32") -> str:
    """Render a 2-arg combine for ``ElementwiseImpl`` reduce ops at ``dt``.

    ``target`` (optional) provides the dtype-specific intrinsic spelling.
    ``None`` keeps the legacy f32 spellings for callers that haven't
    plumbed a target through yet.
    """
    spelling = _REDUCE_SPELLING.get(op.reduce_canon)
    if spelling is None:
        raise ValueError(f"TreeHalve: unsupported op {op.name!r}")
    if spelling.infix is not None:
        return f"{a} {spelling.infix} {b}"
    fn = target.intrinsic(spelling.intrinsic, dt) if target is not None else f"{spelling.intrinsic}f"
    return f"{fn}({a}, {b})"


# ``StridedLoop`` is shared infrastructure — defined in ``ir/stmt.py``
# and re-exported here. Used at Tile IR for cooperative iteration and
# at Kernel IR for cooperative smem loads.


# ---------------------------------------------------------------------------
# Top-level: KernelOp
# ---------------------------------------------------------------------------


# Hard ceiling on the launch grid for ``KernelOp.validate``. The CUDA driver
# allows ~2^31 CTAs per dim; the cap here is sized to allow any realistic
# matmul launch (including K-split fan-in, which multiplies CTA count by
# the split factor — e.g. (M=32, K=18944, N=3584) with BM=BN=16 and
# auto-splitK=37 produces 2 × 224 × 37 ≈ 17 k CTAs of heavy per-CTA work),
# while still rejecting truly degenerate launches that would saturate
# the GPU command processor with light per-CTA work. The autotune-side
# guard against pathological tiny-CTA × huge-grid variants is the
# enumeration gate (``_enumeration._matmul_thread_gate``) + the online
# prior, not this cap.
_MAX_CTAS = 65536


def pack_smem(smems) -> tuple[dict[str, int], int]:  # noqa: ANN001 — smems: Iterable[Smem]
    """Pack ``Smem`` buffers into one contiguous pool, padding each to its
    alignment. Walks the buffers in order, aligns the running cursor to
    ``max(nbytes_of(dtype), s.align)`` before placing each buffer, and returns
    ``({name: byte_offset}, total_padded_bytes)``.

    Single source of truth for the per-CTA smem footprint: ``KernelOp.smem_bytes``
    and ``render._compute_dynamic_smem_offsets`` both call this so the
    static-vs-dynamic gate and the launch-time pool size agree (an unpadded sum
    would under-report when a buffer's alignment exceeds its natural stride —
    e.g. a 512/1024 B swizzle-atom-aligned operand)."""
    from math import prod  # noqa: PLC0415

    from emmy.compiler.backend.cuda.dtype import nbytes_of  # noqa: PLC0415

    offsets: dict[str, int] = {}
    cursor = 0
    for s in smems:
        elements = prod(int(e) for e in s.extents) if s.extents else 1
        align = max(nbytes_of(s.dtype), int(s.align) if s.align else 0)
        if align:
            cursor = (cursor + align - 1) // align * align
        offsets[s.name] = cursor
        cursor += elements * nbytes_of(s.dtype)
    return offsets, cursor


@dataclass
class KernelOp(BodyOp):
    """One ``__global__`` GPU kernel as a Kernel IR program.

    :class:`BodyOp` subclass parallel to ``TileOp`` / ``LoopOp``: lives as
    a graph node, carries a body of Kernel IR stmts plus a kernel name.

    Buffer shapes are *not* baked in — the surrounding graph supplies
    them at render time, same as ``TileOp``. Kernel signature is derived
    from the body: distinct ``Load.input`` names become kernel input
    params, distinct ``Write.output`` names become writeable output
    params, ordered by first appearance. ``Smem`` buffers are excluded.
    ``CpAsyncCopy.src`` and ``TmaDescriptor.src_buf`` also name kernel
    input parameters (the descriptor parameter itself is host-built and
    appended by the CUDA backend's argument pipeline, not a graph
    buffer).

    ``zero_delegated`` lists atomic-accumulator buffers whose per-launch zero-init a
    dataflow-PREDECESSOR kernel carries as a ``ZeroPrologue`` stmt
    (``lowering/cuda/005_delegate_zero_init``) — ``010_lower_kernelop`` drops them from the
    ``CudaOp.zero_outputs`` memset list."""

    zero_delegated: tuple[str, ...] = ()

    @cached_property
    def smem_buffers(self) -> dict[str, Smem]:
        """Every ``Smem`` decl in the body, keyed by name. Cached — KernelOp
        is treated as immutable post-construction (passes return new
        instances), so the deep walk runs at most once per op. The deep
        walk matters because ``Smem`` stmts sit inside ``Tile`` / ``Loop``
        bodies post-lowering; a top-level-only iterator would miss them
        and let oversize kernels slip past :meth:`validate`."""
        return {s.name: s for s in self if isinstance(s, Smem)}

    def smem_bytes(self) -> int:
        """Total static + dynamic ``__shared__`` bytes declared in the body —
        the padding-aware pool footprint from :func:`pack_smem` (each buffer
        aligned to ``max(sizeof(dtype), Smem.align)``), so this matches the
        renderer's pool packer and the launch-time pool size."""
        return pack_smem(self.smem_buffers.values())[1]

    def validate(self, ctx) -> bool:
        """Drop kernels whose launch wouldn't fit the hardware. Runs
        after every tile-level rewrite has settled (the engine calls
        ``validate`` on each op rebind, but only at the ``KernelOp``
        stage are the THREAD axes guaranteed to reflect the final
        per-CTA launch geometry — earlier ``TileOp`` rewrites may still
        split or coalesce them). Three checks:

        - threads ≤ ``ctx.max_threads_per_cta`` (driver-side launch cap),
        - smem ≤ ``ctx.max_dynamic_smem`` (per-block ``cudaFuncSetAttribute``
          opt-in cap; smaller for older / consumer cards),
        - CTAs ≤ ``_MAX_CTAS`` (a hard cap on the launch grid — empirically
          variants with tens of thousands of light CTAs slot into the
          GPU command processor so slowly that the per-launch
          ``_KERNEL_TIMEOUT_MS`` watchdog stops being a useful escape
          hatch; better to drop them at the rule level before benching).

        NOTE: the per-CTA thread / grid-CTA checks walked the now-demolished
        tile-flavor wrappers (``GridTile`` / ``ThreadTile``); they are pending
        rebuild. Only the smem-footprint check survives."""
        if self.smem_bytes() > ctx.max_dynamic_smem:
            return False
        return True


# ---------------------------------------------------------------------------
# Tree walk — shared with Loop IR (drives off ``Stmt.nested``)
# ---------------------------------------------------------------------------


__all__ = [
    # Shared expressions (re-exported)
    "Var",
    "Literal",
    "BinaryExpr",
    "Builtin",
    "FuncCallExpr",
    "TernaryExpr",
    "CastExpr",
    "Expr",
    # Loop-IR leaves + control flow (reused)
    "Load",
    "Assign",
    "Select",
    "SelectBranch",
    "Write",
    "Accum",
    "Cond",
    "Loop",
    # Kernel-IR statements
    "Tile",
    "Smem",
    "Sync",
    "TreeHalve",
    "WarpShuffle",
    "CpAsyncCopy",
    "CpAsyncCommit",
    "CpAsyncWait",
    "TmaDescriptor",
    "TmaLoad",
    "MbarrierInit",
    "MbarrierArriveExpectTx",
    "MbarrierArrive",
    "MbarrierWait",
    "SetMaxNReg",
    "StridedLoop",
    "Stmt",
    # Top-level
    "KernelOp",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]


# ---------------------------------------------------------------------------
# rewrite-dispatch handlers for Kernel-IR stmts
# ---------------------------------------------------------------------------
#
# ``Body.structural_key()`` runs ``normalize_body``, which dispatches
# :func:`emmy.compiler.ir.stmt.passes.rewrite` over every stmt. The
# default dispatch raises ``NotImplementedError``, so we register a handler
# per Kernel-IR stmt here. Buffer names (``Smem.name``, mbarriers, TMA
# descriptors, etc.) are *not* SSA — the ``rename`` callback only canon-
# icalizes SSA tokens — so they pass through unchanged. ``Expr`` fields go
# through ``sigma.apply``; the leaf-only stmts (``Sync`` / ``CpAsyncCommit``
# / ``CpAsyncWait``) are stateless and return themselves.


from emmy.compiler.ir.stmt.passes import rewrite as _rewrite  # noqa: E402


@_rewrite.register
def _(s: Tile, rename, sigma, axis_fn):
    # ``axes`` map through ``axis_fn``; the body's stmts route through the
    # generic per-stmt rewrite so SSA names / Exprs canonicalize inside. The
    # per-CTA thread counts (``block_threads`` / ``aux_threads``) are geometry,
    # not SSA — carry them through verbatim (dropping them silently falls the
    # launch back to the scalar ``_BLOCK_SIZE``, over-launching a cooperative
    # tile), exactly as ``Tile.with_bodies`` preserves them. The rasterization
    # fields are geometry too — ``raster_axes`` names axes, so it follows the
    # ``axis_fn`` renames (dropping it silently reverts the launch order to the
    # ungrouped N-fastest raster — the exact silent-loss this handler's comment
    # has always warned about).
    new_axes = tuple(axis_fn(a) for a in s.axes)
    raster_axes = s.raster_axes
    if raster_axes is not None:
        renames = {old.name: new.name for old, new in zip(s.axes, new_axes, strict=True)}
        raster_axes = (renames.get(raster_axes[0], raster_axes[0]), renames.get(raster_axes[1], raster_axes[1]))
    return Tile(
        axes=new_axes,
        body=Body(tuple(_rewrite(c, rename, sigma, axis_fn) for c in s.body)),
        block_threads=s.block_threads,
        aux_threads=s.aux_threads,
        raster_axes=raster_axes,
        raster_group=s.raster_group,
        raster_orient=s.raster_orient,
    )


@_rewrite.register
def _(s: Smem, rename, sigma, axis_fn):
    return s


@_rewrite.register
def _(s: Sync, rename, sigma, axis_fn):
    return s


@_rewrite.register
def _(s: CpAsyncCommit, rename, sigma, axis_fn):
    return s


@_rewrite.register
def _(s: CpAsyncWait, rename, sigma, axis_fn):
    return s


@_rewrite.register
def _(s: CpAsyncCopy, rename, sigma, axis_fn):
    return CpAsyncCopy(
        smem=s.smem,
        smem_index=tuple(sigma.apply(e) for e in s.smem_index),
        src=s.src,
        src_index=tuple(sigma.apply(e) for e in s.src_index),
        nbytes=s.nbytes,
        swizzle=s.swizzle,
    )


@_rewrite.register
def _(s: TmaDescriptor, rename, sigma, axis_fn):
    return s


@_rewrite.register
def _(s: TmaLoad, rename, sigma, axis_fn):
    return TmaLoad(
        smem=s.smem,
        smem_index=tuple(sigma.apply(e) for e in s.smem_index),
        desc=s.desc,
        coords=tuple(sigma.apply(e) for e in s.coords),
        mbar=s.mbar,
        mbar_slot=sigma.apply(s.mbar_slot) if s.mbar_slot is not None else None,
    )


@_rewrite.register
def _(s: MbarrierInit, rename, sigma, axis_fn):
    return MbarrierInit(
        mbar=s.mbar,
        count=s.count,
        slot=sigma.apply(s.slot) if s.slot is not None else None,
    )


@_rewrite.register
def _(s: MbarrierArriveExpectTx, rename, sigma, axis_fn):
    return MbarrierArriveExpectTx(
        mbar=s.mbar,
        bytes_=s.bytes_,
        slot=sigma.apply(s.slot) if s.slot is not None else None,
    )


@_rewrite.register
def _(s: MbarrierArrive, rename, sigma, axis_fn):
    return MbarrierArrive(
        mbar=s.mbar,
        slot=sigma.apply(s.slot) if s.slot is not None else None,
    )


@_rewrite.register
def _(s: MbarrierWait, rename, sigma, axis_fn):
    return MbarrierWait(
        mbar=s.mbar,
        phase=sigma.apply(s.phase),
        slot=sigma.apply(s.slot) if s.slot is not None else None,
    )


@_rewrite.register
def _(s: SetMaxNReg, rename, sigma, axis_fn):
    return s


@_rewrite.register
def _(s: TreeHalve, rename, sigma, axis_fn):
    # The carried state, the second-operand state, and the combine_states program
    # all reference SSA names — thread them through ``rename`` so the canonicalizer
    # can renumber. ``bufs`` are smem slab names (absent from the rename map → no-op)
    # and ``tid_var`` is a loop var, left untouched.
    return TreeHalve(
        bufs=tuple(rename(b) for b in s.bufs),
        state=tuple(rename(n) for n in s.state),
        state_b=tuple(rename(n) for n in s.state_b),
        combine_states=tuple(_rewrite(a, rename, sigma, axis_fn) for a in s.combine_states),
        length=s.length,
        tid_var=s.tid_var,
        dtype=s.dtype,
        barrier_id=s.barrier_id,
        barrier_count=s.barrier_count,
    )


@_rewrite.register
def _(s: WarpShuffle, rename, sigma, axis_fn):
    return WarpShuffle(
        state=tuple(rename(n) for n in s.state),
        state_b=tuple(rename(n) for n in s.state_b),
        combine_states=tuple(_rewrite(a, rename, sigma, axis_fn) for a in s.combine_states),
        length=s.length,
        dtype=s.dtype,
    )


# --- MMA fragment rewrites -------------------------------------------------
#
# Per-cell replication in ``010_split_register_axes`` calls ``s.rewrite(...)``
# on every body Stmt; the dedup pass and ``100_materialize_tile`` do too.
# Each Mma* Stmt routes its SSA fragment names through ``rename`` and its
# Expr-typed offset fields through ``sigma.apply``.


# --- mma.sync (s16816) register-array rewrites -----------------------------


@_rewrite.register
def _(s: RegFragment, rename, sigma, axis_fn):
    return RegFragment(name=rename(s.name), role=s.role, shape=s.shape, dtype=s.dtype, count=s.count)


@_rewrite.register
def _(s: LdmatrixLoad, rename, sigma, axis_fn):
    return LdmatrixLoad(
        frag=rename(s.frag),
        src_buffer=s.src_buffer,
        src_index=tuple(sigma.apply(e) for e in s.src_index),
        role=s.role,
        ldm=s.ldm,
        swizzle=s.swizzle,
        staged=s.staged,
        gmem_guard=None if s.gmem_guard is None else (sigma.apply(s.gmem_guard[0]), sigma.apply(s.gmem_guard[1])),
        k_zero=None if s.k_zero is None else (sigma.apply(s.k_zero[0]), sigma.apply(s.k_zero[1])),
        b_trans=s.b_trans,
        pair_frag=None if s.pair_frag is None else rename(s.pair_frag),
        byte_slab=s.byte_slab,
    )


@_rewrite.register
def _(s: MmaSyncPtx, rename, sigma, axis_fn):
    return MmaSyncPtx(
        c_frag=rename(s.c_frag), a_frag=rename(s.a_frag), b_frag=rename(s.b_frag), shape=s.shape, ab_dtype=s.ab_dtype, c_dtype=s.c_dtype
    )


@_rewrite.register
def _(s: FragmentPromote, rename, sigma, axis_fn):
    return FragmentPromote(dst=rename(s.dst), src=rename(s.src))


@_rewrite.register
def _(s: RegStore, rename, sigma, axis_fn):
    # The epilogue's chain SSA names are render-local (scoped per element
    # inside the store's block), so only the load index Exprs σ-substitute —
    # that threads the per-cell replication offsets through, exactly like
    # ``dst_index``.
    epilogue = s.epilogue
    if epilogue is not None:
        epilogue = RegEpilogue(
            acc=epilogue.acc,
            loads=tuple(
                EpilogueLoad(name=ld.name, buffer=ld.buffer, index=tuple(sigma.apply(e) for e in ld.index), roles=ld.roles)
                for ld in epilogue.loads
            ),
            ops=epilogue.ops,
            result=epilogue.result,
            # Select predicates carry ``__M__`` / ``__N__`` placeholders (not
            # real partition vars), so they're cell-invariant — pass through; the
            # per-cell M/N offset reaches them via ``dst_index`` at render.
            selects=epilogue.selects,
            # The extra channel accumulators rename with the store's own fragment (they are
            # per-cell C-fragment names too) — dropping them here left the multi-channel combine
            # (the gemma GeGLU ``acc2``) unbound at render.
            extra_accs=tuple((a, rename(fr)) for a, fr in epilogue.extra_accs),
        )

    # Guard base/bound Exprs σ-substitute like ``dst_index`` (the per-cell
    # replicator's offsets must reach the boundary predicate; the bound is a
    # Literal or free symbolic Var, which σ passes through).
    def _sub_guard(g):  # noqa: ANN001, ANN202 — tuple[Expr, Expr] | None
        return None if g is None else (sigma.apply(g[0]), sigma.apply(g[1]))

    return RegStore(
        dst_buffer=s.dst_buffer,
        dst_index=tuple(sigma.apply(e) for e in s.dst_index),
        frag=rename(s.frag),
        shape=s.shape,
        ldm=s.ldm,
        epilogue=epilogue,
        m_guard=_sub_guard(s.m_guard),
        n_guard=_sub_guard(s.n_guard),
        # ``atomic`` MUST thread through: dropping it here silently degraded a rewritten (e.g.
        # loopify-rolled) atomic split-K store to racing plain assigns — the partitions then
        # clobber instead of accumulate, a numerically-wrong kernel with no loud failure.
        atomic=s.atomic,
    )


# --- flash fragment-softmax rewrites ---------------------------------------
#
# The online-softmax ops carried INSIDE a streaming-flash ``TileOp`` (so they flow
# through the kernel passes, not only KernelOp→render). Each routes its SSA names
# through ``rename`` (the SSA canonicalizer / per-cell replicator); ``rename`` is
# identity on non-SSA strings, so a literal scale (``"0.25f"``) passes through
# unchanged while an SSA scalar (``"a0"``) renames.


@_rewrite.register
def _(s: Reassign, rename, sigma, axis_fn):
    return Reassign(name=rename(s.name), value=rename(s.value))


@_rewrite.register
def _(s: FragmentApply, rename, sigma, axis_fn):
    args = tuple((rename(a[0]), rename(a[1])) if k == ROW else rename(a) for a, k in zip(s.args, s.kinds, strict=True))
    return FragmentApply(out=rename(s.out), op=s.op, args=args, kinds=s.kinds, in_place=s.in_place, layout=s.layout, post=s.post)


@_rewrite.register
def _(s: FragmentRepack, rename, sigma, axis_fn):
    # Pure register (the flash P→A handoff): route the dest + the two source C fragments through
    # ``rename`` (SSA canonicalizer / per-cell replicator); no index / axis to σ-substitute.
    return FragmentRepack(frag=rename(s.frag), srcs=(rename(s.srcs[0]), rename(s.srcs[1])), ab_dtype=s.ab_dtype)


@_rewrite.register
def _(s: FragmentRowReduce, rename, sigma, axis_fn):
    return FragmentRowReduce(
        top=rename(s.top), bot=rename(s.bot), frags=tuple(rename(f) for f in s.frags), op=s.op, group=s.group, dtype=s.dtype
    )


@_rewrite.register
def _(s: FragmentMask, rename, sigma, axis_fn):
    # ``frag`` is SSA (the score fragment); the tile-origin bases + the predicate σ-substitute so
    # the canonicalizer renames the query / kv axis vars (``qb``→``a1``, ``kv``→``a3``). The
    # reserved ``__frow`` / ``__fcol`` coordinate vars + any free runtime symbol (``seq_len``) are
    # untouched by σ over the local axes.
    return FragmentMask(
        frag=rename(s.frag),
        mask_when=sigma.apply(s.mask_when),
        col_base=sigma.apply(s.col_base),
        row_base=sigma.apply(s.row_base) if s.row_base is not None else None,
        fill=s.fill,
        layout=s.layout,
    )
