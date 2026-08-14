"""CUDA runtime dispatch for ``Graph[CudaOp]``.

Compiles each unique kernel source via NVRTC (through ``cupy.RawKernel``),
allocates a ``cupy.ndarray`` for every buffer in the graph, and walks
compute nodes in topological order launching kernels directly from Python.
No host ``.cu`` is generated — the only codegen that survives is the
per-kernel ``__global__`` function itself, emitted by ``ir/cuda/emit.py``.

Buffer roles come from the graph: ``graph.inputs`` → input,
``ConstantOp`` → constant, ``graph.outputs`` → output, everything else →
scratch. Launch order is ``graph.topological_order()``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os as _os
import pickle
import signal as _signal
import sys as _sys
import time as _time_module
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from emmy.compiler.backend import BenchmarkResult, LaunchTime, RunResult
from emmy.compiler.backend.cuda import _tma, nvcc
from emmy.compiler.backend.cuda._planner import compute_live_intervals, plan_offsets
from emmy.compiler.backend.cuda.dtype import cupy_dtype
from emmy.compiler.backend.plan import BufferSpec as _Buffer
from emmy.compiler.backend.plan import ExecutionPlan, KernelSpec, plan_from_graph
from emmy.compiler.backend.plan import LaunchSpec as _Launch
from emmy.compiler.graph import Graph

if TYPE_CHECKING:
    import cupy as cp

logger = logging.getLogger(__name__)

# Mirror of ``ir.kernel.render.STATIC_SMEM_CAP`` — kept here to avoid
# pulling the renderer into the runtime path.
_STATIC_SMEM_CAP = 48 * 1024


def _ensure_dynamic_smem_attr(kernel: cp.RawKernel, smem_bytes: int) -> None:
    """Opt this kernel into the device's max dynamic-smem allowance.

    Required when ``smem_bytes`` exceeds the 48 KB static cap. cupy's
    ``RawKernel.max_dynamic_shared_size_bytes`` setter calls
    ``cuFuncSetAttribute(MaxDynamicSharedMemorySize)``; the driver
    clamps to the device's per-block dynamic max (e.g. ~99 KB on
    sm_120). Already-set kernels are skipped.
    """
    if kernel.max_dynamic_shared_size_bytes >= smem_bytes:
        return
    kernel.max_dynamic_shared_size_bytes = smem_bytes


# ---------------------------------------------------------------------------
# Buffer / launch classification
# ---------------------------------------------------------------------------


def _resolved_constants(compiled: _Compiled, sym_values: dict[str, int]) -> dict[str, float]:
    """The constant-value map for this run: the static constants plus each runtime
    ``context_value`` constant evaluated at ``sym_values`` (``float(seq_len)``)."""
    if not compiled.runtime_constants:
        return compiled.constants
    return {**compiled.constants, **{nid: float(expr.eval(sym_values)) for nid, expr in compiled.runtime_constants.items()}}


# ---------------------------------------------------------------------------
# Compiled program: RawKernels + buffer plan + launch list
# ---------------------------------------------------------------------------


@dataclass
class _Compiled:
    bufs: list[_Buffer]
    buf_by_name: dict[str, _Buffer]
    constants: dict[str, float]
    kernels: dict[str, cp.RawKernel]  # kernel_name → RawKernel
    launches: list[_Launch]
    # Symbolic axis name → (input_buf_name, dim_index). Resolved from input
    # array shapes at run-time; empty when the graph has no symbolic dims.
    symbolic_bindings: dict[str, tuple[str, int]] = field(default_factory=dict)
    # Per-symbolic-name buffers whose shape carries that name — used to
    # reshape ``input_data`` to the actual runtime shape before upload.
    symbolic_buf_shape: dict[str, tuple] = field(default_factory=dict)
    # Symbolic axis name → its ``Dim`` hint (default expected size). Used as a
    # fallback concrete value when no ``input_data`` is supplied (the autotuner
    # benches a symbolic graph at the hint size).
    symbolic_hints: dict[str, int] = field(default_factory=dict)
    # Symbolic axis name → its capacity CAP. A capacity-capped kernel (the
    # smem-staged fused symbolic-K SDPA P@V)
    # bakes its smem slab at the ``Dim`` hint and is only correct for runtime
    # extents ≤ that cap, so the launch resolver hard-errors when the supplied
    # input shape exceeds it (rather than reading/writing past the baked slab).
    # Empty for every kernel set that tiles the symbolic extent with a ceil-div
    # grid (those handle any runtime size); populated only by the capped cut.
    symbolic_caps: dict[str, int] = field(default_factory=dict)
    # ConstantOp nid → ``Expr`` (over symbolic-dim names) whose runtime value fills the
    # constant (a dynamic mean's divisor = the runtime reduce-axis size). Resolved per run
    # via :func:`_resolved_constants`.
    runtime_constants: dict = field(default_factory=dict)


def _load_kernel(name: str, spec: KernelSpec):
    """Obtain one launchable kernel from its :class:`KernelSpec`. A ``binary_key`` (the pack
    path) loads the content-addressed cubin straight from the cache; otherwise the ``source``
    compiles through the same cache (``nvcc.load_function``). A key whose cubin has been
    evicted falls back to the source when present, and errors otherwise — the pack loader
    pre-checks cubin existence, so hitting this means the cache was cleared mid-boot."""
    if spec.binary_key is not None:
        path = nvcc.cubin_cache_dir() / f"{spec.binary_key}.cubin"
        if path.exists():
            return nvcc.load_cubin_function(path, name)
        if spec.source is None:
            raise RuntimeError(
                f"kernel {name!r}: cubin {spec.binary_key} is gone from the cache and the plan carries no source — "
                "regenerate the pack or boot without it (full compile)"
            )
    if spec.source is None:
        raise RuntimeError(f"kernel {name!r}: plan carries neither a source nor a cached cubin")
    # ``nvcc.load_function`` returns a cupy ``Function`` — launch-callable and
    # smem-attr settable, compiled via offline nvcc into the content-addressed cache.
    return nvcc.load_function(spec.source, name, _nvrtc_options(uses_tma=spec.uses_tma), uses_tma=spec.uses_tma)


def _load_plan(plan: ExecutionPlan) -> _Compiled:
    """Materialize the runtime object from a plan: load every kernel (cubin-by-key or
    source-via-cache), and adopt the plan's pure-data fields as-is."""
    kernels: dict[str, object] = {name: _load_kernel(name, spec) for name, spec in plan.kernels.items()}
    return _Compiled(
        bufs=list(plan.buffers),
        buf_by_name={b.name: b for b in plan.buffers},
        constants=dict(plan.constants),
        kernels=kernels,
        launches=list(plan.launches),
        symbolic_bindings=dict(plan.symbolic_bindings),
        symbolic_hints=dict(plan.symbolic_hints),
        symbolic_caps=dict(plan.symbolic_caps),
        runtime_constants=dict(plan.runtime_constants),
    )


def _nvrtc_options(*, uses_tma: bool) -> tuple[str, ...]:
    """NVRTC compile options. TMA-using kernels need ``sm_<major><minor>a``
    (the ``a`` arch unlocks ``cp.async.bulk.tensor`` PTX). Non-TMA
    kernels keep the cupy default (capability inferred at runtime)."""
    base = ("--use_fast_math",)
    if not uses_tma:
        return base
    from emmy.compiler.target import compute_capability  # noqa: PLC0415

    major, minor = compute_capability()
    return (*base, f"--gpu-architecture=sm_{major}{minor}a")


# ---------------------------------------------------------------------------
# Buffer materialization
# ---------------------------------------------------------------------------


def _materialize(buf: _Buffer, shape: tuple[int, ...], src: np.ndarray | cp.ndarray | None, constants: dict[str, float]) -> cp.ndarray:
    """Build one device array for ``buf`` at ``shape`` — the single fill
    policy shared by :func:`_allocate` and :meth:`CompiledProgram.rebind`.

    ``src`` may already be a **device** (cupy) array — a constant uploaded once and
    shared across programs (the serving runner's symbolic + decode-bucket twins bind
    the same weights). It is used as-is, no copy, unless the dtype disagrees."""
    import cupy as cp

    cp_dtype = cupy_dtype(buf.dtype)
    np_dtype = buf.dtype.np
    if src is not None:
        if isinstance(src, cp.ndarray):
            if src.dtype != np.dtype(np_dtype):
                src = src.astype(np_dtype)
            return src.reshape(shape) if tuple(src.shape) != tuple(shape) else src
        return cp.asarray(np.ascontiguousarray(src, dtype=np_dtype).reshape(shape))
    if buf.role == "constant" and buf.name in constants:
        v = float(constants[buf.name])
        if getattr(buf.dtype, "name", buf.dtype) == "bf16":
            # bf16 buffers ride as uint16 BITS (``BF16.np``) — casting the float would zero it;
            # encode the value to bf16 bits (round-to-nearest-even on the dropped mantissa half).
            bits = int(np.float32(v).view(np.uint32))
            return cp.full(shape, np.uint16((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16), dtype=cp_dtype)
        return cp.full(shape, v, dtype=cp_dtype)
    if buf.role == "input":
        # Pseudo-random fill for un-supplied inputs (matches old generated program).
        n = 1
        for d in shape:
            n *= int(d)
        # Build the index ramp in int64, not ``np_dtype``: a float16 buffer
        # past 65504 elements would overflow to ``inf`` (then ``inf % 101``
        # → ``nan``). Compute in fp32 and cast the final values — always in
        # ``[-0.5, 0.5]``, so fp16-safe.
        idx = np.arange(n, dtype=np.int64)
        vals = (0.01 * ((idx.astype(np.float32) * 7 + 13) % 101 - 50)).astype(np_dtype)
        return cp.asarray(vals.reshape(shape))
    return cp.zeros(shape, dtype=cp_dtype)


@dataclass
class _SlabPlan:
    """A liveness-planned layout of the program's scratch buffers into one
    persistent device slab. ``offsets`` is the byte offset of each scratch
    buffer's view into ``slab``; ``total_bytes`` is the slab size; ``sym_values``
    are the (capacity) dims the layout was planned at — a re-plan is needed when
    they change (``rebind``). ``naive_bytes`` is the sum of the per-scratch sizes
    this layout replaces (for the reduction-factor log). ``slab`` is held for the
    program lifetime so cupy never reclaims it under the views' baked pointers."""

    offsets: dict[str, int]
    total_bytes: int
    sym_values: dict[str, int]
    naive_bytes: int
    slab: cp.ndarray | None = None


class BufferArena:
    """Cross-program device-buffer pool for programs that run **sequentially** (the
    serving runner's per-layer splits). Named grow-only backings: every program built
    with the same arena takes its scratch slab and its input/output buffers as views
    into per-key backings, so N sequential programs hold ~one program's worth of
    activation memory instead of N (the per-layer capacity-buffer artifact: ~350 MB ×
    48 layers for gemma-4-12B). Growth allocates a fresh backing; programs built on an
    older generation keep their (smaller) views alive, so captured graphs / TMA
    descriptors never dangle. Safe iff programs sharing the arena never run
    concurrently and each program's outputs are consumed before the next program runs
    — the runner's contract. Constants are never pooled (persistent values; weight
    sharing is ``gen_runner._bind_plan_constants``)."""

    def __init__(self) -> None:
        self._backings: dict[str, cp.ndarray] = {}

    def backing(self, key: str, nbytes: int) -> cp.ndarray:
        """A zero-init ``uint8`` backing of at least ``nbytes`` for ``key`` — reused
        while it still fits, reallocated (grow-only) when it doesn't."""
        import cupy as cp  # noqa: PLC0415

        cur = self._backings.get(key)
        if cur is None or cur.nbytes < nbytes:
            cur = cp.zeros(max(nbytes, 1), dtype=cp.uint8)
            self._backings[key] = cur
        return cur


def _scratch_sizes(compiled: _Compiled, sym_values: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    """Byte size + alignment per ``scratch`` buffer at the resolved ``sym_values``."""
    sizes: dict[str, int] = {}
    aligns: dict[str, int] = {}
    for buf in compiled.bufs:
        if buf.role != "scratch":
            continue
        shape = buf.resolve_shape(sym_values) or (1,)
        n = 1
        for d in shape:
            n *= int(d)
        sizes[buf.name] = max(1, n) * buf.dtype.nbytes
        aligns[buf.name] = buf.dtype.nbytes
    return sizes, aligns


def _plan_slab(compiled: _Compiled, sym_values: dict[str, int]) -> _SlabPlan:
    """Liveness-plan the scratch buffers into one slab layout (no allocation)."""
    sizes, aligns = _scratch_sizes(compiled, sym_values)
    intervals = compute_live_intervals(list(sizes), compiled.launches)
    offsets, total = plan_offsets(intervals, sizes, aligns)
    return _SlabPlan(offsets=offsets, total_bytes=total, sym_values=dict(sym_values), naive_bytes=sum(sizes.values()))


def _alloc_slab(
    compiled: _Compiled, sym_values: dict[str, int], arena: BufferArena | None = None
) -> tuple[dict[str, cp.ndarray], _SlabPlan]:
    """Plan + allocate the scratch slab and return each scratch buffer as a typed
    view into it. One zero-init ``uint8`` allocation (or a view into the shared
    ``arena`` backing); each view's device pointer (``slab.data + offset``) is stable
    for the program lifetime (the slab is pinned on the returned plan), so captured
    graphs / TMA descriptors that bake the pointer stay valid across replays."""
    import cupy as cp  # noqa: PLC0415

    plan = _plan_slab(compiled, sym_values)
    if arena is not None:
        plan.slab = arena.backing("scratch-slab", plan.total_bytes)
    else:
        plan.slab = cp.zeros(plan.total_bytes or 1, dtype=cp.uint8)
    views: dict[str, cp.ndarray] = {}
    for buf in compiled.bufs:
        if buf.role != "scratch":
            continue
        shape = buf.resolve_shape(sym_values) or (1,)
        views[buf.name] = cp.ndarray(shape, dtype=cupy_dtype(buf.dtype), memptr=plan.slab.data + plan.offsets[buf.name])
    return views, plan


def _arena_view(arena: BufferArena, buf: _Buffer, shape: tuple[int, ...], src, constants: dict[str, float]) -> cp.ndarray:
    """An input/output buffer as a view into the arena's per-``(role, name)`` backing,
    filled under the same policy as :func:`_materialize`. Keyed by role AND name so an
    input and an output that happen to share a tensor name never alias within one
    program (kernels read inputs while writing outputs); across programs the aliasing
    is the point."""
    import cupy as cp  # noqa: PLC0415

    filled = _materialize(buf, shape, src, constants)
    backing = arena.backing(f"{buf.role}:{buf.name}", filled.nbytes)
    view = cp.ndarray(shape, dtype=filled.dtype, memptr=backing.data)
    view[...] = filled
    return view


def _allocate(
    compiled: _Compiled, input_data: dict[str, np.ndarray] | None, arena: BufferArena | None = None
) -> tuple[dict[str, cp.ndarray], _SlabPlan]:
    """Materialize every buffer. ``scratch`` buffers become typed views into one
    liveness-planned persistent slab (dead intervals share memory);
    input/constant/output buffers stay standalone (persistent across the call) —
    unless an ``arena`` is supplied, in which case input/output buffers (and the
    slab) become views into its cross-program backings; constants stay standalone."""
    input_data = input_data or {}
    sym_values = _resolve_symbolic(compiled, input_data)
    constants = _resolved_constants(compiled, sym_values)
    arrays: dict[str, cp.ndarray] = {}
    # Saturating casts here are intended, not bugs: e.g. an SDPA mask-fill
    # constant (``-1e9``) is meant to become ``-inf`` in fp16 (masked → 0 after
    # softmax). Ignore the over/invalid warnings so genuine output stays clean.
    with np.errstate(over="ignore", invalid="ignore"):
        for buf in compiled.bufs:
            if buf.role == "scratch":
                continue  # placed into the slab below
            shape = buf.resolve_shape(sym_values) or (1,)
            if arena is not None and buf.role in ("input", "output"):
                arrays[buf.name] = _arena_view(arena, buf, shape, input_data.get(buf.name), constants)
            else:
                arrays[buf.name] = _materialize(buf, shape, input_data.get(buf.name), constants)
    # ``scratch`` buffers become views into one zero-init slab. The build-time
    # zero preserves the contract scratch had under ``cp.zeros``; per-launch
    # ``zero_outputs`` re-zeros atomic-reduction outputs, and every other kernel
    # fully overwrites its output (lowering contract), so a reused slot's stale
    # contents are never read — which is also why a reused (stale) arena backing
    # is as safe as a reused slab slot.
    scratch_views, plan = _alloc_slab(compiled, sym_values, arena)
    arrays.update(scratch_views)
    return arrays, plan


def _resolve_symbolic(compiled: _Compiled, input_data: dict[str, np.ndarray]) -> dict[str, int]:
    """Bind every symbolic axis name to a concrete ``int``. Reads the runtime
    value from the supplied input array shape (``compiled.symbolic_bindings``
    says which input + dim each name reads from). When no array is supplied for
    that input — the autotuner benches without real inputs — falls back to the
    ``Dim`` hint so the graph runs at its expected (tuned) size."""
    env: dict[str, int] = {}
    for name, (buf, dim_idx) in compiled.symbolic_bindings.items():
        arr = input_data.get(buf)
        if arr is not None:
            env[name] = int(arr.shape[dim_idx])
            cap = compiled.symbolic_caps.get(name)
            if cap is not None and env[name] > cap:
                raise ValueError(
                    f"symbolic dim {name!r} = {env[name]} exceeds the capacity-capped kernel's hint ({cap}); "
                    f"this build bakes its smem slab at {cap} and cannot run a larger extent — "
                    f"re-trace with a larger --seq-len hint or use the ceil-div (uncapped) lowering"
                )
        elif name in compiled.symbolic_hints:
            env[name] = compiled.symbolic_hints[name]
        else:
            raise ValueError(
                f"symbolic dim {name!r} reads from input {buf!r}.shape[{dim_idx}] but no array was supplied and the dim carries no hint"
            )
    return env


def _launch(
    launch: _Launch,
    compiled: _Compiled,
    arrays: dict[str, cp.ndarray],
    desc_args: dict[str, cp.ndarray] | None = None,
    sym_values: dict[str, int] | None = None,
) -> None:
    from emmy.compiler.ir.cuda.ir import resolve_dim  # noqa: PLC0415

    for zname in launch.zero_outputs:
        # memset, not ``.fill(0)``: fill launches a cupy elementwise kernel (~the cost of the
        # finalize kernel the atomic split saves), while memset_async records as a cheap MEMSET
        # node under CUDA-graph capture. All-zero bytes are 0.0 in every buffer dtype.
        buf = arrays[zname]
        buf.data.memset_async(0, buf.nbytes)
    kernel = compiled.kernels[launch.kernel_name]
    desc_args = desc_args or {}
    sym_values = sym_values or {}
    if launch.indirect_args:
        # Indirect operands: the marked arg's position expands in place to (table, sel, slot)
        # — the kernel resolves ``table[sel[slot]]`` in its body preamble. Table/selector are
        # device arrays the caller binds into ``arrays`` under the spec's names; the slot is a
        # plain ``int`` arg (same packing as the runtime-arg tail).
        indirect = {a: (t, s, sl) for a, t, s, sl in launch.indirect_args}
        parts: list = []
        for name in launch.arg_names:
            entry = indirect.get(name)
            if entry is not None:
                parts.extend((arrays[entry[0]], arrays[entry[1]], entry[2]))
            else:
                parts.append(desc_args.get(name) if name in desc_args else arrays[name])
        args = tuple(parts)
    else:
        args = tuple(desc_args.get(name) if name in desc_args else arrays[name] for name in launch.arg_names)
    # Symbolic axes appear as ``int`` kernel params after buffers + TMA
    # descriptors — append their resolved values to the arg pack.
    if launch.runtime_args:
        args = (*args, *(sym_values[name] for name in launch.runtime_args))
    grid = tuple(resolve_dim(spec, sym_values) for spec in launch.grid)
    block = tuple(resolve_dim(spec, sym_values) for spec in launch.block)
    # Kernels whose Smem footprint exceeds the 48 KB static cap declare
    # an ``extern __shared__`` pool; the launch supplies the byte size
    # via ``shared_mem=`` and (for footprints above 48 KB) opts into the
    # device's larger dynamic-smem allowance via ``cudaFuncSetAttribute``.
    smem_bytes = launch.smem_bytes
    if smem_bytes > _STATIC_SMEM_CAP:
        _ensure_dynamic_smem_attr(kernel, smem_bytes)
        kernel(grid, block, args, shared_mem=smem_bytes)
    else:
        kernel(grid, block, args, shared_mem=0)


def _collapse_inert_dims(arr_shape: tuple[int, ...], box_extents: tuple[int, ...]) -> tuple[int, ...]:
    """Reconstruct the materializer's gap-singleton drop from runtime info.

    The materializer drops gap source dims that are extent-1 singletons
    with literal-0 origin coords (a literal-0 origin can only arise for
    a singleton arr dim, since otherwise IR construction would have
    emitted a ``Var`` or expression). At runtime we don't carry that
    decision explicitly — instead we walk ``arr_shape`` and
    ``box_extents`` innermost-first and drop any arr dim of extent 1
    that lines up with a box dim of extent > 1. Leading singletons
    pair with their (kept) box==1 entry and stay; gap singletons fall
    out exactly where the materializer dropped them.

    The materializer's swizzle-split path may emit a rank-(N+1) box on
    a rank-N source by splitting an inner dim. Reinterpret the array's
    last dim as the matching split before walking, so the rank-match
    check below succeeds and ``encode_tiled`` sees a consistent view."""
    arr_rev = list(reversed(arr_shape))
    box_rev = list(reversed(box_extents))
    if len(box_rev) == len(arr_rev) + 1 and arr_rev and box_rev[0] != 0 and arr_rev[0] % box_rev[0] == 0:
        arr_rev = [box_rev[0], arr_rev[0] // box_rev[0], *arr_rev[1:]]
    # Drop exactly ``arr_rank - box_rank`` inert gap singletons — no more. A
    # *box-carrying* dim can be a runtime-extent-1 (or any extent < its box):
    # a masked dynamic axis (e.g. ``seq_len`` = 1, 31) is legitimately small,
    # and TMA zero-fills the overhang where the box exceeds globalDim. The old
    # "drop every extent-1 aligned with box>1" rule mis-dropped that masked dim
    # whenever its runtime extent hit 1, then failed the rank match (seq_len=1
    # → ``arr=(1, 512)`` vs ``box=(64, 32)``). Shedding only the surplus dims
    # keeps the genuine inner gap-singleton drop (arr_rank > box_rank) intact.
    n_drop = len(arr_rev) - len(box_rev)
    kept: list[int] = []
    bi = 0
    for a in arr_rev:
        if n_drop > 0 and a == 1 and bi < len(box_rev) and box_rev[bi] != 1:
            n_drop -= 1
            continue  # dropped gap singleton
        kept.append(a)
        bi += 1
    if n_drop != 0 or len(kept) != len(box_rev):
        raise ValueError(f"TMA descriptor rank mismatch: arr_shape={arr_shape!r} cannot be collapsed to match box_extents={box_extents!r}")
    return tuple(reversed(kept))


def _prebuild_descriptors(
    compiled: _Compiled,
    arrays: dict[str, cp.ndarray],
    sym_values: dict[str, int] | None = None,
    only_symbolic: bool = False,
) -> dict[int, dict[str, cp.ndarray]]:
    """Encode every TMA ``CUtensorMap`` for ``compiled`` up-front.

    The kernel signature takes ``const CUtensorMap*`` (not a by-value
    ``__grid_constant__`` parameter) because cupy's arg-packing doesn't
    guarantee the 64-byte alignment required for by-value descriptors.
    Placing the descriptor in device memory and passing a pointer
    sidesteps the alignment concern — the TMA load PTX dereferences
    via a generic 64-bit pointer either way.

    Why eagerly: ``cp.asarray(np.frombuffer(...))`` queues an H2D copy on
    the current stream. Building descriptors lazily inside ``_launch``
    means each fresh kernel's H2D races against in-flight TMA loads from
    *previous* launches sharing the same descriptor allocator slab —
    the next allocation can land on cupy-pool memory the prior kernel's
    cp.async.bulk.tensor is still reading, corrupting the descriptor
    and deadlocking the wait. Pre-building once after ``_allocate``
    removes the race entirely; the returned dict is held alive for the
    whole program lifetime, so cupy never reclaims the slab.

    ``sym_values``: encode each SYMBOLIC-shaped source buffer at its RESOLVED
    shape instead of the array's allocated (capacity) shape. On the serving
    capacity-buffer path the live data is prefix-packed at the resolved shape's
    row-major strides, so a descriptor's global strides must follow the resolved
    extents — a capacity-baked stride reads the right rows only for the leading
    index 0, which is why batch>1 miscomputed through every TMA-staged kernel
    while batch-1 serving never noticed. ``only_symbolic=True`` returns just the
    per-sym overlay entries (static-src descriptors are excluded — the prebuilt
    ones stay valid)."""
    import cupy as cp

    out: dict[int, dict[str, cp.ndarray]] = {}
    for li, launch in enumerate(compiled.launches):
        if not launch.tma_descriptors:
            continue
        per_launch: dict[str, cp.ndarray] = {}
        for desc in launch.tma_descriptors:
            arr = arrays[desc.src_buf]
            buf = compiled.buf_by_name[desc.src_buf]
            if only_symbolic and not buf.is_symbolic:
                continue
            if sym_values and buf.is_symbolic:
                base_shape = tuple(int(d) for d in (buf.resolve_shape(sym_values) or (1,)))
            else:
                base_shape = tuple(int(d) for d in arr.shape)
            src_shape = _collapse_inert_dims(base_shape, desc.box_extents)
            desc_bytes = _tma.encode_tiled(
                global_address=int(arr.data.ptr),
                src_shape=src_shape,
                box_extents=desc.box_extents,
                elem_size=int(arr.itemsize),
                swizzle=desc.swizzle,
            )
            per_launch[desc.name] = cp.asarray(np.frombuffer(desc_bytes, dtype=np.uint64))
        if per_launch:
            out[li] = per_launch
    if out:
        cp.cuda.runtime.deviceSynchronize()
    return out


# ---------------------------------------------------------------------------
# Iter-loop policy constants + per-event watchdog
# ---------------------------------------------------------------------------


# Per-launch wall-clock cap. Any single kernel launch exceeding this is
# considered "broken" — too many threads, infinite loop, hung GPU — and
# we bail out via ``HungKernelError`` so the autotune sweep doesn't stall
# on one bad variant.
# 2000, not 1000 (2026-07-22): the long-standing gemma-4 post4096-global "bench hang" was a
# WATCHDOG artifact, not a kernel deadlock — under a 1 s deadline the bench_fails 5/5 at the first
# iteration after the warmup-extension re-calibration (its wait exceeds 1 s), while at ANY deadline
# >= 2 s the same program benches clean 9/9 with no event wait ever reaching even the 0.2 s warning
# threshold (measured at 2/4/15/600 s). The deadline-correlation below the driver line is
# unexplained (suspected interaction between the 1 ms cudaEventQuery poll loop's abort path and
# in-flight graph-exec work on this 9-kernel / 96 KB-smem program); empirically 2 s is past the
# cliff, and a real hung kernel is still evicted in 2 s. ``EMMY_KERNEL_TIMEOUT_MS`` overrides.
_KERNEL_TIMEOUT_MS = float(_os.environ.get("EMMY_KERNEL_TIMEOUT_MS", "2000"))

# First-launch grace multiplier: a program's FIRST uncaptured iteration may stall well past the
# steady-state watchdog without any kernel being hung — lazy module loading (CUDA_MODULE_LOADING=
# LAZY uploads each kernel's SASS on first launch), the smem-carveout reconfig for a 96 KB dynamic-
# smem kernel, and allocator first-touch all land there. A genuinely hung kernel is still caught on
# iter 0, just ``_FIRST_ITER_GRACE`` x later.
_FIRST_ITER_GRACE = 30.0


class HungKernelError(RuntimeError):
    """A kernel launch did not complete within the per-launch watchdog window.

    Distinct from a plain ``RuntimeError`` (a slow-but-completing variant) because a hung
    kernel stays **resident on the device** after we give up polling for it — the in-process
    bench has no way to evict it (only the SIGKILL-isolated tuning worker can reset the
    device). A caller that runs further benches on the same device after catching this must
    treat the device as poisoned and stop, or the next blocking ``synchronize()`` (e.g. the
    torch peer-bench) will block behind the still-running kernel. Subclasses ``RuntimeError``
    so existing ``except RuntimeError`` handlers (the autotune sweep) keep marking the variant
    ``bench_fail`` unchanged."""


class GraphCaptureError(RuntimeError):
    """CUDA graph capture of the bench launch loop failed.

    Raised by :meth:`CompiledProgram.capture_launch_graphs` after draining any
    in-progress capture state, so the stream is clean and the caller can simply
    retry the bench uncaptured. Only the per-kernel reproducer bench enables
    capture (the autotune sweep never does), so this can't be misclassified as
    a ``bench_fail`` there."""


_AUTO_BUDGET_MS = 100.0
# Iter-count cap on ``num_iters="auto"``. Combined with the GPU-time
# target above: whichever fires first wins. The cap is the binding
# constraint for fast kernels (sub-ms / launch, where 100 ms target
# would otherwise mean 100s of iters and the corresponding atomic /
# clock-state pressure on heavy-fanout K-split kernels); the GPU-time
# target is the binding constraint for slow kernels (>= 1 ms / launch,
# where 100 iters would over-measure relative to confidence needs).
_AUTO_MAX_ITERS = 100
# Target per-kernel-position timing window. Sub-millisecond kernels are
# dominated by per-iter Python/cupy framing overhead (~100 µs); we
# amortize it by repeating each launch ``batch_size`` times inside one
# CUDA event window, where ``batch_size = ceil(_BATCH_TARGET_MS /
# per_launch_ms)``. Calibrated after warmup from the last-warmup iter's
# per-launch timings, then held fixed during measurement.
_BATCH_TARGET_MS = 1.0
# Minimum total GPU time the warmup window should cover. sm_120 (and
# other consumer GPUs with auto-boost) take several ms to ramp clocks
# from idle. For tiny kernels the requested ``warmup`` iters may sum
# to << 1 ms — the first measured iters then see mid-ramp clocks and
# the median jitters across runs. After the post-warmup batch-size
# calibration we extend ``warmup`` so total warmup GPU time clears
# this threshold.
_WARMUP_TARGET_MS = 10.0


def _wait_for_event(event, timeout_ms: float, label: str) -> None:
    """Block until ``event`` completes, polling rather than calling the
    blocking ``synchronize()``. Raises ``HungKernelError`` on timeout —
    necessary because once a CUDA kernel is hung, ``synchronize()``
    blocks indefinitely (the driver only resets after minutes), which
    stalls the autotune sweep on a single bad variant.

    Caveat: a hung kernel is still queued on the device after we give
    up here, so the *next* launch queues behind it and may also be
    slow. That's still vastly better than blocking forever in this
    one bench."""
    import time as _time

    import cupy as _cp  # noqa: PLC0415

    start = _time.perf_counter()
    deadline = start + timeout_ms / 1000.0
    next_warn = start + 0.2  # surface kernels stuck >200ms even if they eventually finish
    warned = False
    while not event.done:
        now = _time.perf_counter()
        if now > deadline:
            raise HungKernelError(f"kernel {label!r} did not complete within {timeout_ms:.0f} ms — variant marked bench_fail")
        if now > next_warn:
            logger.warning("[cuda] kernel %r still pending after %.2fs (timeout %.1fs)", label, now - start, timeout_ms / 1000.0)
            warned = True
            next_warn = now + 1.0  # subsequent log every 1s while still stuck
        _time.sleep(0.001)
    elapsed = _time.perf_counter() - start
    if warned:
        logger.warning("[cuda] kernel %r completed after %.2fs of waiting", label, elapsed)
    _cp.cuda.runtime.eventSynchronize(event.ptr)  # cheap post-completion sync


# ---------------------------------------------------------------------------
# CompiledProgram: post-compile GPU state + uniform iter loop
# ---------------------------------------------------------------------------


@dataclass
class CompiledProgram:
    """Post-compile GPU state for one graph: kernels, allocated buffers,
    pre-built TMA descriptors.

    Constructed inside ``gpu_lock()`` by the public entry points
    (:func:`run_program`, :func:`run_program_debug`,
    :func:`benchmark_program`) so every CUDA-touching phase — NVRTC
    compile, cupy alloc, descriptor H2D, kernel-launch loop, output
    ``.get()`` — runs with the lock held. Peer xdist workers never
    interleave with us on the device, which previously surfaced as
    small numerical divergence in multi-kernel attention tests when
    the suite ran in parallel.

    All three entry points walk launches through the same
    :meth:`iter_once`. What differs between them — single pass vs
    warmup+measure vs snapshot-every-launch — collapses to which
    optional callbacks they pass."""

    compiled: _Compiled
    arrays: dict[str, cp.ndarray]
    descs: dict[int, dict[str, cp.ndarray]]
    # Per-symbolic-axis runtime ``int`` resolved at ``build`` time from the
    # supplied input shapes — fed straight to ``_launch`` for grid /
    # block resolution and the runtime-arg tail. Empty for fully-static
    # graphs.
    sym_values: dict[str, int] = field(default_factory=dict)
    # Liveness-planned scratch slab: scratch ``arrays`` are views into
    # ``slab_plan.slab``, pinned here for the program lifetime (``build`` always
    # populates it; the default is only for dataclass construction).
    slab_plan: _SlabPlan | None = None
    # Cross-program :class:`BufferArena` the activation buffers + slab view into
    # (``None`` → standalone allocation, the non-serving default). ``rebind`` re-takes
    # its views from the same arena so sharing survives symbolic re-sizing.
    arena: BufferArena | None = None
    # Per-launch timing events, lazily created on first ``iter_once``
    # and reused across every subsequent call so multi-iter bench loops
    # don't churn the cupy ``Event`` pool (the pre-unification
    # ``benchmark_program`` allocated events once outside the while
    # loop; thrashing them per iter perturbs the tuner's variant
    # ranking — close-latency siblings get reordered run-to-run, which
    # caused ``test_tuned_variant_matches_reference`` to flake ~30%).
    _starts: list = field(default_factory=list, repr=False)
    _stops: list = field(default_factory=list, repr=False)
    # Number of completed ``iter_once`` calls — iter 0 gets the ``_FIRST_ITER_GRACE`` watchdog
    # multiplier (first-launch lazy-load / carveout stalls are not hangs; see the constant's note).
    _iters_done: int = field(default=0, repr=False)
    # Per-launch CUDA graphs (one per launch position, each containing that
    # launch's whole batch) captured by :meth:`capture_launch_graphs`. When
    # set, ``iter_once`` replays ``_graphs[i]`` with one host call instead of
    # the ``batch_sizes[i]``-long Python launch loop, so the CUDA event window
    # measures dense GPU work rather than per-launch dispatch gaps. ``None``
    # (the default) keeps the plain launch loop — ``run_program`` /
    # ``run_program_debug`` and the autotune sweep never capture.
    _graphs: list | None = field(default=None, repr=False)
    _graph_batch_sizes: list[int] | None = field(default=None, repr=False)
    # One CUDA graph holding EVERY launch in program order (batch 1 each),
    # captured by :meth:`capture_program_graph` for the whole-program (e2e)
    # timing windows — the emmy analogue of timing a captured torch
    # forward, so the backend-comparison table is like-for-like.
    _e2e_graph: Any | None = field(default=None, repr=False)
    _e2e_start: Any | None = field(default=None, repr=False)
    _e2e_stop: Any | None = field(default=None, repr=False)
    # Whole-program graphs keyed by the resolved symbolic tuple — the serving
    # captured-replay path holds one captured graph PER seq_len over a single
    # capacity-sized buffer set (a graph baked at seq_len S only replays at S,
    # since each kernel's grid + by-value seq_len are frozen by capture). LRU,
    # bounded by ``_graph_cache_max``. The static-shape bench path uses the one
    # ``()`` key, so it's unaffected.
    _graph_cache: dict = field(default_factory=dict, repr=False)
    _graph_cache_max: int = 64
    # Per-sym-key TMA descriptor overlays (symbolic-src descriptors re-encoded at
    # the RESOLVED shape — the capacity-buffer prefix layout's true strides; see
    # :func:`_prebuild_descriptors`). Keyed like ``_graph_cache``; an entry must
    # outlive any captured graph replaying at its key (the graph bakes the desc
    # device pointers), so eviction only drops keys absent from ``_graph_cache``.
    _sym_descs: dict = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls,
        graph: Graph,
        input_data: dict[str, np.ndarray] | None = None,
        *,
        compile_timeout_s: float | None = None,
        arena: BufferArena | None = None,
    ) -> CompiledProgram:
        """Compile ``graph`` and build — ``plan_from_graph`` + :meth:`build_from_plan`; the
        graph is never consulted after the projection (one runtime path whether the plan came
        from a fresh compile or from a stored pack)."""
        return cls.build_from_plan(plan_from_graph(graph), input_data, compile_timeout_s=compile_timeout_s, arena=arena)

    @classmethod
    def build_from_plan(
        cls,
        plan: ExecutionPlan,
        input_data: dict[str, np.ndarray] | None = None,
        *,
        compile_timeout_s: float | None = None,
        arena: BufferArena | None = None,
    ) -> CompiledProgram:
        """Load every kernel (cubin-by-key or source-via-cache), allocate every
        buffer, pre-build TMA descriptors. ``compile_timeout_s`` bounds the
        setup phase at a C-call boundary: if compile + alloc + descriptor work
        overruns, raise ``RuntimeError`` before the caller proceeds to launches
        so no in-flight kernels are left queued. ``arena`` pools the
        activation buffers + scratch slab across sequentially-run
        programs (see :class:`BufferArena`).

        Caller is expected to hold ``gpu_lock()`` around this call and
        every subsequent method on the returned program."""
        t0 = _time_module.monotonic()
        compiled = _load_plan(plan)
        sym_values = _resolve_symbolic(compiled, input_data or {})
        arrays, slab_plan = _allocate(compiled, input_data, arena)
        descs = _prebuild_descriptors(compiled, arrays)
        elapsed = _time_module.monotonic() - t0
        if compile_timeout_s is not None and elapsed > compile_timeout_s:
            raise RuntimeError(f"compile stage exceeded {compile_timeout_s:.1f}s budget ({elapsed:.2f}s) — variant marked bench_fail")
        logger.info(
            "[cuda] CompiledProgram.build: %d launch(es) compile+alloc=%.2fs kernels=[%s]",
            len(compiled.launches),
            elapsed,
            ", ".join(f"{li}:{lc.kernel_name}" for li, lc in enumerate(compiled.launches)),
        )
        if slab_plan.naive_bytes > 0:
            logger.info(
                "[cuda] buffer-reuse: scratch slab=%.2f GB (%.2f GB naive, %.1fx) over %d scratch buf(s)",
                slab_plan.total_bytes / 1e9,
                slab_plan.naive_bytes / 1e9,
                slab_plan.naive_bytes / max(1, slab_plan.total_bytes),
                len(slab_plan.offsets),
            )
        return cls(compiled=compiled, arrays=arrays, descs=descs, sym_values=sym_values, slab_plan=slab_plan, arena=arena)

    def rebind(self, input_data: dict[str, np.ndarray]) -> None:
        """Re-bind ``input_data`` on an already-built program, re-sizing
        symbolic-shaped buffers to the new runtime dims — the serving path,
        where one compiled dynamic-seq_len program runs request after request.

        Supplied buffers are re-uploaded: in place (``arr.set``) when the
        resolved shape is unchanged, re-allocated otherwise. Un-supplied
        buffers whose shape carries a symbolic dim (scratch/outputs sized by
        seq_len) re-materialize at the new shape under the same fill policy
        as ``build``; static-shaped un-supplied buffers — the weights — keep
        their device arrays untouched (no re-upload). When any array was
        re-allocated, TMA descriptors are rebuilt (they embed device pointers
        and shapes) and captured CUDA graphs are dropped (they bake old
        pointers). Caller must hold ``gpu_lock()``."""
        new_sym = _resolve_symbolic(self.compiled, input_data)
        realloc = False
        reuse = self.slab_plan is not None
        with np.errstate(over="ignore", invalid="ignore"):
            for buf in self.compiled.bufs:
                if reuse and buf.role == "scratch":
                    continue  # slab-managed; re-planned below when dims change
                # A runtime ``context_value`` constant (a dynamic mean's divisor) keeps a
                # static (1,) shape but its VALUE tracks the runtime context — refill in place
                # whenever sym_values change (the shape-change check below would skip it).
                if buf.name in self.compiled.runtime_constants:
                    self.arrays[buf.name].fill(float(self.compiled.runtime_constants[buf.name].eval(new_sym)))
                    continue
                src = input_data.get(buf.name)
                if src is None and not buf.is_symbolic:
                    continue
                shape = buf.resolve_shape(new_sym) or (1,)
                arr = self.arrays[buf.name]
                if tuple(arr.shape) != shape:
                    if self.arena is not None and buf.role in ("input", "output"):
                        self.arrays[buf.name] = _arena_view(self.arena, buf, shape, src, self.compiled.constants)
                    else:
                        self.arrays[buf.name] = _materialize(buf, shape, src, self.compiled.constants)
                    realloc = True
                elif src is not None:
                    arr.set(np.ascontiguousarray(src, dtype=buf.dtype.np).reshape(shape))
        # Scratch slab: re-plan + reallocate at the new dims (sizes scale with the
        # symbolic extent). The fresh slab has new pointers, so descs/graphs must
        # be rebuilt/dropped — handled by the shared ``if realloc:`` block below.
        if reuse and new_sym != self.slab_plan.sym_values:
            scratch_views, self.slab_plan = _alloc_slab(self.compiled, new_sym, self.arena)
            self.arrays.update(scratch_views)
            realloc = True
        self.sym_values = new_sym
        if realloc:
            self.descs = _prebuild_descriptors(self.compiled, self.arrays)
            self._graphs = None
            self._graph_batch_sizes = None
            self._e2e_graph = None
            self._graph_cache.clear()
            self._sym_descs.clear()

    def set_sym_values(self, values: dict[str, int]) -> None:
        """Set the host symbolic values that resolve launch grids + by-value
        kernel args, WITHOUT re-allocating buffers — they stay at the build
        (capacity) shape. The serving capture path: buffers sized once at
        capacity, grids + frozen seq_len baked per request via
        :meth:`capture_program_graph`, results sliced to the real shape by
        ``outputs(sym_values=…)``. Errors if any value exceeds the allocated
        buffer capacity (the caller falls back to ``rebind`` above capacity)."""
        merged = {**self.sym_values, **values}
        for buf in self.compiled.bufs:
            want = buf.resolve_shape(merged) or (1,)
            if math.prod(want) > self.arrays[buf.name].size:
                raise ValueError(
                    f"set_sym_values {values}: buffer {buf.name!r} resolves to {want} "
                    f"({math.prod(want)} elems) > capacity {self.arrays[buf.name].size}"
                )
        self.sym_values = merged

    def run_once(self) -> None:
        """Launch every kernel once in program order with no per-launch event
        record/sync/watchdog — the serving hot path (timing semantics live in
        :meth:`iter_once`). The default stream orders the launches; the
        caller's subsequent ``outputs()`` ``.get()`` synchronizes."""
        descs = self._descs_now()
        for i, launch in enumerate(self.compiled.launches):
            _launch(launch, self.compiled, self.arrays, descs.get(i), self.sym_values)

    def _descs_now(self) -> dict[int, dict[str, cp.ndarray]]:
        """The per-launch TMA descriptors matching the CURRENT ``self.sym_values``:
        the prebuilt (allocation-shaped) entries overlaid with per-sym re-encodes of
        every symbolic-src descriptor. On the capacity-buffer serving path the live
        data is prefix-packed at the resolved shape, so symbolic-src descriptors must
        re-encode per seq_len (cached per sym key). A fully-static program (empty
        key) returns the prebuilt dict unchanged."""
        key = self._sym_key()
        if not key:
            return self.descs
        overlay = self._sym_descs.get(key)
        if overlay is None:
            overlay = _prebuild_descriptors(self.compiled, self.arrays, sym_values=self.sym_values, only_symbolic=True)
            # Bound the cache without ever dropping a key whose captured graph is
            # still alive (its replay dereferences these device buffers).
            evictable = [k for k in self._sym_descs if k not in self._graph_cache]
            while evictable and len(self._sym_descs) >= max(self._graph_cache_max, len(self._graph_cache) + 1):
                self._sym_descs.pop(evictable.pop(0))
            self._sym_descs[key] = overlay
        if not overlay:
            return self.descs
        return {li: {**self.descs.get(li, {}), **overlay.get(li, {})} for li in self.descs.keys() | overlay.keys()}

    def capture_launch_graphs(self, batch_sizes: list[int]) -> None:
        """Capture each launch position's batch into one CUDA graph.

        Stream capture is illegal on the legacy default stream, so each batch is
        captured on a temporary side stream; ``iter_once`` then replays the graph
        on the default stream (``Graph.launch`` targets the *current* stream), so
        the existing cupy/torch event interleaving is untouched. The capture
        window holds only ``_launch`` work — output zeroing + kernel launches on
        prebuilt buffers/descriptors — no allocations, no sync, no event records
        (the dynamic-smem attribute is already set by the uncaptured warmup iters
        that always precede capture).

        Safe to call again when batch sizes change (warmup extension re-fires the
        calibration); a no-op when they match the captured ones. Raises
        :class:`GraphCaptureError` on any failure, after draining the capture
        state so the stream isn't left wedged — the caller retries uncaptured."""
        import cupy as cp

        if self._graphs is not None and self._graph_batch_sizes == list(batch_sizes):
            return
        self._graphs = None
        descs = self._descs_now()
        side = cp.cuda.Stream(non_blocking=True)
        graphs = []
        for i, launch in enumerate(self.compiled.launches):
            try:
                with side:
                    side.begin_capture()
                    for _ in range(batch_sizes[i]):
                        _launch(launch, self.compiled, self.arrays, descs.get(i), self.sym_values)
                    graphs.append(side.end_capture())
            except Exception as exc:
                if side.is_capturing():
                    try:
                        with side:
                            side.end_capture()  # drain capture state; discard the partial graph
                    except Exception:  # noqa: BLE001, S110 — already raising the original failure
                        pass
                raise GraphCaptureError(f"capture failed for launch {i} ({launch.kernel_name!r}): {exc}") from exc
        # One throwaway replay per graph absorbs graphExec instantiation /
        # upload cost so the first measured iter is clean. Buffers get
        # clobbered, which is fine: accuracy was checked before benching.
        for g in graphs:
            g.launch()
        cp.cuda.runtime.deviceSynchronize()
        self._graphs = graphs
        self._graph_batch_sizes = list(batch_sizes)

    def capture_program_graph(self) -> None:
        """Capture every launch (batch 1, program order) into ONE CUDA graph at
        the CURRENT ``self.sym_values``, caching it by the resolved symbolic
        tuple and pointing ``self._e2e_graph`` at it.

        Two callers:
        - The bench's :meth:`time_program_window` (static graph → the single
          ``()`` cache key): one event window around N back-to-back replays, the
          same semantics the captured torch closures get.
        - The serving path: one captured graph PER seq_len over a SHARED
          capacity-sized buffer set. A graph baked at seq_len S only replays at S
          (its grids + by-value seq_len are frozen by capture), so the cache is
          keyed by ``self.sym_values`` and bounded LRU (``_graph_cache_max``).
          Set ``self.sym_values`` (via :meth:`set_sym_values`) and upload the
          request's input prefix (via :meth:`upload_prefix`) before calling.

        Cache hit ⇒ no re-capture. Same error contract as
        :meth:`capture_launch_graphs`: raises :class:`GraphCaptureError` after
        draining any partial capture state."""
        import cupy as cp

        key = self._sym_key()
        cached = self._graph_cache.get(key)
        if cached is not None:
            self._graph_cache[key] = self._graph_cache.pop(key)  # LRU bump
            self._e2e_graph = cached
            return
        descs = self._descs_now()
        side = cp.cuda.Stream(non_blocking=True)
        try:
            with side:
                side.begin_capture()
                for i, launch in enumerate(self.compiled.launches):
                    _launch(launch, self.compiled, self.arrays, descs.get(i), self.sym_values)
                graph = side.end_capture()
        except Exception as exc:
            if side.is_capturing():
                try:
                    with side:
                        side.end_capture()  # drain capture state; discard the partial graph
                except Exception:  # noqa: BLE001, S110 — already raising the original failure
                    pass
            raise GraphCaptureError(f"whole-program capture failed: {exc}") from exc
        # Throwaway replay absorbs graphExec instantiation/upload cost.
        graph.launch()
        cp.cuda.runtime.deviceSynchronize()
        self._graph_cache[key] = graph
        while len(self._graph_cache) > self._graph_cache_max:
            evicted = next(iter(self._graph_cache))
            self._graph_cache.pop(evicted)  # evict LRU
            self._sym_descs.pop(evicted, None)  # its desc overlay is no longer pinned
        self._e2e_graph = graph

    def _sym_key(self) -> tuple:
        """Hashable cache key for the current resolved symbolic values (``()`` for
        a fully-static graph)."""
        return tuple(sorted(self.sym_values.items()))

    def replay_program_graph(self) -> None:
        """Launch the whole-program graph for the current ``self.sym_values`` once
        on the default stream — the serving hot path. The caller sets the request's
        seq_len (:meth:`set_sym_values`), uploads its input prefix
        (:meth:`upload_prefix`), and captures-or-reuses the graph
        (:meth:`capture_program_graph`) first; results come from
        ``outputs(sym_values=…)``. Caller must hold ``gpu_lock()``."""
        if self._e2e_graph is None:
            raise RuntimeError("replay_program_graph called before capture_program_graph")
        self._e2e_graph.launch()

    def upload_prefix(self, input_data: dict[str, np.ndarray]) -> None:
        """H2D each host array into the contiguous PREFIX of its capacity-sized
        device buffer — no re-allocation, so the captured graphs' baked pointers
        stay valid (a logically ``(1, S, …)`` tensor occupies the first ``S*…``
        elements of the ``(1, S_cap, …)`` allocation; the kernels, launched at
        grids for the real S, only touch that prefix). Errors if a host array
        exceeds its buffer's capacity. Caller must hold ``gpu_lock()``."""
        for name, host in input_data.items():
            buf = self.compiled.buf_by_name[name]
            arr = self.arrays[name]
            flat = np.ascontiguousarray(host, dtype=buf.dtype.np).ravel()
            if flat.size > arr.size:
                raise ValueError(f"upload_prefix: {name!r} has {flat.size} elems > capacity {arr.size}")
            arr.ravel()[: flat.size].set(flat)

    def upload_prefix_device(self, input_data: dict[str, cp.ndarray]) -> None:
        """Device-to-device twin of :meth:`upload_prefix`: copy each cupy source
        into the contiguous PREFIX of its capacity-sized device buffer with NO
        host round-trip — the serving zero-copy path, where the sources are cupy
        views of the caller's torch GPU tensors (``cp.from_dlpack``). Same
        prefix-packing contract and capacity check as :meth:`upload_prefix` (the
        kernels launched at the real-S grid read only the prefix). Caller must
        hold ``gpu_lock()`` and order the copy on the replay stream — the serving
        runner enters a cupy external stream bound to torch's current stream so
        the copy, the graph replay, and the output read all enqueue in order."""
        import cupy as cp  # noqa: PLC0415

        for name, src in input_data.items():
            buf = self.compiled.buf_by_name[name]
            arr = self.arrays[name]
            # Self-copy skip: when the caller's source IS this buffer (the serving runner's
            # post→pre chaining rewires a producer's output view onto this input's backing, so
            # the "upload" would copy a buffer onto itself), there is nothing to move — and the
            # skip is what deletes the seam copy from the captured decode graph.
            if src.data.ptr == arr.data.ptr:
                continue
            flat = cp.ascontiguousarray(src, dtype=buf.dtype.np).ravel()
            if flat.size > arr.size:
                raise ValueError(f"upload_prefix_device: {name!r} has {flat.size} elems > capacity {arr.size}")
            arr.ravel()[: flat.size] = flat

    def time_program_window(self, replays: int) -> float:
        """One event window around ``replays`` back-to-back whole-program
        replays of the captured e2e graph; returns per-replay ms. Caller
        must have run :meth:`capture_program_graph` first."""
        import cupy as cp

        if self._e2e_start is None:
            self._e2e_start, self._e2e_stop = cp.cuda.Event(), cp.cuda.Event()
        self._e2e_start.record()
        for _ in range(replays):
            self._e2e_graph.launch()
        self._e2e_stop.record()
        n = max(1, len(self.compiled.launches))
        _wait_for_event(self._e2e_stop, _KERNEL_TIMEOUT_MS * n * replays, "<whole-program e2e window>")
        return cp.cuda.get_elapsed_time(self._e2e_start, self._e2e_stop) / replays

    def iter_once(
        self,
        *,
        batch_sizes: list[int] | None = None,
        pre_iter=None,
        per_launch_hook=None,
    ) -> list[float]:
        """Run every launch once. Returns per-launch wall time in ms,
        already event-synced before return.

        ``batch_sizes[i]`` repeats launch ``i`` ``N`` times inside one
        CUDA event window so per-iter Python/cupy framing overhead
        amortizes across launches when the kernel is faster than the
        framing (a 9 µs kernel measured one iter at a time is mostly
        framing noise). Returned dt is divided by the batch size so
        callers always see per-call ms.

        ``pre_iter(max_batch_size)`` runs once before the launch loop
        and inside the GPU lock the caller is holding — that's where
        ``_bench_interleaved`` issues its peer torch backends so they
        share the same warm GPU state emmy measures from.

        ``per_launch_hook(i, launch)`` runs after each launch's stop
        event has synced. :func:`run_program_debug` uses it to
        snapshot every non-input buffer after each launch.

        Per-kernel sync (the ``_wait_for_event(stop_i, ...)``) makes
        per-launch attribution accurate — without it, one kernel's
        stop event can slide into a downstream kernel's scheduling
        window and the timing for a sub-100µs kernel ends up
        contaminated by 0.5-0.8 ms of phantom stream-stall time. The
        watchdog also catches hung kernels independently per launch."""
        import cupy as cp

        n = len(self.compiled.launches)
        if batch_sizes is None:
            batch_sizes = [1] * n
        if pre_iter is not None:
            pre_iter(max(batch_sizes))
        if not self._starts:
            self._starts = [cp.cuda.Event() for _ in range(n)]
            self._stops = [cp.cuda.Event() for _ in range(n)]
        starts, stops = self._starts, self._stops
        descs = self._descs_now()
        dts = [0.0] * n
        for i, launch in enumerate(self.compiled.launches):
            b = batch_sizes[i]
            starts[i].record()
            if self._graphs is not None:
                # Captured-graph replay: one host call enqueues the whole
                # batch, so the event window measures dense GPU work. The
                # caller (``benchmark_program``) re-captures whenever the
                # batch sizes change, so ``_graphs[i]`` always matches ``b``.
                self._graphs[i].launch()
            else:
                for _ in range(b):
                    _launch(launch, self.compiled, self.arrays, descs.get(i), self.sym_values)
            stops[i].record()
            grace = _FIRST_ITER_GRACE if self._iters_done == 0 else 1.0
            _wait_for_event(stops[i], _KERNEL_TIMEOUT_MS * b * grace, f"{launch.kernel_name} (iter {self._iters_done})")
            elapsed_ms = cp.cuda.get_elapsed_time(starts[i], stops[i])
            # CUDA event timing has sub-µs resolution and a real launch must
            # consume at least one device cycle — a 0.0 reading means the
            # launch was a no-op (degenerate grid like BM=1×BN=128 with the
            # M tile entirely masked out, or a kernel that was fused into
            # nothing). Pinning a 0µs "win" in the autotune DB would lock
            # that variant in as the unbeatable best across re-runs. Treat
            # as bench_fail instead — the existing worker → parent → DB
            # path then records a normal sentinel row.
            if elapsed_ms <= 0.0:
                raise RuntimeError(
                    f"kernel {launch.kernel_name!r} reported {elapsed_ms:.3f}ms elapsed — "
                    "degenerate / no-op launch, variant marked bench_fail"
                )
            dts[i] = elapsed_ms / b
            if per_launch_hook is not None:
                per_launch_hook(i, launch)
        self._iters_done += 1
        return dts

    def outputs(self, sym_values: dict[str, int] | None = None) -> dict[str, np.ndarray]:
        """Copy every output buffer back to host. Caller must hold the
        GPU lock — ``.get()`` is an async D2H copy on the default
        stream, so peer workers' kernels would otherwise interleave
        with our D2H on the shared device.

        ``sym_values`` (serving's capture path) slices each output to its real-S
        shape — the buffer is allocated at capacity but only the
        ``resolve_shape(sym_values)`` prefix holds the request's result; the rest
        is unmasked garbage from the oversized allocation. Without it (the
        default) the whole buffer is returned (the uncaptured rebind path already
        sizes buffers to the request)."""
        out: dict[str, np.ndarray] = {}
        for b in self.compiled.bufs:
            if b.role != "output":
                continue
            arr = self.arrays[b.name]
            if sym_values is not None:
                shape = b.resolve_shape({**self.sym_values, **sym_values})
                n = math.prod(shape) if shape else 1
                out[b.name] = arr.ravel()[:n].get().reshape(shape)
            else:
                out[b.name] = arr.get()
        return out

    def output_prefix_device(self, sym_values: dict[str, int] | None = None) -> dict[str, cp.ndarray]:
        """Device twin of :meth:`outputs`: return each output buffer's real-S
        PREFIX as a cupy view (reshaped to the resolved shape) with NO ``.get()``
        host copy — the serving zero-copy path, where the caller wraps the view as
        a torch tensor (``torch.from_dlpack``) and clones it (the shared buffer is
        overwritten by the next request's replay). ``sym_values`` slices to the
        real shape exactly like :meth:`outputs`; without it the whole buffer view
        is returned. Caller must hold ``gpu_lock()`` (and read on the replay
        stream — see :meth:`upload_prefix_device`)."""
        out: dict[str, cp.ndarray] = {}
        for b in self.compiled.bufs:
            if b.role != "output":
                continue
            arr = self.arrays[b.name]
            if sym_values is not None:
                shape = b.resolve_shape({**self.sym_values, **sym_values})
                n = math.prod(shape) if shape else 1
                out[b.name] = arr.ravel()[:n].reshape(shape)
            else:
                out[b.name] = arr
        return out

    def snapshot(self) -> dict[str, np.ndarray]:
        """Copy every non-input buffer (scratch + constants + outputs)
        to host. Used by :func:`run_program_debug` to capture every
        intermediate state for per-launch comparison against a
        reference backend."""
        input_names = {b.name for b in self.compiled.bufs if b.role == "input"}
        return {name: arr.get() for name, arr in self.arrays.items() if name not in input_names}


# ---------------------------------------------------------------------------
# Public entry points: thin shells around CompiledProgram
# ---------------------------------------------------------------------------


def run_program(
    graph: Graph,
    input_data: dict[str, np.ndarray] | None = None,
    *,
    pre_run=None,
) -> tuple[RunResult, Any]:
    """Run the lowered graph once, return ``(RunResult, pre_run_result)``.

    ``pre_run`` runs once inside the GPU lock, before emmy's
    kernel launches. Its return value flows through as the tuple's
    second element. Tests use this to compute a torch eager reference
    on the same GPU window the emmy launches will see, so peer-
    worker CUDA activity can't interleave the eager forward with the
    emmy comparison."""
    from emmy.compiler.backend.gpu_lock import gpu_lock  # noqa: PLC0415

    with gpu_lock():
        pre_result = pre_run() if pre_run is not None else None
        prog = CompiledProgram.build(graph, input_data)
        dts = prog.iter_once()
        outputs = prog.outputs()
    return RunResult(outputs=outputs, time_ms=sum(dts)), pre_result


@dataclass
class DebugResult:
    outputs: dict[str, np.ndarray]
    per_launch: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)


def run_program_debug(
    graph: Graph,
    input_data: dict[str, np.ndarray] | None = None,
    *,
    pre_run=None,
) -> tuple[DebugResult, Any]:
    """Run the graph once, snapshotting every non-input buffer after
    each launch. Returns ``(DebugResult, pre_run_result)`` — same
    ``pre_run`` semantics as :func:`run_program`."""
    from emmy.compiler.backend.gpu_lock import gpu_lock  # noqa: PLC0415

    per_launch: dict[int, dict[str, np.ndarray]] = {}
    with gpu_lock():
        pre_result = pre_run() if pre_run is not None else None
        # Note: scratch buffers share one reused slab, so a per-launch snapshot of
        # a buffer *after its last use* reflects whatever now occupies that slot.
        # Each kernel's own output is valid at its launch (just written, still
        # live) — the usual debug read.
        prog = CompiledProgram.build(graph, input_data)
        prog.iter_once(per_launch_hook=lambda li, _lc: per_launch.__setitem__(li, prog.snapshot()))
        outputs = prog.outputs()
    return DebugResult(outputs=outputs, per_launch=per_launch), pre_result


def benchmark_program(
    graph: Graph,
    input_data: dict[str, np.ndarray] | None = None,
    warmup: int = 5,
    num_iters: int | str = 20,
    on_iter=None,
    compile_timeout_s: float | None = None,
    run_timeout_s: float | None = None,
    capture_graphs: bool = True,
) -> BenchmarkResult:
    """Time the graph's launches with per-kernel CUDA events.

    Single loop covers warmup + measurement: the first ``warmup`` iters
    are discarded, the rest are counted toward the result. The
    per-launch ``_KERNEL_TIMEOUT_MS`` watchdog (inside
    :meth:`CompiledProgram.iter_once`) runs every iter — warmup or
    measured — so a single hung kernel raises cleanly instead of
    stalling the whole sweep.

    ``num_iters`` accepts an explicit count or the string ``"auto"``.
    In auto mode the loop accumulates measured GPU time until it
    reaches ``_AUTO_BUDGET_MS`` (capped at ``_AUTO_MAX_ITERS`` measured
    iters). For a 7-µs RMSNorm that's ~14k iters; a 1-ms matmul gets
    ~100. The result's per-launch ``time_ms`` is the *median* of
    measured iters (mean was sensitive to single-iter outliers from
    thermal blips and GPU-lock-contention spikes — the autotune
    ``_pick_best_candidate`` selects on the lowest summed latency, so
    noise-driven dips made it pick variants whose post-tune bench was
    slower than the heuristic). Total ``time_ms`` is the sum of
    per-launch medians.

    ``on_iter(batch_size)`` is invoked once at the top of every iter
    inside the GPU lock — that's where ``_bench_interleaved`` runs
    peer torch backends so they time the same number of back-to-back
    calls emmy does per CUDA event window, no warm-vs-cold
    asymmetry.

    ``compile_timeout_s`` bounds NVRTC + alloc + descriptor setup;
    raised inside :meth:`CompiledProgram.build`.

    ``run_timeout_s`` bounds the iter loop on **accumulated GPU time**
    (sum of per-launch CUDA-event measurements), not wall-clock — so
    Python/cupy framing overhead doesn't shrink the budget for tiny
    ops. Catches the gap left by the per-launch ``_KERNEL_TIMEOUT_MS``
    watchdog: a variant where every launch fits under the watchdog but
    summed across iters exceeds the budget (e.g. 999 ms × N iters).
    Checked between iters so no in-flight launch is mid-kernel when
    the function raises.

    ``capture_graphs`` (default on) captures each launch's batch into a CUDA
    graph (:meth:`CompiledProgram.capture_launch_graphs`) once batch sizes are
    calibrated, so the event windows measure dense GPU work instead of
    per-launch dispatch gaps. Warmup iters always run uncaptured, which keeps
    the zero-elapsed degenerate-launch guard and the hung-kernel watchdog
    probing real launches before any graph is built. A capture failure is
    non-fatal: the bench logs a warning and continues uncaptured, reporting it
    via ``BenchmarkResult.captured`` — callers pairing this result with peer
    torch timings (``bench_lowered_vs_torch`` / the e2e comparison) use that
    flag to re-run all-or-nothing so one table never mixes semantics, and the
    tune sweep persists it on each ``perf`` row (captured measurements
    supersede wall-semantics ones on write — see ``SearchDB.record_perf``).

    Multi-launch programs additionally get a WHOLE-program time per measured
    iter — one event window around back-to-back replays of a single CUDA
    graph holding every launch in program order
    (:meth:`CompiledProgram.time_program_window`) — reported as
    ``BenchmarkResult.e2e_ms`` / ``e2e_min_ms``. The per-launch windows each
    replay one kernel solo, so their sum misses cross-kernel cache effects;
    only the whole-program window is comparable against a captured torch
    forward. Automatic when capture holds and the program has more than one
    launch (for a single launch the solo window IS the program time, so the
    fields stay ``None`` and nothing is measured twice — the common case for
    the autotune sweep's single-node slices); also ``None`` when capture is
    off or fell back."""
    from emmy.compiler.backend.gpu_lock import gpu_lock  # noqa: PLC0415

    target_total_ms, max_measured, auto = _resolve_iter_budget(num_iters)

    with gpu_lock():
        prog = CompiledProgram.build(graph, input_data, compile_timeout_s=compile_timeout_s)
        n = len(prog.compiled.launches)
        batch_sizes = [1] * n
        # Per-launch sample list — kept around to compute the median
        # across measured iters (more robust than the arithmetic mean
        # against thermal blips, GPU-lock-contention spikes, and other
        # one-off outliers the autotune's variant ranking previously
        # got confused by; see ``project_..._noise`` write-ups).
        samples: list[list[float]] = [[] for _ in range(n)]
        measure_e2e = n > 1  # single launch: the solo window IS the program time
        e2e_samples: list[float] = []
        e2e_replays = 0  # calibrated lazily on the first measured iter
        iters_run = 0
        measured = 0
        cumulative_gpu_ms = 0.0  # measured-iter GPU time, for the "auto" stop target
        total_gpu_ms = 0.0  # all-iter GPU time (incl. warmup), for the run-stage budget

        def _try_capture(sizes: list[int]) -> bool:
            """Best-effort capture; a failure logs + continues uncaptured."""
            try:
                prog.capture_launch_graphs(sizes)
            except GraphCaptureError as exc:
                logger.warning("[cuda] %s — continuing with uncaptured (dispatch-inclusive) timing", exc)
                return False
            return True

        if capture_graphs and warmup == 0:
            # No warmup → the calibration below never fires; capture the
            # uncalibrated all-1 batches so measurement is still dense.
            capture_graphs = _try_capture(batch_sizes)
        while True:
            iter_dts = prog.iter_once(batch_sizes=batch_sizes, pre_iter=on_iter)
            iters_run += 1
            total_gpu_ms += sum(iter_dts[i] * batch_sizes[i] for i in range(n))
            # GPU-time run budget: bail if the cumulative GPU time
            # across all iters (warmup + measured) exceeds
            # ``run_timeout_s``. Catches the "every launch is just
            # under the per-launch watchdog" pathology. Counts warmup
            # iters too so a slow kernel can't hide behind warmup
            # discards.
            if run_timeout_s is not None and total_gpu_ms > run_timeout_s * 1000.0:
                raise RuntimeError(f"benchmark run stage exceeded {run_timeout_s:.1f}s of GPU time — variant marked bench_fail")
            if iters_run == warmup:
                batch_sizes = _calibrate_batch_sizes(iter_dts)
                if capture_graphs:
                    # Capture (or re-capture) at the calibrated batch sizes.
                    # The warmup extension below can re-fire this calibration
                    # branch with new batch sizes — ``capture_launch_graphs``
                    # no-ops when they're unchanged and re-captures when not,
                    # so graphs and batches never go out of sync.
                    capture_graphs = _try_capture(batch_sizes)
                # Extend warmup until total warmup GPU time clears the
                # clock-ramp floor. Post-batching, each subsequent
                # warmup iter spends roughly
                # ``sum(iter_dts[i] * batch_sizes[i])`` of GPU time —
                # use the just-measured per-launch dts to estimate how
                # many extra iters are needed.
                if total_gpu_ms < _WARMUP_TARGET_MS:
                    per_iter_ms = sum(iter_dts[i] * batch_sizes[i] for i in range(n))
                    if per_iter_ms > 0:
                        warmup += int(math.ceil((_WARMUP_TARGET_MS - total_gpu_ms) / per_iter_ms))
            if iters_run <= warmup:
                continue
            # Measured iter: store per-launch sample (already
            # normalized to per-launch ms inside ``iter_once``).
            # Reduced via median at the end so a single outlier iter
            # can't shift the result.
            for i in range(n):
                samples[i].append(iter_dts[i])
            cumulative_gpu_ms += sum(iter_dts[i] * batch_sizes[i] for i in range(n))
            measured += 1
            # Whole-program window, one per measured iter — shares the same
            # warm GPU state as the per-launch windows and the ``on_iter``
            # torch closures. Counted toward the run-stage GPU budget but NOT
            # the auto-stop target, so it never starves per-launch sampling.
            if measure_e2e and capture_graphs:
                if e2e_replays == 0:
                    try:
                        prog.capture_program_graph()
                    except GraphCaptureError as exc:
                        logger.warning("[cuda] %s — skipping whole-program e2e timing", exc)
                        measure_e2e = False
                    else:
                        iter_ms = sum(iter_dts)
                        e2e_replays = max(1, int(round(_BATCH_TARGET_MS / iter_ms))) if 0 < iter_ms < _BATCH_TARGET_MS else 1
                if measure_e2e:
                    e2e_dt = prog.time_program_window(e2e_replays)
                    e2e_samples.append(e2e_dt)
                    total_gpu_ms += e2e_dt * e2e_replays
            if measured >= max_measured:
                break
            if auto and cumulative_gpu_ms >= target_total_ms:
                break

    return _samples_to_result(samples, prog.compiled.launches, captured=capture_graphs, e2e_samples=e2e_samples)


def _resolve_iter_budget(num_iters: int | str) -> tuple[float, int, bool]:
    """Resolve ``num_iters`` to ``(target_total_ms, max_measured, auto)``."""
    if isinstance(num_iters, str):
        if num_iters != "auto":
            raise ValueError(f"num_iters must be int or 'auto', got {num_iters!r}")
        return (_AUTO_BUDGET_MS, _AUTO_MAX_ITERS, True)
    return (float("inf"), int(num_iters), False)


def _calibrate_batch_sizes(iter_dts: list[float]) -> list[int]:
    """Pick per-position batch sizes so each CUDA event window covers
    ~``_BATCH_TARGET_MS`` of GPU time. Per-position 1 when the kernel
    already exceeds the target — no benefit to batching there."""
    return [max(1, int(round(_BATCH_TARGET_MS / dt))) if 0 < dt < _BATCH_TARGET_MS else 1 for dt in iter_dts]


def _samples_to_result(
    samples: list[list[float]], launches: list[_Launch], *, captured: bool = False, e2e_samples: list[float] | None = None
) -> BenchmarkResult:
    """Collapse per-launch sample lists to a ``BenchmarkResult`` keyed
    on the median of each launch's measured iters."""
    import statistics as _stats  # noqa: PLC0415

    n = len(launches)
    medians = [(_stats.median(samples[i]) if samples[i] else 0.0) for i in range(n)]
    mins = [(min(samples[i]) if samples[i] else 0.0) for i in range(n)]
    per_launch = [
        LaunchTime(
            idx=i,
            kernel_name=launches[i].kernel_name,
            time_ms=medians[i],
            samples=tuple(samples[i]) if samples[i] else None,
        )
        for i in range(n)
    ]
    # ``time_ms`` is the per-launch median (stable for tune's ranking); ``min_ms``
    # is the per-launch best-case (least OS/thermal noise — what ``run --bench``
    # reports, matching tune's min-over-variants reporting).
    return BenchmarkResult(
        time_ms=sum(medians),
        min_ms=sum(mins),
        num_launches=n,
        per_launch=per_launch if per_launch else None,
        captured=captured,
        e2e_ms=_stats.median(e2e_samples) if e2e_samples else None,
        e2e_min_ms=min(e2e_samples) if e2e_samples else None,
    )


# ---------------------------------------------------------------------------
# Subprocess-isolated benchmark worker (single async transport)
# ---------------------------------------------------------------------------


class BenchWorkerJobError(RuntimeError):
    """A worker job that ran and failed (``ok: False`` response — the child is alive).
    ``cache_miss`` marks the one retryable kind: a job referenced a ``run_inputs_key``
    a freshly-respawned child no longer holds."""

    def __init__(self, message: str, *, cache_miss: bool = False) -> None:
        super().__init__(message)
        self.cache_miss = cache_miss


class _AsyncBenchWorker:
    """The parent-side transport for the SIGKILL-able ``_bench_worker`` subprocess —
    the **single** isolated-bench transport (the old sync ``_BenchWorker`` is gone).

    Lets the parent enforce a hard wall-clock cap on a bench: if the worker doesn't
    respond within ``wall_timeout_s``, the parent SIGKILLs it. The dirty CUDA stream
    (and any kernels still queued behind a hung launch) dies with the process, so the
    *next* bench starts on a clean device — fixing the "autotune hangs on the variant
    AFTER a bench_fail" pathology. The worker imports cupy lazily on its first
    request, so spawn cost is just Python startup (~0.2 s).

    Drives the ``_bench_worker`` protocol (``<8-byte LE length><pickle>``, both
    directions) over asyncio streams, so one event loop can keep N device-pinned
    workers benching concurrently — the per-kernel multi-GPU autotune path
    (``two_level._inner_reward_async``). The deployable ``--bench`` comparison awaits
    ``benchmark_compare_isolated_async`` over a one-shot instance (via
    ``_run_job_oneshot``); the autotune sweep awaits a persistent instance per GPU
    directly via ``benchmark_program_isolated_async``.

    Pin a worker to a physical GPU with ``device_id``: the spawn env gets
    ``CUDA_VISIBLE_DEVICES=<id>`` (so the child's logical device 0 *is* that
    GPU — every argumentless ``cp.cuda.Device()`` in the child resolves
    correctly with no other call-site change) and, when a base
    ``EMMY_GPU_LOCK`` is set, a per-device lock path so workers on
    different GPUs take distinct ``FileLock``s instead of serialising. The
    env overlay rides the child only — the parent's ``os.environ`` is never
    mutated (it's shared by every slot on the one event-loop thread).

    The wall-clock cap is :func:`asyncio.wait_for`; on overrun the child is
    SIGKILLed and respawned on the next bench."""

    _WORKER_MODULE = "emmy.compiler.backend.cuda._bench_worker"
    _STDERR_TAIL_CHARS = 4000
    _REAP_TIMEOUT_S = 2.0

    def __init__(self, *, device_id: int | None = None) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._device_id = device_id
        # Bounded tail of the CURRENT child's stderr, fed by a background drain task. A
        # chatty child (HF shard-download progress, nvcc warnings) would otherwise fill
        # the ~64 KB stderr pipe and block mid-job — which the parent misreads as a
        # wall-timeout hang — and the tail is the diagnostic every failure path wants.
        self._stderr_tail = ""
        self._stderr_task: asyncio.Task | None = None
        # The framed protocol is a single request/response stream.  Most tune callers
        # lease a backend through a slot queue, but lifecycle/replay callers are also
        # allowed to use the backend directly.  Serialize here, at the transport
        # boundary, so two legal callers can never interleave frames or race one
        # request's timeout teardown against the other's process handle.
        self._job_lock = asyncio.Lock()
        # True only for children spawned by :meth:`_spawn`.  Test harnesses replace
        # ``_spawn`` with ordinary subprocesses, which must not be mistaken for a
        # process-group leader.
        self._owns_process_group = False
        # ``run_inputs_key``s this child has cached (see ``benchmark_pinned_isolated_async``).
        # Cleared on every (re)spawn — a fresh child holds no cache.
        self.cached_input_keys: set[str] = set()

    def _child_env(self) -> dict:
        env = dict(_os.environ)
        if self._device_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self._device_id)
            from emmy import config  # noqa: PLC0415

            base = config.gpu_lock_path()
            if base:
                # Per-device lock so concurrent device-pinned workers don't
                # serialise on one FileLock (the lock is taken inside the child).
                env["EMMY_GPU_LOCK"] = f"{base}-{self._device_id}"
        return env

    async def _spawn(self) -> None:
        # A timed-out compile can still have nvcc/ptxas descendants running when the
        # worker itself is SIGKILLed.  Give the worker a private process group on
        # POSIX so teardown can kill the complete compiler tree.  Non-POSIX asyncio
        # retains the direct-child fallback below.
        spawn_kwargs = {"start_new_session": True} if _os.name == "posix" else {}
        self._proc = await asyncio.create_subprocess_exec(
            _sys.executable,
            "-m",
            self._WORKER_MODULE,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._child_env(),
            **spawn_kwargs,
        )
        self._owns_process_group = _os.name == "posix"
        self._stderr_tail = ""
        self._stderr_task = asyncio.ensure_future(self._drain_stderr(self._proc))
        self.cached_input_keys.clear()
        logger.info("[bench-worker] spawned (async) pid=%s device=%s", self._proc.pid, self._device_id)

    async def _drain_stderr(self, proc: asyncio.subprocess.Process) -> None:
        """Continuously drain the child's stderr into the bounded tail. Runs for the
        child's whole life (exits on EOF, i.e. child exit / SIGKILL) so the pipe can never
        fill and block the child mid-job."""
        try:
            while True:
                chunk = await proc.stderr.read(4096)
                if not chunk:
                    return
                self._stderr_tail = (self._stderr_tail + chunk.decode(errors="replace"))[-self._STDERR_TAIL_CHARS :]
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the drain is best-effort diagnostics
            return

    @staticmethod
    def _close_stdin(proc: asyncio.subprocess.Process) -> None:
        stdin = getattr(proc, "stdin", None)
        if stdin is not None:
            with contextlib.suppress(Exception):
                stdin.close()

    @staticmethod
    def _close_transport(proc: asyncio.subprocess.Process) -> None:
        """Last-resort pipe cleanup after a bounded reap expires.

        ``asyncio.subprocess.Process`` has no public stream-close API for stdout,
        so closing its transport is the only way to detach inherited pipe handles
        without waiting forever.  This is deliberately a timeout-only fallback.
        """
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            with contextlib.suppress(Exception):
                transport.close()

    @staticmethod
    def _terminate(proc: asyncio.subprocess.Process, *, process_group: bool) -> None:
        if proc.returncode is not None:
            return
        if process_group and _os.name == "posix":
            try:
                _os.killpg(proc.pid, _signal.SIGKILL)
                return
            except ProcessLookupError:
                return
            except OSError:
                # A replaced test spawn (or a platform without the promised group)
                # still gets direct-child cleanup.
                pass
        with contextlib.suppress(ProcessLookupError):
            proc.kill()

    @staticmethod
    async def _reap_adopted_group_children(pgid: int) -> None:
        """Reap compiler descendants adopted by this process after a group kill.

        A container command often runs as PID 1 (or as a child subreaper).  Killing
        the bench worker while it is blocked in nvcc reparents nvcc/ptxas here; no
        asyncio transport owns those adopted processes, so they otherwise remain
        zombies for the rest of a long tune.  The worker is already awaited before
        this helper runs, and its private group cannot contain another asyncio-owned
        direct child, making ``waitpid(-pgid)`` safe.  Ordinary non-reaper parents get
        ``ChildProcessError`` and return immediately.
        """
        if _os.name != "posix":
            return
        for _ in range(5):
            reaped = False
            while True:
                try:
                    pid, _status = _os.waitpid(-pgid, _os.WNOHANG)
                except ChildProcessError:
                    return
                if pid == 0:
                    break
                reaped = True
            if not reaped:
                # Group members receive SIGKILL together but may be reparented a
                # scheduling tick after the direct worker's watcher completes.
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(0)

    def _kill(self) -> None:
        proc = self._proc
        self._proc = None
        process_group = self._owns_process_group
        self._owns_process_group = False
        self.cached_input_keys.clear()
        if proc is None:
            return
        self._close_stdin(proc)
        self._terminate(proc, process_group=process_group)

    def close(self) -> None:
        """Best-effort synchronous kill for callers outside the worker's event loop.

        Normal tune/session teardown must use :meth:`aclose` so process-group reap
        and pipe cleanup are awaited and bounded.
        """
        self._kill()

    async def aclose(self) -> None:
        """Drain an in-flight request, then terminate and boundedly reap the worker."""
        async with self._job_lock:
            await self._aclose_current(graceful=True)

    async def _aclose_current(self, *, graceful: bool) -> None:
        """Close the current child while the caller owns ``_job_lock``.

        Normal session teardown first closes stdin so an idle worker can exit cleanly.
        A timeout skips that grace period and SIGKILLs the private process group.  Both
        paths bound ``Process.wait``: asyncio otherwise waits for every subprocess pipe
        to close, which can be forever when an adopted compiler descendant inherited a
        pipe.  The timeout-only transport close keeps the parent event loop live.
        """
        proc = self._proc
        self._proc = None
        process_group = self._owns_process_group
        self._owns_process_group = False
        stderr_task = self._stderr_task
        self._stderr_task = None
        self.cached_input_keys.clear()
        if proc is not None:
            reaped = False
            self._close_stdin(proc)
            if graceful and proc.returncode is None:
                with contextlib.suppress(TimeoutError, ProcessLookupError):
                    await asyncio.wait_for(proc.wait(), timeout=0.25)
            self._terminate(proc, process_group=process_group)
            try:
                await asyncio.wait_for(proc.wait(), timeout=self._REAP_TIMEOUT_S)
                reaped = True
            except (TimeoutError, ProcessLookupError):
                logger.warning("[bench-worker] subprocess reap exceeded %.1fs; closing its transport", self._REAP_TIMEOUT_S)
                self._close_transport(proc)
            if reaped and process_group:
                await self._reap_adopted_group_children(proc.pid)
        if stderr_task is not None:
            # The drain ends on the killed child's stderr EOF; reap it so no pending
            # task survives the caller's event loop.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(stderr_task, timeout=self._REAP_TIMEOUT_S)
            if not stderr_task.done():
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await stderr_task

    async def _stderr_snapshot(self) -> str:
        """The drained stderr tail, letting the drain task flush briefly first (after a
        kill it ends at EOF almost immediately)."""
        if self._stderr_task is not None and not self._stderr_task.done():
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(self._stderr_task), timeout=0.5)
        return self._stderr_tail

    async def run_job(self, request_obj: dict, *, wall_timeout_s: float) -> dict:
        """Send one request, read the response within ``wall_timeout_s`` (else SIGKILL
        + raise ``RuntimeError``), and return the unpickled response. A stale-worker
        race on send respawns and retries once; a response-side timeout is a hard
        error. A response-side EOF (the child self-destructed mid-job) respawns and
        retries ONCE after a short drain grace — see the handler for why."""
        async with self._job_lock:
            try:
                return await self._run_job_single(request_obj, wall_timeout_s=wall_timeout_s)
            except asyncio.CancelledError:
                # A cancelled reader may have already sent its frame.  Never release
                # the single-flight lock with that response still pending: retire the
                # whole stream so the next caller starts from a clean frame boundary.
                await self._aclose_current(graceful=False)
                raise

    async def _run_job_single(self, request_obj: dict, *, wall_timeout_s: float) -> dict:
        """One serialized framed request.  Caller owns ``_job_lock``."""
        request = pickle.dumps(request_obj, protocol=pickle.HIGHEST_PROTOCOL)
        frame = len(request).to_bytes(8, "little") + request
        deadline = _time_module.perf_counter() + wall_timeout_s
        for attempt in (0, 1):
            if self._proc is None or self._proc.returncode is not None:
                await self._spawn()
            assert self._proc is not None  # for type narrowing
            proc = self._proc
            try:
                remaining = deadline - _time_module.perf_counter()
                if remaining <= 0:
                    raise TimeoutError
                proc.stdin.write(frame)
                await asyncio.wait_for(proc.stdin.drain(), timeout=remaining)
            except TimeoutError as exc:
                await self._aclose_current(graceful=False)
                raise RuntimeError(
                    f"bench worker did not accept the request within {wall_timeout_s:.1f}s wall budget — SIGKILL'd, stream cleaned"
                    f"{self._tail_suffix()}"
                ) from exc
            except (BrokenPipeError, ConnectionResetError) as exc:
                await self._aclose_current(graceful=False)
                if attempt == 1:
                    raise RuntimeError(f"bench worker died during request send: {exc}{self._tail_suffix()}") from exc
                logger.info("[bench-worker] stale async worker on send (%s) — respawning", exc)
                continue

            try:
                remaining = deadline - _time_module.perf_counter()
                if remaining <= 0:
                    raise TimeoutError
                header = await asyncio.wait_for(proc.stdout.readexactly(8), timeout=remaining)
                n = int.from_bytes(header, "little")
                remaining = deadline - _time_module.perf_counter()
                if remaining <= 0:
                    raise TimeoutError
                body = await asyncio.wait_for(proc.stdout.readexactly(n), timeout=remaining)
            except TimeoutError as exc:
                await self._aclose_current(graceful=False)
                raise RuntimeError(
                    f"bench worker exceeded {wall_timeout_s:.1f}s wall budget — SIGKILL'd, stream cleaned{self._tail_suffix()}"
                ) from exc
            except asyncio.IncompleteReadError as exc:
                stderr_tail = await self._stderr_snapshot()
                await self._aclose_current(graceful=False)
                if attempt == 1:
                    raise RuntimeError(f"bench worker EOF before response; stderr tail: {stderr_tail}") from exc
                # The child self-destructs (``os._exit``) on a hung kernel to dodge the cupy
                # atexit deadlock, so a mid-job EOF usually means THIS config hangs — the retry
                # will EOF again and fail loudly, costing one extra watchdog interval. But right
                # after a SIGKILL'd predecessor (a greedy-hang wall kill), the dead child's
                # zombie context can still hold the GPU while the driver tears it down, hanging
                # an INNOCENT first launch on the fresh child — a transient the golden refresh
                # sweeps kept hitting on the row right after a hang. One respawn + retry after a
                # short drain grace tells the two apart (the same row replays clean once the
                # zombie context is gone).
                logger.info("[bench-worker] child EOF'd mid-job — draining the device and retrying once%s", self._tail_suffix())
                await asyncio.sleep(min(2.0, max(0.0, deadline - _time_module.perf_counter() - 1.0)))
                continue

            resp = pickle.loads(body)
            if not resp.get("ok"):
                # The in-child traceback (and the stderr tail, where CLI-style helpers log
                # their cause before exiting) would otherwise be silently discarded.
                if resp.get("traceback"):
                    logger.error("[bench-worker] job failed in the child; traceback:\n%s%s", resp["traceback"], self._tail_suffix())
                raise BenchWorkerJobError(f"bench worker error: {resp.get('error', '?')}", cache_miss=bool(resp.get("cache_miss")))
            if resp.pop("_retire_worker", False):
                # The child returned a completed same-input reference after its later greedy
                # timing hit the hung-kernel watchdog. Retire it before returning the reference:
                # a queued/nonterminating kernel must never share a context with pinned rows.
                await self._aclose_current(graceful=True)
            return resp
        raise RuntimeError("bench worker unreachable")  # both attempts exhausted (defensive)

    async def warmup(self, *, wall_timeout_s: float = 60.0) -> None:
        """Initialize the child CUDA context outside a candidate's wall budget."""
        response = await self.run_job({"worker_warmup": True}, wall_timeout_s=wall_timeout_s)
        if not response.get("warmed"):
            raise RuntimeError("bench worker did not acknowledge CUDA warmup")

    def _tail_suffix(self) -> str:
        """The drained stderr tail as an error-message suffix ('' when the child was quiet)."""
        return f"; child stderr tail:\n{self._stderr_tail}" if self._stderr_tail.strip() else ""


async def benchmark_program_isolated_async(
    graph: Graph,
    *,
    worker: _AsyncBenchWorker,
    wall_timeout_s: float,
    warmup: int = 5,
    num_iters: int | str = 20,
    compile_timeout_s: float | None = None,
    run_timeout_s: float | None = None,
    nvcc_flags: str | None = None,
    capture_graphs: bool = True,
) -> BenchmarkResult:
    """Wall-time-bounded ``benchmark_program`` in a subprocess, benching through a
    caller-supplied device-pinned ``worker`` so one event loop can drive N GPUs
    concurrently — the autotune sweep's transport. The in-worker
    ``compile_timeout_s`` / ``run_timeout_s`` budgets apply, and ``wall_timeout_s`` is
    the SIGKILL backstop for a kernel that keeps the GPU busy past them. No ``on_iter``
    (interleaved ``run --bench`` benches in-process via ``benchmark_program``)."""
    resp = await worker.run_job(
        {
            "graph": graph,
            "nvcc_flags": nvcc_flags,
            "torch_spec": None,  # no torch comparison — pure emmy bench
            "kwargs": {
                "warmup": warmup,
                "num_iters": num_iters,
                "compile_timeout_s": compile_timeout_s,
                "run_timeout_s": run_timeout_s,
                "capture_graphs": capture_graphs,
            },
        },
        wall_timeout_s=wall_timeout_s,
    )
    return resp["result"]


async def benchmark_pinned_isolated_async(
    graph: Graph,
    *,
    worker: _AsyncBenchWorker,
    wall_timeout_s: float,
    run_inputs: dict | None = None,
    run_inputs_key: str | None = None,
    warmup: int = 5,
    num_iters: int | str = 20,
    compile_timeout_s: float | None = None,
    run_timeout_s: float | None = None,
) -> tuple[BenchmarkResult, dict | None]:
    """One ``run --bench`` pinned-row job through the persistent ``worker``: an optional
    single execution on ``run_inputs`` (the greedy run's inputs — the wrong-answer gate's
    measurement side, outputs returned for the parent to compare) followed by the emmy-only
    bench. One job per row over one persistent worker per run session; a hung kernel dies
    with the SIGKILL'd child and the next row's job respawns a clean context.

    ``run_inputs_key`` (a session-unique token) lets the reference inputs cross the pipe
    ONCE per child instead of per row — hundreds of MB on the big ``--code`` shapes. The
    child caches them under the key; later rows send the key alone. The worker tracks which
    keys the CURRENT child holds (``cached_input_keys``, cleared on respawn), and a
    cache-miss response — a respawn raced the tracking — retries once with the inputs
    included."""

    def _request(with_inputs: bool) -> dict:
        return {
            "graph": graph,
            "torch_spec": None,
            "run_inputs": run_inputs if with_inputs else None,
            "run_inputs_key": run_inputs_key,
            "kwargs": {
                "warmup": warmup,
                "num_iters": num_iters,
                "compile_timeout_s": compile_timeout_s,
                "run_timeout_s": run_timeout_s,
            },
        }

    send_inputs = run_inputs is not None and (run_inputs_key is None or run_inputs_key not in worker.cached_input_keys)
    try:
        resp = await worker.run_job(_request(send_inputs), wall_timeout_s=wall_timeout_s)
    except BenchWorkerJobError as exc:
        if not (exc.cache_miss and run_inputs is not None):
            raise
        resp = await worker.run_job(_request(True), wall_timeout_s=wall_timeout_s)
    if run_inputs_key is not None and run_inputs is not None:
        worker.cached_input_keys.add(run_inputs_key)
    return resp["result"], resp.get("run_outputs")


async def benchmark_compare_worker_async(
    *,
    worker: _AsyncBenchWorker,
    lowered: Graph,
    torch_spec: tuple,
    bench_backends: str,
    wall_timeout_s: float,
    warmup: int,
    iters: int,
    seed: int,
    accuracy: bool = False,
    want_ref: bool = False,
    strict_accuracy: bool = False,
) -> dict:
    """``run --bench``'s greedy-row transport: the same comparison job as
    :func:`benchmark_compare_isolated_async` but over a caller-supplied persistent
    ``worker`` (shared with the pinned-row jobs — one worker per run session) and with the
    run path's extras: ``accuracy`` (the in-child real-input emmy-vs-eager verdict) and
    ``want_ref`` (that run's ``(inputs, outputs)`` for the pinned rows' wrong-answer gate).
    Returns the normalized response dict — keys ``results`` / ``result`` /
    ``torch_available`` / ``captured`` / ``accuracy_error`` / ``run_io`` plus the optional
    ``greedy_error`` / ``reference_run_us`` when an embedded Loop reference completed before
    its repeated greedy timing failed."""
    resp = await worker.run_job(
        {
            "graph": lowered,
            "torch_spec": torch_spec,
            "bench_backends": bench_backends,
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
            "accuracy": accuracy,
            "want_ref": want_ref,
            "strict_accuracy": strict_accuracy,
        },
        wall_timeout_s=wall_timeout_s,
    )
    return {
        "results": resp["results"],
        "result": resp["result"],
        "torch_available": resp["torch_available"],
        "captured": resp.get("captured", False),
        "accuracy_error": resp.get("accuracy_error"),
        "run_io": resp.get("run_io"),
        "greedy_error": resp.get("greedy_error"),
        "reference_run_us": resp.get("reference_run_us"),
        "correctness": resp.get("correctness"),
    }


async def _run_job_oneshot(request_obj: dict, *, wall_timeout_s: float, device_id: int | None = None) -> dict:
    """Spawn a fresh ``_AsyncBenchWorker``, run one job, tear it down.
    The transport for the synchronous one-shot bridges below — they each wrap this
    in ``asyncio.run`` (the worker's streams bind to the loop, so it can't persist
    across ``asyncio.run`` calls; the per-call ~0.2 s spawn is negligible against a
    deployable ``--bench``). ``device_id`` keeps the comparison on the selected
    tune GPU instead of silently falling back to ordinal 0."""
    worker = _AsyncBenchWorker(device_id=device_id)
    try:
        return await worker.run_job(request_obj, wall_timeout_s=wall_timeout_s)
    finally:
        await worker.aclose()


async def benchmark_compare_isolated_async(
    *,
    lowered: Graph,
    torch_spec: tuple,
    bench_backends: str,
    wall_timeout_s: float,
    warmup: int,
    iters: int,
    seed: int,
    nvcc_flags: str | None = None,
    device_id: int | None = None,
) -> tuple:
    """Run the deployable eager / torch.compile / emmy comparison in the
    SIGKILL-able worker, awaiting a fresh one-shot :class:`_AsyncBenchWorker`
    (the one transport).

    Unlike the emmy-only autotune bench, the deployable comparison interleaves emmy with
    real torch in one process and so couldn't be isolated before — a hung generated kernel wedged
    the whole run. This ships the *entire* comparison to the worker: a hung kernel hangs the child,
    which the parent SIGKILLs on ``wall_timeout_s``, freeing the device and leaving the parent clean.

    ``torch_spec`` rebuilds the torch side **in the child** (no live module crosses the pipe), reusing
    the same core functions the in-process path uses:

    - ``("trace_args", {code, input, adapter, layer, seq_len, dynamic})`` → ``load_or_trace`` rebuilds the real
      module (an HF model id or a ``--code`` expression), benched via ``bench_full_model_real``.
    - ``("frontend_graph", Graph | None)`` → ``bench_lowered_vs_torch`` (per-kernel reproducer; ``None``
      benches emmy-only when the graph isn't torch-runnable).

    Returns ``(results, bench, torch_available, captured, accuracy_error)`` — the shape
    ``bench_lowered_vs_torch`` returns (``captured``: all backends were timed under CUDA graph
    capture; False means the all-or-nothing fallback ran and the timings include host dispatch).
    ``accuracy_error`` is the non-fatal eager-reference verdict for a frontend reproducer."""
    resp = await _run_job_oneshot(
        {
            "graph": lowered,
            "nvcc_flags": nvcc_flags,
            "torch_spec": torch_spec,
            "bench_backends": bench_backends,
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
        },
        wall_timeout_s=wall_timeout_s,
        device_id=device_id,
    )
    return resp["results"], resp["result"], resp["torch_available"], resp.get("captured", False), resp.get("accuracy_error")
