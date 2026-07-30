"""Op-key derivation + source-chain walking for the search package.

Shared by :mod:`.db`, :mod:`.policy`, and the bench-terminal helper
in :mod:`emmy.compiler.pipeline.pipeline`.

``op_cache_key`` keys any kernel-bearing op:

- ``CudaOp`` — digest of rendered kernel source + launch params (the
  bits that determine runtime behavior).
- ``LoopOp`` / ``TileOp`` / ``KernelOp`` — digest of the dialect tag
  plus :meth:`Body.structural_key` (canonicalizes SSA, axis,
  commutative-arg, and external-buffer names). KernelOp works because
  ``kernel/ir.py`` registers ``rewrite`` handlers for every Kernel-IR
  stmt (Smem, Sync, CpAsync*, Tma*, Mbarrier*, TreeHalve, WarpShuffle),
  letting ``normalize_body`` walk the body without bailing.

Same kernel reached via different rewrite paths produces the same key
— ``Op.source`` is not part of the digest.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from emmy.compiler.structural import digest

Dialect = Literal["loop", "tile", "kernel", "cuda"]


def op_cache_key(op: object) -> str | None:
    """Cache key for any kernel-bearing op, or ``None`` if not cacheable."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
    from emmy.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
    from emmy.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

    if isinstance(op, CudaOp):
        # Name-invariant: the kernel function name is rendered into the source
        # (``void <name>(...)``) but doesn't change runtime behavior. Normalize
        # it out so renaming a kernel (e.g. via op provenance) neither busts the
        # perf cache nor blocks an isolated-kernel tune from transferring to a
        # whole-model compile.
        src = op.kernel_source.replace(op.kernel_name, "_K_") if op.kernel_name else op.kernel_source
        return digest("CudaOp", src, op.arg_order, op.grid, op.block, op.smem_bytes)
    if isinstance(op, (LoopOp, TileOp, KernelOp)):
        # Knobs are part of the key: same-body / different-knobs variants must
        # not collide with their parent in the search tree, or
        # ``SearchTree.expand`` self-parents the node and
        # ``_propagate_expected`` walks the parent chain forever.
        knob_key = tuple(sorted(op.knobs.items())) if op.knobs else ()
        # ``TileOp`` identity is the α-invariant hash of the canonically renumbered TERM (step 7
        # — no longer the lowered loop nest): SSA / buffer names canonicalize positionally, the
        # structure is the stored params verbatim (``ops.term_key``). Shape discrimination rides
        # the stamped ``S_ext_*`` features in ``knob_key``, exactly as before.
        if isinstance(op, TileOp):
            from emmy.compiler.ir.tile.ops import term_key  # noqa: PLC0415

            return digest(type(op).__name__, term_key(op.op), knob_key)
        return digest(type(op).__name__, op.body.structural_key(), knob_key)
    return None


def dialect_of(op: object) -> Dialect | None:
    """Return the dialect tag for any kernel-bearing op, or ``None``."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
    from emmy.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
    from emmy.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

    if isinstance(op, CudaOp):
        return "cuda"
    if isinstance(op, KernelOp):
        return "kernel"
    if isinstance(op, TileOp):
        return "tile"
    if isinstance(op, LoopOp):
        return "loop"
    return None


def _is_kernel_bearing(op: object) -> bool:
    """True for any op that represents one kernel of work in the pipeline
    (lowering states from ``LoopOp`` through ``CudaOp``)."""
    return dialect_of(op) is not None


def source_chain(op: object) -> Iterator[object]:
    """Yield ``op`` and every predecessor along ``Op.source``."""
    cur = op
    while cur is not None:
        yield cur
        cur = getattr(cur, "source", None)
