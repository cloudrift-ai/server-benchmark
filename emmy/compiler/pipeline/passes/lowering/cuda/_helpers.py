"""Shared helpers for the ``lowering/cuda`` rules.

Lives in a ``_``-prefixed module so the pass loader skips it (only ``NNN_<name>.py`` files load
as rules); ``005_delegate_zero_init`` and ``010_lower_kernelop`` both need the atomic-output
walk, so it is defined once here (the ``loop/fusion/_helpers`` precedent).
"""

from __future__ import annotations

from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import RegStore
from emmy.compiler.ir.stmt import RowAccum, Write


def atomic_outputs(kernel: KernelOp) -> tuple[str, ...]:
    """Output buffers an atomic reduce-write (``030_split_reduce``'s atomic finalize) accumulates
    into — they must be zero-init'd before each launch (``CudaOp.zero_outputs``), since every
    contributing CTA ``atomicAdd``\\ s into the same cell. The scalar tier's atomic ``Write``
    survives materialization verbatim; the mma tier's became a ``RegStore(atomic=True)``; a
    ``RowAccum`` (the stat-sink epilogue) accumulates its aux row-stat buffer the same way.
    Dict-keyed for stable order."""
    seen: dict[str, None] = {}
    for s in kernel.body.iter():
        if isinstance(s, Write) and s.atomic:
            seen.setdefault(s.output, None)
        elif isinstance(s, RegStore) and s.atomic:
            seen.setdefault(s.dst_buffer, None)
        elif isinstance(s, RowAccum):
            seen.setdefault(s.dst, None)
    return tuple(seen)
