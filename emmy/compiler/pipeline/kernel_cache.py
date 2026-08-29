"""Session-scoped reuse of fully-lowered kernels — fetch instead of re-lowering.

Greedy lowering of one kernel is a function: the fused Loop-IR program in, the lowered
``KernelOp`` out (the schedule pick baked in). This cache memoizes that function at its natural
boundary, so a compile that meets a kernel it has lowered before — the same computation under
different buffer names, in another program of the same session — splices the finished artifact
instead of enumerating, deciding and materializing again.

The boundary is deliberately the END OF THE KERNEL PASSES, before ``lowering/cuda``: the cuda
pass group holds the per-graph negotiations (``005_delegate_zero_init`` injects a
``ZeroPrologue`` into a *neighbor* kernel; ``010`` renders per-graph buffer names), so artifacts
above the boundary stay a pure function of the kernel while the negotiations run fresh on every
assembled graph. Entries store the ``KernelOp`` with its buffers renamed to positional slots
(``Stmt.rename_buffers``); a hit renames the slots to the consumer's io — sound because the key
folds the io fingerprint, so the io orders correspond positionally.

The key is the exact variant key (``identity_key(structural=False, with_io=True,
with_knobs=True)`` — an artifact is exact code, never a cluster representative) folded with the
symbolic-dim hints and the live pin fingerprint: everything the pick depended on besides the
evidence. Evidence is the SESSION-freeze trade, exactly :mod:`~emmy.compiler.backend.plan_cache`'s:
the cache is process-local and CALLER-owned (installed on ``Context.kernel_cache`` by whoever
owns the session — nothing installs it by default), so one cache spans one compiler/evidence
view and no second persistent validity contract exists. Greedy-only: the tune search strips the
field (``prepare_ctx``) because an artifact bakes a conclusion and exploration must not inherit
conclusions.

A kernel whose lowering produced SEVERAL kernels (a cut, a cross-CTA split) poisons its key:
the artifact of a multi-kernel origin is a graph fragment, deliberately out of scope until the
single-kernel path has earned it.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace

from emmy.compiler.dim import DEFAULT_SEQ_HINT
from emmy.compiler.structural import digest

#: A key whose origin lowered to more than one kernel — never served, never overwritten.
POISON = object()


@dataclass(frozen=True)
class _Entry:
    """One cached lowering: the slot-renamed ``KernelOp``, the slot order (origin inputs then
    outputs — the order a consumer's io maps back onto), and the minting origin's ``id()``
    (how the harvest tells a same-origin second kernel — poison — from an identical twin)."""

    kernel: object
    slots: tuple[str, ...]
    origin_id: int


class KernelCache:
    """``variant key ⊕ hints ⊕ pins → lowered KernelOp`` — LRU, process-local, caller-owned."""

    def __init__(self, cap: int = 128) -> None:
        self.cap = cap
        self._store: OrderedDict[str, object] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(op) -> str | None:
        """The cache key of one fused Loop-IR kernel — ``None`` when the op offers no identity.
        The ONE spelling of the composition (consult and harvest both call this)."""
        from emmy.compiler.pipeline.knob import schedule_pin_fingerprint  # noqa: PLC0415

        identity = op.identity_key(structural=False, with_io=True, with_knobs=True)
        if identity is None:
            return None
        io = (*op.inputs.values(), *op.outputs.values())
        hints = tuple(d.hint or DEFAULT_SEQ_HINT for t in io for d in t.shape if not d.is_static)
        return digest(identity, hints, schedule_pin_fingerprint())

    def fetch(self, key: str, target) -> object | None:
        """The cached lowering rebound to ``target``'s buffer names, or ``None`` (miss /
        poisoned). ``target`` is the consuming fused Loop-IR op; its io orders correspond to the
        stored slots positionally because the key folds the io fingerprint."""
        entry = self._store.get(key)
        if entry is None or entry is POISON:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        names = (*target.inputs, *target.outputs)
        kernel = replace(entry.kernel, body=entry.kernel.body.rename_buffers(dict(zip(entry.slots, names, strict=True))))
        kernel.knobs = dict(entry.kernel.knobs)
        kernel.inputs = dict(target.inputs)
        kernel.outputs = dict(target.outputs)
        return kernel

    def harvest(self, key: str, kernel, origin) -> None:
        """Record one lowered kernel under its origin's key. A second, different kernel from the
        SAME origin means the lowering was multi-kernel — the key poisons. An equal key from a
        DIFFERENT origin is an identical twin: the stored artifact already covers it."""
        held = self._store.get(key)
        if held is POISON:
            return
        if held is not None:
            if held.origin_id == id(origin) and held.kernel is not kernel:
                self._store[key] = POISON
            return
        names = (*origin.inputs, *origin.outputs)
        decls = {n for s in kernel.body.iter() for n in s.local_decls()}
        buffers = {n for s in kernel.body.iter() for n in (*s.external_reads(), *s.external_writes())} - decls
        if not buffers <= set(names):
            return  # the kernel reads buffers beyond its origin's io — not a pure single-kernel lowering
        slots = tuple(f"__kc{i}" for i in range(len(names)))
        slotted = replace(kernel, body=kernel.body.rename_buffers(dict(zip(names, slots, strict=True))))
        slotted.inputs, slotted.outputs = {}, {}
        self._store[key] = _Entry(kernel=slotted, slots=slots, origin_id=id(origin))
        while len(self._store) > self.cap:
            self._store.popitem(last=False)
