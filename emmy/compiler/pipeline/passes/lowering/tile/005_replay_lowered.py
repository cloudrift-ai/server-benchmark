"""Fetch a fused kernel's finished lowering from the session kernel cache — the greedy shortcut.

Fusion has settled the kernel boundary by the time ``lowering/tile`` starts, so this rule sits
before the lift: a ``LoopOp`` whose exact variant key (+ hints + pins) was already lowered this
session rewrites STRAIGHT to the cached, io-rebound ``KernelOp`` — no lift, no enumeration, no
fork, no materialization. Everything downstream treats it as any other kernel: the tile and
kernel passes don't match a ``KernelOp``, and the ``lowering/cuda`` group (zero-init delegation,
rendering) runs fresh on the assembled graph, per graph, as it must (those are the cross-kernel
negotiations the cache boundary deliberately excludes).

Fires only when the caller installed ``Context.kernel_cache`` (see
:mod:`~emmy.compiler.pipeline.kernel_cache` — caller-owned, greedy-only; the tune search strips
the field, and the pricing probes strip it too, since a replayed kernel offers no fork to
price)."""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.kernel_cache import KernelCache

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node, ctx=None):  # -> the cached KernelOp (kernel IR stays unimported here)
    del match
    cache: KernelCache | None = getattr(ctx, "kernel_cache", None)
    if cache is None:
        raise RuleSkipped("no session kernel cache installed")
    key = cache.key_for(root.op)
    kernel = cache.fetch(key, root.op) if key is not None else None
    if kernel is None:
        raise RuleSkipped("kernel not cached")
    return kernel
