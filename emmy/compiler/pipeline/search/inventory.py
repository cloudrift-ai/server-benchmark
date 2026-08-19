"""KernelInventory — the tuner's live kernel roster, as a run-scoped engine-event observer.

The two-level tuner's outer terminals enumerate the kernels that exist when the fused kernel set
settles — but kernels minted DURING the inner loops (a placement cut's fragments, a cross-CTA
split's pieces) were never in that enumeration, so they could not be first-class tuning targets
or golden identities; they existed only inside their parent slice's Σ. The inventory closes that
gap: installed on every inner run (``tune_async(strategies=(inventory,))``), it watches every
Graph splice and hands each genuinely NEW kernel to the enrolling strategy.

Cross-trajectory by design: the MCTS re-minting the same cut on every variant reports it once —
the seen-set spans the whole tune session, seeded with the outer terminal's kernels so pieces
structurally identical to an outer kernel are not re-enrolled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline.strategy import PipelineStrategy, SpliceEvent

if TYPE_CHECKING:
    from collections.abc import Callable

    from emmy.compiler.pipeline.passes.identity import IdentityStrategy


class KernelInventory(PipelineStrategy):
    """Reports each new loop-dialect kernel — one whose structural identity has not been seen —
    to ``on_kernel(node_id, op, fragment)``. Identity is COMPUTED through the
    :class:`IdentityStrategy`'s read API, so nothing here depends on a stamp having happened or
    on strategy dispatch order."""

    def __init__(self, identity: IdentityStrategy, on_kernel: Callable, seen: set[str] | None = None) -> None:
        self.identity = identity
        self.on_kernel = on_kernel
        self.seen = seen if seen is not None else set()

    def on_splice(self, e: SpliceEvent) -> None:
        for nid, node in e.fragment.nodes.items():
            op = node.op
            if not isinstance(op, LoopOp):
                continue
            key = self.identity.op_sig(op, e.fragment)
            if key in self.seen:
                continue
            self.seen.add(key)
            self.on_kernel(nid, op, e.fragment)
