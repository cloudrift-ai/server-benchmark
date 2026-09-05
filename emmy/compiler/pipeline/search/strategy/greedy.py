"""GreedyStrategy — the greedy compile as a search shape (``Pipeline.run``'s orchestration).

The decide callback it resolves with — :func:`~..policy.greedy.greedy_decide` — and the pricing
machinery live in ``policy/greedy.py``: that is the POLICY (what to pick at one fork); this
module is the SHAPE (how many resolves, the blocklist retries, the loud failure).
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING

from emmy.compiler.context import Context
from emmy.compiler.ir.loop.ir import LoopOp
from emmy.compiler.ir.tile.ir import TileOp
from emmy.compiler.pipeline.pipeline import LoweringError, Run
from emmy.compiler.pipeline.search.db import SearchDB
from emmy.compiler.pipeline.search.policy.greedy import greedy_decide, logger, tile_identity
from emmy.compiler.pipeline.search.strategy.base import SearchStrategy

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph


# Greedy compile validity-fallback cap: how many times the strategy re-resolves
# blocklisting a tile that failed ``validate(ctx)``. Each retry blocks ≥1 fresh
# tile or stops, so this only bounds pathological cases (every sibling unviable).
_MAX_GREEDY_RETRIES = 8


class GreedyStrategy(SearchStrategy):
    """The greedy compile as a search shape over the engine's one loop (``Run.resolve``):
    deterministic resolution with :func:`~..policy.greedy.greedy_decide`, plus the retry
    orchestration that used to live inside the engine —

    * **Validity fallback** — the prior can rank a tile that fails ``validate(ctx)`` first;
      greedy benches nothing, so an un-lowered node blocklists its tile and re-resolves onto the
      next prior-ranked leaf. Bounded retries (each adds ≥1 block or stops).
    * **Structural retirement** — a fragment kernel's refused row blocklists at the piece's own
      schedule fork (its node id is stable across re-resolves), so the composed route replays
      while the piece re-ranks; only once no row of the piece binds are structural picks retired
      wholesale (``price_structural=False``) — a fragment can't be blocklisted at the fork site.
    * **Prior-off re-resolve** — when the blocklist budget exhausts, one final resolve without
      the prior (emission-order pick) drops the extrapolation that overflowed; the recorded
      goldens still floor it and ``blocked`` rides along.
    * **Loud failure** — a rejection that left its node un-lowered raises
      :class:`~emmy.compiler.pipeline.pipeline.LoweringError` instead of a downstream
      ``CudaBackend`` mystery.
    """

    def __init__(self, pipeline, *, backend=None, db=None, dump=None) -> None:
        self.pipeline = pipeline
        self.backend = backend
        self.db = db
        self.dump = dump

    def run(self, graph: Graph, ctx=None) -> Graph:
        pipeline, backend, dump = self.pipeline, self.backend, self.dump
        if ctx is None:
            ctx = Context.probe()
        backend_name = getattr(backend, "name", "cuda")
        if ctx.backend_name != backend_name:
            ctx = replace(ctx, backend_name=backend_name)
        db = self.db if self.db is not None else SearchDB()
        t_start = time.monotonic()

        blocked: dict[str, set[frozenset]] = {}
        allow_structural = True
        for _attempt in range(_MAX_GREEDY_RETRIES):
            rejections: list[tuple[str, str, str]] = []
            run = Run(pipeline=pipeline, ctx=ctx, db=db, backend=backend, dump=dump, rejections=rejections)
            decide = greedy_decide(blocked=blocked, price_structural=allow_structural, db=db)
            terminal, trace = run.resolve(graph.copy(), decide)
            failed = _unlowered_tiles(terminal, rejections)
            if not failed:
                break
            new = {nid: ident for nid, ident in failed.items() if ident not in blocked.get(nid, set())}
            if new:  # a refused row re-ranks its own piece; every other pick replays
                for nid, ident in new.items():
                    blocked.setdefault(nid, set()).add(ident)
                continue
            # A re-pick of a blocked row: nothing of the piece binds → revisit a structural pick, once.
            if not (allow_structural and any(d.chosen_kind == "graph" for d in trace)):
                break
            allow_structural = False
        # The prior-ranked tiles all overflowed ``validate(ctx)`` within the retry budget — an
        # *online* prior can extrapolate a large tile onto a small shape, and the blocklist
        # retry exhausts before reaching an in-budget leaf. Re-resolve WITHOUT the prior (the
        # emission-order pick): the point is dropping the extrapolation that overflowed, not
        # the quality of what emission order lands on. When that leaf overflows too the
        # re-resolve stays un-lowered and ``_raise_on_unlowered`` fires below, exactly as
        # before.
        if _unlowered_tiles(terminal, rejections):
            rejections = []
            run = Run(pipeline=pipeline, ctx=ctx, db=db, backend=backend, dump=dump, rejections=rejections)
            terminal, _ = run.resolve(graph.copy(), greedy_decide(blocked=blocked, prior=None, price_structural=False))
        _raise_on_unlowered(terminal, rejections, ctx)
        logger.info("compile: total %.2fs (deterministic resolve)", time.monotonic() - t_start)
        return terminal


def _unlowered_tiles(graph: Graph, rejections: list[tuple[str, str, str]]) -> dict[str, frozenset]:
    """``{node_id: tile_identity}`` for every node a ``validate(ctx)`` rejection
    left un-lowered (still a pre-final ``LoopOp`` / ``TileOp`` at the terminal — the
    over-budget tile→kernel drop leaves a knob-stamped ``TileOp``, a pre-tile drop a
    ``LoopOp``, mirroring :func:`_raise_on_unlowered`'s stuck set). The
    ``tile_identity`` is the offending op's knobs — what the retry loop blocklists
    so the greedy fallback lands on the next prior-ranked sibling."""
    if not rejections:
        return {}

    out: dict[str, frozenset] = {}
    for nid, _pass_label, _reason in rejections:
        node = graph.nodes.get(nid)
        if node is not None and isinstance(node.op, (LoopOp, TileOp)):
            # Key through ``tile_identity`` — the SAME canonicalization ``_tile_blocked``
            # applies to a leaf's fork knobs, so the blocklist actually matches on retry.
            out[nid] = tile_identity(getattr(node.op, "knobs", None) or {})
    return out


def _raise_on_unlowered(graph: Graph, rejections: list[tuple[str, str, str]], ctx) -> None:
    """Fail a greedy compile loudly when a recorded ``validate(ctx)``
    rejection (see :func:`Candidate.try_rewrite`) left its node un-lowered.

    ``rejections`` is ``[(node_id, pass_label, reason), ...]``. A node is
    "stuck" iff it still holds a pre-final dialect op (``LoopOp`` or ``TileOp``)
    at the terminal — if a later rule lowered it anyway the op is a ``KernelOp`` /
    ``CudaOp`` and we stay silent (the rejection was a harmless intermediate
    filter). The over-budget-tile drop leaves a ``TileOp`` (its only
    tile→kernel lowering was filtered); a pre-tile drop leaves a ``LoopOp``. Only
    nodes with a recorded rejection are checked, so partial pipelines that
    legitimately terminate at the loop / tile stage (no lowering pass to drop
    anything) never trip this."""
    if not rejections:
        return

    # Last recorded reason / pass wins (the final pass that tried to lower it).
    reason_by_node: dict[str, str] = {}
    pass_by_node: dict[str, str] = {}
    for nid, pass_label, reason in rejections:
        reason_by_node[nid] = reason
        pass_by_node[nid] = pass_label

    stuck = [nid for nid in reason_by_node if (node := graph.nodes.get(nid)) is not None and isinstance(node.op, (LoopOp, TileOp))]
    if not stuck:
        return
    lines = [f"  - {nid!r}: {pass_by_node[nid]} rejected its only lowering — {reason_by_node[nid]}" for nid in stuck]
    raise LoweringError(
        f"compile: {len(stuck)} node(s) left un-lowered — the chosen tile shape produced a kernel that "
        f"failed validate(ctx) and the deterministic compile had no fallback:\n"
        + "\n".join(lines)
        + "\nPin a fitting tile via EMMY_KNOBS, raise the smem budget, or adjust tile-geometry "
        "scoring so an in-budget variant ranks first."
    )
