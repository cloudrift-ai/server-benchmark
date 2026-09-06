"""GreedyStrategy — the greedy compile as a search shape (``Pipeline.run``'s orchestration).

The decide callback it resolves with — :func:`~..policy.greedy.greedy_decide` — and the pricing
machinery live in ``policy/greedy.py``: that is the POLICY (what to pick at one fork); this
module is the SHAPE (how many resolves, the blocklist retries, the loud failure).
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from emmy.compiler.context import Context
from emmy.compiler.ir.loop.ir import LoopOp
from emmy.compiler.ir.tile.ir import TileOp
from emmy.compiler.pipeline.pipeline import Decision, LoweringError, Run
from emmy.compiler.pipeline.search.db import SearchDB
from emmy.compiler.pipeline.search.policy.greedy import greedy_decide, logger, tile_identity
from emmy.compiler.pipeline.search.strategy.base import SearchStrategy

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph


# Greedy compile validity-fallback cap: how many times the strategy re-resolves
# blocklisting a pick that left its node un-lowered. Each retry blocks ≥1 fresh
# pick or stops, so this only bounds pathological cases (every sibling unviable).
_MAX_GREEDY_RETRIES = 8


class GreedyStrategy(SearchStrategy):
    """The greedy compile as a search shape over the engine's one loop (``Run.resolve``):
    deterministic resolution with :func:`~..policy.greedy.greedy_decide`, plus the retry
    orchestration that used to live inside the engine —

    * **Validity fallback** — the prior can rank a tile that fails ``validate(ctx)`` first;
      greedy benches nothing, so an un-lowered node blocklists the pick the resolve made at it
      (read off the trace — the terminal node's own knob row can carry stamps a later pass
      added, which no leaf spells) and re-resolves onto the next-ranked leaf. Bounded retries
      (each adds ≥1 block or stops).
    * **Structural retirement** — a refused row blocklists at the piece's own schedule fork (its
      node id is stable across re-resolves), so the composed route replays while the piece
      re-ranks; once no row of the piece binds, the cut that minted it is retired at its own
      fork — that one splice withdrawn, every other kernel-set decision still priced by the
      evidence — and the pieces' blocklists go with it.
    * **Prior-off re-resolve** — when the blocklist budget exhausts, one final resolve without
      the prior (emission-order pick) drops the extrapolation that overflowed; the measured
      arms still decide the kernel-set forks they spell, and ``blocked`` rides along.
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
        for _attempt in range(_MAX_GREEDY_RETRIES):
            rejections: list[tuple[str, str, str]] = []
            run = Run(pipeline=pipeline, ctx=ctx, db=db, backend=backend, dump=dump, rejections=rejections)
            terminal, trace = run.resolve(graph.copy(), greedy_decide(blocked=blocked, db=db))
            stuck = _stuck(terminal, rejections)
            if not stuck or not _retire(blocked, trace, stuck):
                break
        # The prior-ranked tiles all overflowed ``validate(ctx)`` within the retry budget — an
        # *online* prior can extrapolate a large tile onto a small shape, and the blocklist
        # retry exhausts before reaching an in-budget leaf. Re-resolve WITHOUT the prior (the
        # emission-order pick): the point is dropping the extrapolation that overflowed, not
        # the quality of what emission order lands on. When that leaf overflows too the
        # re-resolve stays un-lowered and ``_raise_on_unlowered`` fires below, exactly as
        # before.
        if _stuck(terminal, rejections):
            rejections = []
            run = Run(pipeline=pipeline, ctx=ctx, db=db, backend=backend, dump=dump, rejections=rejections)
            terminal, _ = run.resolve(graph.copy(), greedy_decide(blocked=blocked, prior=None, db=db))
        _raise_on_unlowered(terminal, rejections, ctx)
        logger.info("compile: total %.2fs (deterministic resolve)", time.monotonic() - t_start)
        return terminal


def _stuck(graph: Graph, rejections: list[tuple[str, str, str]]) -> dict[str, tuple[str, str]]:
    """``{node_id: (pass_label, reason)}`` for every node a ``validate(ctx)`` rejection
    (see :func:`Candidate.try_rewrite`) left un-lowered — still a pre-final ``LoopOp`` /
    ``TileOp`` at the terminal: the over-budget tile→kernel drop leaves a knob-stamped
    ``TileOp``, a pre-tile drop a ``LoopOp``. A node a later rule lowered anyway is a
    harmless intermediate filter, and partial pipelines that legitimately terminate at the
    loop / tile stage record no rejection at all. The last recorded rejection of a node
    wins (the final pass that tried to lower it)."""
    last = {nid: (pass_label, reason) for nid, pass_label, reason in rejections}
    return {nid: why for nid, why in last.items() if (node := graph.nodes.get(nid)) is not None and isinstance(node.op, (LoopOp, TileOp))}


def _retire(blocked: dict[str, set[frozenset]], trace: list[Decision], stuck: dict[str, tuple[str, str]]) -> bool:
    """Blocklist, for every un-lowered node, the pick that produced it: the last decision the
    resolve made at that node, so a refused row re-ranks its own piece. Once that pick is
    already blocked no row of the piece binds, and the cut that minted the piece is retired at
    its own fork instead — its pieces' blocklists dropped with it, since the node ids they keyed
    now host other kernels. Returns whether anything new was blocked; otherwise the retry has
    nothing left to change."""
    changed = False
    for nid, (pass_label, reason) in stuck.items():
        own = [d for d in trace if d.node_id == nid]
        pick = own[-1] if own else None
        if pick is None or not _block(blocked, pick):
            pick = next((d for d in reversed(trace) if d.chosen_kind == "graph" and nid in d.minted), None)
            if pick is None or not _block(blocked, pick):
                continue
        retired = pick.chosen_kind == "graph"
        logger.log(
            logging.WARNING if retired else logging.INFO,
            "compile: %r left un-lowered (%s: %s) — %s",
            nid,
            pass_label,
            reason,
            f"retiring the cut that minted it at {pick.node_id!r}" if retired else "re-ranking it past the refused row",
        )
        changed = True
    return changed


def _block(blocked: dict[str, set[frozenset]], pick: Decision) -> bool:
    """Blocklist ``pick``'s identity at its node — the same ``tile_identity`` the decide applies
    to a leaf's knob row and to a splice's decision knobs, so the blocklist matches on replay.
    ``False`` when it already was blocked."""
    ident = tile_identity(pick.knob_delta)
    if ident in blocked.get(pick.node_id, ()):
        return False
    for minted in pick.minted:
        blocked.pop(minted, None)
    blocked.setdefault(pick.node_id, set()).add(ident)
    return True


def _raise_on_unlowered(graph: Graph, rejections: list[tuple[str, str, str]], ctx) -> None:
    """Fail a greedy compile loudly when a recorded ``validate(ctx)`` rejection left its node
    un-lowered (:func:`_stuck`) instead of leaking the pre-final op to the backend."""
    stuck = _stuck(graph, rejections)
    if not stuck:
        return
    lines = [f"  - {nid!r}: {pass_label} rejected its only lowering — {reason}" for nid, (pass_label, reason) in stuck.items()]
    raise LoweringError(
        f"compile: {len(stuck)} node(s) left un-lowered — the chosen tile shape produced a kernel that "
        f"failed validate(ctx) and the deterministic compile had no fallback:\n"
        + "\n".join(lines)
        + "\nPin a fitting tile via EMMY_KNOBS, raise the smem budget, or adjust tile-geometry "
        "scoring so an in-budget variant ranks first."
    )
