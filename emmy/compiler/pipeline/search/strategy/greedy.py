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
    * **Loud failure** — a node the settled terminal left un-lowered raises
      :class:`~emmy.compiler.pipeline.pipeline.LoweringError` instead of a downstream
      ``CudaBackend`` mystery. What counts as un-lowered is the whole terminal when the
      pipeline runs to the end of lowering (:attr:`Pipeline.lowers_to_cuda`), not just the
      nodes a rule recorded a rejection for — a materializer that declines a row with an
      ordinary ``RuleSkipped`` records nothing and used to escape every one of the fallbacks
      above, returning a half-lowered graph the compile reported as a success.
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

        # Only a pipeline that runs to the final lowering pass promises a Graph[CudaOp]; a
        # truncated build terminates in an earlier dialect, where a surviving tile is the answer.
        complete = pipeline.lowers_to_cuda
        blocked: dict[str, set[frozenset]] = {}
        allow_structural = True
        for _attempt in range(_MAX_GREEDY_RETRIES):
            rejections: list[tuple[str, str, str]] = []
            run = Run(pipeline=pipeline, ctx=ctx, db=db, backend=backend, dump=dump, rejections=rejections)
            decide = greedy_decide(blocked=blocked, price_structural=allow_structural, db=db)
            terminal, trace = run.resolve(graph.copy(), decide)
            failed = _unlowered_tiles(terminal, rejections, lowers_to_cuda=complete)
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
            # Say so: the keep-fused re-resolve is correct but slower, and a silent retirement
            # leaves a deploy on a kernel set with nothing in the log to explain it.
            logger.warning(
                "compile: the priced kernel-set change left %s un-lowered (%s) — retiring structural picks and "
                "re-resolving down the keep-fused branch",
                ", ".join(repr(nid) for nid in failed),
                "; ".join(reason for _nid, _label, reason in rejections) or "no rule recorded a decline",
            )
            allow_structural = False
        # The prior-ranked tiles all overflowed ``validate(ctx)`` within the retry budget — an
        # *online* prior can extrapolate a large tile onto a small shape, and the blocklist
        # retry exhausts before reaching an in-budget leaf. Re-resolve WITHOUT the prior (the
        # emission-order pick): the point is dropping the extrapolation that overflowed, not
        # the quality of what emission order lands on. When that leaf overflows too the
        # re-resolve stays un-lowered and ``_raise_on_unlowered`` fires below, exactly as
        # before.
        if _unlowered_tiles(terminal, rejections, lowers_to_cuda=complete):
            rejections = []
            run = Run(pipeline=pipeline, ctx=ctx, db=db, backend=backend, dump=dump, rejections=rejections)
            terminal, _ = run.resolve(graph.copy(), greedy_decide(blocked=blocked, prior=None, price_structural=False))
        _raise_on_unlowered(terminal, rejections, lowers_to_cuda=complete)
        logger.info("compile: total %.2fs (deterministic resolve)", time.monotonic() - t_start)
        return terminal


def _unlowered_tiles(graph: Graph, rejections: list[tuple[str, str, str]], *, lowers_to_cuda: bool) -> dict[str, frozenset]:
    """``{node_id: tile_identity}`` for every node the resolution left un-lowered —
    still a pre-final ``LoopOp`` / ``TileOp`` at the terminal. The ``tile_identity`` is
    the offending op's knobs, what the retry loop blocklists so the fallback lands on the
    next prior-ranked sibling.

    ``lowers_to_cuda`` (:attr:`Pipeline.lowers_to_cuda`) decides which nodes are eligible.
    A pipeline that runs to the end of lowering promises a ``Graph`` of ``CudaOp``, so
    EVERY surviving tile is stranded, whether or not a rule recorded a rejection: a rule
    whose materializer declined with an ordinary ``RuleSkipped``, or that never matched,
    strands the node just as thoroughly as an all-options-filtered one and records
    nothing. A truncated pipeline terminates in an earlier dialect by design, so there
    only a node with a recorded rejection counts."""
    stuck = graph.nodes.keys() if lowers_to_cuda else {nid for nid, _pass_label, _reason in rejections}
    out: dict[str, frozenset] = {}
    for nid in stuck:
        node = graph.nodes.get(nid)
        if node is not None and isinstance(node.op, (LoopOp, TileOp)):
            # Key through ``tile_identity`` — the SAME canonicalization ``_tile_blocked``
            # applies to a leaf's fork knobs, so the blocklist actually matches on retry.
            out[nid] = tile_identity(getattr(node.op, "knobs", None) or {})
    return out


def _raise_on_unlowered(graph: Graph, rejections: list[tuple[str, str, str]], *, lowers_to_cuda: bool) -> None:
    """Fail a greedy compile loudly when the settled terminal still holds a pre-final
    dialect op (``LoopOp`` or ``TileOp``) — see :func:`_unlowered_tiles` for which nodes
    are eligible under ``lowers_to_cuda``.

    ``rejections`` is ``[(node_id, pass_label, reason), ...]``, the sink
    :meth:`Candidate.try_rewrite` fills. A node that has one gets the pass and reason that
    declined it; a node stranded silently is reported by the op it is stuck on. If a later
    rule lowered a rejected node anyway its op is a ``KernelOp`` / ``CudaOp`` and we stay
    silent — that rejection was a harmless intermediate filter."""
    stuck = sorted(_unlowered_tiles(graph, rejections, lowers_to_cuda=lowers_to_cuda))
    if not stuck:
        return

    # Last recorded reason / pass wins (the final pass that tried to lower it).
    declined: dict[str, tuple[str, str]] = {nid: (pass_label, reason) for nid, pass_label, reason in rejections}
    lines = [
        f"  - {nid!r}: {declined[nid][0]} rejected its only lowering — {declined[nid][1]}"
        if nid in declined
        else f"  - {nid!r}: no lowering rule produced a kernel — still {type(graph.nodes[nid].op).__name__}"
        for nid in stuck
    ]
    raise LoweringError(
        f"compile: {len(stuck)} node(s) left un-lowered — the deterministic compile exhausted its "
        f"fallbacks and has no kernel for them:\n"
        + "\n".join(lines)
        + "\nPin a fitting tile via EMMY_KNOBS, raise the smem budget, or adjust tile-geometry "
        "scoring so a variant this lowering accepts ranks first."
    )
