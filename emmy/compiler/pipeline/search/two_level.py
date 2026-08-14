"""Two-level autotuning: an outer kernel-placement MCTS whose terminal reward comes from
an inner, separable, per-op search.

``emmy tune`` used to run one SP-MCTS tree over the whole graph. Because
the pipeline applies rules sequentially, op-variant forks (tile / pad / stage
choices for one kernel) and placement forks (which closed seams become kernel boundaries) **nest**
and cross-product under one global patience — so the bottleneck op starves the
deep ops (see ``project_mcts_exploration_limit``). The two kinds of decision
have opposite structure:

- **op-variant forks are separable** — every multi-option fork today is an
  in-place ``Op`` rebind that leaves the graph unchanged, so the whole-graph
  time is ``Σ_k t_k`` with each ``t_k`` depending only on op ``k``'s variant;
- **placement forks are NOT separable** — they change *which kernels exist*.

So we split the search in two, drawing the boundary on the fork's *effect*
(the ``Op``-rebind / ``Graph``-splice classification stamped at the engine's
spawn site), not on a fixed
pass index:

- **Outer** (:func:`run_two_level_tune`) drives ``frontend`` + ``loop`` and the
  recognition half of ``lowering/tile`` (:func:`outer_pipeline`). Fusion stays
  deterministic and maximal; recognition offers ``PLACE`` choices that either
  keep that region or materialize one closed seam. A terminal has every
  placement decision resolved and every kernel represented by an unmapped
  ``TileOp``. Each terminal is a candidate kernel set;
  its reward is ``1 / Σ best-per-op time`` from the inner search,
  backpropagated by the reused :class:`TuningSearch` — so keep-vs-split is an
  outer-terminal comparison, the natural cost model for a kernel-set decision.
  Identical offer sites within a trajectory take the same side — the engine
  replays the decision read off the trajectory's own graph via the
  ``Op.source`` decomposition links and the stamped decision knobs
  (``pipeline._replay_structural_decision``) — keeping the tree linear in
  *unique* kernels. Fusion itself remains deterministic and maximal: it does
  not consult placement, schedule, hardware, or search evidence.
- **Inner** (:func:`_inner_reward_async`) tunes each finalized kernel *independently*
  in its own single-node slice (:func:`single_node_graph`) with a plain
  :class:`TuningSearch` over :data:`LOWERING_PASSES` only. Results key
  structurally (:meth:`~emmy.compiler.ir.base.Op.cache_key`), so they transfer to the assembled graph
  unchanged AND are shared across outer terminals (a shared op is a DB hit).

The inner search runs for **every** op on every pass — it is never skipped on
prior effort. Replay is cheap, not gated: each benched terminal hits the
per-variant ``perf`` cache (:class:`pipeline._TerminalBench`), so a variant
already measured is served from the DB with no GPU bench. An identical re-run
(same prior) re-walks the same deterministic trajectory → every terminal is a
cache hit → zero benches and the same total. But the global online prior keeps
changing (it refits across ops and runs), so the same patience can steer the
MCTS down a *different* trajectory — re-running lets it reach and bench the
genuinely-new variants the improved prior surfaces, replaying the rest for free.
(The old per-op ``op_effort`` "skip already-tuned" gate is gone: it skipped the
whole op, which would suppress exactly that prior-driven re-exploration.)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pass, Pipeline, TuningSearch
from emmy.compiler.pipeline.search.db import PerfStats, SearchDB
from emmy.compiler.pipeline.search.policy.mcts import O3_NVCC_FLAGS
from emmy.compiler.pipeline.search.slice import single_node_graph
from emmy.compiler.structural import digest

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.graph import Graph

logger = logging.getLogger(__name__)

# The structural / op-variant boundary lies between tile recognition and tile
# scheduling. Recognition owns the PLACE fork because it changes the kernel set;
# scheduling and every later lowering fork remain separable per kernel.

# Lowering-only passes after loop fusion: ``recognize → schedule → kernel → cuda``.
# The inner per-op search receives an already-recognized ``TileOp`` slice, so
# ``010_recognize`` cannot re-open its outer-owned placement decision. The
# recognized algebra — and thus its ``Op.cache_key`` — is never re-touched by
# ``loop/fusion``; inner ``perf`` / ``lowering`` rows therefore transfer to the
# assembled graph. Slicing this as the tail of ``CUDA_PASSES`` keeps it aligned
# with pass-list edits and retains the LoopOp guardrail for unrecognized algebra.
LOWERING_PASSES = CUDA_PASSES[len(LOOP_PASSES) :]


def outer_pipeline() -> Pipeline:
    """Drive maximal fusion, algebra recognition, and structural placement only.

    ``010_recognize`` lifts each fused ``LoopOp`` to an unmapped ``TileOp`` and
    offers the fused-vs-materialized ``PLACE`` fork. A cut's new ``LoopOp`` pieces
    re-enter that same rule until the complete kernel set is recognized. The outer
    pipeline stops before ``020_schedule``: tile/reduce/stage choices are inner,
    per-kernel decisions, including graph splices caused by a selected reduce plan.
    """
    passes = [Pass.load(name, i) for i, name in enumerate(LOOP_PASSES)]
    passes.append(Pass.load("lowering/tile", len(passes), {"010_recognize"}))
    return Pipeline(passes=passes)


# Per-op latency stand-in when the inner search produced no clean ``ok``
# measurement — large enough to sink the outer reward, finite so the Σ stays a
# real number for the separability report.
_FAIL_US = 1e12


@dataclass
class OpResult:
    """One unique kernel's inner-search outcome, for the per-op summary.

    ``multiplicity`` is the number of structurally-identical ``LoopOp`` nodes
    in the fused graph that share this ``op_key`` — 24 for a 24-layer
    RMSNorm, 1 for a singleton. The outer reward's ``total_us`` weights
    ``best_us`` by ``multiplicity`` so the Σ across ``per_op`` equals the
    whole-graph latency (every node position counts, even though dedup
    means we only run the inner search and DB lookup once per key).
    """

    name: str
    op_key: str
    best_us: float | None
    multiplicity: int = 1
    # Fastest directly observed terminal from this invocation. Kept separate
    # from ``best_us`` because that aggregate can come from older DB evidence.
    searched_knobs: dict[str, str] | None = None
    searched_us: float | None = None
    searched_cuda_ops: int | None = None


@dataclass
class InnerReward:
    """Result of evaluating one outer terminal: ``Σ best-per-op time``."""

    total_us: float
    ok: bool  # every kernel had a clean ``ok`` measurement
    per_op: list[OpResult] = field(default_factory=list)
    # Online-prior end-of-run sanity block(s) — printed by the command after
    # the progress bar closes.
    prior_summaries: list[str] = field(default_factory=list)

    def searched_winner(self) -> tuple[dict[str, str], float] | None:
        """The actual searched winner when it maps to exactly one CUDA kernel.

        A target can contain repeated/heterogeneous post-fusion kernels, and one
        post-fusion kernel can lower to several CudaOps. In either case there is
        no single golden-format knob row whose latency represents the target, so
        callers must omit a winner annotation instead of inventing one.
        """
        if len(self.per_op) != 1:
            return None
        op = self.per_op[0]
        if op.multiplicity != 1 or op.searched_cuda_ops != 1 or op.searched_knobs is None or op.searched_us is None:
            return None
        return dict(op.searched_knobs), op.searched_us


@dataclass
class TwoLevelResult:
    """Outcome of :func:`run_two_level_tune`."""

    best_fused: Graph | None  # winning fused graph (finalized LoopOps)
    best_reward: InnerReward | None  # its Σ-per-op breakdown
    n_terminals: int  # outer terminals evaluated (1 today)
    assembled: Graph | None  # greedy DB-best Graph[CudaOp] assembled from the bests
    prior_summaries: list[str] = field(default_factory=list)  # online-prior stats


def _point_stats(us: float) -> PerfStats:
    """A degenerate :class:`PerfStats` carrying a single aggregate value
    (``n_samples=0`` marks it as a derived total, not a raw sample set)."""
    return PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=0)


def _mint_run_id() -> str:
    """A sortable, unique tune-session id stamped on this run's ``node`` rows —
    UTC timestamp + a uuid tail (two sessions in the same second stay distinct)."""
    from datetime import UTC, datetime  # noqa: PLC0415
    from uuid import uuid4  # noqa: PLC0415

    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"


def _kernel_nodes(graph: Graph) -> list[tuple[str, object]]:
    """Outer-terminal kernels — one ``(node_id, TileOp)`` per recognized kernel."""
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415

    # ``LoopOp`` is retained as a guardrail for algebra that recognition cannot
    # lift (for example a symbolic loop). Its inner slice retries the complete
    # tile pass and follows the same lowering failure contract as before.
    return [(nid, n.op) for nid, n in graph.nodes.items() if isinstance(n.op, (LoopOp, TileOp))]


async def _inner_reward_async(
    fused_graph,
    *,
    ctx,
    db,
    pool,
    patience,
    ucb_c,
    explore_eps,
    seed,
    progress,
    prior,
    run_id: str = "",
    max_candidates: int | None = None,
    backend_slots=None,
    close_backends: bool = True,
) -> InnerReward:
    """Tune every post-fusion kernel of ``fused_graph`` in its own single-node slice
    and return ``Σ best-per-op time`` — the outer terminal reward.

    One coroutine per unique kernel over a slot queue of ``len(pool)`` device-pinned
    :class:`CudaBackend`s (one in-flight bench per slot), each op's inner search
    running on its slot via :meth:`Pipeline.tune_async`. Single event loop, single
    thread — the shared ``db`` / ``prior`` are touched only between bench ``await``s,
    so they're atomic with no locks. A single-element ``pool`` is the serial case:
    coroutines acquire the lone worker in ``op_idx`` order → strictly sequential.

    ``prior`` (a single shared :class:`~emmy.compiler.pipeline.search.prior.Prior`,
    or ``None``) drives every inner search's PUCT — ONE **global** model across all
    kernels: each op's search trains it on ``archived + this op's tree``; when the op
    finishes its rows are archived and the prior is checkpointed (keyed by regime), so
    a later compile / tune reloads it.

    Every kernel's slice is tuned by a plain inner :class:`TuningSearch` over
    :data:`LOWERING_PASSES` on every pass — never skipped on prior effort. The cost is
    paid at the bench, not gated at the op: :class:`pipeline._TerminalBench` serves any
    already-measured variant from the ``perf`` cache, so an identical re-run benches
    nothing while a prior-shifted trajectory benches only its genuinely-new variants.
    Benches scale as ``Σ_k n_k`` (per op), never the product.

    ``progress`` (a duck-typed :class:`~emmy.commands.tune_progress.TuneProgress`,
    or ``None``) drives the CLI progress bar: one op leaf ticked per *unique* kernel,
    the live tail updated per benched variant. Leaves are deduped by ``Op.cache_key``
    before iteration; multiplicity is preserved so ``total_us`` weights each unique
    kernel's best by its node count (order-stable, identical to per-node iteration).

    ``max_candidates`` caps live measurements per kernel; cached observations do
    not consume it. ``backend_slots`` and ``close_backends=False`` let several
    independent working-file targets share one command-owned device queue.
    """
    from collections import OrderedDict  # noqa: PLC0415

    from emmy.compiler.pipeline.passes.lowering.tile._flash import fused_producer_ids  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import variant_label  # noqa: PLC0415

    ctx_key = ctx.structural_key()
    # The regime the deployable -O3 re-benches (``pipeline._rebench_o3_async``) are
    # keyed under in the node store — the tune context with the re-bench's flags
    # substituted, so an -O3 leaf row never collides with its -O1 twin. ``None``
    # when the sweep itself already runs at -O3 (the re-bench is a no-op there and
    # the two keys would coincide).
    o3_ctx_key = replace(ctx, compile_flags=O3_NVCC_FLAGS).structural_key() if "-O3" not in ctx.compile_flags else None
    backend_name = getattr(pool[0], "name", "cuda")
    # Fold-aware slicing: a flash fold offer site's slice must CARRY the score producer its
    # fusion consumes (``absorbed[producer] = consumer``) — a single-node slice turns the
    # producer into a synthetic input, ``try_flash`` can never re-fuse, and every tune
    # trajectory silently degrades to the cut (benching fragment kernels that greedy deploy
    # never picks). The absorbed producer loses its own slice — the offer site's slice benches
    # the whole attention (fused: one kernel; a cut trajectory: both), so the Σ stays honest.
    absorbed: dict[str, str] = {}
    for nid, _op in _kernel_nodes(fused_graph):
        for pid in fused_producer_ids(fused_graph, fused_graph.nodes[nid]):
            absorbed[pid] = nid
    # Group structurally-identical LoopOps under one ``Op.cache_key`` —
    # insertion order = first occurrence (drives the progress tail name).
    # Ops with no cache key are unreachable through the bench path so they
    # don't enter the dedup map at all (matches the previous filter).
    unique: OrderedDict[str, tuple[str, object, int]] = OrderedDict()
    for nid, op in _kernel_nodes(fused_graph):
        if nid in absorbed:
            continue  # tuned inside its consumer's fold-offer slice
        key = op.cache_key()
        if key is None:
            continue
        if key in unique:
            rep_nid, rep_op, count = unique[key]
            unique[key] = (rep_nid, rep_op, count + 1)
        else:
            unique[key] = (nid, op, 1)
    if progress is not None:
        progress.start_terminal(len(unique))

    # Slot queue: each coroutine pops a device-pinned backend, benches its op's
    # whole inner search on it, returns it. ``len(pool)`` benches run at once.
    slots: asyncio.Queue = backend_slots if backend_slots is not None else asyncio.Queue()
    if backend_slots is None:
        for b in pool:
            slots.put_nowait(b)
    results: dict[int, OpResult] = {}

    async def tune_op(op_idx: int, key: str, nid: str, op, count: int) -> None:
        name = getattr(op, "name", None) or nid
        backend = await slots.get()
        try:
            if progress is not None:
                progress.op_start(name, slot=op_idx)
            sub = single_node_graph(fused_graph, nid, absorb=frozenset(p for p, c in absorbed.items() if c == nid))
            # Base knobs the prior sees on every row: the LoopOp's ``S_*``
            # structural identity (op-aware rows) + the ``H_*`` host/hardware
            # regime (GPU + nvcc opt level), so one global prior spans ops and
            # regimes from the feature vector alone.
            base_knobs = {**ctx.features(), **op.knobs}
            # Per-op RNG seed so each kernel's ε-greedy stream differs yet the run
            # is reproducible AND execution-order-independent (no wall-clock seed):
            # op ``op_idx`` always seeds ``seed + op_idx`` regardless of which slot
            # or completion order it ran in.
            inner = TuningSearch(
                patience=patience,
                ucb_c=ucb_c,
                explore_eps=explore_eps,
                seed=seed + op_idx,
                max_measurements=max_candidates,
                prior_model=prior,
                base_knobs=base_knobs,
            )
            async for cand in Pipeline.build(LOWERING_PASSES).tune_async(sub, search=inner, ctx=ctx, backend=backend, db=db):
                if progress is not None:
                    st = inner.last_stats
                    best_us = (1.0 / inner.tree.best_reward) if inner.tree.best_reward > 0 else None
                    progress.variant(
                        name,
                        variant_label(cand.graph),
                        median_us=st.median if st is not None else None,
                        status=inner.last_status or "",
                        best_us=best_us,
                        slot=op_idx,
                    )
            # The inner MCTS's best reward is ``1 / min whole-slice total``
            # (``_bench_terminal`` sums every CudaOp in the slice, so a split-K
            # main + combine both count). Record that total under the LoopOp
            # key so ``best_per_op_time`` reads the true per-op cost.
            best_total = 1.0 / inner.tree.best_reward if inner.tree.best_reward > 0 else None
            searched = inner.best_realized()
            if best_total is not None:
                # captured=True: the sweep benches under graph capture by default, so
                # this Σ-best bookkeeping row derives from captured measurements (a
                # rare per-variant capture fallback can contaminate the min — accepted;
                # the capture-wins overwrite then lets a re-tune upgrade it).
                db.record_perf(ctx_key, key, backend=backend_name, status="ok", stats=_point_stats(best_total), captured=True)
            if prior is not None:
                # In-flight refit (single-threaded → no lock): stream this op's
                # value-of-position rows (-O1) plus any -O3 winner samples into the
                # global reservoir; refit + checkpoint once enough new rows
                # accumulate (batched — see ``Prior``). Rows arrive in completion
                # order, so the trained ``prior.json`` varies run-to-run; the per-op
                # DB best below does not (distinct ``key`` per op).
                prior.add_rows(inner._collect_rows() + inner.o3_rows)
                if prior.maybe_refit():
                    prior.checkpoint()
            # Persist every search-tree node (partial branches + leaves, plus failed
            # leaves and the near-best configs' -O3-regime re-bench rows) to the
            # keyed/deduped ``node`` table — alongside the reservoir
            # feed above, and independent of whether a prior is attached. ``op_sig``
            # is the op's ``S_*`` structural signature; ``gpu`` is the card identity
            # (folded into the key so same-die SKUs don't collide); ``run_id`` tags
            # the rows with this tune session; the walk threads
            # parent_key/value/visits/stats/depth.
            op_sig = digest(*sorted((k, v) for k, v in op.knobs.items() if k.startswith("S_")))
            gpu = ctx.hardware_id()
            db.record_nodes(
                inner._collect_node_records(context_key=ctx_key, op_sig=op_sig, gpu=gpu, run_id=run_id, o3_context_key=o3_ctx_key)
            )
            best = db.best_per_op_time(ctx_key, key, backend=backend_name)
            searched_knobs = searched_us = searched_cuda_ops = None
            if searched is not None:
                from emmy.compiler.pipeline.knob import stamp_schedule_families  # noqa: PLC0415

                searched_knobs = stamp_schedule_families(searched[0])
                searched_us = searched[1]
                searched_cuda_ops = searched[2]
            results[op_idx] = OpResult(
                name=name,
                op_key=key,
                best_us=best,
                multiplicity=count,
                searched_knobs=searched_knobs,
                searched_us=searched_us,
                searched_cuda_ops=searched_cuda_ops,
            )
            if progress is not None:
                progress.op_done(name, slot=op_idx)
        finally:
            slots.put_nowait(backend)

    try:
        await asyncio.gather(*[tune_op(i, key, nid, op, count) for i, (key, (nid, op, count)) in enumerate(unique.items())])
    finally:
        # SIGKILL + await-reap each slot's async bench worker (the subprocess
        # transports are bound to this event loop; awaiting the reap cleans them
        # between terminals). Backend objects persist — their workers respawn lazily
        # on the next terminal's first ``benchmark_async`` (same loop, since the whole
        # outer drive runs under one ``asyncio.run`` in ``handle_tune``).
        if close_backends:
            for b in pool:
                aclose = getattr(b, "aclose_async_worker", None)
                if aclose is not None:
                    await aclose()

    # Accumulate in ``op_idx`` order so the reward / ``per_op`` order is
    # execution-order-independent (the float sum is order-stable, matching serial).
    total = 0.0
    ok = True
    per_op: list[OpResult] = []
    for op_idx in range(len(unique)):
        r = results[op_idx]
        per_op.append(r)
        if r.best_us is None:
            ok = False
            total += _FAIL_US * r.multiplicity
        else:
            total += r.best_us * r.multiplicity
    return InnerReward(total_us=total, ok=ok, per_op=per_op, prior_summaries=[])


async def run_two_level_tune(
    graph: Graph,
    *,
    ctx: Context,
    db: SearchDB,
    backend=None,
    backends=None,
    patience: int,
    ucb_c: float = TuningSearch.DEFAULT_UCB_C,
    explore_eps: float = 0.0,
    dump=None,
    progress=None,
    prior_seed: int = 0,
    run_id: str | None = None,
    max_candidates: int | None = None,
    prior=None,
    manage_prior: bool = True,
    backend_slots=None,
    close_backends: bool = True,
) -> TwoLevelResult:
    """Drive the outer structural search, scoring each terminal by
    :func:`_inner_reward_async`, then greedy-assemble the DB-best kernels and bench
    the whole graph once for the separability check.

    ``backends`` (a list of device-pinned :class:`CudaBackend`s) fans the inner
    per-kernel search out across GPUs; the default single ``backend`` is the
    one-slot serial pool. A caller driving several independent targets can provide
    one shared ``prior`` and ``backend_slots`` queue, set ``manage_prior=False`` /
    ``close_backends=False``, and own their final checkpoint and worker teardown.

    The outer drives a :class:`Run` directly (manual ``observe``)
    because its terminal reward comes from the inner tuning, not
    ``_bench_terminal_async``. The outer pipeline (:func:`outer_pipeline`) runs
    through the pre-partition tile rules, so each structural fork
    branches the outer tree —
    one terminal per kernel-set, compared by Σ-per-op cost. A graph with no
    structural offers yields a single terminal and this reduces to "tune each
    op once, sum, assemble". Identical offer sites within one trajectory
    replay the first decision (read off the graph —
    ``pipeline._replay_structural_decision``), and a terminal whose kernels
    are all known is a pure DB read, so extra terminals stay cheap."""
    from emmy.compiler import provenance  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415

    provenance.seed(graph)
    # One session id for every node row this run writes — minted by the caller
    # (``handle_tune``: one id per CLI invocation) or here as a fallback.
    run_id = run_id or _mint_run_id()
    # ONE global prior for the whole run — the ``OnlinePrior`` (warm-
    # started from its checkpoint) behind an ``OfflinePrior`` cold-start
    # fallback, so the first op's inner search is heuristic-guided, not uniform.
    # Training (add_rows / maybe_refit / checkpoint) delegates to the online half
    # (lazy import keeps catboost off non-tune callers).
    if prior is None:
        from emmy.compiler.pipeline.search.prior import load_prior  # noqa: PLC0415

        prior = load_prior(seed=prior_seed)
    outer = TuningSearch(patience=patience, ucb_c=ucb_c, prior_model=prior, base_knobs=ctx.features())
    # The outer drives only the graph-changing passes (through the
    # pre-partition tile head) — no dump on this Run; the winning config's
    # full stage artifacts and in-memory frontend provenance slices come
    # from the final assembled CUDA_PASSES run below.
    outer_run = Run(pipeline=outer_pipeline(), ctx=ctx, search=outer, db=db)

    best_fused: Graph | None = None
    best_reward: InnerReward | None = None
    n_terminals = 0
    prior_summaries: list[str] = []
    pool = list(backends) if backends else [backend]
    for token, fused in outer_run.drive(graph):
        n_terminals += 1
        reward = await _inner_reward_async(
            fused.graph,
            ctx=ctx,
            db=db,
            pool=pool,
            patience=patience,
            ucb_c=ucb_c,
            explore_eps=explore_eps,
            seed=prior_seed,
            progress=progress,
            prior=prior,
            run_id=run_id,
            max_candidates=max_candidates,
            backend_slots=backend_slots,
            close_backends=close_backends,
        )
        stats = PerfStats(median=reward.total_us, min=reward.total_us, max=reward.total_us, mean=reward.total_us, variance=0.0, n_samples=0)
        outer.observe(token, stats, "ok" if reward.ok else "bench_fail")
        positions = sum(r.multiplicity for r in reward.per_op)
        logger.info(
            "[tune] fused terminal #%d: Σ per-op = %.2f us (%d unique kernels, %d positions)",
            n_terminals,
            reward.total_us,
            len(reward.per_op),
            positions,
        )
        if best_reward is None or (reward.ok and reward.total_us < best_reward.total_us):
            best_fused, best_reward = fused.graph, reward

    # Placement rows are whole-kernel-set observations: each outer fork prefix is
    # labeled by the best Σ of its descendants. Feed them to the same online prior
    # after the outer walk; inner per-kernel rows deliberately contain no PLACE
    # metadata, so schedule evidence stays transferable between kernel sets.
    if prior is not None:
        prior.add_rows(outer._collect_rows())

    # One global end-of-run sanity block (the prior spans every kernel now).
    if manage_prior and (prior.fitted or prior.trajectory):
        prior_summaries.append(prior.summary("global"))
    # Force a final fit so even a small tune that never crossed a refit tier ends
    # with a usable model, then persist (dataset accumulates across runs).
    if manage_prior:
        prior.maybe_refit(force=True)
        prior.checkpoint()

    assembled: Graph | None = None
    if best_fused is not None:
        # Lower the winning outer terminal directly. This preserves the measured
        # placement choice independent of a not-yet-refitted prior while each
        # kernel's schedule replays from its DB evidence. No backend means the
        # stub bench persists nothing, so the 1.0us stub never clobbers a tuned
        # row. The dump (if any) rides here
        # so it captures the winning config's full stage artifacts. The
        # whole-graph separability bench used to run here too; dropped — its
        # only output was a small advisory gap line that nobody read, and at
        # the tight tune compile budget it could (and did) abort whole-model
        # tunes when a slow-compiling kernel raised. ``--bench`` re-benches
        # the assembled graph at -O3 anyway, which is the deployable number.
        assembled = Pipeline.build(LOWERING_PASSES).run(best_fused.copy(), ctx=ctx, db=db, dump=dump)
    return TwoLevelResult(
        best_fused=best_fused, best_reward=best_reward, n_terminals=n_terminals, assembled=assembled, prior_summaries=prior_summaries
    )
