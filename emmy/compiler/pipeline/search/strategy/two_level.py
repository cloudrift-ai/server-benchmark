"""Two-level autotuning as ONE search strategy over the engine's loop.

``TwoLevelStrategy`` owns the whole two-level design — the engine provides the loop
(``Run.drive`` / ``Pipeline.tune_async``) and nothing else:

- **Outer**: drive the graph-changing passes (``OUTER_PASSES`` — ``frontend`` + ``loop``, the
  strategy's OWN boundary config, never an engine parameter). The outer never ventures into
  Tile IR; a terminal is the fused graph of finalized ``LoopOp``\\ s. Today the outer tree is a
  chain — fusion offers no multi-option forks yet — and nothing here depends on that. A direct
  unscheduled ``TileOp`` target passes through that boundary unchanged and joins the inner
  per-kernel search; an already scheduled ``TileOp`` stays lowering-only.
- **Scoring is DECLARED SEPARABLE**: an outer terminal's reward is the Σ of its unique kernels'
  bests, each kernel measured independently in its own single-node slice
  (:func:`single_node_graph`) by a plain :class:`TuningSearch` (MCTS) over ``INNER_PASSES``.
  A Tile-dialect cross-CTA split is part of a kernel's independent measurement — a slice whose
  kernel set changed benches as the Σ over the pieces it minted.
- **Minted kernels become first-class targets**: the strategy's private splice watcher
  (:class:`_KernelInventory`) rides every inner run; each genuinely new kernel it reports (deduped by structural identity across the
  whole session, outer kernels included) is ENROLLED — tuned in its own slice, its rows keyed
  under its own identity — in waves, until no inner run mints anything new. Enrolled kernels are
  evidence, not reward terms: the parent slice's Σ already priced them, so they stay out of
  ``per_op`` / ``total_us`` (and out of ``searched_winner()``, which golden seeding reads).

Results key structurally (:meth:`~emmy.compiler.ir.base.Op.identity_key`), so inner-tuned ``perf``
/ ``lowering`` rows transfer to the assembled graph unchanged AND are shared across outer
terminals (a shared op is a DB hit). The inner search runs for **every** op on every pass — it is
never skipped on prior effort; replay is cheap (the per-variant ``perf`` cache serves
already-measured variants without a bench).
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from emmy.compiler.context import Context
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pass, Pipeline, TuningSearch
from emmy.compiler.pipeline.knob import stamp_schedule_families
from emmy.compiler.pipeline.passes.identity import IdentityStrategy
from emmy.compiler.pipeline.pipeline import Run, variant_label
from emmy.compiler.pipeline.search.db import PerfStats, SearchDB
from emmy.compiler.pipeline.search.slice import single_node_graph
from emmy.compiler.pipeline.search.strategy.base import SearchStrategy
from emmy.compiler.pipeline.strategy import PipelineStrategy, SpliceEvent, discovered_strategies

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph

logger = logging.getLogger(__name__)

# Lowering-only passes (post-fusion): ``tile → kernel → cuda``. The inner per-op search runs
# these on a single-node slice so the finalized LoopOp body — and thus its ``identity_key(with_io=True, with_knobs=True)`` — is
# never re-touched by ``loop/fusion``, which is what keeps inner-tuned ``perf`` / ``lowering``
# rows transferable to the assembled graph. Sliced as the tail of ``CUDA_PASSES`` so it tracks
# pass-list edits automatically.
LOWERING_PASSES = CUDA_PASSES[len(LOOP_PASSES) :]


def outer_pipeline() -> Pipeline:
    """The graph-changing passes the outer search drives: ``frontend`` + ``loop`` (the fusion
    forks). An outer terminal is a post-fusion graph of finalized ``LoopOp``\\ s; the strategy's
    separable ``evaluate`` picks each up as its own slice (own patience, own progress leaf,
    deduped by ``identity_key(with_io=True, with_knobs=True)``) and tunes it via :data:`LOWERING_PASSES`."""
    passes = [Pass.load(name, i) for i, name in enumerate(TwoLevelStrategy.OUTER_PASSES)]
    return Pipeline(passes=passes, strategies=discovered_strategies())


def _identity() -> IdentityStrategy:
    """The discovered IdentityStrategy instance — the one spelling of structural identity."""
    return next(s for s in discovered_strategies() if isinstance(s, IdentityStrategy))


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
    searched_structural: bool = False


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
        """The actual searched winner when it has one exact replay row.

        A target can contain repeated/heterogeneous post-fusion kernels. One
        post-fusion kernel can also lower to several CudaOps; that winner is
        replayable only when search retained the exact structural row that
        minted its independently scheduled pieces.
        """
        if len(self.per_op) != 1:
            return None
        op = self.per_op[0]
        if (
            op.multiplicity != 1
            or (op.searched_cuda_ops != 1 and not op.searched_structural)
            or op.searched_knobs is None
            or op.searched_us is None
        ):
            return None
        return dict(op.searched_knobs), op.searched_us


@dataclass
class TwoLevelResult:
    """Outcome of :meth:`TwoLevelStrategy.run`."""

    best_fused: Graph | None  # winning outer terminal (normally finalized LoopOps)
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
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"


def _kernel_nodes(graph: Graph) -> list[tuple[str, object]]:
    """Post-fusion kernel nodes — ``(node_id, op)`` for every kernel-bearing op.

    A normal outer terminal sits at the loop dialect's end (:func:`outer_pipeline`), so its
    kernels are finalized ``LoopOp`` instances. A direct post-cut reproducer instead starts at an
    unscheduled ``TileOp``; it enters the same inner search so its rows keep the child identity
    ordinary parent-route replay already consumes. A Tile root whose worker inventory is sealed
    (``work`` set) is already scheduled and stays lowering-only. ``work`` is the only sealed
    signal today, and it is ``None`` for the per-cell / pure-reduce forms too — such a root
    re-enters the per-kernel search and re-decides its schedule from the same evidence, which is
    redundant but not wrong; a dedicated scheduled marker on ``TileOp`` would tighten this."""
    return [(nid, n.op) for nid, n in graph.nodes.items() if isinstance(n.op, LoopOp) or (isinstance(n.op, TileOp) and n.op.work is None)]


@dataclass
class _Work:
    """One inner tuning target: an outer kernel (counts toward the terminal reward) or an
    enrolled minted kernel (evidence only)."""

    key: str  # ``identity_key(with_io=True, with_knobs=True)`` — the perf-row key
    nid: str
    op: object
    src_graph: Graph  # what the slice is cut from: the fused graph, or the minting fragment
    count: int  # graph multiplicity (0 for enrolled — never a reward term)
    enrolled: bool


class _KernelInventory(PipelineStrategy):
    """TwoLevelStrategy's PRIVATE splice watcher — not a composable component: the strategy
    composes one instance into every inner run's pipeline (``Pipeline.with_strategies``) so
    kernels minted during lowering (currently a split's pieces) can be enrolled as
    tuning targets. Reports each new kernel-bearing op — one whose structural identity has not
    been seen — to ``on_kernel(node_id, op, fragment)``. Cross-trajectory by design: the MCTS
    re-minting the same piece on every variant reports it once, and the seen-set is seeded with
    the outer terminal's kernels so pieces structurally identical to an outer kernel are not
    re-enrolled. Identity is COMPUTED through the IdentityStrategy's read API, so nothing here
    depends on a stamp having happened or on strategy dispatch order. It derives from
    PipelineStrategy because the pipeline's strategy set is the channel the engine notifies —
    the event protocol is how a search shape hears about splices."""

    def __init__(self, identity: IdentityStrategy, on_kernel, seen: set[str] | None = None) -> None:
        self.identity = identity
        self.on_kernel = on_kernel
        self.seen = seen if seen is not None else set()

    def on_splice(self, e: SpliceEvent) -> None:
        for nid, node in e.fragment.nodes.items():
            op = node.op
            if op.dialect is None:
                continue
            key = self.identity.op_sig(op, e.fragment)
            if key in self.seen:
                continue
            self.seen.add(key)
            self.on_kernel(nid, op, e.fragment)


class TwoLevelStrategy(SearchStrategy):
    """The two-level search as one strategy composing the engine's loop — see the module
    docstring for the design. Construct once per tune session; ``run`` drives one target."""

    OUTER_PASSES = LOOP_PASSES  # the strategy's own boundary config — never an engine parameter
    INNER_PASSES = LOWERING_PASSES

    def __init__(
        self,
        *,
        db: SearchDB,
        patience: int,
        backend=None,
        backends=None,
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
    ) -> None:
        self.db = db
        self.patience = patience
        self.pool = list(backends) if backends else [backend]
        self.ucb_c = ucb_c
        self.explore_eps = explore_eps
        self.dump = dump
        self.progress = progress
        self.prior_seed = prior_seed
        # One session id for every node row this run writes — minted by the caller
        # (``handle_tune``: one id per CLI invocation) or here as a fallback.
        self.run_id = run_id or _mint_run_id()
        self.max_candidates = max_candidates
        self.prior = prior
        self.manage_prior = manage_prior
        self.backend_slots = backend_slots
        self.close_backends = close_backends

    async def run(self, graph: Graph, ctx: Context | None = None) -> TwoLevelResult:
        """Drive the outer structural search, scoring each terminal by the separable
        :meth:`_evaluate_terminal`, then greedy-assemble the DB-best kernels.

        The outer drives a :class:`Run` directly (manual ``observe``) because its terminal
        reward comes from the inner tuning, not a whole-graph bench. Each fusion fork branches
        the outer tree — one terminal per kernel-set, compared by Σ-per-op cost. A graph with no
        structural offers yields a single terminal and this reduces to "tune each op once, sum,
        assemble". A terminal whose kernels are all known is a pure DB read, so extra terminals
        stay cheap."""
        if ctx is None:
            ctx = Context.probe()
        if self.prior is None and self.manage_prior:
            from emmy.compiler.pipeline.search.prior import load_prior  # noqa: PLC0415

            # ONE global prior for the whole run — warm-started from its checkpoint, so the
            # first op's inner search is heuristic-guided, not uniform. A caller that manages
            # its own prior (or a test wanting the uniform PUCT) passes it explicitly.
            self.prior = load_prior(seed=self.prior_seed)
        outer = TuningSearch(patience=self.patience, ucb_c=self.ucb_c, prior_model=self.prior, base_knobs=ctx.features())
        # The tune ctx relaxation is the policy's own (``TuningSearch.prepare_ctx``); the outer
        # Run is constructed directly, so apply it here.
        ctx = outer.prepare_ctx(ctx)
        # No dump on the outer Run — the winning config's full stage artifacts come from the
        # final assembled CUDA_PASSES run below.
        outer_run = Run(pipeline=outer_pipeline(), ctx=ctx, search=outer, db=self.db)

        best_fused: Graph | None = None
        best_reward: InnerReward | None = None
        n_terminals = 0
        prior_summaries: list[str] = []
        for token, fused in outer_run.drive(graph):
            n_terminals += 1
            reward = await self._evaluate_terminal(fused.graph, ctx)
            stats = _point_stats(reward.total_us)
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

        # One global end-of-run sanity block (the prior spans every kernel now).
        if self.manage_prior and self.prior is not None:
            if self.prior.fitted or self.prior.trajectory:
                prior_summaries.append(self.prior.summary("global"))
            # Force a final fit so even a small tune that never crossed a refit tier ends
            # with a usable model, then persist (dataset accumulates across runs).
            self.prior.maybe_refit(force=True)
            self.prior.checkpoint()

        assembled: Graph | None = None
        if best_fused is not None:
            # Greedy replay over the *original* graph re-derives the same fused LoopOps and
            # lowers each via the DB-best forks the inner search recorded. No backend →
            # nothing persisted (so the 1.0us stub never clobbers a tuned row). The dump (if
            # any) rides here so it captures the winning config's full stage artifacts.
            #
            # The card's recorded goldens are held OUT of this replay. They are a DEPLOY tier —
            # "this config is known good on this card" — and now that the sweep measures in the
            # deployable regime the tier would activate here for the first time and decide ahead
            # of the search's own evidence. A tune must assemble what it measured: ``--output``
            # and ``tune --bench`` read this graph while ``persist_tune_winner`` records the
            # searched winner, so a golden overriding it would report a benched number for a
            # config the tune did not choose.
            from emmy.compiler.pipeline.search.golden import records_override  # noqa: PLC0415

            with records_override([]):  # synchronous body — see the helper's note
                assembled = Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx, db=self.db, dump=self.dump)
        return TwoLevelResult(
            best_fused=best_fused, best_reward=best_reward, n_terminals=n_terminals, assembled=assembled, prior_summaries=prior_summaries
        )

    async def _evaluate_terminal(self, fused_graph: Graph, ctx: Context) -> InnerReward:
        """The separable scoring function: tune every post-fusion kernel of ``fused_graph`` in
        its own single-node slice and return ``Σ best-per-op time`` — the outer terminal reward
        — once all kernel roots are measured. Kernels minted inside the inner loops are enrolled
        in waves (evidence, never reward terms).

        One coroutine per work item over a slot queue of ``len(pool)`` device-pinned backends
        (one in-flight bench per slot). Single event loop, single thread — the shared ``db`` /
        ``prior`` are touched only between bench ``await``\\ s, so they're atomic with no locks.
        ``max_candidates`` caps live measurements per kernel; cached observations do not consume
        it."""
        db, prior, progress = self.db, self.prior, self.progress
        identity = _identity()
        ctx_key = ctx.structural_key()
        backend_name = getattr(self.pool[0], "name", "cuda")
        # Group structurally-identical kernel roots under one ``identity_key(with_io=True, with_knobs=True)`` — insertion order =
        # first occurrence (drives the progress tail name). Ops with no cache key are
        # unreachable through the bench path so they don't enter the dedup map at all.
        unique: OrderedDict[str, tuple[str, object, int]] = OrderedDict()
        for nid, op in _kernel_nodes(fused_graph):
            key = op.identity_key(with_io=True, with_knobs=True)
            if key is None:
                continue
            if key in unique:
                rep_nid, rep_op, count = unique[key]
                unique[key] = (rep_nid, rep_op, count + 1)
            else:
                unique[key] = (nid, op, 1)
        if progress is not None:
            progress.start_terminal(len(unique))

        # The session-wide kernel roster: seeded with the outer kernels' identities so a minted
        # piece structurally identical to an outer kernel is not re-enrolled; installed on every
        # inner run, so an enrolled kernel's own cuts/splits feed the next wave.
        minted: list[tuple[str, object, Graph]] = []
        inventory = _KernelInventory(
            identity,
            lambda nid, op, frag: minted.append((nid, op, frag)),
            seen={identity.op_sig(op) for _, op, _ in unique.values()},
        )

        # Slot queue: each coroutine pops a device-pinned backend, benches its op's whole inner
        # search on it, returns it. ``len(pool)`` benches run at once.
        slots: asyncio.Queue = self.backend_slots if self.backend_slots is not None else asyncio.Queue()
        if self.backend_slots is None:
            for b in self.pool:
                slots.put_nowait(b)
        results: dict[int, OpResult] = {}

        async def tune_op(op_idx: int, work: _Work) -> None:
            name = getattr(work.op, "name", None) or work.nid
            backend = await slots.get()
            try:
                if progress is not None and not work.enrolled:
                    progress.op_start(name, slot=op_idx)
                sub = single_node_graph(work.src_graph, work.nid)
                # Base knobs the prior sees on every row: the LoopOp's ``S_*`` structural
                # identity (op-aware rows) + the ``H_*`` host/hardware regime, so one global
                # prior spans ops and regimes from the feature vector alone.
                base_knobs = {**ctx.features(), **work.op.knobs}
                # Per-op RNG seed so each kernel's ε-greedy stream differs yet the run is
                # reproducible AND execution-order-independent (no wall-clock seed).
                inner = TuningSearch(
                    patience=self.patience,
                    ucb_c=self.ucb_c,
                    explore_eps=self.explore_eps,
                    seed=self.prior_seed + op_idx,
                    max_measurements=self.max_candidates,
                    prior_model=prior,
                    base_knobs=base_knobs,
                )
                inner_pipeline = Pipeline.build(LOWERING_PASSES).with_strategies(inventory)
                async for cand in inner_pipeline.tune_async(sub, search=inner, ctx=ctx, backend=backend, db=db):
                    if progress is not None and not work.enrolled:
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
                # The inner MCTS's best reward is ``1 / min whole-slice total`` (the bench sums
                # every CudaOp in the slice, so a split-K main + combine both count). Record
                # that total under the LoopOp key so ``best_per_op_time`` reads the true
                # per-op cost.
                best_total = 1.0 / inner.tree.best_reward if inner.tree.best_reward > 0 else None
                searched = inner.best_realized()
                if best_total is not None:
                    # captured=True: the sweep benches under graph capture by default, so this
                    # Σ-best bookkeeping row derives from captured measurements.
                    db.record_perf(ctx_key, work.key, backend=backend_name, status="ok", stats=_point_stats(best_total), captured=True)
                if prior is not None:
                    # In-flight refit (single-threaded → no lock): stream this op's rows into
                    # the global reservoir; refit + checkpoint once enough new rows accumulate.
                    prior.add_rows(inner._collect_rows())
                    if prior.maybe_refit():
                        prior.checkpoint()
                # Persist every search-tree node to the keyed/deduped ``node`` table. The
                # ``op_sig`` is the kernel's OWN structural identity — an enrolled piece's rows
                # are its own evidence, never its parent's.
                db.record_nodes(
                    inner._collect_node_records(
                        context_key=ctx_key,
                        op_sig=identity.op_sig(work.op),
                        gpu=ctx.hardware_id(),
                        run_id=self.run_id,
                    )
                )
                if work.enrolled:
                    if best_total is not None:
                        logger.info("[tune] enrolled minted kernel %s: Σ best %.2f us", name, best_total)
                    else:
                        logger.info("[tune] enrolled minted kernel %s: no clean measurement", name)
                    return
                best = db.best_per_op_time(ctx_key, work.key, backend=backend_name)
                searched_knobs = searched_us = searched_cuda_ops = None
                searched_structural = False
                if searched is not None:
                    searched_structural = searched[3]
                    searched_knobs = dict(searched[0]) if searched_structural else stamp_schedule_families(searched[0])
                    searched_us = searched[1]
                    searched_cuda_ops = searched[2]
                results[op_idx] = OpResult(
                    name=name,
                    op_key=work.key,
                    best_us=best,
                    multiplicity=work.count,
                    searched_knobs=searched_knobs,
                    searched_us=searched_us,
                    searched_cuda_ops=searched_cuda_ops,
                    searched_structural=searched_structural,
                )
                if progress is not None:
                    progress.op_done(name, slot=op_idx)
            finally:
                slots.put_nowait(backend)

        n_outer = len(unique)
        wave = [
            _Work(key=key, nid=nid, op=op, src_graph=fused_graph, count=count, enrolled=False) for key, (nid, op, count) in unique.items()
        ]
        op_idx = 0
        try:
            while wave:
                tasks = []
                for work in wave:
                    tasks.append(tune_op(op_idx, work))
                    op_idx += 1
                await asyncio.gather(*tasks)
                # Enrollment wave: everything the inventory reported while this wave ran. Waves
                # terminate because cut/split trees strictly shrink and the seen-set dedups.
                wave = [
                    _Work(key=key, nid=nid, op=op, src_graph=frag, count=0, enrolled=True)
                    for nid, op, frag in minted
                    if (key := op.identity_key(with_io=True, with_knobs=True)) is not None
                ]
                minted.clear()
        finally:
            # SIGKILL + await-reap each slot's async bench worker (the subprocess transports
            # are bound to this event loop; awaiting the reap cleans them between terminals).
            # Backend objects persist — their workers respawn lazily on the next terminal's
            # first ``benchmark_async``.
            if self.close_backends:
                for b in self.pool:
                    aclose = getattr(b, "aclose_async_worker", None)
                    if aclose is not None:
                        await aclose()

        # Accumulate in ``op_idx`` order so the reward / ``per_op`` order is
        # execution-order-independent (the float sum is order-stable, matching serial).
        total = 0.0
        ok = True
        per_op: list[OpResult] = []
        for i in range(n_outer):
            r = results[i]
            per_op.append(r)
            if r.best_us is None:
                ok = False
                total += _FAIL_US * r.multiplicity
            else:
                total += r.best_us * r.multiplicity
        return InnerReward(total_us=total, ok=ok, per_op=per_op, prior_summaries=[])
