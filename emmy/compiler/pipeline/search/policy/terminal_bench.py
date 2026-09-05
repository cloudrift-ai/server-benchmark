"""Bench + DB persistence for one tune terminal — terminal valuation, the policy's half.

What a terminal is worth is search policy, not engine mechanics: ``TuningSearch.evaluate`` drives
:func:`bench_terminal_async` per terminal the engine's loop yields. The engine never benches,
persists, or reads a policy attribute.
"""

from __future__ import annotations

import json
import logging
import re
import statistics

from emmy.compiler.backend.cuda.program import compile_budget_overrun
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.kernel.ir import KernelOp
from emmy.compiler.ir.loop.ir import LoopOp
from emmy.compiler.pipeline.search.db import PerfStats
from emmy.compiler.structural import digest

# The engine logger keeps the existing ``[tune]`` log channel and verbosity toggles.
logger = logging.getLogger("emmy.compiler.pipeline")


class TerminalBench:
    """Shared machinery for benching one terminal candidate's ``CudaOp``s and
    persisting per-kernel ``perf`` / inventory / lowering rows.

    :func:`bench_terminal_async` drives it: the no-cuda / cache-hit / stub
    short-circuits (:meth:`prelude`) and every DB write (:meth:`finalize_result` /
    :meth:`finalize_exc`) live here, so the only awaited step is the device bench."""

    def __init__(self, cand, *, backend, db) -> None:

        self.backend = backend
        self.db = db
        self.graph = cand.graph
        self.context_key = cand.ctx.structural_key()
        order = self.graph.topological_order()
        self.cuda_nodes = [self.graph.nodes[nid] for nid in order if isinstance(self.graph.nodes[nid].op, CudaOp)]
        # Kernel-bearing nodes a rewrite left un-lowered (a validation-filtered rewrite under
        # tune — see ``Candidate.try_rewrite``). The same membership test as the backend's
        # ``_launches`` walk, so anything the backend would refuse is known BEFORE the bench:
        # summing only ``cuda_nodes`` prices the un-lowered kernel at zero, and the cache-hit
        # path would then report the residual kernels' Σ as an ``ok`` terminal measurement (the
        # issue-#327 "impossibly fast" split-K rows — a finalize kernel's cached µs standing in
        # for the whole matmul).
        self.unlowered = [nid for nid in order if not isinstance(self.graph.nodes[nid].op, (CudaOp, InputOp, ConstantOp))]
        self.backend_name = getattr(backend, "name", "stub")
        #: Per-KERNEL measurements, ``[(knobs, median_us, status), ...]`` in launch order — what
        #: the search trains its prior on. A terminal is a Σ over its kernels; when a structural
        #: fork made it several, they hold DIFFERENT rows and there is no single row to attribute
        #: the total to. Each kernel carries its own decisions and earns its own sample.
        self.per_kernel: list[tuple[dict, float, str]] = []
        #: The kernel set's own identity — the digest of its kernels' variant keys — where a
        #: multi-kernel terminal's verdict is filed when no single kernel can be blamed for it
        #: (:func:`persist_bench_failure`). ``None`` for a one-kernel terminal, whose verdict is its kernel's.
        keys = [n.op.identity_key(with_io=True, with_knobs=True) for n in self.cuda_nodes]
        self.set_key = digest("kernel-set", *sorted(keys)) if len(keys) > 1 and None not in keys else None

    def _note(self, op, stats, status: str) -> None:
        self.per_kernel.append((dict(getattr(op, "knobs", None) or {}), float(stats.median), status))

    def _fail_verdict(self, us: float, status: str = "bench_fail"):
        """The terminal's value when it cannot be priced: every kernel at the fail sentinel ``us``
        — the Σ a fresh failure returns, so a replayed one scores the same."""
        return point_stats(us * len(self.cuda_nodes)), status

    def _cached_row(self, node):
        key = node.op.identity_key(with_io=True, with_knobs=True)
        return self.db.lookup_perf(self.context_key, key, backend=self.backend_name) if key is not None else None

    @staticmethod
    def _stats_from_launch(lt):
        return stats_from_launch(lt)

    def _persist(self, cuda_op, *, stats, status: str, captured: bool = False, error: str | None = None) -> None:
        persist_kernel_perf(
            self.db, self.context_key, self.backend_name, cuda_op, stats=stats, status=status, captured=captured, error=error
        )
        logger.info("[tune]   %s @ %.2f us  (%s)", getattr(cuda_op, "kernel_name", "?"), stats.median, status)

    def _accumulate(self, acc, s):

        if acc is None:
            return s
        return PerfStats(
            median=acc.median + s.median,
            min=acc.min + s.min,
            max=acc.max + s.max,
            mean=acc.mean + s.mean,
            variance=acc.variance + s.variance,
            n_samples=min(acc.n_samples, s.n_samples) if acc.n_samples and s.n_samples else (acc.n_samples or s.n_samples),
        )

    def prelude(self):
        """Resolve everything that needs no live bench. Returns ``("done", (stats,
        status))`` when no measurement is needed (no CudaOps / full cache hit /
        stub backend), else ``("bench", None)`` — the caller obtains a
        ``BenchmarkResult`` and calls :meth:`finalize_result` / :meth:`finalize_exc`."""
        # An un-lowered kernel-bearing node is a bench_fail terminal, decided here — never a
        # backend call (it would raise the opaque ``non-CudaOp`` TypeError) and never the
        # cache-hit / no-CudaOp paths below (they see only the RESIDUAL kernels and would report
        # a partial Σ as ``ok``). Nothing is persisted: the residual kernels' own perf rows are
        # honest, and the fail is the terminal's, not theirs.
        if self.unlowered:
            logger.warning(
                "[tune] %d node(s) left un-lowered (%s) — bench_fail without benching",
                len(self.unlowered),
                ", ".join(f"{nid}: {type(self.graph.nodes[nid].op).__name__}" for nid in self.unlowered),
            )
            fail_s = self.backend.bench_run_timeout_s if self.backend is not None else 1.0
            return "done", (point_stats(float(fail_s) * 1_000_000.0), "bench_fail")

        if not self.cuda_nodes:
            return "done", (point_stats(0.0), "ok")

        # Cache lookup. A kernel with a failed row fails every slice it is in — its identity is
        # its rendered source and launch geometry, the same bytes wherever it appears — so one such
        # row decides the slice, blamed exactly as it was recorded, and the other kernels need no
        # row of their own (the all-or-nothing rule below used to re-bench a hang on every fresh
        # session because the innocent kernels had none). An ``ok`` replay still needs every
        # kernel's row: ``backend.benchmark`` runs the whole graph, so a partial cache cannot
        # stand in for the Σ. A verdict filed against the kernel set as a whole (an unblamed wall
        # kill, :meth:`finalize_exc`) is looked up first: it has no kernel behind it.
        if self.set_key is not None:
            row = self.db.lookup_perf(self.context_key, self.set_key, backend=self.backend_name)
            if row is not None:
                logger.info(
                    "[tune] cache hit: this %d-kernel set recorded %s as a whole — skipping bench", len(self.cuda_nodes), row.status
                )
                return "done", self._fail_verdict(row.stats.median, row.status)
        rows = [(node, self._cached_row(node)) for node in self.cuda_nodes]
        failed = [(node, row) for node, row in rows if row is not None and row.status != "ok"]
        if failed:
            logger.info("[tune] cache hit: %d of %d kernel(s) recorded %s — skipping bench", len(failed), len(rows), failed[0][1].status)
            for node, row in failed:
                self._note(node.op, row.stats, row.status)
            return "done", self._fail_verdict(failed[0][1].stats.median, failed[0][1].status)
        if all(row is not None for _node, row in rows):
            logger.info("[tune] cache hit for %d kernel(s) — skipping bench", len(rows))
            agg = None
            for node, row in rows:
                agg = self._accumulate(agg, row.stats)
                self._note(node.op, row.stats, row.status)
                logger.info("[tune]   %s @ %.2f us  (%s, cached)", row.op_key[:12], row.stats.median, row.status)
            return "done", (agg or point_stats(0.0), "ok")

        if self.backend is None:
            # No real measurement → do NOT persist. Writing the 1.0us stub
            # to a shared DB used to clobber tuned ``best_median_us`` values
            # (record_lowering / record_perf keep the minimum), so any plain
            # ``emmy run`` (which routes through ``Pipeline.run`` without
            # a backend) was overwriting real autotune rows with 1.0us stubs.
            # Tests that need lowering edges in stub mode should pass an
            # explicit stub backend.
            agg = None
            for node in self.cuda_nodes:
                agg = self._accumulate(agg, point_stats(1.0))
                self._note(node.op, point_stats(1.0), "ok")
            return "done", (agg or point_stats(0.0), "ok")

        logger.info("[tune] benching %d kernel(s) in graph", len(self.cuda_nodes))
        return "bench", None

    def finalize_exc(self, exc):
        if compile_budget_overrun(exc):
            # Nothing was measured, so nothing is recorded (see ``CompileBudgetExceeded``). The
            # status stays in memory: ``_collect_node_records`` emits fail rows for ``bench_fail``
            # exactly, so this writes no node row either. Loud, because a whole tile family
            # overrunning the budget is a finding, not noise.
            logger.warning(
                "[tune] COMPILE BUDGET EXCEEDED for %d kernel(s) (%s) — nothing recorded; "
                "raise bench_compile_timeout_s if this repeats on a whole tile family",
                len(self.cuda_nodes),
                exc,
            )
            return point_stats(0.0), "compile_timeout"
        fail_us = float(self.backend.bench_run_timeout_s) * 1_000_000.0
        blamed = persist_bench_failure(self.db, self.context_key, self.backend_name, self.cuda_nodes, exc, fail_us)
        logger.warning(
            "[tune] backend.benchmark failed (%s) — pinning bench_fail @ %.1f us for %d of %d kernel(s)",
            exc,
            fail_us,
            len(blamed),
            len(self.cuda_nodes),
        )
        s = point_stats(fail_us)
        for node in blamed:
            self._note(node.op, s, "bench_fail")
        if not blamed and self.set_key is not None:
            # The row carries no knobs: nothing about any kernel is claimed, so the greedy's
            # disqualification index (which joins on ``S_*`` signatures) and the dataset (which
            # joins on ``cuda_op``) never see it — only this cache lookup does.
            error = f"{type(exc).__name__}: {exc}"
            self.db.record_perf(self.context_key, self.set_key, backend=self.backend_name, status="bench_fail", stats=s, error=error)
        return self._fail_verdict(fail_us)

    def finalize_result(self, result):
        agg = None
        per_launch = result.per_launch or []
        if len(per_launch) != len(self.cuda_nodes):
            logger.warning(
                "[tune] per_launch count (%d) != CudaOp node count (%d); falling back to graph time_ms / N",
                len(per_launch),
                len(self.cuda_nodes),
            )
            avg_us = (result.time_ms * 1000.0) / max(len(self.cuda_nodes), 1)
            s = point_stats(avg_us)
            for node in self.cuda_nodes:
                self._persist(node.op, stats=s, status="ok", captured=result.captured)
                self._note(node.op, s, "ok")
                agg = self._accumulate(agg, s)
        else:
            for node, lt in zip(self.cuda_nodes, per_launch, strict=True):
                s = self._stats_from_launch(lt)
                self._persist(node.op, stats=s, status="ok", captured=result.captured)
                self._note(node.op, s, "ok")
                agg = self._accumulate(agg, s)
        try:
            import cupy as _cp  # noqa: PLC0415

            _cp.cuda.runtime.deviceSynchronize()
            _cp.get_default_memory_pool().free_all_blocks()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
        return agg or point_stats(0.0), "ok"


async def bench_terminal_async(cand, *, backend, db):
    """Bench every ``CudaOp`` in ``cand.graph``, persist per-kernel ``perf`` / inventory / lowering
    rows, and return ``(stats, status, measured, per_kernel)``: ``stats`` is the per-kernel
    ``PerfStats`` summed across the graph (the total terminal latency), ``measured`` whether a live
    backend measurement was required, and ``per_kernel`` the ``(knobs, median_us, status)`` of each
    kernel — the terminal's Σ decomposed into the rows that earned it. The
    only ``await`` is the device-pinned bench, so N kernels' benches overlap on one
    event loop; cache-hit / stub / persistence semantics live in :class:`TerminalBench`."""
    b = TerminalBench(cand, backend=backend, db=db)
    kind, payload = b.prelude()
    if kind == "done":
        return *payload, False, b.per_kernel
    try:
        result = await backend.benchmark_async(b.graph, warmup=1, num_iters="auto")
    except Exception as exc:  # noqa: BLE001
        return *b.finalize_exc(exc), True, b.per_kernel
    return *b.finalize_result(result), True, b.per_kernel


def point_stats(us: float) -> PerfStats:
    """A single-sample ``PerfStats`` — the shape a whole-graph time takes when no per-launch samples exist."""
    return PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=0)


def stats_from_launch(lt) -> PerfStats:
    """``PerfStats`` for one benched launch: over its samples when it carries them, else the point time."""
    if lt.samples and len(lt.samples) >= 1:
        us = [s * 1000.0 for s in lt.samples]
        return PerfStats(
            median=statistics.median(us),
            min=min(us),
            max=max(us),
            mean=statistics.fmean(us),
            variance=statistics.pvariance(us) if len(us) > 1 else 0.0,
            n_samples=len(us),
        )
    return point_stats(lt.time_ms * 1000.0)


def record_op_inventory(db, op) -> None:
    """Upsert one op's inventory row (``cuda_op`` / ``kernel_op`` / ``loop_op``) by its variant key."""
    key = op.identity_key(with_io=True, with_knobs=True)
    if key is None:
        return
    if isinstance(op, CudaOp):
        db.record_cuda_op(
            key,
            kernel_source=op.kernel_source,
            arg_order=list(op.arg_order),
            grid=list(op.grid),
            block=list(op.block),
            smem_bytes=op.smem_bytes,
            pretty=op.kernel_source,
        )
    elif isinstance(op, KernelOp):
        db.record_kernel_op(key, _body_json(op, "kernel"), op.pretty_body())
    elif isinstance(op, LoopOp):
        db.record_loop_op(key, _body_json(op, "loop"), op.pretty_body())


def _body_json(op, dialect: str) -> str:
    return json.dumps(
        {
            "dialect": dialect,
            "name": getattr(op, "name", None) or getattr(op, "kernel_name", None) or "?",
            "body_repr": repr(op.body),
        },
        default=str,
    )


def persist_kernel_perf(
    db, context_key: str, backend_name: str, cuda_op, *, stats, status: str, captured: bool = False, error: str | None = None
) -> bool:
    """Persist one measured kernel as deploy evidence: its ``perf`` row under ``context_key``
    (keep-best policy, see :meth:`SearchDB.record_perf`), the inventory rows of every op on its
    source chain, and the ``lowering`` hops between them. The ONE writer for a kernel
    measurement — the tuner's terminal bench and ``run --bench``'s pinned rows both come here, so
    a replayed golden and a searched candidate are indistinguishable to the evidence pick.
    Returns whether a row was written (a kernel with no variant key persists nothing)."""
    cuda_key = cuda_op.identity_key(with_io=True, with_knobs=True)
    if cuda_key is None:
        return False
    chain = [op for op in cuda_op.source_chain() if op.dialect is not None]
    for op in chain:
        record_op_inventory(db, op)
    for parent_op, child_op in zip(chain[1:], chain[:-1], strict=False):
        p_dialect = parent_op.dialect
        c_dialect = child_op.dialect
        if p_dialect is None or c_dialect is None:
            continue
        if p_dialect == c_dialect == "loop":
            # loop→loop source hops are structural/decision hops, not
            # lowering rewrites: the splice attribution stamped by the
            # identity strategy (a decomposition's kernels → the
            # pre-split op), the keep-vs-split rebind, name stamps.
            # A ``lowering`` row holds ONE best child per parent, so
            # recording a multi-kernel decomposition's hops would let
            # ``best_per_op_time``'s chain walk resolve the pre-split
            # op to a single fragment kernel's median — half the work
            # masquerading as the whole op. The decomposition's cost
            # is a Σ, owned by the two-level tuner, never this table.
            continue
        p_key = parent_op.identity_key(with_io=True, with_knobs=True)
        c_key = child_op.identity_key(with_io=True, with_knobs=True)
        if p_key is None or c_key is None:
            continue
        p_knobs = getattr(parent_op, "knobs", None) or {}
        c_knobs = getattr(child_op, "knobs", None) or {}
        knobs_delta = {k: v for k, v in c_knobs.items() if p_knobs.get(k) != v}
        db.record_lowering(
            p_key,
            p_dialect,
            c_key,
            c_dialect,
            knobs=knobs_delta,
            measured_median_us=stats.median if status == "ok" else None,
        )
    knobs = getattr(cuda_op, "knobs", None) or {}
    db.record_perf(context_key, cuda_key, backend=backend_name, status=status, stats=stats, knobs=knobs, captured=captured, error=error)
    return True


#: The kernel a watchdog message NAMES — ``kernel 'k_foo (iter 0)' did not complete …``. The
#: exception class does not survive the bench worker's pipe (it arrives wrapped in a
#: ``BenchWorkerJobError``), so the label is recovered from the text.
_NAMED_KERNEL = re.compile(r"kernel '([A-Za-z_][A-Za-z0-9_]*)")


def persist_bench_failure(db, context_key: str, backend_name: str, cuda_nodes, exc, fail_us: float) -> list:
    """Persist a failed bench as the per-kernel evidence it is: a ``bench_fail`` perf row at the
    ``fail_us`` sentinel for every node the failure is EVIDENCE ABOUT — usually not every kernel
    benched — and return those nodes. The ONE writer for a bench failure, as
    :func:`persist_kernel_perf` is for a measurement: the tuner's terminal bench and
    ``run --bench``'s greedy row both come here, so a hang blames the same kernel whichever
    command measured it.

    A bench runs many kernels together and one of them hanging fails the whole run, so blaming all
    of them records a failure for kernels that were never shown to fail. That is not a cosmetic
    mislabel: those rows are read as deploy evidence, and on DeepSeek-V4's post block 70 recorded
    failures carried only 7 distinct errors — 20 kernels condemned by one hang, and 21 by a
    bench-worker startup timeout that is not a property of any kernel. So blame is recorded only
    where it is unambiguous: the kernel the watchdog named, or the single kernel of a one-kernel
    graph. Otherwise no kernel earns a row — the run failed, but which kernel failed is unknown,
    and unknown is not the same as failed (the tuner files that verdict under the kernel set's
    own key instead)."""
    named = _NAMED_KERNEL.search(str(exc))
    if named is not None:
        blamed = [n for n in cuda_nodes if getattr(n.op, "kernel_name", "") == named.group(1)]
    else:
        blamed = list(cuda_nodes) if len(cuda_nodes) == 1 else []
    stats = point_stats(fail_us)
    error = f"{type(exc).__name__}: {exc}"
    for node in blamed:
        persist_kernel_perf(db, context_key, backend_name, node.op, stats=stats, status="bench_fail", error=error)
    return blamed
