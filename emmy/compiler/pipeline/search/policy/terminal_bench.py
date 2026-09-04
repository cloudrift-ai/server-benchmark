"""Bench + DB persistence for one tune terminal — terminal valuation, the policy's half.

What a terminal is worth is search policy, not engine mechanics: ``TuningSearch.evaluate`` drives
:func:`bench_terminal_async` per terminal the engine's loop yields. The engine never benches,
persists, or reads a policy attribute.
"""

from __future__ import annotations

import logging
import re
import statistics

from emmy.compiler.backend.cuda.program import compile_budget_overrun
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.pipeline.search.db import PerfStats

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
        self.regime = cand.ctx.regime()
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
        #: Per-KERNEL measurements, ``[(knobs, median_us, status), ...]`` in launch order — what
        #: the search trains its prior on. A terminal is a Σ over its kernels; when a structural
        #: fork made it several, they hold DIFFERENT rows and there is no single row to attribute
        #: the total to. Each kernel carries its own decisions and earns its own sample.
        self.per_kernel: list[tuple[dict, float, str]] = []

    #: The kernel a watchdog message NAMES — ``kernel 'k_foo (iter 0)' did not complete …``. The
    #: exception class does not survive the bench worker's pipe (it arrives wrapped in a
    #: ``BenchWorkerJobError``), so the label is recovered from the text.
    _NAMED_KERNEL = re.compile(r"kernel '([A-Za-z_][A-Za-z0-9_]*)")

    def _blamed(self, exc) -> set[int]:
        """``id()``s of the nodes a failure is EVIDENCE ABOUT — usually not every kernel benched.

        A terminal benches many kernels together and one of them hanging fails the whole run, so
        blaming all of them records a failure for kernels that were never shown to fail. That is
        not a cosmetic mislabel: those rows are read as deploy evidence, and on DeepSeek-V4's post
        block 70 recorded failures carried only 7 distinct errors — 20 kernels condemned by one
        hang, and 21 by a bench-worker startup timeout that is not a property of any kernel.

        So blame is recorded only where it is unambiguous: the kernel the watchdog named, or the
        single kernel of a one-kernel terminal. Otherwise nothing is persisted — the run failed,
        but which kernel failed is unknown, and unknown is not the same as failed. The terminal
        still reports ``bench_fail`` either way, so the search treats the candidate as failed and
        moves on; only the durable per-kernel evidence is narrowed to what was actually observed."""
        named = self._NAMED_KERNEL.search(str(exc))
        if named is not None:
            culprit = named.group(1)
            return {id(n) for n in self.cuda_nodes if getattr(n.op, "kernel_name", "") == culprit}
        return {id(self.cuda_nodes[0])} if len(self.cuda_nodes) == 1 else set()

    def _note(self, op, stats, status: str) -> None:
        self.per_kernel.append((dict(getattr(op, "knobs", None) or {}), float(stats.median), status))

    @staticmethod
    def _point_stats(us: float):

        return PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=0)

    @classmethod
    def _stats_from_launch(cls, lt):

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
        return cls._point_stats(lt.time_ms * 1000.0)

    def _persist(self, cuda_op, *, stats, status: str, captured: bool = False, error: str | None = None) -> None:
        """One measured kernel as one ``measurement`` row: its knob-free identity, the arm its
        knobs decided, and the µs. The rewrite chain needs no rows of its own — every stage of it
        keys off the same Loop-IR content, so the row a chain's terminal writes IS the row its
        LoopOp is priced by."""
        identity = cuda_op.identity()
        if identity is None:
            return
        self.db.record_measurement(self.regime, identity, cuda_op.knobs, status=status, stats=stats, captured=captured, error=error)
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
            return "done", (self._point_stats(float(fail_s) * 1_000_000.0), "bench_fail")

        if not self.cuda_nodes:
            return "done", (self._point_stats(0.0), "ok")

        # Cache lookup: if every CudaOp already has a perf row for this
        # (context, backend), skip the benchmark entirely and rebuild the
        # aggregate stats from the DB. Per-kernel partial caching isn't
        # useful here because ``backend.benchmark`` runs the whole graph.
        cached_rows = []
        for node in self.cuda_nodes:
            identity = node.op.identity()
            if identity is None:
                cached_rows = None
                break
            row = self.db.measurement(self.regime, identity, node.op.knobs)
            if row is None:
                cached_rows = None
                break
            cached_rows.append(row)
        if cached_rows is not None:
            logger.info("[tune] cache hit for %d kernel(s) — skipping bench", len(self.cuda_nodes))
            agg = None
            status = "ok"
            for node, row in zip(self.cuda_nodes, cached_rows, strict=True):
                if row.status != "ok":
                    status = row.status
                stats = self._point_stats(row.us_median)
                agg = self._accumulate(agg, stats)
                self._note(node.op, stats, row.status)
                logger.info("[tune]   %s @ %.2f us  (%s, cached)", row.op[:12], row.us_median, row.status)
            return "done", (agg or self._point_stats(0.0), status)

        if self.backend is None:
            # No real measurement → do NOT persist. Writing the 1.0us stub
            # to a shared DB used to clobber tuned ``best_median_us`` values
            # (``record_measurement`` keeps the minimum), so any plain
            # ``emmy run`` (which routes through ``Pipeline.run`` without
            # a backend) was overwriting real autotune rows with 1.0us stubs.
            # Tests that need lowering edges in stub mode should pass an
            # explicit stub backend.
            agg = None
            for node in self.cuda_nodes:
                agg = self._accumulate(agg, self._point_stats(1.0))
                self._note(node.op, self._point_stats(1.0), "ok")
            return "done", (agg or self._point_stats(0.0), "ok")

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
            return self._point_stats(0.0), "compile_timeout"
        fail_us = float(self.backend.bench_run_timeout_s) * 1_000_000.0
        blamed = self._blamed(exc)
        logger.warning(
            "[tune] backend.benchmark failed (%s) — pinning bench_fail @ %.1f us for %d of %d kernel(s)",
            exc,
            fail_us,
            len(blamed),
            len(self.cuda_nodes),
        )
        s = self._point_stats(fail_us)
        agg = None
        for node in self.cuda_nodes:
            if id(node) in blamed:
                self._persist(node.op, stats=s, status="bench_fail", error=f"{type(exc).__name__}: {exc}")
                self._note(node.op, s, "bench_fail")
            agg = self._accumulate(agg, s)
        return agg or self._point_stats(0.0), "bench_fail"

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
            s = self._point_stats(avg_us)
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
        return agg or self._point_stats(0.0), "ok"


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
