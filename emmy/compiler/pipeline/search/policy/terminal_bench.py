"""Bench + DB persistence for one tune terminal — terminal valuation, the policy's half.

What a terminal is worth is search policy, not engine mechanics: ``TuningSearch.evaluate`` drives
:func:`bench_terminal_async` (and the deployable-regime :func:`rebench_o3_async`) per terminal the
engine's loop yields. The engine never benches, persists, or reads a policy attribute.
"""

from __future__ import annotations

import json
import logging
import statistics

from emmy import config
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.kernel.ir import KernelOp
from emmy.compiler.ir.loop.ir import LoopOp
from emmy.compiler.pipeline.search.db import PerfStats

# The engine logger keeps the existing ``[tune]`` log channel and verbosity toggles.
logger = logging.getLogger("emmy.compiler.pipeline")

# The nvcc flags of the deployable -O3 re-bench (:func:`rebench_o3_async`) — also the regime its
# node rows are keyed under (``two_level`` derives their ``context_key`` from the tune context
# with these flags substituted).
O3_NVCC_FLAGS = "-Xcicc -O3"


async def rebench_o3_async(cand, backend):
    """Re-bench an already-lowered tune winner at ``-Xcicc -O3`` (deployable codegen)
    for a clean prior sample, awaiting the device-pinned worker. Returns the -O3
    median latency in µs, or ``None`` when the sweep is already at -O3 or the bench
    errors (best-effort — a re-bench hiccup must never abort the sweep). The winner
    already benched OK at -O1, so the only added cost is one -O3 compile (cubin-cached)."""
    if "-O3" in config.nvcc_flags():
        return None
    try:
        result = await backend.benchmark_async(cand.graph, nvcc_flags=O3_NVCC_FLAGS)
    except Exception:  # noqa: BLE001 — a re-bench failure is non-fatal to tuning
        return None
    return result.time_ms * 1000.0 if result.time_ms else None


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

    @staticmethod
    def _body_json(op, dialect: str) -> str:

        return json.dumps(
            {
                "dialect": dialect,
                "name": getattr(op, "name", None) or getattr(op, "kernel_name", None) or "?",
                "body_repr": repr(op.body),
            },
            default=str,
        )

    def _record_op_inventory(self, op) -> None:

        key = op.cache_key()
        if key is None:
            return
        if isinstance(op, CudaOp):
            self.db.record_cuda_op(
                key,
                kernel_source=op.kernel_source,
                arg_order=list(op.arg_order),
                grid=list(op.grid),
                block=list(op.block),
                smem_bytes=op.smem_bytes,
                pretty=op.kernel_source,
            )
        elif isinstance(op, KernelOp):
            self.db.record_kernel_op(key, self._body_json(op, "kernel"), op.pretty_body())
        elif isinstance(op, LoopOp):
            self.db.record_loop_op(key, self._body_json(op, "loop"), op.pretty_body())

    def _persist(self, cuda_op, *, stats, status: str, captured: bool = False, error: str | None = None) -> None:
        cuda_key = cuda_op.cache_key()
        if cuda_key is None:
            return
        chain = [op for op in cuda_op.source_chain() if op.dialect is not None]
        for op in chain:
            self._record_op_inventory(op)
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
            p_key = parent_op.cache_key()
            c_key = child_op.cache_key()
            if p_key is None or c_key is None:
                continue
            p_knobs = getattr(parent_op, "knobs", None) or {}
            c_knobs = getattr(child_op, "knobs", None) or {}
            knobs_delta = {k: v for k, v in c_knobs.items() if p_knobs.get(k) != v}
            self.db.record_lowering(
                p_key,
                p_dialect,
                c_key,
                c_dialect,
                knobs=knobs_delta,
                measured_median_us=stats.median if status == "ok" else None,
            )
        knobs = getattr(cuda_op, "knobs", None) or {}
        self.db.record_perf(
            self.context_key, cuda_key, backend=self.backend_name, status=status, stats=stats, knobs=knobs, captured=captured, error=error
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
            return "done", (self._point_stats(float(fail_s) * 1_000_000.0), "bench_fail")

        if not self.cuda_nodes:
            return "done", (self._point_stats(0.0), "ok")

        # Cache lookup: if every CudaOp already has a perf row for this
        # (context, backend), skip the benchmark entirely and rebuild the
        # aggregate stats from the DB. Per-kernel partial caching isn't
        # useful here because ``backend.benchmark`` runs the whole graph.
        cached_rows = []
        for node in self.cuda_nodes:
            key = node.op.cache_key()
            row = self.db.lookup_perf(self.context_key, key, backend=self.backend_name) if key is not None else None
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
                agg = self._accumulate(agg, row.stats)
                self._note(node.op, row.stats, row.status)
                logger.info("[tune]   %s @ %.2f us  (%s, cached)", row.op_key[:12], row.stats.median, row.status)
            return "done", (agg or self._point_stats(0.0), status)

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
                agg = self._accumulate(agg, self._point_stats(1.0))
                self._note(node.op, self._point_stats(1.0), "ok")
            return "done", (agg or self._point_stats(0.0), "ok")

        logger.info("[tune] benching %d kernel(s) in graph", len(self.cuda_nodes))
        return "bench", None

    def finalize_exc(self, exc):
        fail_us = float(self.backend.bench_run_timeout_s) * 1_000_000.0
        logger.warning(
            "[tune] backend.benchmark failed (%s) — pinning bench_fail @ %.1f us for %d kernel(s)",
            exc,
            fail_us,
            len(self.cuda_nodes),
        )
        s = self._point_stats(fail_us)
        agg = None
        for node in self.cuda_nodes:
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
        result = await backend.benchmark_async(b.graph, num_iters="auto")
    except Exception as exc:  # noqa: BLE001
        return *b.finalize_exc(exc), True, b.per_kernel
    return *b.finalize_result(result), True, b.per_kernel
