from __future__ import annotations

from types import SimpleNamespace

from emmy.compiler.backend.base import BenchmarkResult, LaunchTime
from emmy.compiler.backend.cuda.program import BenchWorkerJobError, CompileBudgetExceeded
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.pipeline.search.db import SearchDB
from emmy.compiler.pipeline.search.policy.terminal_bench import bench_terminal_async


def _candidate():
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (1,)), node_id="x")
    graph.add_node(
        CudaOp(kernel_source='extern "C" __global__ void k(float* out) {}', kernel_name="k", arg_order=("out",)),
        [],
        Tensor("out", (1,)),
        node_id="out",
    )
    graph.inputs = ["x"]
    graph.outputs = ["out"]
    return SimpleNamespace(graph=graph, ctx=Context.from_target((8, 0)))


class _BudgetedBackend:
    name = "cuda"
    bench_run_timeout_s = 2.0

    def __init__(self, iter_ms: float) -> None:
        self.iter_ms = iter_ms
        self.calls: list[tuple[int, int | str]] = []

    async def benchmark_async(self, _graph, *, warmup: int = 5, num_iters: int | str = 20) -> BenchmarkResult:
        self.calls.append((warmup, num_iters))
        total_ms = 0.0
        for _ in range(warmup):
            total_ms += self.iter_ms
            if total_ms > self.bench_run_timeout_s * 1000.0:
                raise RuntimeError("benchmark run stage exceeded 2.0s of GPU time — variant marked bench_fail")
        total_ms += self.iter_ms
        if total_ms > self.bench_run_timeout_s * 1000.0:
            raise RuntimeError("benchmark run stage exceeded 2.0s of GPU time — variant marked bench_fail")
        launch = LaunchTime(idx=0, kernel_name="k", time_ms=self.iter_ms, samples=(self.iter_ms,))
        return BenchmarkResult(time_ms=self.iter_ms, num_launches=1, per_launch=[launch])


async def test_auto_tune_uses_one_nominal_warmup_before_run_budget() -> None:
    backend = _BudgetedBackend(iter_ms=750.0)

    stats, status, measured, _per_kernel = await bench_terminal_async(_candidate(), backend=backend, db=SearchDB())

    assert backend.calls == [(1, "auto")]
    assert measured is True
    assert status == "ok"
    assert stats.median == 750_000.0


class _RaisingBackend:
    """Fails every bench with ``exc``; counts how many times it was actually asked."""

    name = "cuda"
    bench_run_timeout_s = 2.0

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    async def benchmark_async(self, _graph, *, warmup: int = 5, num_iters: int | str = 20) -> BenchmarkResult:
        self.calls += 1
        raise self.exc


def _perf_row(db: SearchDB, cand):
    return db.lookup_perf(cand.ctx.structural_key(), cand.graph.nodes["out"].op.cache_key(), backend="cuda")


async def test_compile_budget_overrun_records_nothing() -> None:
    """A compile-budget overrun measured no latency, so it must persist none. Recording one is
    not merely a wrong label: ``prelude``'s cache lookup accepts any status, so the row would be
    served as a cache hit forever after."""
    db, cand = SearchDB(), _candidate()
    backend = _RaisingBackend(CompileBudgetExceeded("compile stage exceeded 12.0s budget (13.4s) — nothing measured"))

    stats, status, measured, per_kernel = await bench_terminal_async(cand, backend=backend, db=db)

    assert status == "compile_timeout"
    assert measured is True  # it burned real wall time; it must still spend the candidate budget
    assert stats.median == 0.0  # no latency was measured — none is invented
    assert per_kernel == []
    assert _perf_row(db, cand) is None


async def test_compile_budget_overrun_does_not_become_a_sticky_cache_hit() -> None:
    """The regression this fix exists for: after a compile timeout the SAME config must still be
    benchable. Previously the ``bench_fail`` sentinel row was a ``prelude`` cache hit on every
    later terminal and every later sweep, so the config was never re-benched until ``tune
    --clean`` — a slow compile permanently deleted it from the search."""
    db, cand = SearchDB(), _candidate()
    await bench_terminal_async(cand, backend=_RaisingBackend(CompileBudgetExceeded("budget")), db=db)

    good = _BudgetedBackend(iter_ms=1.0)
    stats, status, measured, _ = await bench_terminal_async(cand, backend=good, db=db)

    assert good.calls == [(1, "auto")], "the retry must reach the device, not be served the earlier failure"
    assert (status, measured) == ("ok", True)
    assert stats.median == 1000.0


async def test_worker_side_compile_budget_overrun_is_recognized() -> None:
    """The sweep benches in a subprocess, where the exception CLASS cannot cross the pipe — the
    kind rides back as a flag on ``BenchWorkerJobError`` and must be treated identically."""
    db, cand = SearchDB(), _candidate()
    exc = BenchWorkerJobError("bench worker error: CompileBudgetExceeded(...)", compile_budget=True)

    _stats, status, _measured, _ = await bench_terminal_async(cand, backend=_RaisingBackend(exc), db=db)

    assert status == "compile_timeout"
    assert _perf_row(db, cand) is None


async def test_a_real_bench_failure_still_records_bench_fail() -> None:
    """The narrowing must not swallow genuine failures: a kernel that compiled and then failed
    IS evidence about the kernel, and stays a recorded ``bench_fail`` at the watchdog sentinel."""
    db, cand = SearchDB(), _candidate()

    stats, status, _measured, _ = await bench_terminal_async(cand, backend=_RaisingBackend(RuntimeError("illegal memory access")), db=db)

    assert status == "bench_fail"
    assert stats.median == 2_000_000.0
    assert _perf_row(db, cand).status == "bench_fail"
