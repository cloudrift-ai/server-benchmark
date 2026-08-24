from __future__ import annotations

from types import SimpleNamespace

from emmy.compiler.backend.base import BenchmarkResult, LaunchTime
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
