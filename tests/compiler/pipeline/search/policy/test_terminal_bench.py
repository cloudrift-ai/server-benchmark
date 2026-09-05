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
    return db.lookup_perf(cand.ctx.structural_key(), cand.graph.nodes["out"].op.identity_key(with_io=True, with_knobs=True), backend="cuda")


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


def _candidate_pair():
    """A terminal with TWO kernels — the shape one hang used to condemn wholesale."""
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (1,)), node_id="x")
    # The bodies must differ materially: ``CudaOp`` identity normalizes the kernel NAME away, so
    # two kernels differing only in name are one identity and would share a single perf row.
    for nid, name, body in (("mid", "k_innocent", "out[0] = 1.0f;"), ("out", "k_culprit", "out[0] = 2.0f;")):
        graph.add_node(
            CudaOp(
                kernel_source=f'extern "C" __global__ void {name}(float* out) {{ {body} }}',
                kernel_name=name,
                arg_order=("out",),
            ),
            [],
            Tensor(nid, (1,)),
            node_id=nid,
        )
    graph.inputs, graph.outputs = ["x"], ["out"]
    return SimpleNamespace(graph=graph, ctx=Context.from_target((8, 0)))


def _fail_rows(db: SearchDB, cand) -> dict[str, str]:
    """``kernel_name -> status`` for every kernel of ``cand`` that has a perf row."""
    out = {}
    for nid in ("mid", "out"):
        op = cand.graph.nodes[nid].op
        row = db.lookup_perf(cand.ctx.structural_key(), op.identity_key(with_io=True, with_knobs=True), backend="cuda")
        if row is not None:
            out[op.kernel_name] = row.status
    return out


async def test_a_hung_kernel_is_blamed_alone() -> None:
    """The watchdog NAMES the kernel that hung, so only that kernel earns the ``bench_fail`` row.

    A terminal benches its kernels together and one hang fails the whole run; recording the
    failure against all of them manufactures evidence about kernels never shown to fail. Measured
    on DeepSeek-V4's post block: 70 recorded failures carrying only 7 distinct errors, 20 kernels
    condemned by a single hang."""
    db, cand = SearchDB(), _candidate_pair()
    exc = BenchWorkerJobError("bench worker error: HungKernelError(\"kernel 'k_culprit (iter 0)' did not complete within 60000 ms\")")

    _stats, status, _measured, per_kernel = await bench_terminal_async(cand, backend=_RaisingBackend(exc), db=db)

    assert status == "bench_fail", "the terminal still failed — the search must move on"
    assert _fail_rows(db, cand) == {"k_culprit": "bench_fail"}, "the innocent kernel must carry no failure"
    assert len(per_kernel) == 1, "only the culprit trains the prior on this failure"


async def test_a_blamed_kernel_replays_the_hang_for_its_slice() -> None:
    """The culprit's row is the slice's evidence: a kernel that hung hangs every slice it is in, so
    on the next session the slice must be served ``bench_fail`` from the cache without the innocent
    kernel needing a row of its own. Before, the all-or-nothing lookup missed on the innocent kernel
    and every fresh session re-benched the hang — about 15 minutes each on the DeepSeek-V4-Flash
    post4096 twin, whose composed fused / serial arms were effectively uncacheable."""
    db, cand = SearchDB(), _candidate_pair()
    hang = BenchWorkerJobError("bench worker error: HungKernelError(\"kernel 'k_culprit (iter 0)' did not complete within 60000 ms\")")
    await bench_terminal_async(cand, backend=_RaisingBackend(hang), db=db)

    retry = _BudgetedBackend(iter_ms=1.0)
    _stats, status, measured, per_kernel = await bench_terminal_async(_candidate_pair(), backend=retry, db=db)

    assert retry.calls == [], "a slice holding a kernel that hung is known-hung — it must not reach the device"
    assert (status, measured) == ("bench_fail", False)
    assert _fail_rows(db, cand) == {"k_culprit": "bench_fail"}, "the replay adds no blame"
    assert [st for _knobs, _us, st in per_kernel] == ["bench_fail"], "only the culprit's row is evidence"


async def test_an_unattributable_failure_blames_no_kernel() -> None:
    """A bench-worker startup timeout is not a property of any kernel — it names none, and with
    several kernels in the terminal there is no unambiguous culprit. Unknown is not failed, so
    nothing is persisted; the terminal still reports ``bench_fail`` and the candidate is spent."""
    db, cand = SearchDB(), _candidate_pair()
    exc = RuntimeError("bench worker did not accept the request within 74.0s wall budget — SIGKILL'd, stream cleaned")

    _stats, status, _measured, per_kernel = await bench_terminal_async(cand, backend=_RaisingBackend(exc), db=db)

    assert status == "bench_fail"
    assert _fail_rows(db, cand) == {}
    assert per_kernel == []
