"""The deployable eager / torch.compile / emmy comparison runs in the SIGKILL-able worker.

This is what makes ``tune --bench`` / ``run --bench`` survive a hung generated kernel: the whole
comparison (emmy + the torch peer-bench, rebuilt in the child from a recipe) runs in the worker,
so a non-terminating kernel hangs the *child* and the parent SIGKILLs it on ``wall_timeout_s`` —
freeing the device and leaving the parent clean, instead of the ~109-minute in-process wedge.

- ``test_compare_in_worker_returns_torch_and_emmy`` — the happy path: a real frontend graph is
  rebuilt + benched against eager torch in the child, numbers come back.
- ``test_worker_hang_is_sigkilled_not_wedged`` — a worker that hangs on a real non-terminating GPU
  kernel is SIGKILLed at the wall-timeout and surfaces a ``RuntimeError`` promptly (not a wedge).
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
import time

from tests.compiler.helpers import requires_cuda


def test_oneshot_compare_worker_uses_selected_device(monkeypatch) -> None:
    from emmy.compiler.backend.cuda import program

    seen = []

    class Worker:
        def __init__(self, *, device_id=None):
            seen.append(("init", device_id))

        async def run_job(self, request, *, wall_timeout_s):
            seen.append(("run", request, wall_timeout_s))
            return {"ok": True}

        async def aclose(self):
            seen.append(("close",))

    monkeypatch.setattr(program, "_AsyncBenchWorker", Worker)

    result = asyncio.run(program._run_job_oneshot({"job": "compare"}, wall_timeout_s=5.0, device_id=3))

    assert result == {"ok": True}
    assert seen == [("init", 3), ("run", {"job": "compare"}, 5.0), ("close",)]


@requires_cuda
def test_compare_in_worker_returns_torch_and_emmy() -> None:
    from emmy.commands.run import _detect_stage, _passes_after_stage
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.backend.cuda.program import benchmark_compare_isolated_async
    from emmy.compiler.pipeline import Pipeline

    # A small torch_ref-runnable op; lowered in-parent (as the per-kernel sweep does), then the
    # frontend snapshot + lowered graph go to the worker, which rebuilds the torch ref and benches.
    g, _, _ = graph_from_code("torch.nn.RMSNorm(512)(torch.randn(8, 512))")
    fe = g.copy()
    tail = _passes_after_stage(_detect_stage(g))
    lowered = Pipeline.build(tail).run(g) if tail else g

    results, bench, torch_available, _captured = asyncio.run(
        benchmark_compare_isolated_async(
            lowered=lowered,
            torch_spec=("frontend_graph", fe),
            bench_backends="eager,emmy",
            wall_timeout_s=180.0,
            warmup=2,
            iters=5,
            seed=0,
            nvcc_flags="",
        )
    )
    assert torch_available, "the worker should have rebuilt the torch reference from the frontend graph"
    assert results.get("Emmy", 0) > 0, f"missing emmy number: {results}"
    assert results.get("Eager PyTorch", 0) > 0, f"missing eager torch number: {results}"
    assert bench is not None


def _scripted_worker(child_src: str):
    """An ``_AsyncBenchWorker`` whose child runs ``child_src`` instead of the real worker
    module — the harness for exercising the transport against controlled child behavior
    (no CUDA needed when the script never touches the GPU)."""
    from emmy.compiler.backend.cuda.program import _AsyncBenchWorker

    worker = _AsyncBenchWorker()

    async def _spawn() -> None:
        import asyncio as _a

        worker._proc = await _a.create_subprocess_exec(
            sys.executable,
            "-c",
            child_src,
            stdin=_a.subprocess.PIPE,
            stdout=_a.subprocess.PIPE,
            stderr=_a.subprocess.PIPE,
        )
        worker._stderr_tail = ""
        worker._stderr_task = _a.ensure_future(worker._drain_stderr(worker._proc))
        worker.cached_input_keys.clear()

    worker._spawn = _spawn  # type: ignore[method-assign]
    return worker


def test_worker_systemexit_surfaces_cause_and_traceback(caplog) -> None:
    """An in-child ``sys.exit`` (a CLI-style helper) no longer surfaces as an opaque
    ``SystemExit(1)``: the response's error names the exit and points at the traceback,
    which the parent logs instead of discarding."""
    import logging

    import pytest

    from emmy.compiler.backend.cuda.program import BenchWorkerJobError

    child = textwrap.dedent(
        """
        import sys
        import emmy.compiler.backend.cuda._bench_worker as w
        async def _boom(req):
            sys.exit(1)
        w._run_job = _boom
        w.main()
        """
    )
    worker = _scripted_worker(child)
    with caplog.at_level(logging.ERROR, logger="emmy.compiler.backend.cuda.program"):
        with pytest.raises(BenchWorkerJobError, match=r"sys\.exit\(1\)"):
            asyncio.run(_job_then_close(worker, {"torch_spec": None, "kwargs": {}}, wall_timeout_s=60.0))
    assert any("traceback" in rec.message.lower() for rec in caplog.records), "the child traceback must be logged, not discarded"


def test_worker_chatty_stderr_does_not_block_the_job() -> None:
    """A child that writes far past the ~64 KB stderr pipe buffer before responding must
    not deadlock mid-job: the background drain keeps the pipe empty (and keeps only a
    bounded tail)."""
    child = textwrap.dedent(
        """
        import sys
        import emmy.compiler.backend.cuda._bench_worker as w
        sys.stderr.write("x" * 262144)
        sys.stderr.flush()
        async def _ok(req):
            return {"result": "R", "results": None, "torch_available": False, "captured": True}
        w._run_job = _ok
        w.main()
        """
    )
    worker = _scripted_worker(child)
    resp = asyncio.run(_job_then_close(worker, {"torch_spec": None, "kwargs": {}}, wall_timeout_s=60.0))
    assert resp["result"] == "R"
    assert len(worker._stderr_tail) <= worker._STDERR_TAIL_CHARS


def test_worker_wall_timeout_error_carries_stderr_tail() -> None:
    """The SIGKILL-on-wall-timeout error includes the drained stderr tail — exactly the
    context that matters when a hung kernel took the child down."""
    import pytest

    child = 'import sys, time; sys.stderr.write("MARKER-STDERR-TAIL"); sys.stderr.flush(); time.sleep(600)'
    worker = _scripted_worker(child)
    with pytest.raises(RuntimeError, match="MARKER-STDERR-TAIL"):
        asyncio.run(_job_then_close(worker, {"torch_spec": None, "kwargs": {}}, wall_timeout_s=2.0))


async def _job_then_close(worker, req: dict, *, wall_timeout_s: float):
    """Run one job and tear the worker down inside the same event loop (its subprocess
    transport binds to the loop)."""
    try:
        return await worker.run_job(req, wall_timeout_s=wall_timeout_s)
    finally:
        await worker.aclose()


class _HangWorker:
    """An ``_AsyncBenchWorker`` whose child's ``_run_job`` launches a non-terminating GPU kernel and
    blocks on it forever — to exercise the parent's wall-timeout SIGKILL on a genuinely hung worker."""

    _CHILD = textwrap.dedent(
        """
        import emmy.compiler.backend.cuda._bench_worker as w
        import cupy
        def _hang(req):
            spin = cupy.RawKernel(r'extern "C" __global__ void spin(volatile int* f){ while(f[0]==0){} }', 'spin')
            flag = cupy.zeros(1, dtype=cupy.int32)   # never set → infinite loop
            spin((1,), (1,), (flag,))
            cupy.cuda.runtime.deviceSynchronize()    # blocks forever on the hung kernel
            return {}
        w._run_job = _hang
        w.main()
        """
    )

    def __init__(self) -> None:
        from emmy.compiler.backend.cuda.program import _AsyncBenchWorker

        self._impl = _AsyncBenchWorker()
        # Override _spawn to launch our hanging child instead of ``-m _bench_worker``.
        self._impl._spawn = self._spawn_hang  # type: ignore[method-assign]

    async def _spawn_hang(self) -> None:
        import asyncio

        self._impl._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            self._CHILD,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    def run_job(self, *a, **k):
        return self._impl.run_job(*a, **k)

    @property
    def proc(self):
        return self._impl._proc


def test_run_job_send_times_out_on_unresponsive_worker() -> None:
    """A worker that is alive but never reads stdin — e.g. wedged in CUDA-context
    teardown behind a hung kernel after a dirty exit — must trip the wall budget
    during the request SEND (``stdin.drain`` blocks once the request exceeds the
    ~64 KB pipe buffer). No CUDA needed: the deaf child is plain Python."""
    import asyncio

    import pytest

    from emmy.compiler.backend.cuda.program import _AsyncBenchWorker

    worker = _AsyncBenchWorker()

    async def _spawn_deaf() -> None:
        worker._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    worker._spawn = _spawn_deaf  # type: ignore[method-assign]
    t0 = time.time()
    with pytest.raises(RuntimeError, match="did not accept the request"):
        asyncio.run(worker.run_job({"blob": b"x" * (1 << 20)}, wall_timeout_s=2.0))  # 1 MB >> pipe buffer
    assert time.time() - t0 < 20.0, "send must respect the wall budget, not block on the pipe"
    assert worker._proc is None, "the unresponsive worker must be killed and its handle released"


def test_run_job_run_inputs_executes_before_bench(monkeypatch) -> None:
    """A spec-None job with ``run_inputs`` executes the graph once on those inputs — the
    pinned-row wrong-answer gate's measurement side — and ships the outputs back beside
    the bench; without ``run_inputs`` no execution is attempted."""
    from types import SimpleNamespace

    import emmy.compiler.backend.cuda.backend as backend_mod
    import emmy.compiler.backend.cuda.program as program_mod
    from emmy.compiler.backend.cuda._bench_worker import _run_job

    calls: list = []

    class _FakeBackend:
        def __init__(self, **kwargs) -> None:
            pass

        def run(self, graph, *, input_data=None):
            calls.append(("run", input_data))
            return SimpleNamespace(outputs={"n0": [1.0]}), None

    def _fake_bench(graph, **kwargs):
        calls.append(("bench", graph))
        return SimpleNamespace(captured=True)

    monkeypatch.setattr(backend_mod, "CudaBackend", _FakeBackend)
    monkeypatch.setattr(program_mod, "benchmark_program", _fake_bench)

    req = {"graph": "G", "torch_spec": None, "run_inputs": {"x": [0.0]}, "kwargs": {"warmup": 1, "num_iters": 2}}
    resp = asyncio.run(_run_job(req))
    assert resp["run_outputs"] == {"n0": [1.0]}
    assert [c[0] for c in calls] == ["run", "bench"]

    calls.clear()
    resp = asyncio.run(_run_job({"graph": "G", "torch_spec": None, "run_inputs": None, "kwargs": {}}))
    assert resp["run_outputs"] is None
    assert [c[0] for c in calls] == ["bench"]


def test_run_job_caches_run_inputs_by_key(monkeypatch) -> None:
    """The reference inputs cross the pipe once per child: a job carrying key + inputs
    caches them, later key-only jobs reuse the cache, and a key the (respawned) child
    doesn't hold raises the typed cache-miss the parent retries on."""
    from types import SimpleNamespace

    import pytest

    import emmy.compiler.backend.cuda._bench_worker as worker_mod
    import emmy.compiler.backend.cuda.backend as backend_mod
    import emmy.compiler.backend.cuda.program as program_mod
    from emmy.compiler.backend.cuda._bench_worker import InputsCacheMissError, _run_job

    seen_inputs: list = []

    class _FakeBackend:
        def __init__(self, **kwargs) -> None:
            pass

        def run(self, graph, *, input_data=None):
            seen_inputs.append(input_data)
            return SimpleNamespace(outputs={"n0": [1.0]}), None

    monkeypatch.setattr(backend_mod, "CudaBackend", _FakeBackend)
    monkeypatch.setattr(program_mod, "benchmark_program", lambda graph, **kwargs: SimpleNamespace(captured=True))
    monkeypatch.setattr(worker_mod, "_RUN_INPUTS_CACHE", {})

    inputs = {"x": [0.0]}
    asyncio.run(_run_job({"graph": "G", "torch_spec": None, "run_inputs": inputs, "run_inputs_key": "K", "kwargs": {}}))
    asyncio.run(_run_job({"graph": "G", "torch_spec": None, "run_inputs": None, "run_inputs_key": "K", "kwargs": {}}))
    assert seen_inputs == [inputs, inputs]  # second job ran on the cached set
    with pytest.raises(InputsCacheMissError):
        asyncio.run(_run_job({"graph": "G", "torch_spec": None, "run_inputs": None, "run_inputs_key": "OTHER", "kwargs": {}}))


def test_benchmark_pinned_isolated_async_ships_inputs_once_and_retries_on_miss() -> None:
    """Parent side of the input cache: the first row sends the inputs, later rows send the
    key alone, and a cache-miss job error (the child respawned after the key was tracked)
    retries once with the inputs included."""
    from emmy.compiler.backend.cuda.program import BenchWorkerJobError, benchmark_pinned_isolated_async

    sent: list = []

    class _FakeWorker:
        def __init__(self) -> None:
            self.cached_input_keys: set[str] = set()
            self.fail_next = False

        async def run_job(self, req, *, wall_timeout_s):
            sent.append(req)
            if self.fail_next:
                self.fail_next = False
                raise BenchWorkerJobError("bench worker error: run_inputs 'K' not cached", cache_miss=True)
            return {"result": "B", "run_outputs": {"o": 1}}

    worker = _FakeWorker()
    inputs = {"x": [0.0]}

    def _row():
        return asyncio.run(
            benchmark_pinned_isolated_async(
                "G", worker=worker, wall_timeout_s=1.0, run_inputs=inputs, run_inputs_key="K", warmup=1, num_iters=1
            )
        )

    _row()
    assert sent[-1]["run_inputs"] == inputs  # first row ships the inputs
    _row()
    assert sent[-1]["run_inputs"] is None and sent[-1]["run_inputs_key"] == "K"  # later rows send the key alone
    worker.fail_next = True  # child respawned since the key was tracked
    bench, outputs = _row()
    assert sent[-2]["run_inputs"] is None and sent[-1]["run_inputs"] == inputs  # miss → one retry with the inputs
    assert bench == "B" and outputs == {"o": 1}


def test_run_job_trace_args_accuracy_gates_the_bench(monkeypatch) -> None:
    """A ``trace_args`` job with ``accuracy``: the child binds the rebuilt module's real
    inputs, runs the emmy program on them, and compares vs eager. A numeric failure ships
    the verdict back WITHOUT benching (the parent aborts on it — a latency table for a
    miscompiling program is meaningless); a pass with ``want_ref`` returns that run's
    ``(inputs, outputs)`` as the pinned rows' wrong-answer reference."""
    from types import SimpleNamespace

    import emmy.commands.compile as compile_mod
    import emmy.commands.run as run_mod
    import emmy.compiler.backend.cuda.backend as backend_mod
    from emmy.compiler.backend.cuda._bench_worker import _run_job

    class _FakeBackend:
        def __init__(self, **kwargs) -> None:
            pass

        def run(self, graph, *, input_data=None):
            return SimpleNamespace(outputs={"n0": [2.0]}), None

    benched: list = []

    async def _fake_full_model(module, args_t, kwargs, graph, backend, *, warmup, iters, bench_backends):
        benched.append(graph)
        return {"Emmy": 1.0}, SimpleNamespace(captured=True), True

    monkeypatch.setattr(compile_mod, "load_or_trace", lambda ns: (None, None, (object(), (), {})))
    monkeypatch.setattr(backend_mod, "CudaBackend", _FakeBackend)
    monkeypatch.setattr(run_mod, "_bind_inputs", lambda g, m, a, k, checkpoint=None: {"x": [0.0]})
    monkeypatch.setattr(run_mod, "_eager_output", lambda m, a, k: "EAGER")
    monkeypatch.setattr(run_mod, "bench_full_model_real", _fake_full_model)

    req = {
        "graph": "G",
        "torch_spec": ("trace_args", {}),
        "bench_backends": "emmy",
        "warmup": 1,
        "iters": 2,
        "accuracy": True,
        "want_ref": True,
    }

    monkeypatch.setattr(run_mod, "_check_accuracy", lambda outs, eager: "accuracy check failed vs eager: output n0")
    resp = asyncio.run(_run_job(dict(req)))
    assert resp["accuracy_error"] is not None and resp["result"] is None and not benched

    monkeypatch.setattr(run_mod, "_check_accuracy", lambda outs, eager: None)
    resp = asyncio.run(_run_job(dict(req)))
    assert resp["accuracy_error"] is None and resp["run_io"] == ({"x": [0.0]}, {"n0": [2.0]})
    assert benched and resp["results"] == {"Emmy": 1.0}


@requires_cuda
def test_worker_hang_is_sigkilled_not_wedged() -> None:
    import asyncio

    import pytest

    worker = _HangWorker()
    t0 = time.time()
    # The child launches an infinite kernel and never responds; the parent must SIGKILL it at the
    # 3 s wall budget and raise — not block forever (the pre-isolation in-process failure mode).
    with pytest.raises(RuntimeError, match="wall budget"):
        asyncio.run(worker.run_job({"graph": None, "torch_spec": None, "kwargs": {}}, wall_timeout_s=3.0))
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"run_job took {elapsed:.1f}s — the wall-timeout SIGKILL did not fire promptly"
    assert worker.proc is None, "the hung worker must be killed and its handle released"
