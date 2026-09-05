"""The persistent bench worker must not serve from a corrupted CUDA context.

A kernel that does an illegal / misaligned memory access leaves the CUDA
context in a *sticky*-error state: every subsequent CUDA call returns the same
error until the context is destroyed. The bench worker is a long-lived
subprocess reused across every autotune config, so a single such crash used to
cascade identical false ``bench_fail``s across all later configs (and ops).

The worker now probes its context after a failure (``_context_dirty``) and
exits if it's poisoned, so the parent (``program.py`` ``_AsyncBenchWorker.run_job``)
respawns a clean context on the next request. A benign failure (NVRTC compile
error, etc.) leaves the context healthy and keeps the worker alive.

These tests drive the real ``_bench_worker.main`` loop over the real
length-prefixed pickle protocol, with ``benchmark_program`` monkeypatched in the
child to either corrupt the context or raise without touching CUDA.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import subprocess
import sys
import textwrap

import pytest

from emmy.compiler.backend import BenchmarkResult
from emmy.compiler.backend.cuda import program as P
from tests.compiler.helpers import requires_cuda

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _spawn(child_src: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(child_src)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=PROJECT_ROOT,
    )


def _send(proc: subprocess.Popen, obj: object) -> None:
    body = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    proc.stdin.write(len(body).to_bytes(8, "little"))
    proc.stdin.write(body)
    proc.stdin.flush()


def _recv(proc: subprocess.Popen) -> dict | None:
    header = proc.stdout.read(8)
    if len(header) < 8:
        return None  # worker exited / closed stdout
    n = int.from_bytes(header, "little")
    return pickle.loads(proc.stdout.read(n))


# The patched ``benchmark_program`` ignores the request graph, so the parent can
# send a dummy request — no real compiled Graph needed.
_DUMMY_REQ = {"graph": None, "kwargs": {}}


# ---------------------------------------------------------------------------
# Parent-side fakes: a worker child as ``_AsyncBenchWorker.run_job`` sees it
# ---------------------------------------------------------------------------


def _wire(resp: dict) -> bytes:
    """One framed response as the child writes it."""
    body = pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL)
    return len(body).to_bytes(8, "little") + body


class _FakeStdin:
    def __init__(self, *, broken: bool) -> None:
        self._broken = broken

    def write(self, _data: bytes) -> None:
        pass

    async def drain(self) -> None:
        if self._broken:
            raise BrokenPipeError("stale worker — read end closed")


class _FakeStdout:
    def __init__(self, wire: bytes) -> None:
        self._buf = bytearray(wire)

    async def readexactly(self, n: int) -> bytes:
        if len(self._buf) < n:
            raise asyncio.IncompleteReadError(bytes(self._buf), n)
        chunk = bytes(self._buf[:n])
        del self._buf[:n]
        return chunk


class _FakeStderr:
    async def read(self) -> bytes:
        return b""


class _FakeProc:
    """``wire`` is what the child's stdout answers (``b""`` = EOF); ``fail_send`` breaks the pipe on
    the first write. ``kill`` records the SIGKILL as ``returncode``."""

    def __init__(self, *, wire: bytes = b"", fail_send: bool = False) -> None:
        self.pid = 1000
        self.returncode = None
        self.stdin = _FakeStdin(broken=fail_send)
        self.stdout = _FakeStdout(wire)
        self.stderr = _FakeStderr()

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return -9


def _spawning(monkeypatch, procs: list[_FakeProc]) -> list[int]:
    """Hand ``procs`` out in order on each ``_spawn``; returns the spawn counter (one cell)."""
    spawned = [0]

    async def fake_spawn(self: P._AsyncBenchWorker) -> None:
        self._proc = procs[spawned[0]]
        spawned[0] += 1

    monkeypatch.setattr(P._AsyncBenchWorker, "_spawn", fake_spawn)
    return spawned


def _job(w: P._AsyncBenchWorker) -> dict:
    return asyncio.run(w.run_job({"graph": None, "torch_spec": None, "kwargs": {}}, wall_timeout_s=5.0))


@requires_cuda
def test_worker_exits_after_context_corruption() -> None:
    # Child: first bench launches an out-of-bounds write (sticky illegal
    # access), then raises. The worker should detect the dirty context, answer
    # the first request with the error, and exit — so the *second* request gets
    # no response (EOF), proving it won't serve from the poisoned context.
    child = """
        import emmy.compiler.backend.cuda.program as program
        def _corrupt(graph, **kw):
            import cupy
            k = cupy.RawKernel(r'extern "C" __global__ void oob(float* p){ p[268435456] = 1.0f; }', 'oob')
            buf = cupy.zeros(8, dtype=cupy.float32)
            k((1,), (1,), (buf,))
            cupy.cuda.runtime.deviceSynchronize()  # surfaces the sticky error
            raise RuntimeError('unreached')
        program.benchmark_program = _corrupt
        from emmy.compiler.backend.cuda._bench_worker import main
        main()
    """
    proc = _spawn(child)
    try:
        _send(proc, _DUMMY_REQ)
        resp1 = _recv(proc)
        assert resp1 is not None and resp1["ok"] is False
        # The worker must not answer a second request — its context is dirty.
        _send(proc, _DUMMY_REQ)
        resp2 = _recv(proc)
        assert resp2 is None, f"worker kept serving from a corrupted context: {resp2}"
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


def test_kill_idempotent_when_no_proc() -> None:
    """``_kill()`` on a worker that was never spawned must be a silent no-op."""
    from emmy.compiler.backend.cuda.program import _AsyncBenchWorker

    w = _AsyncBenchWorker()
    assert w._proc is None
    w._kill()
    assert w._proc is None
    w._kill()  # repeated calls stay no-ops
    assert w._proc is None


def test_kill_releases_already_dead_subprocess() -> None:
    """A worker subprocess that exited on its own (e.g. dirty-context path) is
    still attached to ``self._proc``; ``_kill()`` must release it without raising
    (a dead proc has ``returncode`` set, so no SIGKILL is attempted)."""
    import asyncio

    from emmy.compiler.backend.cuda.program import _AsyncBenchWorker

    async def _run() -> None:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import sys; sys.exit(0)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert await proc.wait() == 0
        w = _AsyncBenchWorker()
        w._proc = proc
        w._kill()
        assert w._proc is None

    asyncio.run(_run())


def test_bench_retries_after_broken_pipe_on_first_write(monkeypatch) -> None:
    """The dirty-context exit path can race a respawn: the first ``stdin.drain``
    raises ``BrokenPipeError`` (the worker's read end is gone). ``run_job`` must
    respawn and retry the send once before surfacing the failure."""
    ok = _wire({"ok": True, "result": BenchmarkResult(time_ms=42.0, num_launches=0)})
    spawned = _spawning(monkeypatch, [_FakeProc(fail_send=True), _FakeProc(wire=ok)])

    resp = _job(P._AsyncBenchWorker())

    assert spawned[0] == 2, "BrokenPipeError on first send must trigger one respawn"
    assert resp["result"].time_ms == 42.0


def test_bench_retries_after_mid_job_eof(monkeypatch) -> None:
    """A response-side EOF (the child ``os._exit``'d mid-job) must respawn and retry ONCE:
    right after a SIGKILL'd predecessor, the dead child's zombie context can still hold the
    GPU while the driver tears it down, hanging an innocent first launch on the fresh child
    (the golden-refresh flake — the same row replays clean once the zombie is gone). A second
    EOF is the config's own hang and stays a hard error (the test below)."""
    ok = _wire({"ok": True, "result": BenchmarkResult(time_ms=17.0, num_launches=0)})

    def _run(wires: list[bytes]):
        spawned = _spawning(monkeypatch, [_FakeProc(wire=w_) for w_ in wires])
        return spawned, _job(P._AsyncBenchWorker())

    # First child EOFs mid-response (empty stdout), second answers: one retry, success.
    spawned, resp = _run([b"", ok])
    assert spawned[0] == 2, "a mid-job EOF must trigger exactly one respawn + retry"
    assert resp["result"].time_ms == 17.0

    # Both children EOF: the config's own hang — a hard error after the single retry.
    with pytest.raises(RuntimeError, match="EOF before response"):
        _run([b"", b""])


# ---------------------------------------------------------------------------
# A hung kernel retires the child: the parent SIGKILLs + reaps it on the error response
# ---------------------------------------------------------------------------


def test_hung_kernel_response_retires_the_child(monkeypatch) -> None:
    """A hung kernel stays resident on the device until its context dies, and the child cannot end
    that context on its own: its interpreter exit blocks in the CUDA teardown behind the kernel.
    Left to itself the child became a zombie that still held the GPU, the next candidate's request
    wedged against it before any launch, and the wall budget priced THAT configuration as a failure
    — the "did not accept the request" rows for kernels that never ran. The error response flags
    the retirement and the parent takes the same teardown a wall overrun takes: SIGKILL + reap, so
    the next request spawns on a clean device."""
    watchdog = "HungKernelError(\"kernel 'k (iter 0)' did not complete\")"
    hung = _wire({"ok": False, "error": watchdog, "traceback": "", "_retire_worker": True})
    ok = _wire({"ok": True, "result": BenchmarkResult(time_ms=3.0, num_launches=0)})
    first, second = _FakeProc(wire=hung), _FakeProc(wire=ok)
    spawned = _spawning(monkeypatch, [first, second])
    w = P._AsyncBenchWorker()

    with pytest.raises(P.BenchWorkerJobError, match="HungKernelError"):
        _job(w)

    assert first.returncode == -9, "the hung child must be SIGKILLed with its resident kernel"
    assert w._proc is None, "and reaped, so nothing is ever sent to the zombie"
    assert _job(w)["result"].time_ms == 3.0
    assert spawned[0] == 2, "the next candidate runs on a fresh child"


def test_benign_error_response_keeps_the_child(monkeypatch) -> None:
    """The retirement is the child's verdict, not the parent's guess: a failure that left the
    context healthy keeps the worker, so rejected configs pay no respawn."""
    benign = _wire({"ok": False, "error": "ValueError('nvrtc')", "traceback": "", "_retire_worker": False})
    proc = _FakeProc(wire=benign)
    _spawning(monkeypatch, [proc])
    w = P._AsyncBenchWorker()

    with pytest.raises(P.BenchWorkerJobError, match="nvrtc"):
        _job(w)

    assert proc.returncode is None and w._proc is proc


def test_worker_flags_a_hung_kernel_for_retirement() -> None:
    """The child's side of the contract: the watchdog's error response carries the retirement flag
    (the parent cannot probe a context a hung kernel holds), and the child serves nothing further."""
    child = """
        import emmy.compiler.backend.cuda.program as program
        def _hang(graph, **kw):
            raise program.HungKernelError("kernel 'k (iter 0)' did not complete within 2000 ms — variant marked bench_fail")
        program.benchmark_program = _hang
        from emmy.compiler.backend.cuda._bench_worker import main
        main()
    """
    proc = _spawn(child)
    try:
        _send(proc, _DUMMY_REQ)
        resp = _recv(proc)
        assert resp is not None and resp["ok"] is False
        assert resp["_retire_worker"] is True, "a hung kernel must ask the parent for the SIGKILL"
        _send(proc, _DUMMY_REQ)
        assert _recv(proc) is None, "the child must not serve another request from that context"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


def test_worker_survives_benign_error() -> None:
    # A failure that never touches CUDA (e.g. an NVRTC compile error) leaves the
    # context healthy — the worker must stay alive and keep serving so the
    # autotune sweep doesn't pay a respawn per rejected config.
    child = """
        import emmy.compiler.backend.cuda.program as program
        def _benign(graph, **kw):
            raise ValueError('benign compile-like failure')
        program.benchmark_program = _benign
        from emmy.compiler.backend.cuda._bench_worker import main
        main()
    """
    proc = _spawn(child)
    try:
        _send(proc, _DUMMY_REQ)
        resp1 = _recv(proc)
        assert resp1 is not None and resp1["ok"] is False
        # Still alive: a second request gets a real response, not EOF.
        _send(proc, _DUMMY_REQ)
        resp2 = _recv(proc)
        assert resp2 is not None and resp2["ok"] is False, "worker exited on a benign error"
    finally:
        proc.stdin.close()
        if proc.poll() is None:
            proc.wait(timeout=5)
        if proc.poll() is None:
            proc.kill()
