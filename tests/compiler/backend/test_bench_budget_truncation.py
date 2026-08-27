"""The run-stage GPU budget truncates instead of failing once enough samples exist, and a heavy
launch collapses the remaining warmup — so slow-but-real kernels yield measurements the tuner can
rank, while a kernel that can't produce ``_MIN_MEASURED_ON_BUDGET`` samples stays a bench_fail.

No CUDA: ``CompiledProgram.build`` and the gpu lock are stubbed; the fake program plays back a fixed
per-launch GPU time per iter.
"""

from __future__ import annotations

import contextlib
import types

import pytest

import emmy.compiler.backend.cuda.program as program_mod


class _FakeProg:
    def __init__(self, per_iter_ms: float) -> None:
        self._dt = per_iter_ms
        launch = types.SimpleNamespace(kernel_name="k_fake")
        self.compiled = types.SimpleNamespace(launches=[launch])
        self.iters = 0

    def iter_once(self, batch_sizes=None, pre_iter=None):  # noqa: ARG002
        self.iters += 1
        return [self._dt]

    def capture_launch_graphs(self, sizes):  # noqa: ARG002
        raise program_mod.GraphCaptureError("no CUDA in this test")

    def time_program_window(self, replays):  # pragma: no cover - single-launch program
        raise AssertionError("single-launch program never times a whole-program window")


@pytest.fixture
def _no_cuda(monkeypatch):
    monkeypatch.setattr(program_mod, "gpu_lock", lambda: contextlib.nullcontext(), raising=False)
    import emmy.compiler.backend.gpu_lock as lock_mod

    monkeypatch.setattr(lock_mod, "gpu_lock", lambda: contextlib.nullcontext())


def _bench(monkeypatch, per_iter_ms: float, *, warmup: int, iters: int, budget_s: float):
    prog = _FakeProg(per_iter_ms)
    monkeypatch.setattr(program_mod.CompiledProgram, "build", staticmethod(lambda *a, **k: prog))
    result = program_mod.benchmark_program(
        graph=None, input_data=None, warmup=warmup, num_iters=iters, run_timeout_s=budget_s, capture_graphs=False
    )
    return prog, result


def test_budget_truncates_with_enough_samples(_no_cuda, monkeypatch):
    # 200 ms/iter against a 2 s budget: 10 iters fit. Warmup collapses after the first slow iter,
    # leaving ~9 measured — well past the floor, so the result is returned truncated, not raised.
    prog, result = _bench(monkeypatch, 200.0, warmup=10, iters=100, budget_s=2.0)
    launch = result.per_launch[0]
    assert launch.samples and program_mod._MIN_MEASURED_ON_BUDGET <= len(launch.samples) < 100
    assert launch.time_ms == pytest.approx(200.0)


def test_budget_still_fails_below_sample_floor(_no_cuda, monkeypatch):
    # 900 ms/iter against a 2 s budget: only ~2 iters fit in total — under the floor, bench_fail.
    with pytest.raises(RuntimeError, match="benchmark run stage exceeded"):
        _bench(monkeypatch, 900.0, warmup=10, iters=100, budget_s=2.0)


def test_slow_launch_collapses_warmup(_no_cuda, monkeypatch):
    # Above _SLOW_LAUNCH_WARMUP_MS the first iter ends warmup: with no budget pressure the
    # requested iters all land as measured samples instead of 10 warmup discards.
    prog, result = _bench(monkeypatch, 60.0, warmup=10, iters=5, budget_s=None)
    assert len(result.per_launch[0].samples) == 5
    assert prog.iters == 6  # 1 collapsed warmup iter + 5 measured


def test_fast_kernels_keep_full_warmup(_no_cuda, monkeypatch):
    # Below the threshold the warmup contract is unchanged (modulo the existing clock-ramp
    # extension, which this per-iter time is large enough not to trigger).
    prog, result = _bench(monkeypatch, 20.0, warmup=4, iters=3, budget_s=None)
    assert len(result.per_launch[0].samples) == 3
    assert prog.iters == 7  # 4 warmup + 3 measured
