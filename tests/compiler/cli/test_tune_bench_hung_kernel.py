"""``tune --bench`` runs its deployable benches in the SIGKILL-able worker, so a hung kernel can't
wedge the run — and, because the parent device stays clean, a failed full-model bench no longer has
to skip the per-kernel sweep (the pre-isolation behavior); it just continues.

A non-terminating kernel trips the worker's per-launch watchdog (``HungKernelError``) and the parent
SIGKILLs the child, surfacing as a ``RuntimeError`` to ``_run_bench``. These tests drive the
``_run_bench`` control flow with the worker call (``benchmark_compare_isolated_async``) and the per-kernel
sweep stubbed — no CUDA. The real-GPU recovery is covered in
``tests/compiler/backend/test_bench_worker_compare.py``.
"""

from __future__ import annotations

import json
import types

import emmy.commands.run as run_mod
import emmy.commands.tune as tune_mod
import emmy.compiler.backend.cuda.backend as backend_mod
import emmy.compiler.backend.cuda.program as program_mod
from emmy import config
from emmy.compiler.backend.cuda.program import HungKernelError


class _DummyBackend:
    """Stands in for ``CudaBackend`` — ``tune_db=None`` keeps ``_run_bench`` off the SearchDB path."""

    def __init__(self, *, tune_db=None, **_kw) -> None:  # noqa: ARG002
        self.tune_db = None


def _args() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        nvcc_flags=None,
        warmup=1,
        iters=1,
        seed=0,
        bench_backends="emmy",
        code=None,
        input="some/model",
        layer=0,
        seq_len=32,
        dynamic=None,
    )


def _patch_common(
    monkeypatch,
    *,
    compare_raises: Exception | None,
    seen_devices: list[int | None] | None = None,
    seen_contexts: list[object] | None = None,
) -> list[bool]:
    """Stub the worker compare call + the per-kernel sweep; return a flag flipped iff per-kernel ran."""
    per_kernel_ran = [False]

    async def _fake_compare(**_kw):
        if seen_devices is not None:
            seen_devices.append(_kw.get("device_id"))
        if compare_raises is not None:
            raise compare_raises
        return {"Emmy": 1.0}, object(), True, False, None  # results, bench, torch_available, captured, accuracy_error

    def _fake_per_kernel(*_a, **_k):
        if seen_devices is not None:
            seen_devices.append(_k.get("device_id"))
        if seen_contexts is not None:
            seen_contexts.append(_k.get("ctx"))
        per_kernel_ran[0] = True
        return [], []

    monkeypatch.setattr(backend_mod, "CudaBackend", _DummyBackend)
    monkeypatch.setattr(program_mod, "benchmark_compare_isolated_async", _fake_compare)
    monkeypatch.setattr(tune_mod, "_bench_per_kernel", _fake_per_kernel)
    monkeypatch.setattr(tune_mod, "_context_for_device", lambda *_args, **_kwargs: "selected-o3-context")
    monkeypatch.setattr(run_mod, "_print_table", lambda *_a, **_k: None)
    monkeypatch.setenv(config.NVCC_FLAGS, "")  # registers cleanup of the flag _run_bench sets
    return per_kernel_ran


def test_hung_kernel_error_is_runtimeerror() -> None:
    # Subclassing RuntimeError keeps every existing ``except RuntimeError`` (the autotune sweep's
    # bench_fail handling, _run_bench's continue) catching it unchanged.
    assert issubclass(HungKernelError, RuntimeError)


def test_run_bench_continues_to_per_kernel_on_full_model_failure(monkeypatch) -> None:
    # The worker SIGKILLs a hung kernel and surfaces a RuntimeError; the parent device is clean
    # (the bench ran in the child), so the per-kernel sweep must still run — no skip.
    ran = _patch_common(monkeypatch, compare_raises=RuntimeError("bench worker exceeded wall budget — SIGKILL'd"))
    dump = types.SimpleNamespace(dir="/tmp/does-not-matter")

    tune_mod._run_bench(_args(), ("module", "args", "kwargs"), assembled=None, dump=dump, html_dir=None)

    assert ran[0] is True, "per-kernel bench must still run after an isolated full-model failure"


def test_run_bench_runs_per_kernel_on_success(monkeypatch) -> None:
    ran = _patch_common(monkeypatch, compare_raises=None)
    dump = types.SimpleNamespace(dir="/tmp/does-not-matter")

    tune_mod._run_bench(_args(), ("module", "args", "kwargs"), assembled=None, dump=dump, html_dir=None)

    assert ran[0] is True, "per-kernel bench must run after a successful full-model bench"


def test_run_bench_pins_full_and_per_kernel_workers(monkeypatch) -> None:
    seen_devices: list[int | None] = []
    seen_contexts: list[object] = []
    _patch_common(monkeypatch, compare_raises=None, seen_devices=seen_devices, seen_contexts=seen_contexts)
    dump = types.SimpleNamespace(dir="/tmp/does-not-matter")

    tune_mod._run_bench(
        _args(),
        ("module", "args", "kwargs"),
        assembled=None,
        dump=dump,
        html_dir=None,
        device_id=3,
    )

    assert seen_devices == [3, 3]
    assert seen_contexts == ["selected-o3-context"]


def test_per_kernel_bench_persists_and_surfaces_accuracy_error(monkeypatch, tmp_path, capsys) -> None:
    """The deployable artifact must retain the correctness verdict returned by its worker."""

    class _Graph:
        def copy(self):
            return self

    async def _fake_compare(**_kwargs):
        return {"Eager PyTorch": 4.0, "Emmy": 2.0}, object(), True, True, "wrong-answer: rel err 1.000"

    dump = types.SimpleNamespace(dir=tmp_path, frontend_reproducers=lambda: {"k_test_abc123": _Graph()})
    monkeypatch.setattr(program_mod, "benchmark_compare_isolated_async", _fake_compare)
    monkeypatch.setattr("emmy.compiler.backend.torch_ref.is_runnable", lambda _graph: True)
    monkeypatch.setattr(run_mod, "_detect_stage", lambda _graph: "frontend")
    monkeypatch.setattr(run_mod, "_passes_after_stage", lambda _stage: [])

    rows, fallback = tune_mod._bench_per_kernel(_args(), dump, db=None)

    assert rows == [("k_test", {"Eager PyTorch": 4.0, "Emmy": 2.0})]
    assert fallback == []
    [record] = json.loads((tmp_path / "62_kernel_bench.json").read_text())
    assert record["reference_available"] is True
    assert record["accuracy_error"] == "wrong-answer: rel err 1.000"
    assert "accuracy failed (wrong-answer: rel err 1.000)" in capsys.readouterr().err
