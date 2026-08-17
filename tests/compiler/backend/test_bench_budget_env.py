"""``EMMY_BENCH_*`` env overrides of the bench budgets — read live through ``emmy.config``.

The budgets are constructor policy on the backend (``tune`` shrinks them, the bench worker
raises them); the paired env vars override every caller uniformly, read on every access
(never cached), so a raised budget reaches the in-child backend and the pinned-row wall
cap alike.
"""

from __future__ import annotations

from emmy import config
from emmy.compiler.backend import Backend
from emmy.compiler.backend.cuda.backend import CudaBackend


class _StubBackend(Backend):
    def compile(self, graph):
        return graph


def test_defaults_unchanged_without_env(monkeypatch):
    monkeypatch.delenv(config.BENCH_COMPILE_TIMEOUT_S, raising=False)
    monkeypatch.delenv(config.BENCH_RUN_TIMEOUT_S, raising=False)
    monkeypatch.delenv(config.BENCH_WALL_TIMEOUT_S, raising=False)
    backend = _StubBackend()
    assert backend.bench_compile_timeout_s == 30.0
    assert backend.bench_run_timeout_s == 10.0
    assert backend.bench_wall_timeout_s is None


def test_env_overrides_constructor_policy(monkeypatch):
    backend = CudaBackend(bench_compile_timeout_s=12.0, bench_run_timeout_s=2.0, bench_wall_timeout_s=16.0)
    monkeypatch.setenv(config.BENCH_COMPILE_TIMEOUT_S, "120")
    monkeypatch.setenv(config.BENCH_RUN_TIMEOUT_S, "300")
    monkeypatch.setenv(config.BENCH_WALL_TIMEOUT_S, "500")
    assert backend.bench_compile_timeout_s == 120.0
    assert backend.bench_run_timeout_s == 300.0
    assert backend.bench_wall_timeout_s == 500.0


def test_reads_are_live_not_cached(monkeypatch):
    backend = CudaBackend()
    monkeypatch.setenv(config.BENCH_RUN_TIMEOUT_S, "90")
    assert backend.bench_run_timeout_s == 90.0
    monkeypatch.delenv(config.BENCH_RUN_TIMEOUT_S)
    assert backend.bench_run_timeout_s == 10.0


def test_wall_env_selects_isolated_budget_over_none(monkeypatch):
    backend = CudaBackend()  # constructor default: None (in-process bench)
    monkeypatch.setenv(config.BENCH_WALL_TIMEOUT_S, "42.5")
    assert backend.bench_wall_timeout_s == 42.5


def test_unparseable_env_falls_back_to_policy_value(monkeypatch):
    backend = CudaBackend(bench_run_timeout_s=2.0)
    monkeypatch.setenv(config.BENCH_RUN_TIMEOUT_S, "not-a-number")
    assert backend.bench_run_timeout_s == 2.0
