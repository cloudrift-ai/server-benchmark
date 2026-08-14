"""Tests for repeated raw benchmark execution."""

import asyncio
from pathlib import Path

from emmy.benchmark.workload import capture_server_log, run_benchmark_workload
from emmy.recipe.types import Recipe


def _stanza(ttft: float, tokps: float) -> str:
    return f"""============ Serving Benchmark Result ============
Successful requests:                     32
Failed requests:                         0
Benchmark duration (s):                  10.00
Output token throughput (tok/s):         {tokps}
Median TTFT (ms):                        {ttft}
==================================================
"""


def _recipe(repeats: int = 1) -> Recipe:
    return Recipe.from_dict(
        {
            "model": {"huggingface": "google/gemma-4-12B-it"},
            "engine": {"llm": {"context_length": 8192, "vllm": {}}},
            "benchmark": {
                "max_concurrency": 1,
                "num_prompts": 32,
                "random_input_len": 4096,
                "random_output_len": 4096,
                "repeats": repeats,
            },
            "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
        }
    )


def test_run_benchmark_workload_repeats_client_runs():
    calls: list[str] = []

    async def fake_run_cmd(command, stream=True, timeout=600):
        calls.append(command)
        return 0, f"client noise\n{_stanza(100.0 + len(calls), 50.0)}", ""

    success, output, _, _ = asyncio.run(run_benchmark_workload(fake_run_cmd, _recipe(3)))
    assert success
    assert len(calls) == 3
    assert output.count("Serving Benchmark Result") == 3
    assert output.count("client noise") == 3


def test_run_benchmark_workload_fails_on_failed_repeat():
    async def fake_run_cmd(command, stream=True, timeout=600):
        return 1, "boom", "err"

    success, output, stderr, _ = asyncio.run(run_benchmark_workload(fake_run_cmd, _recipe(3)))
    assert not success
    assert output == "boom"
    assert stderr == "err"


def test_failed_later_repeat_preserves_earlier_raw_output():
    calls = 0

    async def fake_run_cmd(command, stream=True, timeout=600):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0, _stanza(100.0, 50.0), ""
        return 1, "client failed after partial output", "connection lost"

    success, output, stderr, _ = asyncio.run(run_benchmark_workload(fake_run_cmd, _recipe(3)))
    assert not success
    assert output.count("Serving Benchmark Result") == 1
    assert "client failed after partial output" in output
    assert stderr == "connection lost"


def test_capture_server_log_preserves_raw_evidence(tmp_path: Path):
    async def fake_run_cmd(command, stream=True, timeout=600):
        assert command == "docker compose logs --no-color"
        assert stream is False
        return 0, "selected backend: native\nlatency: 1.2", ""

    path = tmp_path / "server.log"
    result = asyncio.run(capture_server_log(fake_run_cmd, path))
    assert result == {"path": "server.log", "status": "collected", "exit_code": 0}
    assert path.read_text() == "selected backend: native\nlatency: 1.2\n"


def test_capture_server_log_reports_transport_failure(tmp_path: Path):
    async def fake_run_cmd(command, stream=True, timeout=600):
        return 1, "partial", "connection lost"

    path = tmp_path / "server.log"
    result = asyncio.run(capture_server_log(fake_run_cmd, path))
    assert result["status"] == "failed"
    assert path.read_text() == "partial\nconnection lost\n"
