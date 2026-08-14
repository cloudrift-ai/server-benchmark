"""benchmark.repeats: multi-run workload execution and mean/stddev aggregation."""

import asyncio
from dataclasses import replace
from pathlib import Path

from emmy.benchmark.results import (
    BenchmarkMetrics,
    aggregate_metrics,
    compose_json_result,
    parse_repeat_metrics,
)
from emmy.benchmark.workload import capture_server_log, run_benchmark_workload
from emmy.planner import BenchmarkTask
from emmy.planner.variant import Variant
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
            "benchmark": {"max_concurrency": 1, "num_prompts": 32, "random_input_len": 4096, "random_output_len": 4096, "repeats": repeats},
            "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
        }
    )


def _task(recipe: Recipe) -> BenchmarkTask:
    return BenchmarkTask(recipe_dir="r", variant=Variant(params={"deploy.gpu": "NVIDIA GeForce RTX 5090"}), recipe=recipe)


def test_parse_repeat_metrics_splits_stanzas():
    output = "\n\n".join([_stanza(100.0, 50.0), _stanza(110.0, 52.0), _stanza(105.0, 51.0)])
    repeats = parse_repeat_metrics(output)
    assert [r.median_ttft_ms for r in repeats] == [100.0, 110.0, 105.0]
    assert parse_repeat_metrics(_stanza(100.0, 50.0))[0].median_ttft_ms == 100.0
    assert len(parse_repeat_metrics(_stanza(100.0, 50.0))) == 1


def test_aggregate_metrics_mean_stddev():
    repeats = parse_repeat_metrics("\n\n".join([_stanza(100.0, 50.0), _stanza(110.0, 52.0), _stanza(105.0, 51.0)]))
    mean, stddev = aggregate_metrics(repeats)
    assert mean.median_ttft_ms == 105.0
    assert mean.output_token_throughput == 51.0
    assert mean.successful_requests == 32  # identical values keep their type
    assert stddev["median_ttft_ms"] == 5.0
    assert stddev["successful_requests"] == 0.0


def test_aggregate_metrics_skips_fields_missing_in_any_repeat():
    a = BenchmarkMetrics(median_ttft_ms=100.0, output_token_throughput=50.0)
    b = replace(a, median_ttft_ms=None)
    mean, stddev = aggregate_metrics([a, b])
    assert mean.median_ttft_ms is None
    assert "median_ttft_ms" not in stddev
    assert mean.output_token_throughput == 50.0


def test_single_repeat_json_result_unchanged():
    task = _task(_recipe())
    data = compose_json_result(task, _stanza(100.0, 50.0), "compose", "cmd", "")
    assert data["metrics"]["median_ttft_ms"] == 100.0
    assert "metrics_stddev" not in data
    assert "metrics_repeats" not in data


def test_multi_repeat_json_result_aggregates():
    task = _task(_recipe(3))
    output = "\n\n".join([_stanza(100.0, 50.0), _stanza(110.0, 52.0), _stanza(105.0, 51.0)])
    data = compose_json_result(task, output, "compose", "cmd", "")
    assert data["metrics"]["median_ttft_ms"] == 105.0
    assert data["metrics_stddev"]["median_ttft_ms"] == 5.0
    assert len(data["metrics_repeats"]) == 3


def test_run_benchmark_workload_repeats_client_runs():
    calls: list[str] = []

    async def fake_run_cmd(command, stream=True, timeout=600):
        calls.append(command)
        return 0, f"client noise\n{_stanza(100.0 + len(calls), 50.0)}", ""

    success, output, _, _ = asyncio.run(run_benchmark_workload(fake_run_cmd, _recipe(3)))
    assert success
    assert len(calls) == 3
    assert len(parse_repeat_metrics(output)) == 3
    assert output.count("client noise") == 3


def test_run_benchmark_workload_fails_on_failed_repeat():
    async def fake_run_cmd(command, stream=True, timeout=600):
        return 1, "boom", "err"

    success, output, stderr, _ = asyncio.run(run_benchmark_workload(fake_run_cmd, _recipe(3)))
    assert not success
    assert output == "boom"
    assert stderr == "err"


def test_failed_later_repeat_preserves_earlier_observations():
    calls = 0

    async def fake_run_cmd(command, stream=True, timeout=600):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0, _stanza(100.0, 50.0), ""
        return 1, "client failed after partial output", "connection lost"

    success, output, stderr, _ = asyncio.run(run_benchmark_workload(fake_run_cmd, _recipe(3)))
    assert not success
    assert len(parse_repeat_metrics(output)) == 1
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
