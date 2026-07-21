"""benchmark.repeats: multi-run workload execution and mean/stddev aggregation."""

import asyncio
from dataclasses import replace

from emmy.benchmark.results import (
    BenchmarkMetrics,
    aggregate_metrics,
    compose_json_result,
    parse_repeat_metrics,
)
from emmy.benchmark.workload import run_benchmark_workload
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


def test_json_result_carries_gpu_summary():
    task = _task(_recipe())
    gpu = {
        "gpus": [{"index": 0, "avg_power_w": 450.0, "peak_memory_mib": 31000.0, "samples": 2}],
        "total_avg_power_w": 450.0,
        "peak_memory_mib": 31000.0,
    }
    data = compose_json_result(task, _stanza(100.0, 50.0), "compose", "cmd", "", gpu=gpu)
    assert data["gpu"]["total_avg_power_w"] == 450.0
    assert "gpu" not in compose_json_result(task, _stanza(100.0, 50.0), "compose", "cmd", "")


def test_run_benchmark_workload_repeats_client_runs():
    bench_calls: list[str] = []

    async def fake_run_cmd(command, stream=True, timeout=600):
        if "nvidia-smi" in command:
            return 0, "12345", ""
        if command.startswith("kill "):
            return 0, "0, 450.0, 30000\n0, 460.0, 31000\n", ""
        bench_calls.append(command)
        return 0, f"client noise\n{_stanza(100.0 + len(bench_calls), 50.0)}", ""

    success, output, _, _, gpu = asyncio.run(run_benchmark_workload(fake_run_cmd, _recipe(3)))
    assert success
    assert len(bench_calls) == 3
    assert len(parse_repeat_metrics(output)) == 3
    assert "client noise" not in output  # per-repeat outputs are trimmed to the stanza
    assert gpu is not None and gpu.total_avg_power_w == 455.0 and gpu.peak_memory_mib == 31000


def test_run_benchmark_workload_fails_on_failed_repeat():
    async def fake_run_cmd(command, stream=True, timeout=600):
        return 1, "boom", "err"

    success, output, stderr, _, gpu = asyncio.run(run_benchmark_workload(fake_run_cmd, _recipe(3)))
    assert not success
    assert output == "boom"
    assert stderr == "err"
    assert gpu is None
