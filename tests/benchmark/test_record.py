"""Tests for the common YAML experiment record."""

import os
import re
from pathlib import Path

import yaml

from emmy.benchmark.execution import run_execution_group
from emmy.benchmark.record import (
    BenchmarkMetrics,
    command_measurement,
    create_record,
    finish_record,
    inference_measurement,
    missing_command_provenance,
    parse_benchmark_metrics,
    parse_machine_info,
    start_record,
    write_record,
)
from emmy.planner import BenchmarkTask, ExecutionGroup
from emmy.planner.variant import Variant
from emmy.recipe.types import Recipe
from emmy.redact import register_secret

BENCHMARK_OUTPUT = """\
============ Serving Benchmark Result ============
Successful requests:                     80
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  365.50
Total input tokens:                      320000
Total generated tokens:                  320000
Request throughput (req/s):              0.22
Output token throughput (tok/s):         875.52
Peak output token throughput (tok/s):    1096.00
Peak concurrent requests:                16.00
Total token throughput (tok/s):          1751.04
Mean TTFT (ms):                          771.60
Median TTFT (ms):                        845.10
P99 TTFT (ms):                           1492.58
Mean TPOT (ms):                          8.95
Median TPOT (ms):                        8.88
P99 TPOT (ms):                           9.67
Mean ITL (ms):                           8.95
Median ITL (ms):                         8.75
P99 ITL (ms):                            10.60
Mean E2EL (ms):                          4570.00
Median E2EL (ms):                        4600.12
P99 E2EL (ms):                           5200.50
==================================================
"""

MACHINE_RAW = """\
=== HOSTNAME ===
riftvm

=== OS ===
PRETTY_NAME="Ubuntu 24.04.1 LTS"
ID=ubuntu
VERSION_ID="24.04"

=== KERNEL ===
6.8.0-51-generic

=== CPU ===
Architecture:                         x86_64
CPU(s):                               64
Model name:                           AMD EPYC 7702 64-Core Processor
Thread(s) per core:                   2
Core(s) per socket:                   16
Socket(s):                            2
NUMA node(s):                         2

=== MEMORY ===
MemTotal: 52613349376

=== NVIDIA GPUS ===
0, NVIDIA GeForce RTX 5090, GPU-1234, 00000000:01:00.0, 32607, 580.65.06, P0, 42, 2, 2400, 1750, 120.0, 575.0

=== NVIDIA DRIVER ===
| NVIDIA-SMI 580.65.06   Driver Version: 580.65.06   CUDA Version: 13.0 |

=== GPU PCI DEVICES ===
0000:01:00.0,0x10de,0x2b85

=== AMD SMI ===
N/A

=== CUDA COMPILER ===
Cuda compilation tools, release 13.0, V13.0.88

=== HIP COMPILER ===
N/A

=== ROOT FILESYSTEM ===
/dev/sda1 ext4 1000000000 400000000 600000000 40% /

=== UPTIME SECONDS ===
1234.50

=== DOCKER ===
ClientVersion: 28.5.1
ServerVersion: 28.5.1
OperatingSystem: Ubuntu 24.04.1 LTS
"""


def _task(tmp_path: Path) -> BenchmarkTask:
    recipe = Recipe.from_dict(
        {
            "model": {"huggingface": "test-org/test-model"},
            "engine": {"llm": {"context_length": 8192, "vllm": {"image": "vllm/vllm-openai:v0.17.0"}}},
            "benchmark": {"max_concurrency": 8, "num_prompts": 80},
            "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
        }
    )
    variant = Variant(
        params={
            "deploy.gpu": "NVIDIA GeForce RTX 5090",
            "deploy.gpu_count": 1,
            "benchmark.max_concurrency": 8,
        }
    )
    return BenchmarkTask(
        recipe_dir=str(tmp_path),
        variant=variant,
        recipe=recipe,
        run_dir=tmp_path / "results",
    )


def test_parse_benchmark_metrics():
    metrics = parse_benchmark_metrics(BENCHMARK_OUTPUT)
    assert isinstance(metrics, BenchmarkMetrics)
    assert metrics.successful_requests == 80
    assert metrics.failed_requests == 0
    assert metrics.max_request_concurrency == 8
    assert metrics.output_token_throughput == 875.52
    assert metrics.median_ttft_ms == 845.1
    assert metrics.median_tpot_ms == 8.88
    assert metrics.median_e2el_ms == 4600.12


def test_parse_benchmark_metrics_empty():
    metrics = parse_benchmark_metrics("not a result")
    assert all(getattr(metrics, name) is None for name in BenchmarkMetrics.__dataclass_fields__)


def test_parse_machine_info_extracts_common_hardware():
    machine = parse_machine_info(MACHINE_RAW)
    assert machine["hostname"] == "riftvm"
    assert machine["os"] == {
        "name": "Ubuntu 24.04.1 LTS",
        "id": "ubuntu",
        "version": "24.04",
        "kernel": "6.8.0-51-generic",
    }
    assert machine["cpu"]["model"] == "AMD EPYC 7702 64-Core Processor"
    assert machine["cpu"]["logical_count"] == 64
    assert machine["cpu"]["cores_per_socket"] == 16
    assert machine["memory"]["total_bytes"] == 52613349376
    assert len(machine["gpus"]) == 1
    assert machine["gpus"][0]["uuid"] == "GPU-1234"
    assert machine["gpus"][0]["power_limit_w"] == 575.0
    assert machine["software"]["cuda_driver_api"] == "13.0"
    assert machine["software"]["cuda_compiler"].endswith("V13.0.88")
    assert machine["software"]["hip_compiler"] is None
    assert machine["root_filesystem"]["available_bytes"] == 600000000
    assert machine["uptime_seconds"] == 1234.5


def test_parse_machine_info_uses_pci_registry_for_amd():
    raw = MACHINE_RAW.replace(
        "0, NVIDIA GeForce RTX 5090, GPU-1234, 00000000:01:00.0, 32607, 580.65.06, P0, 42, 2, 2400, 1750, 120.0, 575.0",
        "N/A",
    ).replace("0000:01:00.0,0x10de,0x2b85", "0000:41:00.0,0x1002,0x75b0")
    machine = parse_machine_info(raw)
    assert machine["gpus"] == [
        {
            "index": 0,
            "name": "AMD Instinct MI350X",
            "vendor": "AMD",
            "uuid": None,
            "pci_bus_id": "0000:41:00.0",
            "memory_total_mib": None,
            "driver_version": None,
            "performance_state": None,
            "temperature_c": None,
            "utilization_percent": None,
            "sm_clock_mhz": None,
            "memory_clock_mhz": None,
            "power_draw_w": None,
            "power_limit_w": None,
        }
    ]


def test_record_lifecycle_and_yaml_output(tmp_path):
    task = _task(tmp_path)
    task.record = create_record(task, "20260814T120000Z-deadbeef", "deadbeef" * 8)
    assert task.record["schema_version"] == 1
    assert task.record["status"] == "queued"
    assert task.record["experiment"]["variant"]["parameters"]["benchmark.max_concurrency"] == 8
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", task.record["timestamp"])

    start_record(task.record, "benchmark")
    task.record["machine"] = parse_machine_info(MACHINE_RAW)
    register_secret("secret-value")
    task.record["measurement"] = inference_measurement(BENCHMARK_OUTPUT, "token: secret-value", "vllm bench serve")
    finish_record(task.record, success=True, stage="benchmark", timing={"benchmark": 1.5, "total": 1.5})
    write_record(task)

    loaded = yaml.safe_load(task.record_path().read_text())
    assert loaded["status"] == "succeeded"
    assert loaded["execution"]["completed_at"].endswith("Z")
    assert loaded["measurement"]["metrics"]["successful_requests"] == 80
    assert loaded["measurement"]["compose"] == "token: ***"
    assert not task.record_path().with_suffix(".yaml.tmp").exists()


def test_multi_repeat_measurement_keeps_mean_spread_and_raw_rows():
    output = "\n\n".join(
        [
            BENCHMARK_OUTPUT.replace("875.52", "800.00"),
            BENCHMARK_OUTPUT.replace("875.52", "900.00"),
            BENCHMARK_OUTPUT.replace("875.52", "1000.00"),
        ]
    )
    measurement = inference_measurement(output, "services: {}", "vllm bench serve")
    assert measurement["metrics"]["output_token_throughput"] == 900.0
    assert measurement["metrics_stddev"]["output_token_throughput"] == 100.0
    assert len(measurement["repetitions"]) == 3


def test_command_measurement_and_provenance_contract():
    measurement = command_measurement({"rendered_command": "echo test", "exit_code": 0, "result_paths": ["result.csv"]})
    assert measurement["kind"] == "command"
    assert measurement["exit_code"] == 0
    source = {"source_id": "abc", "files": {"emmy/a.py": "hash"}}
    assert missing_command_provenance(parse_machine_info(MACHINE_RAW), source) == []
    assert missing_command_provenance(None, None) == [
        "staged source manifest",
        "GPU provenance",
        "GPU compiler provenance",
    ]


def test_prepare_results_dir_replaces_only_results(tmp_path):
    recipe = tmp_path / "experiment"
    results = recipe / "results"
    results.mkdir(parents=True)
    (results / "stale.log").write_text("old")
    recipe_file = recipe / "recipe.yaml"
    recipe_file.write_text("matrices: {}\n")

    prepared = BenchmarkTask.prepare_results_dir(str(recipe), overwrite=True)

    assert prepared == results.resolve()
    assert list(prepared.iterdir()) == []
    assert recipe_file.exists()


def test_prepare_results_dir_refuses_symlink(tmp_path):
    recipe = tmp_path / "experiment"
    target = tmp_path / "elsewhere"
    recipe.mkdir()
    target.mkdir()
    os.symlink(target, recipe / "results")

    try:
        BenchmarkTask.prepare_results_dir(str(recipe), overwrite=True)
    except ValueError as exc:
        assert "symlinked" in str(exc)
    else:
        raise AssertionError("symlinked results directory was accepted")


async def test_execution_failure_leaves_terminal_yaml_record(tmp_path, monkeypatch):
    task = _task(tmp_path)
    task.run_dir.mkdir()
    group = ExecutionGroup(gpu_name=task.gpu_name, gpu_count=task.gpu_count, tasks=[task])

    async def fail_to_provision(*_args, **_kwargs):
        return None

    monkeypatch.setattr("emmy.benchmark.execution.provision_cloud_vm", fail_to_provision)

    results = await run_execution_group(group, {"benchmark": {}}, "/missing/ssh-key")

    assert len(results) == 1
    _result_task, ok, timing = results[0]
    assert ok is False
    assert timing["total"] >= 0
    record = yaml.safe_load(task.record_path().read_text())
    assert record["status"] == "failed"
    assert record["execution"]["stage"] == "provisioning"
    assert record["execution"]["error"]["message"] == "VM provisioning failed"
