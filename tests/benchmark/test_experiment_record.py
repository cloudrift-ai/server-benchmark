"""Tests for the typed, system-only YAML experiment record."""

import re
from pathlib import Path

import yaml

from emmy.benchmark.execution import run_execution_group
from emmy.benchmark.experiment_record import ExperimentRecord, Infrastructure, Provenance
from emmy.planner import BenchmarkTask, ExecutionGroup
from emmy.planner.variant import Variant
from emmy.recipe.types import Recipe
from emmy.redact import register_secret
from emmy.system_info import SystemInformation

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

=== GPU PCI DEVICES ===
0000:01:00.0,0x10de,0x2b85

=== AMD SMI ===
N/A

=== NVCC VERSION ===
13.0.88

=== CUBLAS VERSION ===
13.1.1.3

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


def test_system_information_extracts_common_hardware():
    system = SystemInformation.from_raw(MACHINE_RAW)
    assert system.hostname == "riftvm"
    assert system.os.name == "Ubuntu 24.04.1 LTS"
    assert system.os.id == "ubuntu"
    assert system.os.version == "24.04"
    assert system.os.kernel == "6.8.0-51-generic"
    assert system.cpu.model == "AMD EPYC 7702 64-Core Processor"
    assert system.cpu.logical_count == 64
    assert system.cpu.cores_per_socket == 16
    assert system.memory.total_bytes == 52613349376
    assert len(system.gpus) == 1
    assert system.gpus[0].uuid == "GPU-1234"
    assert system.gpus[0].power_limit_w == 575.0
    assert system.gpus[0].driver_version == "580.65.06"
    assert system.software.nvcc_version == "13.0.88"
    assert system.software.cublas_version == "13.1.1.3"
    assert system.root_filesystem.available_bytes == 600000000
    assert system.uptime_seconds == 1234.5


def test_system_information_uses_pci_registry_for_amd():
    raw = MACHINE_RAW.replace(
        "0, NVIDIA GeForce RTX 5090, GPU-1234, 00000000:01:00.0, 32607, 580.65.06, P0, 42, 2, 2400, 1750, 120.0, 575.0",
        "N/A",
    ).replace("0000:01:00.0,0x10de,0x2b85", "0000:41:00.0,0x1002,0x75b0")
    system = SystemInformation.from_raw(raw)
    assert len(system.gpus) == 1
    assert system.gpus[0].name == "AMD Instinct MI350X"
    assert system.gpus[0].vendor == "AMD"
    assert system.gpus[0].pci_bus_id == "0000:41:00.0"


async def test_system_information_retrieves_and_parses_host():
    calls = []

    async def run_cmd(command, *, stream, timeout):
        calls.append((command, stream, timeout))
        return 0, MACHINE_RAW, ""

    system = await SystemInformation.retrieve(run_cmd)

    assert system.hostname == "riftvm"
    assert calls[0][1:] == (False, 120)
    assert "nvidia-smi --query-gpu" in calls[0][0]


def test_record_lifecycle_serialization_and_round_trip(tmp_path):
    task = _task(tmp_path)
    task.record = ExperimentRecord.create(task, "20260814T120000Z")
    assert task.record.schema_version == 2
    assert task.record.status == "queued"
    assert task.record.experiment.parameters["benchmark.max_concurrency"] == 8
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", task.record.timestamp)

    task.record.start("benchmark")
    task.record.system = SystemInformation.from_raw(MACHINE_RAW)
    task.record.execution.infrastructure = Infrastructure(
        group="rtx5090_x_1",
        requested_gpu="NVIDIA GeForce RTX 5090",
        requested_gpu_count=1,
        address="host.example",
        ssh_port=22,
    )
    register_secret("secret-value")
    task.record.finish(success=True, stage="benchmark", timing={"benchmark": 1.5, "total": 1.5})
    task.record.write(task.record_path())

    loaded_yaml = yaml.safe_load(task.record_path().read_text())
    loaded = ExperimentRecord.read(task.record_path())
    assert loaded.status == "succeeded"
    assert loaded.execution.completed_at.endswith("Z")
    assert loaded.system.gpus[0].uuid == "GPU-1234"
    assert set(loaded_yaml["provenance"]) == {"git_revision", "git_dirty"}
    assert "artifacts" not in loaded_yaml
    assert "measurement" not in loaded_yaml
    assert "metrics" not in loaded_yaml
    assert "rendered_command" not in loaded_yaml
    assert "secret-value" not in task.record_path().read_text()
    assert not task.record_path().with_suffix(".yaml.tmp").exists()


def test_record_rejects_other_schema_versions(tmp_path):
    path = tmp_path / "old.experiment.yaml"
    path.write_text("schema_version: 1\n", encoding="utf-8")

    try:
        ExperimentRecord.read(path)
    except ValueError as exc:
        assert str(exc) == "unsupported experiment record schema version: 1"
    else:
        raise AssertionError("obsolete experiment record schema was accepted")


def test_record_rejects_removed_artifact_field(tmp_path):
    task = _task(tmp_path)
    record = ExperimentRecord.create(task, "run")
    value = record.to_mapping()
    value["artifacts"] = []
    path = tmp_path / "old.experiment.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")

    try:
        ExperimentRecord.read(path)
    except ValueError as exc:
        assert str(exc) == "experiment record fields do not match schema version 2"
    else:
        raise AssertionError("removed experiment record field was accepted")


def test_strict_command_provenance_uses_typed_system_information(tmp_path):
    record = ExperimentRecord.create(_task(tmp_path), "run")
    record.provenance = Provenance(git_revision="revision", git_dirty=False)
    record.system = SystemInformation.from_raw(MACHINE_RAW)
    assert record.missing_command_provenance() == []

    record.system = None
    record.provenance = Provenance(git_revision=None, git_dirty=None)
    assert record.missing_command_provenance() == [
        "Git provenance",
        "GPU provenance",
        "NVCC provenance",
        "cuBLAS provenance",
    ]


def test_create_run_dir_uses_run_timestamp_without_replacing_prior_runs(tmp_path):
    recipe = tmp_path / "experiment"
    prior = recipe / "2026-08-14_11-59-59"
    prior.mkdir(parents=True)
    marker = prior / "raw.log"
    marker.write_text("old")
    recipe_file = recipe / "recipe.yaml"
    recipe_file.write_text("matrices: {}\n")

    created = BenchmarkTask.create_run_dir(str(recipe), "20260814T120000Z", create=True)

    assert created == (recipe / "2026-08-14_12-00-00").resolve()
    assert list(created.iterdir()) == []
    assert marker.read_text() == "old"
    assert recipe_file.exists()


def test_create_run_dir_dry_run_does_not_create_directory(tmp_path):
    recipe = tmp_path / "experiment"
    recipe.mkdir()

    run_dir = BenchmarkTask.create_run_dir(str(recipe), "20260814T120000Z", create=False)

    assert run_dir == (recipe / "2026-08-14_12-00-00").resolve()
    assert not run_dir.exists()


def test_create_run_dir_refuses_symlink(tmp_path):
    recipe = tmp_path / "experiment"
    target = tmp_path / "elsewhere"
    recipe.mkdir()
    target.mkdir()
    (recipe / "2026-08-14_12-00-00").symlink_to(target)

    try:
        BenchmarkTask.create_run_dir(str(recipe), "20260814T120000Z", create=True)
    except ValueError as exc:
        assert "symlinked" in str(exc)
    else:
        raise AssertionError("symlinked run directory was accepted")


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
    record = ExperimentRecord.read(task.record_path())
    assert record.status == "failed"
    assert record.execution.stage == "provisioning"
    assert record.execution.error.message == "VM provisioning failed"
