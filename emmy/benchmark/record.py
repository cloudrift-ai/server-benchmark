"""The sole structured record produced for an experiment row."""

from __future__ import annotations

import os
import re
import statistics
import subprocess
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from emmy import gpu
from emmy.redact import redact_secrets

if TYPE_CHECKING:
    from emmy.planner import BenchmarkTask

SCHEMA_VERSION = 1
RESULT_MARKER = "============ Serving Benchmark Result ============"


@dataclass
class BenchmarkMetrics:
    """Parsed metrics from an inference workload."""

    successful_requests: int | None = None
    failed_requests: int | None = None
    max_request_concurrency: int | None = None
    benchmark_duration_s: float | None = None
    total_input_tokens: int | None = None
    total_generated_tokens: int | None = None
    request_throughput: float | None = None
    output_token_throughput: float | None = None
    peak_output_token_throughput: float | None = None
    peak_concurrent_requests: float | None = None
    total_token_throughput: float | None = None
    mean_ttft_ms: float | None = None
    median_ttft_ms: float | None = None
    p99_ttft_ms: float | None = None
    mean_tpot_ms: float | None = None
    median_tpot_ms: float | None = None
    p99_tpot_ms: float | None = None
    mean_itl_ms: float | None = None
    median_itl_ms: float | None = None
    p99_itl_ms: float | None = None
    mean_e2el_ms: float | None = None
    median_e2el_ms: float | None = None
    p99_e2el_ms: float | None = None


_METRIC_FIELDS = [
    ("Successful requests", "successful_requests", int),
    ("Failed requests", "failed_requests", int),
    ("Maximum request concurrency", "max_request_concurrency", int),
    ("Benchmark duration (s)", "benchmark_duration_s", float),
    ("Total input tokens", "total_input_tokens", int),
    ("Total generated tokens", "total_generated_tokens", int),
    ("Request throughput (req/s)", "request_throughput", float),
    ("Output token throughput (tok/s)", "output_token_throughput", float),
    ("Peak output token throughput (tok/s)", "peak_output_token_throughput", float),
    ("Peak concurrent requests", "peak_concurrent_requests", float),
    ("Total token throughput (tok/s)", "total_token_throughput", float),
    ("Mean TTFT (ms)", "mean_ttft_ms", float),
    ("Median TTFT (ms)", "median_ttft_ms", float),
    ("P99 TTFT (ms)", "p99_ttft_ms", float),
    ("Mean TPOT (ms)", "mean_tpot_ms", float),
    ("Median TPOT (ms)", "median_tpot_ms", float),
    ("P99 TPOT (ms)", "p99_tpot_ms", float),
    ("Mean ITL (ms)", "mean_itl_ms", float),
    ("Median ITL (ms)", "median_itl_ms", float),
    ("P99 ITL (ms)", "p99_itl_ms", float),
    ("Mean E2EL (ms)", "mean_e2el_ms", float),
    ("Median E2EL (ms)", "median_e2el_ms", float),
    ("P99 E2EL (ms)", "p99_e2el_ms", float),
]


def parse_benchmark_metrics(output: str) -> BenchmarkMetrics:
    """Parse one inference result stanza."""
    parsed = {}
    for label, field_name, typ in _METRIC_FIELDS:
        match = re.search(rf"{re.escape(label)}:\s+([\d.]+)", output)
        if match:
            try:
                parsed[field_name] = typ(match.group(1))
            except (TypeError, ValueError):
                pass
    return BenchmarkMetrics(**parsed)


def parse_repeat_metrics(output: str) -> list[BenchmarkMetrics]:
    """Parse one metrics object per repeated inference measurement."""
    starts = [match.start() for match in re.finditer(re.escape(RESULT_MARKER), output)]
    if len(starts) <= 1:
        return [parse_benchmark_metrics(output)]
    chunks = [output[start:end] for start, end in zip(starts, starts[1:] + [len(output)], strict=True)]
    return [parse_benchmark_metrics(chunk) for chunk in chunks]


def aggregate_metrics(repeats: list[BenchmarkMetrics]) -> tuple[BenchmarkMetrics, dict[str, float]]:
    """Return per-field means and sample standard deviations across repeats."""
    mean_fields: dict[str, int | float] = {}
    stddev: dict[str, float] = {}
    for metric_field in fields(BenchmarkMetrics):
        values = [getattr(repeat, metric_field.name) for repeat in repeats]
        if any(value is None for value in values):
            continue
        present = [value for value in values if value is not None]
        mean_fields[metric_field.name] = present[0] if len(set(present)) == 1 else round(statistics.fmean(present), 4)
        if len(present) > 1:
            stddev[metric_field.name] = round(statistics.stdev(present), 4)
    return BenchmarkMetrics(**mean_fields), stddev


def _get_section(raw_text: str, section_name: str) -> str:
    pattern = rf"=== {re.escape(section_name)} ===\n(.*?)(?=\n=== |\Z)"
    match = re.search(pattern, raw_text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _key_values(text: str, separator: str = ":") -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        if separator not in line:
            continue
        key, value = line.split(separator, 1)
        result[key.strip()] = value.strip().strip('"')
    return result


def _number(value: str | None, typ: type[int] | type[float]) -> int | float | None:
    if not value or value.strip().upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    match = re.search(r"-?[\d.]+", value)
    if not match:
        return None
    try:
        return typ(match.group(0))
    except ValueError:
        return None


def _parse_nvidia_gpus(text: str) -> list[dict[str, Any]]:
    columns = (
        "index",
        "name",
        "uuid",
        "pci_bus_id",
        "memory_total_mib",
        "driver_version",
        "performance_state",
        "temperature_c",
        "utilization_percent",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "power_draw_w",
        "power_limit_w",
    )
    numeric = {
        "index": int,
        "memory_total_mib": int,
        "temperature_c": float,
        "utilization_percent": float,
        "sm_clock_mhz": float,
        "memory_clock_mhz": float,
        "power_draw_w": float,
        "power_limit_w": float,
    }
    gpus = []
    for line in text.splitlines():
        if not line.strip() or line.strip() == "N/A":
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(columns):
            continue
        item: dict[str, Any] = {"vendor": "NVIDIA"}
        for name, value in zip(columns, values, strict=True):
            item[name] = _number(value, numeric[name]) if name in numeric else (None if value.upper() == "N/A" else value)
        gpus.append(item)
    return gpus


def _parse_pci_gpus(text: str) -> list[dict[str, Any]]:
    devices = []
    for line in text.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 3:
            continue
        address, vendor_id, device_id = values
        spec = gpu.by_pci_device_id(device_id)
        devices.append(
            {
                "pci_bus_id": address,
                "vendor_id": vendor_id.removeprefix("0x"),
                "device_id": device_id.removeprefix("0x"),
                "vendor": spec.vendor if spec else None,
                "name": spec.name if spec else None,
            }
        )
    return devices


def _tool_output(raw_text: str, section: str) -> str | None:
    value = _get_section(raw_text, section)
    return None if not value or value == "N/A" else value


def parse_machine_info(raw_text: str) -> dict[str, Any] | None:
    """Parse the portable host snapshot collected by :mod:`system_info`."""
    if not raw_text:
        return None

    os_release = _key_values(_get_section(raw_text, "OS"), "=")
    lscpu = _key_values(_get_section(raw_text, "CPU"))
    memory = _key_values(_get_section(raw_text, "MEMORY"))
    docker = _key_values(_get_section(raw_text, "DOCKER"))
    filesystem = _get_section(raw_text, "ROOT FILESYSTEM").split()
    gpus = _parse_nvidia_gpus(_get_section(raw_text, "NVIDIA GPUS"))
    pci_gpus = _parse_pci_gpus(_get_section(raw_text, "GPU PCI DEVICES"))
    known_pci_addresses = {item.get("pci_bus_id", "").lower()[-12:] for item in gpus}
    for item in pci_gpus:
        if item["pci_bus_id"].lower()[-12:] not in known_pci_addresses:
            gpus.append(
                {
                    "index": len(gpus),
                    "name": item["name"],
                    "vendor": item["vendor"],
                    "uuid": None,
                    "pci_bus_id": item["pci_bus_id"],
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
            )

    root_filesystem = None
    if len(filesystem) >= 7:
        root_filesystem = {
            "device": filesystem[0],
            "type": filesystem[1],
            "total_bytes": _number(filesystem[2], int),
            "used_bytes": _number(filesystem[3], int),
            "available_bytes": _number(filesystem[4], int),
            "mount": filesystem[-1],
        }

    cuda_driver = _get_section(raw_text, "NVIDIA DRIVER")
    cuda_match = re.search(r"CUDA Version:\s+([\d.]+)", cuda_driver)
    uptime = _number(_get_section(raw_text, "UPTIME SECONDS"), float)
    machine = {
        "hostname": _get_section(raw_text, "HOSTNAME") or None,
        "os": {
            "name": os_release.get("PRETTY_NAME"),
            "id": os_release.get("ID"),
            "version": os_release.get("VERSION_ID"),
            "kernel": _get_section(raw_text, "KERNEL") or None,
        },
        "cpu": {
            "model": lscpu.get("Model name"),
            "architecture": lscpu.get("Architecture"),
            "logical_count": _number(lscpu.get("CPU(s)"), int),
            "sockets": _number(lscpu.get("Socket(s)"), int),
            "cores_per_socket": _number(lscpu.get("Core(s) per socket"), int),
            "threads_per_core": _number(lscpu.get("Thread(s) per core"), int),
            "numa_nodes": _number(lscpu.get("NUMA node(s)"), int),
        },
        "memory": {"total_bytes": _number(memory.get("MemTotal"), int)},
        "gpus": gpus,
        "gpu_pci_devices": pci_gpus,
        "software": {
            "cuda_driver_api": cuda_match.group(1) if cuda_match else None,
            "cuda_compiler": _tool_output(raw_text, "CUDA COMPILER"),
            "hip_compiler": _tool_output(raw_text, "HIP COMPILER"),
            "docker_client": docker.get("ClientVersion"),
            "docker_server": docker.get("ServerVersion"),
            "docker_os": docker.get("OperatingSystem"),
        },
        "root_filesystem": root_filesystem,
        "uptime_seconds": uptime,
    }
    amd_smi = _get_section(raw_text, "AMD SMI")
    if amd_smi and amd_smi != "N/A":
        machine["amd_smi"] = amd_smi
    return machine


def missing_command_provenance(machine: dict[str, Any] | None, source: dict | None) -> list[str]:
    """Return missing fields required by the command strictness contract."""
    missing = []
    if not source or not source.get("source_id") or not source.get("files"):
        missing.append("staged source manifest")
    gpus = (machine or {}).get("gpus") or []
    if not gpus or not all(item.get("uuid") or item.get("pci_bus_id") for item in gpus):
        missing.append("GPU provenance")
    software = (machine or {}).get("software") or {}
    if not software.get("cuda_compiler") and not software.get("hip_compiler"):
        missing.append("GPU compiler provenance")
    return missing


def utc_timestamp() -> str:
    """Return an RFC 3339 UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_run_id(code_hash: str) -> str:
    """Return a readable invocation identifier shared by all selected rows."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{code_hash[:8]}"


def _git_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=no")
    return {"git_revision": revision, "git_dirty": bool(status) if status is not None else None}


def _builtins(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _builtins(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtins(item) for item in value]
    return value


def _repo_relative(path: str) -> str:
    resolved = Path(path).resolve()
    repository = Path(__file__).resolve().parents[2]
    try:
        return str(resolved.relative_to(repository))
    except ValueError:
        return str(resolved)


def create_record(task: BenchmarkTask, run_id: str, code_hash: str) -> dict[str, Any]:
    """Create the initial record for one expanded matrix row."""
    timestamp = utc_timestamp()
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp,
        "status": "queued",
        "experiment": {
            "task_id": task.task_id,
            "row_id": task.row_id,
            "directory": _repo_relative(task.recipe_dir),
            "kind": task.recipe.kind,
            "variant": {"name": str(task.variant), "parameters": _builtins(task.variant.params)},
            "recipe": _builtins(asdict(task.recipe)),
        },
        "provenance": {"emmy_code_sha256": code_hash, **_git_provenance(), "source": None},
        "machine": None,
        "execution": {
            "run_id": run_id,
            "stage": "queued",
            "started_at": None,
            "completed_at": None,
            "timing_seconds": {},
            "infrastructure": None,
            "error": None,
        },
        "measurement": None,
        "artifacts": [],
    }


def start_record(record: dict[str, Any], stage: str) -> None:
    """Mark a queued row as running."""
    record["status"] = "running"
    record["execution"]["stage"] = stage
    record["execution"]["started_at"] = record["execution"]["started_at"] or utc_timestamp()


def finish_record(
    record: dict[str, Any],
    *,
    success: bool,
    stage: str,
    timing: dict[str, float],
    error: str | None = None,
) -> None:
    """Finalize one row after success or failure."""
    start_record(record, stage)
    record["status"] = "succeeded" if success else "failed"
    record["execution"]["stage"] = "complete" if success else stage
    record["execution"]["completed_at"] = utc_timestamp()
    record["execution"]["timing_seconds"] = timing
    record["execution"]["error"] = None if success else {"stage": stage, "message": error or f"{stage} failed"}


def inference_measurement(benchmark_output: str, compose_content: str, bench_command: str) -> dict[str, Any]:
    """Build the inference-specific part of a row record."""
    repeats = parse_repeat_metrics(benchmark_output)
    metrics, stddev = aggregate_metrics(repeats)
    result = {
        "kind": "inference",
        "command": bench_command,
        "compose": redact_secrets(compose_content),
        "metrics": asdict(metrics),
    }
    if len(repeats) > 1:
        result["metrics_stddev"] = stddev
        result["repetitions"] = [asdict(repeat) for repeat in repeats]
    return result


def command_measurement(command_info: dict[str, Any]) -> dict[str, Any]:
    """Build the command-specific part of a row record."""
    return {"kind": "command", **_builtins(command_info)}


def artifact(task: BenchmarkTask, path: Path, kind: str, **metadata: Any) -> dict[str, Any]:
    """Describe one raw file relative to the experiment directory."""
    experiment_dir = Path(task.recipe_dir).resolve()
    try:
        relative = path.resolve().relative_to(experiment_dir)
        rendered = str(relative)
    except ValueError:
        rendered = str(path)
    return {"kind": kind, "path": rendered, **metadata}


def write_record_path(path: Path, record: dict[str, Any]) -> None:
    """Atomically persist one YAML experiment record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(record, sort_keys=False, allow_unicode=True, width=120)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(redact_secrets(payload), encoding="utf-8")
    os.replace(temporary, path)


def write_record(task: BenchmarkTask) -> None:
    """Atomically persist the task's current YAML record."""
    if task.record is None:
        raise ValueError("experiment record has not been initialized")
    write_record_path(task.record_path(), task.record)
