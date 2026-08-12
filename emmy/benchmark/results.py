"""Structured JSON benchmark results: dataclasses and parsers."""

import re
import statistics
from dataclasses import asdict, dataclass, fields

from emmy.redact import redact_secrets

RESULT_MARKER = "============ Serving Benchmark Result ============"


@dataclass
class BenchmarkMetrics:
    """Parsed metrics from vLLM bench serve output."""

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


@dataclass
class SystemInfo:
    """Parsed system information from remote server."""

    hostname: str | None = None
    os: str | None = None
    kernel: str | None = None
    cpu_model: str | None = None
    cpu_count: int | None = None
    cpu_arch: str | None = None
    memory_total_gib: float | None = None
    gpu_name: str | None = None
    gpu_memory_mib: int | None = None
    gpu_driver: str | None = None
    cuda_version: str | None = None
    gpu_count: int | None = None
    docker_version: str | None = None
    gpu_provenance: list[str] | None = None
    cuda_compiler: str | None = None


# (label in bench output, dataclass field name, type constructor)
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
    """Parse vLLM bench serve output into BenchmarkMetrics."""
    parsed = {}
    for label, field_name, typ in _METRIC_FIELDS:
        m = re.search(rf"{re.escape(label)}:\s+([\d.]+)", output)
        if m:
            try:
                parsed[field_name] = typ(m.group(1))
            except (ValueError, TypeError):
                pass
    return BenchmarkMetrics(**parsed)


def parse_repeat_metrics(output: str) -> list[BenchmarkMetrics]:
    """Parse one BenchmarkMetrics per result stanza (``benchmark.repeats`` runs emit
    several). Output with no stanza marker parses as a single repeat."""
    starts = [m.start() for m in re.finditer(re.escape(RESULT_MARKER), output)]
    if len(starts) <= 1:
        return [parse_benchmark_metrics(output)]
    chunks = [output[s:e] for s, e in zip(starts, starts[1:] + [len(output)], strict=True)]
    return [parse_benchmark_metrics(c) for c in chunks]


def aggregate_metrics(repeats: list[BenchmarkMetrics]) -> tuple[BenchmarkMetrics, dict[str, float]]:
    """Per-field mean across repeats, plus sample stddev for fields present in every
    repeat. Fields missing in any repeat stay None (mean) / absent (stddev); values
    identical across repeats keep their original type (int fields stay int)."""
    mean_fields: dict = {}
    stddev: dict[str, float] = {}
    for f in fields(BenchmarkMetrics):
        values = [getattr(r, f.name) for r in repeats]
        if any(v is None for v in values):
            continue
        mean_fields[f.name] = values[0] if len(set(values)) == 1 else round(statistics.fmean(values), 4)
        if len(values) > 1:
            stddev[f.name] = round(statistics.stdev(values), 4)
    return BenchmarkMetrics(**mean_fields), stddev


def _get_section(raw_text: str, section_name: str) -> str:
    """Extract content between === SECTION === markers."""
    pattern = rf"=== {re.escape(section_name)} ===\n(.*?)(?=\n=== |\Z)"
    m = re.search(pattern, raw_text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_memory_total(mem_section: str) -> float | None:
    """Parse total memory from `free -h` output to GiB."""
    # Match the Mem: line, e.g. "Mem:  49Gi  ..."
    m = re.search(r"Mem:\s+([\d.]+)\s*([A-Za-z]+)", mem_section)
    if not m:
        return None
    value = float(m.group(1))
    suffix = m.group(2).lower()
    multipliers = {"gi": 1.0, "g": 1.0 / 1.073741824, "mi": 1.0 / 1024, "ti": 1024.0}
    return round(value * multipliers.get(suffix, 1.0), 2)


def parse_system_info(raw_text: str) -> SystemInfo:
    """Parse system info collected via collect_system_info() into SystemInfo."""
    if not raw_text:
        return SystemInfo()

    fields: dict = {}

    # HOSTNAME
    hostname = _get_section(raw_text, "HOSTNAME")
    if hostname:
        fields["hostname"] = hostname

    # OS
    os_section = _get_section(raw_text, "OS")
    m = re.search(r'PRETTY_NAME="(.+?)"', os_section)
    if m:
        fields["os"] = m.group(1)

    # KERNEL
    kernel = _get_section(raw_text, "KERNEL")
    if kernel:
        fields["kernel"] = kernel

    # CPU INFORMATION
    cpu_section = _get_section(raw_text, "CPU INFORMATION")
    m = re.search(r"Model name:\s+(.+)", cpu_section)
    if m:
        fields["cpu_model"] = m.group(1).strip()
    m = re.search(r"Architecture:\s+(\w+)", cpu_section)
    if m:
        fields["cpu_arch"] = m.group(1)

    # CPU COUNT
    cpu_count = _get_section(raw_text, "CPU COUNT")
    if cpu_count:
        try:
            fields["cpu_count"] = int(cpu_count)
        except ValueError:
            pass

    # MEMORY
    mem_section = _get_section(raw_text, "MEMORY")
    mem_total = _parse_memory_total(mem_section)
    if mem_total is not None:
        fields["memory_total_gib"] = mem_total

    # GPU INFORMATION — CSV: name, memory_mib, driver, pstate, temp, util
    gpu_section = _get_section(raw_text, "GPU INFORMATION")
    if gpu_section and gpu_section != "N/A":
        gpu_lines = [line.strip() for line in gpu_section.strip().splitlines() if line.strip()]
        fields["gpu_count"] = len(gpu_lines)
        if gpu_lines:
            parts = [p.strip() for p in gpu_lines[0].split(",")]
            if len(parts) >= 1:
                fields["gpu_name"] = parts[0]
            if len(parts) >= 2:
                try:
                    fields["gpu_memory_mib"] = int(parts[1].split()[0])
                except (ValueError, IndexError):
                    pass
            if len(parts) >= 3:
                fields["gpu_driver"] = parts[2]

    # GPU DETAILS — CUDA version
    gpu_details = _get_section(raw_text, "GPU DETAILS")
    m = re.search(r"CUDA Version:\s+([\d.]+)", gpu_details)
    if m:
        fields["cuda_version"] = m.group(1)

    gpu_provenance = _get_section(raw_text, "GPU PROVENANCE")
    if gpu_provenance and gpu_provenance != "N/A":
        fields["gpu_provenance"] = [line.strip() for line in gpu_provenance.splitlines() if line.strip()]

    cuda_compiler = _get_section(raw_text, "CUDA COMPILER")
    if cuda_compiler and cuda_compiler != "N/A":
        fields["cuda_compiler"] = cuda_compiler

    # DOCKER VERSION
    docker_section = _get_section(raw_text, "DOCKER VERSION")
    m = re.search(r"Docker version ([\d.]+)", docker_section)
    if m:
        fields["docker_version"] = m.group(1)

    return SystemInfo(**fields)


def _task_metadata(task) -> dict:
    return {
        "recipe_dir": task.recipe_dir,
        "variant": str(task.variant),
        "gpu_name": task.gpu_name,
        "gpu_short": task.gpu_short,
        "gpu_count": task.gpu_count,
    }


def compose_json_result(
    task,
    benchmark_output: str,
    compose_content: str,
    bench_command: str,
    system_info_raw: str,
    timing: dict[str, float] | None = None,
) -> dict:
    """Assemble the structured JSON result dict from all benchmark data.

    ``timing`` (a flat ``phase -> seconds`` dict plus a ``total`` key) is added under
    a ``"timing"`` key only when provided. ``model_load_and_warmup`` is the deploy
    window that covers weight load + CUDA graph capture; the optional
    ``weights_load`` / ``cuda_graph_capture`` keys break it down (best-effort) and are
    excluded from ``total``. Note ``timing["benchmark"]`` is wall-clock (incl. docker
    bench-client startup), distinct from ``metrics.benchmark_duration_s`` (the
    server-measured window).
    """
    repeats = parse_repeat_metrics(benchmark_output)
    mean_metrics, stddev = aggregate_metrics(repeats)
    result = {
        "task": _task_metadata(task),
        "recipe": asdict(task.recipe),
        "metrics": asdict(mean_metrics),
        "system": asdict(parse_system_info(system_info_raw)),
        "compose": redact_secrets(compose_content),
        "bench_command": bench_command,
    }
    if len(repeats) > 1:
        result["metrics_stddev"] = stddev
        result["metrics_repeats"] = [asdict(r) for r in repeats]
    if timing is not None:
        result["timing"] = timing
    return result


def compose_command_json_result(
    task,
    command_info: dict,
    system_info_raw: str,
    *,
    success: bool,
    timing: dict[str, float] | None = None,
    source: dict | None = None,
) -> dict:
    """Assemble the standard result for a command-recipe task."""
    result = {
        "task": _task_metadata(task),
        "recipe": asdict(task.recipe),
        "command": redact_secrets(command_info),
        "system": asdict(parse_system_info(system_info_raw)),
        "status": "ok" if success else "failed",
    }
    if timing is not None:
        result["timing"] = timing
    if source is not None:
        result["source"] = source
    return result


def missing_command_provenance(system_info_raw: str, source: dict | None) -> list[str]:
    """Return missing fields required for reproducible command measurements."""
    system = parse_system_info(system_info_raw)
    missing = []
    if not source or not source.get("source_id") or not source.get("files"):
        missing.append("staged source manifest")
    if not system.gpu_provenance:
        missing.append("GPU provenance")
    if not system.cuda_compiler:
        missing.append("CUDA compiler provenance")
    return missing
