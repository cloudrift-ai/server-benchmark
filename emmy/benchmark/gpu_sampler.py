"""GPU power/VRAM sampling during benchmark runs.

Wrap a benchmark invocation in ``gpu_sampling(run_cmd)`` to record average power draw (W)
and peak memory used (MiB) per GPU while the benchmark runs. ``run_cmd`` is the injected
async command runner (see STYLE.md), so sampling happens on whatever host runs the
benchmark — typically a remote server over SSH.

Because ``run_cmd`` is one-shot, sampling is done host-side: on enter, a background
``nvidia-smi -lms`` loop is started with ``nohup``, appending one CSV line per GPU per
tick to a unique temp file; on exit, the loop is killed and the file is read back,
parsed, and deleted.

Sampling is best-effort and never fails the benchmark: if the sampler cannot start or
produces no parseable samples (no nvidia-smi, file lost, all values ``[N/A]``),
``summary()`` returns None and a warning is logged.

Note: ``summary()`` only has data after the ``async with`` block exits — the samples are
collected on exit.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_MS = 500


@dataclass
class GpuStats:
    """Per-GPU aggregates. A field is None when every sample for it was [N/A]."""

    index: int
    avg_power_w: float | None
    peak_memory_mib: float | None
    samples: int


@dataclass
class GpuSamplingSummary:
    """Aggregates over one sampling session.

    ``total_avg_power_w`` sums the per-GPU average power; ``peak_memory_mib`` is the
    max over per-GPU peaks (GPUs with no valid samples for a field are skipped).
    """

    gpus: list[GpuStats]
    total_avg_power_w: float | None
    peak_memory_mib: float | None

    def to_dict(self) -> dict:
        """Plain-dict form for merging into the benchmark JSON result."""
        return asdict(self)


def _parse_value(field: str) -> float | None:
    """A numeric CSV field, or None for [N/A] / truncated garbage."""
    try:
        return float(field)
    except ValueError:
        return None


def parse_gpu_samples(text: str) -> GpuSamplingSummary | None:
    """Parse ``nvidia-smi --query-gpu=index,power.draw,memory.used`` CSV output.

    Tolerates [N/A] values, blank lines, and a truncated last line (the sampler is
    killed mid-write). Returns None when no line yields a valid sample.
    """
    power: dict[int, list[float]] = {}
    memory: dict[int, list[float]] = {}
    counts: dict[int, int] = {}
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        counts[index] = counts.get(index, 0) + 1
        if (p := _parse_value(fields[1])) is not None:
            power.setdefault(index, []).append(p)
        if (m := _parse_value(fields[2])) is not None:
            memory.setdefault(index, []).append(m)

    if not counts:
        return None
    gpus = [
        GpuStats(
            index=index,
            avg_power_w=sum(p) / len(p) if (p := power.get(index)) else None,
            peak_memory_mib=max(m) if (m := memory.get(index)) else None,
            samples=counts[index],
        )
        for index in sorted(counts)
    ]
    avg_powers = [g.avg_power_w for g in gpus if g.avg_power_w is not None]
    peaks = [g.peak_memory_mib for g in gpus if g.peak_memory_mib is not None]
    return GpuSamplingSummary(
        gpus=gpus,
        total_avg_power_w=sum(avg_powers) if avg_powers else None,
        peak_memory_mib=max(peaks) if peaks else None,
    )


def _parse_pid(stdout: str) -> int | None:
    try:
        return int(stdout.split()[-1])
    except (IndexError, ValueError):
        return None


class GpuSampler:
    """Handle yielded by gpu_sampling(). summary() is None until the block exits."""

    def __init__(self):
        self._summary: GpuSamplingSummary | None = None

    def summary(self) -> GpuSamplingSummary | None:
        return self._summary


@asynccontextmanager
async def gpu_sampling(run_cmd, interval_ms: int = SAMPLE_INTERVAL_MS) -> AsyncIterator[GpuSampler]:
    """Sample GPU power/memory on the benchmark host for the duration of the block.

    Usage::

        async with gpu_sampling(run_cmd) as sampler:
            ...run the benchmark...
        summary = sampler.summary()  # GpuSamplingSummary | None

    The temp file name is unique per session, so concurrent benchmarks on one host
    don't collide. Samples are still collected (and the temp file removed) when the
    block raises.
    """
    sampler = GpuSampler()
    remote_file = f"/tmp/emmy_gpu_samples_{uuid.uuid4().hex}.csv"
    start_cmd = (
        f"nohup nvidia-smi --query-gpu=index,power.draw,memory.used "
        f"--format=csv,noheader,nounits -lms {interval_ms} > {remote_file} 2>&1 & echo $!"
    )
    rc, stdout, _ = await run_cmd(start_cmd, stream=False, timeout=60)
    pid = _parse_pid(stdout) if rc == 0 else None
    if pid is None:
        logger.warning("GPU sampler failed to start; benchmark runs without power/memory stats")
    try:
        yield sampler
    finally:
        if pid is not None:
            stop_cmd = f"kill {pid} 2>/dev/null; cat {remote_file} 2>/dev/null; rm -f {remote_file}"
            rc, stdout, _ = await run_cmd(stop_cmd, stream=False, timeout=120)
            summary = parse_gpu_samples(stdout) if rc == 0 else None
            if summary is None:
                logger.warning("No GPU samples collected")
            sampler._summary = summary
