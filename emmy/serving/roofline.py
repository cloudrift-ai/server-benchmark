"""Boot-time roofline audit: surface catastrophically mispicked serving kernels at startup.

Motivating incident (2026-07-29, TinyLlama-1.1B on an RTX 4080): a cold deploy shipped a fused-norm
decode kernel ~150x off the DRAM floor (10 ms/layer — a 54x serving TPOT gap vs stock vLLM) and nothing
surfaced it; the server booted clean and was simply slow. Root-causing it took a per-kernel attribution
session. This audit turns that class of failure into one boot log line.

Mechanism: each STATIC twin program is event-timed once at boot and compared against its
**weight-streaming floor** — every weight must be read at least once per forward, whether it is a
bound constant or a per-launch weight INPUT (a MoE expert program holds no constants at all), so
``floor = weight_bytes / dram_bw``. Bandwidth is self-calibrated with a device-to-device copy (no
per-card peak table to maintain). The comparison is conservative by construction: the weight floor is a
true lower bound (compute-bound programs sit above it) and copy bandwidth undershoots peak DRAM
bandwidth, so real ratios read LOW — a program flagged ``>10x`` is never a healthy program. Symbolic
programs are skipped: at boot their buffers sit at capacity, not at a serving width, so a timing there
measures the wrong shape.

The audit is advisory only — it logs and never raises; any failure inside it is swallowed (a boot
warning must not become a boot blocker).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Programs whose weight floor is below this are skipped: launch overhead and timer noise dominate
# µs-class programs, and a mispick there costs little in absolute terms.
MIN_FLOOR_US = 20.0
# Warn threshold on measured/floor. Healthy tuned programs measure ~1-3x the weight floor (seam
# copies, pointwise glue, sub-peak streaming); the incident class sits at >100x.
WARN_RATIO = 10.0


def measure_copy_bw() -> float:
    """Device-to-device copy bandwidth in bytes/s (read + write both count)."""
    import cupy as cp

    n = 64 * 1024 * 1024
    a = cp.zeros(n, dtype=cp.uint8)
    b = cp.zeros(n, dtype=cp.uint8)
    cp.copyto(b, a)  # warm: allocator + module load out of the window
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    reps = 4
    start.record()
    for _ in range(reps):
        cp.copyto(b, a)
    stop.record()
    stop.synchronize()
    ms = cp.cuda.get_elapsed_time(start, stop)
    return (2.0 * n * reps) / (ms / 1e3)


def time_program_us(program, *, reps: int = 3) -> float:
    """Median wall time of ``program.run_once()`` in µs (one warmup run outside the window)."""
    import cupy as cp

    program.run_once()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for _ in range(reps):
        start.record()
        program.run_once()
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop) * 1e3)
    return sorted(times)[len(times) // 2]


def flag_ratio(measured_us: float, weight_bytes: int, bw_bytes_per_s: float) -> tuple[float, float] | None:
    """``(floor_us, ratio)`` when the measurement warrants a warning, else ``None``.
    Pure decision logic — unit-testable without CUDA."""
    if weight_bytes <= 0 or bw_bytes_per_s <= 0:
        return None
    floor_us = weight_bytes / bw_bytes_per_s * 1e6
    if floor_us < MIN_FLOOR_US:
        return None
    ratio = measured_us / floor_us
    if ratio <= WARN_RATIO:
        return None
    return floor_us, ratio


def audit_boot_programs(named_programs) -> None:
    """Time each ``(label, _Program)`` against its weight-streaming floor and warn on outliers.
    ``_Program`` here is the serving wrapper (``gen_runner._Program``): ``.program`` is the
    ``CompiledProgram`` and ``.weight_bytes`` the per-forward weight footprint — bound constants
    plus the weight INPUTS an expert program takes per launch. Never raises."""
    try:
        from emmy.compiler.backend.gpu_lock import gpu_lock

        with gpu_lock():
            bw = measure_copy_bw()
            flagged = []
            for label, prog in named_programs:
                verdict = flag_ratio(time_program_us(prog.program), prog.weight_bytes, bw)
                if verdict is not None:
                    floor_us, ratio = verdict
                    flagged.append((label, floor_us, ratio))
        for label, floor_us, ratio in flagged:
            logger.warning(
                "[roofline] %s runs %.0fx over its weight-streaming floor (~%.0f us) — a deployed kernel "
                "pick is far off the DRAM floor for this model/GPU. Capture and tune the serving twins "
                "(`emmy tune`; see emmy/serving/ARCHITECTURE.md → 'Tuning what serving actually runs').",
                label,
                ratio,
                floor_us,
            )
        if named_programs and not flagged:
            n = len(named_programs)
            logger.info("[roofline] boot audit clean: %d static program(s) within %sx of the weight floor", n, int(WARN_RATIO))
    except Exception:  # noqa: BLE001 — advisory only, never a boot blocker
        logger.debug("[roofline] boot audit skipped", exc_info=True)
