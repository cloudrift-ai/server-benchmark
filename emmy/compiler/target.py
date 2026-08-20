"""Compile-time hardware target selection.

The tile-IR passes that gate on compute capability (``050_use_tma`` for
TMA, ``060_use_async_copy`` for cp.async) read the target via
:func:`compute_capability`. By default that probes the live CUDA device
through cupy. Callers can override the target via :func:`set_target` so
the compiler emits code for a different architecture than the host —
useful for ``emmy compile`` on a CPU box that wants to see the
sm_120 codegen path, or for cross-compiling for a benchmark target.

CLI commands attach their compute-capability option with :func:`add_target_arg`
and resolve it via :func:`apply_target_arg`. The options accept the
canonical NVIDIA spelling (``sm_80``, ``sm_90``, ``sm_120``); the
optional architecture suffix on Hopper (``sm_90a``) is stripped.
"""

from __future__ import annotations

import functools
import logging
import re
from pathlib import Path

_logger = logging.getLogger(__name__)

_OVERRIDE: tuple[int, int] | None = None


def parse_sm(spec: str) -> tuple[int, int]:
    """Parse ``sm_NN`` / ``sm_NNN`` (with optional arch suffix) into ``(major, minor)``.

    Examples: ``sm_80`` → (8, 0), ``sm_86`` → (8, 6), ``sm_90`` → (9, 0),
    ``sm_90a`` → (9, 0), ``sm_120`` → (12, 0).
    """
    m = re.fullmatch(r"sm_(\d+)[a-z]?", spec.strip().lower())
    if not m:
        raise ValueError(f"invalid SM target {spec!r} — expected e.g. sm_80, sm_90, sm_120")
    digits = m.group(1)
    return (int(digits[:-1]), int(digits[-1]))


def set_target(cap: tuple[int, int] | None) -> None:
    """Set (or clear) the compile-time compute-capability override.

    Pass ``None`` to revert to live device probing. Clears the
    :func:`compute_capability` cache so the next caller sees the change.
    """
    global _OVERRIDE
    _OVERRIDE = cap
    compute_capability.cache_clear()


def add_target_arg(parser, *, dest: str = "target", option: str = "--target") -> None:
    """Add a compute-capability argument to ``parser``.

    The parsed value is a string; pass it to :func:`apply_target_arg` after parsing
    to install the override. Most commands use ``--target``; ``run`` reserves that
    spelling for working-golden selection and uses ``--gpu-arch`` here.
    """
    parser.add_argument(
        option,
        dest=dest,
        default=None,
        metavar="sm_NN",
        help=(
            "Compile-time target compute capability (e.g. sm_80, sm_90, sm_120). "
            "Overrides the live device's capability, so passes that gate on "
            "hardware features (TMA, cp.async) take the same path they would on "
            "the target GPU. Default: probe the active CUDA device."
        ),
    )


def apply_target_arg(args, *, dest: str = "target") -> None:
    """Install the target from a parsed-args namespace, if set."""
    spec = getattr(args, dest, None)
    if spec is None:
        return
    cap = parse_sm(spec)
    set_target(cap)
    _logger.info("compile target set to sm_%d%d (override)", cap[0], cap[1])


@functools.cache
def compute_capability() -> tuple[int, int]:
    """Active compute capability as ``(major, minor)``.

    Returns the override set via :func:`set_target` if any; otherwise
    delegates to :func:`live_compute_capability`. Cached so repeated
    rule firings don't re-query the driver; :func:`set_target` clears
    the cache.
    """
    if _OVERRIDE is not None:
        return _OVERRIDE
    return live_compute_capability()


@functools.cache
def live_device_features() -> dict[str, float]:
    """Physical properties of the live CUDA device — SM count, shared memory per
    SM / per block, register file, warp size — for the online prior's
    hardware-regime features (see :meth:`Context.features`). These are SKU facts
    CUDA reports but compute-capability alone doesn't fix (an sm_120 laptop and an
    sm_120 RTX 5090 differ in SM count). Delegates to
    :func:`emmy.gpu.probe_live_features`, which probes the live device via
    cupy and, when none is visible (GPU-less CI / offline eval), falls back to the
    **memorized** specs of :data:`emmy.gpu.DEFAULT_GPU` — so offline hosts get
    faithful per-SKU features instead of none. Cached — physical and
    target-independent, so :func:`set_target` need not clear it."""
    from emmy import gpu  # noqa: PLC0415

    return gpu.probe_live_features()


@functools.cache
def live_compute_capability() -> tuple[int, int]:
    """The live CUDA device's compute capability, ignoring any
    :func:`set_target` override.

    Returns ``(0, 0)`` when cupy is unavailable. Used by ``Context.probe``
    to size ``max_dynamic_smem`` to what the actual hardware can honor
    even when the target-derived gate cap is higher.
    """
    try:
        import cupy as cp

        dev = cp.cuda.Device()
        # cupy returns the capability as a string ``"MMm"``: ``"86"`` for
        # sm_86, ``"120"`` for sm_12.0. Minor is always the last digit.
        cap = str(dev.compute_capability)
        return (int(cap[:-1]), int(cap[-1]))
    except Exception as e:  # pragma: no cover
        _logger.debug("live_compute_capability query failed (%s)", e)
        return (0, 0)


# Lowest compute capability each NVRTC major version can compile for. CUDA 13
# removed every architecture below sm_75 — Maxwell, Pascal (P100, sm_60) and
# Volta (V100, sm_70) alike; CUDA 12 still targets sm_50 and up. A major absent
# here is not checked.
_NVRTC_MIN_CC: dict[int, tuple[int, int]] = {12: (5, 0), 13: (7, 5)}


def check_nvrtc_supports_live_device() -> None:
    """Abort when the NVRTC cupy loaded is too new to compile for the live GPU.

    ``torch`` depends on ``nvidia-cuda-nvrtc`` 13.x and cupy resolves NVRTC to
    that in preference to any CUDA 12 build, so on a pre-Turing card every cupy
    runtime compile dies with ``NVRTC_ERROR_INVALID_OPTION`` (``invalid value
    for --gpu-architecture``). Left undetected that surfaces as hundreds of
    unrelated-looking bench failures instead of one setup problem. No-op
    without cupy, without a visible device, or on an unrecorded NVRTC major.
    """
    cap = live_compute_capability()
    if cap == (0, 0):
        return  # no device to compile for
    try:
        import cupy as cp

        nvrtc_major = cp.cuda.nvrtc.getVersion()[0]
    except Exception as e:  # noqa: BLE001 — no cupy / no NVRTC ⇒ nothing to check
        _logger.debug("NVRTC version query failed (%s)", e)
        return
    floor = _NVRTC_MIN_CC.get(nvrtc_major)
    if floor is None or cap >= floor:
        return
    preload = next(iter(sorted(Path("/usr/local").glob("cuda-12*/lib64/libnvrtc.so.12"), reverse=True)), None)
    raise SystemExit(
        f"NVRTC {nvrtc_major}.x cannot target this GPU (sm_{cap[0]}{cap[1]}): CUDA {nvrtc_major} dropped every "
        f"architecture below sm_{floor[0]}{floor[1]}, so every cupy runtime compile fails with "
        "NVRTC_ERROR_INVALID_OPTION.\n"
        f"torch depends on nvidia-cuda-nvrtc {nvrtc_major}.x and cupy prefers it over any CUDA 12 NVRTC, so "
        "installing nvidia-cuda-nvrtc-cu12 does not help.\n"
        f"Remedy: run emmy with LD_PRELOAD={preload or '/usr/local/cuda-12.x/lib64/libnvrtc.so.12'}"
    )
