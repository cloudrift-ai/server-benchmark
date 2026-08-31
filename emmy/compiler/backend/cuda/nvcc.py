"""Offline kernel compilation via the ``nvcc`` binary (ptxas).

cupy's NVRTC path was the v1 default; we use ``nvcc --cubin`` exclusively
now because it's ~3× faster than cold NVRTC on the complex tile-search
kernels that dominate autotune, GPU-free for the compile step, and the
cubin loads with no driver JIT (~25 ms). (The original trigger was
historical: cupy's bundled cu13 toolkit lacked ``crt/mma.h``, so the
then-tensor-core ``wmma::*`` kernels couldn't compile through NVRTC. That
node family is gone now — the s16816 ``mma.sync`` path emits pure PTX with
no ``<mma.h>`` — but the perf / cubin-cache wins kept nvcc as the only path.)

The two halves are split on purpose:

- :func:`compile_to_cubin` — ``nvcc --cubin`` into a content-addressed disk
  cache. GPU-free and independent per kernel, so a compile **pool** can warm
  the cache off the GPU (the planned next step).
- :func:`load_function` — ensure the cubin exists, then ``RawModule``-load it
  on the GPU. This is all the bench worker needs once the cache is warm.

``nvcc`` is required — there is no NVRTC fallback. Install the CUDA
toolkit (``nvcc`` on ``$PATH`` or under ``$CUDA_HOME``/``$CUDA_PATH``)
or set ``EMMY_NO_NVCC=1`` and accept the resulting hard error on
any kernel that needs ``<mma.h>``.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from emmy import config

logger = logging.getLogger(__name__)

# Base nvcc flags emmy always compiles with (matches ``program._nvrtc_options``).
_BASE_FLAGS = ["--use_fast_math"]


def effective_flags() -> list[str]:
    """The full nvcc flag list: the base flags plus any extra flags from the
    ``EMMY_NVCC_FLAGS`` env var (space-separated), which the CLI commands
    set — ``tune``, ``compile`` and ``run`` all default to nvcc's own cicc -O3.
    Read fresh each call so a per-invocation override / the bench-worker
    subprocess (which inherits the env) both see the same value, and so the
    flags fold into the cache key."""
    return [*_BASE_FLAGS, *config.nvcc_flags().split()]


def cubin_cache_dir() -> Path:
    """Directory holding the content-addressed cubin cache (``EMMY_CUBIN_CACHE``)."""
    return config.cubin_cache_dir()


def clear_cubin_cache() -> None:
    """Delete the entire cubin cache (used by ``emmy tune --clean``)."""
    shutil.rmtree(cubin_cache_dir(), ignore_errors=True)


@functools.cache
def nvcc_path() -> str | None:
    """Resolve the ``nvcc`` binary (PATH, then ``$CUDA_HOME``/``$CUDA_PATH``),
    or ``None`` when unavailable. Cached — looked up once per process."""
    if config.nvcc_disabled():
        return None
    found = shutil.which("nvcc")
    if found:
        return found
    for env in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env)
        if root and (cand := Path(root) / "bin" / "nvcc").exists():
            return str(cand)
    return None


def device_arch(arch_specific: bool) -> str:
    """The active compiler target, plus the ``a`` suffix when the kernel needs the ARCH-SPECIFIC ISA.

    Two instruction families need it, for unrelated reasons: TMA, and the block-scaled fp4 mma
    (ptxas assembles that one only for an arch-suffixed sm_120a). So the flag names the need, not
    either instruction."""
    from emmy.compiler.target import compute_capability  # noqa: PLC0415

    major, minor = compute_capability()
    cap = f"{major}{minor}"
    return f"sm_{cap}" + ("a" if arch_specific else "")


def _launchable_arch(arch_specific: bool) -> str:
    """The arch for a cubin THIS process is about to launch — the LIVE device's, not the target's.

    ``--target`` means "make every lowering decision as if on that GPU" (TMA / cp.async gating,
    atom family, goldens), and :func:`device_arch` follows it because an ahead-of-time bake really
    is building for that card. A cubin the current process then launches has a harder constraint:
    a cubin is not forward-compatible, so one assembled for a lower target will not load here at
    all (``CUDA_ERROR_NO_BINARY_FOR_GPU``). Only the ISA the ALREADY-CHOSEN instructions are
    assembled into follows the live device; nothing about the lowering changes. Falls back to the
    target when no device answers (there is nothing to launch on anyway)."""
    import cupy as cp  # noqa: PLC0415

    try:
        cap = str(cp.cuda.Device().compute_capability)
    except Exception:  # noqa: BLE001 — no device to probe; the target is the only answer left
        return device_arch(arch_specific)
    return f"sm_{cap}" + ("a" if arch_specific else "")


@functools.cache
def _toolkit_tag() -> str:
    """Short digest of the ``nvcc`` toolchain (``nvcc --version``), folded into
    the cache key so a CUDA upgrade never reuses a cubin compiled by an older
    ptxas (which could emit different / worse SASS for the same source). Run
    once per process."""
    nvcc = nvcc_path()
    try:
        ver = subprocess.run([nvcc, "--version"], check=True, capture_output=True, text=True).stdout
    except Exception:  # noqa: BLE001 — fall back to the path; never block a compile on version probing
        ver = nvcc or "?"
    return hashlib.sha1(ver.encode()).hexdigest()[:12]


def _cubin_key(source: str, name: str, arch: str) -> str:
    # Content-addressed: identical (source, name, arch, toolkit, flags) → same
    # cubin, so the persistent cache is safe to share across (even concurrent)
    # runs. Toolkit + flags are in the key so an nvcc / opt-level / flags change
    # recompiles rather than serving a stale or wrong-opt cubin.
    h = hashlib.sha1()
    for part in (source, name, arch, _toolkit_tag(), "\x1f".join(effective_flags())):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def compile_to_cubin(source: str, name: str, *, arch: str) -> Path:
    """Compile ``source`` to a cubin with ``nvcc --cubin``, content-addressed in
    the on-disk cache. Idempotent + atomic (compile to a temp file, then
    ``os.replace``) so concurrent compilers / the bench loader never observe a
    half-written cubin. GPU-free — safe to call from a worker pool. Raises
    ``RuntimeError`` if ``nvcc`` is unavailable; ``CalledProcessError`` on a
    compile error (caller decides whether to fall back).

    The opt level comes from :func:`effective_flags` (``EMMY_NVCC_FLAGS``). Every command —
    ``tune`` included — defaults to nvcc's own -O3, the deployable regime, so a tuned latency is
    the deployed one. A lower level pinned by hand compiles no faster on current codegen and
    mis-ranks by tile area; see ``backend/cuda/ARCHITECTURE.md``."""
    nvcc = nvcc_path()
    if nvcc is None:
        raise RuntimeError("nvcc unavailable")
    cache = cubin_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{_cubin_key(source, name, arch)}.cubin"
    if out.exists():
        return out
    with tempfile.TemporaryDirectory(dir=cache) as td:
        cu = Path(td) / "k.cu"
        cu.write_text(source)
        tmp_cubin = Path(td) / "k.cubin"
        subprocess.run(
            [nvcc, "--cubin", f"-arch={arch}", *effective_flags(), "-o", str(tmp_cubin), str(cu)],
            check=True,
            capture_output=True,
        )
        os.replace(tmp_cubin, out)  # atomic publish
    return out


def load_function(source: str, name: str, options, *, arch_specific: bool):  # noqa: ARG001 — options kept for call-site compat
    """Compile (via nvcc, cached) + ``RawModule``-load ``name``, returning a
    cupy ``Function`` usable exactly like a ``RawKernel`` at launch (callable,
    and ``max_dynamic_shared_size_bytes`` is settable for the >48KB smem path).

    Raises ``RuntimeError`` if ``nvcc`` is unavailable — the NVRTC fallback
    was dropped (faster compiles, GPU-free, cubin-cacheable; see the module
    docstring), so ``nvcc`` is now a hard dependency.
    """
    if nvcc_path() is None:
        raise RuntimeError(
            "nvcc unavailable — emmy requires the CUDA toolkit's "
            "nvcc binary on PATH / under $CUDA_HOME (the NVRTC fallback was "
            "dropped for faster, GPU-free, cubin-cacheable compiles)"
        )
    try:
        cubin = compile_to_cubin(source, name, arch=_launchable_arch(arch_specific))
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace") if exc.stderr else "(no stderr)"
        logger.error("nvcc compile failed for kernel %r:\n%s", name, detail)
        raise RuntimeError(f"nvcc compile failed for kernel {name!r}: {detail[-400:]}") from exc
    return load_cubin_function(cubin, name)


def load_cubin_function(path: Path | str, name: str):
    """``RawModule``-load an existing cubin and return kernel ``name`` — the load half of
    :func:`load_function`, used directly by the execution-plan path when a plan references the
    cubin by its content-addressed cache key (no source, no compile).

    ONE NAME, ONE SOURCE: launches resolve kernels by function name (first source wins), so two
    kernels compiled from different sources under one name silently run one body twice. Every
    kernel-minting or kernel-rewriting site must keep names unique per distinct source — the cut
    pass suffixes producer names with the seam digest (``lowering/tile/_cut.realize``), and the
    zero-init delegation re-suffixes with the baked word count (``lowering/cuda/005``)."""
    import cupy as cp  # noqa: PLC0415

    return cp.RawModule(path=str(path)).get_function(name)
