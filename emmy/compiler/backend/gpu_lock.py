"""Cross-process GPU lock used by the CUDA backend.

Every entry point that issues CUDA work (NVRTC compile, kernel launch,
bench loop) acquires this lock so that concurrent worker processes —
``make bench-kernels``, ``make bench-kernels-tune``, parallel
``emmy run`` invocations from xdist — never interleave kernels on
the same GPU. Without it, two processes' kernels share clocks / caches /
thermal state and timings turn into noise (we saw 2× variance on tiny
ops like ``rmsnorm`` and ``silu_mul`` at small seqlens).

The lock is scoped per PHYSICAL device (UUID suffix on the lock path):
multi-rank serving workers each own a different card and must never
serialize — or deadlock — on one machine-wide lock.

Activated when ``EMMY_GPU_LOCK`` is set to a path (the perf
conftest exports ``/tmp/emmy-gpu.lock``); otherwise the context
manager is a no-op so ad-hoc ``emmy run`` invocations don't pay
any coordination overhead.

Re-entrant within a single process: the same thread can ``with
gpu_lock():`` nested arbitrarily deep. ``filelock`` already handles this
on a per-instance basis; we share the instance via ``_LOCK_CACHE`` so
nested calls inside the same process don't deadlock against themselves.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

from emmy import config

_LOCK_CACHE: dict[str, object] = {}


def _device_scoped(path: str) -> str:
    """One lock per PHYSICAL device: processes contend only when they share the card.

    The lock exists to keep two processes' kernels off the SAME GPU; ranks of a tensor- or
    pipeline-parallel serving group each own a different card, and a machine-wide lock
    deadlocks them — one rank holds it while its stream waits on a collective whose peer
    needs the lock to reach the matching call. ``CUDA_VISIBLE_DEVICES`` renumbers every
    worker's card to index 0, so the scope key is the device UUID, not the index."""
    with contextlib.suppress(Exception):
        import torch  # noqa: PLC0415 — resolve lazily; the lock is also used before CUDA init

        if torch.cuda.is_available():
            return f"{path}.{torch.cuda.get_device_properties(torch.cuda.current_device()).uuid}"
    return path


def _resolve_lock():
    """Return the cached ``FileLock`` instance, or ``None`` for no-op."""
    path = config.gpu_lock_path()
    if not path:
        return None
    path = _device_scoped(path)
    cached = _LOCK_CACHE.get(path)
    if cached is not None:
        return cached
    from filelock import FileLock  # noqa: PLC0415 — optional dep, deferred

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(path)
    _LOCK_CACHE[path] = lock
    return lock


@contextlib.contextmanager
def gpu_lock() -> Iterator[None]:
    """Hold the cross-process GPU lock for the duration of the block.

    No-op when ``EMMY_GPU_LOCK`` is unset. Otherwise wraps the
    shared ``FileLock`` so any code inside is guaranteed sole access
    to the device across processes."""
    lock = _resolve_lock()
    if lock is None:
        yield
        return
    with lock:
        yield
