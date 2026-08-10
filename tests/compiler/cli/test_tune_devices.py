"""CLI tests for ``emmy tune --gpus / --devices`` device resolution (no GPU).

``_resolve_devices`` maps the flags to a device-id list (``--devices`` wins) and,
for two or more devices, enforces homogeneity (one perf key per tune). The
homogeneity probe needs cupy, so it's exercised here by monkeypatching the
device-properties call.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from emmy.commands import tune


async def test_worker_readiness_ramps_at_two_backends() -> None:
    active = 0
    peak = 0

    class Backend:
        async def warm_async_worker(self):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1

    await tune._warm_tune_backends([Backend() for _ in range(7)])
    assert peak == 2


def _args(*, gpus=None, devices=None):
    return SimpleNamespace(gpus=gpus, devices=devices)


def test_default_is_single_unpinned_slot() -> None:
    assert tune._resolve_devices(_args()) == [None]


def test_gpus_n_expands_to_range(monkeypatch) -> None:
    # This test covers flag expansion; homogeneity has dedicated mocked tests below.
    monkeypatch.setattr(tune, "_require_homogeneous_devices", lambda _devices: None)
    assert tune._resolve_devices(_args(gpus=3)) == [0, 1, 2]


def test_devices_list_wins_over_gpus(monkeypatch) -> None:
    monkeypatch.setattr(tune, "_require_homogeneous_devices", lambda _devices: None)
    assert tune._resolve_devices(_args(gpus=8, devices="0,2,5")) == [0, 2, 5]


def test_single_device_skips_homogeneity() -> None:
    assert tune._resolve_devices(_args(devices="1")) == [1]


def test_bad_devices_exits(capsys) -> None:
    with pytest.raises(SystemExit) as e:
        tune._resolve_devices(_args(devices="0,x,2"))
    assert e.value.code == 2


def test_gpus_below_one_exits() -> None:
    with pytest.raises(SystemExit) as e:
        tune._resolve_devices(_args(gpus=0))
    assert e.value.code == 2


def test_heterogeneous_devices_rejected(monkeypatch) -> None:
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceProperties=lambda d: {"major": 8 if d == 0 else 9, "minor": 0}))
    )
    monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cupy)
    with pytest.raises(SystemExit) as e:
        tune._resolve_devices(_args(devices="0,1"))
    assert e.value.code == 2


def test_homogeneous_devices_accepted(monkeypatch) -> None:
    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceProperties=lambda d: {"major": 9, "minor": 0})))
    monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cupy)
    assert tune._resolve_devices(_args(devices="0,1,2")) == [0, 1, 2]


def test_same_capability_different_gpu_names_rejected(monkeypatch) -> None:
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                getDeviceProperties=lambda d: {"major": 9, "minor": 0, "name": b"NVIDIA H100" if d == 0 else b"NVIDIA H200"}
            )
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cupy)
    with pytest.raises(SystemExit) as exc:
        tune._resolve_devices(_args(devices="0,1"))
    assert exc.value.code == 2


def test_selected_nonzero_device_builds_context_from_that_ordinal(monkeypatch) -> None:
    seen = []

    def properties(device_id):
        seen.append(device_id)
        return {
            "major": 9,
            "minor": 0,
            "name": b"NVIDIA H200",
            "multiProcessorCount": 132,
            "sharedMemPerMultiprocessor": 233472,
            "sharedMemPerBlock": 49152,
            "regsPerBlock": 65536,
            "warpSize": 32,
            "totalGlobalMem": 150754934784,
            "maxThreadsPerBlock": 1024,
        }

    monkeypatch.setattr(tune, "_device_properties", properties)

    ctx = tune._context_for_device(3)

    assert seen == [3]
    assert ctx.compute_capability == (9, 0)
    assert ctx.gpu_name == "NVIDIA H200 141GB"
    assert ctx.hardware_id() == "NVIDIA H200 141GB"
    assert ctx.sm_count == 132
    assert ctx.device_props["total_mem"] == 150754934784.0
