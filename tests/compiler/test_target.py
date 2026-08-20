"""The NVRTC-versus-live-device guard on the compile target."""

import sys
import types

import pytest

from emmy.compiler import target


@pytest.fixture
def nvrtc(monkeypatch):
    """Install a fake live device capability and a fake loaded NVRTC version."""

    def _install(cap, version):
        monkeypatch.setattr(target, "live_compute_capability", lambda: cap)
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(nvrtc=types.SimpleNamespace(getVersion=lambda: version)))
        monkeypatch.setitem(sys.modules, "cupy", fake)

    return _install


@pytest.mark.parametrize(
    ("cap", "version", "rejected"),
    [
        ((7, 0), (13, 0), True),  # V100 under the nvidia-cuda-nvrtc 13 that torch pulls in
        ((6, 0), (13, 0), True),  # P100 fails identically — CUDA 13 dropped Pascal too
        ((7, 5), (13, 0), False),  # Turing is the CUDA 13 floor
        ((7, 0), (12, 9), False),  # a CUDA 12 NVRTC still targets Volta
        ((0, 0), (13, 0), False),  # no visible device: nothing to compile for
        ((7, 0), (14, 0), False),  # unrecorded major: no floor to judge against
    ],
)
def test_guard_rejects_only_untargetable_devices(nvrtc, cap, version, rejected):
    nvrtc(cap, version)

    if not rejected:
        target.check_nvrtc_supports_live_device()
        return
    with pytest.raises(SystemExit):
        target.check_nvrtc_supports_live_device()


def test_rejection_names_the_arch_and_the_remedy(nvrtc):
    nvrtc((7, 0), (13, 0))

    with pytest.raises(SystemExit) as excinfo:
        target.check_nvrtc_supports_live_device()

    message = str(excinfo.value)
    assert "sm_70" in message
    assert "sm_75" in message
    assert "LD_PRELOAD=" in message
    assert "libnvrtc.so.12" in message
