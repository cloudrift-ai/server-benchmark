"""Fabric Manager provisioning: NVSwitch hosts need it before CUDA will initialize."""

from unittest.mock import AsyncMock, patch

import pytest

from emmy.provisioning.remote import _ensure_fabric_manager


class FakeHost:
    """Records commands and answers the two probe commands from a script."""

    name = "fake"

    def __init__(self, *, has_nvswitch: bool, active: list[bool]):
        self.has_nvswitch = has_nvswitch
        self.active = list(active)
        self.commands: list[str] = []

    async def run(self, cmd, *, sudo=False, capture=False, timeout=600):
        self.commands.append(cmd)
        if "nvidia-nvswitch" in cmd:
            return (0 if self.has_nvswitch else 1), ""
        if "systemctl is-active" in cmd:
            return (0 if self.active.pop(0) else 1), ""
        return 0, ""


@pytest.fixture
def patched():
    with (
        patch("emmy.provisioning.remote._current_driver_version", AsyncMock(return_value="580.159.03")),
        patch("emmy.provisioning.remote._setup_nvidia_cuda_repo", AsyncMock()) as repo,
    ):
        yield repo


async def test_no_nvswitch_host_is_left_alone(patched):
    host = FakeHost(has_nvswitch=False, active=[])
    await _ensure_fabric_manager(host)
    assert not any("apt-get install" in c for c in host.commands)
    patched.assert_not_awaited()


async def test_already_running_service_is_not_reinstalled(patched):
    host = FakeHost(has_nvswitch=True, active=[True])
    await _ensure_fabric_manager(host)
    assert not any("apt-get install" in c for c in host.commands)


async def test_nvswitch_host_installs_version_matched_service(patched):
    host = FakeHost(has_nvswitch=True, active=[False, True])
    await _ensure_fabric_manager(host)

    install = next(c for c in host.commands if "apt-get install" in c)
    # Pinned to the running driver, not to a guessed package revision suffix.
    assert 'driver="580.159.03"' in install
    assert 'apt-get install -y "nvidia-fabricmanager=$ver"' in install
    assert "systemctl enable --now nvidia-fabricmanager" in install
    patched.assert_awaited_once()


async def test_service_that_does_not_come_up_is_an_error(patched):
    host = FakeHost(has_nvswitch=True, active=[False, False])
    with pytest.raises(RuntimeError, match="not active"):
        await _ensure_fabric_manager(host)
