"""Unit tests for GPU detection via PCI sysfs."""

from unittest.mock import AsyncMock, patch

import pytest

from emmy.detect import _parse_sysfs_output, detect_local_gpus, detect_remote_gpus
from emmy.system_info import GPU_PCI_INFORMATION_COMMAND

# ── _parse_sysfs_output ────────────────────────────────────────────


def test_detect_local_gpus_nvidia():
    """Parse sysfs output with two identical NVIDIA GPUs."""
    output = "0000:01:00.0,0x10de,0x2b85\n0000:02:00.0,0x10de,0x2b85\n"
    name, count = _parse_sysfs_output(output)
    assert name == "NVIDIA GeForce RTX 5090"
    assert count == 2


def test_detect_local_gpus_mixed_error():
    """Different GPU types raise RuntimeError."""
    output = "0000:01:00.0,0x10de,0x2b85\n0000:02:00.0,0x10de,0x2684\n"
    with pytest.raises(RuntimeError, match="Mixed GPU types"):
        _parse_sysfs_output(output)


def test_detect_local_gpus_no_gpus():
    """No recognized GPUs raise RuntimeError."""
    output = ""
    with pytest.raises(RuntimeError, match="No supported GPUs"):
        _parse_sysfs_output(output)


def test_detect_local_gpus_amd():
    """Parse sysfs output with AMD GPU."""
    output = "0000:41:00.0,0x1002,0x75b0\n"
    name, count = _parse_sysfs_output(output)
    assert name == "AMD Instinct MI350X"
    assert count == 1


def test_detect_local_gpus_empty():
    """Empty output raises RuntimeError."""
    with pytest.raises(RuntimeError, match="No supported GPUs"):
        _parse_sysfs_output("")


# ── detect_local_gpus ──────────────────────────────────────────────


def test_detect_local_gpus_subprocess():
    """detect_local_gpus calls bash and parses output."""
    sysfs_output = "".join(f"0000:{index:02x}:00.0,0x10de,0x2684\n" for index in range(4))
    mock_result = type("Result", (), {"returncode": 0, "stdout": sysfs_output, "stderr": ""})()
    with patch("subprocess.run", return_value=mock_result) as run:
        name, count = detect_local_gpus()
        assert name == "NVIDIA GeForce RTX 4090"
        assert count == 4
    run.assert_called_once_with(
        ["bash", "-c", GPU_PCI_INFORMATION_COMMAND],
        capture_output=True,
        text=True,
    )


# ── detect_remote_gpus ─────────────────────────────────────────────


async def test_detect_remote_gpus():
    """detect_remote_gpus runs SSH and parses output."""
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"0000:01:00.0,0x10de,0x2330\n0000:02:00.0,0x10de,0x2330\n", b"")
    mock_proc.returncode = 0

    with patch("emmy.detect.asyncio.create_subprocess_exec", return_value=mock_proc) as create:
        name, count = await detect_remote_gpus("user@host", "~/.ssh/id_ed25519", 22)
        assert name == "NVIDIA H100 80GB"
        assert count == 2
    assert create.call_args.args[-1] == GPU_PCI_INFORMATION_COMMAND
