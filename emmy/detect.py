"""GPU detection via PCI sysfs device IDs."""

import asyncio
import logging

from emmy.provisioning.ssh_transport import ssh_base_args
from emmy.system_info import GPU_PCI_INFORMATION_COMMAND, SystemInformation

logger = logging.getLogger(__name__)


def _parse_sysfs_output(output: str) -> tuple[str, int]:
    """Parse the shared PCI inventory and return (gpu_name, count)."""
    return SystemInformation.gpu_summary_from_pci(output)


def detect_local_gpus() -> tuple[str, int]:
    """Detect local GPUs by scanning PCI sysfs. Returns (gpu_name, count)."""
    import subprocess

    result = subprocess.run(
        ["bash", "-c", GPU_PCI_INFORMATION_COMMAND],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to scan PCI devices: {result.stderr}")

    return _parse_sysfs_output(result.stdout)


async def detect_remote_gpus(server: str, ssh_key: str, ssh_port: int) -> tuple[str, int]:
    """Detect GPUs on a remote server via SSH. Returns (gpu_name, count)."""
    args = ssh_base_args(server, ssh_key, ssh_port)
    args.append(GPU_PCI_INFORMATION_COMMAND)

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=30)

    if proc.returncode != 0:
        stderr = stderr_bytes.decode() if stderr_bytes else ""
        raise RuntimeError(f"Failed to scan PCI devices on {server}: {stderr}")

    return _parse_sysfs_output(stdout_bytes.decode())
