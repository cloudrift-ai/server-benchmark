"""Typed generic system information and the shared host probe."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from emmy import gpu

logger = logging.getLogger(__name__)


GPU_PCI_INFORMATION_COMMAND = r"""for device in /sys/bus/pci/devices/*; do
  vendor=$(cat "$device/vendor" 2>/dev/null) || continue
  [ "$vendor" = "0x10de" ] || [ "$vendor" = "0x1002" ] || continue
  printf '%s,%s,%s\n' "$(basename "$device")" "$vendor" "$(cat "$device/device")"
done"""

SYSTEM_INFORMATION_COMMAND = (
    r"""echo "=== HOSTNAME ==="
hostname

echo ""
echo "=== OS ==="
cat /etc/os-release 2>/dev/null || echo "N/A"

echo ""
echo "=== KERNEL ==="
uname -r

echo ""
echo "=== CPU ==="
lscpu 2>/dev/null || echo "N/A"

echo ""
echo "=== MEMORY ==="
awk '/^MemTotal:/{printf "MemTotal: %.0f\n", $2 * 1024}' /proc/meminfo 2>/dev/null || echo "N/A"

echo ""
echo "=== NVIDIA GPUS ==="
gpu_fields=index,name,uuid,pci.bus_id,memory.total,driver_version,pstate,temperature.gpu
gpu_fields=$gpu_fields,utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit
nvidia-smi --query-gpu="$gpu_fields" --format=csv,noheader,nounits 2>/dev/null || echo "N/A"

echo ""
echo "=== NVIDIA DRIVER ==="
nvidia-smi 2>/dev/null || echo "N/A"

echo ""
echo "=== GPU PCI DEVICES ==="
"""
    + GPU_PCI_INFORMATION_COMMAND
    + r"""

echo ""
echo "=== AMD SMI ==="
(amd-smi static --json 2>/dev/null || rocm-smi --showproductname --showuniqueid --showmeminfo vram \
  --showdriverversion --showtemp --showuse --showclocks --showpower --json 2>/dev/null) || echo "N/A"

echo ""
echo "=== CUDA COMPILER ==="
(nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null) || echo "N/A"

echo ""
echo "=== HIP COMPILER ==="
(hipcc --version 2>/dev/null || /opt/rocm/bin/hipcc --version 2>/dev/null) || echo "N/A"

echo ""
echo "=== ROOT FILESYSTEM ==="
df -B1 -T / 2>/dev/null | tail -n 1 || echo "N/A"

echo ""
echo "=== UPTIME SECONDS ==="
cut -d' ' -f1 /proc/uptime 2>/dev/null || echo "N/A"

echo ""
echo "=== DOCKER ==="
printf 'ClientVersion: '
docker version --format '{{.Client.Version}}' 2>/dev/null || echo "N/A"
printf 'ServerVersion: '
docker version --format '{{.Server.Version}}' 2>/dev/null || echo "N/A"
printf 'OperatingSystem: '
docker info --format '{{.OperatingSystem}}' 2>/dev/null || echo "N/A"
"""
)


def _section(raw_text: str, name: str) -> str:
    match = re.search(rf"=== {re.escape(name)} ===\n(.*?)(?=\n=== |\Z)", raw_text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _key_values(text: str, separator: str = ":") -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        if separator not in line:
            continue
        key, value = line.split(separator, 1)
        result[key.strip()] = value.strip().strip('"')
    return result


def _number(value: str | None, typ: type[int] | type[float]) -> int | float | None:
    if not value or value.strip().upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    match = re.search(r"-?[\d.]+", value)
    if not match:
        return None
    try:
        return typ(match.group(0))
    except ValueError:
        return None


def _tool_output(raw_text: str, section: str) -> str | None:
    value = _section(raw_text, section)
    return None if not value or value == "N/A" else value


@dataclass
class OperatingSystemInformation:
    name: str | None = None
    id: str | None = None
    version: str | None = None
    kernel: str | None = None


@dataclass
class CpuInformation:
    model: str | None = None
    architecture: str | None = None
    logical_count: int | None = None
    sockets: int | None = None
    cores_per_socket: int | None = None
    threads_per_core: int | None = None
    numa_nodes: int | None = None


@dataclass
class MemoryInformation:
    total_bytes: int | None = None


@dataclass
class GpuInformation:
    index: int
    name: str | None = None
    vendor: str | None = None
    uuid: str | None = None
    pci_bus_id: str | None = None
    memory_total_mib: int | None = None
    driver_version: str | None = None
    performance_state: str | None = None
    temperature_c: float | None = None
    utilization_percent: float | None = None
    sm_clock_mhz: float | None = None
    memory_clock_mhz: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None


@dataclass
class PciGpuDevice:
    pci_bus_id: str
    vendor_id: str
    device_id: str
    vendor: str | None = None
    name: str | None = None


@dataclass
class SoftwareInformation:
    cuda_driver_api: str | None = None
    cuda_compiler: str | None = None
    hip_compiler: str | None = None
    docker_client: str | None = None
    docker_server: str | None = None
    docker_os: str | None = None


@dataclass
class FilesystemInformation:
    device: str
    type: str
    total_bytes: int | None
    used_bytes: int | None
    available_bytes: int | None
    mount: str


@dataclass
class SystemInformation:
    """Generic hardware and software facts retrieved from a host."""

    hostname: str | None
    os: OperatingSystemInformation
    cpu: CpuInformation
    memory: MemoryInformation
    gpus: list[GpuInformation] = field(default_factory=list)
    gpu_pci_devices: list[PciGpuDevice] = field(default_factory=list)
    software: SoftwareInformation = field(default_factory=SoftwareInformation)
    root_filesystem: FilesystemInformation | None = None
    uptime_seconds: float | None = None
    amd_smi: str | None = None

    @staticmethod
    def _parse_nvidia_gpus(text: str) -> list[GpuInformation]:
        columns = (
            "index",
            "name",
            "uuid",
            "pci_bus_id",
            "memory_total_mib",
            "driver_version",
            "performance_state",
            "temperature_c",
            "utilization_percent",
            "sm_clock_mhz",
            "memory_clock_mhz",
            "power_draw_w",
            "power_limit_w",
        )
        numeric = {
            "index": int,
            "memory_total_mib": int,
            "temperature_c": float,
            "utilization_percent": float,
            "sm_clock_mhz": float,
            "memory_clock_mhz": float,
            "power_draw_w": float,
            "power_limit_w": float,
        }
        gpus = []
        for line in text.splitlines():
            if not line.strip() or line.strip() == "N/A":
                continue
            values = [value.strip() for value in line.split(",")]
            if len(values) != len(columns):
                continue
            item: dict[str, Any] = {"vendor": "NVIDIA"}
            for name, value in zip(columns, values, strict=True):
                item[name] = _number(value, numeric[name]) if name in numeric else (None if value.upper() == "N/A" else value)
            gpus.append(GpuInformation(**item))
        return gpus

    @staticmethod
    def parse_pci_gpus(text: str) -> list[PciGpuDevice]:
        """Parse output from the shared PCI inventory command."""
        devices = []
        for line in text.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) != 3:
                continue
            address, vendor_id, device_id = values
            spec = gpu.by_pci_device_id(device_id)
            devices.append(
                PciGpuDevice(
                    pci_bus_id=address,
                    vendor_id=vendor_id.removeprefix("0x"),
                    device_id=device_id.removeprefix("0x"),
                    vendor=spec.vendor if spec else None,
                    name=spec.name if spec else None,
                )
            )
        return devices

    @classmethod
    def gpu_summary_from_pci(cls, text: str) -> tuple[str, int]:
        """Return the one supported GPU model and count in a PCI inventory."""
        found: dict[str, int] = {}
        for device in cls.parse_pci_gpus(text):
            if device.name is not None:
                found[device.name] = found.get(device.name, 0) + 1
        if not found:
            raise RuntimeError("No supported GPUs detected via PCI sysfs")
        if len(found) > 1:
            names = ", ".join(sorted(found))
            raise RuntimeError(f"Mixed GPU types detected: {names}. All GPUs must be the same type.")
        return next(iter(found.items()))

    @classmethod
    def from_raw(cls, raw_text: str) -> SystemInformation | None:
        """Parse the portable host snapshot into the typed schema."""
        if not raw_text:
            return None

        os_release = _key_values(_section(raw_text, "OS"), "=")
        lscpu = _key_values(_section(raw_text, "CPU"))
        memory = _key_values(_section(raw_text, "MEMORY"))
        docker = _key_values(_section(raw_text, "DOCKER"))
        filesystem = _section(raw_text, "ROOT FILESYSTEM").split()
        gpus = cls._parse_nvidia_gpus(_section(raw_text, "NVIDIA GPUS"))
        pci_gpus = cls.parse_pci_gpus(_section(raw_text, "GPU PCI DEVICES"))
        known_pci_addresses = {(item.pci_bus_id or "").lower()[-12:] for item in gpus}
        for item in pci_gpus:
            if item.pci_bus_id.lower()[-12:] not in known_pci_addresses:
                gpus.append(GpuInformation(index=len(gpus), name=item.name, vendor=item.vendor, pci_bus_id=item.pci_bus_id))

        root_filesystem = None
        if len(filesystem) >= 7:
            root_filesystem = FilesystemInformation(
                device=filesystem[0],
                type=filesystem[1],
                total_bytes=_number(filesystem[2], int),
                used_bytes=_number(filesystem[3], int),
                available_bytes=_number(filesystem[4], int),
                mount=filesystem[-1],
            )

        cuda_driver = _section(raw_text, "NVIDIA DRIVER")
        cuda_match = re.search(r"CUDA Version:\s+([\d.]+)", cuda_driver)
        amd_smi = _section(raw_text, "AMD SMI")
        return cls(
            hostname=_section(raw_text, "HOSTNAME") or None,
            os=OperatingSystemInformation(
                name=os_release.get("PRETTY_NAME"),
                id=os_release.get("ID"),
                version=os_release.get("VERSION_ID"),
                kernel=_section(raw_text, "KERNEL") or None,
            ),
            cpu=CpuInformation(
                model=lscpu.get("Model name"),
                architecture=lscpu.get("Architecture"),
                logical_count=_number(lscpu.get("CPU(s)"), int),
                sockets=_number(lscpu.get("Socket(s)"), int),
                cores_per_socket=_number(lscpu.get("Core(s) per socket"), int),
                threads_per_core=_number(lscpu.get("Thread(s) per core"), int),
                numa_nodes=_number(lscpu.get("NUMA node(s)"), int),
            ),
            memory=MemoryInformation(total_bytes=_number(memory.get("MemTotal"), int)),
            gpus=gpus,
            gpu_pci_devices=pci_gpus,
            software=SoftwareInformation(
                cuda_driver_api=cuda_match.group(1) if cuda_match else None,
                cuda_compiler=_tool_output(raw_text, "CUDA COMPILER"),
                hip_compiler=_tool_output(raw_text, "HIP COMPILER"),
                docker_client=docker.get("ClientVersion"),
                docker_server=docker.get("ServerVersion"),
                docker_os=docker.get("OperatingSystem"),
            ),
            root_filesystem=root_filesystem,
            uptime_seconds=_number(_section(raw_text, "UPTIME SECONDS"), float),
            amd_smi=amd_smi if amd_smi and amd_smi != "N/A" else None,
        )

    @classmethod
    async def retrieve(cls, run_cmd) -> SystemInformation | None:
        """Retrieve and parse generic system information from a host."""
        rc, output, _ = await run_cmd(SYSTEM_INFORMATION_COMMAND, stream=False, timeout=120)
        if rc != 0:
            logger.warning("Failed to retrieve system information")
            return None
        return cls.from_raw(output)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> SystemInformation:
        """Deserialize system information from YAML-safe built-in values."""
        root = value.get("root_filesystem")
        return cls(
            hostname=value.get("hostname"),
            os=OperatingSystemInformation(**value["os"]),
            cpu=CpuInformation(**value["cpu"]),
            memory=MemoryInformation(**value["memory"]),
            gpus=[GpuInformation(**item) for item in value.get("gpus", [])],
            gpu_pci_devices=[PciGpuDevice(**item) for item in value.get("gpu_pci_devices", [])],
            software=SoftwareInformation(**value.get("software", {})),
            root_filesystem=FilesystemInformation(**root) if root else None,
            uptime_seconds=value.get("uptime_seconds"),
            amd_smi=value.get("amd_smi"),
        )
