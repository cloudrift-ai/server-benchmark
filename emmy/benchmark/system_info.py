"""Collect a parseable machine snapshot from an experiment host."""

import logging

logger = logging.getLogger(__name__)

SYSTEM_INFO_CMD = r"""
echo "=== HOSTNAME ==="
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
for device in /sys/bus/pci/devices/*; do
  vendor=$(cat "$device/vendor" 2>/dev/null) || continue
  [ "$vendor" = "0x10de" ] || [ "$vendor" = "0x1002" ] || continue
  printf '%s,%s,%s\n' "$(basename "$device")" "$vendor" "$(cat "$device/device")"
done

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


async def collect_system_info(run_cmd) -> str:
    """Collect machine information from an experiment host.

    Returns the output wrapped in section delimiters, or empty string on failure.
    """
    rc, output, _ = await run_cmd(SYSTEM_INFO_CMD, stream=False, timeout=120)
    if rc != 0:
        logger.warning("Failed to collect system info")
        return ""
    return output
