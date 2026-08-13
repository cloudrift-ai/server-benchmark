"""Durable ownership state for interrupt-safe cloud VM lifecycles."""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from emmy.provisioning import cloudrift as cloudrift_provider
from emmy.provisioning import gcp as gcp_provider
from emmy.provisioning.cloud import delete_cloud_vm
from emmy.provisioning.errors import TerminalProvisionError
from emmy.provisioning.types import VMConnectionInfo

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _write_lease(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_owned_lease(path: Path, owner: str) -> dict:
    """Load and validate a VM lease belonging to one exact owner."""
    lease = json.loads(path.read_text())
    if lease.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported VM lease schema at {path}")
    if lease.get("owner") != owner:
        raise RuntimeError(f"Refusing VM operation: {path} belongs to {lease.get('owner')}, not {owner}")
    vm = lease.get("vm", {})
    if vm.get("provider") not in {"cloudrift", "gcp"} or not vm.get("instance_id"):
        raise RuntimeError(f"Invalid VM lease: {path}")
    if vm["provider"] == "gcp" and not vm.get("zone"):
        raise RuntimeError(f"GCP VM lease is missing its zone: {path}")
    return lease


def _delete_info(lease: dict) -> tuple:
    vm = lease["vm"]
    if vm["provider"] == "cloudrift":
        return "cloudrift", vm["instance_id"]
    return "gcp", vm["instance_id"], vm["zone"]


async def lease_is_active(lease: dict) -> bool:
    """Return whether the provider still reports the leased VM as active."""
    vm = lease["vm"]
    if vm["provider"] == "cloudrift":
        api_key = os.environ.get("CLOUDRIFT_API_KEY")
        if not api_key:
            raise RuntimeError("CLOUDRIFT_API_KEY is required to audit the leased CloudRift VM")
        return await cloudrift_provider.instance_is_active(api_key, vm["instance_id"])
    return await gcp_provider.instance_is_active(vm["instance_id"], vm["zone"])


@dataclass(frozen=True)
class VmLeaseObserver:
    """Persist allocation handles before readiness polling can be interrupted."""

    path: Path
    owner: str
    gpu: str
    gpu_count: int

    async def before_allocate(self, provider: str) -> None:
        if not self.path.exists():
            return
        try:
            lease = load_owned_lease(self.path, self.owner)
            active = lease.get("status") != "deleted" and await lease_is_active(lease)
        except Exception as exc:
            raise TerminalProvisionError(f"Cannot safely replace VM lease {self.path}: {exc}") from exc
        if active:
            vm = lease["vm"]
            raise TerminalProvisionError(
                f"Refusing to allocate through {provider} while leased {vm['provider']} VM {vm['instance_id']} remains active"
            )

    async def allocated(self, delete_info: tuple) -> None:
        provider = delete_info[0]
        await self.before_allocate(provider)
        vm = {"provider": provider, "instance_id": delete_info[1]}
        if provider == "gcp":
            vm["zone"] = delete_info[2]
        _write_lease(
            self.path,
            {
                "schema_version": SCHEMA_VERSION,
                "owner": self.owner,
                "request": {"gpu": self.gpu, "gpu_count": self.gpu_count},
                "vm": vm,
                "status": "provisioning",
            },
        )

    async def ready(self, connection: VMConnectionInfo) -> None:
        lease = load_owned_lease(self.path, self.owner)
        if _delete_info(lease) != connection.delete_info:
            raise RuntimeError("Ready VM deletion handle does not match its persisted lease")
        ssh_target = connection.address
        if connection.ssh_port != 22:
            ssh_target = f"{ssh_target}:{connection.ssh_port}"
        lease["vm"].update(
            {
                "ssh_host": connection.host,
                "ssh_port": connection.ssh_port,
                "ssh_user": connection.username,
                "ssh_target": ssh_target,
            }
        )
        lease["status"] = "active"
        _write_lease(self.path, lease)


async def delete_owned_vm(
    path: Path,
    owner: str,
    *,
    retries: int = 3,
    retry_delay: float = 10,
    audit_attempts: int = 6,
    audit_delay: float = 10,
) -> None:
    """Delete only the VM named by an owned lease and verify it is gone."""
    if not path.exists():
        logger.info(f"No lease at {path}; no VM to delete")
        return
    lease = load_owned_lease(path, owner)
    if not await lease_is_active(lease):
        lease["status"] = "deleted"
        _write_lease(path, lease)
        return

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if not await delete_cloud_vm(_delete_info(lease)):
                raise RuntimeError("provider rejected the VM deletion request")
            for _ in range(audit_attempts):
                if not await lease_is_active(lease):
                    lease["status"] = "deleted"
                    _write_lease(path, lease)
                    return
                await asyncio.sleep(audit_delay)
            last_error = RuntimeError("provider still reports the VM as active")
        except Exception as exc:
            last_error = exc
        if attempt < retries:
            await asyncio.sleep(retry_delay)
    raise RuntimeError(f"Leased VM remains after {retries} cleanup attempts: {last_error}")


async def audit_owned_vm(path: Path, owner: str) -> None:
    """Fail if the VM named by an owned lease is still active."""
    if not path.exists():
        logger.info(f"No lease at {path}; no VM to audit")
        return
    lease = load_owned_lease(path, owner)
    if await lease_is_active(lease):
        vm = lease["vm"]
        raise RuntimeError(f"Leased {vm['provider']} VM {vm['instance_id']} is still active")
    logger.info("Leased VM is absent or terminal")
