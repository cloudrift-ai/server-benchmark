"""Tests for interrupt-safe VM allocation leases."""

import json
from unittest.mock import AsyncMock

import pytest

from emmy.provisioning import cloudrift, gcp, lease
from emmy.provisioning.errors import TerminalProvisionError
from emmy.provisioning.types import VMConnectionInfo


def _payload(owner="run-123", status="active"):
    return {
        "schema_version": 1,
        "owner": owner,
        "request": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1},
        "vm": {"provider": "cloudrift", "instance_id": "owned-id"},
        "status": status,
    }


def test_load_owned_lease_rejects_another_owner(tmp_path):
    path = tmp_path / "lease.json"
    lease._write_lease(path, _payload())

    with pytest.raises(RuntimeError, match="Refusing VM operation"):
        lease.load_owned_lease(path, "another-run")


async def test_observer_records_allocation_before_connection_is_ready(tmp_path):
    path = tmp_path / "lease.json"
    observer = lease.VmLeaseObserver(path, "run-123", "NVIDIA H200 141GB", 1)

    await observer.allocated(("gcp", "onboard-123", "us-west1-c"))

    value = json.loads(path.read_text())
    assert value["owner"] == "run-123"
    assert value["vm"] == {"provider": "gcp", "instance_id": "onboard-123", "zone": "us-west1-c"}
    assert value["status"] == "provisioning"


async def test_observer_marks_ready_connection(tmp_path):
    path = tmp_path / "lease.json"
    observer = lease.VmLeaseObserver(path, "run-123", "NVIDIA H200 141GB", 1)
    await observer.allocated(("cloudrift", "owned-id"))

    await observer.ready(
        VMConnectionInfo(
            host="example.test",
            username="riftuser",
            ssh_port=2222,
            delete_info=("cloudrift", "owned-id"),
        )
    )

    value = json.loads(path.read_text())
    assert value["status"] == "active"
    assert value["vm"]["ssh_target"] == "riftuser@example.test:2222"


async def test_observer_refuses_to_overwrite_active_handle(monkeypatch, tmp_path):
    path = tmp_path / "lease.json"
    lease._write_lease(path, _payload())
    observer = lease.VmLeaseObserver(path, "run-123", "NVIDIA H200 141GB", 1)
    monkeypatch.setattr(lease, "lease_is_active", AsyncMock(return_value=True))

    with pytest.raises(TerminalProvisionError, match="remains active"):
        await observer.before_allocate("gcp")


async def test_delete_is_idempotent_when_owned_vm_is_absent(monkeypatch, tmp_path):
    path = tmp_path / "lease.json"
    lease._write_lease(path, _payload())
    monkeypatch.setattr(lease, "lease_is_active", AsyncMock(return_value=False))

    await lease.delete_owned_vm(path, "run-123")

    assert json.loads(path.read_text())["status"] == "deleted"


async def test_delete_waits_for_provider_absence(monkeypatch, tmp_path):
    path = tmp_path / "lease.json"
    lease._write_lease(path, _payload())
    monkeypatch.setattr(lease, "lease_is_active", AsyncMock(side_effect=[True, False]))
    delete = AsyncMock(return_value=True)
    monkeypatch.setattr(lease, "delete_cloud_vm", delete)

    await lease.delete_owned_vm(path, "run-123", audit_delay=0)

    delete.assert_awaited_once_with(("cloudrift", "owned-id"))
    assert json.loads(path.read_text())["status"] == "deleted"


async def test_cloudrift_notifies_observer_after_rent(monkeypatch, tmp_path):
    key = tmp_path / "id.pub"
    key.write_text("ssh-ed25519 AAAA test")
    monkeypatch.setattr(cloudrift, "_rent_instance", AsyncMock(return_value={"instance_ids": ["instance-1"]}))
    monkeypatch.setattr(cloudrift, "wait_for_status", AsyncMock(return_value={"id": "instance-1", "status": "Active"}))
    monkeypatch.setattr(
        cloudrift,
        "_extract_connection_info",
        lambda _info, delete_info=(): VMConnectionInfo("host", "riftuser", delete_info=delete_info),
    )
    monkeypatch.setattr(cloudrift, "_log_connection_info", lambda _info: None)
    observer = AsyncMock()

    connection = await cloudrift.create_instance("key", "h200.1", key, allocation_observer=observer)

    observer.allocated.assert_awaited_once_with(("cloudrift", "instance-1"))
    assert connection.delete_info == ("cloudrift", "instance-1")


async def test_gcp_notifies_observer_after_create(monkeypatch):
    run = AsyncMock(side_effect=[(0, "", ""), (0, "RUNNING\n", ""), (0, "192.0.2.1\n", "")])
    monkeypatch.setattr(gcp, "run_shell_cmd", run)
    observer = AsyncMock()

    connection = await gcp.create_instance("instance-1", "us-west1-c", "a3-highgpu-1g", allocation_observer=observer)

    observer.allocated.assert_awaited_once_with(("gcp", "instance-1", "us-west1-c"))
    assert connection.delete_info == ("gcp", "instance-1", "us-west1-c")


async def test_gcp_audit_accepts_not_found_on_stdout(monkeypatch):
    monkeypatch.setattr(gcp, "run_shell_cmd", AsyncMock(return_value=(1, "instance was not found", "")))

    assert not await gcp.instance_is_active("instance-1", "us-west1-c")


def test_vm_lease_cli_is_idempotent_without_a_lease(run_cli, tmp_path):
    path = tmp_path / "missing.json"

    delete_code, _, _ = run_cli("vm", "delete", "lease", str(path), "--owner", "run-123")
    audit_code, _, _ = run_cli("vm", "audit", "lease", str(path), "--owner", "run-123")

    assert delete_code == 0
    assert audit_code == 0


def test_vm_create_exact_count_rejects_overallocation(run_cli):
    returncode, stdout, _ = run_cli(
        "vm",
        "create",
        "gpu",
        "--gpu",
        "NVIDIA B200",
        "--gpu-count",
        "1",
        "--exact-gpu-count",
        "--dry-run",
    )

    assert returncode == 1
    assert "No provider offers exactly 1 x NVIDIA B200" in stdout
