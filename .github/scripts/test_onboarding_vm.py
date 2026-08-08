import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("onboarding_vm.py")
SPEC = importlib.util.spec_from_file_location("onboarding_vm", MODULE_PATH)
onboarding_vm = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(onboarding_vm)


@pytest.mark.parametrize(
    ("provider", "instance_type", "expected"),
    [
        ("cloudrift", "h200-8-generic.4", 4),
        ("gcp", "a3-ultragpu-8g", 8),
        ("gcp", "g4-standard-192", 4),
    ],
)
def test_candidate_gpu_count(provider, instance_type, expected):
    assert onboarding_vm._candidate_gpu_count(provider, instance_type) == expected


def test_exact_providers_rejects_gcp_overallocation():
    assert onboarding_vm._exact_providers("NVIDIA B200", 1) == []
    assert onboarding_vm._exact_providers("NVIDIA B200", 8) == ["gcp"]


def test_gcp_availability_is_false_without_cli_or_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    def missing(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(onboarding_vm.subprocess, "run", missing)
    assert onboarding_vm._gcp_available() is False


def test_load_owned_lease_rejects_another_run(tmp_path):
    lease = tmp_path / "lease.json"
    onboarding_vm._write_json(
        lease,
        {
            "owner": {"repository": "cloudrift-ai/emmy", "run_id": "123-1"},
            "vm": {"provider": "cloudrift", "instance_id": "owned-id"},
        },
    )

    with pytest.raises(RuntimeError, match="Refusing VM operation"):
        onboarding_vm._load_owned_lease(lease, "cloudrift-ai/emmy", "456-1")


def test_provisional_lease_records_ownership_before_ssh_is_known(tmp_path):
    lease = tmp_path / "lease.json"
    args = type(
        "Args",
        (),
        {
            "lease": lease,
            "repository": "cloudrift-ai/emmy",
            "run_id": "123-1",
            "gpu": "NVIDIA H200 141GB",
            "gpu_count": 1,
        },
    )()

    onboarding_vm._provisional_lease(args, "gcp", "onboard-123", "us-west1-c")

    value = onboarding_vm.json.loads(lease.read_text())
    assert value["owner"] == {"repository": "cloudrift-ai/emmy", "run_id": "123-1"}
    assert value["vm"] == {"provider": "gcp", "instance_id": "onboard-123", "zone": "us-west1-c"}
    assert value["status"] == "provisioning"


@pytest.mark.asyncio
async def test_delete_is_idempotent_when_owned_vm_is_absent(monkeypatch, tmp_path):
    lease_path = tmp_path / "lease.json"
    onboarding_vm._write_json(
        lease_path,
        {
            "owner": {"repository": "cloudrift-ai/emmy", "run_id": "123-1"},
            "vm": {"provider": "cloudrift", "instance_id": "owned-id"},
            "status": "active",
        },
    )

    async def inactive(_lease):
        return False

    monkeypatch.setattr(onboarding_vm, "_is_active", inactive)
    args = type("Args", (), {"lease": lease_path, "repository": "cloudrift-ai/emmy", "run_id": "123-1"})()

    assert await onboarding_vm._delete(args) == 0
    assert onboarding_vm.json.loads(lease_path.read_text())["status"] == "deleted"


@pytest.mark.asyncio
async def test_provision_does_not_overwrite_active_failed_provider_handle(monkeypatch, tmp_path):
    lease_path = tmp_path / "lease.json"
    args = type(
        "Args",
        (),
        {
            "gpu": "NVIDIA H200 141GB",
            "gpu_count": 1,
            "ssh_key": "/tmp/key",
            "repository": "cloudrift-ai/emmy",
            "run_id": "123-1",
            "lease": lease_path,
        },
    )()
    monkeypatch.setenv("CLOUDRIFT_API_KEY", "configured")
    monkeypatch.setattr(onboarding_vm, "_gcp_available", lambda: False)
    attempts = []

    async def failed_provider(provision_args, provider):
        attempts.append(provider)
        onboarding_vm._provisional_lease(provision_args, provider, "still-active")
        raise RuntimeError("orphan deletion failed")

    async def active(_lease):
        return True

    monkeypatch.setattr(onboarding_vm, "_provision_with_early_lease", failed_provider)
    monkeypatch.setattr(onboarding_vm, "_is_active", active)

    with pytest.raises(RuntimeError, match="Refusing provider fallback"):
        await onboarding_vm._provision(args)
    assert attempts == ["cloudrift"]
    assert onboarding_vm.json.loads(lease_path.read_text())["vm"]["instance_id"] == "still-active"


@pytest.mark.asyncio
async def test_provider_retry_does_not_overwrite_active_provisional_handle(monkeypatch, tmp_path):
    lease_path = tmp_path / "lease.json"
    args = type(
        "Args",
        (),
        {
            "gpu": "NVIDIA H200 141GB",
            "gpu_count": 1,
            "ssh_key": "/tmp/key",
            "repository": "cloudrift-ai/emmy",
            "run_id": "123-1",
            "lease": lease_path,
        },
    )()
    rent_calls = []

    async def rent(*_args, **_kwargs):
        rent_calls.append(True)
        return {"instance_ids": [f"instance-{len(rent_calls)}"]}

    async def two_rent_attempts(**_kwargs):
        await onboarding_vm.cloudrift_provider._rent_instance()
        await onboarding_vm.cloudrift_provider._rent_instance()

    async def active(_lease):
        return True

    monkeypatch.setattr(onboarding_vm.cloudrift_provider, "_rent_instance", rent)
    monkeypatch.setattr(onboarding_vm, "provision_cloud_vm", two_rent_attempts)
    monkeypatch.setattr(onboarding_vm, "_is_active", active)

    with pytest.raises(RuntimeError, match="Refusing to overwrite active provisional"):
        await onboarding_vm._provision_with_early_lease(args, "cloudrift")
    assert len(rent_calls) == 1
    assert onboarding_vm.json.loads(lease_path.read_text())["vm"]["instance_id"] == "instance-1"
