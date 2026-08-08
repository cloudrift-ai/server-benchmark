#!/usr/bin/env python3
"""Provision and clean up the single VM owned by a model-onboarding run."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from emmy.hardware import GPU_INSTANCE_TYPES
from emmy.provisioning import cloudrift as cloudrift_provider
from emmy.provisioning import gcp as gcp_provider
from emmy.provisioning.candidates import iter_candidates
from emmy.provisioning.cloud import provision_cloud_vm

TERMINAL_CLOUDRIFT_STATES = {"Deleted", "Inactive", "Terminated"}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_github_outputs(values: dict[str, object]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a") as output:
        for key, value in values.items():
            output.write(f"{key}={value}\n")


def _candidate_gpu_count(provider: str, instance_type: str) -> int:
    if provider == "cloudrift":
        return int(instance_type.rsplit(".", 1)[1])
    if instance_type.startswith("g4-standard-"):
        return int(instance_type.rsplit("-", 1)[1]) // 48
    return int(instance_type.rsplit("-", 1)[1].removesuffix("g"))


def _exact_providers(gpu: str, gpu_count: int) -> list[str]:
    if gpu_count < 1:
        raise ValueError("GPU count must be positive")
    candidates = iter_candidates(gpu, gpu_count, None)
    exact = {
        candidate.provider for candidate in candidates if _candidate_gpu_count(candidate.provider, candidate.instance_type) == gpu_count
    }
    provider_order = [provider for provider, _ in GPU_INSTANCE_TYPES[gpu]]
    return list(dict.fromkeys(provider for provider in provider_order if provider in exact))


def _provisional_lease(args: argparse.Namespace, provider: str, instance_id: str, zone: str | None = None) -> None:
    """Persist the provider handle as soon as allocation starts, before readiness polling."""
    _write_json(
        args.lease,
        {
            "schema_version": 1,
            "owner": {"repository": args.repository, "run_id": args.run_id},
            "request": {"gpu": args.gpu, "gpu_count": args.gpu_count},
            "vm": {"provider": provider, "instance_id": instance_id, "zone": zone},
            "status": "provisioning",
        },
    )


async def _provision_with_early_lease(args: argparse.Namespace, provider: str):
    """Provision through Emmy while intercepting the first provider-owned handle."""

    async def refuse_active_overwrite() -> None:
        if not args.lease.exists():
            return
        provisional = _load_owned_lease(args.lease, args.repository, args.run_id)
        if provisional["vm"]["provider"] != provider or not await _is_active(provisional):
            return
        instance_id = provisional["vm"]["instance_id"]
        raise RuntimeError(
            f"Refusing to overwrite active provisional {provider} VM {instance_id}; "
            "the always-cleanup step must retain and delete this handle"
        )

    if provider == "cloudrift":
        original_rent = cloudrift_provider._rent_instance

        async def rent_and_record(*rent_args, **rent_kwargs):
            await refuse_active_overwrite()
            result = await original_rent(*rent_args, **rent_kwargs)
            instance_ids = (result or {}).get("instance_ids", [])
            if instance_ids:
                _provisional_lease(args, "cloudrift", instance_ids[0])
            return result

        cloudrift_provider._rent_instance = rent_and_record
        try:
            return await provision_cloud_vm(
                gpu_name=args.gpu,
                gpu_count=args.gpu_count,
                ssh_key=args.ssh_key,
                server_name=f"onboard-{args.run_id}",
                provider=provider,
            )
        finally:
            cloudrift_provider._rent_instance = original_rent

    original_create = gcp_provider.create_instance

    async def create_and_record(*create_args, **create_kwargs):
        await refuse_active_overwrite()
        instance = create_kwargs.get("instance", create_args[0] if create_args else None)
        zone = create_kwargs.get("zone", create_args[1] if len(create_args) > 1 else None)
        _provisional_lease(args, "gcp", instance, zone)
        return await original_create(*create_args, **create_kwargs)

    gcp_provider.create_instance = create_and_record
    try:
        return await provision_cloud_vm(
            gpu_name=args.gpu,
            gpu_count=args.gpu_count,
            ssh_key=args.ssh_key,
            server_name=f"onboard-{args.run_id}",
            provider=provider,
        )
    finally:
        gcp_provider.create_instance = original_create


def _gcp_available() -> bool:
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials and Path(credentials).is_file():
        return True
    try:
        result = subprocess.run(
            ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


async def _provision(args: argparse.Namespace) -> int:
    providers = _exact_providers(args.gpu, args.gpu_count)
    if not providers:
        raise RuntimeError(f"No provider offers exactly {args.gpu_count} x {args.gpu}")

    configured = []
    if "cloudrift" in providers and os.environ.get("CLOUDRIFT_API_KEY"):
        configured.append("cloudrift")
    if "gcp" in providers and _gcp_available():
        configured.append("gcp")
    if not configured:
        raise RuntimeError(
            f"No configured provider can rent exactly {args.gpu_count} x {args.gpu}; "
            "CLOUDRIFT_API_KEY or active GCP credentials are required"
        )

    errors = []
    connection = None
    for provider in providers:
        if provider not in configured:
            continue
        try:
            connection = await _provision_with_early_lease(args, provider)
        except Exception as exc:  # keep the second provider available after a provider-specific failure
            errors.append(f"{provider}: {exc}")
            if args.lease.exists():
                provisional = _load_owned_lease(args.lease, args.repository, args.run_id)
                if provisional["vm"]["provider"] == provider and await _is_active(provisional):
                    instance_id = provisional["vm"]["instance_id"]
                    raise RuntimeError(
                        f"Refusing provider fallback because {provider} VM {instance_id} remains active; "
                        "the always-cleanup step must retain and delete this handle"
                    ) from exc
            continue
        if connection is not None:
            break
        errors.append(f"{provider}: no exact capacity")
        if args.lease.exists():
            provisional = _load_owned_lease(args.lease, args.repository, args.run_id)
            if provisional["vm"]["provider"] == provider and await _is_active(provisional):
                instance_id = provisional["vm"]["instance_id"]
                raise RuntimeError(
                    f"Refusing provider fallback because {provider} VM {instance_id} remains active; "
                    "the always-cleanup step must retain and delete this handle"
                )

    if connection is None:
        detail = "; ".join(errors) if errors else "no configured exact candidate"
        raise RuntimeError(f"No exact {args.gpu_count} x {args.gpu} VM is available on CloudRift or GCP ({detail})")

    provider = connection.delete_info[0]
    instance_id = connection.delete_info[1]
    zone = connection.delete_info[2] if provider == "gcp" else None
    ssh_user = connection.username or os.environ.get("USER", "deploy")
    ssh_target = f"{ssh_user}@{connection.host}"
    if connection.ssh_port != 22:
        ssh_target = f"{ssh_target}:{connection.ssh_port}"
    lease = {
        "schema_version": 1,
        "owner": {
            "repository": args.repository,
            "run_id": args.run_id,
        },
        "request": {
            "gpu": args.gpu,
            "gpu_count": args.gpu_count,
        },
        "vm": {
            "provider": provider,
            "instance_id": instance_id,
            "zone": zone,
            "ssh_host": connection.host,
            "ssh_port": connection.ssh_port,
            "ssh_user": ssh_user,
            "ssh_target": ssh_target,
        },
        "status": "active",
    }
    _write_json(args.lease, lease)
    _write_github_outputs(
        {
            "provider": provider,
            "instance_id": instance_id,
            "zone": zone or "",
            "ssh_host": connection.host,
            "ssh_port": connection.ssh_port,
            "ssh_user": ssh_user,
            "ssh_target": ssh_target,
            "lease": args.lease,
        }
    )
    print(json.dumps(lease, sort_keys=True))
    return 0


def _load_owned_lease(path: Path, repository: str, run_id: str) -> dict:
    lease = json.loads(path.read_text())
    owner = lease.get("owner", {})
    if owner.get("repository") != repository or owner.get("run_id") != run_id:
        raise RuntimeError(
            f"Refusing VM operation: {path} belongs to {owner.get('repository')}/{owner.get('run_id')}, not {repository}/{run_id}"
        )
    vm = lease.get("vm", {})
    if vm.get("provider") not in {"cloudrift", "gcp"} or not vm.get("instance_id"):
        raise RuntimeError(f"Invalid VM lease: {path}")
    return lease


async def _is_active(lease: dict) -> bool:
    vm = lease["vm"]
    if vm["provider"] == "cloudrift":
        api_key = os.environ.get("CLOUDRIFT_API_KEY")
        if not api_key:
            raise RuntimeError("CLOUDRIFT_API_KEY is required to audit the workflow-owned CloudRift VM")
        info = await cloudrift_provider._get_instance_info(api_key, vm["instance_id"])
        if info is None:
            return False
        return info.get("status") not in TERMINAL_CLOUDRIFT_STATES

    zone = vm.get("zone")
    if not zone:
        raise RuntimeError("GCP lease is missing its zone")
    result = subprocess.run(
        ["gcloud", "compute", "instances", "describe", vm["instance_id"], "--zone", zone, "--format=value(status)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if "not found" in combined or "was not found" in combined:
        return False
    raise RuntimeError(f"Could not audit GCP VM {vm['instance_id']} in {zone}: {result.stderr.strip()}")


async def _delete_once(lease: dict) -> bool:
    vm = lease["vm"]
    if vm["provider"] == "cloudrift":
        api_key = os.environ.get("CLOUDRIFT_API_KEY")
        if not api_key:
            raise RuntimeError("CLOUDRIFT_API_KEY is required to delete the workflow-owned CloudRift VM")
        return await cloudrift_provider.delete_instance(api_key, vm["instance_id"])
    return await gcp_provider.delete_instance(vm["instance_id"], vm["zone"])


async def _delete(args: argparse.Namespace) -> int:
    if not args.lease.exists():
        print(f"No lease at {args.lease}; no workflow-owned VM to delete")
        return 0
    lease = _load_owned_lease(args.lease, args.repository, args.run_id)
    if not await _is_active(lease):
        lease["status"] = "deleted"
        _write_json(args.lease, lease)
        return 0

    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            await _delete_once(lease)
            for _ in range(args.audit_attempts):
                if not await _is_active(lease):
                    lease["status"] = "deleted"
                    _write_json(args.lease, lease)
                    return 0
                await asyncio.sleep(args.audit_delay)
            last_error = RuntimeError("provider still reports the VM as active")
        except Exception as exc:
            last_error = exc
        if attempt < args.retries:
            await asyncio.sleep(args.retry_delay)
    raise RuntimeError(f"Workflow-owned VM remains after {args.retries} cleanup attempts: {last_error}")


async def _audit(args: argparse.Namespace) -> int:
    if not args.lease.exists():
        print(f"No lease at {args.lease}; no workflow-owned VM to audit")
        return 0
    lease = _load_owned_lease(args.lease, args.repository, args.run_id)
    if await _is_active(lease):
        vm = lease["vm"]
        raise RuntimeError(f"Workflow-owned {vm['provider']} VM {vm['instance_id']} is still active")
    print("Workflow-owned VM is absent or terminal")
    return 0


def _common_owner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    provision = subparsers.add_parser("provision", help="Rent one exact GPU VM and write its ownership lease")
    _common_owner_args(provision)
    provision.add_argument("--gpu", required=True)
    provision.add_argument("--gpu-count", type=int, required=True)
    provision.add_argument("--ssh-key", required=True)

    delete = subparsers.add_parser("delete", help="Delete only the VM identified by an owned lease")
    _common_owner_args(delete)
    delete.add_argument("--retries", type=int, default=3)
    delete.add_argument("--retry-delay", type=float, default=10)
    delete.add_argument("--audit-attempts", type=int, default=6)
    delete.add_argument("--audit-delay", type=float, default=10)

    audit = subparsers.add_parser("audit", help="Fail if the VM identified by an owned lease is still active")
    _common_owner_args(audit)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "provision":
            return asyncio.run(_provision(args))
        if args.command == "delete":
            return asyncio.run(_delete(args))
        return asyncio.run(_audit(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
