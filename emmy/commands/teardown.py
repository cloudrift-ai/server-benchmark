"""Clean up VMs retained by experiment records."""

import asyncio
import logging
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from emmy.benchmark.record import utc_timestamp, write_record_path
from emmy.deploy.orchestrate import run_teardown
from emmy.provisioning.cloud import delete_cloud_vm
from emmy.provisioning.ssh_transport import make_run_cmd

logger = logging.getLogger(__name__)


def _load_active_instances(directory: Path) -> dict[tuple, list[tuple[Path, dict]]]:
    """Group active infrastructure handles from row records."""
    paths = [*directory.glob("*.experiment.yaml"), *(directory / "results").glob("*.experiment.yaml")]
    instances: dict[tuple, list[tuple[Path, dict]]] = defaultdict(list)
    for path in sorted(set(paths)):
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
        infrastructure = (record.get("execution") or {}).get("infrastructure") or {}
        if infrastructure.get("state") != "active":
            continue
        key = (
            infrastructure.get("provider"),
            infrastructure.get("instance_id"),
            infrastructure.get("zone"),
            infrastructure.get("address"),
            infrastructure.get("ssh_port", 22),
        )
        instances[key].append((path, record))
    return instances


def handle_teardown(args):
    """Handle the teardown command."""
    asyncio.run(_handle_teardown(args))


async def _handle_teardown(args):
    experiment_dir = Path(args.experiment_dir)
    ssh_key = args.ssh_key
    instances = _load_active_instances(experiment_dir)
    if not instances:
        logger.error(f"No active instances found in experiment records under {experiment_dir}")
        sys.exit(1)

    logger.info(f"Tearing down {len(instances)} instance(s) from {experiment_dir}")
    errors = []
    for key, records in instances.items():
        provider, instance_id, zone, address, ssh_port = key
        infrastructure = records[0][1]["execution"]["infrastructure"]
        label = infrastructure.get("group", "unknown")
        logger.info(f"[{label}] {address} ({provider}: {instance_id})")

        if address:
            logger.info(f"  Stopping containers on {address}...")
            run_cmd = make_run_cmd(address, ssh_key, ssh_port)
            await run_teardown(run_cmd)

        if provider and instance_id:
            logger.info(f"  Deleting VM ({provider}: {instance_id})...")
            try:
                if provider == "gcp":
                    if not zone:
                        raise ValueError(f"missing zone for GCP instance {instance_id}")
                    delete_info = (provider, instance_id, zone)
                else:
                    delete_info = (provider, instance_id)
                deleted = await delete_cloud_vm(delete_info)
                if deleted is False:
                    raise RuntimeError("provider reported that VM deletion failed")
            except Exception as exc:
                logger.error(f"  ERROR deleting VM: {exc}")
                errors.append(label)
                continue
            logger.info("  VM deleted.")

        for path, record in records:
            record["execution"]["infrastructure"]["state"] = "deleted"
            record["execution"]["infrastructure"]["deleted_at"] = utc_timestamp()
            write_record_path(path, record)

    if errors:
        logger.info(f"Failed to clean up {len(errors)} instance(s): {', '.join(errors)}")
        sys.exit(1)
    logger.info("All retained instances cleaned up; experiment records updated.")


def register_teardown_command(subparsers):
    """Register the teardown subcommand."""
    parser = subparsers.add_parser(
        "teardown",
        help="Tear down VMs retained by 'bench --no-teardown'",
    )
    parser.add_argument(
        "experiment_dir",
        help="Experiment directory containing active experiment records",
    )
    parser.add_argument(
        "--ssh-key",
        default="~/.ssh/id_ed25519",
        help="SSH private key path (default: ~/.ssh/id_ed25519)",
    )
    parser.set_defaults(func=handle_teardown)
