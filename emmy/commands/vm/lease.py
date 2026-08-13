"""CLI handlers for VM lifecycle operations through an owned lease."""

import asyncio
import logging
import sys
from pathlib import Path

from emmy.provisioning.lease import audit_owned_vm, delete_owned_vm

logger = logging.getLogger(__name__)


def _run(coro) -> None:
    try:
        asyncio.run(coro)
    except Exception as exc:
        logger.error(str(exc))
        sys.exit(1)


def _handle_delete(args) -> None:
    _run(
        delete_owned_vm(
            args.lease,
            args.owner,
            retries=args.retries,
            retry_delay=args.retry_delay,
            audit_attempts=args.audit_attempts,
            audit_delay=args.audit_delay,
        )
    )


def _handle_audit(args) -> None:
    _run(audit_owned_vm(args.lease, args.owner))


def _owner_args(parser) -> None:
    parser.add_argument("lease", type=Path, help="VM lease written by `emmy vm create gpu --lease`")
    parser.add_argument("--owner", required=True, help="Exact owner recorded in the lease")


def register_delete_target(subparsers) -> None:
    """Register `vm delete lease`."""
    parser = subparsers.add_parser("lease", help="Delete and verify the VM identified by an owned lease")
    _owner_args(parser)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=10)
    parser.add_argument("--audit-attempts", type=int, default=6)
    parser.add_argument("--audit-delay", type=float, default=10)
    parser.set_defaults(func=_handle_delete)


def register_audit_target(subparsers) -> None:
    """Register `vm audit lease`."""
    parser = subparsers.add_parser("lease", help="Fail if the VM identified by an owned lease remains active")
    _owner_args(parser)
    parser.set_defaults(func=_handle_audit)
