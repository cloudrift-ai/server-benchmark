"""Query recipes and create onboarding stubs."""

import asyncio
import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import httpx

from emmy.provisioning.errors import TerminalProvisionError
from emmy.recipe.bundled import default_recipe_root
from emmy.recipe.catalog import create_recipe_stub, recipe_inventory, recipe_inventory_document
from emmy.recipe.query import (
    build_query_rows,
    enrich_query_rows,
    parse_predicate,
    parse_sort,
    query_document,
    query_rows,
    referenced_fields,
)
from emmy.redact import register_secret

logger = logging.getLogger(__name__)


def _handle_list(args) -> None:
    try:
        with default_recipe_root() as root:
            if root is None:
                raise ValueError("this Emmy installation does not contain recipes")
            document = recipe_inventory_document(root, tuple(args.tag))
    except (OSError, ValueError) as exc:
        logger.error(str(exc))
        raise SystemExit(2) from exc
    if args.json:
        logger.info(json.dumps(document, sort_keys=True))
        return
    for recipe in document["recipes"]:
        logger.info("%s\t%s\t%s", recipe["model_id"], ",".join(recipe["tags"]) or "-", recipe["path"])


def _handle_create(args) -> None:
    try:
        deployments = [{"deploy.gpu": gpu_name, "deploy.gpu_count": int(gpu_count)} for gpu_name, gpu_count in args.deployment]
        path = create_recipe_stub(args.root, args.model_id, args.rationale, args.task, deployments)
        logger.info(str(path))
    except (OSError, ValueError) as exc:
        logger.error(str(exc))
        raise SystemExit(2) from exc


def _resolve_cloudrift_api_key() -> str:
    api_key = os.environ.get("CLOUDRIFT_API_KEY")
    if not api_key:
        raise ValueError("CLOUDRIFT_API_KEY is required by the requested recipe query fields")
    register_secret(api_key)
    return api_key


def _handle_query(args) -> None:
    try:
        if args.limit is not None and args.limit < 1:
            raise ValueError("limit must be positive")
        filters = [parse_predicate(source) for source in args.filters]
        requirements = [parse_predicate(source) for source in args.requirements]
        sorts = [parse_sort(source) for source in args.sorts]
        fields = referenced_fields(filters, requirements, sorts)
        root_context = nullcontext(args.root) if args.root is not None else default_recipe_root()
        with root_context as root:
            if root is None:
                raise ValueError("this Emmy installation does not contain recipes")
            if not root.is_dir():
                raise ValueError(f"Recipe root is not a directory: {root}")
            inventory = recipe_inventory(root)
        expand_deployments = args.gpu is not None or any(field.startswith("deployment.") for field in fields)
        rows = build_query_rows(
            inventory,
            model_id=args.model,
            allow_missing_model=args.allow_missing_model,
            gpu=args.gpu,
            gpu_count=args.gpu_count,
            expand_deployments=expand_deployments,
        )
        needs_cloudrift = any(field.startswith("provider.cloudrift.") or field.endswith(".availability.cloudrift") for field in fields)
        api_key = _resolve_cloudrift_api_key() if needs_cloudrift else None
        asyncio.run(
            enrich_query_rows(
                rows,
                fields,
                cloudrift_api_key=api_key,
                cloudrift_team_id=os.environ.get("CLOUDRIFT_TEAM_ID"),
            )
        )
        selected = query_rows(
            rows,
            filters=filters,
            requirements=requirements,
            sorts=sorts,
            limit=args.limit,
        )
    except (OSError, ValueError, TerminalProvisionError, httpx.HTTPError) as exc:
        logger.error(str(exc))
        raise SystemExit(2) from exc

    document = query_document(selected)
    if args.json:
        logger.info(json.dumps(document, sort_keys=True))
        return
    for row in document["rows"]:
        deployment = row["deployment"] or {}
        gpu = deployment.get("gpu", "-")
        gpu_count = deployment.get("gpu_count", "-")
        logger.info("%s\t%s\tx%s\t%s", row["model_id"], gpu, gpu_count, row["lifecycle"] or "-")


def register_recipe_command(subparsers) -> None:
    parser = subparsers.add_parser("recipe", help="Inspect recipes and create onboarding stubs")
    actions = parser.add_subparsers(dest="recipe_action", required=True)

    list_parser = actions.add_parser("list", help="List recipe metadata")
    list_parser.add_argument("--tag", action="append", default=[], help="Require a tag; repeat to require multiple tags")
    list_parser.add_argument("--json", action="store_true", help="Print the versioned JSON recipe catalog")
    list_parser.set_defaults(func=_handle_list)

    query_parser = actions.add_parser("query", help="Filter and order normalized recipe or deployment rows")
    query_parser.add_argument(
        "--filter",
        action="append",
        default=[],
        dest="filters",
        metavar="EXPRESSION",
        help="Keep matching rows; repeat for logical AND",
    )
    query_parser.add_argument(
        "--require",
        action="append",
        default=[],
        dest="requirements",
        metavar="EXPRESSION",
        help="Fail if any candidate row does not match; repeat for logical AND",
    )
    query_parser.add_argument(
        "--sort",
        action="append",
        default=[],
        dest="sorts",
        metavar="EXPRESSION",
        help="Add a stable sort key in priority order",
    )
    query_parser.add_argument("--limit", type=int, default=None, help="Return at most this many rows")
    query_parser.add_argument("--model", help="Restrict the candidate source to one exact Hugging Face model ID")
    query_parser.add_argument("--gpu", help="Use this exact deployment GPU for --model")
    query_parser.add_argument("--gpu-count", type=int, help="Use this exact deployment GPU count for --model")
    query_parser.add_argument(
        "--allow-missing-model",
        action="store_true",
        help="Create a synthetic onboarding candidate when --model is absent from the catalog",
    )
    query_parser.add_argument(
        "--root",
        type=Path,
        help="Query this recipe root instead of the installation-aware catalog",
    )
    query_parser.add_argument("--json", action="store_true", help="Print the versioned JSON query result")
    query_parser.set_defaults(func=_handle_query)

    create_parser = actions.add_parser("create", help="Create an onboarding/untested recipe stub")
    create_parser.add_argument("model_id", help="Hugging Face model ID in organization/model form")
    create_parser.add_argument("--root", type=Path, default=Path("recipes"), help="Recipe root (default: recipes)")
    create_parser.add_argument("--task", choices=["generate", "embed"], default="generate")
    create_parser.add_argument("--rationale", required=True, help="Why the model is worth onboarding")
    create_parser.add_argument(
        "--deployment",
        action="append",
        nargs=2,
        required=True,
        metavar=("GPU", "COUNT"),
        help="Candidate deploy.gpu and deploy.gpu_count; repeat for up to three setups",
    )
    create_parser.set_defaults(func=_handle_create)
