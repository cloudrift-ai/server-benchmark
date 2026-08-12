"""List recipes and create onboarding stubs."""

import json
import logging
from pathlib import Path

from emmy.recipe.catalog import create_recipe_stub, recipe_inventory

logger = logging.getLogger(__name__)


def _handle_list(args) -> None:
    try:
        inventory = recipe_inventory(args.root, tuple(args.tag))
    except (OSError, ValueError) as exc:
        logger.error(str(exc))
        raise SystemExit(2) from exc
    if args.json:
        logger.info(json.dumps(inventory, sort_keys=True))
        return
    for recipe in inventory:
        logger.info("%s\t%s\t%s", recipe["model_id"], ",".join(recipe["tags"]) or "-", recipe["path"])


def _handle_create(args) -> None:
    try:
        deployments = [{"deploy.gpu": gpu_name, "deploy.gpu_count": int(gpu_count)} for gpu_name, gpu_count in args.deployment]
        path = create_recipe_stub(args.root, args.model_id, args.rationale, args.task, deployments)
        if args.json:
            created = next(recipe for recipe in recipe_inventory(args.root) if recipe["model_id"] == args.model_id)
            logger.info(json.dumps(created, sort_keys=True))
        else:
            logger.info(str(path))
    except (OSError, ValueError) as exc:
        logger.error(str(exc))
        raise SystemExit(2) from exc


def register_recipe_command(subparsers) -> None:
    parser = subparsers.add_parser("recipe", help="Inspect recipes and create onboarding stubs")
    actions = parser.add_subparsers(dest="recipe_action", required=True)

    list_parser = actions.add_parser("list", help="List recipe metadata")
    list_parser.add_argument("root", nargs="?", type=Path, default=Path("recipes"), help="Recipe root (default: recipes)")
    list_parser.add_argument("--tag", action="append", default=[], help="Require a tag; repeat to require multiple tags")
    list_parser.add_argument("--json", action="store_true", help="Print a JSON array of recipe metadata")
    list_parser.set_defaults(func=_handle_list)

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
    create_parser.add_argument("--json", action="store_true", help="Print the created recipe metadata as JSON")
    create_parser.set_defaults(func=_handle_create)
