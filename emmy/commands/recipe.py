"""List recipes and create onboarding stubs."""

import json
import logging
from pathlib import Path

from emmy.recipe.bundled import bundled_root, default_recipe_root
from emmy.recipe.catalog import create_recipe_stub, recipe_inventory_document

logger = logging.getLogger(__name__)


def _handle_list(args) -> None:
    try:
        if args.bundled and args.root is not None:
            raise ValueError("recipe list accepts either ROOT or --bundled, not both")
        if args.root is not None:
            document = recipe_inventory_document(args.root, tuple(args.tag))
        else:
            root_context = bundled_root() if args.bundled else default_recipe_root()
            with root_context as root:
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


def register_recipe_command(subparsers) -> None:
    parser = subparsers.add_parser("recipe", help="Inspect recipes and create onboarding stubs")
    actions = parser.add_subparsers(dest="recipe_action", required=True)

    list_parser = actions.add_parser("list", help="List recipe metadata")
    list_parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        help="Recipe root (default: editable checkout recipes or installed wheel bundle)",
    )
    list_parser.add_argument("--bundled", action="store_true", help="Force recipes shipped with this installation")
    list_parser.add_argument("--tag", action="append", default=[], help="Require a tag; repeat to require multiple tags")
    list_parser.add_argument("--json", action="store_true", help="Print the versioned JSON recipe catalog")
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
    create_parser.set_defaults(func=_handle_create)
