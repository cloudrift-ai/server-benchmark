"""Publish the canonical serving image named by a recipe."""

import logging
import shlex
import subprocess
import sys

from emmy.publish import load_publish_recipe, publish_recipe_image

logger = logging.getLogger(__name__)


def register_publish_command(subparsers):
    parser = subparsers.add_parser("publish", help="Publish a recipe's canonical serving image")
    parser.add_argument("recipe", help="Path to a recipe directory")
    parser.add_argument(
        "--source-image",
        default=None,
        help="Locally built image to retag; defaults to the recipe image",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata and print commands without changing images")
    parser.add_argument("--yes", action="store_true", help="Confirm the noninteractive Docker tag and push")
    parser.set_defaults(func=handle_publish)


def handle_publish(args):
    try:
        recipe = load_publish_recipe(args.recipe)
        plan = publish_recipe_image(
            recipe,
            source_image=args.source_image,
            dry_run=args.dry_run,
            yes=args.yes,
        )
    except (FileNotFoundError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        logger.error("Publish failed: %s", exc)
        sys.exit(1)

    logger.info("Source:      %s", plan.source_image)
    logger.info("Image ID:    %s", plan.source_image_id)
    logger.info("Destination: %s", plan.destination.image)
    if plan.already_published:
        logger.info("Registry:    already published at %s", plan.registry_digest)
    elif args.dry_run:
        logger.info("Registry:    destination is absent")
        for command in plan.commands:
            logger.info("DRY RUN: %s", shlex.join(command))
    else:
        logger.info("Published %s@%s", plan.destination.image, plan.registry_digest)
