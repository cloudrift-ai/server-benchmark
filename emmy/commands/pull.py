"""Download a model from HuggingFace to the standard HF cache."""

import logging
import sys

logger = logging.getLogger(__name__)


def register_pull_command(subparsers):
    parser = subparsers.add_parser("pull", help="Download a model from HuggingFace")
    parser.add_argument(
        "model",
        help="HuggingFace model ID, optionally pinned as <repo>@<branch-or-commit> (e.g. meta-llama/Llama-3.1-8B)",
    )
    parser.set_defaults(func=handle_pull)


def handle_pull(args):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is required: pip install huggingface_hub")
        sys.exit(1)

    from emmy.compiler.loader.safetensors import split_revision

    repo, revision = split_revision(args.model)
    logger.info("Pulling %s...", args.model)
    path = snapshot_download(repo, revision=revision)
    logger.info("Cached at: %s", path)
