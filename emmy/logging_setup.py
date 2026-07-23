"""CLI logging setup: simple %(message)s format for standalone commands."""

import logging
import sys

from emmy.redact import install_redaction


def setup_cli_logging():
    """Configure root logger with plain message format for CLI commands.

    Produces output identical to print(). The bench command's setup_logging()
    overrides this with a prefixed format.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    install_redaction(handler)
    root.addHandler(handler)


def ensure_plugin_logging():
    """Make emmy INFO logs visible when the host process configured no logging.

    Under a bare vLLM entrypoint (``python3 -m vllm.entrypoints.openai.api_server``)
    nothing configures the root logger — vLLM's dictConfig covers only the ``vllm``
    tree — so emmy's INFO records (e.g. the serving runners' "pack hit" line, which
    the gemma4 image's verify.sh gates on via ``docker logs``) are silently dropped
    by Python's WARNING-only last-resort handler. If no handler up the tree would
    emit ``emmy`` records, attach an INFO stderr handler to the ``emmy`` logger;
    no-op when logging is already configured (the CLI path), so nothing double-prints.
    """
    log = logging.getLogger("emmy")
    node = log
    while node is not None:
        if node.handlers:
            return
        node = node.parent if node.propagate else None
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s %(name)s: %(message)s"))
    install_redaction(handler)
    log.addHandler(handler)
    log.setLevel(logging.INFO)
