"""ensure_plugin_logging: emmy INFO logs must surface when the host process configured
no logging (the bare vLLM entrypoint case), and must stay hands-off when logging is
already configured (the CLI path) so nothing double-prints."""

import logging
import subprocess
import sys

from emmy.logging_setup import ensure_plugin_logging


def test_attaches_handler_when_unconfigured():
    # A subprocess is the faithful simulation: in-process, pytest's own logging plugin
    # keeps a capture handler on root, so the "nothing configured" state never exists here.
    code = (
        "import logging\n"
        "from emmy.logging_setup import ensure_plugin_logging\n"
        "ensure_plugin_logging()\n"
        "ensure_plugin_logging()  # idempotent: register() may run more than once\n"
        "handlers = logging.getLogger('emmy').handlers\n"
        "assert len(handlers) == 1, handlers\n"
        "logging.getLogger('emmy.serving.gen_runner').info('pack hit at /opt/emmy/pack/x')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "pack hit at /opt/emmy/pack/x" in out.stderr


def test_noop_when_logging_already_configured():
    root, emmy_log = logging.getLogger(), logging.getLogger("emmy")
    saved = emmy_log.handlers[:]
    guard = logging.NullHandler()
    root.addHandler(guard)
    try:
        ensure_plugin_logging()
        assert emmy_log.handlers == saved
    finally:
        root.removeHandler(guard)
