#!/usr/bin/env python3
"""Patch a tabbyAPI checkout so `vllm bench serve` can drive it.

One server-side defect, recorded in plans/vq-phase0-findings.md §5.1 and reproduced on
tabbyAPI @ d844f705 with exllamav3 1.4.0: the vLLM benchmark client always sends
``"logprobs": null``, and tabby's exllamav3 backend compares that field against an int
without a None guard, so every request dies with ``TypeError: '>' not supported between
instances of 'NoneType' and 'int'``. There is no client-side workaround — the field is
unconditional — so the fix belongs here, as a scripted, idempotent, re-runnable step of
bringing a tabby clone up.

The OTHER Phase 0 defect, the SSE framing wedge, is deliberately NOT patched here. It is a
CLIENT bug (``vllm bench serve`` splits events on ``\\n\\n`` and chokes on CRLF framing and
on keepalive comments), it silently fabricates latency numbers rather than failing, and it
would bite any engine that frames its stream that way — so it is fixed once, engine-
agnostically, in ``scripts/bench_serve_sweep.py``. Setting ``sse_ping_interval: 0`` in
tabby's own config.yml removes the keepalive half without touching its source at all.

A pattern that no longer matches is reported as a warning, not an error: upstream may have
fixed it. The downstream guard is ``bench_serve_sweep.py`` refusing to report any run whose
requests did not all complete, which is what turns a silent failure into a loud one.

    python scripts/patch_tabbyapi.py --path ~/venvs/exl3/tabbyAPI
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# (description, file glob relative to the clone, pattern, replacement)
PATCHES = [
    (
        "guard the None logprobs the vLLM bench client always sends",
        "backends/exllamav3/*.py",
        re.compile(r"\bif params\.logprobs\s*>"),
        "if params.logprobs and params.logprobs >",
    ),
]


def apply_patches(root: Path) -> int:
    """Apply every patch in place. Returns the number that matched nothing."""
    missed = 0
    for description, glob, pattern, replacement in PATCHES:
        hits = 0
        for path in sorted(root.glob(glob)):
            text = path.read_text(encoding="utf-8", errors="surrogateescape")
            patched, count = pattern.subn(replacement, text)
            if count:
                path.write_text(patched, encoding="utf-8", errors="surrogateescape")
                hits += count
                logger.info("patched %s (%d site%s): %s", path.relative_to(root), count, "" if count == 1 else "s", description)
        if not hits:
            missed += 1
            logger.warning(
                "no site matched under %s for: %s — already patched, or upstream changed it. "
                "Verify by hand before trusting a run; bench_serve_sweep.py's completed-request check is the backstop.",
                glob,
                description,
            )
    return missed


def main(argv: list[str] | None = None) -> int:
    from emmy.logging_setup import setup_cli_logging

    setup_cli_logging()
    parser = argparse.ArgumentParser(prog="patch_tabbyapi.py", description=__doc__.splitlines()[0])
    parser.add_argument("--path", required=True, help="path to the tabbyAPI clone")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    root = Path(args.path).expanduser()
    if not (root / "backends").is_dir():
        logger.error("%s does not look like a tabbyAPI clone (no backends/ directory)", root)
        return 2
    apply_patches(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
