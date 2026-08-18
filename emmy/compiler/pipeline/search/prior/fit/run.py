"""The ``emmy fit`` run harness — full-train fit, cross-validation, and metrics assembly as one
pure function, the run structure shared by every trainer and dataset combination.

:func:`run_fit` owns the *shape* of a fit run and none of its *content*: the trainer arrives as two
configured objects (the shippable model's and the CV folds' — they differ only in seeding policy,
which is the caller's to decide and to record in its header), the dataset as pre-built
:class:`~.group.Group` lists, and every non-deterministic input (dates, repo commit, CLI args)
pre-rendered inside ``header``. No I/O, no clock, no argparse: the same inputs produce the same
``(metrics, fit)`` pair, so the harness is testable on synthetic groups with a stub trainer.

It returns the FIT, not an artifact. Assembling one is shipping policy — which dynamic weight set a
fit with no dynamic cases goes out with, what provenance it carries — and that belongs with the
command layer (:mod:`emmy.commands.fit`), which also keeps what ``pipeline/`` must not import: the
snippet-tracing case builder, the CLI, and the file writing.
"""

from __future__ import annotations

import logging

from emmy.compiler.pipeline.search.prior.fit.cv import build_metrics, run_folds
from emmy.compiler.pipeline.search.prior.fit.group import Group


def run_fit(groups: list[Group], skipped: list[tuple[str, str, str]], *, trainer, fold_trainer, folds, header):
    """One complete fit run → ``(metrics, fit)``. ``trainer.fit(groups)`` produces the shippable
    model over every group; ``fold_trainer`` is the same trainer under the fold-seeding policy and
    fits one model per :func:`~.cv.run_folds` fold. ``folds`` is the fold count (``0`` skips
    cross-validation entirely and the metrics carry an empty ``cv`` block)."""
    full = trainer.fit(groups)

    # Each fold re-fits — silence the fit package's per-case rank logging for that stage
    # (the full-train fit above already reported it).
    pkg_logger = logging.getLogger(__package__)
    level = pkg_logger.level
    pkg_logger.setLevel(logging.WARNING)
    try:
        cv = {"shape": run_folds(groups, trainer=fold_trainer, k=folds)} if folds else {}
    finally:
        pkg_logger.setLevel(level)

    return build_metrics(header, groups, skipped, full, cv), full
