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

from emmy.compiler.pipeline.search.prior.fit.cv import build_metrics, run_axis
from emmy.compiler.pipeline.search.prior.fit.group import Group


def run_fit(groups: list[Group], skipped: list[tuple[str, str, str]], *, trainer, fold_trainer, axes, header):
    """One complete fit run → ``(metrics, fit)``. ``trainer.fit(groups)`` produces the shippable
    model over every group; ``fold_trainer`` is the same trainer under the fold-seeding policy and
    fits one model per :func:`~.cv.run_axis` fold."""
    full = trainer.fit(groups)

    # CV folds re-fit ~2x per axis per fold — silence the fit package's per-case rank
    # logging for that stage (the full-train fit above already reported it).
    pkg_logger = logging.getLogger(__package__)
    level = pkg_logger.level
    pkg_logger.setLevel(logging.WARNING)
    try:
        cv = {axis: run_axis(groups, axis, trainer=fold_trainer) for axis in axes}
    finally:
        pkg_logger.setLevel(level)

    return build_metrics(header, groups, skipped, full, cv), full
