"""The ``emmy fit`` run harness — full-train fit, cross-validation, and metrics/artifact
assembly as one pure function, the run structure shared by every trainer and dataset combination.

:func:`run_fit` owns the *shape* of a fit run and none of its *content*: the trainer
arrives as two callables (``full_train_fit`` for the shippable model, ``fit_model`` for
the fold models — the seeding split between them is the caller's policy, recorded in its
header), the dataset as pre-built :class:`~.group.Group` lists, and every
non-deterministic input (dates, repo commit, CLI args) pre-rendered inside ``header`` /
``provenance``. No I/O, no clock, no argparse: the same inputs produce the same
``(metrics, artifact)`` dicts, so the harness is testable on synthetic groups with a
stub trainer. The command layer (:mod:`emmy.commands.fit`) keeps what ``pipeline/`` must
not import — the snippet-tracing case builder — plus the CLI and file writing.
"""

from __future__ import annotations

import logging

import numpy as np

from emmy.compiler.pipeline.search.prior.fit.cv import build_metrics, run_axis
from emmy.compiler.pipeline.search.prior.fit.group import Group


def run_fit(groups: list[Group], skipped: list[tuple[str, str, str]], *, full_train_fit, fit_model, axes, seed, header, params, provenance):
    """One complete fit run → the ``(metrics, artifact)`` dict pair the command layer
    serializes. ``full_train_fit(groups, rng) -> (model, notes)`` fits the shippable
    model over every group (``notes`` is its free-text provenance line);
    ``fit_model(train_groups, rng) -> model`` fits one fold model per
    :func:`~.cv.run_axis` fold. ``params`` carries into the artifact unchanged and
    ``provenance`` is completed here with the case counts and ``notes``."""
    full_model, notes = full_train_fit(groups, np.random.default_rng(seed))

    # CV folds re-fit ~2x per axis per fold — silence the fit package's per-case rank
    # logging for that stage (the full-train fit above already reported it).
    pkg_logger = logging.getLogger(__package__)
    level = pkg_logger.level
    pkg_logger.setLevel(logging.WARNING)
    try:
        cv = {axis: run_axis(groups, axis, fit_model=fit_model, seed=seed) for axis in axes}
    finally:
        pkg_logger.setLevel(level)

    metrics = build_metrics(header, groups, skipped, full_model, cv)
    n_dyn = sum(1 for g in groups if g.tier == "dyn")
    artifact = full_model.to_artifact(
        params=params,
        provenance={**provenance, "cases": {"static": len(groups) - n_dyn, "dynamic": n_dyn}, "notes": notes},
    )
    return metrics, artifact
