"""Fork classification by effect — the ``_is_structural_option`` predicate.

The engine classifies every multi-option fork at the spawn site in ``Run.drive`` — where the raw
option list is concrete — and threads the result through ``Search.push(structural=)``: any
``Graph``-splicing option (a kernel-set change) marks the fork structural; pure ``Op`` rebinds and
the body-move tiling forks are op-variant.

This file pins the predicate itself. The engine-level flag on real emitters was covered here until
the from-scratch Tile IR rebuild (#293) removed those two tests along with the rules they drove;
the surviving end-to-end coverage of structural forks is ``test_fork.py`` and
``test_two_level.py``.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import InputOp
from emmy.compiler.pipeline.fork import Fork, OptionFork
from emmy.compiler.pipeline.pipeline import _is_structural_option


class _BranchFork(Fork):
    """Minimal non-leaf ``Fork`` — untypable without ``expand()``, so never structural."""

    knobs: dict = {}

    def expand(self):
        return []


def test_is_structural_option_predicate() -> None:
    """The Op/Graph return-type split IS the classification: a raw ``Graph`` or
    a leaf ``OptionFork`` wrapping one is structural; an ``Op`` rebind, an
    Op-wrapping leaf, and a branch ``Fork`` (untypable without ``expand()`` —
    today always the partition planner's op-variant tree) are not."""
    assert _is_structural_option(Graph())
    assert _is_structural_option(OptionFork(option=Graph()))
    assert not _is_structural_option(InputOp())
    assert not _is_structural_option(OptionFork(option=InputOp()))
    assert not _is_structural_option(_BranchFork())
