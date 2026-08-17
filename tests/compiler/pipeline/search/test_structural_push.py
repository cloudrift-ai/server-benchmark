"""Fork classification by effect — the ``_is_structural_option`` predicate, and what greedy does
with a structural option it cannot price.

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

from types import SimpleNamespace

from emmy.compiler.context import Context
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


class _Prior:
    """A prior that scores a leaf by its own ``rank`` knob — lower is better, and a leaf with no
    knob row at all (a ``Graph`` splice) reads ``0.0``, the best score of the three."""

    def mean_scores(self, rows):
        return [float(r.get("rank", 0)) for r in rows]


def _fork(options):
    from emmy.compiler.graph import Tensor

    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("n0", (4,)), node_id="n0")
    graph.inputs, graph.outputs = [], ["n0"]
    return SimpleNamespace(
        ctx=Context.from_target((12, 0)),
        options=options,
        root_op=SimpleNamespace(knobs={}),
        node_id="n0",
        score=None,
        match=SimpleNamespace(rule=None, graph=graph),
    )


def test_an_unpriceable_structural_option_is_ranked_not_withheld() -> None:
    """A ``Graph`` leaf nothing can price is just a leaf. Greedy used to FILTER the structural
    options out whenever :func:`greedy._priced_pick` decided nothing — cold prior, unpriceable
    option — so that a compile without evidence never changed the kernel set. That was a
    hand-written preference for the fused form, and it is gone: an option nothing can price
    competes in the ordinary leaf ranking like any other.

    The pricing probes here return ``None`` (the fake leaves have no real graph behind them), so
    this exercises exactly the path that used to filter."""
    from emmy.compiler.pipeline.search.policy.greedy import greedy_decide

    splice = Graph()
    fused = OptionFork(option=InputOp(), knobs={"rank": 1})
    decide = greedy_decide(prior=_Prior())
    assert decide(_fork([fused, splice])) is splice


def test_structural_retirement_still_withdraws_the_splices() -> None:
    """``price_structural=False`` is NOT the deleted filter: it is how ``Pipeline.run`` retires a
    structural pick whose fragment kernel failed to LOWER (a splice mints fresh node ids, so it
    cannot be blocklisted at the fork site), and how a nested price probe avoids re-splitting the
    slice it is pricing. Both are validity mechanics, so they survive — the same fork that ranks
    the splice above must keep the op leaf here."""
    from emmy.compiler.pipeline.search.policy.greedy import greedy_decide

    splice = Graph()
    fused = OptionFork(option=InputOp(), knobs={"rank": 1})
    decide = greedy_decide(prior=_Prior(), price_structural=False)
    assert decide(_fork([fused, splice])) is fused
