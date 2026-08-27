"""SearchStrategy — the ABC for search SHAPES over the engine's loop.

A search strategy owns everything ABOVE the loop: how many loops run, over which pass lists,
with which frontier policy or decide callback inside, and what the terminals' results mean
together. ``GreedyStrategy`` (one deterministic resolve plus retry orchestration) and
``TwoLevelStrategy`` (an outer fusion drive scored by a separable Σ of per-kernel inner loops)
are the two shapes today. Contrast the two protocols a shape composes with:
``search.policy.Search`` answers the questions ONE loop asks (frontier ranking, terminal
valuation), and ``pipeline.strategy.PipelineStrategy`` reacts to the events a loop emits
(provenance, identity, the kernel inventory) without steering it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.graph import Graph


class SearchStrategy(ABC):
    """One search shape: constructor carries the shape's configuration (backends, DB, budgets,
    the pipeline or pass lists it composes); :meth:`run` drives one input graph to the shape's
    result. A tuning shape defines ``run`` as a coroutine (its terminal valuation awaits
    device-pinned benches); a deterministic shape is a plain function."""

    @abstractmethod
    def run(self, graph: Graph, ctx: Context | None = None):
        """Drive ``graph`` to this shape's result — a terminal ``Graph`` for a deterministic
        shape, a result object (e.g. ``TwoLevelResult``) for a tuning shape."""
