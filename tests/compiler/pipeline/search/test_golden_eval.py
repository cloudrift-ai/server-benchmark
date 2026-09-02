"""Focused tests for program-backed golden enumeration."""

from dataclasses import dataclass, field
from types import SimpleNamespace

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph
from emmy.compiler.pipeline.fork import DeferredFork, Fork
from emmy.compiler.pipeline.search.golden_eval import enumerate_graph


@dataclass(frozen=True)
class _EmptyBranch(Fork):
    knobs: dict = field(default_factory=dict)
    is_leaf = False

    def expand(self):
        return []


def test_enumeration_skips_an_empty_pinned_branch(monkeypatch) -> None:
    row = {"WORK": "w1x1", "TILE": "mma"}
    live = DeferredFork(materialize=lambda: None, knobs=row)

    def resolve(_self, graph, decide):
        assert decide(SimpleNamespace(options=[_EmptyBranch(), live])) is live
        return graph, []

    monkeypatch.setattr("emmy.compiler.pipeline.pipeline.Run.resolve", resolve)

    candidates = enumerate_graph(Graph(), Context.from_target((8, 0)))

    assert candidates.rows == [row]
