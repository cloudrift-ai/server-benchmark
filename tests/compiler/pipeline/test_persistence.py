"""Rewrite-chain persistence at the Graph-splice boundary."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import Op
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.pipeline import Pass, Pattern, Pipeline, Rule
from emmy.compiler.pipeline.pipeline import Cursor, Match, Run, _TerminalBench
from emmy.compiler.pipeline.search.candidate import Candidate
from emmy.compiler.pipeline.search.db import SearchDB


@dataclass
class _KeyedTileOp(Op):
    key: str = ""

    @property
    def dialect(self) -> str:
        return "tile"

    def cache_key(self) -> str:
        return self.key


def _candidate_with_root(op: _KeyedTileOp, *, pass_name: str, rule_name: str) -> tuple[Candidate, Match]:
    graph = Graph()
    graph.add_node(op=op, inputs=[], output=Tensor("root", (1,)), node_id="root")
    graph.outputs = ["root"]
    rule = Rule(name=rule_name, pattern=[Pattern("root", _KeyedTileOp)], rewrite=lambda root: root)
    pass_ = Pass(name=pass_name, rules=[rule])
    run = Run(pipeline=Pipeline([pass_]), ctx=Context.from_target((9, 0)))
    candidate = Candidate(run=run, graph=graph, cursor=Cursor(run=run))
    match = Match(
        graph=graph,
        root_node_id="root",
        rule=rule,
        nodes={"root": "root"},
        consumed={"root"},
        _identities={"root": graph.nodes["root"]},
    )
    return candidate, match


def _two_kernel_fragment(prefix: str) -> Graph:
    fragment = Graph()
    fragment.add_node(op=_KeyedTileOp(f"{prefix}-partial"), inputs=[], output=Tensor("partial", (1,)), node_id="partial")
    fragment.add_node(op=_KeyedTileOp(f"{prefix}-residue"), inputs=["partial"], output=Tensor("root", (1,)), node_id="root")
    fragment.outputs = ["root"]
    return fragment


def _persist_chain(cuda: CudaOp, ctx: Context) -> SearchDB:
    graph = Graph()
    graph.add_node(op=cuda, inputs=[], output=Tensor("out", (1,)), node_id="out")
    graph.outputs = ["out"]
    db = SearchDB()
    bench = _TerminalBench(SimpleNamespace(graph=graph, ctx=ctx), backend=None, db=db)
    bench._persist(cuda, stats=bench._point_stats(5.0), status="ok")
    return db


def _lowering_child(db: SearchDB, parent_key: str) -> str | None:
    row = db._conn.execute("SELECT child_key FROM lowering WHERE parent_key = ?", (parent_key,)).fetchone()
    return row[0] if row is not None else None


def test_place_fragment_timing_does_not_price_the_offer_but_schedule_rebind_does() -> None:
    offer = _KeyedTileOp("place-offer")
    candidate, match = _candidate_with_root(offer, pass_name="lowering/tile", rule_name="015_place")
    candidate.apply(match, _two_kernel_fragment("place"))

    fragment = candidate.graph.nodes["root"].op
    assert fragment.source is offer and fragment.source_is_graph_splice

    scheduled = _KeyedTileOp("place-scheduled", knobs={"TILE": "f1x1"})
    schedule_rule = Rule(name="020_schedule", pattern=[Pattern("root", _KeyedTileOp)], rewrite=lambda root: root)
    Pass(name="lowering/schedule", rules=[schedule_rule])
    schedule_match = Match(
        graph=candidate.graph,
        root_node_id="root",
        rule=schedule_rule,
        nodes={"root": "root"},
        consumed={"root"},
        _identities={"root": candidate.graph.nodes["root"]},
    )
    candidate.apply(schedule_match, scheduled)
    assert scheduled.source is fragment and not scheduled.source_is_graph_splice

    cuda = CudaOp(kernel_source='extern "C" __global__ void k() {}', kernel_name="k", source=scheduled)
    db = _persist_chain(cuda, candidate.ctx)

    assert _lowering_child(db, offer.cache_key()) is None
    assert _lowering_child(db, fragment.cache_key()) == scheduled.cache_key()


def test_split_reduce_fragment_timing_does_not_price_the_pre_split_kernel() -> None:
    scheduled = _KeyedTileOp("split-offer", knobs={"REDUCE": "g2k"})
    candidate, match = _candidate_with_root(scheduled, pass_name="lowering/schedule", rule_name="030_split_reduce")
    candidate.apply(match, _two_kernel_fragment("split"))

    partial = candidate.graph.nodes["partial"].op
    assert partial.source is scheduled and partial.source_is_graph_splice

    cuda = CudaOp(kernel_source='extern "C" __global__ void k_partial() {}', kernel_name="k_partial", source=partial)
    db = _persist_chain(cuda, candidate.ctx)

    assert _lowering_child(db, scheduled.cache_key()) is None
