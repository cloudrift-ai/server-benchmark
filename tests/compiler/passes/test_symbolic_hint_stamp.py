"""Symbolic FREE geometry offered to deploy ranking matches scheduler enumeration hints."""

from __future__ import annotations

from emmy.compiler import dtype as _dt
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import Fork
from emmy.compiler.pipeline.pipeline import Run


def test_symbolic_free_hint_facts_survive_materialization() -> None:
    m, n, k = Dim("num_tokens", hint=256), 4096, 4096
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("a", (m, k), _dt.F16), node_id="a")
    graph.add_node(InputOp(), [], Tensor("b", (k, n), _dt.F16), node_id="b")
    graph.add_node(MatmulOp(), ["a", "b"], Tensor("o", (m, n), _dt.F16), node_id="o")
    graph.inputs, graph.outputs = ["a", "b"], ["o"]

    def decide(fp):
        option = fp.options[0]
        while isinstance(option, Fork) and not option.is_leaf:
            option = option.expand()[0]
        return option

    resolved, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((7, 0))).resolve(graph, decide)
    scheduled = [n.op for n in resolved.nodes.values() if "S_hint_free_prod" in getattr(n.op, "knobs", {})]
    assert scheduled, "symbolic matmul produced no hint-stamped schedule"
    assert all(op.knobs["S_hint_n_free_axis"] == 2.0 for op in scheduled)
    assert all(op.knobs["S_hint_free_prod"] == float(256 * 4096) for op in scheduled)
    assert all(op.knobs["S_hint_free_max"] == 4096.0 for op in scheduled)
