"""The ``S_warp_eligible`` stamp must survive onto the MATERIALIZED op.

The scheduler stamps it into every fork option row (branch identity, deploy candidates), but the
materialized op is what ``realized_knobs`` — and therefore every leaf / -O3 evidence row — reads.
When ``_schedule``'s ``_materialize`` dropped it, one op's rows fractured into two ``S_*``
signatures (fork rows stamped, leaf rows not), ``Prior.evidence_pick`` never joined the measured
-O3 rows at deploy time, and greedy shipped the online model's unbenched per-cell extrapolation
(the 2026-07-07 RTX 5090 gate: 1157 µs per-cell b256 vs the 3.5 µs mma golden, ~330x).
"""

from __future__ import annotations

from emmy.compiler import dtype as _dt
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import Fork
from emmy.compiler.pipeline.pipeline import Run


def _matmul_graph(M: int, N: int, K: int, dtype: str) -> Graph:
    graph = Graph()
    dt = _dt.get({"fp16": "f16", "fp32": "f32"}.get(dtype, dtype))
    graph.add_node(InputOp(), [], Tensor("a", (M, K), dt), node_id="a")
    graph.add_node(InputOp(), [], Tensor("b", (K, N), dt), node_id="b")
    graph.add_node(MatmulOp(), ["a", "b"], Tensor("o", (M, N), dt), node_id="o")
    graph.inputs, graph.outputs = ["a", "b"], ["o"]
    return graph


def _resolve_option0(graph, ctx):
    def decide(fp):
        option = fp.options[0]
        while isinstance(option, Fork) and not option.is_leaf:
            option = option.expand()[0]
        return option

    resolved, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(graph, decide)
    return resolved


def test_materialized_op_carries_warp_eligibility_stamp():
    """An fp16 matmul on a tensor-core-capable target offers warp rows, so EVERY materialized
    variant of the kernel — whichever row option-0 lands on — must carry ``S_warp_eligible=1.0``
    in its op knobs, matching the fork rows' stamp (one op = one ``S_*`` signature)."""
    ctx = Context.from_target((12, 0))
    resolved = _resolve_option0(_matmul_graph(512, 512, 512, "fp16"), ctx)
    stamps = [
        node.op.knobs.get("S_warp_eligible")
        for node in resolved.nodes.values()
        if getattr(node.op, "knobs", None) and any(k.split("@")[0] == "TILE" for k in node.op.knobs)
    ]
    assert stamps, "no tile-scheduled op in the resolved graph"
    assert all(v == 1.0 for v in stamps), f"materialized op lost the S_warp_eligible stamp: {stamps}"
