"""Single-node kernel slice (`search.slice.single_node_graph`).

The two-level tuner isolates each post-fusion kernel into its own graph and
tunes it standalone. For inner-tuned ``perf`` / ``lowering`` rows to transfer
back to the assembled graph, the slice must round-trip to the *same*
``Op.cache_key`` — for the finalized ``LoopOp`` and for every ``CudaOp`` it
lowers to — as the full graph. These tests pin that invariant.

Target is forced to sm_80 so the lowering is deterministic and GPU-independent
(no kernel ever executes — only source generation + structural keys).
"""

from __future__ import annotations

import pytest

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tensor.ir import CastOp, ElementwiseOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.search.db import SearchDB
from emmy.compiler.pipeline.search.slice import single_node_graph, topo_order
from emmy.compiler.pipeline.search.two_level import LOWERING_PASSES


@pytest.fixture(autouse=True)
def _force_target():
    from emmy.compiler import target as target_mod

    target_mod.set_target((8, 0))
    yield
    target_mod.set_target(None)


def _two_matmul_graph() -> Graph:
    """Two independent matmuls of distinct shapes — stays two separate
    ``LoopOp`` kernels after fusion (no shared producer/consumer edge)."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (64, 32)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (32, 48)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (64, 48)), node_id="c")
    g.add_node(InputOp(), [], Tensor("d", (16, 8)), node_id="d")
    g.add_node(InputOp(), [], Tensor("e", (8, 24)), node_id="e")
    g.add_node(MatmulOp(), ["d", "e"], Tensor("f", (16, 24)), node_id="f")
    g.inputs = ["a", "b", "d", "e"]
    g.outputs = ["c", "f"]
    return g


def test_slice_topological_order_ignores_keep_set_iteration_order() -> None:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("residual", (4,)), node_id="residual")
    graph.add_node(InputOp(), [], Tensor("to", (4,)), node_id="to")
    graph.add_node(MatmulOp(), ["residual", "to"], Tensor("root", (4,)), node_id="root")

    class ReverseKeep(set):
        def __iter__(self):
            return iter(("to", "residual", "root"))

    assert topo_order(graph, ReverseKeep(graph.nodes)) == ["residual", "to", "root"]


def test_slice_is_standalone_and_preserves_loop_key() -> None:
    fused = Pipeline.build(LOOP_PASSES).run(_two_matmul_graph(), db=SearchDB())
    loops = [(nid, n.op) for nid, n in fused.nodes.items() if isinstance(n.op, LoopOp)]
    assert len(loops) == 2, f"expected two LoopOp kernels, got {len(loops)}"

    for nid, op in loops:
        sub = single_node_graph(fused, nid)
        # Standalone: sole output is the kernel; every other node is a leaf
        # (InputOp stub or constant) so the slice runs in isolation.
        assert sub.outputs == [nid]
        assert isinstance(sub.nodes[nid].op, LoopOp)
        non_root = [n for k, n in sub.nodes.items() if k != nid]
        assert all(isinstance(n.op, InputOp) for n in non_root), "slice ancestors must be InputOp stubs/leaves"
        # Op shared by reference → identical body → identical key.
        assert sub.nodes[nid].op.cache_key() == op.cache_key()


def test_sliced_kernel_lowers_to_same_cuda_keys() -> None:
    """Lowering each slice with the lowering-only passes yields exactly the
    CudaOp keys the full-graph compile produces — the DB-handoff invariant."""
    graph = _two_matmul_graph()
    full = Pipeline.build(CUDA_PASSES).run(graph, db=SearchDB())
    full_keys = sorted(n.op.cache_key() for n in full.nodes.values() if isinstance(n.op, CudaOp))

    fused = Pipeline.build(LOOP_PASSES).run(graph, db=SearchDB())
    slice_keys: list[str] = []
    for nid, n in fused.nodes.items():
        if not isinstance(n.op, LoopOp):
            continue
        sub = single_node_graph(fused, nid)
        lowered = Pipeline.build(LOWERING_PASSES).run(sub, db=SearchDB())
        slice_keys += [x.op.cache_key() for x in lowered.nodes.values() if isinstance(x.op, CudaOp)]

    assert sorted(slice_keys) == full_keys


def test_slice_makes_surviving_cast_a_synthetic_boundary() -> None:
    """Compact storage algebra can leave a value ``CastOp`` between two LoopOps.

    Loop wire format cannot carry the cast as compute. The cast therefore
    becomes an input stub, rather than a retained node with an unbound input.
    """
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("a", (16,), "i32"), node_id="a")
    graph.add_node(InputOp(), [], Tensor("b", (16,), "i32"), node_id="b")
    graph.add_node(ElementwiseOp(op="subtract"), ["a", "b"], Tensor("centered", (16,), "i32"), node_id="centered")
    graph.add_node(CastOp(dtype="f16"), ["centered"], Tensor("values", (16,), "f16"), node_id="values")
    graph.add_node(InputOp(), [], Tensor("scale", (16,), "f16"), node_id="scale")
    graph.add_node(ElementwiseOp(op="multiply"), ["values", "scale"], Tensor("scaled", (16,), "f16"), node_id="scaled")
    graph.inputs, graph.outputs = ["a", "b", "scale"], ["scaled"]

    fused = Pipeline.build(LOOP_PASSES).run(graph, db=SearchDB())
    loops = [nid for nid, node in fused.nodes.items() if isinstance(node.op, LoopOp)]
    assert len(loops) == 2
    producer = next(nid for nid in loops if "centered" in fused.nodes[nid].buffer_names())
    consumer = next(nid for nid in loops if nid != producer)
    sub = single_node_graph(fused, consumer)
    sub.validate()
    assert producer not in sub.nodes
    assert isinstance(sub.nodes["values"].op, InputOp)
    assert "values" in sub.inputs
