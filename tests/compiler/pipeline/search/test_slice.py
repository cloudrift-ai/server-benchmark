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
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.search.db import SearchDB
from emmy.compiler.pipeline.search.slice import single_node_graph
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


def test_flash_slice_absorbs_score_producer() -> None:
    """Fold-aware slicing: a flash fold offer site's slice CARRIES the score producer its fusion
    consumes (``fused_producer_ids`` → ``absorb``) as a REAL ``LoopOp`` — a synthetic-input
    boundary would make ``try_flash`` unfusable in-slice, degrading every tune trajectory to the
    cut. The absorbed slice re-fuses: lowering it yields ONE fused flash kernel."""
    import torch  # noqa: PLC0415

    from emmy.compiler.pipeline.passes.lowering.tile._flash import fused_producer_ids  # noqa: PLC0415
    from emmy.compiler.trace.torch import trace_module  # noqa: PLC0415

    class _Sdpa(torch.nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    q, k, v = (torch.randn(1, 2, 16, 8) for _ in range(3))
    graph = trace_module(_Sdpa().cpu(), (q, k, v))
    fused = Pipeline.build(LOOP_PASSES).run(graph, db=SearchDB())
    loops = [(nid, n) for nid, n in fused.nodes.items() if isinstance(n.op, LoopOp)]
    assert len(loops) == 2, f"outer terminal should hold score + softmax-P@V LoopOps, got {len(loops)}"

    absorbed = {pid: nid for nid, node in loops for pid in fused_producer_ids(fused, node)}
    assert len(absorbed) == 1, f"exactly one score producer should be absorbed, got {absorbed}"
    (producer, consumer) = next(iter(absorbed.items()))

    sub = single_node_graph(fused, consumer, absorb=frozenset({producer}))
    assert isinstance(sub.nodes[producer].op, LoopOp), "the absorbed producer must stay a real LoopOp"
    lowered = Pipeline.build(LOWERING_PASSES).run(sub, db=SearchDB())
    cuda = [n for n in lowered.nodes.values() if isinstance(n.op, CudaOp)]
    assert len(cuda) == 1, f"the absorbed slice must re-fuse to one flash kernel, got {len(cuda)}"

    # Without absorb the producer is a synthetic input and the fusion cannot fire — two kernels.
    plain = single_node_graph(fused, consumer)
    assert isinstance(plain.nodes[producer].op, InputOp), "un-absorbed producer is a synthetic input"
