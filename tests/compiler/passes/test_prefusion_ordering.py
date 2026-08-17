"""``loop/prefusion`` orders the merges so a contraction closes before anything splices into it.

A merge is directional: it makes the SINK the region's output, so the sink's width must then be
written. Splice a compute producer into a still-open product and the free-axes x contraction-axis
outer product lands in gmem — and the fold that would have collapsed it can only arrive afterwards
as a reduce nested in a reduce, which ``loop/fusion`` refuses as an unreadable seam. Nothing
downstream can undo that: the buffer is there at every tile.

Which of the two orders the fixpoint reaches was decided by whichever match the enumeration hit
first. ``loop/prefusion`` drains the narrowing merges to fixpoint first, so the good order is the
one that happens — without refusing anything, since ``loop/fusion`` still offers every widening
merge afterwards.
"""

import numpy as np

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline


def _chained_matmuls(m=8, k0=4, k1=6, n=5) -> Graph:
    """``(x @ w0) @ w1`` — the smallest graph with an intermediate contraction."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (m, k0)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w0", (k1, k0)), node_id="w0")
    g.add_node(InputOp(), [], Tensor("w1", (n, k1)), node_id="w1")
    g.add_node(LinearOp(), ["x", "w0"], Tensor("h", (m, k1)), node_id="h")
    g.add_node(LinearOp(), ["h", "w1"], Tensor("y", (m, n)), node_id="y")
    g.inputs, g.outputs = ["x", "w0", "w1"], ["y"]
    return g


def _kernel_outputs(graph: Graph) -> dict[str, int]:
    out = {}
    for node in graph.nodes.values():
        if isinstance(node.op, LoopOp):
            numel = 1
            for dim in node.output.shape:
                numel *= dim.as_static()
            out[node.id] = numel
    return out


def test_chained_matmul_never_writes_the_outer_product():
    """No kernel may output more than the widest contraction RESULT.

    Without the ordering this graph leaves a ``(m, k1, n)`` buffer behind — 240 elements where 48
    is the budget. On a 1-layer Qwen3-0.6B trunk at seq 512 the same defect planned a 6.006 GiB
    scratch slab instead of 0.026 GiB, and at batch 4 it pushed a buffer past 2^31 elements."""
    m, k1, n = 8, 6, 5
    result = Pipeline.build(LOOP_PASSES).run(_chained_matmuls(m, 4, k1, n))
    budget = max(m * k1, m * n)
    for name, numel in _kernel_outputs(result).items():
        assert numel <= budget, f"{name} writes {numel} elements > {budget} — the outer product reached gmem"


def test_prefusion_defers_widening_merges_without_refusing_them():
    """The pass is an ORDERING, not a gate: every merge ``loop/fusion`` would have taken is still
    offered to it afterwards. Compare the two pipelines — running fusion alone must not produce
    MORE kernels than running prefusion in front of it, i.e. prefusion never costs a merge."""
    without = Pipeline.build([p for p in LOOP_PASSES if p != "loop/prefusion"]).run(_chained_matmuls())
    with_pre = Pipeline.build(LOOP_PASSES).run(_chained_matmuls())
    assert len(_kernel_outputs(with_pre)) <= len(_kernel_outputs(without))


def test_ordering_preserves_numerics():
    """Fusion order is a scheduling choice over the same algebra; the answer cannot move."""
    from emmy.compiler.backend.numpy import NumpyBackend

    rng = np.random.default_rng(0)
    inputs = {
        "x": rng.standard_normal((8, 4)).astype(np.float32),
        "w0": rng.standard_normal((6, 4)).astype(np.float32),
        "w1": rng.standard_normal((5, 6)).astype(np.float32),
    }
    backend = NumpyBackend()
    fused = Pipeline.build(LOOP_PASSES).run(_chained_matmuls())
    got = next(iter(backend.run(backend.compile(fused), input_data=inputs)[0].outputs.values()))
    want = (inputs["x"] @ inputs["w0"].T) @ inputs["w1"].T
    np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-5, atol=1e-5)
