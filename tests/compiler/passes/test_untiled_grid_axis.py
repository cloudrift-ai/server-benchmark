"""Every grid axis the tiled ``(m, n)`` cell does not take stays a declared kernel axis.

A kernel that writes several outputs carries one grid axis per distinct output extent. The tiled
root claims two of them as its ``(m, n)`` cell — and which two is a fact about that CONTRACTION's
own operand axes (``Sched._derive_mn``), not a position in the grid: a fused q/k/v projection whose
q is N 64 beside k/v at N 32 tiles the q pair while the k/v output axis rides the grid beside it.
Everything the cell does not take must still be bound, because the per-cell SSA rename passes those
coordinates through (``_atom._scalar_protected``) and the ``Tile`` declares them.

Reading them off the grid POSITIONALLY breaks on exactly that shape: the untiled axis vanished from
the kernel and the tiled m axis was re-declared beside itself, after which the per-cell rename took
the now-unbound axis for an ordinary SSA name and suffixed it — nvcc: ``identifier "a2__c0_0" is
undefined`` (the serving q/k/v twin on sm_120, three kernels of the generation capture suite).

The oracle is the kernel body's free names: a well-formed kernel reads its own buffers and the
symbolic extents its launch passes, and nothing else.
"""

from __future__ import annotations

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.ir.stmt.body import free_names
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import KERNEL_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.target import set_target

_CAP = (12, 0)
_TOKENS = Dim("num_tokens")
_K, _N_Q, _N_KV = 64, 64, 32  # q's N differs from k/v's — the second output axis on one grid


def _qkv_graph() -> Graph:
    """The fused projection shape: one shared computed operand feeding three linears, q wider than
    k/v. Under a pinned fuse this is ONE kernel whose grid carries both output extents."""
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (_TOKENS, _K), dtype=F32), node_id="x")
    for weight, n in (("wq", _N_Q), ("wk", _N_KV), ("wv", _N_KV)):
        graph.add_node(InputOp(), [], Tensor(weight, (n, _K), dtype=F32), node_id=weight)
    # A computed A cone — what makes the three contractions fuse into one kernel, the way a
    # serving block's shared input norm does.
    graph.add_node(ElementwiseOp("multiply"), ["x", "x"], Tensor("xn", (_TOKENS, _K), dtype=F32), node_id="xn")
    for out, weight, n in (("q", "wq", _N_Q), ("k", "wk", _N_KV), ("v", "wv", _N_KV)):
        graph.add_node(LinearOp(), ["xn", weight], Tensor(out, (_TOKENS, n), dtype=F32), node_id=out)
    graph.inputs, graph.outputs = ["x", "wq", "wk", "wv"], ["q", "k", "v"]
    return graph


def test_a_sibling_outputs_grid_axis_stays_bound() -> None:
    """The oracle is about the FUSED kernel, so the fused arm is pinned rather than assumed.

    Both placement arms are well formed on this shape. Fused, one kernel carries every output
    extent, and the untiled one has to stay a declared axis — the fact this test is about. Cut, the
    q branch and the k/v branch each own their outputs and become a kernel binding its own extent
    on the grid (the output-owning cut: no axis rides all three stores, so the fused kernel promotes
    no sweep and keeps a one-axis placement, while each piece promotes its own). Both lower and
    validate; the pin selects the one whose body the oracle reads."""
    set_target(_CAP)
    try:
        with pinned_knobs({"PLACE": "fuse"}):
            lowered = Pipeline.build(KERNEL_PASSES).run(_qkv_graph(), ctx=Context(compute_capability=_CAP))
    finally:
        set_target(None)
    kernels = [node for node in lowered.nodes.values() if getattr(node.op, "body", None) is not None]

    assert len(kernels) == 1, "the pinned fuse arm must give this shape one kernel for the untiled axis to arise"
    node = kernels[0]
    bound = {*node.inputs, *node.buffer_names(), *_TOKENS.expr.free_vars()}
    free = set().union(*(free_names(stmt) for stmt in node.op.body))
    assert free <= bound, f"kernel body reads names the kernel never binds: {sorted(free - bound)}"
