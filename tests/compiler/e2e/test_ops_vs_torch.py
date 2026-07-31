"""Graph ops across every backend, against PyTorch eager as ground truth.

Two axes, both parametrized: the **backend** (numpy / loop / cuda) comes from the ``run_graph``
fixture in ``conftest.py``, and the **op** comes from the case tables below. For each aten
operation the tracer captures, every backend's output is compared against torch. Requires PyTorch.

Elementwise unary and binary are value-level matrices (one op name + one torch callable per row).
Everything else is a ``Case``: a builder returning ``(graph, run_inputs, expected)`` plus its
tolerance. Adding an op is adding a row — do that rather than adding a bespoke test function.

Builders draw from the module-level ``rng``, which the root conftest reseeds before every test,
so each case sees the same inputs regardless of execution order.
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest
import torch

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.frontend.ir import (
    CatOp,
    LinearOp,
    MatmulOp,
    MeanOp,
    ReshapeOp,
    SdpaOp,
    SliceOp,
    TransposeOp,
    UnsqueezeOp,
)
from emmy.compiler.ir.tensor.ir import ElementwiseOp, GatherOp, ReduceOp

rng = np.random.default_rng(42)


def _torch_to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _run(run_graph, graph: Graph, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Execute through ``run_graph`` fixture and return the single output array."""
    return list(run_graph(graph, inputs).values())[0]


# ---------------------------------------------------------------------------
# Elementwise unary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn,torch_fn",
    [
        ("negative", lambda x: torch.neg(x)),
        ("exp", lambda x: torch.exp(x)),
        ("rsqrt", lambda x: torch.rsqrt(x)),
        ("reciprocal", lambda x: torch.reciprocal(x)),
        ("relu", lambda x: torch.relu(x)),
        ("tanh", lambda x: torch.tanh(x)),
        ("sigmoid", lambda x: torch.sigmoid(x)),
        ("abs", lambda x: torch.abs(x)),
        ("silu", lambda x: torch.nn.functional.silu(x)),
    ],
)
def test_unary(fn, torch_fn, run_graph):
    x_np = rng.uniform(0.1, 5.0, size=(4, 8)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ElementwiseOp(fn), ["x"], Tensor("y", (4, 8)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    expected = _torch_to_np(torch_fn(torch.from_numpy(x_np)))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np}), expected, rtol=2e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Elementwise binary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn,torch_fn",
    [
        ("add", lambda x, y: x + y),
        ("subtract", lambda x, y: x - y),
        ("multiply", lambda x, y: x * y),
        ("divide", lambda x, y: x / y),
    ],
)
def test_binary(fn, torch_fn, run_graph):
    x_np = rng.standard_normal((4, 8)).astype(np.float32)
    y_np = rng.uniform(0.1, 5.0, size=(4, 8)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (4, 8)), node_id="y")
    g.add_node(ElementwiseOp(fn), ["x", "y"], Tensor("z", (4, 8)), node_id="z")
    g.inputs, g.outputs = ["x", "y"], ["z"]
    expected = _torch_to_np(torch_fn(torch.from_numpy(x_np), torch.from_numpy(y_np)))
    rtol = 1e-3 if fn == "divide" else 1e-5
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np, "y": y_np}), expected, rtol=rtol, atol=1e-5)


# ---------------------------------------------------------------------------
# The op case table
# ---------------------------------------------------------------------------
# A node is ``(op, srcs, node_id, shape)``; ``_graph`` adds them in the order given, so a constant
# precedes its consumer exactly as it would when hand-written.


def _graph(nodes, inputs, outputs) -> Graph:
    g = Graph()
    for op, srcs, nid, shape in nodes:
        g.add_node(op, list(srcs), Tensor(nid, shape), node_id=nid)
    g.inputs, g.outputs = list(inputs), list(outputs)
    return g


def _in(name: str, shape: tuple):
    return (InputOp(), [], name, shape)


def _const(name: str, shape: tuple = (1,), **kw):
    return (ConstantOp(name=name, **kw), [], name, shape)


def _normal(*shape) -> np.ndarray:
    return rng.standard_normal(shape if len(shape) > 1 else shape[0]).astype(np.float32)


class Case(NamedTuple):
    """One op under test: ``build()`` returns the graph, the run inputs, and the torch answer."""

    id: str
    build: Callable[[], tuple[Graph, dict[str, np.ndarray], np.ndarray]]
    rtol: float = 1e-5
    atol: float = 1e-5


# --- pointwise with a constant / broadcast ---------------------------------


def _pow():
    x = rng.uniform(0.1, 5.0, size=(4, 8)).astype(np.float32)
    g = _graph([_in("x", (4, 8)), _const("p", value=2.0), (ElementwiseOp("pow"), ["x", "p"], "y", (4, 8))], ["x"], ["y"])
    return g, {"x": x}, _torch_to_np(torch.from_numpy(x).pow(2.0))


def _add_broadcast():
    x, y = _normal(4, 8), _normal(8)
    g = _graph([_in("x", (4, 8)), _in("y", (8,)), (ElementwiseOp("add"), ["x", "y"], "z", (4, 8))], ["x", "y"], ["z"])
    return g, {"x": x, "y": y}, _torch_to_np(torch.from_numpy(x) + torch.from_numpy(y))


# --- reductions ------------------------------------------------------------


def _reduce(op, ref):
    def build():
        x = _normal(4, 8)
        g = _graph([_in("x", (4, 8)), (op, ["x"], "y", (4, 1))], ["x"], ["y"])
        return g, {"x": x}, _torch_to_np(ref(torch.from_numpy(x)))

    return build


# --- layout ----------------------------------------------------------------


def _layout(op, in_shape, out_shape, ref):
    def build():
        x = _normal(*in_shape)
        g = _graph([_in("x", in_shape), (op, ["x"], "y", out_shape)], ["x"], ["y"])
        return g, {"x": x}, _torch_to_np(ref(torch.from_numpy(x)))

    return build


# --- slice / cat / gather --------------------------------------------------


def _slice():
    x = _normal(4, 8)
    nodes = [
        _in("x", (4, 8)),
        _const("dim", value=1.0),
        _const("start", value=2.0),
        _const("end", value=6.0),
        (SliceOp(shape=(4, 4)), ["x", "dim", "start", "end"], "y", (4, 4)),
    ]
    return _graph(nodes, ["x"], ["y"]), {"x": x}, _torch_to_np(torch.from_numpy(x)[:, 2:6])


def _cat():
    a, b = _normal(4, 3), _normal(4, 5)
    nodes = [_in("a", (4, 3)), _in("b", (4, 5)), _const("dim", value=1.0), (CatOp(), ["a", "b", "dim"], "y", (4, 8))]
    expected = _torch_to_np(torch.cat([torch.from_numpy(a), torch.from_numpy(b)], dim=1))
    return _graph(nodes, ["a", "b"], ["y"]), {"a": a, "b": b}, expected


def _gather():
    x = _normal(4, 8)
    idx = rng.integers(0, 8, size=(4, 3))
    g = _graph([_in("x", (4, 8)), _in("idx", (4, 3)), (GatherOp(axis=1), ["x", "idx"], "y", (4, 3))], ["x", "idx"], ["y"])
    expected = _torch_to_np(torch.from_numpy(x).gather(1, torch.from_numpy(idx).long()))
    return g, {"x": x, "idx": idx.astype(np.float32)}, expected


def _embedding():
    # GatherOp with output rank = idx_rank + data_rank - 1 (embedding / index_select semantics).
    # The lift must index data and idx at their ACTUAL ranks — not at the output rank.
    # Qwen3-style: idx (1, S), weight (V, H), output (1, S, H).
    V, H, S = 16, 4, 5
    weight = _normal(V, H)
    idx = rng.integers(0, V, size=(1, S))
    g = _graph([_in("w", (V, H)), _in("idx", (1, S)), (GatherOp(axis=0), ["w", "idx"], "y", (1, S, H))], ["w", "idx"], ["y"])
    expected = _torch_to_np(torch.nn.functional.embedding(torch.from_numpy(idx).long(), torch.from_numpy(weight)))
    return g, {"w": weight, "idx": idx.astype(np.float32)}, expected


def _index_select_middle_axis():
    # index_select with axis != 0 and idx rank 1: output rank equals data rank, with the axis-dim
    # replaced by idx size. Mis-mapping the data axes would corrupt indexing.
    data = _normal(3, 5, 4)
    idx = rng.integers(0, 5, size=(3,))
    g = _graph([_in("d", (3, 5, 4)), _in("idx", (3,)), (GatherOp(axis=1), ["d", "idx"], "y", (3, 3, 4))], ["d", "idx"], ["y"])
    expected = _torch_to_np(torch.index_select(torch.from_numpy(data), 1, torch.from_numpy(idx).long()))
    return g, {"d": data, "idx": idx.astype(np.float32)}, expected


# --- matmul / linear -------------------------------------------------------


def _matmul():
    a, b = _normal(4, 8), _normal(8, 3)
    g = _graph([_in("a", (4, 8)), _in("b", (8, 3)), (MatmulOp(), ["a", "b"], "c", (4, 3))], ["a", "b"], ["c"])
    return g, {"a": a, "b": b}, _torch_to_np(torch.from_numpy(a) @ torch.from_numpy(b))


def _matmul_with_bias():
    a, b, bias = _normal(4, 8), _normal(8, 3), _normal(3)
    nodes = [_in("a", (4, 8)), _in("b", (8, 3)), _in("bias", (3,)), (MatmulOp(has_bias=True), ["a", "b", "bias"], "c", (4, 3))]
    expected = _torch_to_np(torch.addmm(torch.from_numpy(bias), torch.from_numpy(a), torch.from_numpy(b)))
    return _graph(nodes, ["a", "b", "bias"], ["c"]), {"a": a, "b": b, "bias": bias}, expected


def _linear():
    x, w = _normal(2, 8), _normal(4, 8)
    g = _graph([_in("x", (2, 8)), _const("w", (4, 8)), (LinearOp(has_bias=False), ["x", "w"], "y", (2, 4))], ["x"], ["y"])
    expected = _torch_to_np(torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w)))
    return g, {"x": x, "w": w}, expected


def _linear_with_bias():
    x, w, b = _normal(2, 8), _normal(4, 8), _normal(4)
    nodes = [_in("x", (2, 8)), _const("w", (4, 8)), _const("b", (4,)), (LinearOp(has_bias=True), ["x", "w", "b"], "y", (2, 4))]
    expected = _torch_to_np(torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b)))
    return _graph(nodes, ["x"], ["y"]), {"x": x, "w": w, "b": b}, expected


# --- SDPA ------------------------------------------------------------------


def _sdpa(shape, *, causal=False):
    def build():
        q, k, v = _normal(*shape), _normal(*shape), _normal(*shape)
        nodes = [_in("q", shape), _in("k", shape), _in("v", shape), (SdpaOp(is_causal=causal), ["q", "k", "v"], "out", shape)]
        expected = _torch_to_np(
            torch.nn.functional.scaled_dot_product_attention(
                torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v), is_causal=causal
            )
        )
        return _graph(nodes, ["q", "k", "v"], ["out"]), {"q": q, "k": k, "v": v}, expected

    return build


def _sdpa_gqa():
    # GQA: Q has more heads than K/V (28 Q heads, 4 KV heads). The cuda backend reaches the
    # P@V matmul via the planner's fused-prologue chain extension (``_classify_fused_prologue``).
    B, Hq, Hkv, S, D = 1, 28, 4, 8, 16
    q, k, v = _normal(B, Hq, S, D), _normal(B, Hkv, S, D), _normal(B, Hkv, S, D)
    nodes = [
        _in("q", (B, Hq, S, D)),
        _in("k", (B, Hkv, S, D)),
        _in("v", (B, Hkv, S, D)),
        (SdpaOp(), ["q", "k", "v"], "out", (B, Hq, S, D)),
    ]
    # Reference: expand K/V heads to match Q, then standard SDPA.
    group = Hq // Hkv
    k_exp = torch.from_numpy(k).repeat_interleave(group, dim=1)
    v_exp = torch.from_numpy(v).repeat_interleave(group, dim=1)
    expected = _torch_to_np(torch.nn.functional.scaled_dot_product_attention(torch.from_numpy(q), k_exp, v_exp))
    return _graph(nodes, ["q", "k", "v"], ["out"]), {"q": q, "k": k, "v": v}, expected


# --- compound graphs -------------------------------------------------------


def _softmax_graph():
    rows, cols = 4, 8
    x = _normal(rows, cols)
    nodes = [
        _in("x", (rows, cols)),
        (ReduceOp("maximum", -1), ["x"], "mx", (rows, 1)),
        (ElementwiseOp("subtract"), ["x", "mx"], "subtract", (rows, cols)),
        (ElementwiseOp("exp"), ["subtract"], "exp", (rows, cols)),
        (ReduceOp("sum", -1), ["exp"], "sm", (rows, 1)),
        (ElementwiseOp("divide"), ["exp", "sm"], "out", (rows, cols)),
    ]
    return _graph(nodes, ["x"], ["out"]), {"x": x}, _torch_to_np(torch.softmax(torch.from_numpy(x), dim=-1))


def _rmsnorm_graph():
    rows, dim, eps = 8, 64, 1e-6
    X, w = _normal(rows, dim), _normal(dim)
    nodes = [
        _in("X", (rows, dim)),
        _const("eps", value=eps),
        _in("w", (dim,)),
        (ElementwiseOp("multiply"), ["X", "X"], "sq", (rows, dim)),
        (ReduceOp("sum", axis=-1), ["sq"], "red", (rows, 1)),
        (ElementwiseOp("add"), ["red", "eps"], "ae", (rows, 1)),
        (ElementwiseOp("rsqrt"), ["ae"], "rsq", (rows, 1)),
        (ElementwiseOp("multiply"), ["X", "rsq"], "norm", (rows, dim)),
        (ElementwiseOp("multiply"), ["norm", "w"], "out", (rows, dim)),
    ]
    X_t, w_t = torch.from_numpy(X), torch.from_numpy(w)
    expected = _torch_to_np(X_t * torch.rsqrt(X_t.pow(2).sum(-1, keepdim=True) + eps) * w_t)
    return _graph(nodes, ["X", "w"], ["out"]), {"X": X, "w": w}, expected


CASES = [
    Case("pow", _pow),
    Case("add_broadcast", _add_broadcast),
    Case("reduce_sum", _reduce(ReduceOp("sum", -1), lambda t: t.sum(dim=-1, keepdim=True))),
    Case("reduce_max", _reduce(ReduceOp("maximum", -1), lambda t: t.amax(dim=-1, keepdim=True))),
    Case("mean", _reduce(MeanOp(axis=-1), lambda t: t.mean(dim=-1, keepdim=True))),
    Case("transpose", _layout(TransposeOp(axes=(1, 0)), (3, 4), (4, 3), lambda t: t.transpose(0, 1)), 1e-6, 1e-6),
    Case("transpose_perm", _layout(TransposeOp(axes=(0, 2, 1)), (2, 3, 4), (2, 4, 3), lambda t: t.permute(0, 2, 1)), 1e-6, 1e-6),
    Case("reshape", _layout(ReshapeOp(shape=(2, 6)), (3, 4), (2, 6), lambda t: t.reshape(2, 6)), 1e-6, 1e-6),
    Case("unsqueeze", _layout(UnsqueezeOp(dim=0), (4,), (1, 4), lambda t: t.unsqueeze(0)), 1e-6, 1e-6),
    Case("slice", _slice, 1e-6, 1e-6),
    Case("cat", _cat, 1e-6, 1e-6),
    Case("gather", _gather, 1e-6, 1e-6),
    Case("embedding", _embedding, 1e-6, 1e-6),
    Case("index_select_middle_axis", _index_select_middle_axis, 1e-6, 1e-6),
    Case("matmul", _matmul, 5e-5, 2e-6),
    Case("matmul_with_bias", _matmul_with_bias, 5e-5, 2e-6),
    Case("linear", _linear, 1e-4, 2e-6),
    Case("linear_with_bias", _linear_with_bias, 1e-4, 2e-6),
    Case("sdpa", _sdpa((1, 2, 4, 8)), 5e-5, 2e-6),
    # Causal masking: future positions must be masked out.
    Case("sdpa_causal", _sdpa((1, 2, 8, 16), causal=True), 1e-4, 1e-5),
    Case("sdpa_gqa", _sdpa_gqa, 5e-3, 1e-4),
    Case("softmax_graph", _softmax_graph, 2e-4, 1e-5),
    Case("rmsnorm_graph", _rmsnorm_graph, 1e-4, 1e-5),
]


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_op(case: Case, run_graph):
    graph, inputs, expected = case.build()
    np.testing.assert_allclose(_run(run_graph, graph, inputs), expected, rtol=case.rtol, atol=case.atol)
