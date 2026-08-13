"""Regression: the loop-runner JIT cache must key on kernel content, not
object identity.

``execute_loop_op_cpp`` memoizes JIT-compiled kernels so repeated calls with
the same kernel don't re-invoke Cling. The key used to include ``id(loop)``,
which is a use-after-free hazard: CPython recycles the address of a GC'd
``LoopOp``, so a later same-shape LoopOp (e.g. ``tanh`` after a freed
``negative``) could alias the old id and be handed the stale cached kernel —
silently returning ``-x`` for ``tanh``. This surfaced only under randomized /
parallel test ordering (the GC timing that triggers id reuse), never in
isolation.

We force the collision deterministically by pinning every ``id`` the runner
sees to a constant, then check two structurally-different same-shape kernels
each compute their own result. Under the old id-keyed cache this fails (the
second kernel is served the first's compiled function); under content-keying
it passes. cppyy-only — no CUDA required.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp

cppyy = pytest.importorskip("cppyy")


def _build(fn: str) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    g.add_node(op=ElementwiseOp(fn), inputs=["x"], output=Tensor("y", (4, 8)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    return g


def _run_loop(graph: Graph, x: np.ndarray) -> np.ndarray:
    from emmy.compiler.backend.loop import LoopBackend

    be = LoopBackend()
    return be.run(be.compile(graph), input_data={"x": x})[0].outputs["y"]


def test_cache_not_keyed_on_object_identity(monkeypatch):
    from emmy.compiler.ir.loop import runner

    # Pin every id() the runner module evaluates to one constant, forcing the
    # exact id-collision the old cache key was vulnerable to.
    monkeypatch.setattr(runner, "id", lambda _obj: 0xC0FFEE, raising=False)
    monkeypatch.setattr(runner, "_FN_CACHE", {})

    x = np.random.default_rng(0).uniform(0.1, 5.0, size=(4, 8)).astype(np.float32)

    neg = _run_loop(_build("negative"), x)
    np.testing.assert_allclose(neg, -x, rtol=2e-5, atol=1e-5)

    # Same shape, different body — must NOT be served the cached negate kernel.
    tanh = _run_loop(_build("tanh"), x)
    np.testing.assert_allclose(tanh, np.tanh(x), rtol=2e-5, atol=1e-5)


def test_float_precision_boundaries_use_host_cpp_spelling():
    """The Loop runner's float ABI must not inherit CUDA half spellings."""
    from emmy.compiler.backend.loop import LoopBackend
    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.ir.loop.runner import render_loopop_cpp

    graph = Graph()
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("x", (8,), "f32"), node_id="x")
    graph.add_node(op=ElementwiseOp("copy"), inputs=["x"], output=Tensor("narrow", (8,), "f16"), node_id="narrow")
    graph.add_node(op=ElementwiseOp("copy"), inputs=["narrow"], output=Tensor("wide", (8,), "f32"), node_id="wide")
    graph.inputs, graph.outputs = ["x"], ["wide"]

    backend = LoopBackend()
    compiled = backend.compile(graph)
    loop = next(node.op for node in compiled.nodes.values() if isinstance(node.op, LoopOp))
    source = render_loopop_cpp(loop, "precision_copy", {"x": (8,)}, (8,))
    assert "__half" not in source
    assert "__float2half" not in source

    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    actual = backend.run(compiled, input_data={"x": x})[0].outputs["wide"]
    np.testing.assert_array_equal(actual, x)
