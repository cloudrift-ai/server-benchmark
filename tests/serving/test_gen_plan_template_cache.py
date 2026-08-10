"""Generative split integration for structural plan-template reuse (CPU-only)."""

from __future__ import annotations

import numpy as np

from emmy.compiler.backend.plan_cache import PlanTemplateCache
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.cuda import CudaOp


def _split_graph() -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (2, 4)), node_id="x")
    g.add_node(
        op=ConstantOp(name="weight", source_path="weight", source_shape=(4, 4), source_dtype="f32"),
        inputs=[],
        output=Tensor("weight", (4, 4)),
        node_id="weight",
    )
    g.add_node(
        op=CudaOp(
            kernel_source='extern "C" __global__ void k_split() {}',
            kernel_name="k_split",
            arg_order=("x", "weight", "y"),
            grid=((1,), (1,), (1,)),
            block=((32,), (1,), (1,)),
        ),
        inputs=["x", "weight"],
        output=Tensor("y", (2, 4)),
        node_id="y",
    )
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def test_compile_split_reuses_plan_but_builds_fresh_programs_and_weights(monkeypatch):
    import torch

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.serving import gen_runner

    class Wrapper(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((4, 4), value))

        def forward(self, x):
            return x @ self.weight.T

    compile_calls = []
    builds = []

    def compile_graph(_backend, graph):
        compile_calls.append(graph)
        return graph

    def build_from_plan(_cls, plan, feed, **_kwargs):
        program = object()
        builds.append((program, plan, dict(feed)))
        return program

    monkeypatch.setattr(gen_runner, "trace_split", lambda *_args, **_kwargs: _split_graph())
    monkeypatch.setattr(CudaBackend, "compile", compile_graph)
    monkeypatch.setattr(CompiledProgram, "build_from_plan", classmethod(build_from_plan))

    cache = PlanTemplateCache()
    x = torch.zeros(2, 4)
    first, plan0 = gen_runner._compile_split(Wrapper(1.0), [x], None, np.dtype("float32"), plan_cache=cache)
    second, plan1 = gen_runner._compile_split(Wrapper(2.0), [x], None, np.dtype("float32"), plan_cache=cache)

    assert len(compile_calls) == 1
    assert (cache.hits, cache.misses) == (1, 1)
    assert first.program is builds[0][0]
    assert second.program is builds[1][0]
    assert first.program is not second.program
    assert plan0 is not plan1
    assert plan0.weights["weight"].source_path == "weight"
    assert plan1.weights["weight"].source_path == "weight"
    np.testing.assert_array_equal(builds[0][2]["weight"], np.ones((4, 4), dtype=np.float32))
    np.testing.assert_array_equal(builds[1][2]["weight"], np.full((4, 4), 2.0, dtype=np.float32))
