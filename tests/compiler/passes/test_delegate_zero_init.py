"""Unit coverage for delegating an atomic output's zero-init to a predecessor kernel."""

from __future__ import annotations

from importlib import import_module

import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Write, ZeroPrologue
from emmy.compiler.pipeline import Match, Rule, RuleSkipped

delegate = import_module("emmy.compiler.pipeline.passes.lowering.cuda.005_delegate_zero_init")


def _atomic_graph(shape=(32, 512)) -> tuple[Graph, KernelOp, Match]:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (1,), F32), node_id="x")
    graph.add_node(KernelOp(name="k_predecessor"), ["x"], Tensor("mid", (1,), F32), node_id="mid")
    atomic = KernelOp(
        body=Body((Write(output="acc", index=(Literal(0, "int"),), value="v", atomic=True),)),
        name="k_atomic",
    )
    graph.add_node(
        atomic, ["mid"], Tensor("acc", tuple(Dim(size) if isinstance(size, int) else size for size in shape), F32), node_id="acc"
    )
    atomic.populate_io(graph, graph.nodes["acc"])
    match = Match(graph=graph, root_node_id="acc", rule=Rule(name="test", pattern=[]))
    return graph, atomic, match


def test_static_atomic_output_delegates_to_predecessor() -> None:
    graph, _, match = _atomic_graph()
    atomic = delegate.rewrite(match, graph.nodes["acc"])
    predecessor = graph.nodes["mid"].op
    prologues = tuple(stmt for stmt in predecessor.body if isinstance(stmt, ZeroPrologue))

    assert [(stmt.dst, stmt.words) for stmt in prologues] == [("acc", 32 * 512)]
    assert predecessor.name == "k_predecessor__zp16384"
    assert "acc" in predecessor.outputs
    assert atomic.zero_delegated == ("acc",)


def test_large_static_atomic_output_delegates_without_a_size_policy() -> None:
    graph, _, match = _atomic_graph((32, 3840))
    delegate.rewrite(match, graph.nodes["acc"])
    [prologue] = [stmt for stmt in graph.nodes["mid"].op.body if isinstance(stmt, ZeroPrologue)]
    assert prologue.words == 32 * 3840


def test_first_atomic_kernel_keeps_runtime_zero_init() -> None:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (1,), F32), node_id="x")
    atomic = KernelOp(body=Body((Write(output="acc", index=(Literal(0, "int"),), value="v", atomic=True),)), name="k_atomic")
    graph.add_node(atomic, ["x"], Tensor("acc", (32, 512), F32), node_id="acc")
    atomic.populate_io(graph, graph.nodes["acc"])
    match = Match(graph=graph, root_node_id="acc", rule=Rule(name="test", pattern=[]))

    with pytest.raises(RuleSkipped, match="first launch keeps its memset"):
        delegate.rewrite(match, graph.nodes["acc"])


def test_symbolic_atomic_output_keeps_runtime_zero_init() -> None:
    graph, _, match = _atomic_graph((32, Dim("tokens", hint=512)))
    with pytest.raises(RuleSkipped, match="symbolic / already delegated"):
        delegate.rewrite(match, graph.nodes["acc"])
    assert not any(isinstance(stmt, ZeroPrologue) for stmt in graph.nodes["mid"].op.body)
