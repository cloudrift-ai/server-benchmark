"""One broadcast constant feeding TWO sibling cones declares its load ONCE per scope.

A ``Fold``'s operand edges splice independently (``Fold.spliced_step``), so two cones reading the
same 1-element input each carry their own copy of its ``buf[0]`` ``Load`` — same SSA name, same
``(input, index)``. Flattened into one loop body they become two C declarations of one name, which
nvcc rejects (``"in4" has already been declared in the current scope``, 11 errors on the
DeepSeek-V4 MXFP4 expert kernel, whose decode spells eleven shared scalar constants and applies
them to both halves of a fused ``gate_up`` weight).

The graph below is that shape minimized: one constant applied to a fused weight, sliced into the
two channels of one contraction. Capability-independent — it reproduced identically on sm_70,
sm_80, sm_89 and sm_120.
"""

from __future__ import annotations

import re
from importlib import import_module

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.frontend.ir import LinearOp, SliceOp
from emmy.compiler.ir.stmt import Body, Load
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.target import set_target
from tests.compiler.helpers import pin_classic

materialize = import_module("emmy.compiler.pipeline.passes.lowering.kernel.010_materialize")

_CAP = (8, 0)
_M, _K, _N = 1, 32, 16  # M=1 — the decode row, whose contraction folds serially per channel
_DECL = re.compile(r"^\s*(?:const\s+)?(?:float|double|half|__half|int|unsigned|long|bool)\s+(\w+)\s*(?:=|;)")


def _redeclared(source: str) -> list[str]:
    """Names declared twice within ONE brace scope — exactly what nvcc rejects."""
    scopes: list[dict[str, int]] = [{}]
    clashes: list[str] = []
    for line in source.splitlines():
        declared = _DECL.match(line)
        if declared and declared.group(1) in scopes[-1]:
            clashes.append(declared.group(1))
        elif declared:
            scopes[-1][declared.group(1)] = 1
        for char in line:
            if char == "{":
                scopes.append({})
            elif char == "}" and len(scopes) > 1:
                scopes.pop()
    return clashes


def _gate_up_graph() -> Graph:
    """``y = (x @ (w − c)[:N].T) · (x @ (w − c)[N:].T)`` — the fused gate/up shape, ``c`` shared."""
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (_M, _K), dtype=F16), node_id="x")
    graph.add_node(InputOp(), [], Tensor("w", (2 * _N, _K), dtype=F16), node_id="w")
    graph.add_node(InputOp(), [], Tensor("c", (1,), dtype=F16), node_id="c")
    graph.add_node(ElementwiseOp("subtract"), ["w", "c"], Tensor("wq", (2 * _N, _K), dtype=F16), node_id="wq")
    graph.add_node(SliceOp(shape=(_N, _K), dim=0, start=0), ["wq"], Tensor("wg", (_N, _K), dtype=F16), node_id="wg")
    graph.add_node(SliceOp(shape=(_N, _K), dim=0, start=_N), ["wq"], Tensor("wu", (_N, _K), dtype=F16), node_id="wu")
    graph.add_node(LinearOp(), ["x", "wg"], Tensor("yg", (_M, _N), dtype=F16), node_id="yg")
    graph.add_node(LinearOp(), ["x", "wu"], Tensor("yu", (_M, _N), dtype=F16), node_id="yu")
    graph.add_node(ElementwiseOp("multiply"), ["yg", "yu"], Tensor("y", (_M, _N), dtype=F16), node_id="y")
    graph.inputs, graph.outputs = ["x", "w", "c"], ["y"]
    return graph


@pytest.fixture
def _scalar_tier(monkeypatch):
    """Pin the mma family off — the channels then fold serially in ONE loop body, the scope that clashes."""
    pin_classic(monkeypatch, {"TILE": "", "STAGE": ""})


def test_sibling_cones_share_one_declaration_of_a_broadcast_constant(_scalar_tier) -> None:
    set_target(_CAP)
    try:
        lowered = Pipeline.build(CUDA_PASSES).run(_gate_up_graph(), ctx=Context(compute_capability=_CAP))
    finally:
        set_target(None)
    sources = {node.op.kernel_name: node.op.kernel_source for node in lowered.nodes.values() if getattr(node.op, "kernel_source", None)}

    assert sources, "the graph must lower to at least one kernel"
    assert {name: _redeclared(src) for name, src in sources.items() if _redeclared(src)} == {}
    # The two channels read the shared constant through ONE binding, not one per cone.
    assert sum(src.count("= c[0]") for src in sources.values()) == 1


def test_a_name_rebound_to_a_different_address_survives_as_the_fault_it_is() -> None:
    """Only an exact repeat is dead. A same-name rebind of a DIFFERENT address must reach nvcc."""
    zero = (Literal(0, "int"),)
    body = Body((Load(name="in0", input="a", index=zero), Load(name="in0", input="b", index=zero)))

    assert [stmt.input for stmt in materialize._drop_repeated_declarations(body)] == ["a", "b"]
