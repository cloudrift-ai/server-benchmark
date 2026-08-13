"""Loop IR fallback persistence for provenance-less golden targets."""

from __future__ import annotations

import json

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, Write
from emmy.compiler.loop_wire import loop_graph_from_wire, loop_graph_to_wire
from emmy.compiler.tensor import Tensor


def test_loop_graph_round_trip_preserves_structural_body_and_composite_dims() -> None:
    extent = Dim(BinaryExpr("*", Var("seq"), Literal(2, "int")))
    body = Body(
        (
            Loop(
                Axis("a0", extent),
                Body(
                    (
                        Load(name="in0", input="x", index=(Var("a0"),)),
                        Assign(name="v0", op=ElementwiseImpl("relu"), args=("in0",)),
                        Write(output="y", index=(Var("a0"),), value="v0"),
                    )
                ),
            ),
        )
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (extent,), "f16"), node_id="x")
    graph.add_node(LoopOp(body=body, name="k_relu"), ["x"], Tensor("y", (extent,), "f16"), node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y"]

    wire = json.loads(json.dumps(loop_graph_to_wire(graph)))
    restored = loop_graph_from_wire(wire)

    assert restored.nodes["y"].op.body == body
    assert restored.nodes["y"].op.name == "k_relu"
    assert restored.buffer("y").shape == (extent,)
    assert loop_graph_to_wire(restored) == wire
