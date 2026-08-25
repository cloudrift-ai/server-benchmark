"""Canonical Loop IR lifts completely into a Fold tree without recognition."""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure.fold import Channel, Fold
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tensor.ir import ReduceOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.tile._lift import lift_tile


def _tile(body: Body):
    return lift_tile(LoopOp(body=body))


def _matmul_body(epilogue=(), k_extent: int = 128) -> Body:
    m, n, k = Axis("m", Dim(32)), Axis("n", Dim(64)), Axis("k", Dim(k_extent))
    inner = Body(
        (
            Load(name="xv", input="x", index=(Var("m"), Var("k"))),
            Load(name="wv", input="w", index=(Var("n"), Var("k"))),
            Assign(name="prod", op=ElementwiseImpl("multiply"), args=("xv", "wv")),
            Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
        )
    )
    cell = (Loop(axis=k, body=inner), *epilogue, Write(output="out", index=(Var("m"), Var("n")), value="acc"))
    return Body((Loop(axis=m, body=Body((Loop(axis=n, body=Body(cell)),))),))


def test_matmul_cell_lifts_and_canonicalizes_as_contraction() -> None:
    tile = _tile(_matmul_body())
    assert [axis.extent for axis in tile.place.free] == [Dim(32), Dim(64)]
    node = tile.op
    assert isinstance(node, Fold) and node.axis is not None
    assert node.role is AxisRole.CONTRACTION
    assert isinstance(node.a, Load) and node.a.input == "x"
    assert isinstance(node.b, Load) and node.b.input == "w"
    assert len(tile.stores) == 1 and tile.stores[0].sweep is None


def test_epilogue_stays_in_the_projection_body() -> None:
    epilogue = (
        Load(name="bias", input="b", index=(Var("n"),)),
        Assign(name="outv", op=ElementwiseImpl("add"), args=("acc", "bias")),
    )
    m, n, k = Axis("m", Dim(32)), Axis("n", Dim(64)), Axis("k", Dim(128))
    inner = Body(
        (
            Load(name="xv", input="x", index=(Var("m"), Var("k"))),
            Load(name="wv", input="w", index=(Var("n"), Var("k"))),
            Assign(name="prod", op=ElementwiseImpl("multiply"), args=("xv", "wv")),
            Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
        )
    )
    cell = (Loop(axis=k, body=inner), *epilogue, Write(output="out", index=(Var("m"), Var("n")), value="outv"))
    tile = _tile(Body((Loop(axis=m, body=Body((Loop(axis=n, body=Body(cell)),))),)))
    node = tile.op
    assert isinstance(node, Fold) and node.axis is None and len(node.operands) == 1
    assert node.operands[0].axis is not None
    assert {stmt.input for stmt in node.body if isinstance(stmt, Load)} == {"b"}


def test_non_distributing_lift_stays_planar() -> None:
    m, n, k = Axis("m", Dim(32)), Axis("n", Dim(64)), Axis("k", Dim(128))
    inner = Body(
        (
            Load(name="xv", input="x", index=(Var("m"), Var("k"))),
            Load(name="wv", input="w", index=(Var("n"), Var("k"))),
            Assign(name="prod", op=ElementwiseImpl("maximum"), args=("xv", "wv")),
            Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
        )
    )
    cell = (Loop(axis=k, body=inner), Write(output="out", index=(Var("m"), Var("n")), value="acc"))
    tile = _tile(Body((Loop(axis=m, body=Body((Loop(axis=n, body=Body(cell)),))),)))
    node = tile.op
    assert isinstance(node, Fold) and node.axis is not None and node.role is AxisRole.PLANAR
    assert node.semiring is None


def test_contraction_formation_gates_on_the_semiring() -> None:
    import pytest

    k = Axis("k", Dim(16))
    a = Load(name="av", input="a", index=(Var("m"), Var("k")))
    b = Load(name="bv", input="b", index=(Var("n"), Var("k")))
    with pytest.raises(ValueError, match="semiring"):
        Fold.contraction(k_axis=k, a=a, channels=(Channel(b=b, acc="acc"),), product="maximum")
    node = Fold.contraction(k_axis=k, a=a, channels=(Channel(b=b, acc="acc"),))
    assert tuple(op.name for op in node.semiring) == ("multiply", "add")


def test_total_lift_fires_through_the_pipeline() -> None:
    graph = Graph()
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    graph.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    graph.inputs, graph.outputs = ["x"], ["o"]
    lowered = Pipeline.build(TILE_PASSES).run(graph)

    from emmy.compiler.ir.tile import TileOp

    assert any(isinstance(node.op, TileOp) for node in lowered.nodes.values())
