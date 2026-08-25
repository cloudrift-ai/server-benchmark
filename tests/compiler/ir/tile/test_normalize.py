from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Fold, Lambda, M
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import Placement, TileOp, lambda_equivalent_clusters
from emmy.compiler.pipeline.passes.lowering.tile._lift import lift_tile


def _planar_matmul() -> Fold:
    axis = Axis("k", Dim(32))
    body = Body(
        (
            Load(name="left", input="x", index=(Var("m"), Var("k"))),
            Load(name="right", input="w", index=(Var("n"), Var("k"))),
            Assign(name="product", op="multiply", args=("left", "right")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    return Fold(axis=axis, lift=Lambda(params=("k",), body=body, results=("product",)), init=init, combine=combine)


def test_tile_post_init_canonicalizes_contraction() -> None:
    tile = TileOp(
        op=Fold.projection(body=Body((_planar_matmul(),))),
        place=Placement(free=(Axis("m", 8), Axis("n", 16))),
    )

    assert isinstance(tile.op, Fold) and tile.op.role is AxisRole.CONTRACTION
    assert tile.op.a.input == "x"
    assert tile.op.b.input == "w"
    assert TileOp(op=tile.op, place=tile.place).op == tile.op


def test_contraction_clusters_alpha_equivalent_shared_operands() -> None:
    axis = Axis("k", Dim(32))
    body = Body(
        (
            Load(name="left0", input="x", index=(Var("m"), Var("k"))),
            Load(name="right0", input="w0", index=(Var("n"), Var("k"))),
            Assign(name="product0", op="multiply", args=("left0", "right0")),
            Load(name="left1", input="x", index=(Var("m"), Var("k"))),
            Load(name="right1", input="w1", index=(Var("n"), Var("k"))),
            Assign(name="product1", op="multiply", args=("left1", "right1")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), ElementwiseImpl("add"), names=("acc0", "acc1"))
    planar = Fold(
        axis=axis,
        lift=Lambda(params=("k",), body=body, results=("product0", "product1")),
        init=init,
        combine=combine,
    )

    tile = TileOp(
        op=Fold.projection(body=Body((planar,))),
        place=Placement(free=(Axis("m", 8), Axis("n", 16))),
    )

    contraction = tile.op.operands[0]
    assert contraction.role is AxisRole.CONTRACTION
    assert len(contraction.channels) == 2
    assert contraction.a.input == "x"


def test_lambda_equivalent_clusters_include_captured_axes() -> None:
    first = Lambda(
        params=("k",),
        body=Body((Load(name="x", input="q", index=(Var("row"), Var("k"))),)),
        results=("x",),
    )
    second = Lambda(
        params=("depth",),
        body=Body((Load(name="value", input="q", index=(Var("query"), Var("depth"))),)),
        results=("value",),
    )

    assert lambda_equivalent_clusters(((first, ("row", "k")), (second, ("unused", "query", "depth")))) == ((0, 1),)


def test_total_lift_produces_canonical_contraction() -> None:
    m, n, k = Axis("m", Dim(8)), Axis("n", Dim(16)), Axis("k", Dim(32))
    inner = Body(
        (
            Load(name="left", input="x", index=(Var("m"), Var("k"))),
            Load(name="right", input="w", index=(Var("n"), Var("k"))),
            Assign(name="product", op="multiply", args=("left", "right")),
            Accum(name="acc", value="product", op="add", axes=("k",)),
        )
    )
    body = Body(
        (
            Loop(
                axis=m,
                body=Body(
                    (Loop(axis=n, body=Body((Loop(axis=k, body=inner), Write(output="out", index=(Var("m"), Var("n")), value="acc")))),)
                ),
            ),
        )
    )

    tile = lift_tile(LoopOp(body=body))

    assert tile.op.role is AxisRole.CONTRACTION
    assert tile.op.loop.role is AxisRole.CONTRACTION
