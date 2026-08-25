from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Fold, Lambda, M
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import Placement, TileOp, lambda_equivalent_clusters
from emmy.compiler.pipeline import Pipeline


def _lift(body: Body) -> TileOp:
    graph = Graph()
    graph.add_node(LoopOp(body=body), [], Tensor("out", (1,)), node_id="out")
    graph.outputs = ["out"]
    return Pipeline.build(["lowering/tile"], select=["lift"]).run(graph).nodes["out"].op


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


def _computed_matmul(*, computed_a: bool, computed_b: bool) -> TileOp:
    axis = Axis("k", Dim(32))
    body = []
    body.append(Load(name="left", input="x", index=(Var("m"), Var("k"))))
    left = "left"
    if computed_a:
        body.extend(
            (
                Load(name="left_scale", input="xs", index=(Var("k"),)),
                Assign(name="computed_left", op="multiply", args=(left, "left_scale")),
            )
        )
        left = "computed_left"
    body.append(Load(name="right", input="w", index=(Var("k"), Var("n"))))
    right = "right"
    if computed_b:
        body.extend(
            (
                Load(name="right_scale", input="ws", index=(Var("k"),)),
                Assign(name="computed_right", op="multiply", args=(right, "right_scale")),
            )
        )
        right = "computed_right"
    body.append(Assign(name="product", op="multiply", args=(left, right)))
    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    planar = Fold(axis=axis, lift=Lambda(params=("k",), body=Body(body), results=("product",)), init=init, combine=combine)
    return TileOp(
        op=Fold.projection(body=Body((planar,))),
        place=Placement(free=(Axis("m", 8), Axis("n", 16))),
    )


def test_contraction_factors_a_computed_operand_cone() -> None:
    tile = _computed_matmul(computed_a=True, computed_b=False)

    assert tile.op.role is AxisRole.CONTRACTION
    assert isinstance(tile.op.a, Fold) and tile.op.a.axis is None and tile.op.a.out == "computed_left"
    assert isinstance(tile.op.b, Load) and tile.op.b.input == "w"


def test_contraction_factors_b_computed_operand_cone() -> None:
    tile = _computed_matmul(computed_a=False, computed_b=True)

    assert tile.op.role is AxisRole.CONTRACTION
    assert isinstance(tile.op.a, Load) and tile.op.a.input == "x"
    assert isinstance(tile.op.b, Fold) and tile.op.b.axis is None and tile.op.b.out == "computed_right"


def test_contraction_factors_both_computed_operand_cones_idempotently() -> None:
    tile = _computed_matmul(computed_a=True, computed_b=True)

    assert tile.op.role is AxisRole.CONTRACTION
    assert isinstance(tile.op.a, Fold) and isinstance(tile.op.b, Fold)
    assert TileOp(op=tile.op, place=tile.place).op is tile.op


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

    tile = _lift(body)

    assert tile.op.role is AxisRole.CONTRACTION
    assert tile.op.loop.role is AxisRole.CONTRACTION
