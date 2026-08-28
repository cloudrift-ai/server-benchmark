from __future__ import annotations

from pathlib import Path

import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Fold, Lambda, M
from emmy.compiler.ir.schedule import TilePlan
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp, lambda_equivalent_clusters
from emmy.compiler.ir.tile.path import sites
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams
from emmy.compiler.pipeline.search.golden import _lifted_target, load_golden_file, load_golden_records


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


def test_tile_post_init_recovers_an_elided_unit_contraction_row() -> None:
    axis = Axis("k", Dim(16))
    body = Body(
        (
            Load(name="left", input="x", index=(Literal(0, "int"), Var("k"))),
            Load(name="right", input="w", index=(Var("n"), Var("k"))),
            Assign(name="product", op="multiply", args=("left", "right")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    planar = Fold(axis=axis, lift=Lambda(params=("k",), body=body, results=("product",)), init=init, combine=combine)

    tile = TileOp(
        op=planar,
        place=Placement(free=(Axis("n", 16),)),
        output_specs=(OutputSpec(Write(output="out", index=(Literal(0, "int"), Var("n")), value="acc")),),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("_um", "n")
    assert tile.op.role is AxisRole.CONTRACTION


@pytest.mark.parametrize(
    "index",
    (
        (Var("n"), Literal(0, "int")),
        (Literal(0, "int"), Var("n") * 2),
    ),
    ids=("varying-coordinate-before-zero", "strided-column"),
)
def test_tile_post_init_does_not_infer_a_unit_row_from_a_non_dense_boundary(index) -> None:
    axis = Axis("k", Dim(16))
    body = Body(
        (
            Load(name="left", input="x", index=(Literal(0, "int"), Var("k"))),
            Load(name="right", input="w", index=(Var("n"), Var("k"))),
            Assign(name="product", op="multiply", args=("left", "right")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    planar = Fold(axis=axis, lift=Lambda(params=("k",), body=body, results=("product",)), init=init, combine=combine)

    tile = TileOp(
        op=planar,
        place=Placement(free=(Axis("n", 16),)),
        output_specs=(OutputSpec(Write(output="out", index=index, value="acc")),),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("n",)


def test_contraction_promotes_a_shared_store_sweep_once() -> None:
    n = Axis("n", 16)
    stores = tuple(OutputSpec(write=Write(output="out", index=(Var("m"), Var("n")), value="acc"), sweep=n) for _ in range(2))

    tile = TileOp(op=_planar_matmul(), place=Placement(free=(Axis("m", 8),)), output_specs=stores)

    assert tuple(axis.name for axis in tile.place.free) == ("m", "n")
    assert all(store.sweep is None for store in tile.output_specs)


def test_contraction_promotes_a_shared_store_sweep_after_grid_mapping() -> None:
    """A scheduled/reloaded tile keeps promotion as a construction invariant."""
    m, n = Axis("m", 8), Axis("n", 16)
    normalized = TileOp(op=_planar_matmul(), place=Placement(free=(m, n))).op
    tile = TileOp(
        op=normalized,
        place=Placement(free=(m,), grid=(m,), mapped=True),
        schedule={"TILE": TilePlan(regs=(2, 2))},
        output_specs=(OutputSpec(write=Write(output="out", index=(Var("m"), Var("n")), value="acc"), sweep=n),),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("m", "n")
    assert tuple(axis.name for axis in tile.place.grid) == ("m", "n")
    assert tile.place.is_mapped
    assert tile.schedule["TILE"] == TilePlan(regs=(2, 2))
    assert tile.output_specs[0].sweep is None


def test_nested_contraction_promotes_a_shared_store_sweep() -> None:
    """A sibling reduction can be the root-most node while a later contraction reads the sweep axis."""
    m, n = Axis("m", 8), Axis("n", 16)
    stat_axis = Axis("r", 4)
    init, combine = M(ElementwiseImpl("add"), names=("stat",))
    stat = Fold(
        axis=stat_axis,
        lift=Lambda(
            params=("r",),
            body=Body((Load(name="sample", input="s", index=(Var("m"), Var("r"))),)),
            results=("sample",),
        ),
        init=init,
        combine=combine,
    )
    root = Fold.projection(
        operands=(stat, _planar_matmul()),
        body=Body((Assign(name="result", op="add", args=("stat", "acc")),)),
        results=("result",),
    )
    tile = TileOp(
        op=root,
        place=Placement(free=(m,)),
        output_specs=(OutputSpec(write=Write(output="out", index=(Var("m"), Var("n")), value="result"), sweep=n),),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("m", "n")
    assert tile.output_specs[0].sweep is None


def test_nested_contraction_promotes_a_swept_column_beside_an_implicit_unit_row() -> None:
    """A nested linear site turns the swept column into grid placement before scheduling."""
    n, k, r = Axis("n", 16), Axis("k", 32), Axis("r", 4)
    stat_init, stat_combine = M(ElementwiseImpl("add"), names=("stat",))
    stat = Fold(
        axis=r,
        lift=Lambda(
            params=("r",),
            body=Body((Load(name="sample", input="s", index=(Var("r"),)),)),
            results=("sample",),
        ),
        init=stat_init,
        combine=stat_combine,
    )
    linear_init, linear_combine = M(ElementwiseImpl("add"), names=("acc",))
    linear = Fold(
        axis=k,
        lift=Lambda(
            params=("k",),
            body=Body(
                (
                    Load(name="left", input="x", index=(Literal(0, "int"), Var("k"))),
                    Load(name="right", input="w", index=(Var("k"), Var("n"))),
                    Assign(name="product", op="multiply", args=("left", "right")),
                )
            ),
            results=("product",),
        ),
        init=linear_init,
        combine=linear_combine,
    )
    root = Fold.projection(
        operands=(stat, linear),
        body=Body((Assign(name="result", op="add", args=("stat", "acc")),)),
        results=("result",),
    )
    tile = TileOp(
        op=root,
        output_specs=(
            OutputSpec(
                write=Write(
                    output="out",
                    index=(Literal(0, "int"), Literal(0, "int"), Var("n")),
                    value="result",
                ),
                sweep=n,
            ),
        ),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("_um", "n")
    assert tile.output_specs[0].sweep is None
    assert any(site.node.role is AxisRole.CONTRACTION for site in sites(tile.op))
    assert TileOp(op=tile.op, place=tile.place, output_specs=tile.output_specs).op is tile.op


def test_matvec_recovers_an_implicit_unit_row_through_an_output_reshape() -> None:
    """A split head/value boundary is still one varying matrix column coordinate."""
    n, k = Axis("n", 2048), Axis("k", 1024)
    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    product = Fold(
        axis=k,
        lift=Lambda(
            params=("k",),
            body=Body(
                (
                    Load(name="left", input="x", index=(Var("k"),)),
                    Load(name="right", input="w", index=(Var("k"), Var("n"))),
                    Assign(name="product", op="multiply", args=("left", "right")),
                )
            ),
            results=("product",),
        ),
        init=init,
        combine=combine,
    )
    tile = TileOp(
        op=product,
        place=Placement(free=(n,)),
        output_specs=(
            OutputSpec(
                write=Write(
                    output="out",
                    index=(Literal(0, "int"), Literal(0, "int"), Var("n") / 128, Var("n") % 128),
                    value="acc",
                )
            ),
        ),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("_um", "n")
    assert isinstance(tile.op, Fold) and tile.op.role is AxisRole.CONTRACTION


def test_promoted_attention_output_sweep_closes_the_a100_b_seam_idempotently() -> None:
    """The reduced Qwen3 target needs its promoted value-width axis to close computed B."""
    case = Path(__file__).parents[2] / "realization/cases/attention/rmsnorm-gqa-b-cut.yaml"
    (record,) = load_golden_records(load_golden_file(case))

    tile = _lifted_target(record)
    reconstructed = TileOp(op=tile.op, name=tile.name, place=tile.place, output_specs=tile.output_specs)

    assert tuple(axis.name for axis in tile.place.free) == ("a0", "a1", "a6")
    assert all(spec.sweep is None for spec in tile.output_specs)
    assert reconstructed.op is tile.op
    assert "PLACE@map.fold.a.fold.b1" in {seam.spelling for seam in cuttable_seams(tile)}


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


def test_contraction_coalesces_overlapping_equivalent_shared_operands() -> None:
    axis = Axis("k", Dim(32))
    body = Body(
        (
            Load(name="left", input="x", index=(Var("m"), Var("k"))),
            Load(name="scale", input="s", index=(Var("k"),)),
            Assign(name="scaled0", op="multiply", args=("left", "scale")),
            Assign(name="scaled1", op="multiply", args=("left", "scale")),
            Load(name="right0", input="w0", index=(Var("n"), Var("k"))),
            Load(name="right1", input="w1", index=(Var("n"), Var("k"))),
            Assign(name="product0", op="multiply", args=("scaled0", "right0")),
            Assign(name="product1", op="multiply", args=("scaled1", "right1")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), ElementwiseImpl("add"), names=("acc0", "acc1"))
    planar = Fold(
        axis=axis,
        lift=Lambda(params=("k",), body=body, results=("product0", "product1")),
        init=init,
        combine=combine,
    )

    tile = TileOp(op=planar, place=Placement(free=(Axis("m", 8), Axis("n", 16))))

    assert tile.op.role is AxisRole.CONTRACTION
    assert len(tile.op.channels) == 2
    assert tile.op.a.out == "scaled0"
    assert sum(isinstance(stmt, Assign) and stmt.name.startswith("scaled") for stmt in tile.op.a.body) == 1


def test_contraction_orients_a_shared_commutative_argument_first() -> None:
    axis = Axis("k", Dim(32))
    body = Body(
        (
            Load(name="left0", input="x0", index=(Var("m"), Var("k"))),
            Load(name="left1", input="x1", index=(Var("m"), Var("k"))),
            Load(name="right", input="w", index=(Var("k"), Var("n"))),
            Assign(name="product0", op="multiply", args=("left0", "right")),
            Assign(name="product1", op="multiply", args=("left1", "right")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), ElementwiseImpl("add"), names=("acc0", "acc1"))
    planar = Fold(
        axis=axis,
        lift=Lambda(params=("k",), body=body, results=("product0", "product1")),
        init=init,
        combine=combine,
    )

    tile = TileOp(op=planar, place=Placement(free=(Axis("m", 8), Axis("n", 16))))

    assert tile.op.role is AxisRole.CONTRACTION
    assert tuple(op.name for op in tile.op.semiring) == ("multiply", "add")
    assert tile.op.a.input == "w"
    assert [channel.b.input for channel in tile.op.channels] == ["x0", "x1"]
    assert TileOp(op=tile.op, place=tile.place).op is tile.op


def test_contraction_computes_an_equivalent_channel_once() -> None:
    axis = Axis("k", Dim(32))
    body = Body(
        (
            Load(name="left", input="x", index=(Var("m"), Var("k"))),
            Load(name="right", input="w", index=(Var("n"), Var("k"))),
            Assign(name="product0", op="multiply", args=("left", "right")),
            Assign(name="product1", op="multiply", args=("left", "right")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), ElementwiseImpl("add"), names=("acc0", "acc1"))
    planar = Fold(
        axis=axis,
        lift=Lambda(params=("k",), body=body, results=("product0", "product1")),
        init=init,
        combine=combine,
    )

    tile = TileOp(op=planar, place=Placement(free=(Axis("m", 8), Axis("n", 16))))

    assert tile.op.role is AxisRole.CONTRACTION and len(tile.op.channels) == 2
    assert len(tile.op.operands) == 2 and tile.op.channels[0].b is tile.op.channels[1].b
    assert sum(isinstance(stmt, Load) and stmt.input == "w" for stmt in tile.op.loop.body) == 1


def test_projection_keeps_only_the_maximal_shared_operand() -> None:
    small = Fold.projection(body=Body((Load(name="a", input="x", index=(Var("m"),)),)), results=("a",))
    large = Fold.projection(
        operands=(small,),
        body=Body((Assign(name="b", op="copy", args=("a",)),)),
        results=("a", "b"),
    )

    projection = Fold.projection(operands=(small, large), body=Body((Assign(name="o", op="copy", args=("b",)),)))

    assert projection.operands == (large,)
    assert projection.lift.params == ("a", "b")
    assert sum(isinstance(stmt, Load) and stmt.input == "x" for stmt in projection.lower()) == 1


def test_semiring_merges_overlapping_operand_cones_into_one_multi_result_edge() -> None:
    axis = Axis("k", Dim(32))
    body = Body(
        (
            Load(name="left", input="x", index=(Var("m"), Var("k"))),
            Load(name="scale", input="s", index=(Var("k"),)),
            Assign(name="scaled", op="multiply", args=("left", "scale")),
            Load(name="right0", input="w0", index=(Var("k"), Var("n"))),
            Load(name="right1", input="w1", index=(Var("k"), Var("n"))),
            Assign(name="product0", op="multiply", args=("left", "right0")),
            Assign(name="product1", op="multiply", args=("scaled", "right1")),
        )
    )
    init, combine = M(ElementwiseImpl("add"), ElementwiseImpl("add"), names=("acc0", "acc1"))
    planar = Fold(
        axis=axis,
        lift=Lambda(params=("k",), body=body, results=("product0", "product1")),
        init=init,
        combine=combine,
    )

    tile = TileOp(op=planar, place=Placement(free=(Axis("m", 8), Axis("n", 16))))

    shared = next(edge for edge in tile.op.operands if isinstance(edge, Fold) and edge.lift.results == ("left", "scaled"))
    assert tuple(shared.lift.results) == ("left", "scaled")
    assert tuple(tile.op.lift.params) == ("k", "right0", "left", "scaled", "right1")
    assert sum(isinstance(stmt, Load) and stmt.input == "x" for stmt in tile.op.loop.body) == 1


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


def test_contraction_preserves_computed_operand_statement_order() -> None:
    j, k = Axis("j", Dim(32)), Axis("k", Dim(64))
    max_init, max_combine = M(ElementwiseImpl("maximum"), names=("row_max",))
    row_max = Fold(
        axis=j,
        lift=Lambda(
            params=("j",),
            body=Body((Load(name="score", input="s", index=(Var("m"), Var("j"))),)),
            results=("score",),
        ),
        init=max_init,
        combine=max_combine,
    )
    sum_init, sum_combine = M(ElementwiseImpl("add"), names=("value",))
    value = Fold(
        axis=j,
        lift=Lambda(
            params=("j",),
            body=Body(
                (
                    Load(name="v", input="v", index=(Var("j"), Var("k"))),
                    Assign(name="weighted", op="multiply", args=("inv", "v")),
                )
            ),
            results=("weighted",),
        ),
        init=sum_init,
        combine=sum_combine,
    )
    outer_init, outer_combine = M(ElementwiseImpl("add"), names=("out",))
    outer = Fold(
        axis=k,
        lift=Lambda(
            params=("k",),
            body=Body(
                (
                    row_max,
                    Assign(name="inv", op="reciprocal", args=("row_max",)),
                    value,
                    Load(name="weight", input="w", index=(Var("k"), Var("n"))),
                    Assign(name="product", op="multiply", args=("value", "weight")),
                )
            ),
            results=("product",),
        ),
        init=outer_init,
        combine=outer_combine,
    )

    tile = TileOp(op=outer, place=Placement(free=(Axis("m", 8), Axis("n", 16))))

    assert tile.op.role is AxisRole.CONTRACTION
    computed = tile.op.a
    lowered = computed.lower()
    assert isinstance(lowered[0], Loop) and lowered[0].axis.name == "j"
    assert isinstance(lowered[1], Assign) and lowered[1].name == "inv"
    assert isinstance(lowered[2], Loop) and lowered[2].axis.name == "j"
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
