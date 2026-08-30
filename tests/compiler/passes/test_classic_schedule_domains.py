"""Production classic scheduling obeys the independent-domain contract."""

from emmy.compiler.context import Context
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.classic_schedule import (
    ClassicProblem,
    ClassicScheduleCodec,
    ClassicScheduleContext,
    ReductionSchedule,
    enumerate_reference,
)
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Channel, Fold
from emmy.compiler.ir.schedule import Reduce
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.tile import Placement, TileOp
from emmy.compiler.pipeline.passes.lowering.tile._classic import project_domains, schedule
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from emmy.compiler.pipeline.search.space import coop_reduce_moves, scalar_tile_moves


def _signature(codec, assignment) -> tuple[tuple[str, str], ...]:
    return tuple(codec.encode(assignment).items())


def test_production_enumeration_is_the_compatible_independent_product() -> None:
    root = Fold.projection(body=Body((Assign("y", "add", ("x", "x")),)), results=("y",))
    tile = TileOp(op=root, place=Placement(free=(Axis("n", 8),)))
    target = Context.from_target((12, 0))
    problem = ClassicProblem(tile.op, target)
    domains = project_domains(tile, target)
    codec = ClassicScheduleCodec(problem, domains)

    reference = {_signature(codec, assignment) for assignment in enumerate_reference(problem, domains)}
    leaves = schedule(tile, "pointwise", {}, target)

    assert {_signature(codec, leaf.schedule) for leaf in leaves} == reference
    assert len(reference) == domains.product_size == 1
    (materialized,) = leaves[0].expand()
    assert materialized.classic == leaves[0].schedule
    assert materialized.place == tile.place.on_grid()


def test_reduction_enumeration_filters_the_independent_product_by_compatibility() -> None:
    root = fold_from_loop(
        Loop(
            axis=Axis("k", 2048),
            body=Body(
                (
                    Load(name="xv", input="x", index=(Var("k"),)),
                    Accum(name="acc", value="xv", op="add", axes=("k",)),
                )
            ),
            role=AxisRole.PLANAR,
        )
    )
    assert root is not None
    tile = TileOp(op=root, place=Placement(free=(Axis("n", 512),)))
    target = Context.from_target((12, 0))
    problem = ClassicProblem(tile.op, target)
    domains = project_domains(tile, target)
    context = ClassicScheduleContext(problem, domains)
    site = context.index.nodes[0]

    expected_reductions = {Reduce(), *coop_reduce_moves()}
    assert {choice.reduce for choice in domains.nodes[site] if isinstance(choice, ReductionSchedule)} == expected_reductions
    reference = tuple(enumerate_reference(problem, domains))
    leaves = schedule(tile, "reduce", {}, target)
    codec = ClassicScheduleCodec(problem, domains)

    assert {_signature(codec, leaf.schedule) for leaf in leaves} == {_signature(codec, assignment) for assignment in reference}
    assert len(reference) == len(expected_reductions)
    assert domains.product_size > len(reference)


def test_scalar_contraction_enumeration_is_the_compatible_independent_product() -> None:
    m, n, k = Axis("m", 64), Axis("n", 64), Axis("k", 64)
    root = Fold.contraction(
        k_axis=k,
        a=Load(name="a_e", input="a", index=(Var("m"), Var("k"))),
        channels=(Channel(b=Load(name="b_e", input="b", index=(Var("k"), Var("n"))), acc="acc"),),
    )
    tile = TileOp(op=root, place=Placement(free=(m, n)))
    target = Context.from_target((12, 0))
    problem = ClassicProblem(tile.op, target)
    domains = project_domains(tile, target)
    context = ClassicScheduleContext(problem, domains)
    site = context.index.nodes[0]

    choices = domains.nodes[site]
    assert {choice.tile for choice in choices if isinstance(choice, ReductionSchedule) and choice.reduce == Reduce()} == set(
        scalar_tile_moves()
    )
    expected_reductions = {
        Reduce(),
        *(choice for choice in coop_reduce_moves() if choice.coop <= 64 and choice.reg <= 64),
    }
    actual_reductions = {choice.reduce for choice in choices if isinstance(choice, ReductionSchedule) and not choice.tile.is_tiled}
    assert actual_reductions == expected_reductions

    reference = tuple(enumerate_reference(problem, domains))
    leaves = schedule(tile, "matmul", {}, target)
    codec = ClassicScheduleCodec(problem, domains)
    assert {_signature(codec, leaf.schedule) for leaf in leaves} == {_signature(codec, assignment) for assignment in reference}
    assert domains.product_size > len(reference) == len(choices)

    tiled = next(leaf for leaf in leaves if leaf.schedule.nodes[site].tile.is_tiled)
    materialized = tiled.expand()[0]
    assert materialized.materialization.tiles[site].choice == tiled.schedule.nodes[site].tile
