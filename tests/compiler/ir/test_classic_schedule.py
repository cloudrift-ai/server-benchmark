"""The classic scheduling problem, sites, classification, and complete assignment contract."""

import pickle

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.classic_schedule import (
    ClassicMaterialization,
    ClassicProblem,
    ClassicSchedule,
    ClassicScheduleCodec,
    ClassicScheduleContext,
    Contraction,
    EdgeSchedule,
    EdgeSite,
    KernelSchedule,
    NodeId,
    Projection,
    ProjectionSchedule,
    Reduction,
    ReductionSchedule,
    SiteIndex,
    cartesian_assignments,
    classify,
    edge_domain,
    enumerate_classic,
    enumerate_reference,
    kernel_domain,
    node_domain,
)
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Channel, Fold, Lambda, M
from emmy.compiler.ir.schedule import PlacedTile, Raster, Reduce, ResolvedStage, Stage, Tile, Work
from emmy.compiler.ir.stmt import Assign, Body, Load


def _sum(name: str = "sum") -> Fold:
    init, combine = M(ElementwiseImpl("add"), names=(name,))
    return Fold(
        axis=Axis("k", 8),
        operands=(Load(name="x", input="x", index=(Var("k"),)),),
        lift=Lambda(params=("k", "x"), body=Body(), results=("x",)),
        init=init,
        combine=combine,
    )


def _contraction() -> Fold:
    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    return Fold.contraction(
        k_axis=Axis("k", 8),
        a=Load(name="a", input="a", index=(Var("m"), Var("k"))),
        channels=(Channel(Load(name="b", input="b", index=(Var("k"), Var("n"))), "acc"),),
    )


def _direct(context: ClassicScheduleContext) -> ClassicSchedule:
    nodes = {
        site: ProjectionSchedule(Tile()) if isinstance(view, Projection) else ReductionSchedule(Tile(), Reduce())
        for site, view in context.views.items()
    }
    return ClassicSchedule(
        kernel=KernelSchedule(work=Work(), raster=Raster()),
        nodes=nodes,
        edges={edge: EdgeSchedule(stage=Stage.direct()) for edge in context.index.edges},
    )


def test_shared_node_has_one_site_and_each_use_has_an_edge() -> None:
    shared = _sum()
    left = Fold.projection(operands=(shared,), body=Body((Assign("left", "add", ("sum", "sum")),)), results=("left",))
    right = Fold.projection(operands=(shared,), body=Body((Assign("right", "add", ("sum", "sum")),)), results=("right",))
    root = Fold.projection(operands=(left, right), body=Body(), results=("left", "right"))
    index = SiteIndex(root)

    assert len(index.nodes) == 4
    shared_site = index.site(shared)
    uses = tuple(edge for edge in index.edges if index.producer(edge) == shared_site)
    assert uses == (EdgeSite(index.site(left), 0), EdgeSite(index.site(right), 0))


def test_classification_binds_contraction_roles_to_consumer_operands() -> None:
    contraction = _contraction()
    index = SiteIndex(contraction)

    view = classify(index, index.nodes[0])
    assert view == Reduction(Contraction(a=1, channels=(0,)))


def test_classification_does_not_read_the_target() -> None:
    root = Fold.projection(body=Body((Assign("y", "add", ("x", "x")),)))
    index = SiteIndex(root)

    assert classify(index, index.nodes[0]) == Projection()


def test_context_requires_complete_node_and_edge_coverage() -> None:
    context = ClassicScheduleContext(ClassicProblem(_contraction(), target=object()))
    complete = _direct(context)
    assert context.accepts(complete)

    missing_node = ClassicSchedule(complete.kernel, {}, complete.edges)
    refusal = context.accepts(missing_node).refusal
    assert refusal is not None and refusal.reason == "missing node assignment"

    missing_edge = ClassicSchedule(complete.kernel, complete.nodes, {})
    refusal = context.accepts(missing_edge).refusal
    assert refusal is not None and refusal.reason == "missing edge assignment"


def test_context_rejects_a_node_schedule_from_the_wrong_sum_arm() -> None:
    context = ClassicScheduleContext(ClassicProblem(_sum(), target=object()))
    schedule = _direct(context)
    site = context.index.nodes[0]
    wrong = ClassicSchedule(schedule.kernel, {site: ProjectionSchedule(Tile())}, schedule.edges)

    refusal = context.accepts(wrong).refusal
    assert refusal is not None and refusal.reason == "reduction site requires a reduction schedule"


def test_node_ids_reject_negative_ordinals() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        NodeId(-1)


def _signature(schedule: ClassicSchedule) -> tuple:
    return schedule.kernel, tuple(schedule.nodes.items()), tuple(schedule.edges.items())


def test_direct_domains_are_explicit_and_independent() -> None:
    problem = ClassicProblem(_contraction(), target=object())
    context = ClassicScheduleContext(problem)
    site = context.index.nodes[0]
    edge = context.index.edges[0]

    assert kernel_domain(problem) == (KernelSchedule(Work(), Raster()),)
    assert node_domain(problem, site, context.views[site]) == (ReductionSchedule(Tile(), Reduce()),)
    assert edge_domain(problem, edge) == (EdgeSchedule(Stage.direct()),)


def test_reference_enumerator_is_the_accepted_cartesian_subset() -> None:
    problem = ClassicProblem(_contraction(), target=object())

    assignments = list(cartesian_assignments(problem))
    assert [_signature(schedule) for schedule, verdict in assignments if verdict] == [
        _signature(schedule) for schedule in enumerate_reference(problem)
    ]


def test_lazy_enumerator_is_order_independent_and_matches_reference() -> None:
    problem = ClassicProblem(_contraction(), target=object())
    context = ClassicScheduleContext(problem)
    reference = {_signature(schedule) for schedule in enumerate_reference(problem)}
    node_first = {_signature(schedule) for schedule in enumerate_classic(problem)}
    edge_first = {_signature(schedule) for schedule in enumerate_classic(problem, (*context.index.edges, *reversed(context.index.nodes)))}

    assert node_first == edge_first == reference


def test_lazy_enumerator_rejects_incomplete_or_duplicate_traversals() -> None:
    problem = ClassicProblem(_sum(), target=object())
    site = ClassicScheduleContext(problem).index.nodes[0]

    with pytest.raises(ValueError, match="exactly once"):
        list(enumerate_classic(problem, (site, site)))


def test_codec_round_trips_one_canonical_complete_row() -> None:
    problem = ClassicProblem(_contraction(), target=object())
    context = ClassicScheduleContext(problem)
    codec = ClassicScheduleCodec(problem)
    schedule = _direct(context)

    row = codec.encode(schedule)

    assert row == {
        "WORK": "",
        "RASTER": "",
        "TILE@n0": "",
        "REDUCE@n0": "",
        "STAGE@n0.e0": "",
        "STAGE@n0.e1": "",
    }
    assert codec.decode(row) == schedule


def test_codec_has_no_missing_unknown_or_alias_key_path() -> None:
    problem = ClassicProblem(_contraction(), target=object())
    codec = ClassicScheduleCodec(problem)
    row = codec.encode(_direct(codec.context))

    with pytest.raises(ValueError, match="missing STAGE@n0.e0"):
        codec.decode({key: value for key, value in row.items() if key != "STAGE@n0.e0"})
    with pytest.raises(ValueError, match="unknown keys STAGE"):
        codec.decode({**row, "STAGE": ""})


def test_schedule_and_materialization_are_pickle_safe() -> None:
    context = ClassicScheduleContext(ClassicProblem(_sum(), target=None))
    schedule = _direct(context)
    restored = pickle.loads(pickle.dumps(schedule))
    assert restored.kernel == schedule.kernel
    assert dict(restored.nodes) == dict(schedule.nodes)
    assert dict(restored.edges) == dict(schedule.edges)

    site = context.index.nodes[0]
    edge = context.index.edges[0]
    placed = Tile(regs=(2, 1)).at(Axis("m", 8), Axis("n", 8))
    resolved = ResolvedStage(Stage(depth=1, transport="smem"), ("a_smem",), 8)
    materialization = pickle.loads(pickle.dumps(ClassicMaterialization({site: placed}, {edge: resolved})))
    assert materialization.tiles[site] == placed
    assert materialization.stages[edge] == resolved
    assert isinstance(materialization.tiles[site], PlacedTile)
