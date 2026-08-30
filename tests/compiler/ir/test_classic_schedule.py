"""The classic scheduling problem, sites, classification, and complete assignment contract."""

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.classic_schedule import (
    ClassicProblem,
    ClassicSchedule,
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
    classify,
)
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Channel, Fold, Lambda, M
from emmy.compiler.ir.schedule import Reduce, Tile
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
        kernel=KernelSchedule(work=None, raster=None),
        nodes=nodes,
        edges={edge: EdgeSchedule(stage=None) for edge in context.index.edges},
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
