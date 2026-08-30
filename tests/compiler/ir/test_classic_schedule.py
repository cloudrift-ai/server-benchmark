"""The classic scheduling problem, sites, classification, and complete assignment contract."""

import json
import pickle
from itertools import permutations

import pytest

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.classic_schedule import (
    ClassicDomains,
    ClassicMaterialization,
    ClassicProblem,
    ClassicSchedule,
    ClassicScheduleCodec,
    ClassicScheduleContext,
    Contraction,
    EdgeSchedule,
    EdgeSite,
    KernelSchedule,
    LocalSupport,
    NodeId,
    NodeSite,
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
from emmy.compiler.ir.schedule import PlacedTile, Placement, Raster, Reduce, ResolvedStage, Stage, Tile, Work
from emmy.compiler.ir.stmt import Assign, Body, Load
from emmy.compiler.ir.tile import TileOp


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


def test_context_owns_worker_and_transport_compatibility() -> None:
    context = ClassicScheduleContext(ClassicProblem(_contraction(), target=object()))
    direct = _direct(context)
    site = context.index.nodes[0]

    wrong_work = ClassicSchedule(KernelSchedule(Work.parse("t2"), Raster()), direct.nodes, direct.edges)
    refusal = context.accepts(wrong_work).refusal
    assert refusal is not None and refusal.reason == "kernel WORK does not realize the node choices"

    tiled = Tile(units=(1, 2))
    edges = dict(direct.edges)
    edges[context.index.edges[0]] = EdgeSchedule(Stage())
    mixed_transport = ClassicSchedule(
        KernelSchedule(Work.parse("t2"), Raster()),
        {site: ReductionSchedule(tiled, Reduce())},
        edges,
    )
    refusal = context.accepts(mixed_transport).refusal
    assert refusal is not None and refusal.reason == "one contraction currently requires one transport choice across its operands"


def _finite_domains(problem: ClassicProblem) -> ClassicDomains:
    context = ClassicScheduleContext(problem)
    site = context.index.nodes[0]
    direct_node = ReductionSchedule(Tile(), Reduce())
    tiled_node = ReductionSchedule(Tile(units=(1, 2)), Reduce())
    direct_edges = {edge: EdgeSchedule(Stage.direct()) for edge in context.index.edges}
    staged_edges = {edge: EdgeSchedule(Stage()) for edge in context.index.edges}
    return ClassicDomains(
        kernel=(
            KernelSchedule(Work(), Raster()),
            KernelSchedule(Work.parse("t2"), Raster()),
            KernelSchedule(Work.parse("t2"), Raster("m", 8)),
        ),
        nodes={site: (direct_node, tiled_node)},
        edges={edge: (direct_edges[edge], staged_edges[edge]) for edge in context.index.edges},
        supports={
            site: (
                LocalSupport(direct_node, direct_edges),
                LocalSupport(tiled_node, staged_edges, work=Work.parse("t2"), raster_eligible=True),
            )
        },
    )


def _schedule_signature(schedule: ClassicSchedule) -> tuple:
    return schedule.kernel, tuple(sorted(schedule.nodes.items())), tuple(sorted(schedule.edges.items()))


def test_domains_are_independent_projections_of_static_support() -> None:
    problem = ClassicProblem(_contraction(), target=object())
    context = ClassicScheduleContext(problem)
    domains = _finite_domains(problem)
    site = context.index.nodes[0]
    edge = context.index.edges[0]

    assert kernel_domain(problem, domains) == domains.kernel
    assert node_domain(problem, site, context.views[site], domains) == domains.nodes[site]
    assert edge_domain(problem, edge, domains) == domains.edges[edge]


def test_context_indexes_finite_domain_membership(monkeypatch) -> None:
    problem = ClassicProblem(_contraction(), target=object())
    domains = _finite_domains(problem)
    many_kernel_choices = tuple(KernelSchedule(Work(kind="thread", units=(width, 1)), Raster()) for width in range(1, 65)) + (
        KernelSchedule(Work(), Raster()),
    )
    domains = ClassicDomains(many_kernel_choices, domains.nodes, domains.edges, domains.supports)
    context = ClassicScheduleContext(problem, domains)
    equals = KernelSchedule.__eq__
    calls = 0

    def counted(left, right):
        nonlocal calls
        calls += 1
        return equals(left, right)

    monkeypatch.setattr(KernelSchedule, "__eq__", counted)

    assert context.accepts(_direct(context))
    assert calls <= 2


def test_reference_is_the_compatible_cartesian_subset() -> None:
    problem = ClassicProblem(_contraction(), target=object())
    domains = _finite_domains(problem)

    assignments = list(cartesian_assignments(problem, domains))

    assert {_schedule_signature(schedule) for schedule, verdict in assignments if verdict} == {
        _schedule_signature(schedule) for schedule in enumerate_reference(problem, domains)
    }
    assert len(assignments) == domains.product_size == 24


def test_every_lazy_traversal_equals_the_cartesian_reference() -> None:
    problem = ClassicProblem(_contraction(), target=object())
    context = ClassicScheduleContext(problem)
    domains = _finite_domains(problem)
    reference = {_schedule_signature(schedule) for schedule in enumerate_reference(problem, domains)}

    for traversal in permutations((*context.index.nodes, *context.index.edges)):
        assert {_schedule_signature(schedule) for schedule in enumerate_classic(problem, traversal, domains)} == reference


def test_lazy_enumerator_rejects_incomplete_or_duplicate_traversals() -> None:
    problem = ClassicProblem(_sum(), target=object())
    site = ClassicScheduleContext(problem).index.nodes[0]

    with pytest.raises(ValueError, match="exactly once"):
        list(enumerate_classic(problem, (site, site)))


def test_node_ids_reject_negative_ordinals() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        NodeId(-1)
    with pytest.raises(ValueError, match="non-negative"):
        NodeId(True)
    with pytest.raises(ValueError, match="n<ordinal>"):
        NodeId.parse("n00")


def test_edge_sites_have_one_spelling() -> None:
    edge = EdgeSite(NodeSite(NodeId(3)), 2)
    assert EdgeSite.parse(edge.spell()) == edge
    with pytest.raises(ValueError, match="n<ordinal>.e<operand>"):
        EdgeSite.parse("n3.e02")


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
    assert tuple(row) == codec.keys()
    assert codec.decode(row) == schedule


def test_codec_decode_checks_compatibility_once(monkeypatch) -> None:
    codec = ClassicScheduleCodec(ClassicProblem(_contraction(), target=object()))
    schedule = _direct(codec.context)
    row = codec.encode(schedule)
    accepts = codec.context.accepts
    calls = 0

    def counted(candidate):
        nonlocal calls
        calls += 1
        return accepts(candidate)

    monkeypatch.setattr(codec.context, "accepts", counted)

    assert codec.decode(row) == schedule
    assert calls == 1


def test_codec_resolves_explicit_unit_register_tile_against_kernel_work() -> None:
    codec = ClassicScheduleCodec(ClassicProblem(_contraction(), target=object()))
    row = codec.encode(_direct(codec.context))
    row["WORK"] = "t4x2"
    row["TILE@n0"] = "f1"

    schedule = codec.decode(row)

    assert schedule.nodes[codec.context.index.nodes[0]].tile == Tile(units=(2, 4))


def test_codec_has_no_missing_unknown_or_alias_key_path() -> None:
    problem = ClassicProblem(_contraction(), target=object())
    codec = ClassicScheduleCodec(problem)
    row = codec.encode(_direct(codec.context))

    with pytest.raises(ValueError, match="missing STAGE@n0.e0"):
        codec.decode({key: value for key, value in row.items() if key != "STAGE@n0.e0"})
    with pytest.raises(ValueError, match="unknown keys STAGE"):
        codec.decode({**row, "STAGE": ""})


def test_codec_rejects_a_noncanonical_value_spelling() -> None:
    codec = ClassicScheduleCodec(ClassicProblem(_contraction(), target=object()))
    row = codec.encode(_direct(codec.context))

    with pytest.raises(ValueError, match="not canonical"):
        codec.decode({**row, "WORK": "t04"})


def test_context_enforces_kernel_resource_and_producer_band_invariants() -> None:
    context = ClassicScheduleContext(ClassicProblem(_contraction(), target=object()))
    direct = _direct(context)
    site = context.index.nodes[0]
    atom = ATOM_REGISTRY["mma_m16n8k16_f16_f32"]

    oversized = ClassicSchedule(
        KernelSchedule(Work.parse("w33x1"), Raster()),
        {site: ReductionSchedule(Tile(atom=atom, units=(33, 1)), Reduce())},
        direct.edges,
    )
    assert context.accepts(oversized).refusal.reason == "worker inventory exceeds the target thread limit"

    too_many_producers = ClassicSchedule(
        KernelSchedule(Work.parse("w1x1+p2"), Raster()),
        {site: ReductionSchedule(Tile(atom=atom), Reduce())},
        direct.edges,
    )
    assert context.accepts(too_many_producers).refusal.reason == "producer band cannot outnumber the compute band"

    no_tma = ClassicSchedule(
        KernelSchedule(Work.parse("w1x1+p1"), Raster()),
        {site: ReductionSchedule(Tile(atom=atom), Reduce())},
        direct.edges,
    )
    assert context.accepts(no_tma).refusal.reason == "a producer band requires TMA transport at every tiled consumer"


def test_context_requires_a_tiled_contraction_for_grouped_raster() -> None:
    context = ClassicScheduleContext(ClassicProblem(_contraction(), target=object()))
    direct = _direct(context)
    grouped = ClassicSchedule(KernelSchedule(Work(), Raster("m", 8)), direct.nodes, direct.edges)

    assert context.accepts(grouped).refusal.reason == "RASTER requires a tiled contraction site"


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


def test_schedule_and_materialization_reject_untyped_entries() -> None:
    context = ClassicScheduleContext(ClassicProblem(_sum(), target=None))
    schedule = _direct(context)

    with pytest.raises(TypeError, match="kernel assignment must be KernelSchedule"):
        ClassicSchedule(object(), schedule.nodes, schedule.edges)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="node assignments must be keyed by NodeSite"):
        ClassicSchedule(schedule.kernel, {object(): next(iter(schedule.nodes.values()))}, schedule.edges)  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="node assignments must contain"):
        ClassicSchedule(schedule.kernel, {context.index.nodes[0]: object()}, schedule.edges)  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="tiles must map NodeSite to PlacedTile"):
        ClassicMaterialization({context.index.nodes[0]: Tile()}, {})  # type: ignore[dict-item]


def test_tile_requires_complete_materialization() -> None:
    root = _contraction()
    context = ClassicScheduleContext(ClassicProblem(root, target=None))
    site = context.index.nodes[0]
    plan = Tile(units=(1, 2))
    schedule = ClassicSchedule(
        KernelSchedule(Work.parse("t2"), Raster()),
        {site: ReductionSchedule(plan, Reduce())},
        {edge: EdgeSchedule(Stage.direct()) for edge in context.index.edges},
    )

    with pytest.raises(ValueError, match="exactly the tiled node sites"):
        TileOp(op=root, classic=schedule, materialization=ClassicMaterialization({}, {}))


def test_tile_graph_round_trip_uses_the_strict_schedule_codec() -> None:
    import copy

    root = _contraction()
    m, n = Axis("m", 8), Axis("n", 8)
    context = ClassicScheduleContext(ClassicProblem(root, target=None))
    site = context.index.nodes[0]
    plan = Tile(units=(1, 2))
    schedule = ClassicSchedule(
        KernelSchedule(Work.parse("t2"), Raster()),
        {site: ReductionSchedule(plan, Reduce())},
        {edge: EdgeSchedule(Stage.direct()) for edge in context.index.edges},
    )
    tile = TileOp(
        op=root,
        name="classic",
        place=Placement(free=(m, n), grid=(m, n), mapped=True),
        classic=schedule,
        materialization=ClassicMaterialization({site: plan.at(m, n)}, {}),
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("a", (8, 8)), node_id="a")
    graph.add_node(InputOp(), [], Tensor("b", (8, 8)), node_id="b")
    graph.add_node(tile, ["a", "b"], Tensor("out", (8, 8)), node_id="out")

    payload = json.loads(json.dumps(graph.to_dict(), default=str))
    restored = Graph.from_dict(payload).nodes["out"].op

    assert restored.classic == schedule
    assert restored.materialization == tile.materialization
    assert "schedule" not in payload["nodes"]["out"]["op_fields"]

    unknown = copy.deepcopy(payload)
    unknown["nodes"]["out"]["op_fields"]["materialization"]["alias"] = {}
    with pytest.raises(ValueError, match="unknown fields alias"):
        Graph.from_dict(unknown)

    missing = copy.deepcopy(payload)
    del missing["nodes"]["out"]["op_fields"]["materialization"]["stages"]
    with pytest.raises(ValueError, match="missing stages"):
        Graph.from_dict(missing)
