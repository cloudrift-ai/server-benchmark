"""The classic scheduling problem, sites, classification, and complete assignment contract."""

import json
import pickle
from dataclasses import FrozenInstanceError, replace
from itertools import permutations

import pytest

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.schedule import (
    PlacedTile,
    Placement,
    Raster,
    Reduce,
    ResolvedStage,
    Schedule,
    ScheduleContext,
    ScheduleRefused,
    Stage,
    Tile,
    Work,
)
from emmy.compiler.ir.schedule import (
    schedule as advance_schedule,
)
from emmy.compiler.ir.schedule.classic import (
    ClassicAssignment,
    ClassicDomains,
    ClassicMaterialization,
    ClassicScheduleCodec,
    ClassicScheduleContext,
    EdgeSchedule,
    KernelSchedule,
    ProjectionSchedule,
    ReductionSchedule,
    edge_site_spelling,
    node_id_spelling,
    parse_edge_site,
    parse_node_id,
)
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.tile import TileOp, blockify
from emmy.compiler.pipeline.fork import DeferredFork, iter_leaves, schedule_forks
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from tests.compiler.helpers import classic_cartesian_assignments, enumerate_classic_reference
from tests.compiler.terms import contraction, projection

_K = Axis("k", 8)


def _sum(name: str = "sum") -> Fold:
    """``name = Σ_k x[k]`` — lifted from its Loop IR, so the term is the compiler's own spelling."""
    load = Load(name="x", input="x", index=(Var("k"),))
    return fold_from_loop(Loop(axis=_K, body=Body((load, Accum(name=name, value="x", op="add", axes=("k",))))))


def _contraction() -> Fold:
    return contraction(
        _K, Load(name="a", input="a", index=(Var("m"), Var("k"))), (Load(name="b", input="b", index=(Var("k"), Var("n"))), "acc")
    )


def _problem(root: Fold, target=None) -> tuple[TileOp, object]:
    """The ``(tile, target)`` problem the schedule model tests compose over."""
    m, n = Axis("m", 8), Axis("n", 8)
    tile = TileOp(op=root, place=Placement(free=(m, n)), axes=(m, n, _K))
    tile = replace(tile, blocks=blockify(tile))
    return tile, Context.from_target((12, 0)) if target is None else target


def _leaf_nodes(tile: TileOp) -> dict:
    """The one schedule every zero-axis site takes — a slab is a site of its own, so a hand-built
    assignment covers it beside the reduce site it feeds."""
    return {site: ProjectionSchedule(Tile()) for site, view in enumerate(tile.views) if view.axis is None}


def _direct(context: ClassicScheduleContext) -> ClassicAssignment:
    nodes = {**_leaf_nodes(context.tile_op), **{site: ReductionSchedule(Tile(), Reduce()) for site in _reduce_sites(context.tile_op)}}
    return Schedule(
        kernel=KernelSchedule(work=Work(), raster=Raster()),
        nodes=nodes,
        edges={edge: EdgeSchedule(stage=Stage.direct()) for edge in context.tile_op.edge_sites},
    )


def _reduce_sites(tile: TileOp) -> tuple[int, ...]:
    return tuple(site for site, view in enumerate(tile.views) if view.axis is not None)


def test_shared_node_has_one_site_and_each_use_has_an_edge() -> None:
    shared = _sum()
    left = projection((shared,), (Assign("left", "add", ("sum", "sum")),), ("left",))
    right = projection((shared,), (Assign("right", "add", ("sum", "sum")),), ("right",))
    root = projection((left, right), (), ("left", "right"))
    inventory = TileOp(op=root, axes=(_K,))

    assert len(inventory.sites) == 4  # root, left, right, the shared sum — the slab it reads is no site
    assert inventory.node_id(shared) == inventory.node_id(shared)  # one site, however many uses
    uses = tuple(edge for edge in inventory.edge_sites if inventory.sites[edge[0]].node.operands[edge[1]] is shared)
    assert uses == ((inventory.node_id(left), 0), (inventory.node_id(right), 0))


def test_classification_binds_contraction_roles_to_consumer_operands() -> None:
    """A site's classification is the term itself: its bilinear reading names the pair, and the
    roles are the consumer's operand positions — A at ``operands[0]``, the channel's B after it."""
    inventory = TileOp(op=_contraction(), axes=(_K,))

    view = inventory.views[0]
    assert view.as_contraction() is not None and view.axis == "k"
    a, b = (edge.as_slab().load.input for edge in view.operands)
    assert (a, b) == ("a", "b") and inventory.edge_sites == ((0, 0), (0, 1))
    assert inventory.node_sites == (0,)  # the two slabs carry no schedule of their own: no sites


def test_classification_does_not_read_the_target() -> None:
    root = projection((), (Load(name="x", input="x", index=(Var("n"),)), Assign("y", "add", ("x", "x"))))
    assert root.axis is None and root.as_contraction() is None
    assert TileOp(op=root, axes=(Axis("n", 8),)).views == (root,)


def test_context_requires_complete_node_and_edge_coverage() -> None:
    context = ClassicScheduleContext(*_problem(_contraction()))
    complete = _direct(context)
    assert context.extend(complete).assignment == complete

    missing_node = Schedule(complete.kernel, {}, complete.edges)
    with pytest.raises(ScheduleRefused, match="missing node assignment"):
        context.extend(missing_node)

    missing_edge = Schedule(complete.kernel, complete.nodes, {})
    with pytest.raises(ScheduleRefused, match="missing edge assignment"):
        context.extend(missing_edge)


def test_context_rejects_a_node_schedule_from_the_wrong_sum_arm() -> None:
    context = ClassicScheduleContext(*_problem(_sum()))
    schedule = _direct(context)
    site = context.tile_op.node_sites[0]
    wrong = Schedule(schedule.kernel, {**schedule.nodes, site: ProjectionSchedule(Tile())}, schedule.edges)

    with pytest.raises(ScheduleRefused, match="reduction site requires a reduction schedule"):
        context.extend(wrong)


def test_context_owns_worker_and_transport_compatibility() -> None:
    context = ClassicScheduleContext(*_problem(_contraction()))
    direct = _direct(context)
    site = context.tile_op.node_sites[0]

    wrong_work = Schedule(KernelSchedule(Work.parse("t2"), Raster()), direct.nodes, direct.edges)
    with pytest.raises(ScheduleRefused, match="kernel WORK does not realize the node choices"):
        context.extend(wrong_work)

    tiled = Tile(units=(1, 2))
    edges = dict(direct.edges)
    edges[context.tile_op.edge_sites[0]] = EdgeSchedule(Stage())
    mixed_transport = Schedule(
        KernelSchedule(Work.parse("t2"), Raster()),
        {**direct.nodes, site: ReductionSchedule(tiled, Reduce())},
        edges,
    )
    with pytest.raises(ScheduleRefused, match="one contraction currently requires one transport choice across its operands"):
        context.extend(mixed_transport)


def test_independent_nodes_compose_only_at_matching_physical_axis_geometry() -> None:
    """Algebraically reversed axes compose only when their physical widths still agree."""
    m, n = Axis("m", 8), Axis("n", 8)
    k1, k2 = Axis("k1", 8), Axis("k2", 8)
    first = contraction(k1, Load("a1", "a1", (Var("m"), Var("k1"))), (Load("b1", "b1", (Var("k1"), Var("n"))), "left"))
    second = contraction(k2, Load("a2", "a2", (Var("n"), Var("k2"))), (Load("b2", "b2", (Var("k2"), Var("m"))), "right"))
    root = projection((first, second), (), ("left", "right"))
    source = TileOp(op=root, place=Placement(free=(m, n)), axes=(m, n, k1, k2))
    problem = (replace(source, blocks=blockify(source)), Context.from_target((12, 0)))

    def pick(context, node, choice):
        site = context.site(node)
        return Schedule(
            None,
            {site: choice},
            {edge: EdgeSchedule(Stage.direct()) for edge in context.incident_edges(site)},
        )

    def contract(context, node, choice):
        """Pick ``node``'s reduction — its slabs are no sites, so nothing else is left to pick under it."""
        return context.extend(pick(context, node, choice))

    context = ClassicScheduleContext(*problem)
    context = context.extend(pick(context, root, ProjectionSchedule(Tile())))
    context = contract(context, first, ReductionSchedule(Tile(regs=(1, 2)), Reduce()))
    assert contract(context, second, ReductionSchedule(Tile(regs=(2, 1)), Reduce()))
    with pytest.raises(ScheduleRefused, match="pick disagrees on physical-axis geometry"):
        context.extend(pick(context, second, ReductionSchedule(Tile(regs=(1, 2)), Reduce())))


def _finite_domains(problem: tuple[TileOp, object]) -> ClassicDomains:
    context = ClassicScheduleContext(*problem)
    site = context.tile_op.node_sites[0]
    direct_node = ReductionSchedule(Tile(), Reduce())
    tiled_node = ReductionSchedule(Tile(units=(1, 2)), Reduce())
    direct_edges = {edge: EdgeSchedule(Stage.direct()) for edge in context.tile_op.edge_sites}
    staged_edges = {edge: EdgeSchedule(Stage()) for edge in context.tile_op.edge_sites}
    return ClassicDomains(
        kernel=(
            KernelSchedule(Work(), Raster()),
            KernelSchedule(Work.parse("t2"), Raster()),
            KernelSchedule(Work.parse("t2"), Raster("m", 8)),
        ),
        nodes={site: (direct_node, tiled_node), **{leaf: (choice,) for leaf, choice in _leaf_nodes(context.tile_op).items()}},
        edges={edge: (direct_edges[edge], staged_edges[edge]) for edge in context.tile_op.edge_sites},
    )


def _schedule_signature(schedule: ClassicAssignment) -> tuple:
    return schedule.kernel, tuple(sorted(schedule.nodes.items())), tuple(sorted(schedule.edges.items()))


def _enumerate_context(context: ScheduleContext):
    for value in advance_schedule(context):
        if isinstance(value, ScheduleContext):
            yield from _enumerate_context(value)
        else:
            yield value


def test_domains_are_independent_projections_of_static_support() -> None:
    problem = _problem(_contraction())
    context = ClassicScheduleContext(*problem)
    domains = _finite_domains(problem)
    site = context.tile_op.node_sites[0]
    edge = context.tile_op.edge_sites[0]

    projected = ClassicScheduleContext(*problem, domains)
    assert projected.kernels == domains.kernel
    assert projected.node_choices(site) == domains.nodes[site]
    assert projected.edge_choices(edge) == domains.edges[edge]


def test_context_indexes_finite_domain_membership(monkeypatch) -> None:
    problem = _problem(_contraction())
    domains = _finite_domains(problem)
    many_kernel_choices = tuple(KernelSchedule(Work(kind="thread", units=(width, 1)), Raster()) for width in range(1, 65)) + (
        KernelSchedule(Work(), Raster()),
    )
    domains = ClassicDomains(many_kernel_choices, domains.nodes, domains.edges)
    context = ClassicScheduleContext(*problem, domains)
    equals = KernelSchedule.__eq__
    calls = 0

    def counted(left, right):
        nonlocal calls
        calls += 1
        return equals(left, right)

    monkeypatch.setattr(KernelSchedule, "__eq__", counted)

    direct = _direct(context)
    assert context.extend(direct).assignment == direct
    assert calls <= 2


def test_reference_is_the_compatible_cartesian_subset() -> None:
    problem = _problem(_contraction())
    domains = _finite_domains(problem)

    context = ClassicScheduleContext(*problem, domains)
    assignments = list(classic_cartesian_assignments(context))

    assert {_schedule_signature(schedule) for schedule, verdict in assignments if verdict} == {
        _schedule_signature(schedule) for schedule in enumerate_classic_reference(context)
    }
    assert len(assignments) == domains.product_size == 24


def test_every_lazy_traversal_equals_the_cartesian_reference() -> None:
    problem = _problem(_contraction())
    domains = _finite_domains(problem)
    context = ClassicScheduleContext(*problem, domains)
    reference = {_schedule_signature(schedule) for schedule in enumerate_classic_reference(context)}

    for traversal in permutations(context.tile_op.node_sites):
        reordered = ClassicScheduleContext(*problem, domains, order=traversal)
        assert {_schedule_signature(schedule) for schedule in _enumerate_context(reordered)} == reference


def test_every_lazy_traversal_equals_algorithm_one_under_pinned_c() -> None:
    problem = _problem(_contraction())
    domains = _finite_domains(problem)
    pins = {family: () for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")}
    pins["RASTER"] = (("RASTER", ""),)
    context = ClassicScheduleContext(*problem, domains).restrict(pins)
    reference = {_schedule_signature(schedule) for schedule in enumerate_classic_reference(context)}

    for traversal in permutations(context.tile_op.node_sites):
        actual = _enumerate_context(ClassicScheduleContext(*problem, domains, order=traversal).restrict(pins))
        assert {_schedule_signature(schedule) for schedule in actual} == reference


def test_extend_accepts_a_complete_schedule_at_the_root_or_matching_prefix() -> None:
    problem = _problem(_contraction())
    domains = _finite_domains(problem)
    context = ClassicScheduleContext(*problem, domains)
    wanted = next(enumerate_classic_reference(context))

    assert context.extend(wanted).assignment == wanted

    prefix = context.extend(
        next(pick for pick in context.extensions() if pick.nodes == {0: wanted.nodes[0]} and pick.edges == wanted.edges)
    )
    assert prefix.extend(wanted).assignment == wanted


def test_context_rejects_incomplete_or_duplicate_composition_orders() -> None:
    problem = _problem(_sum())
    site = ClassicScheduleContext(*problem).tile_op.node_sites[0]

    with pytest.raises(ValueError, match="exactly once"):
        ClassicScheduleContext(*problem, order=(site, site))


def test_an_authored_tile_bypasses_enumeration_precision_policy() -> None:
    root = _contraction()
    m, n = Axis("m", 8), Axis("n", 8)
    source = TileOp(
        op=root,
        place=Placement(free=(m, n)),
        axes=(m, n, Axis("k", 16)),
        inputs={"a": Tensor("a", (8, 16), "f16"), "b": Tensor("b", (16, 8), "f16")},
        outputs={"out": Tensor("out", (8, 8), "f16")},
    )
    source = replace(source, blocks=blockify(source))
    problem = (source, Context.from_target((12, 0)))
    base = ClassicScheduleContext(*problem)
    site = base.tile_op.node_sites[0]
    tile = Tile(atom=ATOM_REGISTRY["mma_m16n8k16_f16_f16"], units=(1, 4), regs=(2, 2))
    node = ReductionSchedule(tile, Reduce())
    nodes = {**_leaf_nodes(base.tile_op), site: node}
    edges = {edge: EdgeSchedule(Stage.direct()) for edge in base.tile_op.edge_sites}
    domains = ClassicDomains(
        kernel=(KernelSchedule(Work.parse("w1x4"), Raster()),),
        nodes={site: (choice,) for site, choice in nodes.items()},
        edges={edge: (choice,) for edge, choice in edges.items()},
    )
    schedule = Schedule(domains.kernel[0], nodes, edges)

    policy_only = ClassicScheduleContext(*problem, domains).restrict({}, allow_f16_accumulate=False)
    with pytest.raises(ScheduleRefused, match="precision restriction"):
        policy_only.extend(schedule)

    authored = ClassicScheduleContext(*problem, domains).restrict(
        {"TILE": (("TILE", tile.spell()),)},
        allow_f16_accumulate=False,
    )
    assert authored.extend(schedule).assignment == schedule


def test_node_ids_are_integers_with_one_wire_spelling() -> None:
    assert parse_node_id(node_id_spelling(3)) == 3
    with pytest.raises(ValueError, match="non-negative"):
        node_id_spelling(-1)
    with pytest.raises(ValueError, match="non-negative"):
        node_id_spelling(True)
    with pytest.raises(ValueError, match="n<ordinal>"):
        parse_node_id("n00")


def test_edge_sites_have_one_spelling() -> None:
    edge = (3, 2)
    assert parse_edge_site(edge_site_spelling(edge)) == edge
    with pytest.raises(ValueError, match="n<ordinal>.e<operand>"):
        parse_edge_site("n3.e02")


def test_codec_round_trips_one_canonical_complete_row() -> None:
    problem = _problem(_contraction())
    context = ClassicScheduleContext(*problem)
    codec = ClassicScheduleCodec(context)
    schedule = _direct(context)

    row = codec.encode(schedule)

    assert row == {
        "WORK": "",
        "RASTER": "",
        "TILE": "",
        "REDUCE": "",
        "STAGE": "",
    }
    assert tuple(row) == codec.keys()
    assert codec.decode(row) == schedule


def test_codec_decode_checks_compatibility_once(monkeypatch) -> None:
    codec = ClassicScheduleCodec(ClassicScheduleContext(*_problem(_contraction())))
    schedule = _direct(codec.context)
    row = codec.encode(schedule)
    extend = ClassicScheduleContext.extend
    calls = 0

    def counted(context, candidate):
        nonlocal calls
        calls += 1
        return extend(context, candidate)

    monkeypatch.setattr(ClassicScheduleContext, "extend", counted)

    assert codec.decode(row) == schedule
    assert calls == 1


def test_codec_resolves_explicit_unit_register_tile_against_kernel_work() -> None:
    codec = ClassicScheduleCodec(ClassicScheduleContext(*_problem(_contraction())))
    row = codec.encode(_direct(codec.context))
    row["WORK"] = "t4x2"
    row["TILE"] = "f1"

    schedule = codec.decode(row)

    assert schedule.nodes[codec.context.tile_op.node_sites[0]].tile == Tile(units=(2, 4))


def test_codec_has_no_missing_unknown_or_alias_key_path() -> None:
    problem = _problem(_contraction())
    codec = ClassicScheduleCodec(ClassicScheduleContext(*problem))
    row = codec.encode(_direct(codec.context))

    with pytest.raises(ValueError, match="missing STAGE"):
        codec.decode({key: value for key, value in row.items() if key != "STAGE"})
    with pytest.raises(ValueError, match="unknown keys STAGE@map.1/inner"):
        codec.decode({**row, "STAGE@map.1/inner": ""})


def test_codec_rejects_a_noncanonical_value_spelling() -> None:
    codec = ClassicScheduleCodec(ClassicScheduleContext(*_problem(_contraction())))
    row = codec.encode(_direct(codec.context))

    with pytest.raises(ValueError, match="not canonical"):
        codec.decode({**row, "WORK": "t04"})


def test_context_enforces_kernel_resource_and_producer_band_invariants() -> None:
    context = ClassicScheduleContext(*_problem(_contraction()))
    direct = _direct(context)
    site = context.tile_op.node_sites[0]
    atom = ATOM_REGISTRY["mma_m16n8k16_f16_f32"]

    oversized = Schedule(
        KernelSchedule(Work.parse("w33x1"), Raster()),
        {**direct.nodes, site: ReductionSchedule(Tile(atom=atom, units=(33, 1)), Reduce())},
        direct.edges,
    )
    with pytest.raises(ScheduleRefused, match="worker inventory exceeds the target thread limit"):
        context.extend(oversized)

    too_many_producers = Schedule(
        KernelSchedule(Work.parse("w1x1+p2"), Raster()),
        {**direct.nodes, site: ReductionSchedule(Tile(atom=atom), Reduce())},
        direct.edges,
    )
    with pytest.raises(ScheduleRefused, match="producer band cannot outnumber the compute band"):
        context.extend(too_many_producers)

    no_tma = Schedule(
        KernelSchedule(Work.parse("w1x1+p1"), Raster()),
        {**direct.nodes, site: ReductionSchedule(Tile(atom=atom), Reduce())},
        direct.edges,
    )
    with pytest.raises(ScheduleRefused, match="a producer band requires TMA transport at every tiled consumer"):
        context.extend(no_tma)


def test_context_requires_a_tiled_contraction_for_grouped_raster() -> None:
    context = ClassicScheduleContext(*_problem(_contraction()))
    direct = _direct(context)
    grouped = Schedule(KernelSchedule(Work(), Raster("m", 8)), direct.nodes, direct.edges)

    with pytest.raises(ScheduleRefused, match="RASTER requires a tiled contraction site"):
        context.extend(grouped)


def test_problem_context_schedule_and_materialization_are_pickle_safe() -> None:
    context = ClassicScheduleContext(*_problem(_sum()))
    restored_context = pickle.loads(pickle.dumps(context))
    restored_node = restored_context.tile_op.sites[0].node
    assert restored_context.site(restored_node) == 0
    assert restored_context.extend(_direct(restored_context)).assignment.kernel == KernelSchedule(Work(), Raster())

    schedule = _direct(context)
    restored = pickle.loads(pickle.dumps(schedule))
    assert restored.kernel == schedule.kernel
    assert dict(restored.nodes) == dict(schedule.nodes)
    assert dict(restored.edges) == dict(schedule.edges)

    site = context.tile_op.node_sites[0]
    edge = context.tile_op.edge_sites[0]
    placed = Tile(regs=(2, 1)).at(Axis("m", 8), Axis("n", 8))
    resolved = ResolvedStage(Stage(depth=1, transport="smem"), ("a_smem",), 8)
    materialization = pickle.loads(pickle.dumps(ClassicMaterialization({site: placed}, {edge: resolved})))
    assert materialization.tiles[site] == placed
    assert materialization.stages[edge] == resolved
    assert isinstance(materialization.tiles[site], PlacedTile)


def test_classic_types_implement_the_schedule_interfaces() -> None:
    assert issubclass(ClassicScheduleContext, ScheduleContext)


def test_generic_fork_adapter_drives_a_schedule_context_lazily() -> None:
    problem = _problem(_sum())
    inventory = ClassicScheduleContext(*problem)
    direct = _direct(inventory)
    context = ClassicScheduleContext(
        *problem,
        ClassicDomains(
            kernel=(direct.kernel,),
            nodes={site: (choice,) for site, choice in direct.nodes.items()},
            edges={edge: (choice,) for edge, choice in direct.edges.items()},
        ),
    )
    accepted = []

    def leaf(assignment: Schedule) -> DeferredFork:
        accepted.append(assignment)
        return DeferredFork(lambda: context.tile_op)

    forks = schedule_forks(
        context,
        branch_knobs={},
        row_delta=lambda before, after: {"position": str(after.position - before.position)},
        leaf=leaf,
        pool_id="test",
        pool_bound=1,
        pool_descent_bound=1,
    )

    assert forks and not accepted
    assert tuple(iter_leaves(forks))
    assert accepted


def test_schedule_is_immutable_without_schedule_family_mutators() -> None:
    context = ClassicScheduleContext(*_problem(_sum()))
    original = _direct(context)
    site = context.tile_op.node_sites[0]
    edge = context.tile_op.edge_sites[0]
    kernel = KernelSchedule(Work.parse("t2"), Raster())
    node = ReductionSchedule(Tile(units=(1, 2)), Reduce())
    edge_assignment = EdgeSchedule(Stage())

    with pytest.raises(FrozenInstanceError):
        original.kernel = kernel  # type: ignore[misc]
    with pytest.raises(TypeError):
        original.nodes[site] = node  # type: ignore[index]
    with pytest.raises(TypeError):
        original.edges[edge] = edge_assignment  # type: ignore[index]

    assert original == _direct(context)


def test_schedule_and_materialization_reject_untyped_entries() -> None:
    context = ClassicScheduleContext(*_problem(_sum()))
    schedule = _direct(context)

    with pytest.raises(ScheduleRefused, match="classic kernel schedule"):
        context.extend(Schedule(object(), schedule.nodes, schedule.edges))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="non-negative integer sites"):
        Schedule(schedule.kernel, {object(): next(iter(schedule.nodes.values()))}, schedule.edges)  # type: ignore[dict-item]
    with pytest.raises(ScheduleRefused, match="node assignments must contain"):
        context.extend(Schedule(schedule.kernel, {context.tile_op.node_sites[0]: object()}, schedule.edges))  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="tiles must map node ids to PlacedTile"):
        ClassicMaterialization({context.tile_op.node_sites[0]: Tile()}, {})  # type: ignore[dict-item]


def test_tile_op_caches_the_stable_schedule_inventory() -> None:
    shared = _sum()
    left = projection((shared,), (), ("sum",))
    right = projection((shared,), (), ("sum",))
    root = projection((left, right), (), ("sum", "sum"))
    tile = TileOp(op=root, axes=(_K,))

    # one walk per kernel, and every structural reading is a field of its site records
    assert tile.sites is tile.sites
    assert tile.edge_sites is tile.edge_sites
    assert tile.edge_sites == tuple((c, e) for c, s in enumerate(tile.sites) for e in range(len(s.node.operands)))
    assert tuple(tile.node_id(s.node) for s in tile.sites) == tuple(range(len(tile.sites)))

    # the walk's labels are POSITIONS in this kernel's tree, so they belong to the TileOp and not
    # to the shared subterms it is built from: every node but the root sits under a reaching
    # segment path, and a node reached down two paths keeps the first.
    assert all(site.depth > 0 for site in tile.sites[1:])
    assert tile.sites[0].depth == 0
    assert sum(site.node is shared for site in tile.sites) == 1 and tile.sites[tile.node_id(shared)].node is shared


def test_tile_requires_complete_materialization() -> None:
    root = _contraction()
    context = ClassicScheduleContext(*_problem(root))
    site = context.tile_op.node_sites[0]
    m, n = Axis("m", 8), Axis("n", 8)
    plan = Tile(units=(1, 2))
    schedule = Schedule(
        KernelSchedule(Work.parse("t2"), Raster()),
        {**_leaf_nodes(context.tile_op), site: ReductionSchedule(plan, Reduce())},
        {edge: EdgeSchedule(Stage.direct()) for edge in context.tile_op.edge_sites},
    )

    with pytest.raises(ValueError, match="refused classic schedule"):
        TileOp(
            op=root,
            place=Placement(free=(m, n)),
            axes=(m, n, _K),
            blocks=context.tile_op.blocks,
            schedule=schedule,
            materialization=ClassicMaterialization({}, {}),
        )
    placed = plan.at(m, n)
    with pytest.raises(ValueError, match="exactly the tiled node sites"):
        TileOp(
            op=root,
            place=Placement(free=(m, n)),
            axes=(m, n, _K),
            blocks=context.tile_op.blocks,
            schedule=schedule,
            materialization=ClassicMaterialization({site: placed, site + 1: placed}, {}),
        )


def test_tile_graph_round_trip_uses_the_strict_schedule_codec() -> None:
    import copy

    root = _contraction()
    m, n = Axis("m", 8), Axis("n", 8)
    context = ClassicScheduleContext(*_problem(root))
    site = context.tile_op.node_sites[0]
    plan = Tile(units=(1, 2))
    schedule = Schedule(
        KernelSchedule(Work.parse("t2"), Raster()),
        {**_leaf_nodes(context.tile_op), site: ReductionSchedule(plan, Reduce())},
        {edge: EdgeSchedule(Stage.direct()) for edge in context.tile_op.edge_sites},
    )
    tile = TileOp(
        op=root,
        name="classic",
        place=Placement(free=(m, n), grid=(m, n), mapped=True),
        axes=(m, n, _K),
        blocks=context.tile_op.blocks,
        schedule=schedule,
        materialization=ClassicMaterialization({site: plan.at(m, n)}, {}),
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("a", (8, 8)), node_id="a")
    graph.add_node(InputOp(), [], Tensor("b", (8, 8)), node_id="b")
    graph.add_node(tile, ["a", "b"], Tensor("out", (8, 8)), node_id="out")

    payload = json.loads(json.dumps(graph.to_dict(), default=str))
    restored = Graph.from_dict(payload).nodes["out"].op

    assert restored.schedule == schedule
    assert restored.materialization == tile.materialization
    assert "schedule" in payload["nodes"]["out"]["op_fields"]

    unknown = copy.deepcopy(payload)
    unknown["nodes"]["out"]["op_fields"]["materialization"]["alias"] = {}
    with pytest.raises(ValueError, match="unknown fields alias"):
        Graph.from_dict(unknown)

    missing = copy.deepcopy(payload)
    del missing["nodes"]["out"]["op_fields"]["materialization"]["stages"]
    with pytest.raises(ValueError, match="missing stages"):
        Graph.from_dict(missing)
