"""Production classic scheduling obeys the independent-domain contract."""

from dataclasses import replace as dc_replace
from importlib import import_module

from emmy.compiler.context import Context
from emmy.compiler.graph import Tensor
from emmy.compiler.ir.axis import Axis, AxisRole, Window
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Channel, Fold
from emmy.compiler.ir.schedule import Reduce, ScheduleContext, ScheduleRefused, Stage, Tile, Work, schedule
from emmy.compiler.ir.schedule import classic_projection as classic
from emmy.compiler.ir.schedule.catalog import coop_reduce_moves, scalar_tile_moves
from emmy.compiler.ir.schedule.classic import (
    ClassicScheduleCodec,
    ClassicScheduleContext,
    ReductionSchedule,
)
from emmy.compiler.ir.schedule.classic_projection import project_classic
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.tile import Placement, TileOp
from emmy.compiler.ir.tile.ops import carries_partition
from emmy.compiler.pipeline.fork import iter_leaves
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from tests.compiler.helpers import enumerate_classic_reference

classic_forks = import_module("emmy.compiler.pipeline.passes.lowering.tile.040_schedule").classic_forks


def _signature(codec, assignment) -> tuple[tuple[str, str], ...]:
    return tuple(codec.encode(assignment).items())


def _enumerate_context(context: ScheduleContext):
    for value in schedule(context):
        if isinstance(value, ScheduleContext):
            yield from _enumerate_context(value)
        else:
            yield value


def _context(tile, target, domains, *, pins=None):
    return ClassicScheduleContext(tile, target, domains).restrict(
        pins or {},
        split_consumed=carries_partition(tile.op) or tile.split_consumed,
        allow_f16_accumulate=False,
        allow_fp8=False,
        validate_pins=target.validate_pins,
    )


def _reference(tile, target, domains, *, pins=None):
    """Run Algorithm 1(c, p, t) under the same immutable c as production."""
    return enumerate_classic_reference(_context(tile, target, domains, pins=pins))


def _schedule_leaves(tile, name, target):
    """Expand the lazy traversal while retaining its typed schedule leaves."""
    return tuple(iter_leaves(classic_forks(tile, name, {}, target)))


def test_production_enumeration_is_the_compatible_independent_product() -> None:
    root = Fold.projection(body=Body((Assign("y", "add", ("x", "x")),)), results=("y",))
    tile = TileOp(
        op=root,
        place=Placement(free=(Axis("n", 8),)),
        placement_decided=True,
        split_consumed=True,
    )
    target = Context.from_target((12, 0))
    domains = project_classic(tile, target)
    codec = ClassicScheduleCodec(_context(tile, target, domains))

    reference = {_signature(codec, assignment) for assignment in _reference(tile, target, domains)}
    leaves = _schedule_leaves(tile, "pointwise", target)

    assert {_signature(codec, leaf.schedule) for leaf in leaves} == reference
    assert len(reference) == domains.product_size == 3
    (materialized,) = leaves[0].expand()
    assert materialized.schedule == leaves[0].schedule
    assert materialized.place == tile.place.on_grid()
    assert materialized.placement_decided
    assert materialized.split_consumed


def test_complete_c_proves_its_singleton_without_changing_domains() -> None:
    root = fold_from_loop(
        Loop(
            axis=Axis("k", 64),
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
    tile = TileOp(op=root, place=Placement(free=(Axis("n", 64),)))
    target = Context.from_target((12, 0))
    domains = project_classic(tile, target)
    context = ClassicScheduleContext(tile, target, domains)
    codec = ClassicScheduleCodec(context)
    candidates = tuple(enumerate_classic_reference(context))
    wanted = candidates[-1]
    pins = {family: [] for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")}
    for key, value in codec.encode(wanted).items():
        pins[key.partition("@")[0]].append((key, value))
    c = _context(tile, target, domains, pins={family: tuple(values) for family, values in pins.items()})

    assert len(candidates) > 1
    assert project_classic(tile, target) == domains
    assert tuple(_enumerate_context(c)) == (wanted,)
    assert tuple(enumerate_classic_reference(c)) == (wanted,)


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
    domains = project_classic(tile, target)
    context = ClassicScheduleContext(tile, target, domains)
    site = context.tile_op.node_sites[0]

    expected_reductions = {Reduce(), *coop_reduce_moves()}
    assert {choice.reduce for choice in domains.nodes[site] if isinstance(choice, ReductionSchedule)} == expected_reductions
    reference = tuple(_reference(tile, target, domains))
    leaves = _schedule_leaves(tile, "reduce", target)
    codec = ClassicScheduleCodec(_context(tile, target, domains))

    assert {_signature(codec, leaf.schedule) for leaf in leaves} == {_signature(codec, assignment) for assignment in reference}
    assert len(reference) == len(expected_reductions)
    assert domains.product_size > len(reference)


def test_scalar_contraction_enumeration_is_the_compatible_independent_product(monkeypatch) -> None:
    m, n, k = Axis("m", 64), Axis("n", 64), Axis("k", 64)
    root = Fold.contraction(
        k_axis=k,
        a=Load(name="a_e", input="a", index=(Var("m"), Var("k"))),
        channels=(Channel(b=Load(name="b_e", input="b", index=(Var("k"), Var("n"))), acc="acc"),),
    )
    tile = TileOp(op=root, place=Placement(free=(m, n)))
    target = Context.from_target((12, 0))
    monkeypatch.setattr(classic, "stage_moves", lambda *, warp, ctx=None: [])
    domains = project_classic(tile, target)
    context = ClassicScheduleContext(tile, target, domains)
    site = context.tile_op.node_sites[0]

    choices = domains.nodes[site]
    assert {choice.tile for choice in choices if isinstance(choice, ReductionSchedule) and choice.reduce == Reduce()} == set(
        scalar_tile_moves()
    )
    expected_reductions = {Reduce(), *coop_reduce_moves()}
    actual_reductions = {choice.reduce for choice in choices if isinstance(choice, ReductionSchedule) and not choice.tile.is_tiled}
    assert actual_reductions == expected_reductions

    reference = tuple(_reference(tile, target, domains))
    leaves = _schedule_leaves(tile, "matmul", target)
    codec = ClassicScheduleCodec(_context(tile, target, domains))
    assert {_signature(codec, leaf.schedule) for leaf in leaves} == {_signature(codec, assignment) for assignment in reference}
    assert domains.product_size > len(reference)
    assert {choice.raster.spell() for choice in domains.kernel} == {"", "gm8", "gn4", "gn8"}
    assert all(assignment.kernel.raster.is_direct for assignment in reference if not assignment.nodes[site].tile.is_tiled)

    tiled = next(leaf for leaf in leaves if leaf.schedule.nodes[site].tile.is_tiled)
    materialized = tiled.expand()[0]
    assert materialized.materialization.tiles[site].choice == tiled.schedule.nodes[site].tile


def test_overwide_reduction_is_in_the_domain_before_c_restricts_it(monkeypatch) -> None:
    root = fold_from_loop(
        Loop(
            axis=Axis("k", 8),
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
    tile = TileOp(op=root, place=Placement(free=(Axis("n", 8),)))
    target = Context.from_target((12, 0))
    overwide = Reduce.of(coop=128)
    monkeypatch.setattr(classic, "coop_reduce_moves", lambda: [overwide])
    domains = project_classic(tile, target)
    c = _context(
        tile,
        target,
        domains,
        pins={
            "WORK": (("WORK", "t128"),),
            "TILE": (("TILE", ""),),
            "REDUCE": (("REDUCE", "coop"),),
            "STAGE": (("STAGE", ""),),
            "RASTER": (("RASTER", ""),),
        },
    )
    assert any(choice.reduce == overwide for choice in domains.nodes[0])
    assert len(tuple(_enumerate_context(c))) == 1


def test_multi_channel_contraction_domain_contains_per_cell_and_warp_compute_fill() -> None:
    m, n, k = Axis("m", 16), Axis("n", 16), Axis("k", 16)
    root = Fold.contraction(
        k_axis=k,
        a=Load(name="a_e", input="a", index=(Var("m"), Var("k"))),
        channels=(
            Channel(b=Load(name="b0_e", input="b0", index=(Var("k"), Var("n"))), acc="acc0"),
            Channel(b=Load(name="b1_e", input="b1", index=(Var("k"), Var("n"))), acc="acc1"),
        ),
    )
    tile = TileOp(
        op=root,
        place=Placement(free=(m, n)),
        inputs={name: Tensor(name, (16, 16), "f16") for name in ("a", "b0", "b1")},
        outputs={"out": Tensor("out", (16, 16), "f16")},
    )
    target = Context.from_target((12, 0))
    domains = project_classic(tile, target)
    c = ClassicScheduleContext(tile, target, domains)
    compatible = []
    for pick in c.extensions():
        try:
            compatible.append(c.extend(pick))
        except ScheduleRefused:
            pass

    assert any(not choice.tile.is_tiled for choice in domains.nodes[0])
    assert any(choice.tile.is_warp for choice in domains.nodes[0])
    per_cell = [child for child in compatible if not child.assignment.nodes[0].tile.is_tiled]
    warp = [child for child in compatible if child.assignment.nodes[0].tile.is_warp]
    assert per_cell and all(choice.stage.is_direct for child in per_cell for choice in child.assignment.edges.values())
    assert warp and all(choice.stage.transport == "smem" for child in warp for choice in child.assignment.edges.values())


def test_tensor_core_enumeration_is_the_compatible_independent_product(monkeypatch) -> None:
    m, n, k = Axis("m", 128), Axis("n", 128), Axis("k", 132)
    root = Fold.contraction(
        k_axis=k,
        a=Load(name="a_e", input="a", index=(Var("m"), Var("k"))),
        channels=(Channel(b=Load(name="b_e", input="b", index=(Var("k"), Var("n"))), acc="acc"),),
    )
    tile = TileOp(
        op=root,
        place=Placement(free=(m, n)),
        inputs={"a": Tensor("a", (128, 132), "f16"), "b": Tensor("b", (132, 128), "f16")},
        outputs={"out": Tensor("out", (128, 128), "f16")},
    )
    target = Context.from_target((12, 0))
    warp = Tile.parse("mma_m16n8k16_f16_f32/f2x2/k2", Work.parse("w2x2"))
    monkeypatch.setattr(classic, "scalar_tile_moves", lambda: [Tile()])
    monkeypatch.setattr(classic, "warp_tile_moves", lambda atoms: [warp] if warp.atom.name in atoms else [])
    monkeypatch.setattr(classic, "stage_moves", lambda *, warp, ctx=None: [])
    domains = project_classic(tile, target)
    context = ClassicScheduleContext(tile, target, domains)
    site = context.tile_op.node_sites[0]

    warp_choices = tuple(choice for choice in domains.nodes[site] if isinstance(choice, ReductionSchedule) and choice.tile.is_warp)
    assert warp_choices
    assert any(choice.work.kind == "warp" for choice in domains.kernel)

    reference = tuple(_reference(tile, target, domains))
    leaves = _schedule_leaves(tile, "matmul", target)
    codec = ClassicScheduleCodec(_context(tile, target, domains))
    assert {_signature(codec, leaf.schedule) for leaf in leaves} == {_signature(codec, assignment) for assignment in reference}
    assert domains.product_size > len(reference)
    assert {choice.raster.spell() for choice in domains.kernel} == {"", "gm8", "gn4", "gn8"}
    assert all(assignment.kernel.raster.is_direct for assignment in reference if not assignment.nodes[site].tile.is_tiled)

    warp = next(leaf for leaf in leaves if leaf.schedule.nodes[site].tile.is_warp)
    assert warp.expand()[0].materialization.tiles[site].choice == warp.schedule.nodes[site].tile


def test_producer_band_is_a_restricted_kernel_domain_choice(monkeypatch) -> None:
    """Producer bands belong to the fixed kernel factor; edge compatibility admits only TMA."""
    m, n, k = Axis("m", 128), Axis("n", 128), Axis("k", 128)
    root = Fold.contraction(
        k_axis=k,
        a=Load(name="a_e", input="a", index=(Var("m"), Var("k"))),
        channels=(Channel(b=Load(name="b_e", input="b", index=(Var("k"), Var("n"))), acc="acc"),),
    )
    tile = TileOp(
        op=root,
        place=Placement(free=(m, n)),
        inputs={"a": Tensor("a", (128, 128), "f16"), "b": Tensor("b", (128, 128), "f16")},
        outputs={"out": Tensor("out", (128, 128), "f16")},
    )
    target = Context.from_target((12, 0))
    plan = Tile.parse("mma_m16n8k16_f16_f32/f2x2/k2", Work.parse("w2x2"))
    monkeypatch.setattr(classic, "scalar_tile_moves", lambda: [Tile()])
    monkeypatch.setattr(classic, "warp_tile_moves", lambda atoms: [plan] if plan.atom.name in atoms else [])
    monkeypatch.setattr(classic, "stage_moves", lambda *, warp, ctx=None: [Stage.parse("d2/smem-tma")])
    domains = project_classic(tile, target)

    works = {choice.work.spell() for choice in domains.kernel}
    assert {"w2x2", "w2x2+p1", "w2x2+p2"} <= works
    reference = tuple(_reference(tile, target, domains))
    assert any(assignment.kernel.work.producer for assignment in reference)
    assert all(
        all(choice.stage.transport == "smem-tma" for choice in assignment.edges.values())
        for assignment in reference
        if assignment.kernel.work.producer
    )

    monkeypatch.setenv("EMMY_WORK", "w2x2+p1")
    monkeypatch.setenv("EMMY_TILE", plan.spell())
    monkeypatch.setenv("EMMY_REDUCE", "")
    monkeypatch.setenv("EMMY_STAGE", "d2/smem-tma")
    assert project_classic(tile, target) == domains
    leaves = _schedule_leaves(tile, "matmul", target)
    c = _context(
        tile,
        target,
        domains,
        pins={
            "WORK": (("WORK", "w2x2+p1"),),
            "TILE": (("TILE", plan.spell()),),
            "REDUCE": (("REDUCE", ""),),
            "STAGE": (("STAGE", "d2/smem-tma"),),
        },
    )
    codec = ClassicScheduleCodec(c)
    restricted = tuple(enumerate_classic_reference(c))
    assert {_signature(codec, leaf.schedule) for leaf in leaves} == {_signature(codec, assignment) for assignment in restricted}
    assert restricted and all(assignment.kernel.work.spell() == "w2x2+p1" for assignment in restricted)
    assert leaves[0].expand()[0].workers.producer_warps == 1


def test_schedule_parameters_restrict_algorithm_one_without_changing_domains(monkeypatch) -> None:
    """Exact parameters preserve the factors and prune only context composition."""
    m, n, k = Axis("m", 128), Axis("n", 128), Axis("k", 128)
    root = Fold.contraction(
        k_axis=k,
        a=Load(name="a_e", input="a", index=(Var("m"), Var("k"))),
        channels=(Channel(b=Load(name="b_e", input="b", index=(Var("k"), Var("n"))), acc="acc"),),
    )
    tile = TileOp(
        op=root,
        place=Placement(free=(m, n)),
        inputs={"a": Tensor("a", (128, 128), "f16"), "b": Tensor("b", (128, 128), "f16")},
        outputs={"out": Tensor("out", (128, 128), "f16")},
    )
    pinned_plan = Tile.parse("mma_m16n8k16_f16_f32/f2x2/k2", Work.parse("w2x1"))
    monkeypatch.setattr(classic, "scalar_tile_moves", lambda: [Tile()])
    monkeypatch.setattr(
        classic,
        "warp_tile_moves",
        lambda atoms: [pinned_plan] if pinned_plan.atom.name in atoms else [],
    )
    monkeypatch.setattr(classic, "stage_moves", lambda *, warp, ctx=None: [])
    target = Context.from_target((12, 0))
    unpinned = project_classic(tile, target)
    monkeypatch.setenv("EMMY_WORK", "w2x1")
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f32/f2x2/k2")
    pinned = project_classic(tile, target)

    assert pinned == unpinned

    c = _context(
        tile,
        target,
        pinned,
        pins={
            "WORK": (("WORK", "w2x1"),),
            "TILE": (("TILE", "mma_m16n8k16_f16_f32/f2x2/k2"),),
        },
    )
    codec = ClassicScheduleCodec(c)
    assert {pick.nodes[0].tile.spell() for pick in c.extensions()} == {"mma_m16n8k16_f16_f32/f2x2/k2"}
    assignments = tuple(_enumerate_context(c))
    assert {assignment.nodes[0].tile.spell() for assignment in assignments} == {"mma_m16n8k16_f16_f32/f2x2/k2"}
    assert {assignment.kernel.work.spell() for assignment in assignments} == {"w2x1"}
    reference = {_signature(codec, assignment) for assignment in enumerate_classic_reference(c)}
    leaves = _schedule_leaves(tile, "matmul", target)
    assert {_signature(codec, leaf.schedule) for leaf in leaves} == reference
    assert reference
    assert all(dict(row)["WORK"] == "w2x1" and dict(row)["TILE"] == "mma_m16n8k16_f16_f32/f2x2/k2" for row in reference)


def test_bare_kernel_parameter_applies_when_scoped_pin_targets_another_kernel() -> None:
    root = Fold.projection(body=Body((Assign("y", "add", ("x", "x")),)), results=("y",))
    tile = TileOp(op=root, place=Placement(free=(Axis("n", 8),)))
    target = Context.from_target((12, 0))
    domains = project_classic(tile, target)
    pins = {family: () for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")}
    pins["WORK"] = (("WORK", "w1x1"),)
    pins["TILE"] = (("TILE@n9", "mma_m16n8k16_f16_f32/f2x2/k2"),)

    c = _context(tile, target, domains, pins=pins)

    assert tuple(enumerate_classic_reference(c)) == ()


def test_union_parameter_ignores_a_global_value_unsupported_by_this_kernel() -> None:
    """A graph-wide pin may target a sibling kernel in a union compile."""
    root = Fold.projection(body=Body((Assign("y", "add", ("x", "x")),)), results=("y",))
    tile = TileOp(op=root, place=Placement(free=(Axis("n", 8),)))
    target = dc_replace(Context.from_target((12, 0)), validate_pins=False)
    domains = project_classic(tile, target)
    pins = {family: () for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")}
    pins["WORK"] = (("WORK", "w1x1"),)

    c = _context(tile, target, domains, pins=pins)

    assert tuple(enumerate_classic_reference(c))


def test_schedule_restriction_snapshots_parameter_values() -> None:
    root = Fold.projection(body=Body((Assign("y", "add", ("x", "x")),)), results=("y",))
    tile = TileOp(op=root, place=Placement(free=(Axis("n", 8),)))
    target = Context.from_target((12, 0))
    domains = project_classic(tile, target)
    pins = {family: () for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")}
    pins["WORK"] = (("WORK", ""),)
    c = _context(tile, target, domains, pins=pins)
    expected = tuple(enumerate_classic_reference(c))

    pins["WORK"] = (("WORK", "t2"),)

    assert expected
    assert tuple(enumerate_classic_reference(c)) == expected


def test_schedule_restriction_drops_the_structural_split_stage_from_c() -> None:
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
    assert root is not None and root.axis is not None
    parent = root.axis
    root = dc_replace(root, axis=dc_replace(parent, extent=1024, window=Window(parent=parent, partition=True)))
    tile = TileOp(op=root, place=Placement(free=(Axis("n", 512),)))
    target = Context.from_target((12, 0))
    domains = project_classic(tile, target)
    pins = {family: () for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")}
    pins["REDUCE"] = (("REDUCE", "g2k"),)
    c = _context(tile, target, domains, pins=pins)
    assignments = tuple(enumerate_classic_reference(c))
    site = ClassicScheduleContext(tile, target, domains).tile_op.node_sites[0]

    assert assignments
    assert all(assignment.nodes[site].reduce == Reduce() for assignment in assignments)


def test_staged_edges_are_independent_product_factors(monkeypatch) -> None:
    m, n, k = Axis("m", 64), Axis("n", 64), Axis("k", 64)
    root = Fold.contraction(
        k_axis=k,
        a=Load(name="a_e", input="a", index=(Var("m"), Var("k"))),
        channels=(Channel(b=Load(name="b_e", input="b", index=(Var("k"), Var("n"))), acc="acc"),),
    )
    tile = TileOp(
        op=root,
        place=Placement(free=(m, n)),
        inputs={"a": Tensor("a", (64, 64), "f32"), "b": Tensor("b", (64, 64), "f32")},
        outputs={"out": Tensor("out", (64, 64), "f32")},
    )
    target = Context.from_target((12, 0))
    monkeypatch.setattr(classic, "stage_moves", lambda *, warp, ctx=None: [Stage.parse("d1/smem-async")])
    domains = project_classic(tile, target)
    context = ClassicScheduleContext(tile, target, domains)

    assert len(domains.edges) == 2
    assert all({choice.stage.spell() for choice in choices} == {"", "d1/smem-async"} for choices in domains.edges.values())
    reference = tuple(_reference(tile, target, domains))
    leaves = _schedule_leaves(tile, "matmul", target)
    codec = ClassicScheduleCodec(_context(tile, target, domains))
    assert {_signature(codec, leaf.schedule) for leaf in leaves} == {_signature(codec, assignment) for assignment in reference}
    assert domains.product_size > len(reference)
    assert all(len({choice.stage for choice in assignment.edges.values()}) == 1 for assignment in reference)

    staged = next(leaf for leaf in leaves if all(not choice.stage.is_direct for choice in leaf.schedule.edges.values()))
    materialized = staged.expand()[0]
    assert set(materialized.materialization.stages) == set(context.tile_op.edge_sites)
    assert all(stage.choice == staged.schedule.edges[edge].stage for edge, stage in materialized.materialization.stages.items())


def test_compute_fill_edges_remain_independent_product_factors(monkeypatch) -> None:
    m, n, k = Axis("m", 64), Axis("n", 64), Axis("k", 64)
    computed_a = Fold.projection(
        body=Body(
            (
                Load(name="score", input="scores", index=(Var("m"), Var("k"))),
                Assign(name="prob", op="exp", args=("score",)),
            )
        )
    )
    root = Fold.contraction(
        k_axis=k,
        a=computed_a,
        channels=(Channel(b=Load(name="value", input="values", index=(Var("k"), Var("n"))), acc="acc"),),
    )
    tile = TileOp(
        op=root,
        place=Placement(free=(m, n)),
        inputs={"scores": Tensor("scores", (64, 64), "f16"), "values": Tensor("values", (64, 64), "f16")},
        outputs={"out": Tensor("out", (64, 64), "f16")},
    )
    target = Context.from_target((12, 0))
    warp = Tile.parse("mma_m16n8k16_f16_f32/f1x1", Work.parse("w1x1"))
    monkeypatch.setattr(classic, "scalar_tile_moves", lambda: [Tile()])
    monkeypatch.setattr(classic, "warp_tile_moves", lambda atoms: [warp] if warp.atom.name in atoms else [])
    domains = project_classic(tile, target)
    context = ClassicScheduleContext(tile, target, domains)
    contraction = context.tile_op.node_sites[0]

    assert all({choice.stage.spell() for choice in choices} == {"", "d1/smem", "d2/smem"} for choices in domains.edges.values())
    reference = tuple(_reference(tile, target, domains))
    leaves = _schedule_leaves(tile, "computed_a", target)
    codec = ClassicScheduleCodec(_context(tile, target, domains))
    assert {_signature(codec, leaf.schedule) for leaf in leaves} == {_signature(codec, assignment) for assignment in reference}
    assert domains.product_size > len(reference)
    warp_assignments = tuple(assignment for assignment in reference if assignment.nodes[contraction].tile.is_warp)
    assert warp_assignments
    assert all({edge.stage.transport for edge in assignment.edges.values()} == {"smem"} for assignment in warp_assignments)
