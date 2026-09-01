"""Classic schedule domain projection, enumeration, and materialization.

The scheduler has one candidate-space contract: kernel, node, and edge domains are projected
independently from static facts, and enumeration is exactly the compatible subset of their
Cartesian product. Algorithm 1(c, p, t) carries the immutable schedule restriction ``c`` intact
through every context extension. Traversal order may change evaluation cost, never membership.

Projection, plain-reduction, scalar-contraction, precision-gated tensor-core, materialized-operand
copy staging, smem compute-fill staging, and kernel-global raster choices are live. Later schedule
families extend the same independent factors and the one compatibility relation; they do not add
another enumerator.
"""

from __future__ import annotations

from dataclasses import dataclass

from frozendict import frozendict

from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.pure.fold import Fold, edge_refs_axis
from emmy.compiler.ir.schedule import (
    PlacedTile,
    Raster,
    Reduce,
    Stage,
    Tile,
    WarpSpec,
    Work,
    derive_inventory,
)
from emmy.compiler.ir.schedule.classic import (
    ClassicAssignment,
    ClassicDomains,
    ClassicMaterialization,
    ClassicProblem,
    ClassicScheduleCodec,
    ClassicScheduleContext,
    EdgeSchedule,
    KernelSchedule,
    ProjectionSchedule,
    ReductionSchedule,
    _ContractionFacts,
    _kstep_refusal,
    _needs_fill,
    _plan_node_refusal,
    _resolve_stage,
    _warp_atoms,
    edge_site_spelling,
    node_id_spelling,
)
from emmy.compiler.ir.schedule.views import Projection, Reduction
from emmy.compiler.ir.stmt import Loop
from emmy.compiler.ir.stmt.passes import has_contraction_tail
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import Sched, carries_partition, projection_tail, scheduled
from emmy.compiler.pipeline.fork import Fork, iter_leaves, schedule_forks
from emmy.compiler.pipeline.knob import family_pins, schedule_pin_fingerprint
from emmy.compiler.pipeline.search.space import (
    F16_MMA_F32_ACC,
    FP8_MMA,
    WARP_LANES,
    coop_reduce_moves,
    precision_pin,
    producer_band_moves,
    raster_moves,
    scalar_tile_moves,
    stage_moves,
    warp_tile_moves,
)
from emmy.compiler.structural import digest


class _EmptyDomain(RuntimeError):
    """One projected site has no locally supported choice on this structural branch."""


def _inner_free(tile: TileOp):
    """Return the innermost non-unit free axis, if one exists."""
    return next(
        (axis for axis in reversed(tile.place.free) if not (axis.extent.is_static and axis.extent.as_static() == 1)),
        None,
    )


def _transposed_reduction_ok(tile: TileOp) -> bool:
    """Whether this kernel has the structure required by a transposed cooperative band."""
    tail = projection_tail(tile)
    return _inner_free(tile) is not None and not any(isinstance(stmt, Loop) for stmt in tail) and not has_contraction_tail(tail)


def _reduction_domain(tile: TileOp, node) -> tuple[Reduce, ...]:
    """Project one plain reduction's legal choices from node and kernel facts only.

    The catalog is not capped by the axis extent: an over-wide band is legal and idles its extra
    lanes. Keeping it in the independent node domain lets ``c`` restrict an existing assignment
    instead of manufacturing a pin-only choice outside Algorithm 1.
    """
    if node.observed:
        return (Reduce(),)
    if any(spec.sweep is not None and edge_refs_axis(node, spec.sweep.name) for spec in tile.output_specs):
        return (Reduce(),)
    if isinstance(tile.op, Fold) and tile.op.axis is None and not tile.op.operands:
        return (Reduce(),)
    transposed_ok = _transposed_reduction_ok(tile)
    return (
        Reduce(),
        *(choice for choice in coop_reduce_moves() if not choice.coop_transposed or (choice.coop % WARP_LANES == 0 and transposed_ok)),
    )


def _contraction_domain(
    tile: TileOp,
    target,
    node,
    facts: _ContractionFacts,
) -> tuple[ReductionSchedule, ...]:
    """Project one contraction's locally realizable scalar and tensor-core choices."""
    per_cell_reductions = _reduction_domain(tile, node) if facts.k_axis.extent.is_static else (Reduce(),)
    allowed_atoms = _warp_atoms(tile, target, node, facts)
    wide_warp_tiles = tuple(
        plan for name in allowed_atoms if _kstep_refusal(facts.k_axis, (plan := Tile(atom=ATOM_REGISTRY[name], regs=(26, 4), bk=2))) is None
    )
    scalar_tiles = scalar_tile_moves() if len(node.channels) == 1 else (Tile(),)
    catalog = (
        *scalar_tiles,
        *(plan for plan in warp_tile_moves(allowed_atoms) if _kstep_refusal(facts.k_axis, plan) is None),
        *wide_warp_tiles,
    )
    return tuple(
        ReductionSchedule(plan, reduction) for plan in catalog for reduction in (per_cell_reductions if not plan.is_tiled else (Reduce(),))
    )


@dataclass(frozen=True)
class _ProjectionState:
    """Immutable facts shared by every node-domain projection in one kernel."""

    tile: TileOp
    target: object
    context: ClassicScheduleContext
    contraction_facts: dict
    sched: Sched


def _options(state: _ProjectionState, node) -> tuple:
    """Project one independent node factor without crossing it with edge choices."""
    site = state.context.site(node)
    view = state.context.views[site]
    if isinstance(view, Projection):
        if site not in state.context.tile_sites or not state.tile.place.free:
            return (ProjectionSchedule(Tile()),)
        inner = state.tile.place.free[-1]
        extent = inner.extent.as_static() if inner.extent.is_static else 0
        plans = tuple(
            dict.fromkeys(
                plan
                for plan in scalar_tile_moves()
                if plan.units == (1, 1) and plan.reg_m == 1 and (plan.reg_n == 1 or (extent and extent % plan.reg_n == 0))
            )
        )
        return tuple(ProjectionSchedule(plan) for plan in plans)

    choices = (
        _contraction_domain(state.tile, state.target, node, state.contraction_facts[site])
        if view.contraction is not None
        else tuple(ReductionSchedule(Tile(), reduction) for reduction in _reduction_domain(state.tile, node))
    )
    valid_choices = []
    for choice in choices:
        geometry = state.sched.placed(node, choice.tile)
        if choice.tile.is_tiled and not isinstance(geometry, PlacedTile):
            continue
        facts = state.contraction_facts.get(site)
        if (
            isinstance(geometry, PlacedTile)
            and facts is not None
            and _plan_node_refusal(state.tile, node, choice.tile, geometry, facts) is not None
        ):
            continue
        valid_choices.append(choice)
    if not valid_choices:
        raise _EmptyDomain(f"classic site {node_id_spelling(site)} has no locally supported choice")
    return tuple(valid_choices)


def _edge_domain(state: _ProjectionState, site: int, choices: tuple) -> tuple[EdgeSchedule, ...]:
    """Project the independent edge catalog; context composition decides compatibility."""
    node = state.context.node(site)
    view = state.context.views[site]
    if not isinstance(view, Reduction) or view.contraction is None:
        return (EdgeSchedule(Stage.direct()),)
    supported = {}
    direct = EdgeSchedule(Stage.direct())
    catalogs = {
        warp: tuple(stage_moves(warp=warp, ctx=state.target))
        for warp in {choice.tile.is_warp for choice in choices if choice.tile.is_tiled}
    }
    for choice in choices:
        if not choice.tile.is_tiled:
            supported.setdefault(direct, None)
            continue
        if _needs_fill(state.tile, node, choice.tile):
            candidates = (Stage(depth=1), Stage(depth=2))
            facts = state.contraction_facts[site]
            if facts.packed[0] is not None:
                candidates = (*candidates, *catalogs[True])
        else:
            supported.setdefault(direct, None)
            candidates = catalogs[choice.tile.is_warp]
        for stage in candidates:
            supported.setdefault(EdgeSchedule(stage), None)
    if not supported:
        raise _EmptyDomain(f"classic site {node_id_spelling(site)} has no locally supported edge choice")
    return tuple(supported)


def _project_domains(tile: TileOp, target) -> tuple[ClassicProblem, ClassicDomains]:
    """Project independent domains and retain their immutable contraction facts."""
    problem = ClassicProblem.from_tile(tile, target)
    context = ClassicScheduleContext(problem)
    nodes = {}
    work_domain = {Work()}
    contraction_facts = problem.contractions
    state = _ProjectionState(
        tile,
        target,
        context,
        contraction_facts,
        Sched(tile.op, place=tile.place.on_grid()),
    )
    edge_domains = {}
    for site in context.node_sites:
        choices = _options(state, context.node(site))
        nodes[site] = choices
        edge_choices = _edge_domain(state, site, choices)
        edge_domains.update({edge: edge_choices for edge in context.incident_edges(site)})
        work_domain.update(
            work
            for choice in choices
            if (
                work := derive_inventory(
                    (choice.tile,),
                    coop=choice.reduce.coop if isinstance(choice, ReductionSchedule) else 1,
                )
            )
            is not None
        )
    raster_values = (
        raster_moves()
        if any(isinstance(view, Reduction) and view.contraction is not None for view in context.views.values())
        and all(axis.extent.is_static for axis in tile.place.free)
        else [""]
    )
    kernel_work_domain = {
        Work(kind=work.kind, units=work.units, producer=producer)
        for work in work_domain
        for producer in (producer_band_moves() if work.kind == "warp" else (0,))
    }

    return (
        problem,
        ClassicDomains(
            kernel=tuple(
                KernelSchedule(work, Raster.parse(raster))
                for work in sorted(kernel_work_domain, key=lambda work: work.spell())
                for raster in raster_values
            ),
            nodes=nodes,
            edges=edge_domains,
        ),
    )


def project_domains(tile: TileOp, target) -> ClassicDomains:
    """Project independent kernel, node, and edge domains from immutable problem facts.

    Each factor is derived from the problem and its site's node-local classification only. The
    support records state the direct choices' compatibility without changing any public domain.
    """
    _problem, domains = _project_domains(tile, target)
    return domains


@dataclass(frozen=True)
class _ScheduleLeaf(Fork):
    """One accepted schedule assignment, materialized only if search selects it."""

    tile: TileOp
    name: str
    inherited_knobs: dict
    target: object
    schedule: ClassicAssignment
    row: frozendict
    contraction_facts: frozendict
    pool_id: str
    is_leaf = True

    @property
    def knobs(self) -> dict:
        return {**self.inherited_knobs, **self.row}

    def expand(self) -> list[TileOp]:
        source = self.tile
        sched = Sched(source.op, place=source.place.on_grid())
        placed = {}
        resolved = {}
        for site, choice in self.schedule.nodes.items():
            node = source.nodes[site]
            geometry = None
            if choice.tile.is_tiled and isinstance(choice, ReductionSchedule):
                geometry = sched.placed(node, choice.tile)
                if not isinstance(geometry, PlacedTile):
                    raise ValueError(f"accepted TILE at {node_id_spelling(site)} has no placed geometry")
                placed[site] = geometry
            for edge, edge_choice in self.schedule.edges.items():
                if edge[0] != site or edge_choice.stage.is_direct:
                    continue
                if not isinstance(geometry, PlacedTile):
                    raise ValueError(f"accepted STAGE at {edge_site_spelling(edge)} has no placed consumer geometry")
                stage = _resolve_stage(
                    source,
                    self.target,
                    node,
                    choice.tile,
                    geometry,
                    edge_choice.stage,
                    self.contraction_facts[site],
                )
                if stage is None:
                    raise ValueError(f"accepted STAGE at {edge_site_spelling(edge)} did not resolve")
                resolved[edge] = stage
        return [
            scheduled(
                source.op,
                name=self.name,
                place=source.place.on_grid(),
                knobs=self.knobs,
                output_specs=source.output_specs,
                schedule=self.schedule,
                materialization=ClassicMaterialization(placed, resolved),
                workers=WarpSpec(self.schedule.kernel.work.producer) if self.schedule.kernel.work.producer else None,
            )
        ]


class _SampleRow(dict):
    """A streamed canonical row that retains its already-validated typed schedule."""

    __slots__ = ("schedule",)

    def __init__(self, schedule: ClassicAssignment, row: dict) -> None:
        super().__init__(row)
        self.schedule = schedule


def _context_row(before: ClassicScheduleContext, after: ClassicScheduleContext) -> dict[str, str]:
    """The canonical partial row composed between two immutable contexts."""
    row = {}
    assert before.order is not None and after.order is not None
    for site in after.order[before.position : after.position]:
        node = after.assignment.nodes[site]
        if site in after.tile_sites:
            row[after.node_key("TILE", site)] = node.tile.spell()
        if site in after.reduction_sites:
            assert isinstance(node, ReductionSchedule)
            row[after.node_key("REDUCE", site)] = node.reduce.spell()
        staged = tuple(edge for edge in after.incident_edges(site) if edge in after.stage_edges)
        if staged:
            choices = {after.assignment.edges[edge] for edge in staged}
            if len(choices) == 1:
                row[after.stage_key(staged[0])] = choices.pop().stage.spell()
    if after.work is not None:
        row["WORK"] = after.work.spell()
    return row


def classic_context(
    tile: TileOp,
    target,
    domains: ClassicDomains,
    *,
    pins=None,
    problem: ClassicProblem | None = None,
) -> ClassicScheduleContext:
    """Build the immutable classic ``c + p + t`` context for one fixed domain product."""
    requested = {family: family_pins(family) for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")} if pins is None else pins
    context = ClassicScheduleContext(problem or ClassicProblem.from_tile(tile, target), domains).restrict(
        requested,
        split_consumed=carries_partition(tile.op) or tile.split_consumed,
        allow_f16_accumulate=precision_pin(F16_MMA_F32_ACC) is True,
        allow_fp8=precision_pin(FP8_MMA) is True,
        validate_pins=target.validate_pins,
    )
    return context


def classic_forks(tile: TileOp, name: str, knobs: dict, ctx) -> list[Fork]:
    """Run Algorithm 1(c, p, t) over fixed independent domains."""
    try:
        problem, domains = _project_domains(tile, ctx)
    except _EmptyDomain:
        return []
    context = classic_context(tile, ctx, domains, problem=problem)
    codec = ClassicScheduleCodec(context)
    pin_fingerprint = schedule_pin_fingerprint()
    pool_id = digest(
        tile.identity_key(with_io=True) or "",
        ctx.structural_key(),
        tuple((axis.name, repr(axis.extent)) for axis in tile.place.free),
        tuple(codec.keys()),
        pin_fingerprint,
        tile.split_consumed,
    )
    prefix = {"S_warp_eligible": 1.0} if any(choice.tile.is_warp for choices in domains.nodes.values() for choice in choices) else {}
    roots = schedule_forks(
        context,
        branch_knobs={**knobs, **prefix},
        row_delta=_context_row,
        leaf=lambda assignment: _ScheduleLeaf(
            tile,
            name,
            dict(knobs),
            ctx,
            assignment,
            frozendict({**prefix, **codec._encode(assignment)}),
            problem.contractions,
            pool_id,
        ),
        pool_id=pool_id,
        pool_bound=domains.product_size,
        pool_descent_bound=len(domains.kernel)
        + sum(
            len(choices) * max((len(domains.edges[edge]) for edge in domains.edges if edge[0] == site), default=1)
            for site, choices in domains.nodes.items()
        ),
    )
    sample = getattr(ctx, "pool_sample", None)
    if sample is not None:
        drawn = sample.take(_SampleRow(leaf.schedule, dict(leaf.row)) for leaf in iter_leaves(roots) if isinstance(leaf, _ScheduleLeaf))
        sample.totals[pool_id] = drawn.total
        return [
            _ScheduleLeaf(
                tile,
                name,
                dict(knobs),
                ctx,
                row.schedule,
                frozendict(row),
                problem.contractions,
                pool_id,
            )
            for row in drawn.rows
        ]
    return roots


__all__ = ["classic_context", "classic_forks", "project_domains"]
