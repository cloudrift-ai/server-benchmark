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

from dataclasses import dataclass, replace

from frozendict import frozendict

from emmy.compiler.dim import Dim
from emmy.compiler.ir.atom import ATOM_REGISTRY, atoms_for
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.pure.fold import Fold, deep_reads, edge_refs_axis, is_contraction
from emmy.compiler.ir.schedule import (
    PlacedTile,
    Placement,
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
    _atom_refusal,
    _ContractionFacts,
    _kstep_refusal,
    _needs_fill,
    _plan_node_refusal,
    _resolve_stage,
    edge_site_spelling,
    node_id_spelling,
)
from emmy.compiler.ir.schedule.views import Projection, Reduction, ScheduleInventory
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Body, Load, Loop, Write
from emmy.compiler.ir.stmt.passes import has_contraction_tail
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ir import OutputSpec
from emmy.compiler.ir.tile.ops import Sched, carries_partition, cone_seam, edge_dtypes, projection_tail, scheduled
from emmy.compiler.ir.tile.path import sites
from emmy.compiler.pipeline.fork import Fork, iter_leaves, schedule_forks
from emmy.compiler.pipeline.knob import family_pins, schedule_pin_fingerprint
from emmy.compiler.pipeline.passes.lowering._addr import gmem_axis_step
from emmy.compiler.pipeline.passes.lowering._packed import match_packed_b_node, match_packed_pair_node
from emmy.compiler.pipeline.passes.lowering.tile._tree import children, walk
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


def _sibling_fragment_edges(root: Fold, inventory: ScheduleInventory) -> dict[int, str]:
    """Map each sibling-step consumer to the one contraction producing its computed edge."""
    out = {}
    for node, _axes in walk(root):
        if not (isinstance(node, Fold) and node.axis is not None) or is_contraction(node) or node.combine is None:
            continue
        steps = node.step_stmts()
        states = set(node.combine.results)
        for position, consumer in ((i, stmt) for i, stmt in enumerate(steps) if is_contraction(stmt)):
            accumulated = any(
                isinstance(stmt, Accum) and stmt.name in states and stmt.value in consumer.defines() for stmt in steps[position + 1 :]
            )
            reads = {name for edge in consumer.operands if isinstance(edge, Fold) for name in deep_reads(edge.lower())}
            if not accumulated or not reads:
                continue
            cone = Body(tuple(steps[:position])).backward_cone(reads)
            producers = tuple(stmt for stmt in cone.members if is_contraction(stmt))
            if len(producers) == 1:
                out[id(consumer)] = node_id_spelling(inventory.site(producers[0]))
    return out


def _contraction_facts(tile: TileOp, target, inventory: ScheduleInventory) -> dict[int, _ContractionFacts]:
    """Derive contraction facts that are independent of every schedule choice."""
    root = tile.op
    parents: dict[int, Fold] = {}
    for node, _axes in walk(root):
        for child, _child_axes in children(node):
            parents.setdefault(id(child), node)
    derived = {id(site.node) for site in sites(root) if site.derived}
    sibling = _sibling_fragment_edges(root, inventory)
    tail = projection_tail(tile)
    fragment_epilogue = _fragment_epilogue_ok(tail, _fold_states(root))
    facts = {}
    for node, _axes in walk(root):
        if not (isinstance(node, Fold) and node.axis is not None and is_contraction(node)):
            continue
        site = inventory.site(node)
        if site in facts:
            continue
        packed = (match_packed_b_node(node, tile.inputs), match_packed_pair_node(node, tile.inputs))
        refusal = _node_refusal(tile, target, node, fragment_epilogue, packed)
        parent = parents.get(id(node))
        if (
            id(node) in derived
            and node.axis.extent.is_static
            and node.axis.extent.as_static() == 1
            and isinstance(parent, Fold)
            and parent.axis is not None
        ):
            assert parent.combine is not None and node.combine is not None
            seam = ((), (), tuple(parent.combine.results[: -len(node.combine.results)]))
            k_axis = parent.axis
        else:
            seam = cone_seam(node.a, node.axis.name) if isinstance(node.a, Fold) and refusal is None else None
            k_axis = node.axis
        producer = None
        if isinstance(node.a, Fold):
            nested = tuple(site.node for site in sites(node.a) if is_contraction(site.node) and edge_refs_axis(site.node, k_axis.name))
            producer = nested[0] if len(nested) == 1 else None
        need = sibling.get(id(node))
        need_step = need is not None
        if need is None and producer is not None:
            need = node_id_spelling(inventory.site(producer))
        facts[site] = _ContractionFacts(
            k_axis=k_axis,
            seam=seam,
            producer=producer,
            need=need,
            packed=packed,
            need_step=need_step,
        )
    return facts


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


def _fold_states(op) -> frozenset[str]:
    """Return the Fold state names visible to the projection tail."""
    if not isinstance(op, Fold):
        return frozenset()
    if op.axis is not None:
        return frozenset(op.defines())
    return frozenset(name for edge in (*op.operands, *op.body) if isinstance(edge, Fold) for name in edge.defines())


def _fragment_epilogue_ok(tail: list, states: frozenset[str]) -> bool:
    """Whether every output is a straight-line projection of a Fold state."""
    definitions: set[str] = set()
    for stmt in tail:
        if isinstance(stmt, Loop):
            return False
        if isinstance(stmt, Load) and {name for index in stmt.index for name in index.free_vars()} & definitions:
            return False
        definitions.update(stmt.defines())
    body = Body(tail)
    return all(body.backward_cone(stmt.values).external_reads & states for stmt in tail if isinstance(stmt, Write))


def _channel_dtype(tile: TileOp, node, target):
    """Return the one tensor-core dtype shared by the contraction's B channels."""
    dtypes = {edge_dtypes(channel.b, tile.inputs)[0] for channel in node.channels}
    if len(dtypes) == 1:
        return next(iter(dtypes))
    eligible = {dtype for dtype in dtypes if dtype is not None and atoms_for(dtype, ctx=target)}
    return next(iter(eligible)) if len(eligible) == 1 else None


def _node_refusal(tile: TileOp, target, node, fragment_epilogue: bool, packed: tuple = (None, None)) -> str | None:
    """Return why static node facts rule out every tensor-core atom."""
    ring = node.semiring
    if ring is None or tuple(operator.name for operator in ring) != ("multiply", "add"):
        return "the mma atom realizes only the (multiply, add) semiring instance"
    if not tile.inputs:
        return "no typed inputs expose operand dtypes"
    if len(tile.place.free) < 2:
        return "the grid supplies no output-axis pair for a fragment"
    if not fragment_epilogue:
        return "the projection epilogue is not a per-fragment straight-line program"
    if isinstance(node.a, Fold) and node.a.axis is not None:
        return "a nested scheduling site inhabits the A edge"
    if len(node.channels) == 1 and isinstance(node.channels[0].b, Fold) and node.channels[0].b.axis is not None:
        return "a nested scheduling site inhabits the B edge"

    dtype = edge_dtypes(node.a, tile.inputs)[0]
    if packed[1] is not None:
        return None
    if dtype is not None and dtype.logical_elems != 1:
        return f"a packed {dtype} A pairs with no packed peer; no atom multiplies packed codes against decoded ones"
    if dtype is not None and dtype.nbytes == 1:
        if not isinstance(node.a, Load):
            return "fp8 fragment loads require a materialized A edge"
        if _channel_dtype(tile, node, target) != dtype:
            return "fp8 fragment loads require one matching operand dtype"
        if not atoms_for(dtype, ctx=target):
            return f"no tensor-core atom takes a {dtype} multiplicand on this target"
        return None

    atom_dtype = dtype if atoms_for(dtype, ctx=target) else _channel_dtype(tile, node, target)
    if atom_dtype is None:
        return "no operand dtype selects a tensor-core atom family"
    if atom_dtype.nbytes == 1 and atom_dtype != dtype:
        return "a demoting compute fill cannot produce an fp8 fragment"
    if not (atoms_for(atom_dtype, ctx=target) or atoms_for(atom_dtype, acc=atom_dtype, ctx=target)):
        return f"no tensor-core atom takes a {atom_dtype} multiplicand on this target"
    return None


def _atom_families(tile: TileOp, target, node, tail: list, packed: tuple = (None, None)) -> tuple[str, ...]:
    """Project every tensor-core atom allowed by static node and target facts."""
    dtype = edge_dtypes(node.a, tile.inputs)[0]
    a_is_load = isinstance(node.a, Load)
    a_step = gmem_axis_step(node.a, node.axis.name, tile.inputs) if a_is_load else None
    shapes = {**tile.inputs, **tile.outputs}

    def bindable(names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            name for name in names if _atom_refusal(ATOM_REGISTRY[name], dtype, a_step, a_is_load, tail, tile.place.free, shapes) is None
        )

    if (pair := packed[1]) is not None:
        if any(operand.bits is None for operand in pair.b):
            return ()
        weights = {tile.inputs[operand.bits.input].dtype for operand in pair.b}
        return atoms_for(next(iter(weights)), ctx=target) if len(weights) == 1 else ()
    if dtype is not None and dtype.nbytes == 1:
        return bindable(atoms_for(dtype, ctx=target))
    atom_dtype = dtype if atoms_for(dtype, ctx=target) else _channel_dtype(tile, node, target)
    base = bindable(atoms_for(atom_dtype, ctx=target))
    reduced_acc = bindable(atoms_for(atom_dtype, acc=atom_dtype, ctx=target))
    return tuple(dict.fromkeys((*base, *reduced_acc)))


def _tile_refusals(tile: TileOp, target, inventory: ScheduleInventory) -> dict[tuple[int, str], str]:
    """Project problem/target reasons for atom choices absent from a node domain."""
    tail = projection_tail(tile)
    fragment_epilogue = _fragment_epilogue_ok(tail, _fold_states(tile.op))
    shapes = {**tile.inputs, **tile.outputs}
    out = {}
    for site in inventory.tile_sites:
        node = inventory.node(site)
        view = inventory.views[site]
        if not isinstance(view, Reduction) or view.contraction is None:
            continue
        packed = (match_packed_b_node(node, tile.inputs), match_packed_pair_node(node, tile.inputs))
        node_why = _node_refusal(tile, target, node, fragment_epilogue, packed)
        dtype = edge_dtypes(node.a, tile.inputs)[0]
        a_is_load = isinstance(node.a, Load)
        a_step = gmem_axis_step(node.a, node.axis.name, tile.inputs) if a_is_load else None
        for atom in ATOM_REGISTRY.values():
            if not atom.name.startswith("mma_"):
                continue
            if target is not None and not atom.available_on(target):
                cc = target.compute_capability
                why = f"atom {atom.name} requires target feature {atom.target_feature}, which is unavailable on sm_{cc[0]}{cc[1]}"
            else:
                why = node_why
                if why is None and packed[1] is None:
                    why = _atom_refusal(atom, dtype, a_step, a_is_load, tail, tile.place.free, shapes)
            if why is not None:
                out[(site, atom.name)] = why
    return out


def _warp_atoms(tile: TileOp, target, node, packed: tuple = (None, None)) -> tuple[str, ...]:
    """Project tensor-core atoms from contraction, dtype, address, and target facts."""
    tail = projection_tail(tile)
    if _node_refusal(tile, target, node, _fragment_epilogue_ok(tail, _fold_states(tile.op)), packed) is not None:
        return ()
    return _atom_families(tile, target, node, tail, packed)


def _contraction_domain(
    tile: TileOp,
    target,
    node,
    facts: _ContractionFacts,
) -> tuple[ReductionSchedule, ...]:
    """Project one contraction's locally realizable scalar and tensor-core choices."""
    per_cell_reductions = _reduction_domain(tile, node) if facts.k_axis.extent.is_static else (Reduce(),)
    allowed_atoms = _warp_atoms(tile, target, node, facts.packed)
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
    producer_sites: frozenset[str]
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
        frozenset(facts.need for facts in contraction_facts.values() if facts.need is not None),
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
        root_choice = self.schedule.nodes[0]
        if isinstance(root_choice, ProjectionSchedule) and root_choice.tile.is_tiled:
            source = _pointwise_variant(source, root_choice.tile)
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


def _pointwise_variant(tile: TileOp, plan: Tile) -> TileOp:
    """Apply one pure pointwise register strip selected by a projection TILE choice."""
    inner = tile.place.free[-1]
    width = plan.reg_n
    op = tile.op
    ssa = {name for stmt in op.body for name in stmt.defines()}
    loads = []
    computes = []
    stores: list[OutputSpec] = []
    for offset in range(width):

        def rename(name: str, offset: int = offset) -> str:
            return f"{name}__u{offset}" if name in ssa else name

        sigma = Sigma(
            {
                inner.name: BinaryExpr(
                    "+",
                    BinaryExpr("*", Var(inner.name), Literal(width, "int")),
                    Literal(offset, "int"),
                )
            }
        )
        for stmt in op.body:
            rewritten = stmt.rewrite(rename, sigma)
            (loads if isinstance(rewritten, Load) else computes).append(rewritten)
        stores.extend(replace(spec, write=spec.write.rewrite(rename, sigma)) for spec in tile.output_specs)
    new_inner = replace(inner, extent=Dim(inner.extent.as_static() // width))
    free = (*tile.place.free[:-1], new_inner)
    return replace(
        tile,
        op=Fold.projection(body=Body((*loads, *computes))),
        place=Placement(free=free),
        output_specs=tuple(stores),
    )


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
            frozendict({**prefix, **codec._encode_accepted(assignment)}),
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
