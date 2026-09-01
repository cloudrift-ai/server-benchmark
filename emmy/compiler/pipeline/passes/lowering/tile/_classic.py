"""Classic schedule domain projection, enumeration, and materialization.

The scheduler has one candidate-space contract: kernel, node, and edge domains are projected
independently from static facts, and enumeration is exactly the compatible subset of their
Cartesian product. Algorithm 1(c, p, t) carries the immutable schedule restriction ``c`` intact
and evaluates it only on complete assignments. Traversal order may change evaluation cost, never
membership.

Projection, plain-reduction, scalar-contraction, precision-gated tensor-core, materialized-operand
copy staging, smem compute-fill staging, and kernel-global raster choices are live. Later schedule
families extend the same independent factors and the one compatibility relation; they do not add
another enumerator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache

from frozendict import frozendict

from emmy.compiler.ir.atom import ATOM_REGISTRY, AtomKind, atoms_for
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.pure.fold import Fold, deep_reads, edge_refs_axis, is_contraction
from emmy.compiler.ir.schedule import (
    PlacedTile,
    Raster,
    Reduce,
    ResolvedStage,
    Schedule,
    ScheduleContext,
    Stage,
    Tile,
    WarpSpec,
    Work,
    derive_inventory,
)
from emmy.compiler.ir.schedule.classic import (
    AxisAgreement,
    ClassicAssignment,
    ClassicDomains,
    ClassicMaterialization,
    ClassicProblem,
    ClassicScheduleCodec,
    ClassicScheduleContext,
    EdgeSchedule,
    FragmentAgreement,
    KernelSchedule,
    LocalSupport,
    ProjectionSchedule,
    ReductionSchedule,
    edge_site_spelling,
    node_id_spelling,
)
from emmy.compiler.ir.schedule.views import Projection, Reduction
from emmy.compiler.ir.stmt import Accum, Body, Load, Loop, Write
from emmy.compiler.ir.stmt.passes import has_contraction_tail
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import Sched, carries_partition, cone_seam, edge_dtypes, projection_tail, scheduled
from emmy.compiler.ir.tile.path import sites
from emmy.compiler.pipeline.fork import Fork, iter_leaves
from emmy.compiler.pipeline.knob import family_pins, schedule_pin_fingerprint
from emmy.compiler.pipeline.passes.lowering._addr import gmem_axis_step, split_addressable
from emmy.compiler.pipeline.passes.lowering.tile import _staging as staging
from emmy.compiler.pipeline.passes.lowering.tile._tree import children, walk
from emmy.compiler.pipeline.search.space import (
    F16_MMA_F32_ACC,
    FP8_MMA,
    MAX_REGISTERS_PER_CTA,
    MAX_REGISTERS_PER_THREAD,
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


@dataclass(frozen=True)
class _ContractionFacts:
    """The effective reduction axis and carried-state seam of one contraction node."""

    k_axis: Axis
    seam: tuple | None = None
    producer: Fold | None = None
    need: str | None = None
    need_step: bool = False


def _sibling_fragment_edges(root: Fold, context: ClassicScheduleContext) -> dict[int, str]:
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
                out[id(consumer)] = node_id_spelling(context.site(producers[0]))
    return out


def _contraction_facts(tile: TileOp, target, context: ClassicScheduleContext) -> dict[int, _ContractionFacts]:
    """Derive contraction facts that are independent of every schedule choice."""
    root = tile.op
    parents: dict[int, Fold] = {}
    for node, _axes in walk(root):
        for child, _child_axes in children(node):
            parents.setdefault(id(child), node)
    derived = {id(site.node) for site in sites(root) if site.derived}
    sibling = _sibling_fragment_edges(root, context)
    tail = projection_tail(tile)
    fragment_epilogue = _fragment_epilogue_ok(tail, _fold_states(root))
    facts = {}
    for node, _axes in walk(root):
        if not (isinstance(node, Fold) and node.axis is not None and is_contraction(node)):
            continue
        if id(node) in facts:
            continue
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
            seam = (
                cone_seam(node.a, node.axis.name)
                if isinstance(node.a, Fold) and _node_refusal(tile, target, node, fragment_epilogue) is None
                else None
            )
            k_axis = node.axis
        producer = None
        if isinstance(node.a, Fold):
            nested = tuple(site.node for site in sites(node.a) if is_contraction(site.node) and edge_refs_axis(site.node, k_axis.name))
            producer = nested[0] if len(nested) == 1 else None
        need = sibling.get(id(node))
        need_step = need is not None
        if need is None and producer is not None:
            need = node_id_spelling(context.site(producer))
        facts[id(node)] = _ContractionFacts(k_axis, seam, producer, need, need_step)
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
    transposed_ok = _transposed_reduction_ok(tile)
    return (
        Reduce(),
        *(choice for choice in coop_reduce_moves() if not choice.coop_transposed or (choice.coop % WARP_LANES == 0 and transposed_ok)),
    )


def _reduce_moves(state, node, key: str | None) -> list[Reduce]:
    """Compatibility adapter for the plain-reduction catalog boundary.

    The catalog depends only on the fold and immutable kernel facts. ``key`` is accepted because
    callers historically addressed this boundary by schedule key; a restriction is deliberately
    not consulted while projecting the domain.
    """
    del key
    tile = state.tile
    if node.observed or any(spec.sweep is not None and edge_refs_axis(node, spec.sweep.name) for spec in tile.output_specs):
        return [Reduce()]
    transposed_ok = getattr(state, "transposed_ok", None)
    if transposed_ok is None:
        transposed_ok = _transposed_reduction_ok(tile)
    return [
        Reduce(),
        *(choice for choice in coop_reduce_moves() if not choice.coop_transposed or (choice.coop % WARP_LANES == 0 and transposed_ok)),
    ]


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


def _node_refusal(tile: TileOp, target, node, fragment_epilogue: bool) -> str | None:
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


def _split_store_refusal(tail: list, free: tuple, atom_shape: tuple[int, int, int], shapes: dict) -> str | None:
    """Return why an atom cannot address a projection-tail load or store."""
    roles = [(free[-1].name, atom_shape[1], "n", True)]
    if len(free) >= 2:
        roles.append((free[-2].name, atom_shape[0], "m", False))
    for stmt in tail:
        if not isinstance(stmt, (Load, Write)):
            continue
        buffer = stmt.input if isinstance(stmt, Load) else stmt.output
        shape = getattr(shapes.get(buffer), "shape", None)
        for name, extent, role, trailing in roles:
            if not split_addressable(stmt.index, shape, name, extent, trailing):
                return f"warp TILE: the {role} axis reaches {buffer} through an unsupported split dimension"
    return None


def _atom_refusal(
    atom: AtomKind,
    a_dtype,
    a_step,
    a_is_load: bool,
    tail: list,
    free: tuple,
    shapes: dict,
) -> str | None:
    """Return why one otherwise available atom cannot bind this node."""
    converting = a_is_load and a_dtype is not None and a_dtype.nbytes >= 2 and a_dtype != atom.operand_dtype("a")
    if a_is_load and not converting and (a_step is None or a_step[0] != 1 or (a_step[1] and a_step[1] % atom.atom_k)):
        motion = "unknown" if a_step is None else f"{a_step[0]} elements per column"
        return (
            f"warp TILE: A fragment loaders read {atom.atom_k} contraction columns CONTIGUOUSLY, "
            f"but this operand's gmem index moves {motion}"
        )
    return _split_store_refusal(tail, free, atom.shape, shapes)


def _atom_families(tile: TileOp, target, node, tail: list) -> tuple[str, ...]:
    """Project every tensor-core atom allowed by static node and target facts."""
    dtype = edge_dtypes(node.a, tile.inputs)[0]
    a_is_load = isinstance(node.a, Load)
    a_step = gmem_axis_step(node.a, node.axis.name, tile.inputs) if a_is_load else None
    shapes = {**tile.inputs, **tile.outputs}

    def bindable(names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            name for name in names if _atom_refusal(ATOM_REGISTRY[name], dtype, a_step, a_is_load, tail, tile.place.free, shapes) is None
        )

    if dtype is not None and dtype.nbytes == 1:
        return bindable(atoms_for(dtype, ctx=target))
    atom_dtype = dtype if atoms_for(dtype, ctx=target) else _channel_dtype(tile, node, target)
    base = bindable(atoms_for(atom_dtype, ctx=target))
    reduced_acc = bindable(atoms_for(atom_dtype, acc=atom_dtype, ctx=target))
    return tuple(dict.fromkeys((*base, *reduced_acc)))


def _kstep_refusal(k_axis, plan: Tile) -> str | None:
    """Return why an fp8 atom cannot cover the contraction's K extent."""
    if not (plan.is_warp and plan.atom.operand_dtype("a").nbytes == 1):
        return None
    if not k_axis.extent.is_static:
        return f"atom {plan.atom.name}: fp8 fragment loads require a static K"
    step = plan.atom.atom_k * plan.bk
    extent = k_axis.extent.as_static()
    if extent % step == 0:
        return None
    return f"warp TILE K-step {step} does not divide the static contraction K={extent}"


def _computed_edge(node) -> bool:
    """Whether a contraction operand is an inline zero-axis producer cone."""
    return any(isinstance(edge, Fold) and edge.axis is None for edge in (node.a, *(channel.b for channel in node.channels)))


def _supports_scalar(node) -> bool:
    """Whether the scalar atom can carry every contraction channel."""
    return len(node.channels) == 1


def _needs_fill(tile: TileOp, node, plan: Tile) -> bool:
    """Whether one warp choice requires the shared-memory compute fill."""
    return plan.is_warp and (_computed_edge(node) or len(node.channels) > 1 or staging.converting_a(node, plan.atom, tile.inputs))


def _warp_atoms(tile: TileOp, target, node) -> tuple[str, ...]:
    """Project tensor-core atoms from contraction, dtype, address, and target facts."""
    tail = projection_tail(tile)
    if _node_refusal(tile, target, node, _fragment_epilogue_ok(tail, _fold_states(tile.op))) is not None:
        return ()
    return _atom_families(tile, target, node, tail)


def _contraction_domain(
    tile: TileOp,
    target,
    node,
    facts: _ContractionFacts,
) -> tuple[ReductionSchedule, ...]:
    """Project one contraction's locally realizable scalar and tensor-core choices."""
    per_cell_reductions = _reduction_domain(tile, node) if facts.k_axis.extent.is_static else (Reduce(),)
    allowed_atoms = _warp_atoms(tile, target, node)
    wide_warp_tiles = tuple(
        plan for name in allowed_atoms if _kstep_refusal(facts.k_axis, (plan := Tile(atom=ATOM_REGISTRY[name], regs=(26, 4), bk=2))) is None
    )
    catalog = (
        *(scalar_tile_moves() if _supports_scalar(node) else ()),
        *(plan for plan in warp_tile_moves(allowed_atoms) if _kstep_refusal(facts.k_axis, plan) is None),
        *wide_warp_tiles,
    )
    return tuple(
        ReductionSchedule(plan, reduction) for plan in catalog for reduction in (per_cell_reductions if not plan.is_tiled else (Reduce(),))
    )


def _plan_node_refusal(tile: TileOp, node, plan: Tile, placed: PlacedTile, facts: _ContractionFacts) -> str | None:
    """Return why node-local compute-fill facts reject one tensor-core choice."""
    refusal = _kstep_refusal(facts.k_axis, plan)
    if refusal is not None or not _needs_fill(tile, node, plan):
        return refusal
    converting = staging.converting_a(node, plan.atom, tile.inputs)
    return staging.computed_operand_cover(node, placed, converting=converting, k_axis=facts.k_axis) or staging.computed_operand_copy_dtype(
        node,
        placed,
        tile.inputs,
        converting=converting,
    )


def _fragment_agreements(
    site,
    plan: Tile,
    placed,
    stage: ResolvedStage | None,
    facts: _ContractionFacts,
    producer_sites: frozenset[str],
) -> tuple[FragmentAgreement, ...]:
    """Project one node choice's fragment offer and need without selecting its peer."""
    out = []
    key = node_id_spelling(site)
    if key in producer_sites:
        if not plan.is_tiled:
            offer = ("free",)
        elif plan.is_warp:
            offer = ("warp", plan.atom.shape, plan.atom.fragment_layout, placed.n.units, placed.n.tile)
        else:
            offer = ("scalar",)
        out.append(FragmentAgreement("offer", key, offer))
    if facts.need is not None:
        if plan.is_warp and stage is not None and stage.transport == "smem":
            kind = "step" if facts.need_step else "warp"
            need = (kind, plan.atom.shape, plan.atom.fragment_layout, stage.bk_elems)
        else:
            need = ("free",)
        out.append(FragmentAgreement("need", facts.need, need))
    return tuple(out)


def _fragment_registers(atom: AtomKind, role: str) -> int:
    """Return the exact per-lane register count of one emitted mma fragment."""
    explicit = atom.fragment_nregs(role)
    if explicit is not None:
        return explicit
    m, n, k = atom.ptx_shape
    dtype = atom.operand_dtype(role)
    if role == "a":
        return m * k * dtype.nbytes // 128
    if role == "b":
        return n * k * dtype.nbytes // 128
    return m * n // (64 if dtype.nbytes == 2 else 32)


def _paired_budget_refusal(node, producer, placed: PlacedTile, stage: ResolvedStage | None) -> str | None:
    """Return why a compute fill's producer and consumer fragments cannot coexist."""
    if not (placed.is_warp and stage is not None and producer is not None):
        return None
    atom = placed.atom
    if stage.bk_elems % atom.atom_n:
        return None
    a_regs = _fragment_registers(atom, "a")
    b_regs = _fragment_registers(atom, "b")
    c_regs = _fragment_registers(atom, "c")
    if atom.operand_dtype("c").nbytes == 2:
        c_regs += atom.atom_m * atom.atom_n // 32
    depth = max(1, stage.reg_depth)
    channels = len(node.channels)
    consumer_c = channels * placed.reg_m * placed.reg_n * c_regs
    consumer = placed.reg_m * depth * a_regs + channels * (placed.reg_n * depth * b_regs + placed.reg_m * placed.reg_n * c_regs)
    producer_n = stage.bk_elems // atom.atom_n
    producer_regs = placed.reg_m * a_regs + len(producer.channels) * (producer_n * b_regs + placed.reg_m * producer_n * c_regs)
    required = max(consumer, consumer_c + producer_regs)
    available = min(MAX_REGISTERS_PER_THREAD, MAX_REGISTERS_PER_CTA // placed.block_threads)
    if required <= available:
        return None
    return (
        f"paired contractions require at least {required} live fragment registers/thread, over the "
        f"{available}-register envelope at {placed.block_threads} threads/CTA"
    )


def _resolve_stage(
    tile: TileOp,
    target,
    node,
    plan: Tile,
    placed: PlacedTile,
    choice: Stage,
    facts: _ContractionFacts,
) -> ResolvedStage | None:
    """Resolve one copy transport without consulting any selected edge assignment."""
    if _needs_fill(tile, node, plan):
        return staging.resolve_fill_stage(
            node,
            placed,
            target.max_dynamic_smem,
            choice.depth,
            inputs=tile.inputs,
            seam=facts.seam,
            k_axis=facts.k_axis,
            producer=facts.producer,
        )
    if plan.is_warp:
        return staging.resolve_warp_stage(node, placed, choice, target.max_dynamic_smem, tile.inputs)
    return staging.resolve_scalar_stage(node, placed, choice, tile.inputs, target.max_dynamic_smem)


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
        return (ProjectionSchedule(Tile()),)

    choices = (
        _contraction_domain(state.tile, state.target, node, state.contraction_facts[id(node)])
        if view.contraction is not None
        else tuple(ReductionSchedule(Tile(), reduction) for reduction in _reduction_domain(state.tile, node))
    )
    valid_choices = []
    for choice in choices:
        geometry = state.sched.placed(node, choice.tile)
        if choice.tile.is_tiled and not isinstance(geometry, PlacedTile):
            continue
        facts = state.contraction_facts.get(id(node))
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


def _local_support(
    state: _ProjectionState,
    site: int,
    choice,
    edges: Mapping,
) -> LocalSupport | None:
    """Derive one local compatibility record after ``c`` selects its product member."""
    node = state.context.node(site)
    view = state.context.views[site]
    incident = state.context.incident_edges(site)
    if set(edges) != set(incident) or len(set(edges.values())) > 1:
        return None
    geometry = state.sched.placed(node, choice.tile)
    if choice.tile.is_tiled and not isinstance(geometry, PlacedTile):
        return None
    facts = state.contraction_facts.get(id(node))
    if (
        isinstance(geometry, PlacedTile)
        and facts is not None
        and _plan_node_refusal(state.tile, node, choice.tile, geometry, facts) is not None
    ):
        return None
    stage = next(iter(edges.values())).stage if edges else Stage.direct()
    resolved_stage = None
    if not isinstance(view, Reduction) or view.contraction is None or not choice.tile.is_tiled:
        if not stage.is_direct:
            return None
    elif _needs_fill(state.tile, node, choice.tile):
        if stage not in (Stage(depth=1), Stage(depth=2)):
            return None
        resolved_stage = _resolve_stage(state.tile, state.target, node, choice.tile, geometry, stage, facts)
    elif not stage.is_direct:
        resolved_stage = _resolve_stage(state.tile, state.target, node, choice.tile, geometry, stage, facts)
    if not stage.is_direct and (resolved_stage is None or resolved_stage.choice != stage):
        return None
    if (
        facts is not None
        and isinstance(geometry, PlacedTile)
        and _paired_budget_refusal(node, facts.producer, geometry, resolved_stage) is not None
    ):
        return None
    axes = (
        tuple(AxisAgreement(side.axis.name, side.tile, side.units) for side in geometry.mn)
        if choice.tile.is_tiled and isinstance(geometry, PlacedTile)
        else ()
    )
    return LocalSupport(
        choice,
        edges,
        work=derive_inventory((choice.tile,), coop=choice.reduce.coop if isinstance(choice, ReductionSchedule) else 1),
        axes=axes,
        fragments=(
            _fragment_agreements(site, choice.tile, geometry, resolved_stage, facts, state.producer_sites) if facts is not None else ()
        ),
        raster_eligible=choice.tile.is_tiled,
    )


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
        else:
            supported.setdefault(direct, None)
            candidates = catalogs[choice.tile.is_warp]
        for stage in candidates:
            supported.setdefault(EdgeSchedule(stage), None)
    if not supported:
        raise _EmptyDomain(f"classic site {node_id_spelling(site)} has no locally supported edge choice")
    return tuple(supported)


def _project_domains(tile: TileOp, target) -> tuple[ClassicDomains, frozendict]:
    """Project independent domains and retain their immutable contraction facts."""
    problem = ClassicProblem(tile.op, target)
    context = ClassicScheduleContext(problem)
    nodes = {}
    work_domain = {Work()}
    contraction_facts = _contraction_facts(tile, target, context)
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

    @cache
    def cached_support(site, choice, edge_items):
        return _local_support(state, site, choice, frozendict(edge_items))

    def support(site, choice, edges):
        return cached_support(site, choice, tuple(edges.items()))

    return (
        ClassicDomains(
            kernel=tuple(
                KernelSchedule(work, Raster.parse(raster))
                for work in sorted(kernel_work_domain, key=lambda work: work.spell())
                for raster in raster_values
            ),
            nodes=nodes,
            edges=edge_domains,
            _support=support,
        ),
        frozendict(contraction_facts),
    )


def project_domains(tile: TileOp, target) -> ClassicDomains:
    """Project independent kernel, node, and edge domains from immutable problem facts.

    Each factor is derived from the problem and its site's node-local classification only. The
    support records state the direct choices' compatibility without changing any public domain.
    """
    domains, _facts = _project_domains(tile, target)
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
        context = ClassicScheduleContext(ClassicProblem(self.tile.op, target=None))
        sched = Sched(self.tile.op, place=self.tile.place.on_grid())
        placed = {}
        resolved = {}
        for site, choice in self.schedule.nodes.items():
            node = context.node(site)
            geometry = None
            if choice.tile.is_tiled:
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
                    self.tile,
                    self.target,
                    node,
                    choice.tile,
                    geometry,
                    edge_choice.stage,
                    self.contraction_facts[id(node)],
                )
                if stage is None:
                    raise ValueError(f"accepted STAGE at {edge_site_spelling(edge)} did not resolve")
                resolved[edge] = stage
        return [
            scheduled(
                self.tile.op,
                name=self.name,
                place=self.tile.place.on_grid(),
                knobs=self.knobs,
                output_specs=self.tile.output_specs,
                schedule=self.schedule,
                materialization=ClassicMaterialization(placed, resolved),
                workers=WarpSpec(self.schedule.kernel.work.producer) if self.schedule.kernel.work.producer else None,
            )
        ]


@dataclass(frozen=True)
class _ScheduleWalk:
    """Facts shared by every lazy branch of one Algorithm 1 traversal."""

    tile: TileOp
    name: str
    inherited_knobs: dict
    target: object
    domains: ClassicDomains
    codec: ClassicScheduleCodec
    advance: Callable[[ScheduleContext], Iterable]
    pool_id: str
    prefix: dict
    contraction_facts: frozendict

    @property
    def pool_bound(self) -> int:
        return self.domains.product_size

    @property
    def pool_descent_bound(self) -> int:
        return len(self.domains.kernel) + sum(
            len(choices) * max((len(self.domains.edges[edge]) for edge in self.domains.edges if edge[0] == site), default=1)
            for site, choices in self.domains.nodes.items()
        )


@dataclass(frozen=True)
class _ScheduleBranch(Fork):
    """A lazy compatible prefix; expanding it advances to the next real choice."""

    walk: _ScheduleWalk
    context: ClassicScheduleContext
    row: frozendict
    is_leaf = False

    @property
    def knobs(self) -> dict:
        return {**self.walk.inherited_knobs, **self.walk.prefix, **self.row}

    @property
    def pool_id(self) -> str:
        return self.walk.pool_id

    @property
    def pool_bound(self) -> int:
        return self.walk.pool_bound

    @property
    def pool_descent_bound(self) -> int:
        return self.walk.pool_descent_bound

    def expand(self) -> list[Fork]:
        return _walk_step(self.walk, self.context, dict(self.row))


class _SampleRow(dict):
    """A streamed canonical row that retains its already-validated typed schedule."""

    __slots__ = ("schedule",)

    def __init__(self, schedule: ClassicAssignment, row: dict) -> None:
        super().__init__(row)
        self.schedule = schedule


def _context_row(walk: _ScheduleWalk, before: ClassicScheduleContext, after: ClassicScheduleContext) -> dict[str, str]:
    """The canonical partial row composed between two immutable contexts."""
    row = {}
    assert before.order is not None and after.order is not None
    for site in after.order[before.position : after.position]:
        node = after.node_assignment(site)
        if site in after.tile_sites:
            row[after.node_key("TILE", site)] = node.tile.spell()
        if site in after.reduction_sites:
            assert isinstance(node, ReductionSchedule)
            row[after.node_key("REDUCE", site)] = node.reduce.spell()
        staged = tuple(edge for edge in after.incident_edges(site) if edge in after.stage_edges)
        if staged:
            choices = {after.edge_assignment(edge) for edge in staged}
            if len(choices) == 1:
                row[after.stage_key(staged[0])] = choices.pop().stage.spell()
    if after.work is not None:
        row["WORK"] = after.work.spell()
    return row


def _walk_step(
    walk: _ScheduleWalk,
    context: ClassicScheduleContext,
    row: dict[str, str],
) -> list[Fork]:
    """Turn one result of the generic schedule function into pipeline forks."""
    forks = []
    for result in walk.advance(context):
        if isinstance(result, ScheduleContext):
            forks.append(_ScheduleBranch(walk, result, frozendict({**row, **_context_row(walk, context, result)})))
            continue
        assert isinstance(result, Schedule)
        forks.append(
            _ScheduleLeaf(
                walk.tile,
                walk.name,
                walk.inherited_knobs,
                walk.target,
                result,
                frozendict({**walk.prefix, **walk.codec._encode_accepted(result)}),
                walk.contraction_facts,
                walk.pool_id,
            )
        )
    return forks


def classic_context(tile: TileOp, target, domains: ClassicDomains, *, pins=None) -> ClassicScheduleContext:
    """Build the immutable classic ``c + p + t`` context for one fixed domain product."""
    requested = {family: family_pins(family) for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")} if pins is None else pins
    context = ClassicScheduleContext(ClassicProblem(tile.op, target), domains).restrict(
        requested,
        split_consumed=carries_partition(tile.op) or tile.split_consumed,
        allow_f16_accumulate=precision_pin(F16_MMA_F32_ACC) is True,
        allow_fp8=precision_pin(FP8_MMA) is True,
        validate_pins=target.validate_pins,
    )
    return context


def classic_forks(tile: TileOp, name: str, knobs: dict, ctx, *, advance: Callable[[ScheduleContext], Iterable]) -> list[Fork]:
    """Run Algorithm 1(c, p, t) over fixed independent domains."""
    try:
        domains, contraction_facts = _project_domains(tile, ctx)
    except _EmptyDomain:
        return []
    context = classic_context(tile, ctx, domains)
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
    walk = _ScheduleWalk(
        tile,
        name,
        dict(knobs),
        ctx,
        domains,
        codec,
        advance,
        pool_id,
        prefix,
        contraction_facts,
    )
    roots = _walk_step(walk, context, {})
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
                contraction_facts,
                pool_id,
            )
            for row in drawn.rows
        ]
    return roots


__all__ = ["classic_context", "classic_forks", "project_domains"]
