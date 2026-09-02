"""Classic schedule domain projection and materialization.

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

from dataclasses import dataclass

from emmy.compiler.ir.address import gmem_axis_step, split_addressable
from emmy.compiler.ir.atom import ATOM_REGISTRY, AtomKind, atoms_for
from emmy.compiler.ir.pure.fold import Fold, edge_free_axes
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
from emmy.compiler.ir.schedule.catalog import (
    WARP_LANES,
    coop_reduce_moves,
    producer_band_moves,
    raster_moves,
    scalar_tile_moves,
    stage_moves,
    warp_tile_moves,
)
from emmy.compiler.ir.schedule.classic import (
    ClassicAssignment,
    ClassicDomains,
    ClassicMaterialization,
    EdgeSchedule,
    KernelSchedule,
    ProjectionSchedule,
    ReductionSchedule,
    _kstep_refusal,
    _needs_fill,
    _plan_node_refusal,
    _resolve_stage,
    edge_site_spelling,
    node_id_spelling,
)
from emmy.compiler.ir.schedule.views import ContractionFacts, Projection, Reduction
from emmy.compiler.ir.stmt import Body, Load, Loop, Write
from emmy.compiler.ir.stmt.passes import has_contraction_tail
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ir import observed_result_names
from emmy.compiler.ir.tile.ops import Sched, edge_dtypes, projection_tail, scheduled


class ClassicProjectionError(RuntimeError):
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


def _chain_member_serial(tile: TileOp, node) -> bool:
    """Whether a node under a chain-form root realizes the serial fold only.

    A chain-form root is a zero-axis :class:`Fold` with no operand edge — a body-FED or
    sweep-reading member the normalize hoist keeps in place, and the shape every composed-cut and
    split piece binds through. A DIRECT body member binds through the kernel factorizer's chain
    arm, which emits its sibling providers ahead of one shared strided loop, so it carries a
    partition. Three facts keep the serial fold instead: a node nested DEEPER than the body (the
    body recursion emits it serially per cell), a boundary store carrying an output sweep (the wrap
    would re-run a partitioned member per swept cell), and a boundary store that streams into a
    sibling observed member's reduce loop (the trailing splice cannot reach a loop already sitting
    in an earlier segment). Both store facts are read over the WHOLE kernel, not the node —
    a different question from the per-node sweep-reading check above, and the exact gate the
    factorizer's chain arm applies before it gathers members.
    """
    root = tile.op
    if any(spec.sweep is not None for spec in tile.output_specs):
        return True
    if any(set(spec.write.values) <= observed_result_names(root) for spec in tile.output_specs):
        return True
    return not any(node is stmt for stmt in root.body)


def _reduction_domain(tile: TileOp, node) -> tuple[Reduce, ...]:
    """Project one plain reduction's legal choices from node and kernel facts only.

    The catalog is not capped by the axis extent: an over-wide band is legal and idles its extra
    lanes. Keeping it in the independent node domain lets ``c`` restrict an existing assignment
    instead of manufacturing a pin-only choice outside Algorithm 1.

    Shared by the contraction per-cell tier through :func:`_contraction_domain`'s delegation, and
    deliberately so: a contraction is a monoid with a ⊗ lift, so a contraction that is a direct
    chain member inherits the same member catalog, the same nested / swept / streamed serial-only
    exclusions, and the same transposed exclusion, with no carve-out of its own. That inheritance
    is a stated decision, not live behavior: ``normalize_fold_tree``'s hoist absorbs whatever body
    value fed a contraction and moves it onto an operand edge, and a root with an operand edge is
    no longer chain-form — so no tree the compiler builds today reaches this arm carrying one. It
    is written once here so a normalizer that later keeps one in place inherits the reading rather
    than acquiring a different one by omission.
    """
    if node.observed:
        return (Reduce(),)
    if not edge_free_axes(node).isdisjoint(spec.sweep.name for spec in tile.output_specs if spec.sweep is not None):
        return (Reduce(),)
    if isinstance(tile.op, Fold) and tile.op.axis is None and not tile.op.operands:
        if _chain_member_serial(tile, node):
            return (Reduce(),)
        # A transposed band's σ-substitution and guarded close assume the fold is the kernel ROOT,
        # so the chain arm cannot realize one; offering it here would mint one kernel from two
        # knob spellings.
        return (Reduce(), *(choice for choice in coop_reduce_moves() if not choice.coop_transposed))
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


def _warp_atoms(tile: TileOp, target, node) -> tuple[str, ...]:
    """Project tensor-core atoms from contraction, dtype, address, and target facts."""
    tail = projection_tail(tile)
    packed = tile.packed_reading(node)
    if _node_refusal(tile, target, node, _fragment_epilogue_ok(tail, _fold_states(tile.op)), packed) is not None:
        return ()
    return _atom_families(tile, target, node, tail, packed)


def _contraction_domain(
    tile: TileOp,
    target,
    node,
    facts: ContractionFacts,
) -> tuple[ReductionSchedule, ...]:
    """Project one contraction's locally realizable scalar and tensor-core choices."""
    per_cell_reductions = _reduction_domain(tile, node) if facts.k_axis.extent.is_static else (Reduce(),)
    allowed_atoms = _warp_atoms(tile, target, node)
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
    sched: Sched


def _options(state: _ProjectionState, node) -> tuple:
    """Project one independent node factor without crossing it with edge choices."""
    site = state.tile.node_id(node)
    view = state.tile.views[site]
    if isinstance(view, Projection):
        if site not in state.tile.family_sites["TILE"] or not state.tile.place.free:
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
        _contraction_domain(state.tile, state.target, node, state.tile.contractions[site])
        if view.contraction is not None
        else tuple(ReductionSchedule(Tile(), reduction) for reduction in _reduction_domain(state.tile, node))
    )
    valid_choices = []
    for choice in choices:
        geometry = state.sched.placed(node, choice.tile)
        if choice.tile.is_tiled and not isinstance(geometry, PlacedTile):
            continue
        facts = state.tile.contractions.get(site)
        if (
            isinstance(geometry, PlacedTile)
            and facts is not None
            and _plan_node_refusal(state.tile, node, choice.tile, geometry, facts) is not None
        ):
            continue
        valid_choices.append(choice)
    if not valid_choices:
        raise ClassicProjectionError(f"classic site {node_id_spelling(site)} has no locally supported choice")
    return tuple(valid_choices)


def _edge_domain(state: _ProjectionState, site: int, choices: tuple) -> tuple[EdgeSchedule, ...]:
    """Project the independent edge catalog; context composition decides compatibility."""
    node = state.tile.sites[site].node
    view = state.tile.views[site]
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
            if state.tile.packed_reading(node)[0] is not None:
                candidates = (*candidates, *catalogs[True])
        else:
            supported.setdefault(direct, None)
            candidates = catalogs[choice.tile.is_warp]
        for stage in candidates:
            supported.setdefault(EdgeSchedule(stage), None)
    if not supported:
        raise ClassicProjectionError(f"classic site {node_id_spelling(site)} has no locally supported edge choice")
    return tuple(supported)


def project_classic(tile: TileOp, target) -> ClassicDomains:
    """Project the independent kernel, node, and edge domains of one unscheduled tile."""
    nodes = {}
    work_domain = {Work()}
    state = _ProjectionState(tile, target, Sched(tile, place=tile.place.on_grid()))
    edge_domains = {}
    for site in range(len(tile.sites)):
        choices = _options(state, tile.sites[site].node)
        nodes[site] = choices
        edge_choices = _edge_domain(state, site, choices)
        edge_domains.update({edge: edge_choices for edge in tile.incident_edges[site]})
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
        if any(isinstance(view, Reduction) and view.contraction is not None for view in tile.views)
        and all(axis.extent.is_static for axis in tile.place.free)
        else [""]
    )
    kernel_work_domain = {
        Work(kind=work.kind, units=work.units, producer=producer)
        for work in work_domain
        for producer in (producer_band_moves() if work.kind == "warp" else (0,))
    }

    return ClassicDomains(
        kernel=tuple(
            KernelSchedule(work, Raster.parse(raster))
            for work in sorted(kernel_work_domain, key=lambda work: work.spell())
            for raster in raster_values
        ),
        nodes=nodes,
        edges=edge_domains,
    )


def materialize_classic(
    tile: TileOp,
    *,
    name: str,
    knobs: dict,
    target,
    assignment: ClassicAssignment,
) -> TileOp:
    """Materialize one accepted classic assignment into a scheduled TileOp."""
    sched = Sched(tile, place=tile.place.on_grid())
    placed = {}
    resolved = {}
    for site, choice in assignment.nodes.items():
        node = tile.sites[site].node
        geometry = None
        if choice.tile.is_tiled and isinstance(choice, ReductionSchedule):
            geometry = sched.placed(node, choice.tile)
            if not isinstance(geometry, PlacedTile):
                raise ValueError(f"accepted TILE at {node_id_spelling(site)} has no placed geometry")
            placed[site] = geometry
        for edge, edge_choice in assignment.edges.items():
            if edge[0] != site or edge_choice.stage.is_direct:
                continue
            if not isinstance(geometry, PlacedTile):
                raise ValueError(f"accepted STAGE at {edge_site_spelling(edge)} has no placed consumer geometry")
            stage = _resolve_stage(
                tile,
                target,
                node,
                choice.tile,
                geometry,
                edge_choice.stage,
                tile.contractions[site],
            )
            if stage is None:
                raise ValueError(f"accepted STAGE at {edge_site_spelling(edge)} did not resolve")
            resolved[edge] = stage
    return scheduled(
        tile.op,
        name=name,
        place=tile.place.on_grid(),
        knobs=knobs,
        output_specs=tile.output_specs,
        schedule=assignment,
        materialization=ClassicMaterialization(placed, resolved),
        workers=WarpSpec(assignment.kernel.work.producer) if assignment.kernel.work.producer else None,
    )


__all__ = [
    "ClassicProjectionError",
    "materialize_classic",
    "project_classic",
]
