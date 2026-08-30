"""Classic schedule domain projection, enumeration, and materialization.

The scheduler has one candidate-space contract: kernel, node, and edge domains are projected
independently from static facts, and enumeration is exactly the compatible subset of their
Cartesian product. Traversal order may change evaluation cost, never membership.

Projection, plain-reduction, scalar-contraction, precision-gated gmem-direct tensor-core, and
kernel-global raster choices are live. Later schedule families extend the same independent
factors and the one compatibility relation; they do not add another enumerator.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from emmy.compiler.ir.atom import ATOM_REGISTRY, AtomKind, atoms_for
from emmy.compiler.ir.classic_schedule import (
    AxisAgreement,
    ClassicDomains,
    ClassicMaterialization,
    ClassicProblem,
    ClassicSchedule,
    ClassicScheduleCodec,
    ClassicScheduleContext,
    EdgeSchedule,
    KernelSchedule,
    LocalSupport,
    Projection,
    ProjectionSchedule,
    Reduction,
    ReductionSchedule,
    enumerate_classic,
)
from emmy.compiler.ir.pure.fold import Fold, edge_refs_axis
from emmy.compiler.ir.schedule import PlacedTile, Raster, Reduce, Stage, Tile, Work, derive_inventory
from emmy.compiler.ir.stmt import Body, Load, Loop, Write
from emmy.compiler.ir.stmt.passes import has_contraction_tail
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import Sched, edge_dtypes, projection_tail, scheduled
from emmy.compiler.pipeline.fork import Fork
from emmy.compiler.pipeline.knob import family_pins
from emmy.compiler.pipeline.passes.lowering._addr import gmem_axis_step, split_addressable
from emmy.compiler.pipeline.search.space import (
    F16_MMA_F32_ACC,
    FP8_MMA,
    RASTER,
    WARP_LANES,
    coop_reduce_moves,
    precision_pin,
    raster_moves,
    scalar_tile_moves,
    warp_tile_moves,
)


class ClassicScheduleUnavailable(RuntimeError):
    """Classic scheduling has not yet been reconstructed for this term."""


class PinRefused(ValueError):
    """A classic schedule pin may be realizable only after a structural rewrite."""


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
    """Project one plain reduction's choices from node and kernel facts only."""
    if node.observed:
        return (Reduce(),)
    if any(spec.sweep is not None and edge_refs_axis(node, spec.sweep.name) for spec in tile.output_specs):
        return (Reduce(),)
    extent = node.axis.hint_extent
    transposed_ok = _transposed_reduction_ok(tile)
    return (
        Reduce(),
        *(
            choice
            for choice in coop_reduce_moves()
            if choice.coop <= extent
            and choice.reg <= extent
            and (not choice.coop_transposed or (choice.coop % WARP_LANES == 0 and transposed_ok))
        ),
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


def _atom_families(tile: TileOp, target, node, tail: list) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Project policy-offered and pin-only tensor-core atoms from static node facts."""
    dtype = edge_dtypes(node.a, tile.inputs)[0]
    a_is_load = isinstance(node.a, Load)
    a_step = gmem_axis_step(node.a, node.axis.name, tile.inputs) if a_is_load else None
    shapes = {**tile.inputs, **tile.outputs}

    def bindable(names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            name for name in names if _atom_refusal(ATOM_REGISTRY[name], dtype, a_step, a_is_load, tail, tile.place.free, shapes) is None
        )

    if dtype is not None and dtype.nbytes == 1:
        atoms = bindable(atoms_for(dtype, ctx=target))
        return (atoms, ()) if precision_pin(FP8_MMA) else ((), atoms)
    atom_dtype = dtype if atoms_for(dtype, ctx=target) else _channel_dtype(tile, node, target)
    base = bindable(atoms_for(atom_dtype, ctx=target))
    reduced_acc = bindable(atoms_for(atom_dtype, acc=atom_dtype, ctx=target))
    return ((*base, *reduced_acc), ()) if precision_pin(F16_MMA_F32_ACC) else (base, reduced_acc)


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


def _warp_atoms(tile: TileOp, target, node, tile_pins: tuple[tuple[str, str], ...], site) -> tuple[str, ...]:
    """Project direct tensor-core atoms from contraction, dtype, address, and target facts."""
    tail = projection_tail(tile)
    if len(node.channels) != 1 or not isinstance(node.a, Load) or not isinstance(node.channels[0].b, Load):
        return ()
    if _node_refusal(tile, target, node, _fragment_epilogue_ok(tail, _fold_states(tile.op))) is not None:
        return ()
    offered, pin_only = _atom_families(tile, target, node, tail)
    site_key = f"TILE@{site.id.spell()}"
    pinned_atoms = tuple(
        atom for atom in pin_only if any(key in ("TILE", site_key) and value.split("/", 1)[0] == atom for key, value in tile_pins)
    )
    return tuple(dict.fromkeys((*offered, *pinned_atoms)))


def _contraction_domain(tile: TileOp, target, node, tile_pins: tuple[tuple[str, str], ...], site) -> tuple[ReductionSchedule, ...]:
    """Project one contraction's locally realizable direct scalar and tensor-core choices."""
    if len(node.channels) != 1:
        raise ClassicScheduleUnavailable("multi-channel contractions require tensor-core schedules")
    per_cell_reductions = _reduction_domain(tile, node) if node.axis.extent.is_static else (Reduce(),)
    plans = (
        *scalar_tile_moves(),
        *(plan for plan in warp_tile_moves(_warp_atoms(tile, target, node, tile_pins, site)) if _kstep_refusal(node.axis, plan) is None),
    )
    return tuple(
        ReductionSchedule(plan, reduction) for plan in plans for reduction in (per_cell_reductions if not plan.is_tiled else (Reduce(),))
    )


def project_domains(tile: TileOp, target) -> ClassicDomains:
    """Project independent kernel, node, and edge domains from immutable problem facts.

    Each factor is derived from the problem and its site's node-local classification only. The
    support records state the direct choices' compatibility without changing any public domain.
    """
    problem = ClassicProblem(tile.op, target)
    context = ClassicScheduleContext(problem)
    direct_edges = {edge: EdgeSchedule(Stage.direct()) for edge in context.index.edges}
    nodes = {}
    supports = {}
    work_domain = {Work()}
    tile_pins = family_pins("TILE")
    sched = Sched(tile.op, place=tile.place.on_grid())
    for site, view in context.views.items():
        incident = {edge: direct_edges[edge] for edge in context.index.edges if edge.consumer == site}
        if isinstance(view, Projection):
            choices = (ProjectionSchedule(Tile()),)
            local = tuple(LocalSupport(choice, incident) for choice in choices)
        else:
            node = context.index.node(site)
            choices = (
                _contraction_domain(tile, target, node, tile_pins, site)
                if view.contraction is not None
                else tuple(ReductionSchedule(Tile(), reduction) for reduction in _reduction_domain(tile, node))
            )
            local = []
            for choice in choices:
                geometry = sched.placed(node, choice.tile)
                if choice.tile.is_tiled and not isinstance(geometry, PlacedTile):
                    continue
                axes = (
                    tuple(AxisAgreement(side.axis.name, side.tile, side.units) for side in geometry.mn)
                    if isinstance(geometry, PlacedTile)
                    else ()
                )
                local.append(
                    LocalSupport(
                        choice,
                        incident,
                        work=derive_inventory((choice.tile,), coop=choice.reduce.coop),
                        axes=axes,
                        raster_eligible=choice.tile.is_tiled,
                    )
                )
            local = tuple(local)
            choices = tuple(support.node for support in local)
        nodes[site] = choices
        supports[site] = local
        work_domain.update(support.work for support in local if support.work is not None)
    raster_values = (
        raster_moves()
        if any(isinstance(view, Reduction) and view.contraction is not None for view in context.views.values())
        and all(axis.extent.is_static for axis in tile.place.free)
        else [""]
    )
    raster_pin = RASTER.raw()
    if raster_pin is not None and raster_pin not in raster_values:
        Raster.parse(raster_pin)
        raster_values.append(raster_pin)
    return ClassicDomains(
        kernel=tuple(
            KernelSchedule(work, Raster.parse(raster))
            for work in sorted(work_domain, key=lambda work: work.spell())
            for raster in raster_values
        ),
        nodes=nodes,
        edges={edge: (choice,) for edge, choice in direct_edges.items()},
        supports=supports,
    )


@dataclass(frozen=True)
class _ScheduleLeaf(Fork):
    """One accepted schedule assignment, materialized only if search selects it."""

    tile: TileOp
    name: str
    inherited_knobs: dict
    schedule: ClassicSchedule
    row: MappingProxyType
    is_leaf = True

    @property
    def knobs(self) -> dict:
        return {**self.inherited_knobs, **self.row}

    def expand(self) -> list[TileOp]:
        context = ClassicScheduleContext(ClassicProblem(self.tile.op, target=None))
        sched = Sched(self.tile.op, place=self.tile.place.on_grid())
        placed = {}
        for site, choice in self.schedule.nodes.items():
            if not choice.tile.is_tiled:
                continue
            geometry = sched.placed(context.index.node(site), choice.tile)
            if not isinstance(geometry, PlacedTile):
                raise ValueError(f"accepted TILE at {site.id.spell()} has no placed geometry")
            placed[site] = geometry
        return [
            scheduled(
                self.tile.op,
                name=self.name,
                place=self.tile.place.on_grid(),
                knobs=self.knobs,
                output_specs=self.tile.output_specs,
                classic=self.schedule,
                materialization=ClassicMaterialization(placed, {}),
                workers=None,
            )
        ]


def _honors_pins(row: dict[str, str], pins: dict[str, tuple[tuple[str, str], ...]]) -> bool:
    """Whether a complete canonical row satisfies every pin that addresses this kernel."""
    for family, family_values in pins.items():
        row_keys = tuple(key for key in row if key == family or key.startswith(family + "@"))
        for key, value in family_values:
            addressed = row_keys if key == family else ((key,) if key in row else ())
            if addressed and any(row[row_key] != value for row_key in addressed):
                return False
    return True


def schedule(tile: TileOp, name: str, knobs: dict, ctx) -> list[Fork]:
    """Enumerate the compatible subset of the independently projected classic domains."""
    problem = ClassicProblem(tile.op, ctx)
    context = ClassicScheduleContext(problem)
    contractions = tuple(site for site, view in context.views.items() if isinstance(view, Reduction) and view.contraction is not None)
    if len(contractions) > 1:
        raise ClassicScheduleUnavailable("composed contraction domains have not been reconstructed")
    pins = {family: family_pins(family) for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")}
    observed = any(context.index.node(site).observed for site in context.index.nodes)
    if observed and getattr(ctx, "pool_sample", None) is not None:
        raise ClassicScheduleUnavailable("sampled lazy enumeration has not been reconstructed")
    domains = project_domains(tile, ctx)
    codec = ClassicScheduleCodec(problem, domains)
    leaves = []
    for assignment in enumerate_classic(problem, domains=domains):
        row = codec.encode(assignment)
        if observed or _honors_pins(row, pins):
            leaves.append(_ScheduleLeaf(tile, name, dict(knobs), assignment, MappingProxyType(row)))
    return leaves


def _removed(*args, **kwargs):
    """Fail a directly exercised former scheduler seam through the reconstruction boundary."""
    del args, kwargs
    raise ClassicScheduleUnavailable("this classic scheduler seam has not been reconstructed")


_options = _removed
_reduce_moves = _removed
cone_seam = _removed


__all__ = ["ClassicScheduleUnavailable", "PinRefused", "project_domains", "schedule"]
