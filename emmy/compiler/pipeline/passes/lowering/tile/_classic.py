"""Classic schedule domain projection, enumeration, and materialization.

The scheduler has one candidate-space contract: kernel, node, and edge domains are projected
independently from static facts, and enumeration is exactly the compatible subset of their
Cartesian product. Traversal order may change evaluation cost, never membership.

Projection, plain-reduction, scalar-contraction, and gmem-direct tensor-core choices are live.
Later schedule families extend the same independent factors and the one compatibility relation;
they do not add another enumerator.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from emmy.compiler.ir.atom import ATOM_REGISTRY, atoms_for
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
from emmy.compiler.pipeline.search.space import WARP_LANES, coop_reduce_moves, scalar_tile_moves, warp_tile_moves


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


def _split_store_ok(tile: TileOp, atom_name: str) -> bool:
    """Whether one atom's fragment coordinates address every tail load and store."""
    atom = ATOM_REGISTRY[atom_name]
    free = tile.place.free
    shapes = {**tile.inputs, **tile.outputs}
    roles = [(free[-1].name, atom.atom_n, True)]
    if len(free) >= 2:
        roles.append((free[-2].name, atom.atom_m, False))
    for stmt in projection_tail(tile):
        if not isinstance(stmt, (Load, Write)):
            continue
        buffer = stmt.input if isinstance(stmt, Load) else stmt.output
        shape = getattr(shapes.get(buffer), "shape", None)
        if any(not split_addressable(stmt.index, shape, name, extent, trailing) for name, extent, trailing in roles):
            return False
    return True


def _warp_atoms(tile: TileOp, target, node) -> tuple[str, ...]:
    """Project direct tensor-core atoms from contraction, dtype, address, and target facts."""
    ring = node.semiring
    if (
        len(node.channels) != 1
        or ring is None
        or tuple(operator.name for operator in ring) != ("multiply", "add")
        or len(tile.place.free) < 2
        or not tile.inputs
        or not _fragment_epilogue_ok(projection_tail(tile), _fold_states(tile.op))
        or not isinstance(node.a, Load)
        or not isinstance(node.channels[0].b, Load)
    ):
        return ()
    a_dtype = edge_dtypes(node.a, tile.inputs)[0]
    b_dtype = edge_dtypes(node.channels[0].b, tile.inputs)[0]
    if a_dtype != b_dtype:
        return ()
    step = gmem_axis_step(node.a, node.axis.name, tile.inputs)
    return tuple(
        name
        for name in atoms_for(a_dtype, ctx=target)
        if step is not None and step[0] == 1 and (step[1] == 0 or step[1] % ATOM_REGISTRY[name].atom_k == 0) and _split_store_ok(tile, name)
    )


def _contraction_domain(tile: TileOp, target, node) -> tuple[ReductionSchedule, ...]:
    """Project one contraction's locally realizable direct scalar and tensor-core choices."""
    if len(node.channels) != 1:
        raise ClassicScheduleUnavailable("multi-channel contractions require tensor-core schedules")
    per_cell_reductions = _reduction_domain(tile, node) if node.axis.extent.is_static else (Reduce(),)
    plans = (*scalar_tile_moves(), *warp_tile_moves(_warp_atoms(tile, target, node)))
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
    sched = Sched(tile.op, place=tile.place.on_grid())
    for site, view in context.views.items():
        incident = {edge: direct_edges[edge] for edge in context.index.edges if edge.consumer == site}
        if isinstance(view, Projection):
            choices = (ProjectionSchedule(Tile()),)
            local = tuple(LocalSupport(choice, incident) for choice in choices)
        else:
            node = context.index.node(site)
            choices = (
                _contraction_domain(tile, target, node)
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
    return ClassicDomains(
        kernel=tuple(KernelSchedule(work, Raster()) for work in sorted(work_domain, key=lambda work: work.spell())),
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


_atom_families = _removed
_kstep_refusal = _removed
_node_refusal = _removed
_options = _removed
_reduce_moves = _removed
_split_store_refusal = _removed
cone_seam = _removed


__all__ = ["ClassicScheduleUnavailable", "PinRefused", "project_domains", "schedule"]
