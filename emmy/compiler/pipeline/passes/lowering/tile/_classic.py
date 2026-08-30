"""Classic schedule domain projection, enumeration, and materialization.

The scheduler has one candidate-space contract: kernel, node, and edge domains are projected
independently from static facts, and enumeration is exactly the compatible subset of their
Cartesian product. Traversal order may change evaluation cost, never membership.

Projection and plain-reduction choices are live. Later schedule families extend the same
independent factors and the one compatibility relation; they do not add another enumerator.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from emmy.compiler.ir.classic_schedule import (
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
from emmy.compiler.ir.pure.fold import edge_refs_axis
from emmy.compiler.ir.schedule import Raster, Reduce, Stage, Tile, Work
from emmy.compiler.ir.stmt import Loop
from emmy.compiler.ir.stmt.passes import has_contraction_tail
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import projection_tail, scheduled
from emmy.compiler.pipeline.fork import Fork
from emmy.compiler.pipeline.knob import family_pins
from emmy.compiler.pipeline.search.space import WARP_LANES, coop_reduce_moves


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
    for site, view in context.views.items():
        incident = {edge: direct_edges[edge] for edge in context.index.edges if edge.consumer == site}
        if isinstance(view, Projection):
            choices = (ProjectionSchedule(Tile()),)
            local = tuple(LocalSupport(choice, incident) for choice in choices)
        else:
            reductions = _reduction_domain(tile, context.index.node(site))
            choices = tuple(ReductionSchedule(Tile(), reduction) for reduction in reductions)
            local = tuple(
                LocalSupport(choice, incident, work=Work(kind="thread", units=(choice.reduce.coop, 1)))
                if choice.reduce.coop > 1
                else LocalSupport(choice, incident)
                for choice in choices
            )
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
        return [
            scheduled(
                self.tile.op,
                name=self.name,
                place=self.tile.place.on_grid(),
                knobs=self.knobs,
                output_specs=self.tile.output_specs,
                classic=self.schedule,
                materialization=ClassicMaterialization({}, {}),
                workers=None,
            )
        ]


def schedule(tile: TileOp, name: str, knobs: dict, ctx) -> list[Fork]:
    """Enumerate the compatible subset of the independently projected classic domains."""
    problem = ClassicProblem(tile.op, ctx)
    context = ClassicScheduleContext(problem)
    if any(isinstance(view, Reduction) and view.contraction is not None for view in context.views.values()):
        raise ClassicScheduleUnavailable("contraction domains have not been reconstructed")
    pins = {family: family_pins(family) for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")}
    observed = any(context.index.node(site).observed for site in context.index.nodes)
    if observed and getattr(ctx, "pool_sample", None) is not None:
        raise ClassicScheduleUnavailable("sampled lazy enumeration has not been reconstructed")
    if pins["WORK"] and all(isinstance(view, Projection) for view in context.views.values()):
        _key, raw = pins["WORK"][0]
        if Work.parse(raw) != Work():
            return []
    if any(pins.values()) and not observed:
        raise ClassicScheduleUnavailable("classic schedule pin narrowing has not been reconstructed")
    domains = project_domains(tile, ctx)
    codec = ClassicScheduleCodec(problem, domains)
    return [
        _ScheduleLeaf(tile, name, dict(knobs), assignment, MappingProxyType(codec.encode(assignment)))
        for assignment in enumerate_classic(problem, domains=domains)
    ]


def _removed(*args, **kwargs):
    """Fail a directly exercised former scheduler seam through the reconstruction boundary."""
    del args, kwargs
    raise ClassicScheduleUnavailable("this classic scheduler seam has not been reconstructed")


_atom_families = _removed
_fold_states = _removed
_fragment_epilogue_ok = _removed
_kstep_refusal = _removed
_node_refusal = _removed
_options = _removed
_reduce_moves = _removed
_split_store_refusal = _removed
cone_seam = _removed


__all__ = ["ClassicScheduleUnavailable", "PinRefused", "project_domains", "schedule"]
