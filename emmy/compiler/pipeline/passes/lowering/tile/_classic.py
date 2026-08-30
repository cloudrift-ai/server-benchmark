"""Classic schedule domain projection, enumeration, and materialization.

The scheduler has one candidate-space contract: kernel, node, and edge domains are projected
independently from static facts, and enumeration is exactly the compatible subset of their
Cartesian product. Traversal order may change evaluation cost, never membership.

Reconstruction starts with the direct schedule at every site. Later schedule families extend the
same independent factors and the one compatibility relation; they do not add another enumerator.
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
    ReductionSchedule,
    enumerate_classic,
)
from emmy.compiler.ir.schedule import Raster, Reduce, Stage, Tile, Work
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import scheduled
from emmy.compiler.pipeline.fork import Fork
from emmy.compiler.pipeline.knob import family_pins


class ClassicScheduleUnavailable(RuntimeError):
    """Classic scheduling has not yet been reconstructed for this term."""


class PinRefused(ValueError):
    """A classic schedule pin may be realizable only after a structural rewrite."""


def project_domains(problem: ClassicProblem) -> ClassicDomains:
    """Project the initial independent kernel, node, and edge domains.

    Each factor is derived from the problem and its site's node-local classification only. The
    support records state the direct choices' compatibility without changing any public domain.
    """
    context = ClassicScheduleContext(problem)
    direct_edges = {edge: EdgeSchedule(Stage.direct()) for edge in context.index.edges}
    nodes = {}
    supports = {}
    for site, view in context.views.items():
        choice = ProjectionSchedule(Tile()) if isinstance(view, Projection) else ReductionSchedule(Tile(), Reduce())
        incident = {edge: direct_edges[edge] for edge in context.index.edges if edge.consumer == site}
        nodes[site] = (choice,)
        supports[site] = (LocalSupport(choice, incident),)
    return ClassicDomains(
        kernel=(KernelSchedule(Work(), Raster()),),
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
    """Enumerate the compatible subset of the independently projected direct domains."""
    problem = ClassicProblem(tile.op, ctx)
    context = ClassicScheduleContext(problem)
    if any(not isinstance(view, Projection) for view in context.views.values()):
        raise ClassicScheduleUnavailable("reduction and contraction domains have not been reconstructed")
    if any(family_pins(family) for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")):
        raise ClassicScheduleUnavailable("classic schedule pin narrowing has not been reconstructed")
    domains = project_domains(problem)
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
