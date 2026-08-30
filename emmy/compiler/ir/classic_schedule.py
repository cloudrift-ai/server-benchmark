"""The semantic model for the classic grid/CTA/warp/thread/register schedule.

The model contains choices only.  A :class:`ClassicProblem` supplies the immutable Fold tree and
target; :class:`ClassicScheduleContext` derives identities and classification from that problem and
is the only authority that accepts or refuses a complete assignment.  Codecs, enumeration order,
search state, and materialization data do not belong here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType

from emmy.compiler.ir.pure.fold import Fold, is_contraction
from emmy.compiler.ir.schedule import (
    PlacedTile,
    Raster,
    Reduce,
    ResolvedStage,
    Stage,
    Tile,
    Work,
    derive_inventory,
    resolve_site_tile,
)

CLASSIC_NODE_FAMILIES = ("TILE", "REDUCE")
CLASSIC_EDGE_FAMILIES = ("STAGE",)
CLASSIC_FAMILIES = (*CLASSIC_NODE_FAMILIES, *CLASSIC_EDGE_FAMILIES)


@dataclass(frozen=True, order=True)
class NodeId:
    """A problem-local node identity, assigned in stable preorder."""

    ordinal: int

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError(f"node ordinal must be a non-negative integer, got {self.ordinal!r}")

    def spell(self) -> str:
        """Return the one serialized spelling of this identity."""
        return f"n{self.ordinal}"

    @classmethod
    def parse(cls, value: str) -> NodeId:
        """Parse :meth:`spell`, rejecting every other form."""
        if not value.startswith("n") or not value[1:].isdigit() or str(int(value[1:])) != value[1:]:
            raise ValueError(f"node id must be n<ordinal>, got {value!r}")
        return cls(int(value[1:]))


@dataclass(frozen=True, order=True)
class NodeSite:
    """The one schedule site of a Fold node."""

    id: NodeId

    def __post_init__(self) -> None:
        if not isinstance(self.id, NodeId):
            raise TypeError("classic node site requires a NodeId")


@dataclass(frozen=True, order=True)
class EdgeSite:
    """One operand use at a consumer node."""

    consumer: NodeSite
    operand: int

    def __post_init__(self) -> None:
        if not isinstance(self.consumer, NodeSite):
            raise TypeError("classic edge site requires a NodeSite consumer")
        if type(self.operand) is not int or self.operand < 0:
            raise ValueError(f"edge operand must be a non-negative integer, got {self.operand!r}")

    def spell(self) -> str:
        """Return the one serialized spelling of this edge identity."""
        return f"{self.consumer.id.spell()}.e{self.operand}"

    @classmethod
    def parse(cls, value: str) -> EdgeSite:
        """Parse :meth:`spell`, rejecting aliases and noncanonical integers."""
        node, separator, operand = value.partition(".e")
        if separator != ".e" or not operand.isdigit() or str(int(operand)) != operand:
            raise ValueError(f"edge site must be n<ordinal>.e<operand>, got {value!r}")
        return cls(NodeSite(NodeId.parse(node)), int(operand))


@dataclass(frozen=True)
class Projection:
    """The classification of a zero-axis Fold."""


@dataclass(frozen=True)
class Contraction:
    """A reduction's bilinear operand roles, expressed as consumer operand positions."""

    a: int
    channels: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.a) is not int or self.a < 0:
            raise ValueError(f"contraction A role must be a non-negative operand position, got {self.a!r}")
        if not isinstance(self.channels, tuple) or any(type(position) is not int or position < 0 for position in self.channels):
            raise ValueError("contraction channel roles must be non-negative operand positions")


@dataclass(frozen=True)
class Reduction:
    """The classification of an iterating Fold."""

    contraction: Contraction | None = None

    def __post_init__(self) -> None:
        if self.contraction is not None and not isinstance(self.contraction, Contraction):
            raise TypeError("reduction contraction capability must be a Contraction or None")


NodeView = Projection | Reduction


@dataclass(frozen=True)
class KernelSchedule:
    """Kernel-scoped choices."""

    work: Work
    raster: Raster

    def __post_init__(self) -> None:
        if not isinstance(self.work, Work) or not isinstance(self.raster, Raster):
            raise TypeError("classic kernel choices must be Work and Raster values")


@dataclass(frozen=True)
class ProjectionSchedule:
    """The choices of a projection node."""

    tile: Tile

    def __post_init__(self) -> None:
        if not isinstance(self.tile, Tile):
            raise TypeError("classic projection TILE must be an unplaced Tile choice")


@dataclass(frozen=True)
class ReductionSchedule:
    """The choices of a reduction node, including a contraction-capable reduction."""

    tile: Tile
    reduce: Reduce

    def __post_init__(self) -> None:
        if not isinstance(self.tile, Tile):
            raise TypeError("classic reduction TILE must be an unplaced Tile choice")
        if not isinstance(self.reduce, Reduce):
            raise TypeError("classic reduction REDUCE must be a Reduce choice")


NodeSchedule = ProjectionSchedule | ReductionSchedule


@dataclass(frozen=True)
class EdgeSchedule:
    """The transport choice of one operand use."""

    stage: Stage

    def __post_init__(self) -> None:
        if not isinstance(self.stage, Stage):
            raise TypeError("classic edge STAGE must be a Stage choice")


@dataclass(frozen=True)
class ClassicSchedule:
    """One complete classic schedule assignment."""

    kernel: KernelSchedule
    nodes: Mapping[NodeSite, NodeSchedule]
    edges: Mapping[EdgeSite, EdgeSchedule]

    def __post_init__(self) -> None:
        if not isinstance(self.kernel, KernelSchedule):
            raise TypeError(f"classic kernel assignment must be KernelSchedule, got {type(self.kernel).__name__}")
        if not isinstance(self.nodes, Mapping) or not isinstance(self.edges, Mapping):
            raise TypeError("classic node and edge assignments must be mappings")
        if any(not isinstance(site, NodeSite) for site in self.nodes):
            raise TypeError("classic node assignments must be keyed by NodeSite")
        if any(not isinstance(assignment, (ProjectionSchedule, ReductionSchedule)) for assignment in self.nodes.values()):
            raise TypeError("classic node assignments must contain projection or reduction schedules")
        if any(not isinstance(edge, EdgeSite) for edge in self.edges):
            raise TypeError("classic edge assignments must be keyed by EdgeSite")
        if any(not isinstance(assignment, EdgeSchedule) for assignment in self.edges.values()):
            raise TypeError("classic edge assignments must contain edge schedules")
        object.__setattr__(self, "nodes", MappingProxyType(dict(self.nodes)))
        object.__setattr__(self, "edges", MappingProxyType(dict(self.edges)))

    def __reduce__(self):
        """Rebuild read-only mappings after process transport."""
        return type(self), (self.kernel, dict(self.nodes), dict(self.edges))


@dataclass(frozen=True)
class AxisAgreement:
    """One physical-axis geometry claim carried by a local schedule offer."""

    name: str
    tile: int
    units: int


@dataclass(frozen=True)
class FragmentAgreement:
    """One producer or consumer claim at a fragment seam."""

    role: str
    edge: str
    value: tuple

    def __post_init__(self) -> None:
        if self.role not in ("need", "offer"):
            raise ValueError(f"fragment agreement role must be need or offer, got {self.role!r}")


@dataclass(frozen=True)
class LocalSupport:
    """Static support for one node choice and its incident edge choices.

    The public domains are projections of these records.  A support record is not a schedule:
    placed geometry and fragment facts remain derived compatibility evidence and never enter a
    :class:`ClassicSchedule` value.
    """

    node: NodeSchedule
    edges: Mapping[EdgeSite, EdgeSchedule]
    work: Work | None = None
    axes: tuple[AxisAgreement, ...] = ()
    fragments: tuple[FragmentAgreement, ...] = ()
    raster_eligible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.node, (ProjectionSchedule, ReductionSchedule)):
            raise TypeError("classic local support requires a node schedule")
        if not isinstance(self.edges, Mapping) or any(
            not isinstance(edge, EdgeSite) or not isinstance(choice, EdgeSchedule) for edge, choice in self.edges.items()
        ):
            raise TypeError("classic local support edges must map EdgeSite to EdgeSchedule")
        if self.work is not None and not isinstance(self.work, Work):
            raise TypeError("classic local support work must be Work or None")
        object.__setattr__(self, "edges", MappingProxyType(dict(self.edges)))


@dataclass(frozen=True)
class ClassicDomains:
    """Independent finite domains plus their static compatibility support.

    ``kernel``, ``nodes`` and ``edges`` are the literal Cartesian factors.  ``supports`` records
    why combinations of those independently projected choices can coexist; it is consumed only by
    :meth:`ClassicScheduleContext.accepts` and by a private prefix propagator.
    """

    kernel: tuple[KernelSchedule, ...]
    nodes: Mapping[NodeSite, tuple[NodeSchedule, ...]]
    edges: Mapping[EdgeSite, tuple[EdgeSchedule, ...]]
    supports: Mapping[NodeSite, tuple[LocalSupport, ...]]

    def __post_init__(self) -> None:
        if not self.kernel or any(not isinstance(choice, KernelSchedule) for choice in self.kernel):
            raise TypeError("classic kernel domain must contain KernelSchedule choices")
        for name, values, site_type, choice_type in (
            ("node", self.nodes, NodeSite, (ProjectionSchedule, ReductionSchedule)),
            ("edge", self.edges, EdgeSite, EdgeSchedule),
            ("support", self.supports, NodeSite, LocalSupport),
        ):
            if not isinstance(values, Mapping) or any(not isinstance(site, site_type) for site in values):
                raise TypeError(f"classic {name} domains have invalid site keys")
            if any(not choices or any(not isinstance(choice, choice_type) for choice in choices) for choices in values.values()):
                raise TypeError(f"classic {name} domains have invalid choices")
        object.__setattr__(self, "nodes", MappingProxyType({site: tuple(choices) for site, choices in self.nodes.items()}))
        object.__setattr__(self, "edges", MappingProxyType({edge: tuple(choices) for edge, choices in self.edges.items()}))
        object.__setattr__(self, "supports", MappingProxyType({site: tuple(choices) for site, choices in self.supports.items()}))

    @property
    def product_size(self) -> int:
        """Number of assignments in the unfiltered Cartesian product."""
        size = len(self.kernel)
        for choices in (*self.nodes.values(), *self.edges.values()):
            size *= len(choices)
        return size


@dataclass(frozen=True)
class ClassicMaterialization:
    """Placed geometry and resolved transport facts derived from an accepted schedule."""

    tiles: Mapping[NodeSite, PlacedTile]
    stages: Mapping[EdgeSite, ResolvedStage]

    def __post_init__(self) -> None:
        if not isinstance(self.tiles, Mapping) or not isinstance(self.stages, Mapping):
            raise TypeError("classic materialization tiles and stages must be mappings")
        if any(not isinstance(site, NodeSite) or not isinstance(tile, PlacedTile) for site, tile in self.tiles.items()):
            raise TypeError("classic materialization tiles must map NodeSite to PlacedTile")
        if any(not isinstance(edge, EdgeSite) or not isinstance(stage, ResolvedStage) for edge, stage in self.stages.items()):
            raise TypeError("classic materialization stages must map EdgeSite to ResolvedStage")
        object.__setattr__(self, "tiles", MappingProxyType(dict(self.tiles)))
        object.__setattr__(self, "stages", MappingProxyType(dict(self.stages)))

    def __reduce__(self):
        """Rebuild read-only mappings after process transport."""
        return type(self), (dict(self.tiles), dict(self.stages))


@dataclass(frozen=True)
class ClassicProblem:
    """The unscheduled Fold tree and compilation target."""

    root: Fold
    target: object

    def __post_init__(self) -> None:
        if not isinstance(self.root, Fold):
            raise TypeError("classic problem root must be a Fold")


class SiteIndex:
    """Immutable problem-local lookup for node and edge sites."""

    def __init__(self, root: Fold) -> None:
        nodes: list[Fold] = []
        seen: set[int] = set()
        for node in _walk(root):
            if id(node) in seen:
                continue
            seen.add(id(node))
            nodes.append(node)
        self._nodes = tuple(nodes)
        self._sites = tuple(NodeSite(NodeId(index)) for index in range(len(nodes)))
        self._site_by_identity = {id(node): site for node, site in zip(self._nodes, self._sites, strict=True)}
        self._edges = tuple(
            EdgeSite(site, operand) for node, site in zip(self._nodes, self._sites, strict=True) for operand in range(len(node.operands))
        )

    @property
    def nodes(self) -> tuple[NodeSite, ...]:
        """Node sites in canonical order."""
        return self._sites

    @property
    def edges(self) -> tuple[EdgeSite, ...]:
        """Edge sites in canonical consumer/operand order."""
        return self._edges

    def node(self, site: NodeSite) -> Fold:
        """Resolve a node site."""
        if site.id.ordinal >= len(self._nodes):
            raise KeyError(f"unknown node site {site.id.spell()}") from None
        return self._nodes[site.id.ordinal]

    def site(self, node: Fold) -> NodeSite:
        """Return the one site of ``node`` by object identity."""
        try:
            return self._site_by_identity[id(node)]
        except KeyError:
            raise KeyError("Fold is not a node of this classic problem") from None

    def operand(self, edge: EdgeSite):
        """Resolve the producer value used at an edge site."""
        node = self.node(edge.consumer)
        try:
            return node.operands[edge.operand]
        except IndexError:
            raise KeyError(f"unknown operand {edge.operand} at {edge.consumer.id.spell()}") from None

    def producer(self, edge: EdgeSite) -> NodeSite | None:
        """Return the producer node site when this edge reads an inline Fold."""
        value = self.operand(edge)
        return self.site(value) if isinstance(value, Fold) else None


def classify(index: SiteIndex, site: NodeSite) -> NodeView:
    """Classify one node site without reading the target."""
    node = index.node(site)
    if node.axis is None:
        return Projection()
    if not is_contraction(node):
        return Reduction()
    return Reduction(
        Contraction(
            a=_operand_position(node, node.a),
            channels=tuple(_operand_position(node, channel.b) for channel in node.channels),
        )
    )


@dataclass(frozen=True)
class Refusal:
    """A stable explanation of why a complete assignment is not a classic schedule."""

    reason: str
    site: NodeSite | EdgeSite | None = None


@dataclass(frozen=True)
class Acceptance:
    """The result of checking one complete schedule."""

    refusal: Refusal | None = None

    def __bool__(self) -> bool:
        return self.refusal is None


class ClassicScheduleContext:
    """Derived facts and complete-assignment invariants for one classic problem."""

    def __init__(self, problem: ClassicProblem, domains: ClassicDomains | None = None) -> None:
        self.problem = problem
        self.index = SiteIndex(problem.root)
        self.views = MappingProxyType({site: classify(self.index, site) for site in self.index.nodes})
        self._node_sites = frozenset(self.index.nodes)
        self._edge_sites = frozenset(self.index.edges)
        self._tile_sites = frozenset(tile_sites(self))
        self._stage_edges = frozenset(stage_edges(self))
        self._incident_edges = {site: tuple(edge for edge in self.index.edges if edge.consumer == site) for site in self.index.nodes}
        self.domains = domains
        self._kernel_domain = frozenset()
        self._node_domains = {}
        self._edge_domains = {}
        self._support_index = {}
        if domains is not None:
            expected_nodes = self._node_sites
            expected_edges = self._edge_sites
            if set(domains.nodes) != expected_nodes or set(domains.supports) != expected_nodes:
                raise ValueError("classic domains must cover every node site exactly once")
            if set(domains.edges) != expected_edges:
                raise ValueError("classic domains must cover every edge site exactly once")
            self._kernel_domain = frozenset(domains.kernel)
            self._node_domains = {site: frozenset(choices) for site, choices in domains.nodes.items()}
            self._edge_domains = {edge: frozenset(choices) for edge, choices in domains.edges.items()}
            for site in self.index.nodes:
                incident = self._incident_edges[site]
                indexed: dict[tuple, list[LocalSupport]] = {}
                for support in domains.supports[site]:
                    key = (support.node, *(support.edges[edge] for edge in incident))
                    indexed.setdefault(key, []).append(support)
                self._support_index[site] = {key: tuple(values) for key, values in indexed.items()}

    def accepts(self, schedule: ClassicSchedule) -> Acceptance:
        """Accept a complete assignment or return the first semantic incompatibility."""
        if not isinstance(schedule, ClassicSchedule):
            return Acceptance(Refusal("assignment must be a ClassicSchedule"))
        if not isinstance(schedule.kernel.work, Work) or not isinstance(schedule.kernel.raster, Raster):
            return Acceptance(Refusal("kernel assignment must contain explicit Work and Raster choices"))
        expected_nodes = self._node_sites
        actual_nodes = schedule.nodes.keys()
        if missing := expected_nodes - actual_nodes:
            return Acceptance(Refusal("missing node assignment", min(missing)))
        if extra := actual_nodes - expected_nodes:
            return Acceptance(Refusal("node assignment is outside this problem", min(extra)))

        expected_edges = self._edge_sites
        actual_edges = schedule.edges.keys()
        if missing := expected_edges - actual_edges:
            return Acceptance(Refusal("missing edge assignment", min(missing)))
        if extra := actual_edges - expected_edges:
            return Acceptance(Refusal("edge assignment is outside this problem", min(extra)))

        claimed_work: Work | None = None
        for site in self.index.nodes:
            view = self.views[site]
            assignment = schedule.nodes[site]
            if isinstance(view, Projection) and not isinstance(assignment, ProjectionSchedule):
                return Acceptance(Refusal("projection site requires a projection schedule", site))
            if isinstance(view, Reduction) and not isinstance(assignment, ReductionSchedule):
                return Acceptance(Refusal("reduction site requires a reduction schedule", site))
            if isinstance(assignment.tile, PlacedTile):
                return Acceptance(Refusal("node choices cannot contain placed tile geometry", site))
            if site not in self._tile_sites and assignment.tile != Tile():
                return Acceptance(Refusal("this node has no tile choice", site))
            if assignment.tile.is_warp and hasattr(self.problem.target, assignment.tile.atom.target_feature):
                if not assignment.tile.atom.available_on(self.problem.target):
                    return Acceptance(Refusal("tile atom is unavailable on the target", site))
            coop = assignment.reduce.coop if isinstance(assignment, ReductionSchedule) else 1
            try:
                work = derive_inventory((assignment.tile,), coop=coop)
            except ValueError as error:
                return Acceptance(Refusal(str(error), site))
            if work is not None:
                if claimed_work is not None and claimed_work != work:
                    return Acceptance(Refusal("node choices require different worker inventories", site))
                claimed_work = work
        kernel_work = Work(schedule.kernel.work.kind, schedule.kernel.work.units)
        if (claimed_work or Work()) != kernel_work:
            return Acceptance(Refusal("kernel WORK does not realize the node choices"))
        warp_size = getattr(self.problem.target, "warp_size", 32)
        max_threads = getattr(self.problem.target, "max_threads_per_cta", 1024)
        compute_threads = kernel_work.count * (warp_size if kernel_work.kind == "warp" else 1)
        producer_threads = schedule.kernel.work.producer * warp_size
        if producer_threads > compute_threads:
            return Acceptance(Refusal("producer band cannot outnumber the compute band"))
        if compute_threads + producer_threads > max_threads:
            return Acceptance(Refusal("worker inventory exceeds the target thread limit"))
        if schedule.kernel.raster != Raster() and not any(
            isinstance(self.views[site], Reduction) and self.views[site].contraction is not None and schedule.nodes[site].tile.is_tiled
            for site in self.index.nodes
        ):
            return Acceptance(Refusal("RASTER requires a tiled contraction site"))
        for edge in self.index.edges:
            edge_assignment = schedule.edges[edge]
            if not isinstance(edge_assignment, EdgeSchedule):
                return Acceptance(Refusal("edge assignment must be an EdgeSchedule", edge))
            stage = edge_assignment.stage
            if not isinstance(stage, Stage):
                return Acceptance(Refusal("edge assignment must contain an explicit Stage choice", edge))
            if edge not in self._stage_edges and not stage.is_direct:
                return Acceptance(Refusal("this edge has no staged transport choice", edge))
            assignment = schedule.nodes[edge.consumer]
            if not stage.is_direct and not assignment.tile.is_tiled:
                return Acceptance(Refusal("staged transport requires a tiled consumer", edge))
            if not stage.is_direct and hasattr(self.problem.target, "has_cp_async") and not stage.available_on(self.problem.target):
                return Acceptance(Refusal("transport is unavailable on the target", edge))
        for site in self.index.nodes:
            stages = {schedule.edges[edge].stage for edge in self._incident_edges[site]}
            if len(stages) > 1:
                return Acceptance(Refusal("one contraction currently requires one transport choice across its operands", site))
        if schedule.kernel.work.producer:
            for site in self.index.nodes:
                assignment = schedule.nodes[site]
                if not assignment.tile.is_tiled:
                    continue
                if isinstance(assignment, ReductionSchedule) and assignment.reduce.needs_split:
                    return Acceptance(Refusal("a producer band cannot accompany a cross-CTA reduction", site))
                edges = tuple(edge for edge in self._incident_edges[site] if edge in self._stage_edges)
                if not edges or any(schedule.edges[edge].stage.transport != "smem-tma" for edge in edges):
                    return Acceptance(Refusal("a producer band requires TMA transport at every tiled consumer", site))
        if self.domains is not None:
            verdict = _accepts_domains(self, schedule)
            if not verdict:
                return verdict
        return Acceptance()


def _accepts_domains(context: ClassicScheduleContext, schedule: ClassicSchedule) -> Acceptance:
    """Check one complete assignment against independent domains and their support relation."""
    assert context.domains is not None
    if schedule.kernel not in context._kernel_domain:
        return Acceptance(Refusal("kernel choice is outside its independent domain"))
    for site in context.index.nodes:
        if schedule.nodes[site] not in context._node_domains[site]:
            return Acceptance(Refusal("node choice is outside its independent domain", site))
    for edge in context.index.edges:
        if schedule.edges[edge] not in context._edge_domains[edge]:
            return Acceptance(Refusal("edge choice is outside its independent domain", edge))

    candidates: list[tuple[NodeSite, tuple[LocalSupport, ...]]] = []
    for site in context.index.nodes:
        incident = context._incident_edges[site]
        key = (schedule.nodes[site], *(schedule.edges[edge] for edge in incident))
        matches = context._support_index[site].get(key, ())
        if not matches:
            return Acceptance(Refusal("node and incident edge choices have no static support", site))
        candidates.append((site, matches))

    def visit(
        position: int,
        work: Work | None,
        axes: dict[str, tuple[int, int]],
        fragments: dict[tuple[str, str], tuple],
        raster_eligible: bool,
    ) -> bool:
        if position == len(candidates):
            return (work or Work()) == schedule.kernel.work and (schedule.kernel.raster.is_direct or raster_eligible)
        _site, supports = candidates[position]
        for support in supports:
            next_work = work
            if support.work is not None:
                if next_work is not None and next_work != support.work:
                    continue
                next_work = support.work
            next_axes = axes
            rejected = False
            if support.axes:
                next_axes = dict(axes)
                for claim in support.axes:
                    value = (claim.tile, claim.units)
                    if next_axes.setdefault(claim.name, value) != value:
                        rejected = True
                        break
            if rejected:
                continue
            next_fragments = fragments
            if support.fragments:
                next_fragments = dict(fragments)
                for claim in support.fragments:
                    key = (claim.role, claim.edge)
                    if next_fragments.setdefault(key, claim.value) != claim.value:
                        rejected = True
                        break
                    other_role = "need" if claim.role == "offer" else "offer"
                    other = next_fragments.get((other_role, claim.edge))
                    if other is not None:
                        need, offer = (claim.value, other) if claim.role == "need" else (other, claim.value)
                        if not _fragment_seam_ok(need, offer):
                            rejected = True
                            break
            if rejected:
                continue
            if visit(
                position + 1,
                next_work,
                next_axes,
                next_fragments,
                raster_eligible or support.raster_eligible,
            ):
                return True
        return False

    if not visit(0, None, {}, {}, False):
        return Acceptance(Refusal("complete choices violate the classic compatibility relation"))
    return Acceptance()


def _fragment_seam_ok(need: tuple, offer: tuple) -> bool:
    """Whether one consumer fragment requirement composes with a producer offer."""
    if offer[0] == "free":
        return need[0] != "step"
    if need[0] not in ("warp", "step") or offer[0] != "warp":
        return False
    _, shape, layout, bk = need
    _, offer_shape, offer_layout, offer_units_n, offer_tile_n = offer
    return shape == offer_shape and layout == offer_layout and offer_units_n == 1 and offer_tile_n == bk


class ClassicScheduleCodec:
    """Strict wire boundary for complete classic schedules.

    Kernel families are bare. Every node and edge family carries the one canonical site suffix: a
    :class:`NodeId`, with ``.e<N>`` for an operand edge. Decoding accepts no aliases, missing
    direct values, or unknown fields.
    """

    def __init__(self, problem: ClassicProblem, domains: ClassicDomains | None = None) -> None:
        self.context = ClassicScheduleContext(problem, domains)
        self._tile_sites = tile_sites(self.context)
        self._reduce_sites = reduction_sites(self.context)
        self._stage_sites = stage_edges(self.context)
        self._key_order = (
            "WORK",
            "RASTER",
            *(self._node_key("TILE", site, self._tile_sites) for site in self._tile_sites),
            *(self._node_key("REDUCE", site, self._reduce_sites) for site in self._reduce_sites),
            *(self._edge_key(edge) for edge in self._stage_sites),
        )
        self._keys = frozenset(self._key_order)

    def encode(self, schedule: ClassicSchedule) -> dict[str, str]:
        """Encode one accepted typed schedule in canonical scope order."""
        verdict = self.context.accepts(schedule)
        if not verdict:
            raise ValueError(_refusal_message(verdict.refusal))
        return self._encode_accepted(schedule)

    def _encode_accepted(self, schedule: ClassicSchedule) -> dict[str, str]:
        """Encode a schedule whose compatibility was already checked at this boundary."""
        row = {
            "WORK": schedule.kernel.work.spell(),
            "RASTER": schedule.kernel.raster.spell(),
        }
        for site in self._tile_sites:
            row[self._node_key("TILE", site, self._tile_sites)] = schedule.nodes[site].tile.spell()
        for site in self._reduce_sites:
            assignment = schedule.nodes[site]
            assert isinstance(assignment, ReductionSchedule)
            row[self._node_key("REDUCE", site, self._reduce_sites)] = assignment.reduce.spell()
        for edge in self._stage_sites:
            row[self._edge_key(edge)] = schedule.edges[edge].stage.spell()
        return row

    def decode(self, row: Mapping[str, str]) -> ClassicSchedule:
        """Decode one complete canonical row and reject every other key set or assignment."""
        self._check_keys(row)

        work = Work.parse(row["WORK"])
        nodes: dict[NodeSite, NodeSchedule] = {}
        for site in self.context.index.nodes:
            view = self.context.views[site]
            reduce = None
            if isinstance(view, Reduction):
                reduce = Reduce.parse(row[self._node_key("REDUCE", site, self._reduce_sites)], work)
            tile = (
                resolve_site_tile(
                    row[self._node_key("TILE", site, self._tile_sites)],
                    work,
                    reduce.coop if reduce is not None else 1,
                )
                if site in self._tile_sites
                else Tile()
            )
            nodes[site] = ProjectionSchedule(tile) if reduce is None else ReductionSchedule(tile, reduce)
        schedule = ClassicSchedule(
            KernelSchedule(work, Raster.parse(row["RASTER"])),
            nodes,
            {
                edge: EdgeSchedule(Stage.parse(row[self._edge_key(edge)])) if edge in self._stage_sites else EdgeSchedule(Stage.direct())
                for edge in self.context.index.edges
            },
        )
        return self._validate_row(schedule, row)

    def _validate_row(self, schedule: ClassicSchedule, row: Mapping[str, str]) -> ClassicSchedule:
        """Validate a parsed assignment and its claimed canonical row exactly once."""
        verdict = self.context.accepts(schedule)
        if not verdict:
            raise ValueError(_refusal_message(verdict.refusal))
        canonical = self._encode_accepted(schedule)
        if dict(row) != canonical:
            raise ValueError("classic schedule row is not its typed schedule's canonical encoding")
        return schedule

    def _check_keys(self, row: Mapping[str, str]) -> None:
        """Require the codec's exact key set before parsing or validating values."""
        expected = self._keys
        actual = set(row)
        if missing := expected - actual:
            raise ValueError(f"classic schedule row is missing {', '.join(sorted(missing))}")
        if extra := actual - expected:
            raise ValueError(f"classic schedule row has unknown keys {', '.join(sorted(extra))}")

    def keys(self) -> tuple[str, ...]:
        """Return accepted keys in canonical encoding order."""
        return self._key_order

    @staticmethod
    def _node_key(family: str, site: NodeSite, family_sites: Sequence[NodeSite]) -> str:
        return node_key(family, site, family_sites)

    def _edge_key(self, edge: EdgeSite) -> str:
        return edge_key(edge, self._stage_sites)


def _refusal_message(refusal: Refusal | None) -> str:
    if refusal is None:
        return "classic schedule refused"
    if refusal.site is None:
        return refusal.reason
    if isinstance(refusal.site, NodeSite):
        where = refusal.site.id.spell()
    else:
        where = f"{refusal.site.consumer.id.spell()}.e{refusal.site.operand}"
    return f"{where}: {refusal.reason}"


def tile_sites(context: ClassicScheduleContext) -> tuple[NodeSite, ...]:
    """Node sites whose tile domain contains more than the fixed direct choice."""
    out = []
    for site in context.index.nodes:
        view = context.views[site]
        node = context.index.node(site)
        if (isinstance(view, Reduction) and view.contraction is not None) or (
            isinstance(view, Projection) and site == context.index.nodes[0] and not node.operands
        ):
            out.append(site)
    return tuple(out)


def stage_edges(context: ClassicScheduleContext) -> tuple[EdgeSite, ...]:
    """Operand edges whose transport domain belongs to a contraction."""
    return tuple(
        edge
        for edge in context.index.edges
        if isinstance((view := context.views[edge.consumer]), Reduction) and view.contraction is not None
    )


def reduction_sites(context: ClassicScheduleContext) -> tuple[NodeSite, ...]:
    """Node sites whose schedule includes a reduction choice."""
    return tuple(site for site in context.index.nodes if isinstance(context.views[site], Reduction))


def node_key(family: str, site: NodeSite, family_sites: Sequence[NodeSite]) -> str:
    """Return the sole canonical codec key for a node family site."""
    if site not in family_sites:
        raise ValueError(f"{site.id.spell()} is not a {family} site")
    return f"{family}@{site.id.spell()}"


def edge_key(edge: EdgeSite, family_edges: Sequence[EdgeSite]) -> str:
    """Return the sole canonical codec key for an edge family site."""
    if edge not in family_edges:
        raise ValueError(f"{edge.consumer.id.spell()}.e{edge.operand} is not a STAGE site")
    return f"STAGE@{edge.spell()}"


def kernel_domain(problem: ClassicProblem, domains: ClassicDomains | None = None) -> tuple[KernelSchedule, ...]:
    """Return the kernel factor without inspecting a selected node or edge choice."""
    del problem
    return domains.kernel if domains is not None else (KernelSchedule(Work(), Raster()),)


def node_domain(
    problem: ClassicProblem,
    site: NodeSite,
    view: NodeView,
    domains: ClassicDomains | None = None,
) -> tuple[NodeSchedule, ...]:
    """Return one node factor without inspecting any selected schedule."""
    index = SiteIndex(problem.root)
    if classify(index, site) != view:
        raise ValueError(f"view does not classify {site.id.spell()} in this problem")
    if domains is not None:
        try:
            return domains.nodes[site]
        except KeyError:
            raise ValueError(f"classic domains do not contain {site.id.spell()}") from None
    if isinstance(view, Projection):
        return (ProjectionSchedule(Tile()),)
    return (ReductionSchedule(Tile(), Reduce()),)


def edge_domain(
    problem: ClassicProblem,
    edge: EdgeSite,
    domains: ClassicDomains | None = None,
) -> tuple[EdgeSchedule, ...]:
    """Return one edge factor without inspecting any selected schedule."""
    index = SiteIndex(problem.root)
    index.operand(edge)
    if domains is not None:
        try:
            return domains.edges[edge]
        except KeyError:
            raise ValueError(f"classic domains do not contain {edge.spell()}") from None
    return (EdgeSchedule(Stage.direct()),)


def cartesian_assignments(
    problem: ClassicProblem,
    domains: ClassicDomains | None = None,
) -> Iterator[tuple[ClassicSchedule, Acceptance]]:
    """Enumerate the literal independent-domain product and each membership verdict."""
    context = ClassicScheduleContext(problem, domains)
    node_domains = tuple(node_domain(problem, site, context.views[site], domains) for site in context.index.nodes)
    edge_domains = tuple(edge_domain(problem, edge, domains) for edge in context.index.edges)
    for kernel, node_values, edge_values in product(
        kernel_domain(problem, domains),
        product(*node_domains),
        product(*edge_domains),
    ):
        schedule = ClassicSchedule(
            kernel,
            dict(zip(context.index.nodes, node_values, strict=True)),
            dict(zip(context.index.edges, edge_values, strict=True)),
        )
        yield schedule, context.accepts(schedule)


def enumerate_reference(problem: ClassicProblem, domains: ClassicDomains | None = None) -> Iterator[ClassicSchedule]:
    """Yield the compatible subset of the literal Cartesian product."""
    for schedule, verdict in cartesian_assignments(problem, domains):
        if verdict:
            yield schedule


def enumerate_classic(
    problem: ClassicProblem,
    traversal: Sequence[NodeSite | EdgeSite] | None = None,
    domains: ClassicDomains | None = None,
) -> Iterator[ClassicSchedule]:
    """Lazily enumerate complete assignments in any site order.

    This traversal intentionally performs no semantic pruning.  Its set must therefore equal
    :func:`enumerate_reference` for every site order; the production walk may prune prefixes only
    while preserving the same complete set.
    """
    context = ClassicScheduleContext(problem, domains)
    canonical = (*context.index.nodes, *context.index.edges)
    order = tuple(canonical if traversal is None else traversal)
    if len(order) != len(canonical) or set(order) != set(canonical):
        raise ValueError("classic traversal must contain every node and edge site exactly once")

    def visit(position: int, nodes: dict, edges: dict) -> Iterator[ClassicSchedule]:
        if position == len(order):
            for kernel in kernel_domain(problem, domains):
                schedule = ClassicSchedule(kernel, nodes, edges)
                if context.accepts(schedule):
                    yield schedule
            return
        site = order[position]
        if isinstance(site, NodeSite):
            choices: Iterable = node_domain(problem, site, context.views[site], domains)
            for choice in choices:
                yield from visit(position + 1, {**nodes, site: choice}, edges)
            return
        for choice in edge_domain(problem, site, domains):
            yield from visit(position + 1, nodes, {**edges, site: choice})

    yield from visit(0, {}, {})


def _operand_position(node: Fold, wanted) -> int:
    for position, operand in enumerate(node.operands):
        if operand is wanted:
            return position
    raise ValueError("contraction role is not one of the node's operand edges")


def _stmt_nodes(stmt) -> Iterator[Fold]:
    for body in stmt.nested():
        for member in body:
            if isinstance(member, Fold):
                yield member
            else:
                yield from _stmt_nodes(member)


def _children(node: Fold) -> Iterator[Fold]:
    for operand in node.operands:
        if isinstance(operand, Fold):
            yield operand
    for member in node.lift.body:
        if isinstance(member, Fold):
            yield member
        else:
            yield from _stmt_nodes(member)
    stored = {id(value) for value in (*node.operands, *node.lift.body)}
    if node.axis is not None and not is_contraction(node):
        for member in node.step_stmts():
            if id(member) in stored:
                continue
            if isinstance(member, Fold):
                yield member
            else:
                yield from _stmt_nodes(member)


def _walk(root: Fold) -> Iterator[Fold]:
    yield root
    for child in _children(root):
        yield from _walk(child)


__all__ = [
    "Acceptance",
    "AxisAgreement",
    "CLASSIC_EDGE_FAMILIES",
    "CLASSIC_FAMILIES",
    "CLASSIC_NODE_FAMILIES",
    "ClassicProblem",
    "ClassicMaterialization",
    "ClassicDomains",
    "ClassicSchedule",
    "ClassicScheduleCodec",
    "ClassicScheduleContext",
    "Contraction",
    "EdgeSchedule",
    "EdgeSite",
    "KernelSchedule",
    "FragmentAgreement",
    "LocalSupport",
    "NodeId",
    "NodeSchedule",
    "NodeSite",
    "NodeView",
    "Projection",
    "ProjectionSchedule",
    "Reduction",
    "ReductionSchedule",
    "Refusal",
    "SiteIndex",
    "classify",
    "cartesian_assignments",
    "edge_domain",
    "edge_key",
    "enumerate_classic",
    "enumerate_reference",
    "kernel_domain",
    "node_domain",
    "node_key",
    "reduction_sites",
    "stage_edges",
    "tile_sites",
]
