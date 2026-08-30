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
from emmy.compiler.ir.schedule import PlacedTile, Raster, Reduce, ResolvedStage, Stage, Tile, Work


@dataclass(frozen=True, order=True)
class NodeId:
    """A problem-local node identity, assigned in stable preorder."""

    ordinal: int

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError(f"node ordinal must be non-negative, got {self.ordinal}")

    def spell(self) -> str:
        """Return the one serialized spelling of this identity."""
        return f"n{self.ordinal}"

    @classmethod
    def parse(cls, value: str) -> NodeId:
        """Parse :meth:`spell`, rejecting every other form."""
        if not value.startswith("n") or not value[1:].isdigit():
            raise ValueError(f"node id must be n<ordinal>, got {value!r}")
        return cls(int(value[1:]))


@dataclass(frozen=True, order=True)
class NodeSite:
    """The one schedule site of a Fold node."""

    id: NodeId


@dataclass(frozen=True, order=True)
class EdgeSite:
    """One operand use at a consumer node."""

    consumer: NodeSite
    operand: int


@dataclass(frozen=True)
class Projection:
    """The classification of a zero-axis Fold."""


@dataclass(frozen=True)
class Contraction:
    """A reduction's bilinear operand roles, expressed as consumer operand positions."""

    a: int
    channels: tuple[int, ...]


@dataclass(frozen=True)
class Reduction:
    """The classification of an iterating Fold."""

    contraction: Contraction | None = None


NodeView = Projection | Reduction


@dataclass(frozen=True)
class KernelSchedule:
    """Kernel-scoped choices."""

    work: Work
    raster: Raster


@dataclass(frozen=True)
class ProjectionSchedule:
    """The choices of a projection node."""

    tile: Tile


@dataclass(frozen=True)
class ReductionSchedule:
    """The choices of a reduction node, including a contraction-capable reduction."""

    tile: Tile
    reduce: Reduce


NodeSchedule = ProjectionSchedule | ReductionSchedule


@dataclass(frozen=True)
class EdgeSchedule:
    """The transport choice of one operand use."""

    stage: Stage


@dataclass(frozen=True)
class ClassicSchedule:
    """One complete classic schedule assignment."""

    kernel: KernelSchedule
    nodes: Mapping[NodeSite, NodeSchedule]
    edges: Mapping[EdgeSite, EdgeSchedule]

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", MappingProxyType(dict(self.nodes)))
        object.__setattr__(self, "edges", MappingProxyType(dict(self.edges)))

    def __reduce__(self):
        """Rebuild read-only mappings after process transport."""
        return type(self), (self.kernel, dict(self.nodes), dict(self.edges))


@dataclass(frozen=True)
class ClassicMaterialization:
    """Placed geometry and resolved transport facts derived from an accepted schedule."""

    tiles: Mapping[NodeSite, PlacedTile]
    stages: Mapping[EdgeSite, ResolvedStage]

    def __post_init__(self) -> None:
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
    """Derived facts and the sole membership relation for one classic problem."""

    def __init__(self, problem: ClassicProblem) -> None:
        self.problem = problem
        self.index = SiteIndex(problem.root)
        self.views = MappingProxyType({site: classify(self.index, site) for site in self.index.nodes})

    def accepts(self, schedule: ClassicSchedule) -> Acceptance:
        """Accept a complete, in-scope, view-compatible assignment or return its first refusal."""
        if not isinstance(schedule.kernel.work, Work) or not isinstance(schedule.kernel.raster, Raster):
            return Acceptance(Refusal("kernel assignment must contain explicit Work and Raster choices"))
        expected_nodes = set(self.index.nodes)
        actual_nodes = set(schedule.nodes)
        if missing := expected_nodes - actual_nodes:
            return Acceptance(Refusal("missing node assignment", min(missing)))
        if extra := actual_nodes - expected_nodes:
            return Acceptance(Refusal("node assignment is outside this problem", min(extra)))

        expected_edges = set(self.index.edges)
        actual_edges = set(schedule.edges)
        if missing := expected_edges - actual_edges:
            return Acceptance(Refusal("missing edge assignment", min(missing)))
        if extra := actual_edges - expected_edges:
            return Acceptance(Refusal("edge assignment is outside this problem", min(extra)))

        for site in self.index.nodes:
            view = self.views[site]
            assignment = schedule.nodes[site]
            if isinstance(view, Projection) and not isinstance(assignment, ProjectionSchedule):
                return Acceptance(Refusal("projection site requires a projection schedule", site))
            if isinstance(view, Reduction) and not isinstance(assignment, ReductionSchedule):
                return Acceptance(Refusal("reduction site requires a reduction schedule", site))
            if isinstance(assignment.tile, PlacedTile):
                return Acceptance(Refusal("node choices cannot contain placed tile geometry", site))
            if site not in tile_sites(self) and assignment.tile != Tile():
                return Acceptance(Refusal("this node has no tile choice", site))
        for edge in self.index.edges:
            stage = schedule.edges[edge].stage
            if not isinstance(stage, Stage):
                return Acceptance(Refusal("edge assignment must contain an explicit Stage choice", edge))
            if edge not in stage_edges(self) and not stage.is_direct:
                return Acceptance(Refusal("this edge has no staged transport choice", edge))
        return Acceptance()


class ClassicScheduleCodec:
    """Strict wire boundary for complete classic schedules.

    Kernel families are bare. Node and edge families use their bare spelling only when the family
    has exactly one site; otherwise the one canonical suffix is a :class:`NodeId`, with ``.e<N>``
    for an operand edge. Decoding accepts no aliases, missing direct values, or unknown fields.
    """

    def __init__(self, problem: ClassicProblem) -> None:
        self.context = ClassicScheduleContext(problem)
        self._tile_sites = tile_sites(self.context)
        self._reduce_sites = reduction_sites(self.context)
        self._stage_sites = stage_edges(self.context)

    def encode(self, schedule: ClassicSchedule) -> dict[str, str]:
        """Encode one accepted typed schedule in canonical scope order."""
        verdict = self.context.accepts(schedule)
        if not verdict:
            raise ValueError(_refusal_message(verdict.refusal))
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
        expected = self.keys()
        actual = set(row)
        if missing := expected - actual:
            raise ValueError(f"classic schedule row is missing {', '.join(sorted(missing))}")
        if extra := actual - expected:
            raise ValueError(f"classic schedule row has unknown keys {', '.join(sorted(extra))}")

        work = Work.parse(row["WORK"])
        nodes: dict[NodeSite, NodeSchedule] = {}
        for site in self.context.index.nodes:
            view = self.context.views[site]
            reduce = None
            if isinstance(view, Reduction):
                reduce = Reduce.parse(row[self._node_key("REDUCE", site, self._reduce_sites)], work)
            tile = Tile.parse(row[self._node_key("TILE", site, self._tile_sites)], work) if site in self._tile_sites else Tile()
            nodes[site] = ProjectionSchedule(tile) if reduce is None else ReductionSchedule(tile, reduce)
        schedule = ClassicSchedule(
            KernelSchedule(work, Raster.parse(row["RASTER"])),
            nodes,
            {
                edge: EdgeSchedule(Stage.parse(row[self._edge_key(edge)])) if edge in self._stage_sites else EdgeSchedule(Stage.direct())
                for edge in self.context.index.edges
            },
        )
        verdict = self.context.accepts(schedule)
        if not verdict:
            raise ValueError(_refusal_message(verdict.refusal))
        return schedule

    def keys(self) -> set[str]:
        """Return the exact key set accepted by :meth:`decode`."""
        return {
            "WORK",
            "RASTER",
            *(self._node_key("TILE", site, self._tile_sites) for site in self._tile_sites),
            *(self._node_key("REDUCE", site, self._reduce_sites) for site in self._reduce_sites),
            *(self._edge_key(edge) for edge in self._stage_sites),
        }

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
    return family if len(family_sites) == 1 else f"{family}@{site.id.spell()}"


def edge_key(edge: EdgeSite, family_edges: Sequence[EdgeSite]) -> str:
    """Return the sole canonical codec key for an edge family site."""
    if len(family_edges) == 1:
        return "STAGE"
    return f"STAGE@{edge.consumer.id.spell()}.e{edge.operand}"


def kernel_domain(problem: ClassicProblem) -> tuple[KernelSchedule, ...]:
    """Return the kernel choices available from static problem facts.

    The direct choice is the base domain. Hardware work inventories and grouped raster choices are
    added by their own recovery clusters; callers can rely on this function never reading a node or
    edge assignment.
    """
    del problem
    return (KernelSchedule(Work(), Raster()),)


def node_domain(problem: ClassicProblem, site: NodeSite, view: NodeView) -> tuple[NodeSchedule, ...]:
    """Return the site-local choices without inspecting any selected schedule."""
    index = SiteIndex(problem.root)
    if classify(index, site) != view:
        raise ValueError(f"view does not classify {site.id.spell()} in this problem")
    if isinstance(view, Projection):
        return (ProjectionSchedule(Tile()),)
    return (ReductionSchedule(Tile(), Reduce()),)


def edge_domain(problem: ClassicProblem, edge: EdgeSite) -> tuple[EdgeSchedule, ...]:
    """Return the edge-local transport choices without inspecting another choice."""
    index = SiteIndex(problem.root)
    index.operand(edge)  # scope check
    return (EdgeSchedule(Stage.direct()),)


def cartesian_assignments(problem: ClassicProblem) -> Iterator[tuple[ClassicSchedule, Acceptance]]:
    """Enumerate the literal domain product and the semantic verdict for each complete assignment."""
    context = ClassicScheduleContext(problem)
    node_domains = tuple(node_domain(problem, site, context.views[site]) for site in context.index.nodes)
    edge_domains = tuple(edge_domain(problem, edge) for edge in context.index.edges)
    for kernel, node_values, edge_values in product(
        kernel_domain(problem),
        product(*node_domains),
        product(*edge_domains),
    ):
        schedule = ClassicSchedule(
            kernel,
            dict(zip(context.index.nodes, node_values, strict=True)),
            dict(zip(context.index.edges, edge_values, strict=True)),
        )
        yield schedule, context.accepts(schedule)


def enumerate_reference(problem: ClassicProblem) -> Iterator[ClassicSchedule]:
    """Yield the accepted subset of the literal Cartesian product."""
    for schedule, verdict in cartesian_assignments(problem):
        if verdict:
            yield schedule


def enumerate_classic(
    problem: ClassicProblem,
    traversal: Sequence[NodeSite | EdgeSite] | None = None,
) -> Iterator[ClassicSchedule]:
    """Lazily enumerate complete assignments in any site order, with acceptance at every leaf.

    This deliberately carries no public partial context. Later pruning may reject prefixes through
    a private propagator, but the complete leaf remains subject to :meth:`ClassicScheduleContext.accepts`.
    """
    context = ClassicScheduleContext(problem)
    canonical = (*context.index.nodes, *context.index.edges)
    order = tuple(canonical if traversal is None else traversal)
    if len(order) != len(canonical) or set(order) != set(canonical):
        raise ValueError("classic traversal must contain every node and edge site exactly once")

    def visit(position: int, nodes: dict, edges: dict) -> Iterator[ClassicSchedule]:
        if position == len(order):
            for kernel in kernel_domain(problem):
                schedule = ClassicSchedule(kernel, nodes, edges)
                if context.accepts(schedule):
                    yield schedule
            return
        site = order[position]
        if isinstance(site, NodeSite):
            domain: Iterable = node_domain(problem, site, context.views[site])
            for choice in domain:
                yield from visit(position + 1, {**nodes, site: choice}, edges)
            return
        for choice in edge_domain(problem, site):
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
    "ClassicProblem",
    "ClassicMaterialization",
    "ClassicSchedule",
    "ClassicScheduleCodec",
    "ClassicScheduleContext",
    "Contraction",
    "EdgeSchedule",
    "EdgeSite",
    "KernelSchedule",
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
    "cartesian_assignments",
    "classify",
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
