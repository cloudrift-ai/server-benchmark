"""The semantic model for the classic grid/CTA/warp/thread/register schedule.

The model contains choices only.  A :class:`ClassicProblem` supplies the immutable Fold tree and
target; :class:`ClassicScheduleContext` derives identities and classification from that problem and
is the only authority that accepts or refuses a complete assignment.  Codecs, enumeration order,
search state, and materialization data do not belong here.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from emmy.compiler.ir.pure.fold import Fold, is_contraction
from emmy.compiler.ir.schedule import Raster, Reduce, Stage, Tile, Work


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

    work: Work | None
    raster: Raster | None


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

    stage: Stage | None


@dataclass(frozen=True)
class ClassicSchedule:
    """One complete classic schedule assignment."""

    kernel: KernelSchedule
    nodes: Mapping[NodeSite, NodeSchedule]
    edges: Mapping[EdgeSite, EdgeSchedule]

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", MappingProxyType(dict(self.nodes)))
        object.__setattr__(self, "edges", MappingProxyType(dict(self.edges)))


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
        return Acceptance()


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
    "ClassicSchedule",
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
    "classify",
]
