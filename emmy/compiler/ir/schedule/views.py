"""Reusable views of the Fold nodes a schedule addresses."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from frozendict import frozendict

from emmy.compiler.ir.pure.fold import Fold, cone_seam, deep_reads, edge_refs_axis, is_contraction
from emmy.compiler.ir.pure.tree import children, walk
from emmy.compiler.ir.stmt import Accum, Body
from emmy.compiler.structural import instance_memo

type NodeId = int
type EdgeSite = tuple[NodeId, int]


@dataclass(frozen=True)
class Projection:
    """A zero-axis Fold."""


@dataclass(frozen=True)
class Contraction:
    """A reduction's bilinear operand roles, expressed as operand positions."""

    a: int
    channels: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.a) is not int or self.a < 0:
            raise ValueError(f"contraction A role must be a non-negative operand position, got {self.a!r}")
        if any(type(position) is not int or position < 0 for position in self.channels):
            raise ValueError("contraction channel roles must be non-negative operand positions")


@dataclass(frozen=True)
class Reduction:
    """An iterating Fold, optionally viewed as a contraction."""

    contraction: Contraction | None = None

    def __post_init__(self) -> None:
        if self.contraction is not None and not isinstance(self.contraction, Contraction):
            raise TypeError("reduction contraction capability must be a Contraction or None")


@dataclass(frozen=True)
class ContractionFacts:
    """One contraction's schedule-independent structure.

    Every field is read off the Fold root alone: the effective ``k_axis`` (a derived singleton
    marker borrows its enclosing sweep's), the computed-A cone's ``seam``, the single nested
    ``producer`` its A edge contracts, and the ``need`` site whose fragment this one consumes
    (``need_step`` when that need is a sibling step rather than a nested producer).
    """

    k_axis: object
    seam: tuple | None = None
    producer: Fold | None = None
    need: NodeId | None = None
    need_step: bool = False


type NodeView = Projection | Reduction


def node_view(node: Fold) -> NodeView:
    """Classify one Fold without target or schedule input."""
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


def schedule_nodes(root: Fold) -> tuple[Fold, ...]:
    """Return Fold nodes in stable preorder, keeping one entry per object identity."""
    memo = instance_memo(root, "_memo_schedule_views")
    if "nodes" in memo:
        return memo["nodes"]
    nodes = []
    seen = set()
    for node in _walk(root):
        if id(node) not in seen:
            seen.add(id(node))
            nodes.append(node)
    memo["nodes"] = tuple(nodes)
    return memo["nodes"]


def schedule_edges(nodes: tuple[Fold, ...]) -> tuple[EdgeSite, ...]:
    """Return every consumer operand position in stable node order."""
    return tuple((consumer, operand) for consumer, node in enumerate(nodes) for operand in range(len(node.operands)))


@dataclass(frozen=True)
class ClassicSites:
    """The node and edge sites, and their schedule views, derived from one Fold root."""

    root: Fold

    def __post_init__(self) -> None:
        if not isinstance(self.root, Fold):
            raise TypeError("classic sites require a Fold root")

    def _memo(self) -> dict:
        return instance_memo(self.root, "_memo_classic_sites")

    @property
    def nodes(self) -> tuple[Fold, ...]:
        return schedule_nodes(self.root)

    @property
    def node_sites(self) -> tuple[NodeId, ...]:
        return tuple(range(len(self.nodes)))

    @property
    def edge_sites(self) -> tuple[EdgeSite, ...]:
        return schedule_edges(self.nodes)

    @property
    def views(self) -> frozendict[NodeId, NodeView]:
        memo = self._memo()
        if "views" not in memo:
            memo["views"] = frozendict({site: node_view(self.nodes[site]) for site in self.node_sites})
        return memo["views"]

    @property
    def tile_sites(self) -> tuple[NodeId, ...]:
        return tuple(
            site
            for site in self.node_sites
            if (isinstance(self.views[site], Reduction) and self.views[site].contraction is not None)
            or (isinstance(self.views[site], Projection) and site == self.node_sites[0] and not self.nodes[site].operands)
        )

    @property
    def reduction_sites(self) -> tuple[NodeId, ...]:
        return tuple(site for site in self.node_sites if isinstance(self.views[site], Reduction))

    @property
    def stage_edges(self) -> tuple[EdgeSite, ...]:
        return tuple(
            edge for edge in self.edge_sites if isinstance(self.views[edge[0]], Reduction) and self.views[edge[0]].contraction is not None
        )

    def node(self, site: NodeId) -> Fold:
        if type(site) is not int or not 0 <= site < len(self.nodes):
            raise KeyError(f"unknown node site {site!r}")
        return self.nodes[site]

    def site(self, node: Fold) -> NodeId:
        memo = self._memo()
        if "site_by_identity" not in memo:
            memo["site_by_identity"] = {id(value): site for site, value in enumerate(self.nodes)}
        try:
            return memo["site_by_identity"][id(node)]
        except KeyError:
            raise KeyError("Fold is not a node of these classic sites") from None

    def operand(self, edge: EdgeSite):
        consumer, operand = edge
        try:
            return self.node(consumer).operands[operand]
        except (TypeError, IndexError):
            raise KeyError(f"unknown classic edge site {edge!r}") from None

    def producer(self, edge: EdgeSite) -> NodeId | None:
        value = self.operand(edge)
        return self.site(value) if isinstance(value, Fold) else None

    def incident_edges(self, site: NodeId) -> tuple[EdgeSite, ...]:
        self.node(site)
        return tuple(edge for edge in self.edge_sites if edge[0] == site)

    @property
    def contractions(self) -> frozendict[NodeId, ContractionFacts]:
        """The per-contraction structure every schedule choice over this root shares."""
        memo = self._memo()
        if "contractions" not in memo:
            memo["contractions"] = frozendict(_contraction_facts(self))
        return memo["contractions"]


def _sibling_fragment_edges(site_index: ClassicSites) -> dict[int, NodeId]:
    """Map each sibling-step consumer to the one contraction producing its computed edge."""
    out = {}
    for node, _axes in walk(site_index.root):
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
                out[id(consumer)] = site_index.site(producers[0])
    return out


def _contraction_facts(site_index: ClassicSites) -> dict[NodeId, ContractionFacts]:
    """Derive every contraction's :class:`ContractionFacts` from the root alone."""
    from emmy.compiler.ir.tile.path import sites  # noqa: PLC0415 — the tile layer reads these views

    root = site_index.root
    parents: dict[int, Fold] = {}
    for node, _axes in walk(root):
        for child, _child_axes in children(node):
            parents.setdefault(id(child), node)
    derived = {id(site.node) for site in sites(root) if site.derived}
    sibling = _sibling_fragment_edges(site_index)
    facts = {}
    for node, _axes in walk(root):
        if not (isinstance(node, Fold) and node.axis is not None and is_contraction(node)):
            continue
        site = site_index.site(node)
        if site in facts:
            continue
        parent = parents.get(id(node))
        if (
            id(node) in derived
            and node.axis.extent.is_static
            and node.axis.extent.as_static() == 1
            and isinstance(parent, Fold)
            and parent.axis is not None
        ):
            # a derived singleton marker: the enclosing Fold owns the K sweep, and the seam it
            # bridges is that Fold's own leading state rather than a cone read off this node
            assert parent.combine is not None and node.combine is not None
            seam = ((), (), tuple(parent.combine.results[: -len(node.combine.results)]))
            k_axis = parent.axis
        else:
            seam = cone_seam(node.a, node.axis.name) if isinstance(node.a, Fold) else None
            k_axis = node.axis
        producer = None
        if isinstance(node.a, Fold):
            nested = tuple(site.node for site in sites(node.a) if is_contraction(site.node) and edge_refs_axis(site.node, k_axis.name))
            producer = nested[0] if len(nested) == 1 else None
        need = sibling.get(id(node))
        facts[site] = ContractionFacts(
            k_axis=k_axis,
            seam=seam,
            producer=producer,
            need=need if need is not None else (site_index.site(producer) if producer is not None else None),
            need_step=need is not None,
        )
    return facts


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
    yield from (operand for operand in node.operands if isinstance(operand, Fold))
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
    "ClassicSites",
    "Contraction",
    "ContractionFacts",
    "EdgeSite",
    "NodeId",
    "NodeView",
    "Projection",
    "Reduction",
    "node_view",
    "schedule_edges",
    "schedule_nodes",
]
