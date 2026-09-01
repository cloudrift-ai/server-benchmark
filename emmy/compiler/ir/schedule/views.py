"""Reusable views of the Fold nodes a schedule addresses."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from frozendict import frozendict

from emmy.compiler.ir.pure.fold import Fold, is_contraction
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
    "EdgeSite",
    "NodeId",
    "NodeView",
    "Projection",
    "Reduction",
    "node_view",
    "schedule_edges",
    "schedule_nodes",
]
