"""Reusable views of the Fold nodes a schedule addresses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from frozendict import frozendict

from emmy.compiler.ir.fold_tree import walk
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


@dataclass(frozen=True)
class ScheduleInventory:
    """Stable target-independent node, edge, and classification indexes for one Fold tree."""

    nodes: tuple[Fold, ...]
    edges: tuple[EdgeSite, ...]
    site_by_identity: Mapping[int, NodeId]
    views: Mapping[NodeId, NodeView]
    incident: Mapping[NodeId, tuple[EdgeSite, ...]]

    @classmethod
    def from_root(
        cls,
        root: Fold,
        *,
        nodes: tuple[Fold, ...] | None = None,
        edges: tuple[EdgeSite, ...] | None = None,
    ) -> Self:
        nodes = schedule_nodes(root) if nodes is None else nodes
        edges = schedule_edges(nodes) if edges is None else edges
        sites = tuple(range(len(nodes)))
        return cls(
            nodes,
            edges,
            frozendict({id(node): site for site, node in enumerate(nodes)}),
            frozendict({site: node_view(nodes[site]) for site in sites}),
            frozendict({site: tuple(edge for edge in edges if edge[0] == site) for site in sites}),
        )

    def __getstate__(self):
        """Serialize structural indexes; object-identity keys are rebuilt on load."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "views": self.views,
            "incident": self.incident,
        }

    def __setstate__(self, state) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "site_by_identity", frozendict({id(node): site for site, node in enumerate(self.nodes)}))

    @property
    def node_sites(self) -> tuple[NodeId, ...]:
        return tuple(range(len(self.nodes)))

    @property
    def tile_sites(self) -> tuple[NodeId, ...]:
        return tuple(
            site
            for site, view in self.views.items()
            if (isinstance(view, Reduction) and view.contraction is not None)
            or (isinstance(view, Projection) and site == 0 and not self.nodes[site].operands)
        )

    @property
    def reduction_sites(self) -> tuple[NodeId, ...]:
        return tuple(site for site, view in self.views.items() if isinstance(view, Reduction))

    @property
    def stage_edges(self) -> tuple[EdgeSite, ...]:
        return tuple(
            edge for edge in self.edges if isinstance(self.views[edge[0]], Reduction) and self.views[edge[0]].contraction is not None
        )

    def node(self, site: NodeId) -> Fold:
        if type(site) is not int or not 0 <= site < len(self.nodes):
            raise KeyError(f"unknown node site {site!r}")
        return self.nodes[site]

    def site(self, node: Fold) -> NodeId:
        try:
            return self.site_by_identity[id(node)]
        except KeyError:
            raise KeyError("Fold is not a node of this schedule inventory") from None

    def operand(self, edge: EdgeSite):
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise KeyError(f"invalid edge site {edge!r}")
        consumer, operand = edge
        try:
            return self.node(consumer).operands[operand]
        except IndexError:
            raise KeyError(f"unknown operand {operand} at node {consumer}") from None

    def producer(self, edge: EdgeSite) -> NodeId | None:
        value = self.operand(edge)
        return self.site(value) if isinstance(value, Fold) else None

    def incident_edges(self, site: NodeId) -> tuple[EdgeSite, ...]:
        return self.incident[site]


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
    for node, _axes in walk(root):
        if id(node) not in seen:
            seen.add(id(node))
            nodes.append(node)
    memo["nodes"] = tuple(nodes)
    return memo["nodes"]


def schedule_edges(nodes: tuple[Fold, ...]) -> tuple[EdgeSite, ...]:
    """Return every consumer operand position in stable node order."""
    return tuple((consumer, operand) for consumer, node in enumerate(nodes) for operand in range(len(node.operands)))


def _operand_position(node: Fold, wanted) -> int:
    for position, operand in enumerate(node.operands):
        if operand is wanted:
            return position
    raise ValueError("contraction role is not one of the node's operand edges")


__all__ = [
    "Contraction",
    "EdgeSite",
    "NodeId",
    "NodeView",
    "Projection",
    "Reduction",
    "ScheduleInventory",
    "node_view",
    "schedule_edges",
    "schedule_nodes",
]
