"""Reusable views of the Fold nodes a schedule addresses."""

from __future__ import annotations

from dataclasses import dataclass

from frozendict import frozendict

from emmy.compiler.ir.pure.fold import Fold, cone_seam, deep_reads, edge_free_axes, is_contraction
from emmy.compiler.ir.pure.tree import walk
from emmy.compiler.ir.stmt import Accum, Body

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


def _sibling_fragment_edges(owner) -> dict[int, NodeId]:
    """Map each sibling-step consumer to the one contraction producing its computed edge."""
    out = {}
    for site in owner.sites:
        node = site.node
        if node.axis is None or is_contraction(node) or node.combine is None:
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
                out[id(consumer)] = owner.node_id(producers[0])
    return out


def contraction_facts(owner) -> frozendict[NodeId, ContractionFacts]:
    """Derive every contraction's :class:`ContractionFacts` from ``owner``'s term alone.

    ``owner`` is the kernel that indexes the sites — the :class:`~emmy.compiler.ir.tile.TileOp`,
    read through ``nodes`` / ``node_sites`` / ``views`` / ``node_at`` / ``node_id`` / ``parents`` /
    ``derived``, so this layer states the reading without importing the tile layer that owns it.
    """
    sibling = _sibling_fragment_edges(owner)
    facts = {}
    for site in range(len(owner.sites)):
        view = owner.views[site]
        if not isinstance(view, Reduction) or view.contraction is None:
            continue
        record = owner.sites[site]
        node, parent = record.node, record.parent
        if (
            record.derived
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
            nested = tuple(visit.node for visit in walk(node.a) if is_contraction(visit.node) and k_axis.name in edge_free_axes(visit.node))
            producer = nested[0] if len(nested) == 1 else None
        need = sibling.get(id(node))
        facts[site] = ContractionFacts(
            k_axis=k_axis,
            seam=seam,
            producer=producer,
            need=need if need is not None else (owner.node_id(producer) if producer is not None else None),
            need_step=need is not None,
        )
    return frozendict(facts)


def _operand_position(node: Fold, wanted) -> int:
    for position, operand in enumerate(node.operands):
        if operand is wanted:
            return position
    raise ValueError("contraction role is not one of the node's operand edges")


__all__ = [
    "Contraction",
    "ContractionFacts",
    "EdgeSite",
    "NodeId",
    "NodeView",
    "Projection",
    "Reduction",
    "contraction_facts",
    "node_view",
]
