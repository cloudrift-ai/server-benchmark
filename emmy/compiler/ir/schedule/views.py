"""Reusable views of the Fold nodes a schedule addresses."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

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
