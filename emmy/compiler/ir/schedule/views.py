"""Reusable views of the Fold nodes a schedule addresses."""

from __future__ import annotations

from dataclasses import dataclass

from frozendict import frozendict

from emmy.compiler.ir.pure.fold import Fold, _operand_result_names, deep_reads, is_contraction, operand_body, splice_operands
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
            nested = tuple(
                visit.node for visit in walk(node.a) if is_contraction(visit.node) and k_axis.name in edge_axes(visit.node, (k_axis.name,))
            )
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


def edge_axes(edge, axes) -> frozenset[str]:
    """Which of ``axes`` this operand edge references — answered from the DECLARATION.

    The caller already holds the axis set, because the binder handed it down. It therefore has no
    business asking the term to re-derive it by walking a lowered body for names that look
    axis-shaped: a term declares the enclosing coordinates it reads as lift params
    (``Lambda.closing``'s scope), so the answer is an intersection.

    Two edge kinds, one rule. A nested term answers from its params, less the axis it BINDS —
    an edge reducing over its own ``k`` shadows an enclosing one of the same name and does not
    vary with it. A ``Load`` is a leaf with no params, so it answers from its own index exprs;
    that is reading the edge's own data, not interrogating it about its surroundings.
    """
    wanted = frozenset(axes)
    if isinstance(edge, Fold):
        return wanted & (set(edge.lift.params) - edge.binds_axes())
    return wanted & {name for expr in edge.exprs() for name in expr.free_vars()}


# ``cone_seam`` lives HERE, not on the term. "Is this edge row-invariant or does it vary with the
# contraction axis?" is a question about a term's relationship to an enclosing iteration space,
# which the term does not know — it knows the axis it reduces over and nothing more. It sat in
# ``ir/pure/fold`` only because the schedule layer needed it and ``ir/tile/ops`` could not be
# imported from here; the schedule layer is where it belonged all along.
def cone_seam(cone, k_name: str) -> tuple[tuple, tuple, tuple[str, ...]]:
    """The computed-A cone's ``(prologue, cell, stats)`` — read off the NODE BOUNDARY, not by
    scanning stmts: the cone is ``Fold.projection(body=<the per-cell normalize>, operands=(<the row-invariant
    prologue>, <any per-cell producer>…))``, and the prologue node IS the per-row statistic (its
    own zero-axis ``Fold`` over the stat ``Fold``) plus any row-invariant cone prefix, placed there
    when the cone was built (:func:`make_cone` splits at the K seam once, structurally).

    The split is the K SEAM, on the edges as on the stmts: an edge that never indexes the
    contraction axis ``k_name`` is row-invariant and belongs to the prologue; a k-VARYING producer
    edge (the attention score contraction the cone's ``exp(s − m)`` reads) is per-cell and splices
    into the cell ahead of its first use, like any operand edge. Every fused norm→linear cone
    carries the single row-invariant edge, so its seam reads exactly as it always did.

    ``stats`` are the prologue results the cell reads — the values bridged through the stat smem
    rows. Internal definitions are excluded: the prologue and cell may independently use the same
    local SSA name. A prologue whose results go unread is dropped (nothing to bridge). The ONE seam
    both sides read: the scheduler sizes the stat rows into the sync stage's smem budget, the
    materializer fills them (``sync_stat_fill``)."""
    if not isinstance(cone, Fold) or cone.axis is not None or not cone.operands:
        return (), tuple(cone.body) if isinstance(cone, Fold) and cone.axis is None else (), ()
    varying = [k_name in edge_axes(e, (k_name,)) for e in cone.operands]
    pro = tuple(s for e, k in zip(cone.operands, varying, strict=True) if not k for s in operand_body(e))
    cell = splice_operands(tuple(e for e, k in zip(cone.operands, varying, strict=True) if k), tuple(cone.body))
    pro_results = {nm for edge, varies in zip(cone.operands, varying, strict=True) if not varies for nm in _operand_result_names(edge)}
    stats = tuple(sorted(pro_results & deep_reads(list(cell))))
    return (pro, cell, stats) if stats else ((), cell, ())
