"""Reusable views of the Fold nodes a schedule addresses."""

from __future__ import annotations

from dataclasses import dataclass

from frozendict import frozendict

from emmy.compiler.ir.pure.fold import ContractionView, Fold
from emmy.compiler.ir.pure.tree import walk
from emmy.compiler.ir.stmt import Body

type NodeId = int
type EdgeSite = tuple[NodeId, int]


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


def contraction_facts(owner) -> frozendict[NodeId, ContractionFacts]:
    """Derive every contraction's :class:`ContractionFacts` from ``owner``'s term alone.

    ``owner`` is the kernel that indexes the sites — the :class:`~emmy.compiler.ir.tile.TileOp`,
    read through ``nodes`` / ``node_sites`` / ``views`` / ``node_at`` / ``node_id`` / ``parents`` /
    ``derived``, so this layer states the reading without importing the tile layer that owns it.
    """
    facts = {}
    for site in range(len(owner.sites)):
        view = owner.views[site]
        if view.as_contraction() is None:
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
            computed = tuple(edge for edge in node.operands if edge.as_slab() is None)
            seam = cone_seam(computed[0], node.axis.name) if computed else None
            k_axis = node.axis
        # The nested contraction this one consumes — sought over the operand edges, which is where
        # a term's children are. A role was a position into that same tuple, so naming one bought
        # nothing the walk does not already have.
        nested = tuple(
            visit.node
            for edge in node.operands
            if edge.as_slab() is None
            for visit in walk(edge)
            if visit.node.as_contraction() is not None and k_axis.name in visit.node.index_space
        )
        producer = nested[0] if len(nested) == 1 else None
        facts[site] = ContractionFacts(
            k_axis=k_axis,
            seam=seam,
            producer=producer,
            need=owner.node_id(producer) if producer is not None else None,
        )
    return frozendict(facts)


def _operand_position(node: Fold, wanted) -> int:
    for position, operand in enumerate(node.operands):
        if operand is wanted:
            return position
    raise ValueError("contraction role is not one of the node's operand edges")


__all__ = [
    "ContractionView",
    "ContractionFacts",
    "EdgeSite",
    "NodeId",
    "contraction_facts",
]


# ``cone_seam`` lives HERE, not on the term. "Is this edge row-invariant or does it vary with the
# contraction axis?" is a question about a term's relationship to an enclosing iteration space,
# which the term does not know — it knows the axis it reduces over and nothing more. It sat in
# ``ir/pure/fold`` only because the schedule layer needed it and ``ir/tile/ops`` could not be
# imported from here; the schedule layer is where it belonged all along.
def cone_seam(cone, k_name: str) -> tuple[tuple, tuple, tuple[str, ...]]:
    """The computed-A cone's ``(prologue, cell, stats)`` — read off the NODE BOUNDARY, not by
    scanning stmts: the cone is a zero-axis term over ``<the per-cell normalize>`` with ``<the row-invariant
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
        return (), tuple(cone.lift.body) if isinstance(cone, Fold) and cone.axis is None else (), ()
    # Split by DECLARATION: an edge whose index space holds the reduction axis varies with it and
    # rides the cell; the rest are row-invariant and lower once into the prologue. Same reading as
    # ``Fold.lower``'s hoist, asked of the same property.
    varying = [k_name in edge.index_space for edge in cone.operands]
    pro = tuple(s for e, k in zip(cone.operands, varying, strict=True) if not k for s in e.lower())
    cell = tuple(stmt for edge, varies in zip(cone.operands, varying, strict=True) if varies for stmt in edge.lower()) + tuple(
        cone.lift.body
    )
    pro_results = {nm for edge, varies in zip(cone.operands, varying, strict=True) if not varies for nm in edge.exposes}
    stats = tuple(sorted(pro_results & Body(cell).ssa_uses))
    return (pro, cell, stats) if stats else ((), cell, ())
