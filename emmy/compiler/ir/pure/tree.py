"""The ONE walk over a stored Fold tree.

``030_cut`` uses this walk to find structural seams, classic scheduling numbers its sites from it,
and the tree-path codec (``ir/tile/path.py``) spells its segments off it. One walk, so the three
cannot drift.

Four rules that are easy to get wrong separately, which is why they are written once:

- **A stored Fold is not always a direct member.** One can sit inside a plain statement's nested
  body (a composed step reached through a ``Loop``), so the walk alternates node-wise and
  statement-wise. A walker that only reads ``operands`` and ``lift.body`` silently loses those.
- **A DERIVED node is a real site.** A λ-spelled fold's derived evaluation (``Fold.step_stmts``)
  can synthesize a node no stored member carries — flash's PV contraction, memoized on the fold so
  it has one identity — and the schedule must reach it. "Derived" means a node the derived step
  yields that is neither an operand edge nor a literal lift-body member, and it propagates to
  everything below. The cut pass is unaffected — it filters through
  ``path.family_sites("PLACE", …)``, which excludes derived sites, so a derived node contributes
  scopes but never a cuttable edge.
- **A contraction's edges are visited by ROLE, not stored order.** ``Fold.contraction`` stores the
  channels before the shared A edge, while the segment vocabulary spells the roles (``a`` / ``b``).
  Visiting by role is what lets the schedule's integer ids and the codec's segments come off ONE
  walk; in stored order a contraction with two computed edges numbers the other way.
- **Axes in scope accumulate down the walk.** A node's own reduce axis is in scope for everything
  below it, and so is any axis a nesting statement binds. The cut pass needs this to ask whether an
  edge is semantically closed where it sits; nothing else may recompute it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

from emmy.compiler.ir.pure.fold import Fold, is_contraction


class Visit(NamedTuple):
    """One node the walk reaches: the term, the node that reached it, the axes in scope where it is
    read, the segment path addressing it from the root, and whether it lives in a derived
    evaluation. ``parent`` is ``None`` at the root."""

    node: object
    parent: object
    axes: tuple
    segments: tuple[str, ...]
    derived: bool


def segment(node) -> str:
    """The path segment naming one node's kind.

    The ZERO-AXIS reading spells ``map`` and every iterating fold — contraction reading included —
    spells ``fold``, exactly as the three stored kinds did before the collapse, so every stored
    golden / DB key keeps meaning what it always meant.
    """
    return "map" if getattr(node, "axis", None) is None else "fold"


def _descend(member, axes: tuple, label: str | None, derived: bool) -> Iterator[tuple]:
    """The Folds ``member`` carries, each with the segment that addresses it — descending a plain
    statement's nested bodies and picking up the axes each nesting binds."""
    if isinstance(member, Fold):
        yield member, axes, label or segment(member), derived
        return
    axis = getattr(member, "axis", None)
    inner = (*axes, axis) if axis is not None else axes
    for body in member.nested():
        for child in body:
            yield from _descend(child, inner, None, derived)


def children(node, axes: tuple = (), derived: bool = False) -> tuple[tuple, ...]:
    """``node``'s stored Fold children — plus its DERIVED ones — each as
    ``(child, axes in scope, segment, derived)``. Stored members lead, derived members follow."""
    if not isinstance(node, Fold):
        return ()
    inner = axes if node.axis is None else (*axes, node.axis)
    if is_contraction(node):
        # A contraction's children are its ROLE-ordered operand edges. Its derived step repeats
        # those nodes, so reading it here would create duplicate sites.
        labelled = (
            (node.a, "a", derived),
            *((channel.b, "b", derived) for channel in node.channels),
            *((member, None, derived) for member in node.lift.body),
        )
    else:
        members = (*node.operands, *node.lift.body)
        labelled = tuple((member, None, derived) for member in members)
        if node.axis is not None:
            stored = {id(member) for member in members}
            labelled += tuple((s, None, True) for s in node.step_stmts() if id(s) not in stored)
    return tuple(visit for member, label, member_derived in labelled for visit in _descend(member, inner, label, member_derived))


def walk(
    node,
    axes: tuple = (),
    segments: tuple[str, ...] | None = None,
    derived: bool = False,
    parent: object = None,
) -> Iterator[Visit]:
    """``node`` and every stored Fold under it, preorder, each as a :class:`Visit`.

    Preorder is a stable discovery order, not schedule semantics. Any scheduling consumer must
    produce the same compatible assignment set under another traversal."""
    segments = (segment(node),) if segments is None else segments
    yield Visit(node, parent, axes, segments, derived)
    for child, inner, label, child_derived in children(node, axes, derived):
        yield from walk(child, inner, (*segments, label), child_derived, node)


__all__ = ["Visit", "children", "segment", "walk"]
