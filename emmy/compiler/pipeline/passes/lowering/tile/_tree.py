"""The ONE walk over a stored Fold tree.

``030_cut`` uses this walk to find structural seams. Classic scheduling may use the same walk to
discover sites, but its candidate set is the compatible subset of independent node and edge
domains: preorder may affect evaluation cost, never membership. The walk answers which Folds are
stored under this one and what axes are in scope where each is read.

Three rules that are easy to get wrong separately, which is why they are written once:

- **A stored Fold is not always a direct member.** One can sit inside a plain statement's nested
  body (a composed step reached through a ``Loop``), so the walk alternates node-wise and
  statement-wise. A walker that only reads ``operands`` and ``lift.body`` silently loses those.
- **A DERIVED node is a real classic site.** A λ-spelled fold's derived evaluation
  (``Fold.step_stmts``) can synthesize a node no stored member carries — flash's PV contraction,
  memoized on the fold so it has one identity — and the schedule must reach it. "Derived" means a node the
  derived step yields that is neither an operand edge nor a literal lift-body member. The cut pass
  is unaffected — it filters through ``path.family_sites("PLACE", …)``, which excludes derived
  sites, so a derived node contributes scopes but never a cuttable edge.
- **Axes in scope accumulate down the walk.** A node's own reduce axis is in scope for everything
  below it, and so is any axis a nesting statement binds. The cut pass needs this to ask whether an
  edge is semantically closed where it sits; nothing else may recompute it.
"""

from __future__ import annotations

from collections.abc import Iterator

from emmy.compiler.ir.pure.fold import Fold, is_contraction


def _members(members, axes: tuple) -> Iterator[tuple[Fold, tuple]]:
    """Stored Fold occurrences among ``members``, descending a plain statement's nested bodies and
    picking up the axes each nesting binds."""
    for member in members:
        if isinstance(member, Fold):
            yield member, axes
            continue
        axis = getattr(member, "axis", None)
        inner = (*axes, axis) if axis is not None else axes
        for body in member.nested():
            yield from _members(body, inner)


def children(node, axes: tuple = ()) -> tuple[tuple[Fold, tuple], ...]:
    """``node``'s stored Fold children — plus its DERIVED ones — each paired with the axes in scope
    at its incoming edge. Stored members lead, derived members follow, matching the tree-path
    codec's own visit order so the two walks key one node identically."""
    if not isinstance(node, Fold):
        return ()
    inner = axes if node.axis is None else (*axes, node.axis)
    members = (*node.operands, *node.lift.body)
    # A contraction's children are exactly its operand edges. Its derived step repeats those
    # nodes, so reading it here would create duplicate classic sites.
    if node.axis is not None and not is_contraction(node):
        stored = {id(m) for m in members}
        members = (*members, *(s for s in node.step_stmts() if id(s) not in stored))
    return tuple(_members(members, inner))


def walk(node, axes: tuple = ()) -> Iterator[tuple[object, tuple]]:
    """``node`` and every stored Fold under it, preorder, each with its axes in scope.

    Preorder is a stable discovery order, not schedule semantics. Any scheduling consumer must
    produce the same compatible assignment set under another traversal."""
    yield node, axes
    for child, inner in children(node, axes):
        yield from walk(child, inner)


__all__ = ["children", "walk"]
