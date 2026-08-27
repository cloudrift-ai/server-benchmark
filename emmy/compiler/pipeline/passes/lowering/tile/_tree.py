"""The ONE walk over a stored Fold tree.

Both tile passes read the same tree and differ only in what they take from it: ``030_cut`` keeps the
cut forks, ``040_schedule`` keeps the schedule forks. The walk itself is the same question — which
Folds are stored under this one, and what axes are in scope where each is read — so it is asked
once, here.

Two rules that are easy to get wrong separately, which is why they are written once:

- **A stored Fold is not always a direct member.** One can sit inside a plain statement's nested
  body (a composed step reached through a ``Loop``), so the walk alternates node-wise and
  statement-wise. A walker that only reads ``operands`` and ``lift.body`` silently loses those.
- **Axes in scope accumulate down the walk.** A node's own reduce axis is in scope for everything
  below it, and so is any axis a nesting statement binds. The cut pass needs this to ask whether an
  edge is semantically closed where it sits; nothing else may recompute it.
"""

from __future__ import annotations

from collections.abc import Iterator

from emmy.compiler.ir.pure.fold import Fold


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
    """``node``'s stored Fold children, each paired with the axes in scope at its incoming edge."""
    if not isinstance(node, Fold):
        return ()
    inner = axes if node.axis is None else (*axes, node.axis)
    return tuple(_members((*node.operands, *node.lift.body), inner))


def walk(node, axes: tuple = ()) -> Iterator[tuple[object, tuple]]:
    """``node`` and every stored Fold under it, preorder, each with its axes in scope.

    Preorder is not incidental: it is the order the schedule walk decides in and the order
    materialization re-resolves in, so the two cannot disagree about a Fold."""
    yield node, axes
    for child, inner in children(node, axes):
        yield from walk(child, inner)


__all__ = ["children", "walk"]
