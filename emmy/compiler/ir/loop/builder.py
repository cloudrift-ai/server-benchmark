"""LoopBuilder — incremental body construction for loop IR.

Accumulates a ``LoopOp`` body by inserting stmts one at a time at a given
enclosing ``Scope``. Callers drive two concerns:

- **Fresh SSA names**: ``fresh(hint)`` returns an unused name derived from
  ``hint``, reserving it in the builder's name pool.
- **Scope-aware insertion**: ``insert(stmt, scope)`` descends the body
  tree along the enclosing axis path, creates ``Loop`` nodes as needed,
  and prepends the stmt at the leaf.

Insertions are prepend-at-leaf. Callers that want defined-before-use
ordering should insert in reverse-topological order (consumers first,
producers after) — that's what the fusion splicer does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from emmy.compiler.ir.loop.ir import Axis, Loop, Scope, Stmt
from emmy.compiler.ir.stmt import Body


@dataclass
class _MutableLoop:
    """Construction-only loop node; frozen into a ``Loop`` by ``finish``."""

    axis: Axis
    body: list[Stmt | _MutableLoop] = field(default_factory=list)


class LoopBuilder:
    """Mutable accumulator for a ``LoopOp`` body."""

    def __init__(self, used_names: set[str]) -> None:
        self._body: list[Stmt | _MutableLoop] = []
        self._used: set[str] = set(used_names)
        self._next_suffix: dict[str, int] = {}

    def fresh(self, hint: str) -> str:
        """Return an unused name derived from ``hint`` and reserve it."""
        if hint not in self._used:
            self._used.add(hint)
            return hint
        i = self._next_suffix.get(hint, 1)
        while f"{hint}_s{i}" in self._used:
            i += 1
        name = f"{hint}_s{i}"
        self._used.add(name)
        self._next_suffix[hint] = i + 1
        return name

    def insert(self, stmt: Stmt, enclosure: Scope) -> None:
        """Prepend ``stmt`` at the leaf of the path denoted by ``enclosure``."""
        body = self._body
        for axis in enclosure.enclosing:
            loop = next((item for item in body if isinstance(item, _MutableLoop) and item.axis == axis), None)
            if loop is None:
                loop = _MutableLoop(axis)
                body.append(loop)
            body = loop.body
        # Construction resolves consumers before producers. Appending is O(1);
        # ``finish`` reverses each scope to retain prepend-at-leaf order.
        body.append(stmt)

    def finish(self) -> Body:
        """Return the accumulated body."""
        return _freeze(self._body)


def _freeze(body: list[Stmt | _MutableLoop]) -> Body:
    """Freeze the append-built tree in its logical prepend order."""
    return Body(Loop(axis=item.axis, body=_freeze(item.body)) if isinstance(item, _MutableLoop) else item for item in reversed(body))
