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

Construction is MUTABLE and the immutable body is built once, by :meth:`LoopBuilder.finish`.
Descending a scope path is a dict lookup per level, and prepending is an append to a
reverse-ordered list. The straightforward form — rebuild the ``Body`` tuple at every level on
every insert, finding each level's ``Loop`` by scanning for a matching axis — is quadratic in the
program size, and it re-runs ``Loop.__post_init__`` (which normalizes the whole body) once per
level per insert. That is invisible on a small graph and decisive on a large one: DeepSeek-V4's
644-node post twin drove 404k inserts, and its compile did not finish in 25 minutes.
"""

from __future__ import annotations

from emmy.compiler.ir.loop.ir import Axis, Loop, Scope, Stmt
from emmy.compiler.ir.stmt import Body


class _Scope:
    """One level of the body under construction.

    ``entries`` holds this level's statements and nested scopes in REVERSE order, so a
    prepend is an append; ``finish`` reverses once. ``children`` indexes the nested scopes by
    axis, so descending a path costs a dict lookup instead of a scan for a matching ``Loop``."""

    __slots__ = ("axis", "children", "entries")

    def __init__(self, axis: Axis | None = None) -> None:
        self.axis = axis
        self.entries: list = []
        self.children: dict[Axis, _Scope] = {}


class LoopBuilder:
    """Mutable accumulator for a ``LoopOp`` body."""

    def __init__(self, used_names: set[str]) -> None:
        self._root = _Scope()
        self._used: set[str] = set(used_names)
        self._next_suffix: dict[str, int] = {}

    def fresh(self, hint: str) -> str:
        """Return an unused name derived from ``hint`` and reserve it."""
        if hint not in self._used:
            self._used.add(hint)
            self._next_suffix[hint] = 1
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
        scope = self._root
        for axis in enclosure.enclosing:
            child = scope.children.get(axis)
            if child is None:
                child = _Scope(axis)
                scope.children[axis] = child
                scope.entries.append(child)  # a new level prepends, like any other entry
            scope = child
        scope.entries.append(stmt)

    def finish(self) -> Body:
        """Return the accumulated body — the one point the immutable tree is built."""
        return _materialize(self._root)


def _materialize(scope: _Scope) -> Body:
    """The scope tree as loop IR: reverse each level back into insertion order, and give every
    nested scope its ``Loop``. A scope carries only its axis, so a ``Loop`` built here takes the
    defaults for every other field — as it did when each insert rebuilt it."""
    return tuple(
        Loop(axis=entry.axis, body=_materialize(entry)) if isinstance(entry, _Scope) else entry for entry in reversed(scope.entries)
    )
