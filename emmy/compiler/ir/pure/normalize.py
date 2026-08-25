"""Canonical forms for pure Lambda terms.

Like Loop IR's ``normalize_body`` and Tile IR's ``normalize_fold_tree``, construction
normalization is an idempotent transform invoked by the owning IR node's ``__post_init__``.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Assign, Load


def _canonical_body_order(body: Body) -> Body:
    """Return a deterministic dependency-respecting order for a pure ANF body."""
    stmts = tuple(body)
    if len(stmts) <= 1:
        return body

    def token(stmt) -> tuple:
        op = getattr(stmt, "op", None)
        return (
            type(stmt).__name__,
            getattr(op, "name", "") if op is not None else "",
            stmt.input if isinstance(stmt, Load) else "",
            len(getattr(stmt, "args", ()) or ()),
        )

    def reads(stmt) -> set[str]:
        out = set(stmt.deps())
        for nested in stmt.nested():
            for child in nested:
                out |= reads(child)
        return out

    definitions = [
        set(stmt.defines()) | {name for nested in stmt.nested() for child in nested for name in child.defines()} for stmt in stmts
    ]
    dependencies = [reads(stmt) for stmt in stmts]
    placed = []
    remaining = list(range(len(stmts)))
    while remaining:
        remaining_definitions = {name for index in remaining for name in definitions[index]}
        ready = [index for index in remaining if not dependencies[index] & (remaining_definitions - definitions[index])]
        if not ready:
            return body
        selected = min(ready, key=lambda index: (token(stmts[index]), index))
        placed.append(stmts[selected])
        remaining.remove(selected)
    return Body(placed)


def normalize_lambda_body(body: Body) -> Body:
    """Canonicalize context-independent statement order and commutative arguments."""
    ordered = _canonical_body_order(body)
    return Body(
        replace(stmt, args=tuple(sorted(stmt.args))) if isinstance(stmt, Assign) and stmt.op.commutative and len(stmt.args) > 1 else stmt
        for stmt in ordered
    )


__all__ = ["normalize_lambda_body"]
