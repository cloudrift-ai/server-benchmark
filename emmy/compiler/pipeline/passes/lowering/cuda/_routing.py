"""Shared construction helpers for direct deterministic routing kernels."""

from __future__ import annotations

from emmy.compiler.dim import to_dim
from emmy.compiler.ir.expr import Var
from emmy.compiler.pipeline import RuleSkipped


def row_launch(tensor) -> tuple[str, object, tuple[str, ...]]:
    """Return the kernel row expression, grid factor, and runtime arguments."""
    rows = to_dim(tensor.shape[0])
    if rows.is_static:
        value = rows.as_static()
        return str(value), value, ()
    names = tuple(sorted(rows.expr.free_vars()))
    if len(names) != 1 or not isinstance(rows.expr, Var) or rows.expr.name != names[0]:
        raise RuleSkipped(f"direct routing lowering requires one bare symbolic row dimension, got {rows}")
    return names[0], rows.expr, names
