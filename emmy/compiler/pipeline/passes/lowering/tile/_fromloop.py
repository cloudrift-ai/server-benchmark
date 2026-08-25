"""Mechanical Loop IR reduction lifting.

A reduce ``Loop`` already states its fold algebra in its ``Accum`` members. Lift it directly:
recursively replace nested reductions in place, remove the ``Accum`` statements from the step,
and store their operations as the fold's componentwise monoid. There is no shape recognition
and no round-trip gate.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.pure import Lambda, M
from emmy.compiler.ir.pure.fold import Fold, is_contraction
from emmy.compiler.ir.stmt import Accum, Body, Init, Loop
from emmy.compiler.ir.tile.ops import head, reduce_loop


def _stamp_axes(loop: Loop) -> Loop:
    """Attach each accumulator to the reduce axis that contains it."""
    body = tuple(
        _stamp_axes(stmt)
        if isinstance(stmt, Loop)
        else replace(stmt, axes=(loop.axis.name,))
        if isinstance(stmt, Accum) and not stmt.axes
        else stmt
        for stmt in loop.body
    )
    return replace(loop, body=Body(body))


def _lift_body(body) -> Body:
    """Replace every reduction in one statement tree with a ``Fold``, in place."""
    out = []
    for stmt in Body.coerce(body):
        if not isinstance(stmt, Loop):
            out.append(stmt)
            continue
        if stmt.is_reduce:
            fold = fold_from_loop(stmt)
            seeds = set(fold.combine.results)
            out = [s for s in out if not (isinstance(s, Init) and s.name in seeds)]
            out.append(fold)
            continue
        out.append(replace(stmt, body=_lift_body(stmt.body)))
    return Body(tuple(out))


def fold_from_loop(loop: Loop, like: Fold | None = None) -> Fold:
    """Lift one reduction from its explicit ``Accum`` statements, without recognition."""
    del like
    loop = _stamp_axes(loop)
    body = _lift_body(loop.body)
    accums = tuple(stmt for stmt in body if isinstance(stmt, Accum))
    if not accums:
        raise ValueError(f"reduce loop {loop.axis.name!r} has no Accum")
    if any(stmt.base is not None or stmt.dtype is not None for stmt in accums):
        raise ValueError(f"reduce loop {loop.axis.name!r} is not in canonical Loop IR")
    step = Body(stmt for stmt in body if not isinstance(stmt, Accum))
    names = tuple(stmt.name for stmt in accums)
    lift = Lambda(params=(loop.axis.name,), body=step, results=tuple(stmt.value for stmt in accums))
    init, combine = M(*(stmt.op for stmt in accums), names=names)
    return Fold(axis=loop.axis, unroll=loop.unroll, lift=lift, init=init, combine=combine)


def nodify_reduce(op, like=None):
    """Lift the outer reduction used by reduction partitioning."""
    node = head(op)
    if is_contraction(node):
        return op, node
    loop = reduce_loop(op)
    if loop is None:
        return op, None
    red = fold_from_loop(loop, like)
    body = list(op.body)
    index = body.index(loop)
    if index:
        raise ValueError(f"unexpected prologue before reduce loop: {body[:index]}")
    tail = Body(body[index + 1 :])
    return (Fold.projection(operands=(red,), body=tail) if tail else red), red
