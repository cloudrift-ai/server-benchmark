"""Mechanical Loop IR reduction lifting.

A reduce ``Loop`` already states its fold algebra in its ``Accum`` members. Lift it directly:
recursively replace nested reductions in place, remove the ``Accum`` statements from the step,
and store their operations as the fold's componentwise monoid. There is no shape recognition
and no round-trip gate.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Lambda, M
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Select, Stmt
from emmy.compiler.ir.tile import Placement, TileOp, extract_output_specs


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


def lift_body(body) -> Body:
    """Replace every reduction in one statement tree with a ``Fold``, in place."""
    out = []
    for stmt in Body.coerce(body):
        if not isinstance(stmt, Loop):
            nested = stmt.nested()
            if nested:
                stmt = stmt.with_bodies(tuple(lift_body(child) for child in nested))
            out.append(stmt)
            continue
        if stmt.is_reduce:
            fold = fold_from_loop(stmt)
            seeds = set(fold.combine.results)
            out = [s for s in out if not (isinstance(s, Init) and s.name in seeds)]
            out.append(fold)
            continue
        out.append(replace(stmt, body=lift_body(stmt.body)))
    return Body(tuple(out))


def fold_from_loop(loop: Loop) -> Fold:
    """Lift one reduction from its explicit ``Accum`` statements."""
    loop = _stamp_axes(loop)
    body = lift_body(loop.body)
    accums = tuple(stmt for stmt in body if isinstance(stmt, Accum))
    if not accums:
        raise ValueError(f"reduce loop {loop.axis.name!r} has no Accum")
    if any(stmt.base is not None or stmt.dtype is not None for stmt in accums):
        raise ValueError(f"reduce loop {loop.axis.name!r} is not in canonical Loop IR")
    step = Body(stmt for stmt in body if not isinstance(stmt, Accum))
    names = tuple(stmt.name for stmt in accums)
    # An accumulated value defined in an ENCLOSING scope (a loop-invariant accumulate — e.g. the
    # single-key decode softmax's max over an extent-1 axis after maximal fusion) is not a name the
    # step defines, and a Lambda result must be defined by its own body. Alias it through a pure
    # copy in the step: the original loop accumulates the same value every iteration, so a
    # per-iteration copy is faithful for every monoid (max→v, add→N·v, mul→v^N).
    available = Lambda(params=(loop.axis.name,), body=step, results=()).defined
    values: list[str | float] = []
    aliases: list[Assign] = []
    for stmt in accums:
        value = stmt.value
        if isinstance(value, str) and value not in available:
            alias = f"{value}__inv_{loop.axis.name}"
            aliases.append(Assign(name=alias, op="copy", args=(value,)))
            value = alias
        values.append(value)
    if aliases:
        step = Body((*step, *aliases))
    lift = Lambda(params=(loop.axis.name,), body=step, results=tuple(values))
    init, combine = M(*(stmt.op for stmt in accums), names=names)
    return Fold(axis=loop.axis, unroll=loop.unroll, lift=lift, init=init, combine=combine)


def _peel(body: Body) -> tuple[list, list[Stmt]]:
    """Peel the outer parallel loop chain into placement axes."""
    axes = []
    prefix: list[Stmt] = []
    current = list(body)
    while True:
        index = 0
        while index < len(current) and isinstance(current[index], (Load, Assign, Init, Select)):
            index += 1
        head, rest = current[:index], current[index:]
        if len(rest) != 1 or not isinstance(rest[0], Loop) or rest[0].is_reduce:
            return axes, prefix + current
        prefix.extend(head)
        axes.append(rest[0].axis)
        current = list(rest[0].body)


def _raw_loops(body) -> list[Loop]:
    """Return every Loop that survived total reduction lifting."""
    out = []
    for stmt in Body.coerce(body):
        if isinstance(stmt, Loop):
            out.append(stmt)
        for nested in stmt.nested():
            out.extend(_raw_loops(nested))
    return out


def lift_loop_op(op: LoopOp, *, name: str = "") -> TileOp:
    """Peel free axes and lift the complete remaining nest as one Fold tree."""
    free, cell = _peel(op.body)
    split = extract_output_specs(lift_body(cell))
    if split is None:
        raise ValueError("Loop IR effects cannot be represented as output specifications")
    body, output_specs = split
    raw = _raw_loops(body)
    if raw:
        axes = ", ".join(inner.axis.name for inner in raw)
        raise ValueError(f"total lift left raw inner loops: {axes}")
    return TileOp(
        op=Fold.projection(body=Body(body)),
        name=name,
        place=Placement(free=tuple(free)),
        inputs=dict(op.inputs),
        output_specs=output_specs,
    )


__all__ = ["fold_from_loop", "lift_body", "lift_loop_op"]
