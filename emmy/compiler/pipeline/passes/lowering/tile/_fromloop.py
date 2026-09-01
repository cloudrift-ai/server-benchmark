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
from emmy.compiler.ir.pure.fold import Fold, _operand_result_names
from emmy.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Select, Stmt, Write
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


def lift_body(body, axes: tuple[str, ...] = ()) -> Body:
    """Replace every reduction in one statement tree with a ``Fold``, in place.

    ``axes`` names the iteration variables the ENCLOSING loops bind, threaded down from
    :func:`_peel`. A term cannot tell an axis from a value — both are a bare ``Var`` — but the
    binder can, because it bound them; so the classification arrives from above rather than being
    inferred by walking a lowered body for names that look axis-shaped."""
    out = []
    for stmt in Body.coerce(body):
        if not isinstance(stmt, Loop):
            nested = stmt.nested()
            if nested:
                stmt = stmt.with_bodies(tuple(lift_body(child, axes) for child in nested))
            out.append(stmt)
            continue
        if stmt.is_reduce:
            fold, trailing = scan_from_loop(stmt, axes)
            seeds = set(fold.combine.results)
            out = [s for s in out if not (isinstance(s, Init) and s.name in seeds)]
            out.append(fold)
            out.extend(trailing)
            continue
        out.append(replace(stmt, body=lift_body(stmt.body, (*axes, stmt.axis.name))))
    return Body(tuple(out))


def fold_from_loop(loop: Loop) -> Fold:
    """Lift one PURE reduction from its explicit ``Accum`` statements."""
    fold, trailing = scan_from_loop(loop)
    if trailing:
        raise ValueError(f"reduce loop {loop.axis.name!r} carries per-step stores — a scan, not a pure reduction")
    return fold


def scan_from_loop(loop: Loop, axes: tuple[str, ...] = ()) -> tuple[Fold, tuple[Write, ...]]:
    """Lift one reduction from its explicit ``Accum`` statements. A per-step ``Write`` makes it a
    SCAN: the store observes the carried state, so the fold gains an observer — a pure per-step
    tap binding fresh ``<state>__obs`` names — and each store returns rewritten to read the
    observed name. The rewritten stores ride the stream position after the node (the observed
    names are the fold's extra ``defines``), where boundary extraction claims them as ordinary
    ``OutputSpec``\\ s and reconstitution splices them back into the loop."""
    loop = _stamp_axes(loop)
    body = lift_body(loop.body)
    accums = tuple(stmt for stmt in body if isinstance(stmt, Accum))
    if not accums:
        raise ValueError(f"reduce loop {loop.axis.name!r} has no Accum")
    if any(stmt.base is not None or stmt.dtype is not None for stmt in accums):
        raise ValueError(f"reduce loop {loop.axis.name!r} is not in canonical Loop IR")
    writes = tuple(stmt for stmt in body if isinstance(stmt, Write))
    write_ids = {id(stmt) for stmt in writes}
    step = Body(stmt for stmt in body if not isinstance(stmt, Accum) and id(stmt) not in write_ids)
    names = tuple(stmt.name for stmt in accums)
    # FORM the lift closed: the step reads the enclosing grid / sweep axes this loop sits under
    # (a matmul cell's ``a0`` / ``a1``), and a term carries no free names, so they are bound here —
    # at the construction site, which is the one that knows it is turning a Loop into a term.
    # SEPARATE the terms: a nested reduce in the step is an operand EDGE, never a member of the
    # lift body. A Fold tree composes through operands — that is the tree — so a term embedded in
    # a lambda body would be a second, competing composition mechanism. ``splice_operands`` places
    # each edge before its first read, so lifting them out preserves evaluation order.
    edges = tuple(stmt for stmt in step if isinstance(stmt, Fold))
    plain = Body(stmt for stmt in step if not isinstance(stmt, Fold))
    bound = tuple(name for edge in edges for name in _operand_result_names(edge))
    # The enclosing binders' axes are declared because THEY said so, not because this term went
    # looking: an edge's index coordinates are as much a param as anything in the step, and only
    # the binder can say which names are axes at all.
    scope = tuple(axis for axis in axes if axis != loop.axis.name and axis not in bound)
    lift = Lambda.closing((loop.axis.name, *bound, *scope), plain, tuple(stmt.value for stmt in accums))
    init, combine = M(*(stmt.op for stmt in accums), names=names)
    if not writes:
        return Fold(axis=loop.axis, unroll=loop.unroll, operands=edges, lift=lift, init=init, combine=combine), ()
    stored = tuple(dict.fromkeys(value for stmt in writes for value in stmt.values))
    if any(value not in names for value in stored):
        raise ValueError(f"reduce loop {loop.axis.name!r}: a per-step store may only observe the carried state {names}")
    observe = Lambda(
        params=(loop.axis.name, *names),
        body=Body(tuple(Assign(name=f"{value}__obs", op="copy", args=(value,)) for value in stored)),
        results=tuple(f"{value}__obs" for value in stored),
    )
    fold = Fold(axis=loop.axis, unroll=loop.unroll, operands=edges, lift=lift, init=init, combine=combine, observe=observe)
    renamed = tuple(replace(stmt, values=tuple(f"{value}__obs" for value in stmt.values)) for stmt in writes)
    return fold, renamed


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
    split = extract_output_specs(lift_body(cell, tuple(axis.name for axis in free)))
    if split is None:
        raise ValueError("Loop IR effects cannot be represented as output specifications")
    body, output_specs = split
    raw = _raw_loops(body)
    if raw:
        axes = ", ".join(inner.axis.name for inner in raw)
        raise ValueError(f"total lift left raw inner loops: {axes}")
    return TileOp(
        op=Fold.projection(body=Body(body), axes=tuple(axis.name for axis in free)),
        name=name,
        place=Placement(free=tuple(free)),
        inputs=dict(op.inputs),
        output_specs=output_specs,
    )


__all__ = ["fold_from_loop", "lift_body", "lift_loop_op"]
