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


def _unbound(term: Fold) -> frozenset[str]:
    """What ``term``'s subtree still reads that it does not bind — the names its level must supply
    through operands before the term is closed. Trailing lift params past the operand binding and
    the operands' own, less the term's axes, its operands' results and its lift body's defs."""
    lead = 0 if term.axis is None else 1
    arity = sum(len(edge.exposes) for edge in term.operands)
    names = set(term.lift.params[lead + arity :])
    for edge in term.operands:
        names |= _unbound(edge)
    bound = {axis.name for axis in term.axes} | {name for edge in term.operands for name in edge.exposes}
    return frozenset(names - bound - (term.lift.defined - set(term.lift.params)))


def _closed(edges: tuple, stmts: Body, axes: tuple) -> tuple[tuple, Body]:
    """Close every term at one level over what it reads from that level.

    A term is closed by CONSTRUCTION: a statement it reads arrives as an operand — a bare load as a
    slab, a scalar chain as a zero-axis cone over the level's statements — and a sibling term's
    state as that sibling itself, the same object, so sharing keeps it one value. The level keeps
    a moved statement only while its own remaining statements still read it. Coordinates are not
    values: an axis the term reads stays a trailing param, bound by the enclosing loop.
    """
    defined_by = {name: stmt for stmt in stmts for name in stmt.defines()}
    exposed: dict[str, Fold] = {}
    consumed: set[int] = set()
    drained: set[int] = set()
    closed = []
    for edge in edges:
        needed = _unbound(edge) & (set(defined_by) | set(exposed))
        extra: list[Fold] = []
        chain = sorted(name for name in needed if name in defined_by)
        if chain:
            cone = stmts.backward_cone(tuple(chain))
            consumed.update(id(stmt) for stmt in cone.members)
            if len(cone.members) == 1 and isinstance(cone.members[0], Load):
                extra.append(Fold.slab(cone.members[0], axes))
            else:
                operands = tuple(dict.fromkeys(exposed[name] for name in sorted(cone.external_reads) if name in exposed))
                bound = tuple(name for edge in operands for name in edge.exposes)
                extra.append(Fold(operands=operands, lift=Lambda.closing(bound, Body(cone.members), tuple(chain))))
        extra.extend(dict.fromkeys(exposed[name] for name in sorted(needed) if name in exposed))
        drained.update(id(term) for term in extra)
        if extra:
            operands = (*edge.operands, *extra)
            lead = () if edge.axis is None else (edge.axis.name,)
            bound = tuple(name for edge in operands for name in edge.exposes)
            edge = replace(edge, operands=operands, lift=Lambda.closing((*lead, *bound), edge.lift.body, edge.lift.results))
        closed.append(edge)
        exposed.update((name, edge) for name in edge.exposes)
    rest = tuple(stmt for stmt in stmts if id(stmt) not in consumed)
    reads = {name for stmt in rest for name in Body((stmt,)).ssa_uses}
    still = stmts.backward_cone(tuple(reads)).members
    keep = {id(stmt) for stmt in rest} | {id(stmt) for stmt in still}
    # A sibling that moved into a reader stays at the level only while the level's own statements
    # still read it; otherwise its one position is under that reader.
    level = tuple(edge for edge in closed if id(edge) not in drained or reads & set(edge.exposes))
    return level, Body(tuple(stmt for stmt in stmts if id(stmt) in keep))


def lift_body(body, axes: tuple = ()) -> tuple[tuple, Body]:
    """Lift one statement tree into ``(operand terms, statements)`` — SEPARATED, bottom up.

    A reduction becomes an operand EDGE of the level it sat in; it is never substituted into the
    statement sequence where its ``Loop`` stood. So no mixed stmt/term stream exists at any point
    and none crosses a function boundary: a ``Body`` holds statements because that is all it was
    ever given, rather than because a later boundary sorted a sequence that should not have been
    built. Separation IS the construction, and so is closure: a term takes what it reads from its
    level as operands (:func:`_closed`), so the level lowers its operands first and its statements
    after, with nothing left to order by dependency.

    A reduce under an output SWEEP is a term evaluated over the sweep coordinate — attention's
    ``Σ_k P·V`` per output column. It joins the enclosing level's operands with its slabs declaring
    the sweep axis; the sweep keeps its pure cell and its stores, and reconstitution
    (``apply_output_specs``) wraps the term's loop back under it.

    ``axes`` names the iteration variables the ENCLOSING loops bind, threaded down from
    :func:`_peel`. A term cannot tell an axis from a value — both are a bare ``Var`` — but the
    binder can, because it bound them; so the classification arrives from above rather than being
    inferred by walking a lowered body for names that look axis-shaped.
    """
    edges: list = []
    stmts: list = []
    for stmt in Body.coerce(body):
        if isinstance(stmt, Loop) and stmt.is_reduce:
            fold, trailing = scan_from_loop(stmt, axes)
            seeds = set(fold.combine.results)
            stmts = [m for m in stmts if not (isinstance(m, Init) and m.name in seeds)]
            edges.append(fold)
            stmts.extend(trailing)
            continue
        if isinstance(stmt, Loop):
            inner, cell = lift_body(stmt.body, (*axes, stmt.axis))
            edges.extend(inner)
            stmts.append(replace(stmt, body=cell))
            continue
        nested = stmt.nested()
        if nested:
            lifted = tuple(lift_body(child, axes) for child in nested)
            if any(inner for inner, _ in lifted):
                # A reduce under a conditional is not a value the level can evaluate outright.
                # Leave it intact and let the raw-loop check report it.
                stmts.append(stmt)
                continue
            stmt = stmt.with_bodies(tuple(cell for _, cell in lifted))
        stmts.append(stmt)
    return _closed(tuple(edges), Body(stmts), axes)


def fold_from_loop(loop: Loop) -> Fold:
    """Lift one PURE reduction from its explicit ``Accum`` statements."""
    fold, trailing = scan_from_loop(loop)
    if trailing:
        raise ValueError(f"reduce loop {loop.axis.name!r} carries per-step stores — a scan, not a pure reduction")
    return fold


def scan_from_loop(loop: Loop, axes: tuple = ()) -> tuple[Fold, tuple[Write, ...]]:
    """Lift one reduction from its explicit ``Accum`` statements. A per-step ``Write`` makes it a
    SCAN: the store observes the carried state, so the fold gains an observer — a pure per-step
    tap binding fresh ``<state>__obs`` names — and each store returns rewritten to read the
    observed name. The rewritten stores ride the stream position after the node (the observed
    names are the fold's extra ``defines``), where boundary extraction claims them as ordinary
    ``OutputSpec``\\ s and reconstitution splices them back into the loop."""
    loop = _stamp_axes(loop)
    scope = (*axes, loop.axis)
    edges, body = lift_body(loop.body, scope)
    accums = tuple(stmt for stmt in body if isinstance(stmt, Accum))
    if not accums:
        raise ValueError(f"reduce loop {loop.axis.name!r} has no Accum")
    if any(stmt.base is not None or stmt.dtype is not None for stmt in accums):
        raise ValueError(f"reduce loop {loop.axis.name!r} is not in canonical Loop IR")
    writes = tuple(stmt for stmt in body if isinstance(stmt, Write))
    write_ids = {id(stmt) for stmt in writes}
    # Already separated by :func:`lift_body` — ``edges`` are the step's nested reductions, ``plain``
    # its statements. ``splice_operands`` places each edge before its first read, so the split
    # preserves evaluation order without the step ever having been a mixed sequence.
    step = tuple(stmt for stmt in body if not isinstance(stmt, Accum) and id(stmt) not in write_ids)
    # Every ``Load`` in the step becomes a SLAB — a term declaring the coordinates it indexes.
    # This is what makes a semiring fold canonical BY CONSTRUCTION: its product arguments arrive as
    # operand edges, so the lift body is the product alone and there is no non-canonical spelling
    # for a later pass to rewrite. The factoring pass that used to hoist these cones existed only
    # because the representation admitted the unfactored form.
    slabs = tuple(Fold.slab(stmt, scope) for stmt in step if isinstance(stmt, Load))
    plain = Body(stmt for stmt in step if not isinstance(stmt, Load))
    edges = (*edges, *slabs)
    names = tuple(stmt.name for stmt in accums)
    # FORM the lift closed: the step reads the enclosing grid / sweep axes this loop sits under
    # (a matmul cell's ``a0`` / ``a1``), and a term carries no free names, so they are bound here —
    # at the construction site, which is the one that knows it is turning a Loop into a term.
    bound = tuple(name for edge in edges for name in edge.exposes)
    lift = Lambda.closing((loop.axis.name, *bound), plain, tuple(stmt.value for stmt in accums))
    init, combine = M(*(stmt.op for stmt in accums), names=names)
    if not writes:
        return Fold(axes=(loop.axis,), unroll=loop.unroll, operands=edges, lift=lift, init=init, combine=combine), ()
    stored = tuple(dict.fromkeys(value for stmt in writes for value in stmt.values))
    if any(value not in names for value in stored):
        raise ValueError(f"reduce loop {loop.axis.name!r}: a per-step store may only observe the carried state {names}")
    observe = Lambda(
        params=(loop.axis.name, *names),
        body=Body(tuple(Assign(name=f"{value}__obs", op="copy", args=(value,)) for value in stored)),
        results=tuple(f"{value}__obs" for value in stored),
    )
    fold = Fold(axes=(loop.axis,), unroll=loop.unroll, operands=edges, lift=lift, init=init, combine=combine, observe=observe)
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


def _raw_loops(body: Body) -> list[Loop]:
    """Return every Loop that survived total reduction lifting.

    Takes a ``Body``: the lift hands back statements and terms already separated, so this walks a
    statement sequence with statement vocabulary and never meets a term."""
    out = []
    for stmt in body:
        if isinstance(stmt, Loop):
            out.append(stmt)
        for nested in stmt.nested():
            out.extend(_raw_loops(nested))
    return out


def _root_results(body: Body) -> tuple[str, ...]:
    """The names a root projection passes to its consumer — its body's last definition.

    Spelled here, at the one construction site that needs it, rather than as a former's default.
    """
    for stmt in reversed(tuple(body)):
        names = stmt.defines()
        if names:
            return (names[-1],)
    return ()


def lift_loop_op(op: LoopOp, *, name: str = "") -> TileOp:
    """Peel free axes and lift the complete remaining nest as one Fold tree."""
    free, cell = _peel(op.body)
    edges, stmts = lift_body(cell, tuple(free))
    split = extract_output_specs(stmts)
    if split is None:
        raise ValueError("Loop IR effects cannot be represented as output specifications")
    body, output_specs = split
    raw = _raw_loops(Body(body))
    if raw:
        axes = ", ".join(inner.axis.name for inner in raw)
        raise ValueError(f"total lift left raw inner loops: {axes}")
    return TileOp(
        # The root term, constructed DIRECTLY: ``lift_body`` already handed back its operands and
        # its statements apart, so there is nothing for a former to separate, dedup or name. The
        # lift binds one param per operand result component, positionally.
        op=Fold(
            operands=edges,
            lift=Lambda.closing(tuple(name for edge in edges for name in edge.exposes), Body(body), _root_results(Body(body))),
        ),
        name=name,
        place=Placement(free=tuple(free)),
        inputs=dict(op.inputs),
        output_specs=output_specs,
    )


__all__ = ["fold_from_loop", "lift_body", "lift_loop_op"]
