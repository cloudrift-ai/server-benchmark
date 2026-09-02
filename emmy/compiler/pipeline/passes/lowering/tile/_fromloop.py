"""Mechanical Loop IR reduction lifting.

A reduce ``Loop`` already states its fold algebra in its ``Accum`` members. Lift it directly:
recursively replace nested reductions in place, remove the ``Accum`` statements from the step,
and store their operations as the fold's componentwise monoid. There is no shape recognition
and no round-trip gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

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


def _declaring(axes: tuple, lift: Lambda, bound: int) -> tuple:
    """The enclosing axes a lift reads past its operand binding — the term's own ``axes``."""
    return tuple(axis for axis in axes if axis.name in lift.params[bound:])


@dataclass
class _Level:
    """One statement level under construction: the axes in scope and what the level has produced
    so far — its statements, and by name the terms it exposes. A term formed at or below this level
    closes over these (:func:`_supply`); what a reader took is remembered so the level can drop a
    statement or a sibling whose one remaining position is under that reader."""

    axes: tuple
    stmts: list = field(default_factory=list)
    exposed: dict = field(default_factory=dict)
    consumed: set = field(default_factory=set)
    drained: set = field(default_factory=set)


def _supply(names: set[str], levels: tuple[_Level, ...]) -> tuple[Fold, ...]:
    """The operands that close a term over the VALUES it reads from its levels, innermost first.

    A statement a term reads arrives as an operand — a bare load as a slab, a scalar chain as a
    zero-axis cone over that level's statements, itself closed the same way — and a sibling term's
    state as that sibling, the same object, so sharing keeps it one value.
    """
    extra: list[Fold] = []
    pending = set(names)
    for depth in range(len(levels) - 1, -1, -1):
        level = levels[depth]
        defined = {name for stmt in level.stmts for name in stmt.defines()}
        chain = sorted(name for name in pending if name in defined)
        siblings = sorted(name for name in pending if name in level.exposed)
        pending -= set(chain) | set(siblings)
        if chain:
            cone = Body(tuple(level.stmts)).backward_cone(tuple(chain))
            level.consumed.update(id(stmt) for stmt in cone.members)
            if len(cone.members) == 1 and isinstance(cone.members[0], Load):
                extra.append(Fold.slab(cone.members[0], level.axes))
            else:
                values = set(cone.external_reads) - {axis.name for axis in level.axes}
                operands, lift, axes = _close(
                    (), _supply(values, levels[: depth + 1]), Body(cone.members), tuple(chain), level.axes, levels[: depth + 1]
                )
                extra.append(Fold(axes=axes, operands=operands, lift=lift))
        for name in siblings:
            term = level.exposed[name]
            level.drained.add(id(term))
            if all(term is not edge for edge in extra):
                extra.append(term)
        if not pending:
            break
    assert not pending, f"a term reads {sorted(pending)}, which no enclosing level defines"
    return tuple(extra)


def _close(lead: tuple, operands: tuple, body, results: tuple, scope: tuple, levels: tuple) -> tuple[tuple, Lambda, tuple]:
    """Form a lift CLOSED over its levels — ``(operands, lift, axes)``.

    ``Lambda.closing`` binds the operand results positionally and leaves whatever else the body
    reads as trailing params. A VALUE among those arrives as one more operand (:func:`_supply`);
    what remains free are coordinates, and those are the term's own ``axes`` — the rule
    :class:`Fold` states at formation.
    """
    bound = tuple(name for edge in operands for name in edge.exposes)
    lift = Lambda.closing((*lead, *bound), body, results)
    values = set(lift.params[len(lead) + len(bound) :]) - {axis.name for axis in scope}
    if values:
        operands = (*operands, *_supply(values, levels))
        bound = tuple(name for edge in operands for name in edge.exposes)
        lift = Lambda.closing((*lead, *bound), body, results)
    return operands, lift, _declaring(scope, lift, len(lead) + len(bound))


def lift_body(body, axes: tuple = (), levels: tuple = ()) -> tuple[tuple, Body]:
    """Lift one statement tree into ``(operand terms, statements)`` — SEPARATED, bottom up.

    A reduction becomes an operand EDGE of the level it sat in; it is never substituted into the
    statement sequence where its ``Loop`` stood. So no mixed stmt/term stream exists at any point
    and none crosses a function boundary: a ``Body`` holds statements because that is all it was
    ever given, rather than because a later boundary sorted a sequence that should not have been
    built. Separation IS the construction, and so is closure: a term takes what it reads from its
    levels as operands when it is FORMED (:func:`_close`), so the level lowers its operands first
    and its statements after, with nothing left to order by dependency.

    An output SWEEP lifts to terms evaluated over the sweep coordinate: a reduce under it —
    attention's ``Σ_k P·V`` per output column — joins the enclosing level's operands with its slabs
    declaring the sweep axis, and the sweep's own per-cell projection joins beside it as a
    zero-axis term declaring that axis. The sweep keeps its stores alone; the boundary extracts
    them as sweep specs, and ``Fold.lower`` opens the sweep loop around exactly the terms evaluated
    over it.

    ``axes`` names the iteration variables the ENCLOSING loops bind, threaded down from
    :func:`_peel`; ``levels`` are the enclosing levels under construction, the providers a term
    formed here may close over. A term cannot tell an axis from a value — both are a bare ``Var``
    — but the binder can, because it bound them; so the classification arrives from above rather
    than being inferred by walking a lowered body for names that look axis-shaped.
    """
    level = _Level(axes)
    inner_levels = (*levels, level)
    edges: list = []
    for stmt in Body.coerce(body):
        if isinstance(stmt, Loop) and stmt.is_reduce:
            fold, trailing = scan_from_loop(stmt, axes, inner_levels)
            seeds = set(fold.combine.results)
            level.stmts = [m for m in level.stmts if not (isinstance(m, Init) and m.name in seeds)]
            edges.append(fold)
            level.exposed.update((name, fold) for name in fold.exposes)
            level.stmts.extend(trailing)
            continue
        if isinstance(stmt, Loop):
            # An OUTPUT SWEEP. Its reductions are terms of this level (hoisted just above); its
            # per-cell projection — the pure statements its stores read — is one more, a zero-axis
            # term evaluated over the sweep coordinate it declares, closed over the level like any
            # other. The loop keeps only its stores, which the boundary extracts as sweep specs.
            inner, cell = lift_body(stmt.body, (*axes, stmt.axis), inner_levels)
            edges.extend(inner)
            level.exposed.update((name, fold) for fold in inner for name in fold.exposes)
            writes = tuple(member for member in cell if isinstance(member, Write))
            pure = Body(tuple(member for member in cell if not isinstance(member, Write)))
            defined = {name for member in pure for name in member.defines()}
            results = tuple(dict.fromkeys(value for write in writes for value in write.values if value in defined))
            if results:
                operands, lift, sweep_axes = _close((), (), pure, results, (*axes, stmt.axis), inner_levels)
                term = Fold(axes=sweep_axes, operands=operands, lift=lift)
                edges.append(term)
                level.exposed.update((name, term) for name in term.exposes)
                cell = Body(writes)
            level.stmts.append(replace(stmt, body=cell))
            continue
        nested = stmt.nested()
        if nested:
            lifted = tuple(lift_body(child, axes, inner_levels) for child in nested)
            if any(inner for inner, _ in lifted):
                # A reduce under a conditional is not a value the level can evaluate outright.
                # Leave it intact and let the raw-loop check report it.
                level.stmts.append(stmt)
                continue
            stmt = stmt.with_bodies(tuple(cell for _, cell in lifted))
        level.stmts.append(stmt)
    stmts = Body(tuple(level.stmts))
    rest = tuple(stmt for stmt in stmts if id(stmt) not in level.consumed)
    reads = {name for stmt in rest for name in Body((stmt,)).ssa_uses}
    keep = {id(stmt) for stmt in rest} | {id(stmt) for stmt in stmts.backward_cone(tuple(reads)).members}
    # A statement or a sibling that moved under a reader stays at the level only while the level's
    # own remaining statements still read it; otherwise its one position is under that reader.
    remaining = tuple(edge for edge in edges if id(edge) not in level.drained or reads & set(edge.exposes))
    return remaining, Body(tuple(stmt for stmt in stmts if id(stmt) in keep))


def fold_from_loop(loop: Loop) -> Fold:
    """Lift one PURE reduction from its explicit ``Accum`` statements."""
    fold, trailing = scan_from_loop(loop)
    if trailing:
        raise ValueError(f"reduce loop {loop.axis.name!r} carries per-step stores — a scan, not a pure reduction")
    return fold


def scan_from_loop(loop: Loop, axes: tuple = (), levels: tuple = ()) -> tuple[Fold, tuple[Write, ...]]:
    """Lift one reduction from its explicit ``Accum`` statements. A per-step ``Write`` makes it a
    SCAN: the store observes the carried state, so the fold gains an observer — a pure per-step
    tap binding fresh ``<state>__obs`` names — and each store returns rewritten to read the
    observed name. The rewritten stores ride the stream position after the node (the observed
    names are the fold's extra ``defines``), where boundary extraction claims them as ordinary
    ``OutputSpec``\\ s and reconstitution splices them back into the loop."""
    loop = _stamp_axes(loop)
    scope = (*axes, loop.axis)
    edges, body = lift_body(loop.body, scope, levels)
    accums = tuple(stmt for stmt in body if isinstance(stmt, Accum))
    if not accums:
        raise ValueError(f"reduce loop {loop.axis.name!r} has no Accum")
    if any(stmt.base is not None or stmt.dtype is not None for stmt in accums):
        raise ValueError(f"reduce loop {loop.axis.name!r} is not in canonical Loop IR")
    writes = tuple(stmt for stmt in body if isinstance(stmt, Write))
    write_ids = {id(stmt) for stmt in writes}
    # Already separated by :func:`lift_body` — ``edges`` are the step's nested reductions, ``plain``
    # its statements. ``Fold.lower`` places each edge ahead of its reader, so the split preserves
    # evaluation order without the step ever having been a mixed sequence.
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
    # FORM the lift closed: a value the step reads from an enclosing level arrives as an operand,
    # and the coordinates it reads outright (a mask's ``Select``) are axes it declares beside its
    # own — at the construction site, which is the one that knows it is turning a Loop into a term.
    edges, lift, coordinates = _close((loop.axis.name,), edges, plain, tuple(stmt.value for stmt in accums), axes, levels)
    fold_axes = (*coordinates, loop.axis)
    init, combine = M(*(stmt.op for stmt in accums), names=names)
    if not writes:
        return Fold(axes=fold_axes, operands=edges, lift=lift, init=init, combine=combine), ()
    stored = tuple(dict.fromkeys(value for stmt in writes for value in stmt.values))
    if any(value not in names for value in stored):
        raise ValueError(f"reduce loop {loop.axis.name!r}: a per-step store may only observe the carried state {names}")
    observe = Lambda(
        params=(loop.axis.name, *names),
        body=Body(tuple(Assign(name=f"{value}__obs", op="copy", args=(value,)) for value in stored)),
        results=tuple(f"{value}__obs" for value in stored),
    )
    fold = Fold(axes=fold_axes, operands=edges, lift=lift, init=init, combine=combine, observe=observe)
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
    # The root term, constructed DIRECTLY: ``lift_body`` already handed back its operands and its
    # statements apart, so there is nothing for a former to separate, dedup or name. The lift binds
    # one param per operand result component, positionally, and declares the grid axes it reads.
    # It exposes what the kernel stores — its body's last definition, or with no body of its own
    # the operand values the boundary writes — so a wrapper over one operand is the identity
    # projection normalization dissolves, rather than a permanent layer over every bare kernel.
    results = _root_results(Body(body)) or tuple(dict.fromkeys(value for spec in output_specs for value in spec.write.values))
    edges, lift, axes = _close((), edges, Body(body), results, tuple(free), ())
    return TileOp(
        op=Fold(axes=axes, operands=edges, lift=lift),
        name=name,
        place=Placement(free=tuple(free)),
        inputs=dict(op.inputs),
        output_specs=output_specs,
    )


__all__ = ["fold_from_loop", "lift_body", "lift_loop_op"]
