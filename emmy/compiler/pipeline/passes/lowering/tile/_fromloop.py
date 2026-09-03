"""Mechanical Loop IR reduction lifting.

A reduce ``Loop`` already states its fold algebra in its ``Accum`` members. Lift it directly:
recursively replace nested reductions in place, remove the ``Accum`` statements from the step,
and store their operations as the fold's componentwise monoid. The one shape formation reads is
the SEMIRING step (:func:`_factor_products`): its product arguments become operand edges — a slab,
a cone over the step, or the raw slab of a storage decode with the invariant factors hoisted to an
epilogue — so the bilinear reading is canonical by construction. There is no round-trip gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Lambda
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
                extra.append(Fold.slab(cone.members[0]))
            else:
                values = set(cone.external_reads) - {axis.name for axis in level.axes}
                operands, lift = _close(
                    (), _supply(values, levels[: depth + 1]), Body(cone.members), tuple(chain), level.axes, levels[: depth + 1]
                )
                extra.append(Fold(operands=operands, lift=lift))
        for name in siblings:
            term = level.exposed[name]
            level.drained.add(id(term))
            if all(term is not edge for edge in extra):
                extra.append(term)
        if not pending:
            break
    if pending:
        raise ValueError(f"a term reads {sorted(pending)}, which no enclosing level defines")
    return tuple(extra)


def _close(lead: tuple, operands: tuple, body, results: tuple, scope: tuple, levels: tuple) -> tuple[tuple, Lambda]:
    """Form a lift CLOSED over its levels — ``(operands, lift)``.

    ``Lambda.closing`` binds the operand results positionally and leaves whatever else the body
    reads as trailing params. A VALUE among those arrives as one more operand (:func:`_supply`);
    what remains free are coordinates, which a term reads without binding — only the binder can
    tell the two apart, by ``scope``.
    """
    bound = tuple(name for edge in operands for name in edge.exposes)
    lift = Lambda.closing((*lead, *bound), body, results)
    values = set(lift.params[len(lead) + len(bound) :]) - {axis.name for axis in scope}
    if values:
        operands = (*operands, *_supply(values, levels))
        bound = tuple(name for edge in operands for name in edge.exposes)
        lift = Lambda.closing((*lead, *bound), body, results)
    return operands, lift


def product_spine(defs: dict, name: str, *, divide: bool = False):
    """Flatten the ``⊗`` spine defining ``name`` into ``(leaf names, spine statements)`` — the
    spine recognized by the ``semiring_product`` TRAIT, never an op-name list. ``divide``
    additionally admits a division on the numerator side: ``(Σ x)/c`` equals ``Σ (x/c)`` for a
    fold-invariant ``c``, but nothing licenses moving a fold into a denominator, so the divisor is a
    leaf and only the numerator continues the spine. ``None`` when a spine node is not binary; a
    name with no product above it is the one-leaf product."""
    spine: list = []
    leaves: list[str] = []

    def walk(current: str) -> bool:
        stmt = defs.get(current)
        if isinstance(stmt, Assign):
            if stmt.op.semiring_product:
                if len(stmt.args) != 2:
                    return False
                spine.append(stmt)
                return all(walk(arg) for arg in stmt.args)
            if divide and stmt.op.name == "divide" and len(stmt.args) == 2:
                spine.append(stmt)
                leaves.append(stmt.args[1])
                return walk(stmt.args[0])
        leaves.append(current)
        return True

    return (tuple(leaves), tuple(spine)) if walk(name) else None


@dataclass
class _Hoist:
    """Fold-invariant factors commuted off one product argument — ``Σ_k a·(s·w) = s·Σ_k a·w`` —
    the terms and statements that define them, and the factors in spine order, each with whether it
    divides."""

    terms: tuple = ()
    stmts: tuple = ()
    factors: tuple = ()


def _decode_split(arg: str, plain: Body, defs: dict, exposed: dict, k_name: str) -> tuple[Fold, _Hoist] | None:
    """A product argument that is a STORAGE DECODE of one raw slab times fold-invariant factors,
    split into ``(the raw slab, its hoist)`` — or ``None`` for any other cone, which stays whole.

    The decode is absorbed by the slab's storage dtype (every consumer converts a bits-carrier
    element by dtype, the mma fragment loaders included), and the invariant factors commute out
    onto the accumulator: the same reassociation category as split-K. Only a decode licenses it,
    so an ordinary floating-point factor chain is never reassociated here."""
    flattened = product_spine(defs, arg, divide=True)
    leaves, spine = flattened if flattened is not None else ((arg,), ())

    def varies(leaf: str) -> bool:
        if leaf in exposed:
            return k_name in exposed[leaf].free_axes
        if leaf not in defs:
            return False  # an enclosing value — bound above the reduce loop
        reads = plain.backward_cone((leaf,)).external_reads
        return k_name in reads or any(k_name in exposed[name].free_axes for name in reads if name in exposed)

    varying = [leaf for leaf in leaves if varies(leaf)]
    decode = defs.get(varying[0]) if len(varying) == 1 else None
    if not isinstance(decode, Assign) or decode.op.decodes is None or len(decode.args) != 1:
        return None
    slab = exposed[decode.args[0]].as_slab() if decode.args[0] in exposed else None
    if slab is None or (slab.load.dtype is not None and slab.load.dtype.name != decode.op.decodes):
        return None
    seen = {id(decode)} | {id(stmt) for stmt in spine}
    stmts: list[Stmt] = []
    terms: list[Fold] = []
    for leaf in leaves:
        if leaf == varying[0]:
            continue
        if leaf in exposed:
            terms.append(exposed[leaf])
        elif leaf in defs:
            stmts.extend(stmt for stmt in plain.backward_cone((leaf,)).members if id(stmt) not in seen)
            seen.update(id(stmt) for stmt in stmts)
    if any(id(stmt) not in seen for stmt in plain.backward_cone((arg,)).members):
        return None
    terms.extend(exposed[name] for stmt in stmts for name in stmt.deps() if name in exposed and all(exposed[name] is not t for t in terms))
    divisors = {stmt.args[1] for stmt in spine if stmt.op.name == "divide"}
    factors = tuple((leaf, leaf in divisors) for leaf in leaves if leaf != varying[0])
    return exposed[decode.args[0]], _Hoist(tuple(terms), tuple(stmts), factors)


def _factor_products(
    plain: Body, values: tuple, ops: tuple, terms: tuple, scope: tuple, levels: tuple, axes: tuple, *, hoist: bool
) -> tuple:
    """Factor a SEMIRING step into ``(operand edges, products)`` — the bilinear form by construction.

    The step is a semiring step when every accumulated value is one product ``⊗`` of two distinct
    names, all products share the ⊗, and ⊗ distributes over the one commutative-monoid ⊕ the
    accumulators fold through. Each product argument the step computes (the dequant ``w_bits ×
    scale``, the normalized ``x × rsqrt``) is then a cone over the step, hoisted into a zero-axis
    term closed like any other (:func:`_close`); arguments whose cones overlap share one term
    exposing both; a slab or a nested reduce an argument names outright rides as that edge. The
    products are all that remains of the step. Any other step — a non-semiring ⊕, a square
    ``x × x``, a member no product reads — is returned as it came. With ``hoist``, an argument that
    is a storage decode times invariant factors is split (:func:`_decode_split`) and the factors
    are returned per accumulator, for the epilogue :func:`_hoisted` wraps around the fold.

    The pair is ORIENTED here, where the enclosing axis order is known: a product argument shared
    by every channel leads (the fused sibling edge's one A), else the argument carrying the earlier
    output axis does, so ``operands[0]`` is A by construction and placement reads M/N off it."""
    unfactored = (terms, plain, {})
    plus = ops[0]
    if len(set(ops)) != 1 or not (plus.associative and plus.commutative and plus.has_identity):
        return unfactored
    defs = plain.definitions
    products = [defs.get(value) for value in values]
    if any(not isinstance(product, Assign) or len(set(product.args)) != 2 for product in products):
        return unfactored
    if any(product.op != products[0].op or not product.op.distributes_over(plus) for product in products):
        return unfactored
    product_ids = {id(product) for product in products}
    exposed = {name: term for term in terms for name in term.exposes}
    computed = tuple(dict.fromkeys(arg for product in products for arg in product.args if arg in defs))
    if any(id(defs[arg]) in product_ids for arg in computed):
        return unfactored
    cones = {arg: plain.backward_cone((arg,)) for arg in computed}
    covered = product_ids | {id(member) for cone in cones.values() for member in cone.members}
    if any(id(stmt) not in covered for stmt in plain):
        return unfactored
    edge_of: dict[str, Fold] = {}
    hoists: dict[str, _Hoist] = {}
    k_name = scope[-1].name
    for arg in computed if hoist else ():
        split = _decode_split(arg, plain, defs, exposed, k_name)
        if split is not None:
            edge_of[arg], hoists[arg] = split
    # A split argument's product reads the raw slab in its place.
    raw_names = {arg: edge.exposes[0] for arg, edge in edge_of.items()}
    products = [replace(product, args=tuple(raw_names.get(name, name) for name in product.args)) for product in products]
    # Overlapping cones are one term exposing every root they share members with.
    groups: list[list[str]] = []
    for arg in (arg for arg in computed if arg not in edge_of):
        members = {id(member) for member in cones[arg].members}
        touching = [group for group in groups if any(members & {id(member) for member in cones[other].members} for other in group)]
        if touching:
            touching[0].append(arg)
            for group in touching[1:]:
                touching[0].extend(group)
                groups.remove(group)
        else:
            groups.append([arg])
    consumed: set[int] = set()
    for group in groups:
        cone = plain.backward_cone(tuple(group))
        reads = tuple(term for term in terms if any(name in cone.external_reads for name in term.exposes))
        consumed.update(id(term) for term in reads)
        operands, lift = _close((), reads, Body(cone.members), tuple(group), scope, levels)
        edge_of.update((arg, Fold(operands=operands, lift=lift)) for arg in group)
    pairs = [tuple(edge_of.get(arg) or exposed.get(arg) for arg in product.args) for product in products]  # raw names are exposed
    ordered = _orient(pairs, tuple(axis.name for axis in axes))
    split_names = {raw: arg for arg, raw in raw_names.items()}
    consumed.update(id(term) for hoist in hoists.values() for term in hoist.terms)  # a factor's slab rides the epilogue
    consumed -= {id(edge) for edge in ordered}
    leftover = tuple(term for term in terms if id(term) not in consumed and all(term is not edge for edge in ordered))
    per_state = {
        index: tuple(hoists[split_names[name]] for name in product.args if name in split_names) for index, product in enumerate(products)
    }
    hoisted = {index: hoist for index, hoist in per_state.items() if any(h.factors for h in hoist)}  # a bare decode has no epilogue
    return (*ordered, *leftover), Body(tuple(products)), hoisted


def _hoisted(fold: Fold, names: tuple[str, ...], hoists: dict, axes: tuple, levels: tuple) -> Fold:
    """The epilogue projection that applies the hoisted factors to the fold's states, under the
    original state names: the fold reduces into ``<state>__sum``, the projection multiplies (or
    divides) that by each factor in spine order and exposes the result as ``<state>``."""
    inner = {name: f"{name}__sum" for index, name in enumerate(names) if index in hoists}
    fold = replace(fold, combine=fold.combine.rename(inner))
    terms: list[Fold] = [fold]
    epilogue: list[Stmt] = []
    for index, name in enumerate(names):
        current = inner.get(name, name)
        factors = [factor for hoist in hoists.get(index, ()) for factor in hoist.factors]
        for hoist in hoists.get(index, ()):
            terms.extend(term for term in hoist.terms if all(term is not held for held in terms))
            epilogue.extend(stmt for stmt in hoist.stmts if all(stmt is not held for held in epilogue))
        for position, (leaf, divide) in enumerate(factors):
            result = name if position == len(factors) - 1 else f"{name}__c{position}"
            epilogue.append(Assign(name=result, op="divide" if divide else "multiply", args=(current, leaf)))
            current = result
    operands, lift = _close((), tuple(terms), Body(tuple(epilogue)), names, axes, levels)
    return Fold(operands=operands, lift=lift)


def _orient(pairs: list[tuple], axes: tuple[str, ...]) -> tuple[Fold, ...]:
    """The unique edges the products read, A first: the edge every channel shares, else — for one
    channel — the edge whose own output axis comes earlier in ``axes``. An argument no edge here
    supplies (an enclosing value ``_close`` binds later) leaves the order as read."""
    edges: list[Fold] = []
    for pair in pairs:
        edges.extend(edge for edge in pair if edge is not None and all(edge is not seen for seen in edges))
    if any(edge is None for pair in pairs for edge in pair):
        return tuple(edges)
    if len(pairs) > 1:
        shared = [edge for edge in edges if all(any(edge is member for member in pair) for pair in pairs)]
        lead = shared[0] if len(shared) == 1 else None
    else:
        left, right = pairs[0]
        left_only, right_only = left.free_axes - right.free_axes, right.free_axes - left.free_axes
        lead = None
        if len(left_only) == 1 and len(right_only) == 1:
            position = {name: index for index, name in enumerate(axes)}
            lead = left if position.get(next(iter(left_only)), len(axes)) <= position.get(next(iter(right_only)), len(axes)) else right
    return tuple(edges) if lead is None else (lead, *(edge for edge in edges if edge is not lead))


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
    attention's ``Σ_k P·V`` per output column — joins the enclosing level's operands, its slabs
    reading the sweep axis, and the sweep's own per-cell projection joins beside it as a
    zero-axis term evaluated over that axis. The sweep keeps its stores alone; the boundary extracts
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
            seeds = set(fold.exposes)  # the accumulators, or the epilogue's names for them once hoisted factors wrap the fold
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
                operands, lift = _close((), (), pure, results, (*axes, stmt.axis), inner_levels)
                term = Fold(operands=operands, lift=lift)
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
    # Every ``Load`` in the step becomes a SLAB — a term declaring the coordinates it indexes —
    # and a semiring step's product ARGUMENTS become operand edges too (:func:`_factor_products`):
    # a chain the step computes ahead of a product is a zero-axis cone. That is what makes a
    # semiring fold canonical BY CONSTRUCTION: the lift body is the products alone, so the bilinear
    # reading is a reading of the stored term and no later pass rewrites the tree into that form.
    # A load over COORDINATES is a slab; a data-dependent GATHER — an index reading a value the step
    # computes (the packed-pair table read by a decoded code) — is a statement of its cone: the value
    # it reads is not an axis, and a slab would declare it as one.
    defined = {name for stmt in step if not isinstance(stmt, Load) for name in stmt.defines()}
    gathers = {id(stmt) for stmt in step if isinstance(stmt, Load) and any(expr.free_vars() & defined for expr in stmt.index)}
    slabs = tuple(Fold.slab(stmt) for stmt in step if isinstance(stmt, Load) and id(stmt) not in gathers)
    plain = Body(stmt for stmt in step if not isinstance(stmt, Load) or id(stmt) in gathers)
    values, ops = tuple(stmt.value for stmt in accums), tuple(stmt.op for stmt in accums)
    edges, plain, hoists = _factor_products(plain, values, ops, (*edges, *slabs), scope, levels, axes, hoist=not writes)
    names = tuple(stmt.name for stmt in accums)
    # FORM the lift closed: a value the step reads from an enclosing level arrives as an operand;
    # the coordinates it reads outright (a mask's ``Select``) stay free — at the construction
    # site, which is the one that knows it is turning a Loop into a term.
    edges, lift = _close((loop.axis.name,), edges, plain, values, axes, levels)
    if not all(op.has_identity for op in ops):
        raise ValueError(f"reduce loop {loop.axis.name!r}: an Accum op without an identity is not a monoid ⊕")
    init, combine = tuple(op.identity for op in ops), Lambda.componentwise(ops, names)
    if not writes:
        fold = Fold(operands=edges, lift=lift, init=init, combine=combine)
        return (_hoisted(fold, names, hoists, axes, levels) if hoists else fold), ()
    stored = tuple(dict.fromkeys(value for stmt in writes for value in stmt.values))
    if any(value not in names for value in stored):
        raise ValueError(f"reduce loop {loop.axis.name!r}: a per-step store may only observe the carried state {names}")
    observe = Lambda(
        params=(loop.axis.name, *names),
        body=Body(tuple(Assign(name=f"{value}__obs", op="copy", args=(value,)) for value in stored)),
        results=tuple(f"{value}__obs" for value in stored),
    )
    fold = Fold(operands=edges, lift=lift, init=init, combine=combine, observe=observe)
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
    # one param per operand result component, positionally, and reads the grid axes free.
    # It exposes what the kernel stores — its body's last definition, or with no body of its own
    # the operand values the boundary writes — so a wrapper over one operand is the identity
    # projection normalization dissolves, rather than a permanent layer over every bare kernel.
    results = _root_results(Body(body)) or tuple(dict.fromkeys(value for spec in output_specs for value in spec.write.values))
    edges, lift = _close((), edges, Body(body), results, tuple(free), ())
    # The kernel's axis table: the free axes and every loop the nest bound (reduce, sweep), by
    # name — the term names them, the kernel holds their extents.
    axes = {axis.name: axis for axis in (*free, *(loop.axis for loop in Body.coerce(cell).loops))}
    return TileOp(
        op=Fold(operands=edges, lift=lift),
        name=name,
        place=Placement(free=tuple(free)),
        axes=tuple(axes.values()),
        inputs=dict(op.inputs),
        output_specs=output_specs,
    )


__all__ = ["fold_from_loop", "lift_body", "lift_loop_op"]
