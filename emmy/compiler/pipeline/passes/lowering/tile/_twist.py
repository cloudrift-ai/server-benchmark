"""The twisted rewrite over a Tile IR Fold tree — every recipe tried on every reduce that reads a
reduce (:mod:`~emmy.compiler.ir.pure.twist`), the tree's operands rewritten onto each fold that
clicks, to a fixpoint. The one algebra of its own: a factor constant along the fold axis commutes
out of an additive fold, which is what exposes an expectation channel's pattern (attention's
``exp(s − m)·v / l`` streams ``exp(s − m)·v`` and applies ``1/l`` once, after the fold)."""

from __future__ import annotations

import logging
from dataclasses import replace

from emmy.compiler.ir.pure import Fold, Lambda
from emmy.compiler.ir.pure.twist import RECIPES
from emmy.compiler.ir.stmt import Assign, Body, Load
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import product_spine

logger = logging.getLogger(__name__)


def _decline(what: str, why: str) -> None:
    """Record why a cluster that LOOKS like a recipe's did not click. Declining is not neutral:
    the kernel keeps a planar fold, and the only visible symptom is the measurement — so
    ``compile -vv`` names the predicate that refused instead of leaving a silent demotion."""
    logger.debug("twisted rewrite declined (%s): %s", what, why)


def _terms(root: Fold):
    """Every term of the tree, each object once, readers before what they read."""
    seen: set[int] = set()
    pending = [root]
    while pending:
        term = pending.pop()
        if id(term) in seen:
            continue
        seen.add(id(term))
        yield term
        pending.extend(reversed(term.operands))


def _replace(term: Fold, mapping: dict[int, tuple[Fold, dict[int, int]]]) -> Fold:
    """The tree with every operand in ``mapping`` (by identity) replaced by ``(edge, seats)`` — the
    replacement and, per old component, WHICH of its results now carries that value. Binding is
    positional, so a reader keeps its own param names and only their order follows the new
    operands; an operand reached twice binds once, and a component no old edge covered binds a
    fresh name.

    Seats, not an offset: the fused carrier seats each channel where its recipe declares it, so an
    absorbed edge's components can land interleaved rather than as a run."""
    order: list[Fold] = []
    slots: dict[int, list[str | None]] = {}
    for param, edge, index in term.bindings:
        new, seats = mapping.get(id(edge), (edge, None))
        if id(new) not in slots:
            order.append(new)
            slots[id(new)] = [None] * len(new.exposes)
        slots[id(new)][index if seats is None else seats[index]] = param
    operands = tuple(_replace(edge, mapping) for edge in order)
    if len(operands) == len(term.operands) and all(a is b for a, b in zip(operands, term.operands, strict=True)):
        return term
    lead = term.lift.params[:1] if term.axis is not None else ()
    bound = [param or f"_unread{index}" for index, param in enumerate(param for edge in order for param in slots[id(edge)])]
    params = (*lead, *bound, *term.lift.params[len(lead) + len(term.bindings) :])
    return replace(term, operands=operands, lift=replace(term.lift, params=params))


def _varies(fold: Fold, name: str, bound: dict[str, Fold]) -> bool:
    """Whether ``name`` — a lift param or a body definition — changes along the fold's axis."""
    if name == fold.axis:
        return True
    if name in bound:
        return fold.axis in bound[name].free_axes
    if name not in fold.lift.params:
        return any(_varies(fold, read, bound) for read in fold.lift.cone(name).params)
    return False


def _hoist_invariant(fold: Fold) -> tuple[Fold, Fold] | None:
    """``Σ_k c·x_k = c·Σ_k x_k`` for every factor ``c`` of the summand constant along the axis —
    the fold over the varying factors alone and the epilogue projection that applies the rest to
    its state under the original name, or ``None`` when nothing is invariant. Divisors hoist as
    divisions; a varying divisor keeps the fold whole."""
    view = fold.as_reduction()
    if view is None or view.ops is None or len(view.states) != 1 or view.ops[0].reduce_canon != "add":
        return None
    result = fold.lift.results[0]
    flattened = product_spine(fold.lift.body.definitions, result, divide=True)
    if flattened is None:
        return None
    leaves, spine = flattened
    bound = {param: edge for param, edge, _ in fold.bindings}
    divisors = {stmt.args[1] for stmt in spine if stmt.op.name == "divide"}
    invariant = [leaf for leaf in leaves if not _varies(fold, leaf, bound)]
    varying = [leaf for leaf in leaves if leaf not in invariant]
    if not invariant or not varying or any(leaf not in bound for leaf in invariant) or any(leaf in divisors for leaf in varying):
        return None
    spine_ids = {id(stmt) for stmt in spine}
    kept = [stmt for stmt in fold.lift.body if id(stmt) not in spine_ids]
    product: list[Assign] = []
    value = varying[0]
    for index, leaf in enumerate(varying[1:]):
        product.append(Assign(name=f"{result}__k{index}", op="multiply", args=(value, leaf)))
        value = product[-1].name
    state = view.states[0]
    inner_state = f"{state}__sum"
    reads = {name for stmt in (*kept, *product) for name in stmt.deps()} | {value}
    kept_edges = {id(edge) for param, edge, _ in fold.bindings if param in reads}
    operands = tuple(edge for edge in fold.operands if id(edge) in kept_edges)
    lift = Lambda(
        params=(
            fold.axis,
            *(param for param, edge, _ in fold.bindings if id(edge) in kept_edges),
            *fold.lift.params[1 + len(bound) :],
        ),
        body=Body((*kept, *product)),
        results=(value,),
    )
    inner = replace(fold, operands=operands, lift=lift, base=fold.base.rename({state: inner_state}))
    epilogue: list[Assign] = []
    current = inner_state
    for index, leaf in enumerate(invariant):
        name = state if index == len(invariant) - 1 else f"{state}__c{index}"
        epilogue.append(Assign(name=name, op="divide" if leaf in divisors else "multiply", args=(current, leaf)))
        current = name
    edges = (inner, *dict.fromkeys(bound[leaf] for leaf in invariant))
    params = (inner_state, *(param for edge in edges[1:] for param, held, _ in fold.bindings if held is edge))
    projection = Fold(operands=edges, lift=Lambda.closing(params, Body(epilogue), (state,)))
    return inner, projection


def _inline(term: Fold, edge: Fold) -> Fold:
    """``term`` with its zero-axis operand ``edge`` composed into the lift — the reader binds the
    edge's operands in its place and computes the edge's body itself, a projection β-reduced into
    its reader. What lets a recipe see a pivot a projection stands in front of (Welford's
    ``mean = sum / N``); the tree keeps the projection for any other reader."""
    (param,) = [p for p, e, _ in term.bindings if e is edge]
    applied = edge.applied
    loads = tuple(stmt for stmt in applied.body if isinstance(stmt, Load))  # a projection keeps its gmem reads inline
    reader = term.lift.rename({param: applied.results[0]})
    lead = 1 if term.axis is not None else 0
    params: list[str] = list(reader.params[:lead])
    operands: list[Fold] = []
    for operand in term.operands:
        if operand is edge:
            params.extend((*applied.params, *(load.name for load in loads)))
            operands.extend((*edge.operands, *(Fold.slab(load) for load in loads)))
        else:
            params.extend(p for p, e, _ in term.bindings if e is operand)
            operands.append(operand)
    params.extend(reader.params[lead + len(term.bindings) :])
    body = Body((*(stmt for stmt in applied.body if not isinstance(stmt, Load)), *reader.body))
    return replace(term, operands=tuple(operands), lift=Lambda(params=tuple(params), body=body, results=reader.results))


def _reads_reduce(edge: Fold) -> bool:
    return edge.axis is None and any(operand.axis is not None for operand in edge.operands)


def _candidates(term: Fold):
    """The spellings of ``term`` a recipe is tried on, each with the replacement its fusion implies
    — the term itself; with its invariant factors hoisted into an epilogue projection; with a
    projection composed in, so a pivot behind one is an operand, and hoisted again after that."""

    def spellings(candidate: Fold):
        yield candidate, lambda fused, candidate=candidate: (fused, _seats(candidate, fused))
        hoisted = _hoist_invariant(candidate)
        if hoisted is not None:
            inner, epilogue = hoisted
            # The epilogue stands in for the whole candidate, so it exposes one result and seats it first.
            # The epilogue stands in for the whole candidate: it exposes the state under its
            # original name, so the reader above seats that one component first.
            yield (
                inner,
                lambda fused, inner=inner, epilogue=epilogue: (_replace(epilogue, {id(inner): (fused, _seats(inner, fused))}), {0: 0}),
            )

    yield from spellings(term)
    for edge in term.operands:
        if _reads_reduce(edge) and len(edge.exposes) == 1:
            # A factored product argument (flash's P cone) is composed back in, then hoisted like
            # the term itself: the two rewrites compose, so a pivot behind a cone that also carries
            # an invariant factor (P's ``1/l``) is reachable.
            yield from spellings(_inline(term, edge))


def _click(root: Fold, axes: dict) -> dict[int, tuple[Fold, int]] | None:
    """The first fusion some recipe accepts anywhere in the tree, as the operand replacement it
    implies — the dependent and its pivot both become the fused fold, each of their states seated
    where the fused carrier now carries it (:func:`_seats`)."""
    for term in _terms(root):
        # A reduce reading a reduce — directly, or through a projection of one (Welford's mean).
        if term.axis is None or not any(edge.axis is not None or _reads_reduce(edge) for edge in term.operands):
            continue
        for recipe in RECIPES:
            for candidate, stands_in in _candidates(term):
                fused = candidate.fuse(recipe, axes)
                if fused is not None:
                    pivot = _pivot_of(candidate, fused)
                    return {id(term): stands_in(fused), id(pivot): (fused, _seats(pivot, fused))}
    return None


def _seats(edge: Fold, fused: Fold) -> dict[int, int]:
    """Where each of ``edge``'s components sits among ``fused``'s — by NAME, because the fusion
    carries the absorbed states' own spellings and seats them in the recipe's declared order."""
    where = {name: index for index, name in enumerate(fused.exposes)}
    return {index: where[name] for index, name in enumerate(edge.exposes) if name in where}


def _pivot_of(dependent: Fold, fused: Fold) -> Fold:
    """The operand of ``dependent`` the fusion absorbed — the reduce whose whole carrier the fused
    fold now holds, under the same pivot.

    CONTAINMENT, not a prefix: a channel joins the carrier at the position its RECIPE declares, so a
    later fusion can seat a state ahead of one the pivot already carried (softmax's denominator
    lands before an expectation that clicked first)."""
    states = fused.as_reduction().states
    return next(
        edge
        for edge in dependent.operands
        if edge.axis is not None and (carried := edge.as_reduction().states)[0] == states[0] and set(carried) <= set(states)
    )


def _report(root: Fold, axes: dict) -> None:
    """Name every pivot-shaped reduce left beside a same-axis sibling once the fixpoint settles —
    the shape this pass exists for, refusing, and the demotion is otherwise invisible."""
    reduces = [term for term in _terms(root) if term.axis is not None]
    pivots = {recipe.pivot for recipe in RECIPES}
    for fold in reduces:
        view = fold.as_reduction()
        if view.ops is None or view.ops[0].reduce_canon not in pivots:
            continue
        siblings = [other for other in reduces if other is not fold and axes[other.axis].extent == axes[fold.axis].extent]
        if siblings:
            _decline("sibling cluster", f"{view.states[0]!r} keeps {len(siblings)} same-axis sibling(s) no recipe fuses onto it")


def rewrite_twisted(root, axes: tuple):
    """Fuse every two-pass reduce pair a recipe recognizes into its twisted carrier, to a fixpoint.
    ``axes`` is the kernel's axis table: two reduces fuse only over one extent, which the terms
    name and the table holds."""
    if not isinstance(root, Fold):
        return root
    table = {axis.name: axis for axis in axes}
    while (mapping := _click(root, table)) is not None:
        root = _replace(mapping.get(id(root), (root, 0))[0], mapping)
    _report(root, table)
    return root


__all__ = ["rewrite_twisted"]
