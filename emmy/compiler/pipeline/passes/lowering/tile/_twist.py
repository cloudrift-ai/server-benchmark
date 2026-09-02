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
from emmy.compiler.ir.stmt import Assign, Body

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


def _replace(term: Fold, mapping: dict[int, Fold]) -> Fold:
    """The tree with every operand in ``mapping`` (by identity) replaced. A reader rebinds its
    lift's params to the new operands' exposed names — the names themselves are unchanged, a
    fused fold keeps its accumulators' — and an operand reached twice binds once."""
    operands: list[Fold] = []
    for edge in term.operands:
        edge = _replace(mapping.get(id(edge), edge), mapping)
        if all(edge is not held for held in operands):
            operands.append(edge)
    if len(operands) == len(term.operands) and all(a is b for a, b in zip(operands, term.operands, strict=True)):
        return term
    lead = term.lift.params[:1] if term.axis is not None else ()
    arity = sum(len(edge.exposes) for edge in term.operands)
    params = (*lead, *(name for edge in operands for name in edge.exposes), *term.lift.params[len(lead) + arity :])
    return replace(term, operands=tuple(operands), lift=replace(term.lift, params=params))


def _varies(fold: Fold, name: str, bound: dict[str, Fold]) -> bool:
    """Whether ``name`` — a lift param or a body definition — changes along the fold's axis."""
    if name == fold.axis.name:
        return True
    if name in bound:
        return fold.axis.name in bound[name].free_axes
    if name not in fold.lift.params:
        return any(_varies(fold, read, bound) for read in fold.lift.cone(name).params)
    return False


def _product_spine(defs: dict, name: str, *, divide: bool = False):
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


def _hoist_invariant(fold: Fold) -> tuple[Fold, Fold] | None:
    """``Σ_k c·x_k = c·Σ_k x_k`` for every factor ``c`` of the summand constant along the axis —
    the fold over the varying factors alone and the epilogue projection that applies the rest to
    its state under the original name, or ``None`` when nothing is invariant. Divisors hoist as
    divisions; a varying divisor keeps the fold whole."""
    view = fold.as_reduction()
    if view is None or view.ops is None or len(view.states) != 1 or view.ops[0].reduce_canon != "add":
        return None
    result = fold.lift.results[0]
    flattened = _product_spine(fold.lift.body.definitions, result, divide=True)
    if flattened is None:
        return None
    leaves, spine = flattened
    bound = {name: edge for edge in fold.operands for name in edge.exposes}
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
    operands = tuple(edge for edge in fold.operands if set(edge.exposes) & reads)
    lift = Lambda(
        params=(fold.axis.name, *(name for edge in operands for name in edge.exposes), *fold.lift.params[1 + len(bound) :]),
        body=Body((*kept, *product)),
        results=(value,),
    )
    inner = replace(fold, operands=operands, lift=lift, combine=fold.combine.rename({state: inner_state}))
    epilogue: list[Assign] = []
    current = inner_state
    for index, leaf in enumerate(invariant):
        name = state if index == len(invariant) - 1 else f"{state}__c{index}"
        epilogue.append(Assign(name=name, op="divide" if leaf in divisors else "multiply", args=(current, leaf)))
        current = name
    edges = (inner, *dict.fromkeys(bound[leaf] for leaf in invariant))
    projection = Fold(
        operands=edges,
        lift=Lambda.closing(tuple(name for edge in edges for name in edge.exposes), Body(epilogue), (state,)),
    )
    return inner, projection


def _click(root: Fold) -> dict[int, Fold] | None:
    """The first fusion some recipe accepts anywhere in the tree, as the operand replacement it
    implies — the dependent and its pivot both become the fused fold (an epilogue projection
    stands in for the dependent when its invariant factors had to hoist first)."""
    for term in _terms(root):
        if term.axis is None or not any(edge.axis is not None for edge in term.operands):
            continue
        for recipe in RECIPES:
            fused = term.twist(recipe)
            if fused is not None:
                return {id(term): fused, id(_pivot_of(term, fused)): fused}
            hoisted = _hoist_invariant(term)
            if hoisted is None:
                continue
            inner, epilogue = hoisted
            fused = inner.twist(recipe)
            if fused is not None:
                return {id(term): _replace(epilogue, {id(inner): fused}), id(_pivot_of(inner, fused)): fused}
    return None


def _pivot_of(dependent: Fold, fused: Fold) -> Fold:
    """The operand of ``dependent`` the fusion absorbed — the one whose states lead ``fused``'s."""
    states = fused.as_reduction().states
    return next(edge for edge in dependent.operands if edge.axis is not None and edge.as_reduction().states == states[:-1])


def _report(root: Fold) -> None:
    """Name every pivot-shaped reduce left beside a same-axis sibling once the fixpoint settles —
    the shape this pass exists for, refusing, and the demotion is otherwise invisible."""
    reduces = [term for term in _terms(root) if term.axis is not None]
    pivots = {recipe.pivot for recipe in RECIPES}
    for fold in reduces:
        view = fold.as_reduction()
        if view.ops is None or view.ops[0].reduce_canon not in pivots:
            continue
        siblings = [other for other in reduces if other is not fold and other.axis.extent == fold.axis.extent]
        if siblings:
            _decline("sibling cluster", f"{view.states[0]!r} keeps {len(siblings)} same-axis sibling(s) no recipe fuses onto it")


def rewrite_twisted(root):
    """Fuse every two-pass reduce pair a recipe recognizes into its twisted carrier, to a fixpoint."""
    if not isinstance(root, Fold):
        return root
    while (mapping := _click(root)) is not None:
        root = _replace(mapping.get(id(root), root), mapping)
    _report(root)
    return root


__all__ = ["rewrite_twisted"]
