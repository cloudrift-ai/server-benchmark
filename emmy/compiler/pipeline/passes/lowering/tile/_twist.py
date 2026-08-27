"""Canonical exp-family rewrites over a Tile IR Fold tree."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Fold, Lambda, component_ops, is_contraction
from emmy.compiler.ir.pure.algebra import product_spine
from emmy.compiler.ir.pure.carrier import EXP_FAMILY, exp_combine_states
from emmy.compiler.ir.pure.fold import _operand_result_names, edge_refs_axis, operand_name, refs_axis
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Load, Select
from emmy.compiler.ir.stmt.body import _member_reads
from emmy.compiler.ir.tile.normalize import lambda_equivalent_clusters

logger = logging.getLogger(__name__)


def _decline(what: str, why: str) -> None:
    """Record why a cluster that LOOKS like the exp family did not become one.

    Declining is not neutral: the kernel keeps a planar fold that
    ``ir/tile/ARCHITECTURE.md`` calls a coverage bug to investigate, and the only visible symptom
    is the measurement. The same convention the splicer uses for its unsupported patterns —
    ``compile -vv`` names the predicate that refused instead of leaving a silent demotion.
    """
    logger.debug("twisted rewrite declined (%s): %s", what, why)
    return None


@dataclass(frozen=True)
class _Score:
    fn: Lambda
    axes: tuple[str, ...]


@dataclass(frozen=True)
class _NormalizedExp:
    statistic: Fold
    provider: Fold | None
    inverse: str


def _bindings(fold: Fold) -> dict[str, object]:
    return {name: edge for edge in fold.operands for name in _operand_result_names(edge)}


def _score(fold: Fold, result: str, axes: tuple[str, ...]) -> _Score | None:
    """A fold's one-source pure score cone, scoped by its streaming axis."""
    bindings = _bindings(fold)
    members = (bindings[result],) if result in bindings else fold.body.backward_cone((result,)).members
    if not members or any(not isinstance(stmt, (Fold, Load, Assign, Select)) for stmt in members):
        return None

    used_edges = tuple(edge for name, edge in bindings.items() if any(name in _member_reads(stmt) for stmt in members))
    indexed_loads = tuple(stmt for stmt in members if isinstance(stmt, Load) and any(expr.free_vars() for expr in stmt.index))
    nodes = tuple(stmt for stmt in members if isinstance(stmt, Fold))
    if len({id(source) for source in (*used_edges, *indexed_loads, *nodes)}) != 1:
        return None
    return _Score(Lambda(params=(fold.axis.name,), body=Body(members), results=(result,)), (*axes, fold.axis.name))


def _same_score(left: _Score, right: _Score) -> bool:
    return lambda_equivalent_clusters(((left.fn, left.axes), (right.fn, right.axes))) == ((0, 1),)


def _same_axis(left: Fold, right: Fold) -> bool:
    return left.axis.extent == right.axis.extent and left.axis.window == right.axis.window and left.unroll == right.unroll


def _exp_score(defs: dict[str, object], name: str, pivots: frozenset[str]) -> str | None:
    """The score under a stable weight ``ψ(score − pivot)``, or ``None``.

    Spelled from :data:`~emmy.compiler.ir.pure.carrier.EXP_FAMILY` — the same table the combine
    generator emits — so the recognizer cannot drift from what it recognizes.
    """
    exp = defs.get(name)
    is_psi = isinstance(exp, Assign) and exp.op.name == EXP_FAMILY.psi and len(exp.args) == 1
    shift = defs.get(exp.args[0]) if is_psi else None
    if isinstance(shift, Assign) and shift.op.name == EXP_FAMILY.shift and len(shift.args) == 2 and shift.args[1] in pivots:
        return shift.args[0]
    return None


def _maximum(fold: Fold, axes: tuple[str, ...]) -> tuple[str, _Score] | None:
    if fold.axis is None:
        return None
    ops = component_ops(fold.combine)
    if ops is None or len(ops) != 1 or ops[0].reduce_canon != EXP_FAMILY.pivot:
        return None
    result = fold.lift.results[0]
    if not isinstance(result, str):
        return None
    score = _score(fold, result, axes)
    cone = fold.body.backward_cone((result,))
    if score is None or {id(stmt) for stmt in cone.members} != {id(stmt) for stmt in fold.body}:
        return None
    return fold.combine.results[0], score


def _denominator(fold: Fold, pivots: frozenset[str], score: _Score, axes: tuple[str, ...]) -> bool:
    if fold.axis is None:
        return False
    ops = component_ops(fold.combine)
    if ops is None or len(ops) != 1 or ops[0].reduce_canon != EXP_FAMILY.plus:
        return False
    result = fold.lift.results[0]
    if not isinstance(result, str):
        return False
    candidate = _exp_score(fold.body.definitions, result, pivots)
    candidate_score = _score(fold, candidate, axes) if candidate is not None else None
    cone = fold.body.backward_cone((result,))
    return (
        candidate_score is not None
        and _same_score(score, candidate_score)
        and {id(stmt) for stmt in cone.members} == {id(stmt) for stmt in fold.body}
    )


def _twisted_pair(maximum: Fold, denominator: Fold) -> Fold:
    states = (maximum.combine.results[0], denominator.combine.results[0])
    other = tuple(f"{name}__o" for name in states)
    lift = replace(maximum.lift, results=(maximum.lift.results[0], 1.0))
    combine = Lambda(params=states + other, body=Body(exp_combine_states(states, other)), results=states)
    return Fold(
        axis=maximum.axis,
        unroll=maximum.unroll,
        operands=maximum.operands,
        lift=lift,
        init=(maximum.init[0], denominator.init[0]),
        combine=combine,
    )


def _merge_siblings(operands: tuple, body: Body, axes: tuple[str, ...]) -> tuple[tuple, Body]:
    """Join a maximum and its additive exponential denominator wherever they are siblings."""
    items = [("operand", edge) for edge in operands] + [("body", stmt) for stmt in body]
    while True:
        changed = False
        for index, (_, candidate) in enumerate(items):
            found = _maximum(candidate, axes) if isinstance(candidate, Fold) else None
            if found is None:
                continue
            maximum, score = found
            pivots = {maximum}
            for _, member in items[index + 1 :]:
                if isinstance(member, Assign) and member.op.name == EXP_FAMILY.alias and len(member.args) == 1 and member.args[0] in pivots:
                    pivots.add(member.name)
            for position in range(index + 1, len(items)):
                section, denominator = items[position]
                if not isinstance(denominator, Fold) or not _same_axis(candidate, denominator):
                    continue
                if not _denominator(denominator, frozenset(pivots), score, axes):
                    continue
                items[index] = (items[index][0], _twisted_pair(candidate, denominator))
                del items[position]
                changed = True
                break
            if changed:
                break
        if not changed:
            break
    _report_unpaired(items, axes)
    return tuple(value for section, value in items if section == "operand"), Body(value for section, value in items if section == "body")


def _report_unpaired(items: list, axes: tuple[str, ...]) -> None:
    """Name every ``maximum`` fold left with a same-axis sibling once the loop above settles.

    Order-independent on purpose. :func:`_merge_siblings` pairs FORWARD only, because the
    denominator reads the pivot — but a maximum that happens to sit last is exactly as interesting
    to someone asking why their kernel stayed planar, and the forward scan would never mention it.
    Reported after the fixpoint, so a pair that DID merge is never named.
    """
    folds = [value for _, value in items if isinstance(value, Fold) and value.axis is not None]
    for fold in folds:
        found = _maximum(fold, axes)
        if found is None:
            continue
        siblings = [other for other in folds if other is not fold and _same_axis(fold, other)]
        if siblings:
            _decline(
                "sibling cluster",
                f"{found[0]!r} keeps {len(siblings)} same-axis sibling(s) carrying no "
                f"{EXP_FAMILY.psi}-weighted denominator, so the cell stays a planar fold",
            )


def _projection_members(node: Fold) -> Body:
    """Remove zero-axis grouping without lowering any iterating Fold."""
    assert node.axis is None
    members = list(node.body)
    for edge in reversed(node.operands):
        names = set(_operand_result_names(edge))
        position = next((i for i, stmt in enumerate(members) if names & set(_member_reads(stmt))), len(members))
        expanded = _projection_members(edge) if isinstance(edge, Fold) and edge.axis is None else Body((edge,))
        members[position:position] = expanded
    out = []
    for member in members:
        out.extend(_projection_members(member) if isinstance(member, Fold) and member.axis is None else (member,))
    return Body(out)


def _mul_leaves(defs: dict[str, object], name: str) -> tuple[str, ...] | None:
    """The product tree's leaves, read through the shared trait-based spine flattener.

    ``divide=True`` is the SAME reading the storage-decode hoist uses
    (``tile/normalize._decode_split``), so a normalized exponential spelled ``w / d`` parses like
    one spelled ``w * (1/d)``. Without it this depended on ``split_invariant_divides`` having
    rewritten the divide at ``LoopOp`` construction — a Loop-IR hoisting heuristic gated on a
    strict axis-subset, whose firing is not a fact this pass can assume.
    """
    flattened = product_spine(defs, name, divide=True)
    return None if flattened is None else flattened[0]


def _inverse_leaf(defs: dict[str, object], leaves: tuple[str, ...], denominator: str) -> str | None:
    """The leaf carrying ``1/denominator``, under either spelling.

    A ``reciprocal(d)`` binding names itself; a ``divide(w, d)`` records ``d`` as a spine leaf
    (:func:`~emmy.compiler.ir.pure.algebra.product_spine`), so the divisor IS the inverse leaf.
    Returns the name the projection epilogue should bind its reciprocal to.
    """
    bound = [
        leaf
        for leaf in leaves
        if isinstance((stmt := defs.get(leaf)), Assign) and stmt.op.name == EXP_FAMILY.inverse and stmt.args == (denominator,)
    ]
    if len(bound) == 1:
        return bound[0]
    divisors = [leaf for leaf in leaves if leaf == denominator]
    return f"{denominator}__inv" if len(divisors) == 1 and not bound else None


def _varying_score(body: Body, result: str, axis: str, axes: tuple[str, ...]) -> tuple[_Score, Body] | None:
    cone = body.backward_cone((result,))
    varying = {axis}
    members = []
    for stmt in cone.members:
        direct = refs_axis(stmt, axis) or (isinstance(stmt, (Fold, Load)) and edge_refs_axis(stmt, axis))
        if direct or varying & set(_member_reads(stmt)):
            members.append(stmt)
            varying.update(stmt.defines())
    if not members:
        return None
    fn = Lambda(params=(axis,), body=Body(members), results=(result,))
    return _Score(fn, (*axes, axis)), Body(members)


def _assign_cone(defs: dict[str, object], root: str, stops: frozenset[str]) -> frozenset[int]:
    found: set[int] = set()

    def visit(name: str) -> None:
        if name in stops:
            return
        stmt = defs.get(name)
        if not isinstance(stmt, Assign) or id(stmt) in found:
            return
        found.add(id(stmt))
        for arg in stmt.args:
            visit(arg)

    visit(root)
    return frozenset(found)


def _normalized_exp(edge: Fold, axis: str, axes: tuple[str, ...]) -> _NormalizedExp | None:
    """View a canonical pointwise edge as ``exp(score - maximum) / denominator``."""
    members = _projection_members(edge)
    statistics = [
        stmt
        for stmt in members
        if isinstance(stmt, Fold)
        and stmt.axis is not None
        and component_ops(stmt.combine) is None
        and len(stmt.init) == 2
        and stmt.lift.results[1:] == (1.0,)
    ]
    if len(statistics) != 1:
        return _decline("normalized exponential", f"expected one (pivot, denominator) statistic in the cone, found {len(statistics)}")
    statistic = statistics[0]

    maximum, denominator = statistic.combine.results
    pivots = {maximum}
    for stmt in members:
        if isinstance(stmt, Assign) and stmt.op.name == EXP_FAMILY.alias and len(stmt.args) == 1 and stmt.args[0] in pivots:
            pivots.add(stmt.name)

    body = Body(members)
    defs = body.definitions
    probability = operand_name(edge)
    leaves = _mul_leaves(defs, probability)
    if leaves is None or len(leaves) != 2:
        return _decline("normalized exponential", f"{probability!r} is not a two-leaf product spine")
    weights = [(leaf, _exp_score(defs, leaf, frozenset(pivots))) for leaf in leaves]
    weights = [(leaf, score) for leaf, score in weights if score is not None]
    inverse = _inverse_leaf(defs, leaves, denominator)
    if len(weights) != 1 or inverse is None:
        return _decline(
            "normalized exponential",
            f"expected one {EXP_FAMILY.psi}({EXP_FAMILY.shift}(score, pivot)) weight and one 1/{denominator} "
            f"factor; found {len(weights)} weight(s) and inverse={inverse!r} over leaves {leaves}",
        )

    score_name = weights[0][1]
    current = _varying_score(body, score_name, axis, axes)
    reference = _score(statistic, statistic.lift.results[0], axes)
    if current is None or reference is None or not _same_score(reference, current[0]):
        return _decline("normalized exponential", f"the weight's score {score_name!r} is not the statistic's own score")

    free = set(statistic.lift.free_names()) - {*axes, statistic.axis.name}
    provider_cone = body.backward_cone(free)
    captures = tuple(name for stmt in provider_cone.members for name in stmt.defines() if name in free)
    if set(captures) != free:
        return _decline("normalized exponential", f"the statistic captures {sorted(free - set(captures))}, which no operand edge provides")
    provider = Fold.projection(body=Body(provider_cone.members), results=captures) if captures else None

    expression = _assign_cone(defs, probability, frozenset({score_name, maximum, denominator}))
    consumed = {id(statistic), *expression, *(id(stmt) for stmt in current[1]), *(id(stmt) for stmt in provider_cone.members)}
    if any(id(stmt) not in consumed for stmt in members):
        return _decline("normalized exponential", "the cone carries statements the rewrite would drop")
    return _NormalizedExp(statistic=statistic, provider=provider, inverse=inverse)


def _rewrite_axis(stmt, old: str, new: str):
    if old == new:
        return stmt
    sigma = Sigma({old: Var(new)})
    return stmt.rewrite(lambda name: new if name == old else name, sigma)


def _extend_statistic(fold: Fold, view: _NormalizedExp) -> Fold:
    """Add every channel of ``sum(normalized_exp(score) * value)`` to its statistic."""
    statistic = view.statistic
    values = tuple(_rewrite_axis(channel.b, fold.axis.name, statistic.axis.name) for channel in fold.channels)
    operands = (*((view.provider,) if view.provider is not None else ()), *statistic.operands, *values)
    sums = tuple(f"{channel.acc}__sum" for channel in fold.channels)
    states = (*statistic.combine.results, *sums)
    other = tuple(f"{name}__o" for name in states)
    lift = Lambda(
        params=(statistic.axis.name, *(name for edge in operands for name in _operand_result_names(edge))),
        body=statistic.body,
        results=(*statistic.lift.results, *(operand_name(value) for value in values)),
    )
    combine = Lambda(params=states + other, body=Body(exp_combine_states(states, other)), results=states)
    merged = Fold(
        axis=statistic.axis,
        unroll=statistic.unroll,
        operands=operands,
        lift=lift,
        init=(*statistic.init, *((fold.semiring[1].identity,) * len(sums))),
        combine=combine,
    )
    epilogue = [Assign(name=view.inverse, op=EXP_FAMILY.inverse, args=(statistic.combine.results[1],))]
    epilogue.extend(
        Assign(name=channel.acc, op=EXP_FAMILY.product, args=(state, view.inverse))
        for channel, state in zip(fold.channels, sums, strict=True)
    )
    return Fold.projection(operands=(merged,), body=Body(epilogue), results=tuple(fold.defines()))


def _rewrite_fold(fold: Fold, axes: tuple[str, ...]) -> Fold:
    operands = tuple(_rewrite_fold(edge, axes) if isinstance(edge, Fold) else edge for edge in fold.operands)
    body_axes = (*axes, fold.axis.name) if fold.axis is not None else axes
    body = Body(_rewrite_fold(stmt, body_axes) if isinstance(stmt, Fold) else stmt for stmt in fold.body)
    node = replace(fold, operands=operands) if operands != fold.operands else fold
    if body != node.body:
        node = node.with_bodies((body,))

    if node.axis is None:
        new_operands, new_body = _merge_siblings(node.operands, node.body, axes)
        if new_operands == node.operands and new_body == node.body:
            return node
        return Fold.projection(operands=new_operands, body=new_body, results=node.lift.results)

    if is_contraction(node) and isinstance(node.a, Fold) and node.a.axis is None:
        view = _normalized_exp(node.a, node.axis.name, axes)
        if view is not None and _same_axis(view.statistic, node):
            product, plus = node.semiring
            if product.name == "multiply" and plus.reduce_canon == "add":
                return _extend_statistic(node, view)

    _, new_body = _merge_siblings((), node.body, body_axes)
    return node.with_bodies((new_body,)) if new_body != node.body else node


def rewrite_twisted(root, axes=()):
    """Rewrite maximum/normalized-exponential Fold algebra into exp-family monoids."""
    return _rewrite_fold(root, tuple(axes)) if isinstance(root, Fold) else root


__all__ = ["rewrite_twisted"]
