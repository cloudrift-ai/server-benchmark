"""General exp-family monoid rewrite over a Tile IR Fold tree."""

from __future__ import annotations

from dataclasses import dataclass, replace

from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Fold, Lambda, component_ops
from emmy.compiler.ir.pure.fold import _operand_result_names
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Load, Select
from emmy.compiler.ir.stmt.body import _member_reads
from emmy.compiler.ir.tile.normalize import lambda_equivalent_clusters
from emmy.compiler.ir.tile.ops import split_invariant_factors


@dataclass(frozen=True)
class _ScopedLambda:
    fn: Lambda
    axes: tuple[str, ...]


@dataclass(frozen=True)
class _Component:
    state: str
    invariant: tuple[str, ...]
    value_body: Body
    values: tuple[str, ...]


@dataclass(frozen=True)
class _Member:
    section: str
    value: object


def _bindings(fold: Fold) -> dict[str, object]:
    return {name: edge for edge in fold.operands for name in _operand_result_names(edge)}


def _score_lambda(fold: Fold, result: str, axes: tuple[str, ...]) -> _ScopedLambda | None:
    """The one-source pure cone producing a fold's per-element score."""
    bindings = _bindings(fold)
    if result in bindings:
        members = (bindings[result],)
    else:
        members = fold.lift.body.backward_cone((result,)).members
    if not members or any(not isinstance(stmt, (Fold, Load, Assign, Select)) for stmt in members):
        return None

    used_edges = tuple(edge for name, edge in bindings.items() if any(name in stmt.deps() for stmt in members))
    indexed_loads = tuple(stmt for stmt in members if isinstance(stmt, Load) and any(expr.free_vars() for expr in stmt.index))
    inline_nodes = tuple(stmt for stmt in members if isinstance(stmt, Fold))
    if len({id(source) for source in (*used_edges, *indexed_loads, *inline_nodes)}) != 1:
        return None

    fn = Lambda(params=(fold.axis.name,), body=Body(members), results=(result,))
    return _ScopedLambda(fn=fn, axes=(*axes, fold.axis.name))


def _equivalent(left: _ScopedLambda, right: _ScopedLambda) -> bool:
    return lambda_equivalent_clusters(((left.fn, left.axes), (right.fn, right.axes))) == ((0, 1),)


def _exp_score(defs: dict[str, object], name: str, pivots: frozenset[str]) -> str | None:
    exp = defs.get(name)
    subtract = defs.get(exp.args[0]) if isinstance(exp, Assign) and exp.op.name == "exp" and len(exp.args) == 1 else None
    if isinstance(subtract, Assign) and subtract.op.name == "subtract" and len(subtract.args) == 2 and subtract.args[1] in pivots:
        return subtract.args[0]
    return None


def _components(
    fold: Fold,
    maximum: str,
    pivots: frozenset[str],
    score: _ScopedLambda,
    states: tuple[str, ...],
    axes: tuple[str, ...],
) -> tuple[_Component, ...] | None:
    """Read every additive component as weight × value × invariant factors."""
    ops = component_ops(fold.combine)
    if ops is None or any(op.reduce_canon != "add" for op in ops):
        return None

    body = fold.lift.body
    defs = body.definitions
    edge_results = frozenset(_bindings(fold))
    if edge_results and not edge_results <= {name for stmt in body for name in stmt.deps()}:
        return None

    components: list[_Component] = []
    covered: set[int] = set()
    for state, result in zip(fold.combine.results, fold.lift.results, strict=True):
        if not isinstance(result, str):
            return None
        factors = split_invariant_factors(list(body), result, fold.axis.name)
        if factors is None:
            return None
        invariant, local = factors
        if edge_results & set(invariant):
            return None

        weights: list[str] = []
        values: list[str] = []
        for name in local:
            candidate = _exp_score(defs, name, pivots)
            candidate_score = _score_lambda(fold, candidate, axes) if candidate is not None else None
            if candidate_score is not None and _equivalent(score, candidate_score):
                weights.append(name)
            else:
                values.append(name)
        if len(weights) != 1:
            return None

        value_cone = body.backward_cone(values)
        if any(not isinstance(stmt, (Fold, Load, Assign, Select)) for stmt in value_cone.members):
            return None
        banned = {maximum, *states, *fold.combine.results, *edge_results}
        if any(banned & set(_member_reads(stmt)) for stmt in value_cone.members):
            return None

        covered.update(id(stmt) for stmt in body.backward_cone((result,)).members)
        components.append(
            _Component(
                state=state,
                invariant=invariant,
                value_body=Body(value_cone.members),
                values=tuple(values),
            )
        )
    if any(id(stmt) not in covered for stmt in body):
        return None
    return tuple(components)


def _same_axis(left: Fold, right: Fold) -> bool:
    return left.axis.extent == right.axis.extent and left.axis.window == right.axis.window and left.unroll == right.unroll


def _rewrite_axis(stmt, old: str, new: str):
    if old == new:
        return stmt
    sigma = Sigma({old: Var(new)})
    return stmt.rewrite(lambda name: new if name == old else name, sigma)


def _value_term(
    component: _Component,
    fold: Fold,
    target_axis: str,
    edges: list,
    prefix: list,
) -> str | float:
    """Move one residual value cone into the merged lift or onto a direct operand edge."""
    body = [_rewrite_axis(stmt, fold.axis.name, target_axis) for stmt in component.value_body]
    values = tuple(target_axis if name == fold.axis.name else name for name in component.values)
    if not values:
        return 1.0

    if len(values) == 1 and len(body) == 1 and isinstance(body[0], Load) and body[0].name == values[0]:
        edges.append(body[0])
        return values[0]

    prefix.extend(body)
    term = values[0]
    for index, value in enumerate(values[1:]):
        name = f"{component.state}__v{index}"
        prefix.append(Assign(name=name, op="multiply", args=(term, value)))
        term = name
    return term


def _merge(maximum_fold: Fold, rest: list[tuple[int, Fold]], pivots: frozenset[str], axes: tuple[str, ...]):
    ops = component_ops(maximum_fold.combine)
    if ops is None or len(ops) != 1 or ops[0].reduce_canon != "maximum" or len(maximum_fold.combine.results) != 1:
        return None
    maximum = maximum_fold.out
    score_name = maximum_fold.lift.results[0]
    if not isinstance(score_name, str):
        return None
    score = _score_lambda(maximum_fold, score_name, axes)
    if score is None:
        return None
    score_members = maximum_fold.lift.body.backward_cone((score_name,)).members
    if {id(stmt) for stmt in score_members} != {id(stmt) for stmt in maximum_fold.lift.body}:
        return None

    states = [maximum]
    terms: list[str | float] = [score_name]
    prefix = list(maximum_fold.lift.body)
    edges = list(maximum_fold.operands)
    replacements: dict[int, tuple[Assign, ...]] = {}
    consumed: list[int] = []

    for position, fold in rest:
        if not _same_axis(maximum_fold, fold):
            continue
        found = _components(fold, maximum, pivots, score, tuple(states), axes)
        if found is None:
            continue
        epilogue: list[Assign] = []
        for component in found:
            term = _value_term(component, fold, maximum_fold.axis.name, edges, prefix)
            state = f"{component.state}__sum" if component.invariant else component.state
            states.append(state)
            terms.append(term)
            current = state
            for index, factor in enumerate(component.invariant):
                name = component.state if index == len(component.invariant) - 1 else f"{component.state}__c{index}"
                epilogue.append(Assign(name=name, op="multiply", args=(current, factor)))
                current = name
        replacements[position] = tuple(epilogue)
        consumed.append(position)

    if not consumed or len(terms) < 2 or terms[1] != 1.0:
        return None

    from emmy.compiler.ir.pure.carrier import exp_combine_states  # noqa: PLC0415

    names = tuple(states)
    other = tuple(f"{name}__o" for name in names)
    lift = Lambda(
        params=(maximum_fold.axis.name, *(name for edge in edges for name in _operand_result_names(edge))),
        body=Body(prefix),
        results=tuple(terms),
    )
    combine = Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names)
    merged = Fold(
        axis=maximum_fold.axis,
        unroll=maximum_fold.unroll,
        operands=tuple(edges),
        lift=lift,
        init=(ops[0].identity,) + (0.0,) * (len(names) - 1),
        combine=combine,
    )
    return merged, frozenset(consumed), replacements


def _rewrite_members(items: list[_Member], axes: tuple[str, ...]) -> tuple[list[_Member], bool]:
    for index, item in enumerate(items):
        fold = item.value
        if not isinstance(fold, Fold) or fold.axis is None:
            continue
        rest = [
            (position, candidate.value)
            for position, candidate in enumerate(items[index + 1 :], index + 1)
            if isinstance(candidate.value, Fold)
        ]
        pivots = {fold.out}
        for candidate in items[index + 1 :]:
            stmt = candidate.value
            if isinstance(stmt, Assign) and stmt.op.name == "copy" and len(stmt.args) == 1 and stmt.args[0] in pivots:
                pivots.add(stmt.name)
        merged = _merge(fold, rest, frozenset(pivots), axes)
        if merged is None:
            continue
        node, consumed, replacements = merged
        out: list[_Member] = []
        for position, member in enumerate(items):
            if position == index:
                out.append(_Member(member.section, node))
            elif position in consumed:
                out.extend(_Member(member.section, stmt) for stmt in replacements[position])
            else:
                out.append(member)
        return out, True
    return items, False


def _pair_members(items: list[_Member], axes: tuple[str, ...]) -> list[_Member]:
    while True:
        items, changed = _rewrite_members(items, axes)
        if not changed:
            return items


def _rewrite_body(body: Body, axes: tuple[str, ...]) -> Body:
    return Body(_rewrite_fold(stmt, axes) if isinstance(stmt, Fold) else stmt for stmt in body)


def _rewrite_fold(fold: Fold, axes: tuple[str, ...]) -> Fold:
    operands = tuple(_rewrite_fold(edge, axes) if isinstance(edge, Fold) else edge for edge in fold.operands)
    body_axes = (*axes, fold.axis.name) if fold.axis is not None else axes
    body = _rewrite_body(fold.lift.body, body_axes)

    if fold.axis is None:
        items = [*(_Member("operand", edge) for edge in operands), *(_Member("body", stmt) for stmt in body)]
        rewritten = _pair_members(items, axes)
        new_operands = tuple(item.value for item in rewritten if item.section == "operand")
        new_body = Body(item.value for item in rewritten if item.section == "body")
        if new_operands == fold.operands and new_body == fold.lift.body:
            return fold
        return Fold.projection(operands=new_operands, body=new_body, results=fold.lift.results)

    items = _pair_members([_Member("body", stmt) for stmt in body], body_axes)
    new_body = Body(item.value for item in items)
    node = fold
    if operands != fold.operands:
        node = replace(node, operands=operands)
    return node.with_bodies((new_body,)) if new_body != node.lift.body else node


def rewrite_twisted(root, axes=()):
    """Join max/denominator/expectation Fold siblings into exp-family monoids."""
    return _rewrite_fold(root, tuple(axes)) if isinstance(root, Fold) else root


__all__ = ["rewrite_twisted"]
