"""Canonical closure of sibling Fold terms over their pure dependencies."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Fold, _operand_result_names, deep_defines, deep_reads, stmt_axis_names
from emmy.compiler.ir.stmt import Assign, Body, Load, Select


def _captures(fold: Fold) -> set[str]:
    """SSA values read by ``fold``'s lowered subtree but defined outside that subtree."""
    lowered = fold.lower()
    defined = {name for stmt in lowered for name in deep_defines(stmt)}
    return deep_reads(lowered) - defined - stmt_axis_names(lowered)


def _names_of(stmt) -> set[str]:
    """Values a sibling makes available to later siblings."""
    if isinstance(stmt, Fold):
        carried = set(stmt.combine.results) if stmt.combine is not None else set(stmt.lift.results if stmt.axis is None else ())
        return deep_defines(stmt) | carried
    return set(stmt.defines())


def _reads_of(stmt) -> set[str]:
    return _captures(stmt) if isinstance(stmt, Fold) else set(deep_reads([stmt]))


def _grow(items: list, end: int, members: set[int], frontier: list[str]) -> None:
    """Extend ``members`` with the backward cone of ``frontier`` over ``items[:end]``."""
    while frontier:
        name = frontier.pop()
        for index in range(end - 1, -1, -1):
            stmt = items[index]
            if index in members or name not in _names_of(stmt):
                continue
            members.add(index)
            if not isinstance(stmt, Fold):
                frontier.extend(stmt.deps())
            break


def _renamer(mapping: dict[str, str]):
    return lambda name: mapping.get(name, name)


def close_folds(cell: list) -> list:
    """Close every sibling Fold over earlier pure values and Fold states it consumes.

    The dependency cone becomes a zero-axis operand edge. Members used only through that edge
    leave the sibling list; shared pure members stay as an alpha-renamed copy. Captures are read
    from the complete lowered Fold subtree, so a producer nested on an operand cannot hide an
    outer dependency from closure.
    """
    out = list(cell)
    for index, fold in enumerate(out):
        if not isinstance(fold, Fold) or fold.axis is None:
            continue
        free = _captures(fold)
        pure_defs = {
            name: earlier for earlier, stmt in enumerate(out[:index]) if isinstance(stmt, (Assign, Load, Select)) for name in stmt.defines()
        }
        needed = [name for name in free if name in pure_defs]
        members: set[int] = set()
        _grow(out, index, members, list(needed))
        while True:
            added = False
            for earlier, stmt in enumerate(out[:index]):
                if earlier in members or not (isinstance(stmt, Fold) and stmt.axis is not None) or not (free & _names_of(stmt)):
                    continue
                readers = [
                    other for other, user in enumerate(out) if other != earlier and user is not fold and _reads_of(user) & _names_of(stmt)
                ]
                if all(other in members for other in readers):
                    members.add(earlier)
                    needed.extend(sorted(free & _names_of(stmt)))
                    added = True
            if not added:
                break
        if not needed:
            continue

        operands = tuple(out[member] for member in sorted(members) if isinstance(out[member], Fold))
        body = tuple(out[member] for member in sorted(members) if not isinstance(out[member], Fold))
        edge_states = {name for operand in operands for name in _names_of(operand)}
        passthrough = sorted((free & edge_states) - set(needed))
        shared = {
            name
            for member in members
            if not isinstance(out[member], Fold)
            for name in out[member].defines()
            if any(other not in members and other != index and name in _reads_of(user) for other, user in enumerate(out))
        }
        rename = (
            {name: f"{name}__e{index}" for member in members if not isinstance(out[member], Fold) for name in out[member].defines()}
            if shared
            else {}
        )
        rewrite = _renamer(rename)
        results = tuple(rewrite(name) for name in (*sorted(needed), *passthrough))
        if rename:
            body = tuple(stmt.rewrite(rewrite) for stmt in body)
            existing = tuple(operand.rewrite(rewrite) for operand in fold.operands)
            lift = Lambda(
                params=(*(rewrite(param) for param in fold.lift.params), *results),
                body=Body(tuple(stmt.rewrite(rewrite) for stmt in fold.lift.body)),
                results=tuple(rewrite(result) if isinstance(result, str) else result for result in fold.lift.results),
            )
        else:
            existing = fold.operands
            lift = Lambda(params=(*fold.lift.params, *results), body=fold.lift.body, results=fold.lift.results)
        edge = Fold.projection(operands=operands, body=Body(body), results=results)
        edge_at = next(
            (operand_at for operand_at, operand in enumerate(existing) if _captures(operand) & set(results)),
            len(existing),
        )
        before = sum(len(_operand_result_names(operand)) for operand in existing[:edge_at])
        lead = 1 if fold.axis is not None else 0
        params = (*lift.params[: lead + before], *results, *lift.params[lead + before : -len(results)])
        lift = replace(lift, params=params)
        nested = (*existing[:edge_at], edge, *existing[edge_at:])
        out[index] = replace(fold, operands=nested, lift=lift)

    in_edge = {
        id(member)
        for fold in out
        if isinstance(fold, Fold)
        for edge in fold.operands
        if isinstance(edge, Fold) and edge.axis is None
        for member in (*edge.operands, *edge.body)
    }
    live_ids = {id(stmt) for stmt in out if id(stmt) not in in_edge}
    while True:
        grew = False
        for stmt in out:
            if id(stmt) in live_ids:
                continue
            names = _names_of(stmt)
            if any(id(user) in live_ids and user is not stmt and _reads_of(user) & names for user in out):
                live_ids.add(id(stmt))
                grew = True
        if not grew:
            break
    return [stmt for stmt in out if id(stmt) in live_ids]
