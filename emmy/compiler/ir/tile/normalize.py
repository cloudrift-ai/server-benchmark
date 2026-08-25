"""Contextual canonical forms for Tile IR Fold trees.

Each :class:`Fold` already owns context-independent lambda-body ordering. The rewrites here need
the enclosing Tile axes or parent Fold; nonlocal sibling clustering belongs to later algebraic
rewrite passes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import (
    Channel,
    Fold,
    Lambda,
    canonical_lambda_body,
    component_ops,
    is_contraction,
)
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Load
from emmy.compiler.ir.stmt.body import _member_reads


def _lambda_members(body: Body):
    """Walk every binding inside a lambda, including Fold operand edges and algebra bodies."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, Fold):
            for edge in stmt.operands:
                if isinstance(edge, Fold):
                    yield from _lambda_members(Body((edge,)))
                else:
                    yield edge
            yield from _lambda_members(stmt.lift.body)
        else:
            for nested in stmt.nested():
                yield from _lambda_members(nested)


def _canonical_lambda(fn: Lambda, axes: Iterable[str] = ()) -> Lambda:
    """Return an alpha-canonical lambda, including its enclosing iteration axes.

    :meth:`Lambda.canonical` handles names bound by the lambda itself.  A Fold tree also needs
    captured axes canonicalized so equivalent lifts at different tree positions compare equal.
    Unused enclosing axes do not affect the result.
    """
    if any(not stmt.pure for stmt in fn.body):
        raise ValueError("lambda canonicalization requires a pure body")

    body = fn.body
    members = tuple(_lambda_members(body))
    reads = {name for stmt in members for name in _member_reads(stmt)}
    bound_axes = tuple(name for stmt in members for name in stmt.binds_axes())
    axis_order = tuple(dict.fromkeys((*axes, *bound_axes)))
    active_axes = tuple(name for name in axis_order if name in reads or name in fn.params or name in bound_axes)
    names = {name: f"_a{i}" for i, name in enumerate(active_axes)}

    p = 0
    for name in fn.params:
        if name not in names:
            names[name] = f"_p{p}"
            p += 1
    v = 0
    for stmt in members:
        for name in stmt.defines():
            if name not in names:
                names[name] = f"_v{v}"
                v += 1

    def rename(name: str) -> str:
        return names.get(name, name)

    sigma = Sigma({name: Var(names[name]) for name in active_axes})

    def rename_axis(axis: Axis) -> Axis:
        name = names.get(axis.name)
        return replace(axis, name=name) if name is not None else axis

    renamed = Body(stmt.rewrite(rename, sigma, rename_axis) for stmt in body)
    canonical = canonical_lambda_body(renamed)
    return Lambda(
        params=tuple(rename(name) for name in fn.params),
        body=canonical,
        results=tuple(rename(result) if isinstance(result, str) else result for result in fn.results),
    )


def lambda_equivalent_clusters(
    items: Iterable[tuple[Lambda, Iterable[str]]],
) -> tuple[tuple[int, ...], ...]:
    """Partition scoped lambdas into alpha-equivalent clusters, in input order.

    Each item is ``(lambda, enclosing-axis-names)``.  The returned indices let a later pass keep
    its own Fold or graph metadata beside this general equivalence analysis.
    """
    clusters: dict[Lambda, list[int]] = {}
    for index, (fn, axes) in enumerate(items):
        clusters.setdefault(_canonical_lambda(fn, axes), []).append(index)
    return tuple(tuple(cluster) for cluster in clusters.values())


def _load_lambda(load: Load, axes: tuple[str, ...]) -> tuple[Lambda, tuple[str, ...]]:
    refs = set().union(*(expr.free_vars() for expr in load.index)) if load.index else set()
    params = tuple(axis for axis in axes if axis in refs)
    return Lambda(params=params, body=Body((load,)), results=(load.name,)), params


def _load_roles(load: Load, axes: tuple[str, ...]) -> frozenset[str]:
    refs = set().union(*(expr.free_vars() for expr in load.index)) if load.index else set()
    return frozenset(refs & set(axes))


def _canonical_contraction(fold: Fold, axes: tuple[str, ...]) -> Fold:
    """Rewrite one flat semiring Fold into :meth:`Fold.contraction`, or leave it unchanged."""
    if fold.axis is None or fold.operands or is_contraction(fold) or fold.combine is None:
        return fold
    pluses = component_ops(fold.combine)
    if not pluses or len(set(pluses)) != 1:
        return fold
    plus = pluses[0]
    if not (plus.associative and plus.commutative and plus.has_identity):
        return fold
    if fold.init != (plus.identity,) * len(pluses) or fold.lift.params != (fold.axis.name,):
        return fold

    body = fold.lift.body
    defs = body.definitions
    products: list[Assign] = []
    pairs: list[tuple[Load, Load]] = []
    product_op = None
    axis_position = {name: i for i, name in enumerate(axes)}
    for result in fold.lift.results:
        product = defs.get(result) if isinstance(result, str) else None
        if not isinstance(product, Assign) or product.dtype is not None or len(product.args) != 2:
            return fold
        if not product.op.distributes_over(plus) or (product_op is not None and product.op != product_op):
            return fold
        left, right = (defs.get(arg) for arg in product.args)
        if not isinstance(left, Load) or not isinstance(right, Load) or not left.is_scalar or not right.is_scalar:
            return fold
        if fold.axis.name not in set().union(*(expr.free_vars() for expr in left.index)):
            return fold
        if fold.axis.name not in set().union(*(expr.free_vars() for expr in right.index)):
            return fold

        left_roles, right_roles = _load_roles(left, axes), _load_roles(right, axes)
        left_only, right_only = left_roles - right_roles, right_roles - left_roles
        if len(left_only) != 1 or len(right_only) != 1:
            return fold
        left_axis, right_axis = next(iter(left_only)), next(iter(right_only))
        pair = (left, right) if axis_position[left_axis] < axis_position[right_axis] else (right, left)
        pairs.append(pair)  # A (earlier output axis), B (later output axis)
        products.append(product)
        product_op = product.op

    if not pairs or len(fold.combine.results) != len(pairs):
        return fold
    a = pairs[0][0]
    all_axes = (*axes, fold.axis.name)
    a_clusters = lambda_equivalent_clusters(_load_lambda(candidate, all_axes) for candidate, _ in pairs)
    if a_clusters != (tuple(range(len(pairs))),):
        return fold

    consumed = {id(stmt) for stmt in products}
    consumed.update(id(load) for pair in pairs for load in pair)
    if any(id(stmt) not in consumed for stmt in body):
        return fold

    if not product_op.commutative:
        for index, (product, (candidate_a, b)) in enumerate(zip(products, pairs, strict=True)):
            canonical_args = (b.name, candidate_a.name) if index == 0 else (candidate_a.name, b.name)
            if product.args != canonical_args:
                return fold

    channels = tuple(Channel(b=b, acc=acc) for (_, b), acc in zip(pairs, fold.combine.results, strict=True))
    canonical = Fold.contraction(k_axis=fold.axis, a=a, channels=channels, product=product_op, fold_op=plus)
    return replace(canonical, unroll=fold.unroll)


def _normalize_body(body: Body, axes: tuple[str, ...]) -> Body:
    out = []
    for stmt in body:
        if isinstance(stmt, Fold):
            out.append(_normalize_fold(stmt, axes))
            continue
        nested = stmt.nested()
        if not nested:
            out.append(stmt)
            continue
        child_axes = (*axes, *stmt.binds_axes())
        out.append(stmt.with_bodies(tuple(_normalize_body(child, child_axes) for child in nested)))
    return Body(out)


def _hoist_closed_folds(root: Fold, axes: tuple[str, ...]) -> Fold:
    """Move closed child Folds from a zero-axis body onto operand edges."""
    candidates = [stmt for stmt in root.body if isinstance(stmt, Fold) and not (set(stmt.lift.free_names()) - set(axes))]
    if not candidates:
        return root
    candidate_ids = {id(candidate) for candidate in candidates}
    remaining = Body(stmt for stmt in root.body if id(stmt) not in candidate_ids)
    operands = (*root.operands, *candidates)
    if not root.operands and len(candidates) == 1 and not remaining and root.lift.results == candidates[0].defines():
        return candidates[0]
    return Fold.projection(operands=operands, body=remaining, results=root.lift.results)


def _normalize_fold(fold: Fold, axes: tuple[str, ...]) -> Fold:
    operands = tuple(_normalize_fold(edge, axes) if isinstance(edge, Fold) else edge for edge in fold.operands)
    node = replace(fold, operands=operands) if operands != fold.operands else fold
    body_axes = (*axes, node.axis.name) if node.axis is not None else axes
    body = _normalize_body(node.lift.body, body_axes)
    if body != node.lift.body:
        node = node.with_bodies((body,))
    node = _canonical_contraction(node, axes)
    return _hoist_closed_folds(node, axes) if node.axis is None else node


def normalize_fold_tree(root, axes: Iterable[str] = ()):
    """Normalize a complete Tile IR tree bottom-up; ``None`` placeholders pass through."""
    return _normalize_fold(root, tuple(axes)) if isinstance(root, Fold) else root


__all__ = [
    "lambda_equivalent_clusters",
    "normalize_fold_tree",
]
