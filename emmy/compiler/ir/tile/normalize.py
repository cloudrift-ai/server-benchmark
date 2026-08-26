"""Contextual canonical forms for Tile IR Fold trees.

Each :class:`Fold` already owns context-independent lambda-body ordering. The rewrites here need
the enclosing Tile axes or parent Fold; nonlocal sibling clustering belongs to later algebraic
rewrite passes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import (
    Channel,
    Fold,
    Lambda,
    component_ops,
    is_contraction,
)
from emmy.compiler.ir.pure.fold import _operand_result_names, edge_refs_axis, operand_name
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
    return Lambda(
        params=tuple(rename(name) for name in fn.params),
        body=renamed,
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


def _operand_lambda(operand, axes: tuple[str, ...]) -> tuple[Lambda, tuple[str, ...]]:
    params = tuple(axis for axis in axes if edge_refs_axis(operand, axis))
    return Lambda(params=params, body=Body((operand,)), results=(operand_name(operand),)), params


def _operand_roles(operand, axes: tuple[str, ...]) -> frozenset[str]:
    return frozenset(axis for axis in axes if edge_refs_axis(operand, axis))


def _ordered_projection(members: Iterable, results: tuple[str, ...]) -> Fold:
    """Factor an ordered pure cone without moving a Fold ahead of an earlier scalar producer.

    A projection evaluates every operand before its scalar body.  When the source sequence is
    ``Fold; scalar; Fold``, the prefix must therefore become a source projection of the latter
    Fold instead of flattening both Folds into sibling operands.
    """
    members = Body(members)
    scalar_seen = False
    split = None
    for index, stmt in enumerate(members):
        if isinstance(stmt, Fold):
            if scalar_seen:
                split = index
                break
        else:
            scalar_seen = True

    if split is not None:
        prefix, suffix = members[:split], members[split:]
        needed = set(results)
        for stmt in suffix:
            needed.update(_member_reads(stmt))
        bridge = tuple(name for stmt in prefix for name in stmt.defines() if name in needed)
        assert bridge, "a separated pure prefix must feed its suffix"
        source = _ordered_projection(prefix, bridge)
        return _ordered_projection((source, *suffix), results)

    operands = tuple(stmt for stmt in members if isinstance(stmt, Fold))
    body = Body(stmt for stmt in members if not isinstance(stmt, Fold))
    return Fold.projection(operands=operands, body=body, results=results)


def _extract_operand(body: Body, name: str):
    """Factor one product argument into a materialized load or a pure projection edge."""
    cone = body.backward_cone((name,))
    if not cone.members:
        return None
    if len(cone.members) == 1 and isinstance(cone.members[0], Load):
        load = cone.members[0]
        return (load, cone.members) if load.is_scalar and load.name == name else None
    if any(not stmt.pure for stmt in cone.members):
        return None

    if len(cone.members) == 1 and isinstance(cone.members[0], Fold) and operand_name(cone.members[0]) == name:
        return cone.members[0], cone.members
    edge = _ordered_projection(cone.members, (name,))
    return (edge, cone.members) if operand_name(edge) == name else None


@dataclass(frozen=True)
class _SemiringForm:
    plus: object
    product: object
    products: tuple[Assign, ...]
    arguments: tuple[tuple[str, str], ...]


def _semiring_form(fold: Fold) -> _SemiringForm | None:
    """Recognize the componentwise semiring law without imposing an operand-sharing shape."""
    if fold.axis is None or fold.operands or is_contraction(fold) or fold.combine is None:
        return None
    pluses = component_ops(fold.combine)
    if not pluses or len(set(pluses)) != 1:
        return None
    plus = pluses[0]
    if not (plus.associative and plus.commutative and plus.has_identity):
        return None
    if fold.init != (plus.identity,) * len(pluses) or fold.lift.params != (fold.axis.name,):
        return None

    body = fold.lift.body
    defs = body.definitions
    products: list[Assign] = []
    argument_names: list[tuple[str, str]] = []
    product_op = None
    for result in fold.lift.results:
        product = defs.get(result) if isinstance(result, str) else None
        if not isinstance(product, Assign) or len(product.args) != 2:
            return None
        if not product.op.distributes_over(plus) or (product_op is not None and product.op != product_op):
            return None
        argument_names.append(product.args)
        products.append(product)
        product_op = product.op

    if not products or len(fold.combine.results) != len(products):
        return None
    return _SemiringForm(plus=plus, product=product_op, products=tuple(products), arguments=tuple(argument_names))


def _merge_operand_cones(body: Body, extracted: dict[str, tuple]) -> tuple[tuple, dict[str, object]]:
    """Hoist maximal overlapping producer cones once, returning unique operand edges and roots."""
    roots = tuple(extracted)
    groups: list[list[str]] = []
    member_ids = {name: {id(stmt) for stmt in extracted[name][1]} for name in roots}
    for name in roots:
        touching = [i for i, group in enumerate(groups) if any(member_ids[name] & member_ids[other] for other in group)]
        if not touching:
            groups.append([name])
            continue
        first, *rest = touching
        groups[first].append(name)
        for index in reversed(rest):
            groups[first].extend(groups.pop(index))

    operands = []
    by_root = {}
    for group in groups:
        if len(group) == 1:
            edge = extracted[group[0]][0]
        else:
            ids = set().union(*(member_ids[name] for name in group))
            members = tuple(stmt for stmt in body if id(stmt) in ids)
            edge = _ordered_projection(members, tuple(group))
        operands.append(edge)
        by_root.update((name, edge) for name in group)
    return tuple(operands), by_root


def _orient_shared(pairs: list[tuple], product, axes: tuple[str, ...]) -> list[tuple]:
    """Put a product argument shared by every channel first when commutativity permits it."""
    if len(pairs) < 2 or not product.commutative:
        return pairs

    candidates = tuple(edge for pair in pairs for edge in pair)
    clusters = lambda_equivalent_clusters(_operand_lambda(edge, axes) for edge in candidates)
    complete = [cluster for cluster in clusters if {index // 2 for index in cluster} == set(range(len(pairs)))]
    if not complete:
        return pairs

    # Prefer literal reuse over merely equivalent duplicate cones. Ties retain the geometric
    # orientation already chosen below, which keeps an ordinary one-channel matmul unchanged.
    shared = min(complete, key=lambda cluster: (len({id(candidates[index]) for index in cluster}), cluster[0]))
    positions = set(shared)
    return [pair if 2 * index in positions else (pair[1], pair[0]) for index, pair in enumerate(pairs)]


def _canonical_semiring(fold: Fold, axes: tuple[str, ...]) -> Fold:
    """Factor operand cones, coalesce an equivalent shared argument, and orient it as contraction A."""
    form = _semiring_form(fold)
    if form is None:
        return fold

    body = fold.lift.body
    all_axes = (*axes, fold.axis.name)
    extracted = {}
    for name in dict.fromkeys(arg for pair in form.arguments for arg in pair):
        operand = _extract_operand(body, name)
        if operand is None:
            return fold
        extracted[name] = operand

    pairs = []
    axis_position = {name: i for i, name in enumerate(axes)}
    for left_name, right_name in form.arguments:
        left, right = extracted[left_name][0], extracted[right_name][0]
        if not edge_refs_axis(left, fold.axis.name) or not edge_refs_axis(right, fold.axis.name):
            return fold
        left_roles, right_roles = _operand_roles(left, axes), _operand_roles(right, axes)
        left_only, right_only = left_roles - right_roles, right_roles - left_roles
        if len(left_only) != 1 or len(right_only) != 1:
            return fold
        left_axis, right_axis = next(iter(left_only)), next(iter(right_only))
        pair = (left, right) if axis_position[left_axis] < axis_position[right_axis] else (right, left)
        pairs.append(pair)  # A (earlier output axis), B (later output axis)

    pairs = _orient_shared(pairs, form.product, all_axes)

    consumed = {id(stmt) for stmt in form.products}
    consumed.update(id(stmt) for _, members in extracted.values() for stmt in members)
    if any(id(stmt) not in consumed for stmt in body):
        return fold
    roots = tuple(extracted)
    members = tuple(stmt for stmt in body if id(stmt) in consumed and id(stmt) not in {id(product) for product in form.products})
    if not body.defs_die_at(members, roots=roots, allowed=form.products):
        return fold

    a_clusters = lambda_equivalent_clusters(_operand_lambda(candidate, all_axes) for candidate, _ in pairs)
    member_sets = {name: {id(stmt) for stmt in cone_members} for name, (_, cone_members) in extracted.items()}
    names = tuple(member_sets)
    shared_names = {operand_name(candidate) for candidate, _ in pairs}
    foreign_overlap = any(
        member_sets[names[i]] & member_sets[names[j]] and not {names[i], names[j]} <= shared_names
        for i in range(len(names))
        for j in range(i + 1, len(names))
    )
    if a_clusters == (tuple(range(len(pairs))),) and not foreign_overlap:
        if not form.product.commutative:
            for index, (product, (candidate_a, b)) in enumerate(zip(form.products, pairs, strict=True)):
                canonical_args = (
                    (operand_name(b), operand_name(candidate_a)) if index == 0 else (operand_name(candidate_a), operand_name(b))
                )
                if product.args != canonical_args:
                    return fold
        a = pairs[0][0]
        channels = tuple(Channel(b=b, acc=acc) for (_, b), acc in zip(pairs, fold.combine.results, strict=True))
        canonical = Fold.contraction(k_axis=fold.axis, a=a, channels=channels, product=form.product, fold_op=form.plus)
        return replace(canonical, unroll=fold.unroll)

    operands, by_root = _merge_operand_cones(body, extracted)
    # Order unique edges by the contraction-role traversal (B then A per product), retaining one
    # occurrence of every shared edge. This agrees with ``Fold.contraction`` for its shared-A
    # subset while keeping a shared B or overlapping multi-result cone singular.
    ordered = []
    for a, b in pairs:
        for edge in (by_root[operand_name(b)], by_root[operand_name(a)]):
            if not any(edge is current for current in ordered):
                ordered.append(edge)
    ordered.extend(edge for edge in operands if not any(edge is current for current in ordered))
    lift = Lambda(
        params=(fold.axis.name, *(name for edge in ordered for name in _operand_result_names(edge))),
        body=Body(form.products),
        results=fold.lift.results,
    )
    return replace(fold, operands=tuple(ordered), lift=lift)


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


def _edge_free_names(edge) -> frozenset[str]:
    if not isinstance(edge, Fold):
        return frozenset()
    free = set(edge.lift.free_names())
    for operand in edge.operands:
        free.update(_edge_free_names(operand))
    return frozenset(free)


def _close_projection(root: Fold, axes: tuple[str, ...]) -> Fold:
    """Move an enclosing projection's dependencies onto captured contraction operands."""
    assert root.axis is None
    provider_order = tuple(
        dict.fromkeys(
            (
                *(name for edge in root.operands for name in _operand_result_names(edge)),
                *(name for stmt in root.body for name in stmt.defines()),
            )
        )
    )
    provider_names = frozenset(provider_order)
    edge_by_name = {name: edge for edge in root.operands for name in _operand_result_names(edge)}
    moved_members: set[int] = set()
    moved_edges: set[int] = set()

    def provider(names: tuple[str, ...]):
        cone = root.body.backward_cone(names)
        required = set(cone.external_reads) | set(names)
        edges = tuple(dict.fromkeys(edge_by_name[name] for name in provider_order if name in required and name in edge_by_name))
        defined = {name for stmt in cone.members for name in stmt.defines()}
        defined.update(name for edge in edges for name in _operand_result_names(edge))
        if not set(names) <= defined:
            return None
        moved_members.update(id(stmt) for stmt in cone.members)
        moved_edges.update(id(edge) for edge in edges)
        return _hoist_closed_folds(Fold.projection(operands=edges, body=Body(cone.members), results=names), axes)

    def close(node: Fold) -> Fold:
        operands = tuple(close(edge) if isinstance(edge, Fold) else edge for edge in node.operands)
        body = Body(close(stmt) if isinstance(stmt, Fold) else stmt for stmt in node.body)
        current = replace(node, operands=operands) if operands != node.operands else node
        if body != current.body:
            current = current.with_bodies((body,))
        if not is_contraction(current):
            return current

        changed = False
        closed = []
        for edge in current.operands:
            if not isinstance(edge, Fold) or edge.axis is not None:
                closed.append(edge)
                continue
            needed = tuple(name for name in provider_order if name in (_edge_free_names(edge) & provider_names))
            source = provider(needed) if needed else None
            if source is None:
                closed.append(edge)
                continue
            closed.append(Fold.projection(operands=(source, *edge.operands), body=edge.body, results=edge.lift.results))
            changed = True
        return replace(current, operands=tuple(closed)) if changed else current

    rewritten_operands = tuple(close(edge) if isinstance(edge, Fold) else edge for edge in root.operands)
    rewritten_body = Body(close(stmt) if isinstance(stmt, Fold) else stmt for stmt in root.body)
    if not moved_members and not moved_edges:
        if rewritten_operands == root.operands and rewritten_body == root.body:
            return root
        return Fold.projection(operands=rewritten_operands, body=rewritten_body, results=root.lift.results)

    candidates = tuple(stmt for stmt in rewritten_body if id(stmt) in moved_members)
    moved_defs = {name for stmt in candidates for name in stmt.defines()}
    outside_reads = set(root.lift.results)
    outside_reads.update(name for stmt in rewritten_body if id(stmt) not in moved_members for name in _member_reads(stmt))
    remaining_body = (
        Body(stmt for stmt in rewritten_body if id(stmt) not in moved_members) if not (moved_defs & outside_reads) else rewritten_body
    )

    live = set(root.lift.results)
    live.update(name for stmt in remaining_body for name in _member_reads(stmt))
    remaining_operands = tuple(
        edge for edge in rewritten_operands if id(edge) not in moved_edges or live & set(_operand_result_names(edge))
    )
    return Fold.projection(operands=remaining_operands, body=remaining_body, results=root.lift.results)


def _normalize_fold(fold: Fold, axes: tuple[str, ...]) -> Fold:
    operands = tuple(_normalize_fold(edge, axes) if isinstance(edge, Fold) else edge for edge in fold.operands)
    node = replace(fold, operands=operands) if operands != fold.operands else fold
    body_axes = (*axes, node.axis.name) if node.axis is not None else axes
    body = _normalize_body(node.lift.body, body_axes)
    if body != node.lift.body:
        node = node.with_bodies((body,))
    node = _canonical_semiring(node, axes)
    if node.axis is not None:
        return node
    node = _close_projection(node, axes)
    return _hoist_closed_folds(node, axes)


def normalize_fold_tree(root, axes: Iterable[str] = ()):
    """Normalize a complete Tile IR tree bottom-up; ``None`` placeholders pass through."""
    if not isinstance(root, Fold):
        return root
    normalized = _normalize_fold(root, tuple(axes))
    return root if normalized == root else normalized


__all__ = [
    "lambda_equivalent_clusters",
    "normalize_fold_tree",
]
