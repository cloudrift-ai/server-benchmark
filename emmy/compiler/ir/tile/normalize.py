"""Contextual canonical forms for Tile IR Fold trees.

Each :class:`Fold` already owns context-independent lambda-body ordering. The rewrites here need
the enclosing Tile axes or parent Fold.

INVARIANT — normalization ends with same-value cones (alpha-equal, identical captures and
interface names) as ONE shared object (:func:`_share_common_cones`). Object identity is how the
placement machinery recognizes that two consumption sites read one value, so a rewrite that
copies a cone (the close rewrites do, by design) is only sound because this final pass restores
the sharing. Recompute elimination is a
Tile-level placement concern built on that identity: a duplicated value becomes one seam, and a
composed cut materializes it once for every reader. Do NOT patch recompute downstream — a Loop IR
fusion or emission workaround sees one kernel at a time and cannot know two kernels re-derive the
same value; fix the sharing or the seam offer here instead.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from emmy.compiler.ir.expr import affine_form
from emmy.compiler.ir.pure import (
    Channel,
    Fold,
    Lambda,
    component_ops,
    is_contraction,
)
from emmy.compiler.ir.pure.algebra import product_spine
from emmy.compiler.ir.pure.closure import edge_key, equivalent_clusters, term_key
from emmy.compiler.ir.pure.fold import _operand_result_names, _ordered_projection, operand_name
from emmy.compiler.ir.schedule.views import edge_axes
from emmy.compiler.ir.stmt import Assign, Body, Load, refs_axis
from emmy.compiler.structural import instance_memo


def _loads_axis_contiguously(operand, axis: str) -> bool:
    """Whether a materialized operand carries ``axis`` only in its trailing index component.

    The same unit-stride reading one position over is ``reads_declared_rows`` in the lowering
    layer's ``_addr``; it cannot be shared because ``ir/tile`` sits below ``pipeline/passes``.
    This is the ONE layout fact canonicalization reads — see the caller's precedence note."""
    if not isinstance(operand, Load) or not operand.index or any(axis in index.free_vars() for index in operand.index[:-1]):
        return False
    form = affine_form(operand.index[-1], {axis})
    return form is not None and form[1].get(axis) == 1


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
    # Params beyond the axis are the ENVIRONMENT (:attr:`Fold.environment`) — the enclosing axes
    # the lift binds now that a term carries no free names. ``fold.operands`` is empty here, so
    # there is nothing else they could be; requiring a bare ``(axis,)`` would refuse every
    # unfactored contraction that reads an enclosing coordinate, which is all of them.
    if fold.init != (plus.identity,) * len(pluses) or fold.lift.params[:1] != (fold.axis.name,):
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
    clusters = equivalent_clusters(edge_key(edge, axes) for edge in candidates)
    complete = [cluster for cluster in clusters if {index // 2 for index in cluster} == set(range(len(pairs)))]
    if not complete:
        return pairs

    # Prefer literal reuse over merely equivalent duplicate cones. Ties retain the geometric
    # orientation already chosen below, which keeps an ordinary one-channel matmul unchanged.
    shared = min(complete, key=lambda cluster: (len({id(candidates[index]) for index in cluster}), cluster[0]))
    positions = set(shared)
    return [pair if 2 * index in positions else (pair[1], pair[0]) for index, pair in enumerate(pairs)]


def _canonical_semiring(fold: Fold, axes: tuple[str, ...], implicit_axes: frozenset[str] = frozenset()) -> Fold:
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
        if not all(fold.axis.name in edge_axes(edge, (fold.axis.name,)) for edge in (left, right)):
            return fold
        left_roles, right_roles = edge_axes(left, axes), edge_axes(right, axes)
        left_only, right_only = left_roles - right_roles, right_roles - left_roles
        unused_implicit = implicit_axes - left_roles - right_roles
        broadcast_batch = False
        if not left_only and len(right_only) == 1 and len(unused_implicit) == 1:
            left_only = unused_implicit
        elif not right_only and len(left_only) == 1 and len(unused_implicit) == 1:
            right_only = unused_implicit
        one_sided_batch = (len(left_only) > 1 and len(right_only) == 1) or (len(right_only) > 1 and len(left_only) == 1)
        if one_sided_batch:
            # A broadcast batch axis may occur in only one operand. The placement's trailing pair
            # still identifies the contraction's m/n roles; earlier free axes remain ordinary
            # grid dimensions and do not change that orientation.
            matrix_axes = frozenset(axes[-2:])
            left_matrix = left_only & matrix_axes
            right_matrix = right_only & matrix_axes
            if len(left_matrix) == 1 and len(right_matrix) == 1:
                left_only, right_only = left_matrix, right_matrix
                broadcast_batch = True
        if len(left_only) != 1 or len(right_only) != 1:
            return fold
        left_axis, right_axis = next(iter(left_only)), next(iter(right_only))
        pair = (left, right) if axis_position[left_axis] < axis_position[right_axis] else (right, left)
        if broadcast_batch and form.product.commutative:
            # Physical M/N orientation stays a placement fact; this swap decides only which
            # commutative argument SPELLS the shared A slot when the batch axis breaks the
            # symmetric geometry, preferring the operand that reads the reduction axis
            # contiguously so placement can derive a tensor-core-eligible orientation.
            # Precedence with ``_orient_shared`` below: a broadcast-batched product is
            # single-channel, where ``_orient_shared`` is a no-op; with two or more channels its
            # shared-argument rule wins and its tie-break retains the orientation chosen here.
            first, second = pair
            if _loads_axis_contiguously(second, fold.axis.name) and not _loads_axis_contiguously(first, fold.axis.name):
                pair = (second, first)
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

    a_clusters = equivalent_clusters(edge_key(candidate, all_axes) for candidate, _ in pairs)
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
        canonical = Fold.contraction(k_axis=fold.axis, a=a, channels=channels, product=form.product, fold_op=form.plus, axes=axes)
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
    # Rebuilding the contraction re-forms its lift, so the enclosing scope has to be re-declared:
    # constructing the ``Lambda`` directly kept only the operand correspondence and dropped every
    # axis the caller had supplied. ``closing`` re-appends whatever the canonical body still reads
    # on top of it, so the declaration survives the canonicalization rather than being reset by it.
    bound = tuple(name for edge in ordered for name in _operand_result_names(edge))
    scope = tuple(axis for axis in axes if axis != fold.axis.name and axis not in bound)
    lift = Lambda.closing((fold.axis.name, *bound, *scope), Body(form.products), fold.lift.results)
    return replace(fold, operands=tuple(ordered), lift=lift)


def _normalize_body(body: Body, axes: tuple[str, ...], implicit_axes: frozenset[str], sweep_axes: frozenset[str]) -> Body:
    out = []
    for stmt in body:
        if isinstance(stmt, Fold):
            out.append(_normalize_fold(stmt, axes, implicit_axes, sweep_axes))
            continue
        nested = stmt.nested()
        if not nested:
            out.append(stmt)
            continue
        child_axes = (*axes, *stmt.binds_axes())
        out.append(stmt.with_bodies(tuple(_normalize_body(child, child_axes, implicit_axes, sweep_axes) for child in nested)))
    return Body(out)


def _passthrough(node: Fold) -> Fold | None:
    """The single operand an identity projection merely re-exposes, or ``None``.

    A pass-through is shape noise — a closing rewrite can leave one behind — and it is what makes
    two occurrences of the same computation compare unequal, so normalization dissolves it
    wherever a projection is formed or revisited."""
    if node.axis is not None or node.lift.body or len(node.operands) != 1:
        return None
    (operand,) = node.operands
    if isinstance(operand, Fold) and tuple(node.lift.results) == _operand_result_names(operand):
        return operand
    return None


def _flat_members(edge) -> tuple | None:
    """A zero-axis operand edge's statements, when it is a plain scalar wrapper.

    A hoisted factor becomes ordinary epilogue statements rather than a projection root: an
    output-tiled root forest must own a boundary store, so a factor edge left as an operand would
    break the ownership rule the split and kernel-binding passes share.
    """
    if not isinstance(edge, Fold) or edge.axis is not None or edge.operands or len(edge.lift.results) != 1:
        return None
    members = tuple(edge.lift.body)
    if any(not isinstance(stmt, (Load, Assign)) for stmt in members):
        return None
    return members if operand_name(edge) in {name for stmt in members for name in stmt.defines()} else None


def _decode_split(edge, axis_name: str):
    """Split a computed operand cone into its raw storage load and its fold-invariant factors.

    The shape is a STORAGE DECODE (the ``ElementwiseImpl.decodes`` trait, never an op-name list)
    times factors constant along the fold axis. The decode is absorbed by the raw load's storage
    dtype — every consumer converts a bits-carrier element by dtype, the mma fragment loaders
    included — and the invariant factors commute out onto the accumulator:
    ``Sum_k a*(s*w) = s*Sum_k a*w``, the same reassociation category as split-K.

    Returns ``None`` unless the residue really is a decode of one raw load, so an ordinary
    floating-point factor chain keeps its computed-cone form and this never reassociates
    arithmetic a storage decode did not already introduce.
    """
    if not isinstance(edge, Fold) or edge.axis is not None or len(edge.lift.results) != 1:
        return None
    body = edge.lift.body
    if any(not isinstance(stmt, (Load, Assign)) for stmt in body):
        return None
    by_param = {operand_name(operand): operand for operand in edge.operands}
    result = operand_name(edge)
    flattened = product_spine(body.definitions, result, divide=True)
    if flattened is None:  # a bare decode: nothing to hoist, but the decode is still absorbed
        leaves, spine = (result,), ()
    else:
        leaves, spine = flattened

    def varies(leaf: str) -> bool:
        if leaf in by_param:
            return axis_name in edge_axes(by_param[leaf], (axis_name,))
        return any(refs_axis(stmt, axis_name) for stmt in body.backward_cone((leaf,)).members)

    varying = [leaf for leaf in leaves if varies(leaf)]
    if len(varying) != 1 or varying[0] in by_param:
        return None
    invariant = [leaf for leaf in leaves if leaf not in varying]

    cone = body.backward_cone((varying[0],)).members
    if len(cone) != 2 or not isinstance(cone[0], Load) or not isinstance(cone[1], Assign):
        return None
    raw, decode = cone
    # The DECODE OP names the storage dtype; the load's own ``dtype`` is stamped later, in Kernel
    # IR, so it is only a consistency check here and never the authority.
    if decode.op.decodes is None or decode.args != (raw.name,):
        return None
    if raw.dtype is not None and raw.dtype.name != decode.op.decodes:
        return None

    seen = {id(stmt) for stmt in cone} | {id(stmt) for stmt in spine}
    members: tuple = ()
    used_params = 0
    for leaf in invariant:
        if leaf in by_param:
            flat = _flat_members(by_param[leaf])
            if flat is None:
                return None
            members += flat
            used_params += 1
            continue
        # A leaf this cone neither defines nor takes as an operand is an ENCLOSING-scope name.
        # It needs no statements: the epilogue lands in that same scope, so the name stays bound.
        for stmt in body.backward_cone((leaf,)).members:
            if id(stmt) not in seen:
                seen.add(id(stmt))
                members += (stmt,)
    if any(id(stmt) not in seen for stmt in body) or used_params != len(edge.operands):
        return None  # a statement or operand the split does not account for — keep the cone whole
    return raw, members, spine, varying[0], result


def _renamed(stmt, mapping: dict[str, str]):
    """Rename one pure scalar statement's definition and its read names."""
    name = mapping.get(stmt.name, stmt.name)
    if isinstance(stmt, Load):
        return replace(stmt, names=(name,))
    return replace(stmt, name=name, args=tuple(mapping.get(arg, arg) for arg in stmt.args))


def _apply_spine(split, carried: str, out: str, taken: set[str]) -> tuple[tuple, str]:
    """Emit one hoisted factor chain, threading ``carried`` through it and defining ``out``."""
    _raw, members, spine, varying, result = split
    mapping = {varying: carried, result: out}
    emitted: list = []
    for stmt in (*members, *spine):
        if stmt.name != result:
            fresh = stmt.name if stmt.name not in taken else f"{stmt.name}__h{len(taken)}"
            mapping[stmt.name] = fresh
            taken.add(fresh)
        emitted.append(_renamed(stmt, mapping))
    taken.add(out)
    return tuple(emitted), out


def _hoist_decode_factors(node: Fold, taken: set[str]):
    """Bind a quantized contraction's operands as raw storage loads with the factors on the
    epilogue. Returns ``(contraction, epilogue statements)`` or ``None``."""
    if not is_contraction(node) or node.semiring is None:
        return None
    axis_name = node.axis.name
    a_split = _decode_split(node.a, axis_name)
    b_splits = [_decode_split(channel.b, axis_name) for channel in node.channels]
    if a_split is None and not any(b_splits):
        return None
    if a_split is None and not all(b_splits):
        return None  # homogeneous channels only: a half-bound B pair has no single slab dtype

    def hoists(split) -> bool:
        return split is not None and bool(split[1] or split[2])

    product, plus = node.semiring
    accs = tuple(channel.acc for channel in node.channels)
    any_hoist = hoists(a_split) or any(hoists(split) for split in b_splits)
    renamed = tuple(f"{acc}__mh" for acc in accs) if any_hoist else accs

    epilogue: list = []
    for index, (acc, out) in enumerate(zip(renamed, accs, strict=True)):
        chain = [split for split in (a_split, b_splits[index]) if hoists(split)]
        carried = acc
        for position, split in enumerate(chain):
            target = out if position == len(chain) - 1 else f"{out}__mh{position}"
            stmts, carried = _apply_spine(split, carried, target, taken)
            epilogue.extend(stmts)

    rebuilt = Fold.contraction(
        k_axis=node.axis,
        a=a_split[0] if a_split is not None else node.a,
        channels=tuple(
            Channel(b=split[0] if split is not None else channel.b, acc=acc)
            for channel, split, acc in zip(node.channels, b_splits, renamed, strict=True)
        ),
        product=product,
        fold_op=plus,
    )
    return replace(rebuilt, unroll=node.unroll), tuple(epilogue)


def _hoist_decode_root(node: Fold) -> Fold:
    """Hoist a contraction reached with no projection wrapper, creating the one it needs.

    A split piece, or a root whose projection already collapsed, has nowhere to put the factors.
    The wrapper defines exactly the names the contraction defined, so consumers are unaffected."""
    hoisted = _hoist_decode_factors(node, set(node.defines()))
    if hoisted is None:
        return node
    rebuilt, epilogue = hoisted
    if not epilogue:
        return rebuilt
    return Fold.projection(operands=(rebuilt,), body=Body(epilogue), results=node.defines())


def _hoist_decode_operands(root: Fold) -> Fold:
    """Apply the storage-decode hoist to every contraction stored in a projection's body."""
    if root.axis is not None:
        return root
    taken = {name for stmt in root.lift.body for name in stmt.defines()}
    taken.update(operand_name(operand) for operand in root.operands)
    members: list = []
    changed = False
    for stmt in root.lift.body:
        hoisted = _hoist_decode_factors(stmt, taken) if isinstance(stmt, Fold) else None
        if hoisted is None:
            members.append(stmt)
            continue
        rebuilt, epilogue = hoisted
        members.append(rebuilt)
        members.extend(epilogue)
        changed = True
    if not changed:
        return root
    return Fold.projection(operands=root.operands, body=Body(tuple(members)), results=root.lift.results)


def _normalize_fold(fold: Fold, axes: tuple[str, ...], implicit_axes: frozenset[str], sweep_axes: frozenset[str]) -> Fold:
    operands = tuple(_normalize_fold(edge, axes, implicit_axes, sweep_axes) if isinstance(edge, Fold) else edge for edge in fold.operands)
    node = replace(fold, operands=operands) if operands != fold.operands else fold
    if node.axis is None and (collapsed := _passthrough(node)) is not None:
        return collapsed
    body_axes = (*axes, node.axis.name) if node.axis is not None else axes
    body = _normalize_body(node.lift.body, body_axes, implicit_axes, sweep_axes)
    if body != node.lift.body:
        node = node.with_bodies((body,))
    node = _canonical_semiring(node, axes, implicit_axes)
    if node.axis is not None:
        return node
    node = _hoist_decode_operands(node)
    return node


def _share_common_cones(root: Fold) -> Fold:
    """Restore object sharing between same-value cones — the tree-wide half of canonicalization.

    Fusion and the close rewrites inline one value into every consumption site, so a traced value
    consumed twice (attention's softmax statistics, read by the weight cone and the epilogue)
    reappears as equal-but-distinct copies. Everything downstream keys on object identity —
    ``cuttable_seams`` groups occurrences by it, ``realize`` replaces cut values by it — so a
    severed sharing silently turns one value into per-site recompute that no schedule can undo.
    This walk hash-conses every Fold bottom-up: copies UNIFY onto the first occurrence in walk
    order when they are alpha-equal with identical captures (the bucket key adds ``deps`` — the
    K-cone family, alpha-equal under DIFFERENT captures, stays value clustering's job) and
    identical interface names (``defines`` — what sibling members and the consuming lift read),
    so a copy that differs only in internal binder spelling still collapses where plain
    structural equality would silently sever the sharing. Emission is untouched in shape:
    lowering walks tree positions, and every position holds a term of the same value (a unified
    representative may change internal spelling). Identity-preserving off the replacement spine,
    like ``_replace_fold``."""
    canon: dict[tuple, list[Fold]] = {}
    seen: dict[int, Fold] = {}
    unify_keys: dict[int, Lambda] = {}

    def unify_key(fold: Fold) -> Lambda:
        # The whole-term alpha-quotient under an empty environment: free names (the captures) and
        # the bucket-pinned interface names make canonical equality mean equal VALUE.
        if id(fold) not in unify_keys:
            unify_keys[id(fold)] = term_key(fold)
        return unify_keys[id(fold)]

    def member(stmt):
        if isinstance(stmt, Fold):
            return visit(stmt)
        nested = stmt.nested()
        if not nested:
            return stmt
        bodies = tuple(Body(tuple(member(child) for child in body)) for body in nested)
        unchanged = all(
            len(body) == len(original) and all(piece is child for piece, child in zip(body, original, strict=True))
            for body, original in zip(bodies, nested, strict=True)
        )
        return stmt if unchanged else stmt.with_bodies(bodies)

    def visit(node: Fold) -> Fold:
        if id(node) in seen:
            return seen[id(node)]
        operands = tuple(visit(edge) if isinstance(edge, Fold) else edge for edge in node.operands)
        body = tuple(member(stmt) for stmt in node.lift.body)
        current = node
        if any(piece is not edge for piece, edge in zip(operands, node.operands, strict=True)):
            current = replace(current, operands=operands)
        if any(piece is not stmt for piece, stmt in zip(body, node.lift.body, strict=True)):
            current = current.with_bodies((Body(body),))
        bucket = canon.setdefault((current.structural_key(), current.defines()), [])
        for prior in bucket:
            if prior == current or unify_key(prior) == unify_key(current):
                seen[id(node)] = prior
                return prior
        bucket.append(current)
        seen[id(node)] = current
        return current

    return visit(root)


def normalize_fold_tree(root, axes: Iterable[str] = (), implicit_axes: Iterable[str] = (), sweep_axes: Iterable[str] = ()):
    """Normalize a complete Tile IR tree bottom-up; ``None`` placeholders pass through.

    ``sweep_axes`` names the axes bound only by output-sweep reconstitution (never kernel scope);
    :func:`_hoist_closed_folds` keeps a fold reading one as a projection body member.

    The reached fixpoint is STAMPED on the result (per enclosing scope, an
    :func:`~emmy.compiler.structural.instance_memo`): the term is immutable and the rewrite
    idempotent, so a reconstruction under the same scope answers without re-walking — on a large
    fused tree the re-verification, once per ``TileOp`` construction, is what turns the pipeline
    quadratic."""
    if not isinstance(root, Fold):
        return root
    scope = (tuple(axes), frozenset(implicit_axes), frozenset(sweep_axes))
    if scope in instance_memo(root, "_memo_normal_scopes"):
        return root
    normalized = root
    while True:
        # One pass is not always the fixpoint (a close can expose the next pass's move), and the
        # stamp must mean the REACHED fixpoint, so iterate here rather than relying on the next
        # construction to finish the job.
        again = _normalize_fold(normalized, scope[0], scope[1], scope[2])
        # A contraction reached as the whole tree has no projection to host hoisted factors, so
        # the hoist creates one here. Nested contractions are handled by their own projection,
        # which keeps the common shape flat.
        if again.axis is not None:
            again = _hoist_decode_root(again)
        if again == normalized:
            break
        normalized = again
    result = root if normalized == root else normalized
    result = _share_common_cones(result)
    instance_memo(result, "_memo_normal_scopes")[scope] = True
    return result


__all__ = [
    "normalize_fold_tree",
]
