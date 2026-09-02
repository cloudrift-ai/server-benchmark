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
from dataclasses import replace

from emmy.compiler.ir.expr import affine_form
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.pure.algebra import product_spine
from emmy.compiler.ir.stmt import Assign, Body, Load
from emmy.compiler.ir.stmt.body import refs_axis
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
    if isinstance(operand, Fold) and tuple(node.lift.results) == operand.exposes:
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
    return members if edge.exposes[-1] in {name for stmt in members for name in stmt.defines()} else None


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
    by_param = {operand.exposes[-1]: operand for operand in edge.operands}
    result = edge.exposes[-1]
    flattened = product_spine(body.definitions, result, divide=True)
    if flattened is None:  # a bare decode: nothing to hoist, but the decode is still absorbed
        leaves, spine = (result,), ()
    else:
        leaves, spine = flattened

    def varies(leaf: str) -> bool:
        if leaf in by_param:
            return axis_name in (frozenset((axis_name,)) & by_param[leaf].index_space)
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


def _a_leads(node: Fold) -> Fold:
    """Put the contraction's A operand first — ``operands[0]`` IS A, by canonical form.

    A is the operand every product multiplies: with several channels (the fused gate⊗up edge) it
    is the argument their products SHARE, and that names it outright. With a single product both
    arguments are trivially shared, so the layout rule decides: A reads ``[…, k]``, its reduction
    axis last, which is what lets a fragment load stride A's rows contiguously. B carries no such
    constraint — stored ``[k, n]`` or transposed ``[n, k]``, both legal, the atom reads it off the
    edge.

    Binding is positional, so the lift's params move with the operands; the body reads by name and
    is untouched.
    """
    if node.axis is None or len(node.operands) < 2 or any(not isinstance(stmt, Assign) or len(stmt.args) != 2 for stmt in node.lift.body):
        return node
    by_name = {name: edge for edge in node.operands for name in edge.exposes}
    argument_sets = [set(product.args) for product in node.lift.body]
    shared = set.intersection(*argument_sets)

    if len(argument_sets) > 1:
        candidates = [by_name[name] for name in shared if name in by_name]
        if len(candidates) != 1:
            return node
        a_edge = candidates[0]
    else:
        pair = [by_name[name] for name in argument_sets[0] if name in by_name]
        if len(pair) != 2:
            return node

        def k_last(edge: Fold) -> bool:
            # Only a slab has a gmem index to read a layout off; a computed cone answers False.
            return edge.as_slab() is not None and node.axis.name in edge.loads[0].index[-1].free_vars()

        a_edge = next((edge for edge in pair if k_last(edge)), None)
        if a_edge is None:
            return node

    if node.operands[0] is a_edge:
        return node
    reordered = (a_edge, *(edge for edge in node.operands if edge is not a_edge))
    params = (node.axis.name, *(name for edge in reordered for name in edge.exposes))
    candidate = replace(node, operands=reordered, lift=replace(node.lift, params=params))
    return candidate if candidate.as_contraction() is not None else node


def _normalize_fold(fold: Fold, axes: tuple[str, ...], implicit_axes: frozenset[str], sweep_axes: frozenset[str]) -> Fold:
    operands = tuple(_normalize_fold(edge, axes, implicit_axes, sweep_axes) if isinstance(edge, Fold) else edge for edge in fold.operands)
    node = replace(fold, operands=operands) if operands != fold.operands else fold
    node = _a_leads(node)
    if node.axis is None and (collapsed := _passthrough(node)) is not None:
        return collapsed
    body_axes = (*axes, node.axis.name) if node.axis is not None else axes
    body = _normalize_body(node.lift.body, body_axes, implicit_axes, sweep_axes)
    if body != node.lift.body:
        node = replace(node, lift=replace(node.lift, body=Body(body)))
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
    canon: dict[tuple, Fold] = {}
    seen: dict[int, Fold] = {}

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
            current = replace(current, lift=replace(current.lift, body=Body(body)))
        # The canonical form IS the key: a dict lookup and an alpha-equality test are the same
        # operation, so there is no prefilter bucket and no pairwise rescan. The exposed names
        # ride beside it: a consumer's lift reads an edge's results BY NAME, so two values that
        # differ only in what they expose stay distinct — unifying them would re-spell every
        # consumer of the copy (softmax's two reads of one row, spelled ``in0`` and ``in1``).
        prior = canon.setdefault((current.canonical(), current.exposes), current)
        seen[id(node)] = prior
        return prior

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
