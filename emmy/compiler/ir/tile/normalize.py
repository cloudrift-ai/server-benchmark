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
from emmy.compiler.ir.pure import Fold, Lambda
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


def _edge_free_names(edge) -> frozenset[str]:
    """The names an operand edge needs SUPPLIED — :meth:`Fold.deps`, the declared roll-up of its
    own environment and its edges'.

    Returning an empty set here (on the grounds that "a term captures nothing") is what made the
    closing rewrites below inert: `_provider_needs` went empty, every `provider()` returned None,
    and the drain stopped draining. A term declaring a name is exactly what the drain exists to
    resolve — the declaration says WHAT is needed, and closing supplies it through an operand."""
    return frozenset(edge.deps()) if isinstance(edge, Fold) else frozenset()


def _carries_iteration(node) -> bool:
    """Whether a provider chain contains a Fold axis rather than only straight-line code."""
    if getattr(node, "axis", None) is not None:
        return True
    children = (*node.operands, *node.lift.body) if isinstance(node, Fold) else tuple(stmt for body in node.nested() for stmt in body)
    return any(_carries_iteration(child) for child in children)


def _with_source(node: Fold, source) -> Fold:
    """``node`` with ``source`` APPENDED to its operands and its lift rebound positionally.

    Appended, never prepended: the path codec spells a site by its operand role and index, so
    inserting ahead of the existing edges renumbers every child path and every stored ``PLACE@``
    pin addressing them stops resolving. Appending leaves the existing indices alone and gives the
    drained producer the next one. Evaluation order is not tuple order —
    :func:`~emmy.compiler.ir.pure.fold.splice_operands` places each edge's body before its first
    read, providers ahead of dependents — so the producer still lands before the value it feeds.

    A zero-axis node rebuilds as a fresh term over the same operands; a
    reducing node keeps its iteration var first and rebinds the rest, so the formation invariant —
    one lift param per operand result component — holds at either position."""
    operands = (*node.operands, source)
    bound = tuple(name for edge in operands for name in edge.exposes)
    if node.axis is None:
        bound = tuple(name for edge in operands for name in edge.exposes)
    return Fold(operands=operands, lift=Lambda.closing(bound, Body.coerce(node.lift.body), node.lift.results))
    return replace(node, operands=operands, lift=replace(node.lift, params=(node.axis.name, *bound)))


def _close_tree(root: Fold, provider) -> tuple[tuple, Body]:
    """The shared walk of both closing rewrites: establish the closure invariant on EVERY node.

    A computed zero-axis operand may still carry value captures
    (:func:`~emmy.compiler.ir.pure.closure.value_captures` — sibling-defined data rather
    than axes); ``provider(edge, binders)`` returns the source edge that supplies them, or
    ``None`` to leave the operand open. ``binders`` counts only the iteration domains crossed
    BELOW ``root`` — a reducing root already evaluates inside its own axis, and a projection root
    has none — and each caller owns what a provider may take and what happens to the drained
    chain afterwards.

    The walk does not ask what KIND of node it stands on. The closure invariant is a property of
    the term — ``ir/pure/closure``: "a normalized term's values arrive through operand edges, and
    the only names its lift may capture are axes bound by its ancestors" — so a projection's or a
    twisted reduce's operand edge is closed by the same rule that closes a contraction's. The
    drained edge carries its PRODUCER, never a reference to the enclosing scope: an operand that
    named an outer value would be a capture with positional spelling, and the term would still not
    be evaluable from its own parts. Sharing is restored by :func:`_share_common_cones`, and a
    value that now has two producers becomes ONE seam at placement — the module-header rule.
    """

    def close(node: Fold, binders: tuple[str, ...] = ()) -> Fold:
        inner = (*binders, node.axis.name) if node.axis is not None else binders
        operands = tuple(close(edge, inner) if isinstance(edge, Fold) else edge for edge in node.operands)
        body = Body(close(stmt, inner) if isinstance(stmt, Fold) else stmt for stmt in node.lift.body)
        current = replace(node, operands=operands) if operands != node.operands else node
        if body != current.lift.body:
            current = replace(current, lift=replace(current.lift, body=Body(body)))

        changed = False
        closed = []
        for edge in current.operands:
            if not isinstance(edge, Fold) or edge.axis is not None:
                closed.append(edge)
                continue
            source = provider(edge, binders)
            if source is None:
                closed.append(edge)
                continue
            closed.append(_with_source(edge, source))
            changed = True
        current = replace(current, operands=tuple(closed)) if changed else current

        # A fold sitting as a BODY MEMBER captures exactly as an operand edge does — attention's
        # twisted per-key statistic reads the scale and the row maximum its siblings define. It
        # takes them the same way: as operands carrying their producer, at whichever position it
        # sits. Its own axis stays the lift's leading binder, so the drain is one move either way.
        members = []
        for stmt in current.lift.body:
            source = provider(stmt, binders) if isinstance(stmt, Fold) else None
            members.append(stmt if source is None else _with_source(stmt, source))
        if any(fresh is not prior for fresh, prior in zip(members, current.lift.body, strict=True)):
            current = replace(current, lift=replace(current.lift, body=Body(members)))
        return current

    rewritten_operands = tuple(close(edge) if isinstance(edge, Fold) else edge for edge in root.operands)
    rewritten_body = Body(close(stmt) if isinstance(stmt, Fold) else stmt for stmt in root.lift.body)
    return rewritten_operands, rewritten_body


def _provider_needs(edge, provider_order: tuple[str, ...], provider_names: frozenset[str]) -> tuple[str, ...]:
    """The provider names an operand edge captures, in provider order."""
    return tuple(name for name in provider_order if name in (_edge_free_names(edge) & provider_names))


def _close_reduce_body(root: Fold, axes: tuple[str, ...], sweep_axes: frozenset[str]) -> Fold:
    """Move a reducing fold's body-resident producer chain onto a captured contraction operand.

    :func:`_close_projection` closes contraction operands against a zero-axis root, but a chain
    that depends on the fold's own iteration axis lives in the reducing fold's lift body, one
    scope below any projection — attention's per-key statistic and its rsqrt ahead of the score
    dot's B cone. Moving that chain onto the captured edge closes the cone at the contraction's
    fold axis plus the enclosing free axes, which is what lets the placement fork offer the
    operand as a workspace seam. The move is gated on exclusive consumption: every moved
    definition must die into the closed edges, so the step's work is repackaged rather than
    duplicated and a chain a sibling member still reads stays put."""
    if root.axis is None or root.as_contraction() is not None or not root.lift.body:
        return root
    body = root.lift.body
    provider_order = tuple(dict.fromkeys(name for stmt in body for name in stmt.defines()))
    provider_names = frozenset(provider_order)
    moved: set[int] = set()

    def provider(edge, binders: tuple[str, ...]) -> Fold | None:
        names = _provider_needs(edge, provider_order, provider_names)
        if not names:
            return None
        cone = body.backward_cone(names)
        defined = {name for stmt in cone.members for name in stmt.defines()}
        if not set(names) <= defined:
            return None
        if binders and any(_carries_iteration(member) for member in cone.members):
            return None
        moved.update(id(stmt) for stmt in cone.members)
        return Fold(lift=Lambda.closing((), Body(cone.members), names))

    rewritten_operands, rewritten_body = _close_tree(root, provider)
    if not moved:
        return root

    kept = Body(stmt for stmt in rewritten_body if id(stmt) not in moved)
    moved_defs = {name for stmt in body if id(stmt) in moved for name in stmt.defines()}
    outside = {result for result in root.lift.results if isinstance(result, str)}
    outside.update(name for stmt in kept for name in Body((stmt,)).ssa_uses)
    if root.observe is not None:
        outside.update(root.observe.params)  # closed: its reads ARE its params (axis + carried state)
    if moved_defs & outside:
        return root  # the chain does not die into the closed edges — moving it would duplicate work
    return replace(root, operands=rewritten_operands, lift=replace(root.lift, body=kept))


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
    ring = node._semiring
    if ring is None or len(node.operands) < 2:
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
            return edge.is_slab and node.axis.name in edge.loads[0].index[-1].free_vars()

        a_edge = next((edge for edge in pair if k_last(edge)), None)
        if a_edge is None:
            return node

    if node.operands[0] is a_edge:
        return node
    reordered = (a_edge, *(edge for edge in node.operands if edge is not a_edge))
    lead = (node.axis.name,) if node.axis is not None else ()
    params = (*lead, *(name for edge in reordered for name in edge.exposes))
    return replace(node, operands=reordered, lift=replace(node.lift, params=params))


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
    # _canonical_semiring deleted
    if node.axis is not None:
        return _close_reduce_body(node, body_axes, sweep_axes)
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
    canon: dict[Fold, Fold] = {}
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
        # The TERM is the key: a dict lookup and the equality test are one operation, so there is
        # no prefilter bucket and no pairwise rescan. Structural equality only — an α-quotient
        # would need a canonical form, and a canonical form is not available while an edge's
        # result names are still the spelling its consumer binds. This merges strictly less, which
        # is the safe direction: it leaves copies distinct, it never merges distinct values.
        prior = canon.setdefault(current, current)
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
