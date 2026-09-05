"""Body-level normalization passes.

Pure ``body → body`` transforms applied via :func:`normalize_body` from
``LoopOp.__post_init__`` and from :meth:`Body.structural_key`, so a
constructed Loop-IR Op and every identity digest land in canonical form. The
passes operate on the shared Stmt vocabulary (``Loop``, ``Load``, ``Assign``,
``Accum``, ``Select``, ``Write``) and recurse through every block-structured
Stmt (``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond``).

A ``TileOp`` does NOT run these: it normalizes its TERM
(``normalize_fold_tree``), and the kernel it materializes is built straight
from ``Fold.lower``. So a body these passes could improve reaches the emitter
unchanged whenever it comes down the term path — the sibling-loop merge below
is reachable from Loop IR and from the digest, not from a materialized
``KernelOp``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from itertools import count

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Expr, Literal, SimplifyCtx, Var, affine_form
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.body import Body, _exposed_defines, free_names
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Load, Mma, Pack, Select, Unpack, Write

# ---------------------------------------------------------------------------
# Visitor helpers shared by every pass below
# ---------------------------------------------------------------------------


def _identity_rename(n: str) -> str:
    return n


def _make_axis_renamer(old: str, new: Axis) -> Callable[[Axis], Axis]:
    return lambda a: new if a.name == old else a


def normalize_body(
    stmts: Body,
    *,
    hoist: bool = True,
    canonical_buffers: bool = False,
    cluster_ops: bool = False,
) -> Body:
    """Apply the structural and cosmetic normalization passes in order.

    Used by both ``LoopOp.__post_init__`` and ``TileOp.__post_init__`` so
    bodies — Loop-IR and Tile-IR — land in a canonical shape before
    validation.

    ``hoist=False`` skips :func:`hoist_loop_invariants`. TileOp bodies turn
    it off because a Stage binding is scoped to the Loop where it's
    declared — hoisting Loads that read from a staged buffer above the
    Stage decl would leave the read referencing an undeclared name.

    ``canonical_buffers=True`` runs :func:`canonicalize_buffer_names` after
    SSA renaming. Off by default — buffer names bind to graph inputs /
    outputs and are meaningful at the Op boundary. Turned on by
    :attr:`Body.structural_key()` so two bodies that read identical patterns
    from differently-named buffers hash and compare equal.

    ``cluster_ops=True`` runs :func:`canonicalize_op_clusters` after
    buffer renaming. Off by default — collapsing ``sub`` to ``add`` (or
    ``mod`` to ``divide``) destroys semantics, so this is only safe
    when the output is a hash key, never a runnable body. Turned on by
    :attr:`Body.structural_key()` so two bodies that differ only in the
    *kind* of FMA / compare / SFU op at the same position hash equal.
    """
    stmts = Body.coerce(stmts)
    stmts = topo_sort_siblings(stmts)
    stmts = drop_size_one_free_axes(stmts)
    stmts = drop_size_one_reduce_axes(stmts)
    stmts = canonicalize_free_axis_order(stmts)
    stmts = eliminate_copy_aliases(stmts)
    stmts = unify_sibling_reduce_axes(stmts)
    stmts = merge_sibling_reduce_loops(stmts)
    if hoist:
        stmts = split_invariant_divides(stmts)
        stmts = hoist_loop_invariants(stmts)
    stmts = simplify_body(stmts)
    stmts = dedup_loads(stmts)
    stmts = rename_ssa_sequential(stmts)
    if canonical_buffers:
        stmts = canonicalize_buffer_names(stmts)
    if cluster_ops:
        stmts = canonicalize_op_clusters(stmts)
    # Sort runs last so the keys it sorts by are the post-rename canonical
    # SSA / buffer names — that way two bodies that differ only in original
    # argument order produce identical post-normalization arg tuples.
    stmts = sort_commutative_args(stmts)
    return stmts


# ---------------------------------------------------------------------------
# Pass 1: drop size-1 free axes
# ---------------------------------------------------------------------------


def drop_size_one_free_axes(stmts: Body) -> Body:
    """Inline every free ``Loop(axis, extent=1)``: replace it with its body
    after substituting ``Var(axis.name) → Literal(0, "int")``. Reduce Loops
    keep their wrappers because dropping them would remove the accumulator.
    Recurses through StridedLoop / Tile / Cond bodies without rewriting
    those wrappers (their iteration semantics aren't a free Loop).

    Size-1 BLOCK / SPLITK_BLOCK protection used to live here when the
    planner stamped ``Loop.role`` for downstream launch_geometry to
    consume. The planner now constructs ``GridTile`` / ``ThreadTile``
    directly and applies its own size-1 filter (see
    ``010_partition_loops::_wrap_tower``), so by the time
    ``drop_size_one_free_axes`` runs on a LoopOp body, no Loop has any
    binding role — every size-1 free Loop is safely inlinable.
    """
    stmts = Body.coerce(stmts)

    def fn(s: Stmt) -> Stmt | Body:
        # Body.map post-order: ``s.body`` is already recursively mapped.
        if isinstance(s, Loop) and s.axis.extent.is_static and s.axis.extent.as_static() == 1 and not s.is_reduce:
            sub = Sigma({s.axis.name: Literal(0, "int")})
            return tuple(c.substitute(sub) for c in s.body)
        return s

    return stmts.map(fn)


def drop_size_one_reduce_axes(stmts: Body) -> Body:
    """Inline a canonical extent-one reduction as its single update.

    Fusion can hoist a singleton reduction's value into the enclosing scope (decode softmax is
    the common case).  Keeping the reduction wrapper then asks Tile IR to form a fold whose lift
    returns that enclosing value without defining it locally.  An extent-one fold is just one
    application of its monoid, so replace each distinct accumulator with an ordinary pure
    assignment before copy-alias elimination rewires the result.

    Only the canonical one-update form is collapsed.  Scans, nested effects, and repeated updates
    to one accumulator keep their loop because their sequential state is not an alias.
    """
    stmts = Body.coerce(stmts)

    def fn(stmt: Stmt) -> Stmt | Body:
        if not (isinstance(stmt, Loop) and stmt.is_reduce and stmt.axis.extent.is_static and stmt.axis.extent.as_static() == 1):
            return stmt
        accums = tuple(member for member in stmt.body if isinstance(member, Accum))
        if (
            not accums
            or len({accum.name for accum in accums}) != len(accums)
            or any(not (member.pure or isinstance(member, Accum)) for member in stmt.body)
        ):
            return stmt

        sub = Sigma({stmt.axis.name: Literal(0, "int")})
        out: list[Stmt] = []
        for member in stmt.body:
            member = member.substitute(sub)
            if not isinstance(member, Accum):
                out.append(member)
                continue
            if member.base is None or member.base == member.name:
                if not member.has_identity:
                    return stmt
                out.append(Assign(name=member.name, op="copy", args=(member.value,), dtype=member.dtype))
            else:
                out.append(Assign(name=member.name, op=member.op, args=(member.base, member.value), dtype=member.dtype))
        return Body(out)

    return stmts.map(fn)


# ---------------------------------------------------------------------------
# Pass 2: canonical free-axis ordering
# ---------------------------------------------------------------------------


def _recurse_canonicalize(s: Stmt) -> Stmt:
    nested = s.nested()
    if not nested:
        return s
    return s.with_bodies(tuple(canonicalize_free_axis_order(b) for b in nested))


def _output_storage_depth(stmts: Body, axis: str) -> int | None:
    """The row-major output-coordinate depth of one unit-affine free axis."""
    depths = []
    for write in stmts.iter_of_type(Write):
        positions = []
        for position, expr in enumerate(write.index):
            form = affine_form(expr, {axis})
            if form is None:
                return None
            coefficient = form[1].get(axis, 0)
            if coefficient:
                if coefficient != 1:
                    return None
                positions.append(position)
        if len(positions) > 1:
            return None
        if positions:
            depths.append(len(write.index) - positions[0] - 1)
    return depths[0] if depths and len(set(depths)) == 1 else None


def canonicalize_free_axis_order(stmts: Body) -> Body:
    """Sort an outer free-loop chain by row-major output storage order.

    Boundary writes provide the canonical geometry: larger coordinate depth is outer, so the
    innermost loop follows the output's contiguous dimension. If the writes do not totally order
    the chain, axis names provide the deterministic fallback. Recursion continues into terminal
    block bodies (Loop / StridedLoop / Tile / Cond).
    """
    stmts = Body.coerce(stmts)
    chain: list[Loop] = []
    current = stmts
    while len(current) == 1 and isinstance(current[0], Loop):
        loop = current[0]
        if loop.is_reduce:
            break
        chain.append(loop)
        current = loop.body

    terminal = tuple(_recurse_canonicalize(s) for s in current)

    depths = [_output_storage_depth(Body(terminal), loop.axis.name) for loop in chain]
    if all(depth is not None for depth in depths) and len(set(depths)) == len(depths):
        chain_sorted = [loop for _, loop in sorted(zip(depths, chain, strict=True), key=lambda item: -item[0])]
    else:
        chain_sorted = sorted(chain, key=lambda lp: lp.axis.name)
    result: Body = terminal
    for loop in reversed(chain_sorted):
        result = (Loop(axis=loop.axis, body=result, unroll=loop.unroll),)
    return result


# ---------------------------------------------------------------------------
# Pass 3: eliminate `y = copy(x)` identity aliases
# ---------------------------------------------------------------------------


def eliminate_copy_aliases(stmts: Body) -> Body:
    """Collapse ``y = copy(x)`` Assigns. The merge rule plants identity
    copies as bridges between producer writes and consumer reads; a long
    chain stacks them. Every such Assign is dropped and downstream
    references to ``y`` are rewired to the alias root. Pure IR hygiene."""
    stmts = Body.coerce(stmts)
    alias: dict[str, str] = {}

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    def fn(s: Stmt) -> Stmt | None:
        # Body.map post-order: block bodies already recursed; only handle leaves.
        if isinstance(s, (Loop, StridedLoop, Cond)):
            return s
        if isinstance(s, Assign) and s.op.name == "copy" and len(s.args) == 1 and s.dtype is None:
            alias[s.name] = s.args[0]
            return None
        return s.rewrite(resolve)

    return stmts.map(fn)


# ---------------------------------------------------------------------------
# Pass 4: unify sibling reduce-loop axis names
# ---------------------------------------------------------------------------


def unify_sibling_reduce_axes(stmts: Body) -> Body:
    """At every scope, find sibling reduce ``Loop``s whose reduce axes
    index overlapping ``(Load.source, dim)`` positions and rename them
    to a single canonical axis name. Recurses through every block-
    structured Stmt (Loop / StridedLoop / Tile / Cond) to find nested
    scopes."""
    stmts = Body.coerce(stmts)

    def walk(body: Body) -> Body:
        # Recurse into nested bodies first (post-order) via the canonical
        # nested() / with_bodies() descent, then group siblings at this
        # scope. Splitting the recursion from the sibling-grouping keeps
        # this pass's scope-level logic isolated in ``_unify_siblings``.
        recursed: list[Stmt] = []
        for s in body:
            nested = s.nested()
            if nested:
                recursed.append(s.with_bodies(tuple(walk(b) for b in nested)))
            else:
                recursed.append(s)
        return _unify_siblings(Body(recursed))

    return walk(stmts)


def _unify_siblings(body: Body) -> Body:
    """Single-scope sibling grouping: rename reduce-axis vars across
    sibling reduce Loops whose bare-Var Load positions overlap on any
    ``(source, dim)`` pair so they share one canonical axis name.

    Two reduce Loops that bind different axis names but both index the
    same input slot (e.g. ``x[..., a2]`` and ``x[..., a3]`` for the
    same ``x``) are semantically the same reduction dimension. Union-
    find on the overlap relation merges all transitively-connected
    Loops into one group. Within a group, the first Loop's axis name
    wins; later Loops are rewritten to use it.

    Pairing on overlap rather than exact-set equality lets matmul-
    siblings that bring in distinct weight tensors (e.g.
    ``silu(x@Wg) * (x@Wu)`` — both reduce over K and index x, but only
    one indexes Wg and the other Wu) unify on the shared x position;
    the downstream :func:`merge_sibling_reduce_loops` pass then
    concatenates their bodies.
    """
    stmts = list(body)

    # Key on ``Dim.expr`` (the underlying ``Expr``) so structural equality on
    # extents matches both static and symbolic siblings: two ``Dim('seq_len')``
    # siblings unify (both back to ``Var('seq_len')``); two distinct symbolic
    # names don't. ``Expr`` is frozen + hashable so it slots into the tuple key.
    entries: list[tuple[int, str, object, frozenset[tuple[str, int, object, int]]]] = []
    for i, s in enumerate(stmts):
        if isinstance(s, Loop) and s.is_reduce:
            positions = _reduce_axis_source_positions(s.body, s.axis.name)
            if positions:
                entries.append((i, s.axis.name, s.axis.extent.expr, frozenset(positions)))

    if len(entries) < 2:
        return Body(stmts)

    parent = list(range(len(entries)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(entries)):
        for b in range(a + 1, len(entries)):
            if entries[a][2] != entries[b][2]:
                continue
            if entries[a][3] & entries[b][3]:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    for k, (idx, axis_name, extent, _) in enumerate(entries):
        canonical = entries[find(k)][1]
        if canonical == axis_name:
            continue
        loop = stmts[idx]
        assert isinstance(loop, Loop)
        # A RENAME — binders and uses through one map, which is why it may pass through the very
        # scopes it renames. Spelled as a σ plus an axis renamer it read as a substitution, and a
        # substitution must stop at a re-binding scope; these loops ARE the re-binding scopes.
        new_axis = replace(loop.axis, name=canonical, extent=extent)
        renamed = tuple(s.rename({loop.axis.name: canonical}) for s in loop.body)
        stmts[idx] = replace(loop, axis=new_axis, body=renamed)

    return Body(stmts)


def _reduce_axis_source_positions(body: Body, reduce_axis_name: str) -> set[tuple[str, int, object, int]]:
    """Collect ``(source, dim, anchor, coefficient)`` positions where a Load index within ``body``
    is AFFINE in ``Var(reduce_axis_name)`` (recursing into nested blocks).

    A bare ``Var`` is the ``(source, dim, 0, 1)`` case, so this generalizes the original bare-Var
    reading rather than replacing it. Affine matters because a BLOCKED reduce reads its stream at
    ``outer·B + inner``: the axis still walks that dimension, and refusing to see it left sibling
    loops over one block unmergeable for no semantic reason.

    The anchor and coefficient ride the key because that is what makes the reading sound. Two
    siblings indexing ``x[…, o·B + i]`` and ``x[…, o·B + j]`` walk the SAME dimension and unify;
    ``o·B + i`` against ``o·B + 32 + j`` walk different halves and must not. ``(source, dim)`` alone
    cannot tell those apart — seeing through the offset would be a miscompile, not a generalization.
    """
    out: set[tuple[str, int, object, int]] = set()
    for s in body.iter():
        if not isinstance(s, Load):
            continue
        for dim, e in enumerate(s.index):
            form = affine_form(e, {reduce_axis_name})
            if form is None:
                continue
            anchor, coeffs = form
            coeff = coeffs.get(reduce_axis_name, 0)
            if coeff:
                out.add((s.input, dim, anchor, coeff))
    return out


# ---------------------------------------------------------------------------
# Pass 4b: merge sibling reduce Loops with matching axis into one Loop.
# ---------------------------------------------------------------------------
#
# After :func:`unify_sibling_reduce_axes` renames sibling reduce axes
# that index overlapping ``(source, dim)`` positions to one canonical
# name, adjacent reduce Loops with the same axis name/extent become
# structurally identical iteration scopes. Merging concatenates their
# bodies into one Loop so the reduce axis is traversed once instead of
# twice. Later normalization by ``dedup_loads`` collapses the duplicate Loads
# both halves share — e.g. ``load x[0, a0, k]`` in the gated-MLP
# pattern ``silu(x@Wg) * (x@Wu)`` where both matmuls reduce over the
# same K and share x as a Load source. Symmetric staging follows: once
# wu lives in the same K-loop as wg, ``stage_inputs`` / ``use_tma`` /
# ``use_ring_buffers`` apply uniformly.
# ---------------------------------------------------------------------------


def merge_sibling_reduce_loops(stmts: Body) -> Body:
    """Merge sibling reduce ``Loop``s with matching ``axis.name`` and
    ``axis.extent`` into one Loop whose body is the concatenation.

    Gates a merge on three conditions, all phrased over what the second Loop
    reads from its enclosing scope (:func:`free_names` — what it uses and does
    not bind itself):

    1. It reads no SSA name the first Loop's body defines. When it does, the
       two reductions are sequentially dependent — e.g. softmax's sum-exp loop
       reads ``acc_max`` from the preceding max loop. Merging would replace
       that read of the *finalized* max with a read of the in-flight per-iter
       value, changing semantics.
    2. No statement between the two Loops defines a name it reads — otherwise
       the merge would move that read above its def.
    3. The names both bodies happen to bind are a COLLISION, not a dependence,
       and the incoming body's copies rename apart — which is what makes two
       alpha-equal copies of one cone mergeable at all. Only a name the incoming
       Loop still binds after it closes (:func:`_carried_out`) refuses: the
       rename cannot reach the readers that name has outside the loop.

    Statements that sit between the two original Loops stay in their
    original positions in the parent Body. References to the first
    Loop's ``Accum`` remain valid (Accum names cross the Loop
    boundary). References to the second Loop's ``Accum`` from
    statements that originally followed it now resolve to the merged
    Loop above them — still defs-before-uses.

    Recurses through every block-structured Stmt to find nested scopes.
    """
    stmts = Body.coerce(stmts)

    def walk(body: Body) -> Body:
        recursed: list[Stmt] = []
        for s in body:
            nested = s.nested()
            if nested:
                recursed.append(s.with_bodies(tuple(walk(b) for b in nested)))
            else:
                recursed.append(s)
        return _merge_sibling_reduce_loops(Body(recursed))

    return walk(stmts)


def _carried_out(body: Body) -> frozenset[str]:
    """The names a ``Loop`` over ``body`` still binds after it CLOSES.

    :meth:`Loop.render` declares the carriers of the immediate body ahead of the loop, so those —
    and, under a nested loop that does not seed its own, that loop's carriers too — are the names a
    later statement can still read. Every other definition lives inside the block the loop closes,
    which is what makes it renamable when two loops merge.
    """
    out = {name for stmt in body if isinstance(stmt, (Accum, Mma)) for name in stmt.carried_names()}
    for stmt in body:
        if isinstance(stmt, Loop) and not stmt.seed:
            out |= _carried_out(stmt.body)
    return frozenset(out)


def _rename_apart(body: Body, clashing: frozenset[str], taken: frozenset[str]) -> Body:
    """``body`` with each name in ``clashing`` renamed to one neither side spells."""
    used = set(body.ssa_defs) | set(body.ssa_uses) | set(taken)
    mapping: dict[str, str] = {}
    for name in sorted(clashing):
        fresh = next(candidate for n in count(1) if (candidate := f"{name}__m{n}") not in used)
        mapping[name] = fresh
        used.add(fresh)
    return Body(tuple(s.rename(mapping) for s in body))


def _merge_sibling_reduce_loops(body: Body) -> Body:
    items = list(body)
    if len(items) < 2:
        return body

    out: list[Stmt] = []
    consumed: set[int] = set()
    for i, s in enumerate(items):
        if i in consumed:
            continue
        if not (isinstance(s, Loop) and s.is_reduce):
            out.append(s)
            continue
        merged = s
        for j in range(i + 1, len(items)):
            if j in consumed:
                continue
            t = items[j]
            if not (
                isinstance(t, Loop)
                and t.is_reduce
                and t.axis.name == merged.axis.name
                and t.axis.extent == merged.axis.extent
                and t.unroll == merged.unroll
                and t.seed == merged.seed
            ):
                continue
            # ONE reading of what the incoming loop needs from around it: the names it reads and
            # does not bind itself. A name it both defines and uses is its own local — which is
            # what two alpha-equal copies of a single cone always share — and counting those as
            # reads reports a dependence that is not there.
            reads = free_names(t)
            merged_defs = Body.coerce(merged.body).ssa_defs
            if merged_defs & reads:
                continue
            incoming = Body.coerce(t.body)
            # What is left of the shared spellings is a COLLISION, not a dependence: two bodies
            # binding one name for unrelated values. Renaming the incoming body's copy apart is
            # sound for every name the loop closes over; a name it still binds afterwards has
            # readers the rename cannot reach, so that one refuses.
            clashing = merged_defs & incoming.ssa_defs
            if clashing & _carried_out(incoming):
                continue
            between_defs: set[str] = set()
            for k in range(i + 1, j):
                if k in consumed:
                    continue
                between_defs |= Body.coerce(Body((items[k],))).ssa_defs
            if between_defs & reads:
                continue
            merged = Loop(
                axis=merged.axis,
                body=Body(tuple(merged.body) + tuple(_rename_apart(incoming, clashing, merged_defs))),
                unroll=merged.unroll,
                seed=merged.seed,
            )
            consumed.add(j)
        out.append(merged)

    return Body(out)


# ---------------------------------------------------------------------------
# Pass 5a: split loop-invariant divides into reciprocal + multiply.
# ---------------------------------------------------------------------------
#
# ``divide(x, y)`` lowers to a single-precision divide on the XU pipe (the
# same pipe ``exp`` uses). When ``y`` is loop-invariant w.r.t. some
# enclosing Loop and ``x`` is not, the divide can't hoist as-is — its live
# set is the union of x's and y's. Splitting into::
#
#     recip_y = reciprocal(y)        # live = axes_of(y)
#     result  = multiply(x, recip_y) # live = axes_of(x) ∪ {recip_y}
#
# lets the next pass (``hoist_loop_invariants``) move ``recip_y`` out of
# every Loop axis that doesn't appear in ``y``. Inside the loop the
# divide turns into a multiply (FMA pipe), which is typically the
# under-utilized pipe on transcendental-heavy kernels (softmax,
# RMSNorm, attention output). One XU op per outer-axis iteration
# instead of one per inner-axis iteration.
#
# Gate: split iff ``axes_of(y)`` is a strict subset of ``axes_of(x)``.
# That's the precise structural condition for "splitting unblocks at
# least one Loop's worth of hoisting." Skip when y has axes x doesn't
# (no hoisting wins) or when both have identical axes (rcp would stay
# in the same scope as the original divide, no win and slight
# precision drift). When y is a true scalar (axes_of empty), the rcp
# hoists all the way to body root.
# ---------------------------------------------------------------------------


def split_invariant_divides(stmts: Body) -> Body:
    """Rewrite ``divide(x, y)`` → ``reciprocal(y) + multiply(x, recip)``
    when ``y``'s axis-dependency set is a strict subset of ``x``'s.

    Invariance is queried via :attr:`Body.axis_dependencies` over the
    pre-rewrite body. The strict-subset check means there's at least one
    axis ``x`` depends on that ``y`` doesn't — splitting moves the rcp out
    of that axis's Loop while the multiply stays. Generates fresh SSA names
    for the rcp; the trailing :func:`rename_ssa_sequential` pass renumbers
    them into ``vN`` form.
    """
    from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415

    stmts = Body.coerce(stmts)
    if not any(isinstance(stmt, Assign) and stmt.op.name == "divide" for stmt in stmts.iter()):
        return stmts
    axis_dependencies = dict(stmts.axis_dependencies)
    ssa_names: set[str] = set(axis_dependencies)
    fresh_counter = [0]

    def _fresh(prefix: str) -> str:
        while True:
            fresh_counter[0] += 1
            n = f"{prefix}_{fresh_counter[0]}"
            if n not in ssa_names:
                ssa_names.add(n)
                return n

    def _axes_of(name: str) -> frozenset[str]:
        return axis_dependencies.get(name, frozenset())

    def walk(body: Body) -> Body:
        out: list[Stmt] = []
        for s in body:
            nested = s.nested()
            if nested:
                # Generic descent — recurse into every nested body, rebuild
                # the wrapper via with_bodies. The closure was built once
                # over the whole body, so post-Loop Accum bookkeeping is
                # already baked in — no per-wrapper update needed here.
                out.append(s.with_bodies(tuple(walk(b) for b in nested)))
                continue
            if isinstance(s, Assign) and s.op == ElementwiseImpl("divide") and len(s.args) == 2:
                x_name, y_name = s.args
                if _axes_of(y_name) < _axes_of(x_name):  # strict subset → splitting unblocks at least one hoist
                    recip_name = _fresh(f"recip_{y_name}")
                    recip = Assign(name=recip_name, op=ElementwiseImpl("reciprocal"), args=(y_name,))
                    mult = Assign(name=s.name, op=ElementwiseImpl("multiply"), args=(x_name, recip_name))
                    # Patch dependencies for the freshly-introduced rcp so a
                    # later divide reading the same y in the same body
                    # still sees the correct axis set.
                    axis_dependencies[recip_name] = axis_dependencies.get(y_name, frozenset())
                    axis_dependencies[mult.name] = axis_dependencies.get(x_name, frozenset()) | axis_dependencies[recip_name]
                    out.append(recip)
                    out.append(mult)
                    continue
            out.append(s)
        return Body(out)

    return walk(stmts)


# ---------------------------------------------------------------------------
# Pass 5b: loop-invariant code motion
# ---------------------------------------------------------------------------


def hoist_loop_invariants(stmts: Body) -> Body:
    """Move stmts out of ``Loop``s whose axis they don't depend on.

    Hoists ``Load`` / ``Assign`` / ``Select`` (SSA values) and entire
    ``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond`` blocks whose contents
    transitively avoid the outer axis — provided the block contains no
    ``Write`` (a Write hoist would change observable side effects).
    Block-level hoisting is what lets a Loop and its downstream consumer
    move together: hoisting just the consumer would leave it referencing
    an Accum still defined inside the outer Loop body.

    ``Accum`` / ``Init`` / ``Write`` always stay (iteration-tied
    semantics). Loop-invariance is queried via :meth:`Body.depends_on`
    against the body's transitive read closure, so the hoisted set is
    automatically closed under SSA dependencies — no separate ordering
    check is needed.
    """
    stmts = Body.coerce(stmts)
    name_axes = stmts.axis_dependencies
    axis_names = stmts.axis_names
    axis_deps: dict[int, tuple[Stmt, frozenset[str]]] = {}

    def _axis_deps(s: Stmt) -> frozenset[str]:
        """Axes read by one immutable subtree, computed bottom-up once."""
        key = id(s)
        cached = axis_deps.get(key)
        if cached is not None and cached[0] is s:
            return cached[1]
        reads = set(s.deps())
        for expr in s.exprs():
            reads.update(expr.free_vars())
        deps = reads & axis_names
        for name in reads:
            deps.update(name_axes.get(name, frozenset()))
        for child in (child for body in s.nested() for child in body):
            deps.update(_axis_deps(child))
        result = frozenset(deps - s.binds_axes())
        axis_deps[key] = (s, result)
        return result

    def _hoistable(s: Stmt, axis: str) -> bool:
        # Accum / Init are scope-bound to their enclosing Loop's reduction (an Init seeds an
        # Accum or a Carrier's state per output cell) — they can't move alone, but the
        # whole enclosing block can. Side-effecting stmts (Write, or any block containing a
        # Write) pin their iteration count and stay put.
        if isinstance(s, (Accum, Init)) or s.has_side_effects:
            return False
        return axis not in _axis_deps(s)

    def _crossing_a_definition(inner: list[Stmt], hoisted: list[Stmt]) -> list[Stmt]:
        """``hoisted`` less every stmt reading a name the loop body still BINDS.

        Axis-invariance alone does not earn a hoist. A nested reduction can export an
        accumulator that varies with none of the outer axes while its own loop stays pinned
        (attention's denominator is produced inside the value sweep, which is pinned by the
        head-dim axis the value slab reads). Its consumer then reads as invariant and moves
        above the definition. Iterated: un-hoisting one candidate can pin the next."""
        while hoisted:
            ids = {id(c) for c in hoisted}
            bound = {name for c in inner if id(c) not in ids for name in _exposed_defines(c)}
            keep = [c for c in hoisted if not (free_names(c) & bound)]
            if len(keep) == len(hoisted):
                break
            hoisted = keep
        return hoisted

    def walk(body: Body) -> list[Stmt]:
        new_body: list[Stmt] = []
        for s in body:
            if isinstance(s, (Loop, StridedLoop)):
                inner = walk(s.body)
                axis = s.axis.name
                hoisted = _crossing_a_definition(inner, [c for c in inner if _hoistable(c, axis)])
                hoisted_ids = {id(c) for c in hoisted}
                stay = [c for c in inner if id(c) not in hoisted_ids]
                new_body.extend(hoisted)
                new_body.append(replace(s, body=tuple(stay)))
            elif isinstance(s, Cond):
                new_body.append(Cond(cond=s.cond, body=tuple(walk(s.body)), else_body=tuple(walk(s.else_body))))
            else:
                new_body.append(s)
        return new_body

    return tuple(walk(stmts))


# ---------------------------------------------------------------------------
# Pass 6: simplify Exprs inside body Stmts (constant folding, identity collapse,
# range-based comparison folding). The per-Expr rewrite logic lives on each
# ``Expr`` subclass as ``simplify(ctx)``; the walk over Stmts is dispatched
# in :mod:`.passes` (singledispatch + Stage introspection).
# ---------------------------------------------------------------------------


def simplify_body(body: Body) -> Body:
    """Simplify every Expr inside a body. Seeds ``SimplifyCtx`` from
    ``Loop`` / ``StridedLoop`` / ``Tile`` axis extents as the walker descends.
    Tile-IR Stmt registrations are loaded when ``tile.ir`` is imported."""
    from emmy.compiler.ir.stmt.passes import simplify  # noqa: PLC0415

    body = Body.coerce(body)
    ctx = SimplifyCtx.empty()
    return tuple(simplify(s, ctx) for s in body)


# ---------------------------------------------------------------------------
# Pass: deduplicate Load stmts with identical (input, index)
# ---------------------------------------------------------------------------


def dedup_loads(stmts: Body) -> Body:
    """Drop duplicate ``Load`` stmts within nested scopes.

    Two ``Load`` stmts with the same ``(input, index)`` read the same
    value; keep the first and rewire downstream SSA references to its
    name. Operates per-scope: a Load at an outer scope is reused by
    inner siblings (their identical ``index`` doesn't reference any
    inner-axis Var, so the values are equal). Loads inside a nested
    scope are not visible to outer / sibling scopes.

    Hygienic: an inner scope that re-binds a name the outer scope
    deduped keeps its own binding — those are different variables
    (see :func:`~emmy.compiler.ir.stmt.passes.rename_free`)."""
    from emmy.compiler.ir.stmt.passes import rename_free  # noqa: PLC0415

    stmts = Body.coerce(stmts)

    def walk(
        body: Body,
        env: dict[tuple[str, tuple[str, ...]], str],
        parent_alias: dict[str, str],
    ) -> Body:
        local = dict(env)
        alias = dict(parent_alias)

        def rename(n: str) -> str:
            return alias.get(n, n)

        def descend(inner: Body) -> Body:
            """Enter ``inner``'s scope, dropping every alias / kept name whose spelling ``inner``
            re-binds. SSA names bound inside a Loop / Cond body are scoped to it, so such a name is
            a DIFFERENT variable — following it out would rewire the inner arithmetic to the outer
            value and redeclare the survivor."""
            shadowed = Body.coerce(inner).ssa_defs
            return walk(
                inner,
                {k: v for k, v in local.items() if v not in shadowed},
                {k: v for k, v in alias.items() if k not in shadowed},
            )

        out: list[Stmt] = []
        for s in body:
            if isinstance(s, Load):
                # Rewire any SSA names in this Load's *index* to their deduped
                # alias first — a gather ``weight[(int)in0, a]`` whose index
                # Load ``in0`` was itself deduped must follow ``in0`` to the
                # kept name, or the index dangles after the duplicate is
                # dropped. (No-op for plain axis indices: axes aren't aliased.)
                s = s.rewrite(rename)
                key = (s.input, tuple(e.pretty() for e in s.index))
                if key in local:
                    alias[s.name] = local[key]
                    continue
                local[key] = s.name
                out.append(s)
            elif isinstance(s, Loop | StridedLoop):
                out.append(replace(s, body=descend(s.body)))
            elif isinstance(s, Cond):
                out.append(Cond(cond=s.cond, body=descend(s.body), else_body=descend(s.else_body)))
            else:
                # ``rename_free``, not ``rewrite``: identical for a leaf, but a block stmt the
                # ladder above doesn't name (``Tile``) carries scopes the alias must stop at.
                out.append(rename_free(s, alias))
        return tuple(out)

    return walk(stmts, {}, {})


# ---------------------------------------------------------------------------
# Pass: topologically sort siblings so SSA defs precede their uses.
# ---------------------------------------------------------------------------


def topo_sort_siblings(stmts: Body) -> Body:
    """Reorder stmts within each Body so SSA defs precede their uses.

    Recurses into every child body via the Stmt protocol
    (:meth:`Stmt.nested` / :meth:`Stmt.with_bodies`), then runs a stable
    Kahn ordering over the current sibling list. A block stmt
    (``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond``) is opaque at the
    parent level: it ``defs`` any Accum names that escape its body
    (visible to siblings via Loop's cross-boundary Accum semantics) and
    ``uses`` its wrapper-level deps plus any free SSA names referenced
    inside (names referenced inside but not defined inside).

    Splicer worklists (and any future producer that emits stmts with
    sibling-dedup) can land a consumer above an already-emitted producer
    when the producer was reused from an earlier emission. Sorting at
    normalize time decouples final body order from producer subtleties
    and guarantees every constructed ``LoopOp`` / ``TileOp`` lands with
    defs above uses, which downstream passes (validator, renamer,
    codegen) rely on.

    Stable: when the dep edges leave a free choice, the original sibling
    order is preserved (heap-based Kahn with index tiebreak). Idempotent:
    bodies already in topo order round-trip unchanged.
    """
    return _topo(Body.coerce(stmts))


def _topo(body: Body) -> Body:
    import heapq

    items: list[Stmt] = []
    for s in body:
        nested = s.nested()
        if nested:
            items.append(s.with_bodies(tuple(_topo(b) for b in nested)))
        else:
            items.append(s)

    n = len(items)
    if n <= 1:
        return Body(tuple(items))

    defs_uses = [_sibling_defs_uses(s) for s in items]
    # First-writer wins: handles repeated Accum decls (idempotent at the
    # same name) and the rare aliasing edge case without crashing.
    def_idx: dict[str, int] = {}
    for i, (defs, _) in enumerate(defs_uses):
        for name in defs:
            def_idx.setdefault(name, i)

    incoming: list[set[int]] = [set() for _ in range(n)]
    outgoing: list[list[int]] = [[] for _ in range(n)]
    for i, (_, uses) in enumerate(defs_uses):
        for name in uses:
            j = def_idx.get(name)
            if j is not None and j != i and j not in incoming[i]:
                incoming[i].add(j)
                outgoing[j].append(i)

    ready: list[int] = [i for i in range(n) if not incoming[i]]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        i = heapq.heappop(ready)
        order.append(i)
        for k in outgoing[i]:
            incoming[k].discard(i)
            if not incoming[k]:
                heapq.heappush(ready, k)

    if len(order) != n:
        # Cycle through SSA names — leave order untouched so the validator
        # rejects it with a precise message instead of silently shuffling.
        return Body(tuple(items))
    return Body(tuple(items[i] for i in order))


def _sibling_defs_uses(stmt: Stmt) -> tuple[frozenset[str], frozenset[str]]:
    """Names ``stmt`` makes visible to siblings, and names it depends on
    from siblings.

    Leaves: ``defs = stmt.defines()``, ``uses = stmt.deps()``.
    Block stmts: ``defs`` = Accum names escaping the body (recursive);
    ``uses`` = wrapper's own deps ∪ ((all inner uses) − (all inner SSA
    defs)).
    """
    nested = stmt.nested()
    if not nested:
        return frozenset(stmt.defines()), frozenset(stmt.deps())
    defs: set[str] = set()
    all_uses: set[str] = set(stmt.deps())
    all_inner_defs: set[str] = set()
    for b in nested:
        defs |= _exported_accs(b)
        all_uses |= Body.coerce(b).ssa_uses
        all_inner_defs |= Body.coerce(b).ssa_defs
    return frozenset(defs), frozenset(all_uses - all_inner_defs)


def _exported_accs(body: Body) -> frozenset[str]:
    return Body.coerce(body)._exported_accums


# ---------------------------------------------------------------------------
# Pass 7: canonicalize SSA names to sequential v0, v1, ...
# ---------------------------------------------------------------------------


def rename_ssa_sequential(stmts: Body) -> Body:
    """Canonicalize names in a fused body:

    - Axes from every axis-bearing scope (``Loop`` / ``StridedLoop`` /
      ``Tile.axes`` / new tile flavors' axes) renamed to ``a0, a1, ...``
      in pre-order of first declaration. All scopes share one numbering
      namespace so Tile.axes ``a0_o`` and a Loop axis ``a2_o`` don't
      collide on rename.
    - Load SSA names renamed to ``in0, in1, ...`` in definition order.
    - Accum names renamed to ``acc0, acc1, ...`` in definition order.
    - Assign / Select SSA names renamed to ``v0, v1, ...`` in definition
      order.

    Idempotent: bodies already in canonical form round-trip unchanged."""
    stmts = Body.coerce(stmts)
    ssa_rename: dict[str, str] = {}
    axis_rename: dict[str, str] = {}
    expr_sub: dict[str, Expr] = {}
    counters = {"v": 0, "in": 0, "acc": 0}

    def _rename(name: str, prefix: str) -> str:
        new = f"{prefix}{counters[prefix]}"
        ssa_rename[name] = new
        counters[prefix] += 1
        return new

    def _record_axis(name: str) -> None:
        if name in axis_rename:
            return
        new = f"a{len(axis_rename)}"
        axis_rename[name] = new
        if name != new:
            expr_sub[name] = Var(new)

    for stmt in stmts.iter():
        if isinstance(stmt, Load):
            for old in stmt.names:
                if old in ssa_rename:
                    continue
                # Only record the SSA rename in ``ssa_rename`` — NOT in
                # ``expr_sub`` (sigma). The Load/Write rewriters apply
                # ``_rename_ssa_vars_in_expr(sigma.apply(e), rename)`` to index
                # exprs: ``sigma`` is the axis-substitution channel, ``rename``
                # the SSA channel. Putting an SSA rename in *both* renames an
                # indirect (gather) index Var twice. Sequential renumbering can
                # form a chain — e.g. cell-3's index ``in2_3 → in5`` while a
                # pre-existing ``in5`` (a layernorm-weight Load) → ``in26`` —
                # and the double application collapses it (``in2_3 → in5 →
                # in26``), wiring the gather to the wrong row. ``acc`` / ``v``
                # names are likewise kept out of ``expr_sub`` (they reach exprs
                # only via ``rename``), so this keeps Load names consistent.
                _rename(old, "in")
        elif isinstance(stmt, Accum) and stmt.name not in ssa_rename:
            _rename(stmt.name, "acc")
        elif isinstance(stmt, (Assign, Select)) and stmt.name not in ssa_rename:
            _rename(stmt.name, "v")
        elif isinstance(stmt, Unpack):
            # ``low_name`` and ``high_name`` are fresh SSA scalars
            # defined by Unpack — must get rename slots in the ``v`` pool.
            # Without this, they collided with their input's renamed name
            # (e.g. paired Accum ``acc0_acc1_p`` → ``acc0`` makes
            # ``Unpack(low_name="acc0", value="acc0_acc1_p")`` rewrite to
            # ``Unpack(low_name="acc0", value="acc0")`` — self-referential).
            for old in (stmt.low_name, stmt.high_name):
                if old not in ssa_rename:
                    _rename(old, "v")
        elif isinstance(stmt, Pack) and stmt.name not in ssa_rename:
            # ``Pack.name`` defines a fresh f16x2 SSA value consumed by the
            # next Accum. Same reasoning as Assign — give it a ``v`` slot.
            _rename(stmt.name, "v")
        elif isinstance(stmt, (Loop, StridedLoop)):
            _record_axis(stmt.axis.name)

    if all(o == n for o, n in ssa_rename.items()) and all(o == n for o, n in axis_rename.items()):
        return stmts

    # ONE map: renaming an SSA value and renaming an axis are the same operation, and a rename
    # travels through the very binders it renames. Spelled as a σ over axis names it read as a
    # substitution — which must stop at a re-binding scope, and these loops ARE those scopes.
    names = {**ssa_rename, **axis_rename}
    return tuple(s.rename(names) for s in stmts)


# ---------------------------------------------------------------------------
# Pass: sort args of commutative Assigns.
# ---------------------------------------------------------------------------


def sort_commutative_args(stmts: Body) -> Body:
    """Sort ``Assign.args`` for commutative ``op``s so two bodies that
    differ only by argument order land in the same canonical form.

    Acts on ``Assign`` only — Expr-level commutativity (e.g. ``a + b``
    inside a ``Load`` index or ``Cond.cond``) is handled by
    :func:`simplify_body` via the per-Expr ``simplify`` rules. Recurses
    through every block-structured Stmt (``Loop`` / ``StridedLoop`` /
    ``Tile`` / ``Cond``)."""
    stmts = Body.coerce(stmts)

    def fn(s: Stmt) -> Stmt:
        if isinstance(s, Assign) and s.op.commutative and len(s.args) > 1:
            sorted_args = tuple(sorted(s.args))
            if sorted_args != s.args:
                return replace(s, args=sorted_args)
        return s

    return stmts.map(fn)


# ---------------------------------------------------------------------------
# Pass: canonicalize external-buffer names (opt-in via normalize_body flag).
# ---------------------------------------------------------------------------


def canonicalize_buffer_names(stmts: Body) -> Body:
    """Rename ``Load.input`` and ``Write.output`` buffer references to
    ``b0, b1, ...`` in encounter order via :meth:`Body.iter`.

    Off by default — buffer names bind to graph nodes (each ``Load.input``
    matches the producing op's id), so renaming them in a body that's
    still attached to an Op would break that wiring. Used by
    :attr:`Body.structural_key()` for dedup queries where buffer identity
    doesn't matter (two bodies with identical access patterns over
    differently-named inputs are structurally equal)."""
    stmts = Body.coerce(stmts)

    rename: dict[str, str] = {}
    for s in stmts.iter():
        for name in (*s.external_reads(), *s.external_writes()):
            if name not in rename:
                rename[name] = f"b{len(rename)}"

    if all(o == n for o, n in rename.items()):
        return stmts

    return stmts.rename_buffers(rename)


# ---------------------------------------------------------------------------
# Pass: collapse ops to their compute-unit cluster representative
# (opt-in via normalize_body's ``cluster_ops`` flag).
# ---------------------------------------------------------------------------


def canonicalize_op_clusters(stmts: Body) -> Body:
    """Replace every ``ElementwiseImpl`` field on every stmt with its
    cluster representative from :func:`cluster_representative`.

    The pass walks ``stmts`` with :meth:`Body.map` and uses
    ``dataclasses.fields`` to locate any field currently holding an
    ``ElementwiseImpl`` (covers ``Init.op`` / ``Assign.op`` /
    ``Accum.op`` without coupling this module to those IR dialects). A
    fold algebra and the kernel-IR cross-thread combine
    stmts (``WarpShuffle`` / ``TreeHalve``) carry their op inside an
    ``Assign`` program (``merge`` / ``combine_states``), already
    canonicalized at the carrier before lowering. The replacement is
    destructive — the resulting body is only safe to consume from
    :attr:`Body.structural_key()`.
    """
    from dataclasses import fields, is_dataclass  # noqa: PLC0415

    from emmy.compiler.ir.elementwise import ElementwiseImpl, cluster_representative  # noqa: PLC0415

    def fn(s: Stmt) -> Stmt:
        if not is_dataclass(s):
            return s
        changes: dict[str, ElementwiseImpl] = {}
        for f in fields(s):
            val = getattr(s, f.name)
            if isinstance(val, ElementwiseImpl):
                rep = cluster_representative(val)
                if rep != val:
                    changes[f.name] = rep
        if not changes:
            return s
        return replace(s, **changes)

    return stmts.map(fn)
