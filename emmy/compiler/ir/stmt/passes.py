"""Stmt rewrite + simplify, dispatched by type.

Replaces the per-class ``Stmt.rewrite`` overrides on body-carrying and
leaf stmts, and the ``_simplify_stmt`` if-ladder in ``normalize``.
The Stage hierarchy uses ``dataclasses.fields()`` introspection inside
the registered handler — adding a new ``Expr`` / ``Axis`` field on a
Stage subclass is picked up automatically (no override needed, no
silent-drop bug).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass
from functools import singledispatch

from emmy.compiler.ir.axis import Axis, extend_simplify_ctx
from emmy.compiler.ir.expr import Expr, SimplifyCtx, Var
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt.base import Stmt, _axis_identity
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import (
    Accum,
    Assign,
    Init,
    Load,
    Mma,
    Pack,
    Select,
    SelectBranch,
    StateMerge,
    Unpack,
    Write,
)

Rename = Callable[[str], str]
AxisFn = Callable[[Axis], Axis]


def _rename_ssa_vars_in_expr(e: Expr, rename: Rename) -> Expr:
    """Apply ``rename`` to every free ``Var`` leaf inside ``e``.

    Used by ``Load`` / ``Write`` rewriters so that *indirect* indices
    (gather: ``x[a0, (int)in0]``, scatter: ``out[(int)idx_v] = ...``)
    have their SSA-name references rewritten when the enclosing body
    is replicated. Without this, the register-tile replicator in
    ``010_split_register_axes`` suffixes the defining Load's name
    (``in0`` → ``in0_1``) but leaves dependent indirect Loads pointing
    at the original ``in0`` — silently dropping the cross-replica data
    dependency.

    Axis-name Vars (``a0``, ``M_b``, …) are never in the rename map
    (it only carries SSA defines), so ``rename(name) == name`` for
    them and they pass through unchanged.
    """
    mapping = {n: Var(rename(n)) for n in e.free_vars() if rename(n) != n}
    return e.substitute(mapping) if mapping else e


# ---------------------------------------------------------------------------
# Generic walker — recurses tuples + plain dataclasses (Addressing, BoundAxis,
# SelectBranch); applies ``on_expr`` to Expr leaves and ``on_axis`` to Axis.
# Stmt is excluded — Stmt traversal goes through the singledispatch handlers.
# ---------------------------------------------------------------------------


def _walk(value, *, on_expr, on_axis):
    if isinstance(value, Expr):
        return on_expr(value)
    if isinstance(value, Axis):
        return on_axis(value)
    if isinstance(value, tuple):
        return tuple(_walk(v, on_expr=on_expr, on_axis=on_axis) for v in value)
    if is_dataclass(value) and not isinstance(value, Stmt):
        return type(value)(**{f.name: _walk(getattr(value, f.name), on_expr=on_expr, on_axis=on_axis) for f in fields(value)})
    return value


# ---------------------------------------------------------------------------
# rewrite — sigma + axis_fn + SSA renaming
# ---------------------------------------------------------------------------


@singledispatch
def rewrite(stmt: Stmt, rename: Rename, sigma: Sigma = Sigma.IDENTITY, axis_fn: AxisFn = _axis_identity) -> Stmt:
    raise NotImplementedError(f"rewrite not registered for {type(stmt).__name__}")


@rewrite.register
def _(s: Load, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Load(
        names=tuple(rename(n) for n in s.names),
        input=s.input,
        index=tuple(_rename_ssa_vars_in_expr(sigma.apply(e), rename) for e in s.index),
        dtype=s.dtype,
    )


@rewrite.register
def _(s: Pack, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Pack(name=rename(s.name), low=rename(s.low), high=rename(s.high), dtype=s.dtype)


@rewrite.register
def _(s: Unpack, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Unpack(
        low_name=rename(s.low_name),
        high_name=rename(s.high_name),
        value=rename(s.value),
        lane_dtype=s.lane_dtype,
    )


@rewrite.register
def _(s: Assign, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Assign(name=rename(s.name), op=s.op, args=tuple(rename(a) for a in s.args), dtype=s.dtype)


@rewrite.register
def _(s: Accum, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    new_axes = tuple(n for old in s.axes for n in _rewrite_axis_name(old, sigma))
    return Accum(
        name=rename(s.name),
        value=rename(s.value),
        op=s.op,
        dtype=s.dtype,
        axes=new_axes,
        base=rename(s.base) if s.base is not None else None,
    )


@rewrite.register
def _(s: Mma, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    new_axes = tuple(n for old in s.axes for n in _rewrite_axis_name(old, sigma))

    def _g(guard):  # σ-substitute a (base, bound) guard's exprs so axis vars canonicalize
        return None if guard is None else (sigma.apply(guard[0]), sigma.apply(guard[1]))

    return Mma(
        c=rename(s.c),
        a=rename(s.a),
        b=rename(s.b),
        atom=s.atom,
        axes=new_axes,
        b_trans=s.b_trans,
        m_guard=_g(s.m_guard),
        n_guard=_g(s.n_guard),
        k_zero=_g(s.k_zero),
    )


@rewrite.register
def _(s: StateMerge, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    # The renderable cross-partition combine: rename the state / state_b (in the rename map)
    # PLUS the merge-internal temps NOT surfaced via ``defines()`` — so a register-tile
    # replicator that renames the state per cell leaves the temps shared, colliding across
    # replicas. Uniquify the temps with a suffix derived from the renamed first state name
    # whenever the state actually moves (identity rename / pure σ-split leaves them untouched).
    names = s.state
    new_state0 = rename(names[0]) if names else None
    carried = set(names) | set(s.state_b)
    temps = {a.name for a in s.merge} - carried
    overlay = {t: f"{t}__{new_state0}" for t in temps} if new_state0 is not None and new_state0 != names[0] else {}

    def rn(name: str) -> str:
        r = rename(name)
        return r if r != name else overlay.get(name, name)

    return StateMerge(
        state=tuple(rn(n) for n in names),
        merge=tuple(rewrite(m, rn, sigma, axis_fn) for m in s.merge),
        state_b=tuple(rn(n) for n in s.state_b),
    )


def _rewrite_axis_name(name: str, sigma: Sigma) -> tuple[str, ...]:
    """Apply ``sigma`` to an axis name and return the resulting axis
    name(s). Handles three cases:

    - ``sigma`` doesn't touch ``name``: returns ``(name,)``.
    - Pure rename (``Var(old) → Var(new)``): returns ``(new,)``.
    - σ-split (``Var(K) → Var(K_o)*N + Var(K_i)``, etc.): returns the
      free-var names of the substitution expression. An Accum that
      reduced over the original axis now reduces over the split sub-
      axes.
    """
    replacement = sigma.mapping.get(name)
    if replacement is None:
        return (name,)
    return tuple(sorted(replacement.free_vars()))


@rewrite.register
def _(s: Init, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    # ``identity`` is a constant scalar — only the name moves. Renamed in lockstep with
    # the fold's ``Accum`` / ``StateMerge.state`` (registered above) so the seed stays paired.
    return Init(name=rename(s.name), identity=s.identity, dtype=s.dtype)


@rewrite.register
def _(s: Write, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Write(
        output=s.output,
        index=tuple(_rename_ssa_vars_in_expr(sigma.apply(e), rename) for e in s.index),
        values=tuple(rename(n) for n in s.values),
        value_dtype=s.value_dtype,
        atomic=s.atomic,
        swizzle=s.swizzle,
    )


@rewrite.register
def _(s: Select, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Select(
        name=rename(s.name),
        branches=tuple(SelectBranch(value=rename(b.value), select=sigma.apply(b.select)) for b in s.branches),
    )


@rewrite.register
def _(s: Loop, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    # Preserve the reduce ``role`` annotation through σ-offsets / axis-renames. The loop carries
    # no algebra — the fold's ⊕ lives on the ``Fold`` node, whose own rewrite handler renames the
    # stored combine in lockstep (``rename_combine``).
    return Loop(
        axis=axis_fn(s.axis),
        body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body),
        unroll=s.unroll,
        role=s.role,
    )


@rewrite.register
def _(s: StridedLoop, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    step = sigma.apply(s.step) if isinstance(s.step, Expr) else s.step
    return StridedLoop(
        axis=axis_fn(s.axis),
        start=sigma.apply(s.start),
        step=step,
        body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body),
        unroll=s.unroll,
        role=s.role,
        end=sigma.apply(s.end) if s.end is not None else None,
    )


@rewrite.register
def _(s: Cond, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Cond(
        cond=sigma.apply(s.cond),
        body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body),
        else_body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.else_body),
    )


# ---------------------------------------------------------------------------
# simplify — ctx-driven Expr simplification, threading axis ranges
# ---------------------------------------------------------------------------


@singledispatch
def simplify(stmt: Stmt, ctx: SimplifyCtx) -> Stmt:
    # Default: no Expr fields to simplify (Assign / Accum / Init / StateMerge).
    return stmt


@simplify.register
def _(s: Load, ctx: SimplifyCtx) -> Stmt:
    return Load(names=s.names, input=s.input, index=tuple(e.simplify(ctx) for e in s.index), dtype=s.dtype)


@simplify.register
def _(s: Write, ctx: SimplifyCtx) -> Stmt:
    return Write(
        output=s.output,
        index=tuple(e.simplify(ctx) for e in s.index),
        values=s.values,
        value_dtype=s.value_dtype,
        atomic=s.atomic,
        swizzle=s.swizzle,
    )


@simplify.register
def _(s: Select, ctx: SimplifyCtx) -> Stmt:
    return Select(name=s.name, branches=tuple(SelectBranch(b.value, b.select.simplify(ctx)) for b in s.branches))


@simplify.register
def _(s: Loop, ctx: SimplifyCtx) -> Stmt:
    inner = extend_simplify_ctx(ctx, s.axis)
    return Loop(axis=s.axis, body=tuple(simplify(c, inner) for c in s.body), unroll=s.unroll, role=s.role)


@simplify.register
def _(s: StridedLoop, ctx: SimplifyCtx) -> Stmt:
    inner = extend_simplify_ctx(ctx, s.axis)
    step = s.step.simplify(ctx) if isinstance(s.step, Expr) else s.step
    return StridedLoop(
        axis=s.axis,
        start=s.start.simplify(ctx),
        step=step,
        body=tuple(simplify(c, inner) for c in s.body),
        unroll=s.unroll,
        role=s.role,
        end=s.end.simplify(ctx) if s.end is not None else None,
    )


@simplify.register
def _(s: Cond, ctx: SimplifyCtx) -> Stmt:
    return Cond(
        cond=s.cond.simplify(ctx),
        body=tuple(simplify(c, ctx) for c in s.body),
        else_body=tuple(simplify(c, ctx) for c in s.else_body),
    )


# Tile-IR Stmt registrations were DEMOLISHED along with the tile IR; pending
# rebuild.


def has_contraction_tail(stmts) -> bool:
    """True if the post-reduce tail contracts over a NEW free axis — a ``Loop`` whose body holds an
    inner reduce ``Loop``. This is the fused norm→linear shape, distinguished from a plain softmax
    tail (a single sweep over the SAME axis). ``Body.accums`` supplies the deep accumulator scan.

    A statement-SHAPE predicate, so it lives beside :func:`projection_distributes` rather than in
    the scheduler that asks: the reduce tiers read it to price a tail, and the shared-row stage
    gate to decide there is one to share a row with."""
    for s in stmts:
        if isinstance(s, Loop) and any(isinstance(c, Loop) and Body(c.body).accums for c in s.body):
            return True
        if any(has_contraction_tail(list(b)) for b in s.nested()):
            return True
    return False


def projection_distributes(body, states: tuple[str, ...]) -> bool:
    """True if the kernel's projection epilogue is a **linear-homogeneous** map of the carried
    state(s) — i.e. it distributes over the atomic-add combine, so applying it to each CTA's
    partition before the ``atomicAdd`` equals applying it once after the cross-CTA sum
    (``Σ c·xₛ = c·(Σ xₛ)``). A bare state write (``proj = id``) trivially distributes; a constant
    *scale* — ``mean``'s ``×1/N`` — does; an additive offset (a fused bias), a nonlinear unary
    (``relu`` / ``reciprocal`` of the *state*), or a product of two state-derived values do NOT.

    Conservative forward dataflow: ``linear`` is the set of SSA names that are a
    linear-homogeneous function of the state. A value is grown into it only by ``multiply`` with
    a state-independent operand (an arg not itself in ``linear``); any other op that consumes a
    ``linear`` value — or any projection stmt we can't reason about — refuses. The final ``Write``
    must store only ``linear`` values."""
    linear = set(states)
    for s in body:
        if isinstance(s, Write):
            return all(v in linear for v in s.values)
        if isinstance(s, Load):
            continue  # reads memory (the count / a per-output operand) — state-independent
        if not isinstance(s, Assign):
            return False  # an unfamiliar projection stmt — can't prove distributivity
        hot = [a for a in s.args if a in linear]
        if not hot:
            continue  # state-independent — a constant w.r.t. the split
        if s.op.name == "multiply" and len(hot) == 1:
            linear.add(s.name)  # state · constant — still linear-homogeneous
            continue
        return False  # add / divide / nonlinear of a state value breaks distributivity
    return False  # no Write reached
