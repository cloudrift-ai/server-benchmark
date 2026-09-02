"""Shared split-free-axis substitution for canonical Loop bodies and fresh Tile pieces."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.address import split_pair
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import BinaryExpr, CastExpr, Expr, FuncCallExpr, Literal, SimplifyCtx, TernaryExpr, Var
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Load, Loop, Write
from emmy.compiler.ir.stmt.passes import simplify as _simplify_stmt


def _no_divmod_on(e: Expr, name: str) -> bool:
    """No ``/`` / ``%`` subterm of ``e`` has ``name`` in its dividend."""
    if isinstance(e, BinaryExpr):
        if e.op in ("/", "//", "%") and name in e.left.free_vars():
            return False
        return _no_divmod_on(e.left, name) and _no_divmod_on(e.right, name)
    if isinstance(e, TernaryExpr):
        return all(_no_divmod_on(x, name) for x in (e.cond, e.if_true, e.if_false))
    if isinstance(e, CastExpr):
        return _no_divmod_on(e.expr, name)
    if isinstance(e, FuncCallExpr):
        return all(_no_divmod_on(a, name) for a in e.args)
    return True


def _access_ok(index: tuple, shape, fname: str, store: bool = False) -> bool:
    """Whether a rewritten access leaves no unresolved split-axis residue."""
    if all(_no_divmod_on(e, fname) for e in index if fname in e.free_vars()):
        return True
    if store and split_pair(index, fname) is not None:
        return True
    if shape is None or len(shape) != len(index) or not all(getattr(d, "is_static", False) for d in shape):
        return False
    flat: Expr = Literal(0, "int")
    stride = 1
    for e, d in zip(reversed(index), reversed(list(shape)), strict=True):
        flat = BinaryExpr("+", flat, BinaryExpr("*", e, Literal(stride, "int")))
        stride *= d.as_static()
    return _no_divmod_on(flat.simplify(SimplifyCtx.empty()), fname)


def _folds_clean(fused: Loop, fname: str, shapes: dict) -> bool:
    for stmt in Body((fused,)).iter():
        if isinstance(stmt, Load):
            if not _access_ok(stmt.index, shapes.get(stmt.input), fname):
                return False
        elif isinstance(stmt, Write):
            if not _access_ok(stmt.index, shapes.get(stmt.output), fname, store=True):
                return False
        elif any(fname in expr.free_vars() and not _no_divmod_on(expr, fname) for expr in stmt.exprs()):
            return False
    return True


def _fuse_pair(outer: Loop, inner: Loop, shapes: dict, between: tuple[Loop, ...] = ()) -> Loop | None:
    """The fused nest for one clean free-axis pair, or ``None`` when it declines."""
    p, q = outer.axis, inner.axis
    if p.name == q.name or p.window is not None or q.window is not None:
        return None
    if not (p.extent.is_static and q.extent.is_static):
        return None
    big, small = p.extent.as_static(), q.extent.as_static()
    if big <= 1 or small <= 1:
        return None
    for stmt in Body(tuple(inner.body)).iter():
        axis = getattr(stmt, "axis", None)
        if axis is not None and axis.name in (p.name, q.name):
            return None
    fused_var = Var(q.name)
    small_lit = Literal(small, "int")
    sigma = Sigma({p.name: BinaryExpr("//", fused_var, small_lit), q.name: BinaryExpr("%", fused_var, small_lit)})
    body = Body(tuple(stmt.rewrite(lambda name: name, sigma) for stmt in inner.body))
    fused = Loop(
        axis=Axis(q.name, big * small),
        body=body,
        unroll=outer.unroll or inner.unroll,
        role=AxisRole.FREE,
        seed=inner.seed,
    )
    fused = _simplify_stmt(fused, SimplifyCtx.empty())
    if not _folds_clean(fused, q.name, shapes):
        return None
    for middle in reversed(between):
        fused = replace(middle, body=Body((fused,)))
    return fused


def _free_chain(loop: Loop) -> list[Loop]:
    """``loop`` and the perfectly nested free loops under it, outermost first."""
    out = [loop]
    while len(out[-1].body) == 1 and isinstance(out[-1].body[0], Loop) and not out[-1].body[0].is_reduce:
        out.append(out[-1].body[0])
    return out


def _fuse_once(body: Body, shapes: dict) -> Body | None:
    """The body with one clean pair fused, outermost-first and depth-first."""
    for index, stmt in enumerate(body):
        if not isinstance(stmt, Loop) or stmt.is_reduce:
            continue
        chain = _free_chain(stmt)
        for inner_index in range(1, len(chain)):
            fused = _fuse_pair(stmt, chain[inner_index], shapes, tuple(chain[1:inner_index]))
            if fused is not None:
                return Body((*body[:index], fused, *body[index + 1 :]))
        inner = _fuse_once(stmt.body, shapes)
        if inner is not None:
            return Body((*body[:index], replace(stmt, body=inner), *body[index + 1 :]))
    return None


def fuse_split_free_axes(body: Body, shapes: dict) -> Body | None:
    """Return the fixpoint with at least one split free-axis pair fused, else ``None``."""
    result = body
    changed = False
    while (step := _fuse_once(result, shapes)) is not None:
        result, changed = step, True
    return result if changed else None


__all__ = ["fuse_split_free_axes"]
