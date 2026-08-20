"""Re-fuse adjacent free axes that a fused reshape split — the contraction-shape canonicalization.

A reshape fused into a producer splits one of the producer's output axes into a nest of two (an
attention projection's ``view(batch, seq, heads, head_dim)`` carves N = 12288 into 24 × 512), so the
kernel iterates the CONSUMER's post-view axes while its operand loads still address the producer's
single axis through a composite index (``wt[k, h*512 + d]``). Downstream that split is a lockout,
not a slowdown: contraction binding assigns ``(m, n)`` to the trailing free-axis pair, so the split
kernel binds the wrong row, the weight load carries a third grid axis, and the warp/mma tier is
never enumerated — the kernel stays a scalar reduce no amount of tuning can rescue.

This rule restores the canonical spelling: two adjacent free loops ``p`` (extent P) over ``q``
(extent Q) — perfectly nested, both static — fuse into one axis of extent P·Q via the bijective
reindexing ``p → f / Q``, ``q → f % Q``, with every coordinate expression σ-substituted and
re-simplified. The substitution is semantics-preserving unconditionally; whether it lands is a
PROFITABILITY gate checked after the fact: every rewritten access must fold clean. A composite
operand index collapses to the bare axis (``(f/Q)·Q + f%Q → f``, the recomposition fold in
``Expr.simplify``); a store that indexes the pair as separate buffer dims keeps the honest
``[…, f/Q, f%Q]`` spelling, accepted only when the buffer's row-major flatten folds it back to an
affine address (the strides match the split, so the emitted address is byte-identical). Any access
where a div/mod residue would survive — an axis used alone, a permuted-stride split, a predicate
over the pair — declines the pair, and the nest stands.

Runs to fixpoint inside one rewrite (a three-way split fuses pairwise), AFTER ``loop/fusion`` is
quiescent — canonicalizing a producer that still awaits a merge could re-spell the very indices the
splicer composes through — and before ``loop/stamp``, so kernel identity and everything downstream
(classification's trailing pair, shape keys, goldens) see one canonical spelling. Split and unsplit
spellings of the same contraction thereby converge to one kernel identity.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import BinaryExpr, CastExpr, Expr, FuncCallExpr, Literal, SimplifyCtx, TernaryExpr, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Load, Loop, Write
from emmy.compiler.ir.stmt.passes import simplify as _simplify_stmt
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", LoopOp)]


def _no_divmod_on(e: Expr, name: str) -> bool:
    """No ``/`` / ``%`` subterm of ``e`` has ``name`` in its dividend — the residue detector."""
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


def _access_ok(index: tuple, shape, fname: str) -> bool:
    """A rewritten access folds clean when its index exprs carry no div/mod residue on the fused
    axis, or — the split-store spelling ``[…, f/Q, f%Q]`` — when the buffer's row-major flatten
    recomposes the residue to an affine address (needs the full static shape)."""
    if all(_no_divmod_on(e, fname) for e in index if fname in e.free_vars()):
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
    for s in Body((fused,)).iter():
        if isinstance(s, Load):
            if not _access_ok(s.index, shapes.get(s.input), fname):
                return False
        elif isinstance(s, Write):
            if not _access_ok(s.index, shapes.get(s.output), fname):
                return False
        elif any(fname in e.free_vars() and not _no_divmod_on(e, fname) for e in s.exprs()):
            return False
    return True


def _fuse_pair(outer: Loop, inner: Loop, shapes: dict) -> Loop | None:
    """The fused loop for a perfectly-nested free pair, or ``None`` when the pair declines."""
    p, q = outer.axis, inner.axis
    if p.name == q.name or p.window is not None or q.window is not None:
        return None
    if not (p.extent.is_static and q.extent.is_static):
        return None
    big, small = p.extent.as_static(), q.extent.as_static()
    if big <= 1 or small <= 1:
        return None  # a size-1 side is drop_size_one_free_axes' job
    for s in Body(tuple(inner.body)).iter():
        ax = getattr(s, "axis", None)
        if ax is not None and ax.name in (p.name, q.name):
            return None  # an inner loop shadows a pair name — substitution would capture
    f = Var(q.name)
    lit = Literal(small, "int")
    sigma = Sigma({p.name: BinaryExpr("//", f, lit), q.name: BinaryExpr("%", f, lit)})
    body = Body(tuple(s.rewrite(lambda n: n, sigma) for s in inner.body))
    fused = Loop(axis=Axis(q.name, big * small), body=body, unroll=outer.unroll or inner.unroll, role=AxisRole.FREE, seed=inner.seed)
    fused = _simplify_stmt(fused, SimplifyCtx.empty())
    return fused if _folds_clean(fused, q.name, shapes) else None


def _fuse_once(body: Body, shapes: dict) -> Body | None:
    """The body with ONE pair fused (outermost-first, depth-first), or ``None`` when no pair
    fuses. The caller iterates to fixpoint, so an outer pair exposed by an inner fusion is
    picked up on the next round."""
    for i, s in enumerate(body):
        if not isinstance(s, Loop) or s.is_reduce:
            continue
        if len(s.body) == 1 and isinstance(s.body[0], Loop) and not s.body[0].is_reduce:
            fused = _fuse_pair(s, s.body[0], shapes)
            if fused is not None:
                return Body((*body[:i], fused, *body[i + 1 :]))
        inner = _fuse_once(s.body, shapes)
        if inner is not None:
            return Body((*body[:i], replace(s, body=inner), *body[i + 1 :]))
    return None


def rewrite(match: Match, root: Node, ctx=None) -> LoopOp:
    op = root.op
    if not isinstance(op, LoopOp):
        raise RuleSkipped("root is no longer a LoopOp")
    shapes = {name: t.shape for name, t in {**op.inputs, **op.outputs}.items()}
    body = op.body
    fused_any = False
    while (step := _fuse_once(body, shapes)) is not None:
        body, fused_any = step, True
    if not fused_any:
        raise RuleSkipped("no adjacent free-axis pair fuses")
    return replace(op, body=body)
