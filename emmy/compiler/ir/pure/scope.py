"""Context readings OVER pure terms — what an enclosing scope needs to know about one.

Every question here is about a term's relationship to something outside it: which enclosing axes
it references, whether a statement mentions one, what axis names appear in a stream. None of them
belong on :class:`~emmy.compiler.ir.pure.fold.Fold` itself. A term knows the axis it reduces over
and nothing else about its surroundings — the binder is what knows which names are axes at all,
and it supplies them downward (``Lambda.closing``'s scope argument). Asking the term to re-derive
them is the inversion this module exists to keep out of ``fold.py``.

They live here rather than in the schedule layer because they read the pure vocabulary only: an
operand edge's lowering, a statement's ``exprs`` / ``nested`` / ``binds_axes``. A consumer's
reading of an already-built term, in the same spirit as ``ir/schedule/packing``.
"""

from __future__ import annotations

from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Stmt


def free_names(stmts) -> frozenset[str]:
    """Every name ``stmts`` read from an enclosing scope, each stmt less the axis it binds."""
    out: set[str] = set()
    for stmt in stmts:
        names = {name for expr in stmt.exprs() for name in expr.free_vars()}
        for body in stmt.nested():
            names |= free_names(body)
        out |= names - stmt.binds_axes()
    return frozenset(out)


def edge_axes(edge, axes) -> frozenset[str]:
    """Which of ``axes`` this operand edge references — answered from the DECLARATION.

    The caller already holds the axis set, because the binder handed it down. It therefore has no
    business asking the term to re-derive it by walking a lowered body for names that look
    axis-shaped: a term declares the enclosing coordinates it reads as lift params
    (``Lambda.closing``'s scope), so the answer is an intersection.

    Two edge kinds, one rule. A nested term answers from its params, less the axis it BINDS —
    an edge reducing over its own ``k`` shadows an enclosing one of the same name and does not
    vary with it. A ``Load`` is a leaf with no params, so it answers from its own index exprs;
    that is reading the edge's own data, not interrogating it about its surroundings.
    """
    wanted = frozenset(axes)
    if isinstance(edge, Fold):
        return wanted & (set(edge.lift.params) - edge.binds_axes())
    return wanted & {name for expr in edge.exprs() for name in expr.free_vars()}


def refs_axis(s: Stmt, name: str) -> bool:
    """``s`` references axis ``name`` in any carried expr (deep) — ``Stmt.exprs``: a ``Load`` /
    ``Write`` index, a ``Select``'s branch predicates. Both spellings are coordinate reads, so both
    make the stmt vary with the axis; a mask ``Select`` read as invariant would be hoisted out of
    the per-cell body it predicates."""
    if any(name in e.free_vars() for e in s.exprs()):
        return True
    return any(refs_axis(child, name) for b in s.nested() for child in b)


def stmt_axis_names(stmts) -> set[str]:
    """Every loop induction variable bound anywhere in ``stmts`` (deep). A composed structural node
    sitting in the body needs no special case — it is a ``Stmt``, so its children are reached through
    the same ``nested()`` walk as any block stmt's."""
    out: set[str] = set()
    for s in stmts:
        ax = getattr(s, "axis", None)
        if ax is not None and hasattr(ax, "name"):
            out.add(ax.name)
        for b in s.nested():
            out |= stmt_axis_names(b)
    return out


__all__ = ["edge_axes", "free_names", "refs_axis", "stmt_axis_names"]
