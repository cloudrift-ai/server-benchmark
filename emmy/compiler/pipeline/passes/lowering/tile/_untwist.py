"""The twisted rewrite's inverse over Loop IR — the two-pass form a twist recipe's ONLINE loop lifts to.

``Fold.merge`` lowers a twisted carrier to one reduce loop whose channels fold through
``Accum.base`` (``O = α·O + β·v``: the ψ-rescale of the carried value as the ``base``, the
β-weighted injection as the ``value``) and whose pivot folds plainly (``m = max(m, s)``). This reads
that recurrence back against each recipe's step — the ``advance`` cone over the pivot pair, one
``rescale`` per channel — and spells what it certifies: the pivot's own reduce first, then one plain
reduce per channel whose lift is the channel's ``pattern`` at the pivot's final state. That is the
shape the frontend gives a two-pass softmax and the shape ``Fold.twist`` matches, so the online loop
and the two-pass ops meet at one stored tree, and the value channel is a contraction node of it.

Only the LINEAR structure of the recurrence is read here: each channel enters its ⊕ at degree one,
with factors that read the pivot pair alone. The nonlinear identity that makes the block merge
associative (``exp(a)·exp(b) = exp(a + b)``) is the recipe's, so a recurrence no recipe certifies is
left as it came, and the lift refuses it as non-canonical Loop IR.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.twist import RECIPES, Channel, Recipe
from emmy.compiler.ir.stmt import Accum, Assign, Body, Const, Loop, Stmt


def untwist_body(stmts) -> tuple[Stmt, ...]:
    """``stmts`` with every reduce loop spelled as an online twisted carrier expanded, in place,
    into the two-pass loops it certifies; every other statement as it came."""
    out: list[Stmt] = []
    for stmt in Body.coerce(stmts):
        expanded = _untwist(stmt) if isinstance(stmt, Loop) and stmt.is_reduce else None
        out.extend(expanded if expanded is not None else (stmt,))
    return tuple(out)


def _factor(stmt: Stmt | None, name: str) -> str | None:
    """The other argument of the product ``stmt`` multiplying ``name``, or ``None``."""
    if not isinstance(stmt, Assign) or stmt.op.name != "multiply" or len(stmt.args) != 2 or name not in stmt.args:
        return None
    return stmt.args[1] if stmt.args[0] == name else stmt.args[0]


def _untwist(loop: Loop) -> tuple[Loop, ...] | None:
    body = Body.coerce(loop.body)
    accums = [stmt for stmt in body if isinstance(stmt, Accum)]
    channels = [accum for accum in accums if accum.base is not None]
    pivots = [accum for accum in accums if accum.base is None]
    if not channels or len(pivots) != 1:
        return None
    (pivot,) = pivots
    defs = body.definitions
    # Every channel rescales its carried value by ONE factor (``base = state·α``) and weighs its
    # injection by ONE other (``value = inj·β``); the two factors are the advance's.
    alphas = {_factor(defs.get(accum.base), accum.name) for accum in channels}
    if len(alphas) != 1 or None in alphas:
        return None
    (alpha,) = alphas
    # The advance is what the factors read past the score and the carried pivot: cut the cone at
    # both, so a nested score contraction and the pivot's own fold stay outside it.
    score = {id(stmt) for stmt in body.backward_cone((pivot.value,)).members}
    steps = Body(tuple(stmt for stmt in body if id(stmt) not in score and not isinstance(stmt, Accum)))
    for recipe in RECIPES:
        if recipe.advance is None or recipe.rescale is None or pivot.op.reduce_canon != recipe.pivot:
            continue
        for beta in (arg for stmt in (defs.get(accum.value) for accum in channels) if isinstance(stmt, Assign) for arg in stmt.args):
            if beta == alpha or any(_factor(defs.get(accum.value), beta) is None for accum in channels):
                continue
            advance = steps.backward_cone((alpha, beta))
            pivot_step = next(
                (stmt.name for stmt in advance.members if isinstance(stmt, Assign) and stmt.op.reduce_canon == recipe.pivot), None
            )
            if pivot_step is None:
                continue
            spelled = Lambda.closing((pivot.name, pivot.value), Body(advance.members), (pivot_step, alpha, beta))
            if not spelled.alpha_eq(recipe.advance):
                continue
            expanded = _two_pass(loop, body, pivot, channels, recipe, beta, advance.members)
            if expanded is not None:
                return expanded
    return None


def _channel(recipe: Recipe, injection: str, defs: dict) -> Channel | None:
    """The recipe channel an injected value spells: a literal ``1`` is the weight-only channel,
    anything else the expectation over one extra."""
    const = defs.get(injection)
    extras = 0 if isinstance(const, Const) and const.value == 1.0 else 1
    return next((c for c in recipe.channels if c.pattern is not None and len(c.pattern.params) == 2 + extras), None)


def _two_pass(loop: Loop, body: Body, pivot: Accum, channels: list[Accum], recipe: Recipe, beta: str, advance) -> tuple[Loop, ...] | None:
    defs = body.definitions
    # What the two-pass loops do not keep: the advance, each channel's rescale and weighing, and the folds.
    dropped = {id(stmt) for stmt in advance} | {id(pivot)} | {id(accum) for accum in channels}
    dropped |= {id(defs[accum.base]) for accum in channels} | {id(defs[accum.value]) for accum in channels}
    prelude = tuple(stmt for stmt in body if id(stmt) not in dropped)
    loops = [_loop(loop, prelude, pivot)]
    for accum in channels:
        injection = _factor(defs[accum.value], beta)
        channel = _channel(recipe, injection, defs)
        if channel is None:
            return None
        pattern = channel.pattern
        # The channel's per-element map at the pivot's FINAL state — ``(score, pivot, *extras)``
        # by role, its temps namespaced on the state it feeds.
        names = dict(zip(pattern.params, (pivot.value, pivot.name, injection), strict=False))
        names.update((stmt.name, f"{accum.name}__{stmt.name}") for stmt in pattern.body)
        instance = pattern.rename(names)
        loops.append(_loop(loop, (*prelude, *instance.body), replace(accum, value=instance.results[0], base=None)))
    return tuple(loops)


def _loop(loop: Loop, stmts: tuple[Stmt, ...], accum: Accum) -> Loop:
    """``loop`` folding ``accum`` alone, its body the backward cone of the folded value."""
    members = Body((*stmts, accum)).backward_cone((accum.value,)).members
    return replace(loop, body=Body((*(stmt for stmt in stmts if any(stmt is member for member in members)), accum)))


__all__ = ["untwist_body"]
