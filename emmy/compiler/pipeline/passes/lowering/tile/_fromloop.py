"""Loop IR → term: reconstructing a :class:`Fold`'s algebra from a reduce ``Loop``.

This is a PARSER, not part of the term vocabulary — it belongs with the passes that consume it
(``010_recognize``, ``ops.nodify_reduce``), not on the node. Every algebra fact reads off the loop
body's own ``Accum``\\s, so a loop carries no side-band algebra; the byte-identity gate in
:func:`fold_from_loop` is what lets the extractors stay shape-strict without a correctness burden
— a shape that does not read off cleanly returns ``None`` and the caller keeps the raw-loop-IR
escape."""

from __future__ import annotations

from emmy.compiler.ir.stmt import Accum, Assign, Body, Lambda, Load, Loop, component_ops
from emmy.compiler.ir.stmt.algebra import M
from emmy.compiler.ir.tile.ir import Fold, is_contraction
from emmy.compiler.ir.tile.ops import head, reduce_loop


def _extract_lift(loop: Loop, like: Fold | None = None) -> tuple[Lambda, tuple, Lambda] | None:
    """Read the λ spelling off a reduce ``Loop`` whose body is the dissolved ``[pure lift stmts…,
    Accum folds]`` sequence — ANNOTATION-FREE: every fact (accumulator names, channel ops, folded
    values) is read off the body's ``Accum``\\ s themselves, and threads into
    :func:`~emmy.compiler.ir.stmt.algebra.M` with the loop's REAL accumulator names. Returns
    ``(lift, init, combine)`` — the flat ⊕ pair — or ``None`` when the shape does not read
    off cleanly (a composed step, an effectful stmt, an identity-less op) — the caller keeps the
    raw-loop escape. A TWISTED (exp-family) loop has no self-describing ``Accum`` spelling (its
    merge interleaves ``base``-``Accum`` folds with ψ rescales), so it extracts only against a
    ``like`` fold carrying the algebra (:func:`_extract_twisted_lift` — the 030 split-partial
    path; recognition builds twisted folds directly, never through here). :meth:`Fold.from_loop`
    re-derives the loop and keeps the λ spelling only on byte-identity, so this extraction can
    stay shape-strict without a correctness burden."""
    if like is not None and component_ops(like.combine) is None:
        return _extract_twisted_lift(loop, like)
    accums = [s for s in loop.body if isinstance(s, Accum)]
    prefix = [s for s in loop.body if not isinstance(s, Accum)]
    if not accums:
        return None
    if any(a.base is not None for a in accums):
        # A ``base``-``Accum`` is the ψ-rescale signature of a dissolved exp-family merge — the
        # twisted spelling reconstructs from the body alone (the byte-compare is the validator).
        return _extract_twisted_self(loop)
    if any(not s.pure for s in prefix):
        return None  # an effectful / raw-block step is not λ-representable (the flat zero-axis escape)
    names = tuple(a.name for a in accums)
    try:
        lift = Lambda(params=(loop.axis.name,), body=Body(tuple(prefix)), results=tuple(a.value for a in accums))
        init, combine = M(*(a.op for a in accums), names=names)
    except ValueError:
        return None  # an undefined result / identity-less op — not the canonical dissolved shape
    if any(a.dtype is not None for a in accums):
        # A typed accumulator is PRECISION, which the λ spelling does not carry — the derived
        # ``Accum``\\ s come back dtype-free, so the byte-identity gate would reject the fold
        # anyway. Decline here and keep the raw-loop escape rather than silently dropping it.
        return None
    return lift, init, combine


def _extract_twisted_self(loop: Loop) -> tuple[Lambda, tuple, Lambda] | None:
    """Reconstruct the exp-family λ spelling from a TWISTED reduce ``Loop``'s body ALONE — no
    side-band algebra: the state is ``(the maximum-Accum, the add-Accums in body order)`` (the
    generator emits them exactly there), the pivot term is the maximum's folded score, and each
    non-pivot term is either the literal ``1.0`` (a denominator) or an external value name (an
    expectation) — resolved by regenerating the streaming merge per candidate and BYTE-COMPARING
    it against the body's tail (``exp_merge`` is deterministic, so equality is proof). The pure
    prefix ahead of the merge becomes the ``lift`` body. Returns ``None`` when no candidate
    matches — a composed step (flash's in-step folds) or a foreign merge spelling."""
    from itertools import product as _product  # noqa: PLC0415

    from emmy.compiler.ir.stmt.carrier import exp_combine_states, exp_merge  # noqa: PLC0415

    body = tuple(loop.body)
    accums = [s for s in body if isinstance(s, Accum)]
    maxes = [a for a in accums if a.op.reduce_canon == "maximum"]
    adds = [a for a in accums if a.op.reduce_canon == "add"]
    if len(maxes) != 1 or not adds or len(maxes) + len(adds) != len(accums):
        return None
    names = (maxes[0].name, *(a.name for a in adds))
    score = str(maxes[0].value)
    # A non-pivot term is the literal 1.0 (a denominator) or a value NAME — defined by a prefix
    # Load/Assign (flash's ``v``) or read from an enclosing scope. Every such name is a candidate;
    # the byte-compare below is the arbiter, so an over-wide pool costs regenerations, not
    # correctness (exp_merge embeds the term spelling, so no two candidates collide).
    defined = {s.name for s in body if isinstance(s, (Load, Assign))}
    read = {a for s in body for a in (getattr(s, "args", None) or ())}
    cands = sorted((defined | read) - {score} - set(names))
    for combo in _product([1.0, *cands], repeat=len(adds)):
        merge = tuple(exp_merge(names, (score, *combo), key=names[0]))
        if len(body) < len(merge) or body[-len(merge) :] != merge:
            continue
        prefix = body[: -len(merge)]
        if any(not isinstance(s, (Load, Assign)) for s in prefix):
            return None  # a composed step keeps the raw-loop escape
        try:
            lift = Lambda(params=(loop.axis.name,), body=Body(prefix), results=(score, *combo))
        except ValueError:
            return None
        other = tuple(f"{n}__o" for n in names)
        combine = Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names)
        return lift, (float("-inf"),) + (0.0,) * len(adds), combine
    return None


def _extract_twisted_lift(loop: Loop, like: Fold) -> tuple[Lambda, tuple, Lambda] | None:
    """Read the λ spelling off an exp-family TWISTED reduce ``Loop`` against the ``like`` fold
    that carries its algebra (the 030 split partial — the sliced loop's state names are the
    original fold's): the body must be ``[pure score prefix…, the dissolved streaming merge]``
    verbatim (the merge regenerated from ``like``'s stored combine + injection terms), the prefix
    becomes the ``lift`` body and the injected terms its results (the singleton — ``(x, 1)``),
    and the flat ⊕ pair stores ``like``'s combine — the generated cross-partition program over
    the loop's REAL state names (the formation invariant the consuming ``Fold`` asserts).
    ``None`` when the shape does not read off cleanly — a composed step (flash's in-step QK / PV
    folds, which carry their own schedule slices), a foreign merge spelling, a role the singleton
    shape cannot carry. :meth:`Fold.from_loop`'s byte-identity gate stands behind this extraction
    like the degenerate one's."""
    from emmy.compiler.ir.stmt.carrier import exp_merge  # noqa: PLC0415

    names = tuple(like.combine.results)
    terms = tuple(like.lift.results)
    merge = tuple(exp_merge(names, terms, key=names[0]))
    body = tuple(loop.body)
    if len(body) < len(merge) or body[-len(merge) :] != merge:
        return None
    prefix = body[: -len(merge)]
    if any(not isinstance(s, (Load, Assign)) for s in prefix):
        return None  # a composed step keeps the step spelling
    if not terms or not isinstance(terms[0], str):
        return None
    try:
        lift = Lambda(params=(loop.axis.name,), body=Body(prefix), results=terms)
    except ValueError:
        return None
    return lift, (float("-inf"),) + (0.0,) * (len(names) - 1), like.combine


def fold_from_loop(loop: Loop, like: Fold | None = None) -> Fold | None:
    """Build a :class:`Fold` from a reduce ``Loop`` — the loop-to-node constructor,
    ANNOTATION-FREE: every algebra fact reads off the body's own ``Accum``\\ s
    (:func:`_extract_lift`), so the loop carries no side-band algebra. A TWISTED loop has no
    self-describing spelling (its merge interleaves ``base``-``Accum`` folds with ψ
    rescales), so it extracts only against the ``like`` fold that carries its algebra —
    030's split partial, the sliced loop of a known fold. A canonical loop (the dissolved
    ``[pure lift stmts…, fold(s)]`` sequence) stores λ-spelled — ``lift`` + the flat
    ``(init, combine)`` pair threading the loop's real accumulator names — and the
    byte-identity gate settled at construction. A NON-canonical shape (an effectful /
    raw-block step, a non-reproducible derivation) returns ``None`` — the caller keeps the
    raw-loop-IR zero-axis ``Fold`` escape."""
    lifted = _extract_lift(loop, like)
    if lifted is None:
        return None
    lift, init, combine = lifted
    fold = Fold(axis=loop.axis, unroll=loop.unroll, lift=lift, init=init, combine=combine)
    # The byte-identity gate — a non-reproducible BODY keeps the raw-loop escape. The role
    # annotation is the fold's own DERIVED read, deliberately excluded: an unbindable matvec
    # captures a CONTRACTION-shaped loop and derives PLANAR — the 1l demotion, now a
    # formation fact (its loads stay inline in the lift).
    derived = fold.loop
    if (derived.body, derived.axis, derived.unroll) != (loop.body, loop.axis, loop.unroll):
        return None
    return fold


def nodify_reduce(op, like=None):
    """Nodify a kernel op into a :class:`~emmy.compiler.ir.tile.ir.Fold` node the reduce
    partition can key on (the plan itself lives in ``TileOp.schedule`` — the caller stores it
    against the returned fold). Returns ``(op2, fold)``. A stored contraction fold (the
    recognize-side per-cell contraction) is already the node — the coop / ILP K partition treats
    it as the degenerate algebra of its additive fold, any projection riding the wrapping zero-axis fold.
    A flat zero-axis fold holding an annotated reduce ``Loop`` (a split partial) nodifies via
    :meth:`Fold.from_loop`, which reconstructs the identical annotated loop, so the lowering is
    byte-identical; a projection tail (a fused epilogue) rides a wrapping zero-axis fold over the node, a
    bare reduce becomes the root node. ``like`` is the fold whose algebra a TWISTED loop extracts
    against (:meth:`Fold.from_loop`) — defaulted from ``op``'s own head when it is node-form
    (030 passes the pre-slice fold for its flat split partial).

    Used by the scheduler (the coop / ILP-K contraction) and ``030_split_reduce`` (the split partial) so
    EVERY partitioned reduce keys its plan off a node uniformly. For the zero-axis ``Fold`` form the reduce ``Loop``
    must be a top-level stmt of ``op.body`` with no prologue ahead of it (true for a split partial /
    bare sum: the reduce is the body's head); a projection tail after it becomes the wrapping
    zero-axis ``Fold`` body."""
    node = head(op)
    if is_contraction(node):
        return op, node  # a bare node IS its own head — the contraction keys its own plan
    if like is None:
        like = node if isinstance(node, Fold) else None
    rloop = reduce_loop(op)
    red = fold_from_loop(rloop, like)
    if red is None:  # not λ-representable (a raw-block slice) — the caller keeps the flat spelling
        return op, None
    body = list(op.body)
    idx = body.index(rloop)
    pre, tail = body[:idx], body[idx + 1 :]
    assert not pre, f"nodify_reduce: unexpected prologue ahead of the reduce loop: {pre}"
    return (Fold.projection(body=Body(tuple(tail)), operands=(red,)) if tail else red), red
