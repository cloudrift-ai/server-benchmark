"""Kernel identity — the α-invariant key of a stored term.

Split out of ``ops`` (which is the geometry-free COMPUTE layer): what a term hashes to is a
CACHE concern, not a structural read. Nothing here is consulted while lowering; the one export
is :func:`term_key`, read by ``op_cache_key``'s TileOp arm and ``Graph.structural_key``'s op
field."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.stmt import Load
from emmy.compiler.ir.tile.ir import Fold


def _term_names(root) -> tuple[list[str], list[str]]:
    """Every SSA name and buffer name in ``root``'s term, in DETERMINISTIC first-appearance
    order over the canonical walk (operand edges in tuple order, then lift params / body defs /
    results, then the combine results; a zero-axis ``Fold``'s fn params / body / results, then sources).
    The order is a function of the stored params only, so the renumber map — and with it
    :func:`term_key` — is α-invariant."""

    names: list[str] = []
    bufs: list[str] = []
    seen: set[str] = set()
    bseen: set[str] = set()

    def note(n) -> None:
        if isinstance(n, str) and n not in seen:
            seen.add(n)
            names.append(n)

    def note_stmt(s) -> None:
        if isinstance(s, Fold):
            walk(s)
            return
        if isinstance(s, Load) and s.input not in bseen:
            bseen.add(s.input)
            bufs.append(s.input)
        for n in s.defines():
            note(n)
        for b in s.nested():
            for c in b:
                note_stmt(c)

    def walk(node) -> None:
        if not isinstance(node, Fold):
            return
        if node.axis is None:
            # The zero-axis reading names its binder first, then walks the sources — the order
            # the projection wrapper always used, so canonical renumbering is unchanged.
            for p in node.lift.params:
                note(p)
            for s in node.lift.body:
                note_stmt(s)
            for r in node.lift.results:
                note(r)
            for src in node.operands:
                note_stmt(src)
            return
        for e in node.operands:
            note_stmt(e)
        for p in node.lift.params:
            note(p)
        for s in node.lift.body:
            note_stmt(s)
        for r in node.lift.results:
            note(r)
        for r in node.combine.results:
            note(r)

    walk(root)
    return names, bufs


def _canon_order(stmts: tuple) -> tuple:
    """A DETERMINISTIC dependency-respecting order for an ANF stmt sequence — the hash-time
    body-order canonicalization: Kahn's algorithm over the def/use edges, ready stmts
    picked by a NAME-INDEPENDENT token (stmt kind, op spelling, buffer, arity), ties keeping the
    original relative order. Two α-equivalent lift bodies that differ only in the interleaving of
    independent stmts canonicalize to one order; the stored term itself is never reordered — the
    lowered nest depends on storage order, identity does not."""
    stmts = tuple(stmts)
    if len(stmts) <= 1:
        return stmts

    def token(s) -> tuple:

        op = getattr(s, "op", None)
        return (
            type(s).__name__,
            getattr(op, "name", "") if op is not None else "",
            s.input if isinstance(s, Load) else "",
            len(getattr(s, "args", ()) or ()),
        )

    def reads(s) -> set:
        out = set(s.deps())
        for b in s.nested():
            for c in b:
                out |= set(reads(c))
        return out

    defs_of = [set(s.defines()) | {d for b in s.nested() for c in b for d in c.defines()} for s in stmts]
    read_of = [reads(s) for s in stmts]
    placed: list = []
    done: set = set()
    remaining = list(range(len(stmts)))
    while remaining:
        ready = [i for i in remaining if not (read_of[i] & {n for j in remaining if j != i for n in defs_of[j]} - done)]
        if not ready:
            return stmts  # a cycle (state-reading merge material) — keep the stored order
        pick = min(ready, key=lambda i: (token(stmts[i]), i))
        placed.append(stmts[pick])
        done |= defs_of[pick]
        remaining.remove(pick)
    return tuple(placed)


def _canon_tree(node):
    """``node`` with every λ body in its tree re-ordered by :func:`_canon_order` — the hash-side
    normal form ``term_key`` renumbers. Never stored."""

    from emmy.compiler.ir.stmt import Lambda  # noqa: PLC0415
    from emmy.compiler.ir.stmt.body import Body as _B  # noqa: PLC0415

    def canon_stmt(s):
        return _canon_tree(s) if isinstance(s, Fold) else s

    if not isinstance(node, Fold):
        return node
    if node._contraction is not None:
        # The bilinear reading's lift is generated, so only the EDGES canonicalize (reordering a
        # contraction's multiply stmts would be meaningless and would move the term key).
        return replace(node, operands=tuple(canon_stmt(e) for e in node.operands))
    body = tuple(canon_stmt(s) for s in _canon_order(tuple(node.lift.body)))
    lift = node.lift
    if all(st.pure for st in body):
        lift = Lambda(params=lift.params, body=_B(body), results=lift.results)
    return replace(node, lift=lift, operands=tuple(canon_stmt(e) for e in node.operands))


def term_key(root) -> str:
    """The α-INVARIANT identity of a stored term (step 7 — kernel identity keys off the
    canonically renumbered TERM, no longer the lowered loop nest): every SSA name maps to
    ``s<i>`` and every buffer to ``b<i>`` in the deterministic first-appearance order of
    :func:`_term_names`, the rename applied through the ONE ``_rewrite`` registry (the Fold /
    the Fold handler renames lift / combine in lockstep, regenerating the
    exp-family programs over the renamed state), and the renumbered term's ``repr`` is the key
    text. Two terms differing only in SSA / buffer spelling — trace naming, fusion suffixes —
    key identically; any structural difference (an op, an axis, an operand shape) keys apart."""
    from emmy.compiler.ir.sigma import Sigma  # noqa: PLC0415
    from emmy.compiler.ir.stmt.passes import rewrite as _stmt_rewrite  # noqa: PLC0415

    if root is None:
        return ""
    root = _canon_tree(root) if isinstance(root, Fold) else root
    names, bufs = _term_names(root)
    ren = {n: f"s{i}" for i, n in enumerate(names)}
    canon = _stmt_rewrite(root, lambda n: ren.get(n, n), Sigma.IDENTITY, lambda a: a)
    text = repr(canon)
    # Buffer names appear in the repr only as ``input='<buf>'`` (the term carries no ``Write``
    # since 1q) — canonicalize them positionally, longest first so no name prefixes another.
    order = {b: i for i, b in enumerate(bufs)}
    for b in sorted(bufs, key=len, reverse=True):
        text = text.replace(f"input='{b}'", f"input='B{order[b]}'")
    return text
