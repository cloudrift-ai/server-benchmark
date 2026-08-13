"""Kernel identity — the α-invariant structural key of a stored term, computed BOTTOM-UP.

Split out of ``ops`` (which is the geometry-free COMPUTE layer): what a term hashes to is a
CACHE concern, not a structural read. Nothing here is consulted while lowering; the one export
is :func:`structural_key` (the :class:`~emmy.compiler.structural.Structural` implementation
behind ``Fold.structural_key()`` / ``TileOp.structural_key()``), read by ``Op.cache_key``'s
TileOp override and ``Graph.structural_key``'s op field.

Each node's key is a digest folded from its OWN canonical content plus its children's cached
keys — a child is any nested ``Fold`` (a computed operand edge, or a node spliced into a lift
body by the collapse reading) and is never re-serialized at the parent. The term is IMMUTABLE
across the whole schedule search, so the per-node cache is sound, and a reading that rewrites
one level of the tree re-keys that level only: every untouched subtree object answers from its
cache.

Names canonicalize PER NODE — operand edges bind positionally and are closed, so no SSA name
crosses a node boundary except a child's RESULT components (``_operand_result_names``), which
the parent numbers at the child's walk position. Buffer identity is the one genuinely
cross-scope fact (RMSNorm's twice-read row, a squared reduce's ``x·x``): each node reports its
buffers in first-use order, and the parent folds each child's buffer list — mapped into the
parent's own first-use numbering — into the digest beside the child's key, so aliasing between
scopes is part of identity without any global walk. Axis names are NOT renumbered — they are
recognition-canonical and part of identity, exactly as before.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.stmt import Load
from emmy.compiler.ir.stmt.algebra import rename_combine
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.tile.ir import Fold, _operand_result_names
from emmy.compiler.structural import digest

#: Per-instance memo slot (``Fold`` is frozen; the slot rides ``__dict__`` beside the
#: ``cached_property`` entries, set via ``object.__setattr__`` like they are).
_CACHE_ATTR = "_structural_cache"


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


def structural_key(root) -> str:
    """The α-invariant identity of a stored term: every SSA name canonicalizes per node, every
    buffer positionally per scope with the cross-scope aliasing pattern kept, and the key text
    is a sha256 digest folded bottom-up from each node's canonical content plus its children's
    cached keys. Two terms differing only in SSA / buffer spelling — trace naming, fusion
    suffixes — key identically; any structural difference (an op, an axis, an operand shape,
    which scopes share a buffer) keys apart."""
    if root is None:
        return ""
    assert isinstance(root, Fold), f"a tile term root is a Fold, got {type(root).__name__}"
    return _structural(root)[0]


def _structural(node: Fold) -> tuple[str, tuple[str, ...]]:
    """``(key, buffers)`` for one node — the digest plus the node's buffer names in first-use
    order (the parent folds them into ITS numbering to keep cross-scope aliasing in the key).
    Memoized on the instance: the term is immutable across the schedule search, so a subtree
    shared between readings — or between two enumerations of the same op — computes once."""
    cached = node.__dict__.get(_CACHE_ATTR)
    if cached is not None:
        return cached

    # ---- collect: local names in the canonical walk order, buffers in first-use order, and the
    # child nodes at their walk positions. A child contributes its RESULT names (what the
    # enclosing scope binds/reads — the positional-binding contract) and its buffer list; its
    # internals stay behind its own key. ----------------------------------------------------- #
    names: list[str] = []
    seen: set[str] = set()
    bufs: list[str] = []
    bindex: dict[str, int] = {}
    children: list[Fold] = []

    def note(n) -> None:
        if isinstance(n, str) and n not in seen:
            seen.add(n)
            names.append(n)

    def note_buf(b: str) -> None:
        if b not in bindex:
            bindex[b] = len(bufs)
            bufs.append(b)

    def note_stmt(s) -> None:
        if isinstance(s, Fold):
            children.append(s)
            for b in _structural(s)[1]:
                note_buf(b)
            for n in _operand_result_names(s):
                note(n)
            return
        if isinstance(s, Load):
            note_buf(s.input)
        for n in s.defines():
            note(n)
        for b in s.nested():
            for c in b:
                assert not isinstance(c, Fold), "a Fold below a non-Fold stmt is not a stored form"
                note_stmt(c)

    body = tuple(node.lift.body)
    if node._contraction is None:
        # The bilinear reading's lift is generated, so only the EDGES canonicalize (reordering a
        # contraction's multiply stmts would be meaningless and would move the key).
        body = _canon_order(body)
    if node.axis is None:
        # The zero-axis reading names its binder first, then walks the sources — the order the
        # projection wrapper always used.
        for p in node.lift.params:
            note(p)
        for s in body:
            note_stmt(s)
        for r in node.lift.results:
            note(r)
        for e in node.operands:
            note_stmt(e)
    else:
        for e in node.operands:
            note_stmt(e)
        for p in node.lift.params:
            note(p)
        for s in body:
            note_stmt(s)
        for r in node.lift.results:
            note(r)
        for r in node.combine.results:
            note(r)

    # ---- render: the same walk again, each stmt as its renamed repr (buffers positional), each
    # child as (key, its buffers in THIS scope's numbering, its result names in THIS scope's
    # renaming). The render consumes ``children`` sequentially — both walks traverse the same
    # order, so ordinal k is children[k] even when one object appears twice. ------------------ #
    ren = {n: f"s{i}" for i, n in enumerate(names)}

    def rn(n: str) -> str:
        return ren.get(n, n)

    def rebuffered(s):
        if isinstance(s, Load):
            return replace(s, input=f"B{bindex[s.input]}")
        nested = s.nested()
        if not nested:
            return s
        return s.with_bodies(tuple(Body(tuple(rebuffered(c) for c in b)) for b in nested))

    consumed = iter(children)

    def render(s) -> object:
        if isinstance(s, Fold):
            child = next(consumed)
            key, cbufs = _structural(child)
            return ("node", key, tuple(bindex[b] for b in cbufs), tuple(rn(n) for n in _operand_result_names(child)))
        return repr(rebuffered(s).rewrite(rn))

    if node.axis is None:
        body_part = tuple(render(s) for s in body)
        edge_part = tuple(render(e) for e in node.operands)
        combine_part = None
    else:
        edge_part = tuple(render(e) for e in node.operands)
        body_part = tuple(render(s) for s in body)
        # ``rename_combine`` is the SSA-rename lockstep the Fold rewrite applies: a generated
        # twisted program REGENERATES over the renamed state, so the ``__o`` / ``__t`` derived
        # names follow their components without entering the local map.
        c = rename_combine(node.combine, rn)
        combine_part = (c.params, tuple(repr(s) for s in c.body), c.results)

    key = digest(
        "Fold",
        repr(node.axis),
        node.unroll,
        repr(node.init),
        tuple(rn(p) for p in node.lift.params),
        body_part,
        tuple(rn(r) if isinstance(r, str) else r for r in node.lift.results),
        combine_part,
        edge_part,
    )
    out = (key, tuple(bufs))
    object.__setattr__(node, _CACHE_ATTR, out)
    return out
