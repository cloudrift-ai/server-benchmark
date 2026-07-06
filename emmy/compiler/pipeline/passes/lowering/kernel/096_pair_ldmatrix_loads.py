"""Pair slab-adjacent staged ``x2`` B-fragment ``LdmatrixLoad``\\ s into one ``x4``.

Every staged B drain costs one ``ldmatrix.x2``[``.trans``] per fragment, but ``ldmatrix.x4``
loads four 8×8 matrices — two adjacent B fragments — in one instruction, halving the drain's
LSU count. Two emitters produce the fusable pattern (which is why this is a PASS, not an
emitter change — same family as ``050_vectorize_loads``): the warp-flash streaming drains
(``_twist._frag_contraction`` — K's N-adjacent plain-``x2`` pairs and V's col-adjacent
``x2.trans`` pairs) and the matmul tier's staged drains (``_atom._staged_inner_atom_loop`` —
``n.reg`` col-adjacent B fragments per K step).

Legality, judged structurally on the two loads:

- both staged role-``"b"``, same slab / ``ldm`` / ``b_trans`` / swizzle mode (the drain's
  swizzle XOR is per-lane address-based, so it commutes with the paired lane map — each lane
  un-permutes its own 16 B chunk), no masks (staged loads carry none);
- **transposed-B** (N-major slab): equal K col, N row exactly ``+8`` — the pair is one plain
  ``x4`` (lanes 16-31 address the ``+8`` N rows);
- **canonical-B** (K-major slab): equal K row, col exactly ``+8`` — the pair is one
  ``x4.trans`` (lanes 16-31 address the ``+8`` column). The ``+8``-half offsets are 16 B, so
  every paired lane address keeps ldmatrix's 16 B alignment;
- the second load moves UP to the first's position: every intervening stmt must be barrier-free
  straight-line code that neither redefines either fragment nor defines a free var of the moved
  load's index (the flash drain interleaves ``MmaSyncPtx`` between the loads — reading the FIRST
  fragment is fine, the pair fills both before any consumer runs).

The fusion is a pure perf transform: same slab cells, same values, same fragments — the paired
kernel is bit-identical to its unpaired sibling. Idempotent: a paired load (``pair_frag`` set)
is never a candidate, so a re-run finds nothing and skips.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import LdmatrixLoad, MbarrierWait, Sync
from emmy.compiler.ir.stmt import Body, Cond, Loop, Stmt, StridedLoop, Write
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]

#: One 8×8 ldmatrix matrix — the fragment adjacency step, in elements (16 B of b16).
_ATOM8 = 8


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    new_body, changed = _walk(op.body)
    if not changed:
        raise RuleSkipped("no pairable staged ldmatrix loads")
    return KernelOp(body=new_body, name=op.name, knobs=dict(op.knobs))


def _walk(body: Body) -> tuple[Body, bool]:
    stmts: list[Stmt] = []
    changed = False
    for s in body:
        nested = s.nested()
        if nested:
            new_bodies = []
            for b in nested:
                nb, c = _walk(b)
                new_bodies.append(nb)
                changed = changed or c
            if changed:
                s = s.with_bodies(tuple(new_bodies))
        stmts.append(s)
    paired = _pair(stmts)
    if paired is None:
        return Body(tuple(stmts)), changed
    return Body(tuple(paired)), True


def _split(e: Expr) -> tuple[Expr | None, int | None]:
    """``e`` as ``base + constant`` (``base`` None for a bare literal), or ``(e, None)`` when no
    constant tail is recognizable — callers compare bases by frozen-dataclass equality."""
    if isinstance(e, Literal):
        return None, int(e.value)
    if isinstance(e, BinaryExpr) and e.op == "+":
        if isinstance(e.right, Literal):
            return e.left, int(e.right.value)
        if isinstance(e.left, Literal):
            return e.right, int(e.left.value)
    return e, 0


def _delta(a: Expr, b: Expr) -> int | None:
    """The structural constant difference ``b - a``, or ``None`` (not provably constant)."""
    if a == b:
        return 0
    (ab, ac), (bb, bc) = _split(a), _split(b)
    if ab == bb and ac is not None and bc is not None:
        return bc - ac
    return None


def _candidate(s: Stmt) -> bool:
    return isinstance(s, LdmatrixLoad) and s.staged and s.role == "b" and s.pair_frag is None and len(s.src_index) == 2


def _pairs_with(a: LdmatrixLoad, b: LdmatrixLoad) -> bool:
    """``b`` is ``a``'s slab-adjacent partner: the SECOND fragment of one x4 (the ``+8``
    matrices lanes 16-31 address)."""
    if a.src_buffer != b.src_buffer or a.ldm != b.ldm or a.b_trans != b.b_trans or a.swizzle != b.swizzle:
        return False
    row_d, col_d = _delta(a.src_index[0], b.src_index[0]), _delta(a.src_index[1], b.src_index[1])
    if a.b_trans:  # N-major slab: N rows adjacent, same K col
        return row_d == _ATOM8 and col_d == 0
    return row_d == 0 and col_d == _ATOM8  # K-major slab: same K row, cols adjacent


def _blocks(s: Stmt, a_frag: str, moved_frag: str | None, moved_deps: set[str]) -> bool:
    """An intervening stmt that forbids moving the partner load up to the pair position: a
    barrier / control / slab-writing stmt; a redefinition of either fragment; a READ of the
    moved fragment (its pre-move value — e.g. the previous streaming step's — would be
    clobbered early); or a definition of a name the moved load's index reads. Reading the
    KEPT fragment is fine — the pair fills it at its original position."""
    if s.nested() or isinstance(s, (Sync, MbarrierWait, Write, Loop, StridedLoop, Cond)):
        return True
    defs = set(s.defines())
    if a_frag in defs or (moved_frag is not None and moved_frag in defs):
        return True
    if moved_frag is not None and moved_frag in s.deps():
        return True
    return bool(defs & moved_deps)


def _pair(stmts: list[Stmt]) -> list[Stmt] | None:
    out = list(stmts)
    changed = False
    i = 0
    while i < len(out):
        a = out[i]
        if not _candidate(a):
            i += 1
            continue
        j = i + 1
        while j < len(out):
            b = out[j]
            if _candidate(b) and _pairs_with(a, b):
                moved_deps = {v for e in b.src_index for v in e.free_vars()}
                if any(_blocks(s, a.frag, b.frag, moved_deps) for s in out[i + 1 : j]):
                    break
                out[i] = replace(a, pair_frag=b.frag)
                del out[j]
                changed = True
                break
            if _blocks(b, a.frag, None, set()):
                break
            j += 1
        i += 1
    return out if changed else None
