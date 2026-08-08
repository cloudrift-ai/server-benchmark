"""Fuse a body's per-element trellis decodes into ONE tile-column run.

A trellis-coded weight is stored as one tail-biting walk per 16x16 tile, so the 16 k rows of a
tile at one output column share the same handful of stream words. The scalar :class:`TrellisLoad`
leaf ignores that: it re-fetches two words and redoes the whole window derivation per element,
which is ~28 instructions and 4 bytes of code traffic per 2-bit weight.

The decode band the reduce tier offers (``_schedule._decode_band_specs`` — the transposed coop
band with ``reg`` at the tile's 16 rows) puts exactly one tile column in each body: 16 register
copies whose k indices are ``anchor, anchor+1, …, anchor+15`` at a common output column, with the
anchor tile-aligned. This peephole recognizes that set and replaces it with the run
:class:`TrellisLoad` (``names`` = the 16 SSA values, ``index`` = the anchor), which renders as one
``emmy_trellis_decode_col`` call.

Two things make the rewrite safe:

- **The anchor must be tile-aligned.** The run's element ``i`` is the weight at ``k_lo == i``, so
  ``anchor % 16`` has to be provably 0. The pass proves it affinely, reading the enclosing
  ``StridedLoop``'s start/step for the reduce coordinate; an unaligned or non-affine anchor simply
  does not fuse.
- **The decodes may be hoisted to the first of them.** ``TrellisLoad`` is a pure read of a kernel
  INPUT buffer (no kernel writes codes), and the run's index uses only coordinates bound outside
  the body, so moving the later decodes up past the intervening multiply/accumulate cannot change
  any value.

Idempotent by construction: after the rewrite the body holds a run rather than 16 scalars, so a
re-scan matches nothing and the pass declines.
"""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, SimplifyCtx, Var, affine_form
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Stmt, StridedLoop, TrellisLoad
from emmy.compiler.ir.stmt.leaves import TRELLIS_TILE
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def _const(expr: Expr) -> int | None:
    """``expr`` as a plain integer, or ``None`` when it is not a literal one."""
    lit = expr.simplify(SimplifyCtx.empty())
    return int(lit.value) if isinstance(lit, Literal) and isinstance(lit.value, int) else None


def _abstract(expr: Expr) -> Expr:
    """``expr`` with every maximal subterm that :func:`affine_form` cannot decompose replaced by
    an OPAQUE coordinate named after the subterm — so a sum whose other terms are affine still
    splits, and two indices sharing that subterm still compare equal.

    Both producers of a decode column need it. The reduce band's anchor is affine in the loop
    coordinates outright; the warp tier's compute fill derives its run's anchor row by DIVIDING
    the flat run index (``(run / cols) * 16``), which is affine in nothing — but it is the same
    subterm in all 16 of the run's indices, and it carries its own literal tile multiple, which
    is all the two checks below need."""
    if isinstance(expr, BinaryExpr) and expr.op in ("+", "-", "*"):
        return BinaryExpr(expr.op, _abstract(expr.left), _abstract(expr.right))
    if affine_form(expr, set(expr.free_vars())) is not None:
        return expr
    return Var(f"_opaque:{expr.pretty()}")  # planning-only: never rendered, never a real coordinate


def _split(expr: Expr) -> tuple[int, dict[str, int]] | None:
    """``expr`` as ``(constant, {coordinate: coefficient})`` over its opaque-abstracted form, or
    ``None`` when even that is not affine with a literal constant part."""
    abstracted = _abstract(expr)
    form = affine_form(abstracted, set(abstracted.free_vars()))
    if form is None:
        return None
    const = _const(form[0])
    return None if const is None else (const, form[1])


def _aligned(expr: Expr, multiples: dict[str, int]) -> bool:
    """Whether ``expr`` is provably a multiple of :data:`TRELLIS_TILE`. ``multiples`` maps a loop
    coordinate to a known divisor of its every value (its start and step's common one)."""
    form = _split(expr)
    if form is None or form[0] % TRELLIS_TILE:
        return False
    return all((coef * multiples.get(name, 1)) % TRELLIS_TILE == 0 for name, coef in form[1].items())


def _loop_multiple(loop: StridedLoop) -> int:
    """The largest power of two up to :data:`TRELLIS_TILE` dividing every value the loop
    coordinate takes — the common divisor of its start and step. ``1`` when either is opaque."""
    step, start = _split(loop.step), _split(loop.start)
    if step is None or start is None or step[1] or start[1]:
        return 1
    m = TRELLIS_TILE
    while m > 1 and (step[0] % m or start[0] % m):
        m //= 2
    return m


def _fuse(body: Body, multiples: dict[str, int]) -> Body:
    """Recurse into nested bodies, then fuse this scope's tile-column decode sets."""
    descended: list[Stmt] = []
    for s in body:
        nested = s.nested()
        if not nested:
            descended.append(s)
            continue
        inner = dict(multiples)
        if isinstance(s, StridedLoop):
            inner[s.axis.name] = _loop_multiple(s)
        descended.append(s.with_bodies(tuple(_fuse(b, inner) for b in nested)))

    decodes = [(i, s) for i, s in enumerate(descended) if isinstance(s, TrellisLoad) and s.is_scalar]
    if len(decodes) != TRELLIS_TILE:
        return Body(tuple(descended))
    anchor = decodes[0][1]
    if len(anchor.index) != 2 or not _aligned(anchor.index[0], multiples):
        return Body(tuple(descended))
    base = _split(anchor.index[0])
    for off, (_, ld) in enumerate(decodes):
        same = (ld.input, ld.cb, ld.k_bits, ld.n_tiles) == (anchor.input, anchor.cb, anchor.k_bits, anchor.n_tiles)
        shifted = _split(ld.index[0])
        if not same or len(ld.index) != 2 or ld.index[1] != anchor.index[1] or shifted is None:
            return Body(tuple(descended))
        if shifted[1] != base[1] or shifted[0] - base[0] != off:
            return Body(tuple(descended))

    run = TrellisLoad(
        names=tuple(ld.names[0] for _, ld in decodes),
        input=anchor.input,
        index=anchor.index,
        dtype=anchor.dtype,
        cb=anchor.cb,
        k_bits=anchor.k_bits,
        n_tiles=anchor.n_tiles,
    )
    drop = {i for i, _ in decodes[1:]}
    return Body(tuple(run if i == decodes[0][0] else s for i, s in enumerate(descended) if i not in drop))


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    if not any(isinstance(s, TrellisLoad) for s in op.body.iter()):
        raise RuleSkipped("no trellis decode in this kernel")
    new_body = _fuse(op.body, {})
    if new_body == op.body:
        raise RuleSkipped("no tile-column decode set to fuse")
    return KernelOp(body=new_body, name=op.name, knobs=dict(op.knobs))
