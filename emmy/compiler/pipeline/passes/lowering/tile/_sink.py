"""Structural binding for the ``PLACE@stat`` row-statistic sink (``040_sink_row_reduce``).

One probe shared by the offer site (``010_recognize`` — is this kernel's statistic *sinkable*?)
and the realizer (``040_sink_row_reduce`` — re-bind and rewrite), exactly as
``bind_prologue_contraction`` serves the cone cut. The matched shape is the fused norm form: a
``Map(body=[π…, sweep Loop], source=Reduction(PLANAR over n))`` whose reduce folds a pure
per-cell term of ONE tensor ``T`` (``Σ x²`` — a single scalar ``Load`` of ``T``, a pure
``Assign`` chain over it, one additive ``Accum``). Everything is gated in the negative on
structural properties — never on a kernel identity:

- the reduce's flat address in ``T`` must be **affine** with reduce-axis coefficient 1 and the
  free (row) coefficients forming an exact mixed-radix cover of ``T`` — so ``row = flat / N``
  is a bijection onto the rows and a producer can attribute any stored cell to its row by flat
  address alone (the per-head qknorm case falls out for free: its "row" is ``(m, head)`` and
  the head coefficient is the head width).
- the projection body must be ``[pure prefix…, one non-reduce sweep Loop]`` — the re-emitted
  consumer then peels BOTH axes free and lands on a wide 2-D pointwise grid (a second trailing
  loop would strand it on the row-only grid, the grid-32 latency form the sink exists to kill).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, SimplifyCtx, affine_form
from emmy.compiler.ir.stmt import Accum, Assign, Init, Load, Loop, Stmt
from emmy.compiler.ir.tile import Map, Reduction


@dataclass(frozen=True)
class SinkBinding:
    """The re-bindable facts of a sinkable row-stat kernel (see module docstring)."""

    src: str  # the reduced tensor's buffer / producer-node id (``T``)
    n: int  # the reduce (row) width — ``row = flat / n``
    rows: int  # total row count (``__sq`` buffer length)
    row_coeffs: tuple[tuple[str, int], ...]  # free-axis name → its coefficient in the flat ROW index
    load: Load  # the reduce cell's single ``Load`` of ``T``
    chain: tuple[Assign, ...]  # the pure per-cell term chain (``v = x·x``), in program order
    acc: str  # the accumulator SSA name (the carrier state the projection reads)
    value: str  # the SSA name the ``Accum`` folds (the contribution term)

    def row_expr(self) -> Expr:
        """The consumer-side ``__sq`` row index — ``Σ (c_i / n) · free_i`` over the row axes."""
        flat: Expr | None = None
        for name, coeff in self.row_coeffs:
            from emmy.compiler.ir.expr import Var  # noqa: PLC0415 — leaf import, avoids a wide module import

            term: Expr = Var(name) if coeff == 1 else BinaryExpr("*", Var(name), Literal(coeff, "int"))
            flat = term if flat is None else BinaryExpr("+", flat, term)
        return flat if flat is not None else Literal(0, "int")


def flat_index_expr(index: tuple[Expr, ...], shape) -> Expr:
    """Row-major flatten ``index`` over a STATIC ``shape`` as one ``Expr`` (the symbolic sibling
    of ``render_index`` — built pass-side so the stmt carries a shape-free address)."""
    flat: Expr | None = None
    stride = 1
    for e, d in zip(reversed(index), reversed(tuple(shape)), strict=True):
        term: Expr = e if stride == 1 else BinaryExpr("*", e, Literal(stride, "int"))
        flat = term if flat is None else BinaryExpr("+", flat, term)
        stride *= d.as_static()
    return flat if flat is not None else Literal(0, "int")


def _is_zero(e: Expr) -> bool:
    s = e.simplify(SimplifyCtx.empty())
    return isinstance(s, Literal) and s.value == 0


def bind_sinkable_stat(node, free: tuple, inputs: dict) -> SinkBinding | None:
    """Bind ``node`` (a ``TileOp``'s op tree) as the sinkable row-stat form, or ``None``.

    ``free`` is the kernel's free (row) axes; ``inputs`` the matcher-populated buffer table
    (the reduced tensor's static shape comes from it). Pure structural probe — no graph access;
    the producer-side eligibility (non-atomic, settled) is the realizer's concern."""
    if not isinstance(node, Map) or not isinstance(node.source, Reduction):
        return None
    red: Reduction = node.source
    if red.role is not AxisRole.PLANAR or red.source is not None:
        return None
    if len(red.carrier.state.names) != 1:
        return None
    if not red.axis.extent.is_static:
        return None
    if not free or not all(ax.extent.is_static for ax in free):
        return None
    # The projection must be [pure prefix…, one non-reduce sweep Loop] — see module docstring.
    body: tuple[Stmt, ...] = tuple(node.body)
    if len(body) == 0 or not isinstance(body[-1], Loop) or body[-1].is_reduce:
        return None
    if not all(isinstance(s, (Load, Assign, Init)) for s in body[:-1]):
        return None
    # The reduce cell: one scalar Load of T, a pure Assign chain over it, one additive Accum.
    stmts = list(red.partial)
    loads = [s for s in stmts if isinstance(s, Load)]
    accums = [s for s in stmts if isinstance(s, Accum)]
    if len(loads) != 1 or not loads[0].is_scalar or len(accums) != 1:
        return None
    load, accum = loads[0], accums[0]
    if accum.op.name != "add" or accum.base is not None or accum.name != red.carrier.state.names[0]:
        return None
    chain: list[Assign] = []
    allowed = {load.name}
    for s in stmts:
        if s is load or s is accum:
            continue
        if not isinstance(s, Assign) or any(a not in allowed for a in s.args):
            return None
        allowed.add(s.name)
        chain.append(s)
    if accum.value not in allowed:
        return None
    # The flat-address gate: affine, reduce coefficient 1, zero anchor, free coefficients an
    # exact mixed-radix cover of T — the bijection that lets the producer derive the row.
    t = inputs.get(load.input) if inputs else None
    if t is None or not all(d.is_static for d in t.shape) or len(load.index) != len(t.shape):
        return None
    total = prod(d.as_static() for d in t.shape)
    n = red.axis.extent.as_static()
    flat = flat_index_expr(load.index, t.shape)
    names = {red.axis.name, *(ax.name for ax in free)}
    form = affine_form(flat, names)
    if form is None:
        return None
    anchor, coeffs = form
    if not _is_zero(anchor) or coeffs.get(red.axis.name, 0) != 1:
        return None
    terms = sorted([(1, n), *((coeffs.get(ax.name, 0), ax.extent.as_static()) for ax in free)])
    if any(c <= 0 for c, _ in terms):
        return None
    for j in range(1, len(terms)):
        if terms[j][0] != terms[j - 1][0] * terms[j - 1][1]:
            return None
    if terms[-1][0] * terms[-1][1] != total:
        return None
    row_coeffs = tuple((ax.name, coeffs[ax.name] // n) for ax in free)
    return SinkBinding(
        src=load.input,
        n=n,
        rows=total // n,
        row_coeffs=row_coeffs,
        load=load,
        chain=tuple(chain),
        acc=accum.name,
        value=accum.value,
    )


__all__ = ["SinkBinding", "bind_sinkable_stat", "flat_index_expr"]
