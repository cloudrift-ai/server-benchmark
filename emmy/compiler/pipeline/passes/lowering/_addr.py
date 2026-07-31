"""Flat-address `Expr` builders — fold-aware `sum` / `product` over int / `Expr` terms, used to
construct σ-tiled load/store indices in the lowering passes (`enumeration/_build` warp-tier σ-tiling,
`assembly/_assemble` carrier realization), plus `gmem_row_stride` — the flattened row step (`ldm`)
a fragment loader reads off a gmem `Load`'s index and buffer shape. Generic Expr / addressing
algebra — no flash / attention / dialect dependency. Lives in `lowering/` (a sibling of `_masking` /
`_predicates`) so both the enumeration and assembly layers — and both the `tile/` and `kernel/`
lowering stages — import it without crossing the enumeration↔assembly boundary.
"""

from __future__ import annotations

from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, affine_form
from emmy.compiler.ir.stmt import Load


def add(*terms) -> Expr:
    """Sum int / Expr terms into one Expr (dropping literal zeros)."""
    out = None
    for t in terms:
        e = Literal(t, "int") if isinstance(t, int) else t
        if isinstance(e, Literal) and e.value == 0:
            continue
        out = e if out is None else BinaryExpr("+", out, e)
    return out if out is not None else Literal(0, "int")


def mul(a, b: int) -> Expr:
    """``a · b`` as an Expr, folding the ``b in {0, 1}`` degenerate cases."""
    return add() if b == 0 else (a if b == 1 else BinaryExpr("*", a if not isinstance(a, int) else Literal(a, "int"), Literal(b, "int")))


def gmem_row_stride(load: Load, axis_name: str, inputs) -> int | None:
    """The flattened gmem stride between successive ``axis_name`` rows of ``load``'s buffer — the
    fragment loaders' row step (``ldm``). The axis var must appear in exactly ONE index component,
    affinely; the stride is its coefficient times the product of the buffer extents AFTER that
    component. ``head_dim`` for the canonical ``(batch…, row, dd)`` layout; ``H·D`` for an
    un-transposed ``(B, S, H, D)`` trace, where assuming the trailing extent read the WRONG rows
    (the gemma layer-0 NaN: resident-Q fragments at ``ldm=head_dim`` on a 4096-stride layout).
    ``None`` when underivable — axis absent or split across components, a non-affine use, an
    unknown buffer, or a symbolic trailing extent — the schedule gate's decline signal."""
    t = inputs.get(load.input) if inputs else None
    if t is None:
        return None
    positions = [i for i, e in enumerate(load.index) if axis_name in e.free_vars()]
    if len(positions) != 1:
        return None
    pos = positions[0]
    form = affine_form(load.index[pos], {axis_name})
    if form is None:
        return None
    coef = form[1].get(axis_name, 0)
    if coef <= 0:
        return None
    stride = coef
    for d in tuple(t.shape)[pos + 1 :]:
        if not d.is_static:
            return None
        stride *= d.as_static()
    return stride
