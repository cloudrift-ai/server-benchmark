"""Flat-address `Expr` builders — fold-aware `sum` / `product` over int / `Expr` terms, used to
construct σ-tiled load/store indices in the lowering passes (`enumeration/_build` warp-tier σ-tiling,
`assembly/_assemble` carrier realization), plus `gmem_row_stride` — the flattened row step (`ldm`)
a fragment loader reads off a gmem `Load`'s index and buffer shape, plus `BYTE_SLAB_PAD` — the
smem row pad a cp.async-staged byte slab carries. Generic Expr / addressing algebra — no flash /
attention / dialect dependency. Lives in `lowering/` so both the enumeration and assembly layers —
and both the `tile/` and `kernel/` lowering stages — import it without crossing the
enumeration↔assembly boundary.
"""

from __future__ import annotations

from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, affine_form
from emmy.compiler.ir.stmt import Load

# Row pad (in ELEMENTS = bytes) for a cp.async-staged 1-byte (fp8) operand slab. The cooperative
# byte-gather drain (no ldmatrix below sm_100a — the fragment loads are per-lane byte gathers)
# reads the slab at the fragment lane map's row strides, and a dense power-of-two byte row lands
# every 4-row group in the same bank quartet (4-way conflicts measured by the lane→bank oracle on
# both the k16 convert drain and the k32 repack drain, either B orientation). 16 extra bytes per
# row breaks the stride while keeping every 16 B cp.async chunk aligned (the byte-staging
# legality requires the data cols to be 16-divisible, so ``cols + 16`` stays 16 B-periodic).
# Derived, not tuned — the same fixed-pad reasoning as the flash ``_twist._PAD``. cp.async only:
# a TMA box deposit is dense (its byte slab stays unpadded and eats the measured conflicts).
# Shared by the kernel staging pass and the tile-layer byte-staging legality check, so it lives
# here rather than in `kernel/_stage` — a tile pass may not import the kernel pass layer.
BYTE_SLAB_PAD = 16


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
