"""Flat-address `Expr` builders — fold-aware `sum` / `product` over int / `Expr` terms, used to
construct σ-tiled load/store indices in the lowering passes (`enumeration/_build` warp-tier σ-tiling,
`assembly/_assemble` carrier realization), plus `gmem_axis_step` / `gmem_row_stride` — how a gmem
`Load`'s flat address moves along one axis, the row step (`ldm`) a fragment loader reads off the
index and buffer shape and the contiguity its columns need, plus `BYTE_SLAB_PAD` — the smem row pad
a cp.async-staged byte slab carries. Generic Expr / addressing algebra — no flash / attention /
dialect dependency. Lives in `lowering/` so both the enumeration and assembly layers — and both the
`tile/` and `kernel/` lowering stages — import it without crossing the enumeration↔assembly
boundary.
"""

from __future__ import annotations

import math

from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, affine_form, axis_step
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


def gmem_axis_step(load: Load, axis_name: str, inputs) -> tuple[int, int] | None:
    """How ``load``'s FLAT gmem address moves along ``axis_name`` — ``(delta, run)``: the address
    changes by ``delta`` elements per unit step of the axis, and that holds over any ``run``-long
    ``run``-aligned window of the axis (``run == 0`` ⇒ the whole extent). The ONE address-algebra
    read the operand loaders and their schedule gates share; ``None`` when the walk cannot say.

    The flat address is ``Σ_j index[j] · stride_j`` over the buffer's declared strides, so the
    walk is per component (:func:`_axis_step`) and the deltas add. Three index forms answer:

    - the canonical layout, where the axis rides ONE component affinely — ``head_dim`` for
      ``(batch…, row, dd)``, ``H·D`` for an un-transposed ``(B, S, H, D)`` trace, where assuming
      the trailing extent read the WRONG rows (the gemma layer-0 NaN);
    - a BLOCKED read, where an index map splits the axis into a ``/`` block coordinate and a ``%``
      offset — the sdpa→o_proj chain's ``(head, seq, dd)`` gather. The block index does not move
      with the offset, so the offset's delta is exact within one block: that is what ``run``
      carries;
    - a DELINEARIZED read, where a reshape re-strides the rows and the components re-read one flat
      coordinate (:func:`_delinearized`), tried when the per-component walk gives up.

    ``None`` for an unknown buffer, an absent axis, a symbolic trailing extent, or any index the
    two readings do not cover — the schedule gate's decline signal."""
    t = inputs.get(load.input) if inputs else None
    if t is None or len(load.index) != len(tuple(t.shape)):
        return None
    dims = tuple(t.shape)
    delta, run, seen, gave_up = 0, 0, False, False
    for pos, e in enumerate(load.index):
        if axis_name not in e.free_vars():
            continue
        seen = True
        step = axis_step(e, axis_name)
        if step is None:
            gave_up = True
            break
        run = step[1] if run == 0 else run if step[1] == 0 else math.gcd(run, step[1])
        if not step[0]:
            continue  # the component holds still on the axis — its stride never enters the sum
        stride = 1
        for d in dims[pos + 1 :]:
            if not d.is_static:
                return None
            stride *= d.as_static()
        delta += step[0] * stride
    if not seen:
        return None
    if not gave_up:
        return delta, run
    flat = _delinearized(load.index, dims)
    return None if flat is None else axis_step(flat, axis_name)


def gmem_row_stride(load: Load, axis_name: str, inputs) -> int | None:
    """The flattened gmem stride between successive ``axis_name`` rows of ``load``'s buffer — the
    fragment loaders' row step (``ldm``). The UNBOUNDED reading of :func:`gmem_axis_step`: a row
    step that only holds inside a block is no ``ldm``, because the loader applies it across the
    whole tile. ``None`` when underivable — the schedule gate's decline signal."""
    step = gmem_axis_step(load, axis_name, inputs)
    return step[0] if step is not None and step[1] == 0 and step[0] > 0 else None


def reads_declared_rows(load: Load, axis_name: str) -> bool:
    """Whether the operand's row step IS the buffer's DECLARED trailing extent — the reading a
    loader falls back on when it is handed no stride (``ir._resolve_ldm``), and the only one that
    survives a SYMBOLIC extent, which no integer derivation can name. True exactly when the axis
    rides the penultimate index component alone at coefficient 1: then the flat row step is the
    product of the extents after it, which IS the declared trailing extent.

    The escape hatch for :func:`gmem_row_stride` declining on a symbolic dim — a masked / dynamic-M
    matmul reads canonical rows and always did. Not a substitute for the derivation: it says the
    declaration happens to be right here, nothing about any other index."""
    if len(load.index) < 2:
        return False
    form = affine_form(load.index[-2], {axis_name})
    if form is None or form[1].get(axis_name, 0) != 1:
        return False
    return not any(axis_name in e.free_vars() for i, e in enumerate(load.index) if i != len(load.index) - 2)


def _int_lit(e: Expr) -> int | None:
    """``e``'s value when it is an integer :class:`Literal`, else ``None``."""
    return e.value if isinstance(e, Literal) and isinstance(e.value, int) and not isinstance(e.value, bool) else None


def _delinearized(index: tuple, shape: tuple) -> Expr | None:
    """The ONE flat source coordinate a DELINEARIZING index map re-reads, or ``None`` when
    ``index`` is not such a map.

    ``130_reshape`` spells a reshape as ``c_j = (flat / stride_j) % dim_j`` over the SOURCE
    buffer's strides, with no ``% dim_0`` on the leading coordinate. Flattening that back —
    ``Σ_j c_j · stride_j`` — collapses to ``flat`` exactly, for every value, by the mixed-radix
    identity (C division guarantees ``(a/b)·b + a%b == a``). Recovering ``flat`` is what lets a
    caller read the operand's REAL row stride off a reshaped view instead of the buffer's declared
    one — the declared reading walks the wrong rows.

    Size-1 source dims are ``Literal(0)`` (``030_lift_indexmap`` forces them) and contribute
    nothing, so they are skipped. Anything else — a transpose, a cat's select branches, a gather —
    fails the match and declines."""
    if len(index) != len(shape) or len(index) < 2:
        return None
    dims, strides, s = list(shape), [1] * len(shape), 1
    for j in range(len(dims) - 1, 0, -1):
        strides[j - 1] = s = s * (dims[j].as_static() if dims[j].is_static else 0)
        if not s:
            return None
    flat = None
    for j, e in enumerate(index):
        if not dims[j].is_static:
            return None
        if dims[j].as_static() == 1:
            continue  # a size-1 source dim reads Literal(0) and strides nothing
        if j > 0:  # every non-leading coordinate carries its own ``% dim_j``
            if not (isinstance(e, BinaryExpr) and e.op == "%" and _int_lit(e.right) == dims[j].as_static()):
                return None
            e = e.left
        if strides[j] != 1:
            if not (isinstance(e, BinaryExpr) and e.op == "/" and _int_lit(e.right) == strides[j]):
                return None
            e = e.left
        if flat is None:
            flat = e
        elif e != flat:
            return None  # the coordinates delinearize DIFFERENT expressions — not one reshape
    return flat
