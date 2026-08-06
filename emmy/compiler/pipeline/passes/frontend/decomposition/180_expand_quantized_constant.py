"""Expand a quantized weight ``ConstantOp`` into its algebraic dequant cone (M2 of the FP8 plan).

M1 keeps quantization out of the graph: the loader dequantizes at bind time and
the constant materializes at the compute dtype. This rule is the M2 alternative —
gated behind ``EMMY_FP8_EXPAND`` (off by default; see ``config.fp8_expand``) — that
spells the dequant IN the graph so the fp8 bits stay in memory and the cast +
multiply can ride the B-operand cone into the kernel:

    W bits (f8e4m3/f8e5m2 ConstantOp, quant spec CONSUMED)
      → decode cast (``from_f8e4m3`` / ``from_f8e5m2`` — the LUT decode IS the
        cast's semantics; a plain ``copy`` would move uint8 bits, not values)
      → [reshape to the interleaved block grid]
      → multiply (divide on ``weight_scale_inv``) by the broadcast scale, a
        separate ConstantOp reading the spec's ``scale_path``
      → [reshape back]

Granularity stays DERIVED from the two shapes (per the QuantSpec contract):
per-tensor and per-out-channel scales need no reshape pair — every axis is a
whole-axis or per-element block, so a plain broadcast multiply suffices; only a
genuine 2-D block scale takes the interleaved-grid form. The cone's OUTPUT
tensor has exactly the dtype/shape the original constant had, so every pass
before this one is unaffected (the interface invariance of the FP8 plan).

Layout-commute rule: the constant's folded ``load_ops`` chain applies to the fp8
BITS (the W constant keeps it), so the scale constant must receive the chain's
block image. A layout op commutes with dequant iff it maps whole quant blocks to
whole blocks — a transpose always does (the scale grid transposes with it); a
reshape only under a per-tensor scale (implemented conservatively). A chain this
rule cannot commute leaves the constant on the M1 bind-time path — graceful
degradation per constant, never a compile error.

Numbered after the constant folds (``050``/``060``) and the sibling merge
(``035``) so the chain and ``source_parts`` have settled on the plain
``ConstantOp`` before the spec is consumed.
"""

from __future__ import annotations

from dataclasses import replace
from math import prod

from emmy import config
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.ir.frontend.ir import ReshapeOp, TransposeOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import broadcast_to, open_fragment

PATTERN = [Pattern("root", ConstantOp)]

_F8_FMTS = ("f8e4m3", "f8e5m2")


def _scale_grid_through_chain(load_ops: tuple, w_shape: tuple[int, ...], scale_shape: tuple[int, ...]):
    """Push the scale grid through the constant's layout chain.

    Returns ``(grid, scale_ops)``: the post-chain scale grid (one entry per axis
    of the post-chain weight) and the block-image ops the scale constant's own
    ``load_ops`` chain needs. Raises ``RuleSkipped`` for a chain op that does not
    map whole quant blocks to whole blocks.
    """
    per_tensor = prod(scale_shape) == 1
    if per_tensor:
        grid: tuple[int, ...] = (1,) * len(w_shape)
    else:
        if len(scale_shape) != len(w_shape):
            raise RuleSkipped(f"scale rank {len(scale_shape)} != weight rank {len(w_shape)}")
        if any(sd == 0 or wd % sd for wd, sd in zip(w_shape, scale_shape, strict=True)):
            raise RuleSkipped(f"scale shape {scale_shape} does not evenly divide weight shape {w_shape}")
        grid = scale_shape

    scale_ops: list = []
    for lop in load_ops:
        if isinstance(lop, TransposeOp):
            # A transpose maps whole blocks to whole blocks on every axis; the
            # scale grid takes the identical permutation.
            grid = tuple(lop.infer_output_shape([grid]))
            if not per_tensor:
                scale_ops.append(lop)
        elif isinstance(lop, ReshapeOp) and per_tensor:
            # A scalar scale is layout-blind; only the grid's rank tracks the
            # reshaped weight.
            grid = (1,) * len(lop.infer_output_shape([w_shape]))
        else:
            raise RuleSkipped(f"load op {type(lop).__name__} does not commute with the quant blocks")
        w_shape = tuple(lop.infer_output_shape([w_shape]))
    return grid, scale_ops


def rewrite(match: Match, root: Node, out: Tensor) -> Graph:
    op = root.op
    if op.quant is None:
        raise RuleSkipped("not a quantized constant")
    if not config.fp8_expand():
        raise RuleSkipped("EMMY_FP8_EXPAND off — the constant stays on the bind-time dequant path (M1)")
    spec = op.quant
    if spec.fmt not in _F8_FMTS:
        raise RuleSkipped(f"unknown quant storage format {spec.fmt!r}")
    if op.source_path is None or op.source_shape is None or op.source_parts:
        # ``source_parts`` (the sibling-linear concat) never carries a spec in
        # M1/M2a — the merge pass skips quantized weights (per-part scale concat
        # is later work).
        raise RuleSkipped("quant spec on a constant without a single checkpoint source")
    if any(not d.is_static for d in out.shape):
        raise RuleSkipped("symbolic weight shape")

    shape = tuple(d.as_static() for d in out.shape)  # post-chain, what consumers see
    grid, scale_ops = _scale_grid_through_chain(op.load_ops, tuple(op.source_shape), spec.scale_shape)
    if len(grid) != len(shape) or any(s % g for g, s in zip(grid, shape, strict=True)):
        raise RuleSkipped(f"post-chain scale grid {grid} does not tile the constant shape {shape}")
    block = tuple(s // g for g, s in zip(grid, shape, strict=True))
    # Degenerate blocks (per-tensor / per-out-channel): every axis is a whole-axis
    # or per-element block, so the scale broadcasts straight onto the weight shape.
    degenerate = all(g in (1, s) for g, s in zip(grid, shape, strict=True))
    scale_decl = grid if degenerate else tuple(x for g in grid for x in (g, 1))
    # The block-image chain leaves the scale at the post-chain grid; append one
    # reshape when the DECLARED shape differs (a stored ``()``/``(1,)`` per-tensor
    # scale normalized to the weight's rank, or the interleaved block form) so
    # the bound array itself has the declared layout.
    chain_shape = spec.scale_shape if not scale_ops else tuple(scale_ops[-1].infer_output_shape([spec.scale_shape]))
    if tuple(chain_shape) != scale_decl:
        scale_ops.append(ReshapeOp(shape=scale_decl))
    # A bf16-stored scale reads as f32 VALUES through the loader (the bits
    # carrier has no numpy value dtype), so the graph tensor says f32.
    scale_dtype = "f32" if spec.scale_dtype == "bf16" else spec.scale_dtype

    frag = open_fragment(match.graph, [])
    w = frag.add_node(
        # ``replace`` keeps every other field — name, load_ops (the bits are laid
        # out post-chain), source_path/shape — and CONSUMES the spec.
        op=replace(op, quant=None, source_dtype=spec.fmt),
        inputs=[],
        output=Tensor(f"{out.name}_bits", shape, spec.fmt),
    )
    scale = frag.add_node(
        op=ConstantOp(
            name=f"{op.name}_scale",
            source_path=spec.scale_path,
            source_shape=spec.scale_shape,
            source_dtype=spec.scale_dtype,
            load_ops=tuple(scale_ops),
        ),
        inputs=[],
        output=Tensor(f"{out.name}_scale", scale_decl, scale_dtype),
    )
    cast = frag.add_node(
        op=ElementwiseOp(op=f"from_{spec.fmt}"),
        inputs=[w],
        output=Tensor(f"{out.name}_dq", shape, out.dtype),
    )
    mul = ElementwiseOp(op="divide" if spec.inverse else "multiply")
    if degenerate:
        s_bc = broadcast_to(frag, scale, shape)
        final = frag.add_node(op=mul, inputs=[cast, s_bc], output=Tensor(out.name, shape, out.dtype))
    else:
        interleaved = tuple(x for g, b in zip(grid, block, strict=True) for x in (g, b))
        blk = frag.add_node(op=ReshapeOp(shape=interleaved), inputs=[cast], output=Tensor(f"{out.name}_blk", interleaved, out.dtype))
        s_bc = broadcast_to(frag, scale, interleaved)
        scaled = frag.add_node(op=mul, inputs=[blk, s_bc], output=Tensor(f"{out.name}_sblk", interleaved, out.dtype))
        final = frag.add_node(op=ReshapeOp(shape=shape), inputs=[scaled], output=Tensor(out.name, shape, out.dtype))
    frag.outputs = [final]
    return frag
