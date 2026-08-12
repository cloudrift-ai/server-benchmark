"""Lower an eligible static coded GEMV to one CudaOp, else recover generic algebra."""

from __future__ import annotations

import math

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.frontend.ir import Exl3GemvOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment

PATTERN = [Pattern("root", Exl3GemvOp)]

_SQ_COUNTERS_CAP = 4096
_SQ_KSPLIT_CAP = 64
_SQ_WS_RESERVED = _SQ_COUNTERS_CAP + 4 * _SQ_KSPLIT_CAP * 4


def _rows_per(*, size_k: int, size_n: int, grid: int, residual: bool) -> int:
    rows_total = size_k // 16
    r = math.ceil(rows_total * (size_n // 256) / grid)
    rows = (max(r, min(2 * r, 32)) + 7) & ~7
    cap = (80 * 1024) // (32 + 64 * (2 if residual else 1))
    cap &= ~7
    return min(max(rows, 16), cap, 512, (rows_total + 7) & ~7)


def _native_fragment(match: Match, x: Node, codes: Node, suh: Node, svh: Node, out: Tensor, ctx) -> Graph | None:
    op: Exl3GemvOp = match.root.op
    n, k = op.weight_shape
    cc = ctx.compute_capability
    if cc < (7, 0) or op.codebook != 2 or (cc < (8, 0) and op.bits not in (5, 6, 7)):
        return None
    if x.output.dtype.name != "f16" or out.dtype.name not in {"f16", "f32"}:
        return None
    if tuple(d.as_static() for d in codes.output.shape) != (k // 16, n // 16, 16 * op.bits):
        return None
    if tuple(d.as_static() for d in suh.output.shape) != (k,) or tuple(d.as_static() for d in svh.output.shape) != (n,):
        return None

    from emmy.serving.native.exl3 import gemv_source, gemv_symbol  # noqa: PLC0415

    c_fp32 = out.dtype.name == "f32"
    source = gemv_source(op.bits, op.codebook, c_fp32=c_fp32, residual=op.residual, compute_capability=cc)
    kernel = gemv_symbol(op.bits, c_fp32=c_fp32, residual=op.residual)
    grid = max(1, 4 * ctx.sm_count)
    rows = _rows_per(size_k=k, size_n=n, grid=grid, residual=op.residual)
    ksplit = math.ceil((k // 16) / rows)
    workspace_ints = _SQ_WS_RESERVED + ksplit * n * (2 if op.residual else 1)
    stage_words = 8 * 4 * 16 * op.bits if cc >= (8, 0) and op.bits in (3, 5, 7) else 0
    smem = rows * 16 * 2 + rows * 16 * 4 * (2 if op.residual else 1) + stage_words * 4 + 2 * 128 * 4
    if smem > ctx.max_dynamic_smem:
        return None

    frag = open_fragment(match.graph, [x, codes, suh, svh])
    workspace = Tensor(f"{out.name}_exl3_ws", (workspace_ints,), "i32")
    native = frag.add_node(
        op=CudaOp(
            kernel_source=source,
            kernel_name=kernel,
            arg_order=(
                x.id,
                codes.id,
                out.name,
                "size_m",
                "size_k",
                "size_n",
                workspace.name,
                suh.id,
                x.id,
                svh.id,
            ),
            grid=((grid,), (1,), (1,)),
            block=((256,), (1,), (1,)),
            smem_bytes=smem,
            dynamic_smem=True,
            zero_outputs=(workspace.name,),
            scalar_args=(("size_m", "i32", 1), ("size_k", "i32", k), ("size_n", "i32", n)),
            comment=f"native EXL3 K{op.bits}/cb{op.codebook} M1 {k}x{n}",
        ),
        inputs=[x, codes, suh, svh],
        outputs=(Tensor(out.name, out.shape, out.dtype), workspace),
        node_id=out.name,
    )
    frag.outputs = [native]
    return frag


def _generic_fragment(match: Match, x: Node, codes: Node, suh: Node, svh: Node, out: Tensor) -> Graph:
    from emmy.compiler.loader.trellis import spell_factored_linear  # noqa: PLC0415

    op: Exl3GemvOp = match.root.op
    frag = open_fragment(match.graph, [x, codes, suh, svh])
    frag.outputs = [
        spell_factored_linear(
            frag,
            codes.id,
            suh.id,
            svh.id,
            cb=op.codebook,
            weight_shape=op.weight_shape,
            x=x.id,
            bias=None,
            out=out,
            weight_name=f"{out.name}_decoded",
            prefer_native=False,
        )
    ]
    return frag


def rewrite(match: Match, inp_x: Node, inp_codes: Node, inp_suh: Node, inp_svh: Node, out: Tensor, ctx) -> Graph:
    return _native_fragment(match, inp_x, inp_codes, inp_suh, inp_svh, out, ctx) or _generic_fragment(
        match, inp_x, inp_codes, inp_suh, inp_svh, out
    )
