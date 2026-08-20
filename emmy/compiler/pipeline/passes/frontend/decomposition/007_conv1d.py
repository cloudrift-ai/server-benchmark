"""Decompose Conv1dOp into its two honest forms: im2col for dense, shifted taps for depthwise.

Every read of the input is one ``IndexMapOp`` whose length coordinate is
``l * stride + tap * dilation - padding``. Stride, dilation and padding are therefore all
paid for in the same expression, and nothing else in the rule has to know about them.
Padding specifically is a second source rather than a materialized zero-padded tensor: the
map selects the input where that coordinate is in range and a zero constant where it is
not, which is the same shape of trick ``150_cat`` uses and costs no extra buffer.

The two forms differ in what they do with those reads:

* **Dense** (``groups == 1``) builds the im2col matrix ``(N, C_in * K, L_out)`` in a single
  map — the stacked channel ``ck`` splits into ``tap = ck / C_in`` and ``ci = ck % C_in`` —
  and contracts it against the flattened weight with one ``MatmulOp``. A convolution then
  reaches exactly the GEMM path every other projection takes.
* **Depthwise** (``groups == C_in``) has no reduction across channels at all. im2col would
  build a ``(C_out, C_in * K)`` weight that is zero except for one band per channel, making
  the GEMM do ``C_in`` times the necessary work. It instead scales each of the ``K`` window
  reads by that tap's per-channel weight and sums them — pure elementwise, which also lets
  the chain fuse into its neighbours.
"""

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, placeholder
from emmy.compiler.ir.frontend.ir import Conv1dOp, MatmulOp, ReshapeOp, TransposeOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp, IndexSource
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, single_indexmap

PATTERN = [Pattern("root", Conv1dOp)]


def _static(extent) -> int:
    """Unwrap a channel/tap extent to int. These axes are weight-derived and never symbolic."""
    if isinstance(extent, Dim):
        if not extent.is_static:
            raise NotImplementedError(f"aten.conv1d requires static channel and tap extents, got {extent}")
        return extent.as_static()
    return int(extent)


def _read(frag: Graph, x: Node, *, out_shape: tuple, channel: object, length: object, in_length: int, name: str) -> Node:
    """Read ``x[n, channel, length]`` over ``out_shape``, yielding zero where ``length`` is padded off the end."""
    coords = (placeholder(0), channel, length)
    if isinstance(length, Literal) or in_length <= 0:
        return single_indexmap(frag, x, out_shape=out_shape, coord_map=coords, name=name)

    in_bounds = BinaryExpr("&&", BinaryExpr(">=", length, Literal(0, "int")), BinaryExpr("<", length, Literal(in_length, "int")))
    # Clamp the off-domain coord so the post-fusion unconditional Load stays in range; the
    # select is what actually discards the value.
    clamped = TernaryExpr(cond=in_bounds, if_true=length, if_false=Literal(0, "int"))
    zero_id = frag.add_node(
        op=ConstantOp(name=f"{name}_zero", value=0.0),
        inputs=[],
        output=Tensor(f"{name}_zero", (1,), x.output.dtype),
    )
    nid = frag.add_node(
        op=IndexMapOp(
            out_shape=tuple(out_shape),
            sources=(
                IndexSource(input_idx=0, coord_map=(placeholder(0), channel, clamped), select=in_bounds),
                IndexSource(input_idx=1, coord_map=(Literal(0, "int"),)),
            ),
        ),
        inputs=[x, zero_id],
        output=Tensor(name, tuple(out_shape), x.output.dtype),
    )
    return frag.nodes[nid]


def _length_at(tap: object, *, stride: int, dilation: int, padding: int) -> object:
    """``l * stride + tap * dilation - padding`` for a constant or computed ``tap``."""
    expr = placeholder(2)
    if stride != 1:
        expr = BinaryExpr("*", expr, Literal(stride, "int"))
    shift = tap if isinstance(tap, int) else None
    if shift is not None:
        offset = shift * dilation - padding
        return expr if offset == 0 else BinaryExpr("+", expr, Literal(offset, "int"))
    scaled = tap if dilation == 1 else BinaryExpr("*", tap, Literal(dilation, "int"))
    expr = BinaryExpr("+", expr, scaled)
    return expr if padding == 0 else BinaryExpr("-", expr, Literal(padding, "int"))


def rewrite(match: Match, root: Node, inp_x: Node, inp_w: Node, inp_bias: Node | None, out: Tensor) -> Graph | None:
    graph = match.graph
    op: Conv1dOp = root.op
    frag = open_fragment(graph, [inp_x, inp_w] + ([inp_bias] if inp_bias else []))

    taps = _static(inp_w.output.shape[-1])
    channels = _static(inp_x.output.shape[-2])
    # The input length is only needed to bound a padded read. Without padding every
    # coordinate is in range by construction of L_out, so a symbolic length is fine there.
    in_length = 0
    if op.padding:
        extent = inp_x.output.shape[-1]
        if isinstance(extent, Dim) and not extent.is_static:
            raise NotImplementedError(f"aten.conv1d with padding needs a static input length to bound the pad, got {extent}")
        in_length = _static(extent)
    out_shape = tuple(out.shape)
    geometry = {"stride": op.stride, "dilation": op.dilation, "padding": op.padding}

    if op.groups == 1:
        # One map builds the whole im2col matrix: ck = tap * C_in + ci, tap-major.
        stacked = channels * taps
        stacked_coord = placeholder(1)
        tap_expr = BinaryExpr("/", stacked_coord, Literal(channels, "int"))
        col = _read(
            frag,
            inp_x,
            out_shape=(out_shape[0], stacked, out_shape[-1]),
            channel=BinaryExpr("%", stacked_coord, Literal(channels, "int")),
            length=_length_at(tap_expr, **geometry),
            in_length=in_length,
            name=f"{out.name}_im2col",
        )
        w_shape = tuple(_static(d) for d in inp_w.output.shape)
        w_t = frag.add_node(
            op=TransposeOp(axes=(0, 2, 1)),
            inputs=[inp_w],
            output=Tensor(f"{out.name}_w_t", (w_shape[0], w_shape[2], w_shape[1]), inp_w.output.dtype),
        )
        flat_w = frag.add_node(
            op=ReshapeOp(shape=(w_shape[0], stacked)),
            inputs=[w_t],
            output=Tensor(f"{out.name}_w_flat", (w_shape[0], stacked), inp_w.output.dtype),
        )
        acc: Node | str = frag.add_node(
            op=MatmulOp(),
            inputs=[flat_w, col],
            output=Tensor(f"{out.name}_mm" if inp_bias else out.name, out_shape, out.dtype),
        )
    else:
        acc = None
        for tap in range(taps):
            window = _read(
                frag,
                inp_x,
                out_shape=out_shape,
                channel=placeholder(1),
                length=_length_at(tap, **geometry),
                in_length=in_length,
                name=f"{out.name}_win{tap}",
            )
            # out[n, c, l] scales by weight[c, 0, tap] — channel-indexed, constant in n and l.
            tap_w = single_indexmap(
                frag,
                inp_w,
                out_shape=out_shape,
                coord_map=[placeholder(1), Literal(0, "int"), Literal(tap, "int")],
                name=f"{out.name}_w{tap}",
            )
            scaled = frag.add_node(
                op=ElementwiseOp(op="multiply"),
                inputs=[window, tap_w],
                output=Tensor(f"{out.name}_scaled{tap}", out_shape, out.dtype),
            )
            if acc is None:
                acc = scaled
                continue
            last = tap == taps - 1 and not inp_bias
            acc = frag.add_node(
                op=ElementwiseOp(op="add"),
                inputs=[acc, scaled],
                output=Tensor(out.name if last else f"{out.name}_acc{tap}", out_shape, out.dtype),
            )

    if inp_bias:
        shaped = single_indexmap(frag, inp_bias, out_shape=out_shape, coord_map=[placeholder(1)], name=f"{out.name}_bias")
        acc = frag.add_node(
            op=ElementwiseOp(op="add"),
            inputs=[acc, shaped],
            output=Tensor(out.name, out_shape, out.dtype),
        )

    frag.outputs = [acc.id if isinstance(acc, Node) else acc]
    return frag
