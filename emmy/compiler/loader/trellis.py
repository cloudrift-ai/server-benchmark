"""Birth-time spelling of trellis reconstruction into format-neutral tensor IR.

This is loader/frontend code: it knows the checkpoint representation, but every node it
emits is an ordinary range, cast/bitcast, elementwise operation, gather/index map, layout
operation, or matmul. The generic constant folder removes the whole static cone before
Loop IR, so no checkpoint-format operation enters lowering, scheduling, or codegen.
"""

from __future__ import annotations

import numpy as np

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, placeholder
from emmy.compiler.ir.frontend.ir import MatmulOp, ReshapeOp, SliceOp, TransposeOp
from emmy.compiler.ir.tensor.ir import (
    BitcastOp,
    CastOp,
    ElementwiseOp,
    GatherOp,
    IndexMapOp,
    IndexSource,
    RangeOp,
)
from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to


class _Builder:
    """Small node-construction vocabulary for one static algebra spelling."""

    def __init__(self, graph: Graph, prefix: str) -> None:
        self.graph = graph
        self.prefix = prefix

    def node(self, nid: str):
        return self.graph.nodes[nid]

    def scalar(self, suffix: str, value: float | int, dtype: str) -> str:
        return self.graph.add_node(
            op=ConstantOp(name=f"{self.prefix}_{suffix}", value=value),
            inputs=[],
            output=Tensor(f"{self.prefix}_{suffix}", (1,), dtype),
        )

    def cast(self, src: str, dtype: str, suffix: str) -> str:
        return self.graph.add_node(
            op=CastOp(dtype=dtype),
            inputs=[src],
            output=Tensor(f"{self.prefix}_{suffix}", self.node(src).output.shape, dtype),
        )

    def bitcast(self, src: str, dtype: str, suffix: str) -> str:
        return self.graph.add_node(
            op=BitcastOp(dtype=dtype),
            inputs=[src],
            output=Tensor(f"{self.prefix}_{suffix}", self.node(src).output.shape, dtype),
        )

    def reshape(self, src: str, shape: tuple[int, ...], suffix: str) -> str:
        return self.graph.add_node(
            op=ReshapeOp(shape=shape),
            inputs=[src],
            output=Tensor(f"{self.prefix}_{suffix}", shape, self.node(src).output.dtype),
        )

    def ew(self, op: str, inputs: tuple[str, ...], shape: tuple[int, ...], dtype: str, suffix: str) -> str:
        expanded = [broadcast_to(self.graph, src, shape).id for src in inputs]
        return self.graph.add_node(
            op=ElementwiseOp(op=op),
            inputs=expanded,
            output=Tensor(f"{self.prefix}_{suffix}", shape, dtype),
        )

    def binary(self, op: str, lhs: str, rhs: str, shape: tuple[int, ...], dtype: str, suffix: str) -> str:
        return self.ew(op, (lhs, rhs), shape, dtype, suffix)

    def unary(self, op: str, src: str, dtype: str, suffix: str) -> str:
        shape = tuple(self.node(src).output.shape)
        return self.ew(op, (src,), shape, dtype, suffix)

    def range(self, stop: int, suffix: str) -> str:
        return self.graph.add_node(
            op=RangeOp(stop=stop, dtype="i32"),
            inputs=[],
            output=Tensor(f"{self.prefix}_{suffix}", (stop,), "i32"),
        )

    def indexed(self, src: str, shape: tuple[int, ...], coords: tuple, suffix: str) -> str:
        return self.graph.add_node(
            op=IndexMapOp(out_shape=shape, sources=(IndexSource(input_idx=0, coord_map=coords),)),
            inputs=[src],
            output=Tensor(f"{self.prefix}_{suffix}", shape, self.node(src).output.dtype),
        )

    def strided(self, src: str, shape: tuple[int, ...], stride: int, offset: int, suffix: str) -> str:
        last = len(shape) - 1
        coords = tuple(placeholder(i) for i in range(last)) + (placeholder(last) * Literal(stride, "int") + Literal(offset, "int"),)
        return self.indexed(src, shape, coords, suffix)


def _unpack_windows(b: _Builder, codes: str) -> tuple[str, int, int]:
    """Packed int16 code words → one circular uint32 window per tile step."""
    code_shape = tuple(d.as_static() for d in b.node(codes).output.shape)
    kt, nt, packed = code_shape
    K = packed // 16
    word_shape = (kt, nt, 8 * K)
    u16 = b.bitcast(codes, "u16", "codes_u16")
    even = b.cast(b.strided(u16, word_shape, 2, 0, "word_even"), "u32", "word_even_u32")
    odd = b.cast(b.strided(u16, word_shape, 2, 1, "word_odd"), "u32", "word_odd_u32")
    odd_hi = b.binary("left_shift", odd, b.scalar("shift16", 16, "u32"), word_shape, "u32", "word_odd_hi")
    words = b.binary("bitwise_or", even, odd_hi, word_shape, "u32", "words")

    shape = (kt, nt, 256)
    step = placeholder(2)
    modulus = 256 * K
    # Adding one full modulus before ``%`` makes the dividend non-negative
    # for every K/step, so Python evaluation and rendered C agree exactly.
    start = ((step + 1) * K - 16 + modulus) % modulus
    word_idx = BinaryExpr("//", start, Literal(32, "int"))
    next_idx = (word_idx + 1) % (8 * K)
    leading = (placeholder(0), placeholder(1))
    lo = b.indexed(words, shape, leading + (word_idx,), "window_lo")
    hi = b.indexed(words, shape, leading + (next_idx,), "window_hi")

    shift_step = b.range(256, "shift_step")
    one = b.scalar("shift_one", 1, "i32")
    start = b.binary("add", shift_step, one, (256,), "i32", "shift_step1")
    start = b.binary("multiply", start, b.scalar("shift_k", K, "i32"), (256,), "i32", "shift_end")
    start = b.binary("subtract", start, b.scalar("shift_bias", 16, "i32"), (256,), "i32", "shift_start")
    start = b.binary("add", start, b.scalar("shift_modulus", modulus, "i32"), (256,), "i32", "shift_start_nonnegative")
    intra = b.binary("remainder", start, b.scalar("shift_word", 32, "i32"), (256,), "i32", "shift_intra")
    shift = b.binary("subtract", b.scalar("shift_top", 48, "i32"), intra, (256,), "i32", "window_shift")
    lo64, hi64 = b.cast(lo, "u64", "window_lo_u64"), b.cast(hi, "u64", "window_hi_u64")
    joined_hi = b.binary(
        "left_shift",
        lo64,
        b.scalar("shift32_u64", 32, "u64"),
        shape,
        "u64",
        "window_join_hi",
    )
    joined = b.binary("bitwise_or", joined_hi, hi64, shape, "u64", "window_join")
    shifted = b.binary("right_shift", joined, b.cast(shift, "u64", "window_shift_u64"), shape, "u64", "window_shifted")
    masked = b.binary("bitwise_and", shifted, b.scalar("mask16_u64", 0xFFFF, "u64"), shape, "u64", "windows_u64")
    return b.cast(masked, "u32", "windows"), kt, nt


def _codebook(b: _Builder, windows: str, cb: int, shape: tuple[int, ...]) -> str:
    """Computed uint32 codebook algebra → fp16 values."""
    if cb == 2:
        x = b.binary("multiply", windows, b.scalar("cb_mul", 0x83DCD12D, "u32"), shape, "u32", "cb_x")
        mask = b.scalar("byte_mask", 0xFF, "u32")
        byte_sum = b.binary("bitwise_and", x, mask, shape, "u32", "cb_byte0")
        for bits in (8, 16, 24):
            shifted = b.binary(
                "right_shift",
                x,
                b.scalar(f"byte_shift{bits}", bits, "u32"),
                shape,
                "u32",
                f"cb_shift{bits}",
            )
            byte = b.binary("bitwise_and", shifted, mask, shape, "u32", f"cb_byte{bits // 8}")
            byte_sum = b.binary("add", byte_sum, byte, shape, "u32", f"cb_sum{bits // 8}")
        biased = b.binary("add", byte_sum, b.scalar("cb_bias", 0x6400, "u32"), shape, "u32", "cb_packed_half")
        h = b.bitcast(b.cast(biased, "u16", "cb_packed_half_u16"), "f16", "cb_h")
        kmul = b.bitcast(b.scalar("cb_kmul_bits", 0x1EEE, "u16"), "f16", "cb_kmul")
        kadd = b.bitcast(b.scalar("cb_kadd_bits", 0xC931, "u16"), "f16", "cb_kadd")
        h64 = b.cast(h, "f64", "cb_h64")
        prod = b.binary("multiply", h64, b.cast(kmul, "f64", "cb_kmul64"), shape, "f64", "cb_prod")
        value64 = b.binary("add", prod, b.cast(kadd, "f64", "cb_kadd64"), shape, "f64", "cb_value64")
        return b.cast(value64, "f16", "cb_value")

    multiplier = 89226354 if cb == 0 else 0xCBAC1FED
    x = b.binary("multiply", windows, b.scalar("cb_mul", multiplier, "u32"), shape, "u32", "cb_product")
    if cb == 0:
        x = b.binary("add", x, b.scalar("cb_add", 64248484, "u32"), shape, "u32", "cb_affine")
    masked = b.binary("bitwise_and", x, b.scalar("cb_mask", 0x8FFF8FFF, "u32"), shape, "u32", "cb_masked")
    x = b.binary("bitwise_xor", masked, b.scalar("cb_xor", 0x3B603B60, "u32"), shape, "u32", "cb_bits")
    lo = b.binary("bitwise_and", x, b.scalar("half_mask", 0xFFFF, "u32"), shape, "u32", "cb_lo")
    lo = b.bitcast(b.cast(lo, "u16", "cb_lo_u16"), "f16", "cb_lo_half")
    hi = b.binary("right_shift", x, b.scalar("half_shift", 16, "u32"), shape, "u32", "cb_hi")
    hi = b.bitcast(b.cast(hi, "u16", "cb_hi_u16"), "f16", "cb_hi_half")
    return b.binary("add", lo, hi, shape, "f16", "cb_value")


def _tile_order(b: _Builder, values: str, kt: int, nt: int) -> str:
    """MMA-fragment step order → row-major tiled matrix, derived from coordinates."""
    pos = b.range(256, "tile_pos")

    def i(suffix: str, value: int) -> str:
        return b.scalar(f"i32_{value}_{suffix}", value, "i32")

    row = b.binary("floor_divide", pos, i("row", 16), (256,), "i32", "tile_row")
    col = b.binary("remainder", pos, i("col", 16), (256,), "i32", "tile_col")
    row8 = b.binary("remainder", row, i("row8", 8), (256,), "i32", "tile_row8")
    lane_lo = b.binary("floor_divide", row8, i("lane2", 2), (256,), "i32", "tile_lane_lo")
    col8 = b.binary("remainder", col, i("col8", 8), (256,), "i32", "tile_col8")
    lane_hi = b.binary("multiply", col8, i("lane4", 4), (256,), "i32", "tile_lane_hi")
    lane = b.binary("add", lane_hi, lane_lo, (256,), "i32", "tile_lane")
    j0 = b.binary("bitwise_and", row, i("j0", 1), (256,), "i32", "tile_j0")
    row3 = b.binary("right_shift", row, i("row3", 3), (256,), "i32", "tile_row3")
    j1bit = b.binary("bitwise_and", row3, i("j1bit", 1), (256,), "i32", "tile_j1bit")
    j1 = b.binary("multiply", j1bit, i("j1", 2), (256,), "i32", "tile_j1")
    j2bit = b.binary("floor_divide", col, i("j2bit", 8), (256,), "i32", "tile_j2bit")
    j2 = b.binary("multiply", j2bit, i("j2", 4), (256,), "i32", "tile_j2")
    j = b.binary("add", b.binary("add", j0, j1, (256,), "i32", "tile_j01"), j2, (256,), "i32", "tile_j")
    lane8 = b.binary("multiply", lane, i("lane8", 8), (256,), "i32", "tile_lane8")
    step = b.binary("add", lane8, j, (256,), "i32", "tile_step")
    ordered = b.graph.add_node(
        op=GatherOp(axis=-1),
        inputs=[values, step],
        output=Tensor(f"{b.prefix}_tiles_row_major", (kt, nt, 256), "f16"),
    )
    tiles = b.reshape(ordered, (kt, nt, 16, 16), "tiles")
    tiled = b.graph.add_node(
        op=TransposeOp(axes=(0, 2, 1, 3)),
        inputs=[tiles],
        output=Tensor(f"{b.prefix}_hat_grid", (kt, 16, nt, 16), "f16"),
    )
    return b.reshape(tiled, (kt * 16, nt * 16), "hat")


def _hadamard(b: _Builder) -> str:
    """Construct the scaled 128x128 Sylvester Hadamard matrix generically."""
    axis = b.range(128, "had_axis")
    row = b.reshape(axis, (128, 1), "had_row")
    col = b.reshape(axis, (1, 128), "had_col")
    shape = (128, 128)
    bits = b.binary("bitwise_and", row, col, shape, "i32", "had_bits")
    count = b.unary("bitwise_count", bits, "i32", "had_count")
    parity = b.binary("bitwise_and", count, b.scalar("had_parity_mask", 1, "i32"), shape, "i32", "had_parity")
    twice = b.binary(
        "multiply",
        b.cast(parity, "f64", "had_parity64"),
        b.scalar("had_two", 2.0, "f64"),
        shape,
        "f64",
        "had_twice",
    )
    signs = b.binary("subtract", b.scalar("had_one", 1.0, "f64"), twice, shape, "f64", "had_signs")
    scale = b.scalar("had_scale", float(1.0 / np.sqrt(128.0)), "f64")
    return b.binary("multiply", signs, scale, shape, "f64", "had")


def spell_reconstruction(
    graph: Graph,
    codes: str,
    suh: str,
    svh: str,
    *,
    cb: int,
    shape: tuple[int, int],
    name: str,
    dtype,
) -> str:
    """Return the root of the generic reconstruction cone added to ``graph``."""
    b = _Builder(graph, name)
    windows, kt, nt = _unpack_windows(b, codes)
    values = _codebook(b, windows, cb, (kt, nt, 256))
    w_hat = _tile_order(b, values, kt, nt)
    k_pad, n_pad = kt * 16, nt * 16

    had = _hadamard(b)
    w64 = b.cast(w_hat, "f64", "hat64")
    blocks = b.reshape(w64, (k_pad // 128, 128, n_pad), "left_blocks")
    left = graph.add_node(
        op=MatmulOp(),
        inputs=[had, blocks],
        output=Tensor(f"{name}_left", (k_pad // 128, 128, n_pad), "f64"),
    )
    left = b.reshape(left, (k_pad, n_pad), "left_flat")
    right_blocks = b.reshape(left, (k_pad, n_pad // 128, 128), "right_blocks")
    restored = graph.add_node(
        op=MatmulOp(),
        inputs=[right_blocks, had],
        output=Tensor(f"{name}_restored_blocks", (k_pad, n_pad // 128, 128), "f64"),
    )
    restored = b.reshape(restored, (k_pad, n_pad), "restored")
    suh64 = b.reshape(b.cast(suh, "f64", "suh64"), (k_pad, 1), "suh_col")
    svh64 = b.reshape(b.cast(svh, "f64", "svh64"), (1, n_pad), "svh_row")
    restored = b.binary("multiply", restored, suh64, (k_pad, n_pad), "f64", "scaled_in")
    restored = b.binary("multiply", restored, svh64, (k_pad, n_pad), "f64", "scaled_out")
    restored = b.cast(restored, "f16", "decoded_f16")
    transposed = graph.add_node(
        op=TransposeOp(axes=(1, 0)),
        inputs=[restored],
        output=Tensor(f"{name}_decoded_t", (n_pad, k_pad), "f16"),
    )

    n, k = shape
    logical = transposed
    if n != n_pad:
        logical = graph.add_node(
            op=SliceOp(shape=(n, k_pad), dim=0, start=0),
            inputs=[logical],
            output=Tensor(f"{name}_slice_n", (n, k_pad), "f16"),
        )
    if k != k_pad:
        logical = graph.add_node(
            op=SliceOp(shape=(n, k), dim=1, start=0),
            inputs=[logical],
            output=Tensor(f"{name}_slice_k", (n, k), "f16"),
        )
    if dtype.name != "f16":
        logical = graph.add_node(
            op=CastOp(dtype=dtype.name),
            inputs=[logical],
            output=Tensor(name, (n, k), dtype),
        )
    else:
        graph.nodes[logical].outputs = (Tensor(name, (n, k), dtype),)
    return logical
