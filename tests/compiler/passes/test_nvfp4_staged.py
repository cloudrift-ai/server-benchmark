"""The packed byte-slab stage — an NVFP4 weight staged as raw bytes plus its block scales.

A packed-pair weight reaches the tensor cores as a computed B: the checkpoint holds 4-bit codes
two per byte plus one e4m3 scale per 16 of them, and the birth-time speller rewrites the constant
into the decode algebra. The generic reading of that cone runs it per slab cell through the sync
compute-fill, which is correct and moves 16-bit weights. This is the specialized reading: the bits
copy VERBATIM into a byte slab (half the traffic of a 16-bit one), the block scales decode once per
block into a small companion slab, and the fragment drain does the decode-and-scale per element.

The scope is deliberately narrow — cp.async, an N-major packed weight of 16-value blocks under a
16-bit atom whose K step is that same 16 — and everything outside it declines back to the generic
reading rather than lowering something the drain is not written for.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BF16, F8E4M3, F16, F32, F4E2M1x2
from emmy.compiler.graph import Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, CastExpr, Literal, Var
from emmy.compiler.ir.schedule import Stage, TilePlan, Workers
from emmy.compiler.ir.stmt import Assign, Body, Load
from emmy.compiler.ir.tile import Channel, Fold
from emmy.compiler.pipeline.passes.lowering._addr import BYTE_SLAB_PAD
from emmy.compiler.pipeline.passes.lowering.tile._atomize import match_packed_b_node
from emmy.compiler.pipeline.passes.lowering.tile._legality import resolve_warp_stage
from tests.compiler.helpers import requires_cuda

K16 = "mma_m16n8k16_f16_f32"
K16_BF16 = "mma_m16n8k16_bf16_f32"
K32 = "mma_m16n8k32_e4m3_f32"


def _lit(v: int):
    return Literal(v, "int")


def _packed_cone(k: int, *, block: int = 16, k_last: bool = True) -> Fold:
    """The NVFP4 speller's decode cone as the tile lowering sees it — flat reshape arithmetic over
    the checkpoint's three tensors, exactly the shape the whole-graph lowering produces (verified
    against it): a block-scale byte decoded and multiplied by the per-tensor scale, and a packed
    byte copied to an index that gathers the code pair's value.

    ``k_last=False`` swaps the bits load's index so the packed axis is the ROW — the layout the
    drain cannot read."""
    n, kv = Var("n"), Var("k")
    flat = BinaryExpr("+", BinaryExpr("*", n, _lit(k)), kv)
    sblock = BinaryExpr("%", BinaryExpr("/", flat, _lit(block)), _lit(k // block))
    byte = BinaryExpr("%", BinaryExpr("/", flat, _lit(2)), _lit(k // 2))
    pair = BinaryExpr("%", flat, _lit(2))
    bits_index = (n, byte) if k_last else (byte, n)
    body = (
        Load(name="in0", input="w_scale_2", index=(_lit(0), _lit(0)), dtype=None),
        Load(name="in1", input="w_scale_bits", index=(n, sblock), dtype=None),
        Load(name="in2", input="w_bits", index=bits_index, dtype=None),
        Assign(name="v0", op="from_f8e4m3", args=("in1",)),
        Assign(name="v1", op="copy", args=("in2",)),
        Assign(name="v2", op="multiply", args=("in0", "v0")),
        Load(name="in3", input="w_f4_pairs", index=(CastExpr(dtype="int", expr=Var("v1")), pair), dtype=None),
        Assign(name="v3", op="multiply", args=("in3", "v2")),
    )
    return Fold.projection(body=Body(body))


def _node(*, m=512, n=4096, k=4096, block=16, a_dtype=F16, k_last=True):
    """``x[m, k] @ decode(packed w)`` — a contraction whose B is the packed decode cone."""
    axes = (Axis("m", Dim(m)), Axis("n", Dim(n)))
    a = Load(name="a", input="x", index=(Var("m"), Var("k")), dtype=a_dtype)
    node = Fold.contraction(k_axis=Axis("k", Dim(k)), a=a, channels=(Channel(b=_packed_cone(k, block=block, k_last=k_last), acc="acc"),))
    inputs = {
        "x": Tensor("x", (m, k), a_dtype),
        "w_bits": Tensor("w_bits", (n, k // 2) if k_last else (k // 2, n), F4E2M1x2),
        "w_scale_bits": Tensor("w_scale_bits", (n, k // block), F8E4M3),
        "w_scale_2": Tensor("w_scale_2", (1, 1), F32),
        "w_f4_pairs": Tensor("w_f4_pairs", (256, 2), F16),
    }
    return node, inputs, axes


def _tile(atom: str, spec: str, work: str, axes):
    return TilePlan.parse(f"{atom}/{spec}", Workers.parse(work)).at(*axes)


# ===================================================================
# The node recognizer — one question, asked by three consumers
# ===================================================================


def test_match_packed_b_node_reads_the_cone_off_the_contraction():
    node, inputs, _axes = _node()
    packed = match_packed_b_node(node, inputs)
    assert packed is not None
    assert packed.bits.input == "w_bits" and packed.table.input == "w_f4_pairs"
    assert packed.factor == "v2" and packed.block == 16


def test_match_packed_b_node_declines_a_materialized_b():
    """An ordinary matmul has no cone to recognize — the packed reading never applies to it."""
    a = Load(name="a", input="x", index=(Var("m"), Var("k")), dtype=F16)
    b = Load(name="wb", input="w", index=(Var("n"), Var("k")), dtype=F16)
    node = Fold.contraction(k_axis=Axis("k", Dim(4096)), a=a, channels=(Channel(b=b, acc="acc"),))
    inputs = {"x": Tensor("x", (512, 4096), F16), "w": Tensor("w", (4096, 4096), F16)}
    assert match_packed_b_node(node, inputs) is None


def test_match_packed_b_node_declines_a_computed_a():
    """A computed A beside the packed B is outside the staged shape: A rides the copy transport."""
    node, inputs, _axes = _node()
    coned = Fold.contraction(k_axis=node.axis, a=_packed_cone(4096), channels=node.channels)
    assert match_packed_b_node(coned, inputs) is None


# ===================================================================
# The stage resolver
# ===================================================================


@pytest.mark.parametrize(("atom", "a_dtype"), [(K16, F16), (K16_BF16, BF16)])
def test_packed_b_resolves_the_cp_async_byte_slab(atom, a_dtype):
    """cp.async resolves and carries the chunk's LOGICAL K width — the byte halving is the slab's
    geometry, not the schedule's K step. Both 16-bit float fragments hold every e2m1 value
    exactly, so both resolve."""
    node, inputs, axes = _node(a_dtype=a_dtype)
    tile = _tile(atom, "f2x2/k2", "w1x4", axes)
    for spec in ("d1/cp", "d2/cp", "d2/cp/p2"):
        st = resolve_warp_stage(node, tile, Stage.parse(spec), 100 * 1024, inputs)
        assert st is not None, spec
        assert st.bk_elems == tile.bk * 16 and st.transport == "cp.async"


def test_packed_b_declines_sync_and_tma():
    """Only cp.async is built. A ``sync`` or ``tma`` spelling keeps the generic computed-B
    reading, whose compute fill evaluates the same cone."""
    node, inputs, axes = _node()
    tile = _tile(K16, "f2x2/k2", "w1x4", axes)
    for spec in ("d1/sync", "d2/tma", "d1/cp/split"):
        assert resolve_warp_stage(node, tile, Stage.parse(spec), 100 * 1024, inputs) is None, spec


def test_packed_b_budget_carries_the_row_pad_and_the_scale_slab():
    """The budget is the A slab plus the PADDED byte rows per ring slot, plus one single-buffer
    scale slab on top — the scale fill is compute, so ringing it buys no overlap."""
    node, inputs, axes = _node()
    tile = _tile(K16, "f2x2/k2", "w1x4", axes)
    bk_elems = tile.bk * 16
    slot = tile.m.tile * bk_elems * 2 + tile.n.tile * (bk_elems // 2 + BYTE_SLAB_PAD)
    scale = tile.n.tile * (bk_elems // 16) * 2
    assert resolve_warp_stage(node, tile, Stage.parse("d2/cp"), scale + 2 * slot, inputs).depth == 2
    assert resolve_warp_stage(node, tile, Stage.parse("d2/cp"), scale + 2 * slot - 1, inputs).depth == 1
    assert resolve_warp_stage(node, tile, Stage.parse("d1/cp"), scale + slot - 1, inputs) is None
    # Sizing the slot without the pad, or forgetting the scale slab, would each admit this budget.
    dense = tile.m.tile * bk_elems * 2 + tile.n.tile * (bk_elems // 2)
    assert resolve_warp_stage(node, tile, Stage.parse("d1/cp"), dense, inputs) is None


def test_packed_b_declines_a_k_strided_layout():
    """The drain reads N-major rows. A packed weight stored K-major has no fragment loader here."""
    node, inputs, axes = _node(k_last=False)
    tile = _tile(K16, "f2x2/k2", "w1x4", axes)
    assert resolve_warp_stage(node, tile, Stage.parse("d2/cp"), 100 * 1024, inputs) is None


def test_packed_b_declines_a_block_the_drain_does_not_read():
    """The drain's scale column is ``K >> 4``; a 32-value block would read the wrong scale."""
    node, inputs, axes = _node(block=32)
    tile = _tile(K16, "f2x2/k2", "w1x4", axes)
    assert match_packed_b_node(node, inputs).block == 32
    assert resolve_warp_stage(node, tile, Stage.parse("d2/cp"), 100 * 1024, inputs) is None


def test_packed_b_declines_a_non_16_bit_atom():
    """The value table and the scale multiply are 16-bit floats. The fp8 atoms are neither."""
    node, inputs, axes = _node()
    assert resolve_warp_stage(node, _tile(K32, "f2x2/k2", "w1x4", axes), Stage.parse("d2/cp"), 100 * 1024, inputs) is None


def test_packed_b_declines_when_a_and_the_atom_disagree():
    """A is byte-copied into the atom's own slab; a bf16 A under an f16 atom would deposit the
    wrong bits, and the two dtypes are the same width so nothing else catches it."""
    node, inputs, axes = _node(a_dtype=BF16)
    assert resolve_warp_stage(node, _tile(K16, "f2x2/k2", "w1x4", axes), Stage.parse("d2/cp"), 100 * 1024, inputs) is None


def test_packed_b_declines_a_mismatched_a():
    """A is byte-copied into the atom's own slab, so it must already carry the atom's dtype."""
    node, inputs, axes = _node(a_dtype=F32)
    tile = _tile(K16, "f2x2/k2", "w1x4", axes)
    assert resolve_warp_stage(node, tile, Stage.parse("d2/cp"), 100 * 1024, inputs) is None


def test_packed_b_declines_a_byte_row_under_sixteen():
    """A byte row of ``bk_elems / 2`` must stay 16-divisible: the fill copies 16 B chunks and a
    chunk never straddles a row. ``k1`` leaves 8 bytes."""
    node, inputs, axes = _node()
    assert resolve_warp_stage(node, _tile(K16, "f2x2/k1", "w1x4", axes), Stage.parse("d2/cp"), 100 * 1024, inputs) is None


# ===================================================================
# The schedule's offer
# ===================================================================


def _rows(node, inputs, axes, pins=None):
    """The ``STAGE`` rows the schedule offers this node at a warp tile, as resolved spellings."""
    from emmy.compiler.context import Context
    from emmy.compiler.ir.stmt import Write
    from emmy.compiler.ir.tile import Placement, TileOp
    from emmy.compiler.ir.tile.ir import Store
    from emmy.compiler.pipeline.passes.lowering.tile import _schedule
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    write = Write(output="y", index=(Var("m"), Var("n")), value="acc")
    op = TileOp(
        op=Fold.projection(body=Body(()), operands=(node,)),
        name="y",
        place=Placement(free=axes),
        inputs=inputs,
        stores=(Store(write=write),),
    )
    term = _schedule._Term(op, op.place.on_grid(), Context.from_target((8, 9)))
    tile = _tile(K16, "f2x2/k2", "w1x4", axes)
    with pinned_knobs(pins or {}):
        return [st.spell() for st in _schedule._computed_values(term, node, tile)]


def test_the_offer_puts_the_byte_slab_beside_the_compute_fill():
    """Both readings are fork siblings: the compute-fill depths first (the conservative option,
    which every computed cone has), then the byte-slab transports."""
    node, inputs, axes = _node()
    rows = _rows(node, inputs, axes)
    assert any("sync" in r for r in rows), rows
    assert any("cp" in r for r in rows), rows
    assert "sync" in rows[0], rows


def test_a_generic_cone_offers_only_the_compute_fill():
    """A cone the byte slab declines is unchanged: its rows are the compute-fill depths alone."""
    node, inputs, axes = _node(block=32)
    assert all("cp" not in r for r in _rows(node, inputs, axes))


def test_a_cp_pin_names_the_byte_slab_and_a_sync_pin_the_compute_fill():
    node, inputs, axes = _node()
    assert _rows(node, inputs, axes, {"STAGE": "d2/cp"}) == ["d2/cp"]
    assert all("cp" not in r for r in _rows(node, inputs, axes, {"STAGE": "d2/sync"}))


# ===================================================================
# The whole lowering: the spelled checkpoint through to CUDA source
# ===================================================================


def _nvfp4_matmul_graph(tmp_path, *, m, n, k, dtype="f16"):
    """``x[m, k] @ dequant(w)ᵀ`` over a synthetic NVFP4 checkpoint; returns the graph + weights.

    ``dtype`` is the element type the trace promises — the weight constant's, the activation's and
    the output's, exactly as a checkpoint's own config would set it. Qwen models trace bf16."""
    import torch

    from emmy.compiler.graph import Graph
    from emmy.compiler.ir.base import ConstantOp, InputOp
    from emmy.compiler.ir.frontend.ir import MatmulOp, TransposeOp
    from emmy.compiler.loader.quant import spell_quantized_constants
    from tests.compiler.loader.test_quant import _FP4_MODELOPT_QC, _fp8_tensor, _write_checkpoint

    rng = np.random.default_rng(7)
    packed = rng.integers(0, 256, (n, k // 2)).astype(np.uint8)
    scale_bits = rng.integers(0, 0x7F, (n, k // 16)).astype(np.uint8)
    s2 = np.array(0.25, dtype=np.float32)
    _write_checkpoint(
        tmp_path,
        {
            "layer.weight": torch.from_numpy(packed),
            "layer.weight_scale": _fp8_tensor(scale_bits),
            "layer.weight_scale_2": torch.tensor(float(s2), dtype=torch.float32),
        },
        quant_config={**_FP4_MODELOPT_QC, "ignore": ["lm_head"]},
    )
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (m, k), dtype), node_id="x")
    w = g.add_node(
        op=ConstantOp(name="w", source_path="layer.weight", source_shape=(n, k), source_dtype=dtype),
        inputs=[],
        output=Tensor("w", (n, k), dtype),
        node_id="w",
    )
    wt = g.add_node(op=TransposeOp(axes=(1, 0)), inputs=[w], output=Tensor("wt", (k, n), dtype))
    y = g.add_node(op=MatmulOp(), inputs=["x", wt], output=Tensor("y", (m, n), dtype), node_id="y")
    g.inputs, g.outputs = ["x"], [y]
    assert spell_quantized_constants(g, str(tmp_path)) == 1
    return g, (packed, scale_bits, s2)


def _packed_pins(dtype="f16"):
    atom = K16 if dtype == "f16" else K16_BF16
    return {"TILE": f"{atom}/f2x2/k2", "WORK": "w1x4", "REDUCE": "", "STAGE": "d2/cp"}


PACKED_PINS = _packed_pins()


def test_the_spelled_checkpoint_lowers_to_the_packed_drain(tmp_path):
    """The whole path, structurally: a spelled NVFP4 matmul under the packed pins emits three
    slabs — the f16 A tile, the raw packed bytes, the decoded block scales — and a drain that
    reads the last two together. No decoded weight is ever materialized."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    g, _ = _nvfp4_matmul_graph(tmp_path, m=32, n=128, k=128)
    with pinned_knobs(PACKED_PINS):
        lowered = Pipeline.build(CUDA_PASSES).run(g, ctx=Context.from_target((8, 9)))
    src = next(s for node in lowered.nodes.values() if (s := getattr(node.op, "kernel_source", None)))
    assert "emmy_mma_load_b_smem_trans_f4s_f16" in src
    assert "EMMY_F4_LUT_F16" in src
    assert "EMMY_F4_LUT_BF16" not in src, "an f16 kernel must not carry the bf16 drain"
    assert "emmy_mma_load_b_smem_trans_f8_f16" not in src, "the fp8 drain helpers must not ride along"
    # tile_n=64 rows of (bk_elems/2 = 16) bytes + the 16 B pad, two ring slots; the scale slab is
    # 64 rows of bk = 2, single-buffer.
    assert "unsigned char _b_smem[4096]" in src
    assert "__half _bs_smem[128]" in src


def test_a_bf16_checkpoint_lowers_to_the_bf16_packed_drain(tmp_path):
    """The same path at bf16, which is what Qwen models trace: the bf16 drain, its own value
    table, and a bf16 scale slab. The value table differs only in its constants — every e2m1
    value is exact in bf16 too."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    g, _ = _nvfp4_matmul_graph(tmp_path, m=32, n=128, k=128, dtype="bf16")
    with pinned_knobs(_packed_pins("bf16")):
        lowered = Pipeline.build(CUDA_PASSES).run(g, ctx=Context.from_target((8, 9)))
    src = next(s for node in lowered.nodes.values() if (s := getattr(node.op, "kernel_source", None)))
    assert "emmy_mma_load_b_smem_trans_f4s_bf16" in src
    assert "EMMY_F4_LUT_BF16" in src and "EMMY_F4_LUT_F16" not in src
    assert "__nv_bfloat16 _bs_smem[128]" in src
    # 1.0 is 0x3F80 in bf16 and 0x3C00 in f16 — the table really is the other format's.
    assert "0x3F80" in src


@requires_cuda
@pytest.mark.parametrize(
    ("dtype", "m", "n", "k", "tol"),
    [("f16", 32, 128, 128, 1e-3), ("f16", 4, 2048, 2048, 1e-3), ("bf16", 32, 128, 128, 6e-3)],
)
@pytest.mark.xdist_group("cuda")
def test_the_packed_drain_matches_the_decoded_oracle(tmp_path, dtype, m, n, k, tol):
    """Numerical parity on the device: the packed kernel equals ``x @ dequantize_nvfp4(w)ᵀ``.

    Both a compute-shaped and a decode-shaped matmul at f16, since the decode shape is where the
    weight traffic dominates, plus the bf16 fragment. Each bound is roughly 3x the measured error
    on a 4090 (f16 2.5e-4, bf16 2.1e-3); bf16's is the looser one because bf16 carries 8 mantissa
    bits against f16's 11 — the format's own precision, not the drain's."""
    import torch

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.loader.quant import dequantize_nvfp4
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    g, (packed, scale_bits, s2) = _nvfp4_matmul_graph(tmp_path, m=m, n=n, k=k, dtype=dtype)
    rng = np.random.default_rng(11)
    # numpy has no bf16, so a bf16 activation crosses as its uint16 bits and the output comes back
    # the same way — the convention every bf16 device test here follows.
    if dtype == "bf16":
        xt = (torch.from_numpy(rng.standard_normal((m, k))) * 0.05).to(torch.bfloat16)
        x, x_ref = xt.view(torch.uint16).numpy(), xt.float().numpy()
    else:
        x = (rng.standard_normal((m, k)) * 0.05).astype(np.float16)
        x_ref = x.astype(np.float32)

    backend = CudaBackend()
    with pinned_knobs(_packed_pins(dtype)):
        compiled = backend.compile(g)
    src = next(s for node in compiled.nodes.values() if (s := getattr(node.op, "kernel_source", None)))
    assert f"emmy_mma_load_b_smem_trans_f4s_{dtype}" in src, "the packed pins did not reach the byte-slab drain"

    data = bind_constants(compiled, {"layer.weight": packed, "layer.weight_scale": scale_bits, "layer.weight_scale_2": s2})
    result, _ = backend.run(compiled, input_data={**data, "x": x})
    out = result.outputs[compiled.outputs[0]].reshape(m, n)
    if dtype == "bf16":
        y = torch.from_numpy(np.asarray(out).astype(np.uint16)).view(torch.bfloat16).float().numpy()
    else:
        y = out.astype(np.float32)

    ref = x_ref @ dequantize_nvfp4(packed, scale_bits, s2).T
    denom = max(float(np.abs(ref).max()), 1e-9)
    assert float(np.abs(y - ref).max()) / denom < tol
