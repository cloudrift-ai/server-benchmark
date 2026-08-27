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
from emmy.compiler.pipeline.passes.lowering.tile._packed import match_packed_b_node
from emmy.compiler.pipeline.passes.lowering.tile._staging import resolve_warp_stage
from tests.compiler.helpers import requires_cuda

K16 = "mma_m16n8k16_f16_f32"
K16_BF16 = "mma_m16n8k16_bf16_f32"
K32 = "mma_m16n8k32_e4m3_f32"


def _lit(v: int):
    return Literal(v, "int")


def _packed_cone(k: int, *, block: int = 16, k_last: bool = True, row: str = "n", prefix: str = "") -> Fold:
    """The NVFP4 speller's decode cone as the tile lowering sees it — flat reshape arithmetic over
    the checkpoint's three tensors, exactly the shape the whole-graph lowering produces (verified
    against it): a block-scale byte decoded and multiplied by the per-tensor scale, and a packed
    byte copied to an index that gathers the code pair's value.

    ``k_last=False`` swaps the bits load's index so the packed axis is the ROW — the layout the
    drain cannot read."""
    n, kv = Var(row), Var("k")
    q = prefix
    flat = BinaryExpr("+", BinaryExpr("*", n, _lit(k)), kv)
    sblock = BinaryExpr("%", BinaryExpr("/", flat, _lit(block)), _lit(k // block))
    byte = BinaryExpr("%", BinaryExpr("/", flat, _lit(2)), _lit(k // 2))
    pair = BinaryExpr("%", flat, _lit(2))
    bits_index = (n, byte) if k_last else (byte, n)
    body = (
        Load(name=f"{q}in0", input=f"{q}w_scale_2", index=(_lit(0), _lit(0)), dtype=None),
        Load(name=f"{q}in1", input=f"{q}w_scale_bits", index=(n, sblock), dtype=None),
        Load(name=f"{q}in2", input=f"{q}w_bits", index=bits_index, dtype=None),
        Assign(name=f"{q}v0", op="from_f8e4m3", args=(f"{q}in1",)),
        Assign(name=f"{q}v1", op="copy", args=(f"{q}in2",)),
        Assign(name=f"{q}v2", op="multiply", args=(f"{q}in0", f"{q}v0")),
        Load(name=f"{q}in3", input=f"{q}w_f4_pairs", index=(CastExpr(dtype="int", expr=Var(f"{q}v1")), pair), dtype=None),
        Assign(name=f"{q}v3", op="multiply", args=(f"{q}in3", f"{q}v2")),
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


def test_match_packed_b_node_admits_a_computed_a():
    """A computed A beside the packed B MATCHES: the packed reading is about B's shape, not A's.

    A serving program fuses the input norm into its projections, so A arrives as a producer cone
    there. Declining that kept the packed weight off the whole serving path. A copies or
    compute-fills exactly as the smem fill decides (:func:`_atom._a_slab_operand`); only B differs."""
    node, inputs, _axes = _node()
    coned = Fold.contraction(k_axis=node.axis, a=_packed_cone(4096), channels=node.channels)
    assert match_packed_b_node(coned, inputs) is not None


# ===================================================================
# The stage resolver
# ===================================================================


def test_a_computed_a_still_resolves_the_byte_slab():
    """The matcher admitting a computed A is only half of it — the resolver must grant the stage too.

    Otherwise the recognizer says yes and the row still never reaches the materializer, which is
    the shape of the bug this replaced: a serving projection sits behind a fused norm, so its A is
    a cone, and the byte slab has to survive that all the way to a resolved stage."""
    node, inputs, axes = _node()
    coned = Fold.contraction(k_axis=node.axis, a=_packed_cone(4096), channels=node.channels)
    assert match_packed_b_node(coned, inputs) is not None
    tile = _tile(K16, "f2x2/k2", "w1x4", axes)
    st = resolve_warp_stage(coned, tile, Stage.parse("d2/smem-async"), 100 * 1024, inputs)
    assert st is not None and st.transport == "smem-async", "a computed A must not lose the byte slab"


@pytest.mark.parametrize(("atom", "a_dtype"), [(K16, F16), (K16_BF16, BF16)])
def test_packed_b_resolves_the_cp_async_byte_slab(atom, a_dtype):
    """cp.async resolves and carries the chunk's LOGICAL K width — the byte halving is the slab's
    geometry, not the schedule's K step. Both 16-bit float fragments hold every e2m1 value
    exactly, so both resolve."""
    node, inputs, axes = _node(a_dtype=a_dtype)
    tile = _tile(atom, "f2x2/k2", "w1x4", axes)
    for spec in ("d1/smem-async", "d2/smem-async", "d2/smem-async/p2"):
        st = resolve_warp_stage(node, tile, Stage.parse(spec), 100 * 1024, inputs)
        assert st is not None, spec
        assert st.bk_elems == tile.bk * 16 and st.transport == "smem-async"


def test_packed_b_declines_the_compute_fill():
    """The compute fill has nothing to copy under, so the byte slab declines it and the cone takes
    the generic reading instead. (The transport ``split`` case that used to sit here went away with
    ``Stage.split``, which existed only for the warp-flash stream.)"""

    node, inputs, axes = _node()
    tile = _tile(K16, "f2x2/k2", "w1x4", axes)
    for spec in ("d1/smem",):
        assert resolve_warp_stage(node, tile, Stage.parse(spec), 100 * 1024, inputs) is None, spec


def test_packed_b_resolves_tma_with_dense_byte_rows():
    """A TMA box deposits dense, so the byte slab carries no row pad — and the budget must size it
    that way, or the stage claims smem it does not use."""
    node, inputs, axes = _node()
    tile = _tile(K16, "f2x2/k2", "w1x4", axes)
    bk_elems = tile.bk * 16
    dense = tile.m.tile * bk_elems * 2 + tile.n.tile * (bk_elems // 2)
    scale = tile.n.tile * (bk_elems // 16) * 2
    st = resolve_warp_stage(node, tile, Stage.parse("d2/smem-tma"), 100 * 1024, inputs)
    assert st is not None and st.transport == "smem-tma" and st.bk_elems == bk_elems
    assert resolve_warp_stage(node, tile, Stage.parse("d2/smem-tma"), scale + 2 * dense, inputs).depth == 2
    assert resolve_warp_stage(node, tile, Stage.parse("d2/smem-tma"), scale + 2 * dense - 1, inputs).depth == 1
    # The cp.async sibling needs strictly more for the same depth — that is exactly the pad.
    assert resolve_warp_stage(node, tile, Stage.parse("d2/smem-async"), scale + 2 * dense, inputs).depth == 1


def test_packed_b_declines_tma_beyond_the_box_limit():
    """Every TMA box dim must fall inside the hardware's 256; a wide N tile does not."""
    node, inputs, axes = _node()
    wide = _tile(K16, "f2x8/k2", "w1x8", axes)  # tile_n = 8*8*8 = 512
    assert wide.n.tile > 256
    assert resolve_warp_stage(node, wide, Stage.parse("d2/smem-tma"), 400 * 1024, inputs) is None


def test_packed_b_budget_carries_the_row_pad_and_the_scale_slab():
    """The budget is the A slab plus the PADDED byte rows per ring slot, plus one single-buffer
    scale slab on top — the scale fill is compute, so ringing it buys no overlap."""
    node, inputs, axes = _node()
    tile = _tile(K16, "f2x2/k2", "w1x4", axes)
    bk_elems = tile.bk * 16
    slot = tile.m.tile * bk_elems * 2 + tile.n.tile * (bk_elems // 2 + BYTE_SLAB_PAD)
    scale = tile.n.tile * (bk_elems // 16) * 2
    assert resolve_warp_stage(node, tile, Stage.parse("d2/smem-async"), scale + 2 * slot, inputs).depth == 2
    assert resolve_warp_stage(node, tile, Stage.parse("d2/smem-async"), scale + 2 * slot - 1, inputs).depth == 1
    assert resolve_warp_stage(node, tile, Stage.parse("d1/smem-async"), scale + slot - 1, inputs) is None
    # Sizing the slot without the pad, or forgetting the scale slab, would each admit this budget.
    dense = tile.m.tile * bk_elems * 2 + tile.n.tile * (bk_elems // 2)
    assert resolve_warp_stage(node, tile, Stage.parse("d1/smem-async"), dense, inputs) is None


def test_packed_b_declines_a_k_strided_layout():
    """The drain reads N-major rows. A packed weight stored K-major has no fragment loader here."""
    node, inputs, axes = _node(k_last=False)
    tile = _tile(K16, "f2x2/k2", "w1x4", axes)
    assert resolve_warp_stage(node, tile, Stage.parse("d2/smem-async"), 100 * 1024, inputs) is None


def test_packed_b_declines_a_block_the_drain_does_not_read():
    """The drain's scale column is ``K >> 4``; a 32-value block would read the wrong scale."""
    node, inputs, axes = _node(block=32)
    tile = _tile(K16, "f2x2/k2", "w1x4", axes)
    assert match_packed_b_node(node, inputs).block == 32
    assert resolve_warp_stage(node, tile, Stage.parse("d2/smem-async"), 100 * 1024, inputs) is None


def test_packed_b_declines_a_non_16_bit_atom():
    """The value table and the scale multiply are 16-bit floats. The fp8 atoms are neither."""
    node, inputs, axes = _node()
    assert resolve_warp_stage(node, _tile(K32, "f2x2/k2", "w1x4", axes), Stage.parse("d2/smem-async"), 100 * 1024, inputs) is None


def test_packed_b_declines_when_a_and_the_atom_disagree():
    """A is byte-copied into the atom's own slab; a bf16 A under an f16 atom would deposit the
    wrong bits, and the two dtypes are the same width so nothing else catches it."""
    node, inputs, axes = _node(a_dtype=BF16)
    assert resolve_warp_stage(node, _tile(K16, "f2x2/k2", "w1x4", axes), Stage.parse("d2/smem-async"), 100 * 1024, inputs) is None


def test_packed_b_declines_a_mismatched_a():
    """A is byte-copied into the atom's own slab, so it must already carry the atom's dtype."""
    node, inputs, axes = _node(a_dtype=F32)
    tile = _tile(K16, "f2x2/k2", "w1x4", axes)
    assert resolve_warp_stage(node, tile, Stage.parse("d2/smem-async"), 100 * 1024, inputs) is None


def test_packed_b_declines_a_byte_row_under_sixteen():
    """A byte row of ``bk_elems / 2`` must stay 16-divisible: the fill copies 16 B chunks and a
    chunk never straddles a row. ``k1`` leaves 8 bytes."""
    node, inputs, axes = _node()
    assert resolve_warp_stage(node, _tile(K16, "f2x2/k1", "w1x4", axes), Stage.parse("d2/smem-async"), 100 * 1024, inputs) is None


# ===================================================================
# The schedule's offer
# ===================================================================


def _rows(node, inputs, axes, pins=None):
    """The ``STAGE`` rows the schedule offers this node at a warp tile, as resolved spellings."""
    from emmy.compiler.context import Context
    from emmy.compiler.ir.stmt import Write
    from emmy.compiler.ir.tile import Placement, TileOp
    from emmy.compiler.ir.tile.ir import OutputSpec
    from emmy.compiler.pipeline.passes.lowering.tile import _schedule
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from emmy.compiler.pipeline.search.space import STAGE

    write = Write(output="y", index=(Var("m"), Var("n")), value="acc")
    op = TileOp(
        op=Fold.projection(body=Body(()), operands=(node,)),
        name="y",
        place=Placement(free=axes),
        inputs=inputs,
        output_specs=(OutputSpec(write=write),),
    )
    ctx = Context.from_target((8, 9))
    tile = _tile(K16, "f2x2/k2", "w1x4", axes)
    with pinned_knobs(pins or {}):
        state = _schedule._state(op, "y", {}, ctx)
        pin = _schedule._pin(STAGE, state.sched.key("STAGE", node))
        return [st.spell() for st in _schedule._fill_options(state, node, tile, pin, ctx.max_dynamic_smem)]


def _transport(row: str) -> str:
    """The transport of a spelled row (``d2/smem-async/p2`` -> ``smem-async``).

    Compared exactly, never by substring: ``smem`` is a prefix of ``smem-async``, so ``in`` would
    call every byte-slab row a compute fill."""
    return row.split("/")[1]


def test_the_offer_puts_the_byte_slab_beside_the_compute_fill():
    """Both readings are fork siblings: the compute-fill depths first (the conservative option,
    which every computed cone has), then the byte-slab transports."""
    node, inputs, axes = _node()
    rows = _rows(node, inputs, axes)
    assert any(_transport(r) == "smem" for r in rows), rows
    assert any(_transport(r).startswith("smem-") for r in rows), rows
    assert _transport(rows[0]) == "smem", rows


def test_the_offer_adds_no_compute_fill_depth_the_fill_did_not_ask_for():
    """A copy move the byte slab declines falls through to the compute fill at that move's depth, so
    a packed node was picking up d3 and d4 fills nobody offers. The fill names its own depths.

    The byte-slab rows themselves carry whatever depths fit the budget — enumerating those is the
    schedule's job, and evidence picks between them."""
    node, inputs, axes = _node()
    fills = {r for r in _rows(node, inputs, axes) if _transport(r) == "smem"}
    assert fills and all(r.startswith(("d1", "d2")) for r in fills), sorted(fills)


def test_a_generic_cone_offers_only_the_compute_fill():
    """A cone the byte slab declines is unchanged: its rows are the compute-fill depths alone."""
    node, inputs, axes = _node(block=32)
    assert all(_transport(r) == "smem" for r in _rows(node, inputs, axes))


def test_a_cp_pin_names_the_byte_slab_and_a_sync_pin_the_compute_fill():
    node, inputs, axes = _node()
    assert _rows(node, inputs, axes, {"STAGE": "d2/smem-async"}) == ["d2/smem-async"]
    assert all(_transport(r) == "smem" for r in _rows(node, inputs, axes, {"STAGE": "d2/smem"}))


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


def _packed_pins(dtype="f16", stage="d2/smem-async"):
    atom = K16 if dtype == "f16" else K16_BF16
    # ``PLACE=fuse`` keeps the decode cone INSIDE the matmul, which is the whole subject here: a
    # cut lifts the weight into its own kernel and the consumer reads a materialized 16-bit tile,
    # so there is no packed operand left for a ``STAGE`` pin to name.
    return {"PLACE": "fuse", "TILE": f"{atom}/f2x2/k2", "WORK": "w1x4", "REDUCE": "", "STAGE": stage}


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


@pytest.mark.parametrize(("stage", "packed"), [("d2/smem-async", True), ("d1/smem", False)])
def test_the_row_features_the_width_the_weight_really_moves(tmp_path, stage, packed):
    """What tells the priors these two rows are not one kernel with a different transport: on the
    byte slab the weight reaches the slab 4 bits per element, on the compute fill 16. The packed
    dtype alone cannot say it — both rows carry it — so the feature is its product with the
    row's async stage."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
    from emmy.compiler.pipeline.search.features import knob_features
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    g, _ = _nvfp4_matmul_graph(tmp_path, m=32, n=128, k=128)
    with pinned_knobs({**PACKED_PINS, "STAGE": stage}):
        lowered = Pipeline.build(CUDA_PASSES).run(g, ctx=Context.from_target((8, 9)))
    knobs = next(k for node in lowered.nodes.values() if (k := getattr(node.op, "knobs", None)) and "TILE" in k)
    assert knobs["S_dtype_f4e2m1x2"] == 1.0, "the kernel reads the packed weight either way"
    assert knob_features(knobs).get("MMA_b_store_bits") == (4.0 if packed else None)


@requires_cuda
@pytest.mark.parametrize(
    ("dtype", "m", "n", "k", "tol", "stage"),
    [
        ("f16", 32, 128, 128, 1e-3, "d2/smem-async"),
        ("f16", 4, 2048, 2048, 1e-3, "d2/smem-async"),
        ("bf16", 32, 128, 128, 6e-3, "d2/smem-async"),
        ("f16", 32, 128, 128, 1e-3, "d2/smem-tma"),
        ("bf16", 32, 128, 128, 6e-3, "d2/smem-tma"),
    ],
)
@pytest.mark.xdist_group("cuda")
def test_the_packed_drain_matches_the_decoded_oracle(tmp_path, dtype, m, n, k, tol, stage):
    """Numerical parity on the device: the packed kernel equals ``x @ dequantize_nvfp4(w)ᵀ``.

    Both a compute-shaped and a decode-shaped matmul at f16, since the decode shape is where the
    weight traffic dominates, the bf16 fragment, and both copy transports — a TMA box deposits
    dense where cp.async pads, so the two drains read different row strides. Each bound is roughly 3x the measured error
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
    with pinned_knobs(_packed_pins(dtype, stage)):
        compiled = backend.compile(g)
    src = next(s for node in compiled.nodes.values() if (s := getattr(node.op, "kernel_source", None)))
    assert f"emmy_mma_load_b_smem_trans_f4s_{dtype}" in src, "the packed pins did not reach the byte-slab drain"
    if stage.endswith("tma"):
        assert "cp.async.bulk.tensor" in src and "emmy_cp_async" not in src, "the TMA pin must box-copy, not cp.async"

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


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_packed_drain_stages_a_batched_activation_over_tma(tmp_path):
    """A leading unit batch axis on A — the shape every ``emmy compile --layer`` trace carries
    (``[1, seq, K]``) — must box the TMA descriptor at FULL rank. ``_a_slab_operand`` used to
    leave the box at the 2-D slab shape while ``_box_origin`` yielded the full-rank origin, so
    the emitted copy carried more coordinates than the descriptor's encoded rank; TMA treats
    that as an invalid tensor map and the kernel raised ILLEGAL INSTRUCTION from its first
    thread (UTMALDG.4D over a rank-3 map, found by the layer-0 W4A4 parity run). The staged
    matmul tests were all 2-D, and the cp.async transport indexes flat rather than boxing,
    which is why only TMA faulted."""
    import torch

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.graph import Graph
    from emmy.compiler.ir.base import ConstantOp, InputOp
    from emmy.compiler.ir.frontend.ir import LinearOp
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.loader.quant import dequantize_nvfp4, spell_quantized_constants
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from tests.compiler.loader.test_quant import _FP4_MODELOPT_QC, _fp8_tensor, _write_checkpoint

    m, n, k = 16, 1024, 4096
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
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1, m, k), "f16"), node_id="x")
    w = g.add_node(
        op=ConstantOp(name="w", source_path="layer.weight", source_shape=(n, k), source_dtype="f16"),
        inputs=[],
        output=Tensor("w", (n, k), "f16"),
        node_id="w",
    )
    g.add_node(op=LinearOp(), inputs=["x", w], output=Tensor("y", (1, m, n), "f16"), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    assert spell_quantized_constants(g, str(tmp_path)) == 1

    x = (np.random.default_rng(3).standard_normal((1, m, k)) * 0.05).astype(np.float16)
    backend = CudaBackend()
    with pinned_knobs({"TILE": "mma_m16n8k16_f16_f32/f2x4/k8", "WORK": "w1x1", "REDUCE": "g8k", "STAGE": "d1/smem-tma"}):
        compiled = backend.compile(g)
    src = next(s for node in compiled.nodes.values() if (s := getattr(node.op, "kernel_source", None)))
    assert "emmy_mma_load_b_smem_trans_f4s_f16" in src, "the packed pins did not reach the byte-slab drain"
    data = bind_constants(compiled, {"layer.weight": packed, "layer.weight_scale": scale_bits, "layer.weight_scale_2": s2})
    result, _ = backend.run(compiled, input_data={**data, "x": x})
    y = np.asarray(result.outputs[compiled.outputs[0]]).reshape(m, n).astype(np.float32)
    ref = x.reshape(m, k).astype(np.float32) @ dequantize_nvfp4(packed, scale_bits, s2).T
    denom = max(float(np.abs(ref).max()), 1e-9)
    assert float(np.abs(y - ref).max()) / denom < 1e-3


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_packed_drain_composes_with_the_f16_accumulate_atom(tmp_path):
    """The byte slab under the f16-accumulate atom (``FAST_MATH``'s ``F16_MMA_F32_ACC`` member):
    the kernel carries the packed drain, the f16-fragment mma chain and its chunk promote
    together, matches the decoded oracle, and stays close to its f32-accumulate sibling (the same
    schedule with only the atom swapped) — the bound the oracle tolerance alone cannot see. The
    drain's contract names the fragment dtype, not the accumulate, so the composition needs no
    code of its own; this pins that it stays true. Bounds are roughly 4x the measured error on an
    RTX 5090 (oracle 5.0e-4, sibling 7.8e-4)."""
    import torch  # noqa: F401  — the CUDA backend needs it loaded

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.loader.quant import dequantize_nvfp4
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    m, n, k = 4, 2048, 2048
    g, (packed, scale_bits, s2) = _nvfp4_matmul_graph(tmp_path, m=m, n=n, k=k, dtype="f16")
    rng = np.random.default_rng(11)
    x = (rng.standard_normal((m, k)) * 0.05).astype(np.float16)

    backend = CudaBackend()
    outs: dict[str, np.ndarray] = {}
    for atom in ("mma_m16n8k16_f16_f16", K16):
        with pinned_knobs({**PACKED_PINS, "TILE": f"{atom}/f2x2/k2"}):
            compiled = backend.compile(g)
        src = next(s for node in compiled.nodes.values() if (s := getattr(node.op, "kernel_source", None)))
        assert "emmy_mma_load_b_smem_trans_f4s_f16" in src, f"{atom}: the packed pins did not reach the byte-slab drain"
        if atom == K16:
            assert "_ch0_0" not in src, "the f32-accumulate sibling must not declare f16 mma fragments"
        else:
            assert "emmy_mma_m16n8k16_f16_f16(_ch0_0" in src, "the mma chain must target the packed f16 fragments"
            assert "emmy_mma_promote_f16acc" in src, "the chunk promote into the f32 shadow must be emitted"
        data = bind_constants(compiled, {"layer.weight": packed, "layer.weight_scale": scale_bits, "layer.weight_scale_2": s2})
        result, _ = backend.run(compiled, input_data={**data, "x": x})
        outs[atom] = np.asarray(result.outputs[compiled.outputs[0]]).reshape(m, n).astype(np.float32)

    ref = x.astype(np.float32) @ dequantize_nvfp4(packed, scale_bits, s2).T
    denom = max(float(np.abs(ref).max()), 1e-9)
    assert float(np.abs(outs["mma_m16n8k16_f16_f16"] - ref).max()) / denom < 2e-3
    assert float(np.abs(outs["mma_m16n8k16_f16_f16"] - outs[K16]).max()) / denom < 3e-3


@requires_cuda
@pytest.mark.parametrize("stage", ["d2/smem-async", "d2/smem-tma"])
@pytest.mark.xdist_group("cuda")
def test_the_packed_drain_addresses_its_own_split_k_slice(tmp_path, stage):
    """A SPLIT contraction axis must reach the packed bytes of ITS OWN slice.

    A split partition shrinks the K axis and hangs the slice's absolute base on it, so every
    operand's gmem address carries a ``ksplit·(K/w)`` term. The bits operand used to rebuild its
    address from the chunk offset alone, which dropped that base: every partition re-read the
    FIRST slice's bytes and the contraction summed the wrong weights. The block scales were
    always right — they are evaluated by rewriting the decode cone's own body, which carries the
    base — so the two disagreed and the result was silently wrong rather than obviously broken.

    Found on an 8B serving compile, where the down projection is the one the scheduler splits;
    every unsplit shape passes, which is why the isolated drain tests missed it.
    """
    import torch  # noqa: F401  — the CUDA backend needs it loaded

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.loader.quant import dequantize_nvfp4
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    m, n, k = 16, 512, 4096
    g, (packed, scale_bits, s2) = _nvfp4_matmul_graph(tmp_path, m=m, n=n, k=k, dtype="f16")
    rng = np.random.default_rng(23)
    x = (rng.standard_normal((m, k)) * 0.05).astype(np.float16)

    backend = CudaBackend()
    with pinned_knobs({**_packed_pins("f16", stage), "REDUCE": "g8k"}):
        compiled = backend.compile(g)
    src = next(s for node in compiled.nodes.values() if (s := getattr(node.op, "kernel_source", None)))
    assert "emmy_mma_load_b_smem_trans_f4s_f16" in src, "the packed pins did not reach the byte-slab drain"

    data = bind_constants(compiled, {"layer.weight": packed, "layer.weight_scale": scale_bits, "layer.weight_scale_2": s2})
    result, _ = backend.run(compiled, input_data={**data, "x": x})
    y = np.asarray(result.outputs[compiled.outputs[0]]).reshape(m, n).astype(np.float32)
    ref = x.astype(np.float32) @ dequantize_nvfp4(packed, scale_bits, s2).T
    denom = max(float(np.abs(ref).max()), 1e-9)
    assert float(np.abs(y - ref).max()) / denom < 1e-3


# ===================================================================
# The block-scaled pair — the native fp4 cell's four-slab stage
# ===================================================================

K64 = "mma_m16n8k64_e2m1_f32"


def _pair_node(*, m=512, n=4096, k=4096, block=16):
    """Both operands packed — the shape the block-scaled cell reads. A's cone mirrors B's with the
    row axis swapped, which is what the activation speller produces once its codes materialize."""

    axes = (Axis("m", Dim(m)), Axis("n", Dim(n)))
    a_cone = _packed_cone(k, block=block, row="m", prefix="a_")
    node = Fold.contraction(k_axis=Axis("k", Dim(k)), a=a_cone, channels=(Channel(b=_packed_cone(k, block=block), acc="acc"),))
    inputs = {
        "w_bits": Tensor("w_bits", (n, k // 2), F4E2M1x2),
        "w_scale_bits": Tensor("w_scale_bits", (n, k // block), F8E4M3),
        "w_scale_2": Tensor("w_scale_2", (1, 1), F32),
        "w_f4_pairs": Tensor("w_f4_pairs", (256, 2), F16),
        "a_w_bits": Tensor("a_w_bits", (m, k // 2), F4E2M1x2),
        "a_w_scale_bits": Tensor("a_w_scale_bits", (m, k // block), F8E4M3),
        "a_w_scale_2": Tensor("a_w_scale_2", (1, 1), F32),
        "a_w_f4_pairs": Tensor("a_w_f4_pairs", (256, 2), F16),
    }
    return node, inputs, axes


def test_the_pair_reading_splits_each_side_into_codes_scale_and_residue():
    """What the cell takes: the packed codes, the RAW block-scale load, and the k-invariant factor
    the epilogue applies. The per-tensor scale is that residue — it is the one part of the
    operand's chain the instruction has no operand for."""
    from emmy.compiler.pipeline.passes.lowering.tile._packed import match_packed_pair_node

    node, inputs, _axes = _pair_node()
    pair = match_packed_pair_node(node, inputs)
    assert pair is not None and pair.block == 16
    assert pair.a.bits.input == "a_w_bits" and pair.b.bits.input == "w_bits"
    assert pair.a.scale.input == "a_w_scale_bits" and pair.b.scale.input == "w_scale_bits"
    assert [ld.input for ld in pair.a.alpha] == ["a_w_scale_2"]
    assert [ld.input for ld in pair.b.alpha] == ["w_scale_2"]


def test_a_packed_weight_beside_a_materialized_a_is_not_a_pair():
    """The single-sided shape answers ``None`` here and keeps its own reading — the k16 drain,
    which decodes the weight into 16-bit fragments against a 16-bit activation."""
    from emmy.compiler.pipeline.passes.lowering.tile._packed import match_packed_pair_node

    node, inputs, _axes = _node()
    assert match_packed_pair_node(node, inputs) is None


def test_the_block_scaled_stage_resolves_four_byte_slabs_on_cp_async():
    node, inputs, axes = _pair_node()
    tile = _tile(K64, "f1x4/k4", "w1x4", axes)
    st = resolve_warp_stage(node, tile, Stage.parse("d2/smem-async"), 200 * 1024, inputs)
    assert st is not None and st.transport == "smem-async"
    assert st.bk_elems == tile.bk * 64


def test_the_block_scaled_stage_declines_tma_and_a_scale_row_under_the_chunk():
    """Two refusals, both facts rather than preferences. TMA: the four-descriptor box copy is not
    written. The narrow tile: a scale row is ``bk_elems / 16`` bytes and the cp.async fill copies
    16 B chunks, so ``bk_elems`` under 256 leaves a row a chunk cannot fill."""
    node, inputs, axes = _pair_node()
    tile = _tile(K64, "f1x4/k4", "w1x4", axes)
    assert resolve_warp_stage(node, tile, Stage.parse("d2/smem-tma"), 200 * 1024, inputs) is None
    narrow = _tile(K64, "f1x4/k2", "w1x4", axes)
    assert resolve_warp_stage(node, narrow, Stage.parse("d2/smem-async"), 200 * 1024, inputs) is None


# --- the producer band's one illegal partner ---------------------------------------------------
# A producer band splits a block into warps that only fetch and warps that only compute. The
# packed byte slab's TMA lowering returns the plain staged K-loop, which never receives the warp
# inventory and so emits no split — while the block still widens to hold the band. The unsplit
# warps then land in the compute body, where the box copy's elected arming thread is chosen on a
# wrapping lane index, so two threads arm one barrier and its phase parity desynchronizes. That is
# a hang, not a slow kernel: an autotune sweep hit it on 107 rows, every one a packed NVFP4 matmul
# under a band, against 556 identically pinned non-packed rows that all passed.


def test_a_packed_byte_slab_refuses_a_producer_band_under_tma():
    from emmy.compiler.ir.schedule import Workers
    from emmy.compiler.pipeline.passes.lowering.tile import _schedule
    from emmy.compiler.pipeline.search.space import Stage

    tma = Stage(depth=1, transport="smem-tma", bk_elems=64)
    assert _schedule._band_packed_slab_refusal(tma, True) is not None, "the combination hangs; it must decline"
    assert _schedule._band_packed_slab_refusal(tma, False) is None, "a band over an unpacked TMA operand stays legal"
    # The refusal reaches the OFFER, so a row over a packed slab claims no band at all. It is a
    # legality, not a bound: the walk yields no leaf when the pinned inventory is never claimed, so
    # an ``EMMY_WORK=...+p1`` pin here is refused rather than exempted.
    work = Workers(kind="warp", units=(1, 4))
    assert _schedule._producer_bands(work, tma, 128, True) == ()
    assert _schedule._producer_bands(work, tma, 128, False) == (1, 2)
    # Only the BAND is refused, never the staging: the caller asks this at all only for a row whose
    # worker inventory declares producer warps, so the packed byte slab's own TMA transport — which
    # carries the weight's 4-bit traffic — keeps every stage it resolved.
