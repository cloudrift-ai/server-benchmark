"""FP8 operand traits, staged byte slabs, and end-to-end warp lowering."""

from __future__ import annotations

import re

import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F8E4M3, F16, F32
from emmy.compiler.graph import Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.fold import Channel, Fold
from emmy.compiler.ir.schedule import Stage, TilePlan, Workers
from emmy.compiler.ir.stmt import Load
from emmy.compiler.pipeline.passes.lowering.tile._legality import resolve_warp_stage
from tests.compiler.helpers import requires_cuda

# ===================================================================
# The decode trait — the registration a new storage format extends
# ===================================================================


def test_decodes_trait_names_the_storage_dtype():
    """``decodes`` is name-keyed like every ElementwiseImpl trait: the fp8 decode ops name their
    storage dtype, ordinary ops answer ``None``, and the trait survives the name-only pickle
    round-trip (no instance field, so serialization is untouched)."""
    import pickle

    assert ElementwiseImpl("from_f8e4m3").decodes == "f8e4m3"
    assert ElementwiseImpl("from_f8e5m2").decodes == "f8e5m2"
    assert ElementwiseImpl("multiply").decodes is None and ElementwiseImpl("copy").decodes is None
    assert pickle.loads(pickle.dumps(ElementwiseImpl("from_f8e4m3"))).decodes == "f8e4m3"


# ===================================================================
# Staged transports: an f8 B stages as a byte slab; other mismatches refuse
# ===================================================================


def _warp_contraction():
    k = Axis("k", Dim(4096))
    m, n = Axis("m", Dim(512)), Axis("n", Dim(4096))
    a = Load(name="a", input="x", index=(Var("m"), Var("k")), dtype=F16)
    b = Load(name="wb", input="w_bits", index=(Var("k"), Var("n")), dtype=F8E4M3)
    node = Fold.contraction(k_axis=k, a=a, channels=(Channel(b=b, acc="acc"),))
    tile = TilePlan.parse("mma_m16n8k16_f16_f32/f4x1/k4", Workers.parse("w1x8")).at(m, n)
    return node, tile


def test_resolve_warp_stage_offers_the_byte_staged_b():
    """The M2b refusal is replaced by the byte-staged offer: an fp8-stored B under a 16-bit atom
    resolves on every copy transport (the raw byte slab, converted at the drain — the full
    legality/parity battery is ``test_fp8_staged``)."""
    node, tile = _warp_contraction()
    inputs = {"x": Tensor("x", (512, 4096), F16), "w_bits": Tensor("w_bits", (4096, 4096), F8E4M3)}
    for spec in ("d2/smem-async", "d2/smem-tma"):
        assert resolve_warp_stage(node, tile, Stage.parse(spec), 100 * 1024, inputs) is not None


def test_resolve_warp_stage_admits_matched_dtypes():
    node, tile = _warp_contraction()
    inputs = {"x": Tensor("x", (512, 4096), F16), "w_bits": Tensor("w_bits", (4096, 4096), F16)}
    assert resolve_warp_stage(node, tile, Stage.parse("d2/smem-async"), 100 * 1024, inputs) is not None


def test_f8_atoms_are_the_gated_k32_family():
    """The only f8-multiplicand atoms are the native m16n8k32 cells (M3) — offered solely through
    the ``FP8_MMA``-gated enumeration — so an f8 operand still never selects a 16-bit (k16)
    tensor-core cell, and the W8A16 path's picks are untouched."""
    f8 = {name: atom for name, atom in ATOM_REGISTRY.items() if any(dt.name.startswith("f8") for _r, dt in atom.operand_dtypes)}
    assert set(f8) == {"mma_m16n8k32_e4m3_f32", "mma_m16n8k32_e5m2_f32"}
    assert all(atom.shape == (16, 8, 32) and atom.operand_dtype("c") is F32 for atom in f8.values())


# ===================================================================
# The warp tier on CUDA: fp8-B W8A16 — fragment-boundary decode + epilogue scale
# ===================================================================


def _fp8_linear_graph(m=32, n=512, k=512):
    """``x:f16 @ (from_f8e4m3(W bits) · s:(N,1))ᵀ`` — the LinearOp over the in-graph decode cone
    the birth-time speller emits for a per-out-channel fp8 weight."""
    from emmy.compiler.graph import Graph
    from emmy.compiler.ir.base import ConstantOp, InputOp
    from emmy.compiler.ir.frontend.ir import LinearOp
    from emmy.compiler.ir.tensor.ir import ElementwiseOp
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (m, k), "f16"), node_id="x")
    w = g.add_node(
        op=ConstantOp(name="p_w", source_path="layer.weight", source_shape=(n, k), source_dtype="f8e4m3"),
        inputs=[],
        output=Tensor("p_w_bits", (n, k), "f8e4m3"),
    )
    scale = g.add_node(
        op=ConstantOp(name="p_w_scale", source_path="layer.weight_scale", source_shape=(n, 1), source_dtype="f32"),
        inputs=[],
        output=Tensor("p_w_scale", (n, 1), "f32"),
    )
    cast = g.add_node(op=ElementwiseOp(op="from_f8e4m3"), inputs=[w], output=Tensor("p_w_dq", (n, k), "f16"))
    s_bc = broadcast_to(g, scale, (n, k))
    g.add_node(op=ElementwiseOp(op="multiply"), inputs=[cast, s_bc], output=Tensor("p_w", (n, k), "f16"), node_id="p_w")
    g.add_node(op=LinearOp(), inputs=["x", "p_w"], output=Tensor("y", (m, n), F16), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    return g


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_fp8_b_matmul_reaches_warp_tier_cuda():
    """The fragment-convert path (M2b priority 3): under a warp ``TILE`` pin the fp8-B linear
    lands on the mma tier — the gmem-direct B fragment load converts fp8 bytes to f16 per element
    (``emmy_mma_load_b_gmem<__nv_fp8_e4m3, __half>``), the per-out-channel scale rides the f32
    fragment epilogue — and the result matches the dequant reference."""
    import numpy as np

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.dtype import decode_f8
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    m, n, k = 32, 512, 512
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 256, (n, k)).astype(np.uint8)
    bits[bits == 0x7F] = 0x00
    bits[bits == 0xFF] = 0x80
    scale = (np.abs(rng.standard_normal((n, 1))) * 0.005 + 0.002).astype(np.float32)
    x = (rng.standard_normal((m, k)) * 0.05).astype(np.float16)

    backend = CudaBackend()
    # STAGE pinned to gmem-direct: this test anchors the gmem-direct fragment-convert spelling
    # (the staged byte-slab forms are ``test_fp8_staged``'s).
    with pinned_knobs({"TILE": "mma_m16n8k16_f16_f32/f2x2/k2", "WORK": "w1x8", "REDUCE": "", "STAGE": ""}):
        compiled = backend.compile(_fp8_linear_graph(m, n, k))

    sources = [getattr(node.op, "kernel_source", None) for node in compiled.nodes.values()]
    mma_src = next((s for s in sources if s and "mma.sync" in s), None)
    assert mma_src is not None, "no mma kernel — the fp8-B contraction did not reach the warp tier"
    # The fragment-boundary decode. Below sm_90 the 050/060 constant folds don't fire, so B
    # arrives in-graph-transposed and the TRANS fragment helper carries the same per-element
    # convert — verified numerically on a 4090 (max_rel 2.5e-4); either spelling is the warp
    # tier with the decode at the fragment load.
    assert re.search(r"emmy_mma_load_b_gmem(_trans)?<__nv_fp8_e4m3, __half>", mma_src)

    input_data: dict = {"x": x}
    input_data.update(bind_constants(compiled, {"layer.weight": bits, "layer.weight_scale": scale}))
    result, _ = backend.run(compiled, input_data=input_data)
    y = result.outputs[compiled.outputs[0]].reshape(m, n).astype(np.float32)

    w = decode_f8(bits, "f8e4m3").astype(np.float32) * scale
    ref = x.astype(np.float32) @ w.T
    denom = max(float(np.abs(ref).max()), 1e-9)
    assert float(np.abs(y - ref).max()) / denom < 2e-3, "fp8-B warp kernel diverges from the dequant reference"
