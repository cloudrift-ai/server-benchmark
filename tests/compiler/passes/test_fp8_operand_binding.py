"""The k-invariant multiplicative dequant binding (M2b of the FP8 plan).

``_atomize.bind_contraction``'s mul-hoist arm: a computed **B** whose cone is a storage decode
(recognized by the ``ElementwiseImpl.decodes`` trait, never an op-name list) times k-invariant
factors binds as the RAW storage-dtype ``Load`` (the decode absorbed by dtype — every consumer
converts a bits-carrier element by dtype) with the factors moved onto the accumulator in the
epilogue (``Σ_k a·(s·w) = s·Σ_k a·w``). A shape the arm cannot hoist — a k-varying (2-D block)
scale, a non-decode computed B — raises, and the recognizer demotes the cell to ``PLANAR`` (the
guardrail contract: unmapped, never mis-scheduled). A storage-dtype (fp8) B now also STAGES — a
raw byte slab whose drain converts to the atom's fragments (``test_fp8_staged``); a mismatch that
is not a byte slab still refuses and keeps the warp tier gmem-direct.
"""

from __future__ import annotations

import re

import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F8E4M3, F16, F32
from emmy.compiler.graph import Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.schedule import Stage, TilePlan, Workers
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.tile import Channel, Fold
from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_contraction
from emmy.compiler.pipeline.passes.lowering.tile._legality import resolve_warp_stage
from emmy.compiler.pipeline.pipeline import LoweringError

from ..conftest import requires_cuda

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
# bind_contraction: the mul-hoist arm
# ===================================================================


def _dequant_loop(*, scale_index=None, scale_op="multiply", decode="from_f8e4m3", extra_factor=False):
    """The fused dequant matmul loop body the fp8 expansion + loop fusion produce:
    ``acc += x[m,k] · (s[n] ⊗ from_f8*(w[k,n]))`` with the scale load hoistable (k-invariant)
    unless ``scale_index`` says otherwise."""
    k = Axis("k", Dim(64))
    stmts = [
        Load(name="s", input="w_scale", index=scale_index or (Literal(0), Var("n"))),
        Load(name="wb", input="w_bits", index=(Var("k"), Var("n")), dtype=F8E4M3),
        Assign(name="dq", op=decode, args=("wb",)),
        Assign(name="wsc", op=scale_op, args=("dq", "s") if scale_op == "divide" else ("s", "dq")),
    ]
    if extra_factor:
        stmts += [
            Load(name="s2", input="w_scale2", index=(Literal(0), Var("n"))),
            Assign(name="wsc2", op="multiply", args=("s2", "wsc")),
        ]
    lift_b = "wsc2" if extra_factor else "wsc"
    stmts += [
        Load(name="a", input="x", index=(Var("m"), Var("k")), dtype=F16),
        Assign(name="v", op="multiply", args=("a", lift_b)),
        Accum(name="acc", value="v", op=ElementwiseImpl("add")),
    ]
    return Loop(axis=k, body=Body(tuple(stmts)), role=AxisRole.CONTRACTION)


def test_decode_scale_cone_binds_via_mul_hoist():
    a, b, acc, epi = bind_contraction(_dequant_loop(), "m", "n", Body())
    assert isinstance(a, Load) and a.input == "x"
    assert isinstance(b, Load) and b.input == "w_bits"  # the RAW f8 load — decode absorbed by dtype
    assert acc == "acc__mh"  # the channel accumulator renamed; the epilogue defines ``acc``
    stmts = list(epi)
    assert [type(s).__name__ for s in stmts] == ["Load", "Assign"]
    assert stmts[0].input == "w_scale"
    assert stmts[1].name == "acc" and stmts[1].op.name == "multiply" and stmts[1].args == ("acc__mh", "s")


def test_inverse_scale_hoists_as_divide():
    """``weight_scale_inv`` spells the cone with a divide — it commutes out the same way."""
    _a, b, acc, epi = bind_contraction(_dequant_loop(scale_op="divide"), "m", "n", Body())
    assert isinstance(b, Load) and b.input == "w_bits"
    tail = list(epi)[-1]
    assert tail.name == "acc" and tail.op.name == "divide" and tail.args == (acc, "s")


def test_factor_chain_hoists_every_k_invariant_factor():
    _a, _b, acc, epi = bind_contraction(_dequant_loop(extra_factor=True), "m", "n", Body())
    assigns = [s for s in epi if isinstance(s, Assign)]
    assert assigns[-1].name == "acc"  # the chain's last def carries the fold's output name
    assert {s.op.name for s in assigns} == {"multiply"}
    assert {s.input for s in epi if isinstance(s, Load)} == {"w_scale", "w_scale2"}
    assert acc == "acc__mh"


def test_original_epilogue_reads_the_scaled_value():
    orig = Body((Assign(name="out", op="relu", args=("acc",)),))
    _a, _b, _acc, epi = bind_contraction(_dequant_loop(), "m", "n", orig)
    stmts = list(epi)
    scale_def = next(i for i, s in enumerate(stmts) if isinstance(s, Assign) and s.name == "acc")
    relu = next(i for i, s in enumerate(stmts) if isinstance(s, Assign) and s.name == "out")
    assert scale_def < relu  # ``acc`` is defined (scaled) before the projection reads it


def test_k_varying_scale_declines_to_planar():
    """A 2-D block scale is k-indexed — the factor does not commute out of the fold, so the arm
    raises and the recognizer keeps the whole cone inline (the fold derives PLANAR)."""
    loop = _dequant_loop(scale_index=(Var("k"), Var("n")))
    with pytest.raises(LoweringError, match="computed cone"):
        bind_contraction(loop, "m", "n", Body())


def test_non_decode_computed_b_raises_instead_of_positional_misbind():
    """A computed B whose leaf is NOT the fp8 decode must never fall through to the positional
    (n, k)-load rule — that binding silently drops the cone (the M2b probe's wrong kernel)."""
    loop = _dequant_loop(decode="exp")
    with pytest.raises(LoweringError, match="computed cone"):
        bind_contraction(loop, "m", "n", Body())


def test_bare_decode_binds_raw_load_without_epilogue():
    """No k-invariant factor at all: B still binds as the raw f8 load, nothing hoists."""
    k = Axis("k", Dim(64))
    loop = Loop(
        axis=k,
        role=AxisRole.CONTRACTION,
        body=Body(
            (
                Load(name="wb", input="w_bits", index=(Var("k"), Var("n")), dtype=F8E4M3),
                Assign(name="dq", op="from_f8e4m3", args=("wb",)),
                Load(name="a", input="x", index=(Var("m"), Var("k")), dtype=F16),
                Assign(name="v", op="multiply", args=("a", "dq")),
                Accum(name="acc", value="v", op=ElementwiseImpl("add")),
            )
        ),
    )
    _a, b, acc, epi = bind_contraction(loop, "m", "n", Body())
    assert isinstance(b, Load) and b.input == "w_bits" and acc == "acc" and not len(epi)


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
    for spec in ("d2/cp", "d2/tma"):
        assert resolve_warp_stage(node, tile, Stage.parse(spec), 100 * 1024, inputs) is not None


def test_resolve_warp_stage_admits_matched_dtypes():
    node, tile = _warp_contraction()
    inputs = {"x": Tensor("x", (512, 4096), F16), "w_bits": Tensor("w_bits", (4096, 4096), F16)}
    assert resolve_warp_stage(node, tile, Stage.parse("d2/cp"), 100 * 1024, inputs) is not None


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
