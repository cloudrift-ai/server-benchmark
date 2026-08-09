"""The native fp8 mma atoms — W8A8 (M3 of the FP8 plan).

The ``mma_m16n8k32_{e4m3,e5m2}_f32`` atoms consume BOTH multiplicands as raw f8 bytes (four per
32-bit fragment register — the ``_b8`` gmem-direct byte-gather loaders) with an f32 accumulator,
spelled by the bare PTX form (``mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`` — probed:
compiles and verifies on sm_89 AND sm_120; ptxas refuses the Blackwell ``.kind::f8f6f4`` qualifier
on both). Enumeration is precision-gated (``FP8_MMA`` / the ``FAST_MATH`` umbrella — accumulation
precision is arch-dependent, reduced on sm_89) and structurally gated (materialized f8 A, every
channel at the same f8 dtype, static K); a ``TILE`` pin naming the atom bypasses the precision
gate only. The W8A8 form arrives by graph algebra: ``to_f8e4m3`` (the encode twin of the M2b
decode) materializes the quantized activation, and ``bind_contraction``'s mul-hoist — now
side-generic — binds BOTH decode cones as raw f8 loads with the two scale factors composed on the
f32 accumulator epilogue.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F8E4M3, F16, F32, decode_f8, encode_f8
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY, atom_for, atoms_for
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.ir.schedule import Placement, TilePlan, Workers
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.ir.tile import Channel, Fold
from emmy.compiler.ir.tile.ir import TileOp
from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to
from emmy.compiler.pipeline.passes.lowering.tile import _schedule as sched
from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_contraction

from ..conftest import requires_cuda

K32 = "mma_m16n8k32_e4m3_f32"

# ===================================================================
# The atom registry + codec spelling
# ===================================================================


def test_k32_atom_registry_and_geometry():
    atom = ATOM_REGISTRY[K32]
    assert atom.shape == (16, 8, 32) and atom.lanes == 32
    assert atom.operand_dtype("a") is F8E4M3 and atom.operand_dtype("b") is F8E4M3 and atom.operand_dtype("c") is F32
    assert atom.ab_dtype == "e4m3"  # the PTX dtype field — the wrapper/atom-name spelling
    assert not atom.c_to_a_repack  # no lane-map C→A repack at k32 — flash's P→A handoff never applies


def test_k32_tile_codec_round_trips():
    """The ``TILE`` codec names the atom by registry key and the spelling round-trips."""
    work = Workers.parse("w1x8")
    plan = TilePlan.parse(f"{K32}/f2x2/k2", work)
    assert plan.atom is ATOM_REGISTRY[K32] and plan.bk == 2
    assert TilePlan.parse(plan.spell(), work) == plan
    assert atom_for(K32) is ATOM_REGISTRY[K32]


def test_k32_fragment_register_counts():
    """The k32 8-bit fragment ABI packs 4 bytes/reg/lane: A is 4x32-bit regs, B is 2 (the PTX
    m16n8k32 8-bit fragment table); C keeps the k16 f32 map (4 regs)."""
    from emmy.compiler.ir.kernel.ir import _mma_sync_nregs

    assert _mma_sync_nregs("a", (16, 8, 32), F8E4M3) == 4
    assert _mma_sync_nregs("b", (16, 8, 32), F8E4M3) == 2
    assert _mma_sync_nregs("c", (16, 8, 32), F32) == 4
    # the 16-bit family is untouched
    assert _mma_sync_nregs("a", (16, 8, 16), F16) == 4 and _mma_sync_nregs("b", (16, 8, 16), F16) == 2


# ===================================================================
# The encode twin (to_f8e4m3)
# ===================================================================


def test_encode_f8_matches_torch_exhaustively():
    """The numpy encode is bit-exact with torch's cast for every finite in-range f16 value
    (round-to-nearest-even). Overflow deliberately differs: the encode saturates to the max
    finite value (the <cuda_fp8.h> constructor's rounding, which the CUDA spelling invokes)
    where torch maps far-overflow to NaN; NaN maps to the canonical NaN code either way."""
    torch = pytest.importorskip("torch")

    h = np.arange(65536, dtype=np.uint16).view(np.float16)
    x = h.astype(np.float32)
    finite = np.isfinite(x) & (np.abs(x) <= 448.0)
    ours = encode_f8(x, "f8e4m3")
    theirs = torch.from_numpy(x).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    assert not ((ours != theirs) & finite).any()
    assert encode_f8(np.array([np.inf, -np.inf, np.nan]), "f8e4m3").tolist() == [0x7E, 0xFE, 0x7F]


def test_encode_decode_round_trip_is_identity():
    """Every non-NaN code encodes back to itself (−0.0 keeps its sign bit, matching torch)."""
    codes = np.arange(256, dtype=np.uint8)
    codes = codes[(codes & 0x7F) != 0x7F]
    np.testing.assert_array_equal(encode_f8(decode_f8(codes, "f8e4m3"), "f8e4m3"), codes)


def test_to_f8_op_registered_with_numpy_forward():
    op = ElementwiseImpl("to_f8e4m3")
    got = op(np.array([1.0, -0.4375, 448.0], dtype=np.float32))
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(decode_f8(got, "f8e4m3"), [1.0, -0.4375, 448.0])


# ===================================================================
# bind_contraction: the side-generic mul-hoist (A-side + double-cone)
# ===================================================================


def _w8a8_loop(*, a_scale=True, b_scale=True):
    """The fused W8A8 matmul loop body: ``acc += (sa · from_f8(a[m,k])) · (sb[n] · from_f8(w[k,n]))``
    — a storage-decode cone on EACH operand, each times a k-invariant scale factor."""
    k = Axis("k", Dim(64))
    stmts: list = [
        Load(name="ab", input="a_bits", index=(Var("m"), Var("k")), dtype=F8E4M3),
        Assign(name="adq", op="from_f8e4m3", args=("ab",)),
        Load(name="wb", input="w_bits", index=(Var("k"), Var("n")), dtype=F8E4M3),
        Assign(name="wdq", op="from_f8e4m3", args=("wb",)),
    ]
    a_val, b_val = "adq", "wdq"
    if a_scale:
        stmts += [Load(name="sa", input="act_scale", index=(Literal(0), Literal(0))), Assign(name="asc", op="multiply", args=("sa", "adq"))]
        a_val = "asc"
    if b_scale:
        stmts += [Load(name="sb", input="w_scale", index=(Literal(0), Var("n"))), Assign(name="wsc", op="multiply", args=("sb", "wdq"))]
        b_val = "wsc"
    stmts += [
        Assign(name="v", op="multiply", args=(a_val, b_val)),
        Accum(name="acc", value="v", op=ElementwiseImpl("add")),
    ]
    return Loop(axis=k, body=Body(tuple(stmts)), role=AxisRole.CONTRACTION)


def test_a_side_decode_scale_binds_raw_with_hoist():
    """A decode-times-factor cone on the A side (B direct would be the mirror of M2b; here B is a
    bare decode) binds A as the RAW f8 load with the activation scale hoisted to the epilogue."""
    a, b, acc, epi = bind_contraction(_w8a8_loop(b_scale=False), "m", "n", Body())
    assert isinstance(a, Load) and a.input == "a_bits"
    assert isinstance(b, Load) and b.input == "w_bits"
    assert acc == "acc__mh"
    tail = [s for s in epi if isinstance(s, Assign)][-1]
    assert tail.name == "acc" and tail.op.name == "multiply"


def test_double_cone_hoists_both_scales():
    """The W8A8 shape: BOTH operands ride decode-times-factor cones — both bind raw, and the two
    scale factors compose into one epilogue chain on the accumulator."""
    a, b, acc, epi = bind_contraction(_w8a8_loop(), "m", "n", Body())
    assert isinstance(a, Load) and a.input == "a_bits"
    assert isinstance(b, Load) and b.input == "w_bits"
    assert acc == "acc__mh"
    assigns = [s for s in epi if isinstance(s, Assign)]
    assert assigns[-1].name == "acc" and {s.op.name for s in assigns} == {"multiply"}
    assert {s.input for s in epi if isinstance(s, Load)} == {"act_scale", "w_scale"}


def test_bare_double_decode_binds_raw_without_epilogue():
    a, b, acc, epi = bind_contraction(_w8a8_loop(a_scale=False, b_scale=False), "m", "n", Body())
    assert isinstance(a, Load) and a.input == "a_bits" and isinstance(b, Load) and b.input == "w_bits"
    assert acc == "acc" and not len(epi)


# ===================================================================
# Enumeration gating: FP8_MMA / FAST_MATH + the structural requirements
# ===================================================================


def _f8_term(cap=(12, 0), *, a_dtype=F8E4M3, b_dtype=F8E4M3, k=512):
    ka = Axis("k", Dim(k))
    a = Load(name="ab", input="a_bits", index=(Var("m"), Var("k")), dtype=a_dtype)
    b = Load(name="wb", input="w_bits", index=(Var("k"), Var("n")), dtype=b_dtype)
    node = Fold.contraction(k_axis=ka, a=a, channels=(Channel(b=b, acc="acc"),))
    inputs = {"a_bits": Tensor("a_bits", (32, k), a_dtype), "w_bits": Tensor("w_bits", (k, 512), b_dtype)}
    tile = TileOp(op=node, name="lin", place=Placement(free=(Axis("m", Dim(32)), Axis("n", Dim(512)))), inputs=inputs)
    return sched._Term(tile, tile.place.on_grid(), Context.from_target(cap)), node


def test_k32_enumeration_requires_the_precision_gate(monkeypatch):
    monkeypatch.delenv("EMMY_FP8_MMA", raising=False)
    monkeypatch.delenv("EMMY_FAST_MATH", raising=False)
    term, node = _f8_term()
    assert sched._warp_atoms(term, node) == ()
    monkeypatch.setenv("EMMY_FAST_MATH", "1")
    assert sched._warp_atoms(term, node) == (K32,)
    monkeypatch.delenv("EMMY_FAST_MATH")
    monkeypatch.setenv("EMMY_FP8_MMA", "1")
    assert sched._warp_atoms(term, node) == (K32,)
    monkeypatch.setenv("EMMY_FP8_MMA", "0")
    monkeypatch.setenv("EMMY_FAST_MATH", "1")  # the precise pin wins over the umbrella
    assert sched._warp_atoms(term, node) == ()


def test_k32_enumeration_structural_requirements(monkeypatch):
    monkeypatch.setenv("EMMY_FP8_MMA", "1")
    # both operands must carry the SAME f8 dtype — a 16-bit B never rides the byte gather
    term, node = _f8_term(b_dtype=F16)
    assert sched._warp_atoms(term, node) == ()
    # the sm_89 hardware floor is absolute (the bare PTX form does not compile below)
    term, node = _f8_term(cap=(8, 6))
    assert sched._warp_atoms(term, node) == ()
    # a symbolic K declines — no masked-K byte gather
    from emmy.compiler.pipeline.passes.lowering.tile import _legality as legal

    term, node = _f8_term()
    plan = TilePlan.parse(f"{K32}/f2x2/k2", Workers.parse("w1x8"))
    sym = Fold.contraction(
        k_axis=Axis("k", Dim("seq")),
        a=Load(name="ab", input="a_bits", index=(Var("m"), Var("k")), dtype=F8E4M3),
        channels=(Channel(b=Load(name="wb", input="w_bits", index=(Var("k"), Var("n")), dtype=F8E4M3), acc="acc"),),
    )
    assert legal.warp_k_step(sym, plan) is not None


def test_k32_never_offered_on_16bit_operands(monkeypatch):
    """The k16 path's picks are untouched: 16-bit operands never see a k32 atom, with or without
    the gate."""
    monkeypatch.setenv("EMMY_FP8_MMA", "1")
    term, node = _f8_term(a_dtype=F16, b_dtype=F16)
    atoms = sched._warp_atoms(term, node)
    assert K32 not in atoms and "mma_m16n8k16_f16_f32" in atoms
    assert atoms_for(F16) == ("mma_m16n8k16_f16_f32",)


def test_f8_atoms_offer_staged_byte_slabs():
    """ldmatrix has no byte-fragment form on these arches, so the fp8 atoms' staged form is the
    cooperative byte-slab drain — the resolver OFFERS it on the copy transports (the refusal this
    replaced kept every fp8 arm on the transaction-bound gmem-direct path; the parity/legality
    battery is ``test_fp8_staged``)."""
    from emmy.compiler.ir.schedule import Stage
    from emmy.compiler.pipeline.passes.lowering.tile._legality import resolve_warp_stage

    _term, node = _f8_term()
    tile = TilePlan.parse(f"{K32}/f4x1/k4", Workers.parse("w1x8")).at(Axis("m", Dim(512)), Axis("n", Dim(512)))
    inputs = {"a_bits": Tensor("a_bits", (512, 512), F8E4M3), "w_bits": Tensor("w_bits", (512, 512), F8E4M3)}
    for spec in ("d2/cp", "d2/tma"):
        assert resolve_warp_stage(node, tile, Stage.parse(spec), 100 * 1024, inputs) is not None


# ===================================================================
# CUDA: the kernel-level anchor + the W8A8 e2e (5090 / 4090)
# ===================================================================


def _bare_f8_linear_graph(m, n, k, out_dtype="f32"):
    """``from_f8(a8) @ from_f8(w8)ᵀ`` over raw-bit f8 INPUTS — the kernel-level anchor graph."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a8", (m, k), "f8e4m3"), node_id="a8")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("w8", (n, k), "f8e4m3"), node_id="w8")
    ad = g.add_node(op=ElementwiseOp(op="from_f8e4m3"), inputs=["a8"], output=Tensor("a_dq", (m, k), "f16"))
    wd = g.add_node(op=ElementwiseOp(op="from_f8e4m3"), inputs=["w8"], output=Tensor("w_dq", (n, k), "f16"))
    g.add_node(op=LinearOp(), inputs=[ad, wd], output=Tensor("y", (m, n), out_dtype), node_id="y")
    g.inputs, g.outputs = ["a8", "w8"], ["y"]
    return g


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_k32_mma_matches_lut_reference_cuda():
    """The fragment-ABI anchor: fixed fp8 bits through the pinned k32 kernel vs the LUT-decode
    numpy reference. Two bit regimes: an EXACT one (A in {0, ±1}, B in one exponent band — every
    partial sum exactly representable, so even sm_89's reduced-precision accumulate and the f16
    store round nothing) and a random-bits one under the loose arch-covering gate."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    m, n, k = 32, 64, 64
    rng = np.random.default_rng(11)
    backend = CudaBackend()
    with pinned_knobs({"TILE": f"{K32}/f2x2/k1", "WORK": "w1x4", "REDUCE": "", "STAGE": ""}):
        compiled = backend.compile(_bare_f8_linear_graph(m, n, k))
    srcs = [getattr(nd.op, "kernel_source", "") or "" for nd in compiled.nodes.values()]
    mma_src = next((s for s in srcs if "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32" in s), None)
    assert mma_src is not None, "no k32 mma kernel — the f8 pair did not reach the native fp8 tier"
    assert "emmy_mma_load_a_gmem_b8" in mma_src  # the byte-gather fragment loaders, raw bits
    assert "emmy_mma_load_b_gmem_trans_b8" in mma_src or "emmy_mma_load_b_gmem_b8" in mma_src

    # exact regime
    abits = np.take(np.array([0x00, 0x38, 0xB8], dtype=np.uint8), rng.integers(0, 3, (m, k)))
    wbits = ((rng.integers(0, 256, (n, k)).astype(np.uint8)) & 0x87) | 0x30  # values in ±[2^-3, 2^-2)
    result, _ = backend.run(compiled, input_data={"a8": abits, "w8": wbits})
    y = result.outputs["y"].reshape(m, n).astype(np.float32)
    ref = decode_f8(abits, "f8e4m3").astype(np.float32) @ decode_f8(wbits, "f8e4m3").astype(np.float32).T
    np.testing.assert_array_equal(y, ref)

    # random-bits regime (small exponent band avoids f16-range overflow in the store path)
    abits = ((rng.integers(0, 256, (m, k)).astype(np.uint8)) & 0x87) | 0x30
    result, _ = backend.run(compiled, input_data={"a8": abits, "w8": wbits})
    y = result.outputs["y"].reshape(m, n).astype(np.float32)
    ref = decode_f8(abits, "f8e4m3").astype(np.float32) @ decode_f8(wbits, "f8e4m3").astype(np.float32).T
    denom = max(float(np.abs(ref).max()), 1e-9)
    # covers the f16 store rounding plus sm_89's reduced-precision accumulate (~3e-4 measured)
    assert float(np.abs(y - ref).max()) / denom < 2e-3


def _w8a8_graph(m, n, k):
    """The W8A8 static-activation-quantization form, all graph algebra: ``x_f8 = to_f8e4m3(x /
    act_scale)`` materialized (a second graph output pins the kernel boundary — the natural serving
    shape, one quantize feeding several projections), the linear consuming ``from_f8(x_f8) ·
    act_scale`` against the fp8 weight's decode-times-scale cone."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (m, k), "f16"), node_id="x")
    sa = g.add_node(
        op=ConstantOp(name="p_sa", source_path="act.scale", source_shape=(1, 1), source_dtype="f32"),
        inputs=[],
        output=Tensor("p_sa", (1, 1), "f32"),
    )
    inv = g.add_node(op=ElementwiseOp(op="reciprocal"), inputs=[sa], output=Tensor("p_sa_inv", (1, 1), "f32"))
    inv_bc = broadcast_to(g, inv, (m, k))
    xs = g.add_node(op=ElementwiseOp(op="multiply"), inputs=["x", inv_bc], output=Tensor("x_scaled", (m, k), "f32"))
    x8 = g.add_node(op=ElementwiseOp(op="to_f8e4m3"), inputs=[xs], output=Tensor("x_f8", (m, k), "f8e4m3"), node_id="x_f8")
    xd = g.add_node(op=ElementwiseOp(op="from_f8e4m3"), inputs=[x8], output=Tensor("x_dq", (m, k), "f16"))
    sa_bc = broadcast_to(g, sa, (m, k))
    xa = g.add_node(op=ElementwiseOp(op="multiply"), inputs=[xd, sa_bc], output=Tensor("x_a", (m, k), "f16"))
    w = g.add_node(
        op=ConstantOp(name="p_w", source_path="layer.weight", source_shape=(n, k), source_dtype="f8e4m3"),
        inputs=[],
        output=Tensor("p_w_bits", (n, k), "f8e4m3"),
    )
    ws = g.add_node(
        op=ConstantOp(name="p_ws", source_path="layer.wscale", source_shape=(n, 1), source_dtype="f32"),
        inputs=[],
        output=Tensor("p_ws", (n, 1), "f32"),
    )
    wd = g.add_node(op=ElementwiseOp(op="from_f8e4m3"), inputs=[w], output=Tensor("p_w_dq", (n, k), "f16"))
    ws_bc = broadcast_to(g, ws, (n, k))
    g.add_node(op=ElementwiseOp(op="multiply"), inputs=[wd, ws_bc], output=Tensor("p_w", (n, k), "f16"), node_id="p_w")
    g.add_node(op=LinearOp(), inputs=[xa, "p_w"], output=Tensor("y", (m, n), F16), node_id="y")
    g.inputs, g.outputs = ["x"], ["y", "x_f8"]
    return g


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_w8a8_static_act_quant_e2e_cuda():
    """The W8A8 e2e (M3 priority 4): the encode kernel materializes ``x_f8`` via the
    <cuda_fp8.h> constructor, the linear's double decode cone binds BOTH raw f8 operands onto the
    k32 mma, and the epilogue composes ``act_scale ⊗ weight_scale`` on the f32 accumulator.
    Compared against (a) the f32 reference of the SAME quantized computation (tight) and (b) the
    unquantized reference (the activation-quantization error, loose documented gate)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    m, n, k = 32, 512, 512
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 256, (n, k)).astype(np.uint8)
    bits[bits == 0x7F] = 0x00
    bits[bits == 0xFF] = 0x80
    wscale = (np.abs(rng.standard_normal((n, 1))) * 0.005 + 0.002).astype(np.float32)
    act_scale = np.array([[0.02]], dtype=np.float32)
    x = (rng.standard_normal((m, k)) * 0.05).astype(np.float16)

    backend = CudaBackend()
    with pinned_knobs({"TILE": f"{K32}/f2x2/k2", "WORK": "w1x8", "REDUCE": "", "STAGE": ""}):
        compiled = backend.compile(_w8a8_graph(m, n, k))
    srcs = [getattr(nd.op, "kernel_source", "") or "" for nd in compiled.nodes.values()]
    assert any("__nv_fp8_e4m3(" in s and "x_f8[" in s for s in srcs), "no encode kernel materializing x_f8"
    mma_src = next((s for s in srcs if "mma.sync.aligned.m16n8k32" in s), None)
    assert mma_src is not None, "the W8A8 linear did not reach the k32 mma"
    assert "emmy_mma_load_a_gmem_b8" in mma_src, "A did not bind as the raw f8 load"

    input_data: dict = {"x": x}
    input_data.update(bind_constants(compiled, {"act.scale": act_scale, "layer.wscale": wscale, "layer.weight": bits}))
    result, _ = backend.run(compiled, input_data=input_data)
    y = result.outputs["y"].reshape(m, n).astype(np.float32)

    xq = decode_f8(encode_f8(x.astype(np.float32) / act_scale, "f8e4m3"), "f8e4m3") * act_scale
    wq = decode_f8(bits, "f8e4m3").astype(np.float32) * wscale
    ref_q = xq.astype(np.float32) @ wq.T
    ref_u = x.astype(np.float32) @ wq.T
    denom_q = max(float(np.abs(ref_q).max()), 1e-9)
    denom_u = max(float(np.abs(ref_u).max()), 1e-9)
    assert float(np.abs(y - ref_q).max()) / denom_q < 2e-3, "k32 kernel diverges from the same-quantization reference"
    assert float(np.abs(y - ref_u).max()) / denom_u < 5e-2, "activation-quantization error beyond the documented gate"


def _dyn_w8a8_graph(m, n, k):
    """Dynamic per-token activation quantization, all graph algebra: ``s[m] = amax_k |x| / 448``
    (the per-row statistic — the rmsnorm shape the reduce tiers already run), ``x_f8 =
    to_f8e4m3(x / s)`` materialized, the linear consuming ``from_f8(x_f8) · s[m]`` — an m-indexed
    factor is K-FREE, so the same mul-hoist commutes it onto the epilogue beside the weight
    scale."""
    from emmy.compiler.ir.tensor.ir import ReduceOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (m, k), "f16"), node_id="x")
    ax = g.add_node(op=ElementwiseOp(op="abs"), inputs=["x"], output=Tensor("x_abs", (m, k), "f32"))
    amax = g.add_node(op=ReduceOp(op="amax", axis=1), inputs=[ax], output=Tensor("x_amax", (m, 1), "f32"))
    denom = g.add_node(
        op=ConstantOp(name="p_fmax", source_path="fmax", source_shape=(1, 1), source_dtype="f32"),
        inputs=[],
        output=Tensor("p_fmax", (1, 1), "f32"),
    )
    denom_bc = broadcast_to(g, denom, (m, 1))
    s = g.add_node(op=ElementwiseOp(op="divide"), inputs=[amax, denom_bc], output=Tensor("x_s", (m, 1), "f32"), node_id="x_s")
    s_bc = broadcast_to(g, s, (m, k))
    xs = g.add_node(op=ElementwiseOp(op="divide"), inputs=["x", s_bc], output=Tensor("x_scaled", (m, k), "f32"))
    x8 = g.add_node(op=ElementwiseOp(op="to_f8e4m3"), inputs=[xs], output=Tensor("x_f8", (m, k), "f8e4m3"), node_id="x_f8")
    xd = g.add_node(op=ElementwiseOp(op="from_f8e4m3"), inputs=[x8], output=Tensor("x_dq", (m, k), "f16"))
    xa = g.add_node(op=ElementwiseOp(op="multiply"), inputs=[xd, s_bc], output=Tensor("x_a", (m, k), "f16"))
    w = g.add_node(
        op=ConstantOp(name="p_w", source_path="layer.weight", source_shape=(n, k), source_dtype="f8e4m3"),
        inputs=[],
        output=Tensor("p_w_bits", (n, k), "f8e4m3"),
    )
    ws = g.add_node(
        op=ConstantOp(name="p_ws", source_path="layer.wscale", source_shape=(n, 1), source_dtype="f32"),
        inputs=[],
        output=Tensor("p_ws", (n, 1), "f32"),
    )
    wd = g.add_node(op=ElementwiseOp(op="from_f8e4m3"), inputs=[w], output=Tensor("p_w_dq", (n, k), "f16"))
    ws_bc = broadcast_to(g, ws, (n, k))
    g.add_node(op=ElementwiseOp(op="multiply"), inputs=[wd, ws_bc], output=Tensor("p_w", (n, k), "f16"), node_id="p_w")
    g.add_node(op=LinearOp(), inputs=[xa, "p_w"], output=Tensor("y", (m, n), F16), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y", "x_f8", "x_s"]
    return g


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_w8a8_dynamic_per_token_amax_cuda():
    """Dynamic per-token amax (M3 priority 5): the composition falls out of the existing
    machinery — the amax statistic + encode ride their own kernels, the k32 mma consumes the
    materialized bits, and the m-indexed per-token scale hoists like any k-invariant factor.
    Verified against the DEVICE's own encoded bits (tight — isolates the mma path from host/device
    encode boundary-tie divergence at x/s values landing exactly between e4m3 codes) and the host
    quantized reference (loose)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    m, n, k = 32, 512, 512
    rng = np.random.default_rng(9)
    bits = rng.integers(0, 256, (n, k)).astype(np.uint8)
    bits[bits == 0x7F] = 0x00
    bits[bits == 0xFF] = 0x80
    wscale = (np.abs(rng.standard_normal((n, 1))) * 0.005 + 0.002).astype(np.float32)
    x = (rng.standard_normal((m, k)) * 0.05).astype(np.float16)

    backend = CudaBackend()
    with pinned_knobs({"TILE": f"{K32}/f2x2/k2", "WORK": "w1x8", "REDUCE": "", "STAGE": ""}):
        compiled = backend.compile(_dyn_w8a8_graph(m, n, k))
    srcs = [getattr(nd.op, "kernel_source", "") or "" for nd in compiled.nodes.values()]
    mma_src = next((s for s in srcs if "mma.sync.aligned.m16n8k32" in s), None)
    assert mma_src is not None and "emmy_mma_load_a_gmem_b8" in mma_src

    input_data: dict = {"x": x}
    fmax = np.array([[448.0]], dtype=np.float32)
    input_data.update(bind_constants(compiled, {"fmax": fmax, "layer.weight": bits, "layer.wscale": wscale}))
    result, _ = backend.run(compiled, input_data=input_data)
    y = result.outputs["y"].reshape(m, n).astype(np.float32)
    got_s = result.outputs["x_s"].reshape(m, 1).astype(np.float32)
    got_x8 = np.asarray(result.outputs["x_f8"]).reshape(m, k).view(np.uint8)

    wq = decode_f8(bits, "f8e4m3").astype(np.float32) * wscale
    ref_dev = (decode_f8(got_x8, "f8e4m3") * got_s).astype(np.float32) @ wq.T
    assert float(np.abs(y - ref_dev).max()) / max(float(np.abs(ref_dev).max()), 1e-9) < 2e-3
    s = (np.max(np.abs(x.astype(np.float32)), axis=1, keepdims=True) / np.float32(448.0)).astype(np.float32)
    xq = decode_f8(encode_f8((x.astype(np.float32) / s).astype(np.float32), "f8e4m3"), "f8e4m3") * s
    ref_q = xq.astype(np.float32) @ wq.T
    assert float(np.abs(y - ref_q).max()) / max(float(np.abs(ref_q).max()), 1e-9) < 1e-2  # host/device encode boundary ties


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_w8a8_needs_the_gate_to_enumerate_cuda(monkeypatch):
    """Without ``FP8_MMA`` / ``FAST_MATH`` (and no TILE pin) the same graph compiles OFF the
    native fp8 tier — no k32 mma anywhere — and still matches the quantized reference through the
    scalar tier's per-element decode (the conservative default)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.loader.binder import bind_constants

    monkeypatch.delenv("EMMY_FP8_MMA", raising=False)
    monkeypatch.delenv("EMMY_FAST_MATH", raising=False)
    m, n, k = 32, 512, 512
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 256, (n, k)).astype(np.uint8)
    bits[bits == 0x7F] = 0x00
    bits[bits == 0xFF] = 0x80
    wscale = (np.abs(rng.standard_normal((n, 1))) * 0.005 + 0.002).astype(np.float32)
    act_scale = np.array([[0.02]], dtype=np.float32)
    x = (rng.standard_normal((m, k)) * 0.05).astype(np.float16)

    backend = CudaBackend()
    compiled = backend.compile(_w8a8_graph(m, n, k))
    srcs = [getattr(nd.op, "kernel_source", "") or "" for nd in compiled.nodes.values()]
    assert not any("m16n8k32" in s for s in srcs), "the k32 atom enumerated without the precision gate"

    input_data: dict = {"x": x}
    input_data.update(bind_constants(compiled, {"act.scale": act_scale, "layer.wscale": wscale, "layer.weight": bits}))
    result, _ = backend.run(compiled, input_data=input_data)
    y = result.outputs["y"].reshape(m, n).astype(np.float32)
    xq = decode_f8(encode_f8(x.astype(np.float32) / act_scale, "f8e4m3"), "f8e4m3") * act_scale
    wq = decode_f8(bits, "f8e4m3").astype(np.float32) * wscale
    ref_q = xq.astype(np.float32) @ wq.T
    assert float(np.abs(y - ref_q).max()) / max(float(np.abs(ref_q).max()), 1e-9) < 2e-3
