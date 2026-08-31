"""FP8 operand traits, the k-invariant dequant binding, staged byte slabs, and warp lowering.

The binding section below was deleted with ``_classify.bind_bilinear`` and is RESTORED here against
the canonical Fold tree, because the contract it pins is independent of which pass owns it: a
computed **B** whose cone is a storage decode (recognized by the ``ElementwiseImpl.decodes`` trait,
never an op-name list) times k-invariant factors must canonicalize to the RAW storage-dtype
``Load`` — the decode absorbed by dtype, every consumer converting a bits-carrier element — with
the factors moved onto the accumulator in the epilogue (``sum_k a*(s*w) = s*sum_k a*w``). A pure
map that cannot commute out — a k-varying (2-D block) scale, or another computed B — must remain a
closed computed operand rather than being positionally misbound to an interior load, and a B
producer reading the output-row axis must decline outright.

Losing the hoist is not a correctness bug and no numerics assert catches it: the raw f8 load is
what lets the mma tier read B gmem-direct at storage width, while a computed cone routes the same
weights through the smem compute fill.
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
from emmy.compiler.ir.pure.fold import Channel, Fold, is_contraction
from emmy.compiler.ir.schedule import Stage, TilePlan, Workers
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.tile import Placement, TileOp
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import _stamp_axes, fold_from_loop
from emmy.compiler.pipeline.passes.lowering.tile._staging import resolve_warp_stage
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
# The k-invariant multiplicative dequant binding (restored)
# ===================================================================


def _bind(loop, m: str = "m", n: str = "n"):
    """Lift the loop through the ONE parser, then canonicalize it as a stored Fold tree.

    Returns ``(contraction, epilogue)`` — the canonical contraction and the projection statements
    left around it — or ``None`` when the tree does not canonicalize to a contraction at all (the
    PLANAR reading, which is the decline this section's negative cases assert)."""
    fold = fold_from_loop(_stamp_axes(loop))
    assert fold is not None, "the dequant loop must lift"
    tile = TileOp(op=Fold.projection(body=Body((fold,))), place=Placement(free=(Axis(m, Dim(64)), Axis(n, Dim(64)))))
    root = tile.op
    if is_contraction(root):
        return root, ()
    inner = [s for s in root.lift.body if isinstance(s, Fold) and is_contraction(s)]
    inner += [o for o in root.operands if isinstance(o, Fold) and is_contraction(o)]
    if not inner:
        return None
    contraction = inner[0]
    epilogue = tuple(s for s in root.lift.body if s is not contraction)
    return contraction, epilogue


def _dequant_loop(*, scale_index=None, scale_op="multiply", decode="from_f8e4m3", extra_factor=False):
    """The fused dequant matmul loop body the fp8 expansion + loop fusion produce:
    ``acc += x[m,k] * (s[n] (x) from_f8*(w[k,n]))`` with the scale load hoistable (k-invariant)
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
    """The per-out-channel fp8 weight: B is the RAW f8 load, the scale rides the epilogue."""
    bound = _bind(_dequant_loop())
    assert bound is not None, "the dequant contraction demoted to PLANAR"
    con, epi = bound
    a, b = con.a, con.b
    assert isinstance(a, Load) and a.input == "x"
    assert isinstance(b, Load) and b.input == "w_bits"  # the RAW f8 load — decode absorbed by dtype
    scale = [s for s in epi if isinstance(s, Load) and s.input == "w_scale"]
    assert scale, "the k-invariant scale did not hoist to the epilogue"
    tail = [s for s in epi if isinstance(s, Assign)][-1]
    assert tail.op.name == "multiply" and "s" in tail.args


def test_inverse_scale_hoists_as_divide():
    """``weight_scale_inv`` spells the cone with a divide — it commutes out the same way."""
    bound = _bind(_dequant_loop(scale_op="divide"))
    assert bound is not None, "the inverse-scale contraction demoted to PLANAR"
    con, epi = bound
    assert isinstance(con.b, Load) and con.b.input == "w_bits"
    tail = [s for s in epi if isinstance(s, Assign)][-1]
    assert tail.op.name == "divide" and "s" in tail.args


def test_factor_chain_hoists_every_k_invariant_factor():
    """Two k-invariant factors compose into ONE epilogue chain; neither stays in the fold."""
    bound = _bind(_dequant_loop(extra_factor=True))
    assert bound is not None, "the two-factor contraction demoted to PLANAR"
    con, epi = bound
    assert isinstance(con.b, Load) and con.b.input == "w_bits"
    assigns = [s for s in epi if isinstance(s, Assign)]
    assert assigns and {s.op.name for s in assigns} == {"multiply"}
    assert {s.input for s in epi if isinstance(s, Load)} == {"w_scale", "w_scale2"}


def test_original_epilogue_reads_the_scaled_value():
    """The factor chain's last definition carries the value any projection statement reads, so a
    consumer of the fold's output reads the SCALED value, never the bare accumulator."""
    bound = _bind(_dequant_loop())
    assert bound is not None
    con, epi = bound
    assigns = [s for s in epi if isinstance(s, Assign)]
    assert assigns, "no epilogue chain — nothing rescales the accumulator"
    assert con.acc in assigns[0].args or con.out in assigns[0].args


def test_k_varying_scale_binds_as_whole_computed_b_cone():
    """A 2-D scale cannot commute out, so the complete generic map remains the B operand."""
    bound = _bind(_dequant_loop(scale_index=(Var("k"), Var("n"))))
    assert bound is not None, "the k-varying dequant demoted to PLANAR"
    con, epi = bound
    assert isinstance(con.a, Load) and isinstance(con.b, Fold) and con.b.axis is None
    assert not [s for s in epi if isinstance(s, Load) and s.input == "w_scale"], "a k-varying scale must NOT hoist"
    cone = list(con.b.body)
    assert {s.input for s in cone if isinstance(s, Load)} == {"w_scale", "w_bits"}


def test_non_decode_computed_b_preserves_cone_instead_of_positional_misbind():
    """An arbitrary pure map on B binds as a whole; the interior load is never misbound alone."""
    bound = _bind(_dequant_loop(decode="exp"))
    assert bound is not None, "the non-decode cone demoted to PLANAR"
    con, _epi = bound
    assert isinstance(con.a, Load) and isinstance(con.b, Fold)
    assert any(isinstance(s, Assign) and s.op.name == "exp" for s in con.b.body)


def test_m_dependent_b_cone_declines_instead_of_crossing_operand_roles():
    """A B producer that reads the output-row axis is not a separable (k,n) operand — the tree
    keeps its PLANAR reading; nothing is positionally misbound."""
    assert _bind(_dequant_loop(scale_index=(Var("m"), Var("k")))) is None


def test_bare_decode_binds_raw_load_without_epilogue():
    """No k-invariant factor at all: B still binds as the raw f8 load, nothing hoists."""
    loop = Loop(
        axis=Axis("k", Dim(64)),
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
    bound = _bind(loop)
    assert bound is not None, "the bare decode contraction demoted to PLANAR"
    con, epi = bound
    assert isinstance(con.b, Load) and con.b.input == "w_bits"
    assert not [s for s in epi if isinstance(s, Assign)]


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


def test_resolve_warp_stage_declines_packed_pair_b():
    """A packed-pair byte (f4e2m1x2) is not an fp8 byte: one stored element is two logical
    K elements, so granting the fp8 byte slab would halve K. Every copy transport refuses."""
    from emmy.compiler.dtype import F4E2M1x2

    node, tile = _warp_contraction()
    inputs = {"x": Tensor("x", (512, 4096), F16), "w_bits": Tensor("w_bits", (4096, 2048), F4E2M1x2)}
    for spec in ("d2/smem-async", "d2/smem-tma"):
        assert resolve_warp_stage(node, tile, Stage.parse(spec), 100 * 1024, inputs) is None


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


# ===================================================================
# The packed-pair k-block matcher (NVFP4 phase 3c groundwork)
# ===================================================================


def _packed_kblock_body(scale_stride=32, scale_k=None):
    """The NVFP4 speller's cone, hand-built in the lowering's idiom (flat / and % reshape
    arithmetic):
    e4m3 block scale (k under /16), fused-scale multiply, packed byte load, index copy,
    pair-table gather, final value x factor multiply. ``scale_k`` overrides the scale
    Load's k expression for the negative cases."""
    from emmy.compiler.ir.expr import BinaryExpr, CastExpr, Literal

    n, k = Var("a1"), Var("a2")
    flat = BinaryExpr("+", BinaryExpr("*", n, Literal(scale_stride, "int")), k)
    sblock = scale_k if scale_k is not None else BinaryExpr("%", BinaryExpr("/", flat, Literal(16, "int")), Literal(2, "int"))
    byte_idx = BinaryExpr("%", BinaryExpr("/", flat, Literal(2, "int")), Literal(16, "int"))
    return [
        Load(name="in1", input="w_scale_bits", index=(n, sblock), dtype=None),
        Assign(name="v0", op="from_f8e4m3", args=("in1",)),
        Load(name="in2", input="w_bits", index=(n, byte_idx), dtype=None),
        Assign(name="v1", op="multiply", args=("in0", "v0")),
        Assign(name="v2", op="copy", args=("in2",)),
        Assign(name="v3", op="copy", args=("v1",)),
        Load(name="in3", input="w_f4_pairs", index=(CastExpr(dtype="int", expr=Var("v2")), Literal(0, "int")), dtype=None),
        Assign(name="v4", op="multiply", args=("in3", "v3")),
    ]


def _packed_inputs():
    from emmy.compiler.dtype import F4E2M1x2

    # Shapes cohere with the body's K=32 arithmetic: 2 scale blocks and 16 packed
    # bytes per row (the matcher itself reads only the dtypes).
    return {
        "w_scale_bits": Tensor("w_scale_bits", (4096, 2), F8E4M3),
        "w_bits": Tensor("w_bits", (4096, 16), F4E2M1x2),
        "w_f4_pairs": Tensor("w_f4_pairs", (256, 2), F32),
    }


def test_packed_kblock_b_matches_the_spelled_shape():
    from emmy.compiler.pipeline.passes.lowering._packed import match_packed_kblock_b

    body = _packed_kblock_body()
    cone = list(Body(tuple(body)).backward_cone(["v4"]).members)
    got = match_packed_kblock_b(cone, "a2", _packed_inputs())
    assert got is not None
    assert got.bits.input == "w_bits" and got.table.input == "w_f4_pairs"
    assert got.factor == "v3" and got.block == 16


def test_packed_kblock_b_declines_misaligned_row_stride():
    # (n*24 + k)/16 is NOT constant on k blocks of 16 — the k-free addend must be a
    # multiple of the divisor.
    from emmy.compiler.pipeline.passes.lowering._packed import match_packed_kblock_b

    body = _packed_kblock_body(scale_stride=24)
    got = match_packed_kblock_b(list(Body(tuple(body)).backward_cone(["v4"]).members), "a2", _packed_inputs())
    assert got is None


def test_packed_kblock_b_declines_naked_k_scale():
    from emmy.compiler.pipeline.passes.lowering._packed import match_packed_kblock_b

    body = _packed_kblock_body(scale_k=Var("a2"))
    got = match_packed_kblock_b(list(Body(tuple(body)).backward_cone(["v4"]).members), "a2", _packed_inputs())
    assert got is None


def test_packed_kblock_b_declines_without_a_packed_load():
    from emmy.compiler.pipeline.passes.lowering._packed import match_packed_kblock_b

    inputs = _packed_inputs()
    inputs["w_bits"] = Tensor("w_bits", (4096, 16), F8E4M3)
    got = match_packed_kblock_b(list(Body(tuple(_packed_kblock_body())).backward_cone(["v4"]).members), "a2", inputs)
    assert got is None


def test_packed_kblock_b_declines_mixed_guarded_and_naked_k():
    # One index expr holds a guarded division AND a bare k: the naked flag alone must
    # decline (the guard set is non-empty, so the single-block check would pass).
    from emmy.compiler.ir.expr import BinaryExpr, Literal
    from emmy.compiler.pipeline.passes.lowering._packed import match_packed_kblock_b

    n, k = Var("a1"), Var("a2")
    flat = BinaryExpr("+", BinaryExpr("*", n, Literal(32, "int")), k)
    mixed = BinaryExpr("+", BinaryExpr("/", flat, Literal(16, "int")), k)
    body = _packed_kblock_body(scale_k=mixed)
    assert match_packed_kblock_b(list(Body(tuple(body)).backward_cone(["v4"]).members), "a2", _packed_inputs()) is None


def test_packed_kblock_b_declines_squared_gather():
    # multiply(in3, in3): the gathered value on both args leaves no factor — None, not
    # an exception.
    from emmy.compiler.pipeline.passes.lowering._packed import match_packed_kblock_b

    body = _packed_kblock_body()
    body[-1] = Assign(name="v4", op="multiply", args=("in3", "in3"))
    assert match_packed_kblock_b(list(Body(tuple(body)).backward_cone(["v4"]).members), "a2", _packed_inputs()) is None
