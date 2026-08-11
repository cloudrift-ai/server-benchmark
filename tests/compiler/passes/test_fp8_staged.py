"""Staged fp8 transport + the cooperative byte-slab drain (the M2/M3 staged residual).

The warp stage resolver now OFFERS a staged form for 1-byte operands instead of refusing them:
an fp8-stored B under a 16-bit atom stages as a RAW BYTE slab whose drain converts to 16-bit
fragments (W8A16 — the transposed-B slab pair-converts with one hardware ``cvt.rn.f16x2.e4m3x2``
per k-half), and the fp8 (k32) atoms stage BOTH operands as byte slabs whose drain is a byte
repack (contiguous-K lanes load one u32). The transports are the EXISTING d<n>/cp|tma fork
family — no new knob; 16-bit operands resolve exactly as before. A cp.async byte slab pads its
rows (``BYTE_SLAB_PAD``) for the drain's bank spread; a TMA byte slab deposits dense through the
U8 descriptor. Staged is bit-identical to gmem-direct (same converts, same K order, same atoms).
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.diagnostics.bank_conflicts import lane_bank_distribution
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F8E4M3, F16, F32, DataType
from emmy.compiler.graph import Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.schedule import Stage, TilePlan, Workers
from emmy.compiler.ir.stmt import Load
from emmy.compiler.ir.tile import Channel, Fold
from emmy.compiler.pipeline.passes.lowering._addr import BYTE_SLAB_PAD
from emmy.compiler.pipeline.passes.lowering.tile._legality import resolve_warp_stage
from emmy.compiler.pipeline.search.space import stage_moves
from tests.compiler.helpers import requires_cuda

K16 = "mma_m16n8k16_f16_f32"
K32 = "mma_m16n8k32_e4m3_f32"


def _tma_pins() -> tuple[str, ...]:
    """The TMA stage pins, only where the live device has TMA (sm_90+) — below it a TMA pin
    correctly resolves to gmem-direct, which is not the staged form under test."""
    import cupy

    cap = cupy.cuda.Device().compute_capability  # e.g. "89" / "120"
    return ("d2/tma",) if int(cap) >= 90 else ()


def _node(*, a_dtype: DataType = F16, b_dtype: DataType = F8E4M3, m=512, n=4096, k=4096, b_trans=False):
    ka = Axis("k", Dim(k))
    a = Load(name="a", input="x", index=(Var("m"), Var("k")), dtype=a_dtype)
    b_index = (Var("n"), Var("k")) if b_trans else (Var("k"), Var("n"))
    b = Load(name="wb", input="w_bits", index=b_index, dtype=b_dtype)
    node = Fold.contraction(k_axis=ka, a=a, channels=(Channel(b=b, acc="acc"),))
    b_shape = (n, k) if b_trans else (k, n)
    inputs = {"x": Tensor("x", (m, k), a_dtype), "w_bits": Tensor("w_bits", b_shape, b_dtype)}
    return node, inputs, (Axis("m", Dim(m)), Axis("n", Dim(n)))


def _tile(atom: str, spec: str, work: str, mn):
    return TilePlan.parse(f"{atom}/{spec}", Workers.parse(work)).at(*mn)


# ===================================================================
# The resolver's byte-staged offer (replacing the M2b refusal)
# ===================================================================


def test_fp8_b_under_k16_atom_resolves_every_transport():
    """The W8A16 byte-B slab: cp.async AND TMA resolve for both B orientations, carrying the
    resolved ``bk_elems`` (the fp8-off spelling contract)."""
    for b_trans in (False, True):
        node, inputs, mn = _node(b_trans=b_trans)
        tile = _tile(K16, "f4x1/k4", "w1x8", mn)
        for spec in ("d1/cp", "d2/cp", "d2/tma", "d2/cp/p2"):
            st = resolve_warp_stage(node, tile, Stage.parse(spec), 100 * 1024, inputs)
            assert st is not None, (b_trans, spec)
            assert st.bk_elems == tile.bk * 16


def test_fp8_b_slot_bytes_include_the_row_pad():
    """The cp.async byte slab budgets its padded rows: a budget that fits the dense slot but not
    the padded one must clamp the depth (the resolver and the materializer size with ONE rule)."""
    node, inputs, mn = _node(b_trans=True)
    tile = _tile(K16, "f4x1/k4", "w1x8", mn)
    bk = tile.bk * 16
    a_slot = tile.m.tile * bk * 2
    b_slot_padded = tile.n.tile * (bk + BYTE_SLAB_PAD)
    st = resolve_warp_stage(node, tile, Stage.parse("d2/cp"), 2 * (a_slot + b_slot_padded), inputs)
    assert st is not None and st.depth == 2
    st = resolve_warp_stage(node, tile, Stage.parse("d2/cp"), 2 * (a_slot + b_slot_padded) - 1, inputs)
    assert st is not None and st.depth == 1


def test_byte_staging_declines_what_it_cannot_fill():
    """The stated refusals: a non-f8 dtype mismatch, an fp8 A under a 16-bit atom, a canonical
    byte-B whose tile_n or N is not 16-divisible (the 16 B chunk / row-pad rule)."""
    # f32-stored B under a 16-bit atom: not a byte slab, not the atom dtype — decline
    node, inputs, mn = _node(b_dtype=F32)
    assert resolve_warp_stage(node, _tile(K16, "f4x1/k4", "w1x8", mn), Stage.parse("d2/cp"), 100 * 1024, inputs) is None
    # fp8-stored A under a 16-bit atom: only B has the convert drain — decline
    node, inputs, mn = _node(a_dtype=F8E4M3, b_dtype=F16)
    assert resolve_warp_stage(node, _tile(K16, "f4x1/k4", "w1x8", mn), Stage.parse("d2/cp"), 100 * 1024, inputs) is None
    # canonical byte-B, tile_n = 8 (w1x1/f1x1): 16 does not divide the inner span — decline
    node, inputs, mn = _node(n=4096)
    assert resolve_warp_stage(node, _tile(K16, "f1x1/k4", "w1x1", mn), Stage.parse("d2/cp"), 100 * 1024, inputs) is None
    # canonical byte-B whose gmem row stride N is 16-indivisible — decline
    node, inputs, mn = _node(n=4104)
    assert resolve_warp_stage(node, _tile(K16, "f4x1/k4", "w1x8", mn), Stage.parse("d2/cp"), 100 * 1024, inputs) is None


def test_k32_atoms_resolve_staged_byte_slabs():
    """The fp8 (k32) atoms stage both operands as byte slabs — cp.async and TMA resolve; a
    symbolic K still declines (no masked-K byte fill)."""
    node, inputs, mn = _node(a_dtype=F8E4M3, b_dtype=F8E4M3, m=512, n=512, k=512)
    tile = _tile(K32, "f4x1/k4", "w1x8", mn)
    for spec in ("d2/cp", "d2/tma"):
        st = resolve_warp_stage(node, tile, Stage.parse(spec), 100 * 1024, inputs)
        assert st is not None and st.bk_elems == tile.bk * 32
    ka = Axis("k", Dim("seq"))
    sym = Fold.contraction(
        k_axis=ka,
        a=Load(name="a", input="x", index=(Var("m"), Var("k")), dtype=F8E4M3),
        channels=(Channel(b=Load(name="wb", input="w_bits", index=(Var("k"), Var("n")), dtype=F8E4M3), acc="acc"),),
    )
    assert resolve_warp_stage(sym, tile, Stage.parse("d2/cp"), 100 * 1024, inputs) is None


def test_16bit_operands_resolve_exactly_as_without_dtype_info():
    """The fp8-off enumeration is unchanged: for 16-bit operands every catalog move resolves to
    the same spelling with and without the dtype info (the byte arm engages only at nbytes==1),
    so no new forks appear for the 16-bit family."""
    node, inputs, mn = _node(b_dtype=F16)
    tile = _tile(K16, "f4x1/k4", "w1x8", mn)
    for move in stage_moves(warp=True):
        with_info = resolve_warp_stage(node, tile, move, 100 * 1024, inputs)
        without = resolve_warp_stage(node, tile, move, 100 * 1024, None)
        assert (with_info is None) == (without is None)
        if with_info is not None:
            assert with_info.spell() == without.spell()


# ===================================================================
# The drain's bank spread — the BYTE_SLAB_PAD regression (the lane→bank oracle)
# ===================================================================


def _drain_max_way(rows: int, cols: int, pad: int, reads) -> int:
    """Worst per-instruction ``max_way`` over the drain's per-lane slab reads. ``reads`` yields
    ``(row_expr, col_expr)`` per emitted load, in the fragment loaders' lane→element map."""
    taxes = (Axis("_t", Dim(32)),)
    worst = 0
    for row, col in reads:
        d = lane_bank_distribution((row, col), (rows, cols + pad), taxes, elem_bytes=1)
        assert d is not None
        worst = max(worst, d.max_way)
    return worst


def _lane_exprs():
    lane = Var("_t")
    return BinaryExpr("/", lane, Literal(4, "int")), BinaryExpr("%", lane, Literal(4, "int"))


def _k16_trans_reads(grp, tig):
    # emmy_mma_load_b_smem_trans_f8_f16: one fp8x2 (2 B) load per k-half — model its first byte
    # (the pair shares the bank word).
    return [(grp, BinaryExpr("+", BinaryExpr("*", tig, Literal(2, "int")), Literal(8 * i, "int"))) for i in (0, 1)]


def _k32_reads(grp, tig, *, a_side: bool):
    # the u32 loads: (row, 4*tig + 16*half) — A visits rows grp / grp+8, trans-B row grp.
    out = []
    for i in range(4 if a_side else 2):
        row = BinaryExpr("+", grp, Literal(8 * (i & 1), "int")) if a_side else grp
        half = (16 if (i & 2) else 0) if a_side else (16 if i else 0)
        out.append((row, BinaryExpr("+", BinaryExpr("*", tig, Literal(4, "int")), Literal(half, "int"))))
    return out


def test_byte_slab_pad_keeps_the_drain_conflict_free():
    """The 16 B row pad's reason, stated as a regression: at ``BYTE_SLAB_PAD`` every byte-slab
    drain instruction is ≤ 2-way (the oracle's broadcast-corrected worst bank); dense rows
    (pad 0 — the TMA deposit) are 4-way on the same maps."""
    grp, tig = _lane_exprs()
    cases = [
        ("k16 trans-B", 128, 64, _k16_trans_reads(grp, tig)),
        ("k32 trans-B", 128, 64, _k32_reads(grp, tig, a_side=False)),
        ("k32 A", 32, 64, _k32_reads(grp, tig, a_side=True)),
    ]
    for name, rows, cols, reads in cases:
        assert _drain_max_way(rows, cols, BYTE_SLAB_PAD, reads) <= 2, name
        assert _drain_max_way(rows, cols, 0, reads) >= 4, name  # the dense layout the pad replaces


# ===================================================================
# CUDA: staged ≡ gmem-direct, both atom families, cp.async + TMA
# ===================================================================


def _run_w8a16(backend, stage_pin, x, bits, scale, m, n, k):
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    from .test_fp8_operand_binding import _fp8_linear_graph

    pins = {"TILE": f"{K16}/f2x2/k2", "WORK": "w1x8", "REDUCE": "", "STAGE": stage_pin or ""}
    with pinned_knobs(pins):
        compiled = backend.compile(_fp8_linear_graph(m, n, k))
    srcs = [getattr(nd.op, "kernel_source", "") or "" for nd in compiled.nodes.values()]
    mma_src = next((s for s in srcs if "mma.sync" in s), "")
    input_data = {"x": x}
    input_data.update(bind_constants(compiled, {"layer.weight": bits, "layer.weight_scale": scale}))
    result, _ = backend.run(compiled, input_data=input_data)
    return np.asarray(result.outputs[compiled.outputs[0]]).reshape(m, n), mma_src


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_w8a16_staged_bit_identical_to_gmem_direct_cuda():
    """Staged fp8-B (cp.async and TMA) is BIT-identical to the gmem-direct fragment-convert
    kernel — same per-element convert, same K order, same atoms — and the staged source carries
    the byte-slab machinery (the fp8 slab decl + a byte-gather drain)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend

    m, n, k = 32, 512, 512
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 256, (n, k)).astype(np.uint8)
    bits[bits == 0x7F] = 0x00
    bits[bits == 0xFF] = 0x80
    scale = (np.abs(rng.standard_normal((n, 1))) * 0.005 + 0.002).astype(np.float32)
    x = (rng.standard_normal((m, k)) * 0.05).astype(np.float16)

    backend = CudaBackend()
    base, base_src = _run_w8a16(backend, None, x, bits, scale, m, n, k)
    assert "_b_smem" not in base_src
    for pin in ("d1/cp", "d2/cp", "d2/cp/p2", *_tma_pins()):
        y, src = _run_w8a16(backend, pin, x, bits, scale, m, n, k)
        assert "__nv_fp8_e4m3 _b_smem" in src or "__nv_fp8_e4m3* _b_smem" in src, pin
        assert "emmy_mma_load_b_smem_trans_f8_f16" in src or "emmy_mma_load_b_gmem" in src, pin
        assert np.array_equal(y.view(np.uint16), base.view(np.uint16)), pin


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_canonical_byte_b_and_splitk_compose_cuda():
    """The CANONICAL (K-major ``B[k, n]``) byte slab — K-strided bytes, so the drain is the
    scalar convert gather — and split-K composed with the staged byte slab. Each staged form is
    bit-identical to the gmem-direct kernel of the SAME reduce plan (split-K reassociates, so
    the serial kernel is not its baseline)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.graph import Graph
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.frontend.ir import MatmulOp
    from emmy.compiler.ir.tensor.ir import ElementwiseOp
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    m, n, k = 32, 512, 512
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((m, k)) * 0.05).astype(np.float16)
    wbits = ((rng.integers(0, 256, (k, n)).astype(np.uint8)) & 0x87) | 0x30

    def graph():
        g = Graph()
        g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (m, k), "f16"), node_id="x")
        g.add_node(op=InputOp(), inputs=[], output=Tensor("w8", (k, n), "f8e4m3"), node_id="w8")
        wd = g.add_node(op=ElementwiseOp(op="from_f8e4m3"), inputs=["w8"], output=Tensor("w_dq", (k, n), "f16"))
        g.add_node(op=MatmulOp(), inputs=["x", wd], output=Tensor("y", (m, n), "f16"), node_id="y")
        g.inputs, g.outputs = ["x", "w8"], ["y"]
        return g

    backend = CudaBackend()

    def run(stage, red):
        with pinned_knobs({"TILE": f"{K16}/f2x2/k2", "WORK": "w1x8", "REDUCE": red, "STAGE": stage}):
            compiled = backend.compile(graph())
        srcs = [getattr(nd.op, "kernel_source", "") or "" for nd in compiled.nodes.values()]
        src = next((s for s in srcs if "mma.sync" in s), "")
        result, _ = backend.run(compiled, input_data={"x": x, "w8": wbits})
        return np.asarray(result.outputs["y"]).reshape(m, n), src

    bases: dict[str, np.ndarray] = {}
    for stage, red in [("d2/cp", ""), ("d2/cp", "g4k"), *((s, r) for s in _tma_pins() for r in ("", "g4k"))]:
        if red not in bases:
            bases[red] = run("", red)[0]
        y, src = run(stage, red)
        assert "_b_smem" in src and "emmy_mma_load_b_gmem<__nv_fp8_e4m3, __half>" in src, (stage, red)
        assert np.array_equal(y.view(np.uint16), bases[red].view(np.uint16)), (stage, red)


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_k32_staged_bit_identical_to_gmem_direct_cuda():
    """Staged W8A8 (the k32 byte repack — raw bytes both slabs) is BIT-identical to the
    gmem-direct ``_b8`` gathers, and the contiguous-K drains ride the vector (u32) loaders."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    from .test_fp8_mma import _bare_f8_linear_graph

    m, n, k = 32, 512, 512
    rng = np.random.default_rng(11)
    abits = ((rng.integers(0, 256, (m, k)).astype(np.uint8)) & 0x87) | 0x30
    wbits = ((rng.integers(0, 256, (n, k)).astype(np.uint8)) & 0x87) | 0x30
    backend = CudaBackend()

    def run(stage_pin):
        pins = {"TILE": f"{K32}/f2x2/k2", "WORK": "w1x8", "REDUCE": "", "STAGE": stage_pin or ""}
        with pinned_knobs(pins):
            compiled = backend.compile(_bare_f8_linear_graph(m, n, k))
        srcs = [getattr(nd.op, "kernel_source", "") or "" for nd in compiled.nodes.values()]
        src = next((s for s in srcs if "m16n8k32" in s), "")
        result, _ = backend.run(compiled, input_data={"a8": abits, "w8": wbits})
        return np.asarray(result.outputs["y"]).reshape(m, n), src

    base, base_src = run(None)
    assert "_a_smem" not in base_src
    for pin in ("d1/cp", "d2/cp", *_tma_pins()):
        y, src = run(pin)
        assert "emmy_mma_load_a_smem_b8v" in src, pin
        assert np.array_equal(y.view(np.uint32), base.view(np.uint32)), pin
