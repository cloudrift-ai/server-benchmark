"""The block-scaled FP4 atom — ``m16n8k64`` with the scales the instruction applies itself.

Every other registered mma takes three fragments (A, B, and the accumulator standing in for both
C and D). This one takes two more: a packed b32 of ue4m3 block scales per side, one scale per 16
K elements, which the hardware multiplies into each block's partial product. That extra pair is
what the IR node, the wrapper, and the fragment loaders here exist to carry.

It also compiles differently. ptxas assembles this instruction only for the arch-SUFFIXED target
(``sm_120a``) and refuses it on ``sm_100a``, so a kernel carrying it has the same requirement TMA
has, for an unrelated reason — hence the shared ``arch_specific`` flag rather than a TMA-named one.

Nothing offers this atom to the scheduler yet: both operands would have to arrive already
quantized, and quantizing an activation is its own step. These tests cover the pieces that exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.context import Context
from emmy.compiler.dtype import F32, F4E2M1x2
from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.kernel.ir import MmaSyncPtx
from tests.compiler.helpers import requires_cuda

ATOM = "mma_m16n8k64_e2m1_f32"


def test_the_atom_is_registered_with_packed_operands_and_an_f32_accumulator():
    atom = ATOM_REGISTRY[ATOM]
    assert atom.shape == (16, 8, 64)  # k counts VALUES, which is twice the bytes a row holds
    assert atom.operand_dtype("a") is F4E2M1x2
    assert atom.operand_dtype("b") is F4E2M1x2
    assert atom.operand_dtype("c") is F32
    assert (atom.fragment_nregs("a"), atom.fragment_nregs("b")) == (4, 2)
    assert atom.target_feature == "has_block_scaled_f4_mma"


def test_the_atom_name_matches_the_wrapper_the_render_calls():
    # ``ab_dtype`` is the PTX dtype token, and the wrapper name is built from it, so a mismatch
    # here would emit a call to a helper the prelude never defines.
    assert ATOM_REGISTRY[ATOM].ab_dtype == "e2m1"


def test_the_atom_accepts_only_materialized_operands():
    """Conservative first scope: no staged slab drain exists for a packed-plus-scales operand, so
    the atom takes gmem-direct materialized edges only."""
    atom = ATOM_REGISTRY[ATOM]
    assert atom.materialized_edges_only
    assert not atom.sync_copy_staging


@pytest.mark.parametrize(
    ("cc", "available"),
    [((12, 0), True), ((10, 0), False), ((9, 0), False), ((8, 9), False), ((8, 0), False)],
)
def test_only_the_consumer_blackwell_family_offers_the_atom(cc, available):
    """cc 10.0 is datacenter Blackwell, where ptxas refuses this instruction — so the check asks
    for the 12 family rather than 'Blackwell or newer'."""
    assert ATOM_REGISTRY[ATOM].available_on(Context(compute_capability=cc)) is available


def test_the_mma_node_carries_both_scale_fragments():
    node = MmaSyncPtx(c_frag="c", a_frag="a", b_frag="b", shape=(16, 8, 64), ab_dtype="e2m1", sfa_frag="sa", sfb_frag="sb")
    assert node.block_scaled
    # The scales are read, so a reordering pass has to keep them defined before the mma.
    assert node.deps() == ("c", "a", "b", "sa", "sb")


def test_a_plain_mma_node_carries_no_scales():
    node = MmaSyncPtx(c_frag="c", a_frag="a", b_frag="b", shape=(16, 8, 16))
    assert not node.block_scaled
    assert node.deps() == ("c", "a", "b")


def test_the_block_scaled_call_passes_scales_where_the_others_repeat_the_accumulator():
    from emmy.compiler.ir.stmt.base import RenderCtx

    node = MmaSyncPtx(c_frag="c", a_frag="a", b_frag="b", shape=(16, 8, 64), ab_dtype="e2m1", sfa_frag="sa", sfb_frag="sb")
    line = node.render(RenderCtx())[0].strip()
    assert line == "emmy_mma_m16n8k64_e2m1_f32(c, a, b, sa, sb);"


def test_a_kernel_carrying_the_block_scaled_mma_asks_for_the_arch_suffixed_target():
    """TMA announces itself through its descriptors; this instruction has no such marker in a
    plan, so the wrapper name in the rendered source is what identifies it."""
    from emmy.compiler.backend.plan import _needs_arch_isa

    class _Op:
        tma_descriptors = ()
        kernel_source = "... emmy_mma_m16n8k64_e2m1_f32(c, a, b, sa, sb); ..."

    class _Plain:
        tma_descriptors = ()
        kernel_source = "... emmy_mma_m16n8k16_f16_f32(c, a, b, c); ..."

    class _Tma:
        tma_descriptors = ("desc",)
        kernel_source = "... cp.async.bulk.tensor ..."

    assert _needs_arch_isa(_Op())
    assert _needs_arch_isa(_Tma())
    assert not _needs_arch_isa(_Plain())


# The driver around the prelude under test: one warp, one atom cell, gmem-direct operands. A
# packed row is K/2 = 32 bytes, which is what the reused ``_b8`` loaders take as their stride.
_DRIVER = """
extern "C" __global__ void k_f4_block_scaled(float* out, const unsigned char* a,
                                             const unsigned char* b,
                                             const unsigned char* sa, const unsigned char* sb) {
    unsigned ra[4], rb[2];
    emmy_mma_load_a_gmem_b8<unsigned char>(ra, a, 32);
    emmy_mma_load_b_gmem_trans_b8<unsigned char>(rb, b, 32);
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    emmy_mma_m16n8k64_e2m1_f32(d, ra, rb, emmy_mma_load_sfa_f4(sa, 4), emmy_mma_load_sfb_f4(sb, 4));
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    for (int i = 0; i < 4; ++i) out[(grp + ((i & 2) ? 8 : 0)) * 8 + (tig << 1) + (i & 1)] = d[i];
}
"""


def _ue4m3(bits: np.ndarray) -> np.ndarray:
    """Decode UNSIGNED e4m3 — the scale format, where the sign bit the signed form uses is unused."""
    exponent, mantissa = (bits >> 3) & 0xF, bits & 7
    subnormal = mantissa / 8 * 2.0**-6
    normal = (1 + mantissa / 8) * 2.0 ** (exponent.astype(np.int32) - 7)
    return np.where(exponent == 0, subnormal, normal).astype(np.float32)


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_block_scaled_atom_matches_the_decoded_oracle():
    """Compiles the prelude the render emits and runs one atom cell against numpy.

    Both the instruction and the scale placement are exact, so this admits bitwise comparison
    rather than a tolerance: e2m1 values and ue4m3 scales are small integers times powers of two,
    and their products fit f32 exactly.
    """
    import cupy as cp

    from emmy.compiler.backend.cuda import nvcc
    from emmy.compiler.dtype import decode_f4x2
    from emmy.compiler.ir.kernel.render import _MMA_F4_BLOCK_PRELUDE, _MMA_F8_PRELUDE

    if cp.cuda.Device().compute_capability[:2] != "12":
        pytest.skip("the block-scaled fp4 mma assembles only for the consumer Blackwell family")

    rng = np.random.default_rng(23)
    a_bits = rng.integers(0, 256, size=(16, 32), dtype=np.uint8)  # 16 rows x 64 packed values
    b_bits = rng.integers(0, 256, size=(8, 32), dtype=np.uint8)  # N-major, K contiguous
    # Moderate exponents keep every product exactly representable, which is what lets the
    # comparison below be exact.
    sa_bits = (0x30 + rng.integers(0, 0x19, size=(16, 4))).astype(np.uint8)
    sb_bits = (0x30 + rng.integers(0, 0x19, size=(8, 4))).astype(np.uint8)

    fn = nvcc.load_function(_MMA_F8_PRELUDE + _MMA_F4_BLOCK_PRELUDE + _DRIVER, "k_f4_block_scaled", (), arch_specific=True)
    out = cp.zeros((16, 8), dtype=cp.float32)
    args = (out, cp.asarray(a_bits), cp.asarray(b_bits), cp.asarray(sa_bits), cp.asarray(sb_bits))
    fn((1, 1, 1), (32, 1, 1), args)
    got = cp.asnumpy(out)

    a = decode_f4x2(a_bits).astype(np.float32)
    b = decode_f4x2(b_bits).astype(np.float32)
    per_block = (a.reshape(16, 4, 16)[:, None] * b.reshape(8, 4, 16)[None]).sum(-1)
    want = (per_block * _ue4m3(sa_bits)[:, None, :] * _ue4m3(sb_bits)[None, :, :]).sum(-1)
    np.testing.assert_array_equal(got, want)
