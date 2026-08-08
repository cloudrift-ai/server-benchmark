"""Kernel IR → CUDA source.

Builds a CUDA ``RenderCtx`` (intrinsic + GPU-builtin spelling tables,
per-buf shapes), emits the ``extern "C" __global__ __launch_bounds__(N)
void`` signature, then walks the body — every Stmt's own ``render``
method does the per-line emission.
"""

from __future__ import annotations

from emmy.compiler.backend.cuda.dtype import cuda_includes, cuda_name
from emmy.compiler.backend.cuda.dtype import nbytes_of as _nbytes_of
from emmy.compiler.backend.cuda.render_target import CudaRenderTarget
from emmy.compiler.dtype import F32
from emmy.compiler.ir.kernel.ir import LDMATRIX_SWIZZLE_XOR, CpAsyncCopy, KernelOp, LdmatrixLoad, Smem, TmaDescriptor, pack_smem
from emmy.compiler.ir.stmt import RenderCtx, render_body
from emmy.compiler.ir.stmt.leaves import Write
from emmy.compiler.tensor import Tensor

# Per-CTA static-smem hard cap on every CUDA arch we target. Above this,
# PTXAS rejects ``__shared__ T arr[N]`` decls (``uses too much shared
# data``); the only way to use more is dynamic smem (``extern __shared__``
# + ``cudaFuncAttributeMaxDynamicSharedMemorySize``). When the kernel's
# total Smem footprint exceeds this, ``render_kernelop`` switches to a
# single dynamic pool with per-buffer offsets.
STATIC_SMEM_CAP = 48 * 1024

# TMA / mbarrier prelude. NVRTC doesn't ship ``<cuda.h>`` /
# ``<cuda/ptx>`` / ``<cuda/barrier>``, so we can't ``#include`` the
# stock ``cuda::ptx::*`` intrinsics; pulling those headers in via a
# CUDA-toolkit-version-locked include path also drags in libcu++.
# Instead we forward-declare ``CUtensorMap`` as an opaque 128-byte
# aligned struct (the kernel only takes its address) and define small
# ``__forceinline__`` wrappers around the inline-PTX so the body reads
# like ``mbarrier_init(&mbar[s], 1)`` rather than 5 lines of asm. Same
# generated SASS as raw asm — these helpers fold away at compile time.
_TMA_PRELUDE = """\
struct __align__(64) CUtensorMap { unsigned long long opaque[16]; };

static __device__ __forceinline__ void mbarrier_init(unsigned long long* mbar, int count) {
    unsigned int addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\\n" :: "r"(addr), "r"(count) : "memory");
}

static __device__ __forceinline__ void mbarrier_arrive_expect_tx(unsigned long long* mbar, int bytes) {
    unsigned int addr = __cvta_generic_to_shared(mbar);
    unsigned long long state;
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 %0, [%1], %2;\\n"
                 : "=l"(state) : "r"(addr), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void mbarrier_arrive(unsigned long long* mbar) {
    // Simple arrive — no transaction-byte count. Used by warp-specialized
    // consumer warps to signal "slot empty" after the producer's
    // expect-tx round has been consumed. The async-proxy fence orders the
    // consumer's GENERIC-proxy slab reads (ldmatrix) before the producer's
    // next ASYNC-proxy write into the slot (cp.async.bulk.tensor) — the
    // mbarrier release alone does not cross the proxy boundary, and without
    // the fence the refill can overtake in-flight reads (silent corruption
    // under scheduling pressure). Mirrors CUTLASS PipelineTmaAsync's
    // consumer_release (fence_view_async_shared + arrive).
    unsigned int addr = __cvta_generic_to_shared(mbar);
    unsigned long long state;
    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];\\n"
                 : "=l"(state) : "r"(addr) : "memory");
}

static __device__ __forceinline__ void mbarrier_wait_parity(unsigned long long* mbar, int phase) {
    // Issue one ``mbarrier.try_wait`` first — its hint timeout makes the
    // warp suspend rather than spin while the TMA tx drains, freeing the
    // scheduler to run other warps. The PTX-level ``while !try_wait`` loop
    // is required (try_wait can return early); the suspend hint prevents
    // hot-spinning across all 256 CTA threads (~3-4× kernel speedup on
    // small matmuls where the wait-vs-compute ratio is high).
    //
    // The ``"memory"`` clobber prevents the compiler from reordering
    // smem loads across this asm. The primary correctness anchor is
    // the trailing ``__syncthreads()`` materialize emits after each
    // MbarrierWait (see ``100_materialize_tile.py``); the clobber is
    // defensive belt-and-braces so the asm itself reads as a fence
    // even if a future caller forgets the surrounding Sync.
    unsigned int addr = __cvta_generic_to_shared(mbar);
    asm volatile("{.reg .pred P; bw: mbarrier.try_wait.parity.shared.b64 P, [%0], %1; @!P bra bw;}\\n"
                 :: "r"(addr), "r"(phase) : "memory");
}

static __device__ __forceinline__ void cp_async_bulk_tensor_2d(
    void* smem, const CUtensorMap* desc, int c0, int c1, unsigned long long* mbar) {
    unsigned int saddr = __cvta_generic_to_shared(smem);
    unsigned int maddr = __cvta_generic_to_shared(mbar);
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];\\n"
                 :: "r"(saddr), "l"(desc), "r"(c0), "r"(c1), "r"(maddr) : "memory");
}

static __device__ __forceinline__ void cp_async_bulk_tensor_3d(
    void* smem, const CUtensorMap* desc, int c0, int c1, int c2, unsigned long long* mbar) {
    unsigned int saddr = __cvta_generic_to_shared(smem);
    unsigned int maddr = __cvta_generic_to_shared(mbar);
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];\\n"
                 :: "r"(saddr), "l"(desc), "r"(c0), "r"(c1), "r"(c2), "r"(maddr) : "memory");
}

static __device__ __forceinline__ void cp_async_bulk_tensor_4d(
    void* smem, const CUtensorMap* desc, int c0, int c1, int c2, int c3, unsigned long long* mbar) {
    unsigned int saddr = __cvta_generic_to_shared(smem);
    unsigned int maddr = __cvta_generic_to_shared(mbar);
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6];\\n"
                 :: "r"(saddr), "l"(desc), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(maddr) : "memory");
}

static __device__ __forceinline__ void cp_async_bulk_tensor_5d(
    void* smem, const CUtensorMap* desc, int c0, int c1, int c2, int c3, int c4, unsigned long long* mbar) {
    unsigned int saddr = __cvta_generic_to_shared(smem);
    unsigned int maddr = __cvta_generic_to_shared(mbar);
    asm volatile("cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];\\n"
                 :: "r"(saddr), "l"(desc), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(maddr) : "memory");
}

"""

# cp.async prelude — the sm_80+ cooperative gmem→smem copy that bypasses the
# register file (the async analogue of an ``LDG`` + ``STS`` pair). Wraps the
# inline PTX so the fill loop reads as ``emmy_cp_async_cg(&smem[i], &gmem[j])``
# rather than a ``__cvta_generic_to_shared`` + raw ``asm volatile`` pair — the
# same helper style the mma / mbarrier preludes use; same SASS. ``cg`` =
# cache-global / bypass-L1 (16 B, the streaming form); ``ca`` = cache-all (4/8 B).
# ``commit`` closes a batch of issued copies; ``wait<N>`` blocks until ≤ N of
# those batches are still in flight.
_CP_ASYNC_PRELUDE = """\
static __device__ __forceinline__ void emmy_cp_async_cg(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"(addr), "l"(gmem) : "memory");
}

template <int Bytes>
static __device__ __forceinline__ void emmy_cp_async_ca(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\\n" :: "r"(addr), "l"(gmem), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void emmy_cp_async_commit() {
    asm volatile("cp.async.commit_group;\\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void emmy_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\\n" :: "n"(N) : "memory");
}

"""

# Warp-level MMA prelude (the ``s16816`` path) — the sole tensor-core path.
# Pure inline PTX (``ldmatrix`` + ``mma.sync.aligned``), so NVRTC needs no
# ``<mma.h>``. ``__forceinline__`` wrappers keep the kernel body reading as
# ``emmy_mma_m16n8k16_{f16,bf16}_{f32,f16}(c, a, b, c)`` (the atom convention's <ab>_<acc> dtype pair) rather than raw asm; same SASS.
# The ``a``/``b`` operands are ``unsigned`` 32-bit register arrays (two packed
# 16-bit elems each); ``c``/``d`` are ``float`` (f32 accumulate). Lane→element
# layout is the PTX-fixed mma.m16n8k16 fragment map (see the ``LdmatrixLoad`` /
# ``RegStore`` address arithmetic in ``ir/kernel/ir.py``).
# Trellis-coded (EXL3) weight decode — the ``TrellisLoad`` leaf's realization. ``codes`` points
# at ONE 16x16 tile's packed words (``16*kbits`` uint16 = ``8*kbits`` uint32; the int16 buffer
# read as little-endian uint32 IS the SWAP16 pair join); the element's 16-bit window is bits
# ``[(t+1)*kbits - 16, (t+1)*kbits) mod 256*kbits`` of the circular stream, MSB-first, with the
# step ``t`` derived from the in-tile coords through the mma m16n8k16 B-fragment order. The
# 3INST computed codebook follows — no stored LUT anywhere. Exact fp16 arithmetic, widened once
# to float on return (``loader/exl3.py`` is the bit-exact numpy reference).
_TRELLIS_PRELUDE = """\
static __device__ __forceinline__ float emmy_trellis_decode(
        const unsigned short* __restrict__ codes, int kbits, int cb, int k_lo, int n_lo) {
    int lane = ((n_lo & 7) << 2) | ((k_lo & 7) >> 1);
    int j = (k_lo & 1) | (((k_lo >> 3) & 1) << 1) | (((n_lo >> 3) & 1) << 2);
    int t = (lane << 3) | j;
    int nbits = kbits << 8;
    int b0 = ((t + 1) * kbits - 16 + nbits) % nbits;
    int w = b0 >> 5;
    int nwords = kbits << 3;
    const unsigned int* u32 = (const unsigned int*)codes;
    unsigned long long joined = ((unsigned long long)u32[w] << 32) | (unsigned long long)u32[(w + 1) % nwords];
    unsigned int x = (unsigned int)(joined >> (48 - (b0 & 31))) & 0xFFFFu;
    if (cb == 2) {
        x *= 0x83DCD12Du;
        unsigned int s = ((x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24) + 0x6400u) & 0xFFFFu;
        __half h = __ushort_as_half((unsigned short)s);
        return __half2float(__hfma(h, __ushort_as_half((unsigned short)0x1EEEu), __ushort_as_half((unsigned short)0xC931u)));
    }
    x = (cb == 1) ? x * 0xCBAC1FEDu : x * 89226354u + 64248484u;
    x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
    return __half2float(__hadd(__ushort_as_half((unsigned short)(x & 0xFFFFu)),
                               __ushort_as_half((unsigned short)(x >> 16))));
}
"""

_MMA_M8N8K4_PRELUDE = """\
// Volta m8n8k4: one warp instruction performs four independent 8x8 MMAs.
// The lane's four-lane computation group selects one quadrant of a logical
// 16x16 output cell; groups in the same quadrant row/column duplicate A/B.
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma884_load_a_impl(
    unsigned* r, const T* g, int ldm, int rows_left, int k_left) {
    int lane = threadIdx.x & 31;
    int comp = (lane & 15) >> 2;
    int row = ((comp >> 1) << 3) + (lane & 3) + ((lane >> 4) << 2);
    if (row >= rows_left) row = rows_left - 1;
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        int k = p << 1;
        unsigned packed = 0;
        if (k < k_left) ((F*)&packed)[0] = F(g[row * ldm + k]);
        if (k + 1 < k_left) ((F*)&packed)[1] = F(g[row * ldm + k + 1]);
        r[p] = packed;
    }
}

template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma884_load_b_impl(
    unsigned* r, const T* g, int ldm, int cols_left, int k_left, bool trans) {
    int lane = threadIdx.x & 31;
    int comp = (lane & 15) >> 2;
    int col = ((comp & 1) << 3) + (lane & 3) + ((lane >> 4) << 2);
    if (col >= cols_left) col = cols_left - 1;
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        int k = p << 1;
        unsigned packed = 0;
        if (k < k_left) ((F*)&packed)[0] = F(trans ? g[col * ldm + k] : g[k * ldm + col]);
        if (k + 1 < k_left) ((F*)&packed)[1] = F(trans ? g[col * ldm + k + 1] : g[(k + 1) * ldm + col]);
        r[p] = packed;
    }
}

template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma884_load_a_gmem(unsigned* r, const T* g, int ldm) {
    emmy_mma884_load_a_impl<T, F>(r, g, ldm, 16, 4);
}
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma884_load_a_gmem_mclamp(unsigned* r, const T* g, int ldm, int left) {
    emmy_mma884_load_a_impl<T, F>(r, g, ldm, left, 4);
}
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma884_load_a_gmem_kzero(unsigned* r, const T* g, int ldm, int k_left) {
    emmy_mma884_load_a_impl<T, F>(r, g, ldm, 16, k_left);
}
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma884_load_a_gmem_mclamp_kzero(
    unsigned* r, const T* g, int ldm, int left, int k_left) {
    emmy_mma884_load_a_impl<T, F>(r, g, ldm, left, k_left);
}

template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma884_load_b_gmem(unsigned* r, const T* g, int ldm) {
    emmy_mma884_load_b_impl<T, F>(r, g, ldm, 16, 4, false);
}
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma884_load_b_gmem_nclamp(unsigned* r, const T* g, int ldm, int left) {
    emmy_mma884_load_b_impl<T, F>(r, g, ldm, left, 4, false);
}
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma884_load_b_gmem_kzero(unsigned* r, const T* g, int ldm, int k_left) {
    emmy_mma884_load_b_impl<T, F>(r, g, ldm, 16, k_left, false);
}
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma884_load_b_gmem_nclamp_kzero(
    unsigned* r, const T* g, int ldm, int left, int k_left) {
    emmy_mma884_load_b_impl<T, F>(r, g, ldm, left, k_left, false);
}

template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma884_load_b_gmem_trans(unsigned* r, const T* g, int ldm) {
    emmy_mma884_load_b_impl<T, F>(r, g, ldm, 16, 4, true);
}
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma884_load_b_gmem_trans_nclamp(unsigned* r, const T* g, int ldm, int left) {
    emmy_mma884_load_b_impl<T, F>(r, g, ldm, left, 4, true);
}
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma884_load_b_gmem_trans_kzero(unsigned* r, const T* g, int ldm, int k_left) {
    emmy_mma884_load_b_impl<T, F>(r, g, ldm, 16, k_left, true);
}
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma884_load_b_gmem_trans_nclamp_kzero(
    unsigned* r, const T* g, int ldm, int left, int k_left) {
    emmy_mma884_load_b_impl<T, F>(r, g, ldm, left, k_left, true);
}

static __device__ __forceinline__ void emmy_mma_m8n8k4_f16_f32(
    float* d, const unsigned* a, const unsigned* b, const float* c) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%12, %13, %14, %15, %16, %17, %18, %19};\\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
                   "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
                   "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
}
"""


_MMA_SYNC_PRELUDE = """\
static __device__ __forceinline__ void emmy_ldmatrix_x4(unsigned* r, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

static __device__ __forceinline__ void emmy_ldmatrix_x2_trans(unsigned* r, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

// x4.trans: two col-adjacent canonical-B fragments in one ldmatrix — lanes 0-15
// address the 16 K rows at the first fragment's column, lanes 16-31 at col+8
// (the 096_pair_ldmatrix_loads fusion; r[0..1] / r[2..3] are the two fragments).
static __device__ __forceinline__ void emmy_ldmatrix_x4_trans(unsigned* r, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// Paired x4 → two B fragments in one ldmatrix (096_pair_ldmatrix_loads): the four
// loaded registers land DIRECTLY as b0[0..1] (lanes 0-15) and b1[0..1] (lanes 16-31)
// — no staging temp / register shuffle. ``_pair`` is the plain x4 (transposed-B slab,
// N-adjacent pair); ``_trans_pair`` is x4.trans (canonical-B slab, col-adjacent pair).
static __device__ __forceinline__ void emmy_ldmatrix_x4_pair(unsigned* b0, unsigned* b1, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\\n"
                 : "=r"(b0[0]), "=r"(b0[1]), "=r"(b1[0]), "=r"(b1[1]) : "r"(addr));
}

static __device__ __forceinline__ void emmy_ldmatrix_x4_trans_pair(unsigned* b0, unsigned* b1, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\\n"
                 : "=r"(b0[0]), "=r"(b0[1]), "=r"(b1[0]), "=r"(b1[1]) : "r"(addr));
}

// Plain (no .trans) x2: a transposed-B operand staged as its native N-major
// slab (Q@K^T's K rows) — each 8x8 matrix's rows ARE the mma B fragment's
// col-major columns, so no transpose is needed (cf. emmy_mma_load_b_gmem_trans).
static __device__ __forceinline__ void emmy_ldmatrix_x2(unsigned* r, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

// C->A register repack: the m16n8k16 f32 C fragment is lane-map ALIGNED with the
// 16-bit A fragment's k-halves, so two k-adjacent C fragments (c0 low, c1 high)
// convert into one A operand fragment per lane — no shuffle, no smem round-trip.
// cvt.rn packs (hi, lo) with round-to-nearest-even — the same rounding the
// retired smem handoff's RegStore vec2 pack applied, so the repack is bit-identical.
static __device__ __forceinline__ void emmy_c_to_a_f16(unsigned* a, const float* c0, const float* c1) {
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\\n" : "=r"(a[0]) : "f"(c0[1]), "f"(c0[0]));
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\\n" : "=r"(a[1]) : "f"(c0[3]), "f"(c0[2]));
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\\n" : "=r"(a[2]) : "f"(c1[1]), "f"(c1[0]));
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\\n" : "=r"(a[3]) : "f"(c1[3]), "f"(c1[2]));
}

static __device__ __forceinline__ void emmy_c_to_a_bf16(unsigned* a, const float* c0, const float* c1) {
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\\n" : "=r"(a[0]) : "f"(c0[1]), "f"(c0[0]));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\\n" : "=r"(a[1]) : "f"(c0[3]), "f"(c0[2]));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\\n" : "=r"(a[2]) : "f"(c1[1]), "f"(c1[0]));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\\n" : "=r"(a[3]) : "f"(c1[3]), "f"(c1[2]));
}

// gmem-direct fragment loads — the fallback when an mma.sync operand was NOT
// staged into shared memory (ldmatrix is smem-only, so we read the fragment
// straight from gmem instead, replicating the PTX m16n8k16 lane→element map).
// Slower than ldmatrix (no smem reuse) but correct; ``005_lower_atom_tile``
// emits these for an unstaged operand. ``ldm`` is the operand's gmem row stride
// (K for the row-major A[M,K]; N for the row-major B[K,N]); ``g`` points at the
// atom cell's base element, each lane adds its own (row,col) within the tile.
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem(unsigned* r, const T* g, int ldm) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);          // M: groupID, +8 for the second row block
        int col = (tig << 1) + ((i & 2) ? 8 : 0);   // K: 2*threadID_in_group, +8 for the k16 half
        const T* p = g + row * ldm + col;
        unsigned packed;
        ((F*)&packed)[0] = F(p[0]);                     // .f16x2: low half = col, high half = col+1
        ((F*)&packed)[1] = F(p[1]);
        r[i] = packed;
    }
}

template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem(unsigned* r, const T* g, int ldm) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;                                 // N: groupID (0..7)
        int k = (tig << 1) + (i ? 8 : 0);            // K: 2*threadID_in_group, +8 for the k16 half
        unsigned packed;
        ((F*)&packed)[0] = F(g[k * ldm + n]);           // .f16x2: low half = k, high half = k+1
        ((F*)&packed)[1] = F(g[(k + 1) * ldm + n]);
        r[i] = packed;
    }
}

// Masked-tile (M9) variants of the gmem-direct fragment loads: a tile
// straddling a masked axis's bound would read rows / cols past the
// runtime-sized buffer, so the lane coordinate on the gated axis clamps to
// ``left - 1`` (``left`` = in-range elements from the tile base; >= 1 because
// the boundary Cond admitted the tile). Clamped lanes read a duplicate
// in-bounds value — harmless, their stores are masked by the RegStore guard.
// Same contract as the staged path's slab-fill clamp (_clamp_source_index).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem_mclamp(unsigned* r, const T* g, int ldm, int rows_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);
        if (row >= rows_left) row = rows_left - 1;   // M: clamp to the runtime extent
        int col = (tig << 1) + ((i & 2) ? 8 : 0);
        const T* p = g + row * ldm + col;
        unsigned packed;
        ((F*)&packed)[0] = F(p[0]);
        ((F*)&packed)[1] = F(p[1]);
        r[i] = packed;
    }
}

template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_nclamp(unsigned* r, const T* g, int ldm, int cols_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = cols_left - 1;       // N: clamp to the runtime extent
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed;
        ((F*)&packed)[0] = F(g[k * ldm + n]);
        ((F*)&packed)[1] = F(g[(k + 1) * ldm + n]);
        r[i] = packed;
    }
}

// Transposed-B (Q @ K^T): B stored N×K (``g[n][k]``, K contiguous) — the native
// ``mma.row.col`` col-major B, so no ldmatrix ``.trans`` is needed. The fragment
// lane→element map is the same (n = groupID, k = 2·threadID_in_group + k16 half),
// but each lane now reads a contiguous (k, k+1) pair from row ``n`` of B. ``ldm``
// is B's gmem row stride (the K extent). Mirrors ``emmy_mma_load_b_gmem`` with the
// (k, n) index roles swapped to (n, k).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans(unsigned* r, const T* g, int ldm) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;                                 // N: groupID (0..7)
        int k = (tig << 1) + (i ? 8 : 0);            // K: 2*threadID_in_group, +8 for the k16 half
        unsigned packed;
        ((F*)&packed)[0] = F(g[n * ldm + k]);           // .f16x2: contiguous (k, k+1) in row n
        ((F*)&packed)[1] = F(g[n * ldm + k + 1]);
        r[i] = packed;
    }
}

// Masked-tile variant of the transposed-B gmem load: the gated N axis is the
// row (``n``) here, so clamp ``n`` to the runtime extent (cf. _b_gmem_nclamp,
// which clamps N as the column). Clamped lanes read a duplicate in-bounds row;
// their stores are masked by the RegStore guard.
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans_nclamp(unsigned* r, const T* g, int ldm, int cols_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = cols_left - 1;       // N: clamp to the runtime extent
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed;
        ((F*)&packed)[0] = F(g[n * ldm + k]);
        ((F*)&packed)[1] = F(g[n * ldm + k + 1]);
        r[i] = packed;
    }
}

// Masked-K (symbolic reduce) variants of the gmem-direct fragment loads: when the
// operand's REDUCE (K) axis is mask-padded (SDPA P@V over ``seq_len``), the final
// K tile straddles the runtime extent. Unlike the M/N clamp above, a K element
// past the extent must be ZERO-FILLED, not clamped to a duplicate — K is summed
// by the mma, so a duplicate corrupts the reduction. ``k_left`` = in-range K
// elements from the tile base; a half past it reads as +0.0 and is never
// dereferenced. Mirrors the staged path's slab zero-fill (``_stage_expand``).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem_kzero(unsigned* r, const T* g, int ldm, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);
        int col = (tig << 1) + ((i & 2) ? 8 : 0);
        const T* p = g + row * ldm + col;
        unsigned packed = 0;
        if (col < k_left) ((F*)&packed)[0] = F(p[0]);
        if (col + 1 < k_left) ((F*)&packed)[1] = F(p[1]);
        r[i] = packed;
    }
}

// A: masked-M (clamp rows) AND masked-K (zero-fill cols).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem_mclamp_kzero(unsigned* r, const T* g, int ldm, int rows_left, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);
        if (row >= rows_left) row = rows_left - 1;
        int col = (tig << 1) + ((i & 2) ? 8 : 0);
        const T* p = g + row * ldm + col;
        unsigned packed = 0;
        if (col < k_left) ((F*)&packed)[0] = F(p[0]);
        if (col + 1 < k_left) ((F*)&packed)[1] = F(p[1]);
        r[i] = packed;
    }
}

// B (row-major K×N, NOT transposed): K is the row, zero-fill past the extent.
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_kzero(unsigned* r, const T* g, int ldm, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed = 0;
        if (k < k_left) ((F*)&packed)[0] = F(g[k * ldm + n]);
        if (k + 1 < k_left) ((F*)&packed)[1] = F(g[(k + 1) * ldm + n]);
        r[i] = packed;
    }
}

// B: masked-N (clamp col) AND masked-K (zero-fill row).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_nclamp_kzero(unsigned* r, const T* g, int ldm, int cols_left, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = cols_left - 1;
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed = 0;
        if (k < k_left) ((F*)&packed)[0] = F(g[k * ldm + n]);
        if (k + 1 < k_left) ((F*)&packed)[1] = F(g[(k + 1) * ldm + n]);
        r[i] = packed;
    }
}

// Transposed-B (Q@K^T, ``g[n][k]`` with K contiguous) masked-K: zero-fill the K
// halves past the runtime extent. K is summed by the mma, so a past-extent element
// must read as +0.0 (never a duplicate). Mirrors ``emmy_mma_load_b_gmem_kzero`` with
// the (k, n) index roles swapped to (n, k) — cf. ``emmy_mma_load_b_gmem_trans``.
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans_kzero(unsigned* r, const T* g, int ldm, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;                                 // N: groupID (0..7)
        int k = (tig << 1) + (i ? 8 : 0);            // K: 2*threadID_in_group, +8 for the k16 half
        unsigned packed = 0;
        if (k < k_left) ((F*)&packed)[0] = F(g[n * ldm + k]);       // .f16x2: contiguous (k, k+1) in row n
        if (k + 1 < k_left) ((F*)&packed)[1] = F(g[n * ldm + k + 1]);
        r[i] = packed;
    }
}

// Transposed-B: masked-N (clamp the ``n`` row) AND masked-K (zero-fill the contiguous k).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans_nclamp_kzero(
    unsigned* r, const T* g, int ldm, int cols_left, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = cols_left - 1;       // N: clamp to the runtime extent
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed = 0;
        if (k < k_left) ((F*)&packed)[0] = F(g[n * ldm + k]);
        if (k + 1 < k_left) ((F*)&packed)[1] = F(g[n * ldm + k + 1]);
        r[i] = packed;
    }
}

static __device__ __forceinline__ void emmy_mma_m16n8k16_f16_f32(float* d, const unsigned* a, const unsigned* b, const float* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

static __device__ __forceinline__ void emmy_mma_m16n8k16_bf16_f32(float* d, const unsigned* a, const unsigned* b, const float* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// f16-accumulate HMMA (the ``mma_m16n8k16_f16_f16`` atom): on the consumer GeForce dies the .f32-accumulate
// form above runs at HALF the tensor-core rate, so this variant keeps the whole mma chain on the
// full-rate .f16 accumulator — ``c``/``d`` are 2 packed b32 regs (two halfs each, the same
// element map as the four f32 regs, pair-packed). Paired with emmy_mma_promote_f16acc below,
// which periodically folds the f16 partials into the f32 shadow accumulator.
static __device__ __forceinline__ void emmy_mma_m16n8k16_f16_f16(unsigned* d, const unsigned* a, const unsigned* b, const unsigned* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\\n"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "r"(c[0]), "r"(c[1]));
}

// The chunk promote of the f16-accumulate scheme: fold the packed f16 C fragment into the f32
// shadow accumulator and rezero it, bounding the f16 accumulation error to one K chunk. Inline
// PTX cvt (not __half2 intrinsics) keeps this prelude header-free, like the wrappers above.
static __device__ __forceinline__ void emmy_mma_promote_f16acc(float* c, unsigned* h) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        float lo, hi;
        asm("{.reg .b16 lo, hi;\\n\\t"
            "mov.b32 {lo, hi}, %2;\\n\\t"
            "cvt.f32.f16 %0, lo;\\n\\t"
            "cvt.f32.f16 %1, hi;}\\n"
            : "=f"(lo), "=f"(hi) : "r"(h[i]));
        c[2 * i] += lo;
        c[2 * i + 1] += hi;
        h[i] = 0u;
    }
}

"""

# Native fp8 mma prelude (the ``m16n8k32`` atom family, M3 of the FP8 plan) — appended to the mma
# prelude ONLY when the kernel carries an fp8 ``MmaSyncPtx``, so every existing 16-bit mma kernel's
# source stays byte-identical. Two parts:
#
# - ``emmy_mma_m16n8k32_{e4m3,e5m2}_f32``: the bare PTX spelling, probed 2026-08-06 — it compiles
#   AND verifies against the LUT-decode reference on BOTH sm_89 (CUDA 13.3) and sm_120 (CUDA 13.0);
#   ptxas refuses the Blackwell ``.kind::f8f6f4`` qualifier on sm_89 / sm_120 / sm_120a, so the bare
#   form is the one spelling for every supported arch. Effective accumulation precision is
#   arch-dependent: true-f32 on sm_120 (~7e-7 rel), reduced on sm_89 (~3e-4 rel measured).
# - the ``_b8`` gmem-direct fragment loaders: the k32 8-bit fragment map packs FOUR bytes per
#   32-bit register per lane (A: 4 regs — reg i covers row ``grp + ((i&1)?8:0)``, cols
#   ``4·tig + ((i&2)?16:0) + 0..3``; B: 2 regs — reg i covers k ``4·tig + (i?16:0) + 0..3`` at col
#   ``grp``; C/D keep the k16 f32 map), so the 16-bit half-pair loaders above cannot express it.
#   These gather raw bytes — no per-element convert: the mma consumes the storage bits directly.
#   ``ldmatrix`` has no byte-fragment form on these arches, so the fp8 atoms are gmem-direct only
#   (the stage resolvers decline them) and there is no masked-K (``kzero``) variant — the schedule
#   offers the atoms only on a static, step-divisible K.
_MMA_F8_PRELUDE = """\
template <typename T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem_b8(unsigned* r, const T* g, int ldm) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);          // M: groupID, +8 for the second row block
        int col = (tig << 2) + ((i & 2) ? 16 : 0);  // K: 4*threadID_in_group, +16 for the k32 half
        unsigned packed;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ((unsigned char*)&packed)[j] = p[row * ldm + col + j];
        r[i] = packed;
    }
}

template <typename T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_b8(unsigned* r, const T* g, int ldm) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;                                 // N: groupID (0..7)
        int k = (tig << 2) + (i ? 16 : 0);           // K: 4*threadID_in_group, +16 for the k32 half
        unsigned packed;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ((unsigned char*)&packed)[j] = p[(k + j) * ldm + n];
        r[i] = packed;
    }
}

// Transposed-B (``g[n][k]``, K contiguous): same fragment map with the (k, n) index roles swapped —
// each lane reads a contiguous 4-byte K run from row ``n``. Cf. emmy_mma_load_b_gmem_trans.
template <typename T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans_b8(unsigned* r, const T* g, int ldm) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        int k = (tig << 2) + (i ? 16 : 0);
        unsigned packed;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ((unsigned char*)&packed)[j] = p[n * ldm + k + j];
        r[i] = packed;
    }
}

// Masked-tile variants: clamp the lane coordinate on the gated output axis to the runtime extent
// (duplicate reads; the RegStore guard masks their stores) — the b8 mirror of the 16-bit
// ``_mclamp`` / ``_nclamp`` family above.
template <typename T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem_mclamp_b8(unsigned* r, const T* g, int ldm, int rows_left) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);
        if (row >= rows_left) row = rows_left - 1;   // M: clamp to the runtime extent
        int col = (tig << 2) + ((i & 2) ? 16 : 0);
        unsigned packed;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ((unsigned char*)&packed)[j] = p[row * ldm + col + j];
        r[i] = packed;
    }
}

template <typename T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_nclamp_b8(unsigned* r, const T* g, int ldm, int cols_left) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = cols_left - 1;       // N: clamp to the runtime extent
        int k = (tig << 2) + (i ? 16 : 0);
        unsigned packed;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ((unsigned char*)&packed)[j] = p[(k + j) * ldm + n];
        r[i] = packed;
    }
}

template <typename T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans_nclamp_b8(unsigned* r, const T* g, int ldm, int cols_left) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = cols_left - 1;       // N: clamp to the runtime extent
        int k = (tig << 2) + (i ? 16 : 0);
        unsigned packed;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ((unsigned char*)&packed)[j] = p[n * ldm + k + j];
        r[i] = packed;
    }
}

static __device__ __forceinline__ void emmy_mma_m16n8k32_e4m3_f32(float* d, const unsigned* a, const unsigned* b, const float* c) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

static __device__ __forceinline__ void emmy_mma_m16n8k32_e5m2_f32(float* d, const unsigned* a, const unsigned* b, const float* c) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

"""

# Staged fp8 slab drains — appended ONLY when the kernel carries a byte-slab ``LdmatrixLoad``
# (a staged 1-byte operand), so every other kernel's source stays byte-identical. ldmatrix is
# b16-only below sm_100a, so a staged fp8 slab drains through a cooperative per-lane gather with
# the same lane→element maps as the gmem fragment loaders — and, where the slab keeps K
# CONTIGUOUS per lane (the transposed-B N-major slab; every k32 A slab), the per-byte gathers
# vectorize:
#
# - ``emmy_mma_load_b_smem_trans_f8_f16`` (W8A16, k16): each lane's (k, k+1) byte pair loads as
#   ONE fp8x2 and converts with ONE hardware ``cvt.rn.f16x2.e4m3x2`` (the <cuda_fp8.h>
#   ``operator __half2()``, low byte → low half — exactly the fragment's pair order). Same
#   per-element convert as the gmem-direct path, so staged stays bit-identical to it.
# - ``emmy_mma_load_{a,b_trans}_smem_b8v`` (W8A8, k32): each lane's 4-byte K run loads as ONE
#   u32 (little-endian — byte j lands in fragment byte j, matching the ``_b8`` gathers).
#
# Alignment holds by construction: k offsets are 2-/4-element aligned, the padded byte-slab row
# stride (``BYTE_SLAB_PAD``) is 16-divisible, and the slab base is 16 B-aligned (the byte-slab
# ``Smem.align``). The K-STRIDED byte reads (a canonical K-major B slab) have no vector form and
# reuse the gmem loader templates pointed at the slab instead.
_F8_STAGED_PRELUDE = """\
template <typename T, typename T2>
static __device__ __forceinline__ void emmy_mma_load_b_smem_trans_f8_f16(unsigned* r, const T* g, int ldm) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int k = (tig << 1) + (i ? 8 : 0);            // K: 2*threadID_in_group, +8 for the k16 half
        __half2 h2 = __half2(*reinterpret_cast<const T2*>(g + grp * ldm + k));
        r[i] = *reinterpret_cast<unsigned*>(&h2);
    }
}

template <typename T>
static __device__ __forceinline__ void emmy_mma_load_a_smem_b8v(unsigned* r, const T* g, int ldm) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);          // M: groupID, +8 for the second row block
        int col = (tig << 2) + ((i & 2) ? 16 : 0);  // K: 4*threadID_in_group, +16 for the k32 half
        r[i] = *reinterpret_cast<const unsigned*>(p + row * ldm + col);
    }
}

template <typename T>
static __device__ __forceinline__ void emmy_mma_load_b_smem_trans_b8v(unsigned* r, const T* g, int ldm) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int k = (tig << 2) + (i ? 16 : 0);           // K: 4*threadID_in_group, +16 for the k32 half
        r[i] = *reinterpret_cast<const unsigned*>(p + grp * ldm + k);
    }
}

"""


def _swizzle_prelude(kernel_op: KernelOp) -> str:
    """One ``emmy_swizzle_<mode>`` helper per swizzle mode the body uses — on ``LdmatrixLoad``
    drains (undoing the TMA hardware in-copy permutation, or reading back a software-swizzled
    slab), on ``CpAsyncCopy`` fills (the software swizzle's producer side), and on swizzled
    slab ``Write``\\ s (the sync compute-fill's producer side). Built from
    ``LDMATRIX_SWIZZLE_XOR`` (the single source of the shift/mask), so the call sites spell their
    (often long) element index once instead of inlining it twice around the XOR.
    ``__forceinline__``; same SASS as the inlined form."""
    modes = sorted(
        {
            s.swizzle
            for s in kernel_op.body.iter()
            if isinstance(s, (LdmatrixLoad, CpAsyncCopy, Write)) and s.swizzle in LDMATRIX_SWIZZLE_XOR
        }
    )
    chunks = []
    for mode in modes:
        shift, mask = LDMATRIX_SWIZZLE_XOR[mode]
        chunks.append(
            f"// {mode} slab swizzle: XOR a b16 smem element index the way the fill permuted the\n"
            f"// 16-byte chunks (TMA in-copy, or the same XOR on a cp.async fill destination);\n"
            f"// the ldmatrix drain must read them back through the identical permutation.\n"
            f"static __device__ __forceinline__ int emmy_swizzle_{mode.lower()}(int e) {{\n"
            f"    return e ^ (((e >> {shift}) & {mask}) << 3);\n"
            f"}}\n\n"
        )
    return "".join(chunks)


_INTRINSIC_TO_CUDA: dict[str, str] = {
    "exp": "expf",
    "exp_fast": "__expf",
    "rsqrt": "rsqrtf",
    "sin": "sinf",
    "cos": "cosf",
    "tanh": "tanhf",
    "fabs": "fabsf",
    "fmax": "fmaxf",
    "fmin": "fminf",
    "pow": "powf",
    "sqrt": "sqrtf",
    "erf": "erff",
}

_BUILTIN_TO_CUDA: dict[str, str] = {
    "thread_idx.x": "threadIdx.x",
    "thread_idx.y": "threadIdx.y",
    "thread_idx.z": "threadIdx.z",
    "block_idx.x": "blockIdx.x",
    "block_idx.y": "blockIdx.y",
    "block_idx.z": "blockIdx.z",
    "block_dim.x": "blockDim.x",
    "block_dim.y": "blockDim.y",
    "block_dim.z": "blockDim.z",
    "grid_dim.x": "gridDim.x",
    "grid_dim.y": "gridDim.y",
    "grid_dim.z": "gridDim.z",
    "warp_size": "warpSize",
}

# Block size for the linear thread-flattening path (``Tile.block_axes``
# empty); the host-side launcher rounds the grid up to cover all threads.
_BLOCK_SIZE = 256


def render_kernelop(
    kernel_op: KernelOp,
    tensors: dict[str, Tensor] | None = None,
    shapes: dict[str, tuple[int, ...]] | None = None,
    literal_constants: dict[str, float] | None = None,
    runtime_args: tuple[str, ...] = (),
    indirect_inputs: tuple[str, ...] = (),
) -> str:
    """Render a complete ``extern "C" __global__`` CUDA function for a ``KernelOp``.

    ``tensors`` maps each global-buffer name (anything appearing on a
    ``Load.input`` or ``Write.output``) to a :class:`Tensor` describing
    its shape + dtype. The renderer uses the shape to row-major-flatten
    multi-dim indices and the dtype for kernel-signature param types.
    Production callers typically build this from the surrounding graph
    (``{nid: graph.nodes[nid].output for nid in ...}``); tests pass it
    as a literal dict.

    ``shapes`` is the legacy form (shape-only, no dtype) — kept for
    back-compat with tests that pre-date the dtype migration; dtypes
    default to F32 in that path. New callers should prefer ``tensors``.

    ``literal_constants`` maps input-buffer names to scalar values that
    should be embedded in the kernel body as ``float`` literals instead
    of passed as kernel parameters. Loads of those bufs render as
    ``float name = <value>;`` (see ``Load.render``) and the buf is
    excluded from the kernel signature.

    ``indirect_inputs`` names input buffers whose base pointer is an
    **indirect operand**: instead of ``const T* <n>`` the signature takes
    ``const T* const* <n>__table, const int* <n>__sel, int <n>__slot``
    and the body prepends ``const T* <n> = <n>__table[<n>__sel[<n>__slot]];``
    — every downstream load is unchanged (the serving MoE fixed-slot
    dispatch: the kernel fetches its weight base from a device pointer
    table indexed by the router's indices tensor). Names absent from
    ``kernel_op.inputs`` are ignored. Empty (the default) renders exactly
    the historical signature — non-indirect kernel sources stay
    byte-identical.

    Kernel signature is derived from the body: ``kernel_op.inputs``
    (distinct ``Load.input`` names) become input params,
    ``kernel_op.outputs`` (distinct ``Write.output`` names) become
    writeable output params, ordered by first appearance. Parameter
    types come from the per-buffer :class:`Tensor.dtype` (passed in via
    ``tensors=`` / ``shapes=`` — the caller supplies them); literal-constant
    inputs are skipped. Unknown buffers fall back to ``F32``.
    """
    literals = dict(literal_constants or {})
    tmap: dict[str, Tensor] = dict(tensors) if tensors else {}
    if shapes:
        for n, s in shapes.items():
            tmap.setdefault(n, Tensor(n, tuple(s)))
    smem_offsets, smem_total = _compute_dynamic_smem_offsets(kernel_op)
    ctx = RenderCtx(
        target=CudaRenderTarget(),
        shapes={n: tuple(t.shape) for n, t in tmap.items()},
        indent=1,
        intrinsics=_INTRINSIC_TO_CUDA,
        builtins=_BUILTIN_TO_CUDA,
        literal_constants=literals,
        smem_dynamic_offsets=smem_offsets,
        buffer_dtypes={n: t.dtype.name for n, t in tmap.items()},
    )

    def _dtype_for(name: str) -> object:
        return tmap[name].dtype if name in tmap else F32

    indirect = tuple(n for n in kernel_op.inputs if n in indirect_inputs and n not in literals)
    sig_parts = [
        f"const {cuda_name(_dtype_for(n))}* const* {n}__table, const int* {n}__sel, int {n}__slot"
        if n in indirect
        else f"const {cuda_name(_dtype_for(n))}* {n}"
        for n in kernel_op.inputs
        if n not in literals
    ]
    sig_parts.extend(f"{cuda_name(_dtype_for(n))}* {n}" for n in kernel_op.outputs)
    # TMA descriptors are passed as ``__grid_constant__`` value parameters.
    # The kernel only takes their address (``&desc``) for inline asm, so
    # the opaque ``CUtensorMap`` forward decl above suffices.
    desc_names = tuple(dict.fromkeys(s.name for s in kernel_op.body.iter_of_type(TmaDescriptor)))
    # Descriptors are passed by pointer (placed in global memory by the
    # host) rather than ``__grid_constant__`` value parameters: cupy's
    # arg-packing path doesn't preserve the 64-byte alignment that
    # by-value ``CUtensorMap`` parameters require, so this avoids a
    # CUDA_ERROR_MISALIGNED_ADDRESS at launch.
    sig_parts.extend(f"const CUtensorMap* __restrict__ {n}" for n in desc_names)
    # Runtime int args for symbolic axis extents (Dim("seq_len") etc.) come
    # after buffers and TMA descriptors so the launcher can simply tail-append
    # the resolved int values to the kernel-arg pack.
    sig_parts.extend(f"int {n}" for n in runtime_args)
    params_text = ", ".join(sig_parts)
    bounds = _launch_bounds_for(kernel_op)
    launch_bounds = f"\n__launch_bounds__({bounds})"

    body_text = "\n".join(render_body(kernel_op.body, ctx))
    # The cross-thread combine (``WarpShuffle`` / ``TreeHalve``) references the hardware
    # warp ``lane`` / ``warp`` ids; declare them at function entry when the body needs them.
    from emmy.compiler.ir.stmt.blocks import _body_uses_lane_warp  # noqa: PLC0415

    if _body_uses_lane_warp(kernel_op.body):
        body_text = "    int lane = threadIdx.x & 31;\n    int warp = threadIdx.x >> 5;\n" + body_text
    if smem_offsets:
        # All Smem decls were rewritten to pointer aliases into a single
        # dynamic pool; declare the pool at function entry. The pool base must
        # be aligned to the strictest buffer it holds — ≥1024 B whenever a
        # swizzled TMA operand is present, else the per-buffer pad inside the
        # pool would still land on a base that doesn't zero the swizzle's
        # source address bits (16 B otherwise satisfies TMA + FP32/FP64).
        # ``extern __shared__`` arrays must be declared without an explicit
        # size — NVCC otherwise treats the decl as a *static* definition
        # (with the static cap), defeating the whole point of the switch.
        # The runtime size is supplied at launch via ``shared_mem=``.
        pool_align = max(16, *(max(_nbytes_of(s.dtype), int(s.align) if s.align else 0) for s in kernel_op.smem_buffers.values()))
        pool_decl = f"    extern __shared__ __align__({pool_align}) unsigned char _smem_pool[];  // {smem_total} bytes\n"
        body_text = pool_decl + body_text
    if indirect:
        # Indirect-operand preamble: resolve each marked input's base pointer from its device
        # table before any body statement runs; downstream loads use the plain name unchanged.
        body_text = "".join(f"    const {cuda_name(_dtype_for(n))}* {n} = {n}__table[{n}__sel[{n}__slot]];\n" for n in indirect) + body_text
    prelude = _TMA_PRELUDE if desc_names else ""
    sig_dtypes = [_dtype_for(n) for n in kernel_op.inputs if n not in literals]
    sig_dtypes.extend(_dtype_for(n) for n in kernel_op.outputs)
    includes = "".join(f"#include {h}\n" for h in cuda_includes(sig_dtypes))
    # The mma.sync (s16816) tensor-core path is pure inline PTX — its
    # ldmatrix / mma.sync wrappers are emitted in ``_MMA_SYNC_PRELUDE``, so
    # NVRTC needs no ``<mma.h>`` (the legacy ``nvcuda::wmma`` family is gone).
    from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait, MmaSyncPtx  # noqa: PLC0415

    mma_stmts = tuple(s for s in kernel_op.body.iter() if isinstance(s, MmaSyncPtx))
    uses_m8n8k4 = any(s.shape == (8, 8, 4) for s in mma_stmts)
    uses_modern_mma = any(s.shape != (8, 8, 4) for s in mma_stmts)
    mma_sync_prelude = (_MMA_M8N8K4_PRELUDE if uses_m8n8k4 else "") + (_MMA_SYNC_PRELUDE if uses_modern_mma else "")
    # The fp8 wrappers + byte-gather loaders join only when an fp8 mma is present, so every
    # 16-bit mma kernel's source stays byte-identical (the kernel-source digest gate). The
    # staged byte-slab drain helpers likewise join only when a byte-slab drain is present.
    if any(isinstance(s, MmaSyncPtx) and s.ab_dtype in ("e4m3", "e5m2") for s in kernel_op.body.iter()):
        mma_sync_prelude += _MMA_F8_PRELUDE
    if any(isinstance(s, LdmatrixLoad) and s.byte_slab for s in kernel_op.body.iter()):
        mma_sync_prelude += _F8_STAGED_PRELUDE
    uses_cp_async = any(isinstance(s, (CpAsyncCopy, CpAsyncCommit, CpAsyncWait)) for s in kernel_op.body.iter())
    cp_async_prelude = _CP_ASYNC_PRELUDE if uses_cp_async else ""
    # The trellis decode helper joins only when a ``TrellisLoad`` is present (kernel-source
    # digest safety), and its fp16 intrinsics need the fp16 header even when no kernel
    # parameter carries an f16 dtype.
    from emmy.compiler.ir.stmt import TrellisLoad  # noqa: PLC0415

    trellis_prelude = ""
    if any(isinstance(s, TrellisLoad) for s in kernel_op.body.iter()):
        trellis_prelude = _TRELLIS_PRELUDE
        includes = "".join(f"#include {h}\n" for h in cuda_includes([*sig_dtypes, "f16"]))
    preludes = f"{includes}{mma_sync_prelude}{cp_async_prelude}{trellis_prelude}{_swizzle_prelude(kernel_op)}{prelude}"
    header = f'{preludes}extern "C" __global__{launch_bounds} void {kernel_op.name}({params_text})'
    return f"{header} {{\n{body_text}\n}}\n"


def _compute_dynamic_smem_offsets(kernel_op: KernelOp) -> tuple[dict[str, int], int]:
    """Walk the body collecting ``Smem`` decls. If their summed footprint
    exceeds the static cap, return ``({name: byte_offset}, total_bytes)``
    so ``Smem.render`` can emit pool aliases. Otherwise return an empty
    map — the kernel keeps using ``__shared__ T arr[N]``.

    Offsets/total come from :func:`pack_smem` (each buffer aligned to the
    larger of its dtype size and explicit ``align`` field — TMA swizzle slabs
    request 256/512/1024 B), the same packer ``KernelOp.smem_bytes`` uses — so
    the static-vs-dynamic gate here and the launch-time pool size agree."""
    smems: list[Smem] = list(kernel_op.smem_buffers.values())
    if not smems:
        return {}, 0

    offsets, total = pack_smem(smems)
    if total <= STATIC_SMEM_CAP:
        return {}, 0
    return offsets, total


def _launch_bounds_for(kernel_op: KernelOp) -> int:
    """Derive ``__launch_bounds__`` from the (single) ``Tile``'s per-CTA thread count — a
    cooperative tile's ``block_threads`` (``coop · ∏block-cells``), else the scalar tier's
    ``_BLOCK_SIZE``. Matches the ``blockDim`` the cuda lowering picks."""
    from emmy.compiler.ir.kernel.ir import Tile  # noqa: PLC0415

    for s in kernel_op.body.iter_of_type(Tile):
        if s.block_threads is not None:
            return s.block_threads + s.aux_threads  # the warp-specialized producer band launches too
    return _BLOCK_SIZE


__all__ = ["render_kernelop"]
