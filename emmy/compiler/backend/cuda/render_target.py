"""CUDA implementation of :class:`RenderTarget`.

Owns every CUDA C-spelling decision the Kernel-IR renderer makes:
type names (``float`` / ``__half`` / fixed-width integer carriers), conversion intrinsics
(``__half2float`` / ``__float2half``), per-dtype op intrinsics
(``expf`` / ``hexp``, ``fmaxf`` / ``__hmax``, ...), and the set of
ops with native fp16 forms.

The Stmt renderer in :mod:`emmy.compiler.ir.stmt` calls into a
:class:`CudaRenderTarget` instance attached to ``RenderCtx``; the
hardcoded ``__half2``/``__float2half`` strings that used to live next
to ``Load`` / ``Assign`` / ``Write`` are gone.
"""

from __future__ import annotations

_TYPE_NAME: dict[str, str] = {
    "f32": "float",
    "f16": "__half",
    "f16x2": "__half2",
    "f8e4m3": "__nv_fp8_e4m3",
    "f8e5m2": "__nv_fp8_e5m2",
    "i16": "short",
    "i32": "int",
    "i64": "long long",
    "u8": "unsigned char",
    "u16": "unsigned short",
    "u32": "unsigned int",
    "u64": "unsigned long long",
    "bool": "bool",
}

_INTEGER_DTYPES = frozenset({"i16", "i32", "i64", "u8", "u16", "u32", "u64"})
_INTEGER_NATIVE_OPS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "floor_divide",
        "remainder",
        "mod",
        "left_shift",
        "right_shift",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_count",
    }
)

# fp8 storage dtypes (M2/M3 of the FP8 plan). A kernel never computes ON fp8 —
# the only legal uses of an fp8 SSA value are the decode conversion out of it
# and the encode conversion into it, both spelled below in ``convert``.
_F8_DTYPES = ("f8e4m3", "f8e5m2")

# Intrinsic spellings — per dtype.  Keys are abstract op names emitted
# by ``op_to_expr`` (``"exp"``, ``"fmax"``, ``"fabs"``, ...).
_INTRINSIC_F32: dict[str, str] = {
    "exp": "expf",
    "exp_fast": "__expf",
    "log": "logf",
    "log_fast": "__logf",
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

_INTRINSIC_F16: dict[str, str] = {
    "exp": "hexp",
    "exp_fast": "hexp",
    "log": "hlog",
    "sqrt": "hsqrt",
    "rsqrt": "hrsqrt",
    "tanh": "htanh",
    "fmax": "__hmax",
    "fmin": "__hmin",
    "fabs": "__habs",
}

# Per-pair (2-wide vector) fp16 intrinsics. Each entry is the cuda_fp16.h
# ``h2`` / ``__h*2`` form. Used when an Init / Accum / Assign carries
# dtype = F16x2 (paired by the ``070_pack_fp16_pairs`` pass).
_INTRINSIC_F16x2: dict[str, str] = {
    "exp": "h2exp",
    "log": "h2log",
    "sqrt": "h2sqrt",
    "rsqrt": "h2rsqrt",
    "tanh": "h2tanh",
    "fmax": "__hmax2",
    "fmin": "__hmin2",
    "fabs": "__habs2",
}

# Abstract op names with a native fp16 form. Binary operators (+, -, *, /)
# work via cuda_fp16.h's operator overloads on ``__half`` and ``__half2``,
# so they don't need an entry in the intrinsic tables but are still
# "native". The same set covers F16 and F16x2 — every op listed has both
# a per-element (``__hmax``, ``hexp``) and per-pair (``__hmax2``,
# ``h2exp``) form.
_NATIVE_FP16_OPS: frozenset[str] = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "maximum",
        "minimum",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "tanh",
        "fabs",
        "abs",
        "negative",
        "copy",
        "reciprocal",
        "relu",
        "sigmoid",
    }
)


class CudaRenderTarget:
    """CUDA C / cuda_fp16.h spellings for :class:`RenderTarget`.

    The Kernel-IR renderer constructs one per ``render_kernelop`` invocation
    (see ``emmy/compiler/ir/kernel/render.py``). ``compute_capability`` selects
    architecture-specific scalar spellings; the default keeps the portable
    CUDA-header forms used by standalone statement-render tests.
    """

    def __init__(self, compute_capability: tuple[int, int] | None = None) -> None:
        self.compute_capability = compute_capability
        self.uses_sm70_f8_decode = False

    @property
    def prelude(self) -> str:
        return _SM70_F8_DECODE_PRELUDE if self.uses_sm70_f8_decode else ""

    def type_name(self, dtype: str) -> str:
        return _TYPE_NAME.get(dtype, "float")

    def literal(self, text: str, dtype: str) -> str:
        # Numeric literals (``0.0f`` / ``1.0f`` / ``-1e+30f``) wrap in
        # ``__float2half`` (scalar) or ``__float2half2_rn`` (pair-
        # broadcast) when the surrounding expression's result dtype is
        # fp16 / fp16x2 so the call composes with ``__half`` / ``__half2``
        # operands. NVRTC folds the call to a constant at compile time.
        if dtype == "f16":
            return f"__float2half({text})"
        if dtype == "f16x2":
            return f"__float2half2_rn({text})"
        return text

    def convert(self, value: str, src_dt: str, dst_dt: str) -> str:
        if src_dt == dst_dt:
            return value
        if src_dt in _F8_DTYPES and dst_dt in ("f32", "f16"):
            if src_dt == "f8e4m3" and dst_dt == "f16" and self.compute_capability == (7, 0):
                # Volta has no FP8 conversion instruction. cuda_fp8.h's scalar
                # fallback widens through f32; decode the finite-and-NaN E4M3
                # bits directly into their exact f16 value instead.
                self.uses_sm70_f8_decode = True
                return f"emmy_sm70_f8e4m3_to_f16({value})"
            # fp8 decode — the ``from_f8*`` cast's device spelling. The functional
            # cast invokes ``<cuda_fp8.h>``'s explicit conversion operator, which
            # compiles on every arch the header supports (hardware ``cvt`` on
            # sm_89+, C++ emulation below) — no sm gate needed.
            return f"{_TYPE_NAME[dst_dt]}({value})"
        if dst_dt in _F8_DTYPES and src_dt in ("f32", "f16"):
            # fp8 encode — the ``to_f8*`` cast's device spelling: the <cuda_fp8.h>
            # explicit constructor (round-to-nearest-even, saturate-to-finite), the
            # decode's twin. Same no-sm-gate story as above.
            return f"{_TYPE_NAME[dst_dt]}({value})"
        if dst_dt == "f16" and src_dt == "f32":
            return f"__float2half({value})"
        if dst_dt == "f32" and src_dt == "f16":
            return f"__half2float({value})"
        if dst_dt == "f16x2" and src_dt == "f16":
            # Broadcast scalar __half into both lanes of a __half2.
            return f"__half2half2({value})"
        if dst_dt == "f16x2" and src_dt == "f32":
            return f"__float2half2_rn({value})"
        return value

    def bitcast(self, value: str, src_dt: str, dst_dt: str) -> str:
        if src_dt == dst_dt:
            return value
        if dst_dt not in _TYPE_NAME:
            raise ValueError(f"CUDA scalar bitcast does not support destination dtype {dst_dt!r}")
        return f"emmy_bitcast<{_TYPE_NAME[dst_dt]}>({value})"

    def intrinsic(self, op_name: str, result_dt: str) -> str:
        if op_name == "bitwise_count":
            return "__popcll" if result_dt in ("i64", "u64") else "__popc"
        if result_dt == "f16":
            return _INTRINSIC_F16.get(op_name, op_name)
        if result_dt == "f16x2":
            return _INTRINSIC_F16x2.get(op_name, op_name)
        return _INTRINSIC_F32.get(op_name, op_name)

    def has_native_op(self, op_name: str, dtype: str) -> bool:
        if dtype == "f32":
            # f32 has every op natively.
            return True
        if dtype in ("f16", "f16x2"):
            return op_name in _NATIVE_FP16_OPS
        if dtype in _INTEGER_DTYPES:
            return op_name in _INTEGER_NATIVE_OPS
        return False

    def vector_type(self, dtype: str, n: int) -> tuple[str, str] | None:
        if dtype == "f32":
            if n in (2, 4):
                return (f"float{n}", "float")
            return None
        if dtype == "f16":
            # n=2 → 4 B (LDS.32) via ``__half2`` (4-byte alignment).
            # n=4 → 8 B (LDS.64) via ``uint2`` punned to 4 ``__half``.
            # n=8 → 16 B (LDS.128) via ``uint4`` punned to 8 ``__half``.
            # The 009b cooperative-reduce permutation guarantees the
            # base address is 16-byte aligned when n=8 (and 8-byte
            # aligned for n=4); for matmul / other shapes the
            # vectorize pass independently checks the affine form.
            if n == 2:
                return ("__half2", "__half")
            if n == 4:
                return ("uint2", "__half")
            if n == 8:
                return ("uint4", "__half")
            return None
        if dtype in (*_F8_DTYPES, "u8"):
            # Byte-valued storage uses ordinary CUDA integer vector carriers;
            # array-style unpacking restores the source scalar type lane by lane.
            if n == 4:
                return ("unsigned int", _TYPE_NAME[dtype])
            if n == 8:
                return ("uint2", _TYPE_NAME[dtype])
            if n == 16:
                return ("uint4", _TYPE_NAME[dtype])
        return None


_SM70_F8_DECODE_PRELUDE = r"""
static __device__ __forceinline__ __half emmy_sm70_f8e4m3_to_f16(__nv_fp8_e4m3 value) {
    const unsigned char bits = *reinterpret_cast<const unsigned char*>(&value);
    const unsigned short sign = static_cast<unsigned short>(bits & 0x80u) << 8;
    const unsigned short magnitude = static_cast<unsigned short>(bits & 0x7fu);
    if (magnitude == 0x7fu) {
        return __ushort_as_half(sign | 0x7e00u);
    }
    const __half unscaled = __ushort_as_half(sign | (magnitude << 7));
    return __hmul(unscaled, __ushort_as_half(23u << 10));
}
"""
