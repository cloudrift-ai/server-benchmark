"""Generic data-type identity used throughout the compiler.

Holds only generic + numpy information. Backend-specific traits (CUDA C
spelling, cupy dtype, required headers) live in the respective backend
modules (e.g. ``emmy/compiler/backend/cuda/dtype.py``).

Naming convention: the class is ``DataType``; every argument, variable,
and field that carries one is named ``dtype``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np


@dataclass(frozen=True)
class DataType:
    """Identity of a tensor element type.

    ``name`` is the canonical token written on ``Tensor.dtype`` and used
    everywhere the graph compares dtypes (e.g. structural keys).

    Two sub-families (see :func:`is_structured`):

    - **Scalar** — a single logical element per value (``F32`` / ``F16`` /
      ``BF16`` / ``I32`` / ``I64``). These are plain ``DataType`` instances.
    - **Structured** (:class:`StructuredType`) — a register-composite value
      with a hardware-specific layout, e.g. the packed vector ``F16x2``
      (``__half2``), as opposed to a plain scalar element.
    """

    name: str
    np: np.dtype
    nbytes: int

    def __str__(self) -> str:
        return self.name

    @property
    def is_structured(self) -> bool:
        """True for register-composite types (packed vectors, fragments)."""
        return False


@dataclass(frozen=True)
class StructuredType(DataType):
    """A register-composite type — a packed vector like ``F16x2`` — as opposed
    to a plain scalar element.

    It occupies a register with a hardware-specific layout (``__half2``); the
    renderer / lowering keys on the concrete subtype rather than treating it as
    a scalar."""

    @property
    def is_structured(self) -> bool:
        return True


F32 = DataType("f32", np.dtype(np.float32), 4)
F16 = DataType("f16", np.dtype(np.float16), 2)
# BFloat16 — same 2-byte footprint as F16 but different exponent / mantissa
# split (8 / 7 vs 5 / 10). Used by the mma.sync bf16 atom kind (M9 of the MMA
# fragment-factorization plan). NumPy has no first-class bf16; we map to
# the closest carrier (uint16 with the bf16 bit-pattern) so Tensor.dtype
# round-trips through serialization. CUDA-side spelling is
# ``__nv_bfloat16`` (see ``backend/cuda/dtype.py``).
BF16 = DataType("bf16", np.dtype(np.uint16), 2)
# FP8 storage formats (e4m3 = 4 exponent / 3 mantissa bits, e5m2 = 5 / 2).
# Same bits-carrier precedent as BF16: NumPy has no first-class fp8, so
# uint8 carries the bit pattern; the loader decodes to real values at bind
# time (M1) and CUDA-side conversion is a later milestone. CUDA spellings
# are ``__nv_fp8_e4m3`` / ``__nv_fp8_e5m2`` (see ``backend/cuda/dtype.py``).
F8E4M3 = DataType("f8e4m3", np.dtype(np.uint8), 1)
F8E5M2 = DataType("f8e5m2", np.dtype(np.uint8), 1)
# Two ``__half`` values packed into a 32-bit register, semantically a
# 2-wide vector of fp16 — the first ``StructuredType``. Same numpy dtype as
# F16 since numpy doesn't distinguish — packing is a CUDA-side storage detail;
# the canonical IR token "f16x2" is what the renderer keys on.
F16x2 = StructuredType("f16x2", np.dtype(np.float16), 4)


# Integer types — appear on ``input_ids`` placeholders from HF whole-model
# traces. The compiler doesn't generate kernels that compute on them today
# (LM-head gather + embedding lookup is index math); they exist so the
# graph can carry the right Tensor.dtype past the placeholder.
# I16 is the packed-code carrier of trellis-coded (EXL3) checkpoints: the
# stored code words are int16 and bind bit-identically through the loader;
# like I32/I64 it never reaches a kernel (the decode cone folds at bind time).
I16 = DataType("i16", np.dtype(np.int16), 2)
I32 = DataType("i32", np.dtype(np.int32), 4)
I64 = DataType("i64", np.dtype(np.int64), 8)

# Bool — appears on the explicit-mask construction ops in HF whole-model
# traces (comparisons / logical masks feeding the attention bias). Like the
# integer types it exists so the trace can carry the right Tensor.dtype;
# no kernel computes on it today.
BOOL = DataType("bool", np.dtype(np.bool_), 1)


_BY_NAME: dict[str, DataType] = {dt.name: dt for dt in (F32, F16, BF16, F8E4M3, F8E5M2, F16x2, I16, I32, I64, BOOL)}

# Aliases let callers feed PyTorch/numpy-style names without re-canonicalizing
# at every callsite. The canonical name (``F32.name == "f32"``) is what lands
# on ``Tensor.dtype``.
_ALIASES: dict[str, str] = {
    "float32": "f32",
    "float": "f32",
    "float16": "f16",
    "half": "f16",
    "bfloat16": "bf16",
    "float8_e4m3fn": "f8e4m3",  # torch spelling ("fn" — finite + NaN, no inf)
    "float8_e5m2": "f8e5m2",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "long": "i64",
}


# ---------------------------------------------------------------------------
# FP8 numpy decode — the value semantics of the two fp8 bits-carrier dtypes.
# Lives here (the generic + numpy layer) because it is dtype information, not
# loader logic: both the loader's bind-time reads and the ``from_f8*`` decode
# intrinsics in ``ir/elementwise.py`` (the in-graph cast) must share ONE
# table, and this module is a leaf both can import.
# ---------------------------------------------------------------------------

# canonical token → (exponent bits, mantissa bits, exponent bias)
_F8_SPECS: dict[str, tuple[int, int, int]] = {"f8e4m3": (4, 3, 7), "f8e5m2": (5, 2, 15)}


@cache
def _f8_lut(fmt: str) -> np.ndarray:
    """256-entry f32 lookup table for an fp8 format, one value per bit pattern.

    Generated by pure-numpy bit manipulation; unit-tested against torch's
    float8 view of the same codes (``tests/compiler/loader/test_quant.py``).
    """
    exp_bits, man_bits, bias = _F8_SPECS[fmt]
    codes = np.arange(256, dtype=np.uint16)
    sign = np.where(codes >> 7, -1.0, 1.0)
    exp = (codes >> man_bits) & ((1 << exp_bits) - 1)
    man = codes & ((1 << man_bits) - 1)
    max_exp = (1 << exp_bits) - 1
    frac = man.astype(np.float64) / (1 << man_bits)
    # exp == 0 → subnormal (no implicit leading 1, fixed exponent 1 - bias).
    vals = sign * np.where(exp == 0, frac * 2.0 ** (1 - bias), (1.0 + frac) * np.exp2(exp.astype(np.float64) - bias))
    if fmt == "f8e4m3":
        # torch's float8_e4m3fn ("fn" — finite + NaN): no infinities; only the
        # all-ones codes 0x7f / 0xff are NaN, the rest of exp == 15 are normal
        # values up to ±448.
        vals[(exp == max_exp) & (man == (1 << man_bits) - 1)] = np.nan
    else:
        # e5m2 is IEEE-conformant: exp == 31 is ±inf (mantissa 0) or NaN.
        inf_mask = (exp == max_exp) & (man == 0)
        vals[inf_mask] = sign[inf_mask] * np.inf
        vals[(exp == max_exp) & (man != 0)] = np.nan
    return vals.astype(np.float32)


def decode_f8(bits: np.ndarray, fmt: str) -> np.ndarray:
    """Decode an fp8 bit-pattern array (uint8 carrier) to f32 by LUT indexing."""
    return _f8_lut(fmt)[np.asarray(bits).astype(np.uint8, copy=False)]


@cache
def _f8_finite_values(fmt: str) -> tuple[np.ndarray, int]:
    """The ascending finite non-negative values of an fp8 format and the count of finite positive
    codes — the encode's search table. e4m3 (``fn``): codes 0x00..0x7E are finite and ascending
    (only 0x7F is NaN); e5m2 (IEEE): 0x00..0x7B (0x7C is inf, 0x7D..0x7F NaN)."""
    n_finite = 0x7F if fmt == "f8e4m3" else 0x7C
    return _f8_lut(fmt)[:n_finite].astype(np.float64), n_finite


def encode_f8(values: np.ndarray, fmt: str) -> np.ndarray:
    """Encode an f32/f16 array to fp8 bit patterns (uint8 carrier) — round-to-nearest-EVEN onto
    the format's representable set, SATURATING to the max finite value (the ``cuda_fp8.h``
    constructor's ``__NV_SATFINITE`` rounding; torch-verified for in-range values in
    ``tests/compiler/passes/test_fp8_mma.py``), NaN → the canonical all-ones NaN code. The exact
    inverse-of-:func:`decode_f8` semantics the ``to_f8*`` encode intrinsics carry."""
    vals, n_finite = _f8_finite_values(fmt)
    with np.errstate(invalid="ignore"):  # a signaling-NaN payload raises FE_INVALID on the widening cast
        x = np.asarray(values, dtype=np.float64)
    sign = np.signbit(x)
    a = np.abs(x)
    # ``vals`` is exactly the representable set, so nearest-neighbor search with a tie-to-even-code
    # rule IS round-to-nearest-even (adjacent codes differ in the low mantissa bit; across an
    # exponent boundary the higher code has mantissa 0 — even either way).
    idx = np.clip(np.searchsorted(vals, a), 1, n_finite - 1)
    lo, hi = vals[idx - 1], vals[idx]
    take_hi = (a - lo > hi - a) | ((a - lo == hi - a) & (idx % 2 == 0))
    code = np.where(take_hi, idx, idx - 1).astype(np.uint8)
    code = np.where(a >= vals[-1], np.uint8(n_finite - 1), code)  # satfinite
    code = code | (sign.astype(np.uint8) << 7)
    return np.where(np.isnan(x), np.uint8(0x7F), code).astype(np.uint8)


def get(dtype: str | DataType) -> DataType:
    """Resolve a name (canonical or alias) or pass through a ``DataType``."""
    if isinstance(dtype, DataType):
        return dtype
    if dtype in _BY_NAME:
        return _BY_NAME[dtype]
    canonical = _ALIASES.get(dtype)
    if canonical is not None:
        return _BY_NAME[canonical]
    raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(_BY_NAME) + sorted(_ALIASES)}")
