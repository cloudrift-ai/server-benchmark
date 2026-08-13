"""Scalar / structured ``DataType`` hierarchy."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.dtype import BF16, F8E4M3, F8E5M2, F16, F32, I16, DataType, F4E2M1x2, F16x2, StructuredType, decode_f4x2, get


def test_scalars_are_not_structured():
    for dt in (F32, F16, BF16, F8E4M3, F8E5M2):
        assert not dt.is_structured
        assert not isinstance(dt, StructuredType)


def test_f16x2_is_structured():
    assert F16x2.is_structured
    assert isinstance(F16x2, StructuredType)
    # Still a DataType, still resolvable by canonical name.
    assert isinstance(F16x2, DataType)
    assert get("f16x2") is F16x2


def test_f8_bits_carriers():
    # fp8 follows the BF16 bits-carrier precedent: numpy has no fp8, uint8
    # carries the bit pattern; torch dtype spellings resolve as aliases.
    for dt, alias in ((F8E4M3, "float8_e4m3fn"), (F8E5M2, "float8_e5m2")):
        assert dt.np == np.dtype(np.uint8)
        assert dt.nbytes == 1
        assert get(dt.name) is dt
        assert get(alias) is dt


def test_i16_packed_code_carrier():
    # int16 carries trellis-coded (EXL3) checkpoint code words bit-identically
    # through the loader; like i32/i64 it never reaches a kernel.
    assert I16.np == np.dtype(np.int16)
    assert I16.nbytes == 2
    assert get("i16") is I16
    assert get("int16") is I16


def test_structured_keeps_scalar_carrier_info():
    # The packed type still reports a usable numpy dtype + byte width
    # (one 32-bit register = two fp16).
    assert F16x2.np == F16.np
    assert F16x2.nbytes == 4


def test_f4e2m1x2_packed_pair_carrier():
    # One uint8 element carries a PAIR of e2m1 codes — structured, like F16x2.
    assert F4E2M1x2.is_structured
    assert isinstance(F4E2M1x2, StructuredType)
    assert F4E2M1x2.np == np.dtype(np.uint8)
    assert F4E2M1x2.nbytes == 1
    assert get("f4e2m1x2") is F4E2M1x2
    assert get("float4_e2m1fn_x2") is F4E2M1x2  # torch alias


def test_decode_f4x2_all_codes_match_bit_formula():
    """All 16 codes vs an independent sign/exponent/mantissa derivation. torch cannot
    decode fp4 on cpu (``copy_kernel`` not implemented for Float4_e2m1fn_x2), so unlike
    fp8 the oracle is the e2m1 formula itself: (-1)^s * (1 + m/2) * 2^(e-1), with the
    e == 0 subnormal worth m/2."""
    codes = np.arange(16, dtype=np.uint8)
    packed = (codes[1::2] << 4 | codes[0::2]).astype(np.uint8)  # pairs (0,1), (2,3), ...
    s = np.where(codes >> 3, -1.0, 1.0)
    e = (codes >> 1) & 0b11
    m = codes & 1
    ref = s * np.where(e == 0, m / 2.0, (1 + m / 2.0) * np.exp2(e - 1.0))
    np.testing.assert_array_equal(decode_f4x2(packed), ref.astype(np.float32))


def test_decode_f4x2_packing_order_and_shape():
    """b = v_even + 16 * v_odd, last axis doubles, leading axes untouched. The order
    convention is pinned externally in the loader tests against a real NVFP4 checkpoint;
    here 0x21 = (v0=1, v1=2) -> (0.5, 1.0) and 0x43 = (v2=3, v3=4) -> (1.5, 2.0)."""
    bits = np.array([[0x21, 0x43]], dtype=np.uint8)
    np.testing.assert_array_equal(decode_f4x2(bits), np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32))
    assert decode_f4x2(np.zeros((3, 2, 5), dtype=np.uint8)).shape == (3, 2, 10)


def test_decode_f4x2_rejects_non_uint8():
    # The carrier contract is asserted, not coerced — garbage input is an error.
    with pytest.raises(AssertionError):
        decode_f4x2(np.zeros((2, 2), dtype=np.int64))
