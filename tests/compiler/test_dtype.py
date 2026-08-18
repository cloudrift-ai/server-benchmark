"""Scalar / structured ``DataType`` hierarchy."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.dtype import (
    BF16,
    F8E4M3,
    F8E5M2,
    F16,
    F32,
    I16,
    DataType,
    F4E2M1x2,
    F16x2,
    StructuredType,
    decode_f4,
    decode_f4x2,
    encode_f4,
    encode_f4x2,
    get,
)


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


def test_logical_elems_per_stored_element():
    # Packed pairs carry two logical elements per stored element; scalars carry one.
    assert F4E2M1x2.logical_elems == 2
    assert F16x2.logical_elems == 2
    for dt in (F32, F16, BF16, F8E4M3, F8E5M2, I16):
        assert dt.logical_elems == 1


def test_decode_f4_single_codes():
    # The unpacked-code sibling: LUT order matches decode_f4x2's low-nibble decode.
    codes = np.arange(16, dtype=np.int32)
    packed = (codes[1::2] << 4 | codes[0::2]).astype(np.uint8)
    np.testing.assert_array_equal(decode_f4(codes), decode_f4x2(packed))
    with pytest.raises(AssertionError):
        decode_f4(np.array([16], dtype=np.int32))  # upper bits set
    with pytest.raises(AssertionError):
        decode_f4(np.array([1.0]))  # non-integer carrier


def test_encode_f4_round_trips_every_representable_code():
    # encode_f4 is decode_f4's inverse, so every code must come back as itself — including the
    # two zeros, whose sign survives.
    codes = np.arange(16, dtype=np.uint8)
    np.testing.assert_array_equal(encode_f4(decode_f4(codes)), codes)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.25, 0.0), (0.75, 1.0), (1.25, 1.0), (1.75, 2.0), (2.5, 2.0), (3.5, 4.0), (5.0, 4.0)],
)
def test_encode_f4_breaks_ties_to_even(value, expected):
    # Every midpoint of the e2m1 grid, so the round-to-nearest-EVEN claim is pinned rather than
    # asserted. Ties land on the even code, which is the one with a clear low mantissa bit.
    assert float(decode_f4(encode_f4(np.float32(value)))) == expected


def test_encode_f4_saturates_and_keeps_sign():
    # e2m1 has no inf code, so a huge magnitude clamps to +-6 rather than overflowing.
    assert float(decode_f4(encode_f4(np.float32(1e9)))) == 6.0
    assert float(decode_f4(encode_f4(np.float32(-1e9)))) == -6.0
    assert encode_f4(np.float32(-0.0)) == 8  # negative zero has its own code
    with pytest.raises(AssertionError):
        encode_f4(np.array([np.nan], dtype=np.float32))  # nowhere for a NaN to go


def test_encode_f4x2_inverts_decode_f4x2():
    # Bit identity over random packed bytes: the pair order must match, not merely the values.
    bits = np.random.default_rng(0).integers(0, 256, size=(4, 8), dtype=np.uint8)
    np.testing.assert_array_equal(encode_f4x2(decode_f4x2(bits)), bits)


def test_encode_f4x2_rejects_an_odd_last_axis():
    with pytest.raises(AssertionError):
        encode_f4x2(np.zeros((3,), dtype=np.float32))
