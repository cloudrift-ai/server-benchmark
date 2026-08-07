"""Scalar / structured ``DataType`` hierarchy."""

from __future__ import annotations

import numpy as np

from emmy.compiler.dtype import BF16, F8E4M3, F8E5M2, F16, F32, I16, DataType, F16x2, StructuredType, get


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
