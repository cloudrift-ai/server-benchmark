"""Codec parse validation — degenerate / malformed knob pins raise a clear ``ValueError``.

A ``0`` width (``f0`` / ``g0``) used to parse to a level the plan silently dropped — a no-op pin
whose knob column still echoed it — and a missing number (``g``) threw a bare ``int('')`` error.
Each codec rejects empty / non-numeric / ``< 1`` widths uniformly; a ``1`` width stays legal (the
level is off, the identity). Since the values went SITE-LOCAL the retired embedded-worker
spellings are rejected the same way: the worker widths have one home, so a value carrying its own
must not decode into a second, self-contained reading.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.schedule import ReducePlan, TilePlan, Workers

_WARP = Workers.parse("w1x1")
_THREADS = Workers.parse("t8")


@pytest.mark.parametrize("spec", ["g0", "g", "r0", "r", "rx", "coop2", "g2k/b0", "b32", "b256t"])
def test_reduce_codec_rejects_degenerate(spec: str) -> None:
    with pytest.raises(ValueError, match="REDUCE"):
        ReducePlan.parse(spec, _THREADS)


@pytest.mark.parametrize("spec", ["r1", "g1k", ""])  # width 1 / absent = level off — the legal identity
def test_reduce_codec_allows_identity(spec: str) -> None:
    ReducePlan.parse(spec, _THREADS)  # no raise


@pytest.mark.parametrize("spec", ["f0", "f0x4", "f2x0", "f", "fx", "n4x4", "n32x16/f2x4"])
def test_tile_codec_rejects_degenerate(spec: str) -> None:
    with pytest.raises(ValueError, match="TILE|invalid literal"):
        TilePlan.parse(spec, _THREADS)


@pytest.mark.parametrize("spec", ["", "f1", "f2x2"])
def test_tile_codec_allows_valid(spec: str) -> None:
    TilePlan.parse(spec, _THREADS)  # no raise


@pytest.mark.parametrize(
    "spec",
    [
        "mma_m16n8k16_f16_f32/f0x1/k1",
        "mma_m16n8k16_f16_f32/f1x1/k0",
        "mma_m16n8k16_f16_f32/w1x1/f1x1/k1",  # the retired embedded-worker spelling
    ],
)
def test_warp_codec_rejects_degenerate(spec: str) -> None:
    with pytest.raises(ValueError, match="TILE|invalid literal"):
        TilePlan.parse(spec, _WARP)


def test_warp_codec_allows_identity() -> None:
    TilePlan.parse("mma_m16n8k16_f16_f32/f1x1/k1", _WARP)  # all-1 widths — no raise
