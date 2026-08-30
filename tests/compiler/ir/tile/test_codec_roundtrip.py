"""Byte-identical round-trip of every codec over the canonical golden shape corpus.

Each codec's ``parse(s, work).spell() == s`` for the exact string shapes that appear on disk. This
locks the ser/de against the wire format independent of whether a golden currently happens to use a
given shape. ``TILE`` / ``REDUCE`` values are SITE-LOCAL, so they round-trip *against* the
inventory their worker widths live in (``WORK``); ``STAGE`` is inventory-free.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.schedule import Reduce, Stage, Tile, Work


@pytest.mark.parametrize(
    ("spec", "work"),
    [("", ""), ("coop", "t8"), ("coop-t", "t256"), ("r4", ""), ("g2a", ""), ("g2k", ""), ("g4a/coop", "t32"), ("g2k/coop/r4", "t16")],
)
def test_reduce_round_trip(spec: str, work: str) -> None:
    assert Reduce.parse(spec, Work.parse(work)).spell() == spec


@pytest.mark.parametrize("spec", ["coop/g2k", "g02k", "g2/coop", " g2k"])
def test_reduce_noncanonical_spellings_raise(spec: str) -> None:
    with pytest.raises(ValueError):
        Reduce.parse(spec, Work.parse("t8"))


@pytest.mark.parametrize(("spec", "work"), [("", ""), ("", "t4"), ("f2", ""), ("", "t32x16"), ("f2x4", "t32x16"), ("f2x2", "t4x4")])
def test_tile_scalar_round_trip(spec: str, work: str) -> None:
    assert Tile.parse(spec, Work.parse(work)).spell() == spec


@pytest.mark.parametrize(
    ("spec", "work"),
    [
        ("mma_m16n8k16_f16_f32/f1x1", "w1x1"),
        ("mma_m16n8k16_f16_f32/f1x2/k8", "w2x1"),
        ("mma_m16n8k16_bf16_f32/f2x2/k4", "w2x2"),
        ("mma_m16n8k16_f16_f16/f2x2/k4", "w2x2"),
    ],
)
def test_warp_round_trip(spec: str, work: str) -> None:
    assert Tile.parse(spec, Work.parse(work)).spell() == spec


@pytest.mark.parametrize("alias", ["mma_m16n8k16_f16/f1x2/k8", "mma_m16n8k16_bf16/f2x2/k4"])
def test_warp_atom_aliases_raise(alias: str) -> None:
    """The f32 accumulator is part of an atom's one wire spelling."""
    with pytest.raises(ValueError, match="unknown atom kind"):
        Tile.parse(alias, Work.parse("w2x1"))


@pytest.mark.parametrize("spec", ["d1/smem", "d1/smem-async", "d2/smem-async", "d3/smem-tma", "d4/smem-async/p2", "d1/smem-tma"])
def test_stage_round_trip(spec: str) -> None:
    assert Stage.parse(spec).spell() == spec


def test_stage_reordered_tokens_raise() -> None:
    with pytest.raises(ValueError, match="not canonical"):
        Stage.parse("smem-async/d2/p4")


@pytest.mark.parametrize("spec", ["d2/smem-async/d3", "smem/smem-tma", "d1/smem-async/p2/p4", "d1/smem-tma/d2"])
def test_stage_field_spelled_twice_raises(spec: str) -> None:
    """A repeated token has no last-one-wins reading a caller could have meant."""
    with pytest.raises(ValueError, match="spelled twice"):
        Stage.parse(spec)


@pytest.mark.parametrize("spec", ["d0", "zzz", "ring", "d1/smem-async/alt", "p0"])
def test_stage_malformed_raises_value_error_naming_the_codec(spec: str) -> None:
    """Only ``ValueError``, and the message names ``STAGE`` — the featurizers degrade on a
    ``ValueError`` and a bad pin has to name the knob it came from. ``ring`` / ``alt`` are the
    retired grammar: they raise rather than resolving to a neighbouring kernel."""
    with pytest.raises(ValueError, match="STAGE"):
        Stage.parse(spec)
