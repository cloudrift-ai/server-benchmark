"""Byte-identical round-trip of every codec over the canonical golden shape corpus.

Each codec's ``parse(s, work).spell() == s`` for the exact string shapes that appear on disk. This
locks the ser/de against the wire format independent of whether a golden currently happens to use a
given shape. ``TILE`` / ``REDUCE`` values are SITE-LOCAL, so they round-trip *against* the
inventory their worker widths live in (``WORK``); ``STAGE`` is inventory-free.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.schedule import ReducePlan, Stage, TilePlan, Workers


@pytest.mark.parametrize(
    ("spec", "work"),
    [("", ""), ("coop", "t8"), ("coop-t", "t256"), ("r4", ""), ("g2a", ""), ("g2k", ""), ("g4a/coop", "t32"), ("g2k/coop/r4", "t16")],
)
def test_reduce_round_trip(spec: str, work: str) -> None:
    assert ReducePlan.parse(spec, Workers.parse(work)).spell() == spec


@pytest.mark.parametrize(("spec", "work"), [("", ""), ("", "t4"), ("f2", ""), ("", "t32x16"), ("f2x4", "t32x16"), ("f2x2", "t4x4")])
def test_tile_scalar_round_trip(spec: str, work: str) -> None:
    assert TilePlan.parse(spec, Workers.parse(work)).spell() == spec


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
    assert TilePlan.parse(spec, Workers.parse(work)).spell() == spec


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("mma_m16n8k16_f16/f1x2/k8", "mma_m16n8k16_f16_f32/f1x2/k8"),
        ("mma_m16n8k16_bf16/f2x2/k4", "mma_m16n8k16_bf16_f32/f2x2/k4"),
    ],
)
def test_warp_alias_canonicalizes(alias: str, canonical: str) -> None:
    """The historical acc-unspecified atom spellings stay parse ALIASES for the f32-accumulate
    atoms (mma_<shape>_<ab>_<acc> is the canonical convention): an aliased spelling parses and
    re-spells canonically, so old pins / goldens / DB rows join with new rows."""
    assert TilePlan.parse(alias, Workers.parse("w2x1")).spell() == canonical


@pytest.mark.parametrize("spec", ["d1/sync", "d1/cp", "d2/cp", "d3/tma", "d4/cp/p2"])
def test_stage_round_trip(spec: str) -> None:
    assert Stage.parse(spec).spell() == spec
