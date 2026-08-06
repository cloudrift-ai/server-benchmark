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


@pytest.mark.parametrize("spec", ["d1/sync", "d1/cp", "d2/cp", "d3/tma", "d4/cp/p2", "d1/tma/split"])
def test_stage_round_trip(spec: str) -> None:
    assert Stage.parse(spec).spell() == spec


def test_stage_binding_is_order_free() -> None:
    """Tokens carry their own field, so the order they are written in is not part of the value."""
    assert Stage.parse("cp/d2/p4") == Stage.parse("d2/cp/p4")


@pytest.mark.parametrize("spec", ["d2/cp/d3", "sync/tma", "d1/cp/p2/p4", "d1/tma/split/split"])
def test_stage_field_spelled_twice_raises(spec: str) -> None:
    """Binding is order-free, so a repeated token has no last-one-wins reading a caller could have
    meant: ``d2/cp/d3`` would land ``d3/cp`` and ``sync/tma`` would land ``d1/tma``, each quietly
    deploying a kernel the pin did not name."""
    with pytest.raises(ValueError, match="spelled twice"):
        Stage.parse(spec)


@pytest.mark.parametrize("spec", ["d0", "zzz", "ring", "d1/cp/alt", "p0"])
def test_stage_malformed_raises_value_error_naming_the_codec(spec: str) -> None:
    """Only ``ValueError``, and the message names ``STAGE`` — the featurizers degrade on a
    ``ValueError`` and a bad pin has to name the knob it came from. ``ring`` / ``alt`` are the
    retired grammar: they raise rather than resolving to a neighbouring kernel."""
    with pytest.raises(ValueError, match="STAGE"):
        Stage.parse(spec)
