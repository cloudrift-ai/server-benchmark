"""The value grammar's codec core — the ``WORK`` worker-inventory family and the SITE-LOCAL
``TILE`` / ``REDUCE`` value forms.

The grammar stores each kernel-global fact ONCE: the worker inventory spells exactly once
(``WORK: w4x1[+p<n>] / t16x8 / t512`` — the ``+p`` band absorbs the retired ``WSPEC`` family), a
``TILE`` value holds only site-local facts (``<atom>/f<FM>x<FN>[/k<bk>]`` — no worker tokens; the
tier discriminator IS the worker kind), and a ``REDUCE`` value spells the pipeline minus the coop
width (``[g<n>[a|k]][/coop[-t]][/r<n>]``). The GRID finalize letter is KEPT — ``a``/``k`` is the
atomic-vs-deferred finalize MODE, not an axis token; ``g4a`` and ``g2k`` are different rows, both
live in the golden corpus. There is no second, self-contained reading: the retired embedded-worker
spellings raise."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.schedule import Reduce, Tile, Work, resolve_site_tile


def test_work_codec_round_trips() -> None:
    for s in ("w4x1", "w1x8", "w4x1+p1", "w2x2+p2", "t16x8", "t512"):
        w = Work.parse(s)
        assert w is not None and w.spell() == s
    assert Work.parse("") == Work.parse(None) == Work()
    assert Work.parse("t512").units == (512, 1)  # the 1-D cooperative width
    assert Work.parse("w4x1+p1").producer == 1


def test_work_codec_rejects_malformed() -> None:
    for bad in ("x4", "w4", "t16x8+p1", "w4x1+q2", "w4x1x2", "w04x1", "t16x1", "w4x1+p0", " w4x1"):
        with pytest.raises(ValueError):
            Work.parse(bad)


def test_tile_site_value_carries_no_worker_tokens() -> None:
    from emmy.compiler.ir.atom import ATOM_REGISTRY

    atom = ATOM_REGISTRY["mma_m16n8k16_f16_f32"]
    warp = Tile(atom=atom, units=(1, 8), regs=(4, 1), bk=4)
    assert warp.spell() == "mma_m16n8k16_f16_f32/f4x1/k4"  # the w1x8 units live in WORK
    assert Tile.parse(warp.spell(), Work.parse("w1x8")) == warp
    # Both tiers store (m, n); the scalar value spells n-then-m, so ``t16x8``/``f4x8`` is m=8, n=16 / m=8, n=4.
    scalar = Tile(units=(8, 16), regs=(8, 4))
    assert scalar.spell() == "f4x8"
    assert Tile.parse("f4x8", Work.parse("t16x8")) == scalar
    assert Tile.parse("", None) == Tile()  # the per-cell tier


def test_unit_register_thread_tile_has_an_explicit_spelling() -> None:
    work = Work.parse("t4x2")
    tile = Tile(units=(2, 4))

    assert tile.spell() == "f1"
    assert resolve_site_tile("f1", work) == tile
    assert resolve_site_tile("", work) == Tile()


def test_tile_warp_value_needs_a_warp_inventory() -> None:
    with pytest.raises(ValueError, match="warp WORK inventory"):
        Tile.parse("mma_m16n8k16_f16_f32/f4x1", Work.parse("t16x8"))


def test_tile_rejects_the_retired_embedded_worker_spellings() -> None:
    # The worker widths have exactly ONE home; a value that carries its own must not decode into a
    # second, self-contained reading that silently disagrees with WORK.
    for legacy in ("n16x8/f4x8", "a:mma_m16n8k16_f16_f32/w1x8/f4x1/k4"):
        with pytest.raises(ValueError):
            Tile.parse(legacy, Work.parse("w4x1"))


def test_reduce_site_value_sheds_the_coop_width_and_keeps_the_finalize_letter() -> None:
    r = Reduce.of(cta=16, coop=256, coop_transposed=True)
    assert r.spell() == "g16k/coop-t"  # the 256 lives in WORK
    assert Reduce.parse("g16k/coop-t", Work.parse("t256")) == r
    assert Reduce.of(coop=512).spell() == "coop"
    assert Reduce.parse("coop", Work.parse("t512")) == Reduce.of(coop=512)
    assert Reduce.parse("", None) == Reduce()  # the scalar serial fold
    # The finalize letter survives — g4a (atomic) and g2k (deferred kernel) stay distinct rows.
    assert Reduce.parse("g4a", None).spell() == "g4a"
    assert Reduce.parse("g2k", None).spell() == "g2k"
    assert Reduce.parse("g4a", None).finalize == "atomic"
    with pytest.raises(ValueError, match="thread WORK inventory"):
        Reduce.parse("coop", Work.parse("w4x1"))


def test_reduce_rejects_the_retired_coop_width_spelling() -> None:
    for legacy in ("b512", "b256t", "g8k/b128t"):
        with pytest.raises(ValueError, match="unknown token"):
            Reduce.parse(legacy, Work.parse("t512"))
