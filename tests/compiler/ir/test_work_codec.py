"""The step-7 value-grammar split's codec core — the ``WORK`` worker-inventory family and the
SITE-LOCAL ``TILE`` / ``REDUCE`` value forms.

The end-state grammar stores each kernel-global fact ONCE: the worker inventory spells exactly
once (``WORK: w4x1[+p<n>] / t16x8 / t512`` — the ``+p`` band absorbs the retired ``WSPEC``
family), a ``TILE`` value holds only site-local facts (``<atom>/f<FM>x<FN>[/k<bk>]`` — no
``a:``/``w``/``n`` worker tokens; the tier discriminator IS the worker kind), and a ``REDUCE``
value spells the pipeline minus the coop width (``[g<n>[a|k]][/coop[-t]][/r<n>]``). The GRID
finalize letter is KEPT — a deliberate deviation from the plan's re-spell table (``a``/``k`` is
the atomic-vs-deferred finalize MODE, not an axis token; ``g4a`` and ``g2k`` are different rows,
both live in the golden corpus). Legacy embedded-token spellings stay PIN ALIASES that must
agree with the given inventory — the loud one-kernel-one-inventory rule."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.schedule import ReducePlan, TilePlan, Workers


def test_work_codec_round_trips() -> None:
    for s in ("w4x1", "w1x8", "w4x1+p1", "w2x2+p2", "t16x8", "t512"):
        w = Workers.parse(s)
        assert w is not None and w.spell() == s
    assert Workers.parse("") is None and Workers.parse(None) is None
    assert Workers.parse("t512").units == (512, 1)  # the 1-D cooperative width
    assert Workers.parse("w4x1+p1").producer == 1


def test_work_codec_rejects_malformed() -> None:
    for bad in ("x4", "w4", "t16x8+p1", "w4x1+q2", "w4x1x2"):
        with pytest.raises(ValueError):
            Workers.parse(bad)


def test_tile_site_value_sheds_the_worker_tokens() -> None:
    wp = Workers.parse("w1x8")
    p = TilePlan.parse("a:mma_m16n8k16_f16_f32/w1x8/f4x1/k4")
    assert p.spell_site() == "mma_m16n8k16_f16_f32/f4x1/k4"
    assert TilePlan.parse_site(p.spell_site(), wp) == p
    sc = TilePlan.parse("n16x8/f4x8")
    assert sc.spell_site() == "f4x8"
    assert TilePlan.parse_site("f4x8", Workers.parse("t16x8")) == sc
    assert TilePlan.parse_site("", None) == TilePlan()  # the per-cell tier


def test_tile_legacy_alias_must_agree_with_work() -> None:
    p = TilePlan.parse("a:mma_m16n8k16_f16_f32/w1x8/f4x1/k4")
    assert TilePlan.parse_site("a:mma_m16n8k16_f16_f32/w1x8/f4x1/k4", Workers.parse("w1x8")) == p
    with pytest.raises(ValueError, match="one kernel, one inventory"):
        TilePlan.parse_site("a:mma_m16n8k16_f16_f32/w1x8/f4x1/k4", Workers.parse("w4x1"))
    with pytest.raises(ValueError, match="warp WORK inventory"):
        TilePlan.parse_site("mma_m16n8k16_f16_f32/f4x1", Workers.parse("t16x8"))


def test_reduce_site_value_sheds_the_coop_width_and_keeps_the_finalize_letter() -> None:
    r = ReducePlan.parse("g16k/b256t")
    assert r.spell_site() == "g16k/coop-t"
    assert ReducePlan.parse_site("g16k/coop-t", Workers.parse("t256")) == r
    assert ReducePlan.parse("b512").spell_site() == "coop"
    assert ReducePlan.parse_site("coop", Workers.parse("t512")) == ReducePlan.parse("b512")
    # The finalize letter survives — g4a (atomic) and g2k (deferred kernel) stay distinct rows.
    assert ReducePlan.parse("g4a").spell_site() == "g4a"
    assert ReducePlan.parse("g2k").spell_site() == "g2k"
    assert ReducePlan.parse_site("g4a", None).finalize == "atomic"
    with pytest.raises(ValueError, match="thread WORK inventory"):
        ReducePlan.parse_site("coop", Workers.parse("w4x1"))
