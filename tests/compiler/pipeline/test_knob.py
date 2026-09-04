"""Unit tests for ``Knob`` parse / pretty round-trips and the registry."""

from __future__ import annotations

import pytest

import emmy.compiler.pipeline.knob as knob_mod
from emmy.compiler.pipeline.knob import (
    Knob,
    KnobType,
    apply_knobs_env,
    apply_off_defaults,
    axis_of,
    family_of,
    family_value,
    format_tuning_knobs,
    is_off_value,
    pin_key_matches,
    tuning_knob_items,
    values_equal,
)
from emmy.compiler.pipeline.search.features import is_warp, knob_features, mma_atom, tile_signature


def test_int_parse():
    k = Knob("BN", KnobType.INT)
    assert k.parse("64") == 64
    assert k.parse("0x40") == 64
    assert k.parse("  128 ") == 128


def test_int_pretty():
    k = Knob("BN", KnobType.INT)
    assert k.pretty(64) == "64"


def test_bool_parse():
    k = Knob("FLAG", KnobType.BOOL)
    for truthy in ("1", "true", "True", "yes", "on", " TRUE "):
        assert k.parse(truthy) is True
    for falsy in ("0", "false", "no", "off", ""):
        assert k.parse(falsy) is False


def test_bool_parse_rejects_unknown():
    # An unrecognized value used to coerce silently to False (a typo'd ``ture`` disabled the
    # knob with no diagnostic); it now fails loudly.
    k = Knob("FLAG", KnobType.BOOL)
    for bad in ("banana", "ture", "2", "yep"):
        with pytest.raises(ValueError, match="bad BOOL"):
            k.parse(bad)


def test_bool_pretty():
    k = Knob("FLAG", KnobType.BOOL)
    assert k.pretty(True) == "True"
    assert k.pretty(False) == "False"


def test_binmask_parse_binary_string():
    k = Knob("MASK", KnobType.BINMASK)
    # char i = bit i (left-to-right reads as buffer rank 0..n-1)
    assert k.parse("101", width=3) == 0b101
    assert k.parse("000", width=3) == 0
    assert k.parse("111", width=3) == 0b111


def test_binmask_parse_keywords():
    k = Knob("MASK", KnobType.BINMASK)
    assert k.parse("all", width=3) == 0b111
    assert k.parse("all", width=5) == 0b11111
    assert k.parse("none", width=3) == 0


def test_binmask_parse_int_clamps_to_width():
    k = Knob("MASK", KnobType.BINMASK)
    assert k.parse("0xFFFF", width=3) == 0b111
    assert k.parse("5", width=3) == 0b101


def test_binmask_pretty():
    k = Knob("MASK", KnobType.BINMASK)
    assert k.pretty(0b101, width=3) == "101"
    assert k.pretty(0, width=3) == "000"
    assert k.pretty(0b111, width=3) == "111"


def test_binmask_roundtrip():
    k = Knob("MASK", KnobType.BINMASK)
    for mask in range(16):
        assert k.parse(k.pretty(mask, width=4), width=4) == mask


def test_binmask_requires_width():
    k = Knob("MASK", KnobType.BINMASK)
    with pytest.raises(ValueError, match="width"):
        k.parse("101")
    with pytest.raises(ValueError, match="width"):
        k.pretty(5)


def test_env_property():
    assert Knob("BN", KnobType.INT).env == "EMMY_BN"
    assert Knob("MASK", KnobType.BINMASK).env == "EMMY_MASK"


# ---------------------------------------------------------------------------
# Knob.narrow — fold env pin into candidate enumeration
# ---------------------------------------------------------------------------


def test_narrow_unpinned_returns_candidates_unchanged(monkeypatch):
    k = Knob("BN", KnobType.INT)
    monkeypatch.delenv("EMMY_BN", raising=False)
    assert k.narrow((16, 32, 64)) == (16, 32, 64)


def test_narrow_pinned_keeps_matching_candidate(monkeypatch):
    k = Knob("BN", KnobType.INT)
    monkeypatch.setenv("EMMY_BN", "32")
    assert k.narrow((16, 32, 64)) == (32,)


def test_narrow_pinned_out_of_set_is_authoritative(monkeypatch):
    # Hints are guidance, not constraint — an env pin outside the candidate
    # tuple is honored, not silently dropped. Downstream structural gates
    # (divisibility, threads-per-CTA budget, …) still apply.
    k = Knob("BN", KnobType.INT)
    monkeypatch.setenv("EMMY_BN", "128")
    assert k.narrow((16, 32, 64)) == (128,)


def test_narrow_accepts_arbitrary_iterable(monkeypatch):
    k = Knob("BN", KnobType.INT)
    monkeypatch.setenv("EMMY_BN", "16")
    # generator, not a tuple
    assert k.narrow(x for x in (8, 16, 32)) == (16,)


def test_narrow_bool(monkeypatch):
    k = Knob("FLAG", KnobType.BOOL)
    monkeypatch.setenv("EMMY_FLAG", "true")
    assert k.narrow((True, False)) == (True,)
    monkeypatch.setenv("EMMY_FLAG", "0")
    assert k.narrow((True, False)) == (False,)


def test_narrow_binmask_rejected(monkeypatch):
    k = Knob("MASK", KnobType.BINMASK)
    monkeypatch.setenv("EMMY_MASK", "111")
    with pytest.raises(ValueError, match="BINMASK"):
        k.narrow((0b000, 0b111))


# ---------------------------------------------------------------------------
# OFF defaults + tier (is_warp / mma_atom)
# ---------------------------------------------------------------------------


def test_apply_off_defaults_fills_only_unspecified_off_knobs():
    """``apply_off_defaults`` stamps a declared knob's ``off`` when absent, leaves
    present values (incl. a prior OFF fill) untouched, and never fills a knob
    whose ``off`` is unset (the default)."""
    wm = Knob("WM", KnobType.INT, off=0)
    bk = Knob("BK", KnobType.INT)  # no off → never auto-filled
    knobs = {"BK": 64}
    apply_off_defaults(knobs, [wm, bk])
    assert knobs == {"BK": 64, "WM": 0}  # WM filled to off, BK untouched (no off)
    # Idempotent + respects a present (non-OFF) value.
    knobs2 = {"WM": 2, "BK": 64}
    apply_off_defaults(knobs2, [wm, bk])
    assert knobs2 == {"WM": 2, "BK": 64}


def test_is_warp_and_mma_atom_tier_discriminator():
    """The unified ``TILE`` knob self-discriminates: a value carrying a bare atom token is the
    warp fragment (and names the atom); a scalar ``f..`` codec beside thread ``WORK``, empty value,
    or absent value is the scalar tier (no atom)."""
    assert not is_warp({}) and mma_atom({}) is None
    assert not is_warp({"TILE@map.1/inner": ""}) and mma_atom({"TILE@map.1/inner": ""}) is None
    assert not is_warp({"TILE@map.1/inner": "f2x4", "WORK": "t32x8"})  # scalar fragment names no atom
    assert is_warp({"TILE@map.1/inner": "mma_m16n8k16_f16_f32/f2x2/k2", "WORK": "w2x2"})
    assert mma_atom({"TILE@map.1/inner": "mma_m16n8k16_f16_f32/f2x2/k2", "WORK": "w2x2"}) == "mma_m16n8k16_f16_f32"


def test_scalar_tile_features_from_thread_tile():
    """``knob_features`` emits the ``D_*`` occupancy family for a scalar row from its
    ``TILE`` codec free split (``par_n·par_m`` threads, ``par_m·reg_m × par_n·reg_n``
    output) — ``n32x8`` parallel thread-tile, ``f2x4`` register sub-tile."""
    sf = knob_features({"TILE@map.1/inner": "f2x4", "WORK": "t32x8"})
    assert any(k.startswith("D_") for k in sf)
    assert sf["D_threads"] == 32 * 8
    assert sf["D_tile_m"] == 8 * 4 and sf["D_tile_n"] == 32 * 2


def test_warp_tile_features_from_warp_tile():
    """``_warp_tile_features`` builds the ``D_*`` family from the warp form of the ``TILE`` codec
    (``WM·WN·32`` threads, ``WM·FM·atom_m × WN·FN·atom_n`` output) — the atom cell dims (16×8 for
    ``m16n8k16``) are read off the parsed atom. A scalar ``TILE`` value → empty."""
    from emmy.compiler.pipeline.search.features import _warp_tile_features  # noqa: PLC0415

    wf = _warp_tile_features({"TILE@map.1/inner": "mma_m16n8k16_f16_f32/f2x2/k2", "WORK": "w2x2", "S_ext_free_prod": 2048 * 2048})
    assert wf["D_threads"] == 128.0  # WM·WN·32
    assert wf["D_tile_m"] == 2 * 2 * 16  # WM·FM·atom_m
    assert wf["D_tile_n"] == 2 * 2 * 8  # WN·FN·atom_n
    assert "D_log2_ctas" in wf and "D_log2_waves" in wf  # occupancy present (free_prod given)
    assert _warp_tile_features({"TILE@map.1/inner": "f2x4", "WORK": "t32x8"}) == {}  # scalar fragment → empty


# ---------------------------------------------------------------------------
# EMMY_KNOBS aggregate env var
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_emmy_env():
    """``apply_knobs_env`` writes ``EMMY_<K>`` via ``config.set_knob`` —
    a direct ``os.environ`` write monkeypatch can't undo (``delenv`` on an
    absent var records nothing to restore), so splatted pins used to leak
    into later tests in the same xdist worker (pin-sensitive planner tests
    then enumerate under stray ``BK``/``BM``/``BN`` pins). Snapshot + restore
    the whole ``EMMY_*`` namespace around each test."""
    import os  # noqa: PLC0415

    saved = {k: v for k, v in os.environ.items() if k.startswith("EMMY_")}
    yield
    for k in [k for k in os.environ if k.startswith("EMMY_")]:
        if k not in saved:
            del os.environ[k]
    os.environ.update(saved)


def test_apply_knobs_env_splats_into_individual_keys(monkeypatch):
    """Aggregate env var sets ``EMMY_<K>`` per entry."""
    monkeypatch.delenv("EMMY_BK", raising=False)
    monkeypatch.delenv("EMMY_BM", raising=False)
    monkeypatch.delenv("EMMY_BN", raising=False)
    applied = apply_knobs_env("BK=2,BM=16,BN=128")
    assert applied == {"EMMY_BK": "2", "EMMY_BM": "16", "EMMY_BN": "128"}


def test_apply_knobs_env_individual_takes_precedence(monkeypatch):
    """An explicit ``EMMY_<K>`` wins over the aggregate."""
    monkeypatch.setenv("EMMY_BK", "4")
    monkeypatch.delenv("EMMY_BM", raising=False)
    applied = apply_knobs_env("BK=2,BM=16")
    assert "EMMY_BK" not in applied  # not clobbered
    assert applied == {"EMMY_BM": "16"}
    import os

    assert os.environ["EMMY_BK"] == "4"
    assert os.environ["EMMY_BM"] == "16"


def test_apply_knobs_env_tolerates_whitespace(monkeypatch):
    """Whitespace around keys / values / separators is stripped."""
    monkeypatch.delenv("EMMY_BK", raising=False)
    monkeypatch.delenv("EMMY_BM", raising=False)
    applied = apply_knobs_env(" BK = 2 ,  BM=16 ")
    assert applied == {"EMMY_BK": "2", "EMMY_BM": "16"}


def test_apply_knobs_env_skips_empty_entries(monkeypatch):
    """Empty entries (trailing comma, double comma) are skipped."""
    monkeypatch.delenv("EMMY_BK", raising=False)
    applied = apply_knobs_env("BK=2,,")
    assert applied == {"EMMY_BK": "2"}


def test_apply_knobs_env_rejects_missing_equals():
    """An entry without ``=`` is malformed and surfaces an error."""
    with pytest.raises(ValueError, match="missing '='"):
        apply_knobs_env("BK=2,BMnoequals")


def test_apply_knobs_env_rejects_empty_key():
    """An entry like ``=4`` has an empty KEY and is rejected."""
    with pytest.raises(ValueError, match="empty KEY"):
        apply_knobs_env("=4")


def test_parse_knob_spec_grammar():
    """``parse_knob_spec`` is the one owner of the ``K1=V1,K2=V2`` grammar
    (``EMMY_KNOBS`` splat + ``run --ab``): uppercased keys in spec order,
    whitespace tolerated, empties skipped, values kept as raw strings."""
    from emmy.compiler.pipeline.knob import parse_knob_spec

    assert parse_knob_spec(" bk = 2 ,, BM=16, STAGE=d2/smem-async ") == {"BK": "2", "BM": "16", "STAGE": "d2/smem-async"}
    assert parse_knob_spec("") == {}
    with pytest.raises(ValueError, match="missing '='"):
        parse_knob_spec("BK2")


def test_apply_knobs_env_uppercases_key(monkeypatch):
    """Lowercased keys round-trip to the upper-case env-var convention."""
    monkeypatch.delenv("EMMY_BK", raising=False)
    applied = apply_knobs_env("bk=2")
    assert applied == {"EMMY_BK": "2"}


# ---------------------------------------------------------------------------
# knob_features — knob dict → flat numeric feature vector
# ---------------------------------------------------------------------------


def test_knob_features_struct_passthrough():
    feats = knob_features({"S_n_load": 3.0, "S_ext_free_prod": 512.0})
    assert feats["S_n_load"] == 3.0
    assert feats["S_ext_free_prod"] == 512.0


def test_knob_features_typed_knobs(monkeypatch):
    monkeypatch.setattr(
        knob_mod,
        "_REGISTRY",
        {
            "BN": Knob("BN", KnobType.INT),
            "FLAG": Knob("FLAG", KnobType.BOOL),
            "MASK": Knob("MASK", KnobType.BINMASK),
        },
    )
    feats = knob_features({"BN": 64, "FLAG": True, "MASK": "101"})
    assert feats["BN"] == 64.0
    assert feats["FLAG"] == 1.0
    assert feats["MASK_popcount"] == 2.0
    assert feats["MASK_width"] == 3.0
    assert feats["MASK_frac"] == 2 / 3


def test_knob_features_serial_cell_work_divides_by_reduce_coverage():
    """``D_serial_cell_work``: log2 of the stamped worst per-cell serial trips after the row's
    reduce-partition coverage — the trips a thread actually serializes. cta (``g<n>``) and coop
    lanes divide; the register/ILP fold (``r<n>``) does NOT (the same thread walks every trip);
    a serial row divides by 1; no stamp → no key (an absent feature, not a fabricated zero)."""
    import math

    work = float(2**20)
    serial = knob_features({"S_ext_serial_cell_work": work})
    assert serial["D_serial_cell_work"] == pytest.approx(math.log2(1 + work))
    coop = knob_features({"S_ext_serial_cell_work": work, "WORK": "t128", "REDUCE": "coop"})
    assert coop["D_serial_cell_work"] == pytest.approx(math.log2(1 + work / 128))
    cta = knob_features({"S_ext_serial_cell_work": work, "WORK": "t64", "REDUCE": "g4k"})
    assert cta["D_serial_cell_work"] == pytest.approx(math.log2(1 + work / 4))
    reg = knob_features({"S_ext_serial_cell_work": work, "WORK": "t16x16", "REDUCE": "r4"})
    assert reg["D_serial_cell_work"] == pytest.approx(math.log2(1 + work))
    assert "D_serial_cell_work" not in knob_features({"S_n_load": 3.0})


def test_serial_floor_is_the_covered_trips_at_the_per_trip_bound():
    """``serial_floor_us``: a physical lower bound on one kernel-launch — the row's per-thread
    serial trips (stamp ÷ reduce-partition coverage) at the per-trip bound; zero with no stamp."""
    from emmy.compiler.pipeline.search.features import SERIAL_TRIP_FLOOR_US, serial_floor_us

    work = float(2**20)
    assert serial_floor_us({"S_ext_serial_cell_work": work}) == pytest.approx(work * SERIAL_TRIP_FLOOR_US)
    coop = serial_floor_us({"S_ext_serial_cell_work": work, "WORK": "t128", "REDUCE": "coop"})
    assert coop == pytest.approx(work / 128 * SERIAL_TRIP_FLOOR_US)
    assert serial_floor_us({}) == 0.0


def test_knob_features_stage_codec():
    """The ``STAGE`` codec (``d<depth>/copy|reg|cp|tma[/p<reg_depth>]``) featurizes to the
    ``D_stage_*`` family; an absent / gmem-direct stage contributes nothing."""
    feats = knob_features({"STAGE@map.1/inner": "d3/smem-tma"})
    assert feats["D_stage_depth"] == 3.0
    assert feats["D_stage_async"] == 1.0
    assert feats["D_stage_tma"] == 1.0
    assert feats["D_stage_reg_depth"] == 1.0  # no /p<n> ⇒ register pipeline OFF
    sync = knob_features({"STAGE@map.1/inner": "d2/smem-async"})
    assert sync["D_stage_depth"] == 2.0 and sync["D_stage_async"] == 1.0 and sync["D_stage_tma"] == 0.0
    # The smem→register double-buffer (``p<n>``) featurizes orthogonally to the gmem→smem ring.
    pp = knob_features({"STAGE@map.1/inner": "d3/smem-async/p2"})
    assert pp["D_stage_depth"] == 3.0 and pp["D_stage_reg_depth"] == 2.0
    assert not any(k.startswith("D_stage_") for k in knob_features({"STAGE@map.1/inner": ""}))


def test_stage_codec_reg_depth_roundtrip():
    """``Stage.parse``/``spell`` round-trip the ``p<reg_depth>`` token; ``p1`` (the default) is
    omitted so an unstaged-register config spells byte-identical to before the field existed."""
    from emmy.compiler.ir.schedule import Stage  # noqa: PLC0415

    assert Stage.parse("d3/smem-async/p2") == Stage(depth=3, transport="smem-async", reg_depth=2)
    assert Stage.parse("d2/smem-async/p4").reg_depth == 4
    assert Stage.parse("d2/smem-async").reg_depth == 1  # absent ⇒ OFF
    assert Stage(depth=2, transport="smem-async", reg_depth=2).spell() == "d2/smem-async/p2"
    assert Stage(depth=2, transport="smem-async", reg_depth=1).spell() == "d2/smem-async"  # p1 omitted
    # reg_depth is perf-only — NOT part of the structural signature (golden-match stability).
    from emmy.compiler.pipeline.search.features import _stage_sig  # noqa: PLC0415

    assert _stage_sig({"STAGE@map.1/inner": "d2/smem-async/p2"}) == _stage_sig({"STAGE@map.1/inner": "d2/smem-async"})


def test_knob_features_mma_expansion():
    # The warp fragment names its atom on the ``TILE`` codec; ``knob_features``
    # expands its physical cell / dtype properties into the ``MMA_*`` family.
    feats = knob_features({"TILE@map.1/inner": "mma_m16n8k16_f16_f32/f1x1", "WORK": "w1x1"})
    assert feats["MMA_tier"] == 1.0
    assert (feats["MMA_atom_m"], feats["MMA_atom_n"], feats["MMA_atom_k"]) == (16.0, 8.0, 16.0)
    assert feats["MMA_a_bits"] == 16.0  # f16 operand
    assert feats["MMA_acc_bits"] == 32.0  # f32 accumulator


def test_knob_features_scalar_tier_default():
    feats = knob_features({"S_n_load": 2.0})
    assert feats["MMA_tier"] == 0.0  # no atom selected


def test_knob_features_unregistered_numeric_vs_string():
    feats = knob_features({"weird_num": 7, "weird_str": "not_a_number"})
    assert feats["weird_num"] == 7.0
    assert "weird_str" not in feats


def test_knob_features_differs_by_one_knob():
    a = knob_features({"S_n_load": 2.0, "S_n_write": 1.0})
    b = knob_features({"S_n_load": 3.0, "S_n_write": 1.0})
    assert a["S_n_load"] != b["S_n_load"]
    assert a["S_n_write"] == b["S_n_write"]


def test_format_tuning_knobs_skips_struct():
    out = format_tuning_knobs({"BN": 64, "S_n_load": 3.0, "S_ext_free_prod": 512.0})
    assert "S_n_load" not in out and "S_ext_free_prod" not in out
    assert "BN=64" in out


def test_format_tuning_knobs_canonical_order():
    """The codec knobs render in canonical order (``KNOB_ORDER`` = ``TILE``, ``REDUCE``,
    ``STAGE``), not alphabetical — shared with the ``emmy eval`` golden tables."""
    out = format_tuning_knobs({"STAGE": "d2/smem-async", "REDUCE": "coop", "TILE": "mma_m16n8k16_f16_f32/f1x1"})
    assert out == "TILE=mma_m16n8k16_f16_f32/f1x1, REDUCE=coop, STAGE=d2/smem-async"


def test_apply_knobs_env_no_raw_falls_back_to_env(monkeypatch):
    """With no ``raw`` argument, the function reads ``EMMY_KNOBS``."""
    monkeypatch.delenv("EMMY_BK", raising=False)
    monkeypatch.setenv("EMMY_KNOBS", "BK=8")
    applied = apply_knobs_env()
    assert applied == {"EMMY_BK": "8"}


# --- Site-scoped schedule keys ------------------------------------------------


def test_family_and_axis_of():
    assert family_of("TILE@d") == "TILE" and axis_of("TILE@d") == "d"
    assert family_of("TILE") == "TILE" and axis_of("TILE") is None
    # A native ``MOVE@element`` splits on the first ``@`` (the ``.cta`` rides the element).
    assert family_of("REDUCE@k.cta") == "REDUCE" and axis_of("REDUCE@k.cta") == "k.cta"


def test_family_value_reads_bare_or_suffixed():
    assert family_value({"TILE@d": "x"}, "TILE") == "x"
    assert family_value({"TILE": "x"}, "TILE") == "x"
    assert family_value({"REDUCE": "coop"}, "TILE") is None


def test_pin_key_matches():
    """A global pin covers every site; a site pin matches only that exact site."""
    assert pin_key_matches("TILE", "TILE")
    assert pin_key_matches("TILE@map.3/inner", "TILE@map.3/inner")
    assert pin_key_matches("TILE", "TILE@map.3/inner")
    assert not pin_key_matches("TILE@map.3/inner", "TILE")
    assert not pin_key_matches("TILE@map.3/inner", "TILE@map.4/inner")


def _pin_registry(monkeypatch):
    monkeypatch.setattr(
        knob_mod,
        "_REGISTRY",
        {
            "FAST_EXP": Knob("FAST_EXP", KnobType.BOOL, off=False),
            "BN": Knob("BN", KnobType.INT),
            "MASK": Knob("MASK", KnobType.BINMASK),
            "TILE": Knob("TILE", KnobType.STR, off=""),
        },
    )


def test_values_equal_registry_canonical(monkeypatch):
    """Pinned-vs-realized value equality decodes both sides through the registered
    knob's canonical ``parse``, so every accepted pin spelling matches its typed
    realization: BOOL ``1``/``yes``/``on`` vs ``True``, hex INT vs decimal, BINMASK
    ``0x5``/``all`` vs the stamped binary string (width from its length — the
    ``pretty`` storage convention). Unregistered families compare by string only."""
    _pin_registry(monkeypatch)
    for spelling in ("1", "yes", "on", "TRUE"):  # the grammar 085_fast_exp advertises (EMMY_FAST_EXP=1)
        assert values_equal("FAST_EXP", spelling, True)
    assert values_equal("FAST_EXP", "0", False)
    assert not values_equal("FAST_EXP", "1", False)  # genuinely swapped BOOL
    assert values_equal("BN", "0x10", 16)  # hex pin vs decimal realization (int(s, 0))
    assert values_equal("MASK", "0x5", "101")
    assert values_equal("MASK", "all", "111")
    assert not values_equal("MASK", "0x5", 5)  # realized side must be the binary pretty() spelling
    # An @-keyed pin resolves through its family's knob.
    assert values_equal("FAST_EXP@d", "on", True)
    # Unregistered family: casefolded string equality only.
    assert values_equal("NOSUCH", "ABC", "abc")
    assert not values_equal("NOSUCH", "1", True)


def test_is_off_value(monkeypatch):
    """A family's declared OFF value means "declined / not applicable" — never a
    conflicting realization; no declared OFF (or an unregistered family) → False."""
    _pin_registry(monkeypatch)
    assert is_off_value("TILE", "")
    assert is_off_value("FAST_EXP", False)
    assert not is_off_value("TILE", "w2x1")
    assert not is_off_value("BN", 0)  # BN declares no OFF
    assert not is_off_value("NOSUCH", "")


def test_exact_site_featurizes_consistently():
    row = {
        "TILE@map.1/inner": "mma_m16n8k16_f16_f32/f2x2/k2",
        "WORK": "w2x2",
        "STAGE@map.1/inner": "d2/smem-async",
        "S_ext_free_prod": 4096.0,
    }
    from emmy.compiler.pipeline.search.features import node_slices

    (node,) = node_slices(row)
    assert {key for key in node if "@" in key} == {"TILE@map.1/inner", "STAGE@map.1/inner"}
    assert knob_features(row)["D_stage_depth"] == 2.0
    assert tile_signature(row)
    assert is_warp(row) is True
    assert mma_atom(row) == "mma_m16n8k16_f16_f32"


def test_display_renders_keys_as_stored():
    """The view renders keys as stored without collapsing distinct exact sites."""
    one = dict(tuning_knob_items({"TILE": "f2", "REDUCE": "coop", "STAGE": "d2/smem-async"}))
    assert set(one) == {"TILE", "REDUCE", "STAGE"}
    exact = dict(tuning_knob_items({"TILE@map.1/inner": "f2", "TILE@map.2/inner": "f4", "REDUCE@map.2/inner": "coop"}))
    assert set(exact) == {"TILE@map.1/inner", "TILE@map.2/inner", "REDUCE@map.2/inner"}


# --- Per-node featurizer (multi-node pool) -----------------------------------


def test_multinode_flash_keys_apart_and_pools_per_node():
    """Two contraction sites remain distinct and their geometry features sum-pool per node."""
    from emmy.compiler.pipeline.search.features import _schedule_node_features, node_slices

    qk_tile = "mma_m16n8k16_f16_f32/f2x2/k2"  # over the shared w4x1 inventory: 128 threads
    pv_tile = "mma_m16n8k16_f16_f32/f4x1/k2"  # the same inventory, its own register tile
    knobs = {
        "TILE@map.1/inner": qk_tile,
        "STAGE@map.1/inner": "d2/smem-async",
        "REDUCE@map.2/inner": "",
        "TILE@map.2/inner": pv_tile,
        "STAGE@map.2/inner": "d2/smem-async",
        "WORK": "w4x1",  # one kernel, one inventory — both nodes share the warp map
        "S_ext_free_prod": 4096.0,
    }
    assert knobs["TILE@map.1/inner"] != knobs["TILE@map.2/inner"]
    slices = node_slices(knobs)
    assert [sorted(k for k in sl if "@" in k) for sl in slices] == [
        ["STAGE@map.1/inner", "TILE@map.1/inner"],
        ["REDUCE@map.2/inner", "STAGE@map.2/inner", "TILE@map.2/inner"],
    ]
    qk = _schedule_node_features(slices[0])
    pv = _schedule_node_features(slices[1])
    feats = knob_features(knobs)
    assert feats["D_threads"] == qk["D_threads"] + pv["D_threads"] == 256.0
    assert feats["MMA_tier"] == 2.0  # both nodes are warp-tier → pooled tier count


def test_precision_pin_precedence(monkeypatch):
    """The precision-knob precedence (the FAST_MATH family): a knob's own ``EMMY_<NAME>`` pin >
    the ``FAST_MATH`` umbrella > ``None`` (neither set — the caller's conservative default). The
    umbrella itself is ``unfeatured`` (a meta gate over other knobs), so ``knob_features`` must
    never see it as a ranking dimension."""
    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC, FAST_EXP, FAST_MATH, precision_pin

    for var in ("EMMY_FAST_MATH", "EMMY_FAST_EXP", "EMMY_F16_MMA_F32_ACC"):
        monkeypatch.delenv(var, raising=False)
    assert precision_pin(FAST_EXP) is None and precision_pin(F16_MMA_F32_ACC) is None
    monkeypatch.setenv("EMMY_FAST_MATH", "1")
    assert precision_pin(FAST_EXP) is True and precision_pin(F16_MMA_F32_ACC) is True
    monkeypatch.setenv("EMMY_FAST_EXP", "0")
    assert precision_pin(FAST_EXP) is False, "the individual pin must win over the umbrella"
    assert precision_pin(F16_MMA_F32_ACC) is True
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    monkeypatch.setenv("EMMY_F16_MMA_F32_ACC", "1")
    assert precision_pin(F16_MMA_F32_ACC) is True, "the individual pin must win in the enabling direction too"
    assert FAST_MATH.unfeatured, "the umbrella is a meta gate — it must never enter the feature vector"


def test_values_equal_requires_exact_classic_spelling():
    from emmy.compiler.pipeline.knob import values_equal

    assert values_equal("TILE@map.1/inner", "mma_m16n8k16_f16_f32/f2x2/k2", "mma_m16n8k16_f16_f32/f2x2/k2")
    assert not values_equal("TILE", "mma_m16n8k16_f16_f32/f2x2/k2", "mma_m16n8k16_f16_f16/f2x2/k2")
    assert not values_equal("STAGE@map.1/inner", "smem-async/d2", "d2/smem-async")
    assert not values_equal("STAGE@map.1/inner", "d2/smem-async", " d2/smem-async")
    assert not values_equal("STAGE", "d2/smem-async", "d3/smem-async")


def test_knob_pinned_scopes_and_restores(monkeypatch):
    """``Knob.pinned`` pins ``EMMY_<NAME>`` for the block and restores the prior state — absence
    or the previous value — including on an exception (the regime-aware diagnostics' scoped gate)."""
    import pytest

    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC

    monkeypatch.delenv("EMMY_F16_MMA_F32_ACC", raising=False)
    with F16_MMA_F32_ACC.pinned("1"):
        assert F16_MMA_F32_ACC.raw() == "1"
    assert F16_MMA_F32_ACC.raw() is None
    monkeypatch.setenv("EMMY_F16_MMA_F32_ACC", "0")
    with F16_MMA_F32_ACC.pinned("1"):
        assert F16_MMA_F32_ACC.raw() == "1"
    assert F16_MMA_F32_ACC.raw() == "0"
    with pytest.raises(RuntimeError), F16_MMA_F32_ACC.pinned("1"):
        raise RuntimeError("boom")
    assert F16_MMA_F32_ACC.raw() == "0", "the pin must restore on the exception path"


def test_registry_complete_in_a_bare_process():
    """Declaring a ``Knob`` IS registering it, and ``registry()`` imports ``space.py`` itself — so
    a process that loads nothing but the featurizer still sees every canonical declaration. The
    retired module-scan registry silently dropped registry-dispatched features (``D_wspec_warps``)
    in exactly such processes (offline fit/eval tooling) and cached the empty view for the process
    lifetime."""
    import subprocess
    import sys

    code = (
        "from emmy.compiler.pipeline.search.features import knob_features\n"
        "f = knob_features({'WORK': 'w4x1+p2', 'RASTER': 'gm8'})\n"
        "assert f['D_wspec_warps'] == 2.0, f\n"
        "assert f['D_raster_group'] == 8.0, f\n"
        "from emmy.compiler.pipeline.knob import get\n"
        "assert get('UNROLL') is not None  # declared in space.py, no pass module loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_complete_kernel_row_requires_the_emitted_exact_schedule():
    from emmy.compiler.pipeline.knob import complete_kernel_row

    row = {
        "WORK": "w1x1",
        "RASTER": "",
        "TILE": "mma_m16n8k16_f16_f32/f1x1",
        "REDUCE": "",
        "STAGE": "",
        "S_ext_free_prod": 64.0,
    }
    out = complete_kernel_row(row)
    assert out["TILE"] == "mma_m16n8k16_f16_f32/f1x1"
    assert "S_ext_free_prod" not in out
    assert complete_kernel_row(out) == out

    with pytest.raises(ValueError, match="missing RASTER"):
        complete_kernel_row({"WORK": "", "REDUCE": ""})
    with pytest.raises(ValueError, match="not canonical"):
        complete_kernel_row({"WORK": "", "RASTER": "", "REDUCE@row": ""})
    with pytest.raises(ValueError, match="must be bare"):
        complete_kernel_row({"WORK@map.1/inner": "", "WORK": "", "RASTER": "", "REDUCE": ""})
    with pytest.raises(ValueError, match="no node assignment"):
        complete_kernel_row({"WORK": "", "RASTER": "", "STAGE": ""})


def test_knob_features_geometry_memo_is_invisible():
    """The geometry memo (``features._GEOMETRY_MEMO``, keyed on the :class:`NodeSlice` the block
    is computed FROM — input and key are one object) must be byte-invisible, and the slices must
    carry everything the block reads: each row's per-slice blocks equal the block computed from
    the full row dict for schema-agnostic bare analytical rows, and repeat passes (all memo hits)
    equal the first."""
    import emmy.compiler.pipeline.search.features as features_mod
    from emmy.compiler.pipeline.search.features import node_slices

    rows = [
        {"TILE": "f2x4", "WORK": "t32x8"},
        {"TILE": "f2x4", "WORK": "t32x8", "S_ext_free_prod": 2048.0, "H_sm_count": 128.0},
        {"TILE": "f2x4", "WORK": "t32x8", "S_masked_m": 1.0, "S_ext_free_prod": 2048.0, "H_sm_count": 84.0},
        {"TILE": "mma_m16n8k16_f16_f32/f2x2/k2", "WORK": "w2x2", "S_ext_free_prod": 2048.0 * 2048.0, "H_sm_count": 128.0},
        {"TILE": "mma_m16n8k16_f16_f32/f1x1", "REDUCE": "coop", "STAGE": "d2/smem-async", "WORK": "w4x2"},
        {"STAGE": "d3/smem-tma"},
        {
            "TILE@map.1/inner": "f2",
            "TILE@map.2/inner": "f4",
            "REDUCE": "coop",
            "WORK": "t32x8",
            "S_ext_free_prod": 64.0,
            "H_sm_count": 128.0,
        },
        {"TILE": "f2x4", "WORK": "t32x8", "BN": 64, "unrelated": 3},
        {"S_n_load": 3.0, "S_ext_free_prod": 512.0},
    ]
    features_mod._GEOMETRY_MEMO.clear()
    memoized = [knob_features(r) for r in rows]
    repeat = [knob_features(r) for r in rows]  # second pass = all cache hits
    assert memoized == repeat
    # Slice completeness: an ALL-BARE row's single slice must featurize identically to the whole
    # row dict — the geometry block reads nothing the slice projects away.
    for r in rows:
        slices = node_slices(r)
        if len(slices) == 1:
            assert dict(features_mod._schedule_node_features(slices[0])) == dict(features_mod._schedule_node_features(r))
    # Memoized blocks are read-only: a consumer mutating a shared block must raise, not corrupt.
    sl = node_slices(rows[0])[0]
    block = features_mod._node_geometry(sl)
    with pytest.raises(TypeError):
        block["D_threads"] = 0.0  # type: ignore[index]
    # The key must separate rows differing only in a structural extent the block reads.
    a = knob_features({"TILE": "f2x4", "WORK": "t32x8", "S_ext_free_prod": 512.0, "H_sm_count": 128.0})
    b = knob_features({"TILE": "f2x4", "WORK": "t32x8", "S_ext_free_prod": 2048.0, "H_sm_count": 128.0})
    assert any(a.get(k) != b.get(k) for k in a if k.startswith("D_"))


def test_evidence_row_vouches_reads_a_bare_row_key_like_a_bare_pin() -> None:
    """A bare row key against a candidate that spells the family per site reads as the bare pin it
    was measured under: some site carries the value and every other site is OFF."""
    from emmy.compiler.pipeline.knob import evidence_row_vouches

    row = {"WORK": "t32", "REDUCE": "coop"}
    assert evidence_row_vouches({"WORK": "t32", "REDUCE@map.1/twist": "coop", "REDUCE@map.1/twist.1/inner": ""}, row)
    assert not evidence_row_vouches({"WORK": "t32", "REDUCE@map.1/twist": "coop/r2", "REDUCE@map.1/twist.1/inner": ""}, row)
    assert not evidence_row_vouches({"WORK": "t32", "REDUCE@map.1/twist": "", "REDUCE@map.1/twist.1/inner": ""}, row), "no site carries it"
    assert evidence_row_vouches({"WORK": "t32", "REDUCE": "coop", "TILE": "f1"}, row), "an exact key compares exactly; TILE is free"
    assert not evidence_row_vouches({"WORK": "t32", "REDUCE": "r4"}, row)
