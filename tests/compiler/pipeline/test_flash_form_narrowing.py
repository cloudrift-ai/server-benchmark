"""``_narrow_flash_forms`` — per-node ``TILE`` pins select flash fork rows by stamped spelling.

Regression: the fold dispatch used to route ANY live ``TILE`` pin to the warp rows alone, so no
pin could select the CHAIN row (the FA-2 shared-score scalar form) and a scalar pin degraded to
the per-cell tier — the 64×-redundant score recompute. The narrowing compares canonicalized
spellings (``a:scalar`` ≡ ``""``, ``f64x1`` ≡ ``f64``) per flash knob key; greedy prior choice
stays untouched when no pin is live, and an unmatched pin keeps the full fork (graceful degrade).
"""

from __future__ import annotations

from types import SimpleNamespace

from emmy.compiler.pipeline.passes.lowering.tile._schedule import _narrow_flash_forms

_WARP_SPEC = "a:mma_m16n8k16_f16/w1x1/f1x2/k4"


def _forms():
    warp = SimpleNamespace(knobs={"TILE@dd": _WARP_SPEC, "TILE@pj": _WARP_SPEC}, tag="warp")
    chain = SimpleNamespace(knobs={"TILE@dd": "", "TILE@pj": "f64"}, tag="chain")
    cell = SimpleNamespace(knobs={"TILE@dd": "", "TILE@pj": ""}, tag="cell")
    return [warp, chain, cell]


_HEAD = SimpleNamespace(k_axis=SimpleNamespace(name="dd"))
_PV = SimpleNamespace(k_axis=SimpleNamespace(name="pj"))


def _tags(forms):
    return [f.tag for f in forms]


def test_no_pin_keeps_the_fork(monkeypatch):
    monkeypatch.delenv("EMMY_TILE", raising=False)
    assert _tags(_narrow_flash_forms(_forms(), _HEAD, _PV)) == ["warp", "chain", "cell"]


def test_keyed_pin_selects_the_chain(monkeypatch):
    monkeypatch.setenv("EMMY_TILE@PJ", "f64")
    assert _tags(_narrow_flash_forms(_forms(), _HEAD, _PV)) == ["chain"]


def test_scalar_alias_pin_covers_the_head_key(monkeypatch):
    # The article's chain spelling: bare ``a:scalar`` (≡ "" on the score node) + keyed ``f<d>``.
    monkeypatch.setenv("EMMY_TILE", "a:scalar")
    monkeypatch.setenv("EMMY_TILE@PJ", "f64")
    assert _tags(_narrow_flash_forms(_forms(), _HEAD, _PV)) == ["chain"]


def test_bare_scalar_pin_keeps_the_per_cell_tier(monkeypatch):
    monkeypatch.setenv("EMMY_TILE", "a:scalar")
    assert _tags(_narrow_flash_forms(_forms(), _HEAD, _PV)) == ["cell"]


def test_alias_spelling_canonicalizes(monkeypatch):
    monkeypatch.setenv("EMMY_TILE@PJ", "f64x1")  # ≡ the stamped "f64"
    assert _tags(_narrow_flash_forms(_forms(), _HEAD, _PV)) == ["chain"]


def test_unmatched_pin_keeps_the_full_fork(monkeypatch):
    monkeypatch.setenv("EMMY_TILE@PJ", "f32")  # no row stamps f32
    assert _tags(_narrow_flash_forms(_forms(), _HEAD, _PV)) == ["warp", "chain", "cell"]
