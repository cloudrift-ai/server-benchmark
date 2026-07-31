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

_WARP_SPEC = "a:mma_m16n8k16_f16_f32/w1x1/f1x2/k4"
# The realized rows spell SITE values + the ONE ``WORK`` entry (F1); the recorded/legacy pin
# above keeps its embedded-worker spelling and must still select among them.
_WARP_SITE = "mma_m16n8k16_f16_f32/f1x2/k4"


def _forms():
    warp = SimpleNamespace(knobs={"TILE@dd": _WARP_SITE, "TILE@pj": _WARP_SITE, "WORK": "w1x1"}, tag="warp")
    chain = SimpleNamespace(knobs={"TILE@dd": "", "TILE@pj": "f64", "WORK": ""}, tag="chain")
    cell = SimpleNamespace(knobs={"TILE@dd": "", "TILE@pj": "", "WORK": ""}, tag="cell")
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


def test_keyed_only_ignores_the_bare_pin(monkeypatch):
    # ``keyed_only`` is the warp-rows caller's mode: a bare warp pin was already consumed by
    # ``_twisted_warp_options`` and must not be re-matched against the DERIVED pj spelling
    # (it can never equal both axes' spellings — their ``k<bk>`` differ).
    monkeypatch.setenv("EMMY_TILE", _WARP_SPEC)
    assert _tags(_narrow_flash_forms(_forms(), _HEAD, _PV, keyed_only=True)) == ["warp", "chain", "cell"]


def test_keyed_only_still_selects_by_axis_pins(monkeypatch):
    monkeypatch.setenv("EMMY_TILE@DD", _WARP_SPEC)
    monkeypatch.setenv("EMMY_TILE@PJ", _WARP_SPEC)
    assert _tags(_narrow_flash_forms(_forms(), _HEAD, _PV, keyed_only=True)) == ["warp"]


def test_stage_pin_does_not_bypass_keyed_tile_pins(monkeypatch):
    """The findings-5 F2 regression: a static attention golden pins ``TILE@dd``/``TILE@pj`` AND
    ``STAGE``; the dispatch's stage-pinned early return (mma rows alone) used to skip the
    narrowing, so the golden A/B benched the prior's tile under the pinned stage. The full
    trace→resolve path must stamp exactly the pinned pair."""
    import pytest

    torch = pytest.importorskip("torch")  # noqa: F841

    from emmy.commands.run import _pinned_knobs
    from emmy.commands.trace import trace_inline_code
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline
    from emmy.compiler.pipeline.fork import Fork
    from emmy.compiler.pipeline.pipeline import Run

    d = trace_inline_code(
        "F.scaled_dot_product_attention(torch.randn(1,2,128,64,dtype=torch.float16), "
        "torch.randn(1,2,128,64,dtype=torch.float16), torch.randn(1,2,128,64,dtype=torch.float16), is_causal=False)"
    )
    g = d["graph"] if isinstance(d, dict) else d
    pins = {
        "TILE@dd": "a:mma_m16n8k16_f16_f32/w2x1/f1x8/k4",
        "TILE@pj": "a:mma_m16n8k16_f16_f32/w2x1/f1x8/k4",
        "STAGE": "d2/cp/ring",
    }

    def decide(fp):
        o = fp.options[0]
        while isinstance(o, Fork) and not o.is_leaf:
            o = o.expand()[0]
        return o

    with _pinned_knobs(pins):
        out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context(compute_capability=(8, 9))).resolve(g, decide)
    stamped = next(k for _, n in out.nodes.items() if (k := getattr(n.op, "knobs", None)) and "TILE@dd" in k)
    # F1: the stamp is the SITE spelling; the pin's worker half lands in the ONE WORK entry.
    assert stamped["TILE@dd"] == "mma_m16n8k16_f16_f32/f1x8/k4", stamped
    assert stamped["TILE@pj"] == "mma_m16n8k16_f16_f32/f1x8/k4", stamped
    assert stamped["WORK"] == "w2x1", stamped
