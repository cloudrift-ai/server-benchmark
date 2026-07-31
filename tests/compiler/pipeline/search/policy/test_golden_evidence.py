"""The deploy-side GOLDEN evidence tier (``policy/greedy._golden_pick`` /
``_golden_evidence_index``) — the card's recorded goldens decide a greedy compile
before the reservoir / DB tiers and the model. Goldens are the only measured data
that ships with a clone (the reservoir and tune DB are machine-local caches), so
this tier is what a fresh machine deploys from. Consulted, never trained on.

Three sections against the one tier, sharing the q_proj-shaped fixtures below:

- **The tier itself** — index construction, shape/regime/card scoping, and the pick,
  both as a unit and end-to-end through ``greedy_decide``.
- **The prior-less floor** — the two paths that resolve WITHOUT a prior and previously
  skipped the tier entirely: a failed ``load_prior`` (corrupt/unreadable online
  checkpoint), and ``Pipeline.run``'s last-resort emission-order resolve after the
  validity-retry budget exhausts (where one over-budget node used to cost every OTHER
  kernel its verified golden). The prior-loaded path implements the floor by tier order.
- **The live resolve** — the same tier through the REAL greedy resolve: trace the golden
  kind's snippet, run the full ``TILE_PASSES`` resolve with ``greedy_decide``, and assert
  the golden actually decided the fork. The unit sections join against hand-built stamp
  dicts; this one exists because those passed while a live deploy missed (the flash fork's
  root op is the tile pass's RESTRUCTURED twisted op whose knobs carry re-derived extents
  only — no histogram, no ``S_loop_depth`` — so a key classified off final-op stamps never
  matched at the fork: the 4090 evidence-tier verification, attention goldens silently
  skipped at 3.7-4.2x deploys).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from emmy import config
from emmy.compiler.context import Context
from emmy.compiler.pipeline.search import golden as golden_mod
from emmy.compiler.pipeline.search.golden import MatmulGoldenConfig
from emmy.compiler.pipeline.search.policy.greedy import (
    _fork_shape_key,
    _golden_evidence_index,
    _golden_pick,
    greedy_decide,
    tile_identity,
)

CARD = "NVIDIA GeForce RTX 4090"
CAP = (8, 9)

# q_proj-shaped fp16 matmul: M512 x N4096, K3840 — static stamps (S_dtype_f32 absent -> warp tier).
_SIG = {"S_ext_free_prod": 2097152.0, "S_ext_free_max": 4096.0, "S_ext_reduce_max": 3840.0, "S_ext_reduce_prod": 3840.0}
_BASE = {**_SIG, "H_opt": 3.0}

_STD_TILE = "a:mma_m16n8k16_f16_f32/w2x2/f4x4/k2"
_FM_TILE = "a:mma_m16n8k16_f16_f16/w2x2/f4x8/k2"
_W1X1_TILE = "a:mma_m16n8k16_f16_f32/w1x1/f1x1"


def _golden(name="gemma4_12b.q_proj", knobs=None, us=78.0, *, dynamic=False, gpu=CARD, cap=CAP):
    return MatmulGoldenConfig(
        name=name,
        M=512,
        N=4096,
        K=3840,
        dtype="fp16",
        dynamic=dynamic,
        knobs=knobs or {"TILE": _STD_TILE, "STAGE": "d2/cp/ring/p2", "REDUCE": "", "WSPEC": "", "RASTER": ""},
        emmy_us=us,
        gpu_name=gpu,
        compute_cap=cap,
    )


def _index(monkeypatch, goldens, gpu=CARD):
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", list(goldens))
    return _golden_evidence_index(Context.from_target(CAP, gpu_name=gpu))


def _rows(*tunings: dict, base: dict = _BASE) -> list[dict]:
    return [{**base, **t} for t in tunings]


# Candidate rows spell the axis-stamped realization (``TILE@a2``) — the golden's bare
# spelling must match it through ``pin_key_matches`` + ``values_equal``.
_ROW_W1X1 = {"TILE@a2": _W1X1_TILE, "STAGE@a2": "d2/cp/ring", "REDUCE@a2": ""}
_ROW_GOLD = {"TILE@a2": _STD_TILE, "STAGE@a2": "d2/cp/ring/p2", "REDUCE@a2": ""}
_ROW_FM = {"TILE@a2": _FM_TILE, "STAGE@a2": "d2/cp/ring", "REDUCE@a2": "g2k"}


# ---------------------------------------------------------------------------
# The tier: index, scoping, and the pick
# ---------------------------------------------------------------------------


def test_golden_pick_matches_bare_spelling_against_axis_stamped_row(monkeypatch):
    index = _index(monkeypatch, [_golden()])
    got = _golden_pick(index, _rows(_ROW_W1X1, _ROW_GOLD), "n0")
    assert got == (1, 78.0)


def test_golden_pick_regime_guard(monkeypatch):
    """Golden µs is deployable (-O3) truth — it never arbitrates an -O1 compile."""
    index = _index(monkeypatch, [_golden()])
    o1 = [{**r, "H_opt": 1.0} for r in _rows(_ROW_W1X1, _ROW_GOLD)]
    assert _golden_pick(index, o1, "n0") is None


def test_unrealizable_golden_warns_and_returns_none(monkeypatch, caplog):
    """A shape match whose golden realizes against NO offered candidate is enumeration
    drift — loud warning, fall-through (never a silent degraded deploy)."""
    index = _index(monkeypatch, [_golden()])
    with caplog.at_level(logging.WARNING):
        got = _golden_pick(index, _rows(_ROW_W1X1), "n0")
    assert got is None
    assert any("no longer realize" in r.message for r in caplog.records)
    # A shape with no golden at all stays silent — extrapolation is expected there.
    caplog.clear()
    cold = _rows({**_ROW_W1X1, "S_ext_free_prod": 99.0})
    with caplog.at_level(logging.WARNING):
        assert _golden_pick(index, cold, "n0") is None
    assert not caplog.records


def test_card_scoping(monkeypatch):
    other_card = _golden(gpu="NVIDIA GeForce RTX 5090", cap=(12, 0))
    assert _index(monkeypatch, [other_card]) == {}
    # No card identity on the ctx (off-GPU pure-logic run) -> no consultation.
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", [_golden()])
    assert _golden_evidence_index(Context.from_target(CAP)) == {}


def test_static_and_dynamic_twins_do_not_cross(monkeypatch):
    index = _index(monkeypatch, [_golden()])  # static golden
    dyn_base = {**_BASE, "S_ext_free_prod": 4096.0, "S_ext_n_symbolic_axis": 1.0}
    assert _golden_pick(index, _rows(_ROW_GOLD, base=dyn_base), "n0") is None
    dyn_index = _index(monkeypatch, [_golden(name="gemma4_12b.q_proj.dynM", dynamic=True)])
    assert _golden_pick(dyn_index, _rows(_ROW_GOLD, base=dyn_base), "n0") == (0, 78.0)
    assert _golden_pick(dyn_index, _rows(_ROW_GOLD), "n0") is None


def test_fast_math_golden_self_excludes_when_atom_not_offered(monkeypatch):
    """The fm entry records a faster µs, but its f16-accumulate atom is only in the
    offer when the fm gate is on — canonical TILE comparison must NOT collapse the
    two atoms, so a gate-off deploy falls to the std entry."""
    fm = _golden(knobs={"TILE": _FM_TILE, "STAGE": "d2/cp/ring", "REDUCE": "g2k", "WSPEC": "", "RASTER": ""}, us=61.5)
    index = _index(monkeypatch, [_golden(), fm])
    # Gate off: only f32-acc atoms offered -> the std golden decides despite fm being faster.
    assert _golden_pick(index, _rows(_ROW_W1X1, _ROW_GOLD), "n0") == (1, 78.0)
    # Gate on: the fm atom is offered -> the faster fm entry decides.
    assert _golden_pick(index, _rows(_ROW_GOLD, _ROW_FM), "n0") == (1, 61.5)


class _FakeFP(SimpleNamespace):
    score = None


def _decide_once(prior, monkeypatch, goldens, leaves):
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", list(goldens))
    fp = _FakeFP(
        ctx=Context.from_target(CAP, gpu_name=CARD),
        options=list(leaves),
        root_op=SimpleNamespace(knobs=dict(_SIG)),
        node_id="n0",
        match=None,
    )
    # ``Context.from_target`` doesn't stamp H_opt; ensure the deployable regime.
    feats = fp.ctx.features()
    if float(feats.get("H_opt", 3.0)) != 3.0:
        pytest.skip("deploy ctx not in the -O3 regime on this host")
    return greedy_decide(prior=prior)(fp), fp


def test_decide_golden_overrides_model_argmin(monkeypatch):
    """End-to-end through ``greedy_decide``: the model prefers the w1x1 leaf; the
    golden decides the pick and its measured µs lands on ``fp.score``."""
    leaves = [SimpleNamespace(knobs=dict(_ROW_W1X1)), SimpleNamespace(knobs=dict(_ROW_GOLD))]
    calls: list[str] = []

    class SpyPrior:
        def pick(self, rows):
            calls.append("pick")
            return 0, 1.0  # the model's argmin: the degenerate w1x1 leaf

        def evidence_pick(self, rows):
            calls.append("evidence_pick")
            return None

        def mean_scores(self, rows):
            calls.append("mean_scores")
            return [1.0] * len(rows)

        def add_rows(self, *a, **k):  # pragma: no cover - the assertion is that it's never hit
            calls.append("add_rows")

    picked, fp = _decide_once(SpyPrior(), monkeypatch, [_golden()], leaves)
    assert picked is leaves[1]
    assert fp.score == 78.0
    # The golden decided: no training-side method was touched, and neither
    # evidence tier nor the model was needed (test 8: no contamination).
    assert "add_rows" not in calls
    assert "pick" not in calls and "evidence_pick" not in calls


def test_decide_golden_beats_reservoir_evidence(monkeypatch):
    """Precedence: a golden outranks a reservoir -O3 evidence row for the same op —
    it is a verified, reproduced measurement; the reservoir row is one tune sample."""
    leaves = [SimpleNamespace(knobs=dict(_ROW_W1X1)), SimpleNamespace(knobs=dict(_ROW_GOLD))]

    class EvidencePrior:
        def pick(self, rows):
            return 0, 1.0

        def evidence_pick(self, rows):
            return 0, 50.0  # the reservoir would deploy the w1x1 leaf

        def mean_scores(self, rows):
            return [1.0] * len(rows)

    picked, fp = _decide_once(EvidencePrior(), monkeypatch, [_golden()], leaves)
    assert picked is leaves[1] and fp.score == 78.0


def test_decide_without_goldens_is_unchanged(monkeypatch):
    leaves = [SimpleNamespace(knobs=dict(_ROW_W1X1)), SimpleNamespace(knobs=dict(_ROW_GOLD))]

    class ModelPrior:
        def pick(self, rows):
            return 0, 42.0

        def evidence_pick(self, rows):
            return None

        def mean_scores(self, rows):
            return [1.0] * len(rows)

    picked, fp = _decide_once(ModelPrior(), monkeypatch, [], leaves)
    assert picked is leaves[0] and fp.score == 42.0


def test_decide_golden_applies_to_bare_mean_scores_priors(monkeypatch):
    leaves = [SimpleNamespace(knobs=dict(_ROW_W1X1)), SimpleNamespace(knobs=dict(_ROW_GOLD))]

    class BarePrior:
        def mean_scores(self, rows):
            return [0.0, 1.0]  # prefers w1x1

    picked, fp = _decide_once(BarePrior(), monkeypatch, [_golden()], leaves)
    assert picked is leaves[1] and fp.score == 78.0


def test_tune_path_never_routes_through_the_deploy_pick():
    """The golden tier lives inside ``greedy_decide`` (the deploy path). The tune's
    selection (``TuningSearch`` / the two-level driver) must not route through it —
    MCTS explores every sibling and benches; goldens must not steer the search."""
    import inspect

    from emmy.compiler.pipeline.search import two_level
    from emmy.compiler.pipeline.search.policy import mcts

    src = inspect.getsource(two_level) + inspect.getsource(mcts)
    assert "greedy_decide" not in src and "_golden_pick" not in src


# --- attention / norm kinds (the ShapeKey ``kind`` extension) -----------------------------------
# Stamp bases measured off the real traced ops (see test_shape_key_kinds): the flash op is a sweep
# (S_loop_depth < n_free + n_reduce + n_sym) with exp + 3 free loops -> kind="flash"; RMSNorm sweeps
# with rsqrt -> kind="rms_norm". Fork leaves captured off the live flash enumeration key
# TILE@dd + TILE@pj (+ REDUCE@kv / STAGE@kv) for BOTH the static and masked forms.

from emmy.compiler.pipeline.search.golden import AttentionGoldenConfig, RmsNormGoldenConfig  # noqa: E402

_FLASH_SIG = {
    "S_ext_free_prod": 2097152.0,
    "S_ext_reduce_max": 512.0,
    "S_ext_n_free_axis": 3.0,
    "S_ext_n_reduce_axis": 3.0,
    "S_ext_n_symbolic_axis": 0.0,
    "S_loop_depth": 4.0,
    "S_pw_exp": 2.0,
    "S_n_free_loop": 3.0,
    "H_opt": 3.0,
}
_FLASH_DYN_SIG = {
    "S_ext_free_prod": 4096.0,
    "S_ext_reduce_max": 0.0,
    "S_ext_n_free_axis": 2.0,
    "S_ext_n_reduce_axis": 0.0,
    "S_ext_n_symbolic_axis": 4.0,
    "S_loop_depth": 4.0,
    "S_pw_exp": 2.0,
    "S_n_free_loop": 3.0,
    "H_opt": 3.0,
}
_RMS_SIG = {
    "S_ext_free_prod": 1966080.0,
    "S_ext_reduce_max": 3840.0,
    "S_ext_n_free_axis": 2.0,
    "S_ext_n_reduce_axis": 1.0,
    "S_ext_n_symbolic_axis": 0.0,
    "S_loop_depth": 2.0,
    "S_pw_rsqrt": 1.0,
    "S_dtype_f32": 5.0,
    "H_opt": 3.0,
}

# Real recorded spellings (rtx4090 golden files): the known-hang w2x1 form vs the golden w4x1
# form, in the legacy embedded-worker grammar (loudly-validated pin aliases); the realized fork
# ROWS spell the site grammar — site TILE halves + the ONE WORK entry.
_DD_W4X1 = "a:mma_m16n8k16_f16_f32/w4x1/f1x2/k16"
_PJ_W4X1 = "a:mma_m16n8k16_f16_f32/w4x1/f1x32"
_PJ_FM = "a:mma_m16n8k16_f16_f16/w4x1/f1x32"
_DD_W4X1_SITE = "mma_m16n8k16_f16_f32/f1x2/k16"
_DD_W2X1_SITE = "mma_m16n8k16_f16_f32/f1x8/k16"
_PJ_SITE = "mma_m16n8k16_f16_f32/f1x32"
_PJ_FM_SITE = "mma_m16n8k16_f16_f16/f1x32"


def _flash_row(dd, pj, work, stage="d2/cp/ring", base=None):
    return {**(base or _FLASH_SIG), "TILE@dd": dd, "TILE@pj": pj, "REDUCE@kv": "", "STAGE@kv": stage, "WORK": work}


def _attention(name="gemma4_12b.attention.hd256", knobs=None, us=44.3, *, dynamic=False):
    return AttentionGoldenConfig(
        name=name,
        n_heads=16,
        seq=512,
        head_dim=256,
        dtype="fp16",
        dynamic=dynamic,
        knobs=knobs or {"TILE@dd": _DD_W4X1, "TILE@pj": _PJ_W4X1, "STAGE": "d2/cp/ring"},
        emmy_us=us,
        gpu_name=CARD,
        compute_cap=CAP,
    )


def test_attention_static_golden_decides_and_pins_are_all_or_nothing(monkeypatch):
    """The hang-class scenario: the fork offers the known-bad w2x1 form and the golden's
    w4x1 form — the golden must decide. And the static dd+pj contract is all-or-nothing:
    a row matching dd but not pj is a DIFFERENT form and must not be picked."""
    index = _index(monkeypatch, [_attention()])
    rows = [
        _flash_row(_DD_W2X1_SITE, _PJ_SITE, "w2x1"),  # the cold-greedy hang pick from the gemma seeding report
        _flash_row(_DD_W4X1_SITE, "mma_m16n8k16_f16_f32/f1x16", "w4x1"),  # dd matches, pj (regs) does not -> not the golden's form
        _flash_row(_DD_W4X1_SITE, _PJ_SITE, "w4x1"),
    ]
    assert _golden_pick(index, rows, "n0") == (2, 44.3)


def test_attention_dynM_bare_plan_golden_matches_axis_keyed_leaves(monkeypatch):
    """A dynamic attention golden is schema-required to record ONE bare TILE (the pin
    contract), but the masked-flash fork's leaves are axis-keyed dd+pj like the static
    form — the bare plan must be satisfied by ANY same-family realization (the
    pin-resolution semantics), here the dd plan, spelled with the atom ALIAS the YAML
    records (canonical value comparison)."""
    gold = _attention(
        name="attention.hd256.dynM",
        dynamic=True,
        us=40.4,
        knobs={"TILE": "a:mma_m16n8k16_f16/w4x1/f1x2/k16", "STAGE": "d2/cp/ring"},
    )
    index = _index(monkeypatch, [gold])
    rows = [
        _flash_row(_DD_W2X1_SITE, _PJ_SITE, "w2x1", base=_FLASH_DYN_SIG),
        _flash_row(_DD_W4X1_SITE, _PJ_SITE, "w4x1", base=_FLASH_DYN_SIG),
    ]
    assert _golden_pick(index, rows, "n0") == (1, 40.4)
    # The static twin's index never decides the masked op and vice versa.
    assert _golden_pick(_index(monkeypatch, [_attention()]), rows, "n0") is None


def test_attention_fm_pv_plan_golden_self_excludes_gate_off(monkeypatch):
    """A fast-math dynM golden records the sibling PV plan as its bare TILE; with the fm
    gate off no leaf realizes the f16-accumulate pj, so it self-excludes and the std
    entry decides — gate on (fm pj offered), the faster fm entry wins via ANY-axis."""
    std = _attention(
        name="attention.hd256.dynM", dynamic=True, us=44.3, knobs={"TILE": "a:mma_m16n8k16_f16/w4x1/f1x2/k16", "STAGE": "d2/cp/ring"}
    )
    fm = _attention(name="attention.hd256.dynM", dynamic=True, us=36.1, knobs={"TILE": _PJ_FM, "STAGE": "d2/cp/ring"})
    index = _index(monkeypatch, [std, fm])
    gate_off = [_flash_row(_DD_W4X1_SITE, _PJ_SITE, "w4x1", base=_FLASH_DYN_SIG)]
    assert _golden_pick(index, gate_off, "n0") == (0, 44.3)
    gate_on = gate_off + [_flash_row(_DD_W4X1_SITE, _PJ_FM_SITE, "w4x1", base=_FLASH_DYN_SIG)]
    assert _golden_pick(index, gate_on, "n0") == (1, 36.1)


def test_rms_norm_golden_decides_its_op(monkeypatch):
    gold = RmsNormGoldenConfig(name="rms_norm.k3840", M=512, K=3840, knobs={"REDUCE": "b256"}, emmy_us=6.3, gpu_name=CARD, compute_cap=CAP)
    index = _index(monkeypatch, [gold])
    rows = [{**_RMS_SIG, "REDUCE@r0": "b32"}, {**_RMS_SIG, "REDUCE@r0": "b256"}]
    assert _golden_pick(index, rows, "n0") == (1, 6.3)


# The computed-A norm→linear cone stamped on its PRE-SPLIT geometry (real stamps off pre32.json's
# in-model norm→q fork, M32 x N4096, K3840): mixed f16 operands + the f32 statistic constant over an
# add-reduce, the statistic reduce NOT yet lifted (n_reduce_axis == 1) and the rsqrt buried in the A
# sub-body, so the histogram classifier reads a plain scalar matmul (kind="", is_warp=False from the
# f32 signal, nonzero free_max).
_CONE_SIG = {
    "S_ext_free_prod": 131072.0,
    "S_ext_free_max": 4096.0,
    "S_ext_reduce_max": 3840.0,
    "S_ext_reduce_prod": 3840.0,
    "S_ext_n_free_axis": 2.0,
    "S_ext_n_reduce_axis": 1.0,
    "S_loop_depth": 3.0,
    "S_reduce_add": 1.0,
    "S_dtype_f16": 1.0,
    "S_dtype_f32": 1.0,
    "H_opt": 3.0,
}
# The fork ROW additionally carries the cone's unmistakable OFFER: a ``d*/sync`` STAGE candidate —
# the mandatory compute-fill only a computed-A contraction enumerates (``stage_moves`` spells only
# gmem-direct / cp / tma). ``_fork_shape_key`` keys the rebuild on this, not on the dtype stamps.
_CONE_ROW = {**_CONE_SIG, "STAGE@a2": "d1/sync"}


def test_computed_a_cone_fork_rebuilds_to_the_fused_key():
    """The pre-split norm→linear cone stamps like a plain scalar matmul, so its raw key disagrees with
    the fused golden on is_warp and kind. ``_fork_shape_key`` must rebuild it to the fused convention
    (flash-analogous) so the norm→qkv cones join their goldens at cold deploy, CARRYING the stamped
    ``free_max`` through — a fused op is a plain two-free-axis contraction, so both builders agree on
    the aspect and the key must keep it (see the collision test below).
    (``_norm_linear`` / ``_FUSED_SIG`` below are the POST-split fused op the classifier already kinds;
    this is the PRE-split fork the classifier misses.)"""
    from emmy.compiler.pipeline.search.golden import NormLinearGoldenConfig

    fused_key = NormLinearGoldenConfig(name="nl", M=32, H=3840, N=4096, knobs={}).shape_key()
    key = _fork_shape_key([_CONE_ROW])
    assert key == fused_key
    assert key.kind == "fused" and key.is_warp and key.free_max == 4096
    # A MIXED-DTYPE PLAIN MATMUL — the same stamp combination (f16 operands + an f32 bias/residual
    # load over an add-reduce) but no ``d*/sync`` STAGE among its offers — is untouched: the offer
    # signal cannot over-fire on dtype coincidences, so the kernel keeps joining its ``kind=""``
    # goldens (the earlier dtype sniff silently re-keyed it out of them).
    plain = _fork_shape_key([{**_CONE_SIG, "STAGE@a2": "d2/cp/ring"}])
    assert plain.kind == "" and plain.free_max == 4096
    # And ``cp.async``'s substring never false-positives the segment match.
    plain_cp = _fork_shape_key([{**_CONE_SIG, "STAGE@a2": "d2/cp.async/ring"}])
    assert plain_cp.kind == ""


def test_fused_key_separates_aspect_equal_free_prod_cones():
    """Two gemma-4 computed-A cones collide on every other key field: the M=32 LOCAL norm→q
    (32x4096) and the M=256 GLOBAL norm→kv (256x512) share free_prod=131072 AND reduce_max=3840.
    Without the aspect the global prefill cone silently deployed the local decode cone's golden and
    reported its 24.1 us. ``free_max`` must keep them apart on BOTH builders."""
    from emmy.compiler.pipeline.search.golden import NormLinearGoldenConfig

    local_q = NormLinearGoldenConfig(name="q", M=32, H=3840, N=4096, knobs={}).shape_key()
    global_kv = NormLinearGoldenConfig(name="kv", M=256, H=3840, N=512, knobs={}).shape_key()
    assert local_q.free_prod == global_kv.free_prod == 131072
    assert local_q.reduce_max == global_kv.reduce_max == 3840
    assert local_q != global_kv and (local_q.free_max, global_kv.free_max) == (4096, 512)

    # The fork side must agree with the golden side, or the join breaks instead of separating.
    assert _fork_shape_key([{**_CONE_ROW, "S_ext_free_prod": 131072.0, "S_ext_free_max": 512.0}]) == global_kv


def test_dynamic_fused_key_normalizes_the_aspect_away():
    """A symbolic-M cone has no static aspect — ``free_max`` must normalize to 0 so the dynamic twin
    keeps joining its golden (the static/dynamic split stays the only discriminator)."""
    from emmy.compiler.pipeline.search.golden import NormLinearGoldenConfig

    dyn = NormLinearGoldenConfig(name="q", M=512, H=3840, N=4096, knobs={}, dynamic=True).shape_key()  # M = DEFAULT_SEQ_HINT
    assert dyn.is_dyn and dyn.free_max == 0
    # bf16 operands rebuild identically — the offer-keyed rebuild is dtype-blind.
    bf16 = _fork_shape_key([{**{k: v for k, v in _CONE_ROW.items() if k != "S_dtype_f16"}, "S_dtype_bf16": 1.0}])
    assert bf16.kind == "fused" and bf16.is_warp
    # The dynamic cone keeps is_dyn and its contraction reduce_max (unlike the flash reduce, which zeroes).
    dyn = _fork_shape_key([{**_CONE_ROW, "S_ext_free_prod": 4096.0, "S_ext_n_symbolic_axis": 1.0}])
    assert dyn.kind == "fused" and dyn.is_dyn and dyn.reduce_max == 3840


def test_computed_a_cone_fork_deploys_the_fused_golden(monkeypatch):
    """End-to-end on the PRE-split fork: with the fused norm→linear golden in the index, cone rows
    (``n_reduce_axis == 1``, no top-level rsqrt) resolve it via the ``_fork_shape_key`` rebuild — a
    MATCH, not a DRIFT — where ``from_s_features`` alone would key them as a plain matmul and miss."""
    index = _index(monkeypatch, [_norm_linear()])  # records _FUSED_TILE / d2/sync / g4k
    row = {**_CONE_SIG, "TILE@a": _FUSED_TILE, "STAGE@a": "d2/sync", "REDUCE@a": "g4k"}
    assert _golden_pick(index, [row], "n0") == (0, 34.8)


def test_cross_kind_isolation_via_the_kind_discriminator(monkeypatch):
    """An attention golden and a matmul golden with NUMERICALLY IDENTICAL extents never
    cross: the flash op's sweep stamps classify kind="flash", the contraction's stamps
    kind="" — each key joins only its own kind."""
    # Matmul golden whose extents collide with the flash op's (free 2097152, K 512, warp).
    collide = _golden(name="matmul.sq512ish", knobs={"TILE": _STD_TILE, "STAGE": "d2/cp/ring"})
    object.__setattr__(collide, "K", 512)  # frozen dataclass; forge the colliding reduce extent
    index = _index(monkeypatch, [collide])
    flash_rows = [_flash_row(_DD_W4X1_SITE, _PJ_SITE, "w4x1")]
    assert _golden_pick(index, flash_rows, "n0") is None  # matmul golden never decides a flash op
    # And the attention golden never decides a matmul op of the same extents.
    a_index = _index(monkeypatch, [_attention()])
    mm_rows = _rows(
        _ROW_GOLD, base={**_BASE, "S_ext_reduce_max": 512.0, "S_ext_n_free_axis": 2.0, "S_ext_n_reduce_axis": 1.0, "S_loop_depth": 3.0}
    )
    assert _golden_pick(a_index, mm_rows, "n0") is None


# --- fused computed-A (norm_linear) kind ---------------------------------------------------------
# Stamp base captured off the real deploy fork of ``F.rms_norm(x,(H,),nw) @ w`` (M32 x N4096, K3840):
# a sweep (S_loop_depth 3 < n_free 2 + n_reduce 2) with rsqrt AND a SECOND reduce axis (the
# contraction) -> kind="fused". S_dtype_f32 present (the real 12B model's f32 statistic constants):
# is_warp is FORCED True for the kind (a computed-A contraction is a warp mma). The fork leaves are
# axis-keyed TILE@a3 + STAGE@a3 + REDUCE@a3 over the ``sync`` compute-fill (d1/d2/sync, not tma/ring).

from emmy.compiler.pipeline.search.golden import NormLinearGoldenConfig  # noqa: E402

_FUSED_SIG = {
    "S_ext_free_prod": 131072.0,
    "S_ext_free_max": 4096.0,
    "S_ext_reduce_max": 3840.0,
    "S_ext_reduce_prod": 14745600.0,
    "S_ext_n_free_axis": 2.0,
    "S_ext_n_reduce_axis": 2.0,
    "S_ext_n_symbolic_axis": 0.0,
    "S_loop_depth": 3.0,
    "S_pw_rsqrt": 1.0,
    "S_dtype_f16": 6.0,
    "S_dtype_f32": 2.0,
    "H_opt": 3.0,
}
_FUSED_TILE = "a:mma_m16n8k16_f16_f32/w1x8/f2x2/k2"


def _fused_row(tile, stage="d2/sync", reduce="g4k"):
    return {**_FUSED_SIG, "TILE@a3": tile, "STAGE@a3": stage, "REDUCE@a3": reduce, "RASTER": ""}


def _norm_linear(name="gemma4_12b.q_proj_fused", knobs=None, us=34.8):
    return NormLinearGoldenConfig(
        name=name,
        M=32,
        H=3840,
        N=4096,
        dtype="fp16",
        knobs=knobs or {"TILE": _FUSED_TILE, "STAGE": "d2/sync", "REDUCE": "g4k", "RASTER": "", "WSPEC": ""},
        emmy_us=us,
        gpu_name=CARD,
        compute_cap=CAP,
    )


def test_fused_golden_decides_its_op_over_a_warp_sibling(monkeypatch):
    """The fused norm→linear golden decides its computed-A op: the recorded bare knobs match the
    axis-keyed warp leaf (``TILE@a3`` etc.) over an offered sibling on a different tile."""
    index = _index(monkeypatch, [_norm_linear()])
    rows = [_fused_row("a:mma_m16n8k16_f16_f32/w1x1/f1x1", stage="d1/sync", reduce=""), _fused_row(_FUSED_TILE)]
    assert _golden_pick(index, rows, "n0") == (1, 34.8)


def test_fused_golden_never_crosses_rms_norm_or_matmul(monkeypatch):
    """The fused key (``kind="fused"``) never joins a bare RMSNorm sweep nor a plain matmul, even at
    coincident extents — the ``S_ext_n_reduce_axis>=2`` discriminator keeps the families apart."""
    # A fused golden must not decide a bare rms_norm op (one reduce axis, kind="rms_norm").
    index = _index(monkeypatch, [_norm_linear()])
    rms_rows = [{**_RMS_SIG, "REDUCE@r0": "b256"}]
    assert _golden_pick(index, rms_rows, "n0") is None
    # A fused golden must not decide a plain matmul op (no rsqrt, kind="").
    mm_rows = _rows(_ROW_GOLD)
    assert _golden_pick(index, mm_rows, "n0") is None
    # And a bare rms_norm golden must not decide the fused op.
    rms_gold = RmsNormGoldenConfig(name="rms.k3840", M=512, K=3840, knobs={"REDUCE": "b256"}, emmy_us=6.3, gpu_name=CARD, compute_cap=CAP)
    assert _golden_pick(_index(monkeypatch, [rms_gold]), [_fused_row(_FUSED_TILE)], "n0") is None


# ---------------------------------------------------------------------------
# The prior-less floor (``greedy_decide`` with ``prior=None``)
# ---------------------------------------------------------------------------
# The card's recorded goldens are tier-1 evidence independent of any prior, so a fork
# whose golden still realizes deploys the golden — never option-0.


def _decide_no_prior(monkeypatch, goldens, leaves, blocked=None):
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", list(goldens))
    with config.nvcc_flags_override(""):  # the deployable -O3 regime (H_opt=3) golden consultation is gated on
        ctx = Context.from_target(CAP, gpu_name=CARD)
    fp = _FakeFP(ctx=ctx, options=list(leaves), root_op=SimpleNamespace(knobs=dict(_SIG)), node_id="n0", match=None)
    return greedy_decide(blocked, prior=None)(fp), fp


def test_no_prior_resolve_deploys_realizable_golden(monkeypatch, caplog):
    """A resolve with no prior at all must still land on the recorded golden when it
    realizes — previously this path fell straight to option-0 (the degenerate w1x1
    leaf here), silently missing the verified entry. The override is logged loud."""
    leaves = [SimpleNamespace(knobs=dict(_ROW_W1X1)), SimpleNamespace(knobs=dict(_ROW_GOLD))]
    with caplog.at_level(logging.WARNING):
        picked, fp = _decide_no_prior(monkeypatch, [_golden()], leaves)
    assert picked is leaves[1]
    assert fp.score == 78.0
    assert any("golden floor" in r.message for r in caplog.records)


def test_no_prior_drift_warns_and_falls_to_option0(monkeypatch, caplog):
    """A golden keyed to the fork's shape that NO offered candidate realizes is
    enumeration drift — the loud drift warning fires on this path too (it was silent
    here before the floor: the tier was never consulted), then emission order."""
    leaves = [SimpleNamespace(knobs=dict(_ROW_W1X1))]
    with caplog.at_level(logging.WARNING):
        picked, fp = _decide_no_prior(monkeypatch, [_golden()], leaves)
    assert picked is leaves[0]
    assert fp.score is None
    assert any("no longer realize" in r.message for r in caplog.records)


def test_no_prior_without_goldens_is_pure_emission_order(monkeypatch, caplog):
    """No golden for the live card ⇒ the old behavior bit-for-bit: option-0, no
    warnings (parity with ``test_resolve.test_option0_decide_matches_no_prior_greedy``)."""
    other_card = _golden(gpu="NVIDIA GeForce RTX 5090", cap=(12, 0))
    leaves = [SimpleNamespace(knobs=dict(_ROW_W1X1)), SimpleNamespace(knobs=dict(_ROW_GOLD))]
    with caplog.at_level(logging.WARNING):
        picked, fp = _decide_no_prior(monkeypatch, [other_card], leaves)
    assert picked is leaves[0]
    assert fp.score is None
    assert not caplog.records


def test_floor_respects_blocklist(monkeypatch, caplog):
    """``Pipeline.run``'s last-resort resolve threads ``blocked`` through, so the
    floor can never re-pick a tile that already failed ``validate(ctx)`` — with the
    golden's realization blocklisted the shape reads as drift and option-0 stands."""
    blocked = {"n0": {tile_identity(dict(_ROW_GOLD))}}
    leaves = [SimpleNamespace(knobs=dict(_ROW_W1X1)), SimpleNamespace(knobs=dict(_ROW_GOLD))]
    with caplog.at_level(logging.WARNING):
        picked, fp = _decide_no_prior(monkeypatch, [_golden()], leaves, blocked=blocked)
    assert picked is leaves[0]
    assert fp.score is None
    assert any("no longer realize" in r.message for r in caplog.records)


def test_audit_sink_records_unrealized_pin_only_entries(monkeypatch):
    """Under ``golden_audit`` the MATCH record carries ``unrealized`` — the shape's recorded
    entries NO offered candidate realizes (the ``eval golden`` offer audit's pin-only
    signal). The deploy outcome is unchanged: the realizable sibling still floors the pick.
    A shape whose entries are ALL unrealized records DRIFT with every entry listed."""
    from emmy.compiler.pipeline.search.policy.greedy import golden_audit

    # A STAGE these leaves never offer, at the fastest µs → consulted first; must not decide.
    pin_only = _golden(knobs={"TILE": _STD_TILE, "STAGE": "d4/tma/ring"}, us=50.0)
    leaves = [SimpleNamespace(knobs=dict(_ROW_W1X1)), SimpleNamespace(knobs=dict(_ROW_GOLD))]
    records: list[dict] = []
    with golden_audit(records):
        picked, fp = _decide_no_prior(monkeypatch, [pin_only, _golden()], leaves)
    assert picked is leaves[1], "the realizable sibling floors the deploy exactly as before"
    assert fp.score == 78.0
    (rec,) = records
    assert rec["verdict"] == "MATCH"
    assert rec["unrealized"] == [pin_only]

    records.clear()
    with golden_audit(records):
        picked, _ = _decide_no_prior(monkeypatch, [pin_only], leaves)
    assert picked is leaves[0]  # nothing realizes → option-0
    (rec,) = records
    assert rec["verdict"] == "DRIFT"
    assert rec["unrealized"] == [pin_only]


def test_fork_shape_key_flash_sniff_scans_every_row():
    """The flash OFFER signal (the ``TILE@dd`` + ``TILE@pj`` pair) marks the fork when
    ANY row carries it — row 0 is whichever leaf the planner emitted first and need
    not be a warp leaf. A row-0-only sniff mis-keyed such a fork ``kind=""`` (GAP), and
    the attention goldens were silently skipped in favor of the model. The sig is the
    RESTRUCTURED twisted op's: re-derived extents only, no histogram — so the stamp
    classifier cannot fire and the key depends on the offer sniff alone."""
    flash_sig = {
        "S_ext_free_prod": 2097152.0,
        "S_ext_reduce_max": 512.0,
        "S_ext_n_free_axis": 3.0,
        "S_ext_n_reduce_axis": 3.0,
        "S_ext_n_symbolic_axis": 0.0,
        "H_opt": 3.0,
    }
    scalar_row = {**flash_sig, "TILE": "b32x8/f1x1"}  # a bare-TILE non-warp sibling emitted first
    warp_row = {
        **flash_sig,
        "TILE@dd": "a:mma_m16n8k16_f16_f32/w4x1/f1x2/k16",
        "TILE@pj": "a:mma_m16n8k16_f16_f32/w4x1/f1x32",
        "REDUCE@kv": "",
        "STAGE@kv": "d2/cp/ring",
    }
    assert _fork_shape_key([warp_row, scalar_row]).kind == "flash"
    assert _fork_shape_key([scalar_row, warp_row]).kind == "flash", "the sniff must not depend on row order"


# ---------------------------------------------------------------------------
# The live resolve: does the tier fire on a real fork?
# ---------------------------------------------------------------------------
# Fixture knobs are read off the live offer itself, so enumeration drift moves the
# fixture instead of false-failing the join.

_SDPA = (
    "F.scaled_dot_product_attention(torch.randn(1,16,512,256,dtype=torch.float16), "
    "torch.randn(1,16,512,256,dtype=torch.float16), torch.randn(1,16,512,256,dtype=torch.float16), is_causal=True)"
)
_SDPA_DYN = ["seq_len@x0:2", "seq_len@x1:2", "seq_len@x2:2"]
_RMS = "torch.nn.RMSNorm(3840)(torch.randn(512,3840))"


class _StubPrior:
    """Bare-``mean_scores`` prior preferring emission order — any golden pick must
    override it, so a pass proves the tier (not the model) decided."""

    def mean_scores(self, rows):
        return list(range(len(rows)))


def _resolve(snippet: str, dyn: list[str]):
    """One full greedy resolve of the traced snippet under the DEPLOYABLE regime — the
    ctx's compile flags are forced empty (-O3): ``make test`` runs the -Xcicc -O1
    correctness lane, where the golden tier is correctly inert (the regime guard), so
    without this override the suite would exercise nothing. No nvcc runs here (the
    resolve stops at the tile dialect), so the override is purely a feature stamp."""
    from dataclasses import replace

    from emmy.commands.trace import graph_from_code
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline
    from emmy.compiler.pipeline.pipeline import Run
    from emmy.compiler.pipeline.search.policy import greedy as greedy_mod
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

    ds = build_torch_dynamic_shapes(parse_position_specs(dyn)) if dyn else None
    graph, _, _ = graph_from_code(snippet, dynamic_shapes=ds)
    ctx = replace(Context.from_target(CAP, gpu_name=CARD), compile_flags="")
    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(graph, greedy_mod.greedy_decide(prior=_StubPrior(), price_structural=False))
    return graph


def _spying(monkeypatch, rows_sink: list, picks_sink: list):
    from emmy.compiler.pipeline.search.policy import greedy as greedy_mod

    orig = greedy_mod._golden_pick

    def spy(index, rows, node_id):
        got = orig(index, rows, node_id)
        rows_sink.append(rows)
        picks_sink.append((node_id, got))
        return got

    monkeypatch.setattr(greedy_mod, "_golden_pick", spy)


def _offered_flash_form(rows: list[dict]) -> dict:
    """A non-degenerate offered flash form (skip the w1x1 emission-order head) to
    record as the test golden — read off the live enumeration, never hardcoded."""
    for r in rows:
        dd = r.get("TILE@dd", "")
        if "/w1x1/" not in dd and dd:
            return r
    return rows[0]


def _decoy():
    """An off-shape golden keeping the evidence index non-empty during the probe
    resolve — the tier (and so the spy) only runs when the card has recorded goldens."""
    return AttentionGoldenConfig(
        name="attention.hd64.decoy",
        n_heads=2,
        seq=512,
        head_dim=64,
        dtype="fp16",
        knobs={"TILE@dd": "a:mma_m16n8k16_f16_f32/w1x1/f1x2/k16"},
        emmy_us=1.0,
        gpu_name=CARD,
        compute_cap=CAP,
    )


@pytest.mark.parametrize("dyn", [[], _SDPA_DYN], ids=["static", "dynM"])
def test_attention_golden_decides_the_live_flash_fork(monkeypatch, dyn):
    # Probe resolve: capture the flash fork's offered rows (decoy keeps the tier live).
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", [_decoy()])
    rows_sink: list = []
    picks_sink: list = []
    _spying(monkeypatch, rows_sink, picks_sink)
    _resolve(_SDPA, dyn)
    flash_rows = next(r for r in rows_sink if any("TILE@dd" in row for row in r))
    form = _offered_flash_form(flash_rows)
    # Record that form as the golden: static keeps the axis-keyed all-or-nothing pins;
    # the dynamic twin is schema-required to record ONE bare TILE (the dd plan).
    knobs = (
        {"TILE": form["TILE@dd"], "STAGE": form.get("STAGE@kv", "")}
        if dyn
        else {"TILE@dd": form["TILE@dd"], "TILE@pj": form["TILE@pj"], "STAGE": form.get("STAGE@kv", "")}
    )
    gold = AttentionGoldenConfig(
        name="attention.hd256.test",
        n_heads=16,
        seq=512,
        head_dim=256,
        dtype="fp16",
        dynamic=bool(dyn),
        knobs=knobs,
        emmy_us=44.3,
        gpu_name=CARD,
        compute_cap=CAP,
    )
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", [gold])
    rows_sink.clear()
    picks_sink.clear()
    graph = _resolve(_SDPA, dyn)
    hits = [(n, got) for n, got in picks_sink if got is not None]
    assert hits and hits[0][1][1] == 44.3, f"attention golden never decided a fork: {picks_sink}"
    # The resolved graph realized the golden's dd plan on its flash kernel.
    realized = [
        (getattr(n.op, "knobs", {}) or {}).get("TILE@dd") for n in graph.nodes.values() if "TILE@dd" in (getattr(n.op, "knobs", {}) or {})
    ]
    assert form["TILE@dd"] in realized


def test_rms_norm_golden_decides_the_live_reduce_fork(monkeypatch):
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", [_decoy()])
    rows_sink: list = []
    picks_sink: list = []
    _spying(monkeypatch, rows_sink, picks_sink)
    _resolve(_RMS, [])
    reduce_rows = next((r for r in rows_sink if any(k.startswith("REDUCE@") for row in r for k in row)), None)
    if reduce_rows is None:
        pytest.skip("the rms_norm lowering offers no REDUCE fork on this pipeline")
    form = reduce_rows[-1]  # any non-head row: the stub prior prefers emission order
    fam_key = next(k for k in form if k.startswith("REDUCE@"))
    gold = RmsNormGoldenConfig(
        name="rms_norm.k3840.test",
        M=512,
        K=3840,
        knobs={"REDUCE": form[fam_key]},
        emmy_us=6.3,
        gpu_name=CARD,
        compute_cap=CAP,
    )
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", [gold])
    picks_sink.clear()
    _resolve(_RMS, [])
    assert any(got is not None and got[1] == 6.3 for _, got in picks_sink), f"rms_norm golden never decided: {picks_sink}"
