"""The deploy-side GOLDEN evidence tier (``policy/greedy._golden_pick`` /
``_golden_evidence_index``) — the card's recorded goldens decide a greedy compile
before the reservoir / DB tiers and the model. Goldens are the only measured data
that ships with a clone (the reservoir and tune DB are machine-local caches), so
this tier is what a fresh machine deploys from. Consulted, never trained on."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search import golden as golden_mod
from emmy.compiler.pipeline.search.golden import MatmulGoldenConfig
from emmy.compiler.pipeline.search.policy.greedy import _golden_evidence_index, _golden_pick, greedy_decide

CARD = "NVIDIA GeForce RTX 4090"
CAP = (8, 9)

# q_proj-shaped fp16 matmul: M512 x N4096, K3840 — static stamps (S_dtype_f32 absent -> warp tier).
_SIG = {"S_ext_free_prod": 2097152.0, "S_ext_reduce_max": 3840.0, "S_ext_reduce_prod": 3840.0}
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
