"""The blend strategies — how the online and offline halves interact.

Two questions per strategy, and they are answered separately: which half owns the deploy ranking, and what
PUCT explores by. The default ``tilt`` combines the halves only in the second, where both have already been
normalized within the sibling set; ``gate`` is its null hypothesis; ``online`` / ``offline`` are the A/B arms
that ignore the calibration gate on purpose.
"""

from __future__ import annotations

import math

import pytest

from emmy import config
from emmy.compiler.pipeline.search.prior.base import normalize_policy
from emmy.compiler.pipeline.search.prior.blend import BLENDS, load_blend
from emmy.compiler.pipeline.search.prior.fallback import FallbackPrior


class _Half:
    """A prior stub with fixed per-sibling answers (latency-like, lower is better)."""

    def __init__(self, scores, *, trustworthy: bool = True):
        self.scores = scores
        self.trustworthy = trustworthy
        self.fitted = True

    def mean_score(self, knobs):  # noqa: ARG002 — positional answers, not keyed by knobs
        return self.scores[0]

    def mean_scores(self, rows):
        return self.scores[: len(rows)]

    def policy(self, rows):
        return normalize_policy(self.mean_scores(rows))


ROWS = [{}, {}, {}]


# --- normalize_policy, the shared primitive ---------------------------------------


def test_policy_prices_siblings_against_the_best():
    """The best sibling is exactly 1.0 and one priced 10x slower is 0.1 — a scale PUCT can compare against
    ``Q in [0, 1]``, and one that does not shrink as a fork gets wider (which sum-normalization would)."""
    assert normalize_policy([10.0, 100.0, 20.0]) == [1.0, 0.1, 0.5]
    assert max(normalize_policy([3.0, 7.0])) == 1.0


def test_policy_is_uniform_when_the_model_has_no_opinion():
    """A cold model predicts 0.0 for everything; PUCT must fall back to uniform exploration rather than
    dividing by it. A single unknown sibling among known ones stays optimistic (1.0)."""
    assert normalize_policy([0.0, 0.0]) == [1.0, 1.0]
    assert normalize_policy([]) == []
    assert normalize_policy([10.0, 0.0, 20.0]) == [1.0, 1.0, 0.5]


def test_policy_survives_the_full_proxy_range():
    """The offline proxy spans e**±700. Sibling-relative normalization cannot saturate, so extremes still
    order — the property the retired e**±8 clamp destroyed on 255 of 261 goldens."""
    pol = normalize_policy([math.exp(-700.0), 1.0, math.exp(700.0)])
    assert pol[0] == 1.0 and pol[0] > pol[1] > pol[2]


# --- the strategies ---------------------------------------------------------------


def test_every_strategy_is_registered_under_its_own_name():
    assert set(BLENDS) == {"tilt", "gate", "online", "offline"}
    assert all(name == b.name for name, b in BLENDS.items())


def test_unknown_blend_is_a_hard_error():
    """A silently-defaulted arm would report the default's numbers under the arm's label."""
    with pytest.raises(ValueError, match="unknown prior blend"):
        load_blend("geometric")


def test_blend_is_selected_by_env(monkeypatch):
    monkeypatch.setenv("EMMY_PRIOR_BLEND", "gate")
    assert load_blend().name == "gate"
    monkeypatch.delenv("EMMY_PRIOR_BLEND")
    assert load_blend().name == "tilt"


@pytest.mark.parametrize("blend", ["tilt", "gate"])
def test_deploy_falls_back_to_offline_while_quarantined(blend):
    """Both default strategies keep the calibration gate: a fitted-but-mis-calibrated online model does not
    own the deploy ranking."""
    online, offline = _Half([10.0], trustworthy=True), _Half([99.0])
    fb = FallbackPrior(online, offline, blend=load_blend(blend))
    assert fb.mean_score({}) == 10.0
    online.trustworthy = False
    assert fb.mean_score({}) == 99.0


@pytest.mark.parametrize(("blend", "trusted", "expected"), [("online", False, 10.0), ("offline", True, 99.0)])
def test_single_half_arms_ignore_the_calibration_gate(blend, trusted, expected):
    """The A/B arms answer from one half regardless of trust — that is what isolates a change to that half.
    Neither is a deploy default: they discard the guard on purpose."""
    fb = FallbackPrior(_Half([10.0], trustworthy=trusted), _Half([99.0]), blend=load_blend(blend))
    assert fb.mean_score({}) == expected


def test_gate_does_not_combine_the_halves():
    """``gate``'s policy is the live half's alone — the null hypothesis the tilt is measured against."""
    online, offline = _Half([10.0, 20.0, 40.0]), _Half([40.0, 20.0, 10.0])
    fb = FallbackPrior(online, offline, blend=load_blend("gate"))
    assert fb.policy(ROWS) == online.policy(ROWS)


def test_tilt_nudges_the_online_order_without_replacing_it(monkeypatch):
    """The point of the tilt: a sibling the cold heuristic likes and the online model has buried gets pulled
    up, but the online model still sets the order's shape."""
    online, offline = _Half([10.0, 11.0, 12.0]), _Half([100.0, 100.0, 1.0])  # offline loves the last sibling
    fb = FallbackPrior(online, offline, blend=load_blend("tilt"))
    monkeypatch.setattr(config, "offline_tilt", lambda: 0.3)
    tilted = fb.policy(ROWS)
    plain = online.policy(ROWS)
    assert tilted[2] / tilted[0] > plain[2] / plain[0], "the heuristic's favorite must gain ground"
    assert max(tilted) == 1.0, "the result is renormalized, so PUCT's scale against Q is preserved"


def test_tilt_at_zero_weight_is_pure_online(monkeypatch):
    monkeypatch.setattr(config, "offline_tilt", lambda: 0.0)
    online, offline = _Half([10.0, 20.0, 40.0]), _Half([40.0, 20.0, 10.0])
    fb = FallbackPrior(online, offline, blend=load_blend("tilt"))
    assert fb.policy(ROWS) == online.policy(ROWS)


def test_tilt_uses_the_offline_half_alone_while_cold(monkeypatch):
    monkeypatch.setattr(config, "offline_tilt", lambda: 0.3)
    online, offline = _Half([10.0, 20.0, 40.0], trustworthy=False), _Half([40.0, 20.0, 10.0])
    fb = FallbackPrior(online, offline, blend=load_blend("tilt"))
    assert fb.policy(ROWS) == offline.policy(ROWS)
