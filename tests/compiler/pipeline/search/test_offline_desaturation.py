"""The offline prior's proxy must never saturate over live qualities, and the golden
rank must be tie-pessimistic.

The retired ``±80`` QUALITY clip inside the exp squash sat in the middle of the live
range (at scale 0.1 thousands of good warp tiles score past 80): every such row
collapsed onto one ``exp(-8)`` plateau, greedy's argmin fell through to emission
order (the ``w1x1`` gemma misdeploys, 12-29x off the golden on a real 4090), and
``evaluate_golden``'s strictly-greater rank reported 0 for every tied row — the
"prior finds the goldens" illusion. The clip now guards only the exp argument's
float-safety bound, and the rank counts earlier-emitted ties against the golden.
No GPU.
"""

from __future__ import annotations

import math

from emmy.compiler.pipeline.search import golden_eval
from emmy.compiler.pipeline.search.prior import OfflinePrior
from emmy.compiler.pipeline.search.prior.fallback import FallbackPrior

_GATES_OFF = {
    "atomic_free_split_threshold": 0.0,
    "atomic_free_weight": 0.0,
    "scalar_on_warp_weight": 0.0,
    "splitk_roundtrip_weight": 0.0,
}


def _prior(scale: float = 0.1) -> OfflinePrior:
    """A single-weight prior: quality == the row's ``D_q`` value."""
    return OfflinePrior(weights={"D_q": 1.0}, weights_dynamic={"D_q": 1.0}, scale=scale, **_GATES_OFF)


def test_qualities_past_the_old_clip_stay_strictly_ordered():
    """Qualities 85 and 120 both sat past the retired ±80 clip and scored the identical
    exp(-8); they must now rank strictly (higher quality → lower latency proxy)."""
    ap = _prior()
    ok, better = ap.mean_score_features({"D_q": 85.0}), ap.mean_score_features({"D_q": 120.0})
    assert better < ok, "quality 120 must outrank quality 85 — the old clip tied them"
    assert ap.mean_score_features({"D_q": 90.0}) == ap.mean_score_features({"D_q": 90.0})


def test_exp_argument_clips_at_the_float_safety_bound_only():
    """Absurd qualities stay finite (the clip's only remaining job) and the bound sits
    outside anything a real pool produces."""
    ap = _prior()
    assert ap.mean_score_features({"D_q": 1e9}) == math.exp(-700.0)
    assert ap.mean_score_features({"D_q": -1e9}) == math.exp(700.0)
    # just inside the bound: still strictly ordered
    assert ap.mean_score_features({"D_q": 6900.0}) < ap.mean_score_features({"D_q": 6800.0})


def test_fallback_tilt_multiplier_stays_bounded(monkeypatch):
    """The trusted-online blend must not feed the unsaturated proxy (up to e**±700)
    straight into the µs product — the multiplier clamps to e**±8."""
    from emmy import config

    class _Online:
        trustworthy = True

        def score(self, knobs):  # noqa: ARG002 — fixed µs anchor
            return 100.0

    class _Offline:
        def __init__(self, proxy):
            self.proxy = proxy

        def score(self, knobs):  # noqa: ARG002 — an extreme unsaturated proxy
            return self.proxy

    monkeypatch.setattr(config, "offline_tilt", lambda: 1.0)
    lo = FallbackPrior(_Online(), _Offline(math.exp(-700.0))).score({})
    hi = FallbackPrior(_Online(), _Offline(math.exp(700.0))).score({})
    assert lo == 100.0 * math.exp(-8.0), "multiplier must clamp at e**-8, not vanish to ~0"
    assert hi == 100.0 * math.exp(8.0), "multiplier must clamp at e**+8, not blow up the µs anchor"


def test_evaluate_golden_rank_is_tie_pessimistic(monkeypatch):
    """A golden tied with earlier-emitted rows loses the greedy argmin to them — the
    rank must count those ties, so a tie plateau can never report rank 0."""
    rows = [
        {"TILE@a0": "n16x8/f2x4"},
        {"TILE@a0": "n32x8/f4x8"},
        {"TILE@a0": "n32x16/f4x8"},  # the golden, emitted third
        {"TILE@a0": "n64x16/f4x8"},
    ]
    monkeypatch.setattr(golden_eval, "_enumerate", lambda M, N, K, dtype, ctx: (rows, ()))
    golden = {"TILE": "n32x16/f4x8"}

    _, rank_tied, pool, rank_opt = golden_eval.evaluate_golden(1, 1, 1, "fp16", golden, ctx=None, scorer=lambda r: 1.0)
    assert pool == 4
    assert rank_tied == 2, "two earlier-emitted ties must count against the golden"
    assert rank_opt == 0, "the optimistic count forgives the whole tie plateau"
    assert rank_tied - rank_opt == 2, "the pessimistic-optimistic gap IS the earlier-tie plateau width"

    def favors_golden(r):
        return 2.0 if r is rows[2] else 1.0

    _, rank_best, _, rank_best_opt = golden_eval.evaluate_golden(1, 1, 1, "fp16", golden, ctx=None, scorer=favors_golden)
    assert rank_best == 0, "a strict winner still ranks 0"
    assert rank_best_opt == 0, "no plateau, no gap"
