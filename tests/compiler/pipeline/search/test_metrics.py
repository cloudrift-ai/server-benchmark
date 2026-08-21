"""The pure metric layer — ranks, regret and rank correlation over synthetic vectors.

No model, no dataset, no fixtures: every case here is a handful of numbers chosen so the right answer is
obvious by inspection. That is the point of the module being pure — the tie conventions and the guards are
the load-bearing part, and they are far easier to pin on six numbers than on a fitted model's output.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.pipeline.search.metrics import (
    best_dual_rank,
    best_rank,
    dual_rank,
    rank_of_golden,
    spearman,
    topk_pick,
    topk_regret,
)

# --- the rank family ---------------------------------------------------------------


def test_rank_of_golden_counts_ties_against_the_golden():
    """A tie is a miss: greedy breaks score ties by emission order, so the model gets no credit for
    a candidate it expressed no preference over."""
    scores = np.array([5.0, 3.0, 3.0, 1.0])
    assert rank_of_golden(scores, 0) == 0
    assert rank_of_golden(scores, 1) == rank_of_golden(scores, 2) == 2  # both tied rows, both charged
    assert rank_of_golden(np.array([1.0, 1.0, 1.0]), 0) == 2  # a flat scorer ranks LAST, not first


def test_dual_rank_separates_the_tie_plateau_from_the_rows_genuinely_ahead():
    """``rank`` counts ties emitted EARLIER (what greedy actually deploys ahead); ``rank_optimistic``
    counts only strictly-better rows. The gap between them is the plateau width — a saturated scorer
    reads top-1 on the optimistic count while deploying an emission-order pick."""
    scores = [2.0, 1.0, 1.0, 1.0, 3.0]
    assert dual_rank(scores, 2) == (3, 2)  # two rows strictly better, plus the one tie emitted ahead of it
    assert dual_rank(scores, 3) == (4, 2)  # same two ahead, but now two ties precede it
    assert dual_rank(scores, 4) == (0, 0)  # the strict winner has no plateau
    assert dual_rank([1.0, 1.0, 1.0], 0) == (0, 0)  # emitted first, wins its own plateau
    assert dual_rank([1.0, 1.0, 1.0], 2) == (2, 0)  # emitted last, the whole plateau deploys first


def test_best_rank_is_the_single_golden_function_at_one_positive():
    """The set forms generalize the single-index ones without moving any number — which is what kept a
    one-positive fit byte-identical when multi-positive pools arrived."""
    scores = np.array([4.0, 1.0, 3.0, 2.0])
    for i in range(4):
        assert best_rank(scores, (i,)) == rank_of_golden(scores, i)
        assert best_dual_rank(scores, (i,)) == dual_rank(scores, i)


def test_best_rank_takes_the_minimum_because_deploy_ships_one_config():
    """Any acceptable config ranked first is a win, so the per-group term is the best over the
    positives — a mean would spend weights pushing up a runner-up that never ships."""
    scores = np.array([9.0, 1.0, 8.0, 7.0, 10.0])
    assert (rank_of_golden(scores, 1), rank_of_golden(scores, 4)) == (4, 0)  # worst and best row of the pool
    assert best_rank(scores, (1, 4)) == 0
    assert best_dual_rank(scores, (1, 4)) == dual_rank(scores, 4) == (0, 0)
    # Positives that TIE on the objective resolve to the earliest-emitted one — the row greedy would deploy,
    # so the pessimistic count is the smaller of the two rather than an arbitrary pick.
    tied = np.array([1.0, 1.0, 1.0])
    assert best_rank(tied, (1, 2)) == rank_of_golden(tied, 1) == 2
    assert best_dual_rank(tied, (1, 2)) == dual_rank(tied, 1) == (1, 0)


# --- the regret family -------------------------------------------------------------


def test_topk_pick_resolves_a_tie_to_the_worst_measured_candidate():
    """The model said these two were equal, so which one ships is decided by emission order — something
    it neither knows nor controls. Crediting it with the lucky one reports skill it does not have."""
    assert topk_pick([1.0, 1.0, 2.0], [5.0, 9.0, 1.0], 1) == 1  # tied at the top: the slower one
    assert topk_pick([3.0, 1.0, 2.0], [5.0, 9.0, 1.0], 1) == 1  # no tie: the argmin, plainly


def test_top1_regret_is_the_deploy_question():
    """At ``k=1`` the pick ships, so its measured latency IS the cost, as a ratio to the achievable best."""
    assert topk_regret([3.0, 1.0, 2.0], [50.0, 20.0, 10.0], 1) == pytest.approx(2.0)  # picks 20µs, best is 10
    assert topk_regret([1.0, 2.0, 3.0], [10.0, 20.0, 50.0], 1) == pytest.approx(1.0)  # recovers the optimum


def test_topk_regret_widens_the_frontier():
    """Larger ``k`` asks the tuning question: bench the model's top k, keep the best. The ratio is a view
    of the pick, so the two never disagree about which candidate is being priced."""
    pred, meas = [1.0, 2.0, 3.0, 4.0], [90.0, 30.0, 10.0, 80.0]
    for k, expected in ((1, 9.0), (2, 3.0), (3, 1.0)):  # the optimum enters the frontier at k=3
        assert topk_regret(pred, meas, k) == pytest.approx(expected)
        assert meas[topk_pick(pred, meas, k)] / min(meas) == pytest.approx(expected)


def test_topk_regret_refuses_a_group_it_cannot_discriminate():
    """The guard that matters most. A pool of 5 scored at ``k=10`` has its whole self as its top 10, so
    it returns a perfect 1.00 for a reason that has nothing to do with the model — and averaged into a
    report it is indistinguishable from a model that genuinely found the optimum. At the measurement
    freeze's median group size of 5, that describes most of the corpus."""
    pred, meas = [1.0, 2.0, 3.0], [30.0, 20.0, 10.0]
    assert topk_regret(pred, meas, 3) is None  # k == n: the frontier is the pool
    assert topk_regret(pred, meas, 10) is None
    assert topk_regret(pred, meas, 2) is not None
    assert topk_pick(pred, meas, 3) is None  # the pick refuses on the same condition, so they cannot diverge


def test_topk_regret_refuses_a_label_that_is_not_a_latency():
    """A non-positive best would divide into nonsense. A FAILED bench is deliberately NOT this case: it
    carries the positive ``1e9`` sentinel and divides perfectly well, so excluding failures stays the
    caller's status filter rather than something this function can guess at."""
    assert topk_regret([1.0, 2.0], [0.0, 5.0], 1) is None
    assert topk_regret([1.0, 2.0], [1e9, 1e9], 1) == pytest.approx(1.0)  # a ratio, and a meaningless one


def test_the_predicted_side_is_read_ordinally_so_a_rank_trained_model_works():
    """The offline prior is fitted on RANKS: its output is an ordinal quality with no latency meaning, while
    the online prior regresses µs. Both must be answerable here, and they are — the regret family sorts by
    the prediction and takes every value it reports from the MEASUREMENTS, and Spearman is a rank
    correlation. So any monotone-decreasing-in-goodness transform is the same input.

    Pinning it because the alternative is a metric that silently means something different depending on
    which prior produced the vector."""
    rng = np.random.default_rng(4)
    pred = rng.normal(0, 30, 25).tolist()
    meas = rng.uniform(1.0, 500.0, 25).tolist()

    for transform in (lambda v: [x * 7.5 for x in v], lambda v: [x + 1000 for x in v], np.exp):
        same = list(transform(pred))  # strictly increasing: a different scale, the same ORDER
        assert topk_pick(same, meas, 1) == topk_pick(pred, meas, 1)
        assert topk_regret(same, meas, 5) == pytest.approx(topk_regret(pred, meas, 5))
        assert spearman(same, meas) == pytest.approx(spearman(pred, meas))


# --- correlation -------------------------------------------------------------------


def test_spearman_is_positive_when_the_model_orders_the_pool_correctly():
    """Both arguments are latency-like, so a model that agrees with the hardware scores ``+1``. Passing
    a ranking-quality vector here instead would silently invert the sign, which is why the parameters
    are named for their units."""
    assert spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)
    assert spearman([1.0, 2.0, 3.0, 4.0], [10.0, 30.0, 20.0, 40.0]) == pytest.approx(0.8)


def test_spearman_returns_none_where_rho_is_undefined():
    """A constant prediction vector is the feature/label vocabulary collapse the calibration gate exists
    to catch, and it must read as UNMEASURED rather than as ``0.0`` — a real correlation of zero is a
    model that ranks randomly, which is a different finding."""
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert spearman([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) is None
    assert spearman([1.0], [2.0]) is None


def test_spearman_and_regret_answer_different_questions():
    """A model that nails the top pick and shuffles the tail keeps its regret and loses its ρ. Reporting
    only one of them hides half of what a prior can get wrong."""
    pred = [1.0, 2.0, 3.0, 4.0, 5.0]
    meas = [10.0, 90.0, 50.0, 80.0, 60.0]  # best first, then noise
    assert topk_regret(pred, meas, 1) == pytest.approx(1.0)  # perfect where it counts for deploy
    assert spearman(pred, meas) < 0.6  # ...and largely uninformative about the rest of the pool
