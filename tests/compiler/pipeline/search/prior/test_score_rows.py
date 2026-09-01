"""``Prior.score_rows`` — the pool-shaped scoring surface every prior answers.

The dict surfaces (``mean_scores`` / ``mean_scores_features``) cannot be used on the pools a ranking question
is actually asked over: a matmul enumeration runs to ~10^5 rows and the per-row dict representation is what
made the fit OOM. This one takes the packed pool. What each implementation must get right is its OWN column
list and its OWN absent-value fill — which is why it cannot be one shared function — and one polarity, since
the fitted model classes compute quality (higher = faster) while the online model regresses latency.

``Prior.columns`` is the other half of that: each implementation PUBLISHES the complete column list it reads —
weights, the interaction's inputs and the routing stamp — so a pool builder can pack for the models it is about
to score instead of restating their columns or widening to every column the featurizer can spell.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from emmy.compiler.pipeline.search.data.group import GoldenGroup
from emmy.compiler.pipeline.search.features import ROUTING_FEATURES
from emmy.compiler.pipeline.search.prior.fallback import FallbackPrior
from emmy.compiler.pipeline.search.prior.linear_model import GATE_FEATURES, LinearModel, gate_columns, unweighted_cols
from emmy.compiler.pipeline.search.prior.offline import OfflinePrior
from emmy.compiler.pipeline.search.prior.online import OnlinePrior


def _group(feats: list[dict], *, tier: str = "thread") -> GoldenGroup:
    return GoldenGroup.from_dicts("card/k", "k", tier, "card", "k", 0, feats)


def _linear(**weights) -> OfflinePrior:
    return OfflinePrior(model=_model(dict(weights), dict(weights)))


def _model(weights, weights_dynamic, **params) -> LinearModel:
    """A model declaring exactly what it reads, the way a fit writes one."""
    return LinearModel(
        unweighted_cols=unweighted_cols(weights, weights_dynamic),
        weights=weights,
        weights_dynamic=weights_dynamic,
        scale=1.0,
        **{"atomic_free_weight": 0.0, "atomic_free_split_threshold": 4.0, **params},
    )


def _stamp(dynamic: float = 0.0) -> dict:
    """The stamps the featurizer writes on every row: the routing one — 0.0 when no axis is symbolic, never
    absent — and the regime, which ``ctx.features()`` always supplies and the prior's dataset requires."""
    return {"S_ext_n_symbolic_axis": dynamic, "H_opt": 3.0}


def test_the_linear_half_scores_a_whole_pool_by_its_own_weights():
    """Higher quality = predicted faster, and the model reads only the columns it has weights for — extra
    columns in the pool are ignored rather than being an error."""
    prior = _linear(D_a=2.0)
    scores = prior.score_rows(_group([{**_stamp(), "D_a": 1.0, "D_unused": 99.0}, {**_stamp(), "D_a": 3.0}]))

    assert scores.tolist() == [2.0, 6.0]


def test_a_narrower_feature_view_does_not_move_the_linear_halfs_ranks():
    """Why ``eval prior --dataset golden`` may build its pools over just the columns its halves declare, while
    ``emmy fit`` trains under a narrow ``D_*`` view, and both still report the same rank: the model projects the
    pool onto its own weight names, so every column outside them is inert."""
    prior = _linear(D_a=2.0)
    narrow = [{**_stamp(), "D_a": 1.0}, {**_stamp(), "D_a": 3.0}]
    wide = [{**row, "S_ext_free_prod": 4096.0, "H_opt": 3.0, "MMA_a_bits": 16.0} for row in narrow]

    assert prior.score_rows(_group(narrow)).tolist() == prior.score_rows(_group(wide)).tolist()


def test_the_linear_half_declares_every_column_it_reads():
    """:attr:`LinearModel.columns` must cover what ``score_rows`` reads under BOTH weight sets — a builder packs
    one pool without knowing which set it will route to. It also covers the interaction's two inputs whether or
    not a weight names them (a pruned zero weight drops the key, and ``gate_columns`` still reads it), and the
    routing stamp, which the model reads through the pool's own ``dynamic`` label rather than out of the matrix.

    The stamp is the one that used to be missing, and the reason the artifact stores an ``unweighted_cols``
    field at all: it is the part of the answer no weight key can spell."""
    model = _model(
        {"D_a": 1.0},  # no D_finalize_kernel / D_splitk weight — the pruned-gate case
        {"D_a": 1.0, "D_dyn_only": 2.0},
    )

    assert set(model.cols_for(False)) <= set(model.columns)
    assert set(model.cols_for(True)) <= set(model.columns)
    assert set(GATE_FEATURES) <= set(model.columns)
    assert set(ROUTING_FEATURES) <= set(model.columns)
    assert not set(ROUTING_FEATURES) & set(model.cols_for(True))  # it routes; it is not packed into the matmul
    assert "D_dyn_only" in model.columns  # the dynamic set's own column, declared even when scoring a static pool


def test_a_model_with_no_dynamic_weight_set_still_declares_its_static_columns():
    """An incomplete fit (no symbolic-axis cases) declines a dynamic pool but is still a usable static scorer,
    so its declaration is the static set rather than empty. It still declares the routing stamp: reading the
    stamp is how it knows to decline."""
    static_only = _model({"D_a": 1.0}, None)

    assert set(static_only.columns) == {"D_a", *GATE_FEATURES, *ROUTING_FEATURES}


def test_a_pool_that_cannot_route_is_refused_rather_than_scored_as_static():
    """The stamp is not optional. A pool packed under a view that dropped it arrives labelled static — every
    candidate would then be priced by the static weight set whatever its regime, and nothing downstream could
    tell that from a genuinely static pool. So the model refuses it.

    This is the check that replaces the feature view's old exemption of the stamp from every spec: the filter
    now keeps only what its spec names, and the model says so when a builder gets it wrong."""
    prior = _linear(D_a=2.0)
    unstamped = _group([{"D_a": 1.0}, {"D_a": 3.0}])  # no S_ext_n_symbolic_axis anywhere in the pool

    with pytest.raises(ValueError, match="routing stamp"):
        prior.score_rows(unstamped)


def test_the_interaction_is_computed_from_the_pool_not_from_a_default():
    """The atomic-free term is the one part of the score that is not a linear weight, and a per-name default
    used to let it vanish: an absent ``D_finalize_kernel`` read 0.0, which zeroes ``weight · finalize · (…)``
    for every row, while an absent ``D_splitk`` read 1.0, a split count no featurization produces.

    The pool's own values reach the term now — a split count above the threshold rewards the deferred combine
    kernel, one below penalizes it — and an unstamped column reads this model's one absent value, so a pool
    with no finalize kernel prices the term at the formula's own zero rather than at a substituted one."""
    prior = _linear(D_a=0.0)  # weights contribute nothing, so the score IS the interaction
    model = replace(prior.model, atomic_free_weight=3.0)
    rows = [{**_stamp(), "D_finalize_kernel": 1.0, "D_splitk": splitk} for splitk in (2.0, 8.0)]

    assert OfflinePrior(model=model).score_rows(_group(rows)).tolist() == [-3.0, 3.0]
    assert prior.score_rows(_group(rows)).tolist() == [0.0, 0.0]  # a zero interaction weight is a real zero
    # A pool that stamps neither input: 0.0 on both, which is the term's value at ``finalize == 0``.
    assert OfflinePrior(model=model).score_rows(_group([{**_stamp(), "D_a": 5.0}])).tolist() == [0.0]
    assert [float(c[0]) for c in gate_columns(np.zeros((1, 1)), ["D_a"])] == [0.0, 0.0]


def test_the_linear_half_declines_a_pool_it_fitted_no_weight_set_for():
    """``None``, not an exception and not a silent zero: an unfittable cross-validation fold and an
    all-static fit both hit this, and a report has to count the pools rather than average them in."""
    static_only = _model({"D_a": 1.0}, None)
    prior = OfflinePrior(model=static_only)

    assert prior.score_rows(_group([{**_stamp(), "D_a": 1.0}])) is not None
    assert prior.score_rows(_group([{**_stamp(1.0), "D_a": 1.0}], tier="dyn")) is None


def test_the_online_half_returns_quality_not_latency():
    """It regresses ``log(latency µs)``, so the pool-shaped surface has to invert the polarity — otherwise
    every rank the report computes off it is exactly backwards."""
    prior = OnlinePrior(iterations=40)
    # A single feature that IS the latency: BM=8 rows are slow, BM=64 rows are fast.
    prior.add_rows([({**_stamp(), "BM": 8.0}, 100.0)] * 40 + [({**_stamp(), "BM": 64.0}, 5.0)] * 40)
    prior.fit()

    scores = prior.score_rows(_group([{**_stamp(), "BM": 8.0}, {**_stamp(), "BM": 64.0}]))
    assert scores[1] > scores[0]  # the faster config scores HIGHER quality
    assert prior.mean_score({**_stamp(), "BM": 64.0}) < prior.mean_score({**_stamp(), "BM": 8.0})  # µs: lower is better


def test_the_online_half_declares_the_columns_it_regressed_on():
    """The tree indexes features by POSITION, so its column list is as load-bearing as a weight dict's keys.
    Empty until the first fit — an unfit half is dropped by a report rather than asked."""
    prior = OnlinePrior(iterations=40)
    assert prior.columns == ()

    prior.add_rows([({**_stamp(), "BM": 8.0}, 100.0)] * 40 + [({**_stamp(), "BM": 64.0}, 5.0)] * 40)
    prior.fit()

    assert prior.fitted and "BM" in prior.columns


def test_an_unfit_online_half_answers_a_constant_rather_than_failing():
    """A cold checkpoint must render as a model with no ranking ability, not crash a report."""
    assert OnlinePrior().score_rows(_group([{"D_a": 1.0}, {"D_a": 2.0}])).tolist() == [0.0, 0.0]


def test_the_online_half_waits_for_label_variation_before_fitting():
    """One equal-latency batch cannot rank anything; keep it pending until another measurement adds information."""
    prior = OnlinePrior(iterations=5, min_rows=2, refit_every=1)
    prior.add_rows([(_stamp() | {"BM": 8.0}, 10.0)] * 2)

    assert not prior.maybe_refit()
    assert not prior.fitted

    prior.add_rows([(_stamp() | {"BM": 64.0}, 5.0)])

    assert prior.maybe_refit()
    assert prior.fitted


def test_the_composite_answers_with_the_half_that_owns_deploys():
    """``FallbackPrior`` forwards to whichever half is live, exactly as its other scoring surfaces do — so a
    report over the deployed prior decomposes the model that actually makes decisions."""
    offline = _linear(D_a=2.0)
    composite = FallbackPrior(OnlinePrior(), offline)  # online unfit → not trustworthy → offline owns deploys
    group = _group([{**_stamp(), "D_a": 1.0}, {**_stamp(), "D_a": 3.0}])

    assert not composite.trustworthy
    assert np.array_equal(composite.score_rows(group), offline.score_rows(group))


def test_the_composite_declares_the_columns_of_the_half_that_owns_deploys():
    """``columns`` must agree with ``score_rows`` about WHICH half answers. ``__getattr__`` forwards to the
    online half, so leaving this to it would declare one half's columns and score with the other's."""
    offline = _linear(D_a=2.0)
    composite = FallbackPrior(OnlinePrior(), offline)  # online unfit → not trustworthy → offline owns deploys

    assert composite.columns == offline.columns
