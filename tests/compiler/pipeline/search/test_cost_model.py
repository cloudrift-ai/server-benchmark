"""Unit tests for the fitted best-latency estimate and its artifact.

CPU-only. The corpus cases fit a real model over the golden rows, which is seconds rather than
minutes because the dataset is a thousand rows wide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from emmy.compiler.pipeline.search import cost_model
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION


@dataclass(frozen=True)
class _Row:
    """The surface the model reads: a feature dict carrying its roofline, plus a label and the two
    grouping keys."""

    features: dict[str, float]
    best_us: float
    fold: str = "f"
    gpu: str = "CARD"
    kind: str = ""
    members: int = 1


def _synthetic(n: int = 240) -> list[_Row]:
    """Rows whose ratio-to-roofline is a clean function of one feature, so a fit that works must
    recover it and a fit that ignores its input cannot."""
    rows = []
    for i in range(n):
        roofline = 1.0 + (i % 17)
        over = 1.0 + (i % 5)  # the signal: ratio depends on this feature alone
        rows.append(
            _Row(
                features={"R_log_roofline_us": math.log(roofline), "signal": float(over), "noise": float(i % 3)},
                best_us=roofline * over,
                fold=f"g{i % 12}",
                # Two cards with enough rows each that the leave-one-card-out branch runs; without
                # this the headline diagnostic of the module would have no coverage at all.
                gpu="CARD_A" if i % 2 else "CARD_B",
            )
        )
    return rows


# --- what the model predicts ------------------------------------------------------------------


def test_predicts_microseconds_not_a_ratio():
    """The whole point of the design: a structural fork sums these against measured microseconds
    from the evidence tiers, so the model has to hand back microseconds rather than any normalized
    quantity a caller would have to un-normalize."""
    rows = _synthetic()
    predicted = cost_model.fit(rows).predict_us(rows)
    assert all(p > 0 for p in predicted)
    # Every label here is roofline x (1..5), so a prediction in ratio space would land under 6.
    assert max(predicted) > 10.0
    assert math.isclose(sum(predicted) / len(predicted), sum(r.best_us for r in rows) / len(rows), rel_tol=0.25)


def test_the_fit_beats_scaling_the_roofline():
    """The only claim that makes the model worth having: it must beat the roofline scaled by one
    constant, which is the strongest predictor that never looks at the kernel.

    Compared against :func:`cross_val_constant` rather than against the raw floor. The floor
    carries a systematic offset any deployment removes by multiplying, so beating IT would prove
    nothing about the fit."""
    rows = _synthetic()
    err = lambda ps: sum(abs(math.log(p) - math.log(r.best_us)) for p, r in zip(ps, rows, strict=True))  # noqa: E731
    assert err(cost_model.fit(rows).predict_us(rows)) < err(cost_model.cross_val_constant(rows, folds=4)) / 3


def test_a_column_almost_no_row_carries_is_not_fitted_on():
    """The ``S_*`` stamps are a histogram, so a corpus-wide union carries a long tail of columns
    one or two rows populate — four of them appear on a single row of the real 982. Fitting on
    them spends capacity on noise, and filtering at 1% coverage improved median held-out error at
    every fold count tried.

    Filtered by COVERAGE rather than by fitted importance, deliberately: choosing columns by how
    well they predict the label would be selecting on the whole corpus and then cross-validating
    on it, which leaks."""
    rows = _synthetic()
    rare = _Row(features={"R_log_roofline_us": 0.0, "signal": 1.0, "noise": 0.0, "freak": 1.0}, best_us=1.0, fold="g0")
    model = cost_model.fit([*rows, rare])
    assert "freak" not in model.cols, "a column on 1 row of 241 must not be fitted on"
    assert {"signal", "noise", "R_log_roofline_us"} <= set(model.cols)


def test_a_column_the_fit_never_saw_is_ignored_and_a_missing_one_is_nan():
    """``cols`` is positional — the booster knows order, not names — so both directions have to be
    handled at the packing seam rather than by hoping callers match."""
    model = cost_model.fit(_synthetic())
    assert "signal" in model.cols and "unseen" not in model.cols
    extra = _Row(features={"R_log_roofline_us": 0.0, "signal": 2.0, "noise": 1.0, "unseen": 99.0}, best_us=2.0)
    plain = _Row(features={"R_log_roofline_us": 0.0, "signal": 2.0, "noise": 1.0}, best_us=2.0)
    assert model.predict_us([extra]) == model.predict_us([plain])
    packed = model._matrix([_Row(features={"R_log_roofline_us": 0.0, "signal": 2.0}, best_us=1.0)])
    assert math.isnan(packed[0][model.cols.index("noise")])


# --- the artifact -----------------------------------------------------------------------------


def test_artifact_round_trip_preserves_predictions():
    rows = _synthetic()
    model = cost_model.fit(rows)
    artifact = model.to_artifact(provenance={"note": "test"})
    restored = cost_model.CostModel.from_artifact(artifact)
    assert restored.cols == model.cols
    assert restored.predict_us(rows) == pytest.approx(model.predict_us(rows))


def test_the_artifact_is_one_self_contained_object():
    """Inlined rather than a sidecar so the artifact is one file to copy, name and load — and so
    that shipping it never depends on a second path travelling with it."""
    artifact = cost_model.fit(_synthetic()).to_artifact(provenance={})
    assert isinstance(artifact["model"], str) and artifact["model"]  # base64, not a path
    assert "model_file" not in artifact
    import json

    assert json.loads(json.dumps(artifact))["kind"] == "cost"


def test_loading_a_foreign_or_stale_artifact_is_a_hard_error():
    """No silent fallback: a caller that quietly scored with the wrong model would measure
    nothing. A stale ``feat_ver`` is the dangerous case — the columns are spelled in a vocabulary
    that no longer means the same thing, so it would score garbage rather than fail."""
    good = cost_model.fit(_synthetic()).to_artifact(provenance={})
    with pytest.raises(RuntimeError, match="not a cost-model artifact"):
        cost_model.CostModel.from_artifact({**good, "kind": "linear"})
    with pytest.raises(RuntimeError, match="refit it"):
        cost_model.CostModel.from_artifact({**good, "feat_ver": FEATURIZER_VERSION + 1})
    with pytest.raises(KeyError):
        cost_model.CostModel.from_artifact({k: v for k, v in good.items() if k != "cols"})


# --- cross-validation -------------------------------------------------------------------------


def test_cross_validation_holds_out_whole_fold_groups():
    """The module's central honesty claim, asserted on the assignment rather than on the output.

    A kernel recurs across cards and several goldens can share a pool, so a split finer than the
    fold key scores a model on rows it trained on. Checking only that predictions exist would pass
    for a per-row split, which is exactly the leak this forbids."""
    from emmy.compiler.pipeline.search.prior.fit.cv import assign_folds

    rows = _synthetic()
    assigned = assign_folds([r.fold for r in rows], 4)
    for f in range(4):
        train = {r.fold for r in rows if assigned[r.fold] != f}
        held = {r.fold for r in rows if assigned[r.fold] == f}
        assert held and not (train & held), f"fold {f} shares a kernel with its training set"
    predicted = cost_model.cross_val_predict(rows, folds=4)
    assert len(predicted) == len(rows) and all(p > 0 for p in predicted)
    # Deterministic: the fold assignment is a pure function of the keys and their sizes.
    assert predicted == cost_model.cross_val_predict(rows, folds=4)


def test_the_report_holds_out_whole_cards_too():
    """The leave-one-card-out cells are the module's headline finding, so they need coverage.

    Cross-validating by kernel still lets a fit learn a card from that card's OTHER kernels; only
    holding the card out entirely asks whether the card features transfer. The guard below the
    branch skips cards with under 20 rows, so the fixture has to give two real ones."""
    report = cost_model.evaluate(_synthetic(), folds=4)
    held = {s.axes["cell"] for s in report.summaries if s.axes["axis"] == "held_out_card"}
    assert held == {"CARD_A", "CARD_B"}
    assert not [s for s in report.summaries if s.axes["axis"] == "gpu"], "an in-corpus per-card cell would contradict these"


def test_cross_validated_error_is_worse_than_in_sample():
    """The check that the numbers reported are held-out ones. If these matched, the split would be
    leaking and every figure in the report would be optimistic."""
    rows = _synthetic()
    in_sample = cost_model.fit(rows).predict_us(rows)
    held_out = cost_model.cross_val_predict(rows, folds=4)
    err = lambda ps: sum(abs(math.log(p) - math.log(r.best_us)) for p, r in zip(ps, rows, strict=True))  # noqa: E731
    assert err(held_out) > err(in_sample)


# --- the report -------------------------------------------------------------------------------


def test_the_report_carries_the_baseline_beside_every_cell():
    """Without a kernel-blind figure beside it a bias number is uninterpretable — and the baseline
    has to be the STRONGEST such predictor (roofline times one constant), not the bare floor, or
    the comparison flatters the fit on exactly the axis the report exists to be sceptical about.

    Every figure is a median or a quantile: the error distribution is heavy-tailed even in log
    space, so a mean would describe the worst rows rather than the typical one."""
    report = cost_model.evaluate(_synthetic(), folds=4)
    assert report.summaries
    for summary in report.summaries:
        assert set(summary.metrics) == {"model", "baseline"}
        if summary.axes["axis"] == "ordering":
            continue  # pairs, not rows — a different question with its own metric
        # Signed bias, typical error and the tail — and nothing derivable from them: the row
        # count is ``Summary.groups`` and the ratio is ``exp(bias)``, both left to the renderer.
        assert set(summary.metrics["model"]) == {"bias", "err", "p90"}
    overall = next(s for s in report.summaries if s.axes["cell"] == "all")
    assert overall.groups == 240
    assert overall.metrics["model"]["err"] < overall.metrics["baseline"]["err"]


def test_the_report_is_json_serializable():
    import json

    report = cost_model.evaluate(_synthetic(), folds=4)
    assert json.loads(json.dumps(report.to_json()))["header"]["target"] == "log(measured_us)"


def test_ordering_scores_close_pairs_and_ignores_easy_ones():
    """The question a fork actually asks: not whether a microsecond figure is right, but whether a
    comparison between two kernels comes out right.

    Banded to CLOSE pairs deliberately. Over all pairs a model scores near 1.0 because most pairs
    are a fast kernel against a slow one and ordering those takes no skill; a fork is only decided
    by the model when its arms are close, so that is the population worth scoring."""
    rows = _synthetic()
    idx = list(range(len(rows)))
    truth = [r.best_us for r in rows]
    perfect, n_pairs = cost_model._concordance(rows, truth, idx)
    reversed_, _ = cost_model._concordance(rows, [-t for t in truth], idx)
    assert n_pairs > 0
    # Exactly 0.0 for the reversed predictor is the check that TIES are excluded: a tie satisfies
    # the comparison in both directions, so counting them would credit any predictor for free.
    assert perfect == 1.0 and reversed_ == 0.0

    # A pair further apart than the band is not counted at all, however it is ordered.
    far = [
        _Row(features={"R_log_roofline_us": 0.0}, best_us=1.0),
        _Row(features={"R_log_roofline_us": 0.0}, best_us=1000.0),
    ]
    assert cost_model._concordance(far, [2.0, 1.0], [0, 1]) == (None, 0)


def test_the_report_scores_ordering_per_card_and_never_across_them():
    """Two kernels on different cards are not alternatives to each other, so a pair spanning cards
    would score a comparison no fork will ever make."""
    report = cost_model.evaluate(_synthetic(), folds=4)
    ordering = [s for s in report.summaries if s.axes["axis"] == "ordering"]
    assert {s.axes["cell"] for s in ordering} == {"CARD_A", "CARD_B"}
    for summary in ordering:
        assert set(summary.metrics["model"]) == {"concordance"}
        assert 0.0 <= summary.metrics["model"]["concordance"] <= 1.0
        # ``groups`` counts PAIRS here, not pools — and a card's pairs cannot exceed its own rows.
        rows_on_card = sum(1 for r in _synthetic() if r.gpu == summary.axes["cell"])
        assert summary.groups <= rows_on_card * (rows_on_card - 1) // 2
