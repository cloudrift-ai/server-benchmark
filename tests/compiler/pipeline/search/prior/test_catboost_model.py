"""The CatBoost offline model: its artifact round-trip, its absent-feature semantics, its exact
decomposition, and the trainer that produces it. No GPU — CatBoost fits on CPU in a fraction of a second at
these sizes.

The two model classes answer one surface (``mean_score_features`` / ``mean_scores_features`` /
``explain_features``), so most of what matters here is that a tree honours the same contracts the linear model
does, plus the two it deliberately does not: masking is no longer exact, and a fit is no longer byte-identical.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION
from emmy.compiler.pipeline.search.prior import OfflinePrior
from emmy.compiler.pipeline.search.prior.catboost_model import ABSENT, CatBoostModel
from emmy.compiler.pipeline.search.prior.fit.catboost import CatBoostTrainer
from emmy.compiler.pipeline.search.prior.fit.group import Group

FEATURES = ("D_a", "D_b")


def _groups(n_pools: int = 8, n_rows: int = 30, dynamic: bool = False) -> list[Group]:
    """Pools whose golden is the row maximizing ``D_a`` — a signal any working ranker recovers, and one
    that is invisible to ``D_b`` (pure noise), so a fitted model must have learned the right column."""
    rng = np.random.default_rng(0)
    stamp = {"S_ext_n_symbolic_axis": 1.0} if dynamic else {}
    out = []
    for gi in range(n_pools):
        rows = [{"D_a": float(i), "D_b": float(rng.integers(0, 5)), **stamp} for i in range(n_rows)]
        out.append(Group.from_dicts(f"gpuA/p{gi}", f"p{gi}", "dyn" if dynamic else "warp", "gpuA", f"shape{gi}", n_rows - 1, rows))
    return out


def _fit(groups=None, **kw) -> CatBoostModel:
    trainer = CatBoostTrainer(feature_names=FEATURES, iterations=40, negatives=12, **kw)
    return trainer.fit(groups if groups is not None else _groups()).model


# --- the trainer -------------------------------------------------------------------


def test_fit_recovers_the_planted_signal():
    """Every golden ranks first in its own full pool — the fit objective reaching its floor on a dataset
    where the answer is a single feature."""
    fit = CatBoostTrainer(feature_names=FEATURES, iterations=40, negatives=12).fit(_groups())
    assert fit.ranks == [0] * 8
    assert "not byte-reproducible" in fit.notes


def test_routing_stamp_is_an_ordinary_column():
    """The tree prices both regimes in ONE model: the stamp the dataset holds out of its matrix comes back
    as a plain column, so there is no second weight set and no unfittable-dynamic-set fold to exclude."""
    model = _fit(_groups() + _groups(dynamic=True))
    assert model.cols == (*FEATURES, "S_ext_n_symbolic_axis")


def test_trainer_declares_no_fittability_constraint():
    """The linear trainer excludes folds whose training slice cannot fit a weight set it needs. A tree has
    no weight sets, so it declares no constraint and the fold harness reads ``None``."""
    assert not hasattr(CatBoostTrainer(feature_names=FEATURES), "unfittable")


def test_score_rows_covers_the_full_pool_not_the_sample():
    """Training sampled 12 negatives per pool; the metric still ranks the golden among all 30 rows, so the
    reported rank stays exactly the deployed quantity."""
    groups = _groups()
    fit = CatBoostTrainer(feature_names=FEATURES, iterations=40, negatives=12).fit(groups)
    assert fit.rows < sum(len(g.feats) for g in groups)
    assert fit.score_rows(groups[0]).shape == (30,)


def test_hard_negative_mining_grows_the_training_set():
    """Round 0 draws negatives uniformly; each further round adds the rows the current model ranks near the
    golden. More rounds therefore train on strictly more rows — the mechanism, not merely the flag."""
    one = CatBoostTrainer(feature_names=FEATURES, iterations=20, negatives=5, rounds=1).fit(_groups())
    two = CatBoostTrainer(feature_names=FEATURES, iterations=20, negatives=5, rounds=2).fit(_groups())
    assert two.rows > one.rows


# --- the model surface -------------------------------------------------------------


def test_batched_and_single_row_scoring_agree():
    model = _fit()
    rows = [{"D_a": 3.0, "D_b": 1.0}, {"D_a": 29.0, "D_b": 4.0}]
    assert model.mean_scores_features(rows) == [model.mean_score_features(r) for r in rows]
    assert model.mean_scores_features([]) == []


def test_score_polarity_is_lower_is_better():
    """``mean_score`` is a latency proxy in both model classes, so the greedy argmin means the same thing
    whichever one is loaded: the row the ranker prefers must score LOWER."""
    model = _fit()
    assert model.mean_score_features({"D_a": 29.0, "D_b": 1.0}) < model.mean_score_features({"D_a": 0.0, "D_b": 1.0})


def test_absent_features_are_nan_not_zero():
    """The absent/decided-zero distinction, which is the whole reason a tree gets NaN: a row that never
    stamped ``D_b`` must not be read as a row that stamped it as 0.0."""
    model = _fit()
    packed = model.matrix([{"D_a": 5.0}])
    assert math.isnan(packed[0][1]) and packed[0][0] == 5.0
    assert math.isnan(ABSENT)


def test_explain_features_sums_to_the_quality():
    """The exactness contract the blame diagnostics rest on: the per-term decomposition totals the model's
    own ranking quantity, so a two-row term diff IS the model's preference gap."""
    model = _fit()
    row = {"D_a": 20.0, "D_b": 2.0}
    terms = model.explain_features(row)
    quality = model.quality_rows(model.matrix([row]))[0]
    assert sum(terms.values()) == pytest.approx(quality, rel=1e-6)
    assert "base:expected" in terms


def test_masking_is_not_exact_for_a_tree():
    """The linear model's twin is True. The ablation diagnostics read this to decide whether to caveat
    their Δ, and a tree re-routes its splits rather than dropping a term."""
    from emmy.compiler.pipeline.search.prior.linear_model import LinearModel

    assert CatBoostModel.masking_exact is False
    assert LinearModel.masking_exact is True


# --- the artifact ------------------------------------------------------------------


def _artifact(model: CatBoostModel) -> dict:
    return model.to_artifact(provenance={"fitted": "2026-08-13"})


def test_artifact_round_trip_preserves_predictions():
    """The base64 ``cbm`` blob is the model: a reloaded artifact scores identically, so an A/B against a
    written artifact measures the fit rather than the serialization."""
    model = _fit()
    art = json.loads(json.dumps(_artifact(model)))  # through JSON, as the file path does
    assert art["kind"] == "catboost" and art["feat_ver"] == FEATURIZER_VERSION
    reloaded = CatBoostModel.from_artifact(art)
    assert reloaded.cols == model.cols and reloaded.scale == model.scale
    rows = [{"D_a": float(i), "D_b": 1.0} for i in range(10)]
    assert reloaded.mean_scores_features(rows) == model.mean_scores_features(rows)


def test_offline_prior_loads_a_tree_artifact(tmp_path, monkeypatch):
    """The deploy path end to end: ``kind`` dispatches the load, and the prior ranks through the tree with
    no caller aware of which model class it holds."""
    from emmy import storage

    path = tmp_path / "tree.json"
    model = _fit()
    storage.write_json(path, _artifact(model))
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    prior = OfflinePrior()
    assert isinstance(prior.model, CatBoostModel)
    assert prior.masking_exact is False
    fast, slow = {"D_a": 29.0, "D_b": 1.0}, {"D_a": 0.0, "D_b": 1.0}
    assert prior.mean_score_features(fast) < prior.mean_score_features(slow)
    # ``policy`` prices the sibling set relative to its best — the PUCT surface, over a tree.
    assert max(prior.mean_scores_features([fast, slow])) > 0
    assert prior.policy([fast, slow])[0] == 1.0


def test_linear_only_overrides_are_rejected_against_a_tree_artifact(tmp_path, monkeypatch):
    """A per-field override silently ignored would hand an A/B the unmodified model and let it report a
    difference of zero as a real result — the same rule the ``model=``-plus-override case follows."""
    from emmy import storage

    path = tmp_path / "tree.json"
    storage.write_json(path, _artifact(_fit()))
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    with pytest.raises(ValueError, match="no per-field overrides"):
        OfflinePrior(scale=2.0)


def test_unknown_artifact_kind_is_a_hard_error(tmp_path, monkeypatch):
    from emmy import storage

    path = tmp_path / "weird.json"
    storage.write_json(path, {"feat_ver": FEATURIZER_VERSION, "kind": "randomforest", "params": {}})
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    with pytest.raises(RuntimeError, match="kind='randomforest'"):
        OfflinePrior()


def test_tree_artifact_missing_its_own_keys_is_a_hard_error(tmp_path, monkeypatch):
    """Per-kind validation: a tree artifact needs its column order and blob, which the linear key set
    would never have caught."""
    from emmy import storage

    art = _artifact(_fit())
    del art["cols"]
    path = tmp_path / "partial.json"
    storage.write_json(path, art)
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    with pytest.raises(RuntimeError, match="cols"):
        OfflinePrior()
