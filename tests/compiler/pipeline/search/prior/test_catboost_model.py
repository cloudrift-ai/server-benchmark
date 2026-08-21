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
from dataclasses import replace
from pathlib import Path

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
    stamp = {"S_ext_n_symbolic_axis": 1.0 if dynamic else 0.0}
    out = []
    for gi in range(n_pools):
        rows = [{"D_a": float(i), "D_b": float(rng.integers(0, 5)), **stamp} for i in range(n_rows)]
        out.append(Group.from_dicts(f"gpuA/p{gi}", f"p{gi}", "dyn" if dynamic else "warp", "gpuA", f"shape{gi}", n_rows - 1, rows))
    return out


def _fit(groups=None, *, feature_names=FEATURES, **kw) -> CatBoostModel:
    trainer = CatBoostTrainer(feature_names=feature_names, iterations=40, negatives=12, **kw)
    return trainer.fit(groups if groups is not None else _groups()).model


# --- the trainer -------------------------------------------------------------------


def test_fit_recovers_the_planted_signal():
    """Every golden ranks first in its own full pool — the fit objective reaching its floor on a dataset
    where the answer is a single feature."""
    fit = CatBoostTrainer(feature_names=FEATURES, iterations=40, negatives=12).fit(_groups())
    assert fit.ranks == [0] * 8
    assert "not byte-reproducible" in fit.notes


def test_the_tree_prices_the_two_regimes_from_the_routing_column():
    """The capability the packed column restores. Static and dynamic pools here rank OPPOSITELY on ``D_a``,
    which one model can only fit by splitting on the regime. While the column arrived all-NaN every row sat
    in one bucket, the split was unavailable, and the fit had to compromise across the two regimes.

    Guards the fix at the level it operates: not the column's name, its contents."""
    rng = np.random.default_rng(0)

    def pools(dynamic, offset):
        out = []
        for gi in range(8):
            rows = [
                {"D_a": float(i), "D_b": float(rng.integers(0, 5)), "S_ext_n_symbolic_axis": 1.0 if dynamic else 0.0} for i in range(30)
            ]
            pinned = 0 if dynamic else 29  # dynamic pools want the LOW D_a row, static the high one
            out.append(
                Group.from_dicts(
                    f"gpuA/p{gi + offset}", f"p{gi + offset}", "dyn" if dynamic else "warp", "gpuA", f"shape{gi + offset}", pinned, rows
                )
            )
        return out

    groups = pools(False, 0) + pools(True, 8)
    fit = CatBoostTrainer(feature_names=(*FEATURES, "S_ext_n_symbolic_axis"), iterations=60, negatives=12).fit(groups)
    assert fit.ranks == [0] * 16
    importance = dict(zip(fit.model.cols, fit.model.booster.get_feature_importance(type="PredictionValuesChange"), strict=True))
    assert importance["S_ext_n_symbolic_axis"] > 0.0


def test_routing_stamp_is_an_ordinary_column():
    """The tree prices both regimes in ONE model: the routing stamp is a plain packed column it reads like
    any other, so there is no second weight set and no unfittable-dynamic-set fold to exclude.

    The load-bearing assertion is that the column arrives with the STAMP in it. It used to arrive all-NaN:
    the dataset withheld the name, the trainer appended it to ``cols`` anyway, and ``Group.matrix`` filled
    a column it could not find with ``ABSENT``. Every training row landed in one bucket while every live
    candidate carried a real 0.0/1.0 — a train/serve skew that ``cols`` alone could never reveal."""
    routing = ("S_ext_n_symbolic_axis",)
    groups = _groups() + _groups(dynamic=True)
    model = _fit(groups, feature_names=(*FEATURES, *routing))
    assert model.cols == (*FEATURES, *routing)
    static, dynamic = groups[0], groups[-1]
    assert (dynamic.matrix(list(routing), fill=ABSENT) == 1.0).all()
    assert (static.matrix(list(routing), fill=ABSENT) == 0.0).all()


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


# --- several positives per pool ----------------------------------------------------


def test_a_verified_row_is_never_drawn_as_a_negative():
    """A pool the builder matched several goldens into holds several verified-good configs, and none of them
    may be handed to the loss at label 0.0 — teaching the model that a config we measured as good is bad is
    exactly the contamination ``DEFAULT_ROUNDS`` warns about, only self-inflicted.

    Checked on both draws: the small-pool branch, which contributes every row it has rather than sampling,
    and the sampling branch. And on the mined draw, whose window has to widen by the pin count or a pool
    whose pins all rank high would come back short."""
    trainer = CatBoostTrainer(feature_names=FEATURES, negatives=12)
    rng = np.random.default_rng(0)
    pins = (2, 5)
    small = trainer._uniform(8, pins, rng)  # 6 unpinned rows <= 12: the every-row branch
    assert sorted(small) == [0, 1, 3, 4, 6, 7]
    big = trainer._uniform(200, pins, rng)
    assert len(big) == 12 and not set(pins) & set(big.tolist())
    scores = np.arange(50, dtype=float)[::-1]  # row 0 scores highest, so the pins sit inside the window
    mined = trainer._hard(scores, (0, 1))
    assert len(mined) == 12 and not {0, 1} & set(mined.tolist())


def test_row_accounting_covers_every_positive():
    """``rows`` is what the fit reports it trained on, so a pool with two pins must count two — otherwise the
    number silently understates the training set as the corpus grows siblings."""
    trainer = CatBoostTrainer(feature_names=FEATURES, negatives=4)
    groups = [_groups(n_pools=1)[0], replace(_groups(n_pools=1)[0], key="gpuA/p1", pinned=(0, 1, 2))]
    assert trainer._n_rows([np.arange(4), np.arange(4)], groups) == (4 + 1) + (4 + 3)


def test_fit_treats_every_pin_as_a_positive():
    """``QuerySoftMax`` takes several positives per group as they are, so a pool with two verified configs
    needs no reshaping: both pins ride into the training set at label 1.0 and both end up ahead of every
    unpinned row. A run that had drawn one of them as a negative would have pushed it back down."""
    rows = [{"D_a": float(i), "D_b": 0.0} for i in range(20)]
    groups = [Group.from_dicts(f"gpuA/p{i}", f"p{i}", "warp", "gpuA", f"shape{i}", (18, 19), rows) for i in range(6)]
    fit = CatBoostTrainer(feature_names=FEATURES, iterations=40, negatives=8).fit(groups)
    assert fit.rows == 6 * (8 + 2)
    assert set(np.argsort(-fit.score_rows(groups[0]), kind="stable")[:2].tolist()) == {18, 19}
    # Was [1] * 6 while the trainer appended a routing name the matrix could not fill: the resulting
    # all-NaN column collapsed rows 18 and 19 onto one leaf, so the two pins tied and a tie counts against
    # the golden. Without that junk column the tree separates them and each golden ranks first.
    assert fit.ranks == [0] * 6


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


def _artifact(model: CatBoostModel, tmp_path=None, name: str = "weights") -> dict:
    """The artifact dict, and (when given a directory) its sidecar written beside it — the pair the loader
    expects, since the booster no longer rides inside the JSON."""
    art = model.to_artifact(provenance={"fitted": "2026-08-13"}, model_file=f"{name}.cbm")
    if tmp_path is not None:
        (tmp_path / art["model_file"]).write_bytes(model.blob)
    return art


def test_artifact_round_trip_preserves_predictions(tmp_path):
    """The sidecar IS the model: a reloaded artifact scores identically, so an A/B against a written artifact
    measures the fit rather than the serialization."""
    model = _fit()
    art = json.loads(json.dumps(_artifact(model, tmp_path)))  # through JSON, as the file path does
    assert art["kind"] == "catboost" and art["feat_ver"] == FEATURIZER_VERSION
    assert art["model_file"] == "weights.cbm" and "model" not in art, "the booster must not ride in the JSON"
    reloaded = CatBoostModel.from_artifact(art, base_dir=tmp_path)
    assert reloaded.cols == model.cols and reloaded.scale == model.scale
    rows = [{"D_a": float(i), "D_b": 1.0} for i in range(10)]
    assert reloaded.mean_scores_features(rows) == model.mean_scores_features(rows)


def test_sidecar_path_is_relative_so_the_pair_can_move(tmp_path):
    """The recorded path must be relative: an absolute one breaks the moment the JSON + cbm pair is copied into a
    run directory or rsynced to a tuning box with a different scratch root."""
    model = _fit()
    art = _artifact(model, tmp_path)
    assert not Path(art["model_file"]).is_absolute()
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    (moved / "weights.json").write_text(json.dumps(art))
    (moved / art["model_file"]).write_bytes(model.blob)
    assert CatBoostModel.from_artifact(art, base_dir=moved).cols == model.cols


def test_two_artifacts_in_one_directory_do_not_collide(tmp_path):
    """The sidecar is named after ITS OWN json, so a run directory holding several artifacts keeps a distinct
    model per artifact rather than the last write winning."""
    a, b = _fit(), _fit(_groups(n_pools=6))
    art_a, art_b = _artifact(a, tmp_path, "cand-a"), _artifact(b, tmp_path, "cand-b")
    assert art_a["model_file"] != art_b["model_file"]
    assert (tmp_path / "cand-a.cbm").read_bytes() != (tmp_path / "cand-b.cbm").read_bytes()


def test_inline_blob_still_loads(tmp_path):
    """Artifacts written before the sidecar split carry the booster inline as base64. They still load — run
    outputs under ``_tune/`` predate the change and are still referenced by in-flight experiments."""
    from emmy.compiler.pipeline.search.prior.catboost_model import to_b64

    model = _fit()
    legacy = {k: v for k, v in _artifact(model, tmp_path).items() if k != "model_file"}
    legacy["model"] = to_b64(model.booster)
    assert CatBoostModel.from_artifact(legacy).cols == model.cols


def test_offline_prior_loads_a_tree_artifact(tmp_path, monkeypatch):
    """The deploy path end to end: ``kind`` dispatches the load, and the prior ranks through the tree with
    no caller aware of which model class it holds."""
    from emmy import storage

    path = tmp_path / "tree.json"
    model = _fit()
    storage.write_json(path, _artifact(model, tmp_path, "tree"))
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
    storage.write_json(path, _artifact(_fit(), tmp_path, "tree"))
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

    art = _artifact(_fit(), tmp_path, "partial")
    del art["cols"]
    path = tmp_path / "partial.json"
    storage.write_json(path, art)
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    with pytest.raises(RuntimeError, match="cols"):
        OfflinePrior()
