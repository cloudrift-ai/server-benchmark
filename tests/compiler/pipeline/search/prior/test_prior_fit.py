"""The extracted OfflinePrior fit core (``search/prior/fit``) — the two preservation
guarantees the extraction from the original fit script must keep:

1. The fit is a pure, deterministic function of (cases, seed): the same inputs
   produce a byte-identical weights artifact, so a refit is reproducible and an
   A/B between two fits measures the fit inputs, never run-to-run noise.
2. The fit-time rank evaluation and the deployed :class:`OfflinePrior` scoring
   order candidates identically for the shipped incumbent weights — including the
   atomic-free interaction, which both sides now read through one shared definition.
   That is the invariant that makes fit-time golden ranks transfer to ``eval offline``
   and greedy-deploy ranks.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION
from emmy.compiler.pipeline.search.prior.fit import (
    DEFAULT_L2,
    Group,
    LinearFit,
    LinearTrainer,
    feature_matrix,
    fit_weights,
    gate_columns,
    mean_log_rank,
    rank_of_golden,
    raw_weights,
)
from emmy.compiler.pipeline.search.prior.linear_model import LinearModel
from emmy.compiler.pipeline.search.prior.offline import _DEFAULT_FILE, OfflinePrior

# Seed for the fitted scalar params — the interaction OFF at the shipped threshold, the state a
# fresh fit starts from.
SEED_PARAMS = np.array([0.0, 4.0])

# The two features the OfflinePrior reads through the atomic-free interaction rather than as plain
# linear terms — kept out of the synthetic rows below so that comparison isolates the linear part.
_INTERACTION_FEATURES = {"D_finalize_kernel", "D_splitk"}


def _synthetic_cases(n_cases=6, n_rows=40, n_feats=8, seed=1234):
    """A small fixed case set in the fitter's input shape (:class:`Group`), with sparse
    rows so the absent-feature = 0.0 path (the matrix packing's zero columns) is live."""
    rng = np.random.default_rng(seed)
    names = [f"D_f{i}" for i in range(n_feats)]
    cases = []
    for c in range(n_cases):
        feats = []
        for _ in range(n_rows):
            row = {n: float(rng.integers(-4, 5)) for n in names}
            feats.append({k: v for k, v in row.items() if rng.random() > 0.25})
        tier = ["thread", "warp", "reduce", "dyn"][c % 4]
        # The dynamic cases carry the routing stamp on every row, as the golden case builder writes
        # it; ``Group.from_dicts`` lifts it into ``Group.dynamic`` and out of the fitted matrix.
        if tier == "dyn":
            feats = [{**f, "S_ext_n_symbolic_axis": 1.0} for f in feats]
        cases.append(Group.from_dicts(f"x/case{c}", f"case{c}", tier, "x", int(rng.integers(0, n_rows)), feats))
    return cases, names


# The incumbent a synthetic fit chains from: no weights to warm-start (the fits below set
# ``warm_start=False`` anyway), the interaction seeded OFF at the shipped threshold, and the shipped
# exp scale — which the trainer carries into the fitted model rather than fitting.
SEED_MODEL = LinearModel(
    weights={}, weights_dynamic=None, scale=0.1, atomic_free_weight=float(SEED_PARAMS[0]), atomic_free_split_threshold=float(SEED_PARAMS[1])
)


def _trainer(names, samples=200, **kwargs):
    return LinearTrainer(feature_names=tuple(names), init=SEED_MODEL, samples=samples, random_state=0, warm_start=False, **kwargs)


def _run_fit(samples=200):
    """One trainer invocation on the fixed synthetic case set, as artifact JSON bytes."""
    cases, names = _synthetic_cases()
    fit = _trainer(names, samples).fit(cases)
    provenance = {"fitted": "2026-01-01", "script": "test", "args": {"samples": samples, "seed": 0}}
    return json.dumps(fit.model.to_artifact(provenance=provenance), indent=2)


# --- determinism / byte-identity ---------------------------------------------------


def test_fit_twice_is_byte_identical():
    a, b = _run_fit(), _run_fit()
    assert a == b
    art = json.loads(a)
    assert art["weights"] and art["weights_dynamic"]  # a real fit, not an empty pass-through


def test_one_trainer_instance_refits_identically():
    """``fit`` is pure: the same trainer instance fitted twice returns equal models, and fitting
    does not mutate the trainer. That is what lets ONE instance serve every cross-validation fold
    with no copying — the property an sklearn-style ``clone`` would otherwise have to provide."""
    cases, names = _synthetic_cases()
    trainer = _trainer(names, samples=20)
    first, second = trainer.fit(cases), trainer.fit(cases)
    assert first.model == second.model
    assert first.static_ranks == second.static_ranks
    assert trainer == _trainer(names, samples=20)
    # A fit with no dynamic groups leaves that weight set unfitted rather than substituting one,
    # and says so in the provenance line the artifact records.
    static_only = trainer.fit([c for c in cases if not c.dynamic])
    assert static_only.model.weights_dynamic is None and static_only.dyn_ranks is None
    assert "no dynamic cases" in static_only.notes and "dynamic top1" in first.notes


def test_built_artifact_loads_through_offline_prior(tmp_path, monkeypatch):
    """The assembled artifact is loadable by the deployed prior's strict loader
    (current feat_ver, complete params) — a fit output is never a dead file."""
    path = tmp_path / "fitted.json"
    path.write_text(_run_fit(samples=20))
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    p = OfflinePrior()
    assert p.mean_score_features({}) == 1.0  # neutral with no opinion, per the Prior contract
    assert json.loads(path.read_text())["feat_ver"] == FEATURIZER_VERSION


# --- incumbent rank agreement: fit-time eval vs deployed scoring -------------------


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_incumbent_weights_rank_identically_through_fit_eval_and_prior(monkeypatch, dynamic):
    """Score synthetic rows with the SHIPPED incumbent weights two ways — the fit
    module's linear matrix scoring (descending) and ``OfflinePrior``'s latency proxy
    (ascending) — and require identical ranks, ties included. The proxy is a monotone
    squash of the same linear quality, so any divergence is a real drift between the
    fitter's objective and what actually deploys."""
    monkeypatch.delenv("EMMY_OFFLINE_FILE", raising=False)
    art = json.loads(_DEFAULT_FILE.read_text())
    w_set = art["weights_dynamic"] if dynamic else art["weights"]
    names = sorted(k for k in w_set if k not in _INTERACTION_FEATURES)
    w_vec = np.array([w_set[n] for n in names])

    rng = np.random.default_rng(7)
    rows, qualities = [], []
    while len(rows) < 60:
        row = {n: float(rng.integers(-2, 3)) for n in names if rng.random() > 0.3}
        q = sum(w_set[k] * v for k, v in row.items())
        # Keep qualities pairwise-distinct: a float-roundoff "tie" between the two
        # summation orders would flip the tie-pessimistic rank for reasons that have
        # nothing to do with the invariant under test.
        if all(abs(q - other) > 1e-6 for other in qualities):
            rows.append(row)
            qualities.append(q)

    prior = OfflinePrior()
    linear_scores = feature_matrix(rows, names) @ w_vec
    stamp = {"S_ext_n_symbolic_axis": 1.0} if dynamic else {}
    proxies = np.array([prior.mean_score_features({**row, **stamp}) for row in rows])

    for i in range(len(rows)):
        assert rank_of_golden(linear_scores, i) == rank_of_golden(-proxies, i)


@pytest.mark.parametrize("dynamic", [False, True])
def test_dict_and_matrix_paths_score_identically(dynamic, monkeypatch):
    """One definition, two access shapes. :meth:`LinearModel.quality` scores a live candidate from a
    feature dict; the fitter scores a packed pool as a matrix. They must agree row for row, including
    the atomic-free interaction on both sides of its split threshold (scored with the interaction ON,
    since the shipped weight is 0.0 and a zero term would make the comparison vacuous).

    This also pins the routing agreement, which is the part that CAN break: the dict path reads the
    symbolic-axis stamp off the row it is scoring, the matrix path reads ``Group.dynamic`` off the
    pool. Same rows, so they must land on the same weight set."""
    monkeypatch.delenv("EMMY_OFFLINE_FILE", raising=False)
    art = json.loads(_DEFAULT_FILE.read_text())
    params = {"atomic_free_weight": 5.0, "atomic_free_split_threshold": 4.0}
    # ONE model object behind both paths — the prior scores dicts through it, the fit scores matrices.
    model = LinearModel(weights=art["weights"], weights_dynamic=art["weights_dynamic"], scale=0.1, **params)
    prior, fit = OfflinePrior(model=model), LinearFit(model, [], [])

    rng = np.random.default_rng(11)
    names = sorted(set(art["weights"]) | _INTERACTION_FEATURES)
    stamp = {"S_ext_n_symbolic_axis": 1.0} if dynamic else {}
    rows = [
        {
            **{n: float(rng.integers(-2, 3)) for n in names if rng.random() > 0.3},
            "D_finalize_kernel": float(rng.integers(0, 2)),
            "D_splitk": float(rng.choice([1, 2, 4, 8])),  # straddles the threshold in both directions
            **stamp,
        }
        for _ in range(60)
    ]
    group = Group.from_dicts("x/case", "case", "dyn" if dynamic else "warp", "x", 0, rows)
    assert group.dynamic is dynamic  # routed by the rows' stamp, not the tier label
    fitted = fit.score_rows(group)
    deployed = np.array([prior.quality(row) for row in rows])
    assert np.allclose(fitted, deployed)


def test_model_artifact_round_trips():
    """``LinearModel`` → artifact → ``LinearModel`` is exact, so the file a fit ships and the model
    that produced it rank identically. The params block keeps its full key set (order is free — the
    writer preserves insertion order, so only the shape matters here)."""
    art = json.loads(_DEFAULT_FILE.read_text())
    model = LinearModel.from_artifact(art)
    assert LinearModel.from_artifact(model.to_artifact(provenance={})) == model
    assert model.to_artifact(provenance={})["params"] == art["params"]


def test_gate_columns_survive_the_in_place_z_scoring():
    """The interaction's inputs stay in RAW units through the fit's standardization. ``fit_weights``
    z-scores its pools IN PLACE (the memory fix that keeps an 18 GB corpus fittable), so the columns
    must be copied out beforehand: a view would be standardized underneath the comparison, and a split
    COUNT centred near zero never clears its threshold, silently switching the whole term off."""
    names = ["D_finalize_kernel", "D_splitk", "D_other"]
    mat = np.array([[1.0, 8.0, 5.0], [0.0, 2.0, 7.0], [1.0, 4.0, 9.0]])
    fin, spl = gate_columns(mat, names)
    mat -= mat.mean(0)
    mat /= mat.std(0)  # exactly what fit_weights does to its pools afterwards
    assert list(fin) == [1.0, 0.0, 1.0]
    assert list(spl) == [8.0, 2.0, 4.0]


# --- raw-space L2 regularization ---------------------------------------------------


def _cases_with_flat_feature():
    """The synthetic set plus one constant feature: its z-scored column is all zeros, so
    the rank objective is COMPLETELY flat in its weight — the identifiability failure
    behind the D_pow2_threads 686 incident, in miniature."""
    cases, names = _synthetic_cases()
    flat_cases = [
        replace(g, feat_names=(*g.feat_names, "D_flat"), feats=np.hstack([g.feats, np.full((len(g.feats), 1), 3.0)])) for g in cases
    ]
    return flat_cases, [*names, "D_flat"]


def test_l2_heals_poisoned_seed_on_rank_flat_feature():
    """A poisoned incumbent weight on a rank-flat feature survives an unregularized
    descent-from-seed refit untouched (the objective gives it no gradient); with the
    L2 penalty in the loss the descent walks the flat direction down to ~zero without
    giving up rank quality: at the default strength no rank-worsening step is ever
    accepted (the penalty quantum is orders below the rank quantum), so refitting from
    a converged incumbent can only hold or improve the data term."""
    cases, names = _cases_with_flat_feature()
    ones = np.ones(len(names))

    # Converge unregularized from zero, then poison the flat feature — the shipped-686
    # shape: an incumbent that is a data-term optimum plus an unidentified magnitude.
    base_w, _, base_ranks, _, base_sd = fit_weights(
        cases, names, ones, seed_w=np.zeros(len(names)), seed_params=SEED_PARAMS, rng=np.random.default_rng(0), samples=0, l2=0.0
    )
    poisoned = base_w / base_sd  # the incumbent's raw weights
    poisoned[names.index("D_flat")] = 5.0

    healed = {}
    for l2 in (0.0, DEFAULT_L2):
        w, _, ranks, _, sd = fit_weights(
            cases, names, ones, seed_w=poisoned, seed_params=SEED_PARAMS, rng=np.random.default_rng(0), samples=0, l2=l2
        )
        healed[l2] = (raw_weights(names, w, sd), ranks)
    assert abs(healed[0.0][0]["D_flat"] - 5.0) < 1e-9  # unregularized: the poison survives
    assert abs(healed[DEFAULT_L2][0].get("D_flat", 0.0)) < 0.5  # regularized: walked to ~zero
    assert mean_log_rank(healed[DEFAULT_L2][1]) <= mean_log_rank(base_ranks) + 1e-9  # rank quality held or improved


def test_l2_default_is_rank_neutral_on_random_restart():
    """At the declared default strength the penalty is a tie-breaker: any genuine rank
    improvement dwarfs it, so regularizing cannot cost fit quality. Asserted on the
    objective, not on rank-by-rank equality — walking a rank-flat direction toward zero is
    exactly what the penalty is for, and a step that is neutral in rank but better in
    magnitude legitimately sends the descent down a different (here, marginally better) path."""
    cases, names = _synthetic_cases()
    ranks = {}
    for l2 in (0.0, DEFAULT_L2):
        _, _, r, _, _ = fit_weights(
            cases,
            names,
            np.ones(len(names)),
            seed_w=np.zeros(len(names)),
            seed_params=SEED_PARAMS,
            rng=np.random.default_rng(0),
            samples=200,
            l2=l2,
        )
        ranks[l2] = r
    assert mean_log_rank(ranks[DEFAULT_L2]) <= mean_log_rank(ranks[0.0]) + 1e-9
