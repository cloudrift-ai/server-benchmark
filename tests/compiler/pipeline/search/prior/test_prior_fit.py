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

import numpy as np
import pytest

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION
from emmy.compiler.pipeline.search.prior.fit import (
    DEFAULT_L2,
    PARAM_NAMES,
    Group,
    TwoStageFit,
    build_artifact,
    feature_matrix,
    fit_weights,
    gate_columns,
    objective,
    rank_of_golden,
    raw_weights,
)
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
        cases.append(Group.from_dicts(f"x/case{c}", f"case{c}", tier, "x", int(rng.integers(0, n_rows)), feats))
    return cases, names


def _run_fit(samples=200):
    """The script's two-stage fit (static, then dynamic seeded from static) on the
    fixed synthetic case set, assembled into artifact JSON bytes."""
    cases, names = _synthetic_cases()
    static_cases = [c for c in cases if c.tier != "dyn"]
    dyn_cases = [c for c in cases if c.tier == "dyn"]
    rng = np.random.default_rng(0)
    static_w, params, _, _, static_sd = fit_weights(
        static_cases, names, np.ones(len(names)), seed_w=np.zeros(len(names)), seed_params=SEED_PARAMS, rng=rng, samples=samples
    )
    dyn_w, _, _, _, dyn_sd = fit_weights(
        dyn_cases, names, static_sd, seed_w=static_w, seed_params=params, rng=rng, samples=samples, fit_params=False
    )
    artifact = build_artifact(
        weights=raw_weights(names, static_w, static_sd),
        weights_dynamic=raw_weights(names, dyn_w, dyn_sd),
        params={"scale": 0.1, **dict(zip(PARAM_NAMES, params, strict=True))},
        provenance={"fitted": "2026-01-01", "script": "test", "args": {"samples": samples, "seed": 0}},
    )
    return json.dumps(artifact, indent=2)


# --- determinism / byte-identity ---------------------------------------------------


def test_fit_twice_is_byte_identical():
    a, b = _run_fit(), _run_fit()
    assert a == b
    art = json.loads(a)
    assert art["weights"] and art["weights_dynamic"]  # a real fit, not an empty pass-through


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
def test_fit_scores_the_deployed_quality_interaction_included(dynamic, monkeypatch):
    """The fit's ``score_rows`` and the deployed ``OfflinePrior`` are the SAME function, on rows
    that exercise the atomic-free interaction (both sides of its split threshold) — the invariant
    that makes the fitted objective the deployed ranking. Scored with the interaction ON, since
    the shipped weight is 0.0 and a zero term would make the comparison vacuous."""
    monkeypatch.delenv("EMMY_OFFLINE_FILE", raising=False)
    art = json.loads(_DEFAULT_FILE.read_text())
    params = {"atomic_free_weight": 5.0, "atomic_free_split_threshold": 4.0}
    prior = OfflinePrior(weights=art["weights"], weights_dynamic=art["weights_dynamic"], scale=0.1, **params)
    fit = TwoStageFit(art["weights"], [], art["weights_dynamic"], [], params)

    rng = np.random.default_rng(11)
    names = sorted(set(art["weights"]) | _INTERACTION_FEATURES)
    rows = [
        {
            **{n: float(rng.integers(-2, 3)) for n in names if rng.random() > 0.3},
            "D_finalize_kernel": float(rng.integers(0, 2)),
            "D_splitk": float(rng.choice([1, 2, 4, 8])),  # straddles the threshold in both directions
        }
        for _ in range(60)
    ]
    stamp = {"S_ext_n_symbolic_axis": 1.0} if dynamic else {}
    group = Group.from_dicts("x/case", "case", "dyn" if dynamic else "warp", "x", 0, rows)
    fitted = fit.score_rows(group)
    deployed = np.array([prior.quality({**row, **stamp}) for row in rows])
    assert np.allclose(fitted, deployed)


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
        Group(g.key, g.name, g.tier, g.gpu, g.pinned_idx, (*g.feat_names, "D_flat"), np.hstack([g.feats, np.full((len(g.feats), 1), 3.0)]))
        for g in cases
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
    assert objective(healed[DEFAULT_L2][1]) <= objective(base_ranks) + 1e-9  # rank quality held or improved


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
    assert objective(ranks[DEFAULT_L2]) <= objective(ranks[0.0]) + 1e-9
