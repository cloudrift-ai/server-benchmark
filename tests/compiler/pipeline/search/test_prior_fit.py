"""The extracted OfflinePrior fit core (``search/prior/fit``) — the two preservation
guarantees the extraction from ``scripts/golden_knob_heuristics.py`` must keep:

1. The fit is a pure, deterministic function of (cases, seed): the same inputs
   produce a byte-identical weights artifact, so a refit is reproducible and an
   A/B between two fits measures the fit inputs, never run-to-run noise.
2. The fit-time rank evaluation and the deployed :class:`OfflinePrior` scoring
   order candidates identically for the shipped incumbent weights (away from the
   hardcoded interaction gates, which the fit's linear scoring deliberately
   excludes) — the invariant that makes fit-time golden ranks transfer to
   ``eval offline`` and greedy-deploy ranks.
"""

import json

import numpy as np
import pytest

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION
from emmy.compiler.pipeline.search.prior.fit import build_artifact, feature_matrix, fit_weights, rank_of_golden, raw_weights
from emmy.compiler.pipeline.search.prior.fit import linear as fit_linear
from emmy.compiler.pipeline.search.prior.offline import _DEFAULT_FILE, OfflinePrior

# The features the OfflinePrior scores OUTSIDE the linear weights (the hardcoded
# interaction gates) — the fit's linear rank eval deliberately excludes them, so the
# rank-agreement test keeps them out of its synthetic rows.
_GATE_FEATURES = {"D_finalize_kernel", "D_splitk", "D_scalar_on_warp_eligible", "D_splitk_roundtrip"}


def _synthetic_cases(n_cases=6, n_rows=40, n_feats=8, seed=1234):
    """A small fixed case set in the fitter's input shape: (name, tier, golden_idx,
    feature dict rows), with sparse rows so the ``feats.get(k, 0.0)`` path is live."""
    rng = np.random.default_rng(seed)
    names = [f"D_f{i}" for i in range(n_feats)]
    cases = []
    for c in range(n_cases):
        feats = []
        for _ in range(n_rows):
            row = {n: float(rng.integers(-4, 5)) for n in names}
            feats.append({k: v for k, v in row.items() if rng.random() > 0.25})
        tier = ["thread", "warp", "reduce", "dyn"][c % 4]
        cases.append((f"case{c}", tier, int(rng.integers(0, n_rows)), feats))
    return cases, names


def _run_fit(samples=200):
    """The script's two-stage fit (static, then dynamic seeded from static) on the
    fixed synthetic case set, assembled into artifact JSON bytes."""
    cases, names = _synthetic_cases()
    static_cases = [c for c in cases if c[1] != "dyn"]
    dyn_cases = [c for c in cases if c[1] == "dyn"]
    rng = np.random.default_rng(0)
    static_w, _, _, static_sd = fit_weights(static_cases, names, np.ones(len(names)), seed_w=np.zeros(len(names)), rng=rng, samples=samples)
    dyn_w, _, _, dyn_sd = fit_weights(dyn_cases, names, static_sd, seed_w=static_w, rng=rng, samples=samples)
    artifact = build_artifact(
        weights=raw_weights(names, static_w, static_sd),
        weights_dynamic=raw_weights(names, dyn_w, dyn_sd),
        params={
            "scale": 0.1,
            "atomic_free_split_threshold": 4.0,
            "atomic_free_weight": 0.0,
            "scalar_on_warp_weight": 40.0,
            "splitk_roundtrip_weight": 0.25,
        },
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
    names = sorted(k for k in w_set if k not in _GATE_FEATURES)
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


# --- declared weight bounds ---------------------------------------------------------


def test_weight_bounds_heal_seed_and_cap_descent(monkeypatch):
    """A poisoned seed above a declared bound is clamped on entry, and no search step
    escapes it — the raw exported weight lands within the bound in both stages (the
    D_pow2_threads 686-vs-112 incident: the rank objective is flat above the bound, so
    only the bound keeps the magnitude identified)."""
    monkeypatch.setattr(fit_linear, "WEIGHT_BOUNDS", {"D_f0": 0.5})
    cases, names = _synthetic_cases()
    seed = np.zeros(len(names))
    seed[names.index("D_f0")] = 1e6  # poisoned incumbent
    w, _, _, sd = fit_weights(cases, names, np.ones(len(names)), seed_w=seed, rng=np.random.default_rng(0), samples=50)
    raw = raw_weights(names, w, sd)
    assert abs(raw["D_f0"]) <= 0.5 + 1e-9


def test_shipped_artifact_respects_declared_bounds():
    """The repo-checked artifact honors every declared bound in BOTH weight sets — trips
    on a hand-edited or stale artifact that a bounded refit would have healed."""
    art = json.loads(_DEFAULT_FILE.read_text())
    for name, bound in fit_linear.WEIGHT_BOUNDS.items():
        for w_set in (art["weights"], art["weights_dynamic"]):
            if name in w_set:
                assert abs(w_set[name]) <= bound + 1e-9, f"{name} = {w_set[name]} exceeds declared bound {bound}"
