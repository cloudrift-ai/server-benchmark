"""The extracted OfflinePrior fit core (``search/prior/fit``) — the two preservation
guarantees the extraction from the original fit script must keep:

1. The fit is a pure, deterministic function of (cases, seed): the same inputs
   produce a byte-identical weights artifact, so a refit is reproducible and an
   A/B between two fits measures the fit inputs, never run-to-run noise.
2. The fit-time rank evaluation and the deployed :class:`OfflinePrior` scoring
   order candidates identically for the shipped incumbent weights — including the
   atomic-free interaction, which both sides now read through one shared definition.
   That is the invariant that makes fit-time golden ranks transfer to ``eval prior``
   and greedy-deploy ranks.
"""

import json
import math
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.data.group import GoldenGroup, feature_matrix
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION
from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS
from emmy.compiler.pipeline.search.metrics import rank_of_golden
from emmy.compiler.pipeline.search.prior.fit import (
    DEFAULT_L2,
    LinearFit,
    LinearTrainer,
    fit_weights,
    gate_columns,
    mean_log_rank,
    raw_weights,
)
from emmy.compiler.pipeline.search.prior.linear_model import GATE_FEATURES, LinearModel, gate_values, unweighted_cols
from emmy.compiler.pipeline.search.prior.offline import _DEFAULT_FILE, OfflinePrior

# Seed for the fitted scalar params — the interaction OFF at the shipped threshold, the state a
# fresh fit starts from.
SEED_PARAMS = np.array([0.0, 4.0])

# The interaction's inputs (:data:`GATE_FEATURES`) are kept out of the synthetic rows below, so that
# comparison isolates the linear part.
_INTERACTION_FEATURES = set(GATE_FEATURES)


def _synthetic_cases(n_cases=6, n_rows=40, n_feats=8, seed=1234):
    """A small fixed case set in the fitter's input shape (:class:`GoldenGroup`), with sparse
    rows so the absent-feature path (the matrix packing's fill columns) is live."""
    rng = np.random.default_rng(seed)
    names = [f"D_f{i}" for i in range(n_feats)]
    cases = []
    for c in range(n_cases):
        feats = []
        for _ in range(n_rows):
            row = {n: float(rng.integers(-4, 5)) for n in names}
            feats.append({k: v for k, v in row.items() if rng.random() > 0.25})
        tier = ["thread", "warp", "reduce", "dyn"][c % 4]
        # EVERY pool carries the routing stamp, as the featurizer writes it (``passes/identity._extents``
        # emits the key unconditionally, 0.0 when no axis is symbolic) — a static pool stamps 0.0, it does
        # not omit the key. ``GoldenGroup`` packs it like any other column and the linear trainer narrows it out.
        feats = [{**f, "S_ext_n_symbolic_axis": 1.0 if tier == "dyn" else 0.0} for f in feats]
        cases.append(GoldenGroup.from_dicts(f"x/case{c}", f"case{c}", tier, "x", f"shape{c}", int(rng.integers(0, n_rows)), feats))
    # The feature list the trainer is given is the union of the PACKED columns, exactly as
    # ``commands/fit.py`` builds it — so it includes the routing stamp, and the fit has to narrow it out.
    return cases, sorted({n for c in cases for n in c.feat_names})


# The incumbent a synthetic fit chains from: no weights to warm-start (the fits below set
# ``warm_start=False`` anyway), the interaction seeded OFF at the shipped threshold, and the shipped
# exp scale — which the trainer carries into the fitted model rather than fitting.
SEED_MODEL = LinearModel(
    unweighted_cols=unweighted_cols({}, None),
    weights={},
    weights_dynamic=None,
    scale=0.1,
    atomic_free_weight=float(SEED_PARAMS[0]),
    atomic_free_split_threshold=float(SEED_PARAMS[1]),
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
    # The trainer was handed the routing stamp among its feature names (the union of packed columns) and
    # must not have fitted a weight for it: constant within a pool, so any value there is noise, and it
    # would become an additive term on every symbolic-axis kernel in the cross-kernel price sum.
    assert not {n for n in (*art["weights"], *art["weights_dynamic"]) if n.startswith("S_ext")}


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
    assert "no dynamic groups" in static_only.notes and "dynamic top1" in first.notes


def test_fitting_one_slice_does_not_perturb_the_next():
    """Fold independence. ``run_axis`` used to hand each fold a fresh ``default_rng(seed)``; that
    guarantee now lives inside ``fit``, which builds its own RNG from ``random_state``. So fitting
    slice A and then slice B must give B exactly what fitting B alone gives — otherwise a fold's
    result would depend on how many folds ran before it, and adding one golden family would silently
    move every later fold's numbers."""
    cases, names = _synthetic_cases()
    trainer = _trainer(names, samples=20)
    a, b = cases[:4], cases[2:]
    trainer.fit(a)
    after_a = trainer.fit(b)
    assert after_a.model == trainer.fit(b).model  # same trainer, no carried RNG state
    assert after_a.model == _trainer(names, samples=20).fit(b).model  # and none carried on the object either


def test_built_artifact_loads_through_offline_prior(tmp_path, monkeypatch):
    """The assembled artifact is loadable by the deployed prior's strict loader
    (current feat_ver, complete params) — a fit output is never a dead file."""
    path = tmp_path / "fitted.json"
    path.write_text(_run_fit(samples=20))
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    p = OfflinePrior()
    assert p.mean_score_features({}) == 1.0  # neutral with no opinion, per the Prior contract
    assert json.loads(path.read_text())["feat_ver"] == FEATURIZER_VERSION


@pytest.mark.parametrize("path", sorted(_DEFAULT_FILE.parent.glob("offline_weights*.json")), ids=lambda p: p.stem)
def test_every_shipped_artifact_declares_the_columns_it_cannot_weight(path):
    """Every checked-in linear artifact must declare the columns it reads that no weight key can spell: the
    routing stamp it picks a weight set with, and the interaction's two inputs wherever a fit pruned their
    weights away. ``columns`` is the weight keys unioned with ``unweighted_cols``, so this is the whole of what
    the second field is for — a model whose declaration missed one would hand a pool builder a short list and be
    handed back a pool that silently misroutes, or prices the interaction off a zeros column.

    Both files, because both are reachable: the sibling is a scoped experiment selected with
    ``EMMY_OFFLINE_FILE`` / ``--offline-file``, and it is hand-maintained, which is exactly when a schema check
    earns its place."""
    art = json.loads(path.read_text())
    declared = set(OfflinePrior(model=LinearModel.from_artifact(art)).columns)

    assert set(GATE_FEATURES) <= declared
    assert set(features.ROUTING_FEATURES) <= declared
    # And the stored half is only ever the part the weights cannot carry.
    assert set(art["unweighted_cols"]) == declared - set(art["weights"]) - set(art["weights_dynamic"] or {})


def test_every_column_the_shipped_artifact_declares_is_still_produced():
    """A weight for a column the featurizer no longer spells is a DEAD term — it can never be anything but
    ``0.0``, because the pool never stamps the name and ``Group.matrix`` fills it with the linear model's
    absent value. Nothing used to notice, because nothing could ask an artifact which columns it read.

    This is not hypothetical drift. Retiring ``D_stage_ring`` is what collapsed the RTX 5090 matmul fit's
    top-1 from 54 of 242 goldens to 4, and the shipped artifact carried that dead weight — its fourth-largest
    by magnitude — for months, documented but unenforced. The ``FEATURIZER_VERSION`` 4 bump finally migrated
    the coefficient onto ``D_stage_prefetch``, the identical ``depth >= 2`` signal, so the artifact now
    declares nothing the featurizer has stopped producing.

    What counts as "still produced" is measured by re-featurizing every recorded golden's OWN knobs: no
    enumeration, no ``Context``, no GPU, and it enters through ``knob_features``, the same door the golden
    group builder feeds. A column no recorded configuration exhibits is exactly as dead as one the featurizer
    cannot spell."""
    declared = set(OfflinePrior().columns)
    produced: set[str] = set()
    for record in GOLDEN_RECORDS:
        if record.knobs:
            produced |= set(features.knob_features({**record.structural_features, **record.knobs}))
        if declared <= produced:
            break  # every remaining record can only widen ``produced``, so the verdict is already fixed

    assert declared - produced == set()


def test_a_view_that_hides_a_stamped_interaction_input_is_refused():
    """The fit searches the interaction's weight and threshold as descent coordinates. A view that drops
    ``D_finalize_kernel`` while the pools carry it leaves the term at 0.0 for every candidate, so the search
    walks two coordinates that cannot move the objective and the artifact ships whatever they landed on.

    Refused at the trainer, which is the one place that sees both the view and what the pools stamp. A corpus
    that never stamps the column at all is a different thing — nothing to price — and still fits: that is the
    case the synthetic pools above are."""
    cases, names = _synthetic_cases()
    stamped = [replace(c, feat_names=(*c.feat_names, "D_finalize_kernel")) for c in cases]

    _trainer(names).fit(cases)  # no finalize anywhere: nothing to hide, fits normally
    with pytest.raises(ValueError, match="D_finalize_kernel"):
        _trainer(names).fit(stamped)


# --- incumbent rank agreement: fit-time eval vs deployed scoring -------------------


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_incumbent_weights_rank_identically_through_fit_eval_and_prior(dynamic):
    """Score synthetic rows with the SHIPPED incumbent weights two ways — the fit
    module's linear matrix scoring (descending) and ``OfflinePrior``'s latency proxy
    (ascending) — and require identical ranks, ties included. The proxy is a monotone
    squash of the same linear quality, so any divergence is a real drift between the
    fitter's objective and what actually deploys."""
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
    # On every row either way: the featurizer emits the stamp unconditionally, 0.0 when no axis is symbolic,
    # and a pool without it is one no model can route.
    stamp = {"S_ext_n_symbolic_axis": 1.0 if dynamic else 0.0}
    proxies = np.array([prior.mean_score_features({**row, **stamp}) for row in rows])

    for i in range(len(rows)):
        assert rank_of_golden(linear_scores, i) == rank_of_golden(-proxies, i)


@pytest.mark.parametrize("dynamic", [False, True])
def test_dict_and_matrix_paths_score_identically(dynamic):
    """One definition, two access shapes. :meth:`LinearModel.quality` scores a live candidate from a
    feature dict; the fitter scores a packed pool as a matrix. They must agree row for row, including
    the atomic-free interaction on both sides of its split threshold (scored with the interaction ON,
    since the shipped weight is 0.0 and a zero term would make the comparison vacuous).

    This also pins the routing agreement, which is the part that CAN break: the dict path reads the
    symbolic-axis stamp off the row it is scoring, the matrix path reads ``GoldenGroup.dynamic`` off the
    pool. Same rows, so they must land on the same weight set."""
    art = json.loads(_DEFAULT_FILE.read_text())
    params = {"atomic_free_weight": 5.0, "atomic_free_split_threshold": 4.0}
    # ONE model object behind both paths — the prior scores dicts through it, the fit scores matrices.
    model = LinearModel(
        unweighted_cols=tuple(art["unweighted_cols"]),
        weights=art["weights"],
        weights_dynamic=art["weights_dynamic"],
        scale=0.1,
        **params,
    )
    prior, fit = OfflinePrior(model=model), LinearFit(model, [], [])

    rng = np.random.default_rng(11)
    names = sorted(set(art["weights"]) | _INTERACTION_FEATURES)
    # On every row either way: the featurizer emits the stamp unconditionally, 0.0 when no axis is symbolic,
    # and a pool without it is one no model can route.
    stamp = {"S_ext_n_symbolic_axis": 1.0 if dynamic else 0.0}
    rows = [
        {
            **{n: float(rng.integers(-2, 3)) for n in names if rng.random() > 0.3},
            "D_finalize_kernel": float(rng.integers(0, 2)),
            "D_splitk": float(rng.choice([1, 2, 4, 8])),  # straddles the threshold in both directions
            **stamp,
        }
        for _ in range(60)
    ]
    group = GoldenGroup.from_dicts("x/case", "case", "dyn" if dynamic else "warp", "x", "case", 0, rows)
    assert group.dynamic is dynamic  # routed by the rows' stamp, not the tier label
    fitted = fit.score_rows(group)
    deployed = np.array([prior.quality(row) for row in rows])
    assert np.allclose(fitted, deployed)


def test_an_unfittable_dynamic_fold_scores_as_none_rather_than_raising():
    """A model fit with no symbolic-axis cases has no dynamic weight set, and a dynamic pool handed to it
    is unanswerable rather than wrong. ``None`` is the word for that, and it must survive the forward from
    the fit to the model — :func:`~.cv.case_ranks` skips such a group, where a raised exception would abort
    the whole cross-validation run and a zero vector would silently rank the pool by emission order.

    Untested before this: ``weight_set`` RAISES on a missing dynamic set, so the guard that turns that into
    a ``None`` is the only thing standing between an unfittable fold and a crashed run.

    Static pools still score normally on the same model: it is the missing WEIGHT SET that is the limit,
    not the model."""
    static_only = LinearModel(
        unweighted_cols=unweighted_cols({"D_threads": 1.0}, None),
        weights={"D_threads": 1.0},
        weights_dynamic=None,
        scale=0.1,
        atomic_free_weight=0.0,
        atomic_free_split_threshold=4.0,
    )
    rows = [{"D_threads": float(i), "S_ext_n_symbolic_axis": 1.0} for i in range(5)]
    dyn = GoldenGroup.from_dicts("x/d", "d", "dyn", "x", "d", 0, rows)
    static_rows = [{"D_threads": float(i), "S_ext_n_symbolic_axis": 0.0} for i in range(5)]
    static = GoldenGroup.from_dicts("x/s", "s", "warp", "x", "s", 0, static_rows)

    assert dyn.dynamic and not static.dynamic
    for caller in (static_only, LinearFit(static_only, [], [])):  # the model, and the fit that forwards to it
        assert caller.score_rows(dyn) is None
        assert caller.score_rows(static) is not None


def test_the_proxy_warns_when_it_clips_instead_of_collapsing_a_ranking_silently():
    """A clipped exponent destroys a ranking: every row past the bound lands on the same float, so a greedy
    argmin over that plateau falls through to enumeration order. That is the 2026-07 incident, and it was
    silent — the whole point of the warning is that the same regime cannot recur unnoticed.

    Both model classes must go through the one definition, so the tree gets the same guarantee as the linear
    model without a second copy of the transform."""
    from emmy.compiler.pipeline.search.prior import base

    monkey = base._clip_warned
    try:
        base._clip_warned = False
        with patch.object(base.logger, "warning") as warned:
            assert base.latency_proxy(quality=1.0, scale=0.1) == pytest.approx(math.exp(-0.1))
            assert not warned.called  # the reachable regime says nothing
            assert base.latency_proxy(quality=1e5, scale=0.1) == pytest.approx(math.exp(-base.PROXY_CLIP))
            assert warned.called
            assert "enumeration order" in warned.call_args[0][0]
    finally:
        base._clip_warned = monkey


def test_model_artifact_round_trips():
    """``LinearModel`` → artifact → ``LinearModel`` is exact, so the file a fit ships and the model
    that produced it rank identically. The params block keeps its full key set (order is free — the
    writer preserves insertion order, so only the shape matters here)."""
    art = json.loads(_DEFAULT_FILE.read_text())
    model = LinearModel.from_artifact(art)
    assert LinearModel.from_artifact(model.to_artifact(provenance={})) == model
    assert model.to_artifact(provenance={})["params"] == art["params"]


def test_gate_values_and_gate_columns_agree_on_the_one_absent_value():
    """The interaction's two inputs are read one way for a dict and another for a matrix, and both must yield
    the same numbers. There is one absent value now, 0.0 — the same one an absent weight feature scores and the
    same one ``Group.matrix`` fills a column with — so the two shapes cannot disagree the way they could while
    each name carried a default of its own (a fabricated split count of 1, and a finalize zero that silently
    switched the whole term off).

    :data:`GATE_FEATURES` is the order ``atomic_free_term`` takes its positional arguments in, so that is
    pinned here too."""
    assert GATE_FEATURES == ("D_finalize_kernel", "D_splitk")  # finalize first, as the term reads them

    feats = {"D_finalize_kernel": 1.0, "D_splitk": 8.0, "D_other": 3.0}
    names = sorted(feats)
    # Like for like: a pool's column list IS its rows' feature names.
    cols = gate_columns(feature_matrix([feats], names), names)
    assert [float(c[0]) for c in cols] == list(gate_values(feats))
    # A row inside the pool that lacks the keys its siblings carry: 0.0 on both sides, and the term is off
    # because the formula says so at ``finalize == 0``, not because a default stood in for a value.
    assert gate_values({"D_other": 3.0}) == (0.0, 0.0)
    assert [float(c[0]) for c in gate_columns(feature_matrix([{"D_other": 3.0}], names), names)] == [0.0, 0.0]


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
