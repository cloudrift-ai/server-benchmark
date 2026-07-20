"""The deploy-side DB evidence consult (``policy/greedy._db_measured_pick`` /
``_db_measured_index``) — a config the tune *measured* must not lose the deploy to an
unmeasured model extrapolation (eighth golden sweep, finding 2). The index sources the
-O1 ranking lane's ``ok`` perf rows keyed by ``S_*`` signature; matching follows
``Prior.evidence_pick``'s prefix-consistency contract (candidate-specified tunable
knobs must match; undecided knobs are free)."""

from __future__ import annotations

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.db import PerfStats, SearchDB
from emmy.compiler.pipeline.search.policy.greedy import _TUNE_RANKING_FLAGS, _db_measured_index, _db_measured_pick

_SIG = {"S_ext_free_prod": 2048.0, "S_dtype_f32": 1.0}


def _index_of(*measured: tuple[dict, float], deployable: bool = True) -> dict:
    sig = frozenset((k, str(v)) for k, v in _SIG.items())
    return {sig: [({k: str(v) for k, v in tun.items()}, us, deployable) for tun, us in measured]}


def test_db_pick_prefers_measured_min_over_unmeasured():
    index = _index_of(({"TILE": "n16x8/f4x8", "REDUCE": "g2a"}, 47.1), ({"TILE": "n32x8/f4x8", "REDUCE": "g8k"}, 51.7))
    rows = [
        {**_SIG, "TILE": "n32x8/f4x8", "REDUCE": "g8k"},  # the model's would-be argmax
        {**_SIG, "TILE": "n16x8/f4x8", "REDUCE": "g2a"},  # the measured-fastest config
        {**_SIG, "TILE": "n64x16/f4x4", "REDUCE": ""},  # unmeasured — no evidence
    ]
    got = _db_measured_pick(index, rows)
    assert got is not None
    best_i, us = got
    assert best_i == 1 and us == 47.1


def test_db_pick_requires_signature_match():
    index = _index_of(({"TILE": "n16x8/f4x8"}, 10.0))
    other_shape = {"S_ext_free_prod": 512.0, "S_dtype_f32": 1.0, "TILE": "n16x8/f4x8"}
    assert _db_measured_pick(index, [other_shape]) is None


def test_db_pick_tolerates_signature_vocabulary_drift():
    """A deploy candidate's fork-time base may carry ``S_*`` stamps the recorded rows
    predate (#311's ``S_warp_eligible`` is on no perf row recorded before it) — the join
    must match on the shared ``S_*`` keys, or one added feature silently disables the
    whole evidence lane against every existing DB (the ninth-4090-sweep `mlp_gate_up`
    misdeploy: the model's pick beat a measured-faster config the strict join hid)."""
    index = _index_of(({"TILE": "n16x8/f4x8", "REDUCE": "g2a"}, 47.1))
    drifted = {**_SIG, "S_warp_eligible": 1.0, "TILE": "n16x8/f4x8", "REDUCE": "g2a"}
    assert _db_measured_pick(index, [drifted]) == (0, 47.1)


def test_db_pick_drift_tolerance_still_separates_shapes():
    """Shared-key agreement still rejects a different shape: the extent keys are on
    both sides, so drift tolerance never crosses shapes."""
    index = _index_of(({"TILE": "n16x8/f4x8"}, 10.0))
    other = {"S_ext_free_prod": 512.0, "S_dtype_f32": 1.0, "S_warp_eligible": 1.0, "TILE": "n16x8/f4x8"}
    assert _db_measured_pick(index, [other]) is None


def test_db_pick_prefers_deployable_lane_over_faster_ranking_row():
    """-O1 medians are a ranking signal with -O3 inversions: a deployable-lane
    measurement decides even when a ranking-lane row looks faster (an -O1 override of
    the model regressed qkv/mlp_down in the ninth-4090-sweep verification), and the
    ranking lane decides only when no candidate has deployable evidence."""
    sig = frozenset((k, str(v)) for k, v in _SIG.items())
    index = {sig: [({"TILE": "a"}, 40.0, False), ({"TILE": "b"}, 45.0, True)]}
    rows = [{**_SIG, "TILE": "a"}, {**_SIG, "TILE": "b"}]
    assert _db_measured_pick(index, rows) == (1, 45.0)
    ranking_only = {sig: [({"TILE": "a"}, 40.0, False)]}
    assert _db_measured_pick(ranking_only, rows) == (0, 40.0)


def test_db_index_reads_the_o3_twin_as_deployable(tmp_path):
    """The tune's -O3 re-benches record perf rows under the ``-Xcicc -O3`` context
    twin; the deploy (empty ambient flags) must find them and treat them as the
    deployable lane — they are the DB's only deployable-latency truth."""
    from emmy.compiler.pipeline.search.policy.mcts import O3_NVCC_FLAGS

    ctx_deploy = Context.from_target((12, 0))
    ctx_o3 = Context.from_target((12, 0))
    object.__setattr__(ctx_o3, "compile_flags", O3_NVCC_FLAGS)
    ctx_o1 = Context.from_target((12, 0))
    object.__setattr__(ctx_o1, "compile_flags", _TUNE_RANKING_FLAGS)
    db = SearchDB(path=tmp_path / "tune.db")
    stats_o3 = PerfStats(median=45.0, min=44.0, max=46.0, mean=45.0, variance=0.1, n_samples=5)
    stats_o1 = PerfStats(median=40.0, min=39.0, max=41.0, mean=40.0, variance=0.1, n_samples=5)
    db.record_perf(ctx_o3.structural_key(), "op-b", backend="cuda", status="ok", stats=stats_o3, knobs={**_SIG, "TILE": "b"})
    db.record_perf(ctx_o1.structural_key(), "op-a", backend="cuda", status="ok", stats=stats_o1, knobs={**_SIG, "TILE": "a"})
    index = _db_measured_index(db, ctx_deploy)
    rows = [{**_SIG, "TILE": "a"}, {**_SIG, "TILE": "b"}]
    assert _db_measured_pick(index, rows) == (1, 45.0)


def test_db_pick_prefix_consistency_frees_undecided_knobs():
    # The measured row carries a STAGE the candidate hasn't decided — still a match
    # (value-of-position semantics); a *conflicting* decided knob is not.
    index = _index_of(({"TILE": "n16x8/f4x8", "STAGE": "d2/tma/ring"}, 20.0))
    undecided = {**_SIG, "TILE": "n16x8/f4x8"}
    conflicting = {**_SIG, "TILE": "n16x8/f4x8", "STAGE": "d4/tma/ring"}
    assert _db_measured_pick(index, [undecided]) == (0, 20.0)
    assert _db_measured_pick(index, [conflicting]) is None


def test_db_index_reads_the_o1_tune_twin_context(tmp_path):
    """Perf rows recorded under the tune ranking lane's ``-Xcicc -O1`` context must be
    found by a deploy running under different (e.g. -O3) ambient flags — ``context_key``
    folds the flags, so the index queries the twin key beside the deploy's own."""
    ctx_deploy = Context.from_target((12, 0))
    ctx_tune = Context.from_target((12, 0))
    object.__setattr__(ctx_tune, "compile_flags", _TUNE_RANKING_FLAGS)
    db = SearchDB(path=tmp_path / "tune.db")
    stats = PerfStats(median=47.1, min=46.0, max=48.0, mean=47.2, variance=0.1, n_samples=5)
    db.record_perf(
        ctx_tune.structural_key(),
        "op-key-1",
        backend="cuda",
        status="ok",
        stats=stats,
        knobs={**_SIG, "TILE": "n16x8/f4x8", "REDUCE": "g2a"},
    )
    index = _db_measured_index(db, ctx_deploy)
    rows = [{**_SIG, "TILE": "n16x8/f4x8", "REDUCE": "g2a"}]
    assert _db_measured_pick(index, rows) == (0, 47.1)


def test_db_index_skips_failed_rows(tmp_path):
    ctx = Context.from_target((12, 0))
    db = SearchDB(path=tmp_path / "tune.db")
    stats = PerfStats(median=2e6, min=2e6, max=2e6, mean=2e6, variance=0.0, n_samples=1)
    db.record_perf(ctx.structural_key(), "op-key-1", backend="cuda", status="bench_fail", stats=stats, knobs={**_SIG, "TILE": "n16x8/f4x8"})
    assert _db_measured_index(db, ctx) == {}


def test_structural_price_probe_threads_db_evidence(monkeypatch):
    """The structural fork's pricing probe (``_price_kernel``'s nested resolve) must
    carry the same ``db`` evidence as a top-level knob pick. A prediction-only probe
    is how a measured-slower split displaced the measured-faster fused
    ``mlp_gate_up.h4096`` kernel (ninth 4090 sweep): the model's absolute-µs error
    doesn't cancel across kernel families, so a Σ-of-predictions comparison can flip
    a fork the DB's own measurements refute."""
    from types import SimpleNamespace

    from emmy.compiler.pipeline.search.policy import greedy

    seen: dict = {}
    real = greedy.greedy_decide

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(greedy, "greedy_decide", spy)
    graph = SimpleNamespace(nodes={"n0": SimpleNamespace(op=SimpleNamespace())})
    sentinel = object()
    ctx = Context.from_target((12, 0))
    # The dummy graph is unpriceable (no real op) — the probe must still have been
    # built with the caller's db and with structural pricing off (no recursion).
    assert greedy._price_kernel(graph, "n0", ctx, prior=None, memo={}, db=sentinel) is None
    assert seen.get("db") is sentinel
    assert seen.get("price_structural") is False


def test_disjoint_evidence_warns(caplog):
    """When the DB holds measured rows for a fork's structural signature but NO offered
    candidate prefix-matches any of them, the fallback to the model prediction must warn:
    that condition is exactly "the tune measured a schedule tier this compile did not
    offer" (the gemma o_proj scalar misdeploy — the DB held 34 mma rows while the deploy
    fork offered 370 scalar-only candidates). A cold signature (no rows at all) stays
    silent — extrapolation is expected there."""
    import logging

    from emmy.compiler.pipeline.search.policy.greedy import _warn_disjoint_evidence

    index = _index_of(({"TILE": "a:mma_m16n8k16_f16/w2x2/f4x8", "REDUCE": "g2k"}, 225.2))
    scalar_only = [{**_SIG, "TILE": "n32x8/f4x14", "REDUCE": "g2k"}, {**_SIG, "TILE": "n16x16/f4x8", "REDUCE": "b256"}]
    assert _db_measured_pick(index, scalar_only) is None
    with caplog.at_level(logging.WARNING):
        _warn_disjoint_evidence(index, scalar_only, "linear_3")
    assert any("measured a schedule tier" in r.message for r in caplog.records)

    caplog.clear()
    cold = [{"S_ext_free_prod": 99.0, "TILE": "n32x8/f4x14"}]
    with caplog.at_level(logging.WARNING):
        _warn_disjoint_evidence(index, cold, "linear_3")
    assert not caplog.records, "a cold signature (no evidence at all) must stay silent"


def test_db_pick_stale_row_never_vouches_for_the_cut_twin():
    """A perf row recorded before ``PLACE@cone`` existed (any pre-cut tune DB / reservoir)
    measured the FUSED form; under plain value-of-position semantics its absent key would
    match the knob-identical cut candidate too — and the content tie-break PREFERS
    ``('PLACE@cone','cut') < ('PLACE@cone','fuse')``, deploying the catastrophic cold cut
    at the fused row's µs (the 35–142 ms unseeded-consumer class the model-tier withhold
    exists to prevent). The cut may only win where it was actually measured — a row that
    explicitly records ``PLACE@cone: cut`` still vouches for the cut candidate."""
    stale = _index_of(({"TILE": "n16x8/f4x8", "REDUCE": "g2a"}, 47.1))
    twins = [
        {**_SIG, "TILE": "n16x8/f4x8", "REDUCE": "g2a", "PLACE@cone": "fuse"},
        {**_SIG, "TILE": "n16x8/f4x8", "REDUCE": "g2a", "PLACE@cone": "cut"},
    ]
    assert _db_measured_pick(stale, twins) == (0, 47.1), "a stale row vouches only for the fused twin"
    measured_cut = _index_of(({"TILE": "n16x8/f4x8", "REDUCE": "g2a", "PLACE@cone": "cut"}, 12.0))
    assert _db_measured_pick(measured_cut, twins) == (1, 12.0), "an explicit cut row still vouches for the cut"
