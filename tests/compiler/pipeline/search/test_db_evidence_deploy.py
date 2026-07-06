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


def _index_of(*measured: tuple[dict, float]) -> dict:
    sig = frozenset((k, str(v)) for k, v in _SIG.items())
    return {sig: [({k: str(v) for k, v in tun.items()}, us) for tun, us in measured]}


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
