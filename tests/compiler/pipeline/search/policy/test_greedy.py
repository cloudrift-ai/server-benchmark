"""Focused tests for greedy schedule-space traversal."""

from types import SimpleNamespace

import pytest

from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline.fork import DeferredFork, Level, build_fork_tree, flatten_leaves, leaf_knobs
from emmy.compiler.pipeline.knob import canonical_row_key, schedule_row_key
from emmy.compiler.pipeline.search.policy import greedy
from emmy.compiler.pipeline.search.policy.greedy import (
    _db_measured_index_build,
    _direct_measured_pick,
    _stream_tiers,
    _verified_pick,
    golden_audit,
    tile_identity,
)


@pytest.mark.parametrize("route", ({"PLACE": "cut"}, {"PLACE@a": "cut"}, {"PLACE@a": "cut", "WORK": "t32"}))
def test_db_measured_index_excludes_placement_route_totals(route) -> None:
    signature = frozenset({("S_shape", "128")})
    rows = [
        SimpleNamespace(status="ok", stats=SimpleNamespace(median=1.0), knobs={"S_shape": 128, **route}),
        SimpleNamespace(status="ok", stats=SimpleNamespace(median=7.0), knobs={"S_shape": 128, "WORK": "t64"}),
    ]
    db = SimpleNamespace(iter_perf=lambda *_args, **_kwargs: rows)
    ctx = SimpleNamespace(structural_key=lambda: "ctx")

    assert _db_measured_index_build(db, ctx) == {signature: [({"WORK": "t64"}, 7.0)]}


def test_schedule_pick_descends_directly_to_complete_measured_row() -> None:
    materialized = []
    rows = [{"TILE": str(tile), "STAGE": str(stage)} for tile in range(100) for stage in range(100)]
    tree = build_fork_tree(
        params=rows,
        levels=(Level(("TILE",), lambda row: (row["TILE"],)), Level(("STAGE",), lambda row: (row["STAGE"],))),
        materialize=lambda row: materialized.append(row),
    )

    point = SimpleNamespace(
        options=[tree],
        node_id="node",
        root_op=SimpleNamespace(knobs={"S_shape": 128}),
        ctx=SimpleNamespace(features=lambda: {"H_opt": 3.0}),
    )
    index = {frozenset({("S_shape", "128")}): [({"TILE": "42", "STAGE": "73"}, 1.25)]}
    leaf, knobs, price = _direct_measured_pick(point, None, index)

    assert knobs == {"TILE": "42", "STAGE": "73"}
    assert leaf.knobs == knobs
    assert price == 1.25
    assert materialized == []


def test_verified_pick_ignores_feature_keys_in_schedule_branches(monkeypatch) -> None:
    rows = [
        {"S_warp_eligible": 1.0, "RASTER": "", "TILE": "slow"},
        {"S_warp_eligible": 1.0, "RASTER": "", "TILE": "recorded"},
        {"S_warp_eligible": 1.0, "RASTER": "gm8", "TILE": "other"},
    ]
    tree = build_fork_tree(
        params=rows,
        levels=(
            Level(("S_warp_eligible", "RASTER"), lambda row: (row["S_warp_eligible"], row["RASTER"])),
            Level(("TILE",), lambda row: (row["TILE"],)),
        ),
        materialize=lambda _row: None,
    )
    point = SimpleNamespace(
        options=[tree],
        node_id="node",
        root_op=TileOp(op=Fold.projection(body=Body())),
    )
    record = SimpleNamespace(name="recorded-golden", knobs={"RASTER": "", "TILE": "recorded"}, emmy_us=1.25)
    monkeypatch.setattr(greedy, "deploy_identity", lambda _op: "identity")

    leaf, price, knobs = _verified_pick(point, {"identity": [record]}, None)

    assert schedule_row_key(leaf_knobs(leaf)) == schedule_row_key(record.knobs)
    assert price == 1.25
    assert schedule_row_key(knobs) == schedule_row_key(record.knobs)

    audit = []
    with golden_audit(audit):
        audited_leaf, audited_price, audited_knobs = _verified_pick(point, {"identity": [record]}, None)
    assert schedule_row_key(leaf_knobs(audited_leaf)) == schedule_row_key(record.knobs)
    assert audited_price == 1.25
    assert schedule_row_key(audited_knobs) == schedule_row_key(record.knobs)
    assert audit[0]["verdict"] == "MATCH"


def test_verified_pick_defers_a_structural_fork(monkeypatch) -> None:
    structural = DeferredFork(materialize=lambda: None, structural=True)
    point = SimpleNamespace(
        options=[structural],
        node_id="node",
        root_op=TileOp(op=Fold.projection(body=Body())),
    )
    record = SimpleNamespace(name="recorded-golden", knobs={"TILE": "recorded"}, emmy_us=1.25)
    monkeypatch.setattr(greedy, "deploy_identity", lambda _op: "identity")

    assert _verified_pick(point, {"identity": [record]}, None) is None


# ---------------------------------------------------------------------------
# _stream_tiers — the streamed scan must equal the flattened scoring exactly.
# ---------------------------------------------------------------------------


def _score(row: dict) -> float:
    # Deliberately tie-heavy so the content tiebreak (canonical_row_key) decides across chunks.
    return float((int(row["TILE"]) * 3 + int(row["STAGE"]) * 5) % 4)


class _BarePrior:
    """A prior with only ``mean_scores`` (the bare branch of the old flatten path)."""

    def mean_scores(self, rows):
        return [_score(r) for r in rows]


class _EvidencePrior(_BarePrior):
    """Adds the ``pick`` + ``evidence_pick`` surface; ``pick`` must never run streamed."""

    def __init__(self, measured: dict[tuple[str, str], float]):
        self.measured = measured

    def pick(self, rows):
        raise AssertionError("the streamed scan must consult evidence_pick/mean_scores, never pick")

    def evidence_pick(self, rows):
        best = None
        for i, r in enumerate(rows):
            us = self.measured.get((r.get("TILE"), r.get("STAGE")))
            if us is None:
                continue
            if best is None or us < best[1] or (us == best[1] and canonical_row_key(r) < canonical_row_key(rows[best[0]])):
                best = (i, us)
        return best


def _point(rows):
    tree = build_fork_tree(
        params=rows,
        levels=(Level(("TILE",), lambda row: (row["TILE"],)), Level(("STAGE",), lambda row: (row["STAGE"],))),
        materialize=lambda row: (_ for _ in ()).throw(AssertionError("no leaf may materialize during ranking")),
    )
    return SimpleNamespace(
        options=[tree],
        node_id="node",
        root_op=SimpleNamespace(knobs={"S_shape": 128}),
        ctx=SimpleNamespace(features=lambda: {"H_opt": 3.0}),
    )


def _rows(n_tile=18, n_stage=7):
    return [{"TILE": str(t), "STAGE": str(s)} for t in range(n_tile) for s in range(n_stage)]


def test_streamed_model_pick_equals_flattened_argmin(monkeypatch) -> None:
    monkeypatch.setattr(greedy, "_CHUNK", 10)  # force many uneven chunks over the 126-leaf pool
    point = _point(_rows())
    got = _stream_tiers(point, _BarePrior(), None, {})
    assert got is not None
    leaf, knobs, price = got

    base = {"H_opt": 3.0, "S_shape": 128}
    flat = [(o, leaf_knobs(o)) for o in flatten_leaves(point.options)]
    rows = [{**base, **k} for _, k in flat]
    scores = _BarePrior().mean_scores(rows)
    best_i = min(range(len(rows)), key=lambda i: (scores[i], canonical_row_key(rows[i])))
    assert knobs == flat[best_i][1]
    # The lazy walk mints fresh (content-equal) leaf objects per expansion, so identity is by row.
    assert leaf_knobs(leaf) == flat[best_i][1]
    assert price == scores[best_i]


def test_streamed_evidence_beats_model_and_crosses_chunks(monkeypatch) -> None:
    monkeypatch.setattr(greedy, "_CHUNK", 10)
    # Two measured rows in different chunks; the faster one (later in emission order) must win.
    prior = _EvidencePrior({("1", "2"): 9.0, ("15", "4"): 2.5})
    got = _stream_tiers(_point(_rows()), prior, None, {})
    assert got is not None
    leaf, knobs, price = got
    assert knobs == {"TILE": "15", "STAGE": "4"}
    assert price == 2.5


def test_streamed_db_tier_outranks_the_model(monkeypatch) -> None:
    monkeypatch.setattr(greedy, "_CHUNK", 10)

    class _NoEvidence(_EvidencePrior):
        def evidence_pick(self, rows):
            return None

    # The measured DB row must win the deploy even though the model scores other rows better
    # (every row with _score == 0.0 beats the measured row's model score).
    db_idx = {frozenset({("S_shape", "128")}): [({"TILE": "7", "STAGE": "3"}, 2.0)]}
    got = _stream_tiers(_point(_rows()), _NoEvidence({}), None, db_idx)
    assert got is not None
    leaf, knobs, price = got
    assert knobs == {"TILE": "7", "STAGE": "3"}
    assert price == 2.0


def test_streamed_degenerate_pools() -> None:
    # Single-leaf pool: plain (unscored) return of that leaf.
    point = _point([{"TILE": "0", "STAGE": "0"}])
    leaf, knobs, price = _stream_tiers(point, _BarePrior(), None, {})
    assert knobs is None and price is None
    assert leaf_knobs(leaf) == {"TILE": "0", "STAGE": "0"}
    # Every leaf blocklisted: plain return of the first leaf.
    rows = _rows(3, 2)
    point = _point(rows)
    blocked = {tile_identity(dict(r)) for r in rows}
    leaf, knobs, price = _stream_tiers(point, _BarePrior(), blocked, {})
    assert knobs is None and price is None
    assert leaf_knobs(leaf) == rows[0]


def test_streamed_scan_defers_structural_forks_to_the_flatten_path() -> None:
    point = _point(_rows(2, 2))
    point.options = [*point.options, DeferredFork(materialize=lambda: None, structural=True)]
    assert _stream_tiers(point, _BarePrior(), None, {}) is None


def test_budgeted_pool_ranks_a_deterministic_drawn_subset(monkeypatch) -> None:
    """Above the cold-pool budget the scan ranks seeded descents instead of walking: the pick is
    a legal complete row, identical across calls (the RNG seeds from the pool identity), and the
    model scores at most the draw, never the pool."""
    from dataclasses import dataclass, field

    from emmy.compiler.pipeline.fork import Fork

    @dataclass(frozen=True)
    class _BoundedFork(Fork):
        inner: Fork = None
        knobs: dict = field(default_factory=dict)
        expansions: list = field(default_factory=list, compare=False)
        pool_bound = 10**9
        pool_id = "test-pool"
        pool_descent_bound = 100
        is_leaf = False

        def expand(self):
            self.expansions.append(None)
            return self.inner.expand()

    class _CountingPrior(_BarePrior):
        def __init__(self):
            self.scored = 0

        def mean_scores(self, rows):
            self.scored += len(rows)
            return super().mean_scores(rows)

    monkeypatch.setattr(greedy, "_POOL_DRAW", 64)
    rows = _rows(30, 20)  # 600 leaves ≫ the draw
    all_rows = {(r["TILE"], r["STAGE"]) for r in rows}
    point = _point(rows)
    point.options = [_BoundedFork(inner=point.options[0])]
    prior = _CountingPrior()
    got = _stream_tiers(point, prior, None, {})
    assert got is not None
    leaf, knobs, price = got
    assert (knobs["TILE"], knobs["STAGE"]) in all_rows  # a legal complete row off the real tree
    assert prior.scored <= 64  # the draw, never the pool
    prior2 = _CountingPrior()
    again = _stream_tiers(point, prior2, None, {})
    assert again[1] == knobs and again[2] == price  # seeded off the pool identity → reproducible
    monkeypatch.setattr(greedy, "_POOL_DESCENT_WORK", 800)
    bounded = _CountingPrior()
    assert _stream_tiers(point, bounded, None, {}) is not None
    assert bounded.scored == 2

    monkeypatch.setattr(greedy, "_POOL_DESCENT_WORK", 1)
    overwide = _CountingPrior()
    picked = _stream_tiers(point, overwide, None, {})
    assert picked is not None and isinstance(picked[0], Fork) and picked[0].is_leaf
    assert set(leaf_knobs(picked[0])) == {"TILE", "STAGE"}
    assert overwide.scored == 0  # one complete row needs no ranking
    repeated = _stream_tiers(point, _CountingPrior(), None, {})
    assert repeated is not None and leaf_knobs(repeated[0]) == leaf_knobs(picked[0])

    blocked_point = _point(rows)
    wrapper = _BoundedFork(inner=blocked_point.options[0])
    blocked_point.options = [wrapper]
    blocked = {tile_identity(dict(row)) for row in rows}
    with pytest.raises(RuntimeError, match="no live complete row"):
        _stream_tiers(blocked_point, _CountingPrior(), blocked, {})
    assert len(wrapper.expansions) == 1  # no retry and no exhaustive fallback
