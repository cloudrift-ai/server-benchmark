"""Focused tests for greedy schedule-space traversal."""

import logging
import math
from types import SimpleNamespace

import pytest

from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline.fork import DeferredFork, Level, build_fork_tree, flatten_leaves, leaf_knobs
from emmy.compiler.pipeline.knob import canonical_row_key, schedule_row_key
from emmy.compiler.pipeline.pipeline import NO_OPTION, ForkPoint
from emmy.compiler.pipeline.search.golden import GoldenRecord
from emmy.compiler.pipeline.search.policy import greedy
from emmy.compiler.pipeline.search.policy.greedy import (
    _db_measured_index_build,
    _direct_measured_pick,
    _stream_tiers,
    _verified_pick,
    golden_audit,
    tile_identity,
)


def _record(name: str, knobs: dict, *, emmy_us: float = 1.25) -> GoldenRecord:
    """A minimal recorded row — the real ``GoldenRecord``, so ``is_routing`` (which lane may read
    it) is its own derivation from the knobs rather than a stand-in's hand-set flag."""
    return GoldenRecord(
        name=name,
        gpu_name="card",
        compute_cap=(12, 0),
        model=None,
        program_index=0,
        program_wire={},
        origins=(),
        bindings=(),
        pins=(),
        knobs=knobs,
        measurements={"emmy_us": emmy_us, "reference_us": 2.0, "reference_backend": "torch"},
        ranking=None,
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

    assert _db_measured_index_build(db, ctx).ok == {signature: [({"WORK": "t64"}, 7.0)]}


def test_db_measured_index_collects_shapes_whose_every_measured_variant_failed() -> None:
    """A ``bench_fail`` row is evidence too — the watchdog measured that variant not finishing.
    When EVERY measured variant of one structural shape failed, the shape itself is disqualified;
    one surviving ``ok`` variant means only some rows are bad and the shape stays rankable.

    Failures are collected before the placement-route filter the ``ok`` tier applies: a route's
    LATENCY is unattributable without a child-schedule receipt, but a kernel that hung is
    attributable to the kernel whatever route produced it."""
    doomed = frozenset({("S_shape", "4096")})
    mixed = frozenset({("S_shape", "128")})
    rows = [
        SimpleNamespace(status="bench_fail", stats=SimpleNamespace(median=2_000_000.0), knobs={"S_shape": 4096, "WORK": "t32"}),
        SimpleNamespace(status="bench_fail", stats=SimpleNamespace(median=2_000_000.0), knobs={"S_shape": 4096, "PLACE": "fuse"}),
        SimpleNamespace(status="bench_fail", stats=SimpleNamespace(median=2_000_000.0), knobs={"S_shape": 128, "WORK": "t32"}),
        SimpleNamespace(status="ok", stats=SimpleNamespace(median=7.0), knobs={"S_shape": 128, "WORK": "t64"}),
    ]
    db = SimpleNamespace(iter_perf=lambda *_args, **_kwargs: rows)
    ctx = SimpleNamespace(structural_key=lambda: "ctx")

    measured = _db_measured_index_build(db, ctx)
    assert doomed in measured.failed, "every measured variant of this shape hit the watchdog"
    assert mixed not in measured.failed, "a shape with one ok variant is not disqualified"
    assert measured.ok == {mixed: [({"WORK": "t64"}, 7.0)]}


def test_a_shape_whose_every_variant_failed_prices_as_infeasible() -> None:
    """The disqualification's teeth: a slice containing a known-failed kernel prices ``inf``, so
    any structural arm holding it loses the ``_priced_pick`` argmin to an arm that does not.

    Without this the failures are invisible at deploy — the measured index carries ``ok`` rows
    only, so a kernel every one of whose variants hung simply has no evidence and falls through to
    the prior, which is exactly how DeepSeek-V4's post block kept its hanging fused arm across a
    30-minute tune that recorded 40 failures for it."""
    doomed = frozenset({("S_shape", "4096")})
    op = SimpleNamespace(knobs={"S_shape": 4096}, identity_key=lambda **_kw: "k")
    terminal = SimpleNamespace(nodes={"n": SimpleNamespace(op=op)})
    trace = [SimpleNamespace(node_id="n", score=5.0)]
    ctx = SimpleNamespace(features=lambda: {})

    assert greedy._resolved_price(terminal, trace, ctx, None).us == 5.0
    disqualified = greedy._resolved_price(terminal, trace, ctx, None, failed={doomed: [2_000_000.0]})
    assert disqualified == greedy.Price(math.inf, measured=True), "a failure row is a measurement, so the inf is measured"


def test_a_disqualification_condemns_only_the_shape_that_was_measured() -> None:
    """Elimination matches the signature EXACTLY, unlike the ranking tier's drift-tolerant
    :func:`_sig_groups`. There a loose match only widens the candidate pool and a second filter
    still has to agree on the tunable knobs; here nothing follows the match, so condemning every
    shape that merely does not contradict a recorded failure disqualifies the whole program.
    Measured: on DeepSeek-V4's post block the tolerant form priced all 17 leaves of one fork
    ``inf``, which decides nothing at all."""
    recorded = frozenset({("S_shape", "4096"), ("S_dtype_f16", "1.0")})
    # Agrees on every SHARED key, so the drift-tolerant matcher would call it a hit; it is a
    # different shape and must still be priced.
    other = SimpleNamespace(knobs={"S_shape": 4096, "S_n_loop": 9}, identity_key=lambda **_kw: "k")
    terminal = SimpleNamespace(nodes={"n": SimpleNamespace(op=other)})
    trace = [SimpleNamespace(node_id="n", score=5.0)]
    ctx = SimpleNamespace(features=lambda: {})

    assert greedy._sig_groups({recorded: [1.0]}, frozenset({("S_shape", "4096"), ("S_n_loop", "9")})), (
        "the tolerant matcher does hit here — which is exactly why elimination must not use it"
    )
    assert greedy._resolved_price(terminal, trace, ctx, None, failed={recorded: [2_000_000.0]}).us == 5.0


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


def _fork_point(options, *, rule: str, node_id: str = "node") -> ForkPoint:
    """A ForkPoint over ``options`` — the real one, so ``fp.structural`` reads the engine's own
    typed partition of the offers rather than a stand-in's hand-set flag."""
    return ForkPoint(
        match=SimpleNamespace(root_node_id=node_id, rule=SimpleNamespace(name=rule, pass_=None)),
        options=options,
        root_op=TileOp(op=Fold.projection(body=Body())),
        ctx=SimpleNamespace(features=lambda: {"H_opt": 3.0}),
    )


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
    point = _fork_point([tree], rule="040_schedule")
    record = _record("recorded-golden", {"RASTER": "", "TILE": "recorded"})
    monkeypatch.setattr(TileOp, "identity_key", lambda _op, **_kw: "identity")

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


@pytest.mark.parametrize("with_fuse_arm", [False, True])
def test_verified_pick_defers_a_structural_fork(monkeypatch, caplog, with_fuse_arm) -> None:
    """A kernel-SET fork is not this tier's question, whichever shape it takes: the bare structural
    offer, or a placement fork's non-structural FUSE arm beside its structural cut arms
    (``030_cut``). Every arm spells ``PLACE``, so all of them canonicalize to the all-OFF schedule
    row — comparing a recorded schedule row there let an all-OFF recording bind the fuse arm and
    decide a placement off schedule evidence, and made every other recording read as drift on a
    fork it was never about. The tier declines the whole fork, silently, so a drift warning keeps
    meaning "this recording no longer realizes"."""
    cut = DeferredFork(materialize=lambda: None, knobs={"a3": "cut"}, structural=True)
    fuse = DeferredFork(materialize=lambda: None, knobs={"PLACE": "fuse"}, structural=False)
    point = _fork_point([fuse, cut] if with_fuse_arm else [cut], rule="030_cut")
    monkeypatch.setattr(TileOp, "identity_key", lambda _op, **_kw: "identity")

    all_off = _record("all-off-golden", dict.fromkeys(("WORK", "TILE", "REDUCE", "STAGE", "RASTER"), ""))
    scheduled = _record("scheduled-golden", {"WORK": "t512", "TILE": "recorded"})
    for record in (all_off, scheduled):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=greedy.logger.name):
            assert _verified_pick(point, {"identity": [record]}, None) is None, f"{record.name} decided a kernel-set fork"
        assert "drift" not in caplog.text, f"{record.name} was reported as drift on a kernel-set fork"

    audit = []
    with golden_audit(audit):
        assert _verified_pick(point, {"identity": [all_off]}, None) is None
    assert audit == [], "a kernel-set fork is not a SCHEDULE consultation and carries no verdict"


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
    assert _stream_tiers(blocked_point, _CountingPrior(), blocked, {}) == (NO_OPTION, None, None)
    assert len(wrapper.expansions) == 1  # no retry and no exhaustive fallback


# ---------------------------------------------------------------------------
# greedy_decide — an untrustworthy prior may not settle a MIXED kernel-set comparison.
# ---------------------------------------------------------------------------


class _ProxyPrior(_BarePrior):
    """A prior that answers ``trustworthy`` — the surface the competence check reads."""

    def __init__(self, *, trustworthy: bool):
        self.trustworthy = trustworthy


def _kernel_set_fork():
    """One ``030_cut``-shaped fork: a splice against a keep-fused schedule pool.

    Priced the way the RTX 5080 nvfp4 deploy prices its m32 post-attention block — the fused
    kernel at its MEASURED 49.73 µs, the cut's fragments at what the offline prior's ordinal
    proxy makes of them. The proxy is not µs, so its number is five orders of magnitude below a
    real kernel and the cut wins every such fork on arithmetic alone.
    """
    splice = DeferredFork(materialize=lambda: object(), knobs={"PLACE": "cut"}, structural=True)
    fused = [DeferredFork(materialize=lambda: object(), knobs={"TILE": str(t), "STAGE": "0"}, structural=False) for t in range(3)]
    return ForkPoint(
        match=SimpleNamespace(root_node_id="add_2", rule=SimpleNamespace(name="030_cut", pass_=None)),
        options=[splice, *fused],
        root_op=SimpleNamespace(knobs={"S_shape": 128}),
        ctx=SimpleNamespace(features=lambda: {"H_opt": 3.0}),
    ), splice


@pytest.mark.parametrize(("trustworthy", "takes_the_cut"), [(False, False), (True, True)])
def test_a_predicted_cut_may_not_beat_a_measured_fused_side(monkeypatch, trustworthy, takes_the_cut) -> None:
    """A Σ-of-µs comparison across two kernel sets is only a latency comparison when both sides
    are the same kind of number. While the prior is untrustworthy the deploy half is the offline
    one, whose score is an ordinal proxy, so a predicted Σ against a measured one is withdrawn and
    the measured fused side survives; a trustworthy prior still gets to change the kernel set."""
    # The proxy's fantasy against the measured kernel — neither side wholly measured.
    monkeypatch.setattr(greedy, "_price_graph", lambda *_a, **_kw: greedy.Price(0.00039, measured=False))
    monkeypatch.setattr(greedy, "_price_op_leaf", lambda *_a, **_kw: greedy.Price(49.7336, measured=True))
    point, splice = _kernel_set_fork()

    pick = greedy.greedy_decide(prior=_ProxyPrior(trustworthy=trustworthy))(point)

    assert (pick is splice) is takes_the_cut
    if not takes_the_cut:
        assert leaf_knobs(pick).get("PLACE") is None, "the withdrawn class must not reach the ranking either"


@pytest.mark.parametrize(
    ("fused_measured", "fragments_measured", "takes_the_cut"),
    [(True, True, True), (False, False, True), (True, False, False)],
    ids=["both-measured", "both-predicted", "mixed"],
)
def test_a_kernel_set_change_needs_one_kind_of_number_not_a_trusted_prior(
    monkeypatch, fused_measured, fragments_measured, takes_the_cut
) -> None:
    """The competence check is on the MIX, not on the change. Two measured Σ are two real
    latencies and need no trusted model. Two predicted Σ are one proxy read against itself — the
    ordinary unmeasured pick, and the only thing that can cut a fused kernel on a machine where
    nothing was ever measured. Only the mixed comparison lets an arbitrary magnitude decide, so
    only it withdraws."""
    monkeypatch.setattr(greedy, "_price_graph", lambda *_a, **_kw: greedy.Price(12.5, measured=fragments_measured))
    monkeypatch.setattr(greedy, "_price_op_leaf", lambda *_a, **_kw: greedy.Price(49.7336, measured=fused_measured))
    point, splice = _kernel_set_fork()

    pick = greedy.greedy_decide(prior=_ProxyPrior(trustworthy=False))(point)

    assert (pick is splice) is takes_the_cut


def test_a_disqualified_fused_side_loses_to_a_predicted_cut(monkeypatch) -> None:
    """The competence check and the measured disqualification compose. An ``inf`` keep-fused Σ is
    an ELIMINATION — the tune watched every variant of one of its kernels hang — and an
    elimination has no magnitude for the offline proxy to corrupt, so it needs no trusted ranker:
    any finite alternative beats it. Withdrawing the splices here would deploy the very kernel the
    measurement condemned."""
    monkeypatch.setattr(greedy, "_price_graph", lambda *_a, **_kw: greedy.Price(0.00039, measured=False))
    monkeypatch.setattr(greedy, "_price_op_leaf", lambda *_a, **_kw: greedy.Price(math.inf, measured=True))
    point, splice = _kernel_set_fork()

    assert greedy.greedy_decide(prior=_ProxyPrior(trustworthy=False))(point) is splice


def test_price_total_is_measured_only_when_every_summand_is() -> None:
    """The Σ is what gets compared, so one prediction anywhere makes the whole sum a prediction."""
    assert greedy.Price.total([greedy.Price(1.0, True), greedy.Price(2.0, True)]) == greedy.Price(3.0, True)
    assert greedy.Price.total([greedy.Price(1.0, True), greedy.Price(2.0, False)]) == greedy.Price(3.0, False)


# ---------------------------------------------------------------------------
# _price_kernel — the price memo must share across identically computing kernels.
# ---------------------------------------------------------------------------


def test_price_memo_keys_on_exact_identity_not_the_term_hash(monkeypatch) -> None:
    """Pricing a fused matmul chain probes its cut pieces, and mirror pieces (a depth-i prefix
    cone vs a depth-i suffix cone) are the same computation spelled through different term-axis
    ranges: their ``cache_key``s all differ while the α-invariant exact identity unifies them.
    The memo must key on the identity — re-keying it on the term hash re-prices every mirror
    piece and this cardinality gap closes."""
    from emmy.compiler.context import Context
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.frontend.ir import MatmulOp
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline
    from emmy.compiler.pipeline.search.db import SearchDB
    from emmy.compiler.pipeline.search.prior.fallback import FallbackPrior

    # The per-test prior file is empty, so the loaded prior is untrustworthy and greedy withdraws
    # the splices — no probe would fire. This test is about the price memo's KEY, so it hands
    # greedy a prior allowed to price a kernel-set change.
    monkeypatch.setattr(FallbackPrior, "trustworthy", property(lambda _self: True))
    identity_keys, memo_keys, calls = set(), set(), []
    orig = greedy._price_kernel

    def spy(graph, nid, ctx, prior, memo, db=None, decisions=None):
        op = graph.nodes[nid].op
        calls.append(nid)
        identity_keys.add(op.identity_key(structural=False, with_io=True, with_knobs=True))
        out = orig(graph, nid, ctx, prior, memo, db, decisions)
        memo_keys.update(memo)
        return out

    monkeypatch.setattr(greedy, "_price_kernel", spy)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (16, 32), "f16"), node_id="x")
    prev = "x"
    for i in range(4):
        g.add_node(InputOp(), [], Tensor(f"w{i}", (32, 32), "f16"), node_id=f"w{i}")
        g.add_node(MatmulOp(), [prev, f"w{i}"], Tensor(f"o{i}", (16, 32), "f16"), node_id=f"o{i}")
        prev = f"o{i}"
    g.inputs, g.outputs = ["x"] + [f"w{i}" for i in range(4)], [prev]
    Pipeline.build(TILE_PASSES).run(g, ctx=Context.from_target((12, 0)), db=SearchDB())

    assert identity_keys, "the chain must offer structural forks whose pricing probes fire"
    assert len(identity_keys) < len(calls), "mirror cut pieces must unify under the exact identity"
    assert memo_keys == identity_keys, "the memo must key on the exact identity"
