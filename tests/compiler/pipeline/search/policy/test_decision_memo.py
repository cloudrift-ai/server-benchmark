"""Greedy's decision memo — one flatten-and-score per (schedule pool, blocklist) state.

What these pin: (a) SHARING — the second same-shape kernel in one compile replays the first's
decision by tree descent (``_find_decided_leaf``) and deploys the identical row; (b) SCOPE — the
memo keys on the pool key plus the node's blocklist CONTENT, so a validate-retry with a blocked
tile is a different decision, and a non-``TileOp`` fork never enters the memo; (c) the memo is
greedy-internal — nothing rides the shared ``SessionCache``, so MCTS keeps exploring."""

from __future__ import annotations

from types import SimpleNamespace

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.knob import family_of, tuning_knob_items
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.policy import greedy
from emmy.compiler.pipeline.search.policy.greedy import greedy_decide


def _matmul_graph(n: int = 1) -> Graph:
    """``n`` independent same-shape matmuls (distinct inputs, so fusion cannot merge them)."""
    g = Graph()
    outs = []
    for i in range(n):
        g.add_node(InputOp(), [], Tensor(f"a{i}", (1, Dim(64), Dim(64))), node_id=f"a{i}")
        g.add_node(InputOp(), [], Tensor(f"b{i}", (Dim(64), Dim(64))), node_id=f"b{i}")
        g.add_node(MatmulOp(), [f"a{i}", f"b{i}"], Tensor(f"c{i}", (1, Dim(64), Dim(64))), node_id=f"c{i}")
        outs.append(f"c{i}")
    g.inputs, g.outputs = [n_ for i in range(n) for n_ in (f"a{i}", f"b{i}")], outs
    return g


class _CountingPrior:
    """A bare ``mean_scores`` prior (the tests/custom-callers arm): constant scores, so the pick
    falls to ``canonical_row_key`` — deterministic — while counting how often the schedule fork
    class was actually scored."""

    def __init__(self) -> None:
        self.schedule_scores = 0

    def mean_scores(self, rows: list[dict]) -> list[float]:
        if rows and any("TILE" in family_of(k) for k in rows[0]):
            self.schedule_scores += 1
        return [0.0] * len(rows)


def _scheduled_rows(graph: Graph) -> list[tuple]:
    from emmy.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

    return [tuple(tuning_knob_items(n.op.knobs)) for n in graph.nodes.values() if isinstance(n.op, TileOp)]


def test_second_same_shape_kernel_replays_the_first_decision() -> None:
    prior = _CountingPrior()
    replays: list[object] = []
    real = greedy._find_decided_leaf

    def counting(options, want):
        found = real(options, want)
        replays.append(found)
        return found

    greedy._find_decided_leaf, orig = counting, greedy._find_decided_leaf
    try:
        g, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(
            _matmul_graph(n=2), greedy_decide(prior=prior)
        )
    finally:
        greedy._find_decided_leaf = orig
    rows = _scheduled_rows(g)
    assert len(rows) == 2 and rows[0] == rows[1], "both kernels must deploy the identical row"
    assert prior.schedule_scores == 1, "the second schedule fork must replay, not re-score"
    assert [r for r in replays if r is not None], "the replay must go through the tree descent"


def test_golden_probe_decides_without_flattening(monkeypatch) -> None:
    """A golden MATCH on a schedule fork deploys through the pool-row probe: the recorded row is
    matched against the raw pool rows off the tree's root branch, the pick descends to its one
    leaf, and the flatten-and-score never runs. DRIFT / GAP keep the old fall-through (pass 1)."""
    from dataclasses import replace

    # Pass 1 — an index that matches nothing: the probe consults (recording the fork's ShapeKey),
    # records GAP, and falls through to the scored path.
    seen_keys: list = []

    class _Spy(dict):
        def get(self, key, default=None):
            seen_keys.append(key)
            return default

    monkeypatch.setattr(greedy, "_golden_evidence_index", lambda ctx: _Spy(nonempty=1))
    flattens: list[int] = []
    real_flatten = greedy.flatten_leaves
    monkeypatch.setattr(greedy, "flatten_leaves", lambda opts: (flattens.append(1), real_flatten(opts))[1])

    ctx1 = replace(Context.from_target((12, 0)), compile_flags="")  # the -O3 regime the golden tier requires
    prior1 = _CountingPrior()
    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx1).resolve(_matmul_graph(), greedy_decide(prior=prior1))
    assert seen_keys, "the probe must consult the golden index off the raw pool rows"
    assert prior1.schedule_scores == 1, "a GAP falls through to the scored path"
    pass1_flattens = len(flattens)

    # Pass 2 — a golden recording one of the pool's own rows: MATCH, no flatten, no scoring.
    (pool,) = ctx1.session_cache._store.values()
    row = next(r for r in pool.rows if r.get("TILE"))
    gold = SimpleNamespace(knobs={k: v for k, v in row.items() if not str(k).startswith("S_")}, emmy_us=7.5, name="stub.golden")
    monkeypatch.setattr(greedy, "_golden_evidence_index", lambda ctx: {seen_keys[0]: [gold]})
    flattens.clear()
    ctx2 = replace(Context.from_target((12, 0)), compile_flags="")
    prior2 = _CountingPrior()
    records: list[dict] = []
    with greedy.golden_audit(records):
        g2, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx2).resolve(_matmul_graph(), greedy_decide(prior=prior2))
    assert prior2.schedule_scores == 0, "a golden MATCH must never reach the scoring pass"
    assert len(flattens) == pass1_flattens - 1, "the covered schedule fork must not flatten"
    assert [r["verdict"] for r in records] == ["MATCH"] and records[0]["n_rows"] == len(pool.rows)
    assert _scheduled_rows(g2) == [tuple(tuning_knob_items(gold.knobs))], "the deployed row must realize the golden"


def test_decision_key_separates_blocklist_states_and_skips_foreign_roots() -> None:
    from emmy.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

    tile = TileOp()
    rule = SimpleNamespace(pass_=SimpleNamespace(name="lowering/tile"), name="020_schedule")

    def fp(op, node_id="c0"):
        return SimpleNamespace(root_op=op, match=SimpleNamespace(rule=rule), node_id=node_id)

    clean = greedy._decision_key(fp(tile), None)
    assert clean is not None
    assert greedy._decision_key(fp(tile), {"c0": set()}) == clean, "an empty blocklist is the clean state"
    retried = greedy._decision_key(fp(tile), {"c0": {frozenset({("TILE", "f4")})}})
    assert retried != clean, "a blocked tile is a DIFFERENT decision — the validate-retry must re-decide"
    assert greedy._decision_key(fp(tile, node_id="c1"), {"c0": {frozenset({("TILE", "f4")})}}) == clean, (
        "another node's blocklist must not leak into this node's key"
    )
    assert greedy._decision_key(fp(InputOp()), None) is None, "only TileOp-rooted forks enter the memo"
