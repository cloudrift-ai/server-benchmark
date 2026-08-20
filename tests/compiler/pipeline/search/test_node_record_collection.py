"""The post-search node walk survives a leaf that bench-failed before realizing its knobs.

``observe`` sets ``bench_stats`` / ``bench_status`` alongside ``realized_knobs``, but a
variant killed in the run stage — a GPU-time timeout, a missing bench input — records the
failure without ever realizing a knob set. ``_collect_node_records`` keys every row it emits
on those knobs, so such a leaf has nothing to key on and must be skipped rather than
crashing the walk.

This is not hypothetical: it ended a 13-target tune of Qwen3.8's full-attention layer on an
RTX 4090 at target 12, after six run-stage timeouts had produced exactly this node.
"""

from __future__ import annotations

from emmy.compiler.pipeline.search.db import PerfStats
from emmy.compiler.pipeline.search.policy.mcts import SearchNode, SearchTree, TuningSearch


def _bench_fail_leaf(*, realized_knobs: dict | None) -> SearchNode:
    """A leaf whose bench failed, with or without a realized knob set."""
    node = SearchNode(candidate=object())
    node.visits = 1
    node.best_reward = 0.0  # a failed bench anchors no value
    node.realized_knobs = realized_knobs
    node.bench_status = "bench_fail"
    node.bench_stats = PerfStats(median=2_000_000.0, min=2_000_000.0, max=2_000_000.0, mean=2_000_000.0, variance=0.0, n_samples=1)
    return node


def _collect(tree: SearchTree):
    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree
    return search._collect_node_records(context_key="cc89/o1", op_sig="k_linear", gpu="NVIDIA GeForce RTX 4090")


def test_unrealized_bench_fail_leaf_is_skipped_not_keyed() -> None:
    """The walk completes and emits nothing for a leaf with no knobs to key on."""
    tree = SearchTree()
    tree.root.children = [_bench_fail_leaf(realized_knobs=None)]

    assert _collect(tree) == []


def test_realized_bench_fail_leaf_still_records_its_sentinel() -> None:
    """The guard is narrow: a bench failure that did realize its knobs keeps its row."""
    tree = SearchTree()
    tree.root.children = [_bench_fail_leaf(realized_knobs={"WORK": "w2x2", "TILE": "mma_m16n8k16_f16_f32/f4x4/k2"})]

    [row] = _collect(tree)
    assert row.status == "bench_fail"
    assert row.value_us == 2_000_000.0
    assert row.features == {"WORK": "w2x2", "TILE": "mma_m16n8k16_f16_f32/f4x4/k2"}


def test_a_healthy_sibling_survives_an_unrealized_failure() -> None:
    """One unkeyable leaf does not cost the rest of the tree its records."""
    tree = SearchTree()
    good = _bench_fail_leaf(realized_knobs={"WORK": "w4x2"})
    tree.root.children = [_bench_fail_leaf(realized_knobs=None), good]

    rows = _collect(tree)
    assert [row.features for row in rows] == [{"WORK": "w4x2"}]
