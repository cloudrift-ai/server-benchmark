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

from types import SimpleNamespace

import pytest

from emmy.compiler.pipeline.knob import stamp_schedule_families
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


def _ok_leaf(us: float, *, realized_knobs: dict | None, cuda_ops: int, cuda_knobs: list[dict] | None = None) -> SearchNode:
    node = SearchNode(candidate=object())
    node.realized_knobs = realized_knobs
    node.realized_cuda_ops = cuda_ops
    node.realized_cuda_knobs = cuda_knobs
    node.bench_status = "ok"
    node.bench_stats = PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=1)
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


def test_best_realized_does_not_fall_back_from_a_faster_unrepresentable_terminal() -> None:
    tree = SearchTree()
    tree.root.children = [
        _ok_leaf(18.0, realized_knobs={"WORK": "t64"}, cuda_ops=1),
        _ok_leaf(6.0, realized_knobs=None, cuda_ops=2),
    ]

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree

    assert search.best_realized() is None


def test_validated_structural_input_records_the_original_parent_linked_edge() -> None:
    route = {
        "WORK": "w2x1",
        "TILE": "mma_m16n8k16_f16_f32/f4x8/k8",
        "REDUCE": "g4k",
        "STAGE": "d1/smem-async",
        "RASTER": "",
    }
    tree = SearchTree()
    leaf = _ok_leaf(
        59.61,
        realized_knobs=None,
        cuda_ops=2,
        cuda_knobs=[
            {**route, "REDUCE": ""},
            {"WORK": "", "TILE": "", "REDUCE": "", "STAGE": "", "RASTER": ""},
        ],
    )
    leaf.visits = 1
    leaf.best_reward = 1.0 / 59.61
    tree.root.children = [leaf]
    tree.root.visits = 1
    tree.root.best_reward = leaf.best_reward

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree
    search._base_knobs = {"H_opt": 3.0, "S_loop": 1.0}

    assert search.best_realized() is None
    assert search.best_realized(validated_input_route=route) == (stamp_schedule_families(route), 59.61, 2, True)
    rows = search._collect_node_records(
        context_key="cc89/o3",
        op_sig="original-loop",
        gpu="NVIDIA GeForce RTX 4090",
        run_id="proposal-run",
        validated_input_route=route,
    )

    assert len(rows) == 2
    parent = next(row for row in rows if row.parent_key is None)
    branch = next(row for row in rows if row.parent_key is not None)
    assert parent.features == {"H_opt": 3.0, "S_loop": 1.0}
    assert branch.parent_key == parent.node_key
    assert branch.features == {"H_opt": 3.0, "S_loop": 1.0, **stamp_schedule_families(route)}
    assert branch.value_us == pytest.approx(59.61)
    assert branch.is_leaf


def test_best_realized_returns_the_fastest_terminal_with_its_structural_replay_row() -> None:
    tree = SearchTree()
    row = {"WORK": "w1x1", "TILE": "mma_m16n8k16_f16_f32/f1x4/k8", "REDUCE": "g8k", "STAGE": "d1/smem"}
    route = SearchNode(candidate=SimpleNamespace(resolved_knobs=row), parent=tree.root)
    fast = _ok_leaf(
        6.0,
        realized_knobs=None,
        cuda_ops=2,
        cuda_knobs=[
            {**row, "REDUCE": ""},
            {"WORK": "", "TILE": "", "REDUCE": "", "STAGE": "", "RASTER": ""},
        ],
    )
    fast.parent = route
    route.children = [fast]
    tree.root.children = [_ok_leaf(18.0, realized_knobs={"WORK": "t64"}, cuda_ops=1), route]

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree

    assert search.best_realized() == (stamp_schedule_families(row), 6.0, 2, True)


def test_best_realized_rejects_a_structural_parent_that_names_a_different_child_schedule() -> None:
    tree = SearchTree()
    row = {"WORK": "w4x2", "TILE": "mma_m16n8k16_f16_f32/f1x2/k8", "REDUCE": "g8k", "STAGE": "d1/smem"}
    route = SearchNode(candidate=SimpleNamespace(resolved_knobs=row), parent=tree.root)
    fast = _ok_leaf(
        6.0,
        realized_knobs=None,
        cuda_ops=2,
        cuda_knobs=[
            {**row, "WORK": "w1x2", "REDUCE": ""},
            {"WORK": "", "TILE": "", "REDUCE": "", "STAGE": "", "RASTER": ""},
        ],
    )
    fast.parent = route
    route.children = [fast]
    tree.root.children = [_ok_leaf(18.0, realized_knobs={"WORK": "t64"}, cuda_ops=1), route]

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree

    assert search.best_realized() is None


def test_best_realized_uses_a_compatible_multi_cuda_placement_route() -> None:
    tree = SearchTree()
    tree.root.children = [
        _ok_leaf(
            6.0,
            realized_knobs={"WORK": "w1x1", "TILE": "mma_m16n8k16_f16_f32/f1x4/k8", "PLACE@a": "cut"},
            cuda_ops=2,
            cuda_knobs=[{"WORK": "w1x1"}, {"WORK": ""}],
        )
    ]

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree

    assert search.best_realized() == ({"PLACE@a": "cut"}, 6.0, 2, True)


def test_best_realized_keeps_only_the_routing_row_for_a_placement_cut() -> None:
    tree = SearchTree()
    route = SearchNode(candidate=SimpleNamespace(resolved_knobs={"PLACE@a": "cut", "WORK": "w1x1"}), parent=tree.root)
    fast = _ok_leaf(6.0, realized_knobs=None, cuda_ops=2, cuda_knobs=[{"WORK": "w1x1"}, {"WORK": ""}])
    fast.parent = route
    route.children = [fast]
    tree.root.children = [route]

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree

    assert search.best_realized() == ({"PLACE@a": "cut"}, 6.0, 2, True)


def test_best_realized_returns_an_ordinary_one_kernel_row() -> None:
    tree = SearchTree()
    tree.root.children = [_ok_leaf(6.0, realized_knobs={"WORK": "t64"}, cuda_ops=1)]

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree

    assert search.best_realized() == ({"WORK": "t64"}, 6.0, 1, False)
