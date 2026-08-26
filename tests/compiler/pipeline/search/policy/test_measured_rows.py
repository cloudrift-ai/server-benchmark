"""What a benched terminal is recorded as.

A terminal is a Σ over the kernels it lowered to. When a structural fork made it several,
no single row can carry that total, so the measurement is recorded once per KERNEL — each
row carrying that kernel's own knobs, its own µs, and its own shape as its identity, so the
same kernel measured anywhere else shares its key and its candidate pool.

Covers the places the rule has to hold together: what ``observe`` stashes on the terminal's
node, what the reservoir walk (``_collect_rows``) trains on, and what the node walk
(``_collect_node_records``) stores.
"""

from __future__ import annotations

from types import SimpleNamespace

from emmy.compiler.pipeline.passes.identity import kernel_sig
from emmy.compiler.pipeline.search.db import PerfStats, node_key
from emmy.compiler.pipeline.search.policy.mcts import SearchNode, SearchTree, TuningSearch

CTX = "cc120/o3"
GPU = "NVIDIA GeForce RTX 5090"
REGIME = {"H_cc": 12.0, "H_opt": 3.0}

# Two kernels a cross-CTA split minted: each carries its OWN structural stamps (the identity
# strategy stamps a minted piece at birth) and its own schedule decisions.
MAIN = {"S_n_load": 2.0, "S_n_accum": 1.0, "WORK": "w4x2", "REDUCE": "split4"}
COMBINE = {"S_n_load": 1.0, "S_n_accum": 0.0, "WORK": "", "REDUCE": ""}


def _stats(us: float, *, n: int = 32, var: float = 0.5) -> PerfStats:
    return PerfStats(median=us, min=us, max=us, mean=us, variance=var, n_samples=n)


def _search(**kw) -> TuningSearch:
    return TuningSearch(base_knobs={**REGIME, "S_n_load": 2.0}, **kw)


def _multi_kernel_leaf(*, rows, total_us: float, status: str = "ok") -> SearchNode:
    """A benched terminal that lowered to several kernels — no single row of its own."""
    node = SearchNode(candidate=SimpleNamespace(fork=None, resolved_knobs=None))
    node.visits = 1
    node.best_reward = (1.0 / total_us) if status == "ok" else 0.0
    node.realized_knobs = None
    node.realized_cuda_ops = len(rows)
    node.kernel_rows = rows
    node.bench_status = status
    node.bench_stats = _stats(total_us)
    return node


def _collect_nodes(tree: SearchTree, **kw):
    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree
    search._base_knobs = dict(REGIME)
    return search._collect_node_records(context_key=CTX, op_sig="offer-site", gpu=GPU, **kw)


def _collect_reservoir(tree: SearchTree):
    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree
    search._base_knobs = dict(REGIME)
    return search._collect_rows()


# ---------------------------------------------------------------------------
# observe — the terminal's per-kernel receipts reach its node
# ---------------------------------------------------------------------------


def test_observe_stashes_each_kernels_own_row_under_the_run_regime() -> None:
    search = _search()
    token = SearchNode(candidate=None)

    search.observe(token, _stats(12.0), "ok", kernels=[(MAIN, _stats(10.0), "ok"), (COMBINE, _stats(2.0), "ok")])

    assert [feats for feats, _stats_, _status in token.kernel_rows] == [{**REGIME, **MAIN}, {**REGIME, **COMBINE}]
    assert [s.median for _f, s, _st in token.kernel_rows] == [10.0, 2.0]
    assert [st for _f, _s, st in token.kernel_rows] == ["ok", "ok"]


def test_observe_leaves_kernel_rows_unset_when_the_bench_handed_over_none() -> None:
    """A cache hit / stub path with no per-kernel receipts must not fabricate rows."""
    search = _search()
    token = SearchNode(candidate=None)

    search.observe(token, _stats(12.0), "ok")

    assert token.kernel_rows is None


# ---------------------------------------------------------------------------
# the reservoir walk — what the prior trains on
# ---------------------------------------------------------------------------


def test_reservoir_trains_on_each_kernel_not_on_the_sum() -> None:
    tree = SearchTree()
    tree.root.children = [_multi_kernel_leaf(rows=[({**REGIME, **MAIN}, _stats(10.0), "ok"), ({**REGIME, **COMBINE}, _stats(2.0), "ok")], total_us=12.0)]

    rows = _collect_reservoir(tree)

    assert sorted(us for _knobs, us in rows) == [2.0, 10.0]
    assert 12.0 not in [us for _knobs, us in rows]
    assert {"WORK": knobs["WORK"] for knobs, _us in rows} != {}


def test_reservoir_keeps_the_single_kernel_terminals_value_of_position_label() -> None:
    """One kernel means the terminal's own row earned the whole measurement — unchanged."""
    node = SearchNode(candidate=SimpleNamespace(fork=None, resolved_knobs=None))
    node.visits = 1
    node.best_reward = 1.0 / 7.0
    node.realized_knobs = {**REGIME, **MAIN}
    node.realized_cuda_ops = 1
    node.kernel_rows = [({**REGIME, **MAIN}, _stats(7.0), "ok")]
    node.bench_status = "ok"
    node.bench_stats = _stats(7.0)
    tree = SearchTree()
    tree.root.children = [node]

    assert _collect_reservoir(tree) == [({**REGIME, **MAIN}, 7.0)]


def test_reservoir_weights_a_repeated_piece_once() -> None:
    """The combine kernel's own knobs don't vary with the fork being explored, so every
    variant that mints it re-emits the same row. Undeduped it would weight that one config
    by the number of variants."""
    tree = SearchTree()
    tree.root.children = [
        _multi_kernel_leaf(rows=[({**REGIME, **MAIN}, _stats(10.0), "ok"), ({**REGIME, **COMBINE}, _stats(2.0), "ok")], total_us=12.0),
        _multi_kernel_leaf(rows=[({**REGIME, **MAIN, "WORK": "w2x2"}, _stats(9.0), "ok"), ({**REGIME, **COMBINE}, _stats(2.5), "ok")], total_us=11.5),
    ]

    rows = _collect_reservoir(tree)

    combines = [us for knobs, us in rows if not knobs["WORK"]]
    assert len(combines) == 1, "the repeated piece is one config, not one row per variant that minted it"
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# the node walk — what the store keeps
# ---------------------------------------------------------------------------


def test_node_rows_are_one_per_kernel_keyed_by_that_kernels_own_shape() -> None:
    tree = SearchTree()
    tree.root.children = [_multi_kernel_leaf(rows=[({**REGIME, **MAIN}, _stats(10.0), "ok"), ({**REGIME, **COMBINE}, _stats(2.0), "ok")], total_us=12.0)]

    rows = sorted(_collect_nodes(tree), key=lambda r: r.value_us)

    assert [r.value_us for r in rows] == [2.0, 10.0]
    assert [r.op_sig for r in rows] == [kernel_sig(COMBINE), kernel_sig(MAIN)]
    assert [r.features for r in rows] == [{**REGIME, **COMBINE}, {**REGIME, **MAIN}]
    assert all(r.parent_key is None and r.depth == 0 for r in rows), "a measured kernel is not a position in this tree"
    assert all(r.is_leaf and r.status == "ok" and r.visits == 1 for r in rows)
    assert [r.n_samples for r in rows] == [32, 32]


def test_the_same_kernel_from_two_sites_lands_on_one_key() -> None:
    """The whole point of keying by the kernel's own shape: a piece minted here and the same
    kernel tuned as its own enrolled target are one row, not two."""
    tree = SearchTree()
    tree.root.children = [_multi_kernel_leaf(rows=[({**REGIME, **MAIN}, _stats(10.0), "ok"), ({**REGIME, **COMBINE}, _stats(2.0), "ok")], total_us=12.0)]

    [combine] = [r for r in _collect_nodes(tree) if r.op_sig == kernel_sig(COMBINE)]

    assert combine.node_key == node_key(CTX, GPU, kernel_sig(COMBINE), {**REGIME, **COMBINE})


def test_a_terminal_whose_kernels_agree_still_records_per_kernel() -> None:
    """The old rule fired on a knob CONFLICT, so two kernels that happened to agree got one
    row carrying the Σ. Kernel count is what decides, not whether the pieces disagree."""
    node = _multi_kernel_leaf(rows=[({**REGIME, **MAIN}, _stats(10.0), "ok"), ({**REGIME, **MAIN}, _stats(2.0), "ok")], total_us=12.0)
    node.realized_knobs = {**REGIME, **MAIN}  # no conflict, so a merged row exists
    tree = SearchTree()
    tree.root.children = [node]

    rows = _collect_nodes(tree)

    assert [r.value_us for r in rows] == [2.0], "one row per kernel, deduped by identity — never the 12.0 Σ"


def test_a_stampless_kernel_is_skipped_rather_than_keyed_to_an_empty_shape() -> None:
    """No stamps means no shape; digesting the empty set would collide every unstamped
    auxiliary in the store onto one row."""
    tree = SearchTree()
    tree.root.children = [_multi_kernel_leaf(rows=[({**REGIME, **MAIN}, _stats(10.0), "ok"), ({**REGIME, "WORK": ""}, _stats(2.0), "ok")], total_us=12.0)]

    rows = _collect_nodes(tree)

    assert [r.op_sig for r in rows] == [kernel_sig(MAIN)]


def test_a_failed_multi_kernel_terminal_records_nothing() -> None:
    """The watchdog pins one sentinel on every kernel, so per-piece fail rows would each
    carry a number no kernel measured. The failure is the terminal's."""
    sentinel = _stats(2_000_000.0)
    tree = SearchTree()
    tree.root.children = [
        _multi_kernel_leaf(
            rows=[({**REGIME, **MAIN}, sentinel, "bench_fail"), ({**REGIME, **COMBINE}, sentinel, "bench_fail")],
            total_us=2_000_000.0,
            status="bench_fail",
        )
    ]

    assert _collect_nodes(tree) == []


def test_the_fan_out_leaves_an_ancestors_own_row_alone() -> None:
    """Per-kernel rows are parentless, so the branch above keeps its Σ value-of-position and
    the monotone parent <= child invariant the walk asserts still holds."""
    branch = SearchNode(candidate=SimpleNamespace(fork=None, resolved_knobs=None))
    branch.visits = 1
    branch.best_reward = 1.0 / 12.0
    leaf = _multi_kernel_leaf(rows=[({**REGIME, **MAIN}, _stats(10.0), "ok"), ({**REGIME, **COMBINE}, _stats(2.0), "ok")], total_us=12.0)
    leaf.parent = branch
    branch.children = [leaf]
    tree = SearchTree()
    tree.root.children = [branch]

    rows = _collect_nodes(tree)
    branches = [r for r in rows if not r.is_leaf]

    assert [r.value_us for r in branches] == [12.0]
    assert branches[0].op_sig == "offer-site", "the branch is still a position in the offer site's tree"
