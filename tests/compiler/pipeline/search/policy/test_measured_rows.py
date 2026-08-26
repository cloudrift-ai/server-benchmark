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

from emmy.compiler.pipeline.search.db import PerfStats
from emmy.compiler.pipeline.search.policy.mcts import SearchNode, TuningSearch

REGIME = {"H_cc": 12.0, "H_opt": 3.0}

# Two kernels a cross-CTA split minted: each carries its OWN structural stamps (the identity
# strategy stamps a minted piece at birth) and its own schedule decisions.
MAIN = {"S_n_load": 2.0, "S_n_accum": 1.0, "WORK": "w4x2", "REDUCE": "split4"}
COMBINE = {"S_n_load": 1.0, "S_n_accum": 0.0, "WORK": "", "REDUCE": ""}


def _stats(us: float, *, n: int = 32, var: float = 0.5) -> PerfStats:
    return PerfStats(median=us, min=us, max=us, mean=us, variance=var, n_samples=n)


def _search(**kw) -> TuningSearch:
    return TuningSearch(base_knobs={**REGIME, "S_n_load": 2.0}, **kw)


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
