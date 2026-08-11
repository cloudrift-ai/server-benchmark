"""Two-level autotuning: the inner separable per-op reward.

The inner search tunes each post-fusion kernel in its own single-node slice.
These tests pin the separability properties with a fake counting backend (no
GPU): benches scale as ``Σ_k n_k`` not the product, the bests land in the DB,
re-runs are idempotent under the effort gate, a higher patience re-deepens
only under-tuned ops, and a kernel shared by two terminals is tuned once.

Target is forced to sm_80 so lowering is deterministic and GPU-independent —
the fake backend never launches anything, it just hands back per-launch
latencies keyed off each CudaOp's structural key.
"""

from __future__ import annotations

import zlib

import pytest

from emmy.compiler.backend.base import BenchmarkResult, LaunchTime
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline, TuningSearch
from emmy.compiler.pipeline.search.db import SearchDB
from emmy.compiler.pipeline.search.slice import single_node_graph
from emmy.compiler.pipeline.search.two_level import (
    LOWERING_PASSES,
    InnerReward,
    OpResult,
)
from tests.compiler.helpers import drain_tune, run_inner_reward, run_two_level

# Moderate patience: each kernel explores several variants then stops on
# stagnation (the fake backend gives a stable but arbitrary per-variant
# signal). Enough to exercise "Σ_k n_k, not the product" without paying for a
# full tree drain — exhaustion (∞ effort) is covered directly in test_db.py.
_PATIENCE = 8


def test_searched_winner_requires_one_post_fusion_and_one_cuda_kernel() -> None:
    one = OpResult(
        name="k",
        op_key="key",
        best_us=4.0,
        searched_knobs={"TILE": "f2x2"},
        searched_us=5.0,
        searched_cuda_ops=1,
    )
    assert InnerReward(total_us=4.0, ok=True, per_op=[one]).searched_winner() == ({"TILE": "f2x2"}, 5.0)

    multi_cuda = OpResult(**{**one.__dict__, "searched_cuda_ops": 2})
    assert InnerReward(total_us=4.0, ok=True, per_op=[multi_cuda]).searched_winner() is None
    assert InnerReward(total_us=8.0, ok=True, per_op=[one, one]).searched_winner() is None


@pytest.fixture(autouse=True)
def _force_target(monkeypatch, tmp_path):
    from emmy.compiler import target as target_mod

    # Isolate the online-prior checkpoint: ``run_two_level_tune`` trains and
    # checkpoints the global prior, and these fake-backend rows must never
    # pollute the host's real ``~/.cache/emmy/online.json``.
    monkeypatch.setenv("EMMY_ONLINE_FILE", str(tmp_path / "prior.json"))
    target_mod.set_target((8, 0))
    yield
    target_mod.set_target(None)


class _CountingBackend:
    """Fake backend: counts ``benchmark`` calls and returns a deterministic
    per-CudaOp latency keyed off the op's structural key, so the inner search
    sees real signal and a unique best without touching a GPU."""

    name = "cuda"
    bench_run_timeout_s = 1.0

    def __init__(self) -> None:
        self.calls = 0

    def benchmark(self, graph, num_iters="auto") -> BenchmarkResult:  # noqa: ARG002
        self.calls += 1
        cuda = [n for n in graph.nodes.values() if isinstance(n.op, CudaOp)]
        per: list[LaunchTime] = []
        for i, n in enumerate(cuda):
            # crc32, not hash(): str hashes are salted per process (PYTHONHASHSEED),
            # which made the MCTS path — and the bench counts the separability test
            # bounds — vary run to run.
            us = 1.0 + (zlib.crc32(n.op.cache_key().encode()) % 100)
            per.append(LaunchTime(idx=i, kernel_name=getattr(n.op, "kernel_name", "k"), time_ms=us / 1000.0, samples=(us / 1000.0,)))
        return BenchmarkResult(time_ms=sum(p.time_ms for p in per), num_launches=len(per), per_launch=per)

    async def benchmark_async(self, graph, num_iters="auto") -> BenchmarkResult:
        # The two-level driver benches through the async path (``Pipeline.tune_async``);
        # the fake has no real I/O, so delegate to the deterministic sync bench. The
        # signature mirrors ``benchmark`` exactly (no ``nvcc_flags``) so the -O3
        # re-bench rejects the same way → identical bench counts to the sync path.
        return self.benchmark(graph, num_iters=num_iters)


class _O3CountingBackend(_CountingBackend):
    """Counting backend whose async bench ALSO accepts ``nvcc_flags`` — so the -O3
    re-bench path runs instead of rejecting, for the O3-node-row integration test."""

    def __init__(self) -> None:
        super().__init__()
        self.o3_calls = 0

    async def benchmark_async(self, graph, num_iters="auto", nvcc_flags=None) -> BenchmarkResult:
        if nvcc_flags is not None:
            self.o3_calls += 1
        return self.benchmark(graph, num_iters=num_iters)


def _matmul(g: Graph, prefix: str, M: int, K: int, N: int) -> str:
    a, b, c = f"{prefix}a", f"{prefix}b", f"{prefix}c"
    g.add_node(InputOp(), [], Tensor(a, (M, K)), node_id=a)
    g.add_node(InputOp(), [], Tensor(b, (K, N)), node_id=b)
    g.add_node(MatmulOp(), [a, b], Tensor(c, (M, N)), node_id=c)
    return c


def _two_distinct_matmuls() -> Graph:
    g = Graph()
    c1 = _matmul(g, "x", 64, 128, 48)
    c2 = _matmul(g, "y", 96, 64, 32)
    g.inputs = ["xa", "xb", "ya", "yb"]
    g.outputs = [c1, c2]
    return g


def _two_identical_matmuls() -> Graph:
    g = Graph()
    c1 = _matmul(g, "x", 64, 128, 48)
    c2 = _matmul(g, "y", 64, 128, 48)
    g.inputs = ["xa", "xb", "ya", "yb"]
    g.outputs = [c1, c2]
    return g


def _fuse(graph: Graph) -> Graph:
    return Pipeline.build(LOOP_PASSES).run(graph, db=SearchDB())


def _loop_ids(fused: Graph) -> list[str]:
    return [nid for nid, n in fused.nodes.items() if isinstance(n.op, LoopOp)]


def _tune_one_slice(fused: Graph, nid: str, patience: int) -> int:
    """Tune a single kernel's slice in isolation; return the bench count."""
    backend = _CountingBackend()
    sub = single_node_graph(fused, nid)
    pipeline = Pipeline.build(LOWERING_PASSES)
    search = TuningSearch(patience=patience)
    drain_tune(pipeline, sub, search=search, ctx=Context.from_target((8, 0)), backend=backend, db=SearchDB())
    return backend.calls


def test_inner_reward_is_separable_not_a_product() -> None:
    """Total benches across two kernels == n1 + n2 (per-op), never n1 * n2."""
    fused = _fuse(_two_distinct_matmuls())
    loops = _loop_ids(fused)
    assert len(loops) == 2

    n1 = _tune_one_slice(fused, loops[0], _PATIENCE)
    n2 = _tune_one_slice(fused, loops[1], _PATIENCE)
    assert n1 > 1 and n2 > 1, "kernels must have multiple variants to make the point"

    backend = _CountingBackend()
    db = SearchDB()
    reward = run_inner_reward(fused, ctx=Context.from_target((8, 0)), db=db, backend=backend, patience=_PATIENCE)

    # Separability: the shared run must not blow up to the cross-product
    # (n1 * n2 — the old whole-graph SP-MCTS bug this test guards against).
    # Per-op sharing through the DB perf cache is allowed — an already-measured
    # variant replays without a bench. The exact share count is sensitive to
    # MCTS exploration order: the
    # ``_CountingBackend`` fakes latency from ``crc32(Op.cache_key)``, so any
    # structural-digest perturbation (e.g. a Source-field rename) shifts the
    # path and the count by a few benches. The hard guarantee is the
    # cross-product upper bound; tighter ``n1+n2`` / ``max(n1,n2)`` bounds
    # are sanity checks with slack.
    assert backend.calls < n1 * n2, "separable sum must be below the cross-product"
    # Patience-noise slack: per-op MCTS can stretch its patience window by up
    # to a handful of benches when interleaved with another kernel's search
    # in the shared run (measured drift: 42 benches at n1+n2=32 after the
    # dpl_→emmy_ helper rename shifted the crc32 fake latencies).
    slack = max(12, (n1 + n2) // 4)
    assert backend.calls <= n1 + n2 + slack, f"expected ≤ {n1 + n2 + slack} (separable+slack) benches, got {backend.calls}"
    # Every kernel measured; total is the sum of the per-op bests. Two distinct
    # structural keys → two ``per_op`` entries, each at ``multiplicity=1``.
    assert reward.ok
    assert len(reward.per_op) == 2
    assert all(r.best_us is not None for r in reward.per_op)
    assert all(r.multiplicity == 1 for r in reward.per_op)
    assert reward.total_us == pytest.approx(sum(r.best_us for r in reward.per_op))


def test_inner_reward_records_nodes() -> None:
    """The post-search walk persists tree nodes to the ``node`` table — keyed by
    the run's context, one op_sig group per distinct kernel, with every non-null
    parent_key referencing a recorded node (valid ancestry edges) — and each row
    carries the label-quality columns: benched-descendant visits, the session
    run_id, a measurement timestamp, and per-leaf bench stats."""
    fused = _fuse(_two_distinct_matmuls())
    ctx = Context.from_target((8, 0))
    db = SearchDB()
    run_inner_reward(fused, ctx=ctx, db=db, backend=_CountingBackend(), patience=_PATIENCE, run_id="testrun")

    rows = db._conn.execute(
        "SELECT node_key, parent_key, context_key, op_sig, value_us, visits, is_leaf, n_samples, status, run_id, measured_at FROM node"
    ).fetchall()
    assert rows, "expected node rows recorded from the finished search trees"
    assert {r[2] for r in rows} == {ctx.structural_key()}  # all under the run's regime
    assert all(r[4] > 0 for r in rows)  # positive value-of-position
    assert len({r[3] for r in rows}) >= 2  # two distinct matmuls → ≥2 op_sig groups
    keys = {r[0] for r in rows}
    assert all(r[1] in keys for r in rows if r[1] is not None)  # parents reference recorded nodes
    assert all(r[5] > 0 for r in rows)  # every recorded node has a benched descendant
    assert all((r[8], r[9]) == ("ok", "testrun") for r in rows)  # clean benches, session-tagged
    assert all(r[10] for r in rows)  # measured_at stamped
    leaves = [r for r in rows if r[6] == 1]
    assert leaves and all(r[7] is not None for r in leaves)  # leaf rows carry their bench n_samples


def test_inner_reward_records_o3_regime_rows(monkeypatch) -> None:
    """With a backend whose bench accepts ``nvcc_flags`` (the -O3 re-bench path runs),
    the near-best configs' deployable re-benches land in the ``node`` table under
    their OWN context key — the tune context with the -O3 flags substituted — as
    parentless ``H_opt=3`` leaf rows, alongside (not colliding with) the -O1 tree."""
    from dataclasses import replace

    from emmy.compiler.pipeline.search.policy.mcts import O3_NVCC_FLAGS

    monkeypatch.setenv("EMMY_NVCC_FLAGS", "-Xcicc -O1")  # the tune regime; re-bench not skipped
    fused = _fuse(_two_distinct_matmuls())
    ctx = Context.from_target((8, 0))
    db = SearchDB()
    backend = _O3CountingBackend()
    run_inner_reward(fused, ctx=ctx, db=db, backend=backend, patience=_PATIENCE, run_id="o3run")

    assert backend.o3_calls > 0  # the re-bench path actually ran
    o3_key = replace(ctx, compile_flags=O3_NVCC_FLAGS).structural_key()
    rows = db._conn.execute("SELECT context_key, parent_key, is_leaf, features, run_id FROM node").fetchall()
    assert {r[0] for r in rows} == {ctx.structural_key(), o3_key}  # both regimes present, distinct keys
    o3_rows = [r for r in rows if r[0] == o3_key]
    assert o3_rows
    for _, parent_key, is_leaf, feats_json, run_id in o3_rows:
        assert parent_key is None and is_leaf == 1 and run_id == "o3run"
        assert '"H_opt": 3.0' in feats_json  # the deployable-regime stamp


def test_node_rows_are_fold_ready(monkeypatch) -> None:
    """End-to-end group-holdout readiness: after a real tune (O3 re-benches on),
    every node row carries its fold keys, fold-by-op keeps each op atomic —
    the -O1 tree AND its -O3 regime rows in one fold, parent edges resolving
    inside it — and fold-by-gpu puts this single-card run in one fold."""
    from emmy.compiler.pipeline.search.data import Dataset

    monkeypatch.setenv("EMMY_NVCC_FLAGS", "-Xcicc -O1")
    fused = _fuse(_two_distinct_matmuls())
    ctx = Context.from_target((8, 0))
    db = SearchDB()
    run_inner_reward(fused, ctx=ctx, db=db, backend=_O3CountingBackend(), patience=_PATIENCE, run_id="foldrun")

    rows = list(db.iter_nodes())
    assert rows
    assert all(r.op_sig and r.gpu and r.run_id for r in rows)  # every fold key populated

    by_op = Dataset.fold_node_rows(rows, by="op")
    assert len(by_op) >= 2  # two distinct matmuls → at least two op folds
    assert sum(len(v) for v in by_op.values()) == len(rows)  # a partition
    o3_folds = 0
    for fold in by_op.values():
        keys = {r.node_key for r in fold}
        assert all(r.parent_key in keys for r in fold if r.parent_key is not None)  # edges stay inside
        if len({r.context_key for r in fold}) == 2:  # -O1 tree + -O3 regime rows together
            o3_folds += 1
    assert o3_folds > 0

    by_gpu = Dataset.fold_node_rows(rows, by="gpu")
    assert len(by_gpu) == 1  # one card tuned → one gpu fold holding everything
    (gpu_fold,) = by_gpu.values()
    assert len(gpu_fold) == len(rows)


def test_inner_reward_rerun_is_replay_dominated() -> None:
    """A second pass at the same patience is replay-dominated and never regresses:
    the warm perf cache serves almost every terminal, so the rerun benches far
    fewer variants than the cold run, and the per-op best total only improves (or
    ties), never worsens.

    Two things changed vs the old idempotence invariant. Ranking moved from the
    priority-sorted enumeration to the ``Prior`` (``OfflinePrior`` cold), so the
    cold search walks a real prior-ranked frontier instead of finding the best at
    option-0; that frontier interacts with the cache's cross-op kernel sharing, so
    a warm rerun wanders into a handful of new frontier variants while replaying
    the rest. Those extra benches can only LOWER the per-op best — ``record_perf``
    keeps the minimum and ``best_per_op_time`` reads it — so ``second.total_us <=
    first.total_us`` always (it does not converge to the *same* total; it converges
    *downward*). The exact bench count is exploration-order-sensitive (same caveat
    as ``test_inner_reward_separability``), so only the two robust invariants are
    pinned."""
    fused = _fuse(_two_distinct_matmuls())
    db = SearchDB()
    ctx = Context.from_target((8, 0))

    cold_backend = _CountingBackend()
    first = run_inner_reward(fused, ctx=ctx, db=db, backend=cold_backend, patience=_PATIENCE)
    rerun_backend = _CountingBackend()
    second = run_inner_reward(fused, ctx=ctx, db=db, backend=rerun_backend, patience=_PATIENCE)

    # The DB's per-op best is monotone non-increasing — more benches never worsen it.
    assert second.total_us <= first.total_us + 1e-6, "rerun must not regress the per-op best total"
    # Warm rerun replays most terminals from the perf cache → fewer benches than cold.
    # (Only "fewer", not a fixed ratio: the exact count is exploration-order-
    # sensitive — see the docstring — so a `cold // 2`-style bound is not robust.)
    assert rerun_backend.calls < cold_backend.calls, "warm rerun must bench fewer variants than the cold run"


def test_inner_reward_deeper_patience_benches_new_variants() -> None:
    """A higher patience re-runs the search (never skipped) and reaches new
    variants the shallow pass never measured — those miss the perf cache and
    bench, while the already-measured ones replay for free."""
    fused = _fuse(_two_distinct_matmuls())
    db = SearchDB()
    ctx = Context.from_target((8, 0))

    run_inner_reward(fused, ctx=ctx, db=db, backend=_CountingBackend(), patience=1)

    deep_backend = _CountingBackend()
    run_inner_reward(fused, ctx=ctx, db=db, backend=deep_backend, patience=_PATIENCE)
    assert deep_backend.calls > 0, "a deeper pass must bench the new variants it reaches"


def test_inner_reward_shares_identical_kernel() -> None:
    """Two identical kernels in one terminal collapse to a single ``per_op``
    entry under one ``Op.cache_key`` with ``multiplicity=2``. The inner
    search runs once; the outer total still costs 2× the shared best so the
    outer MCTS reward stays bit-for-bit identical to the per-node-iterated
    formulation."""
    fused = _fuse(_two_identical_matmuls())
    loops = _loop_ids(fused)
    assert len(loops) == 2
    # Same body ⇒ same structural key.
    keys = {fused.nodes[nid].op.cache_key() for nid in loops}
    assert len(keys) == 1, "the two matmuls must share one structural key"

    single = _tune_one_slice(fused, loops[0], _PATIENCE)
    backend = _CountingBackend()
    reward = run_inner_reward(fused, ctx=Context.from_target((8, 0)), db=SearchDB(), backend=backend, patience=_PATIENCE)

    assert backend.calls == single, "shared kernel must bench once, not twice"
    assert len(reward.per_op) == 1, "identical kernels collapse to one per_op entry"
    assert reward.per_op[0].multiplicity == 2, "both node positions are counted"
    # Total weights the shared best by its multiplicity — both kernels still cost time.
    assert reward.total_us == pytest.approx(2 * reward.per_op[0].best_us)


def test_inner_reward_parallel_matches_serial(monkeypatch) -> None:
    """The core multi-GPU invariant: tuning the unique kernels concurrently across
    a pool of N device-pinned backends yields the SAME per-op bests and summed
    reward as the one-slot serial path. Each op's search is seeded by ``op_idx``
    (execution-order-independent) and the fake backend's latency keys off
    ``Op.cache_key`` (slot-independent), so completion order can't change the
    result. ``prior=None`` keeps this off the online-prior (catboost) path.

    The tile is pinned to per-cell (``EMMY_TILE=""``) so the matmul enumerates a
    single candidate: the tile move catalog (``search/space.py``) now offers ~20
    scalar tiles per contraction, and under the small ``_PATIENCE`` window the
    patience-limited MCTS explores a *subset* whose membership is sensitive to the
    bench-completion interleaving of the parallel pool — so the exact parallel==serial
    total only holds once the candidate set is fixed. Pinning isolates the
    orchestration invariant this test targets (the reward summing / multiplicity across
    backends); the search-space breadth is covered by ``test_inner_reward_is_separable``."""
    monkeypatch.setenv("EMMY_TILE", "")
    fused = _fuse(_two_distinct_matmuls())
    ctx = Context.from_target((8, 0))

    serial = run_inner_reward(fused, ctx=ctx, db=SearchDB(), backend=_CountingBackend(), patience=_PATIENCE)
    parallel = run_inner_reward(fused, ctx=ctx, db=SearchDB(), backends=[_CountingBackend(), _CountingBackend()], patience=_PATIENCE)

    assert parallel.total_us == pytest.approx(serial.total_us)
    assert parallel.ok == serial.ok
    s_by_key = {r.op_key: (r.best_us, r.multiplicity) for r in serial.per_op}
    p_by_key = {r.op_key: (r.best_us, r.multiplicity) for r in parallel.per_op}
    assert p_by_key == s_by_key, "per-op bests must be identical regardless of slot count"


def test_run_two_level_tune_single_terminal_assembles_bests() -> None:
    """With no fusion forks today the outer yields one terminal; the assembled
    graph greedy-replays the per-op bests."""
    result = run_two_level(
        _two_distinct_matmuls(),
        ctx=Context.from_target((8, 0)),
        db=SearchDB(),
        backend=_CountingBackend(),
        patience=_PATIENCE,
    )
    assert result.n_terminals == 1, "no multi-option fusion forks today → exactly one outer terminal"
    assert result.best_reward is not None and result.best_reward.ok
    assert len(result.best_reward.per_op) == 2

    # The winning fusion was greedy-assembled into a Graph[CudaOp] from the DB.
    assert result.assembled is not None
    assert any(isinstance(n.op, CudaOp) for n in result.assembled.nodes.values())


def test_o3_band_is_per_regime_under_a_precision_gate(monkeypatch):
    """The -O3 rebench band splits by precision regime: when a fast-math row (an f16-accumulate
    TILE) owns the global -O1 best, the best STANDARD row still qualifies for the deployable -O3
    rebench in its own band — otherwise the gate-off deploy lane is left with no -O3 evidence for
    the shape (the qkv.h4096 ~1000x scalar deploy, 2026-07-09 fm sweep). With no gate every row is
    standard and the band equals the global best (unchanged behavior)."""
    from emmy.compiler.pipeline.search.db import PerfStats
    from emmy.compiler.pipeline.search.policy.mcts import SearchNode, TuningSearch

    monkeypatch.setenv("EMMY_O3_TOL", "0.10")  # the tight band that starved the standard lane
    # ``observe`` re-derives the token's knobs (``_node_knobs`` walks the fork prefix); this test
    # drives bare tokens, so read the preset knobs off the token instead.
    monkeypatch.setattr(TuningSearch, "_node_knobs", lambda self, t: t.realized_knobs or {})

    def bench(search, knobs, median):
        node = SearchNode(candidate=None, parent=search.tree.root)  # parented so record_terminal reaches the root
        node.realized_knobs = knobs
        search.observe(node, PerfStats(median=median, min=median, max=median, mean=median, variance=0.0, n_samples=5), "ok")
        return search.last_o3_worthy

    s = TuningSearch()
    fm = {"TILE": "mma_m16n8k16_f16_f16/f2x4/k8", "WORK": "w2x4", "STAGE": "d1/tma"}
    std_best = {"TILE": "mma_m16n8k16_f16_f32/f2x4", "WORK": "w2x4", "STAGE": "d4/tma"}
    std_near = {"TILE": "mma_m16n8k16_f16_f32/f2x4/k2", "WORK": "w2x4", "STAGE": "d2/tma"}
    std_far = {"TILE": "mma_m16n8k16_f16_f32/f1x1", "WORK": "w1x1", "STAGE": ""}
    assert bench(s, fm, 238.0), "the global-best fast-math row rebenches"
    assert bench(s, std_best, 259.0), "the best STANDARD row must rebench in its own band (8% off the fm best)"
    assert bench(s, std_near, 262.0), "a standard row within tol of the standard best qualifies"
    assert not bench(s, std_far, 400.0), "a standard row far outside its own band still does not"
    assert not bench(s, {**fm, "STAGE": "d2/tma"}, 300.0), "a fast-math row competes against the global best only"
