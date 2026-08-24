"""``TwoLevelStrategy`` — the two-level search as one strategy over the engine's loop.

Pins the strategy's contract with a fake counting backend (no GPU): the outer never ventures
into Tile IR and its scoring is DECLARED SEPARABLE (each unique Loop kernel measured in its own
slice, Σ once all are measured); structurally identical kernels dedup with multiplicity; kernels
minted during the inner loops (a pinned cross-CTA split's pieces) are ENROLLED as first-class
tuning targets with their own identity — evidence, never reward terms.

Target forced to sm_80 so lowering is deterministic and GPU-independent — the fake backend never
launches anything, it hands back per-launch latencies keyed off each CudaOp's structural key.
"""

from __future__ import annotations

import logging
import zlib

import pytest

from emmy.compiler.backend.base import BenchmarkResult, LaunchTime
from emmy.compiler.backend.cuda._planner import compute_live_intervals
from emmy.compiler.backend.plan import plan_from_graph
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.search.db import SearchDB
from emmy.compiler.pipeline.search.slice import single_node_graph
from emmy.compiler.pipeline.search.strategy.two_level import InnerReward, OpResult, _KernelInventory
from tests.compiler.helpers import run_inner_reward, run_two_level

# Moderate patience: each kernel explores several variants then stops on stagnation (the fake
# backend gives a stable but arbitrary per-variant signal).
_PATIENCE = 8


@pytest.fixture(autouse=True)
def _force_target(monkeypatch, tmp_path):
    from emmy.compiler import target as target_mod

    # Isolate the online-prior checkpoint — fake-backend rows must never pollute the host's
    # real ``~/.cache/emmy/online.json``.
    monkeypatch.setenv("EMMY_ONLINE_FILE", str(tmp_path / "prior.json"))
    target_mod.set_target((8, 0))
    yield
    target_mod.set_target(None)


class _CountingBackend:
    """Fake backend: counts ``benchmark`` calls and returns a deterministic per-CudaOp latency
    keyed off the op's structural key (crc32, not the salted ``hash``), so the inner search sees
    real signal and a unique best without touching a GPU."""

    name = "cuda"
    bench_run_timeout_s = 1.0

    def __init__(self) -> None:
        self.calls = 0

    def benchmark(self, graph, num_iters="auto") -> BenchmarkResult:  # noqa: ARG002
        self.calls += 1
        cuda = [n for n in graph.nodes.values() if isinstance(n.op, CudaOp)]
        per: list[LaunchTime] = []
        for i, n in enumerate(cuda):
            us = 1.0 + (zlib.crc32(n.op.cache_key().encode()) % 100)
            per.append(LaunchTime(idx=i, kernel_name=getattr(n.op, "kernel_name", "k"), time_ms=us / 1000.0, samples=(us / 1000.0,)))
        return BenchmarkResult(time_ms=sum(p.time_ms for p in per), num_launches=len(per), per_launch=per)

    async def benchmark_async(self, graph, num_iters="auto") -> BenchmarkResult:
        return self.benchmark(graph, num_iters=num_iters)


def _matmul(g: Graph, prefix: str, M: int, K: int, N: int) -> str:
    a, b, c = f"{prefix}a", f"{prefix}b", f"{prefix}c"
    g.add_node(InputOp(), [], Tensor(a, (M, K)), node_id=a)
    g.add_node(InputOp(), [], Tensor(b, (K, N)), node_id=b)
    g.add_node(MatmulOp(), [a, b], Tensor(c, (M, N)), node_id=c)
    return c


def _graph(*specs: tuple[str, int, int, int]) -> Graph:
    g = Graph()
    outs = [_matmul(g, prefix, m, k, n) for prefix, m, k, n in specs]
    g.inputs = [f"{p}{x}" for p, _, _, _ in specs for x in ("a", "b")]
    g.outputs = outs
    return g


def _fuse(graph: Graph) -> Graph:
    return Pipeline.build(LOOP_PASSES).run(graph, db=SearchDB())


def test_searched_winner_requires_one_post_fusion_and_one_cuda_kernel() -> None:
    one = OpResult(name="k", op_key="key", best_us=4.0, searched_knobs={"TILE": "f2x2"}, searched_us=5.0, searched_cuda_ops=1)
    assert InnerReward(total_us=4.0, ok=True, per_op=[one]).searched_winner() == ({"TILE": "f2x2"}, 5.0)
    multi_cuda = OpResult(**{**one.__dict__, "searched_cuda_ops": 2})
    assert InnerReward(total_us=4.0, ok=True, per_op=[multi_cuda]).searched_winner() is None
    assert InnerReward(total_us=8.0, ok=True, per_op=[one, one]).searched_winner() is None


def test_scoring_is_separable_over_unique_kernels() -> None:
    """Two structurally distinct kernels: each measured independently, the terminal reward is
    their Σ, and the bench count stays far below the cross-product (the old whole-graph MCTS
    failure this design guards against)."""
    fused = _fuse(_graph(("x", 64, 128, 48), ("y", 96, 64, 32)))
    backend = _CountingBackend()
    reward = run_inner_reward(fused, ctx=Context.from_target((8, 0)), db=SearchDB(), backends=[backend], patience=_PATIENCE, prior=None)
    assert reward.ok
    assert len(reward.per_op) == 2
    assert all(r.best_us is not None and r.multiplicity == 1 for r in reward.per_op)
    assert reward.total_us == pytest.approx(sum(r.best_us for r in reward.per_op))
    # Patience is stagnation-based, so a per-op window stretches by a handful of benches; the
    # guarantee is linear scaling in kernels, never the cross-product (hundreds here).
    assert backend.calls <= 8 * _PATIENCE, "separable: benches scale per-op, never as the product"


def test_identical_kernels_dedup_with_multiplicity() -> None:
    fused = _fuse(_graph(("x", 64, 128, 48), ("y", 64, 128, 48)))
    backend = _CountingBackend()
    reward = run_inner_reward(fused, ctx=Context.from_target((8, 0)), db=SearchDB(), backends=[backend], patience=_PATIENCE, prior=None)
    assert len(reward.per_op) == 1, "structurally identical kernels share one inner search"
    assert reward.per_op[0].multiplicity == 2
    assert reward.total_us == pytest.approx(2 * reward.per_op[0].best_us)


def test_single_node_slice_declares_unregistered_input_boundaries_in_the_runtime_plan() -> None:
    fused = _fuse(_graph(("x", 64, 128, 48)))
    fused.inputs.remove("xb")
    root = next(node.id for node in fused.nodes.values() if isinstance(node.op, LoopOp))

    sliced = single_node_graph(fused, root)
    lowered = Pipeline.build(CUDA_PASSES).run(sliced, ctx=Context.from_target((8, 0)))
    plan = plan_from_graph(lowered)

    assert set(sliced.inputs) == {"xa", "xb"}
    roles = {buffer.name: buffer.role for buffer in plan.buffers}
    assert roles == {"xa": "input", "xb": "input", "xc": "output"}
    scratch = [buffer.name for buffer in plan.buffers if buffer.role == "scratch"]
    # Exercise the allocator's exact liveness seam: an undeclared InputOp would
    # be scratch here and fail because no CUDA launch produces it.
    assert compute_live_intervals(scratch, plan.launches) == {}


def test_run_drives_outer_scores_separably_and_assembles() -> None:
    """The full strategy: outer chain (fusion offers no forks today) → one terminal, separable
    scoring, greedy DB-best assembly to CudaOps."""
    graph = _graph(("x", 64, 128, 48))
    result = run_two_level(
        graph,
        ctx=Context.from_target((8, 0)),
        db=SearchDB(),
        backend=_CountingBackend(),
        patience=_PATIENCE,
        prior=None,
        manage_prior=False,
    )
    assert result.n_terminals == 1
    assert result.best_reward is not None and result.best_reward.ok
    assert result.assembled is not None
    assert any(isinstance(n.op, CudaOp) for n in result.assembled.nodes.values())


def test_minted_kernels_are_enrolled_as_first_class_targets(monkeypatch, caplog) -> None:
    """A pinned cross-CTA split mints pieces inside the inner loops; the splice watcher reports
    them and the strategy enrolls each — tuned in its own slice, logged as enrolled — while the
    terminal reward keeps only the OUTER kernel (pieces are evidence, not reward terms)."""
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    fused = _fuse(_graph(("x", 64, 512, 48)))
    backend = _CountingBackend()
    with caplog.at_level(logging.INFO, logger="emmy.compiler.pipeline.search.strategy.two_level"):
        reward = run_inner_reward(fused, ctx=Context.from_target((8, 0)), db=SearchDB(), backends=[backend], patience=_PATIENCE, prior=None)
    enrolled = [rec.message for rec in caplog.records if "enrolled minted kernel" in rec.message]
    assert len(enrolled) >= 2, f"the split's partial + finalize must both enroll, saw: {enrolled}"
    assert len(reward.per_op) == 1, "enrolled pieces never join the terminal reward"
    assert reward.per_op[0].multiplicity == 1


def test_inventory_dedups_by_structural_identity() -> None:
    """The inventory reports a kernel once per structural identity, however many trajectories
    re-mint it, and never re-reports a kernel seeded as already known."""
    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.pipeline.strategy import SpliceEvent, discovered_strategies

    fused = _fuse(_graph(("x", 64, 128, 48)))
    identity = next(s for s in discovered_strategies() if type(s).__name__ == "IdentityStrategy")
    loop_node = next(nid for nid, n in fused.nodes.items() if isinstance(n.op, LoopOp))
    reported: list[str] = []
    inventory = _KernelInventory(identity, lambda nid, op, frag: reported.append(nid))
    event = SpliceEvent(match=None, fragment=fused, root_op=fused.nodes[loop_node].op, pass_name="lowering/tile", graph=fused)
    inventory.on_splice(event)
    assert reported == [loop_node], "first sighting reported"
    inventory.on_splice(event)
    assert reported == [loop_node], "re-minting the same identity is silent"
