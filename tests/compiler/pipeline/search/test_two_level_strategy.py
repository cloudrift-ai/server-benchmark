"""``TwoLevelStrategy`` — the two-level search as one strategy over the engine's loop.

Pins the strategy's contract with a fake counting backend (no GPU): the outer never ventures
into Tile IR and its scoring is DECLARED SEPARABLE (each unique Loop kernel measured in its own
slice, Σ once all are measured); structurally identical kernels dedup with multiplicity; kernels
minted during the inner loops (a pinned cross-CTA split's pieces) are ENROLLED as first-class
tuning targets with their own identity — evidence, never reward terms. A direct unscheduled Tile
child enters that same per-kernel path, while a scheduled child stays lowering-only.

Target forced to sm_80 so lowering is deterministic and GPU-independent — the fake backend never
launches anything, it hands back per-launch latencies keyed off each CudaOp's structural key.
"""

from __future__ import annotations

import json
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
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.search.db import SearchDB
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.pipeline.search.slice import single_node_graph
from emmy.compiler.pipeline.search.strategy.two_level import InnerReward, OpResult, _kernel_nodes, _KernelInventory
from tests.compiler.helpers import run_inner_reward, run_two_level

# Moderate patience: each kernel explores several variants then stops on stagnation (the fake
# backend gives a stable but arbitrary per-variant signal).
_PATIENCE = 8

_CHILD_WINNER = {
    "WORK": "t32",
    "TILE": "f1",
    "STAGE": "",
}


def _is_child_winner(knobs) -> bool:
    """Whether a row is the early non-default child schedule priced by the fake backends."""
    return all(knobs.get(key) == value for key, value in _CHILD_WINNER.items())


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
            us = 1.0 + (zlib.crc32(n.op.identity_key(with_io=True, with_knobs=True).encode()) % 100)
            per.append(LaunchTime(idx=i, kernel_name=getattr(n.op, "kernel_name", "k"), time_ms=us / 1000.0, samples=(us / 1000.0,)))
        return BenchmarkResult(time_ms=sum(p.time_ms for p in per), num_launches=len(per), per_launch=per)

    async def benchmark_async(self, graph, num_iters="auto", warmup=5) -> BenchmarkResult:
        del warmup
        return self.benchmark(graph, num_iters=num_iters)


class _RouteBackend(_CountingBackend):
    """Price one exact child schedule tree as the winning placement route."""

    def __init__(self) -> None:
        super().__init__()
        self.measured_route: tuple[str, ...] | None = None

    def benchmark(self, graph, num_iters="auto") -> BenchmarkResult:  # noqa: ARG002
        self.calls += 1
        cuda = [n for n in graph.nodes.values() if isinstance(n.op, CudaOp)]
        if len(cuda) > 1:
            route = tuple(node.op.identity_key(with_io=True, with_knobs=True) for node in cuda)
            first = cuda[0].op.knobs
            fast_route = _is_child_winner(first)
            if fast_route:
                self.measured_route = route
            us = 1.0 if fast_route else 20.0
        else:
            knobs = cuda[0].op.knobs
            fused = knobs.get("S_n_accum") == 1.0 and knobs.get("S_pw_add") == 1.0
            us = 100.0 if fused else (0.25 if _is_child_winner(knobs) else 10.0)
        per = [
            LaunchTime(idx=i, kernel_name=getattr(node.op, "kernel_name", "k"), time_ms=us / 1000.0, samples=(us / 1000.0,))
            for i, node in enumerate(cuda)
        ]
        return BenchmarkResult(time_ms=sum(item.time_ms for item in per), num_launches=len(per), per_launch=per)


class _ChildScheduleBackend(_CountingBackend):
    """Prefer one exact schedule for a directly tuned post-cut child."""

    def benchmark(self, graph, num_iters="auto") -> BenchmarkResult:  # noqa: ARG002
        self.calls += 1
        (cuda,) = [node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
        fast = _is_child_winner(cuda.knobs)
        us = 0.25 if fast else 10.0
        return BenchmarkResult(
            time_ms=us / 1000.0,
            num_launches=1,
            per_launch=(LaunchTime(idx=0, kernel_name=cuda.kernel_name, time_ms=us / 1000.0, samples=(us / 1000.0,)),),
        )


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


def test_searched_winner_requires_one_post_fusion_kernel_and_an_exact_replay_row() -> None:
    one = OpResult(name="k", op_key="key", best_us=4.0, searched_knobs={"TILE@map.1/inner": "f2x2"}, searched_us=5.0, searched_cuda_ops=1)
    assert InnerReward(total_us=4.0, ok=True, per_op=[one]).searched_winner() == ({"TILE@map.1/inner": "f2x2"}, 5.0)
    multi_cuda = OpResult(**{**one.__dict__, "searched_cuda_ops": 2})
    assert InnerReward(total_us=4.0, ok=True, per_op=[multi_cuda]).searched_winner() is None
    split = OpResult(**{**multi_cuda.__dict__, "searched_knobs": {"REDUCE@map.1/inner": "g8k"}, "searched_structural": True})
    assert InnerReward(total_us=4.0, ok=True, per_op=[split]).searched_winner() == ({"REDUCE@map.1/inner": "g8k"}, 5.0)
    assert InnerReward(total_us=8.0, ok=True, per_op=[one, one]).searched_winner() is None


def test_scoring_is_separable_over_unique_kernels() -> None:
    """Two structurally distinct kernels: each measured independently, the terminal reward is
    their Σ, and no whole-graph Cartesian candidate is introduced."""
    specs = (("x", 64, 128, 48), ("y", 96, 64, 32))
    fused = _fuse(_graph(*specs))
    backend = _CountingBackend()
    reward = run_inner_reward(fused, ctx=Context.from_target((8, 0)), db=SearchDB(), backends=[backend], patience=_PATIENCE, prior=None)
    assert reward.ok
    assert len(reward.per_op) == 2
    assert all(r.best_us is not None and r.multiplicity == 1 for r in reward.per_op)
    assert reward.total_us == pytest.approx(sum(r.best_us for r in reward.per_op))
    individual_calls = []
    for spec in specs:
        one = _CountingBackend()
        run_inner_reward(
            _fuse(_graph(spec)), ctx=Context.from_target((8, 0)), db=SearchDB(), backends=[one], patience=_PATIENCE, prior=None
        )
        individual_calls.append(one.calls)
    assert backend.calls == sum(individual_calls), "the joint graph spends exactly the sum of its independent tuning budgets"


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


def _placement_route_graph() -> Graph:
    graph = _graph(("x", 64, 128, 48))
    graph.add_node(InputOp(), [], Tensor("residual", (64, 48)), node_id="residual")
    graph.add_node(ElementwiseOp("add"), ["xc", "residual"], Tensor("out", (64, 48)), node_id="out")
    graph.inputs.append("residual")
    graph.outputs = ["out"]
    return graph


def _persisted_placement_child() -> Graph:
    """Return one unscheduled child as a Tile dump round-trip would load it."""
    fused = Pipeline.build(LOOP_PASSES).run(_placement_route_graph(), ctx=Context.from_target((8, 0)), db=SearchDB())
    with pinned_knobs({"PLACE": "cut", "REDUCE": ""}):
        pieces = Pipeline.build(["lowering/tile"], select={"lift", "cut"}).run(fused, ctx=Context.from_target((8, 0)), db=SearchDB())
    producer = next(node.id for node in pieces.nodes.values() if isinstance(node.op, TileOp) and "__place_" in node.op.name)
    child = single_node_graph(pieces, producer)
    return Graph.from_dict(json.loads(json.dumps(child.to_dict(), default=str)))


def test_persisted_unscheduled_tile_child_tunes_and_replays_in_parent_cut(tmp_path) -> None:
    """A direct post-cut Tile target records ordinary child evidence, which the same child
    consumes when its parent route is compiled later."""
    ctx = Context.from_target((8, 0))
    db = SearchDB(tmp_path / "child.db")
    child = _persisted_placement_child()
    backend = _ChildScheduleBackend()

    with pinned_knobs({"REDUCE": ""}):
        direct = run_two_level(
            child,
            ctx=ctx,
            db=db,
            backend=backend,
            patience=24,
            max_candidates=32,
            prior=None,
            manage_prior=False,
        )
    with pinned_knobs({"PLACE": "cut", "REDUCE": ""}):
        parent = Pipeline.build(CUDA_PASSES).run(_placement_route_graph(), ctx=ctx, db=db)

    assert direct.best_reward is not None and direct.best_reward.ok
    assert len(direct.best_reward.per_op) == 1
    direct_cuda = [node.op for node in direct.assembled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(direct_cuda) == 1
    assert _is_child_winner(direct_cuda[0].knobs)
    parent_cuda = [node.op for node in parent.nodes.values() if isinstance(node.op, CudaOp)]
    producer = next(op for op in parent_cuda if "__place_" in op.kernel_name)
    assert _is_child_winner(producer.knobs)
    db.close()


def test_scheduled_tile_child_is_not_reenrolled_or_rescheduled() -> None:
    """A Tile root whose worker inventory is sealed is already decided."""
    child = _persisted_placement_child()
    with pinned_knobs({"WORK": "t16x8", "STAGE": "d1/smem-async", "REDUCE": ""}):
        scheduled = Pipeline.build(["lowering/tile"]).run(child, ctx=Context.from_target((8, 0)), db=SearchDB())
    tile = next(node.op for node in scheduled.nodes.values() if isinstance(node.op, TileOp))
    assert tile.schedule is not None
    assert _kernel_nodes(scheduled) == []

    backend = _ChildScheduleBackend()
    result = run_two_level(
        scheduled,
        ctx=Context.from_target((8, 0)),
        db=SearchDB(),
        backend=backend,
        patience=_PATIENCE,
        prior=None,
        manage_prior=False,
    )
    assert backend.calls == 0
    (cuda,) = [node.op for node in result.assembled.nodes.values() if isinstance(node.op, CudaOp)]
    assert cuda.knobs["WORK"] == "t16x8"
    assert cuda.knobs["STAGE"] == "d1/smem-async"


def test_placement_route_total_is_not_persisted_without_a_child_schedule_receipt(monkeypatch, tmp_path) -> None:
    """A measured route stays search evidence until its exact child tree can replay."""
    monkeypatch.setenv("EMMY_REDUCE", "")
    graph = _placement_route_graph()
    ctx = Context.from_target((8, 0))
    path = tmp_path / "route.db"
    db = SearchDB(path)
    backend = _RouteBackend()
    result = run_two_level(
        graph,
        ctx=ctx,
        db=db,
        backend=backend,
        patience=_PATIENCE,
        prior=None,
        manage_prior=False,
    )
    assert result.best_reward is not None
    assert result.best_reward.searched_winner() == ({"PLACE": "cut"}, 2.0)
    assert backend.measured_route is not None
    route_rows = [row for row in db.iter_perf(ctx.structural_key(), backend="cuda") if row.knobs.get("PLACE") == "cut"]
    assert route_rows == []
    db.close()


def test_pinned_placement_route_tunes_and_assembles_child_schedules(monkeypatch, caplog) -> None:
    """A pinned cut freezes the kernel set; every minted child is then tuned independently and
    the assembled route reads each child's own schedule evidence."""
    monkeypatch.setenv("EMMY_REDUCE", "")
    backend = _RouteBackend()
    with caplog.at_level(logging.INFO, logger="emmy.compiler.pipeline.search.strategy.two_level"):
        with pinned_knobs({"PLACE": "cut"}):
            result = run_two_level(
                _placement_route_graph(),
                ctx=Context.from_target((8, 0)),
                db=SearchDB(),
                backend=backend,
                patience=_PATIENCE,
                prior=None,
                manage_prior=False,
            )

    assembled = [node.op for node in result.assembled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(assembled) == 2
    assert sum("enrolled minted kernel" in record.message for record in caplog.records) >= 2
    assert _is_child_winner(assembled[0].knobs)
    assert assembled[1].knobs["WORK"] == "" and assembled[1].knobs.get("STAGE", "") == ""


def test_minted_kernels_are_enrolled_as_first_class_targets(monkeypatch, caplog) -> None:
    """A pinned cross-CTA split mints pieces inside the inner loops; the splice watcher reports
    them and the strategy enrolls each — tuned in its own slice, logged as enrolled — while the
    terminal reward keeps only the OUTER kernel (pieces are evidence, not reward terms)."""
    monkeypatch.setenv("EMMY_REDUCE@INNER", "g2k")
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
