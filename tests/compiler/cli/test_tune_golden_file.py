"""Working-golden target and ranking plumbing for ``emmy tune --golden-file``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from emmy.commands import tune
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.passes.identity import IdentityStrategy
from emmy.compiler.pipeline.search.db import PerfStats, SearchDB
from emmy.compiler.pipeline.search.golden import dump_golden_file, load_golden_file
from emmy.compiler.pipeline.search.policy.mcts import SearchNode, SearchTree, TuningSearch
from emmy.compiler.pipeline.search.strategy.two_level import InnerReward, OpResult
from emmy.compiler.pipeline.search.working_golden import (
    WorkingGoldenTarget,
    load_working_targets,
    measure_proposals,
    persist_proposal_rankings,
    persist_tune_winner,
    realized_tuning_knobs,
    validate_working_gpu,
)
from emmy.compiler.pipeline.strategy import discovered_strategies
from emmy.compiler.torch_wire import intern_program


def _args(path, **over):
    values = {
        "golden_file": str(path),
        "dataset": None,
        "kernel": None,
        "code": None,
        "input": None,
        "golden": None,
        "dynamic": None,
    }
    values.update(over)
    return SimpleNamespace(**values)


def _matmul(name: str, *, pins=None, knobs=None, emmy_us=None, cublas_us=None):
    entry = {
        "name": name,
        "bindings": {},
        "pins": {"FAST_MATH": False} if pins is None else pins,
    }
    if knobs is not None:
        entry["knobs"] = knobs
    if emmy_us is not None:
        entry["measurements"] = {
            "emmy_us": emmy_us,
            "reference_us": cublas_us,
            "reference_backend": "cublas",
        }
    return entry


def _document(*entries):
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (16, 64), "f16"), node_id="x")
    graph.add_node(InputOp(), [], Tensor("w", (64, 32), "f16"), node_id="w")
    graph.add_node(MatmulOp(), ["x", "w"], Tensor("matmul", (16, 32), "f16"), node_id="matmul")
    graph.inputs, graph.outputs = ["x", "w"], ["matmul"]
    programs = []
    program_index = intern_program(programs, graph)
    config = {
        "program": program_index,
        "target": {"origins": ["matmul"]},
        "realizations": [dict(entry) for entry in entries],
    }
    return {"compute_cap": [8, 9], "programs": programs, "configs": [config]}


def test_working_file_groups_candidate_rows_and_recovers_embedded_program(tmp_path):
    path = tmp_path / "trace.yaml"
    dump_golden_file(_document(_matmul("mm"), _matmul("mm", knobs={"TILE": "f2x2"})), path)

    document, targets = load_working_targets(path)

    assert document["configs"][0]["realizations"][0]["name"] == "mm"
    assert len(targets) == 1
    mm = targets[0]
    assert mm.code is None and mm.input is None and isinstance(mm.program.nodes["matmul"].op, MatmulOp)
    assert mm.entry_indexes == [(0, 0), (0, 1)]
    assert mm.proposals == [((0, 1), {"TILE": "f2x2"})]


def test_working_file_keeps_distinct_input_pin_regimes_separate(tmp_path):
    path = tmp_path / "trace.yaml"
    dump_golden_file(
        _document(
            _matmul("mm", pins={"FAST_MATH": False}),
            _matmul("mm", pins={"FAST_MATH": True}),
        ),
        path,
    )

    _, targets = load_working_targets(path)

    assert [target.pins for target in targets] == [{"FAST_MATH": False}, {"FAST_MATH": True}]


def test_empty_knob_map_is_a_forkless_proposal_not_inventory(tmp_path):
    path = tmp_path / "working.yaml"
    dump_golden_file(_document(_matmul("mm"), _matmul("mm", knobs={})), path)

    loaded_document, targets = load_working_targets(path)

    assert targets[0].entry_indexes == [(0, 0), (0, 1)]
    assert targets[0].proposals == [((0, 1), {})]
    assert set(loaded_document) == {"compute_cap", "programs", "configs"}


def test_multi_cuda_realized_knobs_must_be_conflict_free():
    graph = Graph()
    graph.add_node(op=CudaOp(kernel_name="a", knobs={"TILE": "f2x2"}), inputs=[], output=Tensor("a", (1,)), node_id="a")
    graph.add_node(op=CudaOp(kernel_name="b", knobs={"TILE": "f4x2"}), inputs=[], output=Tensor("b", (1,)), node_id="b")
    assert realized_tuning_knobs(graph) is None

    graph.nodes["b"].op.knobs["TILE"] = "f2x2"
    assert realized_tuning_knobs(graph)["TILE"] == "f2x2"


def test_working_file_rejects_legacy_reproducer_field(tmp_path):
    path = tmp_path / "trace.yaml"
    document = _document(_matmul("mm"))
    document["configs"][0]["reproducer"] = "missing.json"
    with pytest.raises(ValueError, match="unknown field"):
        dump_golden_file(document, path)


def test_working_file_rejects_missing_yaml_cleanly(tmp_path):
    with pytest.raises(ValueError, match="invalid golden file"):
        load_working_targets(tmp_path / "missing.yaml")


def test_working_file_is_mutually_exclusive_with_direct_input(tmp_path):
    path = tmp_path / "working.yaml"
    dump_golden_file(_document(_matmul("mm")), path)
    with pytest.raises(SystemExit) as exc:
        tune.handle_tune(_args(path, code="torch.ones(1)", max_candidates=None))
    assert exc.value.code == 2


def test_ranking_write_preserves_verified_and_records_actual_searched_winner(tmp_path):
    path = tmp_path / "working.yaml"
    verified = _matmul("mm", knobs={"TILE": "f2x2"}, emmy_us=9.0, cublas_us=10.0)
    proposal = _matmul("mm", knobs={"TILE": "f4x2"})
    document = _document(verified, proposal)
    dump_golden_file(document, path)
    document, targets = load_working_targets(path)
    target = targets[0]
    rankings = [
        {"status": "ok", "latency_us": 8.5, "compile_flags": "-O1", "measured_knobs": {"TILE": "f2x2"}},
        {"status": "ok", "latency_us": 8.0, "compile_flags": "-O1", "measured_knobs": {"TILE": "f4x2"}},
    ]
    # Greedy deploy replay deliberately disagrees with the search result. The
    # working file must pair the search's own knobs and latency, never combine
    # ``assembled`` knobs with ``best_reward`` cost.
    reward = SimpleNamespace(searched_winner=lambda: ({"TILE": "f8x2"}, 7.5))
    result = SimpleNamespace(assembled=SimpleNamespace(deploy_knobs={"TILE": "golden"}), best_reward=reward)
    persist_proposal_rankings(path, document, target, rankings)
    persist_tune_winner(
        path,
        document,
        target,
        result.best_reward.searched_winner(),
        compile_flags="-Xcicc -O1",
    )

    got = load_golden_file(path)
    realizations = got["configs"][0]["realizations"]
    assert "ranking" not in realizations[0]
    assert realizations[0]["measurements"]["emmy_us"] == 9.0
    assert realizations[1]["ranking"]["source"] == "proposal"
    assert realizations[1]["ranking"]["latency_us"] == 8.0
    winner = realizations[2]
    assert winner["knobs"] == {"TILE": "f8x2"}
    assert winner["ranking"]["source"] == "tune"
    assert winner["ranking"]["tune_winner"] is True
    assert "measurements" not in winner


def test_incremental_persist_matches_a_full_dump_and_still_checks_realizations(tmp_path):
    path = tmp_path / "working.yaml"
    dump_golden_file(_document(_matmul("mm"), _matmul("mm", knobs={"TILE": "f2x2"})), path)
    document, targets = load_working_targets(path)
    target = targets[0]
    rankings = [{"status": "ok", "latency_us": 8.0, "compile_flags": "-O1", "measured_knobs": {"TILE": "f2x2"}}] * len(target.proposals)

    # ``tune`` persists per target and reuses the pools it loaded; the file must still be the
    # one a full revalidating dump of the same document writes.
    persist_proposal_rankings(path, document, target, rankings)
    canonical = tmp_path / "canonical.yaml"
    dump_golden_file(document, canonical)
    assert path.read_bytes() == canonical.read_bytes()

    document["configs"][0]["realizations"][0]["pins"] = "not-a-mapping"
    with pytest.raises(ValueError, match="pins must be a mapping"):
        persist_proposal_rankings(path, document, target, rankings)


def test_ambiguous_multi_cuda_winner_is_not_annotated(tmp_path):
    path = tmp_path / "working.yaml"
    dump_golden_file(_document(_matmul("mm")), path)
    document, targets = load_working_targets(path)
    result = SimpleNamespace(best_reward=SimpleNamespace(searched_winner=lambda: None), assembled=object())

    persist_tune_winner(path, document, targets[0], result.best_reward.searched_winner(), compile_flags="-O1")

    got = load_golden_file(path)
    assert len(got["configs"]) == 1
    assert got["configs"][0]["realizations"][0]["name"] == "mm"
    assert got["configs"][0]["target"] == {"origins": ["matmul"]}


def test_structural_multi_cuda_winner_persists_its_exact_replay_row(tmp_path):
    path = tmp_path / "working.yaml"
    dump_golden_file(_document(_matmul("mm")), path)
    document, targets = load_working_targets(path)
    route = {
        "WORK": "w1x1",
        "TILE": "mma_m16n8k16_f16_f32/f1x4/k8",
        "REDUCE": "g8k",
        "STAGE": "d1/smem",
        "RASTER": "",
    }
    reward = InnerReward(
        total_us=6.0,
        ok=True,
        per_op=[
            OpResult(
                name="mm",
                op_key="key",
                best_us=6.0,
                searched_knobs=route,
                searched_us=6.0,
                searched_cuda_ops=2,
                searched_structural=True,
            )
        ],
    )

    persist_tune_winner(path, document, targets[0], reward.searched_winner(), compile_flags="-O1")

    realizations = load_golden_file(path)["configs"][0]["realizations"]
    assert realizations[1]["knobs"] == route
    assert realizations[1]["ranking"] == {
        "status": "ok",
        "latency_us": 6.0,
        "compile_flags": "-O1",
        "measured_knobs": route,
        "source": "tune",
        "tune_winner": True,
    }


def test_structural_multi_cuda_proposal_survives_search_continuation_and_reload(tmp_path, monkeypatch):
    from emmy.compiler.pipeline.search.policy.greedy import _db_measured_index, _db_measured_pick

    route = {
        "WORK": "w2x1",
        "TILE": "mma_m16n8k16_f16_f32/f4x8/k8",
        "REDUCE": "g4k",
        "STAGE": "d1/smem-async",
        "RASTER": "",
    }
    terminal = Graph()
    terminal.add_node(
        CudaOp(kernel_name="partial", knobs={**route, "REDUCE": ""}),
        [],
        Tensor("partial", (1,)),
        node_id="partial",
    )
    terminal.add_node(
        CudaOp(kernel_name="finalize", knobs={key: "" for key in route}),
        [],
        Tensor("finalize", (1,)),
        node_id="finalize",
    )
    path = tmp_path / "working.yaml"
    dump_golden_file(_document(_matmul("mm"), _matmul("mm", knobs=route)), path)
    document, targets = load_working_targets(path)
    stable_graph = targets[0].program
    loop_graph = Pipeline.build(LOOP_PASSES).run(stable_graph.copy(), ctx=Context((8, 9)))
    [original_loop] = [node.op for node in loop_graph.nodes.values() if isinstance(node.op, LoopOp)]
    identity = next(strategy for strategy in discovered_strategies() if isinstance(strategy, IdentityStrategy))
    original_op_sig = identity.op_sig(original_loop, loop_graph)
    structural_features = {key: float(value) for key, value in original_loop.knobs.items() if key.startswith("S_")}
    live_features = {**structural_features, "S_warp_eligible": 1.0}
    active_route = route

    class FakeSearch(TuningSearch):
        def __init__(self, **kwargs):
            assert kwargs["base_knobs"] == ctx.features()
            self._base_knobs = dict(kwargs["base_knobs"])
            self.tree = SearchTree()
            self.last_status = "ok"
            self.last_stats = PerfStats(median=59.61, min=59.61, max=59.61, mean=59.61, variance=0.0, n_samples=1)
            self.o3_rows = []

    class FakePipeline:
        def __init__(self):
            self.strategies = ()

        @classmethod
        def build(cls, _passes):
            return cls()

        def with_strategies(self, *strategies):
            self.strategies = strategies
            return self

        async def tune_async(self, graph, **kwargs):
            assert isinstance(graph.nodes["matmul"].op, MatmulOp)
            event = SimpleNamespace(graph=loop_graph)
            for strategy in self.strategies:
                strategy.on_pass_end(event)
            route_parent = TileOp(knobs=dict(live_features))
            route_parent.knobs.update(active_route)
            fragment = Graph()
            splice = SimpleNamespace(root_op=route_parent, fragment=fragment)
            for strategy in self.strategies:
                strategy.on_splice(splice)
            search = kwargs["search"]
            leaf = SearchNode(candidate=SimpleNamespace(resolved_knobs=None))
            leaf.visits = 1
            leaf.best_reward = 1.0 / 59.61
            leaf.realized_knobs = None
            leaf.realized_cuda_ops = 2
            leaf.realized_cuda_knobs = [dict(node.op.knobs) for node in terminal.nodes.values()]
            leaf.bench_status = "ok"
            leaf.bench_stats = search.last_stats
            search.tree.root.children = [leaf]
            search.tree.root.visits = 1
            search.tree.root.best_reward = leaf.best_reward
            yield SimpleNamespace(graph=terminal)

    monkeypatch.setattr("emmy.compiler.pipeline.TuningSearch", FakeSearch)
    monkeypatch.setattr("emmy.compiler.pipeline.Pipeline", FakePipeline)
    ctx = Context(
        (8, 9),
        compile_flags="-O3",
        gpu_name="NVIDIA GeForce RTX 4090",
        device_props={"sm_count": 128},
    )
    db_path = tmp_path / "proposal.db"
    db = SearchDB(db_path)
    proposals = [((0, 1), route)]
    rankings = asyncio.run(
        measure_proposals(stable_graph, proposals, backend=object(), db=db, ctx=ctx, max_candidates=1, run_id="proposal-run")
    )
    assert rankings == [
        {
            "status": "ok",
            "latency_us": 59.61,
            "compile_flags": "-O3",
            "measured_knobs": route,
        }
    ]
    db.close()
    reloaded_db = SearchDB.open_readonly(db_path)
    measured_nodes = list(reloaded_db.iter_nodes(context_key=ctx.structural_key(), op_sig=original_op_sig))
    assert len(measured_nodes) == 2
    parent = next(row for row in measured_nodes if row.parent_key is None)
    branch = next(row for row in measured_nodes if row.parent_key is not None)
    assert branch.parent_key == parent.node_key
    assert parent.op_sig == branch.op_sig == original_op_sig
    assert branch.features["REDUCE"] == "g4k"
    assert branch.value_us == pytest.approx(59.61)
    route_parent = TileOp(knobs={**live_features, **route})
    assert route_parent.cache_key() != original_loop.cache_key()
    perf = reloaded_db.lookup_perf(ctx.structural_key(), route_parent.cache_key(), backend="cuda")
    assert perf is not None
    assert perf.status == "ok"
    assert perf.stats == PerfStats(median=59.61, min=59.61, max=59.61, mean=59.61, variance=0.0, n_samples=1)
    assert perf.captured is True
    assert perf.knobs == {**ctx.features(), **live_features, **route}
    assert reloaded_db.lookup_perf(ctx.structural_key(), original_loop.cache_key(), backend="cuda") is None
    reloaded_db.close()

    # A later ordinary search keeps its own whole-slice bookkeeping and lowering
    # evidence under the unpinned Loop key. Neither may replace or hide the exact
    # structural parent measured by the proposal.
    db = SearchDB(db_path)
    bookkeeping = PerfStats(median=106.95, min=106.95, max=106.95, mean=106.95, variance=0.0, n_samples=1)
    monolithic = PerfStats(median=153.45, min=153.45, max=153.45, mean=153.45, variance=0.0, n_samples=1)
    fallback = {**route, "REDUCE": ""}
    fallback_key = "monolithic-cuda"
    db.record_perf(ctx.structural_key(), original_loop.cache_key(), backend="cuda", status="ok", stats=bookkeeping, captured=True)
    db.record_lowering(
        original_loop.cache_key(),
        "loop",
        fallback_key,
        "cuda",
        knobs=fallback,
        measured_median_us=monolithic.median,
    )
    db.record_perf(
        ctx.structural_key(),
        fallback_key,
        backend="cuda",
        status="ok",
        stats=monolithic,
        knobs={**ctx.features(), **live_features, **fallback},
        captured=True,
    )
    db.close()
    reloaded_db = SearchDB.open_readonly(db_path)
    route_perf = reloaded_db.lookup_perf(ctx.structural_key(), route_parent.cache_key(), backend="cuda")
    loop_perf = reloaded_db.lookup_perf(ctx.structural_key(), original_loop.cache_key(), backend="cuda")
    assert route_perf is not None and route_perf.stats.median == pytest.approx(59.61)
    assert loop_perf is not None and loop_perf.stats.median == pytest.approx(106.95)
    lowering = reloaded_db.lookup_lowering(original_loop.cache_key())
    assert lowering is not None and lowering.child_key == fallback_key
    candidates = [{**live_features, **fallback}, {**live_features, **route}]
    assert _db_measured_pick(_db_measured_index(reloaded_db, ctx), candidates) == (1, 59.61)
    reloaded_db.close()
    persist_proposal_rankings(path, document, targets[0], rankings)
    reloaded, reloaded_targets = load_working_targets(path)
    proposal = reloaded["configs"][0]["realizations"][1]
    assert proposal["knobs"] == route
    assert proposal["ranking"]["measured_knobs"] == route
    assert proposal["ranking"]["status"] == "ok"
    assert reloaded_targets[0].proposals == [((0, 1), route)]

    nonstructural = {**route, "REDUCE": ""}
    active_route = nonstructural
    negative_db = SearchDB(tmp_path / "negative.db")
    [ambiguous] = asyncio.run(
        measure_proposals(
            stable_graph,
            [((0, 1), nonstructural)],
            backend=object(),
            db=negative_db,
            ctx=ctx,
            max_candidates=1,
            run_id="proposal-run",
        )
    )
    assert negative_db.lookup_perf(ctx.structural_key(), original_loop.cache_key(), backend="cuda") is None
    negative_db.close()
    assert ambiguous["status"] == "ambiguous_multi_kernel"
    assert ambiguous["measured_knobs"] is None


def test_working_file_rejects_canonical_path_and_symlink(monkeypatch, tmp_path):
    from contextlib import contextmanager

    from emmy.compiler.pipeline.search import golden

    hardware_dir = tmp_path / "hardware-goldens"
    hardware_dir.mkdir()
    hardware = hardware_dir / "gpu.yaml"
    recipe_root = tmp_path / "recipes"
    recipe = recipe_root / "model" / "golden" / "gpu.yaml"
    dump_golden_file(_document(_matmul("hardware")), hardware)
    dump_golden_file(_document(_matmul("recipe")), recipe)

    @contextmanager
    def default_recipe_root():
        yield recipe_root

    monkeypatch.setattr(golden, "_HARDWARE_GOLDENS_DIR", hardware_dir)
    monkeypatch.setattr(golden, "default_recipe_root", default_recipe_root)
    alias = tmp_path / "canonical-link.yaml"
    alias.symlink_to(recipe)
    for path in (hardware, recipe, alias):
        with pytest.raises(ValueError, match="canonical repository goldens"):
            load_working_targets(path)


def test_copied_verified_rows_resolve_as_working_candidates(tmp_path):
    document = _document(_matmul("mm", knobs={"TILE": "f2x2"}, emmy_us=9.0, cublas_us=10.0))
    copied = tmp_path / "copied.yaml"
    dump_golden_file(document, copied)

    _loaded, targets = load_working_targets(copied)

    proposal_indexes = {path for target in targets for path, _knobs in target.proposals}
    expected = {
        (config_index, realization_index)
        for config_index, config in enumerate(document["configs"])
        for realization_index, _realization in enumerate(config["realizations"])
    }
    assert proposal_indexes == expected


def test_working_gpu_guard_allows_portable_trace_and_rejects_mismatch():
    ctx = SimpleNamespace(compute_capability=(9, 0), gpu_name="NVIDIA H100 80GB HBM3")
    validate_working_gpu({"compute_cap": [0, 0]}, ctx)
    validate_working_gpu({"compute_cap": [9, 0], "gpu_name": "NVIDIA H100 80GB HBM3"}, ctx)

    with pytest.raises(ValueError, match="compute capability"):
        validate_working_gpu({"compute_cap": [8, 0]}, ctx)
    with pytest.raises(ValueError, match="targets NVIDIA V100"):
        validate_working_gpu({"compute_cap": [9, 0], "gpu_name": "NVIDIA V100"}, ctx)


def test_multi_target_dump_uses_stable_sibling_directories(tmp_path):
    args = SimpleNamespace(dump_dir=str(tmp_path), bench=False)

    first, first_tmp = tune._bench_dump(args, target_dir=tune._target_artifact_name(0, "same/name"))
    marker = first.dir / "marker.txt"
    marker.write_text("keep")
    second, second_tmp = tune._bench_dump(args, target_dir=tune._target_artifact_name(1, "same/name"))

    assert first_tmp is None and second_tmp is None
    assert first.dir == tmp_path / "000_same_name"
    assert second.dir == tmp_path / "001_same_name"
    assert marker.read_text() == "keep"


def test_tune_one_measures_proposals_before_mcts_and_deducts_reserved_slots(monkeypatch):
    events = []
    prior = SimpleNamespace()

    async def fake_measure(_graph, proposals, **kwargs):
        events.append(("proposals", len(proposals), kwargs["max_candidates"], kwargs["prior"]))
        return [{"status": "ok"}] * len(proposals)

    class FakeStrategy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def run(self, _graph, _ctx):
            events.append(("mcts", self.kwargs["max_candidates"], self.kwargs["prior"]))
            return SimpleNamespace(n_terminals=1, prior_summaries=[], best_reward=None, assembled=None)

    monkeypatch.setattr(tune, "measure_proposals", fake_measure)
    monkeypatch.setattr(tune, "load_or_trace", lambda _args: (object(), None, None))
    monkeypatch.setattr("emmy.compiler.pipeline.search.prior.load_prior", lambda seed: prior)
    monkeypatch.setattr("emmy.compiler.pipeline.search.strategy.TwoLevelStrategy", FakeStrategy)
    args = SimpleNamespace(
        patience=10,
        explore_eps=0.0,
        ucb_c=1.4,
        seed=3,
        max_candidates=5,
        verbose=0,
        quiet=False,
    )
    rankings = []

    def persist(measured):
        rankings.extend(measured)
        events.append(("persist", len(measured)))

    tune._tune_one(
        args,
        backends=[object()],
        db=object(),
        ctx=SimpleNamespace(compile_flags="-O1"),
        dump=None,
        proposals=[((0, 0), {"TILE": "a"}), ((0, 1), {"TILE": "b"})],
        proposal_ranking_callback=persist,
    )

    assert events == [("proposals", 2, 5, prior), ("persist", 2), ("mcts", 3, prior)]
    assert rankings == [{"status": "ok"}, {"status": "ok"}]


def test_multi_gpu_working_sweep_shares_slots_and_prior_across_targets(monkeypatch):
    async def warm_worker():
        return None

    backends = [
        SimpleNamespace(name="gpu0", warm_async_worker=warm_worker),
        SimpleNamespace(name="gpu1", warm_async_worker=warm_worker),
    ]
    targets = [
        WorkingGoldenTarget("a", "code-a", None, None),
        WorkingGoldenTarget("b", "code-b", None, None),
    ]
    prior = SimpleNamespace(
        fitted=False,
        trajectory=[],
        maybe_refit=lambda **_kwargs: None,
        checkpoint=lambda: None,
    )
    active = 0
    max_active = 0
    seen_prior = []
    seen_queues = []

    async def fake_measure(*_args, **_kwargs):
        return []

    class FakeStrategy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def run(self, _graph, _ctx):
            nonlocal active, max_active
            kwargs = self.kwargs
            seen_prior.append(kwargs["prior"])
            seen_queues.append(kwargs["backend_slots"])
            backend = await kwargs["backend_slots"].get()
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            kwargs["backend_slots"].put_nowait(backend)
            return SimpleNamespace(prior_summaries=[], best_reward=None, assembled=None)

    target_dirs = []

    def fake_dump(_args, *, target_dir=None):
        target_dirs.append(target_dir)
        return None, None

    monkeypatch.setattr(tune, "_bench_dump", fake_dump)
    monkeypatch.setattr(tune, "load_or_trace", lambda args: (args.code, None, None))
    monkeypatch.setattr(tune, "measure_proposals", fake_measure)
    monkeypatch.setattr(tune, "persist_tune_winner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("emmy.compiler.pipeline.search.prior.load_prior", lambda seed: prior)
    monkeypatch.setattr("emmy.compiler.pipeline.search.strategy.TwoLevelStrategy", FakeStrategy)
    args = SimpleNamespace(
        code=None,
        input=None,
        dynamic=None,
        golden_file="working.yaml",
        patience=4,
        explore_eps=0.0,
        ucb_c=1.4,
        seed=0,
        max_candidates=2,
        output=None,
        bench=False,
    )

    assert tune._tune_working_multi(args, targets, {"configs": []}, backends=backends, db=object(), ctx=object(), run_id="r") == 2
    assert max_active == 2
    assert seen_prior == [prior, prior]
    assert seen_queues[0] is seen_queues[1]
    assert target_dirs == ["000_a", "001_b"]


def test_multi_working_prepare_failure_cleans_command_temp_dump(tmp_path, monkeypatch):
    temp_dump = tmp_path / "emmy-tune-bench-created"
    temp_dump.mkdir()
    target = WorkingGoldenTarget("a", "code-a", None, None)
    monkeypatch.setattr(tune, "_bench_dump", lambda *_args, **_kwargs: (None, temp_dump))
    monkeypatch.setattr(tune, "load_or_trace", lambda _args: (_ for _ in ()).throw(ValueError("trace failed")))
    args = SimpleNamespace(code=None, input=None, dynamic=None, dump_dir=None, bench=True)

    with pytest.raises(ValueError, match="trace failed"):
        tune._tune_working_multi(args, [target], {"configs": []}, backends=[object()], db=object(), ctx=object(), run_id="r")

    assert not temp_dump.exists()
