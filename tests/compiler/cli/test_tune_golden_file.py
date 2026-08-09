"""Working-golden target and ranking plumbing for ``emmy tune --golden-file``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from emmy.commands import tune
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.pipeline.search.golden import dump_golden_file, load_golden_file
from emmy.compiler.pipeline.search.working_golden import (
    WorkingGoldenTarget,
    load_working_targets,
    persist_proposal_rankings,
    persist_tune_winner,
    realized_tuning_knobs,
    validate_working_gpu,
)
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


def _matmul(name: str, *, knobs=None, emmy_us=None, cublas_us=None):
    entry = {
        "name": name,
        "target": {"origins": ["matmul"]},
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
    programs = {}
    program_id = intern_program(programs, graph)
    rows = []
    for entry in entries:
        row = dict(entry)
        row["program"] = program_id
        rows.append(row)
    return {"format_version": 2, "compute_cap": [8, 9], "programs": programs, "configs": rows}


def test_working_file_groups_candidate_rows_and_recovers_embedded_program(tmp_path):
    path = tmp_path / "trace.yaml"
    dump_golden_file(_document(_matmul("mm"), _matmul("mm", knobs={"TILE": "f2x2"})), path)

    document, targets = load_working_targets(path)

    assert document["configs"][0]["name"] == "mm"
    assert len(targets) == 1
    mm = targets[0]
    assert mm.code is None and mm.input is None and isinstance(mm.program.nodes["matmul"].op, MatmulOp)
    assert mm.entry_indexes == [0, 1]
    assert mm.proposals == [(1, {"TILE": "f2x2"})]


def test_empty_knob_map_is_a_forkless_proposal_not_inventory(tmp_path):
    path = tmp_path / "working.yaml"
    dump_golden_file(_document(_matmul("mm"), _matmul("mm", knobs={})), path)

    loaded_document, targets = load_working_targets(path)

    assert targets[0].entry_indexes == [0, 1]
    assert targets[0].proposals == [(1, {})]
    assert loaded_document["format_version"] == 2


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
    assert "ranking" not in got["configs"][0]
    assert got["configs"][0]["measurements"]["emmy_us"] == 9.0
    assert got["configs"][1]["ranking"]["source"] == "proposal"
    assert got["configs"][1]["ranking"]["latency_us"] == 8.0
    winner = got["configs"][2]
    assert winner["knobs"] == {"TILE": "f8x2"}
    assert winner["ranking"]["source"] == "tune"
    assert winner["ranking"]["tune_winner"] is True
    assert "measurements" not in winner


def test_ambiguous_multi_cuda_winner_is_not_annotated(tmp_path):
    path = tmp_path / "working.yaml"
    dump_golden_file(_document(_matmul("mm")), path)
    document, targets = load_working_targets(path)
    result = SimpleNamespace(best_reward=SimpleNamespace(searched_winner=lambda: None), assembled=object())

    persist_tune_winner(path, document, targets[0], result.best_reward.searched_winner(), compile_flags="-O1")

    got = load_golden_file(path)
    assert len(got["configs"]) == 1
    assert got["configs"][0]["name"] == "mm"
    assert got["configs"][0]["target"] == {"origins": ["matmul"]}


def test_working_file_rejects_canonical_path_and_symlink(tmp_path):
    from emmy.compiler.pipeline.search import golden

    canonical = next(golden._GOLDENS_DIR.glob("*.yaml"))
    for path in (canonical, tmp_path / "canonical-link.yaml"):
        if path != canonical:
            path.symlink_to(canonical)
        with pytest.raises(ValueError, match="canonical repository goldens"):
            load_working_targets(path)


def test_copied_repository_rows_resolve_as_working_candidates(tmp_path):
    from emmy.compiler.pipeline.search import golden

    canonical = golden._GOLDENS_DIR / "rtx4080_sm89.yaml"
    document = load_golden_file(canonical)
    copied = tmp_path / canonical.name
    dump_golden_file(document, copied)

    _loaded, targets = load_working_targets(copied)

    proposal_indexes = {i for target in targets for i, _knobs in target.proposals}
    assert proposal_indexes == set(range(len(document["configs"])))


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

    async def fake_two_level(_graph, **kwargs):
        events.append(("mcts", kwargs["max_candidates"], kwargs["prior"]))
        return SimpleNamespace(n_terminals=1, prior_summaries=[], best_reward=None, assembled=None)

    monkeypatch.setattr(tune, "measure_proposals", fake_measure)
    monkeypatch.setattr(tune, "load_or_trace", lambda _args: (object(), None, None))
    monkeypatch.setattr("emmy.compiler.pipeline.search.prior.load_prior", lambda seed: prior)
    monkeypatch.setattr("emmy.compiler.pipeline.search.two_level.run_two_level_tune", fake_two_level)
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
        proposals=[(0, {"TILE": "a"}), (1, {"TILE": "b"})],
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

    async def fake_two_level(_graph, **kwargs):
        nonlocal active, max_active
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
    monkeypatch.setattr("emmy.compiler.pipeline.search.two_level.run_two_level_tune", fake_two_level)
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
