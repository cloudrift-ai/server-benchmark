"""Working-golden execution tests."""

import argparse
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from emmy.commands import run as run_mod


def _parser():
    parser = argparse.ArgumentParser()
    run_mod.register_run_command(parser.add_subparsers())
    return parser


def test_run_golden_schema_has_no_process_wrapper_flags():
    args = _parser().parse_args(["run", "--golden", "working.yaml", "--target", "linear.layer0", "--gpu-arch", "sm_90"])

    assert args.golden == "working.yaml"
    assert args.golden_target == "linear.layer0"
    assert args.gpu_arch == "sm_90"
    for removed in ("all_targets", "repeats", "require_kernel_source", "golden_file"):
        assert not hasattr(args, removed)


def test_target_requires_golden(run_cli):
    rc, stdout, stderr = run_cli("run", "--target", "linear.layer0")

    assert rc == 2
    assert "--target requires --golden PATH" in stdout + stderr


def _args(tmp_path, **updates):
    values = {
        "golden": str(tmp_path / "working.yaml"),
        "golden_target": None,
        "input": None,
        "code": None,
        "ir": None,
        "json": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _patch_records(monkeypatch, names):
    from emmy.compiler.pipeline.search import golden

    monkeypatch.setattr(golden, "load_golden_file", lambda _path: {})
    monkeypatch.setattr(golden, "load_golden_records", lambda _document: [SimpleNamespace(name=name) for name in names])


def test_golden_runs_every_distinct_target_in_process(monkeypatch, tmp_path):
    _patch_records(monkeypatch, ["linear.layer0", "linear.layer0", "linear.layer1"])
    calls = []
    monkeypatch.setattr(run_mod, "_handle_run_once", calls.append)

    run_mod._run_golden_targets(_args(tmp_path))

    assert [args.golden for args in calls] == ["linear.layer0", "linear.layer1"]
    assert all(args.golden_file.endswith("working.yaml") for args in calls)


def test_target_limits_golden_to_one_match(monkeypatch, tmp_path):
    _patch_records(monkeypatch, ["linear.layer0", "linear.layer1"])
    calls = []
    monkeypatch.setattr(run_mod, "_handle_run_once", calls.append)

    run_mod._run_golden_targets(_args(tmp_path, golden_target="layer1", json=str(tmp_path / "result.json")))

    assert len(calls) == 1
    assert calls[0].golden == "layer1"
    assert calls[0].json.endswith("result.json")


def test_multi_target_json_uses_one_readable_file_per_target(monkeypatch, tmp_path):
    _patch_records(monkeypatch, ["linear/layer0", "linear/layer1"])
    calls = []
    monkeypatch.setattr(run_mod, "_handle_run_once", calls.append)
    output = tmp_path / "results"

    run_mod._run_golden_targets(_args(tmp_path, json=str(output)))

    assert [Path(args.json).name for args in calls] == ["000-linear_layer0.json", "001-linear_layer1.json"]
    assert output.is_dir()


def test_strict_result_requires_backends_capture_and_correctness():
    proof = {
        "status": "pass",
        "reference": "eager",
        "rtol": 1e-3,
        "atol": 1e-3,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "max_rel_error": 0.0,
    }
    bench = SimpleNamespace(captured=True, num_launches=1, per_launch=[], e2e_min_ms=None)
    args = SimpleNamespace(bench_backends="eager,tcompile,emmy", ab=None)
    results = {"Eager PyTorch": 20.0, "torch.compile": 15.0, "Emmy": 10.0}

    assert run_mod._strict_benchmark_errors(args, results, bench, True, proof, []) == []
    errors = run_mod._strict_benchmark_errors(args, {"Emmy": 10.0}, bench, True, proof, [])
    assert "torch.compile" in " ".join(errors)


def test_strict_result_requires_every_requested_exact_row():
    args = SimpleNamespace(bench_backends="emmy", ab=["TILE=f2x4"])
    proof = {
        "status": "pass",
        "reference": "eager",
        "rtol": 1e-3,
        "atol": 1e-3,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "max_rel_error": 0.0,
    }
    bench = SimpleNamespace(captured=True)

    errors = run_mod._strict_benchmark_errors(args, {"Emmy": 10.0}, bench, True, proof, [])

    assert "expected 1 exact --ab row(s), got 0" in errors


def test_strict_result_accepts_same_input_greedy_only_for_reference_free_loop():
    args = SimpleNamespace(bench_backends="emmy", ab=None)
    proof = {
        "status": "pass",
        "reference": "same-input-greedy",
        "rtol": 1e-3,
        "atol": 1e-3,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "max_rel_error": 0.0,
    }
    bench = SimpleNamespace(captured=True, num_launches=1, per_launch=[], e2e_min_ms=None)
    results = {"Emmy": 10.0}

    runnable_errors = run_mod._strict_benchmark_errors(args, results, bench, True, proof, [])
    assert "same-input-greedy" not in " ".join(runnable_errors)
    assert "strict eager correctness" in " ".join(runnable_errors)

    missing_errors = run_mod._strict_benchmark_errors(
        args,
        results,
        bench,
        True,
        proof,
        [],
        frontend_runnable=False,
        same_input_reference=False,
    )
    assert "same-input greedy reference is unavailable" in missing_errors

    assert (
        run_mod._strict_benchmark_errors(
            args,
            results,
            bench,
            True,
            proof,
            [],
            frontend_runnable=False,
            same_input_reference=True,
        )
        == []
    )


def test_golden_document_is_parsed_once_for_every_target(monkeypatch, tmp_path):
    """A whole-model inventory must not be re-read and re-validated per target."""
    from emmy.compiler.pipeline.search import golden

    loads = []
    document = {"configs": []}
    monkeypatch.setattr(golden, "load_golden_file", lambda _path: loads.append(_path) or document)
    monkeypatch.setattr(golden, "load_golden_records", lambda _document: [SimpleNamespace(name=name) for name in ("a", "b", "c")])
    calls = []
    monkeypatch.setattr(run_mod, "_handle_run_once", calls.append)

    run_mod._run_golden_targets(_args(tmp_path))

    assert len(loads) == 1
    assert [args._golden_document for args in calls] == [document] * 3


def test_resolve_golden_arg_prefers_the_document_the_caller_loaded(monkeypatch, tmp_path):
    """With a document supplied, resolution must not touch the file at all."""
    from emmy.commands import compile as compile_mod
    from emmy.compiler.pipeline.search import golden

    def _explode(_path, **_kwargs):
        raise AssertionError("load_golden_file must not be called when a document is supplied")

    monkeypatch.setattr(golden, "load_golden_file", _explode)
    monkeypatch.setattr(golden, "load_golden_records", lambda _document: [])

    args = SimpleNamespace(
        golden="missing",
        golden_file=str(tmp_path / "absent.yaml"),
        _golden_document={"configs": []},
        input=None,
        code=None,
        ir=None,
        dynamic=None,
    )
    # No record matches "missing", so resolution exits 2 — reaching that exit proves it
    # resolved against the supplied document instead of reading the absent file.
    with pytest.raises(SystemExit) as excinfo:
        compile_mod.resolve_golden_arg(args)
    assert excinfo.value.code == 2


# --- what a run must be before its benched rows are stored ---------------------------------
#
# ``_record_bench_nodes`` writes into the tune DB, and one of the tables it writes is read by a
# later compile to decide a fork. These cover the conditions that disqualify a whole run, as
# opposed to the per-row integrity flags ``_recordable_bench_leaves`` already filters.


def _recording_args(**updates):
    values = {"no_record_nodes": False, "warmup": 5, "iters": 20}
    values.update(updates)
    return SimpleNamespace(**values)


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    """Capture what ``_record_bench_nodes`` would write, without touching a real tune DB."""
    from emmy.commands import compile as compile_mod
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search import bench_record

    writes = []
    monkeypatch.setattr(bench_record, "record_bench_leaves", lambda *a, **k: writes.append((a, k)) or (0, 0))
    monkeypatch.setattr(compile_mod, "resolve_tune_db", lambda: tmp_path / "autotune.db")
    monkeypatch.setattr(Context, "probe", classmethod(lambda cls: Context((8, 9))))
    monkeypatch.setattr(run_mod, "_recordable_bench_leaves", lambda *_: [object()])
    return writes


def test_a_clean_bench_records_its_rows(recorded):
    run_mod._record_bench_nodes(_recording_args(), [SimpleNamespace()], None)

    assert len(recorded) == 1


def test_a_bench_below_the_tune_standard_records_nothing(recorded, capsys):
    run_mod._record_bench_nodes(_recording_args(warmup=1, iters=3), [SimpleNamespace()], None)

    assert recorded == []
    assert "below the tune bench standard" in capsys.readouterr().out


def test_a_cross_target_run_records_nothing(monkeypatch, recorded, capsys):
    """The kernel ran on this card; the row would key to the target's capability."""
    from emmy.compiler import target

    monkeypatch.setattr(target, "compute_capability", lambda: (9, 0))
    monkeypatch.setattr(target, "live_compute_capability", lambda: (12, 0))

    run_mod._record_bench_nodes(_recording_args(), [SimpleNamespace()], None)

    assert recorded == []
    assert "NOT recorded" in capsys.readouterr().out


def _stub_run_ir(monkeypatch, tmp_path, *, accuracy_error=None, from_ir_file=False):
    """Drive ``_handle_run_ir`` far enough to reach its recording decision, on either input.

    ``CudaBackend`` / ``CompilerDump`` are already parameters of the handler, so the stubs go
    in there; the pass pipeline and the torch-reference probe are the only two module-level
    dependencies left to replace. ``golden_configs`` is empty on purpose — the decision under
    test is whether the recorder is reached at all.
    """
    import json

    from emmy.compiler import pipeline as pipeline_mod
    from emmy.compiler.backend import torch_ref
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp

    monkeypatch.setattr(torch_ref, "is_runnable", lambda _graph: False)
    monkeypatch.setattr(pipeline_mod.Pipeline, "build", classmethod(lambda cls, _passes: SimpleNamespace(run=lambda g, **_kw: g)))
    calls = []
    monkeypatch.setattr(run_mod, "_record_bench_nodes", lambda *a: calls.append(a))

    bench = SimpleNamespace(time_ms=1.0, min_ms=1.0, per_launch=None, num_launches=0, captured=True, e2e_ms=None, e2e_min_ms=None)

    class _Backend:
        bench_compile_timeout_s = 1.0
        bench_run_timeout_s = 1.0
        tune_db = None

        def __init__(self, **_kwargs) -> None:
            pass

        async def benchmark_compare_async(self, *_args, **_kwargs):
            return {
                "results": {"Emmy": 1.0},
                "result": bench,
                "captured": True,
                "torch_available": False,
                "accuracy_error": accuracy_error,
                "run_io": None,
                "reference_run_us": None,
                "sym_env": {},
                "correctness": None,
                "greedy_error": None,
            }

        async def bench_pinned_async(self, _graph, **_kwargs):
            return bench, None

        async def aclose_async_worker(self) -> None:
            pass

    graph = Graph()
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1,)), node_id="x")
    ir_path = None
    if from_ir_file:
        # The live shape of the excluded input: a stage dump plus --ab, which reaches
        # ``ab_benches`` / ``greedy_iso`` through the handler's own --ab branch.
        ir_path = tmp_path / "dump.json"
        ir_path.write_text(json.dumps(graph.to_dict()))
    args = SimpleNamespace(
        _golden_graph=None if from_ir_file else graph,
        ir=str(ir_path) if from_ir_file else None,
        ab=["TILE=f2x4"] if from_ir_file else None,
        golden_configs=[],
        dynamic=None,
        dump_dir=None,
        debug=False,
        bench=True,
        strict_correctness=False,
        warmup=5,
        iters=20,
        seed=0,
        bench_backends="emmy",
        json=None,
        profile=False,
        no_record_nodes=False,
    )
    # A --ab row that cannot realize on this stub graph makes the command exit non-zero; the
    # recording decision is taken before that, which is the part under test.
    with contextlib.suppress(SystemExit):
        run_mod._handle_run_ir(args, _Backend, SimpleNamespace(resolve=staticmethod(lambda _d: None)))
    return calls


def test_a_golden_replay_reaches_the_recorder(monkeypatch, tmp_path):
    assert len(_stub_run_ir(monkeypatch, tmp_path)) == 1


def test_a_golden_replay_that_computed_the_wrong_answer_records_nothing(monkeypatch, tmp_path):
    """Unlike the --code path, this one does not exit on a bad answer without --strict."""
    assert _stub_run_ir(monkeypatch, tmp_path, accuracy_error="emmy vs eager: max_diff 3.2") == []


def test_a_direct_ir_input_records_nothing(monkeypatch, tmp_path):
    """Serialization drops the knobs and the rewrite chain, so a row off a stage dump would name
    only what the remaining passes happened to re-decide. The handler serves both inputs, and
    --ab on a dump reaches the same benched rows, so the exclusion has to hold at the call."""
    assert _stub_run_ir(monkeypatch, tmp_path, from_ir_file=True) == []
