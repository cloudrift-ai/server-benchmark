"""Working-golden execution tests."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from emmy.commands import run as run_mod


def _parser():
    parser = argparse.ArgumentParser()
    run_mod.register_run_command(parser.add_subparsers())
    return parser


def test_run_golden_schema_matches_compile():
    """``run`` spells golden selection the way every replaying command does: ``--golden PATH`` is
    the file, ``--realization NAME`` the row inside it — one pair, one meaning, on ``run`` /
    ``compile`` / ``tune`` / ``serve`` alike."""
    args = _parser().parse_args(["run", "--golden", "working.yaml", "--realization", "linear.layer0", "--gpu-arch", "sm_90"])

    assert args.golden == "working.yaml"
    assert args.realization == "linear.layer0"
    assert args.gpu_arch == "sm_90"
    for removed in ("all_targets", "repeats", "require_kernel_source", "golden_target", "golden_file"):
        assert not hasattr(args, removed)
    with pytest.raises(SystemExit):
        _parser().parse_args(["run", "--golden-file", "working.yaml"])


def _args(tmp_path, **updates):
    values = {
        "golden": str(tmp_path / "working.yaml"),
        "realization": None,
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

    assert [args.realization for args in calls] == ["linear.layer0", "linear.layer1"]
    assert all(args.golden.endswith("working.yaml") and args._explicit_realization is False for args in calls)


def test_naming_one_target_skips_the_multi_target_walk(run_cli):
    """``--realization NAME`` goes straight down the single-run path — the walk is for a bare file."""
    rc, stdout, stderr = run_cli("run", "--realization", "linear.layer0", "--code", "torch.randn(4, 4)")

    assert rc == 2
    assert "mutually exclusive" in stdout + stderr


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
        realization="missing",
        golden=str(tmp_path / "absent.yaml"),
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
