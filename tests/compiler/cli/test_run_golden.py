"""Working-golden execution tests."""

import argparse
from pathlib import Path
from types import SimpleNamespace

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
