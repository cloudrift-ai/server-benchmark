"""CLI tests for direct ``emmy tune`` targets after working-golden unification."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from emmy.commands import tune


def _args(**over):
    base = dict(
        kernel=None,
        golden_file=None,
        code=None,
        input=None,
        dynamic=None,
        output=None,
        ucb_c=1.4142,
        seed=0,
        patience=None,
        max_candidates=None,
        explore_eps=None,
        nvcc_flags=None,
        dump_dir=None,
        quiet=False,
        verbose=0,
        bench=False,
        clean=False,
        bench_backends="eager,emmy",
        warmup=10,
        iters=100,
        gpus=None,
        devices=None,
        target=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_direct_code_target_keeps_dynamic_spec():
    code = "torch.matmul(x, torch.randn(8, 8))"
    assert tune._tune_targets(_args(code=code, dynamic=["seq_len@x:0"])) == [
        (code, code, None, ["seq_len@x:0"]),
    ]


def test_direct_code_and_input_conflict():
    with pytest.raises(SystemExit) as exc:
        tune._tune_targets(_args(code="torch.ones(1)", input="model"))
    assert exc.value.code == 2


def test_tune_parser_drops_legacy_dataset_db_and_golden_flags():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    tune.register_tune_command(subparsers)

    parsed = parser.parse_args(["tune", "model", "--devices", "1"])
    assert parsed.input == "model" and parsed.devices == "1"
    for legacy in ("--dataset", "--db", "--golden"):
        with pytest.raises(SystemExit):
            parser.parse_args(["tune", "model", legacy, "value"])


def _stub_runtime(monkeypatch):
    tuned_codes: list[str] = []

    def fake_tune_one(args, **_kwargs):
        tuned_codes.append(args.code)
        return SimpleNamespace(best_reward=None, assembled=None), None

    monkeypatch.setattr(tune, "_tune_one", fake_tune_one)
    monkeypatch.setattr(tune, "_tune_backend", lambda device_id=None: SimpleNamespace(device_id=device_id))
    monkeypatch.setattr(tune, "_bench_dump", lambda args, **_kwargs: (None, None))
    monkeypatch.setattr(tune, "setup_pipeline_runtime", lambda args: None)
    monkeypatch.setattr(tune, "apply_nvcc_flags", lambda args, default: "-Xcicc -O1")
    monkeypatch.setattr(tune, "resolve_tune_db", lambda: "/tmp/tune-targets.db")
    monkeypatch.setattr(tune, "_context_for_device", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("emmy.compiler.pipeline.search.SearchDB", lambda path: object())
    monkeypatch.setattr(tune, "_exit_flushed", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    return tuned_codes


def test_direct_target_uses_shared_tune_loop(monkeypatch):
    tuned_codes = _stub_runtime(monkeypatch)
    code = "torch.matmul(torch.randn(8, 8), torch.randn(8, 8))"
    with pytest.raises(SystemExit) as exc:
        tune.handle_tune(_args(code=code))
    assert exc.value.code == 0
    assert tuned_codes == [code]


def test_kernel_filter_requires_working_golden_file():
    with pytest.raises(SystemExit) as exc:
        tune.handle_tune(_args(code="torch.ones(1)", kernel="matmul"))
    assert exc.value.code == 2


def test_serial_tune_failure_cleans_command_temp_dump(tmp_path, monkeypatch):
    _stub_runtime(monkeypatch)
    temp_dump = tmp_path / "emmy-tune-bench-created"
    temp_dump.mkdir()
    monkeypatch.setattr(tune, "_bench_dump", lambda *_args, **_kwargs: (None, temp_dump))
    monkeypatch.setattr(tune, "_tune_one", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bench failed")))

    with pytest.raises(SystemExit) as exc:
        tune.handle_tune(_args(code="torch.ones(1)", bench=True))

    assert exc.value.code == 1
    assert not temp_dump.exists()
