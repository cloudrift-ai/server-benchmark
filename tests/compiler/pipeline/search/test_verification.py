"""Fresh-process working-golden verification tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from emmy.compiler.pipeline.search import verification


def _proof(**updates) -> dict:
    proof = {
        "status": "pass",
        "reference": "eager",
        "rtol": 1e-3,
        "atol": 1e-3,
        "max_abs_error": 0.001,
        "mean_abs_error": 0.0001,
        "max_rel_error": 0.002,
    }
    proof.update(updates)
    return proof


def _backend(latency: float, *, compiler: bool = False, emmy: bool = False) -> dict:
    row = {"latency_us": latency, "captured": True, "timing_semantics": "captured_whole_forward"}
    if compiler:
        row["correctness"] = {"status": "pass", "rtol": 1e-3, "atol": 1e-3, "fullgraph": True}
    if emmy:
        row["correctness"] = _proof()
    return row


def _pin(*, status: str = "ok", total_us: float | None = 12.5, flags: list[str] | None = None) -> dict:
    return {
        "kind": "ab",
        "status": status,
        "flags": flags or [],
        "total_us": total_us,
        "captured": True,
        "timing_semantics": "single_launch",
        "num_launches": 1,
        "correctness": _proof(),
    }


def _document() -> dict:
    return {
        "configs": [
            {
                "realizations": [
                    {
                        "name": "linear.layer0",
                        "knobs": {"TILE": "f2x4", "WORK": "t16x8"},
                        "ranking": {
                            "status": "ok",
                            "source": "tune",
                            "tune_winner": True,
                            "measured_knobs": {"TILE": "f2x4", "WORK": "t16x8"},
                        },
                    }
                ]
            }
        ]
    }


def _exact_result() -> dict:
    return {
        "backends": {
            "Eager PyTorch": _backend(20.0),
            "torch.compile": _backend(15.0, compiler=True),
            "Emmy": _backend(13.0, emmy=True),
        },
        "pinned": [_pin()],
    }


def test_searched_winners_require_direct_measured_knobs():
    assert verification.searched_winners(_document()) == [verification.Winner(name="linear.layer0", knobs="WORK=t16x8,TILE=f2x4")]


def test_searched_winners_reject_empty_inventory():
    with pytest.raises(verification.WorkingGoldenVerificationError, match="at least one target"):
        verification.searched_winners({"configs": []})


def test_searched_winners_reject_ambiguous_target():
    document = _document()
    document["configs"][0]["realizations"].append(document["configs"][0]["realizations"][0].copy())

    with pytest.raises(verification.WorkingGoldenVerificationError, match="exactly one"):
        verification.searched_winners(document)


def test_verify_tune_winners_runs_fresh_exact_o3_processes(monkeypatch, tmp_path):
    monkeypatch.setattr(verification, "load_golden_file", lambda _path: _document())
    commands = []

    def fake_run(command, *, env, check):
        commands.append((command, env, check))
        Path(command[command.index("--json") + 1]).write_text(json.dumps(_exact_result()), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    attempts = verification.verify_tune_winners(
        tmp_path / "working.yaml",
        tmp_path / "verification",
        emmy="./venv/bin/emmy",
        repeats=5,
        warmup=10,
        iters=100,
        cuda_visible_devices="3",
        bench_backends=("eager", "tcompile", "emmy"),
        run=fake_run,
    )

    assert len(attempts) == 5
    assert all(attempt.status == "ok" for attempt in attempts)
    assert all(command[1]["EMMY_NVCC_FLAGS"] == "" for command in commands)
    assert all(command[1]["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] == "1" for command in commands)
    assert all(command[1]["CUDA_VISIBLE_DEVICES"] == "3" for command in commands)
    assert all("--strict-correctness" in command[0] for command in commands)
    assert all(command[0][command[0].index("--ab") + 1] == "WORK=t16x8,TILE=f2x4" for command in commands)
    manifest = json.loads((tmp_path / "verification" / "manifest.json").read_text())
    assert manifest["process_repeats"] == 5
    assert manifest["attempts"][0]["winner_total_us"] == 12.5
    assert manifest["attempts"][0]["deploy_emmy_us"] == 13.0


def test_verify_tune_winners_preserves_failed_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(verification, "load_golden_file", lambda _path: _document())

    def fake_run(command, *, env, check):
        del env, check
        result = _exact_result()
        result["pinned"] = [_pin(status="pin_unmatched", total_us=None, flags=["bad pin"])]
        Path(command[command.index("--json") + 1]).write_text(json.dumps(result), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    with pytest.raises(verification.WorkingGoldenVerificationError, match="verification attempts failed"):
        verification.verify_tune_winners(
            tmp_path / "working.yaml",
            tmp_path / "verification",
            emmy="emmy",
            repeats=2,
            warmup=5,
            iters=20,
            cuda_visible_devices=None,
            bench_backends=("eager", "tcompile", "emmy"),
            run=fake_run,
        )

    manifest = json.loads((tmp_path / "verification" / "manifest.json").read_text())
    assert manifest["attempts"][-1]["status"] == "integrity_failed"


def test_validate_ab_json_records_missing_optional_compiler(tmp_path):
    output = tmp_path / "ab.json"
    output.write_text(json.dumps(_exact_result()), encoding="utf-8")

    winner, backends, missing = verification._validate_ab_json(
        output,
        required_backends=("eager", "tcompile", "emmy"),
        optional_backends=("hidet",),
    )

    assert winner == 12.5
    assert backends["torch.compile"] == 15.0
    assert missing == ["Hidet"]


def test_validate_ab_json_requires_e2e_timing_for_multi_launch_winner(tmp_path):
    output = tmp_path / "ab.json"
    result = _exact_result()
    result["pinned"][0].update({"num_launches": 2, "timing_semantics": "per_launch_sum"})
    output.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(verification.WorkingGoldenVerificationError, match="whole_program_e2e"):
        verification._validate_ab_json(
            output,
            required_backends=("eager", "tcompile", "emmy"),
            optional_backends=(),
        )


def test_verify_cold_greedy_uses_unique_empty_local_state(monkeypatch, tmp_path):
    document = {"configs": [{"realizations": [{"name": "linear.layer0"}]}]}
    monkeypatch.setattr(verification, "load_golden_file", lambda _path: document)
    commands = []

    def fake_run(command, *, env, check):
        assert check is False
        commands.append((command, env))
        assert not Path(env["EMMY_TUNE_DB"]).exists()
        assert not Path(env["EMMY_ONLINE_FILE"]).exists()
        assert not Path(env["EMMY_CUBIN_CACHE"]).exists()
        assert "--ab" not in command
        output = Path(command[command.index("--json") + 1])
        output.write_text(
            json.dumps({"backends": {"Emmy": _backend(11.5, emmy=True)}, "pinned": []}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    attempts = verification.verify_cold_greedy(
        tmp_path / "inventory.yaml",
        tmp_path / "greedy",
        emmy="./venv/bin/emmy",
        repeats=5,
        warmup=10,
        iters=100,
        cuda_visible_devices="2",
        run=fake_run,
    )

    assert len(attempts) == 5
    assert all(attempt.greedy_emmy_us == 11.5 for attempt in attempts)
    assert len({command[1]["EMMY_TUNE_DB"] for command in commands}) == 5
    manifest = json.loads((tmp_path / "greedy" / "manifest.json").read_text())
    assert manifest["mode"] == "cold_local_evidence"
