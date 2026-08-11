"""Tests for scripts/verify_working_golden_winners.py."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "verify_working_golden_winners.py"
_SPEC = importlib.util.spec_from_file_location("verify_working_golden_winners", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
verifier = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verifier
_SPEC.loader.exec_module(verifier)


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


def test_searched_winners_require_direct_measured_knobs():
    assert verifier.searched_winners(_document()) == [verifier.Winner(name="linear.layer0", knobs="WORK=t16x8,TILE=f2x4")]


def test_searched_winners_reject_ambiguous_target():
    document = _document()
    document["configs"][0]["realizations"].append(document["configs"][0]["realizations"][0].copy())

    with pytest.raises(verifier.VerificationError, match="exactly one"):
        verifier.searched_winners(document)


def test_verify_runs_fresh_exact_o3_processes_and_checks_json(monkeypatch, tmp_path):
    monkeypatch.setattr(verifier, "load_golden_file", lambda _path: _document())
    commands = []

    def fake_run(command, *, env, check):
        commands.append((command, env, check))
        output = Path(command[command.index("--json") + 1])
        output.write_text(
            json.dumps(
                {
                    "backends": {
                        "Eager PyTorch": _backend(20.0),
                        "torch.compile": _backend(15.0, compiler=True),
                        "Emmy": _backend(13.0, emmy=True),
                    },
                    "pinned": [_pin()],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    attempts = verifier.verify(
        tmp_path / "working.yaml",
        tmp_path / "verification",
        emmy="./venv/bin/emmy",
        repeats=5,
        warmup=10,
        iters=100,
        cuda_visible_devices="3",
        run=fake_run,
    )

    assert len(attempts) == 5
    assert all(attempt.status == "ok" for attempt in attempts)
    assert all(command[1]["EMMY_NVCC_FLAGS"] == "" for command in commands)
    assert all(command[1]["TORCHINDUCTOR_MAX_AUTOTUNE"] == "1" for command in commands)
    assert all(command[1]["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] == "1" for command in commands)
    assert all(command[1]["TORCHINDUCTOR_CUDAGRAPHS"] == "1" for command in commands)
    assert all(command[1]["CUDA_VISIBLE_DEVICES"] == "3" for command in commands)
    assert all("--strict-correctness" in command[0] for command in commands)
    assert all(command[0][command[0].index("--ab") + 1] == "WORK=t16x8,TILE=f2x4" for command in commands)
    manifest = json.loads((tmp_path / "verification" / "manifest.json").read_text())
    assert manifest["repeats"] == 5
    assert manifest["attempts"][0]["winner_total_us"] == 12.5
    assert manifest["attempts"][0]["deploy_emmy_us"] == 13.0


def test_verify_preserves_failed_attempt_and_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(verifier, "load_golden_file", lambda _path: _document())

    def fake_run(command, *, env, check):
        del env, check
        output = Path(command[command.index("--json") + 1])
        output.write_text(
            json.dumps(
                {
                    "backends": {
                        "Eager PyTorch": _backend(20.0),
                        "torch.compile": _backend(15.0, compiler=True),
                        "Emmy": _backend(13.0, emmy=True),
                    },
                    "pinned": [_pin(status="pin_unmatched", total_us=None, flags=["bad pin"])],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    with pytest.raises(verifier.VerificationError, match="verification attempts failed"):
        verifier.verify(
            tmp_path / "working.yaml",
            tmp_path / "verification",
            emmy="emmy",
            repeats=2,
            warmup=5,
            iters=20,
            cuda_visible_devices=None,
            run=fake_run,
        )

    manifest = json.loads((tmp_path / "verification" / "manifest.json").read_text())
    assert manifest["attempts"][-1]["status"] == "integrity_failed"


def test_validate_ab_json_requires_all_comparison_backends(tmp_path):
    output = tmp_path / "ab.json"
    output.write_text(
        json.dumps(
            {
                "backends": {
                    "Eager PyTorch": _backend(20.0),
                    "Emmy": _backend(13.0, emmy=True),
                },
                "pinned": [_pin()],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(verifier.VerificationError, match="torch.compile"):
        verifier._validate_ab_json(
            output,
            required_backends=("eager", "tcompile", "emmy"),
            optional_backends=(),
        )


def test_validate_ab_json_records_missing_optional_compiler(tmp_path):
    output = tmp_path / "ab.json"
    output.write_text(
        json.dumps(
            {
                "backends": {
                    "Eager PyTorch": _backend(20.0),
                    "torch.compile": _backend(15.0, compiler=True),
                    "Emmy": _backend(13.0, emmy=True),
                },
                "pinned": [_pin()],
            }
        ),
        encoding="utf-8",
    )

    winner, backends, missing = verifier._validate_ab_json(
        output,
        required_backends=("eager", "tcompile", "emmy"),
        optional_backends=("hidet",),
    )

    assert winner == 12.5
    assert backends["torch.compile"] == 15.0
    assert missing == ["Hidet"]


def test_validate_ab_json_requires_e2e_timing_for_multi_launch_winner(tmp_path):
    output = tmp_path / "ab.json"
    pin = _pin()
    pin.update({"num_launches": 2, "timing_semantics": "per_launch_sum"})
    output.write_text(
        json.dumps(
            {
                "backends": {
                    "Eager PyTorch": _backend(20.0),
                    "torch.compile": _backend(15.0, compiler=True),
                    "Emmy": _backend(13.0, emmy=True),
                },
                "pinned": [pin],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(verifier.VerificationError, match="whole_program_e2e"):
        verifier._validate_ab_json(
            output,
            required_backends=("eager", "tcompile", "emmy"),
            optional_backends=(),
        )


def test_validate_ab_json_rejects_failed_exact_winner_correctness(tmp_path):
    output = tmp_path / "ab.json"
    pin = _pin()
    pin["correctness"] = _proof(status="fail", error="tolerance exceeded")
    output.write_text(
        json.dumps(
            {
                "backends": {
                    "Eager PyTorch": _backend(20.0),
                    "torch.compile": _backend(15.0, compiler=True),
                    "Emmy": _backend(13.0, emmy=True),
                },
                "pinned": [pin],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(verifier.VerificationError, match="strict eager correctness"):
        verifier._validate_ab_json(
            output,
            required_backends=("eager", "tcompile", "emmy"),
            optional_backends=(),
        )
