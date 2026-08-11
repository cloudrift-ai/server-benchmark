"""Tests for scripts/verify_working_golden_greedy.py."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "verify_working_golden_greedy.py"
_SPEC = importlib.util.spec_from_file_location("verify_working_golden_greedy", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
greedy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = greedy
_SPEC.loader.exec_module(greedy)


def test_verify_runs_before_tuning_with_unique_empty_state(monkeypatch, tmp_path):
    document = {"configs": [{"realizations": [{"name": "linear.layer0"}]}]}
    monkeypatch.setattr(greedy, "load_golden_file", lambda _path: document)
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
            json.dumps(
                {
                    "backends": {
                        "Emmy": {
                            "latency_us": 11.5,
                            "captured": True,
                            "timing_semantics": "captured_whole_forward",
                        }
                    },
                    "pinned": [],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    attempts = greedy.verify(
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
    assert manifest["evidence_state"] == "empty_local_db_online_cubin_before_tuning"
