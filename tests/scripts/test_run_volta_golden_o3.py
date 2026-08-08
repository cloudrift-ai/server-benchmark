import json
from pathlib import Path
from subprocess import CompletedProcess

from scripts import run_volta_golden_o3


def test_run_one_pins_candidate_and_parses_bench_rows(monkeypatch, tmp_path: Path) -> None:
    for child in ("logs", "cubins", "locks", "records"):
        (tmp_path / child).mkdir()
    candidate = {
        "name": "qwen35_122b.test",
        "knobs": {
            "WORK": "w1x2",
            "TILE": "mma_m8n8k4_f16_f32/f2x2/k2",
            "REDUCE": "g2k",
            "STAGE": "",
            "LOOPIFY": "0",
            "RASTER": "",
        },
    }
    captured = {}

    def fake_run(command, *, env, capture_output, text, timeout):
        captured.update(command=command, env=env, capture_output=capture_output, text=text, timeout=timeout)
        record_path = Path(command[command.index("--json") + 1])
        record_path.write_text(
            json.dumps(
                {
                    "golden": "qwen35_122b.test",
                    "gpu": run_volta_golden_o3.EXPECTED_GPU,
                    "warmup": run_volta_golden_o3.WARMUP,
                    "iters": run_volta_golden_o3.ITERS,
                    "backends": {"Eager PyTorch": {"latency_us": 21.5}},
                    "greedy": {
                        "status": "ok",
                        "lane": "std",
                        "total_us": 12.25,
                        "isolated": {
                            "status": "ok",
                            "total_us": 11.5,
                            "flags": [],
                            "kernels": [{"record_knobs": candidate["knobs"]}],
                        },
                    },
                }
            )
        )
        return CompletedProcess(
            command,
            0,
            "Eager PyTorch 21.5 us\nEmmy 12.25 us\ngreedy (isolated) 11.5 --\n",
            "",
        )

    monkeypatch.setattr(run_volta_golden_o3.subprocess, "run", fake_run)
    result = run_volta_golden_o3._run_one(3, tmp_path, candidate, "candidate", 1)

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert captured["env"][run_volta_golden_o3._CHILD_MARKER] == "1"
    assert captured["env"]["EMMY_WORK"] == "w1x2"
    assert captured["env"]["EMMY_REDUCE"] == "g2k"
    assert captured["env"]["EMMY_STAGE"] == ""
    assert captured["command"][-11:-6] == ["--target", "sm_70", "--bench", "--bench-backends", "eager,emmy"]
    assert captured["command"][-6:-4] == ["--warmup", str(run_volta_golden_o3.WARMUP)]
    assert captured["command"][-4:-2] == ["--iters", str(run_volta_golden_o3.ITERS)]
    assert captured["command"][-2] == "--json"
    assert result["eager_us"] == 21.5
    assert result["emmy_us"] == 11.5
    assert result["interleaved_emmy_us"] == 12.25
    assert result["returncode"] == 0
    assert result["valid"] is True and result["integrity_errors"] == []
    assert result["record_knobs"] == candidate["knobs"]
    assert "qwen35_122b.test candidate repeat=1" in (tmp_path / "logs" / "gpu-3.log").read_text()


def test_run_one_rejects_integrity_flags_and_unrealized_knobs(monkeypatch, tmp_path: Path) -> None:
    for child in ("logs", "cubins", "locks", "records"):
        (tmp_path / child).mkdir()
    candidate = {
        "name": "qwen35_122b.test",
        "knobs": {"WORK": "w1x1", "TILE": "mma_m8n8k4_f16_f32/f2x2/k2", "REDUCE": "", "STAGE": "", "LOOPIFY": "0", "RASTER": ""},
    }

    def fake_run(command, *, env, capture_output, text, timeout):
        record_path = Path(command[command.index("--json") + 1])
        record_path.write_text(
            json.dumps(
                {
                    "golden": candidate["name"],
                    "gpu": run_volta_golden_o3.EXPECTED_GPU,
                    "warmup": run_volta_golden_o3.WARMUP,
                    "iters": run_volta_golden_o3.ITERS,
                    "backends": {"Eager PyTorch": {"latency_us": 20.0}},
                    "greedy": {
                        "status": "ok",
                        "lane": "std",
                        "total_us": 10.0,
                        "isolated": {
                            "status": "ok",
                            "total_us": 9.0,
                            "flags": ["wrong-answer"],
                            "kernels": [{"record_knobs": {**candidate["knobs"], "WORK": "w2x2"}}],
                        },
                    },
                }
            )
        )
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(run_volta_golden_o3.subprocess, "run", fake_run)
    result = run_volta_golden_o3._run_one(0, tmp_path, candidate, "candidate", 0)

    assert result["valid"] is False
    assert "isolated flags=['wrong-answer']" in result["integrity_errors"]
    assert "realized WORK='w2x2', expected 'w1x1'" in result["integrity_errors"]


def test_worker_interleaves_three_repeats_per_lane(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run_one(gpu, out_dir, row, lane, repeat):
        calls.append((gpu, row["name"], lane, repeat))
        return {"lane": lane, "repeat": repeat}

    monkeypatch.setattr(run_volta_golden_o3, "_run_one", fake_run_one)
    results = run_volta_golden_o3._worker(2, [{"name": "qwen35_122b.test"}], tmp_path)

    assert calls == [
        (2, "qwen35_122b.test", "bootstrap", 0),
        (2, "qwen35_122b.test", "candidate", 0),
        (2, "qwen35_122b.test", "bootstrap", 1),
        (2, "qwen35_122b.test", "candidate", 1),
        (2, "qwen35_122b.test", "bootstrap", 2),
        (2, "qwen35_122b.test", "candidate", 2),
    ]
    assert results == [
        {"lane": "bootstrap", "repeat": 0},
        {"lane": "candidate", "repeat": 0},
        {"lane": "bootstrap", "repeat": 1},
        {"lane": "candidate", "repeat": 1},
        {"lane": "bootstrap", "repeat": 2},
        {"lane": "candidate", "repeat": 2},
    ]
