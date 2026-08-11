"""Tests for scripts/capture_kernel_environment.py."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "capture_kernel_environment.py"
_SPEC = importlib.util.spec_from_file_location("capture_kernel_environment", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
capture_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = capture_script
_SPEC.loader.exec_module(capture_script)


def test_capture_archives_pinned_config_and_machine_state(monkeypatch, tmp_path):
    source = tmp_path / "source-config.json"
    source.write_text('{"hidden_size":4096}\n', encoding="utf-8")
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(json.dumps({"source_id": "b" * 64}), encoding="utf-8")
    commands = []

    def fake_run(command, *, check, capture_output, text):
        assert check is False and capture_output is True and text is True
        commands.append(command)
        if command[-2:] == ["freeze", "--all"]:
            output = "torch==2.10.0\ntransformers==5.1.0"
        elif command[0] == "nvidia-smi":
            output = "0, NVIDIA H200, GPU-1, 580.1, P0, 30, 1980, 3200, 400, 700"
        else:
            output = "Cuda compilation tools, release 13.0"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    downloads = []

    def fake_download(**kwargs):
        downloads.append(kwargs)
        return str(source)

    monkeypatch.setattr(capture_script, "_package_versions", lambda: {"torch": "2.10.0"})
    record = capture_script.capture(
        "Qwen/Qwen3-0.6B@" + "a" * 40,
        tmp_path / "evidence",
        source_manifest,
        run=fake_run,
        hf_download=fake_download,
    )

    assert downloads == [{"repo_id": "Qwen/Qwen3-0.6B", "filename": "config.json", "revision": "a" * 40}]
    assert record["model"]["revision"] == "a" * 40
    assert record["source"]["source_id"] == "b" * 64
    assert record["gpu_state"][0].startswith("0, NVIDIA H200")
    assert (tmp_path / "evidence" / "model-config.json").read_text() == source.read_text()
    assert json.loads((tmp_path / "evidence" / "environment.json").read_text()) == record
    assert any(command[0] == "nvidia-smi" for command in commands)


def test_capture_rejects_unpinned_model(tmp_path):
    with pytest.raises(capture_script.CaptureError, match="FULL_40_HEX_REVISION"):
        capture_script.capture("Qwen/Qwen3-0.6B", tmp_path, tmp_path / "source.json")
