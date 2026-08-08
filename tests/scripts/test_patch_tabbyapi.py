"""Tests for scripts/patch_tabbyapi.py — the one server-side fix the bench client needs.

tabbyAPI's exllamav3 backend compares the client's unconditional ``"logprobs": null`` against
an int, so every request dies. The patch has to be idempotent (the lane re-runs it on an
existing clone) and has to report a miss rather than pretend success, because a silent miss
puts the crash straight back.
"""

import sys
from pathlib import Path

import pytest


@pytest.fixture
def patcher():
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        import patch_tabbyapi
    finally:
        sys.path.remove(scripts_dir)
    return patch_tabbyapi


@pytest.fixture
def clone(tmp_path):
    backend = tmp_path / "backends" / "exllamav3"
    backend.mkdir(parents=True)
    (backend / "model.py").write_text("async def generate(params):\n    if params.logprobs > 0:\n        pass\n")
    return tmp_path


def test_none_logprobs_gets_a_guard(patcher, clone):
    assert patcher.apply_patches(clone) == 0

    text = (clone / "backends/exllamav3/model.py").read_text()
    assert "if params.logprobs and params.logprobs > 0:" in text


def test_patch_is_idempotent(patcher, clone):
    patcher.apply_patches(clone)
    before = (clone / "backends/exllamav3/model.py").read_text()
    assert patcher.apply_patches(clone) == 1  # nothing left to match — reported, not silent

    assert (clone / "backends/exllamav3/model.py").read_text() == before


def test_missing_pattern_is_reported(patcher, tmp_path, caplog):
    """Upstream may have fixed it. That is a warning plus a hard downstream guard, not an abort."""
    (tmp_path / "backends" / "exllamav3").mkdir(parents=True)
    (tmp_path / "backends/exllamav3/model.py").write_text("pass\n")

    assert patcher.apply_patches(tmp_path) == 1
    assert "no site matched" in caplog.text


def test_rejects_a_directory_that_is_not_a_clone(patcher, tmp_path):
    assert patcher.main(["--path", str(tmp_path)]) == 2
