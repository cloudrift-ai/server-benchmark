"""Tests for scripts/verify_staged_source.py."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "verify_staged_source.py"
_SPEC = importlib.util.spec_from_file_location("verify_staged_source", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
source = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = source
_SPEC.loader.exec_module(source)


def test_load_and_verify_rejects_dirty_or_extra_staged_source(tmp_path):
    (tmp_path / "emmy").mkdir()
    tracked = tmp_path / "emmy" / "run.py"
    tracked.write_text("clean\n", encoding="utf-8")
    files = {"emmy/run.py": source._digest(tracked)}
    manifest = {
        "schema_version": 1,
        "stage_paths": ["emmy"],
        "manifest_path": "source.json",
        "source_id": source._source_id(files),
        "files": files,
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert source.load_and_verify(path, tmp_path)["source_id"] == manifest["source_id"]
    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(source.SourceManifestError, match="digest mismatch"):
        source.load_and_verify(path, tmp_path)

    tracked.write_text("clean\n", encoding="utf-8")
    (tmp_path / "emmy" / "untracked.py").write_text("extra\n", encoding="utf-8")
    with pytest.raises(source.SourceManifestError, match="file set differs"):
        source.load_and_verify(path, tmp_path)


def test_cli_verifies_copied_clean_tree_without_python_cache_contamination(tmp_path):
    (tmp_path / "emmy").mkdir()
    tracked = tmp_path / "emmy" / "run.py"
    tracked.write_text("clean\n", encoding="utf-8")
    files = {"emmy/run.py": source._digest(tracked)}
    manifest = {
        "schema_version": 1,
        "stage_paths": ["emmy"],
        "manifest_path": "source.json",
        "source_id": source._source_id(files),
        "files": files,
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    cache = tmp_path / "emmy" / "__pycache__"
    cache.mkdir()
    (cache / "run.cpython-312.pyc").write_bytes(b"interpreter cache")
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(path), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
