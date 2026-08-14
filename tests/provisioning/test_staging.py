"""Unit tests for the staging module (git ls-files + tar)."""

import asyncio
import io
import subprocess
import tarfile
from pathlib import Path

import pytest

from emmy.provisioning.staging import (
    build_stage_manifest,
    build_stage_tar,
    enumerate_staged_files,
    stage_to_remote,
)


def _git(repo: Path, *args):
    subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": "/tmp",
            "PATH": "/usr/bin:/bin",
        },
    )


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "tracked.py").write_text("print('hi')\n")
    (tmp_path / "scripts" / "untracked.py").write_text("print('new')\n")
    (tmp_path / "scripts" / "ignored.log").write_text("noise\n")
    (tmp_path / ".gitignore").write_text("*.log\n")
    (tmp_path / "other.txt").write_text("outside scope\n")
    _git(tmp_path, "add", "scripts/tracked.py", ".gitignore", "other.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_enumerate_empty_stage_paths(repo):
    files = asyncio.run(enumerate_staged_files(repo, []))
    assert files == []


def test_enumerate_includes_tracked_and_untracked_excludes_ignored(repo):
    files = asyncio.run(enumerate_staged_files(repo, ["scripts"]))
    assert "scripts/tracked.py" in files
    assert "scripts/untracked.py" in files
    assert "scripts/ignored.log" not in files
    # Other path-scoped: should not include other.txt.
    assert "other.txt" not in files


def test_enumerate_dot_includes_everything_tracked(repo):
    files = asyncio.run(enumerate_staged_files(repo, ["."]))
    assert "scripts/tracked.py" in files
    assert "other.txt" in files
    assert "scripts/untracked.py" in files
    assert "scripts/ignored.log" not in files


def test_build_stage_tar_roundtrip(repo):
    files = asyncio.run(enumerate_staged_files(repo, ["scripts"]))
    tar_bytes = build_stage_tar(repo, files)
    members = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for m in tar.getmembers():
            f = tar.extractfile(m)
            if f is not None:
                members[m.name] = f.read().decode()
    assert members["scripts/tracked.py"] == "print('hi')\n"
    assert members["scripts/untracked.py"] == "print('new')\n"
    assert "scripts/ignored.log" not in members


def test_stage_manifest_records_exact_files_revision_and_dirty_state(repo):
    manifest = asyncio.run(build_stage_manifest(repo, ["scripts"]))

    assert manifest["schema_version"] == 1
    assert len(manifest["source_id"]) == 64
    assert len(manifest["git_revision"]) == 40
    assert manifest["clean"] is False
    assert manifest["dirty"] == ["?? scripts/untracked.py"]
    assert set(manifest["files"]) == {"scripts/tracked.py", "scripts/untracked.py"}


def test_stage_manifest_source_id_changes_with_content(repo):
    before = asyncio.run(build_stage_manifest(repo, ["scripts"]))
    (repo / "scripts" / "tracked.py").write_text("print('changed')\n")
    after = asyncio.run(build_stage_manifest(repo, ["scripts"]))

    assert before["source_id"] != after["source_id"]
    assert after["clean"] is False


def test_stage_manifest_excludes_tracked_files_deleted_in_worktree(repo):
    (repo / "scripts" / "tracked.py").unlink()

    manifest = asyncio.run(build_stage_manifest(repo, ["scripts"]))

    assert "scripts/tracked.py" not in manifest["files"]
    assert " D scripts/tracked.py" in manifest["dirty"]


def test_stage_to_remote_clean_gate_runs_before_dry_run_transfer(repo):
    with pytest.raises(RuntimeError, match="requires a clean source tree"):
        asyncio.run(stage_to_remote(repo, ["scripts"], "host", "key", 22, "/remote", dry_run=True, require_clean=True))

    (repo / "scripts" / "untracked.py").unlink()
    manifest = asyncio.run(stage_to_remote(repo, ["scripts"], "host", "key", 22, "/remote", dry_run=True, require_clean=True))
    assert manifest is not None
    assert manifest["clean"] is True
