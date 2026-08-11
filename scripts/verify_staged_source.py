#!/usr/bin/env python3
"""Build or verify a content-addressed manifest for staged experiment source."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path

from emmy.logging_setup import setup_cli_logging

logger = logging.getLogger(__name__)


class SourceManifestError(ValueError):
    """Raised when staged source differs from its preregistered manifest."""


def _source_id(files: dict[str, str]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _listed_files(root: Path, stage_paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *stage_paths],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SourceManifestError(f"git ls-files failed: {result.stderr.strip()}")
    return sorted({line for line in result.stdout.splitlines() if line})


def _walk_files(root: Path, stage_paths: list[str]) -> list[str]:
    files = []
    for relative in stage_paths:
        path = root / relative
        if path.is_file():
            files.append(relative)
        elif path.is_dir():
            files.extend(
                str(item.relative_to(root))
                for item in path.rglob("*")
                if item.is_file() and "__pycache__" not in item.parts and item.suffix not in {".pyc", ".pyo"}
            )
        else:
            raise SourceManifestError(f"staged source path is missing: {relative}")
    return sorted(set(files))


def build(root: Path, stage_paths: list[str], manifest_relative: str) -> dict:
    """Build a manifest using the same tracked/unignored file rule as remote staging."""
    files = {relative: _digest(root / relative) for relative in _listed_files(root, stage_paths) if relative != manifest_relative}
    if not files:
        raise SourceManifestError("source manifest would be empty")
    return {
        "schema_version": 1,
        "stage_paths": stage_paths,
        "manifest_path": manifest_relative,
        "source_id": _source_id(files),
        "files": files,
    }


def load_and_verify(path: Path, root: Path) -> dict:
    """Verify the manifest ID, exact file set, and every staged source digest."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceManifestError(f"cannot load source manifest {path}: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise SourceManifestError("source manifest schema_version must be 1")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SourceManifestError("source manifest files must be a non-empty object")
    if manifest.get("source_id") != _source_id(files):
        raise SourceManifestError("source manifest source_id does not match its file map")
    stage_paths = manifest.get("stage_paths")
    if not isinstance(stage_paths, list) or not all(isinstance(item, str) and item for item in stage_paths):
        raise SourceManifestError("source manifest stage_paths must be non-empty strings")
    manifest_relative = manifest.get("manifest_path")
    actual_files = set(_walk_files(root, stage_paths)) - {manifest_relative}
    if actual_files != set(files):
        missing = sorted(set(files) - actual_files)
        extra = sorted(actual_files - set(files))
        raise SourceManifestError(f"staged source file set differs: missing={missing}, extra={extra}")
    for relative, expected in files.items():
        actual = _digest(root / relative)
        if actual != expected:
            raise SourceManifestError(f"staged source digest mismatch: {relative}")
    return manifest


def main() -> int:
    """Build or verify a staged-source manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--stage-path", action="append", default=[])
    args = parser.parse_args()
    setup_cli_logging()
    try:
        if args.write:
            if not args.stage_path:
                raise SourceManifestError("--write requires at least one --stage-path")
            relative = str(args.manifest.resolve().relative_to(args.root.resolve()))
            manifest = build(args.root, args.stage_path, relative)
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            logger.info("Source manifest %s → %s", manifest["source_id"], args.manifest)
        else:
            manifest = load_and_verify(args.manifest, args.root)
            logger.info("Verified staged source %s", manifest["source_id"])
    except (SourceManifestError, OSError, ValueError) as exc:
        logger.error("Source manifest failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
