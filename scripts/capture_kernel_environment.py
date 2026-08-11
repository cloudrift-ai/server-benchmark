#!/usr/bin/env python3
"""Capture immutable model, software, and GPU provenance for a kernel experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

from emmy.logging_setup import setup_cli_logging

logger = logging.getLogger(__name__)

_MODEL_REF_RE = re.compile(r"(?P<model>[^@]+)@(?P<revision>[0-9a-f]{40})")


class CaptureError(ValueError):
    """Raised when immutable experiment provenance cannot be captured."""


def _command(command: list[str], *, run) -> str:
    result = run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() if isinstance(result.stderr, str) else ""
        raise CaptureError(f"command failed ({' '.join(command)}): {stderr}")
    return result.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("emmy-ml", "hidet", "huggingface-hub", "numpy", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def capture(model_ref: str, output_dir: Path, source_manifest: Path, *, run=subprocess.run, hf_download=None) -> dict:
    """Capture one experiment environment and return its machine-readable record."""
    match = _MODEL_REF_RE.fullmatch(model_ref)
    if match is None:
        raise CaptureError("model must be MODEL_ID@FULL_40_HEX_REVISION")
    model_id = match.group("model")
    revision = match.group("revision")
    if hf_download is None:
        from huggingface_hub import hf_hub_download

        hf_download = hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        source = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot load verified source manifest {source_manifest}: {exc}") from exc
    source_id = source.get("source_id")
    if not isinstance(source_id, str) or not re.fullmatch(r"[0-9a-f]{64}", source_id):
        raise CaptureError("verified source manifest has no valid source_id")
    archived_source_manifest = output_dir / "source-manifest.json"
    shutil.copyfile(source_manifest, archived_source_manifest)
    config_source = Path(hf_download(repo_id=model_id, filename="config.json", revision=revision))
    config_path = output_dir / "model-config.json"
    shutil.copyfile(config_source, config_path)
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()

    freeze = _command([sys.executable, "-m", "pip", "freeze", "--all"], run=run)
    freeze_path = output_dir / "requirements.freeze.txt"
    freeze_path.write_text(freeze + "\n", encoding="utf-8")
    freeze_digest = hashlib.sha256(freeze_path.read_bytes()).hexdigest()

    record = {
        "schema_version": 1,
        "model": {
            "id": model_id,
            "revision": revision,
            "config_path": config_path.name,
            "config_sha256": config_digest,
        },
        "source": {
            "source_id": source_id,
            "manifest_path": archived_source_manifest.name,
            "manifest_sha256": hashlib.sha256(archived_source_manifest.read_bytes()).hexdigest(),
        },
        "packages": _package_versions(),
        "requirements": {"path": freeze_path.name, "sha256": freeze_digest},
        "gpu_state": _command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            run=run,
        ).splitlines(),
        "cuda_compiler": _command(["nvcc", "--version"], run=run),
        "inductor_mode_equivalent": {
            "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
            "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": "1",
            "TORCHINDUCTOR_CUDAGRAPHS": "1",
        },
    }
    (output_dir / "environment.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    """Run the command-line capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model as MODEL_ID@FULL_40_HEX_REVISION")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    setup_cli_logging()
    try:
        capture(args.model, args.output_dir, args.source_manifest)
    except CaptureError as exc:
        logger.error("Environment capture failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
