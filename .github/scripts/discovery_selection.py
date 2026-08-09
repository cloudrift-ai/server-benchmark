#!/usr/bin/env python3
"""Validate one model selected by the discover-models agent run."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

import yaml

HF_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _extract_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1])
    return json.loads(stripped)


def _existing_models(workspace: Path) -> set[str]:
    models = set()
    for recipe in workspace.glob("recipes/*/recipe.yaml"):
        try:
            value = yaml.safe_load(recipe.read_text()) or {}
            model_id = value.get("model", {}).get("huggingface")
        except (OSError, yaml.YAMLError, AttributeError):
            continue
        if model_id:
            models.add(model_id)
    return models


def validate_selection(path: Path, workspace: Path, gpu: str, gpu_count: int) -> dict:
    selection = _extract_object(path.read_text())
    model_id = selection.get("model_id")
    if model_id is None:
        return {"found": False, "model_id": "", "gpu": gpu, "gpu_count": gpu_count, "rationale": ""}
    if not isinstance(model_id, str) or not HF_ID.fullmatch(model_id):
        raise ValueError(f"Invalid Hugging Face model ID: {model_id!r}")
    if model_id in _existing_models(workspace):
        raise ValueError(f"A recipe already exists for {model_id}")
    if selection.get("gpu") != gpu or selection.get("gpu_count") != gpu_count:
        raise ValueError("Discovery selected hardware outside the exact requested target")
    rationale = selection.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Discovery selection is missing its community/serving-value rationale")
    return {
        "found": True,
        "model_id": model_id,
        "gpu": gpu,
        "gpu_count": gpu_count,
        "rationale": rationale.strip(),
    }


def _write_outputs(selection: dict) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a") as output:
        for key in ("found", "model_id", "gpu", "gpu_count"):
            output.write(f"{key}={str(selection[key]).lower() if key == 'found' else selection[key]}\n")
        delimiter = f"discovery_{uuid.uuid4().hex}"
        output.write(f"rationale<<{delimiter}\n{selection['rationale']}\n{delimiter}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        selection = validate_selection(args.input, args.workspace.resolve(), args.gpu, args.gpu_count)
        _write_outputs(selection)
        print(json.dumps(selection, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
