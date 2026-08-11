#!/usr/bin/env python3
"""Validate and apply the model lifecycle manifest from discovery."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

from emmy.recipe.lifecycle import LIFECYCLE_TAGS, ONBOARDING_TAG, UNTESTED_TAG, validate_recipe_tags

HF_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MAX_NEW_MODELS = 3


def _extract_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1])
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("Discovery output must be one JSON object")
    return value


def _inventory(workspace: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted(workspace.glob("recipes/*/recipe.yaml")):
        config = yaml.safe_load(path.read_text()) or {}
        if not isinstance(config, dict):
            raise ValueError(f"Recipe must contain a YAML object: {path.relative_to(workspace)}")
        tags = validate_recipe_tags(config.get("tags"))
        model_id = (config.get("model") or {}).get("huggingface")
        if not model_id:
            continue
        if not isinstance(model_id, str) or not HF_ID.fullmatch(model_id):
            raise ValueError(f"Invalid Hugging Face model ID in {path.relative_to(workspace)}: {model_id!r}")
        if model_id in records:
            raise ValueError(f"Multiple recipes use Hugging Face model ID {model_id}")
        records[model_id] = {"path": path, "config": config, "tags": tags}
    return records


def _unique_model_ids(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    if not all(isinstance(model_id, str) and HF_ID.fullmatch(model_id) for model_id in value):
        raise ValueError(f"{field} must contain exact Hugging Face model IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicates")
    return value


def validate_manifest(path: Path, workspace: Path, gpu: str, gpu_count: int, maintained_count: int) -> dict:
    """Validate one discovery manifest and return its normalized lifecycle decisions."""
    manifest = _extract_object(path.read_text())
    records = _inventory(workspace)
    maintained = _unique_model_ids(manifest.get("maintained_models"), "maintained_models")
    if len(maintained) != maintained_count:
        raise ValueError(f"maintained_models must contain exactly {maintained_count} models")

    missing = set(maintained) - records.keys()
    if missing:
        raise ValueError(f"Maintained models must already have recipes: {', '.join(sorted(missing))}")
    unfinished = [model_id for model_id in maintained if ONBOARDING_TAG in records[model_id]["tags"]]
    if unfinished:
        raise ValueError(f"Untested onboarding shells cannot be maintained: {', '.join(unfinished)}")

    candidates = manifest.get("onboarding_models", [])
    if not isinstance(candidates, list) or len(candidates) > MAX_NEW_MODELS:
        raise ValueError(f"onboarding_models must be a list of at most {MAX_NEW_MODELS} models")
    normalized_candidates = []
    candidate_ids: set[str] = set()
    expected_keys = {"model_id", "task", "gpu", "gpu_count", "rationale"}
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != expected_keys:
            raise ValueError(f"Each onboarding model must contain exactly: {', '.join(sorted(expected_keys))}")
        model_id = candidate["model_id"]
        if not isinstance(model_id, str) or not HF_ID.fullmatch(model_id):
            raise ValueError(f"Invalid onboarding Hugging Face model ID: {model_id!r}")
        if model_id in records:
            raise ValueError(f"A recipe already exists for onboarding model {model_id}")
        if model_id in candidate_ids:
            raise ValueError(f"Duplicate onboarding model {model_id}")
        if candidate["task"] not in ("generate", "embed"):
            raise ValueError(f"Invalid task for {model_id}: {candidate['task']!r}")
        if candidate["gpu"] != gpu or candidate["gpu_count"] != gpu_count:
            raise ValueError(f"Onboarding model {model_id} selected hardware outside the exact requested target")
        rationale = candidate["rationale"]
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 600:
            raise ValueError(f"Onboarding model {model_id} needs a rationale of at most 600 characters")
        candidate_ids.add(model_id)
        normalized_candidates.append({**candidate, "rationale": rationale.strip()})

    existing_onboarding = sorted(model_id for model_id, record in records.items() if ONBOARDING_TAG in record["tags"])
    obsolete = sorted(
        model_id for model_id, record in records.items() if model_id not in maintained and ONBOARDING_TAG not in record["tags"]
    )
    return {
        "maintained_models": maintained,
        "obsolete_models": obsolete,
        "existing_onboarding_models": existing_onboarding,
        "onboarding_models": normalized_candidates,
    }


def _tags_with_lifecycle(tags: tuple[str, ...], lifecycle: str) -> list[str]:
    remaining = [tag for tag in tags if tag not in LIFECYCLE_TAGS and tag != UNTESTED_TAG]
    desired = [lifecycle, *remaining]
    if lifecycle == ONBOARDING_TAG:
        desired.append(UNTESTED_TAG)
    return desired


def _replace_tag_block(text: str, tags: list[str]) -> str:
    lines = text.splitlines(keepends=True)
    block = ["tags:\n", *(f"  - {tag}\n" for tag in tags)]
    start = next((index for index, line in enumerate(lines) if line.startswith("tags:")), None)
    if start is not None:
        end = start + 1
        while end < len(lines) and lines[end].startswith((" ", "\t")):
            end += 1
        return "".join([*lines[:start], *block, *lines[end:]])

    insertion = next(
        (index for index, line in enumerate(lines) if line.strip() and not line.lstrip().startswith("#")),
        len(lines),
    )
    return "".join([*lines[:insertion], *block, "\n", *lines[insertion:]])


def _set_lifecycle(record: dict, lifecycle: str) -> bool:
    path = record["path"]
    before = path.read_text()
    after = _replace_tag_block(before, _tags_with_lifecycle(record["tags"], lifecycle))
    if before == after:
        return False
    path.write_text(after)
    return True


def _shell_directory(workspace: Path, model_id: str) -> Path:
    organization, name = model_id.split("/", 1)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    path = workspace / "recipes" / slug
    if path.exists():
        organization_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", organization).strip("-")
        path = workspace / "recipes" / f"{organization_slug}--{slug}"
    if path.exists():
        raise ValueError(f"Cannot choose a unique recipe directory for {model_id}")
    return path


def _create_shell(workspace: Path, candidate: dict) -> Path:
    path = _shell_directory(workspace, candidate["model_id"])
    path.mkdir(parents=True)
    config = {
        "tags": [ONBOARDING_TAG, UNTESTED_TAG],
        "model": {"huggingface": candidate["model_id"], "task": candidate["task"]},
        "discovery": {
            "target_gpu": candidate["gpu"],
            "target_gpu_count": candidate["gpu_count"],
            "rationale": candidate["rationale"],
        },
    }
    recipe = path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(config, sort_keys=False, width=116))
    return recipe


def _summary(manifest: dict) -> str:
    lines = [
        "## Automated model lifecycle update",
        "",
        f"This rolling PR keeps {len(manifest['maintained_models'])} recipes in the maintained set. ",
        "Obsolete recipes remain in git but are disabled and excluded from package builds.",
        "",
        "### Maintained",
        "",
        *(f"- `{model_id}`" for model_id in manifest["maintained_models"]),
        "",
        "### Obsolete",
        "",
        *(f"- `{model_id}`" for model_id in manifest["obsolete_models"]),
        "",
        "### New onboarding shells",
        "",
    ]
    if manifest["onboarding_models"]:
        for candidate in manifest["onboarding_models"]:
            lines.append(f"- `{candidate['model_id']}` on `{candidate['gpu']} x{candidate['gpu_count']}` — {candidate['rationale']}")
    else:
        lines.append("- None in this run.")
    if manifest["existing_onboarding_models"]:
        lines.extend(["", "### Existing onboarding shells", ""])
        lines.extend(f"- `{model_id}`" for model_id in manifest["existing_onboarding_models"])
    has_run_url = all(os.environ.get(name) for name in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    if has_run_url:
        run_url = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        lines.extend(
            [
                "",
                f"Workflow run: {run_url}",
            ]
        )
    return "\n".join(lines) + "\n"


def apply_manifest(manifest: dict, workspace: Path, summary_path: Path) -> dict:
    """Apply a validated manifest and return change/count outputs."""
    records = _inventory(workspace)
    changed = False
    for model_id in manifest["maintained_models"]:
        changed = _set_lifecycle(records[model_id], "maintained") or changed
    for model_id in manifest["obsolete_models"]:
        changed = _set_lifecycle(records[model_id], "obsolete") or changed
    for candidate in manifest["onboarding_models"]:
        _create_shell(workspace, candidate)
        changed = True
    for plan in workspace.glob("plans/onboard-*.md"):
        plan.unlink()
        changed = True

    summary_path.write_text(_summary(manifest))
    return {
        "changed": changed,
        "maintained_count": len(manifest["maintained_models"]),
        "obsolete_count": len(manifest["obsolete_models"]),
        "onboarding_count": len(manifest["existing_onboarding_models"]) + len(manifest["onboarding_models"]),
    }


def _write_outputs(result: dict) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a") as output:
        for key, value in result.items():
            rendered = str(value).lower() if isinstance(value, bool) else value
            output.write(f"{key}={rendered}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--maintained-count", type=int, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        workspace = args.workspace.resolve()
        manifest = validate_manifest(args.input, workspace, args.gpu, args.gpu_count, args.maintained_count)
        result = apply_manifest(manifest, workspace, args.summary)
        _write_outputs(result)
        print(json.dumps({**manifest, **result}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
