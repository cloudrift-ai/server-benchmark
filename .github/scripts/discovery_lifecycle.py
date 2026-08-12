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

from emmy.recipe.lifecycle import (
    BEST_EFFORT_TAG,
    LIFECYCLE_TAGS,
    MAINTAINED_TAG,
    OBSOLETE_TAG,
    ONBOARDING_TAG,
    UNTESTED_TAG,
    validate_recipe_tags,
)

HF_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MAX_ONBOARDING_MODELS = 3
MANIFEST_FIELDS = frozenset({"maintained_models", "best_effort_models", "obsolete_models", "onboarding_models"})
OBSOLETE_FIELDS = frozenset({"model_id", "replacement_model_id", "rationale"})


def _extract_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1])
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        candidates = []
        for start, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(stripped, start)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and set(candidate) == MANIFEST_FIELDS:
                candidates.append(candidate)
        if len(candidates) != 1:
            raise ValueError("Discovery output must contain exactly one lifecycle JSON object") from None
        value = candidates[0]
    if not isinstance(value, dict) or set(value) != MANIFEST_FIELDS:
        raise ValueError(
            "Discovery lifecycle object must contain exactly maintained_models, best_effort_models, obsolete_models, and onboarding_models"
        )
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


def _obsolete_decisions(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("obsolete_models must be a list")
    decisions = []
    for decision in value:
        if not isinstance(decision, dict) or set(decision) != OBSOLETE_FIELDS:
            raise ValueError(f"Each obsolete model must contain exactly: {', '.join(sorted(OBSOLETE_FIELDS))}")
        model_id = decision["model_id"]
        replacement = decision["replacement_model_id"]
        if not isinstance(model_id, str) or not HF_ID.fullmatch(model_id):
            raise ValueError(f"Invalid obsolete Hugging Face model ID: {model_id!r}")
        if not isinstance(replacement, str) or not HF_ID.fullmatch(replacement):
            raise ValueError(f"Invalid replacement Hugging Face model ID for {model_id}: {replacement!r}")
        rationale = decision["rationale"]
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 600:
            raise ValueError(f"Obsolete model {model_id} needs a rationale of at most 600 characters")
        decisions.append({**decision, "rationale": rationale.strip()})
    return decisions


def validate_manifest(path: Path, workspace: Path, gpu: str, gpu_count: int, maintained_count: int) -> dict:
    """Validate one discovery manifest and return its normalized lifecycle decisions."""
    manifest = _extract_object(path.read_text())
    records = _inventory(workspace)
    maintained = _unique_model_ids(manifest.get("maintained_models"), "maintained_models")
    best_effort = _unique_model_ids(manifest.get("best_effort_models"), "best_effort_models")
    obsolete_decisions = _obsolete_decisions(manifest.get("obsolete_models"))
    obsolete = [decision["model_id"] for decision in obsolete_decisions]
    if len(maintained) != maintained_count:
        raise ValueError(f"maintained_models must contain exactly {maintained_count} models")

    existing_onboarding = sorted(model_id for model_id, record in records.items() if ONBOARDING_TAG in record["tags"])
    if len(existing_onboarding) > MAX_ONBOARDING_MODELS:
        raise ValueError(f"Existing onboarding shells exceed the limit of {MAX_ONBOARDING_MODELS}")

    classified = [*maintained, *best_effort, *obsolete]
    if len(classified) != len(set(classified)):
        raise ValueError("A complete recipe must appear in exactly one lifecycle list")
    unfinished = set(classified) & set(existing_onboarding)
    if unfinished:
        raise ValueError(f"Untested onboarding shells cannot be classified: {', '.join(sorted(unfinished))}")
    complete_models = records.keys() - set(existing_onboarding)
    unknown = set(classified) - complete_models
    if unknown:
        raise ValueError(f"Lifecycle models must have complete existing recipes: {', '.join(sorted(unknown))}")
    unclassified = complete_models - set(classified)
    if unclassified:
        raise ValueError(f"Every complete recipe must be classified: {', '.join(sorted(unclassified))}")

    replacement_models = set(maintained) | set(best_effort)
    for decision in obsolete_decisions:
        model_id = decision["model_id"]
        replacement = decision["replacement_model_id"]
        if model_id == replacement:
            raise ValueError(f"Obsolete model {model_id} cannot replace itself")
        if replacement not in replacement_models:
            raise ValueError(f"Replacement for obsolete model {model_id} must be maintained or best-effort")
        model_task = (records[model_id]["config"].get("model") or {}).get("task", "generate")
        replacement_task = (records[replacement]["config"].get("model") or {}).get("task", "generate")
        if model_task != replacement_task:
            raise ValueError(f"Replacement for obsolete model {model_id} must serve the same task")

    candidates = manifest.get("onboarding_models", [])
    available_onboarding = MAX_ONBOARDING_MODELS - len(existing_onboarding)
    if not isinstance(candidates, list) or len(candidates) > available_onboarding:
        raise ValueError(f"onboarding_models must leave at most {MAX_ONBOARDING_MODELS} total onboarding shells")
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

    return {
        "maintained_models": maintained,
        "best_effort_models": best_effort,
        "obsolete_models": obsolete_decisions,
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


def _model_lines(model_ids: list[str]) -> list[str]:
    return [f"- `{model_id}`" for model_id in model_ids] or ["- None."]


def _obsolete_lines(decisions: list[dict[str, str]]) -> list[str]:
    if not decisions:
        return ["- None."]
    lines = []
    for decision in decisions:
        lines.append(f"- `{decision['model_id']}` → `{decision['replacement_model_id']}` — {decision['rationale']}")
    return lines


def _summary(manifest: dict) -> str:
    lines = [
        "## Automated model lifecycle update",
        "",
        f"This rolling PR keeps {len(manifest['maintained_models'])} recipes in the maintained set.",
        "Best-effort recipes remain runnable but are not selected for periodic testing and optimization. Obsolete recipes",
        "remain in git but are disabled and excluded from package builds.",
        "",
        "### Maintained",
        "",
        *_model_lines(manifest["maintained_models"]),
        "",
        "### Best effort",
        "",
        *_model_lines(manifest["best_effort_models"]),
        "",
        "### Obsolete",
        "",
        *_obsolete_lines(manifest["obsolete_models"]),
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
        changed = _set_lifecycle(records[model_id], MAINTAINED_TAG) or changed
    for model_id in manifest["best_effort_models"]:
        changed = _set_lifecycle(records[model_id], BEST_EFFORT_TAG) or changed
    for decision in manifest["obsolete_models"]:
        changed = _set_lifecycle(records[decision["model_id"]], OBSOLETE_TAG) or changed
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
        "best_effort_count": len(manifest["best_effort_models"]),
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
