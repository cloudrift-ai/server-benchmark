#!/usr/bin/env python3
"""Validate and apply the model lifecycle manifest from discovery."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from emmy import gpu as gpu_registry
from emmy.recipe.catalog import HF_ID, create_recipe_stub, deployment_setups, recipe_catalog, validate_stub_deployments
from emmy.recipe.lifecycle import (
    BEST_EFFORT_TAG,
    LIFECYCLE_TAGS,
    MAINTAINED_TAG,
    OBSOLETE_TAG,
    ONBOARDING_TAG,
    UNTESTED_TAG,
)

MAX_ONBOARDING_MODELS = 3
MANIFEST_FIELDS = frozenset({"maintained_models", "best_effort_models", "obsolete_models", "onboarding_models"})
DECISION_FIELDS = frozenset({"model_id", "rationale"})
OBSOLETE_FIELDS = frozenset({"model_id", "replacement_model_id", "rationale"})
ONBOARDING_FIELDS = frozenset({"model_id", "task", "rationale", "deployments"})


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


def _rationale(value: object, model_id: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 600:
        raise ValueError(f"Model {model_id} needs a rationale of at most 600 characters")
    return value.strip()


def _resolve_existing_model_id(value: object, records: dict[str, dict]) -> object:
    if not isinstance(value, str):
        return value
    if value in records:
        return value
    checkpoint = value.rsplit("/", 1)[-1]
    matches = [model_id for model_id in records if model_id.rsplit("/", 1)[-1] == checkpoint]
    return matches[0] if len(matches) == 1 else value


def _model_decisions(
    value: object,
    field: str,
    records: dict[str, dict],
    *,
    ignore_invalid_ids: bool = False,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    decisions = []
    for decision in value:
        if not isinstance(decision, dict) or set(decision) != DECISION_FIELDS:
            raise ValueError(f"Each {field} entry must contain exactly: {', '.join(sorted(DECISION_FIELDS))}")
        model_id = _resolve_existing_model_id(decision["model_id"], records)
        if not isinstance(model_id, str) or not HF_ID.fullmatch(model_id):
            if ignore_invalid_ids:
                continue
            raise ValueError(f"Invalid Hugging Face model ID in {field}: {model_id!r}")
        decisions.append({"model_id": model_id, "rationale": _rationale(decision["rationale"], model_id)})
    if len(decisions) != len({decision["model_id"] for decision in decisions}):
        raise ValueError(f"{field} must not contain duplicates")
    return decisions


def _obsolete_decisions(value: object, records: dict[str, dict], *, ignore_invalid_ids: bool = False) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("obsolete_models must be a list")
    decisions = []
    for decision in value:
        fields = set(decision) if isinstance(decision, dict) else set()
        if fields not in (DECISION_FIELDS, OBSOLETE_FIELDS):
            raise ValueError("Each obsolete model must contain model_id, rationale, and an optional replacement_model_id")
        model_id = _resolve_existing_model_id(decision["model_id"], records)
        if not isinstance(model_id, str) or not HF_ID.fullmatch(model_id):
            if ignore_invalid_ids:
                continue
            raise ValueError(f"Invalid obsolete Hugging Face model ID: {model_id!r}")
        normalized = {"model_id": model_id, "rationale": _rationale(decision["rationale"], model_id)}
        replacement = _resolve_existing_model_id(decision.get("replacement_model_id"), records)
        if replacement is not None:
            if not isinstance(replacement, str) or not HF_ID.fullmatch(replacement):
                if ignore_invalid_ids:
                    continue
                raise ValueError(f"Invalid replacement Hugging Face model ID for {model_id}: {replacement!r}")
            normalized["replacement_model_id"] = replacement
            if replacement not in normalized["rationale"]:
                normalized["rationale"] = _rationale(
                    f"{replacement} supersedes this recipe: {normalized['rationale']}",
                    model_id,
                )
        decisions.append(normalized)
    return decisions


def _existing_rationale(record: dict) -> str:
    config = record["config"]
    return (config.get("model") or {}).get(
        "rationale"
    ) or "Retained as a useful runnable recipe on a best-effort basis because discovery did not establish that it is obsolete."


def _model_ids(decisions: list[dict[str, str]]) -> list[str]:
    return [decision["model_id"] for decision in decisions]


def _deployment_footprints(config: dict) -> tuple[int, ...]:
    """Return total physical VRAM in MiB for every known deployment variant."""

    footprints = []
    for setup in deployment_setups(config):
        spec = gpu_registry.by_name(setup["deploy.gpu"])
        if spec is None or spec.vram_mib is None:
            continue
        footprints.append(spec.vram_mib * setup["deploy.gpu_count"])
    return tuple(footprints)


def validate_manifest(path: Path, workspace: Path) -> dict:
    """Validate one discovery manifest and return its normalized lifecycle decisions."""
    manifest = _extract_object(path.read_text())
    records = recipe_catalog(workspace / "recipes")
    maintained = _model_decisions(manifest.get("maintained_models"), "maintained_models", records)
    best_effort = _model_decisions(
        manifest.get("best_effort_models"),
        "best_effort_models",
        records,
        ignore_invalid_ids=True,
    )
    obsolete_decisions = _obsolete_decisions(manifest.get("obsolete_models"), records, ignore_invalid_ids=True)
    existing_onboarding = sorted(model_id for model_id, record in records.items() if ONBOARDING_TAG in record["tags"])
    if len(existing_onboarding) > MAX_ONBOARDING_MODELS:
        raise ValueError(f"Existing onboarding shells exceed the limit of {MAX_ONBOARDING_MODELS}")
    existing_onboarding_models = [
        {
            "model_id": model_id,
            "rationale": _existing_rationale(records[model_id]),
            "deployments": validate_stub_deployments(records[model_id]["config"].get("matrices"), model_id),
        }
        for model_id in existing_onboarding
    ]

    proposed = [*_model_ids(maintained), *_model_ids(best_effort), *_model_ids(obsolete_decisions)]
    unfinished = set(proposed) & set(existing_onboarding)
    if unfinished:
        raise ValueError(f"Untested onboarding shells cannot be classified: {', '.join(sorted(unfinished))}")
    complete_models = set(records) - set(existing_onboarding)
    unknown_maintained = set(_model_ids(maintained)) - complete_models
    if unknown_maintained:
        raise ValueError(f"Maintained models must have complete existing recipes: {', '.join(sorted(unknown_maintained))}")
    best_effort = [decision for decision in best_effort if decision["model_id"] in complete_models]
    obsolete_decisions = [decision for decision in obsolete_decisions if decision["model_id"] in complete_models]
    classified = [*_model_ids(maintained), *_model_ids(best_effort), *_model_ids(obsolete_decisions)]
    if len(classified) != len(set(classified)):
        raise ValueError("A complete recipe must appear in exactly one lifecycle list")
    unclassified = complete_models - set(classified)
    best_effort.extend({"model_id": model_id, "rationale": _existing_rationale(records[model_id])} for model_id in sorted(unclassified))

    replacement_models = set(_model_ids(maintained)) | set(_model_ids(best_effort))
    accepted_obsolete = []
    for decision in obsolete_decisions:
        model_id = decision["model_id"]
        replacement = decision.get("replacement_model_id")
        if replacement is None:
            accepted_obsolete.append(decision)
            continue
        if model_id == replacement:
            best_effort.append({"model_id": model_id, "rationale": _existing_rationale(records[model_id])})
            continue
        if replacement not in replacement_models:
            best_effort.append({"model_id": model_id, "rationale": _existing_rationale(records[model_id])})
            continue
        model_task = (records[model_id]["config"].get("model") or {}).get("task", "generate")
        replacement_task = (records[replacement]["config"].get("model") or {}).get("task", "generate")
        if model_task != replacement_task:
            best_effort.append({"model_id": model_id, "rationale": _existing_rationale(records[model_id])})
            continue
        model_footprints = _deployment_footprints(records[model_id]["config"])
        replacement_footprints = _deployment_footprints(records[replacement]["config"])
        if not model_footprints or not replacement_footprints:
            best_effort.append({"model_id": model_id, "rationale": _existing_rationale(records[model_id])})
            continue
        model_min = min(model_footprints)
        replacement_min = min(replacement_footprints)
        if replacement_min > model_min:
            best_effort.append({"model_id": model_id, "rationale": _existing_rationale(records[model_id])})
            continue
        accepted_obsolete.append(decision)
    obsolete_decisions = accepted_obsolete

    candidates = manifest.get("onboarding_models", [])
    available_onboarding = MAX_ONBOARDING_MODELS - len(existing_onboarding)
    if not isinstance(candidates, list) or len(candidates) > MAX_ONBOARDING_MODELS:
        raise ValueError(f"onboarding_models must contain at most {MAX_ONBOARDING_MODELS} candidates")
    candidates = candidates[:available_onboarding]
    normalized_candidates = []
    candidate_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != ONBOARDING_FIELDS:
            raise ValueError(f"Each onboarding model must contain exactly: {', '.join(sorted(ONBOARDING_FIELDS))}")
        model_id = candidate["model_id"]
        if not isinstance(model_id, str) or not HF_ID.fullmatch(model_id):
            raise ValueError(f"Invalid onboarding Hugging Face model ID: {model_id!r}")
        if model_id in records:
            raise ValueError(f"A recipe already exists for onboarding model {model_id}")
        if model_id in candidate_ids:
            raise ValueError(f"Duplicate onboarding model {model_id}")
        if candidate["task"] not in ("generate", "embed"):
            raise ValueError(f"Invalid task for {model_id}: {candidate['task']!r}")
        normalized_deployments = validate_stub_deployments(candidate["deployments"], model_id)
        candidate_ids.add(model_id)
        normalized_candidates.append(
            {
                "model_id": model_id,
                "task": candidate["task"],
                "rationale": _rationale(candidate["rationale"], model_id),
                "deployments": normalized_deployments,
            }
        )

    return {
        "maintained_models": maintained,
        "best_effort_models": best_effort,
        "obsolete_models": obsolete_decisions,
        "existing_onboarding_models": existing_onboarding_models,
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
        while end < len(lines) and lines[end].startswith((" ", "\t", "- ")):
            end += 1
        return "".join([*lines[:start], *block, *lines[end:]])

    insertion = next(
        (index for index, line in enumerate(lines) if line.strip() and not line.lstrip().startswith("#")),
        len(lines),
    )
    return "".join([*lines[:insertion], *block, "\n", *lines[insertion:]])


def _replace_model_rationale(text: str, rationale: str) -> str:
    lines = text.splitlines(keepends=True)
    model_start = next((index for index, line in enumerate(lines) if line.startswith("model:")), None)
    if model_start is None:
        raise ValueError("Recipe is missing its model block")
    model_end = model_start + 1
    while model_end < len(lines) and (not lines[model_end].strip() or lines[model_end].startswith((" ", "\t"))):
        model_end += 1
    rendered = json.dumps(rationale, ensure_ascii=False)
    rationale_line = f"  rationale: {rendered}\n"
    existing = next(
        (index for index in range(model_start + 1, model_end) if lines[index].startswith("  rationale:")),
        None,
    )
    huggingface = next(
        (index for index in range(model_start + 1, model_end) if lines[index].startswith("  huggingface:")),
        model_start,
    )
    if existing is not None:
        lines.pop(existing)
        if existing < huggingface:
            huggingface -= 1
    lines.insert(huggingface + 1, rationale_line)
    return "".join(lines)


def _set_lifecycle(record: dict, lifecycle: str, rationale: str) -> bool:
    path = record["path"]
    before = path.read_text()
    after = _replace_tag_block(before, _tags_with_lifecycle(record["tags"], lifecycle))
    after = _replace_model_rationale(after, rationale)
    if before == after:
        return False
    path.write_text(after)
    return True


def _model_lines(decisions: list[dict[str, str]]) -> list[str]:
    return [f"- `{decision['model_id']}` — {decision['rationale']}" for decision in decisions] or ["- None."]


def _obsolete_lines(decisions: list[dict[str, str]]) -> list[str]:
    if not decisions:
        return ["- None."]
    lines = []
    for decision in decisions:
        replacement = decision.get("replacement_model_id")
        successor = f" → `{replacement}`" if replacement else ""
        lines.append(f"- `{decision['model_id']}`{successor} — {decision['rationale']}")
    return lines


def _onboarding_lines(decisions: list[dict]) -> list[str]:
    lines = []
    for decision in decisions:
        deployments = ", ".join(f"`{deployment['deploy.gpu']} x{deployment['deploy.gpu_count']}`" for deployment in decision["deployments"])
        lines.append(f"- `{decision['model_id']}` on {deployments} — {decision['rationale']}")
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
        lines.extend(_onboarding_lines(manifest["onboarding_models"]))
    else:
        lines.append("- None in this run.")
    if manifest["existing_onboarding_models"]:
        lines.extend(["", "### Existing onboarding shells", ""])
        lines.extend(_onboarding_lines(manifest["existing_onboarding_models"]))
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
    records = recipe_catalog(workspace / "recipes")
    changed = False
    for decision in manifest["maintained_models"]:
        changed = _set_lifecycle(records[decision["model_id"]], MAINTAINED_TAG, decision["rationale"]) or changed
    for decision in manifest["best_effort_models"]:
        changed = _set_lifecycle(records[decision["model_id"]], BEST_EFFORT_TAG, decision["rationale"]) or changed
    for decision in manifest["obsolete_models"]:
        changed = _set_lifecycle(records[decision["model_id"]], OBSOLETE_TAG, decision["rationale"]) or changed
    for decision in manifest["existing_onboarding_models"]:
        changed = _set_lifecycle(records[decision["model_id"]], ONBOARDING_TAG, decision["rationale"]) or changed
    for candidate in manifest["onboarding_models"]:
        create_recipe_stub(
            workspace / "recipes",
            candidate["model_id"],
            candidate["rationale"],
            candidate["task"],
            candidate["deployments"],
        )
        changed = True
    summary_path.write_text(_summary(manifest))
    return {"changed": changed}


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
    parser.add_argument("--summary", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        workspace = args.workspace.resolve()
        manifest = validate_manifest(args.input, workspace)
        result = apply_manifest(manifest, workspace, args.summary)
        _write_outputs(result)
        print(json.dumps({**manifest, **result}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
