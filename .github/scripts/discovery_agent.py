#!/usr/bin/env python3
"""Prepare discovery tasks and assemble lifecycle manifests from agent selections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from emmy.recipe.catalog import recipe_inventory_document
from emmy.recipe.lifecycle import ONBOARDING_TAG

TASK_SCHEMA_VERSION = 1
SELECTION_FIELDS = frozenset({"scores", "maintained_model_ids", "obsolete_models", "new_onboarding_models"})
SCORE_FIELDS = frozenset({"model_id", "rationale", "heat"})
OBSOLETE_FIELDS = frozenset({"model_id", "replacement_model_id"})


def build_task(root: Path, maintained_count: int, batch_size: int) -> dict:
    """Return a compact, versioned task with deterministically batched recipes."""
    if maintained_count < 1:
        raise ValueError("maintained_count must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    inventory = recipe_inventory_document(root)
    recipes = inventory["recipes"]
    maintainable_count = sum(ONBOARDING_TAG not in recipe["tags"] and recipe["runnable"] for recipe in recipes)
    if maintained_count > maintainable_count:
        raise ValueError(f"Cannot select {maintained_count} maintained recipes from {maintainable_count} runnable recipes")

    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "maintained_count": maintained_count,
        "recipe_batches": [recipes[index : index + batch_size] for index in range(0, len(recipes), batch_size)],
    }


def _extract_selection(text: str) -> dict:
    stripped = text.strip()
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
            if isinstance(candidate, dict) and set(candidate) == SELECTION_FIELDS:
                candidates.append(candidate)
        if not candidates:
            raise ValueError("Discovery output must contain a selection JSON object") from None
        value = candidates[-1]
    if not isinstance(value, dict) or set(value) != SELECTION_FIELDS:
        raise ValueError(f"Discovery selection must contain exactly: {', '.join(sorted(SELECTION_FIELDS))}")
    return value


def _task_recipes(task: object) -> tuple[int, list[dict]]:
    if not isinstance(task, dict) or set(task) != {"schema_version", "maintained_count", "recipe_batches"}:
        raise ValueError("Discovery task has an invalid shape")
    if task["schema_version"] != TASK_SCHEMA_VERSION:
        raise ValueError(f"Unsupported discovery task schema version: {task['schema_version']!r}")
    maintained_count = task["maintained_count"]
    if not isinstance(maintained_count, int) or isinstance(maintained_count, bool) or maintained_count < 1:
        raise ValueError("Discovery task maintained_count must be positive")
    batches = task["recipe_batches"]
    if not isinstance(batches, list) or not all(isinstance(batch, list) and batch for batch in batches):
        raise ValueError("Discovery task recipe_batches must contain non-empty lists")
    recipes = [recipe for batch in batches for recipe in batch]
    if not all(isinstance(recipe, dict) for recipe in recipes):
        raise ValueError("Discovery task recipes must be objects")
    model_ids = [recipe.get("model_id") for recipe in recipes]
    if not all(isinstance(model_id, str) for model_id in model_ids):
        raise ValueError("Discovery task contains an invalid model ID")
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Discovery task contains duplicate model IDs")
    return maintained_count, recipes


def _scores(value: object, recipe_ids: set[str]) -> dict[str, dict]:
    if not isinstance(value, list):
        raise ValueError("scores must be a list")
    scores = {}
    for score in value:
        if not isinstance(score, dict) or set(score) != SCORE_FIELDS:
            raise ValueError(f"Each score must contain exactly: {', '.join(sorted(SCORE_FIELDS))}")
        model_id = score["model_id"]
        if model_id not in recipe_ids:
            raise ValueError(f"Score uses an unknown or inexact existing model ID: {model_id!r}")
        if model_id in scores:
            raise ValueError(f"Duplicate score for {model_id}")
        scores[model_id] = score
    missing = recipe_ids - set(scores)
    if missing:
        raise ValueError(f"Existing recipes must be scored: {', '.join(sorted(missing))}")
    return scores


def _model_ids(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(model_id, str) for model_id in value):
        raise ValueError(f"{field} must be a list of exact model IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicates")
    return value


def _obsolete_selections(value: object, complete_ids: set[str]) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("obsolete_models must be a list")
    selections = []
    seen = set()
    for selection in value:
        fields = set(selection) if isinstance(selection, dict) else set()
        if fields not in ({"model_id"}, OBSOLETE_FIELDS):
            raise ValueError("Each obsolete selection needs model_id and an optional replacement_model_id")
        model_id = selection["model_id"]
        if model_id not in complete_ids:
            raise ValueError(f"Obsolete selection must use an exact complete recipe ID: {model_id!r}")
        if model_id in seen:
            raise ValueError(f"Duplicate obsolete selection for {model_id}")
        replacement = selection.get("replacement_model_id")
        if replacement is not None and replacement not in complete_ids:
            raise ValueError(f"Obsolete replacement must use an exact complete recipe ID: {replacement!r}")
        seen.add(model_id)
        selections.append(selection)
    return selections


def assemble_manifest(task: dict, selection_text: str) -> dict:
    """Validate a compact agent selection and deterministically assemble a lifecycle manifest."""
    maintained_count, recipes = _task_recipes(task)
    selection = _extract_selection(selection_text)
    recipe_ids = {recipe["model_id"] for recipe in recipes}
    complete_ids = {recipe["model_id"] for recipe in recipes if ONBOARDING_TAG not in recipe.get("tags", [])}
    onboarding_ids = recipe_ids - complete_ids
    maintainable_ids = {recipe["model_id"] for recipe in recipes if recipe["model_id"] in complete_ids and recipe.get("runnable")}
    scores = _scores(selection["scores"], recipe_ids)

    maintained_ids = _model_ids(selection["maintained_model_ids"], "maintained_model_ids")
    if len(maintained_ids) != maintained_count:
        raise ValueError(f"maintained_model_ids must contain exactly {maintained_count} entries")
    invalid_maintained = set(maintained_ids) - maintainable_ids
    if invalid_maintained:
        raise ValueError(f"Maintained selections must be runnable complete recipes: {', '.join(sorted(invalid_maintained))}")

    obsolete = _obsolete_selections(selection["obsolete_models"], complete_ids)
    obsolete_ids = {item["model_id"] for item in obsolete}
    overlap = set(maintained_ids) & obsolete_ids
    if overlap:
        raise ValueError(f"Maintained recipes cannot also be obsolete: {', '.join(sorted(overlap))}")
    new_onboarding = selection["new_onboarding_models"]
    if not isinstance(new_onboarding, list):
        raise ValueError("new_onboarding_models must be a list")
    new_onboarding = [
        candidate for candidate in new_onboarding if not isinstance(candidate, dict) or candidate.get("model_id") not in recipe_ids
    ]

    maintained_set = set(maintained_ids)
    manifest = {
        "maintained_models": [scores[model_id] for model_id in maintained_ids],
        "best_effort_models": [
            scores[recipe["model_id"]] for recipe in recipes if recipe["model_id"] in complete_ids - maintained_set - obsolete_ids
        ],
        "obsolete_models": [{**scores[item["model_id"]], **item} for item in obsolete],
        "onboarding_models": [],
    }
    for recipe in recipes:
        model_id = recipe["model_id"]
        if model_id not in onboarding_ids:
            continue
        deployments = [
            {"deploy.gpu": deployment["gpu"], "deploy.gpu_count": deployment["gpu_count"]} for deployment in recipe.get("deployments", [])
        ]
        manifest["onboarding_models"].append(
            {
                **scores[model_id],
                "task": recipe.get("task", "generate"),
                "deployments": deployments,
            }
        )
    manifest["onboarding_models"].extend(new_onboarding)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    task = subparsers.add_parser("task", help="Build a compact discovery task")
    task.add_argument("--root", type=Path, default=Path("recipes"))
    task.add_argument("--maintained-count", type=int, required=True)
    task.add_argument("--batch-size", type=int, default=6)

    assemble = subparsers.add_parser("assemble", help="Assemble a lifecycle manifest from agent output")
    assemble.add_argument("--task", type=Path, required=True)
    assemble.add_argument("--selection", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "task":
            result = build_task(args.root, args.maintained_count, args.batch_size)
        else:
            result = assemble_manifest(json.loads(args.task.read_text()), args.selection.read_text())
        sys.stdout.write(json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n")
        return 0
    except Exception as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
