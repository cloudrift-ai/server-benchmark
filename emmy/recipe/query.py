"""Constrained filtering and ordering for compact recipe inventory rows."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cmp_to_key
from pathlib import Path
from typing import Any

from emmy.provisioning.candidates import iter_candidates
from emmy.provisioning.cloudrift import list_available_instance_types, validate_team_id_access
from emmy.recipe.catalog import HF_ID
from emmy.recipe.lifecycle import BEST_EFFORT_TAG, LIFECYCLE_TAGS, MAINTAINED_TAG, OBSOLETE_TAG

QUERY_SCHEMA_VERSION = 1

_FIELD = r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*"
_PREDICATE = re.compile(rf"^\s*({_FIELD})\s*(==|!=|>=|<=|>|<|in|contains|matches)\s*(.+?)\s*$")
_ORDER_SORT = re.compile(rf"^\s*({_FIELD})\s+order\s+(.+?)\s*$")
_DIRECTION_SORT = re.compile(rf"^\s*({_FIELD})\s+(asc|desc)(?:\s+(nulls-first|nulls-last))?\s*$")

RECIPE_FIELDS = frozenset(
    {
        "name",
        "recipe_path",
        "model_id",
        "tags",
        "lifecycle",
        "task",
        "runnable",
        "rationale",
        "operation",
        "expected_lifecycle",
        "deployment.index",
        "deployment.gpu",
        "deployment.gpu_count",
        "deployment.context_length",
        "deployment.availability.cloudrift",
        "results.path",
        "results.last_run_at",
        "provider.cloudrift.team_access",
    }
)


@dataclass(frozen=True)
class Predicate:
    """One validated row predicate."""

    field: str
    operator: str
    value: Any


@dataclass(frozen=True)
class SortExpression:
    """One validated row sort key."""

    field: str
    direction: str | None = None
    nulls: str = "nulls-last"
    order: tuple[Any, ...] | None = None


def _parse_json_value(source: str, *, context: str) -> Any:
    try:
        return json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context} value must be valid JSON: {source!r}") from exc


def _validate_field(field: str) -> None:
    if field not in RECIPE_FIELDS:
        raise ValueError(f"Unknown recipe query field: {field}")


def parse_predicate(source: str) -> Predicate:
    """Parse one constrained filter or requirement expression."""
    match = _PREDICATE.fullmatch(source)
    if match is None:
        raise ValueError(f"Invalid recipe query predicate: {source!r}")
    field, operator, raw_value = match.groups()
    _validate_field(field)
    value = _parse_json_value(raw_value, context="Predicate")
    if operator == "in" and not isinstance(value, list):
        raise ValueError("The 'in' operator requires a JSON array")
    if operator == "matches":
        if not isinstance(value, str):
            raise ValueError("The 'matches' operator requires a JSON string")
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression: {value!r}") from exc
    return Predicate(field, operator, value)


def parse_sort(source: str) -> SortExpression:
    """Parse one direction or explicit-order sort expression."""
    order_match = _ORDER_SORT.fullmatch(source)
    if order_match is not None:
        field, raw_order = order_match.groups()
        _validate_field(field)
        order = _parse_json_value(raw_order, context="Sort order")
        if not isinstance(order, list) or not order:
            raise ValueError("An explicit sort order requires a non-empty JSON array")
        if any(isinstance(value, (dict, list)) for value in order):
            raise ValueError("Explicit sort-order values must be JSON scalars")
        encoded = [json.dumps(value, sort_keys=True) for value in order]
        if len(encoded) != len(set(encoded)):
            raise ValueError("Explicit sort-order values must be unique")
        return SortExpression(field=field, order=tuple(order))

    direction_match = _DIRECTION_SORT.fullmatch(source)
    if direction_match is None:
        raise ValueError(f"Invalid recipe query sort: {source!r}")
    field, direction, nulls = direction_match.groups()
    _validate_field(field)
    return SortExpression(field=field, direction=direction, nulls=nulls or "nulls-last")


def referenced_fields(
    filters: list[Predicate],
    requirements: list[Predicate],
    sorts: list[SortExpression],
) -> frozenset[str]:
    """Return every field needed to evaluate a parsed query."""
    return frozenset([predicate.field for predicate in [*filters, *requirements]] + [sort.field for sort in sorts])


def _lifecycle(tags: list[str]) -> str | None:
    return next((tag for tag in tags if tag in LIFECYCLE_TAGS), None)


def _operation(lifecycle: str | None) -> tuple[str | None, str | None]:
    if lifecycle == MAINTAINED_TAG:
        return "verification", MAINTAINED_TAG
    if lifecycle == BEST_EFFORT_TAG:
        return "verification", BEST_EFFORT_TAG
    if lifecycle == OBSOLETE_TAG:
        return None, None
    return "onboarding", BEST_EFFORT_TAG


def _base_row(record: dict | None, model_id: str) -> dict:
    tags = list(record["tags"]) if record is not None else []
    lifecycle = _lifecycle(tags)
    operation, expected_lifecycle = _operation(lifecycle)
    recipe_path = record["path"] if record is not None else None
    results_path = str(Path(recipe_path).with_name("RESULTS.md")) if recipe_path else None
    return {
        "name": record["name"] if record is not None else None,
        "recipe_path": recipe_path,
        "model_id": model_id,
        "tags": tags,
        "lifecycle": lifecycle,
        "task": record["task"] if record is not None else None,
        "runnable": record["runnable"] if record is not None else False,
        "rationale": record["rationale"] if record is not None else None,
        "operation": operation,
        "expected_lifecycle": expected_lifecycle,
        "results": {"path": results_path, "last_run_at": None},
        "provider": {"cloudrift": {"team_access": None}},
    }


def build_query_rows(
    inventory: list[dict],
    *,
    model_id: str | None = None,
    allow_missing_model: bool = False,
    gpu: str | None = None,
    gpu_count: int | None = None,
    expand_deployments: bool = False,
) -> list[dict]:
    """Build normalized recipe or recipe/deployment rows for query evaluation."""
    if model_id is not None and HF_ID.fullmatch(model_id) is None:
        raise ValueError("model_id must be an exact Hugging Face owner/repository ID")
    if allow_missing_model and model_id is None:
        raise ValueError("--allow-missing-model requires --model")
    if (gpu is None) != (gpu_count is None):
        raise ValueError("--gpu and --gpu-count must be used together")
    if gpu is not None and model_id is None:
        raise ValueError("An explicit deployment requires --model")
    if gpu_count is not None and (isinstance(gpu_count, bool) or gpu_count < 1):
        raise ValueError("gpu_count must be positive")

    by_model = {record["model_id"]: record for record in inventory}
    if model_id is not None:
        record = by_model.get(model_id)
        if record is None and not allow_missing_model:
            raise ValueError(f"No recipe found for model {model_id}")
        records = [(model_id, record)]
    else:
        records = [(record["model_id"], record) for record in inventory]

    rows = []
    for selected_model_id, record in records:
        base = _base_row(record, selected_model_id)
        if gpu is not None:
            deployments = [{"gpu": gpu, "gpu_count": gpu_count, "context_length": None}]
        elif expand_deployments:
            deployments = record["deployments"] if record is not None else []
        else:
            deployments = [None]

        for index, deployment in enumerate(deployments):
            row = {
                **base,
                "results": dict(base["results"]),
                "provider": {"cloudrift": dict(base["provider"]["cloudrift"])},
            }
            if deployment is None:
                row["deployment"] = None
            else:
                row["deployment"] = {
                    "index": index,
                    "gpu": deployment["gpu"],
                    "gpu_count": deployment["gpu_count"],
                    "context_length": deployment.get("context_length"),
                    "availability": {"cloudrift": None},
                }
            rows.append(row)
    return rows


def field_value(row: dict, field: str) -> Any:
    """Read a validated dotted field from a normalized query row."""
    _validate_field(field)
    value: Any = row
    for part in field.split("."):
        if value is None:
            return None
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Recipe query field was not resolved: {field}")
        value = value[part]
    return value


def _matches(row: dict, predicate: Predicate) -> bool:
    current = field_value(row, predicate.field)
    expected = predicate.value
    operator = predicate.operator
    if operator == "==":
        return current == expected
    if operator == "!=":
        return current != expected
    if operator == "in":
        return current in expected
    if operator == "contains":
        try:
            return expected in current
        except TypeError as exc:
            raise ValueError(f"Field {predicate.field} does not support 'contains'") from exc
    if operator == "matches":
        if not isinstance(current, str):
            return False
        return re.search(expected, current) is not None
    try:
        if operator == ">":
            return current > expected
        if operator == ">=":
            return current >= expected
        if operator == "<":
            return current < expected
        if operator == "<=":
            return current <= expected
    except TypeError as exc:
        raise ValueError(f"Field {predicate.field} cannot be compared with {expected!r}") from exc
    raise AssertionError(f"Unhandled predicate operator: {operator}")


def _compare_values(left: Any, right: Any, sort: SortExpression) -> int:
    if left is None or right is None:
        if left is right:
            return 0
        null_first = sort.nulls == "nulls-first"
        return -1 if (left is None) == null_first else 1

    if sort.order is not None:
        ranks = {json.dumps(value, sort_keys=True): index for index, value in enumerate(sort.order)}
        left_key = json.dumps(left, sort_keys=True)
        right_key = json.dumps(right, sort_keys=True)
        left_rank = ranks.get(left_key, len(ranks))
        right_rank = ranks.get(right_key, len(ranks))
        if left_rank != right_rank:
            return -1 if left_rank < right_rank else 1
        if left_key == right_key:
            return 0
        return -1 if left_key < right_key else 1

    try:
        comparison = (left > right) - (left < right)
    except TypeError as exc:
        raise ValueError(f"Field {sort.field} contains values that cannot be ordered together") from exc
    return -comparison if sort.direction == "desc" else comparison


def query_rows(
    rows: list[dict],
    *,
    filters: list[Predicate],
    requirements: list[Predicate],
    sorts: list[SortExpression],
    limit: int | None,
) -> list[dict]:
    """Apply requirements, filters, stable lexicographic ordering, and a limit."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    for requirement in requirements:
        failed = list(dict.fromkeys(row["model_id"] for row in rows if not _matches(row, requirement)))
        if failed:
            raise ValueError(
                f"Recipe query requirement failed for {', '.join(failed)}: "
                f"{requirement.field} {requirement.operator} {json.dumps(requirement.value)}"
            )

    selected = [row for row in rows if all(_matches(row, predicate) for predicate in filters)]

    def compare(left: dict, right: dict) -> int:
        for sort in sorts:
            result = _compare_values(field_value(left, sort.field), field_value(right, sort.field), sort)
            if result:
                return result
        return 0

    if sorts:
        selected.sort(key=cmp_to_key(compare))
    if limit is not None:
        selected = selected[:limit]
    return selected


def _annotate_cloudrift_availability(rows: list[dict], available: set[str]) -> None:
    for row in rows:
        deployment = row["deployment"]
        if deployment is None:
            continue
        try:
            candidates = iter_candidates(
                deployment["gpu"],
                deployment["gpu_count"],
                "cloudrift",
                exact_gpu_count=True,
            )
        except (TypeError, ValueError):
            deployment["availability"]["cloudrift"] = False
            continue
        deployment["availability"]["cloudrift"] = any(candidate.instance_type in available for candidate in candidates)


def _git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("results.last_run_at requires a Git checkout with history")
    return Path(result.stdout.strip()).resolve()


def _annotate_last_recipe_runs(rows: list[dict]) -> None:
    root = _git_root()
    timestamps: dict[str, str | None] = {}
    for row in rows:
        results_path = row["results"]["path"]
        if results_path is None:
            continue
        try:
            git_path = Path(results_path).resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Recipe results path is outside the Git checkout: {results_path}") from exc
        key = str(git_path)
        if key not in timestamps:
            result = subprocess.run(
                ["git", "-C", str(root), "log", "-1", "--format=%ct", "--", key],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                message = result.stderr.strip() or f"git log failed for {key}"
                raise ValueError(message)
            raw_timestamp = result.stdout.strip()
            if raw_timestamp:
                timestamps[key] = datetime.fromtimestamp(int(raw_timestamp), tz=UTC).isoformat().replace("+00:00", "Z")
            else:
                timestamps[key] = None
        row["results"]["last_run_at"] = timestamps[key]


async def enrich_query_rows(
    rows: list[dict],
    fields: frozenset[str],
    *,
    cloudrift_api_key: str | None,
    cloudrift_team_id: str | None,
) -> None:
    """Resolve only the provider and Git fields referenced by a query."""
    needs_team_access = "provider.cloudrift.team_access" in fields
    needs_availability = "deployment.availability.cloudrift" in fields
    if needs_team_access or needs_availability:
        if not cloudrift_api_key:
            raise ValueError("CLOUDRIFT_API_KEY is required by the requested recipe query fields")
        if needs_team_access:
            await validate_team_id_access(cloudrift_api_key, cloudrift_team_id)
            for row in rows:
                row["provider"]["cloudrift"]["team_access"] = True
        if needs_availability:
            available = await list_available_instance_types(cloudrift_api_key)
            _annotate_cloudrift_availability(rows, available)
    if "results.last_run_at" in fields:
        _annotate_last_recipe_runs(rows)


def query_document(rows: list[dict]) -> dict:
    """Wrap query rows in their versioned machine-readable envelope."""
    return {"schema_version": QUERY_SCHEMA_VERSION, "rows": rows}
