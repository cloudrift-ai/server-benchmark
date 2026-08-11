#!/usr/bin/env python3
"""Validate exact stock-versus-Emmy completions across five fresh servers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_EXPECTED_WORKLOADS = {"i256-o256-c64", "i4096-o4096-c1", "i4096-o4096-c8", "i8192-o256-c4"}


class OutputEquivalenceError(ValueError):
    """Raised when a serving output artifact is missing, duplicated, or different."""


def load_prompts(path: Path) -> list[dict]:
    """Load the frozen prompt set and reject ambiguous identifiers."""
    try:
        prompts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise OutputEquivalenceError(f"cannot load prompts {path}: {exc}") from exc
    ids = [item.get("id") for item in prompts]
    if not prompts or any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise OutputEquivalenceError("prompt file must contain unique non-empty string ids")
    if any(not isinstance(item.get("prompt"), str) or not item["prompt"] for item in prompts):
        raise OutputEquivalenceError("every frozen prompt must contain non-empty text")
    if any(not isinstance(item.get("max_tokens"), int) or item["max_tokens"] <= 0 for item in prompts):
        raise OutputEquivalenceError("every frozen prompt must contain positive max_tokens")
    return prompts


def records_from_results(paths: list[Path]) -> tuple[dict[tuple[str, int, str], dict], dict[tuple[str, int, str], dict]]:
    """Extract embedded pre-teardown captures from benchmark JSON artifacts."""
    by_arm = {"stock": {}, "emmy": {}}
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OutputEquivalenceError(f"cannot load benchmark result {path}: {exc}") from exc
        benchmark = document.get("recipe", {}).get("benchmark", {})
        metrics = document.get("metrics", {})
        num_prompts = benchmark.get("num_prompts")
        if metrics.get("successful_requests") != num_prompts or metrics.get("failed_requests") != 0:
            raise OutputEquivalenceError(f"benchmark result {path} did not complete every request successfully")
        records = document.get("output_probe")
        if not isinstance(records, list) or not records:
            raise OutputEquivalenceError(f"benchmark result {path} has no pre-teardown output probe")
        for item in records:
            arm = item.get("arm")
            if item.get("schema_version") != 1 or arm not in by_arm:
                raise OutputEquivalenceError(f"benchmark result {path} has an invalid output-probe record")
            key = (item.get("workload"), item.get("repeat"), item.get("prompt_id"))
            if key in by_arm[arm]:
                raise OutputEquivalenceError(f"duplicate {arm} record for workload/repeat/prompt {key}")
            by_arm[arm][key] = item
    return by_arm["stock"], by_arm["emmy"]


def _validate_records(prompts_path: Path, stock: dict[tuple[str, int, str], dict], emmy: dict[tuple[str, int, str], dict]) -> None:
    """Require complete, request-identical, byte-exact semantic outputs."""
    frozen_prompts = {item["id"]: item for item in load_prompts(prompts_path)}
    prompt_ids = set(frozen_prompts)
    expected = {(workload, repeat, prompt_id) for workload in _EXPECTED_WORKLOADS for repeat in range(5) for prompt_id in prompt_ids}
    if set(stock) != expected:
        raise OutputEquivalenceError(f"stock keys differ from the frozen workload/repeat set: missing={sorted(expected - set(stock))}")
    if set(emmy) != expected:
        raise OutputEquivalenceError(f"Emmy keys differ from the frozen workload/repeat set: missing={sorted(expected - set(emmy))}")
    for key in sorted(expected):
        left = stock[key]
        right = emmy[key]
        prompt_id = key[2]
        frozen = frozen_prompts[prompt_id]
        canonical_request = {
            "model": "google/gemma-4-12B-it",
            "prompt": frozen["prompt"],
            "max_tokens": frozen["max_tokens"],
            "temperature": 0,
            "seed": 0,
            "n": 1,
            "stream": False,
        }
        if left.get("request") != canonical_request or right.get("request") != canonical_request:
            raise OutputEquivalenceError(f"request differs from frozen prompt contract for workload/repeat/prompt {key}")
        if left.get("request") != right.get("request"):
            raise OutputEquivalenceError(f"request mismatch for workload/repeat/prompt {key}")
        fields = ("text", "completion_tokens", "finish_reason")
        differences = [field for field in fields if left.get(field) != right.get(field)]
        if differences:
            raise OutputEquivalenceError(f"semantic output mismatch for workload/repeat/prompt {key}: {', '.join(differences)}")


def validate_results(prompts_path: Path, result_paths: list[Path]) -> None:
    """Validate output probes embedded in benchmark result JSON files."""
    _validate_records(prompts_path, *records_from_results(result_paths))


def main() -> int:
    """Run the validator CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    status = "pass"
    error = None
    try:
        validate_results(args.prompts, args.results)
    except OutputEquivalenceError as exc:
        status = "fail"
        error = str(exc)
    if args.report:
        args.report.write_text(json.dumps({"schema_version": 1, "status": status, "error": error}, indent=2) + "\n", encoding="utf-8")
    if error:
        parser.error(error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
