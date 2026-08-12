"""Built-in deterministic serving-output equivalence gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from emmy.benchmark.workload import (
    load_output_equivalence_prompts,
    output_equivalence_request,
    output_equivalence_workload,
)

if TYPE_CHECKING:
    from emmy.planner import BenchmarkTask


class OutputEquivalenceError(ValueError):
    """Raised when paired serving output is missing, ambiguous, or different."""


def validate_output_equivalence_selection(selected: list[BenchmarkTask], full: list[BenchmarkTask]) -> None:
    """Reject filtered execution of a recipe whose equivalence gate covers its full matrix."""
    prompt_files = {task.recipe.benchmark.output_equivalence_file for task in full}
    if not full or prompt_files == {None}:
        return
    if None in prompt_files or len(prompt_files) != 1:
        raise OutputEquivalenceError("every task in an output-equivalence recipe must use the same prompt file")
    selected_ids = {task.task_id for task in selected}
    full_ids = {task.task_id for task in full}
    if len(selected) != len(full) or selected_ids != full_ids:
        raise OutputEquivalenceError(
            f"output equivalence requires the complete unfiltered recipe matrix ({len(full)} tasks); selected {len(selected)}"
        )


def _expected_keys(tasks: list[BenchmarkTask], prompt_ids: set[str]) -> tuple[list[str], dict[str, set[tuple[str, int, str]]]]:
    by_arm: dict[str, set[tuple[str, int, str]]] = {}
    for task in tasks:
        bench = task.recipe.benchmark
        arm = bench.comparison_arm
        repeat = bench.process_repeat
        if not isinstance(arm, str) or not arm or not isinstance(repeat, int) or repeat < 0:
            raise OutputEquivalenceError(f"{task.task_id} requires benchmark.comparison_arm and a non-negative benchmark.process_repeat")
        keys = {(output_equivalence_workload(bench), repeat, prompt_id) for prompt_id in prompt_ids}
        overlap = by_arm.setdefault(arm, set()) & keys
        if overlap:
            raise OutputEquivalenceError(f"duplicate configured output-equivalence keys for arm {arm}: {sorted(overlap)}")
        by_arm[arm].update(keys)
    arms = sorted(by_arm)
    if len(arms) != 2:
        raise OutputEquivalenceError(f"output equivalence requires exactly two comparison arms, found {arms}")
    if by_arm[arms[0]] != by_arm[arms[1]]:
        raise OutputEquivalenceError("comparison arms do not define the same workload/repeat/prompt keys")
    return arms, by_arm


def validate_output_equivalence(tasks: list[BenchmarkTask], prompt_path: Path) -> dict:
    """Validate all paired task results and return a compact passing report."""
    if not tasks:
        raise OutputEquivalenceError("output equivalence requires benchmark tasks")
    prompt_files = {task.recipe.benchmark.output_equivalence_file for task in tasks}
    resolved_prompt_files = {Path(item).resolve() for item in prompt_files if item is not None}
    if None in prompt_files or resolved_prompt_files != {prompt_path.resolve()}:
        raise OutputEquivalenceError(
            f"output-equivalence tasks must share one prompt file, found {sorted(str(item) for item in prompt_files)}"
        )
    try:
        prompts = load_output_equivalence_prompts(prompt_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise OutputEquivalenceError(f"cannot load output prompts {prompt_path}: {exc}") from exc
    arms, expected = _expected_keys(tasks, set(prompts))
    records: dict[str, dict[tuple[str, int, str], dict]] = {arm: {} for arm in arms}

    for task in tasks:
        path = task.json_result_path()
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OutputEquivalenceError(f"cannot load benchmark result {path}: {exc}") from exc
        bench = task.recipe.benchmark
        metrics = document.get("metrics", {})
        if metrics.get("successful_requests") != bench.num_prompts or metrics.get("failed_requests") != 0:
            raise OutputEquivalenceError(f"benchmark result {path} did not complete every request successfully")
        probe = document.get("output_probe")
        if not isinstance(probe, list) or not probe:
            raise OutputEquivalenceError(f"benchmark result {path} has no pre-teardown output probe")
        arm = bench.comparison_arm
        assert isinstance(arm, str)
        for row in probe:
            key = (row.get("workload"), row.get("repeat"), row.get("prompt_id"))
            if row.get("schema_version") != 1 or row.get("arm") != arm or key not in expected[arm]:
                raise OutputEquivalenceError(f"benchmark result {path} has an invalid output-probe record")
            if key in records[arm]:
                raise OutputEquivalenceError(f"duplicate {arm} output-probe record for {key}")
            prompt = prompts[key[2]]
            if row.get("request") != output_equivalence_request(task.recipe.model_name, prompt):
                raise OutputEquivalenceError(f"request differs from the frozen output prompt for {key}")
            records[arm][key] = row

    for arm in arms:
        if set(records[arm]) != expected[arm]:
            missing = sorted(expected[arm] - set(records[arm]))
            extra = sorted(set(records[arm]) - expected[arm])
            raise OutputEquivalenceError(f"{arm} output-probe keys differ from the configured tasks: missing={missing}, extra={extra}")

    fields = ("request", "text", "completion_tokens", "finish_reason")
    for key in sorted(expected[arms[0]]):
        differences = [field for field in fields if records[arms[0]][key].get(field) != records[arms[1]][key].get(field)]
        if differences:
            raise OutputEquivalenceError(f"output mismatch for {key}: {', '.join(differences)}")
    return {
        "schema_version": 1,
        "status": "pass",
        "arms": arms,
        "paired_records": len(expected[arms[0]]),
        "prompt_file": str(prompt_path),
    }


def run_output_equivalence_gate(tasks: list[BenchmarkTask], prompt_path: Path, report_path: Path) -> bool:
    """Write a report and return whether the configured equivalence gate passed."""
    try:
        report = validate_output_equivalence(tasks, prompt_path)
    except OutputEquivalenceError as exc:
        report = {"schema_version": 1, "status": "fail", "error": str(exc), "prompt_file": str(prompt_path)}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report["status"] == "pass"
