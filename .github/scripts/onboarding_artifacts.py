#!/usr/bin/env python3
"""Validate and stage the artifact manifest returned by the onboard-model skill."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ALLOWED_ARTIFACT_PREFIXES = (
    "docker/vllm-emmy-serve/models/",
    "emmy/compiler/pipeline/search/goldens/",
    "experiments/",
    "recipes/",
)


def _relative_file(workspace: Path, raw_path: str, prefixes: tuple[str, ...]) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts or not path.as_posix().startswith(prefixes):
        raise ValueError(f"Artifact path is outside the allowed onboarding areas: {raw_path}")
    resolved = workspace / path
    if not resolved.is_file():
        raise ValueError(f"Artifact does not exist: {raw_path}")
    return path


def _relative_experiment_recipe(workspace: Path, raw_path: str) -> Path:
    path = _relative_file(workspace, raw_path, ("experiments/",))
    if path.name != "recipe.yaml":
        raise ValueError(f"Only experiment recipe.yaml files may be retained: {raw_path}")
    return path


def _relative_artifact(workspace: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    normalized = path.as_posix()
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Artifact path is not repository-relative: {raw_path}")
    if normalized.startswith(ALLOWED_ARTIFACT_PREFIXES) and (workspace / path).is_file():
        return path
    if normalized.startswith("plans/onboard-") and normalized.endswith(".md") and not (workspace / path).exists():
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", normalized],
            cwd=workspace,
            capture_output=True,
            check=False,
        )
        if tracked.returncode == 0:
            return path
    raise ValueError(f"Artifact path is outside the allowed onboarding areas or does not exist: {raw_path}")


def _invalid_result_artifacts(artifacts: list[Path]) -> list[Path]:
    return [
        path
        for path in artifacts
        if (path.parts[0] == "experiments" and path.name != "recipe.yaml")
        or (path.parts[0] == "recipes" and path.name not in {"recipe.yaml", "RESULTS.md"})
    ]


def validate_summary(
    summary_path: Path,
    workspace: Path,
    model_id: str,
    gpu: str,
    gpu_count: int,
    ssh_target: str,
) -> tuple[dict, list[Path]]:
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "success":
        raise ValueError(f"Onboarding did not succeed: {summary.get('failure')}")
    if summary.get("model_id") != model_id:
        raise ValueError(f"Summary model mismatch: {summary.get('model_id')} != {model_id}")
    target = summary.get("target") or {}
    expected_target = {"gpu": gpu, "gpu_count": gpu_count, "ssh": ssh_target}
    if target != expected_target:
        raise ValueError(f"Summary target mismatch: {target} != {expected_target}")
    cleanup = summary.get("cleanup") or {}
    if cleanup.get("workloads") != "complete" or cleanup.get("docker_logout") is not True:
        raise ValueError(f"Remote workload or Docker credential cleanup is incomplete: {cleanup}")

    recipe = _relative_file(workspace, summary.get("recipe") or "", ("recipes/",))
    report = _relative_file(workspace, summary.get("report") or "", ("recipes/",))
    if report != recipe.with_name("RESULTS.md"):
        raise ValueError(f"Report must be RESULTS.md beside the final recipe: {report}")
    raw_experiment_artifacts = summary.get("experiment_artifacts")
    if not isinstance(raw_experiment_artifacts, list) or not raw_experiment_artifacts:
        raise ValueError("Summary must list the retained experiment recipe in experiment_artifacts")
    experiment_artifacts = [_relative_experiment_recipe(workspace, raw_path) for raw_path in raw_experiment_artifacts]
    experiment_recipe = Path(summary.get("experiment") or "")
    if experiment_recipe not in experiment_artifacts:
        raise ValueError("The experiment recipe must be included in experiment_artifacts")
    raw_artifacts = summary.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("Summary must list every intended repository file in artifacts")
    artifacts = [_relative_artifact(workspace, raw_path) for raw_path in raw_artifacts]
    invalid_result_artifacts = _invalid_result_artifacts(artifacts)
    if invalid_result_artifacts:
        raise ValueError(f"Only recipe.yaml and final recipe RESULTS.md artifacts may be retained: {invalid_result_artifacts}")
    if not {recipe, report, *experiment_artifacts}.issubset(set(artifacts)):
        raise ValueError("artifacts must include the recipe, report, and every experiment_artifacts entry")
    return summary, artifacts


def stage_artifacts(workspace: Path, artifacts: list[Path]) -> None:
    invalid_result_artifacts = _invalid_result_artifacts(artifacts)
    if invalid_result_artifacts:
        raise ValueError(f"Only recipe.yaml and final recipe RESULTS.md artifacts may be retained: {invalid_result_artifacts}")
    tracked_changes = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    ignored_experiments = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--", "experiments"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    allowed = {path.as_posix() for path in artifacts}
    plan_deletions = {path for path in allowed if path.startswith("plans/onboard-") and not (workspace / path).exists()}
    unexpected = (set(tracked_changes) | set(untracked) | set(ignored_experiments)) - allowed
    if unexpected:
        raise ValueError(f"Agent changed paths outside its artifact manifest: {sorted(unexpected)}")

    if plan_deletions:
        subprocess.run(["git", "add", "-u", "--", *sorted(plan_deletions)], cwd=workspace, check=True)
    regular = [str(path) for path in artifacts if path.parts and path.parts[0] not in {"experiments", "plans"}]
    if regular:
        subprocess.run(["git", "add", "--", *regular], cwd=workspace, check=True)
    experiments = [str(path) for path in artifacts if path.parts and path.parts[0] == "experiments"]
    if experiments:
        subprocess.run(["git", "add", "--force", "--", *experiments], cwd=workspace, check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--stage", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        summary, artifacts = validate_summary(
            args.summary,
            args.workspace.resolve(),
            args.model_id,
            args.gpu,
            args.gpu_count,
            args.ssh_target,
        )
        if args.stage:
            stage_artifacts(args.workspace.resolve(), artifacts)
        print(json.dumps(summary, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
