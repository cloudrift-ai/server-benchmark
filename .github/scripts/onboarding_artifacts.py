#!/usr/bin/env python3
"""Validate and stage the artifact manifest returned by the onboard-model skill."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

import yaml

from emmy.hardware import gpu_short_name
from emmy.recipe.catalog import validate_model_heat
from emmy.recipe.lifecycle import validate_recipe_tags

ALLOWED_ARTIFACT_PREFIXES = (
    "docker/vllm-emmy-serve/models/",
    "emmy/compiler/pipeline/search/goldens/",
    "emmy/",
    "experiments/",
    "recipes/",
    "tests/",
)
SUMMARY_TEXT_LIMIT = 1000
MAX_IMPLEMENTATION_FILES = 8
MAX_IMPLEMENTATION_CHANGED_LINES = 500


def _summary_text(summary: dict, field: str) -> str:
    value = summary.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Summary must include a non-empty {field}")
    if "\n" in value or "\r" in value or len(value) > SUMMARY_TEXT_LIMIT:
        raise ValueError(f"Summary {field} must be one line of at most {SUMMARY_TEXT_LIMIT} characters")
    return value


def _relative_file(workspace: Path, raw_path: str, prefixes: tuple[str, ...]) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts or not path.as_posix().startswith(prefixes):
        raise ValueError(f"Artifact path is outside the allowed onboarding areas: {raw_path}")
    resolved = workspace / path
    if not resolved.is_file():
        raise ValueError(f"Artifact does not exist: {raw_path}")
    return path


def _platform_name(gpu: str, gpu_count: int) -> str:
    return f"{gpu_short_name(gpu)}x{gpu_count}"


def _is_platform_record(path: Path, platform: str) -> bool:
    exact_name = path.name == f"{platform}.experiment.yaml"
    expanded_name = path.name.startswith(f"{platform}_") and path.name.endswith(".experiment.yaml")
    return exact_name or expanded_name


def _is_durable_experiment_artifact(path: Path) -> bool:
    named_archive = path.name.startswith("results_") and path.name.endswith(".tar.gz")
    return path.name in {"recipe.yaml", "RESULTS.md"} or named_archive


def _relative_experiment_artifact(workspace: Path, raw_path: str, experiment_dir: Path, platform: str) -> Path:
    path = _relative_file(workspace, raw_path, ("experiments/",))
    allowed_names = {"recipe.yaml", "RESULTS.md", f"results_{platform}.tar.gz"}
    if path.parent != experiment_dir or path.name not in allowed_names:
        raise ValueError(f"Artifact is outside the {platform} durable experiment snapshot: {raw_path}")
    return path


def _is_tracked_deletion(workspace: Path, path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--deleted", "--", str(path)],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and path.as_posix() in result.stdout.splitlines()


def _relative_artifact(workspace: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    normalized = path.as_posix()
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Artifact path is not repository-relative: {raw_path}")
    is_golden = normalized.startswith("emmy/compiler/pipeline/search/goldens/")
    allowed_implementation_file = (
        is_golden or path.parts[0] not in {"emmy", "tests"} or path.suffix == ".py" or path.name == "ARCHITECTURE.md"
    )
    if (
        normalized.startswith(ALLOWED_ARTIFACT_PREFIXES)
        and allowed_implementation_file
        and ((workspace / path).is_file() or _is_tracked_deletion(workspace, path))
    ):
        return path
    raise ValueError(f"Artifact path is outside the allowed onboarding areas or does not exist: {raw_path}")


def _invalid_result_artifacts(workspace: Path, artifacts: list[Path]) -> list[Path]:
    return [
        path
        for path in artifacts
        if (
            path.parts[0] == "experiments"
            and not _is_durable_experiment_artifact(path)
            and not (path.name.endswith(".experiment.yaml") and _is_tracked_deletion(workspace, path))
        )
        or (path.parts[0] == "recipes" and path.name not in {"recipe.yaml", "RESULTS.md"})
    ]


def _archive_platform_records(workspace: Path, archive: Path, platform: str) -> list[PurePosixPath]:
    records = []
    with tarfile.open(workspace / archive, "r:gz") as source:
        for member in source.getmembers():
            member_path = PurePosixPath(member.name.removeprefix("./"))
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe member in exact platform archive: {member.name}")
            if not member.isfile() or not _is_platform_record(Path(member_path.name), platform):
                continue
            contents = source.extractfile(member)
            if contents is None or not isinstance(yaml.safe_load(contents), dict):
                raise ValueError(f"Invalid experiment record in exact platform archive: {member.name}")
            records.append(member_path)
    if not records:
        raise ValueError(f"Exact platform archive contains no {platform} experiment records: {archive}")
    return records


def _is_implementation_artifact(path: Path) -> bool:
    return (
        bool(path.parts) and path.parts[0] in {"emmy", "tests"} and not path.as_posix().startswith("emmy/compiler/pipeline/search/goldens/")
    )


def _validate_implementation_patch(workspace: Path, changed: set[str]) -> None:
    implementation = sorted(Path(path) for path in changed if _is_implementation_artifact(Path(path)))
    if len(implementation) > MAX_IMPLEMENTATION_FILES:
        raise ValueError(f"Onboarding small fix changes too many implementation files: {len(implementation)}")
    if not implementation:
        return
    source_changes = [path for path in implementation if path.parts[0] == "emmy" and path.suffix == ".py"]
    test_changes = [path for path in implementation if path.parts[0] == "tests" and path.suffix == ".py"]
    if source_changes and not test_changes:
        raise ValueError("Onboarding small fix must include a focused test change")

    result = subprocess.run(
        ["git", "diff", "--numstat", "--no-renames", "HEAD", "--", *(str(path) for path in implementation)],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    changed_lines = 0
    counted = set()
    for line in result.stdout.splitlines():
        added, deleted, raw_path = line.split("\t", 2)
        if added == "-" or deleted == "-":
            raise ValueError(f"Onboarding small fix must not add binary implementation artifacts: {raw_path}")
        changed_lines += int(added) + int(deleted)
        counted.add(raw_path)
    for path in implementation:
        if path.as_posix() in counted or not (workspace / path).is_file():
            continue
        contents = (workspace / path).read_bytes()
        if b"\0" in contents:
            raise ValueError(f"Onboarding small fix must not add binary implementation artifacts: {path}")
        changed_lines += len(contents.splitlines())
    if changed_lines > MAX_IMPLEMENTATION_CHANGED_LINES:
        raise ValueError(f"Onboarding small fix changes too many implementation lines: {changed_lines}")


def validate_summary(
    summary_path: Path,
    workspace: Path,
    model_id: str,
    gpu: str,
    gpu_count: int,
    ssh_target: str,
    mode: str,
    expected_tag: str,
    expected_heat: int | None = None,
) -> tuple[dict, list[Path]]:
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "success":
        raise ValueError(f"Onboarding did not succeed: {summary.get('failure')}")
    if summary.get("model_id") != model_id:
        raise ValueError(f"Summary model mismatch: {summary.get('model_id')} != {model_id}")
    if summary.get("mode") != mode:
        raise ValueError(f"Summary mode mismatch: {summary.get('mode')} != {mode}")
    target = summary.get("target") or {}
    expected_target = {"gpu": gpu, "gpu_count": gpu_count, "ssh": ssh_target}
    if target != expected_target:
        raise ValueError(f"Summary target mismatch: {target} != {expected_target}")
    _summary_text(summary, "deployment_summary")
    _summary_text(summary, "performance_summary")
    cleanup = summary.get("cleanup") or {}
    if cleanup.get("workloads") != "complete" or cleanup.get("docker_logout") is not True:
        raise ValueError(f"Remote workload or Docker credential cleanup is incomplete: {cleanup}")

    recipe = _relative_file(workspace, summary.get("recipe") or "", ("recipes/",))
    recipe_config = yaml.safe_load((workspace / recipe).read_text()) or {}
    recipe_model = recipe_config.get("model") or {}
    recipe_model_id = recipe_model.get("huggingface")
    if recipe_model_id != model_id:
        raise ValueError(f"Recipe model mismatch: {recipe_model_id} != {model_id}")
    if expected_heat is not None:
        validate_model_heat(expected_heat, model_id, required=True)
        if recipe_model.get("heat") != expected_heat:
            raise ValueError(f"Recipe must preserve model heat {expected_heat}: {recipe_model.get('heat')!r}")
    recipe_tags = validate_recipe_tags(recipe_config.get("tags"))
    if expected_tag not in recipe_tags:
        raise ValueError(f"Recipe must retain lifecycle tag {expected_tag!r}: {recipe_tags}")
    if mode == "onboarding" and ({"onboarding", "untested"} & set(recipe_tags)):
        raise ValueError(f"Onboarding recipe still has pending lifecycle tags: {recipe_tags}")
    report = _relative_file(workspace, summary.get("report") or "", ("recipes/",))
    if report != recipe.with_name("RESULTS.md"):
        raise ValueError(f"Report must be RESULTS.md beside the final recipe: {report}")
    experiment_recipe = _relative_file(workspace, summary.get("experiment") or "", ("experiments/",))
    if experiment_recipe.name != "recipe.yaml":
        raise ValueError(f"Experiment must identify its recipe.yaml: {experiment_recipe}")
    experiment_dir = experiment_recipe.parent
    platform = _platform_name(gpu, gpu_count)
    raw_experiment_artifacts = summary.get("experiment_artifacts")
    if not isinstance(raw_experiment_artifacts, list) or not raw_experiment_artifacts:
        raise ValueError("Summary must list the retained durable experiment snapshot in experiment_artifacts")
    experiment_artifacts = [
        _relative_experiment_artifact(workspace, raw_path, experiment_dir, platform) for raw_path in raw_experiment_artifacts
    ]
    if experiment_recipe not in experiment_artifacts:
        raise ValueError("The experiment recipe must be included in experiment_artifacts")
    experiment_report = experiment_dir / "RESULTS.md"
    if experiment_report not in experiment_artifacts:
        raise ValueError("The shared experiment RESULTS.md must be included in experiment_artifacts")
    experiment_archive = experiment_dir / f"results_{platform}.tar.gz"
    if experiment_archive not in experiment_artifacts:
        raise ValueError(f"The exact platform archive must be included in experiment_artifacts: {experiment_archive}")
    _archive_platform_records(workspace, experiment_archive, platform)
    raw_artifacts = summary.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("Summary must include a non-empty artifacts list")
    artifacts = list(
        dict.fromkeys(
            [
                *(_relative_artifact(workspace, raw_path) for raw_path in raw_artifacts),
                recipe,
                report,
                *experiment_artifacts,
            ]
        )
    )
    invalid_result_artifacts = _invalid_result_artifacts(workspace, artifacts)
    if invalid_result_artifacts:
        raise ValueError(f"Only durable recipe and experiment artifacts may be retained: {invalid_result_artifacts}")
    invalid_experiment_artifacts = [
        path
        for path in artifacts
        if path.parts[0] == "experiments"
        and not (
            path.parent == experiment_dir
            and (
                path.name in {"recipe.yaml", "RESULTS.md", experiment_archive.name}
                or (_is_platform_record(path, platform) and _is_tracked_deletion(workspace, path))
            )
        )
    ]
    if invalid_experiment_artifacts:
        raise ValueError(
            f"Only the {platform} snapshot may change; other platform results must be preserved: {invalid_experiment_artifacts}"
        )
    return summary, artifacts


def stage_artifacts(workspace: Path, artifacts: list[Path], required_archive: Path | None = None) -> None:
    invalid_result_artifacts = _invalid_result_artifacts(workspace, artifacts)
    if invalid_result_artifacts:
        raise ValueError(f"Only durable recipe and experiment artifacts may be retained: {invalid_result_artifacts}")
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
    unexpected = (set(tracked_changes) | set(untracked) | set(ignored_experiments)) - allowed
    if unexpected:
        raise ValueError(f"Agent changed paths outside its artifact manifest: {sorted(unexpected)}")
    changed = set(tracked_changes) | set(untracked) | set(ignored_experiments)
    _validate_implementation_patch(workspace, changed)
    if required_archive is not None and required_archive.as_posix() not in changed:
        raise ValueError(f"Exact platform results archive was not created or updated: {required_archive}")
    if required_archive is not None:
        attribute = subprocess.run(
            ["git", "check-attr", "filter", "--", str(required_archive)],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if attribute != f"{required_archive}: filter: lfs":
            raise ValueError(f"Exact platform results archive is not tracked by Git LFS: {required_archive}")

    regular = [str(path) for path in artifacts if path.parts and path.parts[0] != "experiments"]
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
    parser.add_argument("--mode", choices=("onboarding", "verification"), required=True)
    parser.add_argument("--expected-tag", choices=("maintained", "best-effort"), required=True)
    parser.add_argument("--expected-heat", type=int)
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
            args.mode,
            args.expected_tag,
            args.expected_heat,
        )
        if args.stage:
            archive_name = f"results_{_platform_name(args.gpu, args.gpu_count)}.tar.gz"
            archive = next(path for path in artifacts if path.parts[0] == "experiments" and path.name == archive_name)
            stage_artifacts(args.workspace.resolve(), artifacts, archive)
        print(json.dumps(summary, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
