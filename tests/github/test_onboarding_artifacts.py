import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / ".github" / "scripts" / "onboarding_artifacts.py"
SPEC = importlib.util.spec_from_file_location("onboarding_artifacts", MODULE_PATH)
onboarding_artifacts = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(onboarding_artifacts)


def _write_artifacts(workspace):
    paths = [
        "recipes/Model/recipe.yaml",
        "recipes/Model/RESULTS.md",
        "experiments/Model/serving_h200/recipe.yaml",
    ]
    for raw_path in paths:
        path = workspace / raw_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("result\n")
    return paths


def test_validate_summary_accepts_exact_manifest(tmp_path):
    paths = _write_artifacts(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "model_id": "org/Model",
                "target": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1, "ssh": "user@host"},
                "recipe": paths[0],
                "report": paths[1],
                "experiment": paths[2],
                "experiment_artifacts": [paths[2]],
                "artifacts": paths,
                "cleanup": {"workloads": "complete", "docker_logout": True},
            }
        )
    )

    _, artifacts = onboarding_artifacts.validate_summary(
        summary_path,
        tmp_path,
        "org/Model",
        "NVIDIA H200 141GB",
        1,
        "user@host",
    )

    assert artifacts == [Path(path) for path in paths]


def test_validate_summary_rejects_experiment_result(tmp_path):
    paths = _write_artifacts(tmp_path)
    raw_result = tmp_path / "experiments" / "Model" / "serving_h200" / "2026-08-08" / "result.json"
    raw_result.parent.mkdir(parents=True)
    raw_result.write_text("result\n")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "model_id": "org/Model",
                "target": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1, "ssh": "user@host"},
                "recipe": paths[0],
                "report": paths[1],
                "experiment": paths[2],
                "experiment_artifacts": [paths[2], str(raw_result.relative_to(tmp_path))],
                "artifacts": [*paths, str(raw_result.relative_to(tmp_path))],
                "cleanup": {"workloads": "complete", "docker_logout": True},
            }
        )
    )

    with pytest.raises(ValueError, match="Only experiment recipe.yaml"):
        onboarding_artifacts.validate_summary(
            summary_path,
            tmp_path,
            "org/Model",
            "NVIDIA H200 141GB",
            1,
            "user@host",
        )


def test_validate_summary_rejects_recipe_run_result(tmp_path):
    paths = _write_artifacts(tmp_path)
    raw_result = tmp_path / "recipes" / "Model" / "2026-08-08" / "result.json"
    raw_result.parent.mkdir(parents=True)
    raw_result.write_text("result\n")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "model_id": "org/Model",
                "target": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1, "ssh": "user@host"},
                "recipe": paths[0],
                "report": paths[1],
                "experiment": paths[2],
                "experiment_artifacts": [paths[2]],
                "artifacts": [*paths, str(raw_result.relative_to(tmp_path))],
                "cleanup": {"workloads": "complete", "docker_logout": True},
            }
        )
    )

    with pytest.raises(ValueError, match="Only recipe.yaml and final recipe RESULTS.md"):
        onboarding_artifacts.validate_summary(
            summary_path,
            tmp_path,
            "org/Model",
            "NVIDIA H200 141GB",
            1,
            "user@host",
        )


def test_validate_summary_rejects_report_outside_final_recipe_dir(tmp_path):
    paths = _write_artifacts(tmp_path)
    nested_report = tmp_path / "recipes" / "Model" / "experiment" / "RESULTS.md"
    nested_report.parent.mkdir(parents=True)
    nested_report.write_text("result\n")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "model_id": "org/Model",
                "target": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1, "ssh": "user@host"},
                "recipe": paths[0],
                "report": str(nested_report.relative_to(tmp_path)),
                "experiment": paths[2],
                "experiment_artifacts": [paths[2]],
                "artifacts": [*paths, str(nested_report.relative_to(tmp_path))],
                "cleanup": {"workloads": "complete", "docker_logout": True},
            }
        )
    )

    with pytest.raises(ValueError, match="beside the final recipe"):
        onboarding_artifacts.validate_summary(
            summary_path,
            tmp_path,
            "org/Model",
            "NVIDIA H200 141GB",
            1,
            "user@host",
        )


def test_stage_artifacts_rejects_unmanifested_agent_changes(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    baseline = tmp_path / "README.md"
    baseline.write_text("baseline\n")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)
    recipe = tmp_path / "recipes" / "Model" / "recipe.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("model: {}\n")
    (tmp_path / "credential.txt").write_text("must not be staged\n")

    with pytest.raises(ValueError, match="outside its artifact manifest"):
        onboarding_artifacts.stage_artifacts(tmp_path, [Path("recipes/Model/recipe.yaml")])


def test_stage_artifacts_rejects_unmanifested_ignored_experiment(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("experiments/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)
    recipe = tmp_path / "experiments" / "Model" / "recipe.yaml"
    extra = tmp_path / "experiments" / "Model" / "scratch.log"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("model: {}\n")
    extra.write_text("exploratory output\n")

    with pytest.raises(ValueError, match="outside its artifact manifest"):
        onboarding_artifacts.stage_artifacts(tmp_path, [Path("experiments/Model/recipe.yaml")])
