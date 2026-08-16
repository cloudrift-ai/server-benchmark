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
        "experiments/Model/serving/recipe.yaml",
        "experiments/Model/serving/RESULTS.md",
        "experiments/Model/serving/results_h200x1.tar.gz",
        "experiments/Model/serving/h200x1_np8_deadbeef.experiment.yaml",
    ]
    for raw_path in paths:
        path = workspace / raw_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if raw_path == paths[0]:
            path.write_text("tags: [best-effort]\nmodel:\n  huggingface: org/Model\n  heat: 77\n")
        elif path.suffixes == [".tar", ".gz"]:
            path.write_bytes(b"compressed results")
        else:
            path.write_text("result\n")
    return paths


def _init_repo(workspace):
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=workspace, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=workspace, check=True)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "non-empty deployment_summary"),
        ("", "non-empty deployment_summary"),
        ("line one\nline two", "one line"),
        ("x" * 1001, "at most 1000"),
    ],
)
def test_summary_text_requires_compact_notification_evidence(value, message):
    with pytest.raises(ValueError, match=message):
        onboarding_artifacts._summary_text({"deployment_summary": value}, "deployment_summary")


def test_validate_summary_accepts_exact_manifest(tmp_path):
    paths = _write_artifacts(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "mode": "onboarding",
                "model_id": "org/Model",
                "target": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1, "ssh": "user@host"},
                "deployment_summary": "vLLM 0.22.1, 32K context, concurrency 8",
                "performance_summary": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, 0 failures",
                "recipe": paths[0],
                "report": paths[1],
                "experiment": paths[2],
                "experiment_artifacts": paths[2:],
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
        "onboarding",
        "best-effort",
        77,
    )

    assert artifacts == [Path(path) for path in paths]


def test_validate_summary_includes_separately_declared_artifacts(tmp_path):
    paths = _write_artifacts(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "mode": "onboarding",
                "model_id": "org/Model",
                "target": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1, "ssh": "user@host"},
                "deployment_summary": "vLLM 0.22.1, 32K context, concurrency 8",
                "performance_summary": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, 0 failures",
                "recipe": paths[0],
                "report": paths[1],
                "experiment": paths[2],
                "experiment_artifacts": paths[2:],
                "artifacts": [paths[0]],
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
        "onboarding",
        "best-effort",
    )

    assert artifacts == [Path(path) for path in paths]


def test_validate_summary_requires_exact_platform_archive(tmp_path):
    paths = _write_artifacts(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "mode": "onboarding",
                "model_id": "org/Model",
                "target": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1, "ssh": "user@host"},
                "deployment_summary": "vLLM 0.22.1, 32K context, concurrency 8",
                "performance_summary": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, 0 failures",
                "recipe": paths[0],
                "report": paths[1],
                "experiment": paths[2],
                "experiment_artifacts": [paths[2], paths[3], paths[5]],
                "artifacts": paths,
                "cleanup": {"workloads": "complete", "docker_logout": True},
            }
        )
    )

    with pytest.raises(ValueError, match="exact platform archive"):
        onboarding_artifacts.validate_summary(
            summary_path,
            tmp_path,
            "org/Model",
            "NVIDIA H200 141GB",
            1,
            "user@host",
            "onboarding",
            "best-effort",
        )


def test_validate_summary_rejects_other_platform_archive(tmp_path):
    paths = _write_artifacts(tmp_path)
    other_archive = tmp_path / "experiments" / "Model" / "serving" / "results_rtx4090x1.tar.gz"
    other_archive.write_bytes(b"other platform")
    other_path = str(other_archive.relative_to(tmp_path))
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "mode": "onboarding",
                "model_id": "org/Model",
                "target": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1, "ssh": "user@host"},
                "deployment_summary": "vLLM 0.22.1, 32K context, concurrency 8",
                "performance_summary": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, 0 failures",
                "recipe": paths[0],
                "report": paths[1],
                "experiment": paths[2],
                "experiment_artifacts": paths[2:],
                "artifacts": [*paths, other_path],
                "cleanup": {"workloads": "complete", "docker_logout": True},
            }
        )
    )

    with pytest.raises(ValueError, match="other platform results must be preserved"):
        onboarding_artifacts.validate_summary(
            summary_path,
            tmp_path,
            "org/Model",
            "NVIDIA H200 141GB",
            1,
            "user@host",
            "onboarding",
            "best-effort",
        )


def test_validate_summary_rejects_experiment_result(tmp_path):
    paths = _write_artifacts(tmp_path)
    raw_result = tmp_path / "experiments" / "Model" / "serving" / "2026-08-08" / "result.json"
    raw_result.parent.mkdir(parents=True)
    raw_result.write_text("result\n")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "mode": "onboarding",
                "model_id": "org/Model",
                "target": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1, "ssh": "user@host"},
                "deployment_summary": "vLLM 0.22.1, 32K context, concurrency 8",
                "performance_summary": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, 0 failures",
                "recipe": paths[0],
                "report": paths[1],
                "experiment": paths[2],
                "experiment_artifacts": [*paths[2:], str(raw_result.relative_to(tmp_path))],
                "artifacts": [*paths, str(raw_result.relative_to(tmp_path))],
                "cleanup": {"workloads": "complete", "docker_logout": True},
            }
        )
    )

    with pytest.raises(ValueError, match="durable experiment snapshot"):
        onboarding_artifacts.validate_summary(
            summary_path,
            tmp_path,
            "org/Model",
            "NVIDIA H200 141GB",
            1,
            "user@host",
            "onboarding",
            "best-effort",
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
                "mode": "onboarding",
                "model_id": "org/Model",
                "target": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1, "ssh": "user@host"},
                "deployment_summary": "vLLM 0.22.1, 32K context, concurrency 8",
                "performance_summary": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, 0 failures",
                "recipe": paths[0],
                "report": paths[1],
                "experiment": paths[2],
                "experiment_artifacts": paths[2:],
                "artifacts": [*paths, str(raw_result.relative_to(tmp_path))],
                "cleanup": {"workloads": "complete", "docker_logout": True},
            }
        )
    )

    with pytest.raises(ValueError, match="Only durable recipe and experiment artifacts"):
        onboarding_artifacts.validate_summary(
            summary_path,
            tmp_path,
            "org/Model",
            "NVIDIA H200 141GB",
            1,
            "user@host",
            "onboarding",
            "best-effort",
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
                "mode": "onboarding",
                "model_id": "org/Model",
                "target": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1, "ssh": "user@host"},
                "deployment_summary": "vLLM 0.22.1, 32K context, concurrency 8",
                "performance_summary": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, 0 failures",
                "recipe": paths[0],
                "report": str(nested_report.relative_to(tmp_path)),
                "experiment": paths[2],
                "experiment_artifacts": paths[2:],
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
            "onboarding",
            "best-effort",
        )


def test_validate_summary_rejects_mode_mismatch(tmp_path):
    paths = _write_artifacts(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "mode": "onboarding",
                "model_id": "org/Model",
                "target": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1, "ssh": "user@host"},
                "deployment_summary": "vLLM 0.22.1, 32K context, concurrency 8",
                "performance_summary": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, 0 failures",
                "recipe": paths[0],
                "report": paths[1],
                "experiment": paths[2],
                "experiment_artifacts": paths[2:],
                "artifacts": paths,
                "cleanup": {"workloads": "complete", "docker_logout": True},
            }
        )
    )

    with pytest.raises(ValueError, match="Summary mode mismatch"):
        onboarding_artifacts.validate_summary(
            summary_path,
            tmp_path,
            "org/Model",
            "NVIDIA H200 141GB",
            1,
            "user@host",
            "verification",
            "best-effort",
        )


def test_validate_summary_rejects_lifecycle_change(tmp_path):
    paths = _write_artifacts(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "mode": "verification",
                "model_id": "org/Model",
                "target": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1, "ssh": "user@host"},
                "deployment_summary": "vLLM 0.22.1, 32K context, concurrency 8",
                "performance_summary": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, 0 failures",
                "recipe": paths[0],
                "report": paths[1],
                "experiment": paths[2],
                "experiment_artifacts": paths[2:],
                "artifacts": paths,
                "cleanup": {"workloads": "complete", "docker_logout": True},
            }
        )
    )

    with pytest.raises(ValueError, match="retain lifecycle tag 'maintained'"):
        onboarding_artifacts.validate_summary(
            summary_path,
            tmp_path,
            "org/Model",
            "NVIDIA H200 141GB",
            1,
            "user@host",
            "verification",
            "maintained",
        )


def test_stage_artifacts_rejects_unmanifested_agent_changes(tmp_path):
    _init_repo(tmp_path)
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
    _init_repo(tmp_path)
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


def test_stage_artifacts_requires_updated_platform_archive(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / ".gitattributes").write_text("experiments/**/results_*.tar.gz filter=lfs diff=lfs merge=lfs -text\n")
    paths = _write_artifacts(tmp_path)
    subprocess.run(["git", "add", "--force", "--", ".gitattributes", *paths], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)
    (tmp_path / paths[3]).write_text("updated report\n")

    with pytest.raises(ValueError, match="was not created or updated"):
        onboarding_artifacts.stage_artifacts(
            tmp_path,
            [Path(path) for path in paths],
            Path(paths[4]),
        )


def test_stage_artifacts_requires_git_lfs_for_platform_archive(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("baseline\n")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)
    paths = _write_artifacts(tmp_path)

    with pytest.raises(ValueError, match="not tracked by Git LFS"):
        onboarding_artifacts.stage_artifacts(
            tmp_path,
            [Path(path) for path in paths],
            Path(paths[4]),
        )


def test_platform_update_preserves_other_platform_snapshot(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / ".gitattributes").write_text("experiments/**/results_*.tar.gz filter=lfs diff=lfs merge=lfs -text\n")
    paths = _write_artifacts(tmp_path)
    experiment_dir = tmp_path / "experiments" / "Model" / "serving"
    other_archive = experiment_dir / "results_rtx4090x1.tar.gz"
    other_record = experiment_dir / "rtx4090x1_np8_feedface.experiment.yaml"
    other_archive.write_bytes(b"preserve archive")
    other_record.write_text("preserve record\n")
    old_record = experiment_dir / "h200x1_np4_cafebabe.experiment.yaml"
    old_record.write_text("old current-platform record\n")
    baseline = [
        ".gitattributes",
        *paths,
        str(other_archive.relative_to(tmp_path)),
        str(other_record.relative_to(tmp_path)),
        str(old_record.relative_to(tmp_path)),
    ]
    subprocess.run(["git", "add", "--force", "--", *baseline], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)

    (tmp_path / paths[3]).write_text("H200 updated; RTX 4090 preserved\n")
    (tmp_path / paths[4]).write_bytes(b"new H200 archive")
    (tmp_path / paths[5]).write_text("new H200 record\n")
    old_record.unlink()
    old_record_path = str(old_record.relative_to(tmp_path))
    summary_path = tmp_path.parent / f"{tmp_path.name}-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "mode": "onboarding",
                "model_id": "org/Model",
                "target": {"gpu": "NVIDIA H200 141GB", "gpu_count": 1, "ssh": "user@host"},
                "deployment_summary": "vLLM 0.22.1, 32K context, concurrency 8",
                "performance_summary": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, 0 failures",
                "recipe": paths[0],
                "report": paths[1],
                "experiment": paths[2],
                "experiment_artifacts": paths[2:],
                "artifacts": [*paths, old_record_path],
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
        "onboarding",
        "best-effort",
    )
    onboarding_artifacts.stage_artifacts(tmp_path, artifacts, Path(paths[4]))

    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert str(other_archive.relative_to(tmp_path)) not in staged
    assert str(other_record.relative_to(tmp_path)) not in staged
    assert other_archive.read_bytes() == b"preserve archive"
    assert other_record.read_text() == "preserve record\n"
    assert old_record_path in staged
