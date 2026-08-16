import importlib.util
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from emmy.recipe.catalog import MAX_STUB_DEPLOYMENTS

MODULE_PATH = Path(__file__).parents[2] / ".github" / "scripts" / "discovery_lifecycle.py"
SPEC = importlib.util.spec_from_file_location("discovery_lifecycle", MODULE_PATH)
discovery_lifecycle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(discovery_lifecycle)

GPU = "NVIDIA H200 141GB"


@pytest.mark.parametrize(
    ("workflow", "message"),
    [
        ("discover-model.yml", '"Complete the attached lifecycle task exactly."'),
        ("onboard-model.yml", '"Complete the attached onboarding task exactly."'),
    ],
)
def test_opencode_message_precedes_variadic_file_option(workflow, message):
    document = yaml.safe_load((Path(__file__).parents[2] / ".github" / "workflows" / workflow).read_text())
    scripts = [step.get("run", "") for job in document["jobs"].values() for step in job["steps"]]
    script = next(script for script in scripts if "opencode run" in script)

    assert script.index(message) < script.index('--file "$AGENT_PROMPT"')


def test_onboarding_loads_control_code_from_exact_workflow_commit():
    document = yaml.safe_load((Path(__file__).parents[2] / ".github" / "workflows" / "onboard-model.yml").read_text())
    steps = document["jobs"]["onboard"]["steps"]
    load_index = next(index for index, step in enumerate(steps) if step.get("name") == "Load exact workflow source")
    lfs_index = next(index for index, step in enumerate(steps) if step.get("name") == "Configure Git LFS")
    selection_index = next(index for index, step in enumerate(steps) if step.get("name") == "Select one available deployment")
    load_script = steps[load_index]["run"]
    validation_script = next(step["run"] for step in steps if step.get("name") == "Validate and stage model artifacts")
    cleanup_script = next(step["run"] for step in steps if step.get("name") == "Cleanup local credentials and output")

    assert load_index < lfs_index < selection_index
    assert 'git archive "$WORKFLOW_SHA"' in load_script
    assert "GIT_LFS_SKIP_SMUDGE" in steps[load_index]["env"]
    assert 'echo "PYTHONPATH=$WORKFLOW_SOURCE" >> "$GITHUB_ENV"' in load_script
    assert 'echo "PYTHONSAFEPATH=1" >> "$GITHUB_ENV"' in load_script
    assert '"$WORKFLOW_SOURCE/.github/scripts/onboarding_artifacts.py"' in validation_script
    assert 'rm -rf -- "$WORKFLOW_SOURCE"' in cleanup_script


@pytest.mark.parametrize(
    ("workflow", "primary_job", "workflow_kind", "notification_name"),
    [
        ("discover-model.yml", "discover", "discover", "Send discovery summary"),
        ("onboard-model.yml", "onboard", "onboard", "Send onboarding summary"),
    ],
)
def test_model_lifecycle_workflow_posts_discord_summary_from_separate_job(workflow, primary_job, workflow_kind, notification_name):
    document = yaml.safe_load((Path(__file__).parents[2] / ".github" / "workflows" / workflow).read_text())
    lifecycle = document["jobs"][primary_job]
    notify = document["jobs"]["notify"]
    lifecycle_pr = next(step for step in lifecycle["steps"] if step.get("id") == "lifecycle-pr")
    checkout, notification = notify["steps"]

    assert notify["needs"] == primary_job
    assert notify["if"] == "${{ always() }}"
    assert notify["runs-on"] == "ubuntu-latest"
    assert notify["permissions"] == {"contents": "read"}
    assert notify["env"]["DISCORD_WEBHOOK_URL"] == "${{ secrets.DISCORD_EMMY_ROBOTS_WEBHOOK_URL }}"
    assert notify["env"]["WORKFLOW_KIND"] == workflow_kind
    assert notify["env"]["WORKFLOW_RESULT"] == f"${{{{ needs.{primary_job}.result }}}}"
    assert checkout["uses"] == "actions/checkout@v4"
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    assert checkout["with"]["persist-credentials"] is False
    assert notification["name"] == notification_name
    assert notification["continue-on-error"] is True
    assert notification["run"] == "python3 .github/scripts/discord_notification.py"
    assert 'echo "number=$PR_NUMBER" >> "$GITHUB_OUTPUT"' in lifecycle_pr["run"]
    assert lifecycle["outputs"]["pr_number"] == "${{ steps.lifecycle-pr.outputs.number || steps.rolling.outputs.number }}"
    if workflow_kind == "discover":
        assert lifecycle["outputs"]["modified_models"] == "${{ steps.lifecycle.outputs.modified_models }}"
        assert notify["env"]["MODIFIED_MODELS"] == "${{ needs.discover.outputs.modified_models }}"
    else:
        artifacts = next(step for step in lifecycle["steps"] if step.get("id") == "artifacts")
        assert lifecycle["outputs"]["deployment_summary"] == "${{ steps.artifacts.outputs.deployment_summary }}"
        assert lifecycle["outputs"]["performance_summary"] == "${{ steps.artifacts.outputs.performance_summary }}"
        assert notify["env"]["DEPLOYMENT_SUMMARY"] == "${{ needs.onboard.outputs.deployment_summary }}"
        assert notify["env"]["PERFORMANCE_SUMMARY"] == "${{ needs.onboard.outputs.performance_summary }}"
        assert "deployment_summary=$(jq -r .deployment_summary" in artifacts["run"]
        assert "performance_summary=$(jq -r .performance_summary" in artifacts["run"]
    assert "DISCORD_EMMY_ROBOTS_ALERT_ROLE_ID" not in (Path(__file__).parents[2] / ".github" / "workflows" / workflow).read_text()


def test_onboarding_requires_platform_results_snapshot_and_git_lfs():
    workspace = Path(__file__).parents[2]
    document = yaml.safe_load((workspace / ".github" / "workflows" / "onboard-model.yml").read_text())
    steps = document["jobs"]["onboard"]["steps"]
    lfs_script = next(step["run"] for step in steps if step.get("name") == "Configure Git LFS")
    host_setup_script = next(step["run"] for step in steps if step.get("name") == "Prepare target GPU host")
    agent_script = next(step["run"] for step in steps if step.get("name") == "Run onboard-model agent")
    cleanup_script = next(step["run"] for step in steps if step.get("name") == "Remove archived task-local raw results")
    validation_script = next(step["run"] for step in steps if step.get("name") == "Validate and stage model artifacts")

    assert "lfs_version=3.7.1" in lfs_script
    assert "sha256sum --check" in lfs_script
    assert 'echo "$lfs_dir" >> "$GITHUB_PATH"' in lfs_script
    assert "git lfs install --local" in lfs_script
    assert "experiments/**/results_*.tar.gz filter=lfs" in lfs_script
    assert "python3.12-venv" in host_setup_script
    assert 'scratch_base="$HOME/.cache/emmy"' in host_setup_script
    assert '"$SSH_USER@$SSH_HOST"' in host_setup_script
    assert '"$SSH_TARGET"' not in host_setup_script
    assert "tmpfs|ramfs" in host_setup_script
    assert "8388608" in host_setup_script
    subprocess.run(["bash", "-n"], input=host_setup_script, text=True, check=True)
    assert "results_<gpu-short>x<gpu-count>.tar.gz" in agent_script
    assert "preserve every other platform archive" in agent_script
    assert '"$WORKFLOW_SOURCE/.agents/skills/onboard-model/SKILL.md"' in agent_script
    assert '"$WORKFLOW_SOURCE/.agents/skills/tune-kernels/SKILL.md"' in agent_script
    assert '"$WORKFLOW_SOURCE/.agents/skills/run-experiment/SKILL.md"' in agent_script
    assert "do not modify or list .gitattributes" in agent_script
    assert 'tarfile.open(temporary_archive, "w:gz")' in cleanup_script
    assert "temporary_roots = verify_archive(temporary_archive)" in cleanup_script
    assert "os.replace(temporary_archive, archive)" in cleanup_script
    assert 'tarfile.open(path, "r:gz")' in cleanup_script
    assert "contents.read(1024 * 1024)" in cleanup_script
    assert "raw_directory.name not in member_roots" in cleanup_script
    assert "shutil.rmtree(raw_directory)" in cleanup_script
    assert "git lfs status" in validation_script
    assert "experiments/**/results_*.tar.gz filter=lfs" in (workspace / ".gitattributes").read_text()


def test_onboarding_removes_only_raw_results_preserved_by_platform_archive(tmp_path):
    document = yaml.safe_load((Path(__file__).parents[2] / ".github" / "workflows" / "onboard-model.yml").read_text())
    step = next(step for step in document["jobs"]["onboard"]["steps"] if step.get("name") == "Remove archived task-local raw results")
    cleanup_source = step["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]

    experiment_dir = tmp_path / "experiments" / "Model" / "serving"
    raw_directory = experiment_dir / "2026-08-16_14-41-08"
    raw_directory.mkdir(parents=True)
    (raw_directory / "benchmark.log").write_text("measured\n")
    (experiment_dir / "recipe.yaml").write_text("name: serving\n")
    (experiment_dir / "RESULTS.md").write_text("# Results\n")
    (experiment_dir / "rtx4090x1_serving.experiment.yaml").write_text("status: succeeded\n")
    archive = experiment_dir / "results_rtx4090x1.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        root_info = tarfile.TarInfo(".")
        root_info.type = tarfile.DIRTYPE
        output.addfile(root_info)
        output.add(raw_directory, arcname=raw_directory.name)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "success",
                "experiment": "experiments/Model/serving/recipe.yaml",
                "experiment_artifacts": ["experiments/Model/serving/results_rtx4090x1.tar.gz"],
                "artifacts": [],
            }
        )
    )

    subprocess.run(
        [sys.executable, "-"],
        cwd=tmp_path,
        env={
            **os.environ,
            "ONBOARD_SUMMARY": str(summary),
            "TARGET_GPU": "NVIDIA GeForce RTX 4090",
            "TARGET_GPU_COUNT": "1",
        },
        input=cleanup_source,
        text=True,
        check=True,
    )

    assert not raw_directory.exists()
    assert archive.is_file()
    updated_summary = json.loads(summary.read_text())
    archive_path = "experiments/Model/serving/results_rtx4090x1.tar.gz"
    assert archive_path in updated_summary["experiment_artifacts"]
    assert archive_path in updated_summary["artifacts"]


def test_onboarding_creates_platform_archive_and_preserves_other_platform(tmp_path):
    document = yaml.safe_load((Path(__file__).parents[2] / ".github" / "workflows" / "onboard-model.yml").read_text())
    step = next(step for step in document["jobs"]["onboard"]["steps"] if step.get("name") == "Remove archived task-local raw results")
    cleanup_source = step["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]

    experiment_dir = tmp_path / "experiments" / "Model" / "serving"
    raw_directory = experiment_dir / "2026-08-16_14-41-08"
    raw_directory.mkdir(parents=True)
    (raw_directory / "benchmark.log").write_text("measured\n")
    (experiment_dir / "recipe.yaml").write_text("name: serving\n")
    (experiment_dir / "RESULTS.md").write_text("# Results\n")
    (experiment_dir / "rtx4090x1_serving.experiment.yaml").write_text("status: succeeded\n")
    archive = experiment_dir / "results_rtx4090x1.tar.gz"
    old_results = tmp_path / "old-results"
    old_results.mkdir()
    (old_results / "obsolete.log").write_text("old measurement\n")
    with tarfile.open(archive, "w:gz") as output:
        output.add(old_results, arcname="2026-08-01_00-00-00")
    other_archive = experiment_dir / "results_h200x1.tar.gz"
    other_archive.write_bytes(b"preserved platform")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "success",
                "experiment": "experiments/Model/serving/recipe.yaml",
                "experiment_artifacts": [
                    "experiments/Model/serving/recipe.yaml",
                    "experiments/Model/serving/RESULTS.md",
                ],
                "artifacts": [],
            }
        )
    )

    subprocess.run(
        [sys.executable, "-"],
        cwd=tmp_path,
        env={
            **os.environ,
            "ONBOARD_SUMMARY": str(summary),
            "TARGET_GPU": "NVIDIA GeForce RTX 4090",
            "TARGET_GPU_COUNT": "1",
        },
        input=cleanup_source,
        text=True,
        check=True,
    )

    assert archive.is_file()
    with tarfile.open(archive, "r:gz") as source:
        assert "2026-08-16_14-41-08/benchmark.log" in source.getnames()
        assert "2026-08-01_00-00-00/obsolete.log" not in source.getnames()
    assert not raw_directory.exists()
    assert other_archive.read_bytes() == b"preserved platform"
    updated_summary = json.loads(summary.read_text())
    expected_snapshot = {
        "experiments/Model/serving/recipe.yaml",
        "experiments/Model/serving/RESULTS.md",
        "experiments/Model/serving/results_rtx4090x1.tar.gz",
        "experiments/Model/serving/rtx4090x1_serving.experiment.yaml",
    }
    assert expected_snapshot <= set(updated_summary["experiment_artifacts"])
    assert expected_snapshot <= set(updated_summary["artifacts"])


def test_onboarding_selects_with_generic_recipe_query():
    document = yaml.safe_load((Path(__file__).parents[2] / ".github" / "workflows" / "onboard-model.yml").read_text())
    script = next(step["run"] for step in document["jobs"]["onboard"]["steps"] if step.get("name") == "Select one available deployment")

    assert "recipe query" in script
    assert "--root recipes" in script
    assert "provider.cloudrift.team_access == true" in script
    assert 'lifecycle in ["onboarding", "maintained"]' in script
    assert "deployment.availability.cloudrift == true" in script
    assert 'lifecycle order ["onboarding", "maintained"]' in script
    assert "results.last_run_at asc nulls-first" in script
    assert "deployment.index asc" in script
    assert "--allow-missing-model" in script
    assert 'lifecycle != "obsolete"' in script
    assert "recipe_inventory_document" not in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_discovery_prompt_keeps_obsolete_classification_conservative():
    document = yaml.safe_load((Path(__file__).parents[2] / ".github" / "workflows" / "discover-model.yml").read_text())
    script = next(step["run"] for step in document["jobs"]["discover"]["steps"] if step.get("name") == "Run discover-models agent")

    assert "A replacement that is merely comparable is not" in script
    assert "read both recipe files" in script
    assert "configured context, concurrency, quantization, hardware support, or model" in script
    assert '"Comparable reasoning"' in script
    assert "replacement_model_id is allowed only in obsolete_models" in script


def _recipe(workspace, name, model_id, tags=None, leading_comment=False, task=None, gpu=GPU, gpu_count=1):
    path = workspace / "recipes" / name / "recipe.yaml"
    path.parent.mkdir(parents=True)
    prefix = "# Keep this qualification note.\n" if leading_comment else ""
    tag_text = "" if tags is None else "tags:\n" + "".join(f"  - {tag}\n" for tag in tags)
    task_text = "" if task is None else f"  task: {task}\n"
    if tags and "onboarding" in tags:
        matrices = f"matrices:\n  - deploy.gpu: {gpu}\n    deploy.gpu_count: {gpu_count}\n"
    else:
        matrices = f"matrices:\n  deploy.gpu: {gpu}\n  deploy.gpu_count: {gpu_count}\n"
    path.write_text(f"{prefix}{tag_text}model:\n  huggingface: {model_id}\n{task_text}engine:\n  llm: {{}}\n{matrices}")
    return path


def _decision(model_id, rationale=None):
    return {"model_id": model_id, "rationale": rationale or f"Rationale for {model_id}."}


def _manifest(path, maintained, best_effort=None, obsolete=None, onboarding=None):
    def normalized(values):
        return [value if isinstance(value, dict) else _decision(value) for value in values or []]

    path.write_text(
        json.dumps(
            {
                "maintained_models": normalized(maintained),
                "best_effort_models": normalized(best_effort),
                "obsolete_models": obsolete or [],
                "onboarding_models": onboarding or [],
            }
        )
    )


def _candidate(model_id="org/new-model"):
    return {
        "model_id": model_id,
        "task": "generate",
        "rationale": "Strong current adoption and serving value.",
        "deployments": [
            {"deploy.gpu": GPU, "deploy.gpu_count": 1},
            {"deploy.gpu": "NVIDIA GeForce RTX 4090", "deploy.gpu_count": 2},
        ],
    }


def _obsolete(model_id="org/old", replacement="org/ready"):
    return {
        "model_id": model_id,
        "replacement_model_id": replacement,
        "rationale": "The replacement is stronger at the same practical VRAM footprint.",
    }


def test_applies_lifecycle_and_creates_onboarding_shell(tmp_path):
    first = _recipe(tmp_path, "first", "org/first", leading_comment=True)
    second = _recipe(tmp_path, "second", "org/second", tags=["maintained"])
    third = _recipe(tmp_path, "third", "org/third", tags=["best-effort"])
    selection = tmp_path / "selection.json"
    _manifest(
        selection,
        ["org/first"],
        best_effort=["org/second"],
        obsolete=[_obsolete("org/third", "org/first")],
        onboarding=[_candidate()],
    )

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)
    result = discovery_lifecycle.apply_manifest(manifest, tmp_path, tmp_path / "summary.md")

    assert result == {"changed": True}
    assert first.read_text().startswith("# Keep this qualification note.\ntags:\n  - maintained\n")
    assert yaml.safe_load(second.read_text())["tags"] == ["best-effort"]
    assert yaml.safe_load(third.read_text())["tags"] == ["obsolete"]
    shell = yaml.safe_load((tmp_path / "recipes" / "new-model" / "recipe.yaml").read_text())
    assert shell["tags"] == ["onboarding", "untested"]
    assert shell["model"] == {
        "huggingface": "org/new-model",
        "rationale": "Strong current adoption and serving value.",
        "task": "generate",
    }
    assert shell["matrices"] == _candidate()["deployments"]
    assert yaml.safe_load(first.read_text())["model"]["rationale"] == "Rationale for org/first."
    assert yaml.safe_load(second.read_text())["model"]["rationale"] == "Rationale for org/second."
    assert yaml.safe_load(third.read_text())["model"]["rationale"] == (
        "org/first supersedes this recipe: The replacement is stronger at the same practical VRAM footprint."
    )
    summary = (tmp_path / "summary.md").read_text()
    assert "`org/new-model`" in summary
    assert "`org/third` → `org/first`" in summary


def test_discovery_workflow_summarizes_tracked_and_new_recipe_changes(tmp_path):
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    existing = _recipe(tmp_path, "existing", "org/existing", tags=["best-effort"])
    subprocess.run(["git", "add", "recipes"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "fixture"], cwd=tmp_path, check=True)
    existing.write_text(existing.read_text().replace("best-effort", "maintained"))
    _recipe(tmp_path, "new", "org/new", tags=["onboarding", "untested"])

    workflow = yaml.safe_load((Path(__file__).parents[2] / ".github" / "workflows" / "discover-model.yml").read_text())
    script = next(step["run"] for step in workflow["jobs"]["discover"]["steps"] if step.get("name") == "Validate and apply model lifecycle")
    source = script.split("./venv/bin/python - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
    output_path = tmp_path / "github-output"
    env = {**os.environ, "GITHUB_OUTPUT": str(output_path)}

    subprocess.run([sys.executable, "-"], cwd=tmp_path, env=env, input=source, text=True, check=True)

    assert output_path.read_text().splitlines() == [
        'modified_models=[{"lifecycle":"maintained","model_id":"org/existing"},{"lifecycle":"onboarding","model_id":"org/new"}]'
    ]


def test_extracts_one_lifecycle_object_from_reasoning_text():
    text = """Analysis before the requested result.
```json
{"maintained_models": [{"model_id": "org/ready", "rationale": "Keep it."}],
 "best_effort_models": [], "obsolete_models": [], "onboarding_models": []}
```
"""

    assert discovery_lifecycle._extract_object(text) == {
        "maintained_models": [{"model_id": "org/ready", "rationale": "Keep it."}],
        "best_effort_models": [],
        "obsolete_models": [],
        "onboarding_models": [],
    }


def test_extracts_last_lifecycle_object_after_an_earlier_draft():
    first = {"maintained_models": [], "best_effort_models": [], "obsolete_models": [], "onboarding_models": []}
    final = {**first, "maintained_models": [_decision("org/final")]}

    assert discovery_lifecycle._extract_object(f"Draft: {json.dumps(first)}\nFinal: {json.dumps(final)}") == final


def test_rejects_extra_top_level_manifest_fields():
    text = json.dumps(
        {
            "maintained_models": [],
            "best_effort_models": [],
            "obsolete_models": [],
            "onboarding_models": [],
            "notes": "not allowed",
        }
    )

    with pytest.raises(ValueError, match="contain exactly"):
        discovery_lifecycle._extract_object(text)


def test_obsolete_recipe_can_become_maintained_again(tmp_path):
    recipe = _recipe(tmp_path, "old", "org/old", tags=["obsolete"])
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/old"])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)
    discovery_lifecycle.apply_manifest(manifest, tmp_path, tmp_path / "summary.md")

    assert yaml.safe_load(recipe.read_text())["tags"] == ["maintained"]


def test_obsolete_recipe_can_become_best_effort_again(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    recipe = _recipe(tmp_path, "old", "org/old", tags=["obsolete"])
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], best_effort=["org/old"])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)
    discovery_lifecycle.apply_manifest(manifest, tmp_path, tmp_path / "summary.md")

    assert yaml.safe_load(recipe.read_text())["tags"] == ["best-effort"]


def test_preserves_existing_onboarding_shell(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    shell = _recipe(tmp_path, "pending", "org/pending", tags=["onboarding", "untested"])
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)
    discovery_lifecycle.apply_manifest(manifest, tmp_path, tmp_path / "summary.md")

    assert yaml.safe_load(shell.read_text())["tags"] == ["onboarding", "untested"]
    assert manifest["existing_onboarding_models"] == [
        {
            "model_id": "org/pending",
            "rationale": (
                "Retained as a useful runnable recipe on a best-effort basis because discovery did not establish that it is obsolete."
            ),
            "deployments": [{"deploy.gpu": GPU, "deploy.gpu_count": 1}],
        }
    ]
    assert "rationale" in yaml.safe_load(shell.read_text())["model"]
    assert "`NVIDIA H200 141GB x1`" in (tmp_path / "summary.md").read_text()


def test_rejects_existing_onboarding_shell_without_deployment_matrix(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    shell = _recipe(tmp_path, "pending", "org/pending", tags=["onboarding", "untested"])
    shell.write_text(shell.read_text().replace("matrices:\n  - deploy.gpu: NVIDIA H200 141GB\n    deploy.gpu_count: 1\n", ""))
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"])

    with pytest.raises(ValueError, match="org/pending needs one to 3 deployments"):
        discovery_lifecycle.validate_manifest(selection, tmp_path)


def test_rewrites_unindented_yaml_tag_lists_without_leaving_duplicate_items(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    shell = _recipe(tmp_path, "pending", "org/pending", tags=["onboarding", "untested"])
    shell.write_text(shell.read_text().replace("  - onboarding\n  - untested\n", "- onboarding\n- untested\n"))
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)
    discovery_lifecycle.apply_manifest(manifest, tmp_path, tmp_path / "summary.md")

    assert yaml.safe_load(shell.read_text())["tags"] == ["onboarding", "untested"]
    assert shell.read_text().count("- onboarding") == 1


def test_moves_existing_rationale_immediately_below_model_id(tmp_path):
    recipe = _recipe(tmp_path, "ready", "org/ready", task="generate")
    recipe.write_text(recipe.read_text().replace("  task: generate\n", "  task: generate\n  rationale: Old rationale.\n"))
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)
    discovery_lifecycle.apply_manifest(manifest, tmp_path, tmp_path / "summary.md")

    model_lines = recipe.read_text().split("model:\n", 1)[1].split("engine:\n", 1)[0].splitlines()
    assert model_lines[:3] == [
        "  huggingface: org/ready",
        '  rationale: "Rationale for org/ready."',
        "  task: generate",
    ]


def test_rejects_onboarding_shell_as_maintained(tmp_path):
    _recipe(tmp_path, "pending", "org/pending", tags=["onboarding", "untested"])
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/pending"])

    with pytest.raises(ValueError, match="cannot be classified"):
        discovery_lifecycle.validate_manifest(selection, tmp_path)


def test_rejects_new_model_with_existing_recipe(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], onboarding=[_candidate("org/ready")])

    with pytest.raises(ValueError, match="recipe already exists"):
        discovery_lifecycle.validate_manifest(selection, tmp_path)


def test_rejects_candidate_with_unknown_hardware(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    selection = tmp_path / "selection.json"
    candidate = _candidate()
    candidate["deployments"] = [{"deploy.gpu": "NVIDIA Imaginary 1TB", "deploy.gpu_count": 1}]
    _manifest(selection, ["org/ready"], onboarding=[candidate])

    with pytest.raises(ValueError, match="selected unknown GPU"):
        discovery_lifecycle.validate_manifest(selection, tmp_path)


def test_rejects_more_than_three_candidate_deployments(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    selection = tmp_path / "selection.json"
    candidate = _candidate()
    candidate["deployments"] = [{"deploy.gpu": GPU, "deploy.gpu_count": count} for count in range(1, MAX_STUB_DEPLOYMENTS + 2)]
    _manifest(selection, ["org/ready"], onboarding=[candidate])

    with pytest.raises(ValueError, match="one to 3 deployments"):
        discovery_lifecycle.validate_manifest(selection, tmp_path)


def test_rejects_lifecycle_decision_without_rationale(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    selection = tmp_path / "selection.json"
    _manifest(selection, [{"model_id": "org/ready"}])

    with pytest.raises(ValueError, match="must contain exactly"):
        discovery_lifecycle.validate_manifest(selection, tmp_path)


def test_existing_onboarding_shells_discard_candidates_beyond_the_pending_limit(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    for index in range(3):
        _recipe(tmp_path, f"pending-{index}", f"org/pending-{index}", tags=["onboarding", "untested"])
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], onboarding=[_candidate()])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)

    assert manifest["onboarding_models"] == []


def test_rejects_more_than_three_onboarding_candidates(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], onboarding=[_candidate(f"org/new-{index}") for index in range(4)])

    with pytest.raises(ValueError, match="at most 3 candidates"):
        discovery_lifecycle.validate_manifest(selection, tmp_path)


def test_unclassified_complete_recipe_defaults_to_best_effort(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    other = _recipe(tmp_path, "other", "org/other")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)
    discovery_lifecycle.apply_manifest(manifest, tmp_path, tmp_path / "summary.md")

    assert [decision["model_id"] for decision in manifest["best_effort_models"]] == ["org/other"]
    assert yaml.safe_load(other.read_text())["tags"] == ["best-effort"]


def test_unknown_lower_priority_model_defaults_real_recipe_to_best_effort(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    _recipe(tmp_path, "other", "org/other")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], best_effort=["org/typo"])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)

    assert [decision["model_id"] for decision in manifest["best_effort_models"]] == ["org/other"]


def test_malformed_lower_priority_ids_default_real_recipes_to_best_effort(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    _recipe(tmp_path, "other", "org/other")
    selection = tmp_path / "selection.json"
    _manifest(
        selection,
        ["org/ready"],
        best_effort=["abbreviated-model"],
        obsolete=[_obsolete("org/other", "abbreviated-replacement")],
    )

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)

    assert [decision["model_id"] for decision in manifest["best_effort_models"]] == ["org/other"]
    assert manifest["obsolete_models"] == []


def test_unknown_maintained_model_is_rejected(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/typo"])

    with pytest.raises(ValueError, match="Maintained models must have complete existing recipes: org/typo"):
        discovery_lifecycle.validate_manifest(selection, tmp_path)


@pytest.mark.parametrize("selected", ["ready", "wrong-org/ready"])
def test_unique_checkpoint_name_in_maintained_set_is_normalized(tmp_path, selected):
    _recipe(tmp_path, "ready", "org/ready")
    selection = tmp_path / "selection.json"
    _manifest(selection, [selected])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)

    assert manifest["maintained_models"] == [_decision("org/ready", f"Rationale for {selected}.")]


def test_rejects_duplicate_lifecycle_classification(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], best_effort=["org/ready"])

    with pytest.raises(ValueError, match="exactly one lifecycle list"):
        discovery_lifecycle.validate_manifest(selection, tmp_path)


def test_obsolete_recipe_without_active_replacement_defaults_to_best_effort(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    _recipe(tmp_path, "old", "org/old")
    _recipe(tmp_path, "older", "org/older")
    selection = tmp_path / "selection.json"
    _manifest(
        selection,
        ["org/ready"],
        obsolete=[_obsolete("org/old", "org/older"), _obsolete("org/older", "org/ready")],
    )

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)

    assert [decision["model_id"] for decision in manifest["best_effort_models"]] == ["org/old"]
    assert [decision["model_id"] for decision in manifest["obsolete_models"]] == ["org/older"]


def test_obsolete_recipe_with_other_task_replacement_defaults_to_best_effort(tmp_path):
    _recipe(tmp_path, "ready", "org/ready", task="embed")
    _recipe(tmp_path, "old", "org/old", task="generate")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], obsolete=[_obsolete("org/old", "org/ready")])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)

    assert [decision["model_id"] for decision in manifest["best_effort_models"]] == ["org/old"]
    assert manifest["obsolete_models"] == []


def test_obsolete_recipe_with_larger_replacement_defaults_to_best_effort(tmp_path):
    _recipe(tmp_path, "ready", "org/ready", gpu=GPU)
    _recipe(tmp_path, "old", "org/old", gpu="NVIDIA GeForce RTX 4090")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], obsolete=[_obsolete("org/old", "org/ready")])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)

    assert [decision["model_id"] for decision in manifest["best_effort_models"]] == ["org/old"]
    assert manifest["obsolete_models"] == []


def test_obsolete_recipe_replacement_may_use_less_total_vram(tmp_path):
    _recipe(tmp_path, "ready", "org/ready", gpu="NVIDIA GeForce RTX 4090")
    _recipe(tmp_path, "old", "org/old", gpu=GPU)
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], obsolete=[_obsolete("org/old", "org/ready")])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)

    assert manifest["obsolete_models"][0]["model_id"] == "org/old"


def test_comparable_replacement_defaults_to_best_effort(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    _recipe(tmp_path, "old", "org/old")
    selection = tmp_path / "selection.json"
    decision = _obsolete("org/old", "org/ready")
    decision["rationale"] = "The replacement has comparable quality at a lower VRAM footprint."
    _manifest(selection, ["org/ready"], obsolete=[decision])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)

    assert [item["model_id"] for item in manifest["best_effort_models"]] == ["org/old"]
    assert manifest["obsolete_models"] == []


def test_replacement_with_less_serving_capacity_defaults_to_best_effort(tmp_path):
    ready = _recipe(tmp_path, "ready", "org/ready")
    old = _recipe(tmp_path, "old", "org/old")
    ready.write_text(ready.read_text().replace("llm: {}", "llm: {context_length: 1024, max_concurrent_requests: 8}"))
    old.write_text(old.read_text().replace("llm: {}", "llm: {context_length: 2048, max_concurrent_requests: 16}"))
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], obsolete=[_obsolete("org/old", "org/ready")])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)

    assert [item["model_id"] for item in manifest["best_effort_models"]] == ["org/old"]
    assert manifest["obsolete_models"] == []


def test_obsolete_recipe_may_include_drop_rationale_without_replacement(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    _recipe(tmp_path, "old", "org/old")
    selection = tmp_path / "selection.json"
    _manifest(
        selection,
        ["org/ready"],
        obsolete=[{"model_id": "org/old", "rationale": "The checkpoint cannot be served by a supported engine."}],
    )

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)

    assert manifest["obsolete_models"] == [{"model_id": "org/old", "rationale": "The checkpoint cannot be served by a supported engine."}]


def test_obsolete_rationale_names_exact_replacement_model(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    _recipe(tmp_path, "old", "org/old")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], obsolete=[_obsolete()])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)

    assert manifest["obsolete_models"][0]["rationale"].startswith("org/ready supersedes this recipe:")
