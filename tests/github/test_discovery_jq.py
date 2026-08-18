import json
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).parents[2]
TASK_FILTER = WORKSPACE / ".github" / "workflows" / "scripts" / "discovery_task.jq"
MANIFEST_FILTER = WORKSPACE / ".github" / "workflows" / "scripts" / "discovery_manifest.jq"
GPU = "NVIDIA H200 141GB"


def _query_row(model_id, *, index=0, tags=None, runnable=True, task="generate", gpu_count=1):
    name = model_id.rsplit("/", 1)[-1]
    return {
        "name": name,
        "recipe_path": f"recipes/{name}/recipe.yaml",
        "model_id": model_id,
        "tags": tags or [],
        "task": task,
        "runnable": runnable,
        "rationale": "Existing rationale.",
        "heat": 50,
        "deployment": {
            "index": index,
            "gpu": GPU,
            "gpu_count": gpu_count,
            "context_length": 8192,
            "availability": {"cloudrift": None},
        },
    }


def _run_task(rows, *, maintained_count=1, batch_size=2):
    result = subprocess.run(
        [
            "jq",
            "--argjson",
            "maintained_count",
            str(maintained_count),
            "--argjson",
            "batch_size",
            str(batch_size),
            "-f",
            str(TASK_FILTER),
        ],
        input=json.dumps({"schema_version": 1, "rows": rows}),
        text=True,
        capture_output=True,
    )
    return result, json.loads(result.stdout) if result.returncode == 0 else None


def _recipe(model_id, *, tags=None, runnable=True, task="generate", gpu_count=1):
    name = model_id.rsplit("/", 1)[-1]
    return {
        "name": name,
        "path": f"recipes/{name}/recipe.yaml",
        "model_id": model_id,
        "tags": tags or [],
        "task": task,
        "runnable": runnable,
        "rationale": "Existing rationale.",
        "heat": 50,
        "deployments": [{"gpu": GPU, "gpu_count": gpu_count, "context_length": 8192}],
    }


def _task(*recipes, maintained_count=1):
    return {"schema_version": 1, "maintained_count": maintained_count, "recipe_batches": [list(recipes)]}


def _score(model_id, heat=50):
    return {"model_id": model_id, "rationale": f"Current evidence for {model_id}.", "heat": heat}


def _selection(scores, maintained, obsolete=None, new_onboarding=None):
    return {
        "scores": scores,
        "maintained_model_ids": maintained,
        "obsolete_models": obsolete or [],
        "new_onboarding_models": new_onboarding or [],
    }


def _run_manifest(task, selection):
    result = subprocess.run(
        ["jq", "--arg", "selection", selection, "-f", str(MANIFEST_FILTER)],
        input=json.dumps(task),
        text=True,
        capture_output=True,
    )
    return result, json.loads(result.stdout) if result.returncode == 0 else None


def test_task_filter_groups_query_deployments_and_batches_recipes():
    rows = [
        _query_row("org/model-a"),
        _query_row("org/model-a", index=1, gpu_count=2),
        _query_row("org/model-b"),
        _query_row("org/model-c", tags=["onboarding", "untested"], runnable=False, task="embed"),
    ]

    result, task = _run_task(rows)

    assert result.returncode == 0, result.stderr
    assert task["schema_version"] == 1
    assert task["maintained_count"] == 1
    assert [[recipe["model_id"] for recipe in batch] for batch in task["recipe_batches"]] == [
        ["org/model-a", "org/model-b"],
        ["org/model-c"],
    ]
    assert task["recipe_batches"][0][0]["deployments"] == [
        {"gpu": GPU, "gpu_count": 1, "context_length": 8192},
        {"gpu": GPU, "gpu_count": 2, "context_length": 8192},
    ]


def test_manifest_filter_restores_existing_onboarding_and_filters_repeated_candidate():
    ready = _recipe("org/ready")
    pending = _recipe("org/pending", tags=["onboarding", "untested"], runnable=False, task="embed")
    selection = _selection(
        [_score("org/ready", 70), _score("org/pending", 85)],
        ["org/ready"],
        new_onboarding=[
            {
                "model_id": "org/pending",
                "task": "generate",
                "rationale": "Agent repeated an existing onboarding shell.",
                "heat": 85,
                "deployments": [{"deploy.gpu": GPU, "deploy.gpu_count": 2}],
            }
        ],
    )

    result, manifest = _run_manifest(_task(ready, pending), f"Draft:\n```json\n{json.dumps(selection)}\n```")

    assert result.returncode == 0, result.stderr
    assert manifest["maintained_models"] == [_score("org/ready", 70)]
    assert manifest["best_effort_models"] == []
    assert manifest["obsolete_models"] == []
    assert manifest["onboarding_models"] == [
        {
            **_score("org/pending", 85),
            "task": "embed",
            "deployments": [{"deploy.gpu": GPU, "deploy.gpu_count": 1}],
        }
    ]


def test_manifest_filter_derives_best_effort_and_obsolete_lists():
    ready = _recipe("org/ready")
    useful = _recipe("org/useful")
    old = _recipe("org/old")
    selection = _selection(
        [_score("org/ready", 80), _score("org/useful", 40), _score("org/old", 10)],
        ["org/ready"],
        obsolete=[{"model_id": "org/old", "replacement_model_id": "org/ready"}],
    )

    result, manifest = _run_manifest(_task(ready, useful, old), json.dumps(selection))

    assert result.returncode == 0, result.stderr
    assert manifest["best_effort_models"] == [_score("org/useful", 40)]
    assert manifest["obsolete_models"] == [{**_score("org/old", 10), "replacement_model_id": "org/ready"}]


def test_manifest_filter_accepts_obsolete_selection_without_replacement():
    ready = _recipe("org/ready")
    old = _recipe("org/old")
    selection = _selection(
        [_score("org/ready", 80), _score("org/old", 10)],
        ["org/ready"],
        obsolete=[{"model_id": "org/old"}],
    )

    result, manifest = _run_manifest(_task(ready, old), json.dumps(selection))

    assert result.returncode == 0, result.stderr
    assert manifest["obsolete_models"] == [_score("org/old", 10)]


def test_manifest_filter_rejects_inexact_score_coverage():
    task = _task(_recipe("org/ready"), _recipe("org/pending", tags=["onboarding", "untested"], runnable=False))
    selection = _selection([_score("org/ready")], ["org/ready"])

    result, manifest = _run_manifest(task, json.dumps(selection))

    assert manifest is None
    assert result.returncode != 0
    assert "Scores must cover every exact recipe ID once" in result.stderr
