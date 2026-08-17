import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / ".github" / "scripts" / "discovery_agent.py"
SPEC = importlib.util.spec_from_file_location("discovery_agent", MODULE_PATH)
discovery_agent = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(discovery_agent)

GPU = "NVIDIA H200 141GB"


def _recipe(workspace, name, model_id, tags=None, task="generate", heat=50):
    path = workspace / "recipes" / name / "recipe.yaml"
    path.parent.mkdir(parents=True)
    tag_text = "" if tags is None else "tags:\n" + "".join(f"  - {tag}\n" for tag in tags)
    if tags and "onboarding" in tags:
        body = f"matrices:\n  - deploy.gpu: {GPU}\n    deploy.gpu_count: 1\n"
    else:
        body = f"engine:\n  llm:\n    tensor_parallel_size: 1\nmatrices:\n  deploy.gpu: {GPU}\n  deploy.gpu_count: 1\n"
    path.write_text(
        f"{tag_text}model:\n  huggingface: {model_id}\n  rationale: Existing rationale.\n  heat: {heat}\n  task: {task}\n{body}"
    )


def _score(model_id, heat=50):
    return {"model_id": model_id, "rationale": f"Current evidence for {model_id}.", "heat": heat}


def _selection(scores, maintained, obsolete=None, new_onboarding=None):
    return json.dumps(
        {
            "scores": scores,
            "maintained_model_ids": maintained,
            "obsolete_models": obsolete or [],
            "new_onboarding_models": new_onboarding or [],
        }
    )


def test_builds_deterministic_recipe_batches(tmp_path):
    for index in range(5):
        _recipe(tmp_path, f"model-{index}", f"org/model-{index}")

    task = discovery_agent.build_task(tmp_path / "recipes", maintained_count=2, batch_size=2)

    assert task["schema_version"] == 1
    assert task["maintained_count"] == 2
    assert [[recipe["model_id"] for recipe in batch] for batch in task["recipe_batches"]] == [
        ["org/model-0", "org/model-1"],
        ["org/model-2", "org/model-3"],
        ["org/model-4"],
    ]


def test_assembles_existing_onboarding_from_task_instead_of_agent_output(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    _recipe(tmp_path, "pending", "org/pending", tags=["onboarding", "untested"], task="embed")
    task = discovery_agent.build_task(tmp_path / "recipes", maintained_count=1, batch_size=1)
    selection = _selection([_score("org/ready", 70), _score("org/pending", 85)], ["org/ready"])

    manifest = discovery_agent.assemble_manifest(task, selection)

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


def test_derives_best_effort_and_obsolete_decisions(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    _recipe(tmp_path, "useful", "org/useful")
    _recipe(tmp_path, "old", "org/old")
    task = discovery_agent.build_task(tmp_path / "recipes", maintained_count=1, batch_size=2)
    scores = [_score("org/old", 10), _score("org/ready", 80), _score("org/useful", 40)]
    selection = _selection(
        scores,
        ["org/ready"],
        obsolete=[{"model_id": "org/old", "replacement_model_id": "org/ready"}],
    )

    manifest = discovery_agent.assemble_manifest(task, selection)

    assert manifest["maintained_models"] == [_score("org/ready", 80)]
    assert manifest["best_effort_models"] == [_score("org/useful", 40)]
    assert manifest["obsolete_models"] == [{**_score("org/old", 10), "replacement_model_id": "org/ready"}]


def test_rejects_missing_existing_recipe_score(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    _recipe(tmp_path, "pending", "org/pending", tags=["onboarding", "untested"])
    task = discovery_agent.build_task(tmp_path / "recipes", maintained_count=1, batch_size=2)
    selection = _selection([_score("org/ready")], ["org/ready"])

    with pytest.raises(ValueError, match="Existing recipes must be scored: org/pending"):
        discovery_agent.assemble_manifest(task, selection)


def test_extracts_last_complete_selection_from_agent_text(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    task = discovery_agent.build_task(tmp_path / "recipes", maintained_count=1, batch_size=1)
    first = _selection([_score("org/ready", 40)], ["org/ready"])
    final = _selection([_score("org/ready", 80)], ["org/ready"])

    manifest = discovery_agent.assemble_manifest(task, f"Draft: {first}\nFinal: {final}\nIncomplete: {{")

    assert manifest["maintained_models"] == [_score("org/ready", 80)]
