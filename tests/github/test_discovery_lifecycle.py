import importlib.util
import json
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
    plan = tmp_path / "plans" / "onboard-old.md"
    plan.parent.mkdir()
    plan.write_text("old plan")
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

    assert result == {
        "changed": True,
        "maintained_count": 1,
        "best_effort_count": 1,
        "obsolete_count": 1,
        "onboarding_count": 1,
    }
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
    assert not plan.exists()
    summary = (tmp_path / "summary.md").read_text()
    assert "`org/new-model`" in summary
    assert "`org/third` → `org/first`" in summary


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


def test_moves_legacy_onboarding_rationale_under_model(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    shell = _recipe(tmp_path, "pending", "org/pending", tags=["onboarding", "untested"])
    shell.write_text(shell.read_text() + "discovery:\n  rationale: Strong recent adoption.\n")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path)
    discovery_lifecycle.apply_manifest(manifest, tmp_path, tmp_path / "summary.md")

    text = shell.read_text()
    assert yaml.safe_load(text)["model"]["rationale"] == "Strong recent adoption."
    assert "discovery:" not in text


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
