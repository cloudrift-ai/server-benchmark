import importlib.util
import json
from pathlib import Path

import pytest
import yaml

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
    path.write_text(
        f"{prefix}{tag_text}model:\n  huggingface: {model_id}\n{task_text}engine:\n  llm: {{}}\nmatrices:\n"
        f"  deploy.gpu: {gpu}\n  deploy.gpu_count: {gpu_count}\n"
    )
    return path


def _manifest(path, maintained, best_effort=None, obsolete=None, onboarding=None):
    path.write_text(
        json.dumps(
            {
                "maintained_models": maintained,
                "best_effort_models": best_effort or [],
                "obsolete_models": obsolete or [],
                "onboarding_models": onboarding or [],
            }
        )
    )


def _candidate(model_id="org/new-model"):
    return {
        "model_id": model_id,
        "task": "generate",
        "gpu": GPU,
        "gpu_count": 1,
        "rationale": "Strong current adoption and serving value.",
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

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 1)
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
    assert shell["model"] == {"huggingface": "org/new-model", "task": "generate"}
    assert shell["discovery"]["target_gpu"] == GPU
    assert not plan.exists()
    summary = (tmp_path / "summary.md").read_text()
    assert "`org/new-model`" in summary
    assert "`org/third` → `org/first`" in summary


def test_extracts_one_lifecycle_object_from_reasoning_text():
    text = """Analysis before the requested result.
```json
{"maintained_models": ["org/ready"], "best_effort_models": [], "obsolete_models": [], "onboarding_models": []}
```
"""

    assert discovery_lifecycle._extract_object(text) == {
        "maintained_models": ["org/ready"],
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

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 1)
    discovery_lifecycle.apply_manifest(manifest, tmp_path, tmp_path / "summary.md")

    assert yaml.safe_load(recipe.read_text())["tags"] == ["maintained"]


def test_obsolete_recipe_can_become_best_effort_again(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    recipe = _recipe(tmp_path, "old", "org/old", tags=["obsolete"])
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], best_effort=["org/old"])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 1)
    discovery_lifecycle.apply_manifest(manifest, tmp_path, tmp_path / "summary.md")

    assert yaml.safe_load(recipe.read_text())["tags"] == ["best-effort"]


def test_preserves_existing_onboarding_shell(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    shell = _recipe(tmp_path, "pending", "org/pending", tags=["onboarding", "untested"])
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 1)
    discovery_lifecycle.apply_manifest(manifest, tmp_path, tmp_path / "summary.md")

    assert yaml.safe_load(shell.read_text())["tags"] == ["onboarding", "untested"]
    assert manifest["existing_onboarding_models"] == ["org/pending"]


def test_rejects_wrong_maintained_count(tmp_path):
    _recipe(tmp_path, "one", "org/one")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/one"])

    with pytest.raises(ValueError, match="exactly 2"):
        discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 2)


def test_rejects_onboarding_shell_as_maintained(tmp_path):
    _recipe(tmp_path, "pending", "org/pending", tags=["onboarding", "untested"])
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/pending"])

    with pytest.raises(ValueError, match="cannot be classified"):
        discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 1)


def test_rejects_new_model_with_existing_recipe(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], onboarding=[_candidate("org/ready")])

    with pytest.raises(ValueError, match="recipe already exists"):
        discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 1)


def test_rejects_candidate_on_other_hardware(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    selection = tmp_path / "selection.json"
    candidate = _candidate()
    candidate["gpu_count"] = 2
    _manifest(selection, ["org/ready"], onboarding=[candidate])

    with pytest.raises(ValueError, match="outside the exact requested target"):
        discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 1)


def test_existing_onboarding_shells_consume_the_pending_limit(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    for index in range(3):
        _recipe(tmp_path, f"pending-{index}", f"org/pending-{index}", tags=["onboarding", "untested"])
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], onboarding=[_candidate()])

    with pytest.raises(ValueError, match="at most 3 total"):
        discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 1)


def test_unclassified_complete_recipe_defaults_to_best_effort(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    other = _recipe(tmp_path, "other", "org/other")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 1)
    discovery_lifecycle.apply_manifest(manifest, tmp_path, tmp_path / "summary.md")

    assert manifest["best_effort_models"] == ["org/other"]
    assert yaml.safe_load(other.read_text())["tags"] == ["best-effort"]


def test_rejects_duplicate_lifecycle_classification(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], best_effort=["org/ready"])

    with pytest.raises(ValueError, match="exactly one lifecycle list"):
        discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 1)


def test_obsolete_recipe_requires_active_replacement(tmp_path):
    _recipe(tmp_path, "ready", "org/ready")
    _recipe(tmp_path, "old", "org/old")
    _recipe(tmp_path, "older", "org/older")
    selection = tmp_path / "selection.json"
    _manifest(
        selection,
        ["org/ready"],
        obsolete=[_obsolete("org/old", "org/older"), _obsolete("org/older", "org/ready")],
    )

    with pytest.raises(ValueError, match="Replacement for obsolete model org/old must be maintained or best-effort"):
        discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 1)


def test_obsolete_recipe_replacement_must_serve_same_task(tmp_path):
    _recipe(tmp_path, "ready", "org/ready", task="embed")
    _recipe(tmp_path, "old", "org/old", task="generate")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], obsolete=[_obsolete("org/old", "org/ready")])

    with pytest.raises(ValueError, match="must serve the same task"):
        discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 1)


def test_obsolete_recipe_replacement_must_fit_same_or_less_total_vram(tmp_path):
    _recipe(tmp_path, "ready", "org/ready", gpu=GPU)
    _recipe(tmp_path, "old", "org/old", gpu="NVIDIA GeForce RTX 4090")
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], obsolete=[_obsolete("org/old", "org/ready")])

    with pytest.raises(ValueError, match="needs more total VRAM"):
        discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 1)


def test_obsolete_recipe_replacement_may_use_less_total_vram(tmp_path):
    _recipe(tmp_path, "ready", "org/ready", gpu="NVIDIA GeForce RTX 4090")
    _recipe(tmp_path, "old", "org/old", gpu=GPU)
    selection = tmp_path / "selection.json"
    _manifest(selection, ["org/ready"], obsolete=[_obsolete("org/old", "org/ready")])

    manifest = discovery_lifecycle.validate_manifest(selection, tmp_path, GPU, 1, 1)

    assert manifest["obsolete_models"][0]["model_id"] == "org/old"
