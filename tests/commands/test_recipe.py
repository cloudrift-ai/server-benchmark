"""The recipe inventory and onboarding-stub CLI."""

import json

import yaml

GPU = "NVIDIA H200 141GB"


def test_recipe_create_json_and_list_by_tag(run_cli, tmp_path):
    root = tmp_path / "recipes"
    returncode, stdout, stderr = run_cli(
        "recipe",
        "create",
        "org/new-model",
        "--root",
        str(root),
        "--rationale",
        "Strong current serving demand.",
        "--deployment",
        GPU,
        "1",
        "--json",
    )

    assert returncode == 0, stderr
    created = json.loads(stdout)
    assert created["model_id"] == "org/new-model"
    assert created["tags"] == ["onboarding", "untested"]
    assert created["deployments"] == [{"deploy.gpu": GPU, "deploy.gpu_count": 1}]
    assert yaml.safe_load((root / "new-model" / "recipe.yaml").read_text())["matrices"] == created["deployments"]

    returncode, stdout, stderr = run_cli("recipe", "list", str(root), "--tag", "onboarding", "--json")

    assert returncode == 0, stderr
    inventory = json.loads(stdout)
    assert [recipe["model_id"] for recipe in inventory] == ["org/new-model"]


def test_recipe_list_excludes_recipes_without_requested_tag(run_cli, tmp_path):
    root = tmp_path / "recipes"
    recipe = root / "ready" / "recipe.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("tags: [best-effort]\nmodel:\n  huggingface: org/ready\n")

    returncode, stdout, stderr = run_cli("recipe", "list", str(root), "--tag", "maintained", "--json")

    assert returncode == 0, stderr
    assert json.loads(stdout) == []
