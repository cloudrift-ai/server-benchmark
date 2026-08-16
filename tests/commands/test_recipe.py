"""The recipe inventory and onboarding-stub CLI."""

import json

import yaml

GPU = "NVIDIA H200 141GB"


def test_recipe_create(run_cli, tmp_path):
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
    )

    assert returncode == 0, stderr
    recipe_path = root / "new-model" / "recipe.yaml"
    assert stdout.strip() == str(recipe_path)
    assert yaml.safe_load(recipe_path.read_text())["matrices"] == [{"deploy.gpu": GPU, "deploy.gpu_count": 1}]


def test_recipe_list_by_tag(run_cli):
    returncode, stdout, stderr = run_cli("recipe", "list", "--tag", "best-effort", "--json")

    assert returncode == 0, stderr
    document = json.loads(stdout)
    assert document["schema_version"] == 1
    assert document["recipes"]
    assert all("best-effort" in recipe["tags"] for recipe in document["recipes"])


def test_recipe_list_excludes_recipes_without_requested_tag(run_cli):
    returncode, stdout, stderr = run_cli("recipe", "list", "--tag", "does-not-exist", "--json")

    assert returncode == 0, stderr
    assert json.loads(stdout) == {"schema_version": 1, "recipes": []}


def test_recipe_list_rejects_catalog_selection_arguments(run_cli):
    for arguments in (("recipes",), ("--bundled",)):
        returncode, _stdout, stderr = run_cli("recipe", "list", *arguments, "--json")

        assert returncode == 2
        assert "unrecognized arguments" in stderr


def test_recipe_query_filters_sorts_and_limits_catalog_rows(run_cli):
    returncode, stdout, stderr = run_cli(
        "recipe",
        "query",
        "--filter",
        'lifecycle == "best-effort"',
        "--sort",
        "results.last_run_at asc nulls-first",
        "--sort",
        "model_id asc",
        "--limit",
        "2",
        "--json",
    )

    assert returncode == 0, stderr
    document = json.loads(stdout)
    assert document["schema_version"] == 1
    assert 0 < len(document["rows"]) <= 2
    assert all(row["lifecycle"] == "best-effort" for row in document["rows"])
    assert [row["model_id"] for row in document["rows"]] == sorted(row["model_id"] for row in document["rows"])


def test_recipe_query_rejects_non_positive_limit(run_cli):
    returncode, stdout, _stderr = run_cli("recipe", "query", "--limit", "0", "--json")

    assert returncode == 2
    assert "limit must be positive" in stdout
