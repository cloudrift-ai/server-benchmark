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


def test_recipe_list_json_contains_complete_catalog(run_cli):
    returncode, stdout, stderr = run_cli("recipe", "list", "--json")

    assert returncode == 0, stderr
    document = json.loads(stdout)
    assert document["schema_version"] == 1
    assert document["recipes"]
    assert all("deployments" in recipe for recipe in document["recipes"])


def test_recipe_list_rejects_catalog_selection_arguments(run_cli):
    for arguments in (("recipes",), ("--bundled",), ("--tag", "maintained")):
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


def test_recipe_query_hydrates_external_candidate(run_cli):
    returncode, stdout, stderr = run_cli(
        "recipe",
        "query",
        "--candidate",
        "org/new-model",
        GPU,
        "1",
        "--filter",
        'lifecycle != "obsolete"',
        "--json",
    )

    assert returncode == 0, stderr
    row = json.loads(stdout)["rows"][0]
    assert row["model_id"] == "org/new-model"
    assert row["operation"] == "onboarding"
    assert row["deployment"]["gpu"] == GPU


def test_recipe_query_exposes_one_external_candidate_option(run_cli):
    returncode, stdout, stderr = run_cli("recipe", "query", "--help")

    assert returncode == 0, stderr
    assert "--candidate MODEL GPU COUNT" in stdout
    for removed_option in ("--require", "--model MODEL", "--gpu GPU", "--gpu-count", "--allow-missing-model"):
        assert removed_option not in stdout
