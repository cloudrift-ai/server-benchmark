"""Recipe catalog inventory and onboarding-stub creation."""

import yaml

from emmy.recipe.catalog import CATALOG_SCHEMA_VERSION, create_recipe_stub, recipe_inventory, recipe_inventory_document

GPU = "NVIDIA H200 141GB"


def _write_recipe(root, name, model_id, tags, *, rationale="Useful model."):
    path = root / name / "recipe.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "tags": tags,
                "model": {"huggingface": model_id, "rationale": rationale, "task": "generate"},
                "engine": {"llm": {"context_length": 8192, "vllm": {"image": "example/vllm"}}},
                "matrices": [
                    {"deploy.gpu": GPU, "deploy.gpu_count": 1},
                    {"deploy.gpu": "NVIDIA B200", "deploy.gpu_count": 2, "engine.llm.context_length": 16384},
                ],
            },
            sort_keys=False,
        )
    )
    return path


def test_recipe_inventory_filters_tags_and_reports_deployments(tmp_path):
    root = tmp_path / "recipes"
    _write_recipe(root, "ready", "org/ready", ["maintained"])
    _write_recipe(root, "other", "org/other", ["best-effort"])

    inventory = recipe_inventory(root, ("maintained",))

    assert inventory == [
        {
            "path": str(root / "ready" / "recipe.yaml"),
            "name": "ready",
            "model_id": "org/ready",
            "tags": ["maintained"],
            "task": "generate",
            "runnable": True,
            "deployments": [
                {"gpu": GPU, "gpu_count": 1, "context_length": 8192},
                {"gpu": "NVIDIA B200", "gpu_count": 2, "context_length": 16384},
            ],
            "rationale": "Useful model.",
        }
    ]


def test_recipe_inventory_document_is_versioned(tmp_path):
    root = tmp_path / "recipes"
    _write_recipe(root, "ready", "org/ready", ["maintained"])

    document = recipe_inventory_document(root)

    assert document["schema_version"] == CATALOG_SCHEMA_VERSION == 1
    assert [recipe["name"] for recipe in document["recipes"]] == ["ready"]


def test_disabled_recipe_is_not_runnable_but_keeps_proposed_deployments(tmp_path):
    root = tmp_path / "recipes"
    recipe = _write_recipe(root, "old", "org/old", ["obsolete"])

    record = recipe_inventory(root)[0]

    assert record["runnable"] is False
    assert record["deployments"][0]["context_length"] == 8192
    assert recipe.is_file()


def test_create_recipe_stub_writes_native_deployment_matrix(tmp_path):
    root = tmp_path / "recipes"
    recipe = create_recipe_stub(
        root,
        "org/new-model",
        "Strong current serving demand.",
        "generate",
        [
            {"deploy.gpu": GPU, "deploy.gpu_count": 1},
            {"deploy.gpu": "NVIDIA B200", "deploy.gpu_count": 2},
        ],
    )

    config = yaml.safe_load(recipe.read_text())
    assert config == {
        "tags": ["onboarding", "untested"],
        "model": {
            "huggingface": "org/new-model",
            "rationale": "Strong current serving demand.",
            "task": "generate",
        },
        "matrices": [
            {"deploy.gpu": GPU, "deploy.gpu_count": 1},
            {"deploy.gpu": "NVIDIA B200", "deploy.gpu_count": 2},
        ],
    }
    assert list(config["model"])[:2] == ["huggingface", "rationale"]


def test_create_recipe_stub_uses_organization_when_checkpoint_directory_exists(tmp_path):
    root = tmp_path / "recipes"
    _write_recipe(root, "new-model", "other/new-model", ["best-effort"])

    recipe = create_recipe_stub(
        root,
        "org/new-model",
        "Different organization checkpoint.",
        "generate",
        [{"deploy.gpu": GPU, "deploy.gpu_count": 1}],
    )

    assert recipe == root / "org--new-model" / "recipe.yaml"
