import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / ".github" / "scripts" / "discovery_selection.py"
SPEC = importlib.util.spec_from_file_location("discovery_selection", MODULE_PATH)
discovery_selection = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(discovery_selection)


def test_validate_selection_returns_at_most_one_exact_target(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "model_id": "org/new-model",
                "gpu": "NVIDIA H200 141GB",
                "gpu_count": 1,
                "rationale": "Strong current adoption and high serving value.",
            }
        )
    )

    result = discovery_selection.validate_selection(selection, tmp_path, "NVIDIA H200 141GB", 1)

    assert result["found"] is True
    assert result["model_id"] == "org/new-model"


def test_validate_selection_rejects_model_with_recipe(tmp_path):
    recipe = tmp_path / "recipes" / "model" / "recipe.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("model:\n  huggingface: org/existing\n")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "model_id": "org/existing",
                "gpu": "NVIDIA H200 141GB",
                "gpu_count": 1,
                "rationale": "Popular.",
            }
        )
    )

    with pytest.raises(ValueError, match="recipe already exists"):
        discovery_selection.validate_selection(selection, tmp_path, "NVIDIA H200 141GB", 1)
