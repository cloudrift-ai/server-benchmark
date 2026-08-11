"""Tests for scripts/validate_gemma_serving_ab.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validate_gemma_serving_ab.py"
_SPEC = importlib.util.spec_from_file_location("validate_gemma_serving_ab", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
validator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validator
_SPEC.loader.exec_module(validator)


def test_validate_gemma_same_image_balanced_ab():
    recipe = PROJECT_ROOT / "experiments/golden-bench-2026/serving_gemma4_rtx5090"
    provenance = recipe / "IMAGE_PROVENANCE.json"

    summary = validator.validate(recipe, provenance)

    assert summary["tasks"] == 40
    assert summary["workloads"] == 4
    assert summary["image"].startswith("cloudriftai/vllm-emmy-gemma-4-12b-it@sha256:")


def test_validate_gemma_rejects_different_image(monkeypatch):
    recipe = PROJECT_ROOT / "experiments/golden-bench-2026/serving_gemma4_rtx5090"
    provenance = recipe / "IMAGE_PROVENANCE.json"
    real_enumerate = validator.enumerate_tasks
    tasks = real_enumerate([str(recipe)])
    tasks[0].recipe.engine.llm.vllm.image = "different/image@sha256:" + "0" * 64
    monkeypatch.setattr(validator, "enumerate_tasks", lambda _paths: tasks)

    with pytest.raises(validator.GemmaABError, match="same immutable image"):
        validator.validate(recipe, provenance)
