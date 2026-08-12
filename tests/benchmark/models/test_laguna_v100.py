"""Durable pins for the measured Laguna V100 serving lane and its evidence."""

from pathlib import Path

import yaml


def test_laguna_v100_recipe_matches_the_measured_experiment(project_root) -> None:
    """The recommended recipe stays identical to the measured lane except for workload and lifecycle metadata."""
    production = yaml.safe_load(Path(project_root, "recipes/Laguna-S-2.1-FP8/recipe.yaml").read_text())
    experiment = yaml.safe_load(Path(project_root, "experiments/Laguna-S-2.1-FP8/serving_v100_sxm3/recipe.yaml").read_text())

    assert "benchmark" not in production
    serving_config = {key: value for key, value in production.items() if key != "tags"}
    serving_config["model"].pop("rationale", None)
    assert serving_config == {key: value for key, value in experiment.items() if key != "benchmark"}

    llm = production["engine"]["llm"]
    assert llm["tensor_parallel_size"] == 8
    assert llm["pipeline_parallel_size"] == 1
    assert llm["context_length"] == 4096
    assert llm["max_concurrent_requests"] == 1
    assert production["matrices"] == {
        "deploy.gpu": "NVIDIA Tesla V100 SXM3 32GB",
        "deploy.gpu_count": 8,
    }


def test_laguna_v100_recipe_pins_the_qualified_dequant_fallback(project_root) -> None:
    """The native SM70 FP8 MoE route faults; never silently restore it in the recipe."""
    recipe = yaml.safe_load(Path(project_root, "recipes/Laguna-S-2.1-FP8/recipe.yaml").read_text())
    vllm = recipe["engine"]["llm"]["vllm"]

    assert vllm["image"] == "cloudriftai/1cat-vllm-sm70:1.2.2-cloudrift"
    assert recipe["model"]["revision"] == "9e0b8ba630080b0e6f20a7b43294a9f2232fd247"
    assert "--revision" not in vllm["extra_args"]
    assert vllm["extra_env"] == {
        "VLLM_SM70_FLASH_ATTN_V100": "1",
        "VLLM_SM70_FP8_TURBOMIND": "0",
        "VLLM_SM70_FP8_MOE_DEQUANT_FALLBACK": "1",
        "VLLM_SM70_FP8_MOE_BATCHED_GEMM": "0",
    }
