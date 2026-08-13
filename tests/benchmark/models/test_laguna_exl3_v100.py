"""Durable pins for the measured Laguna EXL3 V100 serving lane."""

from pathlib import Path

import yaml


def test_laguna_exl3_v100_recipe_matches_the_measured_experiment(project_root) -> None:
    """The highest-precision V100 recipe stays identical to its benchmark lane."""
    production = yaml.safe_load(Path(project_root, "recipes/Laguna-S-2.1-exl3-5bpw/recipe.yaml").read_text())
    experiment_path = Path(project_root, "experiments/Laguna-S-2.1-exl3-5bpw/serving_v100_sxm2/recipe.yaml")
    experiment = yaml.safe_load(experiment_path.read_text())

    assert "benchmark" not in production
    assert production == {key: value for key, value in experiment.items() if key != "benchmark"}

    assert experiment["benchmark"] == {
        "random_input_len": 64,
        "random_output_len": 16,
        "max_concurrency": 1,
        "num_prompts": 4,
        "num_warmups": 4,
        "seed": 0,
        "temperature": 0,
        "ignore_eos": True,
        "repeats": 3,
    }

    llm = production["engine"]["llm"]
    assert llm["tensor_parallel_size"] == 1
    assert llm["pipeline_parallel_size"] == 8
    assert llm["context_length"] == 262144
    assert llm["max_concurrent_requests"] == 1
    assert llm["gpu_memory_utilization"] == 0.84
    assert llm["vllm"]["extra_env"]["EMMY_PACK_DIR"] == "/opt/emmy/pack"
    assert production["model"]["revision"] == "3469659b2f9a1656805250880c6ea9760f9626ed"
    assert production["matrices"] == {
        "deploy.gpu": "NVIDIA Tesla V100 SXM2 16GB",
        "deploy.gpu_count": 8,
    }
