"""Controlled-workload bench flag emission."""

from emmy.benchmark.workload import build_bench_command
from emmy.recipe.types import Recipe


def _recipe(task: str, **bench) -> Recipe:
    return Recipe.from_dict(
        {
            "model": {"huggingface": "google/gemma-4-12B-it", "task": task},
            "engine": {"llm": {"context_length": 8192, "vllm": {}}},
            "benchmark": {"max_concurrency": 1, "num_prompts": 32, "random_input_len": 4096, "random_output_len": 4096, **bench},
        }
    )


def test_hygiene_flags_absent_by_default():
    cmd = build_bench_command(_recipe("generate"))
    assert "--seed" not in cmd
    assert "--temperature" not in cmd
    assert "--ignore-eos" not in cmd
    assert "--num-warmups" not in cmd


def test_hygiene_flags_emitted_when_set():
    cmd = build_bench_command(_recipe("generate", seed=0, temperature=0, ignore_eos=True, num_warmups=8))
    assert "--seed 0" in cmd
    assert "--temperature 0" in cmd
    assert "--ignore-eos" in cmd
    assert "--num-warmups 8" in cmd


def test_embedding_recipes_skip_generation_only_flags():
    cmd = build_bench_command(_recipe("embed", seed=7, temperature=0, ignore_eos=True, num_warmups=8))
    assert "--seed 7" in cmd
    assert "--num-warmups 8" in cmd
    assert "--temperature" not in cmd
    assert "--ignore-eos" not in cmd
