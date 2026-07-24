"""Dry-run + expansion tests for the gemma-4-12B-it A/B benchmark recipes.

The three recipes (stock vLLM, vLLM + emmy plugin, emmy + FAST_MATH) share one
workload grid — a 7-point concurrency-1 input-size sweep plus a 2-point batching
sweep = 9 variants each. These tests pin the variant count and the per-engine
config that distinguishes the three lanes.
"""

import os

from emmy.benchmark.tasks import enumerate_tasks

RECIPES = [
    "gemma-4-12B-it",
    "gemma-4-12B-it-emmy",
    "gemma-4-12B-it-emmy-fastmath",
]


def test_recipes_dry_run(run_cli, make_bench_config, recipes_dir, tmp_path):
    """Each recipe deploys + benchmarks 9 variants under the dry-run machinery."""
    config_path = make_bench_config(tmp_path)
    for name in RECIPES:
        recipe = os.path.join(recipes_dir, name)
        rc, stdout, stderr = run_cli("bench", recipe, "--config", config_path, "--dry-run")
        assert rc == 0, f"{name} stderr: {stderr}\nstdout: {stdout}"
        assert stdout.count("bench serve") == 9, f"{name}: expected 9 bench-serve tasks"
        assert "docker compose pull" in stdout
        assert "docker compose down" in stdout


def test_recipes_expand_to_expected_variants(recipes_dir):
    """Every recipe expands to the same 9-point workload grid."""
    for name in RECIPES:
        tasks = enumerate_tasks([os.path.join(recipes_dir, name)])
        assert len(tasks) == 9, f"{name}: expected 9 variants, got {len(tasks)}"

        # Controlled-workload knobs are pinned on every point.
        for t in tasks:
            b = t.recipe.benchmark
            assert b.seed == 0 and b.temperature == 0 and b.ignore_eos is True

        # Concurrency-1 input-size sweep: input zipped with context (context ≥ in+out).
        sweep = {
            (t.recipe.benchmark.random_input_len, t.recipe.engine.llm.context_length)
            for t in tasks
            if t.recipe.benchmark.max_concurrency == 1
        }
        assert sweep == {
            (512, 8192),
            (2048, 8192),
            (4096, 8192),
            (8192, 16384),
            (16384, 24576),
            (32768, 40960),
            (65536, 73728),
        }, f"{name}: sweep grid mismatch"

        # Batching sweep: num_prompts ≥ max(32, 8 × concurrency).
        batching = {
            (t.recipe.benchmark.max_concurrency, t.recipe.benchmark.num_prompts)
            for t in tasks
            if t.recipe.benchmark.random_output_len == 128
        }
        assert batching == {(4, 32), (8, 64)}, f"{name}: batching grid mismatch"


def test_engine_lanes_differ_only_where_intended(recipes_dir):
    """The three lanes: stock, emmy plugin, emmy + FAST_MATH."""
    stock = enumerate_tasks([os.path.join(recipes_dir, "gemma-4-12B-it")])[0].recipe.engine.llm
    emmy = enumerate_tasks([os.path.join(recipes_dir, "gemma-4-12B-it-emmy")])[0].recipe.engine.llm
    fast = enumerate_tasks([os.path.join(recipes_dir, "gemma-4-12B-it-emmy-fastmath")])[0].recipe.engine.llm

    # Stock: plain vLLM image, fp16, prefix caching off, standard 0.9 budget.
    assert "vllm/vllm-openai" in stock.image
    assert "--dtype float16" in stock.extra_args
    assert "--no-enable-prefix-caching" in stock.extra_args
    assert stock.gpu_memory_utilization == 0.9
    assert stock.extra_env == {}

    # Emmy: the plugin image + EmmyGenModel override + the prefill-chunk cap, 0.97 budget.
    assert "cloudriftai/vllm-emmy" in emmy.image
    assert "EmmyGenModel" in emmy.extra_args
    assert "--max-num-batched-tokens 4096" in emmy.extra_args
    assert "--enforce-eager" in emmy.extra_args
    assert emmy.gpu_memory_utilization == 0.97
    assert emmy.extra_env == {}

    # FAST_MATH: identical to emmy but for the one env var that enables the fork.
    assert fast.image == emmy.image
    assert fast.extra_args == emmy.extra_args
    assert fast.extra_env == {"EMMY_FAST_MATH": "1"}
