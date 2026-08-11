"""Dry-run + expansion tests for the gemma-4-12B-it serving recipe.

`recipes/` holds **deployment** configs — the recommended way to serve a model, one variant,
no workload grid. The A/B grids that justify those choices live in `experiments/`. These
tests pin that distinction for gemma-4-12B-it, because the recipe drifting back into a
benchmark sweep is exactly how it rotted the first time.

The serving config's value is that it matches the shape the published image was warmed at;
a recipe that silently drifts off it still deploys, but re-runs the compiler frontend on
every boot. So the shape is asserted here rather than left to a comment.
"""

import os

import pytest
import yaml

from emmy.benchmark.tasks import enumerate_tasks
from emmy.recipe.lifecycle import recipe_is_runnable

RECIPE = "gemma-4-12B-it"
BASE_RECIPE = "gemma-4-12B"

# The shape docker/vllm-emmy-serve/models/gemma-4-12b-it.env pins, and therefore the shape
# the published image's execution-plan pack is keyed on.
WARMED_MAX_MODEL_LEN = 16384
WARMED_MAX_NUM_BATCHED_TOKENS = 4128
WARMED_DECODE_BUCKET = "32"


def _require_runnable(recipes_dir, name):
    directory = os.path.join(recipes_dir, name)
    with open(os.path.join(directory, "recipe.yaml")) as recipe_file:
        config = yaml.safe_load(recipe_file)
    if not recipe_is_runnable(config):
        pytest.skip(f"{name} is disabled by its lifecycle tag")
    return directory


def _tasks(recipes_dir, name=RECIPE):
    return enumerate_tasks([_require_runnable(recipes_dir, name)])


def test_recipe_dry_run(run_cli, make_bench_config, recipes_dir, tmp_path):
    """The recipe deploys under the dry-run machinery as a single unit."""
    config_path = make_bench_config(tmp_path)
    recipe = _require_runnable(recipes_dir, RECIPE)
    rc, stdout, stderr = run_cli("bench", recipe, "--config", config_path, "--dry-run")
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "docker compose pull" in stdout
    assert "docker compose down" in stdout


def test_recipe_is_a_single_serving_variant(recipes_dir):
    """One variant, no sweep. A recipe that expands to many is an experiment in the wrong
    directory — `experiments/gemma-4-12B/` is where workload grids belong."""
    tasks = _tasks(recipes_dir)
    assert len(tasks) == 1, f"serving recipe expanded to {len(tasks)} variants"
    assert tasks[0].recipe.model.revision == "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"


def test_recipe_carries_no_benchmark_grid(recipes_dir):
    """No `benchmark:` block and no benchmark.* matrix keys: this file configures a server,
    not a measurement. (`enumerate_tasks` supplies BenchmarkConfig defaults regardless, so
    assert on the recipe SOURCE, which is what a reader and a reviewer actually see.)"""
    raw = open(os.path.join(recipes_dir, RECIPE, "recipe.yaml")).read()
    body = "\n".join(ln for ln in raw.splitlines() if not ln.lstrip().startswith("#"))
    assert "benchmark:" not in body, "serving recipe grew a benchmark block"
    assert "benchmark." not in body, "serving recipe grew benchmark.* matrix keys"


def test_serving_config_matches_the_image_warmed_shape(recipes_dir):
    """The point of the serving recipe: its shape equals the image's warmed shape, so a boot
    hits the execution-plan pack instead of re-running the compiler frontend per program."""
    recipe = _tasks(recipes_dir)[0].recipe
    llm = recipe.engine.llm
    assert llm.context_length == WARMED_MAX_MODEL_LEN
    assert f"--max-num-batched-tokens {WARMED_MAX_NUM_BATCHED_TOKENS}" in llm.vllm.extra_args
    assert llm.vllm.extra_env.get("EMMY_GEN_DECODE_BUCKET") == WARMED_DECODE_BUCKET


def test_serves_emmy_on_the_fast_math_fork(recipes_dir):
    """The recommended stack: the published emmy image, EmmyGenModel, fast math on."""
    llm = _tasks(recipes_dir)[0].recipe.engine.llm
    assert "cloudriftai/vllm-emmy" in llm.vllm.image
    assert "EmmyGenModel" in llm.vllm.extra_args
    assert llm.vllm.extra_env.get("EMMY_FAST_MATH") == "1"
    # The emmy generative path claims cupy residents before vLLM profiles memory, so the
    # utilization budget has to cover them — 0.9 fails the KV min-fit check at boot.
    assert llm.gpu_memory_utilization == 0.97


def test_recipe_shape_agrees_with_the_release_config(project_root, recipes_dir):
    """The recipe and docker/vllm-emmy-serve/models/<slug>.env must not drift apart: the
    .env is what the image was warmed with, the recipe is what gets deployed against it."""
    env_path = os.path.join(project_root, "docker/vllm-emmy-serve/models/gemma-4-12b-it.env")
    values = dict(
        ln.split("=", 1) for ln in open(env_path).read().splitlines() if ln.strip() and not ln.lstrip().startswith("#") and "=" in ln
    )
    llm = _tasks(recipes_dir)[0].recipe.engine.llm
    assert values["SERVE_MODEL"] == "google/gemma-4-12B-it"
    assert int(values["SERVE_MAX_MODEL_LEN"]) == llm.context_length
    assert f"--max-num-batched-tokens {values['SERVE_MAX_NUM_BATCHED_TOKENS']}" in llm.vllm.extra_args
    assert values["SERVE_DECODE_BUCKET"] == llm.vllm.extra_env["EMMY_GEN_DECODE_BUCKET"]


def test_base_recipe_is_one_pinned_completion_server(recipes_dir):
    """The separately requested base checkpoint has no chat contract or benchmark grid."""
    directory = _require_runnable(recipes_dir, BASE_RECIPE)
    tasks = _tasks(recipes_dir, BASE_RECIPE)
    assert len(tasks) == 1

    recipe = tasks[0].recipe
    assert recipe.model.huggingface == "google/gemma-4-12B"
    assert recipe.model.revision == "023679ed352de9bb66cc873c9009ce3482585c08"
    assert recipe.model.smoke_test == "completion"
    assert recipe.engine.llm.context_length == 16384
    assert recipe.engine.llm.gpu_memory_utilization == 0.95
    assert "vllm/vllm-openai@sha256:" in recipe.engine.llm.vllm.image

    raw = open(os.path.join(directory, "recipe.yaml")).read()
    body = "\n".join(ln for ln in raw.splitlines() if not ln.lstrip().startswith("#"))
    assert "benchmark:" not in body
