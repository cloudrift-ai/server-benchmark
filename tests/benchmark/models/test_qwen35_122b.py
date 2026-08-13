"""Configuration gates for the Qwen3.5 122B V100 workloads."""

from pathlib import Path

import yaml

from emmy.benchmark.tasks import enumerate_tasks
from emmy.recipe import load_recipe


def _path(project_root: str, relative: str) -> str:
    return str(Path(project_root) / relative)


def test_volta_command_workloads_are_strict_and_single_gpu(project_root) -> None:
    for relative in ("experiments/qwen35-122b/stack_v100", "experiments/qwen35-122b/mma_smoke_v100"):
        recipe = load_recipe(_path(project_root, relative))
        assert recipe.kind == "command"
        assert recipe.command.run.startswith("set -euo pipefail\n")
        tasks = enumerate_tasks([_path(project_root, relative)])
        assert len(tasks) == 1
        assert tasks[0].recipe.deploy.gpu == "NVIDIA Tesla V100 SXM3 32GB"
        assert tasks[0].recipe.deploy.gpu_count == 1

    stack = load_recipe(_path(project_root, "experiments/qwen35-122b/stack_v100"))
    assert "docker build \\\n  --provenance=false" in stack.command.run


def test_onecat_image_is_reproducibly_pinned(project_root) -> None:
    dockerfile = Path(project_root, "docker/1cat-vllm-sm70/Dockerfile").read_text()
    assert "12.8.1-devel-ubuntu24.04@sha256:4b9ed5fa" in dockerfile
    assert "ONECAT_REPO=https://github.com/cloudrift-ai/1Cat-vLLM.git" in dockerfile
    assert "ONECAT_COMMIT=91aca502d2bb1f05d9208ab2edec9fae53ff0d0b" in dockerfile
    assert "ONECAT_BASE_REF=cloudrift/v1.2.2" in dockerfile
    assert "ONECAT_BASE_TAG=v1.2.2" in dockerfile
    assert 'fetch --depth 16 origin "refs/heads/${ONECAT_BASE_REF}:refs/tags/${ONECAT_BASE_TAG}"' in dockerfile
    assert 'test "$(git -C /src rev-parse HEAD)" = "${ONECAT_COMMIT}"' in dockerfile
    assert 'merge-base --is-ancestor "${ONECAT_COMMIT}" "${ONECAT_BASE_TAG}"' in dockerfile
    assert "python -m build --wheel --no-isolation --outdir /wheels" in dockerfile
    assert "CMAKE_BUILD_TYPE=Release python -m build" in dockerfile
    assert "--mount=from=builder,source=/wheels,target=/tmp/wheels,ro" in dockerfile
    assert "TORCH_CUDA_ARCH_LIST=7.0" in dockerfile
    assert "FLASH_ATTN_V100_CUDA_ARCH_LIST=7.0" in dockerfile
    assert "catboost==1.2.10" in dockerfile
    assert "cupy-cuda12x==14.1.1" in dockerfile
    assert "apt-get install -y --no-install-recommends curl" in dockerfile
    assert "strip --strip-unneeded" in dockerfile
    assert "VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL=0" in dockerfile
    assert "patch_onecat.py" not in dockerfile


def test_onecat_stack_qualifies_fork_fixes_on_sm70(project_root) -> None:
    recipe = load_recipe(_path(project_root, "experiments/qwen35-122b/stack_v100"))
    run = recipe.command.run
    assert "cloudriftai/1cat-vllm-sm70:1.2.2-cloudrift" in run
    assert 'importlib.metadata.version("tilelang") == "0.1.10"' in run
    assert 'importlib.metadata.version("apache-tvm-ffi") == "0.1.10"' in run
    assert '"_dummy_run return non-last PP rank" in runner_source' in run
    assert '"-arch=sm_70"' in run
    assert "curl --version" in run


def test_qwen_serving_recipe_is_one_exact_fp16_pp8_tp2_variant(project_root) -> None:
    relative = "recipes/Qwen3.5-122B-A10B"
    recipe = enumerate_tasks([_path(project_root, relative)])[0].recipe
    source = Path(project_root, relative, "recipe.yaml").read_text()
    assert "benchmark:" not in "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert recipe.model.huggingface == "Qwen/Qwen3.5-122B-A10B"
    assert recipe.model.revision == "dc4d348443bc740c68e2d77492492c11606384d5"
    assert recipe.engine.llm.pipeline_parallel_size == 8
    assert recipe.engine.llm.tensor_parallel_size == 2
    assert recipe.deploy.gpu_count == 16
    args = recipe.engine.llm.vllm.extra_args
    for flag in (
        "--dtype half",
        "--language-model-only",
        "--enforce-eager",
        "--attention-backend FLASH_ATTN_V100",
    ):
        assert flag in args


def test_serving_smoke_matches_the_serving_configuration(project_root) -> None:
    serving = yaml.safe_load(Path(project_root, "recipes/Qwen3.5-122B-A10B/recipe.yaml").read_text())
    probe = yaml.safe_load(Path(project_root, "experiments/qwen35-122b/serving_smoke_v100/recipe.yaml").read_text())
    assert {key: value for key, value in serving["model"].items() if key != "rationale"} == probe["model"]
    for section in ("engine", "matrices"):
        assert probe[section] == serving[section]
    assert probe["benchmark"] == {
        "max_concurrency": 1,
        "num_prompts": 1,
        "random_input_len": 32,
        "random_output_len": 16,
        "seed": 0,
        "temperature": 0.0,
    }


def test_serving_harness_freezes_three_repeated_workloads(project_root) -> None:
    tasks = enumerate_tasks([_path(project_root, "experiments/qwen35-122b/serving_v100")])
    assert len(tasks) == 3
    assert [(task.recipe.benchmark.random_input_len, task.recipe.benchmark.random_output_len) for task in tasks] == [
        (32, 256),
        (4096, 256),
        (32000, 256),
    ]
    for task in tasks:
        benchmark = task.recipe.benchmark
        assert benchmark.repeats == 3
        assert benchmark.max_concurrency == benchmark.num_prompts == 1
        assert benchmark.seed == 0 and benchmark.temperature == 0.0 and benchmark.ignore_eos is True
        assert task.recipe.engine.llm.pipeline_parallel_size == 8
        assert task.recipe.engine.llm.tensor_parallel_size == 2
        assert task.recipe.deploy.gpu_count == 16
