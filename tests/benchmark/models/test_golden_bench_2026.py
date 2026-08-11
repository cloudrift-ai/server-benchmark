"""Configuration gates for the 2026 compiler-submission experiments."""

from pathlib import Path

from emmy.benchmark.tasks import enumerate_tasks
from emmy.recipe import load_recipe

EXP = Path("experiments/golden-bench-2026")


def _experiment(project_root: str, name: str) -> str:
    return str(Path(project_root) / EXP / name)


def test_common_kernel_corpus_is_small_and_identical(project_root) -> None:
    platforms = {
        "kernels_v100": ("NVIDIA Tesla V100 SXM3 32GB", 16),
        "kernels_rtx4090": ("NVIDIA GeForce RTX 4090", 1),
        "kernels_rtx5090": ("NVIDIA GeForce RTX 5090", 1),
        "kernels_h200": ("NVIDIA H200 141GB", 4),
        "kernels_b200": ("NVIDIA B200", 4),
    }

    for name, (gpu, gpu_count) in platforms.items():
        recipe_dir = _experiment(project_root, name)
        recipe = load_recipe(recipe_dir)
        tasks = enumerate_tasks([recipe_dir])
        assert recipe.kind == "command"
        assert len(tasks) == 1
        assert tasks[0].recipe.deploy.gpu == gpu
        assert tasks[0].recipe.deploy.gpu_count == gpu_count

        run = recipe.command.run
        assert "for seq_len in 1 512; do" in run
        assert "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca" in run
        assert "--max-candidates 12 --patience 4 --seed 0" in run
        assert "--warmup 10 --iters 100" in run
        assert "artifacts.tar.gz" in run
        assert recipe.command.result_files == ["seq-*/artifacts.tar.gz"]


def test_serving_systems_are_pinned_and_controlled(project_root) -> None:
    systems = {
        "serving_deepseek_v4_flash_0731_v100x16": (
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            "NVIDIA Tesla V100 SXM3 32GB",
            16,
        ),
        "serving_qwen36_27b_awq_rtx4090": (
            "cyankiwi/Qwen3.6-27B-AWQ-INT4",
            "e5cc0400fb2403c437c2c40a7c52fb5ae93fda18",
            "NVIDIA GeForce RTX 4090",
            1,
        ),
        "serving_qwen36_27b_nvfp4_rtx5090": (
            "nvidia/Qwen3.6-27B-NVFP4",
            "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
            "NVIDIA GeForce RTX 5090",
            1,
        ),
        "serving_deepseek_v4_flash_h200x4": (
            "deepseek-ai/DeepSeek-V4-Flash",
            "60d8d70770c6776ff598c94bb586a859a38244f1",
            "NVIDIA H200 141GB",
            4,
        ),
        "serving_kimi_k25_nvfp4_b200x4": (
            "nvidia/Kimi-K2.5-NVFP4",
            "0fd0a5e6879298d3476e3b61852a79792a35ae3d",
            "NVIDIA B200",
            4,
        ),
    }

    for name, (model, revision, gpu, gpu_count) in systems.items():
        tasks = enumerate_tasks([_experiment(project_root, name)])
        assert len(tasks) == 3
        for task in tasks:
            assert task.recipe.model.huggingface == model
            assert task.recipe.model.revision == revision
            assert task.recipe.deploy.gpu == gpu
            assert task.recipe.deploy.gpu_count == gpu_count
            benchmark = task.recipe.benchmark
            assert benchmark.seed == 0
            assert benchmark.temperature == 0
            assert benchmark.ignore_eos is True
            assert benchmark.repeats == 3


def test_qwen_nvfp4_qualification_uses_modelopt(project_root) -> None:
    tasks = enumerate_tasks([_experiment(project_root, "serving_qwen36_27b_nvfp4_rtx5090")])
    for task in tasks:
        llm = task.recipe.engine.llm
        assert llm.tensor_parallel_size == 1
        assert llm.context_length == 32768
        assert llm.vllm.image == "vllm/vllm-openai:nightly"
        assert "--quantization modelopt" in llm.vllm.extra_args


def test_gemma_serving_ab_has_four_points_per_lane(project_root) -> None:
    tasks = enumerate_tasks([_experiment(project_root, "serving_gemma4_rtx5090")])
    assert len(tasks) == 12

    stock = [task for task in tasks if "vllm-openai" in task.recipe.engine.llm.vllm.image]
    emmy = [task for task in tasks if "vllm-emmy" in task.recipe.engine.llm.vllm.image]
    fast_math = [task for task in emmy if "EMMY_FAST_MATH=1" in (task.recipe.engine.llm.vllm.extra_env or "")]
    assert len(stock) == 4
    assert len(emmy) == 8
    assert len(fast_math) == 4

    expected_points = {(256, 256, 64), (4096, 4096, 1), (4096, 4096, 8), (8192, 256, 4)}
    for lane in (stock, emmy):
        points = {
            (
                task.recipe.benchmark.random_input_len,
                task.recipe.benchmark.random_output_len,
                task.recipe.benchmark.max_concurrency,
            )
            for task in lane
        }
        assert points == expected_points
        assert all(task.recipe.benchmark.repeats == 3 for task in lane)
