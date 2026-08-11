"""Configuration gates for the 2026 compiler-submission experiments."""

from pathlib import Path

from emmy.benchmark.tasks import enumerate_tasks
from emmy.recipe import load_recipe

EXP = Path("experiments/golden-bench-2026")


def _experiment(project_root: str, name: str) -> str:
    return str(Path(project_root) / EXP / name)


def _process_repeat(task) -> str:
    value = task.recipe.engine.llm.vllm.extra_env
    if isinstance(value, dict):
        return value["EMMY_BENCH_PROCESS_REPEAT"]
    for assignment in value.split():
        if assignment.startswith("EMMY_BENCH_PROCESS_REPEAT="):
            return assignment.split("=", 1)[1]
    raise AssertionError(f"missing process repeat in {value!r}")


def test_common_kernel_corpus_is_small_and_identical(project_root) -> None:
    platforms = {
        "kernels_v100": ("NVIDIA Tesla V100 SXM3 32GB", 16),
        "kernels_a100": ("NVIDIA A100 80GB", 8),
        "kernels_rtx4090": ("NVIDIA GeForce RTX 4090", 1),
        "kernels_rtx5090": ("NVIDIA GeForce RTX 5090", 1),
        "kernels_h200": ("NVIDIA H200 141GB", 8),
        "kernels_b200": ("NVIDIA B200", 8),
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
        pinned_model = "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"
        assert pinned_model in run
        assert '"$$seq_len" 12 4 0' in run
        assert "scripts/capture_kernel_environment.py" in recipe.command.stage
        assert "run_submission_kernel_case.sh" in run
        assert "overall_status=0" in run
        assert 'exit "$$overall_status"' in run
        assert "scripts/run_submission_kernel_case.sh" in recipe.command.stage
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
        "serving_deepseek_v4_flash_0731_exl3_a100x8": (
            "turboderp/DeepSeek-V4-Flash-0731-exl3",
            "80c463d631f03ae6ba35029929a04e8651c5276e",
            "NVIDIA A100 80GB",
            8,
        ),
        "serving_glm52_fp8_h200x8": (
            "zai-org/GLM-5.2-FP8",
            "ba978f7d347eaf65d22f1a86833408afdb953541",
            "NVIDIA H200 141GB",
            8,
        ),
        "serving_glm52_nvfp4_b200x8": (
            "nvidia/GLM-5.2-NVFP4",
            "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa",
            "NVIDIA B200",
            8,
        ),
    }

    for name, (model, revision, gpu, gpu_count) in systems.items():
        tasks = enumerate_tasks([_experiment(project_root, name)])
        assert len(tasks) == 15
        repeats_by_point = {}
        for task in tasks:
            assert task.recipe.model.huggingface == model
            assert task.recipe.model.revision == revision
            assert task.recipe.deploy.gpu == gpu
            assert task.recipe.deploy.gpu_count == gpu_count
            benchmark = task.recipe.benchmark
            assert benchmark.seed == 0
            assert benchmark.temperature == 0
            assert benchmark.ignore_eos is True
            assert benchmark.repeats == 1
            point = (benchmark.random_input_len, benchmark.random_output_len, benchmark.max_concurrency)
            repeats_by_point.setdefault(point, set()).add(_process_repeat(task))
            assert "--no-enable-prefix-caching" in task.recipe.engine.llm.vllm.extra_args
            if name != "serving_deepseek_v4_flash_0731_v100x16":
                assert "@sha256:" in task.recipe.engine.llm.vllm.image
        assert len(repeats_by_point) == 3
        assert all(repeats == {"0", "1", "2", "3", "4"} for repeats in repeats_by_point.values())


def test_qwen_nvfp4_qualification_uses_modelopt(project_root) -> None:
    tasks = enumerate_tasks([_experiment(project_root, "serving_qwen36_27b_nvfp4_rtx5090")])
    for task in tasks:
        llm = task.recipe.engine.llm
        assert llm.tensor_parallel_size == 1
        assert llm.context_length == 32768
        assert llm.vllm.image == "vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f"
        assert "--quantization modelopt" in llm.vllm.extra_args


def test_large_layer_corpus_is_bounded_and_not_labeled_tp8(project_root) -> None:
    for name, gpu in (("large_layer_h200", "NVIDIA H200 141GB"), ("large_layer_b200", "NVIDIA B200")):
        recipe_dir = _experiment(project_root, name)
        recipe = load_recipe(recipe_dir)
        tasks = enumerate_tasks([recipe_dir])
        assert len(tasks) == 1
        assert tasks[0].recipe.deploy.gpu == gpu
        assert tasks[0].recipe.deploy.gpu_count == 8

        run = recipe.command.run
        assert "for layer in 0 3; do" in run
        assert "for seq_len in 1 512; do" in run
        pinned_model = "Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
        assert pinned_model in run
        assert '"$$seq_len" 8 3 0' in run
        assert "run_submission_kernel_case.sh" in run
        assert "overall_status=0" in run
        assert recipe.command.result_files == ["layer-*-seq-*/artifacts.tar.gz"]


def test_convergence_check_is_one_shape_and_three_seeds(project_root) -> None:
    recipe_dir = _experiment(project_root, "kernels_convergence_h200")
    recipe = load_recipe(recipe_dir)
    tasks = enumerate_tasks([recipe_dir])
    assert len(tasks) == 1
    assert tasks[0].recipe.deploy.gpu == "NVIDIA H200 141GB"
    assert tasks[0].recipe.deploy.gpu_count == 8

    run = recipe.command.run
    assert "for seed in 0 1 2; do" in run
    assert "Qwen/Qwen3-0.6B@" in run
    assert '0 512 12 4 "$$seed"' in run
    assert "overall_status=0" in run
    assert recipe.command.result_files == ["seed-*/artifacts.tar.gz"]


def test_h200_independent_compiler_and_search_ablation_are_executable(project_root) -> None:
    common = load_recipe(_experiment(project_root, "kernels_h200")).command.run
    assert 'pip install "hidet==0.6.1"' in common
    assert '"$gpu_device_ids" default hidet' in common

    ablation_dir = _experiment(project_root, "kernels_search_ablation_h200")
    ablation = load_recipe(ablation_dir)
    tasks = enumerate_tasks([ablation_dir])
    assert len(tasks) == 1
    assert tasks[0].recipe.deploy.gpu == "NVIDIA H200 141GB"
    assert tasks[0].recipe.deploy.gpu_count == 8
    run = ablation.command.run
    assert "for budget_and_patience in 0:0 4:4 12:4 48:12; do" in run
    assert '512 "$$budget" "$$patience" 0' in run
    assert "run_submission_kernel_case.sh" in run
    assert ablation.command.result_files == ["budget-*/artifacts.tar.gz"]


def test_gemma_serving_ab_has_four_points_per_lane(project_root) -> None:
    tasks = enumerate_tasks([_experiment(project_root, "serving_gemma4_rtx5090")])
    assert len(tasks) == 40

    stock = [task for task in tasks if task.recipe.engine.llm.vllm.entrypoint is not None]
    emmy = [task for task in tasks if task.recipe.engine.llm.vllm.entrypoint is None]
    assert len(stock) == 20
    assert len(emmy) == 20
    assert {task.recipe.engine.llm.vllm.image for task in tasks} == {
        "cloudriftai/vllm-emmy-gemma-4-12b-it@sha256:5add12d3b7f4673790b435b76635082433538e3615fbc40227fa1c0db64c9ff3"
    }

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
        assert all(task.recipe.benchmark.repeats == 1 for task in lane)
        assert all(
            task.recipe.benchmark.output_probe_file == "experiments/golden-bench-2026/quality_gemma4_rtx5090/prompts.jsonl" for task in lane
        )

    expected_tokens = {
        (256, 256, 64): 2112,
        (4096, 4096, 1): 4128,
        (4096, 4096, 8): 2056,
        (8192, 256, 4): 4104,
    }
    repeats_by_lane_and_point = {}
    for task in tasks:
        point = (
            task.recipe.benchmark.random_input_len,
            task.recipe.benchmark.random_output_len,
            task.recipe.benchmark.max_concurrency,
        )
        assert f"--max-num-batched-tokens {expected_tokens[point]}" in task.recipe.engine.llm.vllm.extra_args
        lane = "stock" if task in stock else "emmy"
        repeats_by_lane_and_point.setdefault((lane, point), set()).add(_process_repeat(task))
    assert len(repeats_by_lane_and_point) == 8
    assert all(repeats == {"0", "1", "2", "3", "4"} for repeats in repeats_by_lane_and_point.values())


def test_gemma_image_provenance_pins_shared_vllm_revision(project_root) -> None:
    provenance = Path(project_root) / EXP / "serving_gemma4_rtx5090" / "IMAGE_PROVENANCE.md"
    text = provenance.read_text(encoding="utf-8")
    assert "91df0fad4dc98a67c7659d9dbd915245d5c43d96" in text
    assert "sha256:3a1e7f5904e1a1192a02aa0086ceaffc33985d7044c7bb25b3a43d61bdbe3ac0" in text
    assert "sha256:5add12d3b7f4673790b435b76635082433538e3615fbc40227fa1c0db64c9ff3" in text
