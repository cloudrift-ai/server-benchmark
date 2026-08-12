"""Configuration gates for the 2026 compiler-submission experiments."""

import json
import subprocess
from pathlib import Path

from emmy.benchmark.command_workload import build_substitution_map, render_command
from emmy.benchmark.execution import validate_server_log_patterns
from emmy.benchmark.tasks import enumerate_tasks
from emmy.recipe import load_recipe

EXP = Path("experiments/golden-bench-2026")


def _experiment(project_root: str, name: str) -> str:
    return str(Path(project_root) / EXP / name)


def _kernel_tasks(project_root: str, study: str):
    tasks = enumerate_tasks([_experiment(project_root, "kernels")])
    return [task for task in tasks if task.variant.params["study"] == study]


def test_common_kernel_corpus_is_small_and_identical(project_root) -> None:
    platforms = {
        "NVIDIA Tesla V100 SXM3 32GB": (1, "none"),
        "NVIDIA A100 80GB": (1, "none"),
        "NVIDIA GeForce RTX 4090": (1, "none"),
        "NVIDIA GeForce RTX 5090": (1, "none"),
        "NVIDIA H200 141GB": (1, "hidet"),
        "NVIDIA B200": (1, "none"),
    }
    recipe_dir = _experiment(project_root, "kernels")
    recipe = load_recipe(recipe_dir)
    tasks = _kernel_tasks(project_root, "common")
    assert recipe.kind == "command"
    assert len(tasks) == len(platforms) * 2
    assert {task.recipe.deploy.gpu: (task.recipe.deploy.gpu_count, task.variant.params["optional_backend"]) for task in tasks} == platforms
    assert {task.variant.params["seq_len"] for task in tasks} == {1, 512}
    assert {task.variant.params["model_ref"] for task in tasks} == {"Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"}
    assert all(task.variant.params["budget"] == 12 for task in tasks)
    assert all(task.variant.params["patience"] == 4 for task in tasks)

    run = recipe.command.run
    assert "./venv/bin/emmy trace" in run
    assert "./venv/bin/emmy tune" in run
    assert "./venv/bin/emmy run" in run
    assert "--verify-working-golden tune-winners" in run
    assert 'pip install "hidet==0.6.1"' in run
    assert "scripts/" not in run
    assert recipe.command.stage == [
        "emmy",
        "pyproject.toml",
        "requirements.txt",
        "Makefile",
        "experiments/golden-bench-2026/kernels/recipe.yaml",
    ]
    assert recipe.command.require_clean_stage is True
    assert recipe.command.require_result_files is True
    assert recipe.command.require_provenance is True
    assert recipe.command.result_files == ["artifacts.tar.gz"]
    assert "pip freeze --all" in run
    assert "tar -C $task_dir" in run


def test_native_fp8_kernel_corpus_is_separate_and_identical(project_root) -> None:
    tasks = _kernel_tasks(project_root, "fp8-common")
    assert len(tasks) == 8
    assert {task.recipe.deploy.gpu for task in tasks} == {
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 5090",
        "NVIDIA H200 141GB",
        "NVIDIA B200",
    }
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)
    assert {task.variant.params["seq_len"] for task in tasks} == {1, 512}
    assert {task.variant.params["model_ref"] for task in tasks} == {
        "RedHatAI/Qwen3-0.6B-FP8-dynamic@068a9040b238d65f5c1064f10232bb15b96c0ff0"
    }
    assert all(task.variant.params["input_dtype"] == "f8e4m3" for task in tasks)
    assert all(task.variant.params["fp8_mma"] == 1 for task in tasks)
    for task in tasks:
        command = render_command(
            task.recipe.command.run,
            build_substitution_map(task.variant, [0], "/repo", "/task"),
        )
        assert "--input-dtype" in command
        assert "--require-kernel-source" in command
        assert "EMMY_FP8_MMA=1" in command


def test_native_fp8_large_layer_supplement_is_bounded(project_root) -> None:
    tasks = _kernel_tasks(project_root, "fp8-large-layer")
    assert len(tasks) == 4
    assert {task.recipe.deploy.gpu for task in tasks} == {"NVIDIA H200 141GB", "NVIDIA B200"}
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)
    assert {task.variant.params["seq_len"] for task in tasks} == {1, 512}
    assert {task.variant.params["layer"] for task in tasks} == {0}
    assert {task.variant.params["model_ref"] for task in tasks} == {
        "RedHatAI/Qwen3-32B-FP8-dynamic@c6732fc26128341172e4005bad34aafa51c32866"
    }
    assert all(task.variant.params["budget"] == 8 for task in tasks)
    assert all(task.variant.params["input_dtype"] == "f8e4m3" for task in tasks)


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
        "serving_qwen3_8b_nvfp4_rtx5090": (
            "nvidia/Qwen3-8B-NVFP4",
            "ccd10a893cbca613259517c3efe08e151ddf2b8e",
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
            repeats_by_point.setdefault(point, set()).add(benchmark.process_repeat)
            assert "--no-enable-prefix-caching" in task.recipe.engine.llm.vllm.extra_args
            if name != "serving_deepseek_v4_flash_0731_v100x16":
                assert "@sha256:" in task.recipe.engine.llm.vllm.image
        assert len(repeats_by_point) == 3
        assert all(repeats == {0, 1, 2, 3, 4} for repeats in repeats_by_point.values())


def test_qwen_nvfp4_qualification_is_w4a16_marlin(project_root) -> None:
    tasks = enumerate_tasks([_experiment(project_root, "serving_qwen36_27b_nvfp4_rtx5090")])
    for task in tasks:
        llm = task.recipe.engine.llm
        assert llm.tensor_parallel_size == 1
        assert llm.context_length == 32768
        assert llm.vllm.image == "vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f"
        assert "--quantization modelopt" in llm.vllm.extra_args
        benchmark = task.recipe.benchmark
        assert benchmark.required_server_log_patterns
        assert any("marlin" in pattern.lower() for pattern in benchmark.required_server_log_patterns)
        assert any("weight-only fp4" in pattern.lower() for pattern in benchmark.required_server_log_patterns)
        assert any("emulat" in pattern.lower() for pattern in benchmark.forbidden_server_log_patterns)


def test_qwen3_native_nvfp4_qualification_requires_optimized_w4a4_kernel(project_root) -> None:
    tasks = enumerate_tasks([_experiment(project_root, "serving_qwen3_8b_nvfp4_rtx5090")])
    for task in tasks:
        llm = task.recipe.engine.llm
        assert llm.tensor_parallel_size == 1
        assert "--quantization modelopt" in llm.vllm.extra_args
        benchmark = task.recipe.benchmark
        assert any("cutlassnvfp4" in pattern.lower() for pattern in benchmark.required_server_log_patterns)
        assert any("marlin" in pattern.lower() for pattern in benchmark.forbidden_server_log_patterns)
        assert any("emulation" in pattern.lower() for pattern in benchmark.forbidden_server_log_patterns)
        for backend in ("FlashInferCutlassNvFp4LinearKernel", "CutlassNvFp4LinearKernel"):
            logs = f"Detected ModelOpt NVFP4 checkpoint (quant_algo=NVFP4).\nUsing {backend} for NVFP4 GEMM"
            assert (
                validate_server_log_patterns(
                    logs,
                    benchmark.required_server_log_patterns,
                    benchmark.forbidden_server_log_patterns,
                )["status"]
                == "pass"
            )
        for backend in ("MarlinNvFp4LinearKernel", "EmulationNvFp4LinearKernel"):
            logs = f"Detected ModelOpt NVFP4 checkpoint (quant_algo=NVFP4).\nUsing {backend} for NVFP4 GEMM"
            assert (
                validate_server_log_patterns(
                    logs,
                    benchmark.required_server_log_patterns,
                    benchmark.forbidden_server_log_patterns,
                )["status"]
                == "fail"
            )


def test_b200_nvfp4_qualification_requires_native_backend_logs(project_root) -> None:
    tasks = enumerate_tasks([_experiment(project_root, "serving_glm52_nvfp4_b200x8")])
    for task in tasks:
        benchmark = task.recipe.benchmark
        assert any("vllm_cutlass" in pattern.lower() for pattern in benchmark.required_server_log_patterns)
        assert all("modeloptnvfp4" not in pattern.lower() for pattern in benchmark.required_server_log_patterns)
        assert any("falling back" in pattern.lower() for pattern in benchmark.forbidden_server_log_patterns)
        assert any("marlin" in pattern.lower() for pattern in benchmark.forbidden_server_log_patterns)


def test_large_layer_corpus_is_bounded_and_not_labeled_tp8(project_root) -> None:
    tasks = _kernel_tasks(project_root, "large-layer")
    assert len(tasks) == 8
    assert {task.recipe.deploy.gpu for task in tasks} == {"NVIDIA H200 141GB", "NVIDIA B200"}
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)
    assert {task.variant.params["layer"] for task in tasks} == {0, 3}
    assert {task.variant.params["seq_len"] for task in tasks} == {1, 512}
    assert {task.variant.params["model_ref"] for task in tasks} == {"Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"}
    for task in tasks:
        command = render_command(
            task.recipe.command.run,
            build_substitution_map(task.variant, list(range(8)), "/repo", "/task"),
        )
        assert "--loop-targets" not in command
        assert "--verify-working-golden tune-winners" in command
        assert "--bench-backends eager,tcompile,emmy" in command


def test_convergence_check_is_one_shape_and_three_seeds(project_root) -> None:
    tasks = _kernel_tasks(project_root, "convergence")
    assert len(tasks) == 3
    assert {task.variant.params["seed"] for task in tasks} == {0, 1, 2}
    assert all(task.recipe.deploy.gpu == "NVIDIA H200 141GB" for task in tasks)
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)
    assert all(task.variant.params["seq_len"] == 512 for task in tasks)

    fp8_tasks = _kernel_tasks(project_root, "fp8-convergence")
    assert len(fp8_tasks) == 3
    assert {task.variant.params["seed"] for task in fp8_tasks} == {0, 1, 2}
    assert all(task.variant.params["model_ref"].startswith("RedHatAI/Qwen3-0.6B-FP8-dynamic@") for task in fp8_tasks)
    assert all(task.variant.params["input_dtype"] == "f8e4m3" for task in fp8_tasks)


def test_h200_independent_compiler_and_search_ablation_are_executable(project_root) -> None:
    common_dir = _experiment(project_root, "kernels")
    common = load_recipe(common_dir).command.run
    h200 = next(task for task in _kernel_tasks(project_root, "common") if task.recipe.deploy.gpu == "NVIDIA H200 141GB")
    assert 'pip install "hidet==0.6.1"' in common
    assert h200.variant.params["optional_backend"] == "hidet"

    tasks = _kernel_tasks(project_root, "search-ablation")
    assert len(tasks) == 4
    assert {(task.variant.params["budget"], task.variant.params["patience"]) for task in tasks} == {
        (0, 0),
        (4, 4),
        (12, 4),
        (48, 12),
    }
    cold = next(task for task in tasks if task.variant.params["budget"] == 0)
    assert cold.variant.params["verification_mode"] == "cold-greedy"
    assert all(task.recipe.deploy.gpu == "NVIDIA H200 141GB" for task in tasks)
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)


def test_every_command_variant_renders(project_root) -> None:
    root = Path(project_root) / EXP
    rendered = 0
    for recipe_path in sorted(root.glob("*/recipe.yaml")):
        for task in enumerate_tasks([str(recipe_path.parent)]):
            if task.recipe.command is None:
                continue
            substitutions = build_substitution_map(
                task.variant,
                list(range(task.recipe.deploy.gpu_count)),
                "/repo",
                "/task",
            )
            command = render_command(task.recipe.command.run, substitutions)
            assert "/repo" in command
            assert "/task" in command
            subprocess.run(["bash", "-n"], input=command, text=True, check=True)
            rendered += 1
    assert rendered == 42


def test_gemma_serving_ab_has_four_points_per_lane(project_root) -> None:
    tasks = enumerate_tasks([_experiment(project_root, "serving_gemma4_rtx5090")])
    assert len(tasks) == 40

    stock = [task for task in tasks if task.recipe.benchmark.comparison_arm == "stock"]
    emmy = [task for task in tasks if task.recipe.benchmark.comparison_arm == "emmy"]
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
        assert all(task.recipe.benchmark.require_complete_requests is True for task in lane)
        assert all(task.recipe.benchmark.require_output_equivalence is True for task in lane)
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
        lane = task.recipe.benchmark.comparison_arm
        repeats_by_lane_and_point.setdefault((lane, point), set()).add(task.recipe.benchmark.process_repeat)
    assert len(repeats_by_lane_and_point) == 8
    assert all(repeats == {0, 1, 2, 3, 4} for repeats in repeats_by_lane_and_point.values())
    assert all(task.recipe.aggregate is None for task in tasks)
    for point in expected_points:
        point_tasks = [
            task
            for task in tasks
            if (task.recipe.benchmark.random_input_len, task.recipe.benchmark.random_output_len, task.recipe.benchmark.max_concurrency)
            == point
        ]
        assert [task.recipe.benchmark.comparison_order for task in point_tasks] == list(range(10))


def test_every_publication_serving_recipe_requires_complete_requests(project_root) -> None:
    root = Path(project_root) / EXP
    recipe_dirs = sorted(path.parent for path in root.glob("serving_*/recipe.yaml"))
    assert len(recipe_dirs) == 8
    for recipe_dir in recipe_dirs:
        tasks = enumerate_tasks([str(recipe_dir)])
        assert tasks
        assert all(task.recipe.benchmark.require_complete_requests is True for task in tasks), recipe_dir.name


def test_gemma_image_provenance_pins_shared_vllm_revision(project_root) -> None:
    directory = Path(project_root) / EXP / "serving_gemma4_rtx5090"
    text = (directory / "IMAGE_PROVENANCE.md").read_text(encoding="utf-8")
    provenance = json.loads((directory / "IMAGE_PROVENANCE.json").read_text(encoding="utf-8"))
    tasks = enumerate_tasks([str(directory)])
    assert "91df0fad4dc98a67c7659d9dbd915245d5c43d96" in text
    assert "sha256:3a1e7f5904e1a1192a02aa0086ceaffc33985d7044c7bb25b3a43d61bdbe3ac0" in text
    assert "sha256:5add12d3b7f4673790b435b76635082433538e3615fbc40227fa1c0db64c9ff3" in text
    assert {task.recipe.engine.llm.vllm.image for task in tasks} == {provenance["image"]}
    assert {task.recipe.engine.llm.vllm.entrypoint for task in tasks if task.recipe.benchmark.comparison_arm == "stock"} == {
        provenance["stock_entrypoint"]
    }
    assert all(
        f'"architectures":["{provenance["emmy_architecture"]}"]' in task.recipe.engine.llm.vllm.extra_args
        for task in tasks
        if task.recipe.benchmark.comparison_arm == "emmy"
    )
