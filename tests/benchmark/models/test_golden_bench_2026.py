"""Configuration checks for the 2026 compiler-submission experiments."""

import subprocess
from pathlib import Path

from emmy.benchmark.command_workload import build_substitution_map, render_command
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
        "NVIDIA Tesla V100 SXM3 32GB",
        "NVIDIA A100 80GB",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 5090",
        "NVIDIA H200 141GB",
        "NVIDIA B200",
    }
    recipe_dir = _experiment(project_root, "kernels")
    recipe = load_recipe(recipe_dir)
    tasks = _kernel_tasks(project_root, "common")
    assert recipe.kind == "command"
    assert len(tasks) == len(platforms) * 2
    assert {task.recipe.deploy.gpu for task in tasks} == platforms
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)
    assert {task.variant.params["seq_len"] for task in tasks} == {1, 512}
    assert {task.variant.params["model_ref"] for task in tasks} == {"Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"}
    a100_tasks = [task for task in tasks if task.recipe.deploy.gpu == "NVIDIA A100 80GB"]
    searched_tasks = [task for task in tasks if task.recipe.deploy.gpu != "NVIDIA A100 80GB"]
    assert {task.variant.params["golden"] for task in a100_tasks} == {
        "qwen3-06b-s1_a100",
        "qwen3-06b-s512_a100",
    }
    assert all(task.variant.params["budget"] == 0 for task in a100_tasks)
    assert all(task.variant.params["patience"] == 0 for task in a100_tasks)
    assert all(task.variant.params["golden"] == "" for task in searched_tasks)
    assert all(task.variant.params["budget"] == 12 for task in searched_tasks)
    assert all(task.variant.params["patience"] == 4 for task in searched_tasks)

    run = recipe.command.run
    assert "./venv/bin/emmy trace" in run
    assert "./venv/bin/emmy tune" in run
    assert "./venv/bin/emmy run" in run
    assert "for repeat in 0 1 2 3 4" in run
    assert "--golden $task_dir/working.yaml --bench --strict" in run
    assert "scripts/" not in run
    assert recipe.command.stage == [
        "emmy",
        "pyproject.toml",
        "requirements.txt",
        "Makefile",
        "experiments/golden-bench-2026/kernels/recipe.yaml",
        "experiments/golden-bench-2026/kernels/golden",
    ]
    assert recipe.command.strict is True
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
    assert all(task.variant.params["fp8_mma"] == 1 for task in tasks)
    for task in tasks:
        command = render_command(
            task.recipe.command.run,
            build_substitution_map(task.variant, [0], "/repo", "/task"),
        )
        assert "--require-kernel-source" not in command
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


def test_serving_systems_are_pinned_and_controlled(project_root) -> None:
    systems = {
        "serving_deepseek_v4_flash_0731_v100x16": (
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            "NVIDIA Tesla V100 SXM3 32GB",
            16,
        ),
        "serving_qwen38_27b_awq_rtx4090": (
            "philbert440/Qwen3.8-27B-W4A16-AWQ",
            "7908d42a71077a5e4dc458f273682b12dfe384a0",
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
            repeats_by_point.setdefault(point, set()).add(task.variant.params["repeat"])
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


def test_qwen3_native_nvfp4_qualification_requires_optimized_w4a4_kernel(project_root) -> None:
    tasks = enumerate_tasks([_experiment(project_root, "serving_qwen3_8b_nvfp4_rtx5090")])
    for task in tasks:
        llm = task.recipe.engine.llm
        assert llm.tensor_parallel_size == 1
        assert task.recipe.model.revision == "ccd10a893cbca613259517c3efe08e151ddf2b8e"
        assert llm.vllm.image == "vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f"
        assert "--quantization modelopt" in llm.vllm.extra_args


def test_b200_nvfp4_qualification_pins_native_capable_configuration(project_root) -> None:
    tasks = enumerate_tasks([_experiment(project_root, "serving_glm52_nvfp4_b200x8")])
    for task in tasks:
        llm = task.recipe.engine.llm
        assert task.recipe.model.revision == "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
        assert llm.tensor_parallel_size == 8
        assert "--enable-expert-parallel" in llm.vllm.extra_args
        assert llm.vllm.image == "vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f"


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
        assert "--golden /task/working.yaml --bench --strict" in command
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


def test_search_ablation_is_executable(project_root) -> None:
    tasks = _kernel_tasks(project_root, "search-ablation")
    assert len(tasks) == 4
    assert {(task.variant.params["budget"], task.variant.params["patience"]) for task in tasks} == {
        (0, 0),
        (4, 4),
        (12, 4),
        (48, 12),
    }
    assert all("--golden $task_dir/working.yaml --bench --strict" in task.recipe.command.run for task in tasks)
    assert all(task.recipe.deploy.gpu == "NVIDIA H200 141GB" for task in tasks)
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)


def test_mpk_megakernel_lane_is_pinned_and_paired(project_root) -> None:
    directory = _experiment(project_root, "serving_mpk_qwen3_8b_a100")
    tasks = enumerate_tasks([directory])
    assert len(tasks) == 1
    task = tasks[0]
    recipe = load_recipe(directory)
    assert recipe.kind == "command"
    assert task.recipe.deploy.gpu == "NVIDIA A100 80GB"
    assert task.recipe.deploy.gpu_count == 1

    run = recipe.command.run
    # Pinned external sources: the mirage mpk-branch revision and the Qwen3-8B checkpoint revision.
    assert "5c28cc68dc621cc9448c5c9882ef9e21fdc85884" in run
    assert "b968826d9c46dd6066d109eabc6255188de91218" in run
    # The MPK baseline/megakernel demo pair and stock vLLM are the only paths.
    assert "demo/qwen3/demo.py" in run
    assert "--use-mirage" in run
    assert "vllm==0.23.0" in run
    assert "emmy serve" not in run
    assert "EMMY_GEN_DECODE_BUCKET" not in run
    assert "for repeat in 0 1 2 3 4" in run
    # The suite's deterministic serving controls on the single-stream decode point.
    assert "--ignore-eos --temperature 0 --seed 0" in run
    assert "--max-concurrency 1" in run
    assert "--no-enable-prefix-caching" in run


def test_neptune_emmy_pytorch_a100_share_one_experiment(project_root) -> None:
    directory = _experiment(project_root, "compiler_neptune_emmy_pytorch_a100")
    tasks = enumerate_tasks([directory])
    recipe = load_recipe(directory)
    assert recipe.kind == "command"
    assert {task.recipe.deploy.gpu for task in tasks} == {"NVIDIA A100 80GB"}
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)

    # One row per (lane, operator): the artifact's ten operators, and the five common operators
    # compared against current PyTorch from a pre-tuned golden.
    common = ("prefill_global", "prefill_causal", "prefill_gqa", "decode_causal", "decode_gqa")
    artifact_only = ("prefill_alibi", "decode_alibi", "prefill_softcap", "decode_softcap", "prefill_windowed")
    lanes: dict[str, set[str]] = {}
    for task in tasks:
        lanes.setdefault(task.variant.params["lane"], set()).add(task.variant.params["operator"])
    assert lanes == {
        "neptune": set(common + artifact_only),
        "emmy": set(common),
    }
    assert len(tasks) == 15

    run = recipe.command.run
    assert "evanzhao16/neptune-env@sha256:724d07594bc817f0fe94267b2d0dbdc6e29d3ae4a7e3516e553a6d9327bfebca" in run
    assert "3aa55c12ac822337e630b809b0d9eabb11eee5d3" in run
    assert "torch==2.13.0" in run
    assert "EMMY_TUNE_DB=$task_dir/autotune.db" in run
    assert "pip freeze --all" in run
    assert "tar -C $task_dir" in run
    # Each row runs exactly one lane for exactly one operator.
    assert 'case "$lane" in' in run
    assert 'bash /experiment/run.sh "$operator"' in run
    assert '"$$EXPERIMENT/run_emmy.sh" $repo_dir/venv/bin/emmy "$operator"' in run
    # Tuning is the tune-kernels skill's job: the recipe only replays what it committed.
    assert "emmy tune" not in run
    assert recipe.command.stage == [
        "emmy",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "experiments/golden-bench-2026/compiler_neptune_emmy_pytorch_a100/run.sh",
        "experiments/golden-bench-2026/compiler_neptune_emmy_pytorch_a100/run_neptune.py",
        "experiments/golden-bench-2026/compiler_neptune_emmy_pytorch_a100/operators.sh",
        "experiments/golden-bench-2026/compiler_neptune_emmy_pytorch_a100/run_emmy.sh",
        "experiments/golden-bench-2026/compiler_neptune_emmy_pytorch_a100/run_pytorch.py",
        "experiments/golden-bench-2026/compiler_neptune_emmy_pytorch_a100/golden",
    ]
    assert recipe.command.strict is True
    assert recipe.command.result_files == ["artifacts.tar.gz"]
    assert "git apply" not in run

    assert not (Path(directory) / "neptune-inductor.patch").exists()
    assert not (Path(project_root) / EXP / "compiler_neptune_inductor_a100").exists()
    assert not (Path(project_root) / EXP / "compiler_neptune_emmy_tcompile_a100").exists()

    # The artifact runner takes its one operator from the row instead of looping over a private list.
    neptune_runner = (Path(directory) / "run.sh").read_text()
    assert "operator=$1" in neptune_runner
    assert "operators=(" not in neptune_runner
    assert "sequence_lengths=(256 512 1024 2048 4096 8192 16384 32768)" in neptune_runner
    assert "--n-trials 128" in neptune_runner
    assert 'nsys profile -o "/results/profiles/$setup" --trace=cuda,nvtx,osrt --wait=primary' in neptune_runner
    assert 'profile "$operator" "1,$sequence_length" --repeat 15' in neptune_runner
    assert "neptune-setup-status.tsv" in neptune_runner
    assert "tune_status=ok:no-valid-schedule" in neptune_runner
    assert 'profile_status="$profile_status:mismatch"' in neptune_runner
    assert 'profile_status="$profile_status:runner-failure"' in neptune_runner
    assert 'test "$successful_profiles" -gt 0' in neptune_runner
    assert 'torch.__version__.split("+")[0]' in neptune_runner
    assert '"2.6.0"' in neptune_runner
    assert "git apply" not in neptune_runner

    neptune_entry = (Path(directory) / "run_neptune.py").read_text()
    assert "NVIDIA A100-SXM4-80GB" in neptune_entry
    assert "NVIDIA A100-SXM4-40GB" in neptune_entry
    assert "NeptuneGQARunner.e = ours.NeptuneGQARunner.create_flex_from_schedulers" in neptune_entry
    assert 'runpy.run_module("scripts.neptune_bench", run_name="__main__")' in neptune_entry

    # One checked-in definition of the common operators, read both by the comparison lane and by the
    # trace that produces a committed golden, so the tuned and benched programs cannot drift apart.
    operators_path = Path(directory) / "operators.sh"
    assert operators_path.stat().st_mode & 0o111
    operators = operators_path.read_text()
    for operator in common:
        assert operator in operators
    for excluded in artifact_only:
        assert excluded not in operators
    assert "SEQUENCE_LENGTHS=(256 512 1024 2048 4096 8192 16384 32768)" in operators
    assert "enable_gqa=True" in operators
    assert "q_length=1" in operators
    assert "q.reshape(1,8,8,1,128)" in operators
    assert "is_causal=False).reshape(1,64,1,128)" in operators
    assert 'operator_code "$1" "$2" || exit' in operators
    assert not (Path(directory) / "run_tune.sh").exists()

    # The comparison lane only measures: it replays the committed golden and never tunes.
    emmy_runner_path = Path(directory) / "run_emmy.sh"
    assert emmy_runner_path.stat().st_mode & 0o111
    emmy_runner = emmy_runner_path.read_text()
    assert "operators.sh" in emmy_runner
    assert '"$emmy" run --golden "$golden" --bench --bench-backends emmy' in emmy_runner
    assert 'run --golden "$golden" --bench --strict' not in emmy_runner
    assert '"$emmy" run -c "$source_code" --bench --strict --bench-backends eager,tcompile,emmy' in emmy_runner
    assert "emmy tune" not in emmy_runner
    assert "timeout --signal=TERM --kill-after=30s 600s" in emmy_runner
    assert "setup-status.tsv" in emmy_runner
    assert 'test "$missing_goldens" -eq 0' in emmy_runner
    assert 'test "$successful_setups" -gt 0' in emmy_runner
    assert "run_pytorch.py" in emmy_runner
    assert 'reference="pytorch-only:emmy-failed:$reference_status"' in emmy_runner

    pytorch_runner = (Path(directory) / "run_pytorch.py").read_text()
    assert 'mode="max-autotune-no-cudagraphs"' in pytorch_runner
    assert "torch.testing.assert_close" in pytorch_runner
    assert '"captured_whole_forward"' in pytorch_runner


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
    assert rendered == 58


def test_gemma_serving_ab_has_four_points_per_lane(project_root) -> None:
    tasks = enumerate_tasks([_experiment(project_root, "serving_gemma4_rtx5090")])
    assert len(tasks) == 40

    stock = [task for task in tasks if task.variant.params["arm"] == "stock"]
    emmy = [task for task in tasks if task.variant.params["arm"] == "emmy"]
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
        lane = task.variant.params["arm"]
        repeats_by_lane_and_point.setdefault((lane, point), set()).add(task.variant.params["repeat"])
    assert len(repeats_by_lane_and_point) == 8
    assert all(repeats == {0, 1, 2, 3, 4} for repeats in repeats_by_lane_and_point.values())


def test_gemma_arms_share_one_immutable_image(project_root) -> None:
    directory = Path(project_root) / EXP / "serving_gemma4_rtx5090"
    tasks = enumerate_tasks([str(directory)])
    assert {task.recipe.engine.llm.vllm.image for task in tasks} == {
        "cloudriftai/vllm-emmy-gemma-4-12b-it@sha256:5add12d3b7f4673790b435b76635082433538e3615fbc40227fa1c0db64c9ff3"
    }
    assert {task.recipe.engine.llm.vllm.entrypoint for task in tasks if task.variant.params["arm"] == "stock"} == {
        "python3 -m vllm.entrypoints.openai.api_server"
    }
    assert all(
        '"architectures":["EmmyGenModel"]' in task.recipe.engine.llm.vllm.extra_args
        for task in tasks
        if task.variant.params["arm"] == "emmy"
    )
