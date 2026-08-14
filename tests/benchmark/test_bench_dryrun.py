"""Dry-run tests for the bench command."""

import os
import re
from pathlib import Path


def test_bench_dry_run_basic(run_cli, make_bench_config, recipes_dir, tmp_path):
    config_path = make_bench_config(tmp_path)
    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench",
        recipe,
        "--config",
        config_path,
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "[dry-run]" in stdout


def test_bench_dry_run_deploy_then_benchmark(run_cli, make_bench_config, recipes_dir, tmp_path):
    config_path = make_bench_config(tmp_path)
    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench",
        recipe,
        "--config",
        config_path,
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"

    # Verify deploy steps appear
    assert "docker compose pull" in stdout
    assert "docker compose up" in stdout

    # Verify benchmark step appears with recipe params
    assert "bench serve" in stdout
    assert "--random-input-len 4000" in stdout
    assert "--random-output-len 4000" in stdout

    # Verify teardown appears
    assert "docker compose down" in stdout

    # Verify order: pull before bench, bench before teardown
    pull_idx = stdout.index("docker compose pull")
    bench_idx = stdout.index("bench serve")
    assert pull_idx < bench_idx


def test_bench_dry_run_reports_timing(run_cli, make_bench_config, recipes_dir, tmp_path):
    """The end-of-run summary includes a TIMING breakdown for successful tasks."""
    config_path = make_bench_config(tmp_path)
    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench",
        recipe,
        "--config",
        config_path,
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "TIMING" in stdout


def test_bench_multiple_recipes(run_cli, make_bench_config, recipes_dir, tmp_path):
    config_path = make_bench_config(tmp_path)
    recipe1 = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    # GLM-5.1-FP8, not 4.6: the 4.6 recipe was removed from the repo, and the test only kept
    # passing on dev machines where the directory survives as a shell of old bench run dirs.
    recipe2 = os.path.join(recipes_dir, "GLM-5.1-FP8")
    rc, stdout, stderr = run_cli(
        "bench",
        recipe1,
        recipe2,
        "--config",
        config_path,
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"


def test_bench_network_flag_dry_run(run_cli, make_bench_config, recipes_dir, tmp_path):
    """--network propagates into the CloudRift rent payload."""
    config_path = make_bench_config(tmp_path)
    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench",
        recipe,
        "--config",
        config_path,
        "--network",
        "public",
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert '"network": "public"' in stdout


def test_bench_network_flag_with_null_cloudrift_config(run_cli, recipes_dir, tmp_path):
    """`providers.cloudrift: null` (commented children only) must not crash with --network."""
    import yaml

    config = {
        "benchmark": {
            "model_dir": "/hf_models",
        },
        "providers": {"cloudrift": None},
    }
    config_path = os.path.join(str(tmp_path), "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench",
        recipe,
        "--config",
        config_path,
        "--network",
        "public",
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert '"network": "public"' in stdout


def test_bench_no_teardown_dry_run(run_cli, make_bench_config, recipes_dir, tmp_path):
    config_path = make_bench_config(tmp_path)
    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench",
        recipe,
        "--config",
        config_path,
        "--dry-run",
        "--no-teardown",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"

    # With --no-teardown, per-task teardown should be skipped
    assert "bench serve" in stdout
    assert "Tearing down..." not in stdout
    assert "Skipping VM deletion (--no-teardown)" in stdout


def test_bench_reports_timestamped_run_directory(run_cli, make_bench_config, recipes_dir, tmp_path):
    """Every run targets a timestamped directory under the recipe."""
    config_path = make_bench_config(tmp_path)
    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench",
        recipe,
        "--config",
        config_path,
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert str(Path(recipe).resolve()) in stdout
    assert "Run directory:" in stdout
    assert re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", stdout)


def test_bench_experiment_dry_run(run_cli, make_bench_config, project_root, tmp_path):
    """Experiment recipe runs successfully in dry-run mode."""
    config_path = make_bench_config(tmp_path)
    experiment = os.path.join(
        project_root,
        "experiments",
        "Qwen3-Coder-30B-A3B-Instruct-AWQ",
        "optimal_mcr_rtx5090",
    )
    rc, stdout, stderr = run_cli(
        "bench",
        experiment,
        "--config",
        config_path,
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    # Should have multiple benchmark tasks from the sweep
    assert stdout.count("bench serve") >= 2
    assert str(Path(experiment).resolve()) in stdout


def test_bench_dry_run_preserves_prior_run_directory(run_cli, make_bench_config, tmp_path):
    """A dry run must not change a prior local run directory."""
    import yaml

    # Create a minimal recipe in tmp_path so we don't pollute the repo
    recipe_dir = tmp_path / "TestRecipe"
    recipe_dir.mkdir()
    recipe = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "context_length": 8192,
                "vllm": {"image": "vllm/vllm-openai:v0.17.0"},
            }
        },
        "benchmark": {
            "max_concurrency": 128,
            "num_prompts": 256,
            "random_input_len": 4000,
            "random_output_len": 4000,
        },
        "matrices": [
            {"deploy.gpu": "NVIDIA GeForce RTX 5090", "deploy.gpu_count": 1},
        ],
    }
    (recipe_dir / "recipe.yaml").write_text(yaml.dump(recipe))
    prior_run_dir = recipe_dir / "2026-08-13_12-00-00"
    prior_run_dir.mkdir()
    marker = prior_run_dir / "last-run.log"
    marker.write_text("keep\n")

    config_path = make_bench_config(tmp_path)
    rc, stdout, stderr = run_cli(
        "bench",
        str(recipe_dir),
        "--config",
        config_path,
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert marker.read_text() == "keep\n"
    assert list(prior_run_dir.iterdir()) == [marker]


def test_bench_command_recipe_dry_run(run_cli, make_bench_config, tmp_path):
    """A command recipe expands its template and dispatches the command path."""
    import yaml

    recipe_dir = tmp_path / "CmdRecipe"
    recipe_dir.mkdir()
    recipe = {
        "command": {
            "stage": [],
            "run": "echo marker=$marker > $task_dir/result.csv\n",
            "result_files": ["result.csv"],
            "timeout": 30,
        },
        "matrices": [
            {
                "deploy.gpu": "NVIDIA GeForce RTX 5090",
                "deploy.gpu_count": 1,
                "marker": ["a", "b", "c"],
            }
        ],
    }
    (recipe_dir / "recipe.yaml").write_text(yaml.dump(recipe))

    config_path = make_bench_config(tmp_path)
    rc, stdout, stderr = run_cli(
        "bench",
        str(recipe_dir),
        "--config",
        config_path,
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    # Each variant's rendered command should appear with $marker substituted.
    assert "marker=a" in stdout
    assert "marker=b" in stdout
    assert "marker=c" in stdout
    # Inference-path machinery should not be invoked for command recipes.
    assert "bench serve" not in stdout
    assert "docker compose pull" not in stdout


def test_bench_help(run_cli):
    rc, stdout, _ = run_cli("bench", "--help")
    assert rc == 0
    assert "recipes" in stdout
    assert "--ssh-key" in stdout
    assert "--dry-run" in stdout
    assert "--config" in stdout
    assert "--max-workers" in stdout
    assert "--no-teardown" in stdout


def test_teardown_help(run_cli):
    rc, stdout, _ = run_cli("teardown", "--help")
    assert rc == 0
    assert "experiment_dir" in stdout
    assert "--ssh-key" in stdout
