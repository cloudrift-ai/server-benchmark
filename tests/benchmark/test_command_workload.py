"""Unit tests for the command workload module."""

from pathlib import Path

from pathlib import Path

import pytest

from emmy.benchmark.command_workload import _local_result_name, build_substitution_map, render_command, run_command_workload
from emmy.planner import BenchmarkTask
from emmy.planner.variant import Variant
from emmy.recipe.types import CommandConfig, Recipe


def _variant(params):
    return Variant(params=params)


def test_local_result_name_top_level_file_keeps_flat_name():
    assert _local_result_name("rtx5090x1", "accum_error.json") == "rtx5090x1_accum_error.json"


def test_local_result_name_subdir_files_do_not_collide():
    std = _local_result_name("rtx4090x1", "std/golden_bench.json")
    fm = _local_result_name("rtx4090x1", "fm/golden_bench.json")
    assert std == "rtx4090x1_std_golden_bench.json"
    assert fm == "rtx4090x1_fm_golden_bench.json"
    assert std != fm


def test_build_substitution_map_flattens_dot_keys():
    v = _variant({"deploy.gpu": "NVIDIA GeForce RTX 5090", "deploy.gpu_count": 1, "marker": "a"})
    subs = build_substitution_map(v, [0], repo_dir="/tmp/repo", task_dir="/tmp/task")
    assert subs["gpu"] == "NVIDIA GeForce RTX 5090"
    assert subs["gpu_count"] == "1"
    assert subs["marker"] == "a"
    assert subs["task_dir"] == "/tmp/task"
    assert subs["repo_dir"] == "/tmp/repo"
    assert subs["gpu_device_ids"] == "0"


def test_build_substitution_map_no_repo_dir():
    v = _variant({"deploy.gpu": "x", "deploy.gpu_count": 1})
    subs = build_substitution_map(v, [0, 1], repo_dir=None, task_dir="/t")
    assert "repo_dir" not in subs
    assert subs["gpu_device_ids"] == "0,1"


def test_build_substitution_map_leaf_conflict():
    v = _variant({"deploy.gpu": "x", "deploy.gpu_count": 1, "extra.gpu": "y"})
    with pytest.raises(ValueError, match="conflicting leaf name 'gpu'"):
        build_substitution_map(v, [0], repo_dir=None, task_dir="/t")


def test_render_command_basic():
    out = render_command("echo $marker > $task_dir/out", {"marker": "a", "task_dir": "/tmp"})
    assert out == "echo a > /tmp/out"


def test_render_command_missing_var():
    with pytest.raises(ValueError, match=r"undefined variable: \$missing"):
        render_command("echo $missing", {"task_dir": "/tmp"})


def test_render_command_passes_through_shell_metachars():
    """`$(...)`, `${VAR:-default}`, `$1`, and `$$` must survive rendering."""
    # Note: `$$` is Template's escape for a literal `$`, so it renders as `$`.
    out = render_command(
        'echo $marker $(hostname) ${OTHER:-x} "$1" $$',
        {"marker": "a"},
    )
    assert out == 'echo a $(hostname) ${OTHER:-x} "$1" $'


def test_render_command_repo_dir_unavailable():
    """When staging is empty, $repo_dir is omitted from subs and triggers a friendly error."""
    v = _variant({"deploy.gpu": "x", "deploy.gpu_count": 1})
    subs = build_substitution_map(v, [0], repo_dir=None, task_dir="/t")
    with pytest.raises(ValueError, match=r"undefined variable: \$repo_dir"):
        render_command("cd $repo_dir && make", subs)


@pytest.mark.asyncio
async def test_failed_command_still_pulls_preserved_result(monkeypatch, tmp_path):
    recipe = Recipe(command=CommandConfig(run="false", result_files=["case/artifacts.tar.gz"]))
    variant = _variant({"deploy.gpu": "NVIDIA GeForce RTX 5090", "deploy.gpu_count": 1})
    task = BenchmarkTask(recipe_dir="experiment", variant=variant, recipe=recipe, run_dir=tmp_path)

    async def run_cmd(command, **_kwargs):
        return (7, "", "failed") if command == "false" else (0, "", "")

    async def fake_scp(server, ssh_key, ssh_port, remote, local):
        del server, ssh_key, ssh_port, remote
        Path(local).write_bytes(b"archive")
        return 0, ""

    monkeypatch.setattr("emmy.provisioning.ssh_transport.scp_from_remote", fake_scp)
    success, info = await run_command_workload(
        task,
        run_cmd,
        repo_dir=None,
        task_dir="/remote/task",
        gpu_device_ids=[0],
        server="host",
        ssh_key="key",
        ssh_port=22,
    )

    assert success is False
    assert info["exit_code"] == 7
    assert info["result_paths"] == [str(tmp_path / "rtx5090x1_case_artifacts.tar.gz")]
    assert (tmp_path / "rtx5090x1_case_artifacts.tar.gz").read_bytes() == b"archive"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "scp"])
async def test_strict_result_transfer_fails_closed(monkeypatch, tmp_path, failure):
    recipe = Recipe(command=CommandConfig(run="true", result_files=["artifacts/*.tar.gz"], strict=True))
    task = BenchmarkTask(
        recipe_dir="experiment",
        variant=_variant({"deploy.gpu": "NVIDIA GeForce RTX 5090", "deploy.gpu_count": 1}),
        recipe=recipe,
        run_dir=tmp_path,
    )

    async def run_cmd(command, **_kwargs):
        if "for f in" in command:
            return 0, "" if failure == "missing" else "artifacts/results.tar.gz\n", ""
        return 0, "", ""

    async def fake_scp(*_args):
        return 1, "transfer failed"

    monkeypatch.setattr("emmy.provisioning.ssh_transport.scp_from_remote", fake_scp)
    success, info = await run_command_workload(
        task,
        run_cmd,
        repo_dir=None,
        task_dir="/remote/task",
        gpu_device_ids=[0],
        server="host",
        ssh_key="key",
        ssh_port=22,
    )

    assert success is False
    assert info["result_errors"]
