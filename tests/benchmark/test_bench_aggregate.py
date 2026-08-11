from pathlib import Path
from types import SimpleNamespace

import pytest

from emmy.commands.bench import _raise_on_bench_failure, _run_recipe_aggregate
from emmy.recipe.types import AggregateConfig, Recipe


def test_failed_aggregate_propagates_to_bench_exit(tmp_path):
    recipe = Recipe(aggregate=AggregateConfig(run="validate $run_dir", timeout=60))
    commands = []

    def failed_run(command, *, shell, timeout):
        commands.append((command, shell, timeout))
        return SimpleNamespace(returncode=7)

    passed = _run_recipe_aggregate(
        "experiments/gemma",
        Path(tmp_path),
        recipe,
        dry_run=False,
        logger=SimpleNamespace(info=lambda _message: None, error=lambda _message: None),
        run=failed_run,
    )

    assert passed is False
    assert commands == [(f"validate {tmp_path}", True, 60)]
    with pytest.raises(SystemExit) as exc:
        _raise_on_bench_failure(task_failed=False, aggregate_failed=True)
    assert exc.value.code == 1


def test_failed_task_propagates_to_bench_exit():
    with pytest.raises(SystemExit) as exc:
        _raise_on_bench_failure(task_failed=True, aggregate_failed=False)
    assert exc.value.code == 1


def test_successful_tasks_and_aggregate_do_not_raise():
    _raise_on_bench_failure(task_failed=False, aggregate_failed=False)
