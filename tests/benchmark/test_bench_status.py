"""Generic benchmark process-status tests."""

import pytest

from emmy.commands.bench import _raise_on_bench_failure


def test_failed_task_propagates_to_bench_exit() -> None:
    with pytest.raises(SystemExit, match="1"):
        _raise_on_bench_failure(task_failed=True)


def test_successful_tasks_do_not_raise() -> None:
    _raise_on_bench_failure(task_failed=False)
