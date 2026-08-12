"""Tests for the built-in serving-output equivalence gate."""

import json

import pytest

from emmy.benchmark.output_equivalence import (
    OutputEquivalenceError,
    validate_output_equivalence,
    validate_output_equivalence_selection,
)
from emmy.planner import BenchmarkTask
from emmy.planner.variant import Variant
from emmy.recipe.types import Recipe


def _task(tmp_path, arm: str, repeat: int, input_len: int = 256) -> BenchmarkTask:
    recipe = Recipe.from_dict(
        {
            "model": {"huggingface": "org/model"},
            "engine": {"llm": {"vllm": {"image": "image"}}},
            "benchmark": {
                "num_prompts": 8,
                "random_input_len": input_len,
                "random_output_len": 64,
                "max_concurrency": 4,
                "output_equivalence_file": str(tmp_path / "prompts.jsonl"),
                "comparison_arm": arm,
                "process_repeat": repeat,
            },
            "deploy": {"gpu": "NVIDIA H200", "gpu_count": 1},
        }
    )
    variant = Variant(
        params={
            "deploy.gpu": "NVIDIA H200",
            "deploy.gpu_count": 1,
            "arm": arm,
            "repeat": repeat,
            "input_len": input_len,
        }
    )
    return BenchmarkTask("experiment", variant, recipe, run_dir=tmp_path)


def _write_result(task: BenchmarkTask, *, text: str = "world", seed: int = 0) -> None:
    bench = task.recipe.benchmark
    record = {
        "schema_version": 1,
        "arm": bench.comparison_arm,
        "repeat": bench.process_repeat,
        "workload": f"i{bench.random_input_len}-o{bench.random_output_len}-c{bench.max_concurrency}",
        "prompt_id": "p",
        "request": {
            "model": "org/model",
            "prompt": "hello",
            "max_tokens": 8,
            "temperature": 0,
            "seed": seed,
            "n": 1,
            "stream": False,
        },
        "text": text,
        "completion_tokens": 1,
        "finish_reason": "stop",
    }
    task.json_result_path().write_text(
        json.dumps(
            {
                "metrics": {"successful_requests": bench.num_prompts, "failed_requests": 0},
                "output_probe": [record],
            }
        ),
        encoding="utf-8",
    )


def _paired_tasks(tmp_path) -> list[BenchmarkTask]:
    tasks = [_task(tmp_path, arm, repeat, input_len) for input_len in (256, 4096) for repeat in (0, 1) for arm in ("a", "b")]
    for task in tasks:
        _write_result(task)
    return tasks


def test_output_equivalence_accepts_complete_paired_tasks(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"p","prompt":"hello","max_tokens":8}\n', encoding="utf-8")

    report = validate_output_equivalence(_paired_tasks(tmp_path), prompts)

    assert report["status"] == "pass"
    assert report["arms"] == ["a", "b"]
    assert report["paired_records"] == 4


def test_output_equivalence_rejects_semantic_difference(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"p","prompt":"hello","max_tokens":8}\n', encoding="utf-8")
    tasks = _paired_tasks(tmp_path)
    changed = next(task for task in tasks if task.recipe.benchmark.comparison_arm == "b")
    _write_result(changed, text="different")

    with pytest.raises(OutputEquivalenceError, match="output mismatch"):
        validate_output_equivalence(tasks, prompts)


def test_output_equivalence_rejects_same_wrong_request_in_both_arms(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"p","prompt":"hello","max_tokens":8}\n', encoding="utf-8")
    tasks = _paired_tasks(tmp_path)
    for task in tasks:
        _write_result(task, seed=1)

    with pytest.raises(OutputEquivalenceError, match="frozen output prompt"):
        validate_output_equivalence(tasks, prompts)


def test_output_equivalence_rejects_filtered_matrix(tmp_path):
    full = _paired_tasks(tmp_path)

    with pytest.raises(OutputEquivalenceError, match="complete unfiltered recipe matrix"):
        validate_output_equivalence_selection(full[:4], full)

    validate_output_equivalence_selection(full, full)
