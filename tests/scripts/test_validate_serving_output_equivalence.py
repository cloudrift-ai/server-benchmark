import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("validate_serving_output_equivalence", SCRIPTS / "validate_serving_output_equivalence.py")
assert _SPEC is not None and _SPEC.loader is not None
validator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validator
_SPEC.loader.exec_module(validator)


def _write_prompts(path):
    path.write_text('{"id":"p","prompt":"hello","max_tokens":8}\n', encoding="utf-8")


def _write_results(directory, arm, *, changed=None, wrong_seed=False):
    paths = []
    for workload in sorted(validator._EXPECTED_WORKLOADS):
        for repeat in range(5):
            record = {
                "schema_version": 1,
                "arm": arm,
                "repeat": repeat,
                "workload": workload,
                "prompt_id": "p",
                "request": {
                    "model": "google/gemma-4-12B-it",
                    "prompt": "hello",
                    "max_tokens": 8,
                    "temperature": 0,
                    "seed": 1 if wrong_seed else 0,
                    "n": 1,
                    "stream": False,
                },
                "text": "different" if changed == (workload, repeat) else "world",
                "completion_tokens": 1,
                "finish_reason": "stop",
            }
            path = directory / f"{arm}-{workload}-{repeat}.json"
            path.write_text(
                json.dumps(
                    {
                        "recipe": {"benchmark": {"num_prompts": 8}},
                        "metrics": {"successful_requests": 8, "failed_requests": 0},
                        "output_probe": [record],
                    }
                ),
                encoding="utf-8",
            )
            paths.append(path)
    return paths


def test_validate_accepts_exact_five_repeat_outputs(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts)
    results = _write_results(tmp_path, "stock") + _write_results(tmp_path, "emmy")

    validator.validate_results(prompts, results)


def test_validate_rejects_one_semantic_difference(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts)
    workload = next(iter(validator._EXPECTED_WORKLOADS))
    results = _write_results(tmp_path, "stock") + _write_results(tmp_path, "emmy", changed=(workload, 3))

    with pytest.raises(validator.OutputEquivalenceError, match="semantic output mismatch"):
        validator.validate_results(prompts, results)


def test_validate_rejects_same_wrong_request_in_both_arms(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts)
    results = _write_results(tmp_path, "stock", wrong_seed=True) + _write_results(tmp_path, "emmy", wrong_seed=True)

    with pytest.raises(validator.OutputEquivalenceError, match="frozen prompt contract"):
        validator.validate_results(prompts, results)
