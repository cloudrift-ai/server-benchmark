#!/usr/bin/env python3
"""Validate the same-image Gemma stock/Emmy serving A/B contract."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from emmy.benchmark.tasks import enumerate_tasks
from emmy.logging_setup import setup_cli_logging

logger = logging.getLogger(__name__)

_IMAGE_RE = re.compile(r".+@sha256:[0-9a-f]{64}")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_EXPECTED_TOKENS = {
    (256, 256, 64): 2112,
    (4096, 4096, 1): 4128,
    (4096, 4096, 8): 2056,
    (8192, 256, 4): 4104,
}
_OUTPUT_PROBE = "experiments/golden-bench-2026/quality_gemma4_rtx5090/prompts.jsonl"


class GemmaABError(ValueError):
    """Raised when the Gemma A/B includes an image, scheduler, or order confounder."""


def _env(value) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    result = {}
    for assignment in (value or "").split():
        if "=" in assignment:
            key, item = assignment.split("=", 1)
            result[key] = item
    return result


def validate(recipe_dir: Path, provenance_path: Path) -> dict:
    """Validate the expanded task matrix and return a compact summary."""
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GemmaABError(f"cannot load image provenance: {exc}") from exc
    if provenance.get("schema_version") != 1:
        raise GemmaABError("image provenance schema_version must be 1")
    image = provenance.get("image")
    if not isinstance(image, str) or not _IMAGE_RE.fullmatch(image):
        raise GemmaABError("image provenance must pin one immutable image")
    entrypoint = provenance.get("stock_entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise GemmaABError("image provenance must pin the stock entrypoint")
    if not _REVISION_RE.fullmatch(str(provenance.get("vllm_source_revision", ""))):
        raise GemmaABError("image provenance must pin the full vLLM source revision")

    tasks = enumerate_tasks([str(recipe_dir)])
    if len(tasks) != 40:
        raise GemmaABError(f"expected 40 fresh-process tasks, got {len(tasks)}")
    grouped = defaultdict(list)
    for task in tasks:
        recipe = task.recipe
        llm = recipe.engine.llm
        benchmark = recipe.benchmark
        if llm.vllm.image != image:
            raise GemmaABError("all stock and Emmy tasks must use the same immutable image")
        if benchmark.repeats != 1 or benchmark.seed != 0 or benchmark.temperature != 0 or not benchmark.ignore_eos:
            raise GemmaABError("benchmark client controls are not frozen")
        if benchmark.output_probe_file != _OUTPUT_PROBE:
            raise GemmaABError("every fresh task must capture the frozen output probe before teardown")
        point = (benchmark.random_input_len, benchmark.random_output_len, benchmark.max_concurrency)
        if point not in _EXPECTED_TOKENS:
            raise GemmaABError(f"unexpected workload point {point}")
        if f"--max-num-batched-tokens {_EXPECTED_TOKENS[point]}" not in llm.vllm.extra_args:
            raise GemmaABError(f"scheduler token cap is not matched for {point}")
        env = _env(llm.vllm.extra_env)
        arm = env.get("EMMY_BENCH_ARM")
        if arm not in {"stock", "emmy"}:
            raise GemmaABError("every task must declare its arm")
        if "EMMY_FAST_MATH" in env:
            raise GemmaABError("fast math is outside the preregistered causal A/B")
        if arm == "stock":
            if llm.vllm.entrypoint != entrypoint or "EmmyGenModel" in llm.vllm.extra_args:
                raise GemmaABError("stock must use the same image's stock entrypoint without Emmy routing")
        elif llm.vllm.entrypoint is not None or "EmmyGenModel" not in llm.vllm.extra_args:
            raise GemmaABError("Emmy must use the image entrypoint and EmmyGenModel route")
        try:
            repeat = int(env["EMMY_BENCH_PROCESS_REPEAT"])
            order = int(env["EMMY_BENCH_ORDER_INDEX"])
        except (KeyError, ValueError) as exc:
            raise GemmaABError("every task must record integer repeat and order fields") from exc
        grouped[point].append((order, repeat, arm))

    expected_order = [
        (0, 0, "stock"),
        (1, 0, "emmy"),
        (2, 1, "emmy"),
        (3, 1, "stock"),
        (4, 2, "stock"),
        (5, 2, "emmy"),
        (6, 3, "emmy"),
        (7, 3, "stock"),
        (8, 4, "stock"),
        (9, 4, "emmy"),
    ]
    if set(grouped) != set(_EXPECTED_TOKENS):
        raise GemmaABError("matrix must contain every preregistered workload")
    for point, order in grouped.items():
        if order != expected_order:
            raise GemmaABError(f"workload {point} does not use the balanced preregistered arm order")
    aggregate = tasks[0].recipe.aggregate
    if aggregate is None or "validate_serving_output_equivalence.py" not in aggregate.run or "--results" not in aggregate.run:
        raise GemmaABError("recipe must aggregate every task's output-equivalence evidence")
    return {"image": image, "tasks": len(tasks), "workloads": len(grouped), "order": expected_order}


def main() -> int:
    """Run the command-line validator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe_dir", type=Path)
    parser.add_argument("provenance", type=Path)
    args = parser.parse_args()
    setup_cli_logging()
    try:
        summary = validate(args.recipe_dir, args.provenance)
    except GemmaABError as exc:
        logger.error("Gemma A/B validation failed: %s", exc)
        return 1
    logger.info("Validated %d same-image tasks across %d workloads", summary["tasks"], summary["workloads"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
