#!/usr/bin/env python3
"""Build the Qwen3.5-on-Volta article manifest from Emmy benchmark JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shlex
import sys
from copy import deepcopy
from pathlib import Path

from emmy.logging_setup import setup_cli_logging

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3.5-122B-A10B"
MODEL_REVISION = "dc4d348443bc740c68e2d77492492c11606384d5"
ENGINE_NAME = "1Cat-vLLM"
ENGINE_REVISION = "91aca502d2bb1f05d9208ab2edec9fae53ff0d0b"
IMAGE_REFERENCE = "cloudriftai/1cat-vllm-sm70:1.2.2-cloudrift"
GPU_NAME = "NVIDIA Tesla V100 SXM3 32GB"

WORKLOADS = {
    (32, 256): "short_decode",
    (4096, 256): "input_4k_output_256",
    (32000, 256): "near_32k_output_256",
}
WORKLOAD_ORDER = {name: index for index, name in enumerate(WORKLOADS.values())}

REQUIRED_ENGINE_ARGS = {
    "--attention-backend": "FLASH_ATTN_V100",
    "--dtype": "half",
    "--mamba-cache-mode": "align",
    "--max-num-batched-tokens": "4096",
    "--reasoning-parser": "qwen3",
    "--revision": MODEL_REVISION,
    "--tool-call-parser": "qwen3_coder",
}
REQUIRED_ENGINE_SWITCHES = {
    "--enable-auto-tool-choice",
    "--enforce-eager",
    "--language-model-only",
}
REQUIRED_ENGINE_ENV = {
    "VLLM_SM70_ENABLE_DENSE_F16_FASTPATH": "0",
    "VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL": "0",
    "VLLM_SM70_UNQUANTIZED_MOE_0DOT3_CONFIG": "1",
    "VLLM_SM70_UNQUANTIZED_MOE_0DOT3_FUNCTIONAL": "0",
}
REQUIRED_METRICS = {
    "failed_requests",
    "mean_tpot_ms",
    "mean_ttft_ms",
    "output_token_throughput",
    "request_throughput",
    "total_token_throughput",
}

_FULL_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}")
_IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-fA-F]{64}")
_PHASE_TEXT_RE = re.compile(r"(?:^|[^a-z0-9])phase[-_ ]?([12])(?:[^a-z0-9]|$)")


class ManifestError(ValueError):
    """Raised when raw benchmark evidence cannot form a valid manifest."""


def _mapping(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    return value


def _load_result(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {path}: {exc}") from exc
    return _mapping(data, str(path))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full_revision(value: str, label: str) -> str:
    if not _FULL_REVISION_RE.fullmatch(value):
        raise ManifestError(f"{label} must be a full 40-hex revision")
    return value.lower()


def _image_digest(value: str) -> str:
    if not _IMAGE_DIGEST_RE.fullmatch(value):
        raise ManifestError("image digest must be sha256 followed by exactly 64 hex characters")
    return value.lower()


def _phase_value(value, label: str) -> int:
    if isinstance(value, bool):
        raise ManifestError(f"{label} phase must be 1 or 2")
    if isinstance(value, int) and value in (1, 2):
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"(?:phase[-_ ]?)?([12])", value.strip().lower())
        if match:
            return int(match.group(1))
    raise ManifestError(f"{label} phase must be 1 or 2")


def _source_phase(path: Path, data: dict) -> int:
    """Return the phase stamped in raw metadata or its recipe path, rejecting conflicts."""
    labels: set[int] = set()
    for name, container in [
        ("root", data),
        ("metadata", data.get("metadata")),
        ("evidence", data.get("evidence")),
        ("task", data.get("task")),
    ]:
        if isinstance(container, dict) and "phase" in container:
            labels.add(_phase_value(container["phase"], f"{path}:{name}"))

    task = data.get("task") if isinstance(data.get("task"), dict) else {}
    phase_text = " ".join((path.as_posix(), str(task.get("recipe_dir", "")))).lower()
    labels.update(int(match.group(1)) for match in _PHASE_TEXT_RE.finditer(phase_text))

    if not labels:
        raise ManifestError(f"{path}: source is not Phase1- or Phase2-labeled")
    if len(labels) != 1:
        raise ManifestError(f"{path}: conflicting source phase labels {sorted(labels)}")
    return labels.pop()


def _engine_args(extra_args: str) -> tuple[dict[str, str], set[str]]:
    """Parse the small flag subset needed to validate the frozen serving contract."""
    values: dict[str, str] = {}
    switches: set[str] = set()
    tokens = shlex.split(extra_args)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if "=" in token:
            flag, value = token.split("=", 1)
            values[flag] = value
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            values[token] = tokens[index + 1]
            index += 1
        else:
            switches.add(token)
        index += 1
    return values, switches


def _validate_engine(llm: dict, source: Path) -> dict:
    expected_scalars = {
        "context_length": 32768,
        "gpu_memory_utilization": 0.88,
        "max_concurrent_requests": 1,
        "pipeline_parallel_size": 8,
        "tensor_parallel_size": 2,
    }
    for field, expected in expected_scalars.items():
        if llm.get(field) != expected:
            raise ManifestError(f"{source}: engine.llm.{field} must be {expected!r}, got {llm.get(field)!r}")

    vllm = _mapping(llm.get("vllm"), f"{source}:recipe.engine.llm.vllm")
    if vllm.get("image") != IMAGE_REFERENCE:
        raise ManifestError(f"{source}: image must be {IMAGE_REFERENCE!r}, got {vllm.get('image')!r}")

    extra_args = vllm.get("extra_args", "")
    if not isinstance(extra_args, str):
        raise ManifestError(f"{source}: vLLM extra_args must be a string")
    values, switches = _engine_args(extra_args)
    for flag, expected in REQUIRED_ENGINE_ARGS.items():
        if values.get(flag) != expected:
            raise ManifestError(f"{source}: {flag} must be {expected!r}, got {values.get(flag)!r}")
    missing_switches = sorted(REQUIRED_ENGINE_SWITCHES - switches)
    if missing_switches:
        raise ManifestError(f"{source}: missing required engine switches: {', '.join(missing_switches)}")

    extra_env = _mapping(vllm.get("extra_env", {}), f"{source}:recipe.engine.llm.vllm.extra_env")
    for name, expected in REQUIRED_ENGINE_ENV.items():
        if str(extra_env.get(name)) != expected:
            raise ManifestError(f"{source}: {name} must be {expected!r}, got {extra_env.get(name)!r}")

    return {
        **expected_scalars,
        "extra_args": extra_args,
        "extra_env": deepcopy(extra_env),
        "image_reference": IMAGE_REFERENCE,
    }


def _validate_workload(benchmark: dict, source: Path) -> tuple[str, int, int]:
    expected = {
        "ignore_eos": True,
        "max_concurrency": 1,
        "num_prompts": 1,
        "repeats": 3,
        "seed": 0,
        "temperature": 0.0,
    }
    for field, value in expected.items():
        if benchmark.get(field) != value:
            raise ManifestError(f"{source}: benchmark.{field} must be {value!r}, got {benchmark.get(field)!r}")

    point = (benchmark.get("random_input_len"), benchmark.get("random_output_len"))
    if point not in WORKLOADS:
        raise ManifestError(f"{source}: unexpected workload {point}; expected {sorted(WORKLOADS)}")
    return WORKLOADS[point], point[0], point[1]


def _failed_request_summary(repetitions: list[dict]) -> dict:
    failed = [repeat.get("failed_requests") for repeat in repetitions]
    numeric = all(isinstance(value, int) and not isinstance(value, bool) for value in failed)
    total = sum(failed) if numeric else None
    return {
        "all_requests_succeeded": total == 0 if total is not None else None,
        "failed_requests_by_repetition": failed,
        "total_failed_requests": total,
    }


def _source_record(path: Path, data: dict, requested_phase: int, final: bool) -> tuple[dict, dict]:
    source_phase = _source_phase(path, data)
    if final and source_phase == 1:
        raise ManifestError(f"{path}: final manifests cannot consume Phase1-labeled benchmark evidence")
    if source_phase != requested_phase:
        raise ManifestError(f"{path}: source phase {source_phase} does not match requested phase {requested_phase}")

    task = _mapping(data.get("task"), f"{path}:task")
    recipe = _mapping(data.get("recipe"), f"{path}:recipe")
    model = _mapping(recipe.get("model"), f"{path}:recipe.model")
    if model.get("huggingface") != MODEL_ID:
        raise ManifestError(f"{path}: model must be {MODEL_ID!r}, got {model.get('huggingface')!r}")

    deploy = _mapping(recipe.get("deploy"), f"{path}:recipe.deploy")
    if deploy.get("gpu") != GPU_NAME or deploy.get("gpu_count") != 16:
        raise ManifestError(f"{path}: deploy target must be 16x {GPU_NAME}")
    if task.get("gpu_count") != 16:
        raise ManifestError(f"{path}: task.gpu_count must be 16")

    engine = _mapping(recipe.get("engine"), f"{path}:recipe.engine")
    llm = _mapping(engine.get("llm"), f"{path}:recipe.engine.llm")
    serving = _validate_engine(llm, path)
    benchmark = _mapping(recipe.get("benchmark"), f"{path}:recipe.benchmark")
    name, input_tokens, output_tokens = _validate_workload(benchmark, path)

    metrics = _mapping(data.get("metrics"), f"{path}:metrics")
    missing_metrics = sorted(REQUIRED_METRICS - metrics.keys())
    if missing_metrics:
        raise ManifestError(f"{path}: aggregate metrics missing {', '.join(missing_metrics)}")
    repetitions = data.get("metrics_repeats")
    if not isinstance(repetitions, list) or len(repetitions) != 3:
        raise ManifestError(f"{path}: metrics_repeats must contain exactly three raw repetitions")
    for index, repeat in enumerate(repetitions):
        _mapping(repeat, f"{path}:metrics_repeats[{index}]")
    metrics_stddev = _mapping(data.get("metrics_stddev"), f"{path}:metrics_stddev")

    record = {
        "aggregate_metrics": deepcopy(metrics),
        "errors": _failed_request_summary(repetitions),
        "input_tokens": input_tokens,
        "metrics_stddev": deepcopy(metrics_stddev),
        "name": name,
        "output_tokens": output_tokens,
        "protocol": {
            "ignore_eos": True,
            "max_concurrency": 1,
            "num_prompts": 1,
            "repeats": 3,
            "seed": 0,
            "temperature": 0.0,
        },
        "raw_repetitions": deepcopy(repetitions),
        "source": {
            "file": path.name,
            "phase": source_phase,
            "sha256": _sha256(path),
        },
        "system": deepcopy(data.get("system")),
        "timing": deepcopy(data.get("timing")),
    }
    return record, serving


def build_manifest(
    result_paths: list[Path],
    *,
    phase: int,
    final: bool,
    baseline_reason: str,
    emmy_revision: str,
    host_id: str,
    image_digest: str,
    engine_revision: str = ENGINE_REVISION,
    model_revision: str = MODEL_REVISION,
) -> dict:
    """Validate raw Emmy results and return the deterministic article manifest."""
    phase = _phase_value(phase, "manifest")
    if final and phase != 2:
        raise ManifestError("only a Phase 2 manifest can be marked final")
    if not baseline_reason.strip():
        raise ManifestError("baseline reason must be non-empty when baseline results are null")
    if not host_id.strip():
        raise ManifestError("host id must be non-empty")

    emmy_revision = _full_revision(emmy_revision, "Emmy revision")
    engine_revision = _full_revision(engine_revision, "engine revision")
    model_revision = _full_revision(model_revision, "model revision")
    if engine_revision != ENGINE_REVISION:
        raise ManifestError(f"engine revision must be the pinned 1Cat-vLLM v1.2.2 revision {ENGINE_REVISION}")
    if model_revision != MODEL_REVISION:
        raise ManifestError(f"model revision must be the pinned checkpoint revision {MODEL_REVISION}")
    image_digest = _image_digest(image_digest)

    if len(result_paths) != len(WORKLOADS):
        raise ManifestError(f"expected {len(WORKLOADS)} result files, got {len(result_paths)}")

    records: list[dict] = []
    serving_configs: list[dict] = []
    for path in sorted((Path(path) for path in result_paths), key=lambda item: item.as_posix()):
        record, serving = _source_record(path, _load_result(path), phase, final)
        records.append(record)
        serving_configs.append(serving)

    names = [record["name"] for record in records]
    if set(names) != set(WORKLOADS.values()) or len(set(names)) != len(names):
        raise ManifestError(f"results must contain each workload exactly once; got {sorted(names)}")
    if any(config != serving_configs[0] for config in serving_configs[1:]):
        raise ManifestError("benchmark sources disagree on the frozen serving configuration")
    records.sort(key=lambda record: WORKLOAD_ORDER[record["name"]])

    return {
        "baseline": {
            "reason": baseline_reason.strip(),
            "results": None,
        },
        "metadata": {
            "final": final,
            "phase": phase,
            "source_kind": "emmy_benchmark_json",
        },
        "provenance": {
            "emmy_revision": emmy_revision,
            "engine": {
                "image_digest": image_digest,
                "image_reference": IMAGE_REFERENCE,
                "name": ENGINE_NAME,
                "revision": engine_revision,
            },
            "host_id": host_id.strip(),
            "model": {
                "checkpoint_dtype": "bfloat16",
                "id": MODEL_ID,
                "revision": model_revision,
                "serving_dtype": "float16",
            },
        },
        "schema_version": 1,
        "serving_config": serving_configs[0],
        "workloads": records,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    """Write canonical, deterministic JSON with no generated timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="Three Phase-labeled Emmy *_benchmark.json files")
    parser.add_argument("--output", type=Path, required=True, help="article-manifest.json output path")
    parser.add_argument("--phase", type=int, choices=(1, 2), required=True, help="Evidence phase")
    parser.add_argument("--final", action="store_true", help="Mark a Phase 2 manifest as final publication evidence")
    parser.add_argument("--baseline-reason", required=True, help="Why no like-for-like baseline result exists")
    parser.add_argument("--emmy-revision", required=True, help="Full Emmy Git revision used for the run")
    parser.add_argument("--host-id", required=True, help="Stable identifier for the benchmark host")
    parser.add_argument("--image-digest", required=True, help="Resolved serving image digest (sha256:...)")
    parser.add_argument("--engine-revision", default=ENGINE_REVISION, help="Full 1Cat-vLLM source revision")
    parser.add_argument("--model-revision", default=MODEL_REVISION, help="Full Hugging Face checkpoint revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_cli_logging()
    args = _parser().parse_args(argv)
    try:
        manifest = build_manifest(
            args.results,
            phase=args.phase,
            final=args.final,
            baseline_reason=args.baseline_reason,
            emmy_revision=args.emmy_revision,
            host_id=args.host_id,
            image_digest=args.image_digest,
            engine_revision=args.engine_revision,
            model_revision=args.model_revision,
        )
        write_manifest(args.output, manifest)
    except ManifestError as exc:
        logger.error("Manifest rejected: %s", exc)
        return 2
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
