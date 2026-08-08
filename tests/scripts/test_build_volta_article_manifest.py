"""Tests for scripts/build_volta_article_manifest.py."""

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_volta_article_manifest.py"
_SPEC = importlib.util.spec_from_file_location("build_volta_article_manifest", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
manifest_builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(manifest_builder)

EMMY_REVISION = "a" * 40
IMAGE_DIGEST = "sha256:" + "b" * 64


def _repeat(index: int) -> dict:
    return {
        "failed_requests": 0,
        "mean_tpot_ms": 20.0 + index,
        "mean_ttft_ms": 100.0 + index,
        "output_token_throughput": 12.0 + index,
        "request_throughput": 0.1 + index / 100,
        "successful_requests": 1,
        "total_token_throughput": 13.0 + index,
    }


def _write_result(tmp_path: Path, input_tokens: int, *, phase: int = 2, repeats: int = 3, pp: int = 8) -> Path:
    raw_repeats = [_repeat(index) for index in range(repeats)]
    data = {
        "metadata": {"phase": phase},
        "task": {
            "gpu_count": 16,
            "gpu_name": manifest_builder.GPU_NAME,
            "recipe_dir": f"_tune/volta-qwen35/final-bench/phase{phase}-serving",
            "variant": f"v100x16_ril{input_tokens}_rol256",
        },
        "recipe": {
            "model": {"huggingface": manifest_builder.MODEL_ID, "task": "generate"},
            "engine": {
                "llm": {
                    "context_length": 32768,
                    "gpu_memory_utilization": 0.88,
                    "max_concurrent_requests": 1,
                    "pipeline_parallel_size": pp,
                    "tensor_parallel_size": 2,
                    "vllm": {
                        "image": manifest_builder.IMAGE_REFERENCE,
                        "extra_args": (
                            "--dtype half "
                            f"--revision {manifest_builder.MODEL_REVISION} "
                            "--language-model-only "
                            "--attention-backend FLASH_ATTN_V100 "
                            "--mamba-cache-mode align "
                            "--max-num-batched-tokens 4096 "
                            "--enforce-eager "
                            "--enable-auto-tool-choice "
                            "--tool-call-parser qwen3_coder "
                            "--reasoning-parser qwen3"
                        ),
                        "extra_env": dict(manifest_builder.REQUIRED_ENGINE_ENV),
                    },
                }
            },
            "benchmark": {
                "ignore_eos": True,
                "max_concurrency": 1,
                "num_prompts": 1,
                "random_input_len": input_tokens,
                "random_output_len": 256,
                "repeats": 3,
                "seed": 0,
                "temperature": 0.0,
            },
            "deploy": {"gpu": manifest_builder.GPU_NAME, "gpu_count": 16},
        },
        "metrics": _repeat(1),
        "metrics_repeats": raw_repeats,
        "metrics_stddev": {"mean_ttft_ms": 1.0},
        "system": {"gpu_count": 16, "gpu_name": "Tesla V100-SXM3-32GB", "hostname": "nvlink-v100"},
        "timing": {"benchmark": 10.0, "model_load_and_warmup": 300.0, "total": 310.0},
    }
    path = tmp_path / f"phase{phase}_input{input_tokens}_benchmark.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _paths(tmp_path: Path, *, phase: int = 2) -> list[Path]:
    return [_write_result(tmp_path, input_tokens, phase=phase) for input_tokens in (32, 4096, 32000)]


def _build(paths: list[Path], *, phase: int = 2, final: bool = True) -> dict:
    return manifest_builder.build_manifest(
        paths,
        phase=phase,
        final=final,
        baseline_reason="No like-for-like FP16 baseline ran on the qualified host.",
        emmy_revision=EMMY_REVISION,
        host_id="nvlink-v100-16x",
        image_digest=IMAGE_DIGEST,
    )


def test_manifest_is_deterministic_and_preserves_raw_repetitions(tmp_path):
    paths = _paths(tmp_path)
    manifest = _build(list(reversed(paths)))

    assert manifest["metadata"] == {"final": True, "phase": 2, "source_kind": "emmy_benchmark_json"}
    assert manifest["baseline"]["results"] is None
    assert manifest["baseline"]["reason"].startswith("No like-for-like")
    assert manifest["provenance"]["emmy_revision"] == EMMY_REVISION
    assert manifest["provenance"]["engine"]["image_digest"] == IMAGE_DIGEST
    assert manifest["provenance"]["model"]["revision"] == manifest_builder.MODEL_REVISION
    assert [item["name"] for item in manifest["workloads"]] == list(manifest_builder.WORKLOADS.values())
    assert manifest["workloads"][0]["raw_repetitions"] == [_repeat(0), _repeat(1), _repeat(2)]
    assert manifest["workloads"][0]["errors"]["all_requests_succeeded"] is True

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    manifest_builder.write_manifest(first, manifest)
    manifest_builder.write_manifest(second, _build(paths))
    assert first.read_bytes() == second.read_bytes()


def test_phase1_diagnostic_manifest_is_nonfinal(tmp_path):
    manifest = _build(_paths(tmp_path, phase=1), phase=1, final=False)
    assert manifest["metadata"]["phase"] == 1
    assert manifest["metadata"]["final"] is False
    assert {workload["source"]["phase"] for workload in manifest["workloads"]} == {1}


def test_final_manifest_rejects_phase1_labeled_sources(tmp_path):
    with pytest.raises(manifest_builder.ManifestError, match="final manifests cannot consume Phase1"):
        _build(_paths(tmp_path, phase=1), phase=2, final=True)


def test_manifest_requires_three_raw_repetitions(tmp_path):
    paths = _paths(tmp_path)
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    data["metrics_repeats"] = data["metrics_repeats"][:2]
    paths[0].write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(manifest_builder.ManifestError, match="exactly three raw repetitions"):
        _build(paths)


def test_manifest_rejects_serving_configuration_drift(tmp_path):
    paths = _paths(tmp_path)
    data = json.loads(paths[1].read_text(encoding="utf-8"))
    data["recipe"]["engine"]["llm"]["pipeline_parallel_size"] = 4
    paths[1].write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(manifest_builder.ManifestError, match="pipeline_parallel_size must be 8"):
        _build(paths)


@pytest.mark.parametrize("digest", ["", "sha256:abc", "c" * 64])
def test_manifest_requires_full_image_digest(tmp_path, digest):
    with pytest.raises(manifest_builder.ManifestError, match="image digest"):
        manifest_builder.build_manifest(
            _paths(tmp_path),
            phase=2,
            final=True,
            baseline_reason="No baseline.",
            emmy_revision=EMMY_REVISION,
            host_id="nvlink-v100-16x",
            image_digest=digest,
        )
