"""Tests for scripts/validate_serving_kernel_manifest.py."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validate_serving_kernel_manifest.py"
_SPEC = importlib.util.spec_from_file_location("validate_serving_kernel_manifest", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validator)


def _operand_signature(*, tail: bool = False) -> tuple:
    input_rows = 8 if tail else 1
    return (
        "gemm",
        "torch.linear",
        (
            ("input", (input_rows, 4096), (4096, 1), "row_major_contiguous", "bfloat16"),
            ("weight", (4096, 1024), (1024, 1), "row_major_contiguous", "fp8_e4m3fn"),
        ),
        "fp8_block",
        "cutlass",
    )


def _manifest(tp_size: int = 4, *, rank_policy: str = "uniform_tp") -> dict:
    workloads = [
        {
            "name": "decode",
            "phase": "decode",
            "excluded_phases": ["prefill"],
            "input_tokens": 256,
            "output_tokens": 256,
            "concurrency": 8,
            "requests": 40,
            "measured_iterations": 10,
            "seed": 0,
            "temperature": 0,
            "ignore_eos": True,
            "client_command": "vllm bench serve --dataset-name random --seed 0",
            "torch_run_id": "decode-torch-run",
            "nsight_run_id": "decode-nsight-run",
        },
        {
            "name": "prefill",
            "phase": "prefill",
            "excluded_phases": ["decode"],
            "input_tokens": 4096,
            "output_tokens": 1,
            "concurrency": 1,
            "requests": 20,
            "measured_iterations": 10,
            "seed": 0,
            "temperature": 0,
            "ignore_eos": True,
            "client_command": "vllm bench serve --dataset-name random --seed 0",
            "torch_run_id": "prefill-torch-run",
            "nsight_run_id": "prefill-nsight-run",
        },
    ]
    workload_by_name = {workload["name"]: workload for workload in workloads}
    records = []
    selection_by_workload = {}
    for workload in ("decode", "prefill"):
        by_rank = {}
        for rank in range(tp_size):
            by_rank[str(rank)] = {
                "case_ids": [f"{workload}.main" if rank_policy == "uniform_tp" else f"{workload}.main.rank{rank}"],
                "runtime_fraction": 0.92,
            }
            for suffix, duration in (("main", 92.0), ("tail", 8.0)):
                tail = suffix == "tail"
                case_id = f"{workload}.{suffix}"
                if rank_policy == "rank_local_ep":
                    case_id += f".rank{rank}"
                run = workload_by_name[workload]
                records.append(
                    {
                        "record_id": f"{case_id}.rank{rank}",
                        "case_id": case_id,
                        "workload": workload,
                        "phase": workload,
                        "rank": rank,
                        "family": "gemm",
                        "operator": "torch.linear",
                        "kernel": "cutlass_kernel",
                        "operands": [
                            {
                                "role": "input",
                                "shape": [8 if tail else 1, 4096],
                                "strides": [4096, 1],
                                "layout": "row_major_contiguous",
                                "dtype": "bfloat16",
                            },
                            {
                                "role": "weight",
                                "shape": [4096, 1024],
                                "strides": [1024, 1],
                                "layout": "row_major_contiguous",
                                "dtype": "fp8_e4m3fn",
                            },
                        ],
                        "quantization": {"method": "fp8_block", "backend": "cutlass"},
                        "launch_count": 10,
                        "total_cuda_us": duration,
                        "reproducer": {"source": f"reproducers/{case_id}.json", "sha256": "d" * 64},
                        "supported": True,
                        "failure": None,
                        "evidence": {
                            "torch_run_id": run["torch_run_id"],
                            "nsight_run_id": run["nsight_run_id"],
                            "operator_metadata_run_ids": [run["torch_run_id"], run["nsight_run_id"]],
                            "torch_launch_count": 10,
                            "nsight_launch_count": 10,
                            "reconciliation_key": validator._reconciliation_key(workload, workload, _operand_signature(tail=tail)),
                        },
                    }
                )
        selection_by_workload[workload] = {"by_rank": by_rank}
    accounting = {
        "by_workload": {
            workload: {
                "by_rank": {
                    str(rank): {
                        "nsight_whole_window_cuda_us": 110.0,
                        "nsight_model_forward_phase_cuda_us": 100.0,
                        "nsight_excluded_cuda_us": 10.0,
                        "recorded_cuda_us": 100.0,
                        "nsight_model_forward_phase_launches": 20,
                        "recorded_launches": 20,
                    }
                    for rank in range(tp_size)
                }
            }
            for workload in ("decode", "prefill")
        }
    }
    return {
        "schema_version": 1,
        "status": "measured",
        "model": {"id": "org/model", "revision": "a" * 40, "config_sha256": "9" * 64},
        "engine": {"revision": "b" * 40, "image": "org/engine@sha256:" + "c" * 64},
        "compiler": {"revision": "e" * 40},
        "platform": {
            "gpu_name": "NVIDIA H200 141GB",
            "gpu_count": tp_size,
            "tensor_parallel_size": tp_size,
            "ranks": list(range(tp_size)),
            "gpu_uuids": [f"GPU-{rank}" for rank in range(tp_size)],
            "driver": "580.1",
            "cuda": "13.0",
            "parallel_policy": f"tp{tp_size}",
            "rank_policy": rank_policy,
        },
        "serving": {
            "recipe": "experiments/golden-bench-2026/serving_glm52_fp8_h200x8/recipe.yaml",
            "server_command": "vllm serve org/model --tensor-parallel-size 8",
        },
        "capture": {
            source: {
                "command": f"capture-{source}",
                "artifacts": [{"path": f"raw/{source}.json", "sha256": "f" * 64}],
            }
            for source in ("operator_metadata", "torch_profiler", "nsight_systems", "engine_log", "workload_client")
        },
        "workloads": workloads,
        "records": records,
        "accounting": accounting,
        "selection": {"minimum_runtime_fraction": 0.9, "by_workload": selection_by_workload},
    }


@pytest.mark.parametrize("tp_size", [4, 8])
def test_validator_accepts_complete_uniform_tp_manifests(tp_size):
    summaries = validator.validate_manifest(_manifest(tp_size))

    assert summaries["decode"]["by_rank"]["0"] == {
        "case_ids": ["decode.main"],
        "runtime_fraction": 0.92,
    }


def test_validator_accepts_rank_asymmetric_ep_manifest():
    manifest = _manifest(8, rank_policy="rank_local_ep")

    summaries = validator.validate_manifest(manifest)

    assert summaries["prefill"]["by_rank"]["7"]["case_ids"] == ["prefill.main.rank7"]


def test_uniform_tp_validator_requires_every_rank_for_each_case():
    manifest = _manifest()
    manifest["records"] = [record for record in manifest["records"] if record["record_id"] != "decode.main.rank3"]

    with pytest.raises(validator.ManifestError, match="exactly one record for every TP rank"):
        validator.validate_manifest(manifest)


def test_validator_fails_closed_on_missing_runtime_metadata():
    manifest = _manifest()
    del manifest["records"][0]["operands"][0]["layout"]

    with pytest.raises(validator.ManifestError, match="layout must be a non-empty string"):
        validator.validate_manifest(manifest)


def test_validator_recomputes_selection_instead_of_trusting_manifest():
    manifest = _manifest()
    manifest["selection"]["by_workload"]["decode"]["by_rank"]["0"]["case_ids"] = ["decode.tail"]

    with pytest.raises(validator.ManifestError, match="must equal deterministic selection"):
        validator.validate_manifest(manifest)


def test_validator_counts_unsupported_hot_cases_against_coverage():
    manifest = _manifest()
    for record in manifest["records"]:
        if record["case_id"] == "decode.main" and record["rank"] == 0:
            record["supported"] = False
            record["failure"] = "Emmy reproducer cannot express the runtime layout"

    with pytest.raises(validator.ManifestError, match="rank 0 supported cases cover only 0.080"):
        validator.validate_manifest(manifest)


def test_omitted_kernel_cannot_shrink_independent_nsight_denominator():
    manifest = _manifest()
    manifest["records"] = [record for record in manifest["records"] if record["record_id"] != "decode.tail.rank0"]
    manifest["platform"]["rank_policy"] = "rank_local_ep"

    with pytest.raises(validator.ManifestError, match="records must reconcile to recorded CUDA time"):
        validator.validate_manifest(manifest)


def test_phase_contamination_is_rejected():
    manifest = _manifest()
    manifest["records"][0]["phase"] = "prefill"

    with pytest.raises(validator.ManifestError, match="must isolate the 'decode' phase"):
        validator.validate_manifest(manifest)


def test_cross_run_evidence_must_reconcile_by_stable_signature():
    manifest = _manifest()
    manifest["records"][0]["evidence"]["reconciliation_key"] = "0" * 64

    with pytest.raises(validator.ManifestError, match="stable operand signature"):
        validator.validate_manifest(manifest)


def test_cross_run_launch_counts_must_reconcile_by_stable_signature():
    manifest = _manifest()
    manifest["records"][0]["evidence"]["torch_launch_count"] = 9

    with pytest.raises(validator.ManifestError, match="profiler launch counts must reconcile"):
        validator.validate_manifest(manifest)


def test_protocol_only_manifest_is_not_publication_evidence():
    manifest = _manifest()
    manifest["status"] = "protocol_only"

    with pytest.raises(validator.ManifestError, match="not publication evidence"):
        validator.validate_manifest(manifest)


def test_loaded_manifest_requires_digest_matching_artifacts(tmp_path):
    manifest = _manifest(2)
    artifacts = {artifact["path"] for capture in manifest["capture"].values() for artifact in capture["artifacts"]} | {
        record["reproducer"]["source"] for record in manifest["records"]
    }
    for relative in artifacts:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        digest = hashlib.sha256(relative.encode()).hexdigest()
        for capture in manifest["capture"].values():
            for artifact in capture["artifacts"]:
                if artifact["path"] == relative:
                    artifact["sha256"] = digest
        for record in manifest["records"]:
            if record["reproducer"]["source"] == relative:
                record["reproducer"]["sha256"] = digest
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validator.load_and_validate(manifest_path)
    (tmp_path / "raw" / "engine_log.json").write_text("tampered", encoding="utf-8")

    with pytest.raises(validator.ManifestError, match="digest mismatch"):
        validator.load_and_validate(manifest_path)
