#!/usr/bin/env python3
"""Validate a measured distributed serving-kernel manifest and its runtime selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

from emmy.logging_setup import setup_cli_logging

logger = logging.getLogger(__name__)

REQUIRED_WORKLOADS = {"decode", "prefill"}
WORKLOAD_PHASES = {"decode": "decode", "prefill": "prefill"}
RANK_POLICIES = {"uniform_tp", "rank_local_ep"}
MINIMUM_RUNTIME_FRACTION = 0.90
ACCOUNTING_REL_TOL = 1e-9
ACCOUNTING_ABS_TOL_US = 1e-6
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_IMAGE_RE = re.compile(r".+@sha256:[0-9a-f]{64}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """Raised when a serving manifest is incomplete or selects cases non-deterministically."""


def _mapping(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    return value


def _nonempty_string(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a non-empty string")
    return value


def _positive_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ManifestError(f"{label} must be a positive finite number")
    return float(value)


def _nonnegative_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise ManifestError(f"{label} must be a non-negative finite number")
    return float(value)


def _positive_integer(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError(f"{label} must be a positive integer")
    return value


def _revision(value, label: str) -> str:
    value = _nonempty_string(value, label).lower()
    if not _REVISION_RE.fullmatch(value):
        raise ManifestError(f"{label} must be a full 40-hex revision")
    return value


def _validate_artifacts(capture: dict, source: str) -> None:
    record = _mapping(capture.get(source), f"capture.{source}")
    _nonempty_string(record.get("command"), f"capture.{source}.command")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ManifestError(f"capture.{source}.artifacts must be a non-empty list")
    for index, artifact in enumerate(artifacts):
        artifact = _mapping(artifact, f"capture.{source}.artifacts[{index}]")
        _nonempty_string(artifact.get("path"), f"capture.{source}.artifacts[{index}].path")
        digest = _nonempty_string(artifact.get("sha256"), f"capture.{source}.artifacts[{index}].sha256").lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ManifestError(f"capture.{source}.artifacts[{index}].sha256 must be 64 hex characters")


def _validate_provenance(manifest: dict) -> tuple[set[int], str]:
    if manifest.get("schema_version") != 1:
        raise ManifestError("schema_version must be 1")
    if manifest.get("status") != "measured":
        raise ManifestError("status must be 'measured'; protocol-only or synthetic data is not publication evidence")

    model = _mapping(manifest.get("model"), "model")
    _nonempty_string(model.get("id"), "model.id")
    _revision(model.get("revision"), "model.revision")
    config_digest = _nonempty_string(model.get("config_sha256"), "model.config_sha256").lower()
    if not _SHA256_RE.fullmatch(config_digest):
        raise ManifestError("model.config_sha256 must be 64 hex characters")

    engine = _mapping(manifest.get("engine"), "engine")
    _revision(engine.get("revision"), "engine.revision")
    image = _nonempty_string(engine.get("image"), "engine.image").lower()
    if not _IMAGE_RE.fullmatch(image):
        raise ManifestError("engine.image must contain an immutable @sha256 digest")

    compiler = _mapping(manifest.get("compiler"), "compiler")
    _revision(compiler.get("revision"), "compiler.revision")

    platform = _mapping(manifest.get("platform"), "platform")
    _nonempty_string(platform.get("gpu_name"), "platform.gpu_name")
    gpu_count = platform.get("gpu_count")
    tensor_parallel_size = platform.get("tensor_parallel_size")
    if isinstance(tensor_parallel_size, bool) or not isinstance(tensor_parallel_size, int) or tensor_parallel_size < 2:
        raise ManifestError("platform.tensor_parallel_size must be an integer of at least 2")
    if gpu_count != tensor_parallel_size:
        raise ManifestError("platform.gpu_count must equal platform.tensor_parallel_size for this single-replica audit")
    required_ranks = set(range(tensor_parallel_size))
    if platform.get("ranks") != list(range(tensor_parallel_size)):
        raise ManifestError("platform.ranks must enumerate every tensor-parallel rank in order")
    gpu_uuids = platform.get("gpu_uuids")
    if not isinstance(gpu_uuids, list) or len(gpu_uuids) != tensor_parallel_size:
        raise ManifestError("platform.gpu_uuids must contain one UUID per tensor-parallel rank")
    for index, uuid in enumerate(gpu_uuids):
        _nonempty_string(uuid, f"platform.gpu_uuids[{index}]")
    if len(set(gpu_uuids)) != tensor_parallel_size:
        raise ManifestError("platform.gpu_uuids must be unique")
    for field in ("driver", "cuda", "parallel_policy"):
        _nonempty_string(platform.get(field), f"platform.{field}")
    rank_policy = platform.get("rank_policy")
    if rank_policy not in RANK_POLICIES:
        raise ManifestError(f"platform.rank_policy must be one of {sorted(RANK_POLICIES)}")

    serving = _mapping(manifest.get("serving"), "serving")
    _nonempty_string(serving.get("recipe"), "serving.recipe")
    _nonempty_string(serving.get("server_command"), "serving.server_command")

    capture = _mapping(manifest.get("capture"), "capture")
    for source in ("operator_metadata", "torch_profiler", "nsight_systems", "engine_log", "workload_client"):
        _validate_artifacts(capture, source)
    return required_ranks, rank_policy


def _validate_workloads(manifest: dict) -> dict[str, dict]:
    workloads = manifest.get("workloads")
    if not isinstance(workloads, list):
        raise ManifestError("workloads must be a JSON array")
    by_name: dict[str, dict] = {}
    for index, workload in enumerate(workloads):
        workload = _mapping(workload, f"workloads[{index}]")
        name = _nonempty_string(workload.get("name"), f"workloads[{index}].name")
        if name in by_name:
            raise ManifestError(f"duplicate workload {name!r}")
        by_name[name] = workload
        if workload.get("phase") != WORKLOAD_PHASES.get(name):
            raise ManifestError(f"workloads[{index}].phase must be {WORKLOAD_PHASES.get(name)!r}")
        excluded_phases = workload.get("excluded_phases")
        other_phase = next(iter(REQUIRED_WORKLOADS - {name})) if name in REQUIRED_WORKLOADS else None
        if not isinstance(excluded_phases, list) or other_phase not in excluded_phases:
            raise ManifestError(f"workloads[{index}].excluded_phases must retain {other_phase!r} outside selection")
        _positive_number(workload.get("input_tokens"), f"workloads[{index}].input_tokens")
        _positive_number(workload.get("output_tokens"), f"workloads[{index}].output_tokens")
        _positive_number(workload.get("concurrency"), f"workloads[{index}].concurrency")
        _positive_number(workload.get("requests"), f"workloads[{index}].requests")
        _positive_number(workload.get("measured_iterations"), f"workloads[{index}].measured_iterations")
        if workload.get("seed") != 0:
            raise ManifestError(f"workloads[{index}].seed must be 0")
        if workload.get("temperature") != 0:
            raise ManifestError(f"workloads[{index}].temperature must be 0")
        if workload.get("ignore_eos") is not True:
            raise ManifestError(f"workloads[{index}].ignore_eos must be true")
        _nonempty_string(workload.get("client_command"), f"workloads[{index}].client_command")
        torch_run_id = _nonempty_string(workload.get("torch_run_id"), f"workloads[{index}].torch_run_id")
        nsight_run_id = _nonempty_string(workload.get("nsight_run_id"), f"workloads[{index}].nsight_run_id")
        if torch_run_id == nsight_run_id:
            raise ManifestError(f"workloads[{index}] profiler run IDs must be distinct")
    if set(by_name) != REQUIRED_WORKLOADS:
        raise ManifestError(f"workloads must be exactly {sorted(REQUIRED_WORKLOADS)}")
    return by_name


def _validate_operand(operand, label: str) -> tuple:
    operand = _mapping(operand, label)
    role = _nonempty_string(operand.get("role"), f"{label}.role")
    dtype = _nonempty_string(operand.get("dtype"), f"{label}.dtype")
    layout = _nonempty_string(operand.get("layout"), f"{label}.layout")
    shape = operand.get("shape")
    strides = operand.get("strides")
    if not isinstance(shape, list) or not shape:
        raise ManifestError(f"{label}.shape must be a non-empty list")
    if not isinstance(strides, list) or len(strides) != len(shape):
        raise ManifestError(f"{label}.strides must have one entry per shape dimension")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise ManifestError(f"{label}.shape dimensions must be positive integers")
    if any(isinstance(stride, bool) or not isinstance(stride, int) or stride < 0 for stride in strides):
        raise ManifestError(f"{label}.strides must be non-negative integers")
    return role, tuple(shape), tuple(strides), layout, dtype


def _reconciliation_key(workload: str, phase: str, signature: tuple) -> str:
    payload = json.dumps([workload, phase, signature], sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_record(record, index: int, required_ranks: set[int], workloads: dict[str, dict]) -> dict:
    label = f"records[{index}]"
    record = _mapping(record, label)
    record_id = _nonempty_string(record.get("record_id"), f"{label}.record_id")
    case_id = _nonempty_string(record.get("case_id"), f"{label}.case_id")
    workload = _nonempty_string(record.get("workload"), f"{label}.workload")
    if workload not in REQUIRED_WORKLOADS:
        raise ManifestError(f"{label}.workload must be decode or prefill")
    phase = _nonempty_string(record.get("phase"), f"{label}.phase")
    if phase != workloads[workload]["phase"]:
        raise ManifestError(f"{label}.phase must isolate the {workloads[workload]['phase']!r} phase")
    rank = record.get("rank")
    if rank not in required_ranks:
        raise ManifestError(f"{label}.rank must be one of {sorted(required_ranks)}")

    family = _nonempty_string(record.get("family"), f"{label}.family")
    operator = _nonempty_string(record.get("operator"), f"{label}.operator")
    _nonempty_string(record.get("kernel"), f"{label}.kernel")
    operands = record.get("operands")
    if not isinstance(operands, list) or not operands:
        raise ManifestError(f"{label}.operands must be a non-empty list")
    operand_signature = tuple(
        _validate_operand(operand, f"{label}.operands[{operand_index}]") for operand_index, operand in enumerate(operands)
    )

    quantization = _mapping(record.get("quantization"), f"{label}.quantization")
    quant_method = _nonempty_string(quantization.get("method"), f"{label}.quantization.method")
    quant_backend = _nonempty_string(quantization.get("backend"), f"{label}.quantization.backend")

    launch_count = _positive_integer(record.get("launch_count"), f"{label}.launch_count")
    total_cuda_us = _positive_number(record.get("total_cuda_us"), f"{label}.total_cuda_us")
    reproducer = _mapping(record.get("reproducer"), f"{label}.reproducer")
    _nonempty_string(reproducer.get("source"), f"{label}.reproducer.source")
    digest = _nonempty_string(reproducer.get("sha256"), f"{label}.reproducer.sha256").lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ManifestError(f"{label}.reproducer.sha256 must be 64 hex characters")

    supported = record.get("supported")
    if not isinstance(supported, bool):
        raise ManifestError(f"{label}.supported must be a boolean")
    failure = record.get("failure")
    if not supported:
        _nonempty_string(failure, f"{label}.failure")
    elif failure not in (None, ""):
        raise ManifestError(f"{label}.failure must be empty for a supported record")

    signature = (family, operator, operand_signature, quant_method, quant_backend)
    evidence = _mapping(record.get("evidence"), f"{label}.evidence")
    torch_run_id = workloads[workload]["torch_run_id"]
    nsight_run_id = workloads[workload]["nsight_run_id"]
    if evidence.get("torch_run_id") != torch_run_id or evidence.get("nsight_run_id") != nsight_run_id:
        raise ManifestError(f"{label}.evidence must reference the workload's separate profiler runs")
    if evidence.get("operator_metadata_run_ids") != [torch_run_id, nsight_run_id]:
        raise ManifestError(f"{label}.evidence.operator_metadata_run_ids must cover both profiler runs in order")
    torch_launch_count = _positive_integer(evidence.get("torch_launch_count"), f"{label}.evidence.torch_launch_count")
    nsight_launch_count = _positive_integer(evidence.get("nsight_launch_count"), f"{label}.evidence.nsight_launch_count")
    if torch_launch_count != launch_count or nsight_launch_count != launch_count:
        raise ManifestError(f"{label}.evidence profiler launch counts must reconcile for the stable operand signature")
    expected_key = _reconciliation_key(workload, phase, signature)
    if evidence.get("reconciliation_key") != expected_key:
        raise ManifestError(f"{label}.evidence.reconciliation_key does not match the stable operand signature")

    return {
        "record_id": record_id,
        "case_id": case_id,
        "workload": workload,
        "rank": rank,
        "signature": signature,
        "duration": total_cuda_us,
        "launch_count": launch_count,
        "supported": supported,
    }


def _expected_selection(case_rows: dict[str, tuple[float, bool]], threshold: float, denominator: float) -> tuple[list[str], float]:
    ordered = sorted(
        ((case_id, duration, supported) for case_id, (duration, supported) in case_rows.items()),
        key=lambda item: (-item[1], item[0]),
    )
    selected: list[str] = []
    selected_us = 0.0
    for case_id, duration, supported in ordered:
        if not supported:
            continue
        selected.append(case_id)
        selected_us += duration
        if selected_us / denominator >= threshold:
            break
    return selected, selected_us / denominator


def _accounting_totals(manifest: dict, required_ranks: set[int], record_totals: dict) -> dict[tuple[str, int], float]:
    accounting = _mapping(manifest.get("accounting"), "accounting")
    by_workload = _mapping(accounting.get("by_workload"), "accounting.by_workload")
    if set(by_workload) != REQUIRED_WORKLOADS:
        raise ManifestError(f"accounting.by_workload must contain exactly {sorted(REQUIRED_WORKLOADS)}")
    denominators: dict[tuple[str, int], float] = {}
    for workload in sorted(REQUIRED_WORKLOADS):
        workload_accounting = _mapping(by_workload[workload], f"accounting.by_workload.{workload}")
        by_rank = _mapping(workload_accounting.get("by_rank"), f"accounting.by_workload.{workload}.by_rank")
        if set(by_rank) != {str(rank) for rank in required_ranks}:
            raise ManifestError(f"accounting.by_workload.{workload}.by_rank must contain every rank")
        for rank in sorted(required_ranks):
            label = f"accounting.by_workload.{workload}.by_rank.{rank}"
            row = _mapping(by_rank[str(rank)], label)
            whole = _positive_number(row.get("nsight_whole_window_cuda_us"), f"{label}.nsight_whole_window_cuda_us")
            phase = _positive_number(row.get("nsight_model_forward_phase_cuda_us"), f"{label}.nsight_model_forward_phase_cuda_us")
            excluded = _nonnegative_number(row.get("nsight_excluded_cuda_us"), f"{label}.nsight_excluded_cuda_us")
            recorded = _positive_number(row.get("recorded_cuda_us"), f"{label}.recorded_cuda_us")
            nsight_launches = _positive_integer(
                row.get("nsight_model_forward_phase_launches"), f"{label}.nsight_model_forward_phase_launches"
            )
            recorded_launches = _positive_integer(row.get("recorded_launches"), f"{label}.recorded_launches")
            actual_us, actual_launches = record_totals.get((workload, rank), (0.0, 0))
            comparisons = (
                (whole, phase + excluded, "whole-window CUDA time must equal selected phase plus retained exclusions"),
                (recorded, phase, "recorded CUDA time must reconcile to the independent Nsight phase total"),
                (actual_us, recorded, "manifest records must reconcile to recorded CUDA time"),
            )
            for left, right, reason in comparisons:
                if not math.isclose(left, right, rel_tol=ACCOUNTING_REL_TOL, abs_tol=ACCOUNTING_ABS_TOL_US):
                    raise ManifestError(f"{label}: {reason}; got {left:.12g} vs {right:.12g}")
            if recorded_launches != nsight_launches or actual_launches != recorded_launches:
                raise ManifestError(f"{label}: record launch counts must reconcile to the independent Nsight phase total")
            denominators[workload, rank] = phase
    return denominators


def validate_manifest(manifest: dict) -> dict[str, dict]:
    """Validate one measured manifest and return recomputed per-workload, per-rank selections."""
    manifest = _mapping(manifest, "manifest")
    required_ranks, rank_policy = _validate_provenance(manifest)
    workloads = _validate_workloads(manifest)

    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ManifestError("records must be a non-empty JSON array")
    record_ids: set[str] = set()
    case_data: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rank_cases: dict[tuple[str, int], dict[str, tuple[float, bool]]] = defaultdict(dict)
    record_totals: dict[tuple[str, int], tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    signature_cases: dict[tuple[str, tuple], str] = {}
    for index, record in enumerate(records):
        row = _validate_record(record, index, required_ranks, workloads)
        if row["record_id"] in record_ids:
            raise ManifestError(f"duplicate record_id {row['record_id']!r}")
        record_ids.add(row["record_id"])
        key = (row["workload"], row["case_id"])
        case_data[key].append(row)
        rank_key = (row["workload"], row["rank"])
        if row["case_id"] in rank_cases[rank_key]:
            raise ManifestError(f"case {row['case_id']!r} must have at most one record on rank {row['rank']}")
        rank_cases[rank_key][row["case_id"]] = (row["duration"], row["supported"])
        duration, launches = record_totals[rank_key]
        record_totals[rank_key] = (duration + row["duration"], launches + row["launch_count"])
        policy_scope = None if rank_policy == "uniform_tp" else row["rank"]
        signature_key = (row["workload"], policy_scope, row["signature"])
        previous_case = signature_cases.setdefault(signature_key, row["case_id"])
        if previous_case != row["case_id"]:
            raise ManifestError(f"workload {row['workload']} splits one measured operand signature across case IDs")

    for (workload, case_id), rows in case_data.items():
        ranks = {row["rank"] for row in rows}
        signatures = {row["signature"] for row in rows}
        if len(signatures) != 1:
            raise ManifestError(f"case {case_id!r} in {workload} has inconsistent operand or quantization metadata")
        if rank_policy == "uniform_tp" and ranks != required_ranks:
            raise ManifestError(f"case {case_id!r} in {workload} must have exactly one record for every TP rank")

    denominators = _accounting_totals(manifest, required_ranks, record_totals)
    selection = _mapping(manifest.get("selection"), "selection")
    threshold = selection.get("minimum_runtime_fraction")
    if threshold != MINIMUM_RUNTIME_FRACTION:
        raise ManifestError(f"selection.minimum_runtime_fraction must be {MINIMUM_RUNTIME_FRACTION}")
    declared = _mapping(selection.get("by_workload"), "selection.by_workload")
    if set(declared) != REQUIRED_WORKLOADS:
        raise ManifestError(f"selection.by_workload must contain exactly {sorted(REQUIRED_WORKLOADS)}")

    summaries: dict[str, dict] = {}
    for workload in sorted(REQUIRED_WORKLOADS):
        workload_selection = _mapping(declared[workload], f"selection.by_workload.{workload}")
        declared_by_rank = _mapping(workload_selection.get("by_rank"), f"selection.by_workload.{workload}.by_rank")
        if set(declared_by_rank) != {str(rank) for rank in required_ranks}:
            raise ManifestError(f"selection.by_workload.{workload}.by_rank must contain every rank")
        rank_summaries = {}
        for rank in sorted(required_ranks):
            expected_ids, runtime_fraction = _expected_selection(rank_cases[workload, rank], threshold, denominators[workload, rank])
            if runtime_fraction < threshold:
                raise ManifestError(
                    f"{workload} rank {rank} supported cases cover only {runtime_fraction:.3f} of independent Nsight "
                    f"phase time; required {threshold:.3f}"
                )
            label = f"selection.by_workload.{workload}.by_rank.{rank}"
            rank_selection = _mapping(declared_by_rank[str(rank)], label)
            if rank_selection.get("case_ids") != expected_ids:
                raise ManifestError(f"{label}.case_ids must equal deterministic selection {expected_ids}")
            declared_fraction = rank_selection.get("runtime_fraction")
            if not isinstance(declared_fraction, int | float) or not math.isclose(
                declared_fraction, runtime_fraction, rel_tol=0, abs_tol=1e-9
            ):
                raise ManifestError(f"{label}.runtime_fraction must equal recomputed value {runtime_fraction:.12g}")
            rank_summaries[str(rank)] = {"case_ids": expected_ids, "runtime_fraction": runtime_fraction}
        summaries[workload] = {"by_rank": rank_summaries}
    return summaries


def load_and_validate(path: Path) -> dict[str, dict]:
    """Load a manifest, validate its schema, and verify every referenced artifact digest."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {path}: {exc}") from exc
    summaries = validate_manifest(manifest)
    references = []
    for capture in manifest["capture"].values():
        references.extend((artifact["path"], artifact["sha256"]) for artifact in capture["artifacts"])
    references.extend((record["reproducer"]["source"], record["reproducer"]["sha256"]) for record in manifest["records"])
    for relative, expected_digest in references:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ManifestError(f"artifact path must stay below the manifest directory: {relative}")
        artifact_path = path.parent / relative_path
        try:
            actual_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ManifestError(f"cannot read manifest artifact {artifact_path}: {exc}") from exc
        if actual_digest != expected_digest:
            raise ManifestError(f"manifest artifact digest mismatch for {artifact_path}")
    return summaries


def main() -> int:
    """Run the command-line validator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    setup_cli_logging()
    try:
        summaries = load_and_validate(args.manifest)
    except ManifestError as exc:
        logger.error("Invalid serving kernel manifest: %s", exc)
        return 1
    for workload, summary in summaries.items():
        fractions = [row["runtime_fraction"] for row in summary["by_rank"].values()]
        selected = {case_id for row in summary["by_rank"].values() for case_id in row["case_ids"]}
        logger.info(
            "%s: %.1f%% minimum per-rank runtime coverage from %d selected cases",
            workload,
            100 * min(fractions),
            len(selected),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
