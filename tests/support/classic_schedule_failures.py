# ruff: noqa: E501 — exact collected node IDs are intentionally indivisible
"""Strict acceptance registry for the clean-slate classic scheduler reconstruction.

Every entry is one exact collected pytest node ID. Patterns and path-wide marks are deliberately impossible here:
each row is an acceptance obligation, and a recovered test must be removed. The initial count is a hard ceiling;
recovery may shrink the registry or split a cluster, but may never add another failure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from frozendict import frozendict


@dataclass(frozen=True)
class RecoveryCluster:
    """A coherent group recovered in one reconstruction phase."""

    name: str
    phase: int
    reason: str
    nodeids: frozenset[str]
    accepted: tuple[str, ...] = ("ClassicScheduleUnavailable",)


@dataclass(frozen=True)
class ReconstructionFailure:
    """The exact recovery obligation attached to one collected test."""

    cluster: str
    phase: int
    reason: str
    accepted: tuple[str, ...]


RECOVERY_CLUSTERS = (
    RecoveryCluster(
        name="composed-kernels",
        phase=5,
        reason="restore compatible schedules for fused and multi-node kernels",
        nodeids=frozenset(
            (
                "tests/serving/generation/test_gen_capture_gpu.py::test_moe_fixed_slot_decode_step_inside_outer_capture_replays_live",
                "tests/serving/generation/test_gen_capture_gpu.py::test_rider_split_inside_outer_capture_replays_live",
                "tests/serving/generation/test_gen_runner_gpu.py::test_adopted_raw_embed_table_matches_folded_gather",
                "tests/serving/generation/test_gen_runner_gpu.py::test_decode_twin_shares_weight_buffers",
                "tests/serving/generation/test_gen_runner_gpu.py::test_device_residents_allocated_eagerly",
                "tests/serving/generation/test_gen_runner_gpu.py::test_expert_program_fp8_indirect_compose",
                "tests/serving/generation/test_gen_runner_gpu.py::test_expert_program_fp8_inputs_match_reference",
                "tests/serving/generation/test_gen_runner_gpu.py::test_gen_runner_device_path_matches_host",
                "tests/serving/generation/test_gen_runner_gpu.py::test_gen_runner_moe_stitch_matches_eager",
                "tests/serving/generation/test_gen_runner_gpu.py::test_gen_runner_stitch_matches_eager",
                "tests/serving/generation/test_gen_runner_gpu.py::test_host_mapped_embed_table_gathers_identically_and_costs_no_vram",
                "tests/serving/generation/test_gen_runner_gpu.py::test_layers_share_activation_arena",
                "tests/serving/generation/test_gen_runner_gpu.py::test_moe_expert_m256_twin_matches_eager_across_experts",
                "tests/serving/generation/test_gen_runner_gpu.py::test_moe_expert_program_takes_the_widest_admitted_step",
                "tests/serving/generation/test_gen_runner_gpu.py::test_moe_expert_shape_groups_compile_and_dispatch_per_layer",
                "tests/serving/generation/test_gen_runner_gpu.py::test_moe_fixed_slot_combine_matches_routed_oracle",
                "tests/serving/generation/test_gen_runner_gpu.py::test_moe_indirect_slot_matches_direct_expert_bit_exact",
                "tests/serving/test_attention_split_gpu.py::test_gemma_post_wrapper_compiles_and_runs_dynamic",
                "tests/serving/test_attention_split_gpu.py::test_post_wrapper_compiles_with_shared_dim",
                "tests/serving/test_attention_split_gpu.py::test_pre_wrapper_compiles_and_runs_dynamic",
            )
        ),
    ),
    RecoveryCluster(
        name="runtime-and-serving-integration",
        phase=8,
        reason="restore CUDA worker, program-pack, capture, and serving integration",
        nodeids=frozenset(
            (
                "tests/serving/generation/test_gen_pack_gpu.py::test_gen_pack_key_separates_quantized_rungs",
                "tests/serving/generation/test_gen_pack_gpu.py::test_gen_pack_second_boot_hits_and_matches",
            )
        ),
    ),
    RecoveryCluster(
        name="corpus-replay",
        phase=8,
        reason="restore strict realization-corpus replay at every declared stage",
        nodeids=frozenset(
            (
                "tests/compiler/realization/test_realization.py::test_realization[attention/rmsnorm-qk-sdpa-composed-cut.yaml-built]",
                "tests/compiler/realization/test_realization.py::test_realization[attention/rmsnorm-qk-sdpa-composed-cut.yaml-correct]",
                "tests/compiler/realization/test_realization.py::test_realization[attention/rmsnorm-qk-sdpa-composed-cut.yaml-offered]",
                "tests/compiler/realization/test_realization.py::test_realization[attention/rmsnorm-qk-sdpa-composed-cut.yaml-realized]",
                "tests/compiler/realization/test_realization.py::test_realization[attention/rmsnorm-qk-sdpa-output-axis.yaml-built]",
                "tests/compiler/realization/test_realization.py::test_realization[attention/rmsnorm-qk-sdpa-output-axis.yaml-correct]",
                "tests/compiler/realization/test_realization.py::test_realization[attention/rmsnorm-qk-sdpa-stat-b-cut.yaml-built]",
                "tests/compiler/realization/test_realization.py::test_realization[attention/rmsnorm-qk-sdpa-stat-b-cut.yaml-correct]",
                "tests/compiler/realization/test_realization.py::test_realization[attention/rmsnorm-qk-sdpa-stat-cut.yaml-built]",
                "tests/compiler/realization/test_realization.py::test_realization[attention/rmsnorm-qk-sdpa-stat-cut.yaml-correct]",
                "tests/compiler/realization/test_realization.py::test_realization[attention/rmsnorm-qk-sdpa-stat-route.yaml-built]",
                "tests/compiler/realization/test_realization.py::test_realization[attention/rmsnorm-qk-sdpa-stat-route.yaml-correct]",
                "tests/compiler/realization/test_realization.py::test_realization[fused/gate-up-distinct-a_xfail_realized.yaml-realized]",
                "tests/compiler/realization/test_realization.py::test_realization[fused/gate-up-shared-a.yaml-built]",
                "tests/compiler/realization/test_realization.py::test_realization[fused/gate-up-shared-a.yaml-correct]",
                "tests/compiler/realization/test_realization.py::test_realization[fused/norm-linear-f16-warp-masked-m.yaml-built]",
                "tests/compiler/realization/test_realization.py::test_realization[fused/norm-linear-f16-warp-masked-m.yaml-correct]",
                "tests/compiler/realization/test_realization.py::test_realization[fused/norm-weight-linear-f16-computed-b.yaml-built]",
                "tests/compiler/realization/test_realization.py::test_realization[fused/norm-weight-linear-f16-computed-b.yaml-correct]",
                "tests/compiler/realization/test_realization.py::test_realization[matmul/f16-matvec-reshaped-output-tma.yaml-built]",
                "tests/compiler/realization/test_realization.py::test_realization[matmul/f16-matvec-reshaped-output-tma.yaml-correct]",
                "tests/compiler/realization/test_realization.py::test_realization[matmul/f32-scalar-masked-n-staged_xfail_offered.yaml-offered]",
                "tests/compiler/realization/test_realization.py::test_realization[qwen3emb/gated-mlp-s128.yaml-built]",
                "tests/compiler/realization/test_realization.py::test_realization[qwen3emb/gated-mlp-s128.yaml-correct]",
                "tests/compiler/realization/test_realization.py::test_realization[qwen3emb/gated-mlp-s32.yaml-built]",
                "tests/compiler/realization/test_realization.py::test_realization[qwen3emb/gated-mlp-s32.yaml-correct]",
                "tests/compiler/realization/test_realization.py::test_realization[qwen3emb/gated-mlp-s512.yaml-built]",
                "tests/compiler/realization/test_realization.py::test_realization[qwen3emb/gated-mlp-s512.yaml-correct]",
                "tests/compiler/realization/test_realization.py::test_realization[reduce/attention-coop-warp.yaml-built]",
                "tests/compiler/realization/test_realization.py::test_realization[reduce/attention-coop-warp.yaml-correct]",
            )
        ),
        accepted=(
            "ClassicScheduleUnavailable",
            "no enumerated row carries the pin",
            "unreproducible pin:",
            "lowering produced no CUDA kernel",
            "pinned but unstamped:",
            "cp.async / TMA staging is single-fold",
            "kernel binder refuses this row's projection ownership",
        ),
    ),
)


_INITIAL_FAILURE_COUNT = 1304
REMAINING_FAILURE_COUNT = 52


def _failures() -> Mapping[str, ReconstructionFailure]:
    failures: dict[str, ReconstructionFailure] = {}
    for cluster in RECOVERY_CLUSTERS:
        for nodeid in cluster.nodeids:
            if nodeid in failures:
                raise RuntimeError(f"duplicate classic schedule reconstruction failure: {nodeid}")
            failures[nodeid] = ReconstructionFailure(cluster.name, cluster.phase, cluster.reason, cluster.accepted)
    if len(failures) != REMAINING_FAILURE_COUNT:
        raise RuntimeError(
            f"classic schedule reconstruction registry has {len(failures)} entries; REMAINING_FAILURE_COUNT is {REMAINING_FAILURE_COUNT}"
        )
    if len(failures) > _INITIAL_FAILURE_COUNT:
        raise RuntimeError("classic schedule reconstruction failure registry may only shrink")
    return frozendict(failures)


FAILURES = _failures()

__all__ = ["FAILURES", "RECOVERY_CLUSTERS", "REMAINING_FAILURE_COUNT", "ReconstructionFailure"]
