"""Registry of tests expected to fail while the tile SCHEDULER is absent.

The tile scheduler — candidate enumeration and composition (``_schedule.py`` / ``_view.py`` and the
``020_schedule`` rule) — was removed to clear the ground for the demand-driven recursive
enumerator. Recognition (``010_recognize``), the split rewrite (``030_split_reduce``), the knob /
path codec and the whole ``lowering/kernel`` materializer are untouched; what is gone is the step
that decides a ``TileOp``'s ``place`` / ``schedule``. Every test below exercises a compile that
reaches that step, so it fails until enumeration returns.

The registry is deliberately a LIST OF EXACT NODE IDS, not a module or path glob: each id is a
concrete acceptance obligation for the new scheduler, and the marker is non-strict-XPASS-visible
(``strict=False`` so a partially restored scheduler does not turn recovery into a failure, but
``-rX`` lists every id that has started passing). Porting a phase means deleting the ids it
restores — the file shrinking to empty IS the completion gate.

Two limits to know about:

- **Collection errors cannot be xfailed.** Three modules unit-tested private helpers of the deleted
  scheduler; those cases were deleted with the helpers (the surviving cases in
  ``test_node_vs_slice.py`` / ``test_split_cast_from_indexmap.py``, and the one end-to-end pin
  contract in ``test_flash_form_narrowing.py``, stayed).
- **GPU-gated ids are missing.** The list was measured on a CPU-only box, where the CUDA suite
  skips. Ids observed failing on a GPU machine get appended here with the same reason.

Applied by the root ``conftest.py``'s ``pytest_collection_modifyitems``.
"""

from __future__ import annotations

REASON = "tile scheduler removed - awaiting the generic demand-driven enumerator"

# Node ids that fail because scheduling is missing. Sorted by module, then case.
NODEIDS: frozenset[str] = frozenset(
    {
        # tests/compiler/backend/test_source_determinism.py
        "tests/compiler/backend/test_source_determinism.py::test_kernel_source_identical_across_processes",
        # tests/compiler/cli/test_compile.py
        "tests/compiler/cli/test_compile.py::test_compile_dynamic_emits_runtime_arg",
        "tests/compiler/cli/test_compile.py::test_compile_golden_substring_resolves_dynamic",
        # tests/compiler/cli/test_eval.py
        "tests/compiler/cli/test_eval.py::test_offer_audit_flags_pin_only_and_fall_through",
        # tests/compiler/e2e/test_attention_coverage.py
        "tests/compiler/e2e/test_attention_coverage.py::test_bare_sibling_pin_selects_the_f16acc_pv_plan[dynM]",
        "tests/compiler/e2e/test_attention_coverage.py::test_bare_sibling_pin_selects_the_f16acc_pv_plan[static]",
        "tests/compiler/e2e/test_attention_coverage.py::test_flash_form_fork_offers_f16acc_pv",
        "tests/compiler/e2e/test_attention_coverage.py::test_flash_form_fork_offers_geometry_grid",
        # tests/compiler/e2e/test_knob_pinning.py
        "tests/compiler/e2e/test_knob_pinning.py::test_sgemm_inner_reduce_is_unrolled",
        # tests/compiler/e2e/test_matmul_coverage.py
        "tests/compiler/e2e/test_matmul_coverage.py::test_batched_symbolic_mk_reaches_warp",
        "tests/compiler/e2e/test_matmul_coverage.py::test_cp_staged_slab_is_swizzled",
        "tests/compiler/e2e/test_matmul_coverage.py::test_f16acc_enumeration_gate",
        "tests/compiler/e2e/test_matmul_coverage.py::test_masked_symbolic_m_structure[cp]",
        "tests/compiler/e2e/test_matmul_coverage.py::test_masked_symbolic_m_structure[tma]",
        "tests/compiler/e2e/test_matmul_coverage.py::test_pinned_transport_and_shape_fire[dynamic-cp.async]",
        "tests/compiler/e2e/test_matmul_coverage.py::test_pinned_transport_and_shape_fire[dynamic-tma]",
        "tests/compiler/e2e/test_matmul_coverage.py::test_pinned_transport_and_shape_fire[static-cp.async]",
        "tests/compiler/e2e/test_matmul_coverage.py::test_pinned_transport_and_shape_fire[static-tma]",
        "tests/compiler/e2e/test_matmul_coverage.py::test_raster_default_is_the_flat_order",
        "tests/compiler/e2e/test_matmul_coverage.py::test_raster_fork_offers_both_orders",
        "tests/compiler/e2e/test_matmul_coverage.py::test_raster_gm_pin_groups_the_launch_order",
        "tests/compiler/e2e/test_matmul_coverage.py::test_raster_gn_pin_groups_the_transpose",
        "tests/compiler/e2e/test_matmul_coverage.py::test_raster_symbolic_grid_stays_flat",
        "tests/compiler/e2e/test_matmul_coverage.py::test_scalar_masked_n_stage_declines",
        "tests/compiler/e2e/test_matmul_coverage.py::test_scalar_matmul_stages_through_pipeline",
        "tests/compiler/e2e/test_matmul_coverage.py::test_tile_block_over_thread_limit_rejected",
        "tests/compiler/e2e/test_matmul_coverage.py::test_tma_stage_declines_below_sm90",
        "tests/compiler/e2e/test_matmul_coverage.py::test_tma_staged_slab_is_swizzled",
        "tests/compiler/e2e/test_matmul_coverage.py::test_trans_b_offers_staged_rows",
        "tests/compiler/e2e/test_matmul_coverage.py::test_transposed_b_symbolic_k_zero_fills",
        "tests/compiler/e2e/test_matmul_coverage.py::test_warp_matmul_stamps_wspec",
        "tests/compiler/e2e/test_matmul_coverage.py::test_warp_static_k_indivisible_rejected",
        # tests/compiler/passes/test_delegate_zero_init.py
        "tests/compiler/passes/test_delegate_zero_init.py::test_first_atomic_keeps_its_memset",
        "tests/compiler/passes/test_delegate_zero_init.py::test_oversized_accumulator_keeps_its_memset",
        "tests/compiler/passes/test_delegate_zero_init.py::test_second_atomic_delegates_to_first",
        # tests/compiler/passes/test_move_catalog.py
        "tests/compiler/passes/test_move_catalog.py::test_bare_reduce_forks_the_coop_catalog",
        "tests/compiler/passes/test_move_catalog.py::test_schedule_leaf_set_equals_catalog",
        "tests/compiler/passes/test_move_catalog.py::test_schedule_leaves_key_tile_canonically",
        "tests/compiler/passes/test_move_catalog.py::test_warp_staged_rows_fit_the_smem_budget",
        # tests/compiler/passes/test_recognize_boundary_rules.py
        "tests/compiler/passes/test_recognize_boundary_rules.py::test_mlp_gate_up_nodifies_as_two_channel_product_contraction",
        "tests/compiler/passes/test_recognize_boundary_rules.py::test_norm_linear_cone_is_an_inline_node_tree",
        "tests/compiler/passes/test_recognize_boundary_rules.py::test_norm_linear_fp32_keeps_map_rows_only",
        "tests/compiler/passes/test_recognize_boundary_rules.py::test_norm_linear_offers_map_rows_then_warp_contraction_rows",
        "tests/compiler/passes/test_recognize_boundary_rules.py::test_norm_linear_symbolic_m_offers_warp_rows",
        "tests/compiler/passes/test_recognize_boundary_rules.py::test_wide_m1_flinear_uses_single_warp_k_fold",
        # tests/compiler/passes/test_warp_eligible_stamp.py
        "tests/compiler/passes/test_warp_eligible_stamp.py::test_materialized_op_carries_warp_eligibility_stamp",
        # tests/compiler/pipeline/search/policy/test_dit_golden_deploy.py
        "tests/compiler/pipeline/search/policy/test_dit_golden_deploy.py::test_the_shipped_dit_flash_golden_decides_the_live_deploy",
        "tests/compiler/pipeline/search/policy/test_dit_golden_deploy.py::test_the_shipped_dit_golden_decides_the_live_deploy[dit_xl_2.attn_out_proj.s256-256-1152-1152]",
        "tests/compiler/pipeline/search/policy/test_dit_golden_deploy.py::test_the_shipped_dit_golden_decides_the_live_deploy[dit_xl_2.ff_out_proj.s256-256-1152-4608]",
        # tests/compiler/pipeline/search/policy/test_golden_evidence.py
        "tests/compiler/pipeline/search/policy/test_golden_evidence.py::test_attention_golden_decides_the_live_flash_fork[dynM]",
        "tests/compiler/pipeline/search/policy/test_golden_evidence.py::test_attention_golden_decides_the_live_flash_fork[static]",
        # tests/compiler/pipeline/search/prior/test_offline_prior.py
        "tests/compiler/pipeline/search/prior/test_offline_prior.py::test_offline_ranks_mma_above_every_scalar_split[dynamic]",
        "tests/compiler/pipeline/search/prior/test_offline_prior.py::test_offline_ranks_mma_above_every_scalar_split[static]",
        "tests/compiler/pipeline/search/prior/test_offline_prior.py::test_warp_eligible_stamp_fp16_present_fp32_absent",
        # tests/compiler/pipeline/search/test_bench_record.py
        "tests/compiler/pipeline/search/test_bench_record.py::test_bench_leaves_keys_by_offer_site",
        "tests/compiler/pipeline/search/test_bench_record.py::test_mma_path_records_and_joins_the_scalar_pool",
        # tests/compiler/pipeline/search/test_golden_spelling_canonical.py
        "tests/compiler/pipeline/search/test_golden_spelling_canonical.py::test_every_stored_golden_spelling_is_canonical",
        # tests/compiler/pipeline/search/test_keys.py
        "tests/compiler/pipeline/search/test_keys.py::test_static_and_symbolic_twins_never_collide_cuda_stage",
        # tests/compiler/pipeline/search/test_two_level.py
        "tests/compiler/pipeline/search/test_two_level.py::test_inner_reward_deeper_patience_benches_new_variants",
        "tests/compiler/pipeline/search/test_two_level.py::test_inner_reward_is_separable_not_a_product",
        # tests/compiler/pipeline/test_flash_form_narrowing.py
        "tests/compiler/pipeline/test_flash_form_narrowing.py::test_stage_pin_does_not_bypass_keyed_tile_pins",
        # tests/compiler/pipeline/test_golden_attention_pins.py
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[attention.hd128@4090_0]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[attention.hd128@4090_1]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[attention.hd128@4090_2]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[attention.hd128@5090_0]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[attention.hd128@5090_1]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[attention.hd128@5090_2]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[attention.hd128@5090_3]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[attention.hd64@4090_0]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[attention.hd64@4090_1]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[attention.hd64@4090_2]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[attention.hd64@5090_0]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[attention.hd64@5090_1]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[dit_xl_2.attn.s256@4080]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256.s2048@4090_0]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256.s2048@4090_1]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256.s2048@5090_0]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256.s2048@5090_1]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256.s4096@4090_0]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256.s4096@4090_1]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256.s4096@5090_0]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256.s4096@5090_1]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256@4090_0]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256@4090_1]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256@4090_2]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256@4090_3]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256@5090_0]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256@5090_1]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256@5090_2]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256@5090_3]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd256@5090_4]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd512.s2048@4090_0]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd512.s2048@4090_1]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd512.s2048@5090]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd512.s4096@4090_0]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd512.s4096@4090_1]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd512.s4096@5090]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd512@4090_0]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd512@4090_1]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd512@4090_2]",
        "tests/compiler/pipeline/test_golden_attention_pins.py::test_static_attention_golden_pins_bind[gemma4_12b.attention.hd512@5090]",
        # tests/compiler/pipeline/test_resolve.py
        "tests/compiler/pipeline/test_resolve.py::test_decide_score_lands_on_trace",
        "tests/compiler/pipeline/test_resolve.py::test_resolve_applies_in_place",
        "tests/compiler/pipeline/test_resolve.py::test_trace_records_partition_fork",
        # tests/compiler/test_golden_configs.py
        "tests/compiler/test_golden_configs.py::test_fast_math_golden_ranks_in_gated_enumeration",
        # tests/compiler/test_golden_drift_gate.py
        "tests/compiler/test_golden_drift_gate.py::test_gemma4_goldens_deploy_in_serving_twins[rtx4090]",
        "tests/compiler/test_golden_drift_gate.py::test_gemma4_goldens_deploy_in_serving_twins[rtx5090]",
    }
)


def scheduling_xfail(nodeid: str) -> str | None:
    """The xfail reason for ``nodeid``, or ``None`` if it is not a scheduler casualty."""
    return REASON if nodeid in NODEIDS else None
