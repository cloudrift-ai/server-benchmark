"""The total-lift rebuild registry — the whole-subsystem transition mechanism from
``tests/ARCHITECTURE.md``: the recognize-time binders (``_atomize``, ``_softmax``, the old
``_lift`` classification) were deleted and recognition is being rebuilt as classification passes
over the total lift's Fold tree. Every casualty is registered here BY EXACT NODE ID and applied
as a strict xfail from the root ``conftest.py``; files that import the deleted binders directly
are skipped wholesale. Delete entries as the classification passes land; delete this module when
it empties.

Casualty classes:
- contraction binding (``bind_contraction`` / warp tier): every fold now derives PLANAR until the
  tree-level binder lands, so warp/mma/fp8/dynamic-shape kernels lose their tensor-core tier;
- online-softmax pairing (``_fuse``): two-pass softmax stays two sibling folds;
- the monoid-producer composition (``bind_prologue_contraction``): no fused norm->linear view, no
  cut-fork routing.
"""

# Files importing deleted recognize-time symbols -- skipped at collection (paths relative to tests/).
IGNORED_FILES = [
    "compiler/e2e/test_reduce_coverage.py",
    "compiler/passes/test_fp8_mma.py",
    "compiler/passes/test_fp8_operand_binding.py",
    "compiler/passes/test_online_softmax_channels.py",
]

# Exact node ids xfailed (strict) until their subsystem is rebuilt.
XFAIL_NODES = {
    "tests/compiler/ir/test_dynamic_shapes.py::test_cuda_sdpa_over_symbolic_seq_len",
    "tests/compiler/ir/test_dynamic_shapes.py::test_cuda_softmax_over_symbolic_seq_len",
    "tests/compiler/ir/test_dynamic_shapes.py::test_qwen_batched_dynamic_matches_eager_b2",
    "tests/compiler/ir/test_dynamic_shapes.py::test_qwen_batched_dynamic_matches_eager_b32",
    "tests/compiler/ir/test_dynamic_shapes.py::test_qwen_batched_dynamic_matches_eager_b4",
    "tests/compiler/ir/test_dynamic_shapes.py::test_qwen_layer_dynamic_compiles_and_matches_eager",
    "tests/compiler/ir/test_dynamic_shapes.py::test_qwen_whole_model_capture_replay_cache_matches_eager",
    "tests/compiler/ir/test_dynamic_shapes.py::test_qwen_whole_model_dynamic_compiles_and_matches_eager",
    "tests/compiler/passes/test_fp8_staged.py::test_canonical_byte_b_and_splitk_compose_cuda",
    "tests/compiler/passes/test_fp8_staged.py::test_k32_staged_bit_identical_to_gmem_direct_cuda",
    "tests/compiler/passes/test_fp8_staged.py::test_w8a16_staged_bit_identical_to_gmem_direct_cuda",
    "tests/compiler/passes/test_move_catalog.py::test_schedule_leaf_set_equals_catalog",
    "tests/compiler/passes/test_move_catalog.py::test_warp_staged_rows_fit_the_smem_budget",
    "tests/compiler/passes/test_placement_routing.py::test_a_cut_taken_at_a_fork_mid_batch_still_reaches_the_stamp",
    "tests/compiler/passes/test_placement_routing.py::test_norm_linear_cone_cut_recurses_to_the_full_cascade",
    "tests/compiler/passes/test_placement_routing.py::test_place_sites_are_the_non_root_nodes",
    "tests/compiler/passes/test_placement_routing.py::test_rms_norm_place_cut_splits_stat_and_scale",
    "tests/compiler/passes/test_placement_routing.py::test_scoped_place_pin_from_replay_context_cuts_the_cone",
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_bind_contraction_declined_cone_raises_not_positional",
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_channels_with_agreeing_b_layouts_form_one_product_node",
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_channels_with_disagreeing_b_layouts_never_group",
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_duplicated_cone_with_commuted_args_still_shares_one_a",
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_lift_partitions_independent_reduce_and_epilogue_preamble",
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_lift_recognizes_contraction_between_views_of_same_packed_buffer",
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_masked_score_cone_keeps_its_predicate_per_cell",
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_masked_sdpa_reaches_the_computed_a_contraction[False]",
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_masked_sdpa_reaches_the_computed_a_contraction[True]",
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_mlp_gate_up_nodifies_as_two_channel_product_contraction",
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_norm_linear_cone_is_an_inline_node_tree",
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_norm_linear_offers_both_the_map_rows_and_the_warp_contraction_rows",
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_norm_linear_symbolic_m_offers_warp_rows",
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_online_softmax_pairs_two_composed_passes",
    "tests/compiler/passes/test_schedule_pool_cache.py::test_a_dtype_change_keys_a_different_pool",
    "tests/compiler/passes/test_schedule_pool_cache.py::test_a_live_pin_keys_a_different_pool",
    "tests/compiler/passes/test_split_fresh_kernels.py::test_each_piece_decides_its_own_row",
    "tests/compiler/passes/test_volta_mma.py::test_modern_computed_a_edge_keeps_the_cp_async_peer_copy",
    "tests/compiler/passes/test_volta_mma.py::test_modern_mma_source_does_not_gain_the_volta_prelude",
    "tests/compiler/passes/test_volta_mma.py::test_sm70_computed_a_edge_stages_through_the_smem_compute_fill[d1/smem]",
    "tests/compiler/passes/test_volta_mma.py::test_sm70_computed_a_edge_stages_through_the_smem_compute_fill[d2/smem]",
    "tests/compiler/passes/test_volta_mma.py::test_sm70_register_tile_keeps_the_volta_fragment_layout_through_the_reroll",
    "tests/compiler/passes/test_volta_mma.py::test_sm70_rejects_a_pinned_modern_atom",
    "tests/compiler/passes/test_volta_mma.py::test_sm70_rejects_a_pinned_newer_stage[d1/smem-async-cp.async requires sm_80]",
    "tests/compiler/passes/test_volta_mma.py::test_sm70_rejects_a_pinned_newer_stage[d1/smem-tma-TMA requires sm_90]",
    "tests/compiler/passes/test_volta_mma.py::test_sm70_source_uses_only_the_volta_mma_family[False]",
    "tests/compiler/passes/test_volta_mma.py::test_sm70_source_uses_only_the_volta_mma_family[True]",
    "tests/compiler/passes/test_volta_mma.py::test_sm70_sync_copy_composes_ring_and_register_pipelines",
    "tests/compiler/passes/test_volta_mma.py::test_sm70_sync_copy_stages_fragments_without_newer_instructions[False]",
    "tests/compiler/passes/test_volta_mma.py::test_sm70_sync_copy_stages_fragments_without_newer_instructions[True]",
    "tests/compiler/passes/test_warp_eligible_stamp.py::test_materialized_op_carries_warp_eligibility_stamp",
}
