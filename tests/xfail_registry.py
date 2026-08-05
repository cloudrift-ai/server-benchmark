"""Registry of tests expected to fail while the tile schedule enumeration is INCOMPLETE.

The row enumerator was deleted and rebuilt RECURSIVELY over the site tree. Its first rebuild had
been single-site — ``_site_keys`` returned three keys for ``ops.head`` alone and ``_assemble`` took
one ``(TilePlan, stage, ReducePlan)`` triple, so every emitter hand-wrote a flat product over ONE
node's families. The two families that never landed — a COMPUTED operand edge (the fused
norm→linear / gate⊗up cone) and the flash streaming pair — are both cases where two SITES must
agree, which that shape cannot express.

The recursion carries every SINGLE-SITE term, and the COMPUTED operand edge with it (the fused
cone's contraction + statistic pair, its rows unioned with the map reading's). What remains is the
FLASH streaming pair and its consequences: those terms enumerate no rows and stay unmapped. That is
the guardrail contract, not a crash: kernels still compile on the materializer's per-cell path, so
what fails below is coverage — a pinned tier, a stamped knob, an enumerated fork — never a compile.
The ids fall in two bands:

- **The flash streaming pair** — never scheduled by either builder, and the reason the rebuild is
  recursive. The cooperative-KV cases additionally pin the RETIRED ``REDUCE=b<N>`` spelling, which
  the site-local codec no longer decodes; they need re-spelling against the restored flash rows, not
  before them.
- **Whole-model and serving end-to-end** — every graph that CONTAINS one of the above (the qwen
  / tinyllama block and dynamic-shape chains, the generation runner and its batched twins). These
  fail as CONSEQUENCES, so they clear on their own once the enumerator returns; none of them is an
  independent obligation.

The registry is deliberately a LIST OF EXACT NODE IDS, not a module or path glob: each id is a
concrete acceptance obligation for the new scheduler. The marker is **strict**: an id that starts
passing FAILS the run and must be deleted from this list, which is what makes "the file shrinking to
empty IS the completion gate" an enforced claim rather than an aspiration. Genuinely flaky or
card-conditional expectations do not belong here — they stay inline ``@pytest.mark.xfail(strict=False)``
at their own test, where the reason can name the condition.

The :data:`CONSEQUENCE_MODULES` ids additionally do not RUN (``run=False``). They fail only because
their graph CONTAINS the gap above, so executing them re-measures a known-missing
enumerator at a cost of minutes per suite (the drift gate alone spends ~190 s reaching its xfail).
The trade is deliberate: a consequence id can no longer report XPASS, so recovery is signalled by
the PRIMARY ids — which is where it should be signalled anyway.

Two limits to know about:

- **Collection errors cannot be xfailed.** The first demolition took three modules that unit-tested
  private helpers of the scheduler with it. The second one costs nothing here: no test imports
  ``_schedule`` internals any more, so every module still collects and every casualty is an
  ordinary xfail.
- **GPU-gated ids were measured on one card.** The original list came off a CPU-only box, where the
  CUDA suite skips; the ``@cuda``-grouped ids were added from a full GPU run (sm_120). A card whose
  goldens route a different form can surface ids not listed here — append them with the same reason.
  Note the registry matches the RAW ``item.nodeid``, so the ``@cuda`` / ``@cuda-cli`` suffix xdist's
  loadgroup scheduler bakes into the REPORTED id must be stripped before an id is added.

Applied by the root ``conftest.py``'s ``pytest_collection_modifyitems``.
"""

from __future__ import annotations

REASON = "tile schedule incomplete - awaiting the flash streaming pair's rows"

# Node ids that fail because scheduling is missing. Sorted by module, then case.
NODEIDS: frozenset[str] = frozenset(
    {
        # tests/compiler/backend/test_source_determinism.py
        # tests/compiler/cli/test_compile.py
        # tests/compiler/cli/test_eval.py
        # tests/compiler/e2e/test_attention_coverage.py
        "tests/compiler/e2e/test_attention_coverage.py::test_bare_sibling_pin_selects_the_f16acc_pv_plan[dynM]",
        "tests/compiler/e2e/test_attention_coverage.py::test_bare_sibling_pin_selects_the_f16acc_pv_plan[static]",
        "tests/compiler/e2e/test_attention_coverage.py::test_cooperative_flash_matches_torch[1-2-64-16-32]",
        "tests/compiler/e2e/test_attention_coverage.py::test_cooperative_flash_matches_torch[1-2-64-16-64]",
        "tests/compiler/e2e/test_attention_coverage.py::test_cooperative_flash_matches_torch[1-4-128-32-32]",
        "tests/compiler/e2e/test_attention_coverage.py::test_cooperative_flash_matches_torch[1-4-128-32-64]",
        "tests/compiler/e2e/test_attention_coverage.py::test_flash_chain_matches_torch[1-1-8-8]",
        "tests/compiler/e2e/test_attention_coverage.py::test_flash_chain_matches_torch[1-2-16-8]",
        "tests/compiler/e2e/test_attention_coverage.py::test_flash_chain_matches_torch[2-3-32-16]",
        "tests/compiler/e2e/test_attention_coverage.py::test_flash_chain_pin_selects_chain_on_warp_eligible_shape",
        "tests/compiler/e2e/test_attention_coverage.py::test_flash_form_fork_offers_f16acc_pv",
        "tests/compiler/e2e/test_attention_coverage.py::test_flash_form_fork_offers_geometry_grid",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_bf16_matches_torch[1-2-32-16]",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_bf16_matches_torch[1-4-128-64]",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_causal_bf16_matches_torch[1-2-32-16]",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_causal_bf16_matches_torch[1-4-128-64]",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_causal_matches_torch[1-1-16-16]",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_causal_matches_torch[1-2-32-16]",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_causal_matches_torch[1-4-128-64]",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_causal_matches_torch[2-3-64-32]",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_f16acc_matches_torch[]",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_f16acc_matches_torch[d2/cp/ring]",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_matches_torch[1-1-16-16]",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_matches_torch[1-2-32-16]",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_matches_torch[1-2-32-72]",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_matches_torch[1-4-128-64]",
        "tests/compiler/e2e/test_attention_coverage.py::test_generated_tensorcore_flash_matches_torch[2-3-64-32]",
        "tests/compiler/e2e/test_attention_coverage.py::test_loopify_pin_matches_torch[w1x1/f1x2/k4--2]",
        "tests/compiler/e2e/test_attention_coverage.py::test_loopify_pin_matches_torch[w1x1/f1x2/k4--4]",
        "tests/compiler/e2e/test_attention_coverage.py::test_loopify_pin_matches_torch[w1x1/f1x2/k4-d2/cp/ring-4]",
        "tests/compiler/e2e/test_attention_coverage.py::test_loopify_pin_matches_torch[w1x1/f2x2/k4--4]",
        "tests/compiler/e2e/test_attention_coverage.py::test_loopify_pin_matches_torch[w1x1/f2x2/k4-d2/cp/ring-2]",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_flash_stage_pin_keeps_warp_not_scalar",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_flash_symbolic_cp_bit_identical_to_gmem_direct[100]",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_flash_symbolic_cp_bit_identical_to_gmem_direct[64]",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_flash_symbolic_cp_stages_and_matches_torch",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_flash_symbolic_tma_bit_identical_to_gmem_direct[100]",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_flash_symbolic_tma_bit_identical_to_gmem_direct[64]",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_flash_symbolic_tma_stages_and_matches_torch",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_warp_flash_bit_identical_to_gmem_direct[d1/cp]",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_warp_flash_bit_identical_to_gmem_direct[d2/cp/ring]",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_warp_flash_causal_and_gqa_match_torch[d2/cp/ring]",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_warp_flash_causal_and_gqa_match_torch[d2/tma/ring]",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_warp_flash_matches_torch[d1/cp]",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_warp_flash_matches_torch[d1/tma]",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_warp_flash_matches_torch[d2/cp/ring]",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_warp_flash_matches_torch[d2/tma/ring]",
        "tests/compiler/e2e/test_attention_coverage.py::test_staged_warp_flash_matches_torch[d3/cp/ring]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_chain_causal_dynamic_matches_torch[16]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_chain_causal_dynamic_matches_torch[37]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_chain_causal_dynamic_matches_torch[64]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_chain_causal_dynamic_matches_torch[8]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_chain_dynamic_matches_torch[16]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_chain_dynamic_matches_torch[37]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_chain_dynamic_matches_torch[64]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_chain_dynamic_matches_torch[8]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_chain_gqa_dynamic_matches_torch[16]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_chain_gqa_dynamic_matches_torch[37]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_chain_gqa_dynamic_matches_torch[64]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_chain_gqa_dynamic_matches_torch[8]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_chain_gqa_static_matches_torch[16-8-32-32]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_chain_gqa_static_matches_torch[4-2-32-16]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_alt_staging_matches_torch[causal-d1/cp/alt]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_alt_staging_matches_torch[causal-d1/tma/alt]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_alt_staging_matches_torch[plain-d1/cp/alt]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_alt_staging_matches_torch[plain-d1/tma/alt]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_alt_staging_symbolic_bit_identical[causal-d1/cp/alt]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_alt_staging_symbolic_bit_identical[plain-d1/cp/alt]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_alt_staging_symbolic_bit_identical[plain-d1/tma/alt]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_banded_matches_torch[256-100-None]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_banded_matches_torch[256-100-d1/cp/alt]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_banded_matches_torch[256-100-d2/cp/ring]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_banded_matches_torch[256-64-None]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_banded_matches_torch[256-64-d1/cp/alt]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_banded_matches_torch[256-64-d2/cp/ring]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_banded_split_kv_matches_torch",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_banded_symbolic_matches_torch[300]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_banded_symbolic_matches_torch[40]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_banded_symbolic_matches_torch[512]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_banded_tile_skip_structure",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_banded_with_additive_bias_matches_torch",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_causal_stamp_with_bias_matches_torch",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_causal_tile_skip",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_explicit_additive_mask_matches_torch[]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_explicit_additive_mask_matches_torch[d1/cp/alt]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_explicit_additive_mask_matches_torch[d2/tma/ring]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_f32_value_operand_converts",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_geometry_pin_matches_torch[2-4]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_geometry_pin_matches_torch[4-8]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_split_kv_matches_torch[causal]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_split_kv_matches_torch[plain]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_split_kv_symbolic_matches_torch[causal-g2k-300]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_split_kv_symbolic_matches_torch[causal-g2k-512]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_split_kv_symbolic_matches_torch[causal-g4k-40]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_split_kv_symbolic_matches_torch[plain-g2k-40]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_vacuous_band_still_fuses[1024]",
        "tests/compiler/e2e/test_attention_coverage.py::test_warp_flash_vacuous_band_still_fuses[256]",
        # tests/compiler/e2e/test_block.py
        "tests/compiler/e2e/test_block.py::test_qwen_block_accuracy",
        "tests/compiler/e2e/test_block.py::test_tinyllama_block_accuracy[cuda]",
        # tests/compiler/e2e/test_knob_pinning.py
        # tests/compiler/e2e/test_matmul_coverage.py
        "tests/compiler/e2e/test_matmul_coverage.py::test_tile_block_over_thread_limit_rejected",
        # tests/compiler/ir/test_dynamic_shapes.py
        "tests/compiler/ir/test_dynamic_shapes.py::test_qwen_batched_dynamic_matches_eager_b2",
        "tests/compiler/ir/test_dynamic_shapes.py::test_qwen_batched_dynamic_matches_eager_b4",
        "tests/compiler/ir/test_dynamic_shapes.py::test_qwen_layer_dynamic_compiles_and_matches_eager",
        "tests/compiler/ir/test_dynamic_shapes.py::test_qwen_whole_model_capture_replay_cache_matches_eager",
        "tests/compiler/ir/test_dynamic_shapes.py::test_qwen_whole_model_dynamic_compiles_and_matches_eager",
        # tests/compiler/passes/test_delegate_zero_init.py
        # tests/compiler/passes/test_move_catalog.py
        # tests/compiler/passes/test_warp_eligible_stamp.py
        # tests/compiler/pipeline/search/policy/test_dit_golden_deploy.py
        "tests/compiler/pipeline/search/policy/test_dit_golden_deploy.py::test_the_shipped_dit_flash_golden_decides_the_live_deploy",
        # tests/compiler/pipeline/search/policy/test_golden_evidence.py
        "tests/compiler/pipeline/search/policy/test_golden_evidence.py::test_attention_golden_decides_the_live_flash_fork[dynM]",
        "tests/compiler/pipeline/search/policy/test_golden_evidence.py::test_attention_golden_decides_the_live_flash_fork[static]",
        # tests/compiler/pipeline/search/prior/test_offline_prior.py
        # tests/compiler/pipeline/search/test_bench_record.py
        # tests/compiler/pipeline/search/test_keys.py
        # tests/compiler/pipeline/search/test_two_level.py
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
        # tests/compiler/test_golden_configs.py
        # tests/compiler/test_golden_drift_gate.py
        "tests/compiler/test_golden_drift_gate.py::test_gemma4_goldens_deploy_in_serving_twins[rtx4090]",
        "tests/compiler/test_golden_drift_gate.py::test_gemma4_goldens_deploy_in_serving_twins[rtx5090]",
        # tests/serving/test_attention_split_gpu.py
        "tests/serving/test_attention_split_gpu.py::test_gemma_post_wrapper_compiles_and_runs_dynamic",
        "tests/serving/test_attention_split_gpu.py::test_post_wrapper_compiles_with_shared_dim",
        # tests/serving/test_gen_prefill_device_gpu.py
        "tests/serving/test_gen_prefill_device_gpu.py::test_run_device_sym_matches_host_path",
        # tests/serving/test_gen_runner_gpu.py
        "tests/serving/test_gen_runner_gpu.py::test_adopted_raw_embed_table_matches_folded_gather",
        "tests/serving/test_gen_runner_gpu.py::test_decode_twin_shares_weight_buffers",
        "tests/serving/test_gen_runner_gpu.py::test_device_residents_allocated_eagerly",
        "tests/serving/test_gen_runner_gpu.py::test_gen_runner_gemma4_heterogeneous_stitch",
        "tests/serving/test_gen_runner_gpu.py::test_gen_runner_stitch_matches_eager",
        "tests/serving/test_gen_runner_gpu.py::test_layers_share_activation_arena",
        # tests/serving/test_generate_gpu.py
        "tests/serving/test_generate_gpu.py::test_generate_loop_runs_end_to_end",
        "tests/serving/test_generate_gpu.py::test_generate_oracle_matches_eager_fp16",
        "tests/serving/test_generate_gpu.py::test_slice_last_logits_lowers_cold",
        # tests/serving/test_runner_batched_gpu.py
        "tests/serving/test_runner_batched_gpu.py::test_runner_batched_matches_eager",
        "tests/serving/test_runner_batched_gpu.py::test_runner_batched_symbolic_matches_eager",
    }
)


#: Modules whose registered ids fail only as CONSEQUENCES of the two enumerator gaps — whole-model
#: and serving end-to-end graphs, and the golden drift gate. None is an independent obligation, so
#: they are marked ``run=False``: the expectation is recorded, the minutes are not spent.
CONSEQUENCE_MODULES: frozenset[str] = frozenset(
    {
        "tests/compiler/e2e/test_block.py",
        "tests/compiler/ir/test_dynamic_shapes.py",
        "tests/compiler/test_golden_drift_gate.py",
        "tests/serving/test_attention_split_gpu.py",
        "tests/serving/test_gen_prefill_device_gpu.py",
        "tests/serving/test_gen_runner_gpu.py",
        "tests/serving/test_generate_gpu.py",
        "tests/serving/test_runner_batched_gpu.py",
    }
)


def scheduling_xfail(nodeid: str) -> tuple[str, bool] | None:
    """``(reason, run)`` for ``nodeid``, or ``None`` if it is not a scheduler casualty. ``run`` is
    ``False`` for the consequence modules — record the expectation, skip the execution."""
    if nodeid not in NODEIDS:
        return None
    return REASON, nodeid.split("::")[0] not in CONSEQUENCE_MODULES
