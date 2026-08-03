"""Registry of tests expected to fail while the tile scheduler is INCOMPLETE.

The tile scheduler was deleted wholesale and is being rebuilt as ONE generic row enumerator
(``_schedule.py`` + the ``020_schedule`` rule): sites → per-family typed slices → assembled rows →
one ``build_fork_tree``. The first cut restored the roles whose operand edges are all MATERIALIZED —
``FREE`` (pointwise + the register strip), ``PLANAR`` / ``TWISTED`` (the reduce partition) and
``CONTRACTION`` (the tile × stage × reduce × raster product, scalar and warp tiers, split-K) — and
those ids were deleted from this list.

What remains below is what the enumerator still declines to schedule, leaving the term unmapped:

- **COMPUTED operand edges** — the fused norm→linear / gate⊗up cone, whose A edge is an inline node
  rather than a gmem ``Load`` (the sync compute-fill rows and the two-reading merge).
- **The flash streaming pair** — the ``TWISTED`` warp / chain / split-KV forms over a hoisted QK
  operand edge and a derived PV contraction.

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
        # tests/compiler/e2e/test_attention_coverage.py
        "tests/compiler/e2e/test_attention_coverage.py::test_bare_sibling_pin_selects_the_f16acc_pv_plan[dynM]",
        "tests/compiler/e2e/test_attention_coverage.py::test_bare_sibling_pin_selects_the_f16acc_pv_plan[static]",
        "tests/compiler/e2e/test_attention_coverage.py::test_flash_form_fork_offers_f16acc_pv",
        "tests/compiler/e2e/test_attention_coverage.py::test_flash_form_fork_offers_geometry_grid",
        # tests/compiler/passes/test_recognize_boundary_rules.py
        "tests/compiler/passes/test_recognize_boundary_rules.py::test_mlp_gate_up_nodifies_as_two_channel_product_contraction",
        "tests/compiler/passes/test_recognize_boundary_rules.py::test_norm_linear_cone_is_an_inline_node_tree",
        "tests/compiler/passes/test_recognize_boundary_rules.py::test_norm_linear_offers_map_rows_then_warp_contraction_rows",
        "tests/compiler/passes/test_recognize_boundary_rules.py::test_norm_linear_symbolic_m_offers_warp_rows",
        # tests/compiler/pipeline/search/policy/test_dit_golden_deploy.py
        "tests/compiler/pipeline/search/policy/test_dit_golden_deploy.py::test_the_shipped_dit_flash_golden_decides_the_live_deploy",
        # tests/compiler/pipeline/search/policy/test_golden_evidence.py
        "tests/compiler/pipeline/search/policy/test_golden_evidence.py::test_attention_golden_decides_the_live_flash_fork[dynM]",
        "tests/compiler/pipeline/search/policy/test_golden_evidence.py::test_attention_golden_decides_the_live_flash_fork[static]",
        # tests/compiler/pipeline/search/test_golden_spelling_canonical.py
        "tests/compiler/pipeline/search/test_golden_spelling_canonical.py::test_every_stored_golden_spelling_is_canonical",
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
        # tests/compiler/test_golden_drift_gate.py
        "tests/compiler/test_golden_drift_gate.py::test_gemma4_goldens_deploy_in_serving_twins[rtx4090]",
        "tests/compiler/test_golden_drift_gate.py::test_gemma4_goldens_deploy_in_serving_twins[rtx5090]",
    }
)


def scheduling_xfail(nodeid: str) -> str | None:
    """The xfail reason for ``nodeid``, or ``None`` if it is not a scheduler casualty."""
    return REASON if nodeid in NODEIDS else None
