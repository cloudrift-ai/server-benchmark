"""The registry of known-failing tests, applied as ``xfail`` marks at collection.

A test lands here when the behavior it asserts was deliberately removed or is knowingly broken,
and the test is being kept — not deleted — because the behavior may come back. Keeping the body
intact means restoring the feature restores its coverage; the registry entry is the audit trail
in the meantime.

``pytest_collection_modifyitems`` in ``tests/conftest.py`` reads this module and marks each listed
item. Matching is on the pytest node id relative to the repo root, either exactly
(``tests/foo.py::test_bar``) or on the un-parametrized prefix (``tests/foo.py::test_bar`` also
covers ``tests/foo.py::test_bar[case]``) so a whole parametrization can be listed once.

``strict`` (default ``True``) makes an unexpected PASS a failure — that is the ratchet: when the
removed behavior returns, or the test is repaired, the registry entry must be deleted. Set it to
``False`` only for a test that still passes but does so **vacuously** (its subject is gone, so the
assertions no longer prove anything); those are listed so the reader knows the coverage is hollow.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Reasons ----------------------------------------------------------------

_PLACE_REMOVED = (
    "the PLACE structural-placement knob was removed (branch feature/remove-place-knob): "
    "the pin family, its realizer passes (020_cut_edge / 025_sink_row_reduce / 032_fuse_finalize) "
    "and the evidence/golden plumbing that carried PLACE@ stamps are all gone, so recognition "
    "always takes the built-in default placement"
)

_PLACE_VACUOUS = (
    "the PLACE knob was removed (branch feature/remove-place-knob): this test still passes, but "
    "VACUOUSLY — the pin it sets is inert and the alternative placement it contrasts against no "
    "longer exists, so the assertions prove nothing"
)

_PLACE_GOLDENS_COMMENTED = (
    "the PLACE knob was removed (branch feature/remove-place-knob) and the golden entries that "
    "recorded a PLACE@ placement are commented out in search/goldens/*.yaml, so the shapes they "
    "covered are uncovered again"
)


_GOLDEN_EV = "tests/compiler/pipeline/search/test_golden_evidence_deploy.py"


@dataclass(frozen=True)
class XfailEntry:
    reason: str
    strict: bool = True


# --- The registry -----------------------------------------------------------
#
# Keyed by node id relative to the repo root; a bare ``file::test`` key also covers every
# parametrization of that test.

XFAIL_TESTS: dict[str, XfailEntry] = {
    # -- PLACE@cone: the producer-cone cut ------------------------------------
    # test_place_cone_cut_splits_the_kernels / test_place_cone_cut_degenerate_m1 came BACK with
    # the phase-4 routing realizer (lowering/tile/_cut.py) — their entries are deleted per the
    # registry contract. The multi-fold variant stays xfailed: the plan deliberately forfeits
    # per-component separation at tile level (the sanctioned fused-edge cut is the shared `a`
    # operand; the old 020 realizer's N-channel split measured null — #389).
    "tests/compiler/e2e/test_fused_edge.py::test_place_cone_cut_splits_multi_fold": XfailEntry(_PLACE_REMOVED),
    "tests/compiler/passes/test_fusion_rules.py::test_cut_workspace_producer_never_refuses": XfailEntry(_PLACE_REMOVED),
    "tests/compiler/passes/test_recognize_boundary_rules.py::test_norm_linear_warp_pick_is_computed_a_contraction": XfailEntry(
        _PLACE_REMOVED
    ),
    "tests/compiler/test_golden_configs.py::test_fused_golden_requires_a_cone_anchor": XfailEntry(_PLACE_REMOVED),
    # (The PLACE@stat stat-sink and PLACE@fin fuse-finalize test files were DELETED, not
    # registered: they imported the deleted realizer modules and built the retired Fold step
    # spelling, so their bodies could never be repaired by restoring the feature.)
    # -- PLACE@fold / PLACE@tuple: the flash + online-softmax escapes --------
    "tests/compiler/cli/test_compile.py::test_compile_unfused_softmax_loopify_survives_dim": XfailEntry(_PLACE_VACUOUS, strict=False),
    # -- The evidence / featurizer plumbing that carried PLACE@ stamps -------
    "tests/compiler/pipeline/test_knob.py::test_knob_features_cut_roundtrip": XfailEntry(_PLACE_REMOVED),
    "tests/compiler/pipeline/search/test_db_evidence_deploy.py::test_db_pick_stale_row_never_vouches_for_the_cut_twin": XfailEntry(
        _PLACE_REMOVED
    ),
    _GOLDEN_EV + "::test_place_golden_does_not_match_a_fork_that_never_offered_it": XfailEntry(_PLACE_REMOVED),
    _GOLDEN_EV + "::test_cut_row_is_evidence_only_never_a_model_pick": XfailEntry(_PLACE_REMOVED),
    # The schedule fork tree lost its ``PLACE@fin`` level, so the catalog leaf count no longer
    # includes the fuse-mirror rows this test counts.
    "tests/compiler/passes/test_move_catalog.py::test_schedule_leaf_set_equals_catalog": XfailEntry(_PLACE_REMOVED),
    # -- Collateral: goldens that recorded a PLACE@ placement ----------------
    # The pin-only offer audit and the in-model coverage ratchet both read the golden set; the
    # commented-out PLACE@ entries leave their shapes uncovered.
    "tests/compiler/cli/test_eval.py::test_offer_audit_flags_pin_only_and_fall_through": XfailEntry(_PLACE_REMOVED),
    "tests/compiler/test_golden_drift_gate.py::test_gemma4_goldens_deploy_in_serving_twins": XfailEntry(_PLACE_GOLDENS_COMMENTED),
}


def lookup(nodeid: str) -> XfailEntry | None:
    """The registry entry for ``nodeid`` — an exact match, else a match on the node id with its
    ``[param]`` suffix stripped (so one key covers a whole parametrization). ``None`` when the
    test is not registered."""
    hit = XFAIL_TESTS.get(nodeid)
    if hit is not None:
        return hit
    base, sep, _ = nodeid.partition("[")
    return XFAIL_TESTS.get(base) if sep else None
