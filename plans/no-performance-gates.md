# No performance gates: passes enumerate, evidence decides

Working plan for branch `feature/replayable-golden-rows` (PR #513). This PR is **pure removal** on top of main:
no new machinery. The invariant it enforces — stated in
`passes/ARCHITECTURE.md` — is that a pass never refuses, caps, or truncates a semantically legal alternative for
speed. Deployed models answer every choice from their tuned per-GPU golden
file; cold compiles answer through the deploy evidence hierarchy ending at the offline prior, which owns
cold-deploy quality.

## Removed in this PR

- Fusion's duplicated-contraction guard + its online-softmax exemption + the activation/product ordering rule.
  `_BLOWUP_FACTOR` stays strictly as compile boundedness (measured: without it a TinyLlama layer splices into one
  57-loop nest with a ~10¹³-iteration statistic replay; with it, 6 kernels — kernels 0–3 byte-identical to main).
- The trusted-online-prior requirement on structural pricing (any loaded prior prices kernel sets).
- `_SPLITK_MAX_CTAS` (contraction + flash split-KV), `_CHAIN_MAX_D`, the coop-band layout filter (both lane
  orientations always enumerated; B's classified layout only orders cold option-0), the `f16acc_is_faster`
  profitability conjunct and its `Context` property, `_MAX_DELEGATED_WORDS`.
- Every doc/comment description of the removed gating, so no textual precedent remains.

## Follow-ups (separate PRs)

- Replayable golden rows: recorded rows carry placement, exact fail-closed decode, deletion of the fuzzy golden
  matching (`_golden_matches_row`, offer sniffs, any-of/bare-wildcard semantics), per-card corpus migration. Needs
  the kernel-set persistence foundation, reviewed on its own.
- Delegation as a knob (`ZERO_DELEGATE`) and the other declared-but-unsearched policy BOOLs becoming search
  dimensions; the below-codec constants on the guardrail allowlist (slab pads, `setmaxnreg` split,
  `_F16ACC_STEPS`) gaining codec spellings.
- `space.py` ladder extensions (f8 strip, `gn<G>` rasters, wider split-K, fragment-cell cap) with MAX_ROWS
  headroom checks; unconditional perf rewrites (`035_merge_sibling_linears`, `007_sink_narrowing_cast`,
  `040_demote_to_write_dtype`, `096_pair_ldmatrix_loads`, `_fold_constant`'s weight-layout gate) becoming
  two-sided forks; decode-bucket reconciliation; a layout term in `ShapeKey` so cross-orientation evidence rows
  cannot tie.
- Gate removal re-keys some shapes (bigger fused regions): per-card golden re-verification on the available boxes
  (5090 local, V100, A100) before the next golden-bench campaign.
