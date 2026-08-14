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

## Adversarial audit round 2 (2026-08-14, landed on this PR)

- The flash pattern compiler is deleted end to end (recognition/re-synthesis, the bespoke twisted
  emitter, the streaming schedule tiers, the flash split-KV arm, the shape-key sniffs, the STAGE
  codec's dead `split` token, the flash golden rows). SDPA lowers through the generic path —
  verified: plain/causal/GQA/transposed-output each match torch at ~4e-7 in one generic kernel.
  `_softmax`'s streaming fusion stays (algebra, no model vocabulary).
- Placement cuts are pin-only (the routing-evidence consult is gone; no pass reads goldens).
- The sibling-axis σ fix landed for the scalar gmem-direct leaves (#507's named adjacent gap) —
  repairs the block-accuracy tests and the whole-model serving lane on the generic attention path.

## Schedule codec plan (revised 2026-08-14: transport-based, no READING family)

Tile IR is the canonical representation: ONE tree per kernel, and a scheduler that needs to
rewrite the tree is an IR/codec gap, not a mechanism. The readings machinery is that gap's
workaround, and it dissolves into the transport dimension of `STAGE`.

Transport naming (LANDED): the token names the intermediate storage plus its fill mechanism —
`smem` (the synchronous thread fill: a byte copy of materialized edges, or the compute fill
evaluating a computed edge into its slab; the term picks which), `smem-async` (cp.async),
`smem-tma` (TMA). An EMPTY `STAGE` is no intermediate at all: gmem→register on a materialized
operand, register-to-register on a computed one. Old spellings `copy`/`reg` merged into `smem`;
`cp` → `smem-async`, `tma` → `smem-tma`; goldens, tests and docs migrated.

Remaining dissolution steps:

- The mixed-A PROMOTION reading dies first: a materialized f32 edge under `smem` IS the
  converting evaluation — no tree rewrite.
- The COLLAPSE (`demoted`) reading dies next: the canonical tree keeps the cone nested; a
  per-cell row spells an empty `STAGE` on the computed edge (register-to-register) and
  materialization lowers it in place (the per-cell evaluation code already exists — it moves out
  of the reading into the row's decode).
- The MONOID-producer composition is the one reading not yet proven reducible; it goes last and
  the attempt is the verdict. If it resists, a one-value `READING` key is the fallback for that
  case alone.
- With the readings gone, the `owner` row→tree side table, the cross-reading collision raise,
  and the provably-no-op "`S_*` stamp" doctrine text are deleted — replay exactness holds by
  construction rather than by label.

Landed with the rename: `S_computed_a` stamped at enumeration (replaces greedy's transport-token
sniff, which the `copy`/`reg` merge made ambiguous; stored compute-fill golden rows carry the
stamp in their knob dicts). Still adopted:
the scalar tier's slab K-chunk on TILE's existing `/k<bk>` token; `ZERO_DELEGATE` +
`VECTORIZE_STORES` knobs; `f8` restored to `map_tile_moves`. Sequenced with the placement PR:
per-edge `STAGE@a`/`STAGE@b` keys and the `PLACE` fork level + a ShapeKey layout term.
Documented hardcoded exceptions (99% rule): `_softmax` streaming, the stat-fill partition,
`_pick_coop` ordering constants, the derived codegen constants (pads, `setmaxnreg` split,
swizzle picks, the f16-acc promote cadence — precision, not perf), the bit-identical peepholes.

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
