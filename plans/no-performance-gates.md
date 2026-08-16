# No performance gates: passes enumerate, evidence decides

Working plan for branch `feature/replayable-golden-rows` (PR #513). This PR is **pure removal** on top of main:
no new machinery. The invariant it enforces — stated in
`passes/ARCHITECTURE.md` — is that a pass never refuses, caps, or truncates a semantically legal alternative for
speed. Compiles answer every choice through the deploy evidence hierarchy (measured evidence → prior → option-0)
or an explicit pin; the offline prior owns cold-deploy quality. Recorded goldens are exactly replayable pinned
rows, never consulted by an unpinned compile.

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
- Placement cuts: the routing-evidence consult is gone (no pass reads goldens); pins are authoritative, and
  unpinned placement is now an enumerated STRUCTURAL fork (fused option-0 + one fragment per legal seam), so tune
  discovers cuts and a chosen cut records as `PLACE@<seam>: cut` on the parent piece — replayable as the exact pin.
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

Dissolution verdict (EXECUTED):

- The mixed-A PROMOTION is dead: a materialized edge the atom cannot bind directly takes the
  CONVERTING smem compute fill on the one tree (`_needs_fill` / `_converting_a`, the fill
  resolver's sync-side classification, the one-`Load` cone normalization at the materializer's
  decode boundary). No tree rewrite; row-count parity verified against the old two-reading fork.
- The COLLAPSE and the MONOID composition are honest DERIVED VIEWS, not transports: each changes
  the loop structure (the splice re-evaluates the cone per cell; the composition re-associates the
  monoid), so they cannot be spelled as a `STAGE` value. They remain as the two documented pure
  derivations of the one stored canonical tree (`_views`), and the row DECODES its view by `WORK`
  tier — the derived contraction view is warp-only by construction, the per-cell view never is.
- The identity machinery is deleted outright: the `owner` row→tree side table, the reading-index
  `origin`, the cross-reading collision raise, the pool `n_readings` assert, and the cached
  resolved slices. A pool is keys + rows; materialization re-resolves every slice from the row's
  spellings through the same dispatches enumeration used — replay is a function of
  `(stored op, row)` and nothing else.

Superseding the rename's classifier bridge (Dmitry: "drop everything, strongly prioritize clean
design; pinned kernel performance is the only thing we care about"): the classified-ShapeKey
golden DEPLOY TIER is deleted wholesale — greedy's golden consult (`_golden_pick`,
`_fork_shape_key`, `_golden_matches_row`, the audit sink), `_golden_shape_key` with its
offer-signal and `PLACE@a` special cases, the `S_computed_a` stamp (code + YAML), `search/audit.py`
and the `eval golden` deploy/offer audits. Goldens remain as data: named pinned rows replayed
exactly (`run --golden NAME`, `--ab`), offline-fit training data, eval datasets. The deploy
hierarchy is now measured reservoir/DB evidence → prior → option-0; deployed-model parity is
restored later by the strict structural-identity decode (below), never by re-adding fuzzy
matching. **Both are now LANDED**: the verified tier deploys on strict identity + exact row decode
(`_verified_index` / `_verified_pick`), and `search/audit.py`, the greedy verdict sink, and the
`eval golden` serving/offer audits are rebuilt on top of it — verdicts keyed by `deploy_identity`,
never by a shape. Still adopted:
the scalar tier's slab K-chunk on TILE's existing `/k<bk>` token (an added tuning dimension, not a
correction — deferred); per-edge `STAGE@a`/`STAGE@b` keys (same verdict: the single key is coherent —
the term decides each edge's fill — so the split only widens the space; deferred); `ZERO_DELEGATE` +
`VECTORIZE_STORES` knobs; `f8` restored to `map_tile_moves`. The `PLACE` fork level is LANDED
(placement is an enumerated structural fork; pins authoritative). Verdicts from the baseline sweep:
`Stage.smem` / `bk_elems` are documented resolution-derived fields on the resolved slice, not codec
pollution — keep; `ShapeKey` stays as the eval-side histogram descriptor (the deploy-join classifier
and its offer-signal special cases are what was junk, and they are gone); `evidence_row_vouches` is
principled prefix consistency for multi-pass decisions — keep; the `goldens_for_live_gpu` union
fallback serves explicit cross-card `--golden` replay (re-benched live) — keep; QuantSpec retirement
verified already landed (quant is in-graph algebra from birth, no IR metadata).
Documented hardcoded exceptions (99% rule): `_softmax` streaming, the stat-fill partition, the
derived codegen constants (pads, `setmaxnreg` split, swizzle picks, the f16-acc promote cadence —
precision, not perf), the bit-identical peepholes. The reduce tier's `_pick_coop` ordering
constants were on this list and are now DELETED — greedy takes no hand-written help at all, so
there is no ordering exception left to document.

## One algebra recognizer (LANDED on this PR)

Recognition now reads through two shared parsers and nothing else — the λ-fold reading
(`fold_from_loop`, byte-identity-gated) and the ⊗-lift reading (`_bilinear_reads`, shared by
`bind_contraction` and `bind_prologue_contraction`). The recognition-side contraction parser
(`_is_clean_contraction`'s clean/computed/both-computed decision tree) is deleted: candidacy is
liberal (`_bilinear_candidate` — one additive fold, a distributing two-argument lift, a K-indexed
load) and the ONE binder arbitrates, with the role geometry hardened there (role-exclusive A/B
leaves; cone arms require a K-walking load). The online-softmax pairing states its condition on
λ-fold results (raw loops normalized to the dissolved Accum-axes spelling before the gate). All
verified emission-neutral by the pinned-kernel digest gate (byte-identical across every step).
Remaining dispatch (which composition applies) is structural and small; the raw-loop escape and
symbolic-axis deferral are the recognizer's stated incompleteness, dying with its growth.

## Strict structural-identity decode (LANDED on this PR)

The verified tier is restored on exact identity: `deploy_identity` (algebra digest + dtype fingerprint) joins a
fork to records whose identity derives from their OWN persisted program through the shared recognition core;
`schedule_row_key` (the recording canonicalizer restricted to the schedule families) decodes fail-closed —
identity-match without an equal enumerated row warns and decides nothing. Routing records decide the placement
fork by parent-piece PLACE stamps; fused records hold the fused side. The corpus is migrated set by set (tokens +
strict decode + prune, tripwired per file in `test_golden_decode`); flash-era attention rows prune (SDPA is two
kernels now — re-seed as per-kernel loop targets in the next campaign).

## Follow-ups (separate PRs)
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
