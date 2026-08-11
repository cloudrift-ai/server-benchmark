# Merge-loop gate removal experiment

Date: 2026-08-10

Base: `origin/main` at `2856a8e6` (`Prune model tests and speed up test infrastructure (#481)`)

The initial baseline artifacts were captured on its parent `9db3f1aa`. #481 only
changes golden loading and test infrastructure; after rebasing, structural
replay still has zero errors and exactly matches that 1,196-entry baseline.

Branch: `experiment/merge-loop-cuts`

## Experiment

`010_merge_loop_ops.py` was reduced to one policy cost gate: reject a merge when
`post_work > 8 * pre_work`. The following refusals were removed:

- reduce-heavy producers read more than once;
- cast/index-map and pending-contraction boundaries;
- flash/softmax recognition boundaries;
- reconvergent attention and interior-reducer boundaries;
- expensive-operation, read-traffic, and broadcast-materialization growth.

Closed-region ownership, a real N-way splicer rejection, and the `__cut_`
workspace fence remain structural invariants. They are not choices between a
legal fused and materialized realization.

## Baseline

Latest main structurally resolved every checked-in record:

| Card | Target regimes | Entries | Structural errors |
| --- | ---: | ---: | ---: |
| RTX 4080 sm_89 | 9 | 10 | 0 |
| RTX 4090 sm_89 | 279 | 297 | 0 |
| RTX 5090 sm_120 | 502 | 520 | 0 |
| RTX Pro 6000 sm_120 | 9 | 10 | 0 |
| Tesla V100 sm_70 | 359 | 359 | 0 |
| Total | 1,158 | 1,196 | 0 |

On the live RTX 5090, greedy reproduction selected an exact recorded schedule
for 120 of 294 matmul/fast-math regimes. This is a selection baseline, not a
golden-health percentage: most non-exact rows still realize through an offered
recorded sibling.

Main already had one unrelated golden-floor defect:
`gemma4_12b.gate_up_cat.m1024.lin` has no unpinned offered sibling and falls
through. Its recorded row is pin-only. This predates the experiment.
Overall, 4 of 520 baseline entries are pin-only: that row,
`gemma4_12b.gate_up_cat.m512.lin`, and the fused/unfused
`gemma4_12b.mlp_down.m4096` pair. The latter three retain an offered deploy
floor.

## Initial gate-free result

The change is isolated to attention at the standalone-target level:

| Card | Resolving regimes | Error regimes | Resolving entries | Error entries |
| --- | ---: | ---: | ---: | ---: |
| RTX 4080 sm_89 | 8 | 1 | 9 | 1 |
| RTX 4090 sm_89 | 260 | 19 | 268 | 29 |
| RTX 5090 sm_120 | 483 | 19 | 491 | 29 |
| RTX Pro 6000 sm_120 | 9 | 0 | 10 | 0 |
| Tesla V100 sm_70 | 359 | 0 | 359 | 0 |
| Total | 1,119 | 39 | 1,137 | 59 |

All 59 failures are SDPA records. The recorded
`scaled_dot_product_attention` provenance resolves to two structural targets,
so its stable target selector is ambiguous. The affected names are the generic
`attention.hd*` family plus the Gemma 4 `attention.hd256/hd512` fixed,
dynamic-M, s2048, and s4096 variants present on each card.

Every one of the other 1,137 entries keeps exactly the same `ShapeKey` as main.
On RTX 5090, the full 294-row greedy reproduction table is byte-for-byte
identical to the baseline, including the 120 exact matches. There is no new
non-attention schedule-knob drift.

The filtered 5090 offer audit is also unchanged: the same 4 entries are
pin-only and `gemma4_12b.gate_up_cat.m1024.lin` is still the sole fall-through
(4 of 491 audited entries after quarantining the 29 broken attention entries).
All 128 remaining fast-math entries realize unpinned.

The ordinary gate-free `eval golden` command aborts while building its config
list because it evaluates an attention `shape_key`. Deployment has a broader
failure mode: `_golden_evidence_index` builds a card in one exception boundary,
so one ambiguous attention record empties schedule-golden evidence for the
whole 5090 card. The filtered comparison quarantined the 29 broken 5090
attention entries to measure the remaining 491 entries independently.

For a concrete `attention.hd64` record:

- main recognizes one flash `TileOp` with the QK contraction absorbed;
- gate-free recognition produces three generic reduction `TileOp`s and no
  flash kernel.

The serving-twin drift gate found no new drift, compile failure, or uncovered
fork. It instead failed its reverse ratchets because the more aggressive fusion
closed several previously acknowledged coverage gaps on both 4090 and 5090.

## Can YAML placement pins repair it?

Not for the two newly exposed boundary classes.

- On the real `attention.hd64` golden, `PLACE=cut` changes nothing. Placement
  routing runs after `try_flash`; once loop fusion has destroyed the flash
  recognition boundary, a later cut can only split the generic recognized tree.
  It cannot reconstruct the missed flash composite. A routing golden derived
  from the same frontend provenance would also fail to derive its own shape key.
- The exp-to-linear regression has one bare cuttable `PLACE` site, but realizing
  it leaves a parent residue with no `Write` and compilation raises
  `ValueError: LoopOp body has no Write`. There is no alternate named site.

The placement mechanism itself remains healthy: all placement-routing tests
pass, including recursive cuts, authoritative fuse pins, and card-scoped routing.
No golden YAML pin was added because none repairs a newly degraded golden.

A minimal loop-fusion counterfactual does repair the attention family: restoring
only the refusal for a reduce-heavy producer read through more than one `Load`
returns all 1,196 records to structural resolution and restores
`attention.hd64` to one flash kernel. That probe was reverted; the experiment
branch intentionally remains compute-cap-only. This is the smallest observed
non-placement exception if the branch is hardened later.

## Flash recognizer follow-up

The existing `try_flash` / `_recognize` path was extended across the boundaries
created by gate-free loop fusion. No second recognizer or pass was added. The
same flash fragment builder now accepts:

- the original materialized-score boundary;
- a bare P×V root whose probability producer contains rowmax, softmax, and
  repeated inlined Q×K contractions;
- a fully fused small SDPA root, plus the split normalization boundary used by
  banded attention;
- exact computed Q/K operand edges, including fused RMSNorm cones. A closed
  pure-map V cone is factored into a canonical feeder workspace so the flash
  expectation operand remains materialized and stageable.

That recovers all 59 initially broken SDPA entries without restoring a merge
gate or adding a YAML pin. All 1,196 checked-in entries now resolve, and all
1,158 target regimes derive exactly the same `ShapeKey` as the main baseline.
Legacy dynamic flash records use a structural certification marker because
their older bare `TILE` rows depended on an exp histogram that gate-free fusion
moves from the consumer into its producer.

The reduce-heavy-producer counterfactual above is therefore no longer needed
for flash. The remaining observed gate-removal failures are outside attention:
transcendental duplication across contraction columns and the tiny two-linear
runtime miscompile.

## Verification

- `ruff check` and `git diff --check`: pass.
- The combined fusion, recognition-boundary, placement-routing, golden-config,
  and golden-policy suite: 127 pass, 1 skip, and 1 expected experiment failure
  (transcendental duplication across a contraction).
- Attention coverage: all 123 attention cases pass. The sole failure in that
  file is the unrelated tiny two-linear runtime miscompile.
- All-card structural replay: 1,196/1,196 entries resolve; 1,158/1,158 regime
  keys exactly match the baseline.
- Serving-twin drift still reports the known gate-free Gemma MLP offer drift;
  it is not caused by flash recognition.
- #481's duration guard also reports slow tests absent from the newly pruned
  `tests/durations.json`; those bookkeeping errors are independent of results.

Raw, ignored artifacts are under `_tune/merge-loop-cuts/`, including
`baseline.log`, the before/after all-card key maps, structural summaries, the
gate-free abort, and the filtered gate-free reproduction/offer audits.
