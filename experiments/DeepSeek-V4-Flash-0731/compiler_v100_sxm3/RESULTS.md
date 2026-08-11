# DeepSeek-V4-Flash-0731 compiler qualification on 16× V100 SXM3 32 GB

Status: promotion-qualified. The complete static-sequence-512 architecture inventory is now stored at
`emmy/compiler/pipeline/search/goldens/v100_sm70_deepseek_v4_flash_0731.yaml`. The earlier partial artifacts remain
below as historical evidence and must not be substituted for the canonical file.

## Scope

- Qualification date: 2026-08-10
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Immutable model revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- Hardware: 16× `Tesla V100-SXM3-32GB`, compute capability 7.0, 512 GB aggregate GPU memory
- Trace profile: static sequence length 512, FP16 architecture and shape coverage, target `sm_70`

These artifacts use architecture-derived FP16 representative constants. They do not establish checkpoint
representation support, whole-model numerical parity, serving eligibility, or end-to-end serving performance.

## Canonical promotion

The corrected provider traced the four config-derived decoder profiles plus 11 model seams into 13 programs and 279
exact Loop configurations. All rows have paired positive deployable O3 `emmy_us` and live `emmy-greedy`
`reference_us` measurements; the canonical file contains no search ranking. Full collision qualification retained a
universally realized candidate for every one of the 59 ShapeKeys.

Provider fixes after the historical partial run preserve routed-expert clamp-10 semantics, materialize the real
sliding causal mask, keep HCA/CSA compressor block bias live, fail closed when representative MoE replacement or CSA
full-selection equivalence cannot be proved, and model static slice updates without observable alias mutation. A
bounded retrace then repeated O3 candidate-versus-greedy qualification for every changed target and every member of
its affected ShapeKey bucket.

The final uncovered in-model ShapeKey was a fused `bmm_reduce` present once in each decoder profile. Its natural
sync-stage candidate was measured directly from the freshly captured full-graph Loop node in two isolated repeats;
persisted standalone and origins-derived Loop forms were rejected because their structural identities differed. The
candidate median was 14.603–14.625 ms across the four occurrences versus 16.597–16.717 ms for live greedy, with exact
outputs and less than 0.10% candidate repeat range.

Final gates on the exact canonical hash `99578815fad014595a74fa53df0fe42f991562085abe3ba2a58c0a6ddc6fa309`
passed PROMOTION and REPOSITORY validation, unpinned offer replay for 279/279 entries with no fall-through, and the
pinned offline in-model audit at revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`: 255 matches, zero drift, zero
compile failures, and zero major gaps. `canonical_qualification.json` records the full contract and evidence hashes.

## Architecture inventory and semantic boundary

The inventory covered every config-derived decoder path and model seam:

- layers 0–1: sliding HCA plus hash MoE
- layer 2: CSA ratio 4 plus hash MoE
- odd layers 3–41: HCA ratio 128 plus learned MoE
- even layers 4–42: CSA ratio 4 plus learned MoE
- DSpark remapped layers 43–45
- token embedding, initial HC broadcast, final HC collapse, final norm, LM head, DSpark combine, Markov embedding,
  Markov bias, and confidence seams

The alias-aware run initially produced 575 raw entries, deduplicated to 69 exact single-Loop targets. All 69 lowered
and their 96 unique CUDA sources compiled for `sm_70` at O1. That is syntactic compiler evidence only: a later exact
semantic audit showed that target 34 had been generated from an unsound functional interpretation of `aten.copy_`.

The layer-2 compressor creates `new_kv`/`new_gate` bases, writes through destination slices with `copy_`, then reads
the original bases. The previous tracer failed closed at the first observable mutation:

```text
NotImplementedError: aten.copy_ observable alias mutation is unsupported: later live node 'slice_15' reads original
destination alias 'new_zeros'; functional copy_ is supported only through its returned value
```

The tracer now reassembles a static, unit-step slice update rooted at a local `new_zeros` or `new_full` allocation as
a two-source index map and versions sequential writes. Inputs, parameters, dynamic or strided slices, pre-existing
aliases, and used mutation returns remain fail-closed. An independent audit added an empty-slice no-op regression so
the new path never loads from a zero-sized source.

An intermediate exact offline retrace of layer 2 at sequence length 512 retained 612 FX nodes, pruned 226 dead nodes,
produced 1,120 Graph IR nodes, and saved 89 distinct exact Loop targets. All 58 focused remote trace tests passed.
Those then-unmeasured rows were superseded by the architecture-wide retrace and collision qualification described
above. Exact intermediate counts and hashes remain in `retrace-summary.json`.

## Equal-budget tuning

Both final arms started cold from isolated databases, online priors, cubin caches, and output directories. They used
all 16 homogeneous GPUs, seed 731, two candidates per target, patience 2, and O1 ranking compiles. A prior
contention-affected attempt was excluded before these final arms.

| Arm | Targets complete | Successful rows | `bench_fail` rows | Ranked winners |
| --- | ---: | ---: | ---: | ---: |
| MCTS-only | 69/69 | 129 | 109 | 21 |
| Model proposals plus MCTS | 69/69 | 97 | 120 | 21 |

The exact-Loop finalist selector chose 29 searched schedules and 40 measured greedy fallbacks. These results were
measured against the then-current 69-target inventory and remain useful continuation evidence, but the semantic trace
failure active at that time prevented model-wide promotion.

### Focused follow-up tuning

A follow-up first searched the six failed or unfinished targets on six GPUs. Three large linear-matmul reductions
still exceeded the watchdog; the other targets produced mean, linear, and RMSNorm candidates. It then targeted the
12 kernels responsible for 97.89% of the historical partial inventory's summed O3 time. Those two arms used isolated
databases, online priors, cubin caches, 12 GPUs, seed 731, `max-candidates=16`, patience 8, and the same O1 ranking
regime:

| Arm | Attempted rows | Successful rows | `bench_fail` rows |
| --- | ---: | ---: | ---: |
| MCTS-only | 148 | 104 | 44 |
| Recorded proposals plus MCTS | 148 | 105 | 43 |

Nineteen distinct winners then received two fresh deployable O3 pinned comparisons against isolated greedy selection
with identical deterministic inputs, 10 warmups, and 100 measured iterations. Eight were accepted, five had no
consistent O3 win, two candidate schedules hit the watchdog, and four remained unresolved because the isolated
greedy reference exceeded the aggregate watchdog.

| Accepted target | Candidate mean | Greedy mean | Speedup |
| --- | ---: | ---: | ---: |
| `k_matmul_reduce_9eab8c` | 5.365 ms | 11.350 ms | 2.12x |
| `k_rms_norm_linear_reduce_25fe50` | 15.224 us | 22.028 us | 1.45x |
| `k_matmul_86b937` | 42.368 ms | 54.057 ms | 1.28x |
| `k_matmul_9e323a` | 34.135 ms | 43.578 ms | 1.28x |
| `k_mean_0d59f1` | 2.285 us | 2.826 us | 1.24x |
| `k_matmul_f35d97` | 28.021 ms | 32.824 ms | 1.17x |
| `k_linear_reduce_5b3de5` | 54.911 ms | 57.204 ms | 1.04x |
| `k_matmul_softmax_reduce_152eac` | 34.376 ms | 34.412 ms | 1.001x |

The MCTS tensor-core schedule for `k_matmul_reduce_9eab8c` materially beat the recorded-proposal arm; that arm's
winning O1 tile was about twice as slow as greedy at O3. This is why no O1 result was promoted directly. At this
historical stage, three large linear-matmul reductions still exceeded the per-launch watchdog and duplicate ShapeKeys
still needed cross-target A/B. The later canonical run closed both requirements. Historical budgets, knobs, repeat
records, correctness status, artifact hashes, and unresolved rows are in `tuning-closure-summary.json`.

## Deployable O3 evidence

Every O3 attempt used one exact single-Loop program, explicit realized knobs, production O3 compiler flags, 10 warmup
iterations, 100 measured iterations, and accuracy comparison with its Loop reference. Once the semantic trace issue
invalidated complete coverage, three remaining long-running task shards were stopped to free the serving host.

- source inventory: 69 targets
- semantically invalid target excluded: 1 (index 34, `k_softmax_mean_reduce_342131.4dea1e03816a`)
- partial inventory: 68 targets
- O3 and Loop-reference verified: 62 targets, all with captured execution and positive timings
- verified source split: 27 searched schedules and 35 greedy fallbacks
- O3 watchdog failure: 1 target (index 31, runtime exceeded the 2-second per-launch guard)
- explicitly unfinished: 5 targets (indices 33, 49, 57, 63, and 65)

The 62 verified rows contain no accuracy error. Their reference backend is the Loop interpreter, so their timing
ratios are kernel qualification evidence rather than an end-to-end model speedup claim. Failed and unfinished targets
have no fabricated timing or knob record.

## Artifacts

- `partial_exact_loop.yaml`: 68-target non-canonical continuation artifact; 62 targets have explicit verified O3 rows
- `partial_o3_report.json`: per-target source, knobs, timings, accuracy status, failure, and unfinished records
- `partial_coverage.json`: decoder/seam inventory, semantic exclusion, and exact partial counts
- `evidence_hashes.json`: SHA-256 hashes of the source inventory, selection, shard records, and partial outputs
- `retrace-summary.json`: exact post-fix layer-2 trace counts, source hashes, and remote evidence hashes
- `tuning-closure-summary.json`: equal-budget focused search plus two-repeat O3 evidence for all 19 distinct winners
- `canonical_qualification.json`: canonical inventory, bounded closure contracts, fused-gap measurements, gate counts,
  source hashes, and remote artifact hashes
- `emmy/compiler/pipeline/search/goldens/v100_sm70_deepseek_v4_flash_0731.yaml`: canonical static-sequence-512 V100
  deployment evidence

The 132 MB raw search databases, priors, logs, pinned-run JSON, and retrace YAML are mirrored on the supplied node at
`/home/riftuser/onecat-dsv4-0731/optimization/compiler-tuning-closure`; paths in the summary are relative to that
directory.

The canonical V100 golden is promotion-qualified only for this FP16 static-sequence-512 architecture inventory. It is
deliberately separate from the 1Cat serving cache and endpoint qualification evidence.
