# DeepSeek-V4-Flash-0731 compiler qualification on 16× V100 SXM3 32 GB

Status: partial compiler experiment. The architecture-wide trace exposed an output-live aliased mutation that Graph IR
cannot represent. The partial YAML preserves valid tuning and O3 evidence, but it is not a complete model golden and
must not be copied into `emmy/compiler/pipeline/search/goldens/`.

## Scope

- Date: 2026-08-09
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Immutable model revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- Hardware: 16× `Tesla V100-SXM3-32GB`, compute capability 7.0, 512 GB aggregate GPU memory
- Trace profile: static sequence length 512, FP16 architecture and shape coverage, target `sm_70`

These artifacts use architecture-derived FP16 representative constants. They do not establish checkpoint
representation support, whole-model numerical parity, serving eligibility, or end-to-end serving performance.

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
the original bases. Graph IR can model consumers of the returned `copy_` value, but it has no functional slice update
that reassembles the base. The tracer now fails closed at the first observable mutation:

```text
NotImplementedError: aten.copy_ observable alias mutation is unsupported: later live node 'slice_15' reads original
destination alias 'new_zeros'; functional copy_ is supported only through its returned value
```

The sparse `topk`/`scatter_` branch investigated earlier is output-dead and remains correctly pruned. The live
`copy_` mutation above is the honest first unsupported operation. A focused regression proves the failure, while the
existing direct-return functional `copy_`, mutation-through-returned-view liveness, and dead-local-mutation cases
remain green.

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
failure prevents model-wide promotion.

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

No canonical V100 golden is produced. Complete promotion requires a semantically correct functional slice-update/base
reassembly representation, a fresh full trace, and deployable O3/reference measurements for every resulting target.
