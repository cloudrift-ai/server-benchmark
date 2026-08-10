# Ling 3.0 Flash FP8 compiler qualification on 8× V100 SXM2 16 GB

Status: **partial compiler coverage with partial tuning; no repository golden**. The exact model-level MTP
token-roll seam is unsupported by the current frontend. The successful diagnostic inventory is preserved in
[`partial_traced_working.yaml`](partial_traced_working.yaml), but it is not deploy evidence and must not be copied
into the repository golden directory.

## Scope

- Date: 2026-08-09
- Repository revision: `9daede61bab735aca99b1f4afc4b0f4af905fa74`, plus the compiler changes under review
- Model: `inclusionAI/Ling-3.0-flash-fp8`
- Immutable model revision: `a5d248fcca98b9d9a0c225cc22372f2fd1b3540b`
- Hardware: 8× `Tesla V100-SXM2-16GB`, compute capability 7.0, driver 580.159.03
- Trace shape: batch 1, sequence length 8, hidden size 2560
- Environment: Python 3.12, PyTorch `2.13.0+cu126`, Transformers `4.56.2`, FLA `0.5.2`, CUDA 12.9, and
  CuPy `13.6.0` with CUDA 12.9 NVRTC

The checkpoint contains 128,443,021,752 tensor bytes (119.62 GiB). Perfectly balanced over eight 16 GiB cards,
weights alone consume about 14.95 GiB per GPU and leave about 1.05 GiB before CUDA contexts, collectives,
workspaces, attention state, and KV cache. The exact checkpoint therefore failed the serving fit gate independently
of compiler eligibility. No V100 serving recipe or image is published.

## Coverage manifest

The immutable configuration defines 42 decoder layers and one MTP layer. The full machine-readable mapping is in
[`coverage.json`](coverage.json).

| Architecture path | Layer indices | Traced representative |
| --- | --- | --- |
| Dense SwiGLU + KDA | 0–1 | Layer 0 |
| Sparse MoE + KDA | 2–4, 6–10, 12–16, 18–22, 24–28, 30–34, 36–40 | Layer 2 |
| Sparse MoE + MLA | 5, 11, 17, 23, 29, 35, 41 | Layer 41 |
| MTP + MLA + sparse MoE | 42 | Layer 42 |

The model-level inventory also covers token embedding, rotary generation, final RMSNorm, the main output head,
shifted-token embedding, the MTP transition, and the MTP output head. The representative sparse block contains one
exact routed expert and the always-on shared expert. Routing, top-k, sort, and weighted combination remain host
orchestration under Emmy's representative-MoE trace contract.

The diagnostic whole-architecture artifact records checkpoint-compatible source paths for
`model.word_embeddings.weight`, `model.rotary_emb.inv_freq`, `model.norm.weight`, `lm_head.weight`, and layers 0,
2, 41, and 42. It spells 38 checkpoint weights as FP8 E4M3 constants from immutable safetensors metadata while
preserving the checkpoint's dynamic-activation, 128×128 weight-block layout.

## Successful diagnostic trace

Three library-only surfaces use semantically equivalent Torch algebra in the diagnostic artifact:

- FLA short depthwise causal convolution uses explicit causal slices, concatenation, and a weighted sum.
- FLA KDA and fused gated RMSNorm use the exact no-cache recurrence, safe lower-bound gate, L2 normalization,
  beta update, and RMSNorm/gate equations.
- The rotary autocast wrapper uses the exact configuration-specific matrix product, concatenate, cosine/sine, and
  scale algebra.

On the target GPU, original FLA KDA and its replacement are finite and agree with maximum absolute error
0.00390625, mean absolute error 0.00015323, and maximum error divided by reference peak 0.0007788. The comparison
passes FP16 `rtol=5e-3, atol=5e-3`. Original rotary and its replacement agree bit-for-bit: both cosine and sine have
maximum absolute error 0.0.

| Artifact | Frontend nodes | FP8 constants | Targets | SM70 CUDA lowering |
| --- | ---: | ---: | ---: | ---: |
| Whole architecture with diagnostic token-roll replacement | 1,978 | 38 | 341 | 341 / 341 |
| Model seams + MTP representative | 314 | 9 | 54 | 54 / 54 |
| MTP representative alone | 256 | 9 | 43 | 43 / 43 |

The seam wrapper contributes 11 post-fusion targets relative to the MTP representative alone. The preserved
whole-architecture artifact has 231 provenance-selected targets and 110 Loop IR fallbacks. All 341 targets
reconstruct and lower to at least one CUDA kernel on exact SM70.

## Unsupported exact path

The exact MTP source calls `roll_tensor`, exporting this live mutation sequence:

```text
aten.roll -> aten.select -> aten.fill_
```

The returned value aliases the mutated roll result. Alias-aware output liveness correctly retains the mutation, and
three focused trace regressions pass. The exact trace then stops at:

```text
ValueError: cannot map fallback aten.select as elementwise for output shape (1,)
```

Without the alias-aware liveness fix, the select/write was incorrectly pruned and CUDA lowering instead stopped at
`render: elementwise fn='roll' not supported`. A durable fix needs a semantic decomposition of the static
`roll/select/fill` sequence, or general `aten.roll` plus select/write support.

For diagnostic coverage only, the trace replaces the shift with equivalent slice/concatenate algebra. For the probe
input `[1,2,3,4,5,6,7,8]`, both paths produce `[2,3,4,5,6,7,8,0]` exactly. This replacement demonstrates that every
other inventoried path lowers; it does not satisfy the exact-path coverage gate.

## Partial tuning

The final-code comparison used all eight GPUs and matched cold starting state, compiler revision, O1 compile lane,
seed 0, patience 1, and a one-candidate budget per target. Both arms started without a tuning DB, online prior, or
cubin cache. The cold arm had no proposals; the hybrid arm had one explicit greedily realized proposal per target.
Pins are process-global, so hybrid proposal measurements ran serially while rotating over the eight GPU workers.
The full machine-readable comparison is in [`tuning_summary.json`](tuning_summary.json).

| O1 ranking result | Cold MCTS | Hybrid |
| --- | ---: | ---: |
| Working-golden targets visited | 341 / 341 | 341 / 341 |
| Targets with `ok` ranking | 237 | 300 |
| Ambiguous multi-kernel ranking | Not persisted as a winner | 39 |
| Target-level ranking watchdog failure | Not persisted as a winner | 2 |
| Unique DB performance rows | 582 | 463 |
| Successful DB rows | 579 | 457 |
| DB watchdog rows | 3 | 6 |
| Wall time | 312 s | 2,423 s |

There are 237 targets with a positive O1 latency in both arms. Hybrid is faster on 22, cold MCTS is faster on 215,
and 201 realize the same knobs. Across that paired set, the geometric mean hybrid/MCTS latency ratio is 1.115.
These are `-Xcicc -O1` ranking signals, not deployable performance.

The durable partial YAML chooses the lower positive O1 row where both arms succeeded: 215 rows come from cold MCTS
and 126 from hybrid, including hybrid-only and unsuccessful diagnostic rows. Its final status is 300 `ok`, 39
`ambiguous_multi_kernel`, and 2 `bench_fail`. Every one of its 341 targets reconstructs and lowers again on SM70;
see [`lowering_summary.json`](lowering_summary.json). Rankings remain in the working YAML intentionally. It has no
canonical measurement blocks.

The first attempt exposed `nvcc` outside the non-login PATH and was reset before the final cold run. The standard
CLI also rewrites the large working YAML once per target; a task-local batch-persistence wrapper retained identical
search and ranking logic while writing the final document once. Neither issue changed candidate budgets.

## Representative O3 accuracy and timing

Three representative finalists were pinned with their exact realized knobs and measured twice at nvcc default O3,
using five warmups and 20 iterations. All 10 pinned rows report `status=ok`, exact pin realization, and no integrity
flags. The complete sanitized measurements are in [`o3_summary.json`](o3_summary.json).

| Representative target | Eager PyTorch, two runs | Cold-MCTS pin, two runs | Hybrid pin, two runs |
| --- | ---: | ---: | ---: |
| Largest hybrid O1 improvement, `k_linear_7c6322` | 44.03 / 45.06 µs | 508.42 / 535.04 µs | 4.21 / 4.43 µs |
| Hybrid O1 regression, reshape/slice/reduce | 109.07 / 106.83 µs | 9.01 / 9.06 µs | 19.54 / 19.48 µs |
| Same-pin Loop control | Reference unavailable | 1.741 / 1.743 µs | Same realized pin |

The two Torch-runnable representatives passed the live eager comparison without a wrong-answer flag. The Loop
fallback control has no runnable eager reconstruction, so it establishes only repeated O3 compile, pin realization,
and execution. Full-model generation accuracy was not run because the exact checkpoint failed the serving fit gate.
Replacement-level accuracy remains: KDA passes FP16 tolerance, and rotary and token shift are exact.

## Promotion decision

Coverage is partial, so no file was added under `emmy/compiler/pipeline/search/goldens/`. Emmy is ineligible on this
platform: the first serving gate is memory fit, and the independent compiler gate fails at the exact MTP token-roll
seam. The partial YAML's O1 rankings and representative O3 rows must not be presented as complete deploy evidence.

After the exact seam is implemented, retrace this immutable revision, require every target to lower on SM70, then
repeat complete tuning. Promote only after every retained target has explicit realized knobs, repeated deployable
O3 accuracy, positive Emmy/reference timings, and a measured correct greedy fallback for every search miss.
