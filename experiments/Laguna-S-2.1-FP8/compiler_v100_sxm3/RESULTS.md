# Laguna-S-2.1-FP8 compiler qualification on 8× V100 SXM3 32 GB

Status: tuned working golden produced for an FP16 architecture twin of layer 1. It is not a canonical golden or
checkpoint-accuracy claim. The independently qualified 1Cat serving fallback is documented in
[`recipes/Laguna-S-2.1-FP8/RESULTS.md`](../../../recipes/Laguna-S-2.1-FP8/RESULTS.md).

## Scope

- Date: 2026-08-09
- Model: `poolside/Laguna-S-2.1-FP8`
- Immutable revision: `9e0b8ba630080b0e6f20a7b43294a9f2232fd247`
- Emmy base revision: `4438c84a2027b87091fefd43f5cbbd5ea2bb4a5f`, plus this PR's trace and lowering fixes
- Hardware: 8× `Tesla V100-SXM3-32GB`, compute capability 7.0, driver 580.159.03
- Trace profile: layer 1, static sequence length 512, FP16 architecture twin, target `sm_70`

The exact checkpoint contains 117,561,977,600 parameters and occupies 131,264,796,160 bytes (122.25 GiB). Routed
expert matrices in sparse layers 1 through 43 use dynamic-activation, 128×128 block-scaled E4M3 FP8; attention,
shared experts, layer 0, and layers 44 through 47 remain BF16 or are excluded by the quantization configuration.

## Trace and eligibility

The first exact trace exposed a rotary-label mismatch: Laguna attention stores `layer_idx` but not `layer_type`,
while its rotary module is keyed by the labels in `config.layer_types`. The compiler now derives that label and
passes the one `(cos, sin)` tuple the block consumes. The fixed run traced 121 FX nodes into 171 Graph IR nodes and
emitted 20 distinct kernel targets in
[layer1_tuned_working_golden.yaml](layer1_tuned_working_golden.yaml).

Laguna also exposed two omitted block semantics in the serving split wrappers. The post-attention program now
retains the softplus per-head `g_proj` gate, and routed expert results receive `routed_scaling_factor=2.5` before the
unscaled shared expert is added. Focused dense and MoE parity tests cover both changes.

The working golden remains architecture-derived. The checkpoint stores separate per-expert projection and inverse
scale tensors, while the trace exposes packed expert inputs; Emmy does not preserve that checkpoint representation.
The baked Emmy serving runner is also single-GPU. Actual TP8 serving was therefore qualified independently through
the pinned 1Cat engine.

## Equal-budget tuning

Both arms began from the same 20-target inventory, empty arm-specific databases and online priors, empty cubin
caches, seed 0, six candidates per target, patience 4, all eight GPUs, and `-Xcicc -O1` ranking compiles.

| Arm | Successful rows | `bench_fail` rows | O1 search winners | Prior benches | Post-fit Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| MCTS-only | 187 | 26 | 10 | 399 | +0.88 |
| Model proposals plus MCTS | 182 | 22 | 11 | 381 | +0.89 |

The hybrid arm reserved three proposal rows. Every proposal was `pin_unmatched` and produced no CUDA kernel, so none
was eligible for promotion. The committed working golden is the clean MCTS-only artifact: all 20 inventory targets
plus ten O1 ranking winners. O1 costs rank search candidates only; they are not deployable performance evidence.

## O3 verification

The one extra hybrid search finalist, `k_linear_29ca57` at `WORK=t16x8,TILE=f2x4`, took 33.34 and 34.79 ms versus
0.387–0.390 ms eager and 13.47–13.48 ms for greedy Emmy. The alternate `TILE=f1x8` took 23.26–23.28 ms. The sliced
reference also exceeded the eager accuracy tolerance twice, so both schedules were rejected.

The remaining MCTS changes were rebuilt twice at O3 after the lowering fixes:

| Kernel | MCTS winner, µs | Comparison, µs | Live eager, µs | Decision |
| --- | ---: | ---: | ---: | --- |
| Cast pointwise | `f2`: 12.573 / 12.567 | `f4`: 12.352 / 12.352 | unavailable | Reject; not best |
| Linear pointwise | empty: 1.721 / 1.487 | `f4`: 1.399 / 1.388 | unavailable | Reject; not best |
| Softplus pointwise | empty: 1.553 / 1.552 | `f4`: 1.536 / 1.535 | 26.867 / 26.864 | Reject; not best |
| Transpose | `f2`: 26.652 / 26.652 | `f4`: 31.282 / 31.232 | 4.960 / 2.341 | Reject; slower than eager |

No canonical V100 golden was changed: all proposed changes either failed to realize, lost the O3 comparison, were
slower than eager, or failed accuracy.

## Accuracy limit

An exact full-checkpoint layer-1 parity run did not reach GPU execution. Host materialization grew to 653,208,220
KiB RSS (93.1% of host RAM), leaving about 39 GiB and no swap; it was stopped before host OOM. The trace and working
golden are valid operation-inventory evidence, but they do not imply checkpoint or whole-model Emmy accuracy.

## Sources

- [Exact model card](https://huggingface.co/poolside/Laguna-S-2.1-FP8)
- [Exact configuration](https://huggingface.co/poolside/Laguna-S-2.1-FP8/blob/9e0b8ba630080b0e6f20a7b43294a9f2232fd247/config.json)
- [OpenMDW-1.1 license](https://huggingface.co/poolside/Laguna-S-2.1-FP8/blob/9e0b8ba630080b0e6f20a7b43294a9f2232fd247/LICENSE.md)
