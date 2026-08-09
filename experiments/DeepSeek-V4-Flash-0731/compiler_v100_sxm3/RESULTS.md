# DeepSeek-V4-Flash-0731 compiler qualification on 16× V100 SXM3 32 GB

Status: reduced-scope compiler qualification. The exact checkpoint cannot be served on the requested Volta system.
The committed working golden covers an FP16 architecture twin of layer 0; it is not whole-model or serving evidence.

## Scope

- Date: 2026-08-09
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Immutable model revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- Emmy base revision: `4438c84a2027b87091fefd43f5cbbd5ea2bb4a5f`, plus the trace and lowering fixes in this PR
- Hardware: 16× `Tesla V100-SXM3-32GB`, compute capability 7.0, driver 580.173.02
- Trace profile: layer 0, static sequence length 512, FP16 architecture twin, target `sm_70`

The [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) and exact configuration describe a
304.18B-parameter, 43-layer model with a 1,048,576-token native context. The checkpoint combines BF16 residual
tensors, block-scaled FP8 attention and shared weights with E8M0 scales, and packed FP4 experts stored in I8 carriers.

## Serving and checkpoint gates

The live qualification used all 16 GPUs and the official
`vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404` image at digest
`sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8`. vLLM resolved the architecture and
checkpoint format, then stopped before weight loading:

```text
The quantization method deepseek_v4_fp8 is not supported for the current GPU.
Minimum capability: 75. Current capability: 70.
```

Emmy also fails checkpoint-representation eligibility. Its serving loader does not preserve the checkpoint's E8M0
scale encoding or packed-I8 FP4 experts. Dequantizing 304.18B parameters to FP16 would require more than 608 GB for
weights alone, exceeding the supplied 512 GB aggregate GPU memory before runtime state and KV cache.

No serving recipe, serving experiment, canonical golden, checkpoint accuracy claim, or Docker image is therefore
published for this model and hardware.

## Trace coverage

Layer 0 traced successfully: 547 FX nodes became 928 stable Torch IR nodes and 77 distinct kernel targets. The
self-contained result is [layer0_tuned_working_golden.yaml](layer0_tuned_working_golden.yaml).

A representative compressed-attention layer 10 advanced through the constructor, functional copy, no-grad export,
and ternary masked-fill fixes in this PR. It then reached the architecture-specific sparse indexer and stopped at the
remaining multi-output boundary:

```text
NotImplementedError: no tracer mapping for multi-output op aten.topk (2 outputs)
```

The working golden consequently covers only layer 0. Treating it as model-wide coverage would omit the compressed
attention path used by the checkpoint.

## Equal-budget tuning

Both final arms used all 16 GPUs, seed 731, two candidates per target, patience 2, separate empty databases and online
priors, separate cubin caches, and `-Xcicc -O1` ranking compiles.

| Arm | Successful rows | `bench_fail` rows | O1 search winners | Prior benches | Post-fit Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| MCTS-only | 90 | 107 | 31 | 325 | +0.71 |
| Model proposals plus MCTS | 128 | 75 | 50 | 334 | +0.57 |

The hybrid arm reserved six model-proposal rows. Four were `bench_fail`, two were `pin_unmatched`, and none won. Its
50 winners came from search. O1 measurements rank candidates only and are not deployable performance measurements.
The committed working golden is the MCTS-only artifact, retaining all 77 targets and its 31 ranking winners.

## Compiler fixes found by the run

Three independently reproduced lowering defects now have focused regressions:

- Scalar per-cell projection copied an output-sweep coordinate as per-cell SSA and emitted undefined component axes.
- A zero-axis projection root reassembled its boundary store outside the output sweep.
- An inapplicable graph-wide warp pin left a mixed-dtype linear term unmapped; materialization then omitted its free
  axes. The scalar fallback now binds both axes, and the exact source compiles with
  `nvcc --cubin -arch=sm_70 --use_fast_math -Xcicc -O1`.

The original arm measurements were preserved. Fix validation happened after each fair arm completed.

## O3 accuracy and latency

Finalists were rebuilt with production O3 flags using a fresh cubin cache, 10 warmup iterations, 100 measured
iterations, exact realized-knob checks, and the eager accuracy gate.

| Finalist | Eager | Greedy Emmy | Pinned finalist | Decision |
| --- | ---: | ---: | ---: | --- |
| Fused div-reduce, run 1 | 56.073844 µs | 1.361582 µs | 1.378462 µs | Accuracy pass; retain evidence |
| Fused div-reduce, run 2 | 56.326978 µs | 1.362930 µs | 1.377694 µs | Accuracy pass; retain evidence |
| Attention linear | 145.408005 µs | 861.184001 µs | 1451.007962 µs | Accuracy pass; reject schedule |
| RMS mean | unavailable | 22.072889 µs | 151.259422 µs | Reject; random input is outside eager domain |

The fused div-reduce median is 56.200411 µs eager, 1.362256 µs greedy Emmy, and 1.378078 µs pinned. Both pinned runs
reported `status: ok`, exact realized knobs, empty integrity flags, and no accuracy, NaN, or wrong-answer warning.

No row is promoted to the canonical V100 golden: the exact checkpoint representation and compressed layers remain
ineligible, and one representative kernel's O3 evidence cannot establish deployable model coverage.

## Limitations

- The working golden is architecture-derived FP16 layer-0 evidence at sequence length 512.
- Compressed-attention layers remain blocked at multi-output `aten.topk` tracing.
- The checkpoint cannot run on SM70 through the tested mainstream or Emmy serving paths.
- Whole-model accuracy and serving performance were not measured and are not implied by the kernel results above.
