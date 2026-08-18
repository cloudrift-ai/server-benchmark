# DeepSeek V4 Flash 0731 on 16× V100 SXM3 32 GB

Status: serving-qualified with the 1Cat/vLLM engine pinned by the recipe. A local broad Emmy/1Cat image is also
qualified for every pure compiler-eligible serving cone. 1Cat retains scheduling, collectives, and stateful paged
sparse-attention/cache operations. The maintained published recipe remains on the stock image pending publication
approval for the derived image.

## Qualified deployment

| Item | Value |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Model revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Hardware | 16× Tesla V100-SXM3-32GB, compute capability 7.0 |
| Driver / CUDA | 580.173.02 / 13.0 |
| Engine | 1Cat/vLLM `d76126608155c334df7c2fb9b75096f879624859` |
| Image | `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` |
| Serving shape | TP8, PP2, context 1,048,576, concurrency 8, FP8 KV cache |

The recipe disables process-local SM70 MXFP4 small-shape timing selection. The timed selector chose different W13
expert GEMM configurations after fresh starts and changed greedy output. The fixed default dispatch returned the exact
same token and logits in six probes across three fresh TP8×PP2 processes, and again after the final full-context
restart. `EMMY_FAST_MATH` is not set because this is not an Emmy serving engine.

## Best recipe performance

Measured 2026-08-11 with the final 1,048,576-context recipe. The reported three steady repeats followed one
unreported post-restart priming repeat. Each repeat used eight unique 1,024-token prompts at concurrency 8 and
requested 64 output tokens with greedy decoding and ignored EOS. All 24 reported requests completed with exact token
counts. Spread is the population standard deviation across the three repeats.

| Metric | Three-repeat mean ± standard deviation |
| --- | ---: |
| Successful / failed requests | 24 / 0 |
| Benchmark duration | 14.9793 ± 0.1260 s |
| Request throughput | 0.5341 ± 0.0045 requests/s |
| Output throughput | 34.1830 ± 0.2886 tokens/s |
| Total token throughput | 581.1111 ± 4.9058 tokens/s |
| Mean TTFT | 2,580.88 ± 163.94 ms |
| Mean TPOT / ITL | 195.949 ± 4.113 ms |

## Context and accuracy

The engine allocated KV capacity for 4,184,221 tokens on PP0 and 4,220,292 tokens on PP1. An exact
1,048,575-token prompt plus one decode token completed with HTTP 200 in 909.724 seconds and reported 1,048,576 total
tokens. Peak physical allocation left 361 MiB on PP0 and 589 MiB on PP1; the run had zero preemptions, allocator
errors, or OOMs.

The exact arithmetic probe returned `323` with identical logits twice before the boundary request, twice immediately
after it, and twice after a fresh full-context restart. The same deployment passed factual completion, thinking-off
chat, separated reasoning, and structured tool-call probes. The tool parser returned `multiply(a=17, b=19)`.

## Compiler qualification

The [canonical V100 golden](golden/v100_sm70.yaml)
contains 279 exact Loop realizations across 13 programs. Every retained realization has positive deployable O3 and
reference timings. The current schema passed all 279 stored-record and offer checks. The in-model audit traced four
exact representative DeepSeek graphs with 945, 1,156, 1,087, and 1,156 Graph IR nodes. Its layer-0 Loop pass exposed
dependency growth that ran for more than 22 minutes 59 seconds before the existing normalized-work guard. The
model-agnostic builder and structural-bound fix now completes that exact graph in 3.825650 seconds (3.848140-second
repeat), a censored improvement greater than 360×. Full in-model serving-twin qualification remains incomplete.

## Bounded live Emmy qualification

The 2026-08-17 experiment derives a local image from the pinned 1Cat base and replaces only the final decode RMSNorm
with a guarded Emmy-compiled SM70 program. 1Cat still owns model loading, TP8×PP2, attention/cache state, routing, and
FP4/FP8 kernels. The adapter activated on all eight TP ranks of the last PP stage and survived CUDA graph capture.
Unsupported calls and build or first-use parity failures fall back to the original runtime implementation.

The deployed `WORK=t256, REDUCE=coop` realization measured 4.5692 ± 0.0021 µs over three fresh deployable-O3 runs,
1.578× faster than Emmy's 7.2084 µs greedy realization for the leaf. A matched same-host endpoint comparison was a
tie: stock produced 34.1841 ± 0.2505 tokens/s and 197.414 ± 1.514 ms TPOT; the bounded Emmy lane produced
34.1984 ± 0.1387 tokens/s and 197.376 ± 0.666 ms TPOT. Each lane completed 24/24 steady requests, and deterministic
chat content and token usage matched. A separate recipe-driven run completed 32/32 requests and reached 34.44
tokens/s in its final repeat.

The maintained recipe remains on the published stock image because the derived image is local. Registry publication
requires separate human approval. Raw evidence and the exact scope are in
[`emmy_rmsnorm_v100_sxm3`](../../experiments/DeepSeek-V4-Flash-0731/emmy_rmsnorm_v100_sxm3/RESULTS.md).

## Broad live Emmy qualification

The final local image was built from `434577de` over the pinned 1Cat base and enabled
`EMMY_ONECAT_DEEPSEEK_V4=1`. Emmy owns the pure GPU compute after loader graph birth: normalization and RoPE,
unquantized and retained-FP8 projections, all five mHC boundaries, local vocabulary/output work, indexer-Q, learned
and hash routing, and the retained-MXFP4 expert projection/activation/weighted-combine path. The loader validates and
exports the sole physical FP8 or MXFP4 carrier; no decoded copy crosses the runtime boundary.

The release-blocking manifest contains 188 exact programs:

| Family | Profiles |
| --- | ---: |
| Normalization and RoPE | 4 |
| Unquantized projections | 45 |
| Retained physical FP8 projections | 54 |
| mHC boundaries | 45 |
| Output leaves | 3 |
| Local vocabulary and compact top-1 | 17 |
| C4 indexer-Q | 1 |
| Learned and hash routing | 18 |
| Retained compact experts | 1 |

All 188 programs were realized and strictly reloaded on a V100 before server startup. The baked image contains 301
cubins. Fingerprint-identical compiler-denial wrappers observed zero external compiler invocations during strict
replay and the complete TP8×PP2 run; the runtime cache stayed byte-identical to the image. Startup reached health in
244 seconds with no adapter error. The vLLM monitor nevertheless reported nine first-use native Triton
specializations during inference, so the image does not yet pass the stricter zero-JIT release gate. Missing or
damaged packs, unsupported contracts, capture-cold calls, and pre-mutation parity failures retain the exact 1Cat
operation. A failure after KV-cache mutation is surfaced rather than replayed.

The final continuous-batching workload ran one priming repeat followed by two steady repeats. Each repeat used eight
unique 1,024-token prompts at concurrency 8 and forced 64 tokens with temperature zero. All 24 requests completed.
The priming repeat reached 7.689 output tokens/s and 386.253 ms TPOT because every concrete runtime width executes its
first-use parity gate. The steady result was 20.714 ± 0.052 output tokens/s and 220.238 ± 0.026 ms TPOT. Compared with
the published stock baseline, broad Emmy coverage reduced output throughput by 39.4% and increased TPOT by 12.4%.
This is a coverage and correctness result, not a performance win.

The steady benchmark exercised irregular continuous-batch widths, including the previously failing M=3077 expert
case, with zero fallback and zero external compiler invocation. The exact arithmetic probe returned `323` twice with
identical logprobs, and the structured parser returned `multiply(a=17, b=19)`. A 1,048,575-token prompt plus one
decode token remained healthy at full GPU utilization with stable memory and no preemption, OOM, or adapter error for
3,600 seconds, but did not finish before the client timeout. Broad Emmy full-context qualification therefore remains
open; the successful 1,048,576-token evidence in the earlier section applies to the maintained stock recipe.

Representative component results explain the endpoint regression:

| Exact boundary | Emmy result | Matched 1Cat result | Outcome |
| --- | ---: | ---: | --- |
| Final RMSNorm | 4.569 µs | bounded endpoint tie | 1.578× over Emmy greedy |
| Fused Q/KV RMSNorm | 6.82–7.62 µs | 1.65–3.07 µs | 2.46–4.48× slower |
| Inverse RoPE | 1.67–4.35 µs | 1.50–3.43 µs | 9–27% slower |
| C4/C128 compressor projections | 12.51–40.59 µs | 11.93–29.70 µs | 5–49% slower |
| Fused mHC transition, M=1 | 21.254 µs | 12.831 µs | 1.66× slower |
| Retained FP8 projections | exact component parity | matched physical carriers | 4–14× slower in tested shapes |
| Route plus compact experts, M=1/M=8 | exact IDs and FP16 parity | recipe-default stock | 1.34–1.60× faster |

The mHC result uses eight launches. A generic `FixedSinkhornOp` collapses the fixed 20-round 4×4 normalization from
39 launches to one; node-specific evidence selects serial work for short reductions and `t512/coop` for the 16K and
4K reductions. A single global `t512` pin takes 107.520 µs and is rejected. The exact checkpoint's 9.26 billion expert
scale bytes are codes 118–126, inside the validated FP16-exact 113–142 interval.

This work also fixed model-agnostic compiler defects exposed by the live boundaries: non-unit Torch slice steps were
dropped, scoped tile paths could not replay a literal axis named `a0`, compound symbolic shapes lost their source
symbol, and typed 64-bit Loop inputs used a host ABI spelling cppyy could not bind on Darwin. The DeepSeek layer-0
Loop pass now completes in 3.826 seconds instead of running beyond 22 minutes 59 seconds.

1Cat intentionally retains scheduler and continuous-batching orchestration, CUDA-graph lifecycle, TP/PP and expert
collectives, HCA/CSA and sparse/recurrent attention state, paged FP8 KV-cache allocation and mutation, checkpoint
loading and tensor lifetime, and the API/sampling stack. This is full coverage of the eligible pure kernels, not an
`EmmyGenModel` replacement for stateful distributed serving.

The maintained recipe still points to the published stock image and does not enable the broad opt-in. The derived
image is local and unpublished; registry publication requires separate human approval.

## Reproduce

```bash
emmy bench experiments/DeepSeek-V4-Flash-0731/serving_v100_sxm3 --ssh riftuser@185.165.50.61
```

The experiment runs four client repeats. The first warms the complete unique prompt set after deployment; use repeats
two through four to reproduce the reported steady result. Use `$run-experiment` to retain the latest raw results,
system-only experiment records, and factual artifact index.
