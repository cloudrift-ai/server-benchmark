# DeepSeek V4 Flash 0731 on 16× V100 SXM3 32 GB

Status: serving-qualified with the 1Cat/vLLM engine pinned by the recipe. A bounded live Emmy integration is also
qualified for the final decode RMSNorm; full `EmmyGenModel` serving remains ineligible because the DeepSeek V4
compressor, hyper-connection state, quantized checkpoint path, and TP trunk lack executable serving-twin coverage.

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

## Exact-boundary compiler follow-up

After rebasing onto `650e196d`, the follow-up expanded the exact runtime-boundary compiler inventory instead of
treating the fixed-S512 architecture golden as serving coverage. The new traces preserve the deployed FP32 mHC
parameters, strided Q/KV views, interleaved inverse-RoPE indexing, and compact expert bytes. Every candidate passed its
component numerical gate on a V100, but none beat the corresponding 1Cat hot path, so the maintained recipe does not
enable them.

| Exact boundary | Emmy result | Matched 1Cat result | Decision |
| --- | ---: | ---: | --- |
| Fused Q/KV RMSNorm, `M=1/2/4/8/128` | 6.82–7.62 µs | 1.65–3.07 µs | Retain compiler coverage; 2.46–4.48× slower |
| Inverse RoPE, `M=1/2/4/8/128` | 1.67–4.35 µs | 1.50–3.43 µs | Retain compiler coverage; 9–27% slower |
| C4 compressor projection, `N=2048`, measured `M=1/2/4` | 21.98–40.59 µs | 20.79–29.70 µs | Retain; 5–37% slower |
| C128 compressor projection, `N=1024`, measured `M=1/4/8` | 12.51–29.27 µs | 11.93–19.61 µs | Retain; 5–49% slower |
| Full fused mHC transition, `M=1`, mixed per-node schedules | 21.254 µs | 12.831 µs | Retain; 1.66× slower |
| Routed compact MXFP4 W13/W2, six experts with 48 rows | 333.824 / 236.288 µs | 125.261 / 43.735 µs | Retain; 2.67× / 5.40× slower |

The mHC result uses eight launches. A generic bounded `FixedSinkhornOp` collapses the fixed 20-round 4×4 normalization
from 39 launches to one, while per-node evidence selects serial work for the short reductions and `t512/coop` for the
16K and 4K reductions. Applying one global `t512` pin instead takes 107.520 µs, which confirms that schedule evidence
must remain node-specific. The compact MXFP4 loader spelling consumes the checkpoint's packed E2M1 bytes and UE8M0
scales directly; format-specific behavior disappears into generic integer, bitcast, gather, and contraction algebra
at graph birth. The exact checkpoint's 9.26 billion expert scale bytes are all codes 118–126, inside the validated
FP16-exact 113–142 interval.

This work also fixed three model-agnostic compiler defects exposed by the live boundaries: non-unit Torch slice steps
were previously dropped, scoped tile paths could not replay a literal axis named `a0`, and typed 64-bit Loop inputs
used a host ABI spelling that cppyy could not bind on Darwin. These fixes have focused numerical and wire-compatibility
tests. The follow-up does not claim that stateful HCA/CSA cache mutation, sparse attention, expert routing, TP/PP
collectives, or the complete model execute in Emmy.

The final broad-adapter build gate realized and strictly replayed all 114 declared external programs on the supplied
V100 host, producing 114 execution-plan packs and 215 cubins. The inventory adds rank-specific TP-local embedding and
compact LM-head top-1 programs around the original collectives, plus the pure C4 indexer-Q RoPE and weight-scaling
transform. A direct `M=17` V100 check loaded only the persisted packs and matched embedding/rank selection exactly;
local logits and indexer-Q matched their FP32 accumulation references within their declared tolerances. Runtime
compilation is not reachable from these adapters: a missing or damaged pack keeps the corresponding 1Cat operation.

The retained TurboMind FP8 carrier layout was also reconstructed and checked byte-for-byte at graph birth. Emmy used
the caller-owned `uint8` weight and FP16 scale carriers without a decoded weight or scratch buffer and matched the
1Cat result, but measured 4–14× slower for the tested fused-QKV and shared gate/up shapes. The runtime hook and its 54
candidate profiles were therefore removed. Routed-expert prototypes were likewise excluded after failing the
whole-model latency budget. The maintained recipe still does not enable `EMMY_ONECAT_DEEPSEEK_V4`; the 114-profile
result is component and offline-release evidence, not broad endpoint or full-model serving qualification.

## Reproduce

```bash
emmy bench experiments/DeepSeek-V4-Flash-0731/serving_v100_sxm3 --ssh riftuser@185.165.50.61
```

The experiment runs four client repeats. The first warms the complete unique prompt set after deployment; use repeats
two through four to reproduce the reported steady result. Use `$run-experiment` to retain the latest raw results,
system-only experiment records, and factual artifact index.
