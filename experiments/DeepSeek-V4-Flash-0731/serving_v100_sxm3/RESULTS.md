# DeepSeek V4 Flash 0731 serving qualification on V100 SXM3

The exact checkpoint revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` served successfully on sixteen V100
SXM3 32 GB GPUs with TP8 and PP2. The tested 1Cat revision was
`d76126608155c334df7c2fb9b75096f879624859`; its focused CLI metadata and pipeline-parallel profiling fixes are in
[cloudrift-ai/1Cat-vLLM pull request 2](https://github.com/cloudrift-ai/1Cat-vLLM/pull/2).

## Qualification result

The normal `vllm serve` entrypoint reached HTTP 200 health after loading all 48 shards in 19.83 seconds and completing
engine profiling, KV-cache creation, and warmup in 84.05 seconds. Route-log audit confirmed all intended Volta paths:

- SM70 FP16 Triton sparse MLA with packed FP8 KV, `fp8_ds_mla`, and FP8 Lightning Indexer cache;
- SM70 TurboMind FP8 W8A16 dense, grouped-BMM, and gated-SiLU single-layout paths; and
- SM70 TurboMind MXFP4 MoE for all 256 local experts.

Deterministic probes passed for capital-of-France (`Paris`), small arithmetic (`4`), terse 17-by-19 arithmetic
(`323`), exact-response generation (`OK`), and structured tool calling (`multiply` with integer arguments 17 and
19). One alternative direct arithmetic prompt returned the correct `323` and then repeated malformed reasoning
markup until the 32-token limit; the result was byte-identical on repeat. This qualification therefore establishes
coherent deterministic answers and structured tool calls, with a response-formatting caveat for that prompt form.

Three serialized warm `OK` probes returned HTTP 200 and two completion tokens each:

| Probe | End-to-end latency |
| --- | ---: |
| 1 | 0.879345 s |
| 2 | 1.160202 s |
| 3 | 1.152433 s |
| Mean | 1.063993 s |

This is a bounded health and determinism benchmark, not a throughput claim. The experiment recipe provides the
closest reproducible Emmy random-input workload with the same concurrency and token counts; the live qualification
used a fixed `OK` request. Consolidated machine-readable evidence is in [qualification.json](qualification.json).
Full server logs and response bodies remain on the VM under `/home/riftuser/onecat-dsv4-0731/artifacts`.

## Remaining caveats

The first request JIT-compiled 16 Triton kernels that startup warmup did not cover, including slot mapping, prefill
metadata, qnorm/RoPE, FP8 KV insert and gather, sparse gathered and paged attention, and SWA index kernels. A release
image should extend its warm workload before claiming zero-JIT startup. The tested image remains locally tagged on
the VM because Docker Hub rejected the requested push attempt with `insufficient_scope`.

The task container was removed after the final probe. The teardown audit found zero task containers and zero GPU
compute processes; the exact checkpoint and image were intentionally retained. Compiler tracing and tuning are
reported independently in the [compiler experiment](../compiler_v100_sxm3/RESULTS.md).
