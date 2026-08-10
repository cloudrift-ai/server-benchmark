# Laguna S 2.1 FP8 on 8x V100 SXM3 32GB

Measured on 2026-08-09 with model revision `9e0b8ba630080b0e6f20a7b43294a9f2232fd247`, eight 32 GB V100 SXM3
GPUs, driver 580.159.03, and the 1Cat image built from commit `91aca502d2bb1f05d9208ab2edec9fae53ff0d0b`.
The local image manifest digest was
`sha256:8405bb60d24610417d0d6da278a753e2c968bfd1e0d7ff7f79cd6601a038b2be` and its size was
10,801,482,122 bytes.

Docker Hub publication was attempted after the serving gates passed. Authentication succeeded, but Docker Hub
rejected `cloudriftai/1cat-vllm-sm70:1.2.2-cloudrift` with `insufficient_scope`; the available credential cannot
create or push that namespace repository. The tested image remains on the supplied host, and the recipe's tag is
reserved for publication after a `cloudriftai` repository grant. No credential remains on the host.

## Recommended configuration

The checkpoint requires TP8 on V100. The native 1Cat SM70 block-FP8 MoE route loaded all 49 checkpoint shards and
selected `FLASH_ATTN_V100`, but failed during FP8 expert post-processing with a CUDA invalid argument followed by an
illegal memory access. The recipe therefore uses the engine's documented conservative fallback: routed experts are
dequantized to FP16 after loading and executed by the unquantized Triton MoE path.

At 4096 context and one-request concurrency, the fallback loaded successfully in 29.2 seconds and completed engine
warmup in 13.5 seconds. Model loading used 27.9 GiB per rank; the healthy service held about 29.1 GiB per GPU and
reported 0.46 GiB of KV-cache memory. The engine exposed a 37,792-token KV pool, but the recipe intentionally keeps
the qualified request limit at 4096 because the native FP8 route is not usable and memory headroom is narrow.

## Accuracy and capability checks

Two deterministic chat checks passed: the model answered the capital-of-France prompt with `Paris` and computed
17 multiplied by 19 as `323`. A tool-use prompt emitted the correct Poolside markup for
`get_weather(city="Paris")`, but this 1Cat build left the markup in the response `content` rather than populating the
OpenAI `tool_calls` field. The reasoning parser also logged that automatic reasoning-token initialization failed, so
structured tool calls and parsed reasoning are not qualified by this result.

The [compiler golden](../../emmy/compiler/pipeline/search/goldens/v100_sm70_laguna_s_2_1_fp8.yaml)
covers all 48 decoder layers plus token embedding, final normalization, and the output head. It is
architecture-derived rather than an Emmy serving path: the exact checkpoint's per-expert FP8 tensors do not map to
Emmy's packed traced expert inputs, the baked Emmy runner is single-GPU, and a full-checkpoint layer parity run
exceeded 622 GiB of host RAM before reaching GPU execution. Serving accuracy is therefore established only for the
measured 1Cat fallback above; compiler scope and per-target O3 evidence are documented in the
[compiler experiment](../../experiments/Laguna-S-2.1-FP8/compiler_v100_sxm3/RESULTS.md).

## One-request benchmark

The reproducible workload used one 32-token random prompt, requested 16 output tokens, set temperature to zero, and
used concurrency one.

| Metric | Result |
| --- | ---: |
| Successful / failed requests | 1 / 0 |
| Benchmark duration | 3.03 s |
| Request throughput | 0.33 requests/s |
| Output throughput | 5.27 tokens/s |
| Total token throughput | 15.82 tokens/s |
| Mean time to first token | 949.39 ms |
| Mean time per output token | 138.84 ms |
| Mean inter-token latency | 138.84 ms |

Raw results are in
`experiments/Laguna-S-2.1-FP8/serving_v100_sxm3/2026-08-09_15-31-12_1614afd1/v100x8_vllm_benchmark.json`.
