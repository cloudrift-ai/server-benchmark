# Laguna S 2.1 EXL3 5.01 bpw on 8× V100 SXM2 16 GB

Status: serving-qualified with the Emmy/1Cat engine and highest-precision official checkpoint pinned by the recipe.

## Qualified deployment

| Item | Value |
| --- | --- |
| Model | `turboderp/Laguna-S-2.1-exl3` |
| Model revision | `3469659b2f9a1656805250880c6ea9760f9626ed` |
| Checkpoint precision | official 5.00 bpw branch; 5.01 average bpw in `config.json` |
| Safetensor size | 74,445,864,293 bytes across 10 files |
| Hardware | 8× NVIDIA Tesla V100 SXM2 16 GB, compute capability 7.0 |
| Driver / CUDA | 580.159.03 / 13.0 driver, CUDA 12.8 runtime |
| Engine | Emmy with CloudRift 1Cat/vLLM `f0bd304b9d` |
| Emmy revision | `f81bbe18c48fee147a25adcdd46a3ff94dde8760` |
| Image | `emmy-round2-laguna5-v100:f81bbe18` (local, not published) |
| Image ID | `sha256:6d3555d81642169205388cdb65570f1665bd2294d2fab8e3bf99b1d81647117d` |
| Serving shape | TP1, PP8, context 262,144, concurrency 1, standard math, 16-token prefill chunks |
| KV capacity | 272,400 tokens at `gpu_memory_utilization=0.84` |

Pipeline stages own six transformer layers each, covering intervals 0, 6, 12, 18, 24, 30, 36, and 42. Each rank
loads only its stage-local attention sinks and layer weights; only the last rank owns the coded output head. Pipeline
hidden-state transport remains float32, while norms, QKV, paged attention, and routed intermediates remain float16.

This is the highest-published-precision checkpoint for the model. The complete snapshot is 74,511,639,693 bytes and
fits across the eight 16 GB cards without decoding the EXL3 trunk or output head. The selected image keeps the trunk
on the qualified generic compressed path and keeps the 6-bit output head coded.

## Best recipe performance

Measured 2026-08-12 with three steady repeats. Each repeat excluded four warm-up requests, then measured four random
64-token prompts requesting 16 output tokens at concurrency 1 with greedy decoding and EOS ignored. Values are the
mean across the three repeats; `±` is the sample standard deviation.

| Metric | Three-repeat mean ± standard deviation |
| --- | ---: |
| Successful / failed requests | 12 / 0 |
| Benchmark duration | 131.66 ± 0.12 s |
| Request throughput | 0.03 ± 0.00 requests/s |
| Output throughput | 0.49 ± 0.00 tokens/s |
| Total token throughput | 2.43 ± 0.00 tokens/s |
| Mean TTFT | 18,243.44 ± 29.95 ms |
| Mean TPOT / ITL | 978.05 ± 0.05 ms |

## Context and accuracy

An exact 262,143-token prompt plus one output token completed with HTTP 200 in 11,842.918 seconds,
reported exact 262,144 total-token usage, and incurred zero preemptions, allocator retries, or OOMs. The tightest
rank retained 651 MiB of measured physical headroom during the material request. An exact 262,144-token prompt plus
one output token was rejected with HTTP 400. Immediate post-boundary arithmetic, reasoning, and streaming/non-streaming
tool probes remained healthy.

The exact arithmetic oracle returns `42`. Its first-token log probability is `-0.2779814005`, and its top-10 token
order begins `4`, `The`, `I`, `1`, `To`, `3`, `Four`, `7`, `Fort`, and `\\`. The reasoning probe computes
`9 + 7 - 4 = 12`; the Poolside parser returns structured `get_weather(city="Paris")` calls in both response modes.

`EMMY_FAST_MATH` is explicitly disabled. It changed the exact checkpoint top-10 ordering without a useful serving
gain, so standard math is the selected quality lane.

## Compiler qualification

The tuning pass covered 83 current compiler targets and produced 161 independently replayed realizations. Its
full-program offer audit had 136 matches with zero gaps, drift, fall-through, or compile failures, but the resulting
serving pack changed the exact first-token distribution and achieved only 0.51 output tok/s. Those schedules therefore
failed the model-level accuracy and performance gates and are not shipped. The selected image contains the ordinary
standard-math execution plans that restore the reference oracle; no model-specific V100 golden is retained.

The final local image embeds the model, 91 Emmy cubins, 232 execution-plan files, 96 Triton files, and nine CuPy
cubins. A fresh offline container booted with no host bind mounts and a read-only root filesystem. All eight pipeline
ranks loaded their execution plans, and the cubin, plan, and Triton manifests were byte-identical before and after the
capability suite.

## Reproduce

The exact local image tag above must exist on the target host. The retained experiment performs three measured
repeats, each after four excluded warm-up requests:

```bash
emmy bench experiments/Laguna-S-2.1-exl3-5bpw/serving_v100_sxm2 --ssh <user@host>
```

The command writes ignored local output. Do not use `--commit-results`; this file is the only retained benchmark
report.

## Limits

- The serving image is retained only on the qualified host and was not published.
- Concurrency above one is not qualified. PP rank 0, rather than the coded-head rank, sets the context ceiling.
- Sixteen-token chunked prefill makes material 262K requests slow; the context gate measures fit and correctness, not
  interactive long-prompt latency.
- The generic compressed trunk remains selected. The experimental native block-GEMV body changed exact logits and
  reduced end-to-end speed, so it is structurally disabled; only the performance-neutral coded head is retained.
