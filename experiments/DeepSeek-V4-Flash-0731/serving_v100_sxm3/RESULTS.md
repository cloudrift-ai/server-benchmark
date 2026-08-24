# DeepSeek V4 Flash 0731 — serving experiment results

One representative short-context workload for the zero-JIT TP8×PP2 configuration, run per exact GPU platform. Each
platform section below describes the archive named in it and nothing else.

## NVIDIA Tesla V100 SXM3 32GB × 16 — `results_v100x16.tar.gz`

**Question.** Does the maintained TP8×PP2 recipe still deploy, serve, and perform on 16× V100 SXM3, and how stable is
the result across repeats?

**Protocol.** `emmy bench experiments/DeepSeek-V4-Flash-0731/serving_v100_sxm3` against a pre-allocated 16× V100 SXM3
host over SSH. One matrix row (`v100x16`). Four client repeats, each preceded by 8 warm-up requests; each repeat sends
8 unique 1,024-token prompts at concurrency 8 and requests 64 output tokens with greedy decoding (`temperature: 0.0`,
`seed: 731`) and `ignore_eos`, so every request produces exactly 64 tokens. Repeat 1 primes the prompt set after
deployment; repeats 2–4 are the steady-state result. Spread is the population standard deviation across those three.

**Run.** Timestamp `2026-08-21T02:57:30Z`, run ID `20260821T025730Z`, repository revision `fd7b09041` (clean tree),
row `v100x16` / `661253606d45`, status `succeeded`. Archive members:

```
2026-08-21_02-57-30/benchmark.log
2026-08-21_02-57-30/benchmark_v100_x_16.log
2026-08-21_02-57-30/v100x16_661253606d45.benchmark.log
2026-08-21_02-57-30/v100x16_661253606d45.experiment.yaml
2026-08-21_02-57-30/v100x16_661253606d45.server.log
```

**Machine and software.** Ubuntu 24.04.1, kernel 6.8.0-124-generic, 2× Intel Xeon Platinum 8168 (80 logical CPUs),
1,338 GB RAM, 16× Tesla V100-SXM3-32GB behind 12 NVSwitches, driver 580.159.03. Engine image
`cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608`, vLLM `v1.2.3.dev87+gd76126608`. Model revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`, FP16 weights with `deepseek_v4_fp8` quantization and an FP8 KV cache,
TP8 × PP2, context 1,048,576, `gpu_memory_utilization` 0.90.

**Result.** The row succeeded. All 32 requests across the four repeats completed; 0 failed.

| Repeat | Duration (s) | req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (priming) | 23.86 | 0.34 | 21.46 | 364.83 | 8,326.03 | 245.23 |
| 2 | 16.67 | 0.48 | 30.72 | 522.23 | 3,322.63 | 210.94 |
| 3 | 16.52 | 0.48 | 31.00 | 526.98 | 3,242.14 | 210.25 |
| 4 | 16.70 | 0.48 | 30.66 | 521.30 | 3,830.34 | 204.17 |

Steady state over repeats 2–4 (24 requests, 0 failures):

| Metric | Mean ± population SD |
| --- | ---: |
| Benchmark duration | 16.6300 ± 0.0787 s |
| Request throughput | 0.4800 ± 0.0000 req/s |
| Output token throughput | 30.7933 ± 0.1482 tok/s |
| Total token throughput | 523.5033 ± 2.4875 tok/s |
| Mean TTFT | 3,465.04 ± 260.39 ms |
| Mean TPOT / ITL | 208.4533 ± 3.0418 ms |

**Repeat variation.** Throughput is tight across repeats 2–4 (duration 0.5%, output throughput 0.5%). TTFT is not:
the three steady repeats span 3,242–3,830 ms, a 17% spread, against 27 ms of spread in the 2026-08-19 run. The
priming repeat is a clear outlier and is excluded, which the protocol anticipates.

**Deployment timing.** `remote_provision` 6.95 s, `image_pull` 2.44 s and `model_download` 3.68 s (both cached on this
host), `weights_load` 27.00 s, `cuda_graph_capture` 6.00 s, `engine_warmup` 23.73 s, `startup` 89.14 s,
`model_load_and_warmup` 145.87 s, `smoke_test` 4.29 s, `benchmark` 241.09 s, `total` 404.32 s.

**Comparison with the 2026-08-19 run.** Same host, same recipe, same image; the repository revision moved from
`12bb850e` to `fd7b09041` (16 merged pull requests, including a recognition rebuild and two compiler-cache changes).

| Metric | 2026-08-19 (`12bb850e`) | 2026-08-21 (`fd7b09041`) | Change |
| --- | ---: | ---: | ---: |
| Output token throughput | 30.9800 ± 0.0942 tok/s | 30.7933 ± 0.1482 tok/s | −0.6% |
| Total token throughput | 526.6367 ± 1.6402 tok/s | 523.5033 ± 2.4875 tok/s | −0.6% |
| Mean TTFT | 3,838.67 ± 26.92 ms | 3,465.04 ± 260.39 ms | −9.7% |
| Mean TPOT / ITL | 201.360 ± 0.493 ms | 208.453 ± 3.042 ms | +3.5% |

Throughput is unchanged within noise. TTFT improved by about 10% but its spread grew tenfold, so the two runs' TTFT
distributions overlap and the shift is not established by this evidence. The engine image is byte-identical between
the runs, so nothing in the serving stack changed; these are host-level run-to-run differences.

**Zero-JIT claim.** Unchanged from the previous run: eight distinct Triton kernels still JIT-compile once during the
first repeat's warm-up and none recurs, so they cost the priming repeat only.

**Context.** The engine again allocated KV capacity for 4,244,903 tokens on PP0 and 4,281,497 on PP1.

**Conclusion.** The maintained recipe deploys and serves correctly on 16× V100 SXM3 at the current repository
revision, with zero failed requests and steady-state throughput indistinguishable from the previous run.

**Limitations.** One workload shape (1,024 in / 64 out, concurrency 8) on one matrix row. `emmy bench` deploys a fresh
server, so the numbers are cold-cache-then-warmed rather than sustained production load. The TTFT comparison above is
directional only. No Emmy lane exists for this model, so there is no compiler-versus-stock comparison.
