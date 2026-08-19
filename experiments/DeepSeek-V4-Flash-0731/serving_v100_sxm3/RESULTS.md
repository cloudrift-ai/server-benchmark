# DeepSeek V4 Flash 0731 — serving experiment results

One representative short-context workload for the zero-JIT TP8×PP2 configuration, run per exact GPU platform. Each
platform section below describes the archive named in it and nothing else.

## NVIDIA Tesla V100 SXM3 32GB × 16 — `results_v100x16.tar.gz`

**Question.** Does the maintained TP8×PP2 recipe still deploy, serve, and perform on a 16× V100 SXM3 host, and how
stable is the result across repeats?

**Protocol.** `emmy bench experiments/DeepSeek-V4-Flash-0731/serving_v100_sxm3` against a pre-allocated 16× V100 SXM3
host over SSH. One matrix row (`v100x16`). Four client repeats, each preceded by 8 warm-up requests; each repeat sends
8 unique 1,024-token prompts at concurrency 8 and requests 64 output tokens with greedy decoding (`temperature: 0.0`,
`seed: 731`) and `ignore_eos`, so every request produces exactly 64 tokens. Repeat 1 primes the prompt set after
deployment; repeats 2–4 are the steady-state result. Spread is the population standard deviation across those three
repeats.

**Run.** Timestamp `2026-08-19T19:14:45Z`, run ID `20260819T191445Z`, repository revision `12bb850e` (clean tree),
row `v100x16` / `661253606d45`, status `succeeded`. Archive members:

```
2026-08-19_19-14-45/benchmark.log
2026-08-19_19-14-45/benchmark_v100_x_16.log
2026-08-19_19-14-45/v100x16_661253606d45.benchmark.log
2026-08-19_19-14-45/v100x16_661253606d45.experiment.yaml
2026-08-19_19-14-45/v100x16_661253606d45.server.log
```

**Machine and software.** Ubuntu 24.04.1, kernel 6.8.0-124-generic, 2× Intel Xeon Platinum 8168 (80 logical CPUs),
1.44 TB RAM, 16× Tesla V100-SXM3-32GB behind 12 NVSwitches, driver 580.159.03, CUDA 13.0. Engine image
`cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608`
(`sha256:276240257b224097876b5b6db8f0d32484dff6a6f168d6b03d6df188e5c65bc1`), vLLM
`1.2.3.dev87+gd76126608.d20260810`. Model revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, FP16 weights with
`deepseek_v4_fp8` quantization and an FP8 KV cache, TP8 × PP2, context 1,048,576, `gpu_memory_utilization` 0.90.

**Result.** The row succeeded. All 32 requests across the four repeats completed; 0 failed.

| Repeat | Duration (s) | req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (priming) | 23.72 | 0.34 | 21.59 | 367.00 | 8,266.85 | 243.89 |
| 2 | 16.59 | 0.48 | 30.86 | 524.56 | 3,876.69 | 201.79 |
| 3 | 16.47 | 0.49 | 31.09 | 528.57 | 3,821.34 | 200.67 |
| 4 | 16.52 | 0.48 | 30.99 | 526.78 | 3,817.99 | 201.62 |

Steady state over repeats 2–4 (24 requests, 0 failures):

| Metric | Mean ± population SD |
| --- | ---: |
| Benchmark duration | 16.5267 ± 0.0492 s |
| Request throughput | 0.4833 ± 0.0047 req/s |
| Output token throughput | 30.9800 ± 0.0942 tok/s |
| Total token throughput | 526.6367 ± 1.6402 tok/s |
| Mean TTFT | 3,838.6733 ± 26.9166 ms |
| Mean TPOT / ITL | 201.3600 ± 0.4928 ms |

**Repeat variation.** Repeats 2–4 are very tight: duration varies by 0.3%, output throughput by 0.3%, and mean TPOT by
0.2%. The priming repeat is a clear outlier (44% longer, TTFT 2.2× the steady mean) and is excluded, which the protocol
anticipates.

**Deployment timing.** `remote_provision` 7.04 s, `image_pull` 2.36 s and `model_download` 3.41 s (both cached from an
earlier deployment on this host), `weights_load` 24.32 s, `cuda_graph_capture` 7.00 s, `engine_warmup` 22.52 s,
`startup` 92.09 s, `model_load_and_warmup` 145.92 s, `smoke_test` 4.10 s, `benchmark` 242.21 s, `total` 405.04 s.

**Comparison with the 2026-08-11 qualification.** That run measured the same recipe and workload on a *different* 16×
V100 SXM3 host running driver 580.173.02. This is a directional comparison across two machines, not a controlled A/B:
only one variable was intended to change (the host), but driver version changed with it.

| Metric | 2026-08-11 | 2026-08-19 | Change |
| --- | ---: | ---: | ---: |
| Benchmark duration | 14.9793 s | 16.5267 s | +10.3% |
| Request throughput | 0.5341 req/s | 0.4833 req/s | −9.5% |
| Output token throughput | 34.1830 tok/s | 30.9800 tok/s | −9.4% |
| Total token throughput | 581.1111 tok/s | 526.6367 tok/s | −9.4% |
| Mean TTFT | 2,580.88 ms | 3,838.67 ms | +48.7% |
| Mean TPOT / ITL | 195.949 ms | 201.360 ms | +2.8% |

The gap is concentrated in prefill: decode cost (TPOT) is within 2.8%, while TTFT is roughly half again as large. Both
runs are internally stable, so the difference is a property of the two machines rather than measurement noise. The
harness does not isolate host from driver, so this experiment cannot attribute the prefill difference to either one.

**Zero-JIT claim.** The recipe describes this configuration as having a zero-JIT request cache. That does not hold on
this deployment: the server log records eight distinct Triton kernels JIT-compiling during inference —
`_build_c128a_topk_metadata_kernel`, `_build_prefill_chunk_metadata_kernel`, `_combine_topk_swa_indices_kernel`,
`_compute_prefill_metadata_kernel`, `_dequantize_and_gather_k_kernel`, `_sm70_qnorm_rope_kernel`,
`_sm70_sparse_gathered_kernel`, and `quantize_and_insert_k_kernel`. All of them compile once,
during the first repeat's warm-up phase, and none recurs; they inflate the priming repeat and leave repeats 2–4
unaffected.

**Conclusion.** The maintained recipe deploys and serves correctly on 16× V100 SXM3 with zero failed requests and
highly reproducible steady-state throughput. Short-context performance on this host is about 9% below the 2026-08-11
measurement, with the shortfall almost entirely in prefill latency.

**Limitations.** One workload shape only (1,024 in / 64 out, concurrency 8) on one matrix row — it justifies the
recommended configuration but says nothing about long-context or high-concurrency behaviour. `emmy bench` deployed a
fresh server for this run, so the numbers are cold-cache-then-warmed, not steady-state under sustained production load.
The cross-run comparison above changes two variables at once. No Emmy lane exists for this model, so there is no
compiler-vs-stock comparison to report.
