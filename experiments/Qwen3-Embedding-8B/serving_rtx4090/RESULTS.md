# Qwen/Qwen3-Embedding-8B — RTX 4090 Verification Results

**Date:** 2026-08-16
**Repository revision:** `a3d9e021`
**Model revision:** `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` (Hugging Face commit)
**Target GPU:** 1× NVIDIA GeForce RTX 4090 (sm_89, ~24 GB)
**Driver:** 580.65.06 (Ubuntu 24.04)
**CUDA:** 12.9.86 (nvcc)
**vLLM image:** `vllm/vllm-openai:v0.22.1`
**SSH host:** riftuser@211.21.50.85:57001

## Protocol

Verification of the maintained Qwen3-Embedding-8B recipe on a single RTX 4090. The experiment recipe defines a
zipped matrix of 2 images × 2 input lengths (4 combinations), but the Emmy compiled-kernel image
(`cloudriftai/vllm-emmy:0.22.1-90c70d5e`) is not yet published to Docker Hub for this target — pulling it fails
with `denied: requested access to the resource is denied`.

Two rows (stock vLLM lane, input_len=32 and 512) deployed and benchmarked successfully. The two Emmy lane rows
deployed but failed at image pull.

## Benchmark results — stock vLLM (pooling runner)

| Regime | Input len | Prompts | Concurrency | Throughput (req/s) | Token throughput (tok/s) | Mean E2EL (ms) | P99 E2EL (ms) |
|---|---|---|---|---|---|---|---|
| Short query | 32 | 256 | 32 | 87.98 | 2,815.28 | 338.24 | 636.03 |
| Document | 512 | 256 | 32 | 20.08 | 10,281.71 | 1,500.78 | 1,801.83 |

All 512 requests across both regimes succeeded (0 failures). Model load and warmup time was ~66 s start-up for the
first deployment (model download ~132 s on fresh pull, ~4 s on cache hit for the second row).

## Emmy lane

**Status:** Failed — image `cloudriftai/vllm-emmy:0.22.1-90c70d5e` not available on Docker Hub (pull denied).
Publication was not authorized for this run. Both input_len=32 and input_len=512 Emmy lane rows failed at image
pull.

## Compiler qualification

The Qwen3-Embedding-8B model traces successfully as a Qwen3 causal-LM architecture. A whole-model trace of layer 0
emitted 6 distinct kernel targets, all with fp16 dtype and the standard `FAST_MATH: false` pin:

1. `k_linear_mean_reduce_2622cc` — input layernorm + QKV projection
2. `k_sdpa_mean_linear_reduce_ae9858` — RoPE + SDPA + attention output
3. `k_sdpa_linear_reduce_7764a1` — attention output projection
4. `k_linear_reduce_3b6443` — attention residual add
5. `k_linear_mean_reduce_b03b1d` — FFN (gate + silu + up + down)
6. `k_linear_reduce_8eb37b` — FFN residual add

Full O3 tuning and canonical golden promotion were not completed in this verification run.

## Emmy eligibility

**Ineligible** on this run. The first failed gate is gate 4 (compiler coverage): although the model has a complete
trace with 6 targets, full O3 verification measurements and a canonical golden have not been produced, and the
prebuilt serving image is not yet published.

## Experiment artifacts

- **Archive:** `results_rtx4090x1.tar.gz` (Git LFS-tracked)
- **Run ID:** `20260816T195056Z`
- **2 succeeded:** `rtx4090x1_ril32_ear-p_ivllm-vllm-oai-v0.22.1`, `rtx4090x1_ril512_ear-p_ivllm-vllm-oai-v0.22.1`
- **2 failed:** Emmy lanes — `denied` image pull for `cloudriftai/vllm-emmy:0.22.1-90c70d5e`
