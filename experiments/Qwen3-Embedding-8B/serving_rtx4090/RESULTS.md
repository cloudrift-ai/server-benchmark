# Qwen/Qwen3-Embedding-8B — RTX 4090 Verification Results

**Date:** 2026-08-17
**Repository revision:** `0afb4b06`
**Model revision:** `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` (Hugging Face commit)
**Target GPU:** 1× NVIDIA GeForce RTX 4090 (sm_89, ~24 GB)
**Driver:** 580.65.06 (Ubuntu 24.04)
**vLLM image:** `vllm/vllm-openai:v0.22.1` (commit `0decac0d`)

## Protocol

Verification of the maintained Qwen3-Embedding-8B recipe on a single RTX 4090. The experiment recipe defines a zipped
matrix of 2 images × 2 input lengths (4 combinations), but the Emmy compiled-kernel image
(`cloudriftai/vllm-emmy:0.22.1-90c70d5e`) is not yet published to Docker Hub for this target — pulling it fails with
`denied: requested access to the resource is denied`.

Two rows (stock vLLM lane, input_len=32 and 512) deployed and benchmarked successfully. The two Emmy lane rows failed
at image pull. Compiler qualification traced 6 kernel targets on layer 0; full O3 tuning did not complete due to remote
nvcc availability and memory constraints on the supplied host.

## Benchmark results — stock vLLM (pooling runner)

| Regime | Input len | Prompts | Concurrency | Throughput (req/s) | Token throughput (tok/s) | Mean E2EL (ms) | P99 E2EL (ms) |
|---|---|---|---|---|---|---|---|
| Short query | 32 | 256 | 32 | 99.62 | 3,187.68 | 297.85 | 670.16 |
| Document | 512 | 256 | 32 | 20.74 | 10,620.61 | 1,451.46 | 1,701.63 |

All 512 requests across both regimes succeeded (0 failures).

## Previous run comparison (2026-08-16, rev `a3d9e021`)

| Regime | Input len | req/s (prev) | req/s (new) | tok/s (prev) | tok/s (new) |
|---|---|---|---|---|---|
| Short query | 32 | 87.98 | 99.62 | 2,815.28 | 3,187.68 |
| Document | 512 | 20.08 | 20.74 | 10,281.71 | 10,620.61 |

The new run shows a modest improvement (~12% higher short-query throughput, ~3% higher document throughput), likely
due to model weight caching and engine warmup state.

## Emmy lane

**Status:** Failed — image `cloudriftai/vllm-emmy:0.22.1-90c70d5e` not available on Docker Hub (pull denied).
Publication was not authorized for this run. Both input_len=32 and input_len=512 Emmy lane rows failed at image
pull.

## Compiler qualification

A layer-0 trace of the Qwen3-Embedding-8B model emitted 6 distinct kernel targets:

1. `k_linear_mean_reduce_2622cc` — input layernorm + QKV projection
2. `k_sdpa_mean_linear_reduce_ae9858` — RoPE + SDPA + attention output
3. `k_sdpa_linear_reduce_7764a1` — attention output projection
4. `k_linear_reduce_3b6443` — attention residual add
5. `k_linear_mean_reduce_b03b1d` — FFN (gate + silu + up + down)
6. `k_linear_reduce_8eb37b` — FFN residual add

Full O3 tuning and canonical golden promotion were not completed. The MCTS search ran 4 of 6 targets before crashing
(memory allocation failure on target 5). The working golden remains at `experiments/` task scratch.

## Emmy eligibility

**Ineligible** — first failed gate: gate 4 (compiler coverage). The model has a layer-0 trace with 6 targets but full
O3 verification and a canonical golden were not produced. The Emmy serving image is also unavailable.

## Experiment artifacts

- **Archive:** `results_rtx4090x1.tar.gz` (Git LFS-tracked, `filter: lfs`)
- **Run ID:** `20260817T003524Z`
- **2 succeeded:** `rtx4090x1_ril32_ear-p_ivllm-vllm-oai-v0.22.1`, `rtx4090x1_ril512_ear-p_ivllm-vllm-oai-v0.22.1`
- **2 failed:** Emmy lanes — `denied` image pull for `cloudriftai/vllm-emmy:0.22.1-90c70d5e`
