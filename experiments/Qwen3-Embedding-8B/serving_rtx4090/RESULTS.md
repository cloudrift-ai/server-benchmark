# Qwen/Qwen3-Embedding-8B — RTX 4090 Verification Results

**Date:** 2026-08-17
**Repository revision:** `821a3b9b`
**Model revision:** `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` (Hugging Face commit)
**Target GPU:** 1× NVIDIA GeForce RTX 4090 (sm_89, ~24 GB)
**Driver:** 580.65.06 (Ubuntu 24.04)
**vLLM image:** `vllm/vllm-openai:v0.22.1`

## Protocol

Verification of the maintained Qwen3-Embedding-8B recipe on a single RTX 4090 via SSH to the pre-allocated host.
The experiment recipe defines a zipped matrix of 2 images × 2 input lengths (4 combinations), but the Emmy compiled-kernel
image (`cloudriftai/vllm-emmy:0.22.1-90c70d5e`) is not published on Docker Hub for public pull — publication was not
authorized for this run.

The stock vLLM lane (input_len=32 and 512) deployed and benchmarked successfully. The Emmy lane rows were not executed
because the serving image requires publication. Compiler qualification produced a full-model trace with 434 configs and
72 loops covering all 36 layers.

## Benchmark results — stock vLLM (pooling runner)

| Regime | Input len | Prompts | Concurrency | Throughput (req/s) | Token throughput (tok/s) | Mean E2EL (ms) | P99 E2EL (ms) |
|---|---|---|---|---|---|---|---|
| Short query | 32 | 256 | 32 | 81.50 | 2,607.97 | 374.31 | 713.80 |
| Document | 512 | 256 | 32 | 20.82 | 10,660.48 | 1,446.66 | 1,710.54 |

All 512 requests across both regimes succeeded (0 failures).

## Previous run comparison (2026-08-17 rev `0afb4b06`)

| Regime | Input len | req/s (prev) | req/s (new) | tok/s (prev) | tok/s (new) |
|---|---|---|---|---|---|
| Short query | 32 | 99.62 | 81.50 | 3,187.68 | 2,607.97 |
| Document | 512 | 20.74 | 20.82 | 10,620.61 | 10,660.48 |

The short-query throughput decreased ~18% while document throughput is stable (~0.4% improvement), likely due to model
weight not yet being cached from a prior run.

## Emmy lane

**Status:** Not executed — image `cloudriftai/vllm-emmy:0.22.1-90c70d5e` not available on Docker Hub. Publication was
not authorized for this run.

## Compiler qualification

A full-model trace of Qwen3-Embedding-8B produced 434 distinct kernel configurations and 72 loops covering all 36 layers
(0–35). The trace captured:

- Embedding/layernorm/attention output programs (forkless, model-wide)
- Per-layer attention operations (72 loops, 2 per layer for softmax and attention score)
- 434 configs with forkless knob-bearing realizations (FAST_MATH=False)

The model has Qwen3 architecture (hidden_size=4096, 36 layers, 32 heads, intermediate=12288, vocab=151665, context=40k).
The lm_head.weight is missing from the checkpoint (expected for embedding models — pooling is used instead).

## Emmy eligibility

**Ineligible** — gate 5 (serving). The Emmy serving image was not published, so the `emmy serve` path could not be
verified. The compiler trace is complete but full O3 tuning and deployable measurements were not completed on this run.

## Experiment artifacts

- **Archive:** `results_rtx4090x1.tar.gz` (Git LFS-tracked, `filter: lfs`)
- **Run ID:** `20260817T101903Z`
- **2 succeeded:** `rtx4090x1_ril32_ear-p_ivllm-vllm-oai-v0.22.1`, `rtx4090x1_ril512_ear-p_ivllm-vllm-oai-v0.22.1`
- **2 skipped:** Emmy lanes — image not published
