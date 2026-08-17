# Qwen/Qwen3-Embedding-8B — RTX 4090 Verification Report

**Date:** 2026-08-17
**Repository revision:** `0afb4b06`
**Model revision:** `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`
**Target GPU:** 1× NVIDIA GeForce RTX 4090 (sm_89, 24 GB)
**Driver:** 580.65.06
**vLLM image:** `vllm/vllm-openai:v0.22.1` (commit `0decac0d`)

## Summary

Verification of the maintained Qwen3-Embedding-8B recipe on a single RTX 4090. The stock vLLM pooling lane deployed,
passed health checks, and benchmarked at both short-query (input_len=32) and document (input_len=512) regimes. The Emmy
compiled-kernel lane could not be measured because the prebuilt serving image
(`cloudriftai/vllm-emmy:0.22.1-90c70d5e`) requires publication — publication was not authorized for this run. Compiler
qualification traced 6 kernel targets on layer 0; full O3 tuning did not complete.

## Benchmark results — stock vLLM (pooling runner)

| Regime | Input len | Prompts | Concurrency | Throughput (req/s) | Token throughput (tok/s) | Mean E2EL (ms) | P99 E2EL (ms) |
|---|---|---|---|---|---|---|---|
| Short query | 32 | 256 | 32 | 99.62 | 3,187.68 | 297.85 | 670.16 |
| Document | 512 | 256 | 32 | 20.74 | 10,620.61 | 1,451.46 | 1,701.63 |

All 512 requests across both regimes succeeded (0 failures).

## Emmy eligibility

**Ineligible** — compiler qualification traced 6 targets on layer 0 but O3 tuning and a canonical golden were not
completed on this run. The image lane is also blocked by the missing serving image publication.

## Recipe

**File:** `recipes/Qwen3-Embedding-8B/recipe.yaml`
**Lifecycle:** maintained (preserved)

The recipe defines two engine lanes within its matrix:

1. **Stock vLLM:** `vllm/vllm-openai:v0.22.1` with `--runner pooling` — **verified on this run**
2. **Emmy:** `cloudriftai/vllm-emmy:0.22.1-90c70d5e` with `--runner pooling --enforce-eager --hf-overrides`
   — pending image publication

## Serving configuration

- `tensor_parallel_size: 1`, `gpu_memory_utilization: 0.92`
- `context_length: 4096` (model native max: 40,960)
- `max_concurrent_requests: 32`
- Embedding task via vLLM pooling runner on the Qwen3 causal-LM base

## Reproduction

```bash
# Stock vLLM lane (verified)
emmy bench recipes/Qwen3-Embedding-8B --ssh riftuser@HOST:PORT

# Filter to 4090 only
emmy bench recipes/Qwen3-Embedding-8B --ssh riftuser@HOST:PORT --filter "deploy.gpu=*4090*"
```

## Limitations

- Emmy compiled-kernel lane unverified until serving image is published.
- Context length capped at 4096 (emmy's dynamic-dim maximum); the model supports 40k.
