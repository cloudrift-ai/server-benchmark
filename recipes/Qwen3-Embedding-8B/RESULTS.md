# Qwen/Qwen3-Embedding-8B — RTX 4090 Verification Report

**Date:** 2026-08-16
**Repository revision:** `1e9e92fc`
**Model revision:** `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`
**Target GPU:** 1× NVIDIA GeForce RTX 4090 (sm_89, 24 GB)
**Driver:** 580.65.06
**vLLM image:** `vllm/vllm-openai:v0.22.1` (commit `0decac0d`)

## Summary

Verification of the maintained Qwen3-Embedding-8B recipe on a single RTX 4090. The stock vLLM pooling lane
deployed, passed health checks, and benchmarked at both short-query (input_len=32) and document (input_len=512)
regimes. The Emmy compiled-kernel lane could not be measured because the prebuilt serving image
(`cloudriftai/vllm-emmy:0.22.1-90c70d5e`) is not yet published for this target — publication was not authorized.

## Benchmark results — stock vLLM (pooling runner)

| Regime | Input len | Prompts | Concurrency | Throughput (req/s) | Token throughput (tok/s) | Mean E2EL (ms) | P99 E2EL (ms) |
|---|---|---|---|---|---|---|---|
| Short query | 32 | 256 | 32 | 81.83 | 2,618.69 | 360.72 | 701.34 |
| Document | 512 | 256 | 32 | 20.09 | 10,284.78 | 1,500.07 | 1,799.21 |

All 512 requests across both regimes succeeded (0 failures). The model loaded in ~5 s and compiled (torch.compile)
in ~28 s on first boot.

## Emmy eligibility

**Ineligible** — compiler qualification (trace + tune + O3 verify) could not be completed because the GPU host
lacks Python venv/pip support and the required ML stack (PyTorch, transformers) for the Emmy tracer. A complete
Emmy golden for the Qwen3 pooling path requires a working CUDA compilation environment on the target GPU.

The first failed eligibility gate is: gate 4 (compiler coverage — the architecture has a trace path via
`EmmyEmbedModel` but the toolchain was not available on the target to produce a working golden).

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
- No Emmy compiler golden on this run — a full trace of the EmmyEmbedModel pooling path requires a CUDA-equipped
  environment with torch and transformers available.
