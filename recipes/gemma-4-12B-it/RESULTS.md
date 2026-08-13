# Gemma 4 12B IT with Emmy FAST_MATH on one RTX 5090

Status: serving-qualified with the Emmy generation plugin and `EMMY_FAST_MATH=1` at a 16,384-token context. No quality
regression was observed in the published accuracy checks, so FAST_MATH remains the deployment default.

## Qualified deployment

| Item | Value |
| --- | --- |
| Model | `google/gemma-4-12B-it` |
| Model revision | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Hardware | 1× NVIDIA GeForce RTX 5090, compute capability 12.0 |
| Driver / CUDA | 580.159.03 / 13.0 |
| Engine | vLLM 0.23.0 with `EmmyGenModel` |
| Image | `cloudriftai/vllm-emmy-gemma-4-12b-it:latest` |
| Resolved image ID | `sha256:5add12d3b7f4673790b435b76635082433538e3615fbc40227fa1c0db64c9ff3` |
| Serving shape | TP1, context 16,384, FP16 weights/KV, FAST_MATH, decode bucket 32 |

## Best Emmy serving performance

The selected-lane measurements use greedy decoding, ignored EOS, unique seeded prompts, and no prefix caching. These
are the best published results for the recipe engine and precision lane.

| Input / output tokens | Concurrency | Output throughput | Median TTFT | Median TPOT |
| ---: | ---: | ---: | ---: | ---: |
| 256 / 256 | 64 | 1,218.6 tokens/s | 1,841 ms | 28.1 ms |
| 4,096 / 4,096 | 1 | 54.5 tokens/s | 471 ms | 18.2 ms |
| 4,096 / 4,096 | 4 | 205.2 tokens/s | 1,070 ms | 19.2 ms |
| 4,096 / 4,096 | 8 | 375.9 tokens/s | 1,007 ms | 21.0 ms |
| 8,192 / 256 | 4 | 112.5 tokens/s | 2,176 ms | 26.6 ms |

The full measurement protocol is published in the
[Gemma 4 optimization record](https://riftstack.ai/research/optimizing-gemma-4-12b-rtx).
The grid held context at 16,384 and used the same Emmy FAST_MATH engine, while selecting decode and prefill buckets
for each concurrency point. The final recipe keeps the warmed interactive default, decode bucket 32; use the retained
experiment YAML to reproduce the full performance grid.

## FAST_MATH accuracy and compiler qualification

Hybrid accumulation held relative L2 error near `3.3e-4` across the measured K range. The FAST_MATH lane scored
`0.695 ± 0.033` exact match on the 200-question GSM8K check, with no measured quality regression. The deployed
inventory had 226 of 276 measured realizations at or above eager performance and a 1.30× geometric-mean ratio.

The current [RTX 5090 Gemma 4 golden](../../emmy/compiler/pipeline/search/goldens/rtx5090_sm120_gemma4.yaml) contains
281 self-contained programs. Representative selected FAST_MATH projection times at sequence length 512 were 61.5 µs
for Q, 36.4 µs for KV, 64.1 µs for output, 362.4 µs for gate/up, and 214.2 µs for down projection.

## Reproduce

```bash
emmy bench experiments/gemma-4-12B/serving_rtx5090 --ssh dikobraz@kenshin \
  --filter 'engine.llm.vllm.extra_env=EMMY_FAST_MATH=1*'
```

The filter selects only the recipe's Emmy FAST_MATH engine lane. Output remains ignored locally; do not use
`--commit-results`.

## Limits

- The mutable release tag must be resolved to an immutable image identity for every new performance run.
- The 32 GB card does not fit the model's full native context at FP16; the recipe remains capped at 16,384.
- The plugin retains a per-token integration boundary and does not replace vLLM attention or batch composition.
- Re-run the exact accuracy gate before retaining FAST_MATH after an image, model, serving-shape, or driver change.
