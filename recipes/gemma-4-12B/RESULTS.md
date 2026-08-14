# Gemma 4 12B with Emmy FAST_MATH on one RTX 5090

Status: serving-qualified with the Emmy generation plugin and `EMMY_FAST_MATH=1` at a 131,072-token context. The
base checkpoint serves text continuations through `/v1/completions`; this result does not claim the instruction-tuned
checkpoint's chat or multimodal contract.

## Qualified deployment

| Item | Value |
| --- | --- |
| Model | `google/gemma-4-12B` |
| Model revision | `023679ed352de9bb66cc873c9009ce3482585c08` |
| Hardware | 1× NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0 |
| Driver / CUDA | 580.173.02 / 13.0 |
| Engine | Emmy generation plugin with CloudRift 1Cat/vLLM 0.23.0 |
| Image | `cloudriftai/vllm-emmy-gemma-4-12b:0.23.0-5208dcf6` |
| Resolved image ID | `sha256:e0e9cf3e67960c2c0ea093b7d38d36350b07d1389824b87f99e00674dae053c7` |
| CloudRift 1Cat revision | `e2f29175989c0eaff249e9078191610679b8c7e8` |
| Serving shape | TP1, context 131,072, concurrency 64, FP16 weights/KV, V2 runner, FAST_MATH |

The image is retained locally on `kenshin`; it was not published. A fresh-container verification served standard and
FAST_MATH requests from the baked execution pack with all 336 cubins unchanged and no request-time Triton JIT.

## Best Emmy serving performance

Measured 2026-08-12 with the recipe's selected FAST_MATH engine, greedy decoding, ignored EOS, unique seeded random
prompts, and no prefix caching. These five workloads match the published Gemma 4 RTX measurement protocol.

| Input / output tokens | Concurrency | Output throughput (tokens/s) | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 256 / 256 | 64 | 1,382.71 | 1,322.06 | 30.06 |
| 4,096 / 4,096 | 1 | 54.93 | 465.64 | 18.10 |
| 4,096 / 4,096 | 4 | 206.18 | 1,254.15 | 19.10 |
| 4,096 / 4,096 | 8 | 375.68 | 1,205.50 | 21.01 |
| 8,192 / 256 | 4 | 103.02 | 3,117.18 | 26.67 |

The retained experiment YAML preserves the per-lane decode and prefill buckets from the
[published protocol](https://riftstack.ai/research/optimizing-gemma-4-12b-rtx). The final deployment recipe keeps the
warmed interactive default, decode bucket 32.

## Precision, quality, and context gates

Standard and FAST_MATH both scored `0.620 ± 0.034` strict and `0.630 ± 0.034` flexible exact match on the same
seeded 200-question GSM8K run. This was within the predeclared maximum 0.035 absolute FAST_MATH deficit on both
metrics, with an observed deficit of zero. On the matched 256/256 concurrency-one check, FAST_MATH reduced median
TTFT from 70.76 to 64.83 ms while holding output throughput at 58.01 versus 57.96 tokens/s. It also reduced the
131,071-token prefill wall time from 48.18 to 43.12 seconds. FAST_MATH is therefore the deployment default.

The selected image completed a material 131,071-token prompt plus one generated token with exact usage of 131,072
tokens, then passed six direct continuation and factual probes. The card reported 213,443 tokens of KV capacity in
the FAST_MATH configuration; 131,072 is the largest qualified power-of-two context below that capacity.

## Compiler qualification

The base-checkpoint [RTX 5090 Gemma 4 golden](../../emmy/compiler/pipeline/search/goldens/rtx5090_sm120_gemma4_base.yaml)
contains 25 exact Loop targets and 200 realizations. Standard and FAST_MATH full-model audits each reported 99
matches, zero gaps, zero drift, and zero compile failures. The base and instruction-tuned checkpoints have identical
execution geometry; their only configuration difference affecting generation is `eos_token_id`, so the baked
execution pack is shared while the base weights and model revision remain exact.

## Reproduce

```bash
emmy bench experiments/gemma-4-12B/serving_base_rtx5090 --ssh dikobraz@kenshin
```

The command uses the retained experiment YAML and writes ignored local output; do not use `--commit-results`. The
qualified local image must already exist on the target host because registry publication was intentionally omitted.

## Published comparison

The three 4,096/4,096 output-throughput lanes reproduced the published Emmy results within 0.8%. The 256/256
concurrency-64 lane improved output throughput by 13.5%, while the 8,192/256 concurrency-four lane was 8.4% lower.
Median TPOT stayed within 0.6% on all four long-prompt or long-generation lanes; the concurrency-64 lane was 7.0%
higher. Median TTFT deltas, in table order, were -28.2%, -1.1%, +17.2%, +19.7%, and +43.3%.

## Limits

- Context above 131,072 tokens is not qualified on this card.
- Chat, image, video, audio, and instruction-following behavior are not qualified for the base checkpoint.
- Re-run the exact accuracy and zero-JIT gates after an image, model, serving-shape, or driver change.
