# Gemma 4 12B with vLLM on one RTX 5090

Status: serving-qualified for FP16 text completion at a 16,384-token context. This is the base checkpoint, so the
recipe uses `/v1/completions` and does not claim chat or multimodal generation.

## Qualified deployment

| Item | Value |
| --- | --- |
| Model | `google/gemma-4-12B` |
| Model revision | `023679ed352de9bb66cc873c9009ce3482585c08` |
| Hardware | 1× NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0 |
| Driver / CUDA | 580.173.02 / 13.0 |
| Engine | vLLM 0.23.0 |
| Image | `vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f` |
| Serving shape | TP1, context 16,384, concurrency 1, FP16 weights and KV cache |

`EMMY_FAST_MATH` is not set because the recipe uses standard vLLM without the Emmy generation plugin.

## Best recipe performance

Measured 2026-08-11 with four unique random 256-token prompts, 128 requested output tokens, concurrency 1, greedy
decoding, and ignored EOS. The semantic completion smoke test passed before the benchmark.

| Metric | Result |
| --- | ---: |
| Successful / failed requests | 4 / 0 |
| Benchmark duration | 8.51 s |
| Request throughput | 0.47 requests/s |
| Output throughput | 60.15 tokens/s |
| Total token throughput | 180.45 tokens/s |
| Median TTFT | 56.60 ms |
| Median TPOT / ITL | 16.18 / 16.31 ms |

The prompt `2 + 2 =` returned the correct continuation. A 0.90 exploratory boot could not allocate the native
262,144-token KV cache, while 0.96 exceeded workstation headroom. The successful 0.95 recipe qualifies 16,384 tokens;
it does not prove that native context is impossible at every memory setting.

## Compiler qualification

The shared [RTX 5090 Gemma 4 golden](../../emmy/compiler/pipeline/search/goldens/rtx5090_sm120_gemma4.yaml) contains
281 self-contained programs and reconstructs under the current schema. It qualifies the model geometry, but this base
checkpoint was not separately served through the Emmy generation plugin.

## Reproduce

```bash
emmy bench experiments/gemma-4-12B/serving_base_rtx5090 --ssh dikobraz@kenshin
```

The command uses the retained experiment YAML and writes ignored local output; do not use `--commit-results`.

## Limits

- Context beyond 16,384 tokens is not qualified on this card.
- Chat, image, video, and audio generation are not qualified for the base checkpoint.
- The result covers one exact single-request workload, not a concurrency sweep.
