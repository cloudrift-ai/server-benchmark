# Laguna S 2.1 FP8 on 8× V100 SXM3 32 GB

Status: serving-qualified with the native SM70 FP8 MoE path in the 1Cat/vLLM engine pinned by the recipe.

## Qualified deployment

| Item | Value |
| --- | --- |
| Model | `poolside/Laguna-S-2.1-FP8` |
| Model revision | `06d71e91db70a11b08ee6a09c3c4818c85a61953` |
| Hardware | 8× Tesla V100-SXM3-32GB, compute capability 7.0 |
| Driver | 580.159.03 |
| Engine source | CloudRift 1Cat/vLLM `96f26179bf28aaea645635b8ec6f26c98360e0c2` |
| Image | `emmy-round2-laguna-fp8-sm70:96f26179bf28` (local, not published) |
| Image ID | `sha256:ae3fcf3a781eb33db58447fd10afb650b9684937211e8783d64dbe314dbcdca6` |
| Serving shape | TP8, 1,048,576-token context, concurrency 1, native SM70 FP8 MoE |
| Steady GPU memory | 29,844–29,846 MiB per rank before requests |
| KV cache | 12.08 GiB per rank; 2,101,882-token aggregate capacity |

The new model revision is a documentation-only update from `9e0b8ba630080b0e6f20a7b43294a9f2232fd247`: all 49
safetensor blob OIDs and the config, index, tokenizer, and operational blob OIDs are identical. `EMMY_FAST_MATH` is
not set because the selected serving engine is 1Cat rather than Emmy.

## Best recipe performance

Measured on 2026-08-12 with five random 32-token prompts per repeat, 64 requested output tokens, concurrency 1,
greedy decoding, and ignored EOS. Two client warmup repeats were excluded. The table reports the mean of the next four
steady repeats; all 20 measured requests succeeded.

| Lane | Output tok/s | Mean TTFT | Mean TPOT / ITL | Requests/s | Requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1Cat native SM70 FP8 MoE, TP8 | 8.65 | 150.00 ms | 115.12 / 115.12 ms | 0.14 | 20 / 20 |

The four output-throughput repeats were 8.65, 8.66, 8.66, and 8.61 tok/s. Their mean duration was 37.01 seconds and
their total input-plus-output throughput was 12.97 tok/s. The server log confirmed the native TurboMind batched FP8
MoE path; the FP16 expert fallback was disabled.

## Correctness and context

- Chat returned `323` for `17 * 19`; raw completion and top log probabilities remained finite.
- The reasoning parser returned a non-empty reasoning field for a multi-step train problem.
- Non-streaming and streaming tool calls both parsed `get_weather` with `{"city":"Paris"}`.
- The short completion oracle returned ` Paris` before the context test with finite log probability
  `-0.0025943215005099773`.
- A material 1,048,575-token prompt exercised the complete uncached prefill in 3,121.29 seconds with zero preemptions.
  The identical no-logprobs generation returned HTTP 200 with exact 1,048,575-input-plus-one-output usage; its fully
  cached repeat took 1.72 seconds. A one-token-over-boundary request failed cleanly with HTTP 400, and the short
  completion oracle was unchanged after the boundary request.

The native path retains Laguna's FP32 residual and down-output contract while keeping norms, QKV, and MoE
intermediates in FP16. The 1Cat changes also fill FP8 expert pointer tables directly into a PyTorch-owned CUDA tensor,
so allocation failure is reported synchronously instead of surfacing later as an illegal memory access.

## Compiler qualification

The [canonical V100 golden](../../emmy/compiler/pipeline/search/goldens/v100_sm70_laguna_s_2_1_fp8.yaml) contains 22
current compiler targets and 25 verified realizations with paired deployable O3 and reference timings. All 15
full-program deploy offers matched stored realizations with zero gaps, drift, fall-through, or compile failures.
Representative exact-layer traces and all checkpoint tensor OIDs established that the documentation-only model
revision does not change the traced inventory.

## Reproduce

The exact local image tag above must be present on the target host. Run the retained six-repeat experiment with the
existing Emmy CLI, discard its first two client warmups, and report repeats three through six:

```bash
emmy bench experiments/Laguna-S-2.1-FP8/serving_v100_sxm3 --ssh riftuser@66.172.10.131
```

The command writes ignored local output. Do not use `--commit-results`; `RESULTS.md` is the only retained benchmark
artifact.

## Limits

- The serving image is available only as the pinned local image on the qualified host; it was not published.
- Optional logprob reporting is not qualified for long contexts. At both 131,071 and 1,048,575 input tokens,
  `logprobs=5` completed generation but the API returned HTTP 400 while serializing a NaN logprob. The identical
  no-logprobs requests returned HTTP 200, and ordinary generation remained healthy.
- A concurrent full-model compiler trace reached about 539 GiB RSS and was terminated before host OOM. Qualification
  used sequential exact-layer traces and the complete current-inventory deploy audit instead.
