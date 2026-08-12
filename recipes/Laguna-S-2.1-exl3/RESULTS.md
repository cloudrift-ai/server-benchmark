# Laguna S 2.1 EXL3 on one RTX 5090

Status: serving-qualified with the Emmy/vLLM engine and official 2.01 bpw checkpoint pinned by the recipe.

## Qualified deployment

| Item | Value |
| --- | --- |
| Model | `turboderp/Laguna-S-2.1-exl3` |
| Model revision | `348cf8c2cdea2326fde367399c8f0ff6e1dea842` |
| Checkpoint size | 30,674,709,145 bytes |
| Hardware | 1× NVIDIA GeForce RTX 5090 32 GB, compute capability 12.0 |
| Driver / CUDA | 580.173.02 / 13.0 |
| Engine | Emmy with 1Cat/vLLM 0.23.0 |
| Image | `emmy-laguna-exl3:849cd680` |
| Image ID used | `sha256:5bf66e18d7de24bb87ed72690d75f6fbcce75f844b557dbd1a9af7875f64a092` |
| Serving shape | TP1, context 16,384, concurrency 1, standard math, width-1 CUDA graph |

This is the highest-bpw official checkpoint that fits the card. The 2.50 and 3.00 bpw branches are 37.910 GB and
45.287 GB, respectively, so they cannot fit in 32 GB. The qualified 2.01 bpw checkpoint loads in 26.79 GiB and leaves
a 19,483-token KV capacity (1.19× the configured context) with 1,589 MiB of measured live GPU headroom.

## Best recipe performance

Measured 2026-08-11 with three complete repeats. Each repeat used one warm-up request followed by three unique
55-token prompts at concurrency 1, requested 64 output tokens, used greedy decoding, and ignored EOS. All nine
requests completed with exact token counts. Values are the mean across repeats; `±` is the sample standard deviation.

| Metric | Three-repeat mean ± standard deviation |
| --- | ---: |
| Successful / failed requests | 9 / 0 |
| Benchmark duration | 62.9385 ± 0.0550 s |
| Request throughput | 0.047666 ± 0.000042 requests/s |
| Output throughput | 3.05060 ± 0.00266 tokens/s |
| Total token throughput | 5.71987 ± 0.00499 tokens/s |
| Mean TTFT | 10,162.96 ± 6.54 ms |
| Mean TPOT / ITL | 171.6872 ± 0.1954 ms |

The same image completed a 16,383-token prompt plus one output token in 2,798.08 seconds with HTTP 200, zero
preemptions, and no allocator retry or OOM. The exact short oracle remained `42` afterward. Two additional clean
process restarts preserved the exact first-token top-10 ordering, returned the structured
`get_weather(city="Paris")` tool call, and produced coherent reasoning that reached `13 × 7 = 91`.

Pinned official ExLlamaV3 1.4.0 reached about 60 output tokens/s on a reasoning decode probe. This is reference-runtime
evidence rather than the selected deployment lane; its different server and workload make it non-comparable to the
table above. The matched serving gap remains dominated by Emmy host orchestration: one captured decode replay takes
about 169 ms even though profiled GPU kernels take about 6.15 ms.

## Compiler and accuracy qualification

The [canonical RTX 5090 golden](../../emmy/compiler/pipeline/search/goldens/rtx5090_sm120_laguna_s_2_1_exl3.yaml)
contains 10 standard-math realizations across five representative programs. Every retained realization has positive
deployable O3 and reference measurements; the measured speedups over the previous greedy realizations span
4.38–90.85×.

Emmy keeps the official EXL3 checkpoint compressed, uses the pinned native sparse-expert decoder, and keeps the
6-bit output head coded. Laguna's residual stream and expert down outputs remain float32 as required by the pinned
official ExLlamaV3 implementation; norms, attention Q/K/V, and intermediate activations remain float16. The exact
56-token arithmetic oracle returns token IDs `[89, 87]` (`42`) with first-token top-10 IDs
`[89, 785, 86, 1078, 88, 110, 605, 129, 87, 3589]`.

`EMMY_FAST_MATH` is intentionally absent. It produced no speed improvement and changed the exact top-10 ordering, so
the standard-math lane is the qualified recipe.

## Reproduce

```bash
emmy bench experiments/Laguna-S-2.1-exl3/serving_rtx5090 --ssh dikobraz@kenshin
```

The command uses the retained experiment YAML and writes ignored local output; do not use `--commit-results`.

## Limits

- The image is validated locally and was not published because publication was not authorized.
- The native sparse-expert path is selected only for single-token decode; wider concurrent decode uses the slower
  fallback, so concurrency above one is not qualified.
- Two-token chunked prefill preserves the 16,384-token context but makes long-prompt TTFT high.
- The remaining steady-decode gap is host orchestration around the captured graph, not sparse-expert GPU work.
