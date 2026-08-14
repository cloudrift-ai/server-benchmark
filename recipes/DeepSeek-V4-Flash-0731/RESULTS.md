# DeepSeek V4 Flash 0731 on 16× V100 SXM3 32 GB

Status: serving-qualified with the 1Cat/vLLM engine pinned by the recipe. Emmy serving is ineligible because the
DeepSeek V4 compressor and hyper-connection path has no executable external-attention serving ABI.

## Qualified deployment

| Item | Value |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Model revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Hardware | 16× Tesla V100-SXM3-32GB, compute capability 7.0 |
| Driver / CUDA | 580.173.02 / 13.0 |
| Engine | 1Cat/vLLM `d76126608155c334df7c2fb9b75096f879624859` |
| Image | `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` |
| Serving shape | TP8, PP2, context 1,048,576, concurrency 8, FP8 KV cache |

The recipe disables process-local SM70 MXFP4 small-shape timing selection. The timed selector chose different W13
expert GEMM configurations after fresh starts and changed greedy output. The fixed default dispatch returned the exact
same token and logits in six probes across three fresh TP8×PP2 processes, and again after the final full-context
restart. `EMMY_FAST_MATH` is not set because this is not an Emmy serving engine.

## Best recipe performance

Measured 2026-08-11 with the final 1,048,576-context recipe. The reported three steady repeats followed one
unreported post-restart priming repeat. Each repeat used eight unique 1,024-token prompts at concurrency 8 and
requested 64 output tokens with greedy decoding and ignored EOS. All 24 reported requests completed with exact token
counts. Spread is the population standard deviation across the three repeats.

| Metric | Three-repeat mean ± standard deviation |
| --- | ---: |
| Successful / failed requests | 24 / 0 |
| Benchmark duration | 14.9793 ± 0.1260 s |
| Request throughput | 0.5341 ± 0.0045 requests/s |
| Output throughput | 34.1830 ± 0.2886 tokens/s |
| Total token throughput | 581.1111 ± 4.9058 tokens/s |
| Mean TTFT | 2,580.88 ± 163.94 ms |
| Mean TPOT / ITL | 195.949 ± 4.113 ms |

## Context and accuracy

The engine allocated KV capacity for 4,184,221 tokens on PP0 and 4,220,292 tokens on PP1. An exact
1,048,575-token prompt plus one decode token completed with HTTP 200 in 909.724 seconds and reported 1,048,576 total
tokens. Peak physical allocation left 361 MiB on PP0 and 589 MiB on PP1; the run had zero preemptions, allocator
errors, or OOMs.

The exact arithmetic probe returned `323` with identical logits twice before the boundary request, twice immediately
after it, and twice after a fresh full-context restart. The same deployment passed factual completion, thinking-off
chat, separated reasoning, and structured tool-call probes. The tool parser returned `multiply(a=17, b=19)`.

## Compiler qualification

The [canonical V100 golden](../../emmy/compiler/pipeline/search/goldens/v100_sm70_deepseek_v4_flash_0731.yaml)
contains 279 exact Loop realizations across 13 programs. Every retained realization has positive deployable O3 and
reference timings. The current schema passed all 279 stored-record and offer checks. The in-model audit traced four
exact representative DeepSeek graphs with 945, 1,156, 1,087, and 1,156 Graph IR nodes, but `audit_card` returned no
verdict within the bounded 106-minute replay. External sampling localized the compile-time hotspot to merge-region
dependency resolution in the Loop splicer. In-model compiler qualification therefore remains incomplete, and this
compiler evidence does not establish an Emmy serving path for the checkpoint.

## Reproduce

```bash
emmy bench experiments/DeepSeek-V4-Flash-0731/serving_v100_sxm3 --ssh riftuser@185.165.50.61
```

The experiment runs four client repeats. The first warms the complete unique prompt set after deployment; aggregate
repeats two through four to reproduce the reported steady result. The command writes ignored local output; do not use
`--commit-results`.
