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
| Serving shape | TP8, PP2, context 4,096, concurrency 8, FP8 KV cache |

`EMMY_FAST_MATH` is not set because this is not an Emmy serving engine.

## Best recipe performance

Measured 2026-08-11 with three repeats. Each repeat used eight unique 1,024-token prompts at concurrency 8 and
requested 64 output tokens with greedy decoding and ignored EOS. All 24 requests completed with exact token counts.

| Metric | Three-repeat mean |
| --- | ---: |
| Successful / failed requests | 24 / 0 |
| Benchmark duration | 16.4633 s |
| Request throughput | 0.4933 requests/s |
| Output throughput | 31.71 tokens/s |
| Total token throughput | 539.0533 tokens/s |
| Mean TTFT | 3,857.82 ms |
| Mean TPOT / ITL | 198.7433 / 198.7433 ms |

The same deployment passed completion, chat, 4,096-token-boundary, and structured tool-call probes. Tool calling
returned `multiply(a=17, b=19)` correctly. Fully separated reasoning output remains unqualified because one direct
probe left a stray `</think>` marker in assistant content.

## Compiler qualification

The [canonical V100 golden](../../emmy/compiler/pipeline/search/goldens/v100_sm70_deepseek_v4_flash_0731.yaml)
contains 279 exact Loop realizations across 13 programs. Every retained realization has positive deployable O3 and
reference timings. The schema-migrated file passed 279/279 stored-record and offer checks, but its final cold-deploy
in-model replay did not finish before the 106-minute cutoff; the complete qualification predates that schema rewrite.
This compiler evidence does not establish an Emmy serving path for the checkpoint.

## Reproduce

```bash
emmy bench experiments/DeepSeek-V4-Flash-0731/serving_v100_sxm3 --ssh riftuser@185.165.50.61
```

The command uses the retained experiment YAML and writes ignored local output; do not use `--commit-results`.

## Limits

- The qualified lane is the recipe's 1Cat/vLLM engine, not Emmy.
- Context beyond 4,096 tokens and fully parsed reasoning output are not qualified.
- The workload is a serving qualification, not a broad concurrency or context sweep.

## Current regression

Against the prior comparable complete run, the selected 2026-08-11 result improved output throughput from 29.09 to
31.71 tokens/s (+9.0%) and mean TPOT from 222.34 to 198.74 ms (-10.6%), but mean TTFT regressed from 3,528.52 to
3,857.82 ms (+9.3%). The newer complete run remains the main result because its sustained generation metrics are
better; the TTFT regression remains open.
