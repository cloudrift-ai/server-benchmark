# Laguna S 2.1 FP8 on 8× V100 SXM3 32 GB

Status: serving-qualified with the 1Cat/vLLM engine pinned by the recipe. Emmy compiler coverage is complete, but the
checkpoint is ineligible for Emmy serving because its routed FP8 expert storage is not supported by the executable
Emmy generation path.

## Qualified deployment

| Item | Value |
| --- | --- |
| Model | `poolside/Laguna-S-2.1-FP8` |
| Model revision | `9e0b8ba630080b0e6f20a7b43294a9f2232fd247` |
| Hardware | 8× Tesla V100-SXM3-32GB, compute capability 7.0 |
| Driver | 580.159.03 |
| Engine | 1Cat/vLLM `91aca502d2bb1f05d9208ab2edec9fae53ff0d0b` |
| Image | `cloudriftai/1cat-vllm-sm70:1.2.2-cloudrift` |
| Image ID used | `sha256:8405bb60d24610417d0d6da278a753e2c968bfd1e0d7ff7f79cd6601a038b2be` |
| Serving shape | TP8, context 4,096, concurrency 1, FP16 expert fallback |

`EMMY_FAST_MATH` is not set because this is not an Emmy serving engine and its precision-trading MMA lanes do not
apply to V100 compute capability 7.0.

## Best recipe performance

Measured 2026-08-09 with one random 32-token prompt, 16 requested output tokens, concurrency 1, greedy decoding, and
ignored EOS. The deployment passed its chat smoke test and completed without failure.

| Metric | Result |
| --- | ---: |
| Successful / failed requests | 1 / 0 |
| Benchmark duration | 3.03 s |
| Request throughput | 0.33 requests/s |
| Output throughput | 5.27 tokens/s |
| Total token throughput | 15.82 tokens/s |
| Mean TTFT | 949.39 ms |
| Mean TPOT / ITL | 138.84 / 138.84 ms |

## Compiler qualification

The [canonical V100 golden](../../emmy/compiler/pipeline/search/goldens/v100_sm70_laguna_s_2_1_fp8.yaml) covers 36
retained targets from all 48 decoder layers plus embedding, final normalization, and the output head. All 41 stored
realizations have positive deployable O3 and reference timings. Five large computed-operand reductions use selected
two-kernel placement routes; their best repeated O3 totals are 13.844, 82.204, 15.308, 27.663, and 18.306 ms,
respectively, for improvements of 3.05–10.74× over the former fused routes.

This architecture-derived compiler evidence does not preserve the checkpoint's routed-expert storage in a served
Emmy graph. The recipe therefore uses the compatible 1Cat FP16 expert-dequantization fallback.

## Reproduce

```bash
emmy bench experiments/Laguna-S-2.1-FP8/serving_v100_sxm3 --ssh riftuser@66.172.10.131
```

The command uses the retained experiment YAML and writes ignored local output; do not use `--commit-results`.

## Limits

- The native SM70 FP8 MoE path faults during expert post-processing, so the recipe dequantizes experts to FP16.
- Structured tool calls, parsed reasoning, context beyond 4,096, and concurrency above one are not qualified.
- No Emmy serving image was published for this checkpoint.

## Current regression

The 2026-08-11 same-recipe revalidation completed correctly but did not recover the best result. Output throughput
fell from 5.27 to 4.72 tokens/s (-10.4%), mean TPOT rose from 138.84 to 161.05 ms (+16.0%), and mean TTFT rose from
949.39 to 969.49 ms (+2.1%). Each run contained one request, so the cause remains unresolved and may include run
variance; the 2026-08-09 complete run remains the main result.
