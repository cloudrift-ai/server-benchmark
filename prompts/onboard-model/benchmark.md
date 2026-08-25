# Practical Model Benchmarking

Use this guidance when onboarding or verification reproduces, compares, or substantiates model performance. For
generative text serving, start with the canonical workload matrix below so results stay comparable across recipes and
platforms. Apply the capacity adjustment instead of inventing a different grid. Omit a row only when the model's
validated context, modality, or serving path makes that workload inapplicable, and explain the omission. The
onboarding skill's deadlines, artifact rules, accuracy gates, and cleanup requirements remain authoritative.

## Make the comparison interpretable

- Record the exact model revision, GPU name and count, driver, CUDA and framework versions, engine images, dtype,
  quantization, KV cache dtype, context limit, and important serving flags.
- Prefer a paired comparison between the stock serving engine and the candidate configuration using the same engine
  version, model, tokenizer, request client, capacity settings, and workload. Add another engine only when it provides
  a useful sanity check or represents a realistic alternative.
- Keep the standard numerical path separate from optional faster precision or speculative-decoding lanes. Treat each
  as its own configuration and require an appropriate correctness check before recommending it.
- Warm each lane before recording it and repeat measurements when time permits. Preserve failures and variability
  instead of reporting only the best attempt.

## Run the canonical serving matrix

Use exact random-token input and output lengths, and begin with these target concurrency, request-count, and repeat
values:

| Input tokens | Output tokens | Target concurrency | Requests | Repeats | Purpose |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 4096 | 1 | 8 | 3 | Single-stream long-context latency and decode. |
| 4096 | 4096 | 4 | 32 | 1 | Moderate batch scaling. |
| 4096 | 4096 | 8 | 64 | 1 | Higher batch scaling. |
| 8192 | 256 | 4 | 16 | 1 | Long-input, short-output prefill and retrieval-style serving. |
| 256 | 256 | 64 | 256 | 1 | Saturated short-turn API, agent, classification, or extraction traffic. |

Use seed `0`, temperature `0`, ignore EOS, unique seeded prompts, and no prefix-cache reuse. Use the same generated
inputs, client, and benchmark controls across every lane. Configure the server for at least 8,448 total tokens before
running this matrix; if the model's validated context is smaller, retain the rows it can serve and record the others
as inapplicable rather than shortening them. Embedding and materially different multimodal workloads need an analogous
task-specific matrix instead of pretending that token generation measures them.

### Adjust concurrency for capacity

Treat the concurrency values as targets. If any compared lane cannot admit or complete a row because of VRAM, KV cache
capacity, or an out-of-memory failure, halve that row's concurrency until every compared lane succeeds. Starting with
the row's target, follow `64 → 32 → 16 → 8 → 4 → 2 → 1` from that point. Keep input and output lengths
unchanged. Use the same reduced concurrency for all lanes on that platform. Record both the target and actual
concurrency plus the first capacity failure. If concurrency `1` cannot run, preserve the failure and omit the row.

After reducing concurrency, keep the original number of steady-state waves: use `requests = concurrency × 8` for
the 4,096-input / 4,096-output rows and `requests = concurrency × 4` for the other two rows. Use three repeats for a
single-stream 4,096 / 4,096 row and one repeat otherwise.

Report output throughput together with time to first token (TTFT), time per output token (TPOT) or inter-token
latency, failure count, and useful latency percentiles. Separate prefill-dominated and decode-dominated conclusions.
For the short-turn saturation row, use the table's four-wave run for throughput and TPOT. Measure TTFT separately with
`requests = concurrency` so every request belongs to the first admitted wave; do not present later queued requests as
pure prefill time.

## Measure kernels when they inform the serving result

When Emmy kernels are part of the candidate, benchmark a broad deployed inventory rather than a few favorable shapes.
Include the relevant decode buckets, prefill widths, symbolic shapes, and precision lanes. Compare against suitable
reference backends such as eager execution, `torch.compile`, or the vendor library used by the serving baseline.

Summarize coverage and distribution, not only peak speedup: case count, cases at or above baseline, geometric-mean
ratio, a useful upper percentile, best and worst cases, and every omitted or failed case. Use deployable compiler
settings for reported performance. Explain why kernel-level gains do or do not transfer to end-to-end serving,
including integration, attention, scheduling, batching, launch, or memory-bound costs that remain outside the
optimized kernels.

## Check correctness and preserve reproduction

- Run service smoke probes and the repository's kernel accuracy gates before interpreting performance.
- For a materially different numerical path, compare against the standard lane with a checkpoint-appropriate task or
  quality evaluation. Add focused numerical-error measurements when they help explain the tradeoff.
- Use the same evaluation inputs and seed across configurations, report uncertainty when available, and avoid calling
  small score differences meaningful when they fall within expected variation.
- Keep the experiment runnable with one documented command. Preserve exact commands, configuration, revisions,
  environment facts, raw results, and failed rows in the repository's normal experiment artifacts.
- State fit limits, out-of-memory lanes, unsupported features, incomplete coverage, and other caveats. Do not imply
  that a result transfers to another checkpoint, GPU, precision policy, or serving stack without measurement.
