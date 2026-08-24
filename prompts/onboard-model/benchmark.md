# Practical Model Benchmarking

Use this guidance when onboarding or verification reproduces, compares, or substantiates model performance. Apply
only the parts that fit the model, target hardware, serving path, and available time. A smaller representative matrix
with explained omissions is better than a nominally complete run that cannot finish or does not answer the decision
at hand. The onboarding skill's deadlines, artifact rules, accuracy gates, and cleanup requirements remain
authoritative.

## Make the comparison interpretable

- Record the exact model revision, GPU name and count, driver, CUDA and framework versions, engine images, dtype,
  quantization, KV-cache dtype, context limit, and important serving flags.
- Prefer a paired comparison between the stock serving engine and the candidate configuration using the same engine
  version, model, tokenizer, request client, capacity settings, and workload. Add another engine only when it provides
  a useful sanity check or represents a realistic alternative.
- Keep the standard numerical path separate from optional faster precision or speculative-decoding lanes. Treat each
  as its own configuration and require an appropriate correctness check before recommending it.
- Warm each lane before recording it and repeat measurements when time permits. Preserve failures and variability
  instead of reporting only the best attempt.

## Cover useful serving regimes

Choose a compact set of workloads that represents the ways the model is expected to be used. Useful regimes include:

- single-stream latency with a substantial prompt and output;
- moderate concurrency to show batch and throughput scaling;
- a long-input, short-output case that emphasizes prefill and retrieval-style use;
- high-concurrency short requests for API, agent, classification, or extraction traffic.

These are examples, not a required grid. Adjust token lengths, concurrency, and request count to the model's context,
modality, memory limits, and intended task. When practical, use unique seeded inputs, deterministic decoding, fixed
input and output work, the same client and seed across lanes, and no prefix-cache reuse. If the engine cannot match a
control, record the difference.

Report output throughput together with time to first token (TTFT), time per output token (TPOT) or inter-token
latency, failure count, and useful latency percentiles. Separate prefill-dominated and decode-dominated conclusions.
For saturated workloads, distinguish first-wave service latency from later requests that include queue drain; do not
present queued latency as pure prefill time.

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
