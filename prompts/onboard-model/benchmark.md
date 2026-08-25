# Practical Model Benchmarking

Use this guidance when onboarding or verification reproduces, compares, or substantiates model performance. For
generative text serving, use the applicable workload matrix as a set of comparable measurement anchors. GPU class
selects a useful workload profile, not the final serving configuration. Choose concurrency, memory policy, scheduler
settings, and parallelism from the model, hardware, intended traffic, and measured behavior. Omit a row when it does
not inform the deployment decision or when the model's validated context, modality, or serving path makes it
inapplicable, and explain the omission. The onboarding skill's deadlines, artifact rules, accuracy gates, and cleanup
requirements remain authoritative.

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

## Choose the serving matrix

Use the consumer matrix for GeForce and ordinary single-GPU workstation onboarding. Use the datacenter matrix for
server accelerators and multi-GPU server systems. For an ambiguous professional GPU, choose based on the intended
deployment; when that is unspecified, use the consumer matrix for a single-GPU target and the datacenter matrix for a
multi-GPU target. Record which matrix was selected. Treat this classification as a workload-profile choice, not a
claim that every GPU in that class should use the same concurrency or deployment strategy.

Use exact random-token input and output lengths, and begin with the selected target concurrency, request-count, and
repeat values when those rows are relevant. Consumer serving is commonly single-user or interactive, so treat the
concurrency-`1` row as its primary operating point. The higher-concurrency consumer rows are recommended scaling
checks when the intended service expects concurrent traffic; do not require a separate capacity sweep for an ordinary
single-user deployment.

### Consumer GPUs

| Input tokens | Output tokens | Target concurrency | Requests | Repeats | Purpose |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 4096 | 1 | 8 | 3 | Single-stream long-context latency and decode. |
| 4096 | 4096 | 4 | 32 | 1 | Moderate batch scaling. |
| 4096 | 4096 | 8 | 64 | 1 | Higher batch scaling. |
| 8192 | 256 | 4 | 20 | 1 | Long-input, short-output prefill and retrieval-style serving. |
| 256 | 256 | 64 | 320 | 1 | Saturated short-turn API or agent traffic. |

### Datacenter GPUs

| Input tokens | Output tokens | Target concurrency | Requests | Repeats | Purpose |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 4096 | 1 | 8 | 3 | Single-stream long-context latency and decode. |
| 4096 | 4096 | 16 | 128 | 1 | Moderate datacenter batch scaling. |
| 4096 | 4096 | 32 | 256 | 1 | Higher datacenter batch scaling. |
| 8192 | 256 | 16 | 80 | 1 | Long-input, short-output prefill and retrieval-style serving. |
| 1024 | 1024 | 64 | 320 | 1 | Balanced shared-service traffic. |
| 256 | 256 | 256 | 1280 | 1 | Saturated short-turn API or agent traffic. |

Use seed `0`, temperature `0`, ignore EOS, unique seeded prompts, and no prefix-cache reuse. Use the same generated
inputs, client, and benchmark controls across every lane. Configure the server for at least 8,448 total tokens before
running this matrix; if the model's validated context is smaller, retain the rows it can serve and record the others
as inapplicable rather than shortening them. Embedding and materially different multimodal workloads need an analogous
task-specific matrix instead of pretending that token generation measures them.

### Select a serving configuration

Treat matrix concurrency values as starting targets, not universal limits or required final settings. Before selecting
the final configuration, inspect the engine's reported KV-cache or token capacity, scheduler admission limit, memory
headroom, and behavior under load. Consider the engine controls that materially affect the result, such as vLLM's
sequence and batched-token limits or SGLang's running-request, memory-pool, and chunked-prefill settings. Use current
engine guidance and the relevant runtime repository from the README project map rather than prescribing one setting
for every model.

For a normal capacity or comparison row, align client concurrency with the server's request-admission limit so the
measurement represents the intended batch rather than an accidental queue. A separately labeled overload row may
exceed the server limit when queueing behavior is the question. If a row cannot run cleanly because of memory, KV
capacity, preemption, request retraction, sustained queue growth, or an out-of-memory failure, choose a lower load or
different serving configuration and preserve the evidence. Keep the input and output lengths unchanged when comparing
lanes, and use the same supported concurrency across those lanes.

Choose the recommended configuration from useful throughput under the intended latency objective, not merely from
successful completion or the highest concurrency. When no latency objective is supplied, explain the balance selected
from throughput, TTFT, TPOT or inter-token latency, queueing, and capacity evidence. Stop exploring once the retained
measurements support a clear decision; a fixed sweep is not required. Consumer onboarding normally needs no additional
concurrency search beyond the applicable recommended rows. For datacenter deployment, consider whether tensor,
pipeline, expert, or data parallelism, or independent replicas, better match model fit, interconnect, throughput, and
per-user latency. Let measured evidence determine the strategy rather than assuming that the smallest fitting tensor
parallel size is the final deployment.

Use at least five steady-state waves for a throughput row. The table request counts already meet that requirement.
One run is enough for an exploratory point; repeat the selected configuration and its decision-relevant comparison
lane at least three times, balancing run order when practical. Preserve variability rather than reporting only the
best run.

Report output throughput together with time to first token (TTFT), time per output token (TPOT) or inter-token
latency, failure count, and useful latency percentiles. Separate prefill-dominated and decode-dominated conclusions.
When they affect the recommendation, also report observed running requests, queue depth, KV usage, and
preemption or retraction evidence rather than only the requested client load.

Use concurrency `1` to measure intrinsic prefill latency. At higher concurrency, report TTFT as end-to-end TTFT that
may include admission and queueing; do not describe it as pure prefill time. If a first-admitted-batch measurement is
useful, bound it by the engine's observed admission capacity and state how that capacity was established.

The unique-prompt, no-cache matrix is the reproducible baseline. When the final recipe enables prefix or radix caching,
or the intended traffic has shared system prompts or documents, add a separate representative cache-reuse lane and
record its prefix shape or observed hit rate. Do not mix cached and uncached results in one comparison. Use a
task-specific shorter-output workload when classification or extraction is an intended deployment rather than
describing the 256-output-token row as that task.

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
