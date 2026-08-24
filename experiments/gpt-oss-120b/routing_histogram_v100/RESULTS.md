# gpt-oss-120b routed input-slice histogram on 4× V100 SXM3

## Result

The routing distribution is skewed enough to continue toward an RTX 5090 + CPU DRAM prototype, but this experiment
does not establish its end-to-end speedup and the refreshed serving path fails generation correctness. Within each
workload window, the 40 most-selected input slices per layer cover 80.1–84.0% of routed-row selections while occupying
17.75 GiB of packed expert storage. Forty-eight slices cover 85.4–88.6% and occupy 21.30 GiB. Code is the least
concentrated workload.

The hot sets also move with the workload. At K=40, mean per-layer Jaccard overlap is 69.2% for chat versus math, 54.0%
for chat versus code, and 58.0% for math versus code. A single K=40 set selected over the combined trace covers only
75.0% of code selections, versus 80.1% for the code-local set. This supports testing an online, per-layer top-K policy
rather than assuming a fixed global placement.

These percentages are ideal frequency coverage, not measured cache hit rates. The counter records routed rows, not
the ordered sequence of buffer references or PCIe transfers. The next experiment must retain reference order and run
the actual cache on the RTX 5090 before making a latency claim.

## Protocol

`emmy bench experiments/gpt-oss-120b/routing_histogram_v100` ran the exact native-MXFP4
`openai/gpt-oss-120b` checkpoint at revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`. This is the checkpoint
intended for the hybrid deployment, not a separately requantized derivative. OpenAI describes gpt-oss-120b as a
mixture-of-experts model whose weights use MXFP4 and fit on one 80 GB GPU
([model announcement](https://openai.com/index/introducing-gpt-oss/)).

The server used pipeline parallelism across four V100s. Emmy kept each routed matrix in its native uint8 blocks and
uint8 E8M0 scales, decoded those values inside compiled programs, and updated one cumulative GPU counter for every
`(layer, routed input slice)` selection. Counter snapshots were emitted before uncaptured forwards; CUDA-graph decode
replays updated the same counters on-device.

After a delimiter probe, the client sent the first 16 rows from each fixed dataset, sequentially and greedily:

| Window | Dataset | Requests | Forward rows per layer | Routed selections across 36 layers |
| --- | --- | ---: | ---: | ---: |
| Chat | `HuggingFaceH4/ultrachat_200k`, `train_sft` | 16 | 4,541 | 653,904 |
| Math | `openai/gsm8k`, `test` | 16 | 2,719 | 391,536 |
| Code | `openai/openai_humaneval`, `test` | 16 | 3,236 | 465,984 |

Every request allowed 32 completion tokens at temperature 0. The final probe caused the complete code-window snapshot
to be logged. Snapshot subtraction matched `4 * (usage.total_tokens + 1)` at all 36 layers and all four workload
boundaries; the full counts and validation offsets are in `v100x4_routing_histograms.json` inside the archive.

## Run and platform

The successful refresh started at `2026-08-23T23:38:04Z`, completed at `2026-08-24T00:08:36Z`, and has run ID
`20260823T233804Z`. The row `v100x4` / `611ccd6e5f6e` succeeded: 50/50 requests returned HTTP 200 and no workload
process remained after the cleanup trap. The repository revision was `ce918dbc` after rebasing onto main revision
`1ee50309`; the working tree held the refreshed image pin and onboarding recipe update.

The supplied host ran Ubuntu 24.04.1 and kernel 6.8.0-124-generic with 334.4 GiB RAM, an Intel Xeon Platinum 8168,
four Tesla V100-SXM3-32GB GPUs, driver 580.159.03, CUDA 12.9.86, and Docker 29.5.3. The task-local image was
`sha256:9408e965af48ddafc76172945da356a7293510dd1ac44ae723b342ee947a9d98`; it contains Emmy on the supplied
Volta-compatible 1Cat vLLM base, vLLM `1.2.3.dev87+gd76126608.d20260810`.

The model loaded with 15.33 GiB per pipeline rank. During graph preparation the four processes used 28.6–29.7 GiB
each; PP0 later reported 11.88 GiB available for KV cache. Startup took about ten minutes, including 441 seconds for
model loading and cold Emmy compilation. gpt-oss attention sinks are not supported by the optimized Flash-V100 path,
so this trace explicitly allowed its Triton fallback. Latency from this run is not a serving-performance result.

## Usage histograms

Each histogram bins all 4,608 `(layer, input slice)` buffers by selection count divided by that layer's mean. Values
below 1× are colder than uniform routing for their layer. The first bin includes buffers with zero selections.

| Multiple of layer mean | Chat buffers | Math buffers | Code buffers |
| --- | ---: | ---: | ---: |
| 0–0.1× | 28.23% | 32.23% | 27.17% |
| 0.1–0.25× | 15.23% | 12.24% | 11.44% |
| 0.25–0.5× | 14.93% | 12.59% | 13.91% |
| 0.5–1× | 15.45% | 15.52% | 17.97% |
| 1–2× | 12.93% | 13.35% | 15.02% |
| 2–4× | 8.29% | 8.72% | 10.16% |
| 4–8× | 3.60% | 3.99% | 3.47% |
| ≥8× | 1.32% | 1.37% | 0.87% |

| Distribution metric | Chat | Math | Code |
| --- | ---: | ---: | ---: |
| Buffers selected at least once | 86.57% | 81.75% | 86.63% |
| Gini coefficient | 0.710 | 0.706 | 0.661 |
| Mean effective slices per layer | 49.77 | 50.46 | 58.20 |
| Mean slices needed for 80% of selections | 34.44 | 34.53 | 40.03 |
| Mean slices needed for 90% of selections | 50.83 | 49.92 | 56.31 |

The high Gini values and effective counts well below 128 reject the roughly-uniform-use hypothesis on these windows.
The large nonzero coverage also rejects a simpler conclusion that most buffers can remain permanently absent: many
slices are cold rather than unused.

### Ideal per-window top-K coverage

This table selects the best K slices independently in every layer and workload. It is an ideal LFU coverage bound,
not an observed cache policy.

| K per layer | Packed expert VRAM | GPU floor including other weights | Chat | Math | Code |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 7.10 GiB | 11.06 GiB | 59.44% | 58.29% | 52.97% |
| 24 | 10.65 GiB | 14.61 GiB | 70.09% | 69.63% | 64.78% |
| 32 | 14.20 GiB | 18.16 GiB | 77.89% | 77.85% | 73.50% |
| 40 | 17.75 GiB | 21.71 GiB | 83.68% | 83.95% | 80.14% |
| 48 | 21.30 GiB | 25.26 GiB | 88.14% | 88.64% | 85.36% |
| 56 | 24.85 GiB | 28.81 GiB | 91.54% | 92.22% | 89.46% |
| 64 | 28.40 GiB | 32.37 GiB | 94.09% | 94.87% | 92.66% |

One native-MXFP4 `(layer, input slice)` allocation is 13,236,480 bytes: gate/up and down blocks, their E8M0 scales,
and both biases. All 4,608 routed allocations occupy 56.80 GiB; the remaining checkpoint tensors occupy 3.96 GiB,
for 60.77 GiB total.

On a 32 GiB RTX 5090, K=64 is impossible before runtime state because its 32.37 GiB floor already exceeds card
capacity. K=48 leaves only 6.74 GiB for workspaces, CUDA graphs, and KV cache. K=40 leaves 10.29 GiB and is the safer
first prototype. If hot allocations exist only in VRAM rather than being duplicated in host memory, K=40 leaves
43.02 GiB of checkpoint storage in CPU DRAM; K=48 leaves 39.47 GiB. Both fit in 64 GiB with materially more headroom
than retaining a complete 60.77 GiB host copy.

### Workload movement

| K | Chat–math Jaccard | Chat–code Jaccard | Math–code Jaccard |
| ---: | ---: | ---: | ---: |
| 16 | 56.1% | 47.7% | 58.7% |
| 32 | 65.7% | 51.5% | 57.1% |
| 40 | 69.2% | 54.0% | 58.0% |
| 48 | 71.6% | 57.4% | 61.2% |

A static K=40 set chosen from all three windows covers 81.83% of chat, 81.85% of math, and 75.05% of code. At K=48
the corresponding coverage is 86.18%, 86.55%, and 80.89%. The gap from the per-window bounds is direct evidence that
the placement should adapt as the workload changes.

## Trace and tuning qualification

The exact checkpoint trace produced 18 frontend graphs and 24 distinct post-fusion Loop targets. Its SHA-256,
`df9301d20bfd212d260f34cf27abdc84ffce832b2572eb08e57f5b79584b6924`, is byte-identical to the pre-rebase
working golden, so the histogram protocol remained unchanged. M=1, M=16, and dynamic variants yielded 72 tuning
targets.

Both official arms started with empty DB, online-checkpoint, and compiled-kernel paths. They used four V100s, seed 17,
three candidate slots per target, patience 3, the O1 ranking lane, and the same 7,200-second deadline. MCTS-only
completed 72/72 targets in 651 seconds with 2,526 O1 benches and 64 clean searched winners. The corrected
proposal-seeded arm completed 72/72 in 656 seconds with 1,896 O1 benches, 70 clean searched winners, 48 clean
proposals, 20 unmatched proposals, and four ambiguous multi-kernel proposals. Candidate-specific failures were slow
kernels, watchdogs, or unmatched input names rather than trace failures.

Representative finalists from each arm and the model proposal were then recompiled twice at deployable O3. Every
pinned result had status `ok` and no integrity flags:

| Routed target | MCTS-only | Seeded search | Model proposal | Result |
| --- | ---: | ---: | ---: | --- |
| M=1 gate/up | 491.01 µs | 813.57 µs | 164.95 µs | Greedy, about 134 µs |
| M=1 down | 463.10 µs | 1,203.20 µs | 72.37 µs | Greedy, about 66–68 µs |
| M=16 gate/up | 5,506.05 µs | 11,701.25 µs | 1,872.90 µs | Proposal is greedy `t128` family |
| M=16 down | 3,912.70 µs | 3,914.24 µs | 859.65 µs | Proposal is greedy `t128` family |

The O1 search ranking did not transfer reliably to O3, and the proposed M=16 pins reduced to the same fully realized
`t128` family as greedy. No searched result was promoted to a canonical golden. The raw working goldens, databases,
online checkpoints, logs, and 24 O3 JSON records are retained as qualification evidence.

## Onboarding status

Native MXFP4 loading, tracing, compilation, four-stage API startup, CUDA-graph-safe usage tracking, and a complete
50-request experiment are qualified on the supplied V100 host. Generation correctness is not. With the GPT-OSS
reasoning parser enabled, chat, tool-choice, and a 2,590-token context probe all exhausted their output limits with
null content, reasoning, and tool calls. An eight-token diagnostic returned token ID 0 eight times, and requesting
logprobs failed because the response contained NaNs. The long prompt executed but missed the decode twin at width 30
and fell to a cold symbolic path; it did not demonstrate context recall.

This is not a maintained public serving recipe. The diagnostic image is local to the supplied host and was not
published because registry publication requires separate approval. The onboarding shell now pins the immutable model
revision and the measured 4× V100 target, but remains untested rather than claiming valid generation, a reproducible
public image, or a canonical tuned golden.

## Archive and limitations

`results_v100x4.tar.gz` contains the successful timestamped directory, including the system-only experiment record,
server and client logs, exact prompts and responses, image inspection, full per-layer histogram JSON, byte-identical
trace, both official tuning arms, O3 checks, and the generation-correctness probes.

The sample has 16 prompts per workload, one fixed ordering, greedy 32-token completions, and no concurrency. All 50
responses reached the length cap with null content. The histogram therefore includes routed rows from a broken token-0
decode path as well as the workload prompts; it is placement-frequency evidence, not answer-quality validation. The
small math and code deltas from the previous run reinforce that the reported coverage is an empirical bound, not a
fixed model property.

Histogram aggregation also loses temporal locality and cannot predict actual PCIe miss cost, overlap, prefetch
efficiency, or cache churn. The V100 Triton attention fallback and untuned greedy expert programs make request latency
non-representative. The next gate is to diagnose the NaN logits and rerun correct generation; ordered-reference cache
replay and the actual K=40/K=48 hybrid implementation on the RTX 5090 + 64 GiB host follow that gate.
