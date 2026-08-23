# gpt-oss-120b routed input-slice histogram on 4× V100 SXM3

## Result

The routing distribution is skewed enough to justify an RTX 5090 + CPU DRAM prototype, but this experiment does not
yet establish its end-to-end speedup. Within each workload window, the 40 most-selected input slices per layer cover
80.2–84.0% of routed-row selections while occupying 17.75 GiB of packed expert storage. Forty-eight slices cover
85.4–88.7% and occupy 21.30 GiB. Code is the least concentrated workload.

The hot sets also move with the workload. At K=40, mean per-layer Jaccard overlap is 68.1% for chat versus math, 53.8%
for chat versus code, and 57.7% for math versus code. A single K=40 set selected over the combined trace covers only
75.1% of code selections, versus 80.2% for the code-local set. This supports an online, per-layer top-K policy rather
than a fixed global placement.

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

The successful run started at `2026-08-23T21:35:26Z`, completed at `22:05:39Z`, and has run ID
`20260823T213526Z`. The row `v100x4` / `611ccd6e5f6e` succeeded: 50/50 requests returned HTTP 200 and no workload
process remained after the cleanup trap. The repository base revision was `fca481253` with the tracking and MXFP4
changes applied in the working tree.

The supplied host ran Ubuntu 24.04.1 and kernel 6.8.0-124-generic with 334.4 GiB RAM, an Intel Xeon Platinum 8168,
four Tesla V100-SXM3-32GB GPUs, driver 580.159.03, CUDA 12.9.86, and Docker 29.5.3. The task-local image was
`sha256:3db6adc6528fce02cb0bc24afd59353f90677b0fd753461de9d8390f65e1da9d`; it contains Emmy on the supplied
Volta-compatible 1Cat vLLM base, vLLM `1.2.3.dev87+gd76126608.d20260810`.

The model loaded with 15.33 GiB per pipeline rank. During graph preparation the four processes used 28.6–29.7 GiB
each; PP0 later reported 11.88 GiB available for KV cache. Startup took about ten minutes, including 434 seconds for
model loading and cold Emmy compilation. gpt-oss attention sinks are not supported by the optimized Flash-V100 path,
so this trace explicitly allowed its Triton fallback. Latency from this run is not a serving-performance result.

## Usage histograms

Each histogram bins all 4,608 `(layer, input slice)` buffers by selection count divided by that layer's mean. Values
below 1× are colder than uniform routing for their layer. The first bin includes buffers with zero selections.

| Multiple of layer mean | Chat buffers | Math buffers | Code buffers |
| --- | ---: | ---: | ---: |
| 0–0.1× | 28.23% | 32.34% | 27.15% |
| 0.1–0.25× | 15.23% | 12.37% | 11.70% |
| 0.25–0.5× | 14.93% | 12.54% | 13.72% |
| 0.5–1× | 15.45% | 15.32% | 18.06% |
| 1–2× | 12.93% | 13.41% | 14.82% |
| 2–4× | 8.29% | 8.66% | 10.22% |
| 4–8× | 3.60% | 4.08% | 3.43% |
| ≥8× | 1.32% | 1.28% | 0.91% |

| Distribution metric | Chat | Math | Code |
| --- | ---: | ---: | ---: |
| Buffers selected at least once | 86.57% | 80.86% | 86.39% |
| Gini coefficient | 0.710 | 0.708 | 0.662 |
| Mean effective slices per layer | 49.77 | 50.38 | 58.02 |
| Mean slices needed for 80% of selections | 34.44 | 34.44 | 39.92 |
| Mean slices needed for 90% of selections | 50.83 | 49.67 | 56.14 |

The high Gini values and effective counts well below 128 reject the roughly-uniform-use hypothesis on these windows.
The large nonzero coverage also rejects a simpler conclusion that most buffers can remain permanently absent: many
slices are cold rather than unused.

### Ideal per-window top-K coverage

This table selects the best K slices independently in every layer and workload. It is an ideal LFU coverage bound,
not an observed cache policy.

| K per layer | Packed expert VRAM | GPU floor including other weights | Chat | Math | Code |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 7.10 GiB | 11.06 GiB | 59.44% | 58.35% | 53.11% |
| 24 | 10.65 GiB | 14.61 GiB | 70.09% | 69.72% | 64.86% |
| 32 | 14.20 GiB | 18.16 GiB | 77.89% | 77.93% | 73.55% |
| 40 | 17.75 GiB | 21.71 GiB | 83.68% | 84.03% | 80.18% |
| 48 | 21.30 GiB | 25.26 GiB | 88.14% | 88.68% | 85.42% |
| 56 | 24.85 GiB | 28.81 GiB | 91.54% | 92.25% | 89.51% |
| 64 | 28.40 GiB | 32.37 GiB | 94.09% | 94.88% | 92.68% |

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
| 16 | 55.0% | 47.3% | 57.7% |
| 32 | 65.6% | 51.7% | 57.3% |
| 40 | 68.1% | 53.8% | 57.7% |
| 48 | 70.8% | 57.6% | 60.9% |

A static K=40 set chosen from all three windows covers 81.89% of chat, 81.87% of math, and 75.08% of code. At K=48
the corresponding coverage is 86.21%, 86.58%, and 80.93%. The gap from the per-window bounds is direct evidence that
the placement should adapt as the workload changes.

## Trace and tuning qualification

The exact checkpoint trace produced 18 frontend graphs and 24 distinct post-fusion Loop targets. M=1, M=16, and
dynamic variants yielded 72 tuning targets. A four-GPU MCTS-only arm completed all 72 targets, produced 64 ranked
variants, and left eight without a ranking. Its database retained 514 successful and 11 failed performance rows. A
same-budget arm with one model-proposed configuration per target also completed all 72 targets; 48 proposals ranked
cleanly, four mapped ambiguously to multiple kernels, and 20 did not match a generated pin. Its database retained 380
successful and 18 failed performance rows. Candidate-specific failures were timeouts or unmatched input names rather
than trace failures.

The search used `-Xcicc -O1` to rank a small candidate budget. Representative finalists were then recompiled and
measured twice at deployable O3:

| Routed target | Greedy O3 | MCTS finalist | Proposed finalist | Result |
| --- | ---: | ---: | ---: | --- |
| M=1 gate/up | 136.92 µs | 513.54 µs | 161.88 µs | Greedy |
| M=1 down | 67.65 µs | 486.66 µs | 71.57 µs | Greedy |
| M=16 gate/up | 1,985.54 µs | 5,500.93 µs | 1,874.43 µs | Same `t128` family; variance |
| M=16 down | 907.78 µs | 3,916.29 µs | 849.41 µs | Same `t128` family; variance |

The O1 search ranking did not transfer reliably to O3, and the proposed M=16 pins reduced to the same `t128` family
as greedy. No searched result was promoted to a canonical golden. The raw working goldens, databases, online
checkpoints, logs, and all eight O3 measurements are retained as qualification evidence.

## Onboarding status

Native MXFP4 loading, tracing, compilation, four-stage serving, CUDA-graph-safe usage tracking, and a complete
50-request experiment are qualified on the supplied V100 host. Four failed launches are also retained: they exposed
and fixed dual FP8/MXFP4 spelling, capture-size assumptions, stage-local sink naming, and the Volta attention-sink
fallback requirement.

This is not yet a maintained public serving recipe. The successful image is local to the supplied host and was not
published because registry publication requires separate approval. The existing onboarding recipe remains untested
rather than claiming a reproducible public image or a canonical tuned golden.

## Archive and limitations

`results_v100x4.tar.gz` contains the successful timestamped directory, including the system-only experiment record,
server and client logs, exact prompts and responses, image inspection, full per-layer histogram JSON, trace inventory,
tuning databases and logs, O3 checks, and the four diagnostic serving attempts.

The sample has 16 prompts per workload, one fixed ordering, greedy 32-token completions, and no concurrency. All
responses reached the 32-token length cap and exposed no final message content; the server had no reasoning parser
enabled. This run validates execution and routing accounting, not answer quality. Histogram aggregation loses temporal
locality and cannot predict actual PCIe miss cost, overlap, prefetch efficiency, or cache churn. The V100
Triton attention fallback and untuned greedy expert programs also make request latency non-representative. The next
decision gate is an ordered-reference cache replay followed by the actual K=40/K=48 hybrid implementation on the
RTX 5090 + 64 GiB host.
