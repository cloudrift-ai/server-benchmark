# gpt-oss-120b routed input-slice histogram on 4× V100 SXM3

## Result

The corrected native-MXFP4 serving path shows substantial but workload-dependent routing skew. The best 40 input
slices in each layer cover 81.08% of chat, 79.53% of math, and 76.72% of code selections while occupying 17.75 GiB
of packed expert storage. K=48 covers 85.87%, 84.86%, and 82.32% at 21.30 GiB. The mean per-layer K=40 hot-set
Jaccard overlap falls to 47.15% between chat and code.

This supports building an adaptive K=40 prototype on the RTX 5090 with cold routed slices in CPU DRAM. K=40 leaves a
10.29 GiB GPU budget after the other checkpoint tensors and needs 39.06 GiB of host memory for the cold routed
slices. K=48 leaves only 6.73 GiB on the GPU. These are placement floors before workspaces, CUDA graphs, and KV cache,
so K=40 is the safer first end-to-end configuration.

This run does not measure an actual cache hit rate or transfer cost. The counter records frequency but not reference
order, and no weights moved between GPU and CPU during the experiment. The result justifies the prototype; it does
not establish its latency or throughput.

## Protocol

`emmy bench experiments/gpt-oss-120b/routing_histogram_v100` ran the exact native-MXFP4
`openai/gpt-oss-120b` checkpoint at revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`. This is the checkpoint
intended for the hybrid deployment, not a separately requantized derivative. OpenAI describes gpt-oss-120b as a
mixture-of-experts model whose MXFP4 weights fit on one 80 GB GPU
([model announcement](https://openai.com/index/introducing-gpt-oss/)).

The server used pipeline parallelism across four V100s. Emmy kept routed matrices in their native uint8 blocks and
uint8 E8M0 scales, decoded them inside compiled programs, and updated one cumulative GPU counter for every
`(layer, routed input slice)` selection. Counter snapshots were emitted before uncaptured forwards; CUDA-graph decode
replays updated the same counters on-device.

The workload image's one-time CUDA-graph capture execution was present in every cumulative snapshot and therefore
canceled from every workload delta. The final implementation clears that capture baseline before its first snapshot;
a V100 regression separately verifies that subsequent outer-graph replays still increment the device counter.

After a delimiter probe, the client sent the first 16 rows from each fixed dataset, sequentially and greedily:

| Window | Dataset | Requests | Forward rows per layer | Routed selections across 36 layers |
| --- | --- | ---: | ---: | ---: |
| Chat | `HuggingFaceH4/ultrachat_200k`, `train_sft` | 16 | 4,541 | 653,904 |
| Math | `openai/gsm8k`, `test` | 16 | 2,719 | 391,536 |
| Code | `openai/openai_humaneval`, `test` | 16 | 3,236 | 465,984 |

Every request allowed 32 completion tokens at temperature 0 and returned token IDs. The final probe caused the
complete code-window snapshot to be logged. Snapshot subtraction matched
`4 * (response.usage.total_tokens + 1)` at all 36 layers and all four workload boundaries.

The response audit found 50/50 HTTP 200 responses, 50 finite JSON records, and 1,600 token IDs matching reported
completion usage exactly. The IDs contained 577 distinct values and no zero tokens. Every response reached the
32-token cap and contained content, reasoning, or a tool call; this is a structural check, not a semantic evaluation
of all 48 workload answers. The separate qualification probes below provide the generation-correctness gate.
An independent raw recomputation matched every counter delta, histogram bin, distribution metric, top-K result,
combined-static result, and Jaccard value in the derived JSON without discrepancy.

## Run and platform

The successful run started at `2026-08-24T10:31:35Z`, completed at `2026-08-24T11:01:52Z`, and has run ID
`20260824T103135Z`. The `v100x4` / `611ccd6e5f6e` row succeeded in 1,813.48 seconds. An initial attempt was excluded
after its SSH transport timed out during the workload; the retained recipe emits a heartbeat, and the clean rerun
completed without a transport, command, or cleanup error. The supplied host was left running.

The experiment record names revision `fc62d470`, after the compiler and serving changes had been rebased onto main
revision `bfb9150c`. Its dirty flag covers the retained recipe refresh and onboarding text committed with this result.
Later rebases onto main revisions `765959b5`, `c39d405c`, and `4db44ff1` each produced a byte-identical trace, so the
compiler boundaries and retained histogram recipe are unchanged. The final feature branch is based on main revision
`7921d7fe`. Main revision `5642d020` merged the extracted rider-dtype and Boolean A/B replay fixes already present
byte-for-byte in the qualification image. The later whole-step capture changes leave MoE capture on its existing
single-token decode branch, and this recipe disables the prefill twin, so they do not change the qualified serving
path. The retained qualification trace is 495,244 bytes and also matches all target identities and realization rows.
Grouped-placement preservation and the one-warmup terminal benchmark landed after this V100 evidence freeze; the
latest branch passes the repository suite, but this report does not attribute the older V100 timings to those changes.

The host ran Ubuntu 24.04.1 and kernel 6.8.0-124-generic with 334.4 GiB RAM, an Intel Xeon Platinum 8168, four Tesla
V100-SXM3-32GB GPUs, driver 580.159.03, CUDA 12.9.86, and Docker 29.5.3. The final corrected task-local image was
`sha256:69603c33aa9bc096a4cb2fb0a23cc58e8497968474cbf42a5a84c6235025fd1f`; it contains Emmy on the supplied
Volta-compatible 1Cat vLLM base, vLLM `1.2.3.dev87+gd76126608.d20260810`. The histogram workload itself used the
earlier corrected diagnostic image `sha256:d34833c60f2a0c757b260381564378b1b7ca2981bdef193ee6c8b06a6d29c9ac`.

## Usage histograms

Each histogram bins all 4,608 `(layer, input slice)` buffers by selection count divided by that layer's mean. Values
below 1× are colder than uniform routing for their layer. The first bin includes buffers with zero selections.

| Multiple of layer mean | Chat buffers | Math buffers | Code buffers |
| --- | ---: | ---: | ---: |
| 0–0.1× | 22.63% | 25.02% | 21.59% |
| 0.1–0.25× | 15.65% | 12.67% | 12.15% |
| 0.25–0.5× | 17.10% | 14.41% | 14.97% |
| 0.5–1× | 17.08% | 17.38% | 19.88% |
| 1–2× | 14.15% | 16.62% | 18.19% |
| 2–4× | 8.68% | 9.48% | 9.38% |
| 4–8× | 3.45% | 3.32% | 3.02% |
| ≥8× | 1.26% | 1.09% | 0.82% |

| Distribution metric | Chat | Math | Code |
| --- | ---: | ---: | ---: |
| Buffers selected at least once | 92.49% | 89.95% | 93.38% |
| Gini coefficient | 0.670 | 0.650 | 0.619 |
| Mean effective slices per layer | 55.13 | 58.58 | 63.63 |
| Mean slices needed for 80% of selections | 38.64 | 40.86 | 44.72 |
| Mean slices needed for 90% of selections | 56.64 | 57.86 | 62.28 |

The effective counts are far below 128 and the Gini values reject roughly uniform use in these windows. The high
nonzero coverage also rejects permanent removal of most slices: many are cold rather than unused.

### Ideal per-window top-K coverage

This table selects the best K slices independently in every layer and workload. It is an ideal LFU coverage bound,
not an observed cache policy.

| K per layer | Packed expert VRAM | GPU floor including other weights | Chat | Math | Code |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 7.10 GiB | 11.06 GiB | 55.98% | 52.46% | 49.57% |
| 24 | 10.65 GiB | 14.61 GiB | 67.04% | 64.06% | 60.90% |
| 32 | 14.20 GiB | 18.16 GiB | 75.01% | 72.78% | 69.68% |
| 40 | 17.75 GiB | 21.71 GiB | 81.08% | 79.53% | 76.72% |
| 48 | 21.30 GiB | 25.27 GiB | 85.87% | 84.86% | 82.32% |
| 56 | 24.85 GiB | 28.82 GiB | 89.58% | 89.04% | 86.90% |
| 64 | 28.40 GiB | 32.37 GiB | 92.45% | 92.34% | 90.60% |

One native-MXFP4 `(layer, input slice)` allocation is 13,236,480 bytes: gate/up and down blocks, their E8M0 scales,
and both biases. All 4,608 routed allocations occupy 56.81 GiB; the remaining checkpoint tensors occupy 3.96 GiB,
for 60.77 GiB total.

K=64 is impossible on a 32 GiB RTX 5090 before runtime state because its 32.37 GiB floor exceeds card capacity. K=48
leaves 6.73 GiB for workspaces, CUDA graphs, and KV cache; K=40 leaves 10.29 GiB. If hot slices are not duplicated in
host memory, the cold routed store requires 39.06 GiB at K=40 or 35.51 GiB at K=48, leaving 24.94 or 28.49 GiB of the
64 GiB host budget for the operating system, staging, metadata, and runtime allocations.

### Workload movement

| K | Chat–math Jaccard | Chat–code Jaccard | Math–code Jaccard |
| ---: | ---: | ---: | ---: |
| 16 | 49.0% | 39.0% | 47.5% |
| 32 | 57.9% | 43.6% | 50.2% |
| 40 | 60.6% | 47.1% | 52.2% |
| 48 | 64.1% | 50.9% | 54.5% |

A static K=40 set selected over all three windows covers 78.48% of chat, 76.16% of math, and only 69.84% of code,
versus 81.08%, 79.53%, and 76.72% for the window-local sets. At K=48, the static set covers 83.35%, 81.67%, and
75.78%, versus local coverage of 85.87%, 84.86%, and 82.32%. The movement supports an online per-layer placement
policy rather than one fixed set.

## Trace and tuning qualification

The exact checkpoint trace on main revision `4db44ff1` completed in 26.67 seconds and produced 18 frontend graphs,
24 distinct post-fusion Loop targets, and 72 M=1, M=16, and dynamic realizations. Its SHA-256 is
`63d28a6b5cbbdd054fbfe48cabc8f2565eda48eb63e2b65cc72ecb411e08e151`, byte-identical to the trace taken before
the rebase. A source-hash receipt proves runtime byte identity after the two extracted fixes merged in main revision
`5642d020`. The later whole-step capture changes are inactive for this MoE recipe, which retains single-token decode
capture and disables the prefill twin. The subsequently merged grouped-placement and terminal-warmup changes are not
part of this retained tuning result. This trace and serving boundary is why the histogram recipe did not need a
new workload run.

Both official tuning arms started with empty DB, online-checkpoint, and compiled-kernel paths. Each used all four
V100s, seed 0, at most 40 candidates per target, patience 12, and the O1 ranking lane:

| Arm | Completed targets | Wall time | Successful O1 rows | Failed candidate rows | Usable winner rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Proposal-seeded | 72/72 | 3,929 s | 2,451 | 56 | 29 |
| MCTS-only | 72/72 | 3,420 s | 2,451 | 53 | 26 |

Candidate failures were bounded slow-kernel or benchmark failures; neither arm lost a target. The two arms had the
same complete set of 72 base identities. The proposal-seeded working golden retained 101 ranked rows after its 72
inventory rows: 52/72 proposals matched a realized pin and 20 did not. Under the latest structural admission, no
proposal remained an arm winner and search contributed 29 winners, down from 32 on `c39d405c` and 51 on `765959b5`.
MCTS-only retained 26 winners, unchanged from `c39d405c` and down from 47 on `765959b5`.

Deployable O3 replay deduplicated the two arms and proposals to 84 candidates over 72 targets. Two independent fresh-
cache rounds used 10 warmups and 100 iterations. Each completed in 740 seconds and produced exactly 72 JSON documents
and 72 status records with no process timeout; the manifests, candidate inventories, and source hashes match across
rounds. Both rounds had 47 successful target commands and 25 expected strict- or benchmark-failure aggregates. Exact
candidate pins, realized knobs, same-input proof, and source identity passed twice for 67/84 candidates.

The conservative correctness, source-stability, repeat-at-most-20%, and paired-at-most-5%-regression gates left 26
candidates across 18/72 targets. Requiring all three realizations of a serving configuration left nine rows across
three complete configurations. Their selected median paired candidate/reference ratio was 0.9959; repeat delta had
median 0.106% and maximum 2.42%. Selected origins overlap because the same pin can appear in more than one arm:
three proposal rows, six proposal-seeded search rows, and three MCTS-only search rows.

The partial golden's structural and offer audit recognized all nine leaves, but the authoritative GPU serving-coverage
gate reported four matches, no drift, and 126 reachable gaps, then rejected it. A partial golden would silently enter
the verified tier without covering the deployed graph, so no canonical V100 golden was promoted or installed. The
partial file remains evidence only inside the archive; the recipe has no `golden/` directory and makes no fresh-clone
tuned-performance claim.

The O3 audit also exposed an A/B replay defect: registered Boolean pins such as `FAST_MATH=False` were discarded with
render-only Boolean markers. Explicit A/B rows now keep Boolean input pins separate from schedule pins, preserve false
values in JSON, and report the lane realized by the compiled graph. Focused tests cover ordinary and embedded-IR
replay. This fix prevents a nominal standard-precision row from measuring a different input regime.

## Generation correctness

The earlier image produced repeated token ID 0 and NaN logprobs. Layer-by-layer diagnosis found the fp16 residual
stream reaching about 65,000 by layer 34 and overflowing at the next residual add. Native-MXFP4 gpt-oss now keeps the
embedding and inter-layer residual in fp32 while retaining fp16 projection, router, and expert activations. The pack
identity includes `gpt-oss-mxfp4-fp32-residual-v1`, so stale compiled programs cannot survive the precision change.

The same investigation fixed an integer-dtype loss in readable expression inlining. An inlined shift feeding an
MXFP4 bit mask could otherwise be rendered as a logical rather than integer operation. The regression test asserts
the exact integer shift-and-mask spelling.

The eager/reference loaders had a separate native-MXFP4 correctness defect. Whole-model loading did not decode
expert block/scale pairs into the architecture twin, leaving its logical expert values at initialization, while the
selected-layer reference left those expert parameters on the meta device. The loader now decodes the logical values,
restores the checkpoint's gate/up interleaving for the eager architecture, and attaches selected-layer values
strictly. It also makes the decoded down projection contiguous, matching the architecture parameter layout rather
than preserving the decoder's transposed NumPy stride. Hermetic checkpoint tests compare every affected tensor
exactly and assert the selected parameters' layout.

Two fresh CPU-only audits loaded layer 0 from the real checkpoint under a 48 GiB memory cap, with networking disabled,
no GPU devices, and a read-only model cache. Both opened only shard 9 of 14 and completed in 54 and 53 seconds. All
four expert parameters had their exact expected shapes, were contiguous CPU fp16 tensors, and contained only finite,
nontrivial values. Fixed 4,096-value samples matched across the two runs; peak process RSS was 36.81 GiB. This
defect was in validation references; the compressed serving path already loaded the packed expert store directly.

The final image passed six independent API probes:

- 6/6 HTTP 200 responses and no server-side OOM, NaN, or non-finite report;
- nonzero raw token IDs `[200005, 35644, 200008, 976, 1825, 5003, 25, 392]`;
- two deterministic plain responses whose decoded content was exactly `ready`;
- finite token logprobs;
- the exact tool call `get_weather({"location":"Paris, France"})`; and
- exact recall of `ZEPHYR-48291` from a 2,695-token context.

The corrected 50-request histogram run independently produced 1,600 nonzero token IDs and no runtime error. It
therefore measures the corrected routing path rather than the old token-0 failure mode.

## Serving performance and native vLLM boundary

The model loaded 15.33 GiB of weights per pipeline rank. Ready-state GPU memory was 29.03, 27.95, 27.95, and 29.02 GiB
across the four 32 GiB V100s, and PP0 reported 11.88 GiB available for KV cache. Cold startup took 659 seconds,
including 483.87 seconds for model loading and fresh Emmy compilation. gpt-oss attention sinks are unsupported by the
optimized Flash-V100 path, so the run explicitly used its Triton fallback. The expert programs remained on the
measured greedy/task-local evidence path because no canonical golden passed the full-coverage gate.

`vllm bench serve` used the OpenAI chat endpoint, a random 128-input/32-output-token dataset, 16 measured requests,
two warmups, temperature 0, ignored EOS, and concurrency 1:

| Metric | Corrected Emmy image |
| --- | ---: |
| Successful requests | 16/16 |
| Request throughput | 0.0426 req/s |
| Output-token throughput | 1.362 tok/s |
| Total-token throughput | 9.660 tok/s |
| Mean / median / p99 TTFT | 22.223 / 22.194 / 22.963 s |
| Mean TPOT | 41.187 ms/token |
| Mean ITL | 44.026 ms |

Mean TPOT corresponds to about 24.28 decode tokens/s after the first token. The much lower end-to-end output rate is
dominated by roughly 22-second TTFT. Against the prior corrected Emmy repeat, output throughput changed by -0.10%,
mean TTFT by +0.11%, and mean TPOT by -0.07%. These are absolute task-local numbers, not a stock-vLLM speedup claim.

The same exact checkpoint and revision were then started through the native vLLM codepath in the task-local custom
SM70 base image. vLLM `1.2.3.dev87+gd76126608.d20260810` rejected `gpt_oss_mxfp4` during configuration because it
requires compute capability 8.0 and V100 is 7.0. It did not start workers, load weights, or serve a request. The valid
comparison on this hardware is therefore a support boundary: Emmy serves the checkpoint on 4× V100 while native
vLLM rejects it. No stock-vLLM throughput, latency ratio, or output-parity result exists on V100.

## Archive and limitations

`results_v100x4.tar.gz` contains only the successful timestamped run. It includes the system-only experiment record,
exact prompts and responses, image inspection, server logs, full per-layer histogram JSON, byte-identical trace,
equal-budget tuning DBs and working goldens, both O3 replay rounds, failed partial-promotion audits, final image build,
correctness probes, serving benchmark, and native-vLLM rejection evidence. Compiled cubins and exploratory scripts are
excluded.

The routing sample has 16 prompts per workload, one fixed ordering, greedy 32-token completions, no concurrency, and
no repeated or randomized run. Category prompt lengths differ. The combined-static ranking weights each window by
its routed-row volume rather than giving the three workloads equal weight. All responses reached the output cap, so
their generated continuations influence the counters. Frequency aggregation loses temporal locality and cannot
predict PCIe miss cost, overlap, prefetch efficiency, or churn. The next experiment must retain ordered references
and run the actual K=40/K=48 cache on the RTX 5090 with 64 GiB host RAM.

This is still an onboarding recipe, not a maintained public serving recipe. The diagnostic image is local to the
supplied host and was not published because registry publication requires separate approval. A full-coverage
canonical V100 golden and an RTX 5090 hybrid deployment remain open work.
