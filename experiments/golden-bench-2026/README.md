# Golden bench 2026

This experiment suite supports the Emmy compiler submission. Raw measurements remain in benchmark run directories;
the repository holds only recipes, frozen inputs, and the scientific protocol. A recipe is not evidence until its
required artifacts exist and an intelligent reviewer accepts them against the checklist below.

## Evidence sets

| Evidence set | Workload | Platforms | Permitted interpretation |
| --- | --- | --- | --- |
| Common kernel corpus | Qwen3-0.6B layer 0, sequence lengths 1 and 512 | V100, A100, RTX 4090, RTX 5090, H200, B200 | Identical, portable model-derived kernel comparison |
| Dynamic-FP8 checkpoint layer | Qwen3-0.6B-FP8-dynamic layer 0, sequence lengths 1 and 512 | RTX 4090, RTX 5090, H200, B200 | Complete layer inventory; W8A8-only claim deferred |
| Dynamic-FP8 large-layer trace | Qwen3-32B-FP8-dynamic layer 0, sequence lengths 1 and 512 | H200 and B200 | Complete large-layer inventory; W8A8-only claim deferred |
| Large-layer shape stress | Qwen3.6-27B layers 0 and 3, sequence lengths 1 and 512 | H200 and B200 | Unsharded BF16 large-shape stress only |
| End-to-end serving | Pinned recipes below | Consumer single GPU; datacenter TP8 except the V100 TP8xPP2 lane | System performance for explicitly matched stock and Emmy arms |

The BF16 sets produce separate tables and separate geometric means. The unsharded large-layer corpus is not TP8,
quantization, or serving evidence and cannot explain an end-to-end result. Dynamic-FP8 layer traces are preserved as
complete inventories but do not support a W8A8-only table or geometric mean until a separate golden-filtering tool
freezes that denominator. B200 and A100 are stretch platforms; drop B200 first if access is limited.

## Kernel methodology

The single `kernels` recipe uses its `study` matrix field to keep the evidence and diagnostic groups distinct. Its
`common` tasks run
the same layer-0 targets at static sequence lengths 1 and 512 on every platform. The two shapes approximate decode
and ordinary prefill while keeping the initial search bounded. Search uses at most 12 candidates per kernel,
patience 4, and seed 0. Every kernel task requests and tunes on one GPU; multi-GPU allocation is not part of the
kernel claim. The `convergence` tasks repeat only the sequence-512 family at seeds 0, 1, and 2; this is a
search-stability diagnostic and adds no workload to the headline geometric mean.

The `fp8-common` tasks trace the exact `RedHatAI/Qwen3-0.6B-FP8-dynamic` revision at the same layer and sequence
lengths on the four GPUs with native FP8 tensor cores. Checkpoint ingestion spells both the declared per-channel FP8
weights and the declared per-token dynamic activation quantization into runnable graph algebra. The working YAML
records the checkpoint-declaration digest but intentionally retains every post-fusion layer target. `EMMY_FP8_MMA=1`
offers the native W8A8 candidate where applicable.

These tasks are complete-layer engineering evidence, not yet a W8A8-only compiler comparison. Do not compute an FP8
geometric mean or require native FP8 instructions from unrelated targets. A later reusable golden-filtering tool must
select targets by stable graph properties, freeze the selected denominator, and preserve unsupported targets before
the paper admits a W8A8 table. The three `fp8-convergence` rows and the Qwen3-32B `fp8-large-layer` supplement have
the same limitation; they do not currently support W8A8-only stability or large-shape claims.

The `large-layer` tasks create H200 and B200 variants that trace Qwen3.6-27B layers 0 and 3 at the pinned revision.
Layer 0 represents the 48 of 64 linear-attention layers; layer 3 represents the 16 of 64 full-attention layers. Both
use sequence lengths 1 and 512, at most eight candidates, and patience 3. This supplement tests large GEMM and
attention shapes without multiplying the main corpus. Its post-fusion targets are selected by provenance from the
exact full-layer trace and re-resolved in that fusion context; they are not standalone Loop IR artifacts. A target
that cannot retain a runnable eager/Inductor reference fails the task and makes the supplement incomplete.

Each search has an isolated tuning database, online checkpoint, and cubin cache. The positional trace input and model
provenance use the same full revision. `emmy bench` requires a clean staged source tree and records its Git revision,
content-addressed file manifest, package freeze, GPU UUID/state, driver, CUDA compiler, command status, online
checkpoint, and raw tuning artifacts. Every matrix row is a separate task, so a failed case preserves its partial
evidence and does not prevent later rows from running.

The directly searched winner must match its measured knob map exactly. The recipe invokes
`emmy run --golden working.yaml --strict` five times at deployable `-O3`, with 10 warmups and 100 measured iterations.
Each ordinary invocation is an independent process; the CLI contains no repetition or child-process wrapper. It records the exact searched winner and deploy-path
Emmy timing and compares with eager PyTorch and Inductor. Inductor uses the installed PyTorch equivalent of
`mode="max-autotune"` with `fullgraph=True`.
Inductor must compile the full graph and match eager output on the same inputs before its latency is accepted. Any
failed, ambiguous, unmatched, uncaptured, or non-whole-program winner fails after archiving diagnostics.

Report per-kernel latency distributions and a per-platform geometric mean over the identical common corpus. Report
the three-seed convergence distribution separately. Do not pool platforms or count the large-layer supplement in the
headline mean. Eager PyTorch supplies the framework/vendor-library reference; Inductor is the compiler comparison.

The `search-ablation` tasks execute the only search ablation: cold deploy greedy (budget zero), bounded budgets 4 and
12, and budget 48 with patience 12, all on the sequence-512 common family. Budget zero runs before tuning in fresh
processes with empty local DB, online-checkpoint, and cubin state. Its manifest is separate from searched
`winner_total_us`; no post-search result supplies the zero-budget value. Do not claim a prior, lowering, or hardware
ablation without a corresponding executable recipe.

### Kernel estimator and claim rule

For target `t`, process repeat `j`, and baseline `b`, define the paired speedup as
`r[t,j,b] = latency[t,j,b] / winner_total_us[t,j]`; values above one favor Emmy. The backend latency estimate reported
beside each target is the median of its five fresh-process latencies, and the target speedup is the median of the five
paired ratios. The per-platform headline is `exp(mean(log(r[t,b])))` over the fixed common corpus, with each target
weighted once. Do not pool platforms or weight by runtime.

Eager, Inductor, and the exact Emmy winner must succeed with the frozen timing/correctness semantics for every traced
common target. `run --strict` requires deploy Emmy and every exact O3 winner to match eager on the
same deterministic inputs at `rtol=atol=1e-3`, with max/mean/relative errors recorded. Otherwise that platform is
incomplete: publish its failures and coverage denominator, but publish no headline geometric mean or superiority
claim. Descriptive win/tie/loss uses `r > 1.02`, `0.98 <= r <= 1.02`, and `r < 0.98`.

Report all five paired ratios, the target median and range, and a two-stage percentile bootstrap interval for the
platform geometric mean: resample targets, then the five paired process repeats within each sampled target, for
10,000 draws with seed 0. The phrase "faster than baseline" requires full required-backend support and a two-sided
95% interval whose lower endpoint is above one. Never delete an outlier or rerun based on latency.

## Datacenter kernel claim boundary

Configuration arithmetic cannot recover engine layouts, per-rank sharding, quantization backends, expert routing, or
the actual hot-kernel mix. Therefore `--serving-twins` is not used as TP8 serving evidence. This suite makes only the
unsharded common-corpus and large-shape kernel claims above; the 8-GPU recipes are end-to-end system qualification.
A later TP8 kernel study must first add runtime capture as a reusable Emmy CLI feature and independently reconcile
decode and prefill coverage. It must not synthesize serving shapes from model configuration alone.

## NVFP4 kernel admission lane

The NVFP4 kernel lane is protocol-only until Emmy can ingest the exact ModelOpt NVFP4 weight, scale, and activation
semantics as runnable graph algebra. It produces no compiler timing today. Admission requires one pinned
`nvidia/Qwen3-8B-NVFP4@ccd10a893cbca613259517c3efe08e151ddf2b8e` layer at sequence lengths 1 and 512 on RTX 5090
and B200, a denominator fixed from graph
properties before tuning, correctness against the same quantized computation, a separate BF16 quality delta, and a
native vendor or framework kernel baseline. Every accepted winner must preserve compressed storage, reject decoded
BF16/Marlin fallback, and record both native FP4 instruction evidence and proof that the timed launch executed that
cubin. Only the same-model RTX 5090/B200 set may support a cross-platform NVFP4 kernel summary.

The executable end-to-end NVFP4-checkpoint recipes are narrower system qualifications. The pinned Qwen3.6
checkpoint is mixed precision: FP8 attention and W4A16 NVFP4 MLP projections. Its exact vLLM route deliberately uses
`MarlinNvFp4LinearKernel`, so the RTX 5090 result is labeled W4A16-NVFP4/Marlin compatibility and throughput, not
native Blackwell FP4-MMA evidence. The separate Qwen3-8B checkpoint declares W4A4 for every transformer Linear and is
the native RTX 5090 qualification. Intelligent review must confirm an exact optimized NVFP4 GEMM selection and reject
Marlin and emulation. The B200 GLM checkpoint uses W4A4 routed experts. Review of the complete run logs must confirm
the exact optimized RTX 5090 GEMM and native B200 NVFP4 MoE selections and reject Marlin, emulation,
unsupported-hardware, or fallback evidence where those paths contradict the lane's claim. These stock lanes do not
measure Emmy compiler speedup and are not inputs to the protocol-only kernel lane.

## End-to-end matrix

| Platform | Recipe | Purpose | Claim status |
| --- | --- | --- | --- |
| RTX 4090 | Qwen3.6-27B AWQ, TP1 | Recent consumer qualification | Stock baseline until an Emmy arm exists |
| RTX 5090 | Gemma-4-12B-it, TP1 | Same-image stock and Emmy A/B | Primary matched-system result after semantic review |
| RTX 5090 | Qwen3.6-27B mixed FP8/W4A16-NVFP4, TP1 | Requested quantized checkpoint | W4A16/Marlin compatibility and throughput only |
| RTX 5090 | Qwen3-8B NVFP4, TP1 | Native W4A4 consumer qualification | Stock capability result until an Emmy arm exists |
| 16x V100 | DeepSeek-V4-Flash-0731, TP8xPP2 | New checkpoint on the proven SM70 serving path | Portability result until a matched stock arm exists |
| 8x A100 | DeepSeek-V4-Flash-0731 EXL3 3.04 bpw, TP8 | New checkpoint on an older serving platform | Stretch compatibility/refusal study; requires an Emmy arm |
| 8x H200 | GLM-5.2 FP8, TP8 | Primary datacenter serving system | Stock qualification until an Emmy arm and TP8 manifest exist |
| 8x B200 | GLM-5.2 NVFP4, TP8 with expert parallelism | Same architecture on Blackwell | Optional stock qualification until matched evidence exists |

All serving points disable prefix caching and use seed 0, temperature 0, and ignored EOS. Each point expands to five
tasks with `benchmark.repeats: 1`, so every observation receives a fresh deployed server instead of five clients
against one process. Preserve latency, time to first token, inter-token latency, throughput, engine logs, image
digests, driver/CUDA state, and failures. A compatibility fallback is not native NVFP4 or EXL3 evidence. General fast
math is outside this preregistered suite; the exact `FP8_MMA` pin is confined to the dynamic-FP8 checkpoint layer
traces and does not establish a W8A8-only result without the deferred target filter.

The Gemma stock and Emmy arms use identical per-workload `--max-num-batched-tokens` settings. Their immutable images
also record the same vLLM source revision. The
[image provenance contract](serving_gemma4_rtx5090/IMAGE_PROVENANCE.md) is checked against the recipe before a run; an
intelligent reviewer rejects the A/B if the final evidence shows different scheduler settings, runtime revisions, or
model semantics. `emmy bench` does not compare outputs or make this scientific decision.

The Gemma delta supports a matched end-to-end serving-system speedup claim. Stock uses vLLM's native route while Emmy
uses `EmmyGenModel`, so this A/B does not isolate compiler kernels alone. A compiler-caused end-to-end claim requires
a same-`EmmyGenModel` reference-kernel or compiled-kernels-off arm. The remaining recipes qualify systems or
compatibility and deliberately fail to imply a compiler speedup by themselves.

### Gemma estimator and claim rule

Pair stock and Emmy by workload and the neutral matrix label `repeat`, independent of their balanced execution order. The
primary metric is output-token throughput for `(256,256,64)`, median end-to-end latency for `(4096,4096,1)`,
output-token throughput for `(4096,4096,8)`, and median time to first token for `(8192,256,4)`. Other recorded
throughput, TTFT, TPOT, ITL, and latency fields are secondary diagnostics and cannot substitute for a primary metric.

For throughput, the paired improvement ratio is `Emmy / stock`; for latency it is `stock / Emmy`, so values above one
always favor Emmy. A point estimate is the median of its five paired ratios. Report all ratios, the median and range,
and a 10,000-draw seed-0 paired-repeat percentile bootstrap 95% interval. A point is "faster" only when the interval's
lower endpoint exceeds one. The equal-weight four-point summary is the geometric mean of the point medians, with a
10,000-draw seed-0 bootstrap that resamples the five pairs within each point. "Faster across the matrix" requires
all four points and the summary to meet the same lower-bound rule.

For every serving task, intelligent review must confirm `successful_requests == num_prompts`, `failed_requests == 0`,
the complete preregistered matrix, the intended backend from the raw logs, and plausible outputs and metrics. `emmy
bench` records these facts but does not accept or reject them. No performance outlier is removed. A machine-readable
deployment, client, or network failure before a complete metric may trigger one rerun of the entire stock/Emmy pair
for that workload/repeat; retain and disclose both failed originals. A second failure makes the point incomplete. A
semantic mismatch or post-metric performance anomaly is never a rerun reason; after a code/configuration fix, restart
the entire 40-task matrix under a new source ID.

## Intelligent publication review

- Every kernel result has five exact fresh-process `-O3` replays and passes correctness/integrity checks.
- Kernel artifacts include the pinned positional model, clean staged-source ID, package freeze, online checkpoint,
  and GPU/software environment; a provenance mismatch invalidates the result.
- The common-corpus table includes all traced targets or reports every unsupported target in the denominator.
- The dynamic-FP8 runs retain the complete layer inventory and checkpoint-declaration digest. They produce no
  W8A8-only table, geometric mean, or convergence claim until a separate tool freezes the eligible target set.
- The H200 convergence diagnostic reports all three seeds, including failures.
- Every serving image is resolved to a digest in the evidence manifest; the private V100 recipe tag must be resolved
  on the authorized host before publication.
- The suite makes no TP8 kernel claim; datacenter kernel results retain the stated unsharded corpus boundary.
- End-to-end speedup claims require matched hardware, model revision, engine revision, workload, and stock/Emmy arms.
- The Gemma system table additionally requires matched scheduler settings, image provenance, and a documented
  intelligent semantic comparison; it is not labeled compiler-caused without a same-route reference arm.
- Native end-to-end quantization claims require the reviewer to identify the exact method and backend in raw logs and
  reject fallback paths; a recipe alone is planned evidence only.
- Datacenter claims remain per-system. Results are not generalized from TP8 to TP4, or across GPU generations.
