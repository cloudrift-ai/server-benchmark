# Golden bench 2026

This experiment suite supports the Emmy compiler submission. Raw measurements remain in benchmark run directories;
the repository holds only recipes, frozen inputs, and the scientific protocol. A recipe is not evidence until its
required artifacts exist and pass the gates below.

## Evidence sets

| Evidence set | Workload | Platforms | Permitted interpretation |
| --- | --- | --- | --- |
| Common kernel corpus | Qwen3-0.6B layer 0, sequence lengths 1 and 512 | V100, A100, RTX 4090, RTX 5090, H200, B200 | Identical, portable model-derived kernel comparison |
| Native FP8 kernel corpus | Qwen3-0.6B-FP8-dynamic layer 0, sequence lengths 1 and 512 | RTX 4090, RTX 5090, H200, B200 | Separate checkpoint-derived W8A8 comparison |
| Native FP8 large-layer shape stress | Qwen3-32B-FP8-dynamic layer 0, sequence lengths 1 and 512 | H200 and B200 | Supplemental dense W8A8 large-shape comparison |
| Large-layer shape stress | Qwen3.6-27B layers 0 and 3, sequence lengths 1 and 512 | H200 and B200 | Unsharded BF16 large-shape stress only |
| End-to-end serving | Pinned recipes below | Consumer single GPU; datacenter TP8 except the V100 TP8xPP2 lane | System performance for explicitly matched stock and Emmy arms |

These sets produce separate tables and separate geometric means. The unsharded large-layer corpus is not TP8,
quantization, or serving evidence and cannot explain an end-to-end result. The BF16 common corpus and native FP8
corpus have different hardware denominators and must never be pooled. B200 and A100 are stretch platforms; drop B200
first if access is limited.

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
weights and the declared per-token dynamic activation quantization into runnable graph algebra. The trace then retains only
post-fusion targets reading `f8e4m3`; the working YAML records the checkpoint-declaration digest. `EMMY_FP8_MMA=1`
enables the native W8A8 candidate, and every exact winner must contain
`mma.sync.aligned.m16n8k32` in a generated CUDA kernel. Source hashes and pattern verdicts are part of each replay.

This is a same-quantized-computation kernel comparison, not a BF16 quality study, native-framework FP8 serving
comparison, or proof that the small model represents datacenter-scale FP8 layers. Report it in a separate table and
geometric mean. The three `fp8-convergence` rows repeat only H200 sequence 512 at seeds 0, 1, and 2. A datacenter-scale
shape check comes only from the separate `fp8-large-layer` supplement: one dense Qwen3-32B layer at sequence lengths
1 and 512 on H200 and B200, with at most eight candidates and patience 3. It is reported separately from the portable
FP8 geometric mean and still is not an end-to-end FP8 serving result. The BF16 supplement cannot supply FP8 evidence.

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

The directly searched winner must match its measured knob map exactly. `emmy run --verify-working-golden` launches
five fresh processes at
deployable `-O3`, with 10 warmups and 100 measured iterations. It records the exact searched winner and deploy-path
Emmy timing and compares with eager PyTorch and Inductor. Inductor uses the installed PyTorch equivalent of
`mode="max-autotune"`: max autotune, coordinate-descent tuning, and CUDA graphs are all enabled before process start.
Inductor must compile the full graph and match eager output on the same inputs before its latency is accepted. Any
failed, ambiguous, unmatched, uncaptured, or non-whole-program winner fails after archiving diagnostics.

Report per-kernel latency distributions and a per-platform geometric mean over the identical common corpus. Report
the three-seed convergence distribution separately. Do not pool platforms or count the large-layer supplement in the
headline mean. Eager PyTorch supplies the framework/vendor-library reference; Inductor is the all-platform compiler
comparison. Hidet 0.6.1 with search space 2 is an independent compiler comparison on the H200 common corpus. It also
must compile the full graph and match eager output. Keep all traced targets in its denominator and report unsupported
or failed targets instead of dropping them.

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
common target. The verifier runs `--strict-correctness`: deploy Emmy and every exact O3 winner must match eager on the
same deterministic inputs at `rtol=atol=1e-3`, with max/mean/relative errors recorded. Otherwise that platform is
incomplete: publish its failures and coverage denominator, but publish no headline geometric mean or superiority
claim. Hidet is explicitly conditional coverage; report its successful/total
count and a common-support geometric mean without treating missing targets as wins. Descriptive win/tie/loss uses
`r > 1.02`, `0.98 <= r <= 1.02`, and `r < 0.98`.

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
the native RTX 5090 qualification; its gate requires an exact optimized NVFP4 GEMM selection and rejects Marlin and
emulation. The B200 GLM checkpoint uses W4A4 routed experts; its gate requires the exact vLLM selection line for an
allowed native NVFP4 MoE backend and rejects Marlin, emulation, unsupported-hardware, and fallback evidence. A failed
gate preserves the benchmark but invalidates its stated qualification. These stock lanes do not measure Emmy compiler
speedup and are not inputs to the protocol-only kernel lane.

## End-to-end matrix

| Platform | Recipe | Purpose | Claim status |
| --- | --- | --- | --- |
| RTX 4090 | Qwen3.6-27B AWQ, TP1 | Recent consumer qualification | Stock baseline until an Emmy arm exists |
| RTX 5090 | Gemma-4-12B-it, TP1 | Same-image stock and Emmy A/B | Primary matched-system result after semantic gating |
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
math is outside this preregistered suite; the exact `FP8_MMA` pin is used only in the separate same-quantized W8A8
kernel corpus.

The Gemma stock and Emmy arms use identical per-workload `--max-num-batched-tokens` settings. Their immutable images
also record the same vLLM source revision. The
[image provenance gate](serving_gemma4_rtx5090/IMAGE_PROVENANCE.md) is executable; a difference in scheduler settings
or base revision invalidates the A/B. The frozen
[output-equivalence gate](quality_gemma4_rtx5090/PROTOCOL.md) requires exact deterministic completions on all five
fresh stock and Emmy servers.

The Gemma delta supports a matched end-to-end serving-system speedup claim. Stock uses vLLM's native route while Emmy
uses `EmmyGenModel`, so this A/B does not isolate compiler kernels alone. A compiler-caused end-to-end claim requires
a same-`EmmyGenModel` reference-kernel or compiled-kernels-off arm. The remaining recipes qualify systems or
compatibility and deliberately fail to imply a compiler speedup by themselves.

### Gemma estimator and claim rule

Pair stock and Emmy by workload and `benchmark.process_repeat`, independent of their balanced execution order. The
primary metric is output-token throughput for `(256,256,64)`, median end-to-end latency for `(4096,4096,1)`,
output-token throughput for `(4096,4096,8)`, and median time to first token for `(8192,256,4)`. Other recorded
throughput, TTFT, TPOT, ITL, and latency fields are secondary diagnostics and cannot substitute for a primary metric.

For throughput, the paired improvement ratio is `Emmy / stock`; for latency it is `stock / Emmy`, so values above one
always favor Emmy. A point estimate is the median of its five paired ratios. Report all ratios, the median and range,
and a 10,000-draw seed-0 paired-repeat percentile bootstrap 95% interval. A point is "faster" only when the interval's
lower endpoint exceeds one. The equal-weight four-point summary is the geometric mean of the point medians, with a
10,000-draw seed-0 bootstrap that resamples the five pairs within each point. "Faster across the matrix" requires
all four points and the summary to meet the same lower-bound rule.

Every serving recipe enables the built-in request-completeness gate: every repeat must report
`successful_requests == num_prompts` and `failed_requests == 0`, and every output probe must pass. The equivalence
gate rejects a filtered partial Gemma matrix before provisioning. No performance outlier is removed. A
machine-readable deployment, client, or network failure before a complete metric may trigger one rerun of the entire
stock/Emmy pair for that workload/repeat; retain and disclose both failed originals. A second failure makes the point
incomplete. A semantic mismatch or post-metric performance anomaly is never a rerun reason; after a code/configuration
fix, restart the entire 40-task matrix under a new source ID.

## Publication gates

- Every kernel result has five exact fresh-process `-O3` replays and passes correctness/integrity checks.
- Kernel artifacts include the pinned positional model, clean staged-source ID, package freeze, online checkpoint,
  and GPU/software environment; a provenance mismatch invalidates the result.
- The common-corpus table includes all traced targets or reports every unsupported target in the denominator.
- The FP8 table is separate, includes only the predeclared `f8e4m3`-input target inventory, records the checkpoint
  declaration digest, and requires native W8A8 instruction evidence in every exact winner.
- The FP8 large-layer table is supplemental and includes all retained targets from the one pinned Qwen3-32B layer;
  it is not pooled into the portable FP8 geometric mean.
- The H200 Hidet table uses the identical common targets and counts every missing/failed target against coverage.
- The H200 convergence diagnostic reports all three seeds, including failures.
- Every serving image is resolved to a digest in the evidence manifest; the private V100 recipe tag must be resolved
  on the authorized host before publication.
- The suite makes no TP8 kernel claim; datacenter kernel results retain the stated unsharded corpus boundary.
- End-to-end speedup claims require matched hardware, model revision, engine revision, workload, and stock/Emmy arms.
- The Gemma system table additionally requires matched scheduler settings, image provenance, and exact frozen-prompt
  outputs; it is not labeled compiler-caused without a same-route reference arm.
- Native end-to-end quantization claims require the built-in server-log gate to name the exact method and backend and
  reject fallback paths; a recipe without a passing gate is planned evidence only.
- Datacenter claims remain per-system. Results are not generalized from TP8 to TP4, or across GPU generations.
