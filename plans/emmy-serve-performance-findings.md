# Why `emmy serve` can be slower than stock vLLM

Recorded 2026-07-28 from the current serving implementation and benchmark
findings in this repository.

## Executive summary

The diagnosis depends on the serving mode:

- **Embeddings:** a large high-concurrency slowdown is currently expected.
  Emmy processes sequences individually by default and does not yet have
  vLLM's packed variable-length flash-attention path.
- **Generation:** an orders-of-magnitude slowdown is not expected on a tuned,
  supported configuration. It usually indicates an unfair comparison,
  disabled CUDA graph capture, a poor decode/prefill bucket configuration,
  missing shape-specific tuning, or startup/capture time being included in the
  measurement.

`emmy serve` is not only a command-line wrapper. It retains vLLM's HTTP API,
tokenizer, scheduler, and pooling or attention layers, but replaces the model
trunk with Emmy-compiled programs.

## Embedding serving

### Primary bottleneck

The default embedding path uses a symbolic sequence-length program with
`batch_cap = 1`. vLLM supplies a packed batch, but `EmmyEmbedModel` splits it
into spans and invokes the compiled runner once per sequence.

Stock vLLM instead benefits from packed variable-length batch processing and
optimized attention. Therefore, high-concurrency throughput measures a major
integration difference in addition to kernel performance.

`EMMY_SERVING_BATCHED=1` is an opt-in batched symbolic-sequence mode. It pads a
scheduler step to its longest sequence and executes one batched forward.
However, the embedding trunk's current SDPA path does not reach the
packed/flash tiles. Its dominant attention kernel is effectively serial
`O(B * H * S^2)`, so increasing the batch multiplies an already expensive
operation.

### Measured result

RTX 5090, Qwen3-Embedding-0.6B, 512 tokens/request, concurrency 32,
`--max-model-len 1024`, 256 prompts:

| Arm | Requests/s | Median E2E |
| --- | ---: | ---: |
| Emmy batched, scheduler steps filled | 2.46 | 12.3 s |
| Emmy per-sequence default | 2.33 | 13.7 s |
| Emmy batched, starved at 2048 batched tokens | 0.63 | 49.3 s |
| Stock vLLM | 250.9 | 0.10 s |

In the batched Emmy run, `k_sdpa_linear_reduce` accounted for approximately
76% of runtime. A batched-shape tuning pass alone cannot close this structural
gap.

### Improvements

Immediate operational choices:

1. Prefer stock vLLM for high-throughput embedding production.
2. For low-concurrency Emmy serving, set `--max-model-len` close to the real
   workload and warm representative sequence lengths.
3. Test `EMMY_SERVING_BATCHED=1` only for mostly uniform-length requests.
   Always set a workload-sized `--max-num-seqs`; vLLM's default of 256 can
   allocate excessive `(max_num_seqs, max_model_len)` capacity.
4. Use `EMMY_SERVING_STATIC=1` only for genuinely fixed-length workloads. It
   pads every step to both the full batch capacity and `max_model_len`.

Required implementation work for a substantial throughput improvement:

1. Add packed variable-length flash attention using `cu_seqlens`.
2. Consume vLLM's packed spans in one compiled invocation rather than looping
   over sequences.
3. Remove the remaining `positions.cpu()` synchronization used to discover
   span boundaries.
4. Reduce per-sequence output clones and the final concatenate/cast.

## Generative serving

On the tuned Gemma-4-12B/RTX 5090 path, the current results are generally
within a few percent of stock vLLM at concurrency 1-8 and approximately 14%
behind at concurrency 64. A much larger gap suggests a configuration or
measurement problem.

### Common causes

- `--generate` was omitted. Bare `emmy serve MODEL` selects pooling rather
  than generative serving.
- `--enforce-eager` was supplied. This disables the default whole-step decode
  CUDA graphs.
- A caller-supplied `--compilation-config` replaced Emmy's
  `FULL_DECODE_ONLY` graph configuration without preserving equivalent
  capture behavior.
- `EMMY_GEN_DECODE_BUCKET=0` disabled the static decode twins and forced the
  slower symbolic/eager path.
- The decode bucket is much smaller or larger than the active decode batch.
  The default is 16, while measured workload-specific configurations used 8
  for concurrency 4-8 and 64 for concurrency 64.
- `EMMY_GEN_PREFILL_BUCKET` and `--max-num-batched-tokens` create an
  inefficient chunk quantum, freeze decode riders during prefill, or reduce
  KV-cache admission.
- The target model, GPU architecture, or serving shapes do not match the
  tuned golden configurations. Generic fallback schedule choices can be much
  slower.
- The direct-vLLM comparison uses a different vLLM version, dtype,
  quantization, tensor-parallel configuration, or Python environment. This
  repository pins vLLM to `>=0.23,<0.24`, and Emmy generation requires FP16.
- First-boot compilation or first-use graph capture is included in the
  latency measurement.

### Workload-specific example

For Gemma-4 on an RTX 5090, the measured concurrency-4/8 configuration used:

```bash
EMMY_GEN_DECODE_BUCKET=8 \
EMMY_GEN_PREFILL_BUCKET=2048 \
emmy serve MODEL --generate \
  --dtype float16 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2056 \
  --gpu-memory-utilization 0.96
```

These values are specific to that model, GPU, and workload. They should be
benchmarked rather than treated as universal defaults.

## Fair comparison procedure

Use `emmy serve --stock` for the baseline so both arms select the same vLLM
binary, runner type, endpoint, model-length cap, and benchmark implementation.
Pass the same dtype, scheduler limits, concurrency, prompt lengths, and output
lengths to both arms.

Generative A/B:

```bash
emmy serve MODEL --generate --bench --stock COMMON_FLAGS...
emmy serve MODEL --generate --bench COMMON_FLAGS...
```

Embedding A/B:

```bash
emmy serve MODEL --bench --stock COMMON_FLAGS...
emmy serve MODEL --bench COMMON_FLAGS...
```

Warm each server before recording results. Report TTFT, TPOT, request or token
throughput, concurrency, input/output lengths, GPU, model, vLLM version, and
all scheduling flags.

## Identify which kind of slowdown is occurring

### Slow startup

The first boot traces and compiles the model. Persist `EMMY_CUBIN_CACHE` across
restarts and configure `EMMY_PACK_DIR` so matching execution plans can skip
tracing, pipeline passes, fork resolution, and code generation.

These caches improve startup, not steady-state inference throughput.

### Slow first request or new sequence lengths

The embedding runner caches a CUDA graph per distinct sequence length. A new
length pays a capture cost of roughly one forward before subsequent requests
reuse it. Warm representative lengths and exclude warmup from measurements.

### Slow sustained embedding throughput

This is primarily the known packed-variable-length attention gap. Scheduler
flags or additional generic tuning will not close the observed difference
until the attention and packed-batch execution path is implemented.

### Slow sustained generation

First verify that whole-step decode graphs are enabled, the decode bucket is
nonzero and appropriate for concurrency, the prefill chunk quantum is
appropriate, and the target GPU is using matching tuned kernels. Then profile
TTFT and TPOT separately to distinguish prefill scheduling from decode kernel
or graph latency.

## Related repository evidence

- `emmy/commands/serve.py` — command construction, runner selection, graph
  configuration, dtype, batching, and memory defaults.
- `emmy/serving/ARCHITECTURE.md` — execution model, batched modes, known
  integration costs, caches, and generative serving configuration.
- `emmy/serving/vllm_model.py` — packed-span splitting and per-sequence versus
  batched runner dispatch.
- `plans/serving-gap-closure-findings.md` — embedding and generative A/B
  measurements and kernel attribution.
- The Roadmap section below — latest Gemma-4 TTFT, TPOT, and throughput
  scoreboard.

## Local RTX 4080 verification

Run on 2026-07-28 with:

- NVIDIA GeForce RTX 4080, 16,376 MiB, compute capability 8.9
- PyTorch 2.11.0 with CUDA 13.0
- vLLM 0.23.0

### Correctness results

The CPU command-construction suite passed:

```text
tests/serving/test_serve_command.py
23 passed in 0.24s
```

The batched embedding and outer-CUDA-graph suites passed:

```text
tests/serving/test_runner_batched_gpu.py
tests/serving/test_gen_capture_gpu.py
5 passed in 35.35s
```

The two-layer end-to-end generative plugin test also passed:

```text
tests/serving/test_vllm_plugin_gen_gpu.py
1 passed in 105.92s
```

That test compares the complete greedy token sequence with Hugging Face eager.
It covers the prefill token and subsequent KV-cache decode tokens. The run
emitted only non-fatal deprecation and NCCL process-group cleanup warnings.

### Embedding A/B

Model and workload:

- Qwen/Qwen3-Embedding-0.6B
- FP16
- `--max-model-len 512`
- 255 effective input tokens/request
- `--max-num-seqs 4`
- `--max-num-batched-tokens 2048`
- concurrency 4, 64 prompts

| Engine | Requests/s | Median E2E | Mean E2E | P99 E2E |
| --- | ---: | ---: | ---: | ---: |
| Stock vLLM | 145.74 | 16.20 ms | 26.88 ms | 192.04 ms |
| Emmy per-sequence | 1.67 | 2,361.56 ms | 2,372.32 ms | 2,976.68 ms |
| Emmy batched symbolic-seq | 1.08 | 3,587.67 ms | 3,669.97 ms | 5,371.64 ms |

The stock benchmark lasted only 0.44 seconds, so its absolute throughput is
noisy. The size and direction of the gap are nevertheless unambiguous:
batching did not unlock a faster attention path and was slower than processing
sequences individually.

An earlier concurrency-8 run gave:

| Engine | Requests/s | Median E2E |
| --- | ---: | ---: |
| Stock vLLM | 92.14 | 37.05 ms |
| Emmy per-sequence | 1.67 | 4,708.33 ms |

The batch-8 Emmy arm did not reach serving. At `(max_num_seqs=8,
max_model_len=512)`, its compiled scratch slab requested 17,205,035,008 bytes
after approximately 1.2 GB had already been allocated, exceeding the 4080's
16 GB VRAM. Reducing the batch cap to four allowed the server to start.

These results locally verify:

1. The sustained embedding throughput gap does not require an RTX 5090 to
   reproduce.
2. `EMMY_SERVING_BATCHED=1` is not a remedy while the embedding attention path
   remains non-flash and effectively serial.
3. The `(max_num_seqs, max_model_len)` capacity warning is operationally
   important; a seemingly modest batch/length combination can OOM a 16 GB
   card.

### Generative A/B

Model and workload:

- TinyLlama/TinyLlama-1.1B-Chat-v1.0
- FP16
- `--max-model-len 512`
- 63 effective input tokens and 128 output tokens/request
- `--max-num-seqs 8`
- `--max-num-batched-tokens 512`
- concurrency 8, 32 prompts
- `--gpu-memory-utilization 0.85`

| Engine | Requests/s | Output tok/s | Median TTFT | Median TPOT |
| --- | ---: | ---: | ---: | ---: |
| Stock vLLM | 15.59 | 1,995.01 | 28.02 ms | 3.76 ms |
| Emmy | 0.29 | 36.87 | 224.35 ms | 217.63 ms |

The Emmy command used `FULL_DECODE_ONLY` CUDA graphs, capture sizes
`[1, 2, 4, 8, 16]`, the fused rotary custom operation, FP16, and no
`--enforce-eager`. Therefore, disabled graph capture was not the explanation
for this approximately 54x output-throughput gap.

The Emmy cold boot took approximately 16 minutes because it compiled symbolic
pre/post and decode programs for all 22 layers. The timed benchmark began only
after `/health`, so that compile time is not included in the throughput or
latency figures.

This result qualifies the earlier generative diagnosis: a dramatic gap is
possible on an **untuned model/GPU/shape**, even when the graph configuration
is correct. The repository's RTX 4080 golden file is a small generic kernel
set, while the extensive sm_89 serving-specific goldens target Gemma-4 rather
than TinyLlama. The likely contributors are generic fallback schedule choices
and the default decode bucket of 16 padding an active concurrency of 8.

The bucket mismatch can contribute overhead but cannot by itself explain the
roughly 58x TPOT ratio. A definitive attribution would require:

1. Re-running with `EMMY_GEN_DECODE_BUCKET=8`. **Done 2026-07-29 — ruled out; see the attribution below.**
2. Profiling captured decode steps to identify the dominant kernels and
   in-graph gaps. **Done 2026-07-29 — one kernel; see the attribution below.**
3. Capturing and tuning TinyLlama serving twins on the RTX 4080, then repeating
   the same stock/Emmy A/B.

### Generative gap attribution (RTX 4080, 2026-07-29)

**Bucket-8 re-run.** Same protocol with `EMMY_GEN_DECODE_BUCKET=8` (capture sizes `[1, 2, 4, 8]`): 0.29 req/s,
37.12 out tok/s, median TPOT 215.77 ms, P99 216.33 ms — statistically identical to the bucket-16 run. The decode
bucket is ruled out. The near-zero TPOT variance already pointed at a constant per-step cost.

**Standalone reproduction.** A vLLM-free probe (`EmmyGenRunner.create(..., decode_bucket=8, max_tokens=512)`,
simulated decode step: `embed_device` + per-layer `forward_layer_pre_device`/`forward_layer_post_device` at T=8 +
`final_norm_device`) reproduces the full gap: 214.3 ms/step. CUDA-event attribution: `pre` is 35 µs/layer
(healthy); `post` is a uniform 9.70 ms on every one of the 22 layers. This is NOT the known 4080 one-launch
~5.4 ms bench anomaly (that is a single flaky stall per run; this is deterministic, uniform, and matches the
served TPOT).

**Per-kernel attribution** (`CompiledProgram.iter_once`, batched x20, per-launch sync, layer-0 post twin —
5 launches):

| # | kernel | per-launch | grid x block |
| --- | --- | ---: | --- |
| 00 | `k_linear_14d0e2__partial` | 11.3 us | (128,) x (32,) |
| 01 | `k_linear_14d0e2` | 4.5 us | (64,) x (256,) |
| 02 | `k_linear_mean_reduce_e962dd` | **10,085.5 us** | **(8,) x (128,)** |
| 03 | `k_linear_reduce_49faf4__partial` | 61.8 us | (128,) x (32,) |
| 04 | `k_linear_reduce_49faf4` | 12.1 us | (64,) x (256,) |

The fused norm→gate/up contraction (`k_linear_mean_reduce`) deployed a serial per-row schedule: grid (8,) x
block (128,) = 1,024 threads on a 76-SM card, one block per token row, each serially reducing K=2048 for ~11k
outputs. 10.08 ms x 22 layers = 221.8 ms — the entire TPOT. The DRAM floor for that matmul at M=8 (~46 MB of
weights) is ~66 us, so the deployed schedule is ~150x off the floor while every other kernel in the program is
us-class.

**Conclusion:** the 54x generative gap on this untuned model/GPU is ONE catastrophic cold-prior fork pick on the
fused-norm matmul, not graph capture, not the decode bucket, and not a fixed stall. It should be fully closable
by attribution step 3 (capture TinyLlama twins, `emmy tune` them on the 4080, re-run the A/B) — the same op
class runs us-class on the tuned gemma-4-12B path.

### Attribution step 3 executed: twin tune closes the gap (RTX 4080, 2026-07-29)

Captured the four TinyLlama serving twins (`scripts/capture_gen_twins.py --decode-bucket 8 --prefill-bucket 0`
→ `pre8`/`post8` static M=8 + `pre-sym`/`post-sym` symbolic) and tuned each against a twin-local DB
(`EMMY_TUNE_DB`/`EMMY_ONLINE_FILE` → `_tune/tinyllama-twins-4080/twins/`), ~22 min for post8 and less for the
rest. The -O3 `--bench` re-bench of the tuned post8 twin:

| Kernel | eager | tcompile | emmy tuned | previously deployed |
| --- | ---: | ---: | ---: | ---: |
| `k_linear_mean_reduce` | 87 us | 14 us | **33 us** | 10,085 us |
| `k_linear_reduce` | 76 us | 66 us | 135 us | 74 us-class |
| `k_linear` | 8 us | 8 us | 7 us | — |

`pre8` is at/above eager parity throughout; the symbolic twins keep 0.6-0.9x losses at the 512 hint
(prefill-side, secondary here). Residual tuned loss: `k_linear_reduce` (down proj) 0.56x eager at M=8.

**Serving A/B re-run** (same protocol, `EMMY_GEN_DECODE_BUCKET=8`, twin-local DB/online exported to the server):

| Engine | Requests/s | Output tok/s | Median TTFT | Median TPOT |
| --- | ---: | ---: | ---: | ---: |
| Stock vLLM (recorded 2026-07-28) | 15.59 | 1,995.01 | 28.02 ms | 3.76 ms |
| Emmy untuned (2026-07-28/29, either bucket) | 0.29 | ~37 | 224-236 ms | ~216 ms |
| Emmy tuned twins (2026-07-29) | **14.02** | **1,794.29** | **30.37 ms** | **4.21 ms** |

The 54x gap is confirmed to be entirely the cold-prior pick: one twin tune takes TPOT 215.77 → 4.21 ms (51x)
and lands at 1.12x stock TPOT / 0.90x stock req/s — the same ratio class as the tuned Gemma-4/5090 path
(1.14x TPOT). The generative diagnosis in this document holds: on an untuned model/GPU the gap can be dramatic,
and capturing + tuning the serving twins is the remedy, not scheduler flags or graph-capture changes.

### Prior forensics: the OFFLINE half mispriced the fork (2026-07-29)

`emmy eval online --dataset nodes --kernel matmul --blame [--ablate]` against the twin-local DB
(`_tune/tinyllama-twins-4080/twins/`), 4080 @ -O1 node rows, per prior half:

| metric (matmul free=11264 red=2048 — the fused norm→gate/up) | offline prior | online prior |
| --- | ---: | ---: |
| fork sibling regret, PLACE+REDUCE+STAGE+TILE | **114.87x** | 1.20x |
| fork sibling regret, REDUCE+TILE | 11.41x | 12.30x |
| leaf reachability for the shape | 4.97x (pick 3801 vs best 765 us) | 1.03x mean (all ops) |
| leaf calibration (median Spearman) | +0.51 | +0.94 |

Blame is squarely on the **offline (cold-start) half**: at the two missed PLACE+REDUCE+STAGE+TILE forks
(regret-weight 227.74), the `D_pow2_threads` term pushed the wrong (serial) child at **+150,012.73** —
~30x the next contributor (`D_splitk_deficit` +5,161); the counteracting terms (`D_ctas_ge_sm` -5,928,
`MMA_tier` -4,645) were an order of magnitude too weak. Separately, the PLACE+RASTER+REDUCE family is BLIND
(17/17 missed forks, no term separates pick from best) — a featurizer gap, not a weight problem. The online
half is exonerated: one tune's evidence brings the same fork to 1.20x and calibration to +0.94.

Actions this points at:

1. Offline-prior fix: re-examine the `D_pow2_threads` weight (or its feature definition) at TILE-family forks —
   feeds directly into `plans/analytic-prior-catboost-rework.md`. This TinyLlama twin DB is the repro dataset.
   **DONE 2026-07-30** (branch `fix/offline-prior-pow2-refit`): declared weight bounds in the fitter
   (`fit/linear.py::WEIGHT_BOUNDS`, `D_pow2_threads: 112` — the measured golden-objective saturation point) +
   a bounded refit of the shipped artifact. Golden metrics unchanged, twin fork regret 114.87x → 8.30x. Full
   incident write-up in the rework plan's Update 2026-07-30 section.
2. Featurizer gap: add a separating feature for PLACE+RASTER+REDUCE siblings (currently BLIND).
3. Ops mitigation independent of the prior: merge these ~5k 4080 node rows into the global DB
   (`scripts/merge_node_db.py`) so cold deploys on this card inherit the evidence, and consider a boot-time
   roofline warning in serving ("deployed pick >10x off the DRAM floor — run emmy tune").
   **DONE 2026-07-30**: node rows merged (4080 now 8,466 rows; backup `autotune.db.bak-20260729`), and the boot
   audit landed as `emmy/serving/roofline.py` (branch `feature/serving-roofline-warning`) — static twins timed
   against the self-calibrated weight-streaming floor at every `EmmyGenRunner` boot. Live-validated on this
   incident's shapes: tuned twin evidence audits clean; a cold boot flags the M=1 post twin at 68x with the
   `emmy tune` pointer.

### Verification conclusion

An RTX 5090 is required only to reproduce the repository's exact 5090
Gemma-4 figures and `sm_120` cache artifacts. An RTX 4080 is sufficient to
verify serving correctness, expose the embedding batching/attention gap,
measure the capacity-buffer memory hazard, and evaluate the actual performance
of an sm_89 deployment.

## Roadmap: beat stock vLLM on every Gemma-4 metric

Added 2026-07-29 from the final RTX 5090 fast-math scoreboard (2026-07-26 consolidated re-bench, per-cell
configs, util 0.96 — the TTFT/TPOT parity campaign's exit state; that plan is executed and pruned, its exact
targets are reproduced below).

The tuned Gemma-4 path is already close to stock. Beating stock on every
recorded row requires three principal improvements:

1. Remove slightly more than 1.02 ms from the worst decode TPOT row.
2. Remove more than 8.7 ms from short-concurrency-1 prefill TTFT.
3. Increase concurrency-64 saturated output throughput by more than
   205.6 tokens/s, or 16.9% relative to Emmy's current result.

Everything else already wins or should benefit from the common decode-step
improvement.

### Exact targets

| Metric | Current Emmy | Stock | Required improvement |
| --- | ---: | ---: | ---: |
| Worst TPOT gap, ragged c4 | 27.26 ms | 26.24 ms | >1.02 ms |
| Short-c1 TTFT | 65.0 ms | 56.3 ms | >8.7 ms |
| c64 output throughput | 1,219.5 tok/s | 1,425.1 tok/s | >205.6 tok/s |
| c64 TPOT | 28.75 ms | 28.02 ms | >0.73 ms |

TTFT already beats stock on the other five rows. Ragged-c4 throughput already
beats stock.

### Workstream 1: close TPOT with a hybrid norm path

The highest-value common target is the pair of sandwich residual/norm
operations in every layer. Existing profiling attributes approximately
1.24 ms per decode step to their in-graph behavior at M=1. The large
projection and MLP matmuls are already near their weight-streaming floors.

Build a focused serving A/B with two candidate implementations:

1. Use vLLM's fused residual-add/RMSNorm kernel at the pre/post seam.
2. Implement an equivalent dedicated Emmy kernel that performs residual add,
   statistic reduction, scaling, and output without writing the statistic
   through global memory.

Keep the existing Emmy projection and MLP kernels. Select the better norm
implementation per serving width rather than requiring one implementation to
win at every M.

Acceptance gates:

- At least 1.1 ms TPOT reduction at c1.
- At least 0.3 ms reduction at c8.
- At least 0.75 ms reduction at c64.
- No TTFT or throughput regression.
- Greedy-token and lm-eval correctness remain green.

Do not resume generic finalize/stat launch fusion without a new implementation
strategy. Earlier attempts increased duplicated work and regressed TPOT;
`plans/decode-parity-closers-findings.md` records those results.

### Workstream 2: capture the complete short-prefill step

The short-c1 loss is isolated: the nominal 256-token prompt becomes a
257-token prefill and rides the symbolic path.

Build an exact short-prefill tier:

1. Capture static pre/post twins at the actual 257-token width.
2. Tune those real serving graphs on the RTX 5090.
3. Capture the whole step: embedding, all layers, attention, final norm,
   logits, and first-token sampling.
4. Route exact or nearby short prefills to this tier while retaining the
   symbolic path as fallback.

Whole-step capture is essential. A static twin alone may retain the per-layer
DLPack, locking, clone, and dispatch overhead responsible for the short-step
gap.

Acceptance gate: median TTFT below 56.3 ms while the TPOT improvements from
Workstream 1 remain intact.

### Workstream 3: attribute and close c64 throughput

The c64 throughput gap cannot be explained by TPOT alone:

- TPOT is only 2.6% slower.
- Output throughput is 14.4% below stock.

Run synchronized stock and Emmy nsys traces using the exact final-scoreboard
c64 configuration. Record:

- Scheduler step-width histogram.
- Active sequences and scheduled tokens per step.
- Prefill, decode, fill, and drain intervals.
- GPU idle time between full-graph replays.
- Logits and sampler time.
- KV-block availability and admission delays.
- CUDA graph node time versus wall time.
- Output tokens per fully occupied decode step.

The trace should explain essentially the full 205.6 tok/s deficit before
implementation begins.

Route the result by cause:

- **Underfilled steps or scheduler idle:** adjust admission,
  `max_num_batched_tokens`, or decode-rider scheduling.
- **Full steps with low GPU occupancy:** tune or redesign the M=64 twins.
- **Gaps between graph replays:** remove synchronization and host-visible
  operations between steps.
- **Excessive fill/drain cost:** improve wave scheduling or capture transition
  and prefill steps.
- **Logits/sampler bottleneck:** share the same optimized vLLM implementation
  and data path used by stock.

Even after a 0.75-1.1 ms TPOT improvement, c64 may need another 10-14% of
system-level throughput from scheduling, occupancy, or fill/drain work.

### Workstream 4: make serving tiers automatic

The final scoreboard uses workload-specific configurations:

- c1: decode bucket 32.
- c4/c8: decode bucket 8, prefill bucket 2048, batched-token cap 2056.
- c64: decode bucket 64.
- Ragged c4: decode bucket 8 with 4096-scale rider scheduling.

A general production server cannot assume the future workload or require an
operator to restart it with different environment variables.

Build shared-weight execution tiers:

- Decode: M=1, 8, 32, and 64.
- Prefill: M=257, 2048, and 4096.
- Symbolic fallback for uncommon widths.

Route each step to the smallest suitable tier. Share weights and the buffer
arena, and load all validated tiers from a prebuilt execution pack.

The scheduler policy also needs a product decision: c4/c8 prefer a 2056-token
chunk quantum while c64 prefers 4096. Either make chunking adaptive or expose
separate validated deployment profiles rather than presenting a per-cell
benchmark configuration as one universal server configuration.

### Preserve the proven kernel structure

Do not reopen these paths without new end-to-end evidence:

- Generic golden retuning when matmuls are already at the DRAM floor.
- Removing launches by duplicating computation.
- Fused computed-A gate/up at prefill widths; splitting into fast
  single-channel matmuls was proven faster.
- Landing isolated microbenchmark wins without a full serving-step A/B.

The remaining Gemma-4 work is primarily runtime composition, specialized
fused norms, graph coverage, and c64 scheduling rather than ordinary matmul
tuning.

### Execution and acceptance protocol

Execute in this order:

1. Hybrid fused-norm seam.
2. Short-prefill whole-step capture.
3. c64 trace and attributed fix.
4. Automatic execution-tier routing.
5. Full RTX 5090 acceptance matrix.

For each change:

1. Run the affected stock/Emmy cell at least three times on the same RTX 5090.
2. Use identical vLLM version, FP16 mode, scheduler limits, prompt set, seed,
   and memory utilization.
3. Use fresh or correctly invalidated execution packs and empty online
   evidence unless the test explicitly evaluates tuned evidence.
4. Run greedy correctness, in-model golden drift, lm-eval, and the full test
   suite before expanding to the complete matrix.

Final gates:

- Every TTFT value is strictly below stock.
- Every TPOT value is strictly below stock.
- Every throughput value is strictly above stock.
- P99 latency is also below stock.
- KV capacity and peak VRAM are no worse than stock.
- Greedy correctness and lm-eval remain green.
- The baked cubin/pack image starts without runtime compilation.

The strongest current scoreboard uses the fast-math lane. Any production claim
must name that numerical mode explicitly and keep its accuracy gates as part
of the acceptance criteria.

## 2026-08-12 RTX 5090 re-baseline (fcbc880f + pack fix) and step attribution

Re-measured the July losing cells on a rented 5090 (vast.ai, driver 580.126.09, torch 2.11+cu130, vllm 0.23.0,
transformers pinned 5.12.1, `VLLM_USE_FLASHINFER_SAMPLER=0`), 3 runs per cell per arm, empty online evidence,
repo goldens only. Measured at **#483 (fcbc880f)**, not main: #482 re-orphaned the gemma-4 fused geglu goldens
in-model (cold deploys hit 39,623x–104,464x roofline picks; fix in PR #490, pack-import crash fix in PR #488).
The absolute July targets do NOT transplant to this host — both arms shifted (stock small_c1 TPOT 16.28 → 18.41)
— so each arm's own stock run is the reference.

| cell | metric | emmy fm | stock | ratio (July ratio) |
| --- | --- | --: | --: | --- |
| small_c1 256/256 | TTFT ms | 115.6 | 74.2 | 1.56x (1.15x) — WORSE |
| small_c1 | TPOT ms | 19.10 | 18.41 | 1.04x (1.05x) — held |
| rag_c4 8192/256 | TPOT ms | 27.13 | 28.62 | **0.95x WIN** (1.04x) |
| rag_c4 | tok/s | 109.3 | 105.2 | **1.04x WIN** (1.02x) |
| rag_c4 | TTFT ms | 2413 | 2407 | 1.00x (0.89x) — win lost |
| c64 np256 | TPOT ms | 35.08 | 32.09 | 1.09x (1.03x) |
| c64 np256 | tok/s | 1195 | 1368 | 0.87x (0.86x) — unchanged |
| c64 np256 | TTFT ms | 1457 | 1415 | 1.03x |

c64 ran with `--max-num-seqs 64` (without it the 128-capture rung admits over-bucket symbolic decode steps —
`_warn_symbolic_decode` fired — though re-running capped changed TPOT < 0.1 ms, so that leak was immaterial here).

### nsys step attribution (steady-state c64, 25 s windows, `--cuda-graph-trace=node`)

| per decode step | emmy fm | stock |
| --- | --: | --: |
| wall | 40.6 ms | 36.7 ms |
| GPU busy | **29.9 ms** | 30.9 ms |
| idle (host gaps) | **10.7 ms** | 5.8 ms |
| kernels/step | 841 | 658 |
| D2D copied | **170 MB (69 copies)** | 1.1 MB |

**Emmy's in-graph GPU time already beats stock's** (29.9 vs 30.9 ms busy). The whole c64 TPOT loss is
(a) +4.9 ms/step of host idle on eager mixed prefill+decode steps (gpu_lock + DLPack dispatch + per-step
staging: the 170 MB/step D2D torrent — the post→pre chaining from vLLM-integration Milestone A2 is inert on
eager steps by design), and (b) modest kernel headroom inside busy time (corrected 2026-08-13, see below):

- `k_linear_mean_reduce` (fused norm→gate_up matmul) — 164.2 µs x 48/step = 7.9 ms/step at 1.22x its ~134 µs
  weight-streaming floor: ~1.4 ms/step of headroom.
- `k_to_4__cut_acc0` (the geglu-cut down matmul) — 76.8 µs x 48/step = 3.7 ms/step at 1.14x its ~68 µs
  weight-streaming floor: ~0.4 ms/step. The cut's actual glue kernels are negligible: `k_to_4` (the
  down-output f32→f16 cast, [64, 3840]) runs 0.8 µs and the geglu/norm value piece `__cut_v9` 1.7 µs,
  ~0.12 ms/step for the whole cut-glue class.

At c1, M=1 decode is healthy: the big weight streams run at 1.08–1.10x their DRAM floors, GPU busy 16.8 ms vs
a ~12 ms aggregate weight floor, TPOT within 0.7 ms of stock. The 262k-vocab lm_head costs ~1.26 ms/step in
BOTH arms (near-peak bandwidth — a shared floor, not a gap). The small_c1 TTFT loss (and rag's lost TTFT win)
is the eager symbolic-prefill burst: geglu-cone kernels ~4–5x floor at prefill widths plus per-layer eager
framing; the boot roofline also flags the m4096 chunk twins at 24–28x floor (overstated — see the correction:
those shapes are compute-bound, ~1.4–1.5x their compute floor).

### Correction (2026-08-13): the "to_4 cut-cast" item was a name-collision misread

The original attribution called `k_to_4__cut_acc0` "the materialized accumulator cast of the geglu cut,
95 µs x ~45/step = 4.3 ms/step of pure glue at ~8x its copy floor (moving ~12 MB)" and ranked fixing it as
the top c64 action, claiming it flips the TPOT row on its own. That reading is wrong. The nsys per-kernel
summary groups by kernel NAME, and emmy kernel names recur across serving shapes (the same graph node
compiles once per twin), so that 95 µs "per-launch average" mixed two different kernels sharing a name:

- 27,542 m64 decode launches at 76.8 µs (grid 240x256, 59 regs, tight 74–80 µs range) — the geglu-cut down
  matmul streaming its 118 MB weight slab at 1.14x floor, consistent with the recorded
  `gemma4_12b.mlp_down_fused.m64.lin.cut` golden (84.2 µs). Healthy; nothing to tune.
- 336 m4096 chunk-prefill launches at 1.62 ms (grid 480x256, 249 regs) — the separately-tracked chunk-twin
  prefill item, which lands in the ~7 mixed steps of the window, not in every decode step.

Weighted: (27542·76.8 + 336·1621.9) / 27878 = 95.4 µs — exactly the misread figure. The "[64, 30720] f32→f16
cast" it was mistaken for does not exist as a kernel: the only [64, 30720] value in the layer is the gate_up
output, written f16 in-kernel by `k_linear_mean_reduce`'s epilogue; the geglu cut's f32 workspace is the down
accumulator [64, 3840], and its cast (`k_to_4`) runs 0.8 µs. There is no multi-ms cast-glue lever at c64
decode — the busy-time table above (emmy 29.9 < stock 30.9 ms) was already inconsistent with one.

Same-session second correction: the boot roofline's "m4096 chunk twins at 24–28x floor" compares
compute-bound prefill matmuls against a weight-streaming floor. Against the compute floor (~210 TFLOPS dense
f16 with f32 acc, ~2x that in the fm f16-acc lane) the measured chunk kernels are ~1.4–1.5x: down m4096
1.62 ms vs ~1.15 ms fm floor, gate_up m4096 3.52 ms vs ~2.3 ms. Real headroom, a fraction of "24x".

### Ranked next actions (goal: emmy >= stock on every row) — re-ranked 2026-08-13 per the correction

1. Whole-chunk-step capture (the integration plan's promoted item) — removes the ~5 ms/step host idle and the
   per-step staging D2D, activates A2 chaining; the only item big enough to flip c64 TPOT (gap 3.0 ms), and
   the main TTFT lever together with:
2. Prefill-side kernel quality — sym/chunk geglu cone and the m4096 chunk twins (~1.4–1.5x compute floor; the
   roofline's 24–28x overstates them).
3. Fused-norm matmul retune at m64/fm — ~1.4 ms/step headroom on `k_linear_mean_reduce` (1.22x floor) plus
   ~0.4 ms/step on the geglu-cut down matmul (1.14x): worth a pass only after 1 lands.

Protocol traps recorded for the next measurement session: nsys defaults hide captured-graph kernels
(`--cuda-graph-trace=node` required, and sampling flags belong on `nsys start`, not `launch`); time the capture
window against the bench duration (~55 s at c64 np256); `EMMY_PACK_DIR` must be wiped when switching trees on a
box (pack keys do not hash compiler internals); the 5090 needs `VLLM_USE_FLASHINFER_SAMPLER=0`; transformers
must satisfy the PR #491 window; and never read per-kernel launch economics off a name-keyed nsys summary —
split by launch geometry (grid) first, because decode and chunk shapes of the same graph node share one kernel
name in a serving process (the root of the 2026-08-13 correction above).
