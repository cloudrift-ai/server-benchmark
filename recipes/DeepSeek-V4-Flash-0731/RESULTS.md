# DeepSeek V4 Flash 0731 on 16× V100 SXM3 32 GB

Status: serving-qualified with the 1Cat/vLLM engine pinned by the recipe. Reverified 2026-08-21 on 16× V100 SXM3 at
repository revision `fd7b09041`. Emmy serving became eligible on 2026-08-26: the runner reads the checkpoint's MXFP4
experts, shards them across the tensor-parallel group, and `EmmyGenModel` serves this checkpoint at TP8 × PP2 while
hosting the fork's attention sublayer (see "Emmy serving lane" below). What is still missing is an equal-envelope A/B
against the numbers here, and a prebuilt serving image.

## Qualified deployment

| Item | Value |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Model revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Hardware | 16× Tesla V100-SXM3-32GB, compute capability 7.0, 12 NVSwitches |
| Driver / CUDA | 580.159.03 / 13.0 |
| Engine | 1Cat/vLLM `1.2.3.dev87+gd76126608.d20260810` |
| Image | `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` |
| Image digest | `sha256:276240257b224097876b5b6db8f0d32484dff6a6f168d6b03d6df188e5c65bc1` |
| Serving shape | TP8, PP2, context 1,048,576, concurrency 8, FP8 KV cache |

The recipe disables process-local SM70 MXFP4 small-shape timing selection, because the timed selector chose different
W13 expert GEMM configurations after fresh starts and changed greedy output. Greedy decoding was stable across this
run's probes.

### Host prerequisite: NVIDIA Fabric Manager

These 16 GPUs sit behind 12 NVSwitches. Until Fabric Manager trains that fabric, `nvidia-smi` lists all 16 GPUs while
every engine worker dies at `cudaGetDeviceCount()` with `error 802: system not yet initialized`, which reads like an
engine or model fault rather than a missing host service. A freshly provisioned host for this run had no Fabric Manager
installed and could not deploy at all. `emmy deploy` now installs and starts it automatically on NVSwitch hosts, pinned
to the running driver's exact version; no recipe change is required. Anyone deploying this recipe outside Emmy must
ensure `nvidia-fabricmanager` matching the driver is running first.

## Best recipe performance

Measured 2026-08-21 with the pinned 1,048,576-context recipe at repository revision `fd7b09041`. Four client repeats;
the first primes the prompt set after deployment and is excluded. Each repeat used eight unique 1,024-token prompts at
concurrency 8 and requested 64 output tokens with greedy decoding and ignored EOS. All 24 reported requests completed
with exact token counts. Spread is the population standard deviation across the three steady repeats.

| Metric | Three-repeat mean ± standard deviation |
| --- | ---: |
| Successful / failed requests | 24 / 0 |
| Benchmark duration | 16.6300 ± 0.0787 s |
| Request throughput | 0.4800 ± 0.0000 requests/s |
| Output throughput | 30.7933 ± 0.1482 tokens/s |
| Total token throughput | 523.5033 ± 2.4875 tokens/s |
| Mean TTFT | 3,465.04 ± 260.39 ms |
| Mean TPOT / ITL | 208.453 ± 3.042 ms |

Throughput is unchanged from the 2026-08-19 measurement (30.9800 ± 0.0942 tokens/s) across 16 merged pull requests.
TTFT reads about 10% lower and TPOT about 3% higher, but the steady TTFT spread grew from 27 ms to 260 ms, so those
two distributions overlap and neither shift is established here. The engine image is byte-identical between the runs.

The recipe's zero-JIT intent is still not fully met: eight Triton kernels JIT-compile once during the first repeat's
warm-up and none recurs, so they cost the priming repeat only.

## These numbers are vLLM without Emmy kernels

Every serving number in this report was produced by vLLM alone, with no Emmy kernels in the process. The deployed
container runs 1Cat/vLLM; the recipe sets no `EMMY_*` variable, passes no Emmy plugin, and the archived server log for
the reported run contains no mention of Emmy at all. The `VLLM_SM70_*` variables it does set are the 1Cat fork's own
Volta features — flash attention, and the turbomind FP8 / MXFP4 quantized paths — not Emmy code. So the baseline a
reader usually wants, "vLLM without Emmy", is exactly what the throughput, TTFT and TPOT figures above already measure.

The other arm — vLLM **with** Emmy kernels — now exists, but is not yet measured at this report's envelope. Emmy's
serving A/B is `emmy serve <model> --bench` against `emmy serve <model> --bench --stock`, and only the second of those
two lanes is still impossible here.

**The Emmy side serves as of 2026-08-26** (see "Emmy serving lane" below): `EmmyGenModel` hosts the fork's attention
sublayer per layer and owns everything else — hyper-connection stream mixing, norms, shared and routed experts — at
TP8 × PP2 on this host, in the pinned 1Cat image. What it does not yet have is an equal-envelope comparison against
the numbers above, a prebuilt image, or a serving config: `docker/vllm-emmy-serve/models/` still holds none for this
model, and no `cloudriftai/vllm-emmy-deepseek-v4-flash-0731` image exists.

**The stock side has no Volta kernels.** Measured on the target host with `vllm/vllm-openai@sha256:03768d94…`, the
exact stock image the sibling `DeepSeek-V4-Flash` recipe pins for this checkpoint on H200 (vLLM
`0.22.1rc1.dev332+g2c9c07c85`, torch `2.11.0+cu130`):

| Check | Result |
| --- | --- |
| `torch.cuda.get_device_capability(0)` | `(7, 0)` — Tesla V100-SXM3-32GB |
| `torch.cuda.get_arch_list()` | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120` — **no `sm_70`** |
| A plain 256×256 fp16 matmul | `CUDA error: no kernel image is available for execution on the device` |

PyTorch warns the device is unsupported before any vLLM code runs, and a single matmul fails, so the engine never
reaches model loading. That is an architecture gap rather than anything specific to DeepSeek V4, and it is why every
V100 recipe in this repository pins a 1Cat SM70 build instead of a stock image. `emmy serve` invokes `vllm serve` from
the Python environment (`vllm` is the optional `serving` extra), so its stock lane would hit exactly this wall; the
SM70 build that does run Volta is delivered as a container, not a wheel.

**What the numbers therefore are.** A pure vLLM result on the only engine build that runs this checkpoint on Volta,
measured against itself across repository revisions. They are not a speedup over anything: there is no Emmy-accelerated
arm to beat, and no stock arm that survives the architecture gap. The `emmy bench` reproduction accordingly has no
second engine lane to filter to. The compiler work recorded below is kernel-level evidence (the golden) and is
independent of serving eligibility.

## Emmy serving lane (2026-08-26)

`EmmyGenModel` serves this checkpoint at TP8 × PP2 on this host, inside the same pinned 1Cat image, with the fork's
attention sublayer hosted per layer and Emmy owning the hyper-connection stream mixing, norms, shared expert and
routed experts. Serving shape: `--max-model-len 4096 --kv-cache-dtype fp8 --block-size 256
--gpu-memory-utilization 0.90`, eager (decode capture is unsupported for this architecture — the routed combine
host-syncs every step).

| Item | Emmy lane | Plain 1Cat, same shape |
| --- | --- | --- |
| Boot, engine init → serving | ~19 min (55 s load, ~5 min compile on a warm cubin cache, ~12 min profile + KV) | ~3 min |
| Free for KV after residents | 12.99 GiB | 5.68 GiB |
| KV capacity | 78,730 tokens (PP0) / 81,190 (PP1) | 34,397 / 35,472 |
| Single-stream decode | ~3.6 tok/s | ~8.8 tok/s |
| Mixed prefill/decode | 8 concurrent requests, prompts 5–361 tokens, outputs 8 and 128: 544 output tokens in 101.9 s | not measured at this shape |

The KV difference is structural, not tuning: Emmy shards the 256 routed experts across the tensor-parallel group
(32 per rank) and completes the partial sums with the group all-reduce, while the fork replicates all 256 experts on
every rank (`local_experts=256` in its own log). At equal `--gpu-memory-utilization` that buys the Emmy lane 2.3× the
KV capacity, which is the ceiling on context and concurrency. It costs decode throughput today, and the comparison
above is indicative rather than a protocol A/B — the same-envelope measurement with repeats is still to come.

**Greedy agreement against the fork's own implementation.** Both arms served the same checkpoint revision at the same
shape and answered a fixed four-prompt corpus at temperature 0, 32 tokens each. Three prompts — including a
361-token one that spills past the 128-token sliding window into the compressed and indexed attention layers, and a
code prompt — agree on all 32 token ids exactly. The fourth diverges at token 6, where the fork's own distribution
puts its pick and Emmy's 0.125 nats apart (` Italy` −1.3242 against ` Spain` −1.4492), and Emmy's logprob for its own
pick lands within 0.089 of the fork's: a near-tie between two continuations the model has no real preference between,
both of which continue coherently. Each arm is individually deterministic — two runs of each agree on every token —
so the single divergence is arm-to-arm numerics at a tie, not run-to-run noise.

**The routed-expert kernels have no golden coverage.** The committed golden is the per-layer compiler-qualification
trace below — layers 0/2/3/4 plus the model seam, expert weights as dense values. Serving's routed-expert program is
input-sourced instead: packed MXFP4 blocks and E8M0 scales that decode in-graph. Goldens key on strict structural
kernel identity, so that program matches nothing in the file and resolves its forks from measured or prior evidence.
Serving correctness is unaffected — the agreement above was measured in exactly that state — but a release that warms
and bakes this model seals whatever those forks resolved to, unqualified.

Closing it needs a *serving* golden, which this model does not have: `emmy trace --serving-twins` derives its width
and pin matrix from `models/<slug>.env`, and that config is what the image stage's headroom sweep produces. So the
serving golden comes with the image work rather than before it.

This is what makes the output the load-bearing evidence: a transposed expert matrix or a mis-scaled MXFP4 decode
yields fluent-looking garbage, not correct capitals, a valid Python guard clause, and 101 of the corpus's 128 token
ids identical to the reference implementation.

## Context and accuracy

The engine allocated KV capacity for 4,244,903 tokens on PP0 and 4,281,497 tokens on PP1, reporting 4.05×/4.08×
maximum concurrency at the full 1,048,576-token context. An exact 1,048,575-token prompt plus one decode token
completed with HTTP 200 in 1,331.6 s and reported 1,048,576 total tokens, with no preemption, allocator error, or OOM;
peak physical allocation reached 32,206 MiB of 32,768 MiB per GPU. The prompt used random token IDs so that
prefix-cache block deduplication could not shrink the KV footprint under test.

Capability probes on the same pinned recipe: factual completion returned `Paris`; the exact arithmetic probe returned
`323` identically across repeated requests; tool calling returned a structured `multiply(a=17, b=19)` call; and
reasoning was separated into the engine's `reasoning` field (400 characters of reasoning against 197 characters of
content).

Reasoning separation is opt-in per request. The `deepseek_v4` reasoning parser resolves to vLLM's
`DeepSeekV3ReasoningParser`, which delegates to the R1 parser only when the request passes
`chat_template_kwargs: {"thinking": true}` (or `enable_thinking`); otherwise it installs the identity parser and
returns `reasoning: null`. With thinking explicitly disabled, the identity parser also leaves the template's stray
closing `</think>` marker at the head of `content`. Both behaviours are upstream parser semantics, not recipe faults,
but clients that expect separated reasoning must send the flag. The server logs a benign startup warning
(`Auto-initialization of reasoning token IDs failed`) because the identity parser exposes no reasoning delimiters.

These probes and the context measurement were taken on a separate deployment of this exact recipe, image, and serving
shape, immediately before the benchmark deployment; the performance table above comes solely from the archived
benchmark run.

## Compiler qualification

### Coverage

All 43 decoder layers reduce to three distinct traced graphs, set by `compress_ratios` in the model config. Tracing
seven layers and comparing Graph IR node counts closes the manifest empirically:

| Class | `compress_ratio` | Representative | Graph IR nodes | Verified identical |
| --- | ---: | --- | ---: | --- |
| layers 0–1 | 0 | layer 0 | 945 | layer 1 |
| even layers 2–42 | 4 | layer 2 | 1,156 | layers 4, 42 |
| odd layers 3–41 | 128 | layer 3 | 1,087 | layer 41 |

Layer 41 and layer 42 are `dspark_target_layer_ids`, and they trace identically to their ordinary siblings, so the
dspark specialization is not visible as a distinct architecture path. The committed golden's fourth representative
(layer 4) is redundant with layer 2. Non-layer seams are covered by the `model-seam` targets.

One path is **not** covered: the model declares `num_nextn_predict_layers: 1`, but the MTP head is not exposed as a
decoder layer — `emmy trace --layer 43` fails with "layer 43 not found (model has 43 layers)" — so it cannot be
traced through this interface. The committed golden does not contain it either.

A whole-model architecture trace is not bounded on this checkpoint: it grew past 830 GiB of host RAM, climbing about
45 GiB/minute, before it was stopped. Per-layer tracing is bounded (about 1 GiB), which is why the inventory is built
from representatives. Per-layer tracing is nonetheless dominated by merge-region dependency resolution in the Loop
splicer (`ir/loop/splicer.py`, `_ensure_dep` under `build_merged_region`); three representative layers ran 4h45m in
that phase without emitting a post-fusion inventory, confirming the 2026-08-11 report's attribution. The tuning below
therefore ran against the committed inventory rather than a freshly re-derived one.

### Verification on the target GPU

| Gate | Checked against | Result |
| --- | --- | --- |
| Repository-level validation | committed file | passes |
| Strict decode of every realization | committed file | 279 / 279 |
| Reconstruct and lower every target | Tesla V100-SXM3-32GB (sm_70) | 279 / 279, exit 0, 3 min 16 s |

### `REDUCE=g2a`: a validator false negative, now fixed and re-measured

The nine realizations the golden pinned to `REDUCE=g2a` all recorded worse-than-greedy numbers, and an exact `--ab`
pin reported `unreproducible pin: REDUCE=g2a realized (off)`, so the row would not bench at all. Neither symptom
meant what it looked like.

`g2a` decodes as a cross-CTA split of width 2 with atomic finalize, and it does realize: pinning it halves the
emitted K loop from 16,384 to 8,192, adds the partition axis and closes with `atomicAdd`. What changed is where the
receipt lives. #539 made a split mint brand-new kernels and had `knob.consume_kernel_row` strip their schedule row,
so no piece may carry the `g<n>` it came from — `test_split_fresh_kernels` asserts that outright. The partition is
recorded by the piece's sliced reduce axis, not by a stamp, exactly as a realized `PLACE` cut is. The
realized-vs-pinned gate reads only stamps, so it saw the pieces' `REDUCE=(off)` and called a realized pin dropped.
The golden was recorded 2026-08-11, seven days before #539, which is why its rows predate the problem.

The gate now skips the `g<n>` stage the way it already skips `PLACE`, and still gates the rest of the value. With
that, all eight surviving `g2a` rows bench cleanly (16/16 runs, no unreproducible-pin error) and were re-measured on
the target GPU. `g2a` and greedy are indistinguishable on every one of them:

| Realization | `g2a` (µs) | greedy (µs) |
| --- | ---: | ---: |
| `layer0.i003` | 913.4 | 914.4 |
| `layer0.i052` | 914.4 | 916.5 |
| `layer2.i003` | 914.4 | 914.4 |
| `layer2.i063` | 916.5 | 916.5 |
| `layer3.i003` | 914.4 | 916.5 |
| `layer3.i064` | 915.5 | 915.5 |
| `layer4.i003` | 915.5 | 916.5 |
| `layer4.i063` | 912.4 | 912.4 |

Each row keeps its `g2a` knobs — the configuration is real and ties with greedy — and carries the slowest of its two
runs on each side. Across the file, realizations materially slower than greedy fall from 41 to 11, and the remaining
eleven are sub-microsecond pointwise kernels.

### Tuning: equal-budget hybrid versus MCTS-only (2026-08-21, whole model)

The previous round had to scope its A/B to nine targets because `emmy tune` re-validated and rewrote the whole
document on every incremental persist. With that fixed upstream, both arms now sweep the **entire 279-target
inventory** at roughly 15 measured rows per minute against about 3.4 before — 1 h 18 m per arm instead of an
estimated ten hours.

Both arms started from the same inventory-only base (no knobs, timings, or ranking), an empty tune DB and online
prior, separate empty cubin caches, `--max-candidates 8`, `--patience 4`, `--seed 731`, all 16 GPUs, same compiler
revision, MCTS arm first. The hybrid arm added 18 knob proposals across 8 realizations, each reserving a candidate
slot before MCTS. The arms came out closely matched, which is what makes the comparison fair:

| Arm | Wall clock | Benches | Measured rows | ok / bench_fail | Prior calibration |
| --- | ---: | ---: | ---: | ---: | ---: |
| MCTS-only | 1:17:57 | 9,156 | 3,312 | 2,279 / 1,033 | +0.96 |
| Hybrid | 1:18:56 | 9,200 | 3,343 | 2,296 / 1,047 | +0.93 |

**Outcome: no winner, and nothing promoted.** Across 279 targets the O1 ranking lane put hybrid ahead on 44, MCTS
ahead on 44, and tied the remaining 169. The golden is unchanged by this round.

That result is less interesting than why. After the previous round resolved the `g2a` cluster, this golden loses
19.2 ms to greedy in total, and **19.2 ms of it sits in one realization** — `model-seam.k_linear_d74dc7` at 0.876×
(154.2 ms against greedy's 135.0 ms). Every other slower-than-greedy row is under 3 µs. It is also the last
`model-seam` row still on the Volta MMA warp tile, while its winning siblings all use a cooperative reduce over a
thread work inventory — the same swap that won 2.82× on `k_linear_db1eb0` last round. So the hybrid arm proposed
exactly that.

**Every candidate for that target was killed by a watchdog rather than measured.** In the tune lane the proposals hit
the 2.0 s GPU-time budget; only the alternative MMA work shape survived, at 176.2 ms, worse than the incumbent. The
O3 verification lane cannot settle it either: there the *greedy baseline itself* exceeds the 100 s wall budget and is
SIGKILL'd, so the run produces no comparable output at all. This kernel is simply too large for the budgets the
benchmark harness applies, and its 19.2 ms of headroom is currently unreachable by tuning — not refuted, unmeasurable.

**Bench failures.** About 31% of rows in each arm failed to bench, near-identically in both, so they do not bias the
comparison:

| Class | MCTS | Hybrid |
| --- | ---: | ---: |
| nvcc compile failed | 451 | 462 |
| bench worker exceeded the 16 s wall budget | 434 | 460 |
| benchmark run exceeded the 2.0 s GPU-time budget | 137 | 129 |
| hung kernel | 7 | 3 |

Every one of the nvcc failures is the same defect: generated CUDA for a `k_div_*` kernel emits `float v0 = in0 + in2;`
where `in0` was never declared, and nvcc rejects it (2,173 occurrences in the hybrid log, one kernel family, no other
compiler message). It reaches only searched variants — the golden's own recorded configurations compile and replay
cleanly — so it costs search coverage rather than deployed correctness. Knob values do not separate failing from
passing rows and all 462 failing rows are distinct ops, so the trigger is structural to that kernel family.

### Reproducing the compiler work

Emmy needs `nvcc` on `PATH`, and the CUDA **12.9** toolkit specifically: CUDA 13 dropped Volta. Two further host
conditions cost hours before they were found. PyYAML silently falls back to its pure-Python loader without `libyaml`,
which alone cost more than 13 minutes per parse of this 3.4 MB golden. And `torch 2.13` installs
`nvidia-cuda-nvrtc 13.0.88`, which `cupy-cuda12x` then resolves in preference to any CUDA 12 NVRTC; because CUDA 13
has no Volta support, every CuPy JIT path dies with `invalid value for --gpu-architecture` and all benchmarking fails.
Running with `LD_PRELOAD=/usr/local/cuda-12.9/lib64/libnvrtc.so.12` restores it.

## Reproduce

```bash
emmy bench experiments/DeepSeek-V4-Flash-0731/serving_v100_sxm3 --ssh <user>@<16x-v100-host>
```

The experiment runs four client repeats. The first warms the complete unique prompt set after deployment; use repeats
two through four to reproduce the reported steady result. Use `$run-experiment` to retain the latest raw results,
system-only experiment records, and factual artifact index.

## Limitations

The performance table covers one short-context shape (1,024 in / 64 out at concurrency 8); long-context and
high-concurrency serving are validated for capacity and correctness but not for throughput. The cross-run comparison
against 2026-08-11 changes host and driver together. The Emmy lane's figures are indicative single-run measurements at
a 4,096-token context, not the protocol A/B (equal envelope, one priming plus three steady repeats, spread reported),
so they support "it serves, correctly, and where it stands roughly" and nothing finer.
