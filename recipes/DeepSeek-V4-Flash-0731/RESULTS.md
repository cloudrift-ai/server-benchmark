# DeepSeek V4 Flash 0731 on 16× V100 SXM3 32 GB

Status: serving-qualified with the 1Cat/vLLM engine pinned by the recipe. Verification refresh of 2026-08-19 on a
second 16× V100 SXM3 host. Emmy serving remains ineligible: the DeepSeek V4 compressor and hyper-connection path has
no executable external-attention serving ABI, so `EMMY_FAST_MATH` is not set and there is no Emmy comparison lane.

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

Measured 2026-08-19 with the pinned 1,048,576-context recipe at repository revision `12bb850e`. Four client repeats;
the first primes the prompt set after deployment and is excluded. Each repeat used eight unique 1,024-token prompts at
concurrency 8 and requested 64 output tokens with greedy decoding and ignored EOS. All 24 reported requests completed
with exact token counts. Spread is the population standard deviation across the three steady repeats.

| Metric | Three-repeat mean ± standard deviation |
| --- | ---: |
| Successful / failed requests | 24 / 0 |
| Benchmark duration | 16.5267 ± 0.0492 s |
| Request throughput | 0.4833 ± 0.0047 requests/s |
| Output throughput | 30.9800 ± 0.0942 tokens/s |
| Total token throughput | 526.6367 ± 1.6402 tokens/s |
| Mean TTFT | 3,838.67 ± 26.92 ms |
| Mean TPOT / ITL | 201.360 ± 0.493 ms |

This is about 9% below the 2026-08-11 qualification on the previous 16× V100 host (driver 580.173.02), which measured
34.1830 ± 0.2886 output tokens/s. Decode cost is nearly unchanged (TPOT 201.36 ms vs 195.949 ms, +2.8%); essentially
all of the gap is prefill (TTFT 3,838.67 ms vs 2,580.88 ms, +48.7%). Both runs are internally stable, so this is a
machine-level difference rather than noise; the two runs differ in both host and driver version, and this evidence
cannot separate them.

The recipe's zero-JIT intent is not fully met. Eight Triton kernels JIT-compile once during the first repeat's
warm-up — including the prefill-metadata and SM70 quantized-attention paths — and none recurs, so they cost the
priming repeat only. The per-kernel list is in the experiment report.

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
| Reconstruct and lower every target | Tesla V100-SXM3-32GB (sm_70) | 279 / 279, exit 0, 3 min 54 s |

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

### Tuning: equal-budget hybrid versus MCTS-only

Scope: the 9 `g2a` realizations. The other 270 are at or above greedy and carry no headroom, and a full 279-target arm
measured out at roughly ten hours per arm. This is a scoped kernel result, not a whole-model performance claim.

Both arms started from the same inventory-only base (no knobs, timings, or ranking), an empty tune DB and online prior,
separate empty cubin caches, `--max-candidates 8`, `--patience 4`, `--seed 731`, 9 GPUs, same compiler revision, MCTS
arm first. The hybrid arm added 4 knob proposals per target, each reserving a candidate slot before MCTS. MCTS
completed 9/9 targets with 145 benches; hybrid 9/9 with 171. On the O1 ranking lane MCTS produced the better candidate
on 6 targets, hybrid on 1, with 2 ties.

The O1 ranking lane misranks this model badly, so every conclusion below is from repeated deployable O3 measurement:

| Target | Candidate | O3 run 1 | O3 run 2 |
| --- | --- | ---: | ---: |
| `model-seam.k_linear_db1eb0` | greedy (isolated) | 2,854.9 µs | 2,855.9 µs |
| | MCTS winner `TILE=f2x8, WORK=t64x8` | 2,804.7 µs | 2,510.8 µs |
| | **hybrid proposal `REDUCE=coop, WORK=t128`** | **1,014.8 µs** | **902.1 µs** |
| `layer0.i003.k_linear_reduce_f6a146` | greedy (isolated) | 910.3 µs | 910.3 µs |
| | MCTS winner `TILE=f1x2, WORK=t32x8` | 2,526.2 µs | 2,520.1 µs |
| | hybrid `REDUCE=coop, TILE=f2x2, WORK=t16x8` | 2,367.5 µs | 2,487.3 µs |
| `layer3.i064.k_linear_reduce_f6a146` | greedy (isolated) | 912.4 µs | 909.3 µs |
| | MCTS winner `TILE=f1x2, WORK=t32x8` | 2,401.3 µs | 2,522.1 µs |

The hybrid proposal wins decisively on the one target with real headroom — 2.8–3.2× faster than greedy and about
2.5× faster than the MCTS-only winner — and it is exactly the hypothesis the `g2a` evidence suggested: drop `g2a`
and use the cooperative reduce with a thread work inventory. It is promoted into the golden with its two O3
measurements.

The eight `k_linear_reduce_f6a146` realizations keep their `g2a` knobs. No searched or proposed candidate beat
greedy there, and once the validator fix let the incumbent bench, `g2a` measured as a tie with greedy on all
eight — so the recorded configuration stands and only its stale measurements needed refreshing.

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
against 2026-08-11 changes host and driver together. No Emmy lane exists for this checkpoint, so no compiler-versus-
stock comparison is available.
