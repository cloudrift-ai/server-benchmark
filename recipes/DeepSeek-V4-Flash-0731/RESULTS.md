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

The [canonical V100 golden](golden/v100_sm70.yaml) contains 279 exact Loop realizations across 13 programs, each with
positive deployable O3 and reference timings. Coverage spans the layer paths and the non-layer seams (`layer0`,
`layer2`, `layer3`, `layer4`, and `model-seam` targets).

Verified on 2026-08-19 against the exact target hardware:

| Gate | Result |
| --- | --- |
| Repository-level validation | passes |
| Strict decode of every realization | 279 / 279 |
| Paired positive O3 / reference timings | 279 / 279 |
| Whole-file replay on Tesla V100-SXM3-32GB (sm_70) | all 279 targets reconstructed and lowered, exit 0 |

`emmy run --golden recipes/DeepSeek-V4-Flash-0731/golden/v100_sm70.yaml` completed in 4 min 19 s with no compile,
lowering, or execution failure, using the CUDA 12.9 toolchain (CUDA 13 cannot target sm_70). Emmy serving eligibility
is unchanged, and this compiler evidence does not establish an Emmy serving path for the checkpoint.

Reaching that verdict required fixing the replay itself. `emmy run --golden` re-read and re-validated the whole
document once per target; at roughly 15 s per load for this 3.4 MB inventory, a 279-target replay spent over an hour
re-parsing the same file before doing any useful work. That is consistent with the 2026-08-11 note that the in-model
audit returned no verdict within a bounded 106-minute replay, which was attributed to the Loop splicer. The replay now
parses the document once and shares it across targets.

Two host conditions also matter for anyone reproducing this: Emmy needs `nvcc` on `PATH`, and the CUDA 12.9 toolkit
must be selected because CUDA 13 dropped Volta; and PyYAML silently falls back to its pure-Python loader when
`libyaml` is absent, which alone cost more than 13 minutes per parse on this host before it was corrected.

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
