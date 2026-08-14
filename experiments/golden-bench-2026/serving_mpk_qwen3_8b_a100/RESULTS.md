# MPK Qwen3-8B on A100

## Conclusion

Within Mirage's paired demo harness, the persistent-kernel path materially outperformed Mirage's own
kernel-per-operator path. Across five fresh-process repeats, mean reported per-token latency fell from 33.136 ms to
9.804 ms: a 3.38x speedup, or a 70.4% latency reduction. The repeat ranges do not overlap, so this result is clear for
the exact single-request Mirage workload tested here.

The stock vLLM reference reported 10.752 ms/token. Mirage's persistent-kernel latency was 8.8% lower, but this is only
directional: the Mirage and vLLM lanes used different prompts, generation lengths, termination rules, client paths,
and metric implementations. This run does not establish that MPK outperforms vLLM in an equivalent serving workload.

Correctness remains the main unresolved issue. The Mirage baseline produced the same 274-token response in every
repeat, while the persistent-kernel path produced 258–273 tokens with visible wording changes despite temperature
zero. The responses were coherent and on-topic on inspection, but the harness performed no token, logit, or numerical
equivalence check. The run therefore demonstrates a strong latency result, not correctness parity.

## Measurements

| Lane | Repeats | Reported latency, mean | Per-repeat range |
| --- | ---: | ---: | ---: |
| Mirage kernel-per-operator | 5 | 33.136 ms/token | 32.632–33.442 ms/token |
| Mirage persistent kernel | 5 | 9.804 ms/token | 9.785–9.865 ms/token |
| Stock vLLM | 5 | 10.752 ms/token | 10.74–10.76 ms/token |

All five stock vLLM repeats completed eight requests with zero failed requests. Its per-token latency range spans only
0.02 ms/token, while the persistent-kernel latency range spans 0.080 ms/token. The directly comparable Mirage baseline
and persistent-kernel lanes are therefore both repeatable enough that run-to-run variation cannot explain their gap.

## Protocol and limitations

- All lanes ran Qwen3-8B on the same NVIDIA A100-SXM4-80GB. Mirage was pinned to revision
  `5c28cc68dc621cc9448c5c9882ef9e21fdc85884`; stock vLLM used version 0.23.0 and model revision
  `b968826d9c46dd6066d109eabc6255188de91218`.
- The two Mirage lanes are the controlled comparison: one fixed 39-token prompt, one request, natural EOS, and the
  same demo with only `--use-mirage` changed. Each repeat used a fresh Python process.
- The stock lane used the HTTP serving path with eight sequential requests per repeat, fixed 128-token inputs, fixed
  512-token outputs, concurrency one, and EOS ignored. Each repeat started a fresh vLLM server.
- Mirage's model load did not explicitly pass the recorded model revision. Both Mirage lanes shared the same cached
  model, preserving their paired comparison, but exact model-revision parity with stock vLLM is not proven.
- The two harnesses do not define per-token latency identically, and the vLLM client/server path adds work absent from
  the in-process Mirage demo. Their cross-system comparison should not be treated as a serving claim.
- Every persistent-kernel repeat compiled successfully but emitted the same NVCC warnings, including a warning about
  dynamic initialization of function-scope static shared memory. No runtime failure followed, but the run contains no
  separate validation of that warning's effect.

## Run and system

- Status: succeeded
- Timestamp: 2026-08-14T20:24:20Z
- Run ID: `20260814T202420Z`
- Experiment row: `serving_mpk_qwen3_8b_a100/a100x1`
- Git revision: `9a485df4229e0529720a3b46e1d2fc482e97a394`; dirty: false
- Host: `riftvm`; Ubuntu 24.04.1 LTS; kernel `6.8.0-51-generic`
- CPU: AMD EPYC 7742 64-Core Processor, x86_64, 15 logical CPUs; memory: 221634367488 bytes
- GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, UUID `GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`
- NVIDIA driver: `580.65.06`; NVCC: `12.9.86`; cuBLAS: `12.9.1.4`
- Docker client/server: `28.5.1` / `28.5.1`

## Durable files

- Experiment record: `a100x1_e246bb6279fd.experiment.yaml`
- Raw-results archive: `results.tar.gz`
- Archive root retained locally for inspection: `2026-08-14_20-24-20/`
- Raw measurements: `a100x1_mpk_base_r0.txt` through `a100x1_mpk_base_r4.txt`,
  `a100x1_mpk_mega_r0.txt` through `a100x1_mpk_mega_r4.txt`, and `a100x1_stock_r0.txt` through
  `a100x1_stock_r4.txt`
- Supporting evidence: runner/server logs, environment freezes, model-cache inventory, and MPK installation log are
  included in the archive.
