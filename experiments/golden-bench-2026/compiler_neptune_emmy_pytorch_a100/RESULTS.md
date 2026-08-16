# Neptune, modern PyTorch, and untuned Emmy on A100

## Conclusion

The starter comparison is feasible and reproducible, but it does not support a broad Neptune advantage over current
PyTorch. Across the three prefill families shared by both systems, PyTorch 2.13 Inductor was faster than the fastest
available Neptune schedule by 11–20% geometric mean. Neptune was approximately tied on ordinary decode: its best
available schedules were 1.07x faster than Inductor, while its fixed manual schedules were 0.99x as fast.

Decode GQA is the clear Neptune result. Neptune was 3.01x faster than Inductor by geometric mean across all eight
sequence lengths, with per-shape speedups from 1.68x to 4.54x. Inductor itself was 7.67x faster than eager PyTorch on
that family, so Neptune's advantage remains after using a recent compiler baseline rather than the PyTorch 2.6 stack
inside the published artifact.

Untuned Emmy is not competitive in this run, as expected. It produced correct captured timings for 22 of 24 prefill
setups, but those rows were much slower than PyTorch. Its two largest causal/GQA prefill kernels tripped the watchdog,
and all 16 decode setups failed strict eager correctness. Those failures are retained as results, not silently dropped;
an independent PyTorch fallback preserved eager and Inductor measurements for every affected setup.

## Common operator measurements

The Neptune columns report `Inductor latency / Neptune latency`, so values above 1.00x favor Neptune. "Best" selects
the fastest measured manual or tuned Neptune schedule; "manual" uses only the artifact's fixed manual schedules. Each
summary is the geometric mean over eight sequence lengths from 256 through 32768.

| Operator | Inductor vs eager | Neptune vs Inductor, best | Neptune vs Inductor, manual | Full tunes | Valid Emmy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill global | 1.02x | 0.84x | 0.82x | 8/8 | 8/8 |
| Prefill causal | 1.05x | 0.90x | 0.89x | 5/8 | 7/8 |
| Prefill GQA | 1.05x | 0.86x | 0.84x | 6/8 | 7/8 |
| Decode causal | 1.01x | 1.07x | 0.99x | 7/8 | 0/8 |
| Decode GQA | 7.67x | 3.01x | 2.90x | 8/8 | 0/8 |

Best-available Neptune beat Inductor on 15 of the 40 individual shapes: two prefill, five decode-causal, and all eight
decode-GQA setups. The decode-GQA speedup increased with context, from 1.68x at sequence 256 to 4.54x at sequence
32768. By contrast, Neptune's best prefill results ranged from 0.76x to 1.19x Inductor across individual shapes.

## Published artifact coverage

All ten operator families and all eight sequence lengths produced Nsight profiles. Sixty-four of 80 tuning jobs
completed their 128-trial search; 16 reached the 30-minute per-setup limit. Timed-out rows still profile the available
manual and partial tuned schedules and are not described as fully tuned.

The speedup column compares the fastest available Neptune schedule with the fastest valid non-Neptune runner in the
published artifact. Values above 1.00x favor Neptune. The artifact uses PyTorch 2.6, so this table characterizes
Neptune's original comparison environment; the modern Inductor comparison above is the more relevant baseline.

| Published operator | Full tunes | Profiles | Neptune vs fastest valid artifact runner | Validity note |
| --- | ---: | ---: | ---: | --- |
| Prefill global | 8/8 | 8/8 | 0.78x | Excludes the mismatching Tri Dao Triton rows at 256–1024 |
| Prefill causal | 5/8 | 8/8 | 0.78x | Excludes the mismatching Tri Dao Triton runner |
| Prefill GQA | 6/8 | 8/8 | 0.76x | Excludes the mismatching Tri Dao Triton runner |
| Decode causal | 7/8 | 8/8 | 1.04x | No cross-runner mismatch |
| Decode GQA | 8/8 | 8/8 | 1.21x | No cross-runner mismatch |
| Prefill ALiBi | 6/8 | 8/8 | Excluded | Flex and CUTLASS disagree with Neptune on all shapes |
| Decode ALiBi | 8/8 | 8/8 | Excluded | Flex and CUTLASS disagree with Neptune on all shapes |
| Prefill softcap | 4/8 | 8/8 | 1.04x | Compared with Flex |
| Decode softcap | 8/8 | 8/8 | 4.96x | Compared with Flex |
| Prefill windowed | 4/8 | 8/8 | 0.68x | Compared with CUTLASS |

The harness treats a Neptune manual schedule as its correctness reference. Agreement from the other runners supports
the non-ALiBi rows, but it is not an independent oracle for Neptune. ALiBi is therefore excluded rather than assigning
the disagreement to either side.

## Protocol and limitations

- Neptune ran revision `3aa55c12ac822337e630b809b0d9eabb11eee5d3` in the pinned image
  `evanzhao16/neptune-env@sha256:724d07594bc817f0fe94267b2d0dbdc6e29d3ae4a7e3516e553a6d9327bfebca`.
  The artifact environment recorded PyTorch 2.6.0 with CUDA 12.4 and Nsight Systems 2025.3.1.
- The common lane reconstructed global, causal, and GQA attention for prefill and decode through `emmy run -c ...
  --bench`. It used PyTorch 2.13.0 with CUDA 13.0, full-graph Inductor in `max-autotune-no-cudagraphs` mode, untuned
  Emmy, one warmup, 15 measured iterations, and strict correctness.
- When Emmy failed before producing a shared table, the fallback measured only eager and Inductor after checking the
  compiled output against eager at `rtol=1e-3, atol=1e-3`. All 40 Inductor rows passed, and every PyTorch timing used
  CUDA-graph-captured whole-forward semantics.
- Neptune latency is the minimum projected GPU time over 15 measured NVTX ranges. The PyTorch/Emmy lane uses the
  minimum CUDA-event time over 15 interleaved or fallback measurements. Both are GPU-time measurements, but they came
  from separate processes and software environments; cross-system ratios should be treated as kernel-level evidence,
  not an end-to-end application result.
- The softcap, ALiBi, and windowed families have no current PyTorch/Emmy twin in this experiment. Their table only
  reproduces the runners shipped in Neptune's artifact.

## Run and system

- Status: succeeded
- Result timestamp: 2026-08-16T00:41:38Z; run ID: `20260816T004138Z`
- Experiment row: `compiler_neptune_emmy_pytorch_a100_recovery/a100x1`; row ID: `e246bb6279fd`
- Git revision: `2550211d9c93e522ea4f9eb81e39735f4ab64d07`; dirty: false
- Host: `riftvm`; Ubuntu 24.04.1 LTS; kernel `6.8.0-51-generic`
- CPU: AMD EPYC 7742 64-Core Processor, x86_64, 15 logical CPUs; memory: 221634367488 bytes
- GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, UUID `GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`
- NVIDIA driver: `580.65.06`; host NVCC: `12.9.86`; host cuBLAS: `12.9.1.4`
- Docker client/server: `28.5.1` / `28.5.1`

The source run (`20260815T040818Z`) completed all Neptune work in 71393.66 seconds, then failed because the host lane
started outside the staged repository. The successful 2353.12-second recovery verified the immutable source archive's
SHA-256 (`775fb71d3eac78f0371c1014b9945b29d17f41202347f8115e8703db5a4c14ca`), retained its failed status, and ran
only the missing host lane. The durable `recipe.yaml` contains the corrected working directory for clean future runs.

## Durable files

- Experiment record: `a100x1_e246bb6279fd.experiment.yaml`
- Raw-results archive: `results.tar.gz`; SHA-256
  `617f44ee6c6c5cff6dc637d1924a8071b8bc547f6edbb77d3e61e7548bbfee03`
- Archived root: `2026-08-16_00-41-38/`
- Composite task artifact: `a100x1_artifacts.tar.gz`; SHA-256
  `015951d7cccf187c69dd2712bcaf966f3de179b53508942312a0e8e6cc31e4b5`
- Raw evidence includes 80 `.nsys-rep` profiles, 80 CSV exports, all tune/profile logs, 40 modern PyTorch JSON rows,
  Emmy dumps and logs, environment freezes, runner hashes, source/recovery status files, and both run records/logs.
