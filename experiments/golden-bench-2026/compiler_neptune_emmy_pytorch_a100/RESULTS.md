# Neptune, modern PyTorch, and untuned Emmy on A100

## Conclusion

The starter comparison is feasible and reproducible, but it does not support a broad Neptune advantage over current
PyTorch. Across the three prefill families shared by both systems, PyTorch 2.13 Inductor was faster than the fastest
available Neptune schedule by 11–20% geometric mean. Neptune was approximately tied on ordinary decode: its best
available schedules were 1.07x faster than Inductor, while its fixed manual schedules were 0.99x as fast.

This prefill result agrees with the Neptune paper once its baselines are separated correctly. The paper compares
Neptune with tensor compilers in Table 2 and with manually optimized libraries in Table 4. On A100, its library table
also reports that the best library beats Neptune on global, causal, and GQA prefill. The current PyTorch lane is a new
comparison with `torch.compile`, not a reproduction of the paper's tensor-compiler table.

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

## Alignment with the Neptune paper

The [Neptune paper](https://arxiv.org/abs/2510.08726) reports the geometric-mean speedup of Neptune over the fastest
manually optimized library in Table 4. To compare with that table, the replay values below use the arithmetic mean of
the 15 projected GPU times for each implementation and then the geometric mean across the eight sequence lengths.
This is the closest aggregation available in the durable Nsight exports to the paper's mean-of-15 kernel rule. Values
above 1.00x favor Neptune.

| Operator | Paper, A100 | This artifact replay | Difference |
| --- | ---: | ---: | ---: |
| Prefill global | 0.84x | 0.79x | -5.5% |
| Prefill causal | 0.81x | 0.79x | -2.7% |
| Prefill GQA | 0.80x | 0.78x | -3.1% |
| Decode causal | 0.99x | 1.05x | +6.1% |
| Decode GQA | 1.24x | 1.19x | -3.9% |
| Prefill windowed | 0.70x | 0.68x | -2.3% |

The six comparable library results agree within 2.3–6.1%. In particular, both the paper and this replay find that
optimized libraries beat Neptune on the three common A100 prefill operators, while Neptune is competitive on causal
decode and ahead on GQA decode. The paper used an A100-SXM4-40GB, whereas this run used an A100-SXM4-80GB; the GPUs
have the same compute architecture but different memory systems, so exact latency equality is not expected.

The paper's Table 2 reports Neptune relative to Triton, FlexAttention, TVM, and Mirage, rather than to the manually
optimized libraries. The pinned artifact revision leaves its TVM runners disabled, so this experiment cannot claim a
complete reproduction of every Table 2 cell. Its PyTorch 2.6 runners also select specialized SDPA, cuDNN, or CUTLASS
paths; they are not equivalent to the full-graph PyTorch 2.13 lane added here.

The paper does not publish per-shape latency tables or raw plot data. Its absolute attention results are throughput
plots at sequence length 8192 over varying batch sizes. Representative absolute minima from this run are below,
reported as `Neptune / torch.compile` in microseconds. These use the experiment's minimum-of-15 convention, not the
paper-style means in the alignment table.

| Operator | Sequence 2048 | Sequence 32768 |
| --- | ---: | ---: |
| Prefill global | 479.8 / 364.5 | 83,978.9 / 75,066.4 |
| Prefill causal | 298.0 / 243.7 | 45,798.3 / 40,020.0 |
| Prefill GQA | 488.7 / 405.5 | 91,127.6 / 79,070.2 |
| Decode causal | 35.3 / 30.7 | 320.7 / 322.6 |
| Decode GQA | 15.8 / 41.0 | 111.1 / 503.8 |

The absolute trend is coherent with the ratios: prefill remains close but favors current PyTorch, causal decode
converges toward parity, and Neptune's decode-GQA advantage grows with context length.

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

SoftCap decode is the remaining performance-reproduction outlier. This replay reports 4.96x over Flex, while the
paper's A100 compiler table reports 1.86x over its best compiler baseline. The modern PyTorch lane does not implement
SoftCap, so this result should not be treated as reproduced until that difference is explained.

## Tuned Emmy on the same A100 (2026-08-21)

### Question

The 2026-08-16 run measured UNTUNED Emmy. This run replays a committed tuned golden per setup, so the schedules
Emmy deploys are the ones a full hybrid search selected on this exact card.

### Protocol

Each setup's program was traced from `operators.sh`, hybrid-tuned (agent proposals drawn from this card's own
measured schedules, then MCTS at 48 candidates / patience 12 / seed 0 under a per-setup wall budget), and its
deployable `-O3` finalists re-measured against a cold greedy pick; the winner is committed under
`golden/<operator>-b1-s<sequence_length>.golden.yaml`. The lane then runs two arms per setup: the golden replay
times Emmy alone, and an `emmy run -c` reference arm re-traces the same snippet for eager, torch.compile, and the
strict Emmy-vs-eager correctness proof. Both arms use one warmup and 15 measured iterations and take the minimum,
matching the earlier run and the Neptune convention. Compiler revision `0f9c3026`+ (goldens) / `080bde08` (lane).

### Result (µs, minimum of 15; Neptune from the archived 2026-08-16 profiles)

`emmy tuned` sums each setup's post-fusion targets, taking the committed pin where one exists and the greedy pick
otherwise. Ratios above 1.00x favour Emmy.

| setup | eager | torch.compile | emmy untuned | emmy tuned | Neptune | tc/emmy | Neptune/emmy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode_causal 256 | 12.24 | 12.18 | 25.77 | 23.46 | 10.91 | 0.52x | 0.47x |
| decode_causal 512 | 16.01 | 15.96 | 44.48 | 40.96 | 14.59 | 0.39x | 0.36x |
| decode_causal 1024 | 21.02 | 20.84 | 84.57 | 78.92 | 21.38 | 0.26x | 0.27x |
| decode_causal 2048 | 29.37 | 28.89 | 299.52 | 158.16 | 35.26 | 0.18x | 0.22x |
| decode_causal 4096 | 52.99 | 54.39 | 593.92 | 585.73 | 57.85 | 0.09x | 0.10x |
| decode_causal 8192 | 91.96 | 92.77 | 1184.77 | 1174.53 | 94.53 | 0.08x | 0.08x |
| decode_causal 16384 | 167.94 | 167.42 | 2361.34 | 2336.77 | 165.69 | 0.07x | 0.07x |
| decode_causal 32768 | 318.46 | 318.46 | 4713.47 | 4668.42 | 320.74 | 0.07x | 0.07x |
| decode_gqa 256 | 68.85 | 12.59 | 21.14 | 47.53 | 10.34 | 0.26x | 0.22x |
| decode_gqa 512 | 100.17 | 21.40 | 27.44 | 61.64 | 10.75 | 0.35x | 0.17x |
| decode_gqa 1024 | 186.71 | 25.84 | 36.65 | 85.04 | 12.58 | 0.30x | 0.15x |
| decode_gqa 2048 | 348.62 | 33.12 | 62.52 | 147.49 | 15.78 | 0.22x | 0.11x |
| decode_gqa 4096 | 670.23 | 71.90 | 138.68 | 334.97 | 22.98 | 0.21x | 0.07x |
| decode_gqa 8192 | 1308.21 | 115.76 | 320.51 | 816.98 | 40.10 | 0.14x | 0.05x |
| decode_gqa 16384 | 2663.38 | 236.81 | 629.25 | 1662.98 | 67.61 | 0.14x | 0.04x |
| decode_gqa 32768 | 5869.13 | 448.22 | 1403.90 | 3979.26 | 111.07 | 0.11x | 0.03x |
| prefill_global 256 | 16.13 | 15.91 | 93.37 | 32.21 | 19.01 | 0.49x | 0.59x |
| prefill_global 512 | 43.26 | 42.89 | 165.89 | 89.97 | 50.37 | 0.48x | 0.56x |
| prefill_global 1024 | 122.37 | 120.32 | 3254.27 | 597.45 | 153.82 | 0.20x | 0.26x |
| prefill_global 2048 | 342.02 | 330.75 | 12595.20 | 1990.66 | 479.81 | 0.17x | 0.24x |
| prefill_global 4096 | 1269.76 | 1256.45 | 49434.62 | 7092.22 | 1586.33 | 0.18x | 0.22x |
| prefill_global 8192 | 4728.83 | 4728.83 | 197223.42 | 197455.88 | 5988.16 | 0.02x | 0.03x |
| prefill_global 16384 | 18829.31 | 18853.89 | 783868.96 | 711223.27 | 23607.12 | 0.03x | 0.03x |
| prefill_causal 256 | 19.35 | 19.07 | 95.88 | 37.17 | 20.89 | 0.51x | 0.56x |
| prefill_causal 512 | 49.15 | 49.15 | — | 103.95 | 43.17 | 0.47x | 0.42x |
| prefill_causal 1024 | 100.86 | 96.77 | 3297.28 | 627.66 | 102.11 | 0.15x | 0.16x |
| prefill_causal 2048 | 272.38 | 272.38 | — | 2021.38 | 297.98 | 0.13x | 0.15x |
| prefill_causal 4096 | 772.10 | 766.98 | 49837.06 | 7477.25 | 887.29 | 0.10x | 0.12x |
| prefill_causal 8192 | 2959.36 | 2984.96 | — | 198784.00 | 3499.08 | 0.02x | 0.02x |
| prefill_causal 16384 | 10240.00 | 10198.02 | — | 711098.39 | 13229.56 | 0.01x | 0.02x |
| prefill_gqa 256 | 31.74 | 31.74 | — | 65.26 | 28.61 | 0.49x | 0.44x |
| prefill_gqa 512 | 58.37 | 57.09 | 546.82 | 328.19 | 64.10 | 0.17x | 0.20x |
| prefill_gqa 1024 | 152.58 | 150.53 | — | 1043.46 | 173.06 | 0.14x | 0.17x |
| prefill_gqa 2048 | 424.96 | 411.65 | 5637.12 | 3658.75 | 488.70 | 0.11x | 0.13x |
| prefill_gqa 4096 | 1531.90 | 1531.90 | — | 10569.73 | 1667.73 | 0.14x | 0.16x |
| prefill_gqa 8192 | 5177.34 | 5180.42 | — | 354388.98 | 6796.48 | 0.01x | 0.02x |
| prefill_gqa 16384 | 20008.96 | 19987.46 | 1574933.47 | — | 26118.12 | — | — |

The three sequence-32768 prefill setups and `prefill_gqa 16384` produced no Emmy replay: the lane's ten-minute
per-setup budget expires while the kernel is still running. Their goldens exist and their targets are pinned, so
they are unmeasured rather than untuned.

### Conclusion

Two things changed against the 2026-08-16 run. All sixteen decode setups now pass strict Emmy-vs-eager
correctness — the earlier failure was an SDPA decomposition that read the key length off the query, so decode
scored only the first key — and tuning is worth 1.2x to 6.6x on the prefill families it reaches
(`prefill_global 2048` 12595 → 1991 µs, `prefill_causal 4096` 49837 → 7477 µs).

Emmy is nevertheless behind on every one of the 36 measured setups: 0.03x–0.59x of Neptune and 0.01x–0.52x of
current torch.compile. The gap is smallest at sequence 256–512 (roughly 0.5x) and widens with context, because
Emmy's attention still materializes the full score matrix — its two post-fusion targets are a scores/statistics
kernel and a softmax·V kernel — while Neptune, FlashInfer, and the SDPA library backends all run a fused streaming
attention that never writes it. That is an algorithm the compiler does not yet express, not a schedule the tuner
can find: correcting the decode key length made decode correct and 3–10x slower than the incorrect version, which
is the same wall from the other side.

### Limitations

- One card, one batch size, one head dimension; minimum-of-15 with a single warmup, matching the artifact.
- The Neptune column is replayed from the 2026-08-16 archive rather than re-measured, so it carries that run's
  software stack (PyTorch 2.6.0 inside the pinned image) while this arm is PyTorch 2.13.0.
- The `emmy untuned` column comes from the reference arm's greedy deploy in the same process; it is blank where
  that arm failed strict and the PyTorch-only fallback supplied eager and Inductor.
- Four setups are unmeasured on the Emmy side (above); their rows are not evidence of a result either way.

### Run and system

- Status: succeeded (5/5 rows)
- Result timestamp: 2026-08-21T16:20:56Z; run ID: `20260821T162056Z`
- Rows: decode_causal `9f8816b4a4fb`, decode_gqa `a5484cfcec5d`, prefill_causal `606182d43644`,
  prefill_global `c40c2a24baf1`, prefill_gqa `2df1673a2808`
- Git revision: `080bde08`; dirty: false
- GPU: NVIDIA A100-SXM4-80GB, UUID `GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`; PyTorch 2.13.0+cu130
- Raw-results archive: `results_emmy_a100x1.tar.gz`; archived root `2026-08-21_16-20-56/`

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
