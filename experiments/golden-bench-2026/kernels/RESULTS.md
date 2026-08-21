# Golden-bench kernel corpus: retuned golden datasets after the first fix wave (main @ 001d4f44)

## What this refresh is

Second measurement pass over the corpus, after the maintainer's fixes landed (#561 split-axis re-fusion, #556
conv1d/einsum lowering, #549/#547 FA restoration, #513-era search changes). Same three GPUs, fresh hosts,
recipe budgets, three `-O3` repeats per target. Each `goldens/<model>_<gpu>.yaml` was rebuilt from a fresh
trace + tune + 3-repeat verification on `001d4f44`; the previous values remain in git history for comparison.

Runs were driven by an updated benchmark flow — one row per committed golden on its exact GPU, three `-O3`
repeats of `emmy run --golden <file> --bench --bench-backends eager,tcompile,emmy`, no tracing or tuning at
measurement time — kept UNCOMMITTED in this PR per review direction; `recipe.yaml` in-tree is unchanged.

## Before/after (sum of measured kernel targets, median of 3 repeats, µs; old = previous committed goldens)

| platform | model/seq | old emmy | new emmy | gain | new vs eager |
| --- | --- | ---: | ---: | ---: | ---: |
| v100x1 | 0.6B s512 | 71714 | 58981 | 1.2x | 0.07x |
| v100x1 | 0.6B-FP8 s512 | 147160 | 71084 | 2.1x | 0.10x |
| v100x1 | 32B-FP8 s512 | 408241 | 395773 | 1.0x | 0.06x |
| rtx4090x1 | 0.6B s512 | 35305 | 34466 | 1.0x | 0.03x |
| rtx4090x1 | 0.6B-FP8 s512 | 34496 | 24525 | 1.4x | 0.04x |
| rtx4090x1 | 32B-FP8 s512 | 650248 | 117839 | **5.5x** | 0.08x |
| rtx5090x1 | 0.6B s512 | 32870 | 27620 | 1.2x | 0.03x |
| rtx5090x1 | 0.6B-FP8 s512 | 28830 | 17843 | 1.6x | 0.10x |

Decode (s1) rows improved 4-13x on the FP8 corpora but changed measurement coverage (19 -> 8-11 targets, from
new fusion identity plus bench failures), so their sums are not clean ratios; per-target values are in the
archives. Matched-kernel gains on the V100 0.6B corpus: geomean 2.6x, led by q_proj 4.5x (992 -> 219 µs, 0.88x
eager on Volta — #561's tensor-core unlock confirmed in silicon), k/v_proj 3.4x, and the SDPA matmul fusions
21.5x / 16.8x (FA restoration).

## Why layer totals still trail torch.compile: the remaining defects, diagnosed

1. **RoPE-fusion statistic replay (dominates every s512 total; unchanged).** `k_sdpa_mean_reduce`'s loop nest
   recomputes the k-norm statistic (a full 512x128 reduce) inside every q-row iteration — a 512x replay
   (~23000-29300 µs of each card's s512 total; the s1 variant is fine, replay factor 1). Consistent with the
   #513 guard removal enumerating this fused form and cold greedy deploying it. Fix directions: hoist
   loop-invariant statistics in loop/canonicalize, or make the placement-cut alternative evidence-reachable
   cold. Diagnosis-only here per review direction.
2. **Computed-A (fused norm+gate/up) misdeploys, worst at decode.** New extreme case: on the rtx4090,
   `k_linear_mean_reduce_549927.s1` deploys KNOBLESS at **116445 µs vs eager 108** (~1000x); the V100 s512
   sibling regressed 679 -> 1035 µs. The search reaches no schedule for this form and the fallback is
   catastrophic.
3. **Qwen3.6-27B capture advanced one op and is blocked again**: conv1d now lowers (#556), the trace now stops
   at `aten.masked_fill requires resolved self, mask, and fill inputs`. Still no 27B golden.
4. **Hung kernel under the 32B corpus on V100**: `k_mul_12__partial` exceeds the 2 s bench watchdog in the
   tuned deploy (16/19 targets measured around it).
5. **Unmeasured golden rows are real bench failures, kept as inventory**: fp8 files carry 11-22 unmeasured
   realizations each (hangs, compile failures, or the coverage change); only the 0.6B BF16 files validate at
   REPOSITORY level on all three cards, the rest at WORKING level.

## Environment caveats (hosts are rented and heterogeneous)

- The rtx4090 host ran an old driver (CUDA 12.2-era) and nvcc 12.1: the default cu130 torch cannot initialize
  (fixed with the cu126 wheel + matching cu12 libraries), and a subset of fp8 kernels fail to COMPILE under
  nvcc 12.1 that compiled under CUDA 13.3 on the first pass — its fp8 numbers carry that asterisk.
- V100 requires torch cu126 and `cupy-cuda12x==13.6.0` + `fastrlock` (nvrtc 13 dropped Volta), as before.
- Pre-run canaries must check BOTH `cupy.full` AND `torch.cuda.is_available()`; a cupy-only canary passed on
  the old-driver 4090 while every torch-side measurement failed.

## Platform sections

### v100x1 — full pipeline (rebench + retune, 4 models attempted, 27B blocked at trace)
Goldens: `qwen3-06b_v100.yaml` (18/18, REPOSITORY), `qwen3-06b-fp8_v100.yaml` (27/38), `qwen3-32b-fp8_v100.yaml`
(23/38). Archive: `results_v100x1.tar.gz`.

### rtx4090x1 — full pipeline; measurements re-run after the driver fix
Goldens: `qwen3-06b_rtx4090.yaml` (18/18, REPOSITORY), `qwen3-06b-fp8_rtx4090.yaml` (16/38),
`qwen3-32b-fp8_rtx4090.yaml` (19/38). Archive: `results_rtx4090x1.tar.gz`.

### rtx5090x1 — full pipeline on a replacement host (first instance had unstable SSH and a failing toolchain)
Goldens: `qwen3-06b_rtx5090.yaml` (18/18, REPOSITORY), `qwen3-06b-fp8_rtx5090.yaml` (27/38).
Archive: `results_rtx5090x1.tar.gz`. Large models excluded by RAM fit (30 GB host), as before.

## Limitations

Layer-0 evidence only; `-O3` numbers throughout; tcompile per-target values live in the archives (its lane
fails on some SDPA targets); s1 sums are not comparable across passes due to coverage changes; multi-kernel
targets record `knobs: {}` with per-kernel `record_knobs` in the archives.

## Platform h200x8 — earlier failed attempt (retained)

### Conclusion

The latest non-dry invocation failed before tracing, tuning, or kernel benchmarking began. It produced no latency,
correctness, or coverage measurements and supports no kernel-performance claim. The failure is retained because
dry-run validation is not a result.

### Protocol and failure

The invocation selected one `common` row on a pre-allocated host detected as eight NVIDIA H200 141GB GPUs. The task
used one GPU and targeted `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, layer 0, sequence length 512,
with search budget 12, patience 4, and seed 0.

Remote setup reached the command after staging a clean source tree. The command then failed because `make` was not
installed. Its exit trap encountered a second error because `task_dir` was unset, so the intended `artifacts.tar.gz`
was never created or retrieved. The command result records exit code 1 and the missing-result transfer error. The
runner summary is 0/1 successful tasks.

### System and provenance

- Hardware detected: NVIDIA H200 141GB x8; the task requested one GPU.
- Timestamp: `2026-08-13_16-08-20_e1c8d16a`.
- Git revision: `030b6d58182bb3da1748c4954d7d2fd0211e8d3b`; staged source was clean.
- Workload status: failed before measurement.
- The legacy command result has no assembled experiment record and no complete typed system-information section.

### Durable files

- Raw-results archive: `results.tar.gz`.
- Archived root: `2026-08-13_16-08-20_e1c8d16a/`.
- Supporting evidence: runner logs, the executed recipe snapshot, the task manifest, and the command result JSON are
  in the archive.
