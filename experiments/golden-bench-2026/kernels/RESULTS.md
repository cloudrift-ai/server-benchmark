# Golden-bench kernel corpus: tuned golden datasets on RTX 4090 / RTX 5090 / V100 SXM3

## What this directory now contains

Per-GPU **tuned golden datasets** for the corpus models, produced so compiler improvements have measured baselines
to beat — not to claim wins. Each `goldens/<model>_<gpu>.yaml` embeds every post-fusion kernel target of the
model's layer 0 (seq_len 512 and 1 as separate realizations), the **deployed knobs after tuning**
(single-kernel targets; multi-kernel targets record `knobs: {}` and their per-kernel breakdown lives in the
archives), and **median 3-repeat `-O3` measurements**: `emmy_us` vs `reference_us` (`torch-eager`), losses
included. Raw evidence per platform: `results_<platform>.tar.gz` (traces, tune logs with search feedback,
per-target verification JSONs incl. `torch.compile` numbers, system info, progress log).

Run: 2026-08-19/20, revision `c1abcc91`, recipe budgets (`--max-candidates 12/8, --patience 4/3, --seed 0`),
one caller-supplied single-GPU host per platform. Model x GPU cells follow raw memory fit, not the recipe's GPU
matrix (per request): 0.6B + 0.6B-FP8 everywhere; 32B-FP8 on 4090 + V100; Qwen3.6-27B attempted on V100 only.

## Layer-level summary (sum over kernel targets, median of 3 `-O3` repeats, µs)

`tcompile` is `torch.compile(mode="max-autotune")` and its column sums only the kernels where its lane measured
(it fails on some SDPA targets), so ratios are approximate but directionally unambiguous.

| platform | model / seq | eager | tcompile | emmy (tuned deploy) |
| --- | --- | ---: | ---: | ---: |
| rtx5090x1 | 0.6B s512 / s1 | 918 / 467 | 181 / 42 | 32868 / 11108 |
| rtx5090x1 | 0.6B-FP8 s512 / s1 | 1789 / 1208 | 153 / 23 | 28835 / 482 |
| rtx4090x1 | 0.6B s512 / s1 | 1027 / 444 | 210 / 40 | 35285 / 9988 |
| rtx4090x1 | 0.6B-FP8 s512 / s1 | 2287 / 1452 | 173 / 10 | 34540 / 759 |
| rtx4090x1 | 32B-FP8 s512 / s1 | 57546 / 48455 | 259 / 19 | 650274 / 2957 |
| v100x1 | 0.6B s512 / s1 | 4418 / 974 | 899 / 59 | 71727 / 17246 |
| v100x1 | 0.6B-FP8 s512 / s1 | 7367 / 3320 | 821 / 30 | 147232 / 2628 |
| v100x1 | 32B-FP8 s512 / s1 | 24357 / 55282 | 3876 / 45 | 408201 / 16696 |

Reading: tuning improves individual reachable kernels (agent-seeded warp/staged proposals were injected per
linear target before each s512 tune), but the layer totals stay 1-3 orders behind `torch.compile` because the
dominant kernels cannot enter the search at all (below). These datasets are the baselines that make that gap
concrete per kernel, per card.

## Failures and defects surfaced (all reproduced in the archives)

1. **MCTS crash on the 32B-FP8 s512 corpus, both sm_89 and sm_70**: `AttributeError: 'NoneType' object has no
   attribute 'items'` at `emmy/compiler/pipeline/search/policy/mcts.py:518` (`_node_key`), mid-search after ~5
   shapes. Deterministic across cards on `c1abcc91`. Tune logs: `tune-qwen3-32b-fp8-s512.log` in the rtx4090/v100
   archives.
2. **Qwen3.6-27B is untraceable**: `ValueError: cannot map fallback aten.conv1d as elementwise for output shape
   (1, 10240, 515)` (`emmy/compiler/trace/torch.py`). Frontend coverage gap; no 27B golden exists.
   Log: `trace-qwen36-27b-s512.log` in the v100 archive.
3. **Known schedule lockouts dominate the totals** (prior sessions' diagnosis, still true on `c1abcc91`): the
   q/k-proj kernels fuse a reshape-to-heads that splits the output axis, so the warp tiler offers no rows and the
   contraction demotes to a scalar loop; SDPA online-softmax splices carry no structural identity and every
   candidate dies at the `005_stamp` assertion. The goldens record these kernels' measured scalar deploys as the
   baseline to improve.
4. **V100 32B-FP8 s512 verification exited nonzero on all 3 repeats** while still writing 16/19 target JSONs;
   the 4 unmeasured realizations are kept (inventory-only) in `qwen3-32b-fp8_v100.yaml`, which therefore
   validates at WORKING level; the other seven golden files pass REPOSITORY validation.
5. Host provisioning traps (recorded for reuse): fresh CloudRift boxes need `apt-get update` before
   `python3.12-venv`; V100 needs torch cu126 + `cupy-cuda12x==13.6.0` + `fastrlock` (nvrtc 13 dropped Volta).

## Platform a100x1 — replayed tuned golden (2026-08-21)

### Question

Does a fully tuned committed golden change the A100 picture for the common corpus, and which kernels does the
tuner still fail to reach? This is the first `common` row that replays a committed golden instead of searching
inside the recipe, so the searched schedules and the measured schedules are the same by construction.

### Protocol

Both rows replay `golden/qwen3-06b-s{1,512}_a100.golden.yaml` and run five fresh `emmy run --golden ... --bench
--strict --bench-backends eager,tcompile,emmy --warmup 10 --iters 100` processes with an empty per-task tuning DB,
online checkpoint, and cubin cache. The goldens came from a hybrid search on the same A100: `emmy trace` inventory,
agent-proposed warp/staged and cooperative rows from the card's own knob vocabulary, `emmy tune --max-candidates 48
--patience 12 --seed 0` per target with a 900-second per-target wall budget, then a deployable `-O3` A/B of the
searched winner and every measured proposal against a cold greedy pick. A realization is committed only when it
replays a knob map that every one of the target's CUDA kernels realized, and only when it beat that cold greedy
pick; the other targets keep an inventory realization and deploy the greedy schedule.

The attention statistics targets re-keyed twice while fusion learned to refuse a per-step statistic chained into a
fold inside a free sweep, and again when a computing sink stopped absorbing the same splice. At sequence 512 the
target now splits into two RoPE gather kernels, a k-norm/RoPE kernel, and an mma score kernel; at sequence 1 into a
q/k-norm reduce and a score kernel. Each re-keyed target was tuned fresh; every other realization carried its
earlier committed schedule over verbatim and was re-measured beside the fresh ones.

### Per-kernel result (median of five processes, µs)

`untuned` is the cold greedy pick re-benched in the same process; `tuned` is the committed realization where one
exists and the greedy pick otherwise.

| seq | target | role | eager | torch.compile | untuned | tuned | vs eager | vs tcompile |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | k_mean_20f978 | input RMSNorm | 191.66 | 8.76 | 3.83 | 3.17 | 60.53x | 2.77x |
| 512 | k_linear_reduce_06a42b | v_proj | 12.89 | 13.40 | 74.59 | 17.44 | 0.74x | 0.77x |
| 512 | k_linear_1fd3d5 | q_proj + reshape to heads | 100.50 | 21.50 | 132.68 | 22.32 | 4.50x | 0.96x |
| 512 | k_linear_a09c5a | k_proj + reshape to heads | 59.86 | 14.73 | 69.85 | 17.27 | 3.47x | 0.85x |
| 512 | k_unsqueeze_636992 | RoPE cos gather | 1.49 | 1.48 | 1.47 | 1.32 | 1.13x | 1.12x |
| 512 | k_unsqueeze_636992.unsqueeze_1 | RoPE sin gather | 1.48 | 1.54 | 1.46 | 1.32 | 1.12x | 1.17x |
| 512 | k_sdpa_linear_reduce_c0a378 | softmax·V, computed V | 45.70 | 46.08 | 159.57 | 159.57 | 0.29x | 0.29x |
| 512 | k_linear_sdpa_reduce_e24efe | o_proj + residual | 60.19 | 61.27 | 315.73 | 315.73 | 0.19x | 0.19x |
| 512 | k_linear_mean_reduce_dc067d | post-attn norm + gate/up + SiLU | 246.47 | 49.94 | 67.17 | 67.24 | 3.67x | 0.74x |
| 512 | k_linear_6b4b5f | down_proj + residual | 30.38 | 35.50 | 243.46 | 46.52 | 0.65x | 0.76x |
| 512 | k_mean_6cb2c7 | k-norm + RoPE | 271.52 | failed | 6.34 | 5.43 | 50.04x | — |
| 1 | k_mean_b8e46d | input RMSNorm | 121.81 | 2.87 | 3.07 | 2.38 | 51.22x | 1.20x |
| 1 | k_linear_reduce_7ef15d | v_proj | 7.56 | 7.52 | 11.15 | 4.74 | 1.59x | 1.59x |
| 1 | k_linear_49a16b | q_proj + reshape to heads | 35.11 | 6.77 | 21.57 | 4.84 | 7.26x | 1.40x |
| 1 | k_linear_dfb21f | k_proj + reshape to heads | 33.63 | 5.25 | 11.13 | 4.74 | 7.09x | 1.11x |
| 1 | k_mean_37b56e | q/k norm + RoPE | 34.57 | 4.28 | 1.61 | 1.52 | 22.71x | 2.81x |
| 1 | k_sdpa_linear_reduce_d0f5c0 | softmax·V, computed V | 11.91 | 10.61 | 25.22 | 25.22 | 0.47x | 0.42x |
| 1 | k_linear_sdpa_reduce_14c8c7 | o_proj + residual | 14.05 | 12.16 | 36.86 | 36.86 | 0.38x | 0.33x |
| 1 | k_linear_mean_reduce_549927 | post-attn norm + gate/up + SiLU | 154.62 | 16.38 | 3615.74 | 393.22 | 0.39x | 0.04x |
| 1 | k_linear_2dcd0c | down_proj + residual | 9.92 | 7.70 | 33.97 | 9.45 | 1.05x | 0.81x |
| 1 | k_sdpa_mean_reduce_5dfb29 | scores + softmax stats | 300.65 | failed | 39.40 | 39.40 | 7.63x | — |

The sequence-512 `k_sdpa_mean_reduce_7968d0` mma score kernel is the twelfth target and measured in none of the five
repeats: `emmy run` exits at the first strict failure and `k_mean_6cb2c7` already fails on its missing Inductor
reference. Its cold greedy deploy timed 114.05 µs in the tuning A/B.

Layer totals as the sum of those medians: sequence 512 is 1022.13 eager, 254.19 Inductor (two targets missing), and
657.33 Emmy over the eleven measured targets; sequence 1 is 723.83 eager, 73.53 Inductor (one target missing), and
522.37 Emmy. Emmy's sequence-512 layer was 21744 µs before the attention fusion fixes and the re-tune, so the corpus
moved 33x; sequence 1 moved from 4028 untuned to 522.

### Repeat variation

Every measured target's five paired latencies agree to within 0.4% of their median, and the pinned realizations
reproduce their tuning-time -O3 measurement to within 0.5%.

### Conclusion

Emmy beats eager on both layer totals (1.56x at sequence 512, 1.39x at sequence 1) and beats or matches Inductor on
nine of the twenty-one measured targets. Every remaining loss is one of three structural cases:

1. `softmax·V` and `o_proj + residual` lower to three CUDA kernels each whose schedule knobs disagree, so the golden
   format — one knob map per realization — cannot carry any searched schedule and they deploy the greedy pick.
   `emmy eval variants` shows the softmax·V kernel offered only the scalar tier (seven configs, all
   `REDUCE@a2=coop` with `WORK` from `t4` to `t128`; no warp/mma row exists), while the o_proj kernel's searched
   pick is rank 3 of 10 at 1.72x the best measured config. A uniform pin does make the map conflict-free but only
   until the schedule splits a fourth, knob-free epilogue kernel out.
2. The sequence-1 fused post-attention norm + gate/up + SiLU still runs a 393 µs cooperative fold against 155 µs
   eager and 16 µs Inductor, even though tuning already took it from 3616 µs.
3. The remaining projection losses (v_proj and down_proj at 512, down_proj at 1) are code-generation quality: the
   correct staged warp tier is reached and still trails cuBLAS by 1.2–1.5x.

### Limitations

- Layer-0 evidence only, one model, one card; never a whole-model claim.
- Both rows report `failed`. `torch.compile` cannot compile the reference for the RoPE-bearing attention targets on
  PyTorch 2.13.0 (`InductorError: ValueError: The argument '((0)) + 64' is not comparable` from
  `sizevars._stride_vars`), so `--strict` rejects them in every repeat. Those targets are ordered last in both
  goldens so the rest still measure; at sequence 512 the second such target is therefore never reached.
- The Inductor column is missing for those targets, so a geometric mean over the full corpus is not available; the
  measured denominators are stated above.
- The five repeats share one deployed host and run back to back, so they capture process-level, not day-level,
  variation.

### Run and system

- Status: failed (2/2 rows, `torch.compile` reference unavailable on the RoPE targets; all other targets measured)
- Result timestamp: 2026-08-21T10:01:28Z; run ID: `20260821T100128Z`
- Rows: `...sl1_scommon` (row ID `551082cef77b`, 493.44 s) and `...sl512_scommon` (row ID `3a4d139974b8`, 1891.78 s)
- Git revision: `e2f61e2e`; dirty: false
- Host: `riftvm`; Ubuntu 24.04.1 LTS; kernel `6.8.0-51-generic`; AMD EPYC 7742 64-Core Processor
- GPU: NVIDIA A100-SXM4-80GB, UUID `GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`
- Host NVCC `12.9.86`; cuBLAS `12.9.1.4`; PyTorch 2.13.0+cu130

### Durable files

- Raw-results archive: `results_a100x1.tar.gz`; archived root `2026-08-21_10-01-28/`
- Members: both `*.experiment.yaml` records, both `*_artifacts.tar.gz` task archives (per-repeat verification JSON
  per target, per-repeat logs and exit statuses, package freeze), and the two runner logs
- Committed goldens: `golden/qwen3-06b-s1_a100.golden.yaml`, `golden/qwen3-06b-s512_a100.golden.yaml`

## Platform sections

### rtx5090x1
0.6B + 0.6B-FP8 (large models excluded by RAM fit: 30 GB host). 4 traces, 4 tunes, 12 verification runs, zero
chain failures. Goldens: `qwen3-06b_rtx5090.yaml` (18/18 measured), `qwen3-06b-fp8_rtx5090.yaml` (38/38).
Archive: `results_rtx5090x1.tar.gz` (root `gb-rtx5090x1/`).

### rtx4090x1
0.6B, 0.6B-FP8, 32B-FP8. The 32B s512 tune crashed (defect 1); verification still measured every target, so its
golden records greedy/partially-tuned deploys. Goldens: `qwen3-06b_rtx4090.yaml` (18/18),
`qwen3-06b-fp8_rtx4090.yaml` (38/38), `qwen3-32b-fp8_rtx4090.yaml` (38/38).
Archive: `results_rtx4090x1.tar.gz` (root `gb-rtx4090x1/`).

### v100x1
0.6B, 0.6B-FP8, 32B-FP8 (no fp8-mma on sm_70 — FP8 checkpoints run the dequant path), 27B attempted (defect 2).
Goldens: `qwen3-06b_v100.yaml` (18/18), `qwen3-06b-fp8_v100.yaml` (38/38), `qwen3-32b-fp8_v100.yaml` (38
realizations, 34 measured). Archive: `results_v100x1.tar.gz` (root `gb-v100x1/`).

## Limitations

- Layer-0 evidence only; never a whole-model claim (lm_head/embedding uncovered).
- Tune-lane `-O1` rankings appear inside the working files' `ranking` blocks as search feedback; every number in
  the goldens' `measurements` and in this file is deployable `-O3`.
- The tcompile comparison column is partial where its lane failed; per-target values are in the verification JSONs.
- `knobs: {}` on multi-kernel targets is a format limitation, not "no schedule"; see the archives for per-kernel
  `record_knobs`.

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
