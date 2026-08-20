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
