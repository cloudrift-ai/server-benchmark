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

## Platform a100x1 — latest tuned routing evidence (2026-08-23)

The A100 corpus now includes two positive routing realizations measured at deployable `-O3` on the exact
NVIDIA A100-SXM4-80GB (`GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`):

| seq | target | routing realization | Emmy | fused/cold Emmy | eager | result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 512 | score + softmax statistics | `PLACE@b=cut` | 299.69 µs | 21232 µs | 745.47 µs | 2.49x eager |
| 1 | post-attention norm + gate/up + SiLU | `PLACE@map=cut` | 66.97 µs | 4612 µs | 154.62 µs | 2.31x eager |

Both routing realizations passed strict Emmy-versus-eager correctness. All nine sequence-1 targets and seven of nine
sequence-512 targets now carry verified pins in the committed experiment goldens; the sequence-512 softmax·V and
post-attention norm + gate/up + SiLU targets remain unpinned.

This is direct tuning evidence, not yet a matching experiment snapshot. The recipe replays only these committed
goldens and performs no search, but the clean latest-`main` replay is still pending. The historical section and
`results_a100x1.tar.gz` below remain the 2026-08-21 run and do not measure the two new routing realizations.

## Platform a100x1 — historical chain-form replay (2026-08-21)

### Question

`main` now carries the chain root formation: a fold closes over the values its projection body defines, so the RoPE
gathers and the k-norm no longer survive as their own kernels — they become part of the score kernel's tree. That
re-keys every computed-A attention target. Which committed realizations survive the re-key, what do the re-keyed
targets cost once they are tuned again on this card, and — now that every fold is a node with a `PLACE` seam — does
any cut beat the fused form on the targets that lose?

### Identity diff

Both inventories were re-traced from `Qwen/Qwen3-0.6B@c1899de2` layer 0 and every target name was compared with the
committed golden. A surviving identity kept its committed knobs and measurements verbatim.

| seq | committed | re-traced | carried verbatim | re-keyed or unpinned | absorbed by the score kernel |
| --- | ---: | ---: | ---: | ---: | --- |
| 512 | 12 | 9 | 6 | 3 | RoPE cos gather, RoPE sin gather, k-norm + RoPE |
| 1 | 10 | 9 | 6 | 3 | q/k norm + RoPE |

The three re-keyed targets per sequence are the score/statistics kernel, softmax·V, and o_proj + residual. The last
two keep their names but had no committed schedule (their knob maps have never been recordable), so they were tuned
from scratch as well.

### Protocol

Each re-keyed target was hybrid-tuned on this card: agent proposals drawn from the card's own measured schedules plus
every `PLACE` seam the recognize rule enumerates, then `emmy tune --max-candidates 48 --patience 12 --seed 0` under a
per-target wall budget, with an isolated tuning DB, online checkpoint, and cubin cache. Every finalist was re-measured
at deployable `-O3` against the cold greedy pick, then verified in a fresh `emmy run --golden … --target … --bench
--strict` process. The recipe then replayed both committed goldens in five fresh
`emmy run --golden … --bench --strict --bench-backends eager,tcompile,emmy --warmup 10 --iters 100` processes with an
empty per-task tuning DB, online checkpoint, and cubin cache.

### Per-kernel result (median of five processes, µs)

`greedy` is the cold pick re-benched in the same process; `deployed` is the committed realization where one exists and
the greedy pick otherwise.

| seq | target | role | eager | torch.compile | greedy | deployed | vs eager | vs tcompile |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | k_mean_20f978 | input RMSNorm | 191.54 | 8.69 | 3.82 | 3.17 | 60.4x | 2.74x |
| 512 | k_linear_reduce_06a42b | v_proj | 12.97 | 13.70 | 83.03 | 17.50 | 0.74x | 0.78x |
| 512 | k_linear_1fd3d5 | q_proj + reshape to heads | 100.64 | 21.65 | 132.97 | 22.35 | 4.50x | 0.97x |
| 512 | k_linear_a09c5a | k_proj + reshape to heads | 59.86 | 14.73 | 71.21 | 17.28 | 3.46x | 0.85x |
| 512 | k_sdpa_linear_reduce_c0a378 | softmax·V, computed V | 45.78 | 46.04 | 159.57 | 159.57 | 0.29x | 0.29x |
| 512 | k_linear_sdpa_reduce_e24efe | o_proj + residual | 60.16 | 61.27 | 315.39 | 190.46 | 0.32x | 0.32x |
| 512 | k_linear_mean_reduce_dc067d | post-attn norm + gate/up + SiLU | 246.39 | 49.62 | 67.27 | 67.27 | 3.66x | 0.74x |
| 512 | k_linear_6b4b5f | down_proj + residual | 32.09 | 37.21 | 303.10 | 46.57 | 0.69x | 0.80x |
| 512 | k_sdpa_mean_reduce_29d3df | q/k norm + RoPE + scores + softmax stats | 745.29 | failed | 21286.91 | 19609.60 | 0.04x | — |
| 1 | k_mean_b8e46d | input RMSNorm | 121.85 | 2.89 | 3.05 | 2.38 | 51.2x | 1.21x |
| 1 | k_linear_reduce_7ef15d | v_proj | 7.42 | 7.21 | 12.39 | 4.74 | 1.57x | 1.52x |
| 1 | k_linear_49a16b | q_proj + reshape to heads | 35.09 | 6.79 | 21.45 | 4.84 | 7.25x | 1.40x |
| 1 | k_linear_dfb21f | k_proj + reshape to heads | 33.67 | 5.25 | 11.06 | 4.74 | 7.10x | 1.11x |
| 1 | k_sdpa_linear_reduce_d0f5c0 | softmax·V, computed V | 11.91 | 10.62 | 25.22 | 20.72 | 0.57x | 0.51x |
| 1 | k_linear_sdpa_reduce_14c8c7 | o_proj + residual | 14.06 | 12.16 | 36.86 | 24.44 | 0.58x | 0.50x |
| 1 | k_linear_mean_reduce_549927 | post-attn norm + gate/up + SiLU | 154.62 | 16.38 | 4605.95 | 393.22 | 0.39x | 0.04x |
| 1 | k_linear_2dcd0c | down_proj + residual | 9.57 | 7.64 | 35.13 | 9.46 | 1.01x | 0.81x |
| 1 | k_sdpa_mean_reduce_0a2624 | q/k norm + RoPE + scores + softmax stats | 334.55 | failed | 39.82 | 36.53 | 9.16x | — |

Every target measured in all five repeats of both rows: the two `torch.compile`-less targets are ordered last in their
golden, so their strict failure no longer costs the later targets their measurement.

Layer totals as the sum of those medians: sequence 512 is 1494.7 eager, 252.9 Inductor (eight of nine targets), 22423
untuned Emmy and 20134 deployed Emmy; sequence 1 is 722.7 eager, 68.9 Inductor (eight of nine), 4791 untuned and 501
deployed. Sequence 1 improves on the previous revision (522 → 501 µs, 1.44x eager). Sequence 512 does not: one target,
the fused score/statistics kernel, is 19610 µs of the 20134 µs total. Over the other eight targets the sequence-512
layer is 524 µs against 749 µs eager, or 1.43x.

### Why the fused score kernel costs 19.6 ms

Its tile IR is unambiguous. The kernel places `free=(head, query)` and sweeps the key axis on the store; inside that
sweep it runs the whole k cone per cell — a 128-element k-norm fold over `to_4`, then the k RoPE — so every k vector is
recomputed once per query row rather than once per key. At sequence 512 that is 512x redundant arithmetic, and it is
the whole cost: 21243.9 µs of the 21286.9 µs untuned total is that single kernel, at 94% occupancy and 34 registers.
No thread tier changes it (`WORK=t256/t512/t1024` measure 21558 / 21835 / 22126 µs; `coop-t` measures 37873 µs).

### Cut options

The recognize rule enumerates three seams on the score root — bare `PLACE` (the score dot `acc2`), `PLACE@a2` (the
q-norm fold), and `PLACE@fold.fold.a4` (the k-norm fold) — plus one on the softmax·V root. Every one was measured at
deployable `-O3` against the fused form in the same process.

| seq | target | fused | cut option | cut | verdict |
| --- | --- | ---: | --- | ---: | --- |
| 512 | k_sdpa_mean_reduce_29d3df | 21289.98 | `PLACE@fold.fold.a4=cut` (k-norm fold) | 19625.98 | 1.08x — committed |
| 512 | k_sdpa_mean_reduce_29d3df | 21289.98 | `PLACE@a2=cut` (q-norm fold) | 335166.47 | 0.06x |
| 512 | k_sdpa_mean_reduce_29d3df | 21289.98 | both seams | 335165.44 | 0.06x |
| 512 | k_sdpa_linear_reduce_c0a378 | 159.57 | `PLACE=cut` | 4466.69 | 0.04x |
| 512 | k_linear_sdpa_reduce_e24efe | 315.39 | `PLACE=cut` | 366.59 | 0.86x |
| 1 | k_sdpa_linear_reduce_d0f5c0 | 25.17 | `PLACE=cut` | 26.22 | 0.96x |
| 1 | k_linear_sdpa_reduce_14c8c7 | 36.82 | `PLACE=cut` | 37.56 | 0.98x |
| 1 | k_sdpa_mean_reduce_0a2624 | 39.58 | no legal seam on this tree | — | — |

One cut wins, and it is the k-norm fold: materializing that reduction once removes 8% of the redundant work and is now
the first committed placement routing in the corpus. The seam that would matter is not in the set. Cutting `a2`
promotes the key axis to a free axis of the residue, whose grid becomes 4.2M blocks and whose cost rises 16x, because
the k cone is then recomputed with no reuse at all. What the target needs is the RoPE'd k vector materialized once as
the dot's B operand; that is a binding the contraction binder still declines, not a seam the placement fork can spell.
For the softmax·V and o_proj forms the cut is a straight loss: it splits a working mma contraction into a scalar
producer plus a workspace zero-fill (`__zp524288`, 48.5 µs on its own in the o_proj cut).

### Schedules that measure but cannot be recorded

Four targets measured a deployable win that the golden's one-knob-map-per-realization format rejects. A multi-kernel
target whose kernels include a knob-free one (an elementwise epilogue such as `k_add_5`, or `k_sdpa_reduce_fe4eb9` at
sequence 1) always fails the merge: that kernel records the empty value for every schedule family while the others
record the pinned value, so `realized_tuning_knobs` sees `WORK: '' != 't512'` and returns nothing. The pin itself is
uniform and replays, so these four realizations record the exact `--ab` pin instead of the merged realized map, and
each was re-verified by replaying the committed golden in a fresh strict process.

| seq | target | greedy | recorded pin | deployed |
| --- | --- | ---: | --- | ---: |
| 512 | k_linear_sdpa_reduce_e24efe | 315.39 | `WORK=w8x2,TILE=mma_m16n8k16_f16_f32/f1x8/k8,STAGE=d2/smem` | 190.46 |
| 1 | k_linear_sdpa_reduce_14c8c7 | 36.86 | `WORK=t512,REDUCE=coop-t` | 24.44 |
| 1 | k_sdpa_linear_reduce_d0f5c0 | 25.22 | `WORK=t512,REDUCE=coop-t` | 20.72 |
| 1 | k_sdpa_mean_reduce_0a2624 | 39.82 | `WORK=t128,REDUCE=coop/r2` | 36.53 |

### Repeat variation

Every target's five paired latencies agree to within 0.6% of their median (0.57% at sequence 1, 0.52% at 512), and
every committed realization reproduces its tuning-time `-O3` measurement to within 0.5%.

### Defects this round surfaced

1. **The online-prior refit aborts the tune.** `emmy tune` raises `_catboost.CatBoostError: All features are either
   constant or ignored` from `OnlinePrior.fit`, reached through `measure_proposals`' `prior.maybe_refit()`, when the
   first measured proposal contributes a run of rows whose feature vectors are identical. It killed the whole
   invocation for four of the six re-keyed targets on a cold online checkpoint. Re-running the same command against a
   checkpoint that already carries a varied dataset succeeds, which is the workaround used here.
2. **The tune-lane bench watchdog censors an expensive target completely.** Every candidate of the 21 ms fused score
   kernel exceeds the 2 s accumulated-GPU-time budget and is marked `bench_fail`, so a full 1800 s search ranked
   nothing at all and wrote no `ranking` block. `EMMY_BENCH_RUN_TIMEOUT_S` raises the budget; at 60 s the search
   instead spent its whole 2400 s in re-lowering without completing a candidate, so this target's schedules were
   priced by direct `emmy run --ab` instead.
3. **`torch.compile` still cannot compile the RoPE-bearing attention reference** on PyTorch 2.13.0, so `--strict`
   rejects those two targets in every repeat and both rows report `failed`. This is the same Inductor limitation the
   previous run recorded.
4. **A bare `PLACE=cut` pin re-cuts every piece it produces.** Because the pin is authoritative on each freshly
   recognized tree, the resolution recurses through the fragments; on the sequence-512 score tree a single
   `emmy compile --ir tile` under that pin had not terminated after ten minutes. Named seams resolve promptly.

### Conclusion

The re-key is mostly benign: six of nine realizations per sequence carried over verbatim and reproduce their committed
numbers, and sequence 1 is faster than the previous revision (1.44x eager). Sequence 512 regressed by construction —
folding the k cone into the score kernel made it recompute that cone once per (query, key) pair, which costs 19.6 ms
against 745 µs eager and turns a 1.56x layer win into a 0.07x loss. The placement fork is real and usable: its seams
enumerate, resolve by name, and one of them is now committed evidence, but the seam that would undo this particular
regression is a contraction binding rather than a placement.

### Limitations

- Layer-0 evidence only, one model, one card; never a whole-model claim.
- Both rows report `failed` because `--strict` requires a `torch.compile` latency the two RoPE-bearing targets cannot
  produce. Every other target passed strict Emmy-vs-eager correctness in all five repeats.
- The Inductor column is missing for those two targets, so no geometric mean over the full corpus is available; the
  measured denominators are stated above.
- A target's program includes the producers its output needs, so the attention targets overlap and the layer total is
  a sum of overlapping sub-programs on both the Emmy and the eager side, not a disjoint decomposition.
- The five repeats share one deployed host and run back to back, so they capture process-level, not day-level,
  variation.

### Run and system

- Status: failed (2/2 rows, `torch.compile` reference unavailable on the two RoPE-bearing targets; every target
  measured in every repeat)
- Result timestamp: 2026-08-21T23:03:34Z; run ID: `20260821T230334Z`
- Rows: `…sl1_scommon` (row ID `551082cef77b`, 440.74 s) and `…sl512_scommon` (row ID `3a4d139974b8`, 2131.30 s)
- Git revision: `213c443a`; dirty: false
- Host: `riftvm`; Ubuntu 24.04.1 LTS; AMD EPYC 7742 64-Core Processor
- GPU: NVIDIA A100-SXM4-80GB, UUID `GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`; PyTorch 2.13.0+cu130

### Durable files

- Raw-results archive: `results_a100x1.tar.gz`; archived root `2026-08-21_23-03-34/`
- Members: both `*.experiment.yaml` records, both `*_artifacts.tar.gz` task archives (per-repeat verification JSON per
  target, per-repeat logs and exit statuses, package freeze, replayed working golden), and the two runner logs
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
