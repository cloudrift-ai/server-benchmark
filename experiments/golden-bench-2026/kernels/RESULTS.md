# Cycle 3: retune on main @ 3aa618af — tunable kernels now beat eager; fusion's cold deploys got worse

## Scope and protocol

Third pass, after 70 further commits (#629 computed-A recognition, #633 Volta A-reuse, #628/#636 structural
winner replay, FA follow-ups). RTX 4090 (128 vCPU, CUDA 12.8, driver 590) and RTX 5090 (192 vCPU, CUDA 12.8)
hosts; V100 not provided this cycle (its cycle-2 goldens stand). 0.6B + 0.6B-FP8 only — both hosts had 32 GB
disks, excluding the 32B checkpoint by fit. Same trace -> tune (budget 12) -> 3x `-O3` verify flow, with one
deliberate protocol change: **`EMMY_BENCH_RUN_TIMEOUT_S=15`** for tune and verify (the tune lane hardcodes a 2 s
bench watchdog at `emmy/commands/tune.py:188`; 15 s lets slow-but-real candidates rank). Runs were driven by the
same uncommitted benchmark-from-goldens flow as cycle 2; in-tree `recipe.yaml` unchanged.

## Headline, stated carefully

**Both directions moved.** The measured majority of kernels improved dramatically — on the 4090 the tunable
kernels now sum to 553 µs (s512) / 148 µs (s1) and **beat eager 1.4x / 2.7x**, the first eager win of any
cycle; cycle 2's 116 ms knobless decode misdeploy is gone. But the new fusion (kernel count changed 18 -> 21;
every cycle-2 SDPA/computed-A kernel identity is gone) produces cold deploys so slow they exceed even the 15 s
bench cap. Measured with 10-iter runs (single repeat, marked as such in the golden):

| kernel (4090) | emmy µs | eager µs | ratio |
| --- | ---: | ---: | ---: |
| `k_linear_mean_reduce_d623c5...s512` (computed-A) | 422477 | 220 | ~1900x |
| `k_sdpa_linear_reduce_5b0de4...s512` | 233695 | 41 | ~5700x |
| `k_linear_sdpa_reduce_fc07d0...s512` | 233954 | 54 | ~4300x |
| `k_linear_mean_reduce_c1b0ae...s1` | 136830 | 110 | ~1250x |

All deploy with empty schedule knobs on trivial grids (the 422 ms kernel runs on a **2-block grid**). True s512
layer total on the 4090 is therefore ~891 ms vs eager ~1.5 ms — **worse than cycle 2's 34.5 ms**, because the
fused forms got heavier while staying schedule-less. A naive old-vs-new sum over *measured* rows reads as a 62x
improvement; that is coverage illusion, and this file reports it only to warn against it.

## Findings

1. **Greedy-maximal fusion emits cold-catastrophic fused forms and the deploy path has no defense.** The new
   SDPA/computed-A kernels are knobless at deploy (search reaches no schedule; candidates exceed any sane bench
   budget). Raising the watchdog 2 s -> 15 s helped everything measurable, but these need ~100+ s per bench —
   the earlier observation stands that raising it further just moves cost into re-lowering.
2. **Golden drift is now structural and fast**: the #580 goldens no longer replay on this main ("provenance
   target no longer resolves after lowering"; only 4/18 resolved), and the kernel identities changed again
   within this cycle. Any golden corpus needs re-tracing per compiler epoch.
3. **The 2 s tune watchdog is still hardcoded** (`tune.py:188`) with only the env override; cycle-3 tunes ran at
   15 s and that protocol note travels with these numbers.
4. 5090 mirror: same shape (15/21 and 26/38 measured; FP8 traces needed `HF_HUB_DISABLE_XET=1` after xet
   download failures). Its unmeasured rows are the same over-budget fused kernels, not measured separately.
5. Host notes: both cycle-3 hosts were clean (no driver/toolchain traps); Vast containers reap tmux/setsid on
   SSH resets — **supervisor** is the reliable launcher there.

## Files

Goldens refreshed for rtx4090/rtx5090 (0.6B, 0.6B-FP8; the six over-budget 4090 rows carry single-repeat
10-iter measurements, all other rows are median-of-3). Archives: `results_rtx4090x1.tar.gz`,
`results_rtx5090x1.tar.gz` (raw traces, tune logs with 15 s-watchdog search feedback, verify JSONs, slow-kernel
logs). V100 files untouched from cycle 2.

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
