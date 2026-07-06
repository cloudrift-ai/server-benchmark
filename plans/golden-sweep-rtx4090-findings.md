# Golden sweep findings — RTX 4090 (sm_89), 2026-07-01 → 2026-07-06

- GPU: NVIDIA GeForce RTX 4090, sm_89, driver 580.65.06, CUDA 12.9, VM `vm-perf-tuning` (`~/emmy` at `272153d3`).
- Sweep: `emmy tune --dataset golden --clean` (29 shapes, ~171 min compute, 2026-07-01) → forecast
  (`eval prior --dataset golden`) → per-shape A/B `emmy run --bench --golden NAME` (29 shapes) → confirmation re-runs
  of every win candidate (2 extra runs each) → regression bisect on a pre-#269 clone (`~/emmy-pre269`).
  Logs under the gitignored `_tune/golden-sweep-rtx4090/` (`tune.log`, `ab-pass1.log`, `ab-pass2.log`,
  `eval-failures.log`).
- Branch: `feature/golden-sweep-rtx4090-findings`. This file supersedes the 2026-06-19 sweep report (in git history:
  `git show 272153d3:plans/golden-sweep-rtx4090-findings.md`). The recorded YAML numbers ARE that sweep's
  `run --bench` output, same methodology and same GPU as this one, which is what makes the drift comparison in
  Finding 1 apples-to-apples.
- **Category tally (live A/B): 21 worse / 2 same / 6 better.** The YAML is **re-baselined** to the live post-#269
  numbers (Finding 1: the regression is a known, accepted consequence of the rewrite — the team's plan is to
  reintroduce the dropped optimizations gradually): every entry keeps its knobs with `emmy_us` re-recorded live, and
  `square.512` additionally takes the confirmed `SPLITK=4` knob win. The pre-#269 numbers — the long-term target —
  stay recoverable via `git show 272153d3:emmy/compiler/pipeline/search/goldens/rtx4090_sm89.yaml` (noted in the
  YAML header). All A/B numbers are -O3 `run --bench`, never the -O1 tune DB.

## Headline — Finding 1: PR #269 (Block-DAG Tile IR) is a 1.66× median codegen regression (known/accepted)

The recorded golden knob sets, re-benched live on today's compiler, run at **median 1.66× their recorded latency**
(worst `square.512.fp16` **4.07×**, `o_proj.s32` 2.85×, `q_proj.s32` 2.58×; every one of the 29 shapes drifted
≥1.09×; full column in the table below). The cuBLAS reference reproduces its recorded value on all 29 shapes
(e.g. 320.0 → 311, 20.2 → 20), so the harness, clocks, and measurement semantics are unchanged — the emmy kernels
themselves got slower at identical knobs.

Bisect (pinned golden knobs, `square.512.fp16`, -O3 `run --bench` on the VM):

| commit | date | golden-row µs |
|---|---|---:|
| `7b14aff6` (#270, parent of the rewrite) | 06-25 | **6.1** (recorded: 5.9 — reproduces) |
| `8c7d2d54` **(#269, Block-DAG Tile IR)** | 06-26 | **24.0** |
| `272153d3` (HEAD) | 07-01 | 24.0 |

`o_proj.s32` confirms on the same clone: 13.9 µs at the parent (recorded 11.2) vs 31.9 today. The regression enters
at exactly the tile-lowering rewrite and is single-commit.

Mechanism (identical-knob CUDA diff, `WN=1,WM=2,FM=1,FN=4,BK=2,RING=4,STAGE=11,MMA=mma_m16n8k16_f16`; dumps kept on
the VM as `~/cuda-pre269.cu` / `~/cuda-head.cu`): the warp-tile mma math and epilogue are identical, but the new
assembly drops three capabilities the old one applied under the same knobs —

1. **The A operand is never staged into smem.** HEAD emits the `dpl_mma_load_a_gmem` fallback (its own header
   comment: "the fallback when an mma.sync operand was NOT staged into shared memory … Slower than ldmatrix (no smem
   reuse)"), re-reading A fragments from global memory inside every mma step. Visible in the bench table as an empty
   `PLACE@x0` cell on the golden row. Pre-#269 staged A into a 4 KB slab read via `ldmatrix.x4`.
2. **`RING=4` / `STAGE=11` produce no pipeline.** Pre-#269: a true 4-deep cp.async ring (3-stage prologue,
   `(a6+3)%4` indexing, `cp.async.wait_group 3` overlap, drain tail). HEAD: single 2 KB buffer, scalar `__half`
   copies, `__syncthreads()`-load-`__syncthreads()`-compute, 16 serial iterations. The knobs are accepted but inert.
3. **The `GROUP_M=8` L2 CTA swizzle is gone** (plain row-major block decode).

The resource fingerprints tell the same story from the bench table alone: same knobs, pre-#269 16 KB smem / 72 regs /
25% occ vs HEAD 2 KB / 48 regs / 88% occ — high occupancy with no work per thread. The fp32 shapes drift too (same
transport machinery), just less violently (1.09–1.65×).

**Status: known and accepted.** The rewrite deliberately dropped pre-existing optimizations; the plan is to
reintroduce them gradually and reach — then surpass — pre-#269 performance. This sweep's contribution is the
concrete work-list and its per-shape cost: the three dropped capabilities above (A-operand staging is the largest,
then the cp.async ring, then the swizzle), the per-shape drift column below as the scoreboard each reintroduction
should move toward 1.0× (against the pre-#269 numbers at `git show 272153d3:...goldens/rtx4090_sm89.yaml`), and the
bisect clone + identical-knob CUDA dumps on the VM as the reference codegen. Guard for the future: the drift check —
live `run --bench --golden` vs recorded `emmy_us` per shape — belongs in an `eval` view so progress (and any new
regression) is visible per sweep (see Workflow notes).

## Per-shape outcomes (-O3 `run --bench` A/B, pass 1; win candidates re-confirmed twice in pass 2)

| shape | greedy µs | live-golden µs | greedy/golden | recorded µs | live/recorded | cuBLAS µs | greedy vs cuBLAS | category |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| square.512 | 31.8 | 19.3 | 1.65 | 13.5 | 1.43 | 10.8 | 2.94 | worse -> leave |
| square.1024 | 87.0 | 77.6 | 1.12 | 71.0 | 1.09 | 45.4 | 1.92 | worse -> leave |
| square.2048 | 744.4 | 556.5 | 1.34 | 383.3 | 1.45 | 320.0 | 2.33 | worse -> leave |
| square.4096 | 5118.0 | 4609.9 | 1.11 | 4004.9 | 1.15 | 2458.6 | 2.08 | worse -> leave |
| square.512.fp16 | 11.1 | 24.0 | 0.46 | 5.9 | 4.07 | 5.8 | 1.91 | better (not recorded) |
| square.1024.fp16 | 50.6 | 54.6 | 0.93 | 29.3 | 1.86 | 18.1 | 2.80 | better (not recorded) |
| square.2048.fp16 | 261.4 | 285.4 | 0.92 | 119.3 | 2.39 | 115.2 | 2.27 | better (not recorded) |
| square.4096.fp16 | 2263.0 | 2001.9 | 1.13 | 889.9 | 2.25 | 822.3 | 2.75 | worse -> leave |
| qwen3_06b.q_proj.s32 | 28.1 | 20.4 | 1.38 | 7.9 | 2.58 | 9.9 | 2.84 | worse -> leave |
| qwen3_06b.kv_proj.s32 | 17.3 | 13.8 | 1.25 | 6.5 | 2.12 | 6.9 | 2.51 | worse -> leave |
| qwen3_06b.o_proj.s32 | 36.9 | 31.9 | 1.16 | 11.2 | 2.85 | 9.9 | 3.73 | worse -> leave |
| qwen3_06b.gate_up_proj.s32 | 35.3 | 21.4 | 1.65 | 12.1 | 1.77 | 11.3 | 3.12 | worse -> leave |
| qwen3_06b.down_proj.s32 | 47.1 | 35.3 | 1.33 | 15.6 | 2.26 | 13.0 | 3.62 | worse -> leave |
| qwen3_06b.q_proj.s128 | 59.5 | 38.0 | 1.57 | 20.9 | 1.82 | 20.2 | 2.95 | worse -> leave |
| qwen3_06b.kv_proj.s128 | 35.3 | 21.9 | 1.61 | 12.9 | 1.70 | 12.4 | 2.85 | worse -> leave |
| qwen3_06b.o_proj.s128 | 66.6 | 39.7 | 1.68 | 23.0 | 1.73 | 19.0 | 3.51 | worse -> leave |
| qwen3_06b.gate_up_proj.s128 | 48.7 | 53.7 | 0.91 | 33.0 | 1.63 | 25.5 | 1.91 | better (not recorded) |
| qwen3_06b.down_proj.s128 | 114.9 | 55.0 | 2.09 | 33.1 | 1.66 | 24.5 | 4.69 | worse -> leave |
| qwen3_06b.q_proj.s512 | 123.9 | 89.4 | 1.39 | 55.6 | 1.61 | 53.3 | 2.32 | worse -> leave |
| qwen3_06b.kv_proj.s512 | 60.1 | 52.8 | 1.14 | 34.3 | 1.54 | 38.9 | 1.54 | worse -> leave |
| qwen3_06b.o_proj.s512 | 123.5 | 90.3 | 1.37 | 55.6 | 1.62 | 67.8 | 1.82 | worse -> leave |
| qwen3_06b.gate_up_proj.s512 | 140.3 | 124.9 | 1.12 | 103.7 | 1.20 | 85.5 | 1.64 | worse -> leave |
| qwen3_06b.down_proj.s512 | 178.0 | 131.1 | 1.36 | 79.7 | 1.64 | 113.2 | 1.57 | worse -> leave |
| square.512.dynM | 25.6 | 23.1 | 1.11 | 12.8 | 1.80 | 10.8 | 2.37 | worse -> leave |
| qwen3_06b.q_proj.s512.dynM | 96.6 | 89.2 | 1.08 | 63.8 | 1.40 | 53.0 | 1.82 | worse -> leave |
| qwen3_06b.kv_proj.s512.dynM | 50.8 | 51.5 | 0.99 | 33.4 | 1.54 | 37.0 | 1.37 | same |
| qwen3_06b.o_proj.s512.dynM | 91.3 | 117.8 | 0.78 | 62.0 | 1.90 | 67.1 | 1.36 | better (not recorded) |
| qwen3_06b.gate_up_proj.s512.dynM | 121.6 | 129.7 | 0.94 | 91.1 | 1.42 | 85.6 | 1.42 | better (not recorded) |
| qwen3_06b.down_proj.s512.dynM | 130.0 | 129.4 | 1.00 | 92.3 | 1.40 | 119.4 | 1.09 | same |

Columns: `greedy/golden` is the live A/B (both sides benched this run, >1 = greedy slower); `live/recorded` is the
Finding-1 drift (live golden re-bench over the YAML's recorded `emmy_us`); `greedy vs cuBLAS` is greedy µs over the
recorded `cublas_us` (>1 = emmy slower than PyTorch). The `better` rows all reproduced 3/3 (pass 2 re-ran each twice;
golden rows were rock-steady, e.g. 24.0/23.9/23.9 and 117.8/117.9/117.9), so they are genuine on the current
compiler — every one is a shape whose recorded knobs regressed hardest (`live/recorded` 4.07, 1.86, 2.39, 1.63,
1.90, 1.42), i.e. greedy found configs that dodge the dropped optimizations. Their knob dicts could not be recorded
this sweep — the kernel table's new schema doesn't expose the full YAML knob vocabulary and `eval golden` is broken
(Finding 5) — so their re-baselined entries keep the old knobs at live µs; extracting greedy's configs for these six
is a follow-up once Finding 5 is fixed. The `greedy vs cuBLAS` column is uniformly bad (1.09–4.69, vs 0.70–1.71 in
the 2026-06-19 report) — the same regression seen from the absolute side.

## Finding 2 — greedy trails even the regressed goldens on 21/29 shapes; split-K probe wins on `square.512` (P1)

Independent of Finding 1, the deployed greedy pick loses the live A/B on 21/29 shapes (median ~1.35×, worst
`down_proj.s128` 2.09×). Drill-down on `square.512` (fp32): greedy deploys `FM=2, s32 K-chunk, SPLITK=1` at 100%
occupancy — 30.5–32.1 µs across three runs — while the golden knobs (`FM=4, s64`, 50% occ) hold 19.3. The inner loop
of the greedy kernel does 2 FMAs per 3 smem reads (vs the golden's 8 per 6): the prior over-values occupancy and
under-values per-thread reuse, the same `_W_A` mis-pricing family as the 06-19 report's Findings 1/4 (that refit is
now unactioned across three reports).

A manual `--ab` lever sweep on the same shape found `SPLITK=4` on top of the golden knobs (grid 256 → 1024, atomic
combine epilogue) benches **17.3–17.6 µs across three runs** vs the live golden's 19.3 — a reproducible ~10% win and
the best currently-reachable config for the shape (0.63× cuBLAS). `SPLITK=8` is slightly worse (18.0/18.9), bigger
register tiles much worse (FM8/FN4 36.9, FM8/FN8 61.1 — occupancy collapse). Not recorded either: the recorded
pre-#269 golden hit 13.5 µs *without* split-K, so the split-K win is likely compensating for regression-inflated
per-CTA cost (an underfilled grid hurts more when each CTA is slower), and may evaporate once Finding 1 is fixed.

**Recommendation (P1):** re-run this sweep after each optimization-reintroduction lands (the drift column is the
scoreboard). Independently, action the `_W_A` analytic refit (`scripts/golden_knob_heuristics.py`) the last two
reports asked for — the occupancy-over-reuse mis-pricing is visible on current codegen regardless — and check
whether small-shape SPLITK preference belongs in the analytic occupancy term.

## Finding 3 — pinning `SPLITK=6` dies late with an opaque lowering error (P2)

`emmy run --bench --golden square.512 --ab "SPLITK=6,FM=4,FN=2"` fails with `CudaBackend: node 'matmul' has
non-CudaOp 'TileGraphOp'; lowering must produce Graph[CudaOp]` — the tile pass declines the non-power-of-two split
and leaves the op unlowered instead of rejecting the knob value up front. The search never offers SPLITK=6 (tune is
unaffected); only pinned paths (`--ab`, `EMMY_KNOBS`) hit it.

**Recommendation (P2):** validate pinned knob values against the enumeration's candidate domain when the pin is
applied, and fail with "SPLITK=6 not in candidates {1,2,4,8} for this op" instead of a backend invariant violation.

## Finding 4 — fp16 sweep hit an nvcc compile failure; the recorded error hides the real diagnostics (P2)

One config of `square.2048.fp16` (`k_matmul_bed174`; knobs per `eval failures`: `FM=8, FN=2,
ATOM@out=mma_m16n8k16_f16, REDUCE@a2=s1/f1/c1/t1, SPLIT@a0=1x8, SPLIT@a1=8x2, INTERLEAVE_LOADS, VECTORIZE_LOADS`)
failed nvcc with "3 errors detected", but the `perf.error` column only preserves the first diagnostic — the unused
`dpl_mma_m16n8k16_bf16` helper (`ir/kernel/render.py:322-336` emits both the f16 and bf16 helpers into every MMA
kernel unconditionally, so the unused twin is present in *every* fp16 kernel; it cannot alone explain a
config-specific failure). The two real errors are lost — the temp `k.cu` is deleted and the error text truncated.
The search recovered (config pinned `bench_fail @ 2e6`), so impact on the sweep was one lost variant.

**Recommendation (P2):** record the full nvcc stderr in the `error` column (or at least the *last* diagnostics, not
the first), and emit only the selected atom's helper. Repro: pin the knobs above on the `square.2048.fp16` snippet.

## Finding 5 — `eval prior --dataset golden` renders the greedy pick's knobs as all-dashes (P3)

In the post-sweep forecast, every shape's "found" knob cells printed `-` (`m/t 0/10` with `-/8`, `-/16`, …) while the
same-process `run --bench` resolves and prints the greedy knobs fine. Either the view's greedy-pick resolution or its
rendering is broken; as-is the per-knob found/golden diff — the view's whole point — is unreadable. (Its `vs gold`
ratios were also silently inflated by Finding 1, since the denominator is the recorded µs; that is by design, but a
drift flag would have surfaced the regression right there.)

**Recommendation (P3):** fix the found-knob resolution in the golden view; consider a `live/recorded` drift column
sourced from the reservoir's `H_opt=3` rows.

## Workflow notes

- **The drift check that caught Finding 1 was hand-rolled** (a python+yaml join of `ab-pass1.log` against the golden
  YAML). Nothing in the CLI compares live golden re-benches to recorded `emmy_us`. *Improvement:* a per-shape drift
  column in `eval prior --dataset golden` (or a dedicated `eval drift`), plus a documented "if median drift exceeds
  the noise band, stop and bisect before recording" rule in the tune-golden skill — this sweep nearly recorded
  regressed numbers as wins.
- **Tune compute grew 1.7× vs the 06-19 sweep** (~171 min vs ~102 min for the same 29 shapes; per-shape bench counts
  62–144). Unclear whether patience, the `EMMY_O3_TOL` re-bench band, or the richer post-#269 enumeration is
  responsible. *Improvement:* log per-shape bench/wall-time split so growth is attributable.
- **`--dataset golden` and `--golden` disagree on the name universe.** Scripting the A/B loop from
  `GOLDEN_CONFIGS` names crashed on `pointwise.*` entries that `run --golden` rejects (matmul-only). *Improvement:*
  a `--golden list` (or promote the error message's "Available:" list to a first-class command) — the loop had to be
  rebuilt from the error message.
- **`eval variants` still cannot filter by shape** (06-19 note, still open, hit again): groups key on the kernel
  C-hash and merge shapes, so mapping "the square.512 group" required fingerprint guessing, and the deploy-vs-search
  shortfall diagnosis fell back to live `--ab` runs. *Improvement:* unchanged — join through `ShapeKey`.
- **Golden A/B rows were far more stable than the documented 10–13% noise band** this sweep (pinned golden rows
  repeated within ~1%: 19.3/19.3/19.3, 24.0/23.9/23.9, 117.8/117.9/117.9), so 3-run confirmation was cheap and
  unambiguous. The band may be a property of older uncaptured timing; worth re-measuring where it stands now.
- **Status of the 06-19 report's notes:** the option-0 fallback **held** (29/29 shapes, zero `LoweringError`
  crashes); the `cudaErrorMisalignedAddress` golden-pin contamination **did not recur**; the nvcc-on-PATH fail-fast
  is **untested** (nvcc was on PATH throughout); `eval variants` shape filtering **still open** (above); the `_W_A`
  analytic refit **still unactioned** (Finding 2).
