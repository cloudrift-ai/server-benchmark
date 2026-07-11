# Golden sweep findings — NVIDIA GeForce RTX 4080 (sm_89)

**Date:** 2026-07-10  **GPU:** NVIDIA GeForce RTX 4080 (`rtx4080_sm89.yaml`)  **Where:** local dev box (the 4080),
cold in-repo sweep; all data under `_tune/golden-sweep-rtx4080/2026-07-10/` (tune log, per-shape A/B JSON, eval dumps,
pre/post-sweep prior + DB backups).

**Sweep command (cold, one invocation):**

```
EMMY_O3_TOL=0.10 emmy tune --dataset golden --clean     # patience 50 (default); O3_TOL trims -O3 recompiles of fp16 non-contenders
```

**Wall time:** ~33 min for **9** tune targets — the live-card-filtered golden set (4 fp32 squares, 4 fp16 squares,
1 `attention.hd256.dynM`). Per-target tune (`done: … in N s`): 512=28 s, 1024=50 s, 2048=51 s, 4096=142 s,
**512.fp16=437 s**, 1024.fp16=294 s, 2048.fp16=380 s, 4096.fp16=308 s, attn.hd256.dynM=269 s. `bench_fails`:
`square.4096` (a `HungKernelError` + a `CUDA_ERROR_ILLEGAL_ADDRESS` variant on the big fp32 tiles) and
`attention.hd256.dynM` (12 hung wide-attention variants at the 1000 ms guard) — the expected big-shape /
wide-attention bench guards, isolated in the bench-worker subprocess; the tune completed **9/9**.

**A/B method:** `emmy run --bench --golden NAME --json` per shape — pass 1 for all 9, passes 2–3 for all 9 (cubins
cached from the tune, so the 18 re-passes cost ~2 min), plus **5 extra passes on `square.2048.fp16`** to resolve a
bimodal greedy. All µs below are the live **-O3** A/B (greedy = deploy pick vs the recorded golden, both benched this
run), medianed over passes. Golden rows were rock-stable (0.1–2.2 % spread); the noise lived on the greedy side.

## Category tally (9 shapes)

| Category | N | Action |
| --- | --- | --- |
| **REPLACE** (greedy >3 % faster, reproduced above noise) | 1 | recorded |
| ADD | 0 | — |
| same / parity (≤~5 %, or identical knobs) | 3 | none |
| **SLOWER** (prior couldn't reach golden, >3 %) | 5 | findings |

**Recorded (1 replacement):**

| shape | old knobs → new | emmy_us | vs live golden | note |
| --- | --- | --- | --- | --- |
| `square.2048.fp16` | `w2x1/f4x4/k2 d1/cp` → `w1x4/f4x4 d2/cp/ring` | 207.1 → **188.3** | 0.90× | greedy ≥ golden in **all 8** passes; 10 % faster in 5/8 (bimodal 188 / 209 — see Finding 3 / Workflow) |

> **Re-validated on `main` (#339 f16-accumulate) 2026-07-10:** this sweep ran on `bad4e89d`; #339 later renamed the
> fp16 mma atom `_f16`→`_f16_f32` and reshaped the fp16 enumeration. Re-A/B on #339 code: the pinned config
> `w1x4/f4x4 d2/cp/ring` reproduces at **188.4 µs** (flags clean, ≈ the pre-#339 188.3) and still beats the recorded
> golden (209.3) and eager cuBLAS (198) — so the win holds. Note the #339 *greedy* no longer reaches it (it picks
> `w4x1/f2x4/k2 d1/cp` @ 218–242 µs, slower than the golden) — a reachability regression on this shape worth its own
> look. The YAML records the config under the `_f16_f32` atom name at 188.4 µs.

New ratio `cublas/emmy = 197.0/188.3 = 1.046` (was 0.951) — the win crosses emmy from just-below to just-above cuBLAS
HGEMM. The old `w2x1/f4x4/k2 d1/cp` entry is >3 % slower than the new one, so it was **deleted** (replace, not add).

## Update — full cold re-sweep on `main` (#339 f16-accumulate), 2026-07-10

After #339 landed (renamed the fp16 mma atom `_f16`→`_f16_f32`, registered an f16-accumulate atom, reshaped the
fp16 enumeration), the whole 4080 golden set was **re-swept cold on `main` (f7f35a38)** (`emmy tune --dataset
golden --clean`, ~34 min). **Result: zero YAML changes — every recorded golden is confirmed on #339.** What the
re-sweep established:

- **No f16-accumulate win — and enabling `FAST_MATH` makes these kernels SLOWER, not faster.** By default the
  f16-accumulate atom `mma_m16n8k16_f16_f16` (#339's headline, ~2× the f32-accum mma-chain rate on the 4080's
  consumer die) is off — its enumeration is gated behind `F16_MMA_F32_ACC` / `FAST_MATH`
  ([`emmy/compiler/ir/atom.py:135`](emmy/compiler/ir/atom.py#L135); consumer-die gate `_f16acc_allowed`,
  [`tile/_schedule.py:107`](emmy/compiler/pipeline/passes/lowering/tile/_schedule.py#L107)). **Exploration
  (2026-07-10, `EMMY_FAST_MATH=1`):** re-tuning `2048.fp16` / `4096.fp16` with it on enumerated 32 f16-accum
  configs head-to-head, and the tuner **still picked f32-accum** — because f16-accum is **17–23 % slower** on
  these shapes. A direct multi-tile A/B on `2048.fp16` confirmed: every f16-accum tile 220–231 µs vs the
  f32-accum golden 188.6 µs, with **register pressure blown to 228+ regs → 17 % occupancy** (the `FragmentPromote`
  keeps both f16 accumulators and f32 shadows live). These GEMMs are occupancy/bandwidth-bound at their optimal
  tiles, so the 2× mma-chain rate is wasted and the promote overhead dominates. Accuracy stayed within emmy's
  wrong-answer gate (flags clean). **Conclusion: the f32-accum goldens are correct; `FAST_MATH` is a net loss on
  the squares.**
- **Probe — where f16-accum DOES win: K-heavy, small-output-tile shapes (modest +3–9 %).** A follow-up probe
  (matched-tile f16-accum-vs-f32-accum A/Bs) found the sign flips with the tile's occupancy profile. Small output
  tile (`w2x1/f4x4`, ~6 extra regs → 33 % occ preserved) + deep K (mma-bound): synthetic **512²×K8192 +9 %**
  (161.9 vs 178.3 µs), **256²×K16384 +7 %** (83.0 vs 89.5), gemma **`down_proj` K=15360 +3.3 %** (1272.8 vs
  1316.2). Large output tile (the squares, gate/up N=15360): the `FragmentPromote` shadows crush occupancy to
  17 % → **loss**. Even the wins are far below the theoretical 2× (memory traffic, smem staging, promote, split-K
  reduction all dilute the mma-chain speedup). So f16-accum is **situational** — worth enabling only for genuinely
  mma-bound, register-light kernels (deep-K reductions with small output tiles); correctly off-by-default, since a
  blanket `FAST_MATH` slows more kernels than it speeds. Data: `_tune/fastmath-explore/`.
- **The retrained #339 prior fixed the fp32 greedy reachability.** `square.1024` / `square.2048` greedy now
  *reach* their goldens (0.97–1.02× vs 1.09–1.14× on bad4e89d) — the two fp32 findings below are resolved on
  main (the picks match the golden knobs; nothing to record).
- **`square.2048.fp16` golden independently re-confirmed** at 188.6 µs (≈ the recorded 188.4); the greedy still
  cannot reach it (218 µs, `w2x4/f4x4` — a persistent reachability regression on this one shape, worth its own
  look).
- **Left as marginal (sub-5 % floor):** `square.1024.fp16` greedy 3.2 % faster (31.9 vs 32.9, 0.3 % spread over
  4 passes) and `square.512` 4.9 % faster — both reproducible but below the recording threshold.

The findings and fork-regret analysis below are from the original `bad4e89d` sweep; the fp32 reachability
findings are superseded by the re-sweep (fixed), the fp16 tier findings still stand (f16-accum still gated).

## Fork sibling regret (`emmy eval prior --dataset nodes`, -O1 block, 3173 nodes / 55 forks)

The command prints every metric twice: `=== analytic prior ===` is the cold-start ranking that decides what a cold
sweep measures at all; `=== learned prior ===` is the CatBoost this sweep trained. Columns below are the two halves on
the same 4080 nodes.

| metric (-O1, 55 forks) | analytic prior | learned prior (CatBoost) |
| --- | --- | --- |
| TILE fork regret (median) | **2.35×** | **1.07×** |
| STAGE fork regret (median) | 1.00× | 1.00× |
| structural PLACE+R+S+T (median, 1 fork) | 6.61× | 1.00× |
| worst TILE fork | 10.34× `free=4096 red=4096` | 1.44× `free=2048 red=2048` |
| 2nd-worst TILE fork | 2.58× `free=2048 red=2048` | 1.26× `free=4096 red=4096` |
| 3rd-worst TILE fork | 2.50× `free=512 red=512` | 1.19× `free=1024 red=1024` |
| leaf reachability (mean / median / worst) | 2.89× / 1.35× / 18.47× | 1.10× / 1.01× / 1.35× |
| leaf calibration (median per-op Spearman) | +0.13 | +0.89 |

The worst analytic reachability (18.47×, best 5212 µs on `free=4096 red=4096`) is **not** a degenerate baseline:
2·4096³ / 5212 µs = 26.4 TFLOP/s ≈ 54 % of the 4080's ~48.7 TFLOP/s fp32 CUDA-core peak, physically real. The 96 267 µs
`pick` is a genuine pathological -O1 tile the cold ranking chose, corrected by the -O3 recompile + search. No `(*)`
footnote needed — the medians *and* the worst-case are trustworthy here (cleaner than the 4090's degenerate 2200 TFLOP/s
outlier).

**Diagnosis — the 4080 is the inverse of the 4090.** The analytic half misprices TILE (2.35× median, +0.13
calibration) — the same cold-start weakness the 4090 report found. But **unlike the 4090, the learned half is
well-calibrated on the node store** (TILE 1.07×, +0.89, reachability median 1.01×). So node-store fork regret does
**not** explain the deploy shortfalls. Two mechanisms *outside* the fork-regret lens do: **(1)** 4080 tune-time bench
noise polluting the node store (the golden `square.1024` config was recorded at -O3 93.3 µs during the tune but
benches a stable **83.9 µs** in three fresh A/B passes — so the search saw greedy ≈ golden and deployed greedy), and
**(2)** the learned prior's **full-enumeration extrapolation** burying the analytic-preferred fp16 goldens
(`square.512.fp16` ranks **0** under analytic but **7482/9846** under the learned prior). The `--blame --ablate`
attribution backs this: the analytic TILE column's regret is driven by small *actively-misleading* features
(`D_tile_m` −0.12×, `D_aspect` / `D_near_intensity` / `D_threads` / `D_near_threads` −0.08× each), while the learned
TILE column has **no** strong misleading feature (its −0.01× entries are noise) — the learned model is fine on what it
measured; its misses are in the region it never measured.

## Per-shape outcome table (live -O3 A/B, greedy vs best live golden; cuBLAS = live Eager row this run)

| shape | cuBLAS µs | greedy µs | golden µs | greedy/golden | greedy/cuBLAS | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| `square.2048` | 498.7 | 659.5 | 578.0 | **1.141** | 1.32 | SLOWER (finding) |
| `attention.hd256.dynM` | 55.1 | 77.4 | 68.5 | **1.131** | 1.40 | SLOWER (finding) |
| `square.512.fp16` | 5.7 | 9.3 | 8.3 | **1.119** | 1.63 | SLOWER (finding) |
| `square.1024` | 76.6 | 91.3 | 83.9 | **1.089** | 1.19 | SLOWER (finding) |
| `square.1024.fp16` | 27.1 | 35.3 | 32.9 | **1.073** | 1.30 | SLOWER (finding) |
| `square.4096` | 3829.7 | 4632.6 | 4463.6 | 1.038 | 1.21 | same knobs → bench-order noise |
| `square.4096.fp16` | 1347.5 | 1425.4 | 1400.8 | 1.018 | 1.06 | same (parity, diff knobs, greedy 1.8 % slower) |
| `square.512` | 17.1 | 14.1 | 14.8 | 0.951 | 0.82 | same (greedy 4.9 % faster, sub-5 % — see below) |
| `square.2048.fp16` | 197.8 | 188.3 | 209.5 | **0.899** | 0.95 | FASTER → **recorded (replace)** |

**`square.512` is a stale-golden, not a shortfall.** `eval variants` (the `-O3 us` column) shows the greedy pick
`n16x16/f4x4` is **-O1 rank 11** (23.9 µs, looks terrible) yet **-O3 rank 1 (14.1 µs)** — it is the true -O3 optimum,
beating the golden `n32x8/f2x8` (-O1 rank 1, -O3 14.8). The recorded golden is a **-O1-pinned** config that lost the
-O3 crown. The greedy edge (4.9 %) is dead-stable (0.1 % spread over 3 passes) but sits below the skill's 5 % recording
floor, so I left it. **Recommendation (low priority):** re-pin `square.512` to `n16x16/f4x4 d2/cp/ring` @ 14.1 in a
future pass — it is what deploys and it is the -O3 winner; the current entry mis-records both the knobs and the µs.

`square.4096` reproduces its golden knobs exactly (`n32x16/f4x10 d2/cp/ring`, 2/2 in `eval golden`); its 3.8 % A/B gap
is bench-order noise (greedy 4632 µs matches the recorded `emmy_us` 4636.7; the golden row's 4463 µs was the fast side
of the same-knob re-bench). Nothing to do.

## Finding 1 — fp16 squares (`512.fp16` 1.119×, `1024.fp16` 1.073×): learned prior buries the analytic-preferred golden

This is the 4080's headline and the **inverse** of the 4090's analytic-TILE story.

| shape | greedy knobs | golden knobs | analytic rank | learned rank |
| --- | --- | --- | --- | --- |
| `square.512.fp16` | `w4x1/f2x4/k2 d2/cp/ring` | `w2x1/f1x4/k2 d4/cp/ring` | **0** / 9846 | **7482** / 9846 |
| `square.1024.fp16` | `w4x1/f2x4/k2 d1/cp` | `w4x1/f4x4/k2 d2/cp/ring/p2` | **0** / 8838 | 138–183 / 8838 |

`eval golden` per-knob: `512.fp16` 0/2 (TILE *and* STAGE miss), `1024.fp16` 1/2. The common thread — greedy picks a
**narrow `f2x4` N-fragment** where both goldens want `f4x4` (and greedy drops the `d4/cp/ring` / `d2/cp/ring/p2` STAGE
refinements). The analytic prior ranks **both goldens #0** over enumerations of ~9–10 k configs; the learned prior
buries `512.fp16` at 7482/9846. Since the learned TILE fork regret on the *measured* node store is a healthy 1.07×,
this is **not** a measured-fork mispricing — it is extrapolation into the huge fp16 warp-TILE enumeration the sweep
sampled only sparsely (and the learned model inherits the analytic's censoring of that region… except here the
analytic is *right*, so the censoring argument doesn't even apply — the learned model simply mis-extrapolates where the
analytic is optimal).

**Recommendation (high priority):** the fix is on the **learned** side, not an analytic weight refit (analytic already
ranks these #0). Options, in order: (a) have the deploy prior **defer to the analytic ranking in regions the learned
model is out-of-distribution** (few measured neighbors) — the analytic would deploy both goldens directly; (b) add a
`D_*` feature that separates the mma **N-fragment size** (`f2` vs `f4`) so the learned model can tell the goldens'
`f4x4` from greedy's `f2x4`; (c) more tune-time exploration in the `f4x4` fp16 region (512.fp16 already costs 437 s, so
target it via `--kernel`, don't widen globally).

## Finding 2 — fp32 squares (`square.2048` 1.141×, `square.1024` 1.089×): -O1/-O3 top-of-ranking gap + 4080 bench noise

| shape | greedy TILE | golden TILE | greedy -O3 | golden -O3 (A/B) | golden -O3 (node store) |
| --- | --- | --- | --- | --- | --- |
| `square.1024` | `n16x16/f4x8` (BN16) | `n32x16/f4x8` (BN32) | 90.9 | **83.9** (stable) | **93.3** ← mis-measured |
| `square.2048` | `n32x16/f4x8` (BM16) | `n32x8/f4x8` (BM8) | 658.4 | 578.0 | — |

`square.1024` is the clearest 4080-bench-noise case: the greedy `n16x16/f4x8` is **-O1 rank 1** (`k_matmul_262948`,
-O3 90.9) and the golden `n32x16/f4x8` is **-O1 rank 2** (-O3 recorded **93.3** in the node store). At tune time the
golden looked *slower* (93.3 > 90.9), so the search deployed greedy — correctly, by the data it had. But three fresh
A/B passes bench the same golden config at a stable **83.9 µs** (8 % faster than greedy), so the node-store 93.3 was a
**mis-measurement** (the known 4080 bench flakiness — [[rtx4080-bench-anomaly]]). Both priors rank the golden shallow
(analytic 1, learned 2), so this is **not** a steering or patience failure — it is bad tune-time data. `square.2048`
is a cleaner genuine miss: greedy `n32x16/f4x8` (-O1 rank 8, 1.19× of -O1 best; -O3 658) vs golden `n32x8/f4x8`
(-O3 578) — a real BM (8 vs 16) miss the -O1→-O3 gap hides.

**Recommendation:** (a) **high** — add a tune-time re-bench-median gate on this card (median of K isolated re-benches
before a config's -O3 latency enters the node store); the `square.1024` miss is purely bad data, and it would also
catch the `square.2048.fp16` bimodality. (b) **medium** — `square.2048`'s BM miss is a top-of-ranking -O1/-O3
calibration gap (golden -O1 rank 8, -O3 best); a patience bump won't help (golden ranks shallow), and the analytic TILE
weights *are* mispriced (2.35× median) — a `scripts/golden_knob_heuristics.py` refit that down-weights the
actively-misleading `D_tile_m` / `D_aspect` / `D_near_*` features (per the `--blame --ablate` table) is the durable
fix, but it is secondary to the bench-noise gate.

## Finding 3 — `attention.hd256.dynM` (1.131×): split-TILE schema + wide-attention bench censoring

Greedy pick: a **split TILE** — `fuse`, with `a:mma_m16n8k16_f16/w4x1/f1x4/k16` on one matmul and
`a:mma_m16n8k16_f16/w4x1/f1x32/k2` on the other, STAGE `d1/cp`; golden: a single unified
`w4x1/f1x2/k16 d2/cp/ring`. This is the **same schema gap the 4090 report flagged** for `attention.*.dynM`: the dynM
attention YAML records one unified `TILE`, but the greedy pick carries per-matmul split TILE — not representable, so
even a win here couldn't be recorded (it's a 1.13× loss, so moot this sweep). The attention tune logged **12
`bench_fail`s** (hung wide-attention variants at the 1000 ms guard) and the node store has **no -O3 measurement** for
this shape (all `—`), so the search space is heavily censored. This is a masked-tile dynamic (3 symbolic seq axes) —
the static↔dynamic gap the skill calls out.

**Recommendation (low priority — 1 shape, and it's the 4090's open item):** (a) unify the attention knob schema to
accept split (dd/pj) TILE for `.dynM` attention, **or** document the single-TILE constraint and prune split forks for
dynM attention at tune time so the sweep doesn't land on an unrepresentable pick; (b) the wide-attention `bench_fail`s
censor reachability on hd256 — a longer per-variant guard or a memory-tiled hd256 attention variant would let the
search actually measure the region. Cross-card recurring gap; track with the 4090's.

## Workflow notes

Retrospective for whoever maintains the CLI + this skill. One prior RTX 4080 sweep report exists (deleted in
`4346a889`); diffs against its notes below.

- **Tune dominated (~33 min) but tuned exactly the right 9 shapes — the prior report's top complaint is FIXED.** That
  report flagged "the tune re-runs 34 shapes though only 23 are recorded for this GPU; the 11 extra dynM/pointwise/
  GPU-agnostic produce no 4080 golden to A/B." `live_recorded_goldens()` now scopes `--dataset golden` to the live
  card's own recordings, so this sweep ran 9 targets, zero correctness-irrelevant. The golden set was also trimmed
  (23 → 9) by the poisoned-data cleanup (#343). Fix held; nothing more to do here.
- **`eval variants` still can't be reached by golden name — UNFIXED from the prior report.** `--kernel square.512.fp16`
  returns "no measured variants"; the view keys on the DB kernel hash (`k_matmul_262948`, `k_matmul_bed174`, …). I had
  to `eval variants --top 0` dump the whole thing and map shape → hash by matching the rank-1 TILE knobs — the exact
  multi-command detour the last report described. *Improvement (repeat):* accept a golden name in
  `eval variants --kernel NAME` and resolve it to the hash (the plumbing half-exists — it already errors "`--dataset
  golden has no per-variant measurements`").
- **4080 tune-time bench noise polluted the node store — this is the dominant *analysis* hazard on this card.** The
  golden `square.1024` config recorded -O3 93.3 µs at tune time but benches a stable 83.9 µs in three fresh A/B passes
  (11 % mis-measurement, and it flipped the deploy decision — Finding 2); greedy `square.2048.fp16` is bimodal (188 µs
  in 5/8 passes, 209 in 3/8). Consistent with the known 4080 anomaly ([[rtx4080-bench-anomaly]]). *Improvement:* a
  re-bench-median gate before a config's -O3 latency enters the node store — it would have prevented the `square.1024`
  finding entirely.
- **The `-O3 us` column in `eval variants` was decisive** — it exposed `square.512` as a -O1/-O3 inversion (greedy
  `n16x16/f4x4` is -O1 rank 11 / 23.9 µs but -O3 rank 1 / 14.1 µs). Without it I'd have mis-filed `square.512` as a
  shortfall instead of a stale -O1-pinned golden. Keep surfacing -O3 next to the -O1 rank; it is the single most useful
  column in this workflow.
- **`--json` is still the reliable A/B capture.** I broke my own text capture (`tee … | tail -0` → SIGPIPE → 0-byte
  `.txt` files), but every one of the 27 passes was recovered from JSON alone — `greedy` / `pinned` / `backends` /
  `flags`, all present, `flags: []` everywhere (no integrity issues, realized == pinned knobs). Reconfirms the prior
  report: the text tables are disposable, JSON is the primitive.
- **Noise-floor re-runs were cheap and decisive** (cubins cached from the tune): 9 × 3 passes + 5 extra on the one
  bimodal shape, ~4 min total. Only `square.2048.fp16` needed the extra passes (to prove greedy ≥ golden in every one
  of 8 passes despite the 209 outliers). The ~10–13 % band held for the small shapes; the golden rows themselves were
  the *stable* side this sweep (0.1–2.2 % spread) — the greedy pick was the noisy one, the opposite of the skill's
  stated failure mode, and worth flagging in step 4's guidance.
