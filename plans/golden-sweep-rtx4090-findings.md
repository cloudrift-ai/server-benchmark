# Golden sweep findings — NVIDIA GeForce RTX 4090 (sm_89)

**Date:** 2026-07-11  **GPU:** RTX 4090 (`rtx4090_sm89.yaml`)  **Where:** rented CloudRift box
(`riftuser@211.21.50.85:57008`, normal-billed, torn down after), repo @ **`main` `cbc9805d` (#347 —
analytic-prior weights moved to `analytic_weights.json`)**, emmy set up + tuned + A/B'd over SSH. Full data under
`_tune/golden-full-4090/` (all 43 A/B JSON, the 7 `eval` outputs, the sweep log, the harvested tune DB + prior).

> This supersedes the partial `#339` comparison-subset report; it is the **full** workflow — all 43 shapes A/B'd,
> noise-floored, with the fork-regret + `eval` analytic/learned/blame views on the live DB, on **latest main
> (#347)**.

**Sweep command (cold, one invocation):**

```
EMMY_O3_TOL=0.10 emmy tune --dataset golden --clean     # 43 shapes
```

**Run stats:** tune wall **~180 min** (43/43 shapes; the `h4096` matmuls + `.dynM` twins dominate —
`mlp_down.h4096` 1003 s, `mlp_gate_up.h4096` 723 s, `square.1024.fp16` 875 s; memory-bound kinds ~16–27 s), **6
bench_fail** (`square.4096` hung / `CUDA_ERROR_ILLEGAL_ADDRESS`, wide-attention compile-budget). A/B: all 43 at
-O3 (`run --bench --golden … --json`), 1 pass, + 3 noise-floor passes on the win candidates. Node store: **7 842
nodes, 92 forks**.

## Category tally (43 shapes, after noise-floor)

| Category | N | Action |
| --- | --- | --- |
| **FASTER** (greedy >3% below golden, confirmed) | **1** | recorded (replace) |
| same / parity (≤3%, incl. memory-bound kinds) | 24 | none |
| **SLOWER** (prior couldn't reach golden, >3%) | 18 | findings |

**Recorded (1 replacement):**

| shape | old knobs → new | emmy_us | note |
| --- | --- | --- | --- |
| `square.1024` (fp32) | `n16x8/f4x8` → `n32x8/f2x8` | 55.4 → **57.9** | greedy **0.891×** the *live* golden (57.9 vs 65.0), stable (2% spread / 4 passes). The old config's recorded 55.4 now benches **65.0** on #347 — see Finding 3 (drift). |

**Noise-floor caught two false positives:** `attention.hd256.dynM` (pass-1 0.924× was the *golden* benching slow
that pass; 4-pass median **1.008×** = parity) and `softmax.k8192.dynM` (0.988×, sub-5%). Both left unrecorded —
exactly the step-4 failure mode the skill warns about.

## Fork sibling regret (`emmy eval prior --dataset nodes`, -O1, 92 forks)

`=== analytic prior ===` is the cold-start ranking (decides what the cold sweep measures); `=== learned prior ===`
is the CatBoost this sweep trained.

| metric (-O1, 92 forks) | analytic prior | learned prior (CatBoost) |
| --- | --- | --- |
| **TILE fork regret (median)** | **2.04×** | **1.45×** |
| REDUCE / STAGE (median) | 1.01× / 1.00× | 1.00× / 1.00× |
| structural PLACE+R+S+T (median) | 3.19× | 1.01× |
| worst TILE fork: `free=4096 red=14336` (mlp_down) | 8.97× (*) | 1.68× |
| 2nd-worst TILE: `free=12288 red=4096` (qkv) | 6.03× | 1.56× |
| 3rd-worst TILE: `free=4096 red=4096` (o_proj) | 3.11× | 1.51× |
| `free=2048 red=2048` TILE (high in *both*) | 2.32× | 2.32× |
| leaf reachability (mean / median / worst) | 1.26× / 1.10× / 3.68× | 1.07× / 1.03× / 1.53× |
| leaf calibration (median per-op Spearman) | +0.54 | +0.90 |

(*) the `h4096` `.dynM` masked-tile forks; worst-case baselines sit on the fastest measured sibling — the medians
are the robust read.

**Diagnosis — #347 sharply improved the analytic, but the *learned* prior is now the mispriced half over the full
enumeration.** The analytic TILE regret is **2.04×**, down from the earlier 4090 sweep's **3.62×** (#347's refit
worked), and the analytic ranks goldens beautifully: `eval analytic` median golden rank **0**, **top1 = 17/20**.
But the **learned prior's** golden ranks are deep — median **182**, burying `square.4096` at **458/542** and
`square.1024.fp16` at **1131/8838** (both of which the analytic ranks #0). The learned half is well-calibrated *on
the node store* (+0.90, reachability 1.03×) but **extrapolates poorly to the unmeasured full enumeration** — the
same failure the 4080 report found, more pronounced here. So on the 4090 the deploy misses come mostly from the
learned prior, not the cold analytic (except `square.512`, below).

## Per-shape outcome (live -O3 A/B, greedy vs recorded golden; SLOWER shapes are the findings)

| shape | eager | greedy | golden | greedy/gold | outcome |
| --- | --- | --- | --- | --- | --- |
| `square.512` (fp32) | 10.3 | 14.7 | 10.1 | **1.466** | SLOWER — split-K golden unreachable (Finding 1) |
| `square.4096` (fp32) | 2299 | 2957 | 2536 | 1.166 | SLOWER — learned buries golden (rank 458/542) |
| `square.1024.fp16` | 16.1 | 26.0 | 23.3 | 1.117 | SLOWER (learned rank 1131) |
| `rms_norm.k8192` | 13.8 | 14.8 | 13.3 | 1.116 | SLOWER (memory-bound) |
| `square.2048` (fp32) | 308 | 387 | 360 | 1.073 | SLOWER |
| `mlp_gate_up.h4096.dynM` | 748 | 1031 | 967 | 1.067 | SLOWER (masked TILE) |
| `mlp_down.h4096` | 376 | 384 | 366 | 1.049 | SLOWER (same knobs → codegen) |
| `qkv.h4096` | 327 | 368 | 354 | 1.042 | SLOWER (same knobs) |
| `square.1024` (fp32) | 43.4 | 57.9 | 65.0 | **0.891** | **FASTER → recorded** |
| `mlp_gate_up.h4096` | 748 | 964 | 969 | 0.995 | same |
| memory-bound (reduce/rms_norm/softmax/pointwise, static+dynM) | — | — | — | ~1.00 | same (beat cuBLAS 0.5–0.95×) |

Absolute latencies are ~1.6× the 4080's across the board (e.g. `square.2048.fp16` 112 vs 188 µs) — see
[`rtx4080-vs-4090-comparison.md`](rtx4080-vs-4090-comparison.md). TILE was the systematically-missed knob:
`eval golden` matched TILE in only **4/17** matmul shapes (STAGE 15/17).

## Finding 1 — `square.512` fp32 (1.47×): the `g2k` split-K golden is unreachable by the cold prior

**Symptom.** The recorded golden is `n16x8/f4x8` **`g2k`** (split-K) @ 10.1 µs; greedy deploys the **non-split**
`n32x8/f2x8` @ 14.7 µs — **46% slower**, the single worst miss of the sweep.

**Root cause.** `eval variants` marks the greedy `n32x8/f2x8` as tune-**rank 1/61** (it's the best *non-split*
config the tune measured), but the split-K golden is not what the cold prior reaches: `eval analytic` ranks the
`g2k` golden at **197/2372** — the **lone deep outlier** in an otherwise excellent analytic (every other golden
ranks 0–1). So the cold analytic does **not value split-K for this small square**. This is **4090-specific**: at
512³ on a **128-SM** card the non-split grid badly underfills the SMs, so `g2k` split-K (more CTAs → more resident
SMs) is a large win — but the analytic's cold ranking, tuned across cards, doesn't offer/price it for small
free-dims on a high-SM part.

**Recommendation (high priority).** Teach the analytic to value `g2k` split-K when the non-split grid underfills
the card (a `D_*` feature keyed on `ctas < sm_count` for small free×red shapes), and refit
(`scripts/golden_knob_heuristics.py`). This is the highest-leverage 4090 gap — a 46% deploy miss on a canonical
shape.

## Finding 2 — the learned prior buries goldens the analytic nails (`square.4096`, `square.1024.fp16`)

**Symptom.** `square.4096` fp32 greedy 1.17×, `square.1024.fp16` 1.12× — both SLOWER, and both have the golden
ranked **#0 by the analytic** but **458/542** and **1131/8838** by the *learned* prior. The learned prior's median
golden rank is **182** vs the analytic's **0**.

**Root cause.** The learned CatBoost is well-calibrated on the *measured* node store (calibration +0.90,
reachability median 1.03×) but mis-ranks over the *full gated enumeration* it only sparsely sampled — it
extrapolates worse than the (now-good, #347) analytic. Since `compile`/`run` deploy from the learned prior over the
full enumeration, its extrapolation error is the deploy miss.

**Recommendation.** Have the deploy prior **defer to the analytic in regions where the learned model is
out-of-distribution** (few measured neighbors) — on this sweep the analytic would have deployed both goldens
directly. Failing that, a learned-prior refit / more measured coverage of the fp16 + big-square enumeration.
(Lower-priority than Finding 1, which the analytic *also* misses.)

## Finding 3 — recorded `emmy_us` values have drifted on #347 (codegen regression on `square.1024`)

**Symptom.** The `square.1024` golden config `n16x8/f4x8` was recorded at **55.4 µs** but benches a stable **65.0
µs** on #347 (17% slower, 0.3% spread over 4 passes). Greedy found `n32x8/f2x8` @ 57.9 µs — 10.9% below the *live*
old golden, but still 4.5% above the stale recorded 55.4. So the shape's best-achievable **regressed** somewhere
between the golden's recording and #347.

**Root cause / status.** Not root-caused this sweep (would need the pre-#347 build to bisect). It's recorded as a
replace (the new config *is* the current deployable best), but the emmy_us going **up** (55.4→57.9) is the tell:
several recorded goldens predate recent codegen changes and their `emmy_us` are stale. The A/B's golden re-bench is
the trustworthy number; the recorded values are not.

**Recommendation.** A full `emmy_us` refresh of `rtx4090_sm89.yaml` from this sweep's -O3 A/B (the recorded values
drift across codegen changes); and a bisect of the `square.1024` fp32 regression (55.4 → 57.9 best-achievable).

## Repro / artifacts

- Work dir `_tune/golden-full-4090/`: `ab/*.json` (43 shapes + noise-floor passes), `eval/*.txt` (the 7 views),
  `logs/sweep.log`, `autotune.db` (863 MB — harvested, so the CPU-only `eval` views re-run offline), `prior.json`.
- Re-run any ranking view offline: `EMMY_TUNE_DB=_tune/golden-full-4090/autotune.db
  EMMY_PRIOR_FILE=_tune/golden-full-4090/prior.json emmy eval analytic` (GPU-benching views like `eval prior
  --dataset golden` need the live 4090).

## Workflow notes

- **Slowest step: the 43-shape sweep (~180 min)** — the 4090 golden set is ~5× the 4080's; the `h4096` matmuls +
  `.dynM` twins are ~10–17 min each. The A/B pass was ~30 min (cached cubins). Lever: the memory-bound `.dynM`
  twins reproduce their static twins within ~2% (all "same") — skipping them would trim ~10 shapes with no loss, as
  the skill notes.
- **Noise-floor is essential and cheap.** It reclassified 2 of 3 "wins" to parity (the golden benching slow in
  pass 1). 3 extra passes on the candidates cost ~5 min and prevented recording two non-wins.
- **A remote-run redirect bug bit me:** I launched the A/B driver with `> log 2>&1 … >/dev/null 2>&1`, and the
  second redirect silently won, so the completion-marker poll never fired. The 43 JSONs were complete regardless
  (I verified via a parse), but the poller had to be killed manually. Lever: one redirect per detached launch;
  verify the log grows before arming a poller on it.
- **Harvest the DB, not just the reports.** Pulling `autotune.db` (863 MB) back means the CPU-only `eval` views
  (analytic rank, fork regret, golden diff, variants) re-run offline after teardown — only the GPU-benching views
  (`eval prior --dataset golden`) need the live card. Worth the transfer on a rented, torn-down box.
