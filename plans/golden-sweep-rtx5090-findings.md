# Golden sweep — RTX 5090 (sm_120), 2026-07-06

Full re-tune + greedy-vs-golden A/B of all 36 recorded goldens on the local RTX 5090 (sm_120). Numbers are `-O3`
deployable; the A/B rows are live re-benches from `emmy run --bench --golden NAME`. `ratio` in the outcome table is
`greedy / best-golden` (>1 = greedy slower); `vs cuBLAS` is `greedy / cublas_us` (>1 = emmy slower than
PyTorch/cuBLAS).

- **Sweep command:** `emmy tune --dataset golden --clean` (39 tune targets), then `emmy run --bench --golden <name>`
  per shape (36 shapes), with 3× re-benches on every win candidate for the noise floor.
- **Wall time:** ≈40 min — the tune dominated (≈30 min, 39 shapes in one in-process loop); the 36-shape A/B + the
  candidate re-runs were ≈10 min.
- **Category tally:** **5 replaced / 0 added / 26 unchanged / 5 worse.**
  - **Replaced (5):** `reduce.k2048`, `reduce.k8192`, `reduce.k2048.dynM`, `reduce.k8192.dynM` (all `b32`→wider fold),
    `attention.hd64` (bare `TILE`→per-axis `TILE@dd`/`TILE@pj`).
  - **Worse (5), left untouched — the findings below:** `matmul.mlp_down.h4096`, `matmul.mlp_down.h4096.dynM`,
    `attention.hd128.dynM`, `softmax.k2048`, `softmax.k2048.dynM`.

The five wins all reproduced clearly above the ~10–13% small-shape noise band (3/3 re-benches); after recording, the
`golden NAME` row re-benches identically to the greedy pick (verified: `attention.hd64` 8.6/8.5, `reduce.k8192` 3.3/3.3,
`reduce.k2048` 1.5/1.5, same knobs). No matmul goldens changed, so `scripts/golden_knob_heuristics.py` (the `_W_A` /
`_W_A_DYN` analytic refit) did **not** need re-running this sweep.

## What the reduce wins mean

The four `reduce` goldens were pinned at `REDUCE=b32` — the old capped coop ladder. The `b512` extension (that fixed
`rms_norm`/`softmax` in PR #317) also widened the fold the tuner enumerates for bare reductions, and the greedy prior
now reaches `b64` (K=2048) / `b128` (K=8192) — the per-K optimum for a bare `torch.sum`, which needs far fewer
warps/row than the fused normalizers. `reduce` is a **weak** baseline (unfused `torch.sum`, so ratios stay ≥5×), but
the greedy pick is a genuine 10–23% step over the recorded `b32` golden and is what deploys, so the goldens now match
it. `attention.hd64` is the other win: the greedy found a per-axis `w4x1` MMA tile (`TILE@dd`/`TILE@pj`) that beats the
old single-`TILE` `w1x1` golden by ~12% (8.6 vs 9.6 µs).

## Full outcome table (all 36 shapes)

`greedy µs` = the deployed pick (e2e where split-K, else kernel-sum TOTAL); `golden µs` = recorded `emmy_us` (post-edit
for the 5 replaced); `cuBLAS µs` = recorded `cublas_us` (config-independent torch reference). Reference per kind: matmul
fp16 → HGEMM (fp32 `square.512` → SGEMM, a *soft* ref); attention → torch SDPA; softmax/rms_norm → torch **fused**
eager; reduce → `torch.sum` (unfused, **weak**); pointwise → torch `relu`.

| shape | S/D | greedy µs | golden µs | ratio g/gold | cuBLAS µs | vs cuBLAS | category |
|---|---|--:|--:|--:|--:|--:|---|
| matmul.square.512 (fp32) | sta | 10.2 | 10.24 | 1.00 | 12.28 | 0.83 | same (same knobs) |
| matmul.square.512.fp16 | sta | 4.2 | 3.62 | 1.16 | 6.14 | 0.68 | same (stale golden µs, see note) |
| matmul.square.1024 | sta | 15.4 | 15.59 | 0.99 | 14.51 | 1.06 | same (noise) |
| matmul.square.2048 | sta | 97.7 | 101.95 | 0.96 | 98.36 | 0.99 | same (noise) |
| matmul.square.4096 | sta | 659.3 | 640.83 | 1.03 | 640.93 | 1.03 | same |
| matmul.square.512.dynM | dyn | 6.1 | 6.14 | 0.99 | 6.14 | 0.99 | same (prev finding #1, resolved) |
| matmul.qkv.h4096 | sta | 250.8 | 258.47 | 0.97 | 250.54 | 1.00 | same (noise) |
| matmul.o_proj.h4096 | sta | 102.3 | 101.54 | 1.01 | 95.22 | 1.07 | same |
| matmul.mlp_gate_up.h4096 | sta | 604.0 | 595.79 | 1.01 | 557.94 | 1.08 | same |
| matmul.mlp_down.h4096 | sta | 346.7 | 303.60 | **1.14** | 296.95 | **1.17** | **worse — Finding 1** |
| matmul.qkv.h4096.dynM | dyn | 252.3 | 258.46 | 0.98 | 249.82 | 1.01 | same (noise) |
| matmul.o_proj.h4096.dynM | dyn | 102.1 | 100.93 | 1.01 | 95.08 | 1.07 | same |
| matmul.mlp_gate_up.h4096.dynM | dyn | 623.4 | 596.30 | 1.05 | 553.70 | 1.13 | same (borderline) |
| matmul.mlp_down.h4096.dynM | dyn | 375.6 | 305.10 | **1.23** | 295.03 | **1.27** | **worse — Finding 1** |
| softmax.k2048 | sta | 3.9 | 3.7 | 1.05 | 4.1 | 0.95 | **worse — Finding 3** |
| reduce.k2048 | sta | 1.5 | 1.5 | 1.00 | 16.38 | 0.09 | **replaced** (b32→b64) |
| softmax.k8192 | sta | 10.5 | 10.3 | 1.02 | 14.29 | 0.73 | same (same knobs) |
| reduce.k8192 | sta | 3.3 | 3.3 | 1.00 | 16.39 | 0.20 | **replaced** (b32→b128) |
| rms_norm.k2048 | sta | 3.9 | 3.8 | 1.03 | 4.1 | 0.95 | same (same knobs) |
| rms_norm.k4096 | sta | 6.6 | 6.7 | 0.99 | 6.14 | 1.07 | same (b512≈b256, noise) |
| rms_norm.k8192 | sta | 10.2 | 10.2 | 1.00 | 10.24 | 1.00 | same (same knobs) |
| pointwise.n4096 | sta | 3.3 | 3.3 | 1.00 | 4.1 | 0.80 | same (same knobs) |
| pointwise.n16384 | sta | 11.3 | 11.1 | 1.02 | 12.51 | 0.90 | same (same knobs) |
| softmax.k2048.dynM | dyn | 3.9 | 3.7 | 1.05 | 4.1 | 0.95 | **worse — Finding 3** |
| reduce.k2048.dynM | dyn | 1.5 | 1.5 | 1.00 | 16.38 | 0.09 | **replaced** (b32→b64) |
| softmax.k8192.dynM | dyn | 10.4 | 10.1 | 1.03 | 14.27 | 0.73 | same (same knobs) |
| reduce.k8192.dynM | dyn | 3.3 | 3.3 | 1.00 | 16.39 | 0.20 | **replaced** (b32→b128) |
| rms_norm.k2048.dynM | dyn | 4.0 | 3.9 | 1.03 | 4.1 | 0.98 | same (b256 vs golden b512, noise) |
| rms_norm.k4096.dynM | dyn | 6.3 | 6.7 | 0.94 | 6.14 | 1.03 | same (b512≈b256, noise) |
| rms_norm.k8192.dynM | dyn | 10.1 | 10.4 | 0.97 | 10.25 | 0.99 | same (same knobs) |
| pointwise.n4096.dynM | dyn | 3.3 | 3.3 | 1.00 | 4.1 | 0.80 | same (same knobs) |
| pointwise.n16384.dynM | dyn | 11.3 | 11.1 | 1.02 | 12.52 | 0.90 | same (same knobs) |
| attention.hd64 | sta | 8.6 | 8.6 | 1.00 | 10.24 | 0.84 | **replaced** (w1x1→per-axis w4x1) |
| attention.hd128 | sta | 16.5 | 16.5 | 1.00 | 18.42 | 0.90 | same (same knobs) |
| attention.hd64.dynM | dyn | 19.7 | 19.5 | 1.01 | 10.23 | **1.93** | same (masked-flash residual, Finding 2) |
| attention.hd128.dynM | dyn | 59.6 | 29.6 | **2.01** | 18.4 | **3.24** | **worse — Finding 2** |

## Finding 1 — greedy still misses `g8k` split-K on the K-heavy `mlp_down` GEMM

**UPDATE — split-K candidacy feature + golden re-recorded (branch `feature/dynamic-flash-staging`).** The winning
`mlp_down` config (`w4x4/f2x2/k4 + g8k`, ~303 µs) had been lost from the golden — re-recorded to the unstaged greedy
tile at 346 µs; restored for static + `.dynM`. Root cause of the prior miss: split-K carried ONLY penalties in the
feature space (`D_splitk_le2` / `D_splitk_excess`), never a reward when justified, and the occupancy-only `needed`
credited zero split for K-heavy shapes whose free dims already fill the SMs. Added `D_splitk_deficit`: K-heaviness
`K/√(M·N)` folds a second floor into `needed`, and under-splitting past it is a penalty (verified: `mlp_down` g8 in the
sweet spot, balanced GEMMs still prefer g1). Masked-tier cold rank `mlp_down.dynM` 3833→1351 at no aggregate-median
cost. **Static cold-prior limit:** the LINEAR `_W_A` cannot isolate the `g8k` geometry among the many balanced GEMMs
(rank stays ~10k; any stronger deficit trades the aggregate median for a modest gain) — the static split-K deploy is a
CatBoost lever (the refit's `build_cases` is matmul-only, and the nonlinear prior is the model that can price the
interaction). The feature is in place for that retrain.

## 2 — deployed greedy picks a suboptimal MMA tile for `square.512.dynM`

Evidence:

- `eval golden --kernel mlp_down` → `found/golden` = `-/g8k` (static) and `g2k/g8k` (dynamic). The split-K knob is
  exactly what the greedy pick gets wrong; the MMA tile is right.
- `eval prior --dataset golden` `vs gold` (-O3) = **1.14×** (static) / **1.18×** (dynamic) — reproduces the A/B.
- `eval analytic --kernel mlp_down` ranks the `g8k` golden **9116 / 14805** (static), **2226 / 14805** (dynamic). The
  cold analytic prior badly misprices split-K on this shape family — it can't reach it cold.
- `eval prior --dataset golden` learned rank = **528 / 14805** (static). The learned prior lifts it from 9116→528 but
  still scores 527 configs above `g8k`, so the greedy pick never deploys it.

This is the same standing lever the PR #317 report flagged ("teach the search to try split-K on K-heavy static
GEMMs") — it is *unfixed*: the golden still holds because neither the analytic nor the learned prior surfaces `g8k`.
**Recommendation (high):** add an engineered feature that fires split-K candidacy on high-K/N-ratio GEMMs (a `D_*`
feature on `K / max(M,N)`), then refit the analytic weights over it — a 9116-deep analytic rank is a heuristic
mispricing, not a search-patience problem, so patience bumps alone won't reach it.

**UPDATE — softmax coop-reduce ladder extended to `b512`; golden re-recorded.** `softmax.k2048` had regressed to
`REDUCE=b32` (9.5 µs) because `space.coop_reduce_moves` capped the ladder at `b32`, so `b512` (3.7 µs — **2.6×**, 0.9×
cuBLAS) was unreachable by the search. Restored the wide folds `b64`–`b512` (the scheduler's `_coop_reduce_spec`
already gates per-shape legality, so it's safe on small K); re-recorded `softmax.k2048`(.dynM) at `b512`. **Greedy now
DEPLOYS `b256` (3.9 µs)** — a 2.4× deploy win over the `b32` it shipped before — with `b512` the golden. (`rms_norm`'s
bandwidth gap at wide K is untouched — a separate codegen lead.)

## 4 — attention golden re-benches are pathological on the current build

Non-causal static flash is *faster* than torch SDPA (`hd64` 0.84×, `hd128` 0.90× vs cuBLAS — both now at/above
parity). The masked `.dynM` twins are far behind: `hd64.dynM` sits at **1.93× cuBLAS** (19.5 µs golden vs 10.2 µs
SDPA) and `hd128.dynM` at **3.24×** (29.6 µs golden vs 18.4 µs SDPA). Two distinct problems stacked here:

- **Masked-streaming-flash residual (both twins):** even the golden loses ~2–3× to static SDPA. On `hd64.dynM` the
  greedy pick ≈ golden (19.7 vs 19.5, within noise), so this is *not* a config miss — it is genuine masked-flash work
  in the streaming inner loop. Unchanged from PR #317's Finding 2.
- **Prior shortfall on top (`hd128.dynM`):** the greedy pick (59.6 µs) is a further **2.01×** behind its own golden
  (29.6 µs). The greedy `TILE@dd=w2x1/f1x8/k8` / `TILE@pj=w2x1/f1x16/k4` splits the K-chunk the wrong way vs the
  golden's uniform `w2x1/f1x16/k8` and falls to a scalar path. `eval analytic`/`eval variants` return **empty** for
  attention kernels (the matmul-only enumeration doesn't cover flash), so there is no rank/reachability view to lean
  on here — the A/B is the only signal.

**Recommendation (high for the residual, medium for the pick):** profile the masked-streaming-flash inner loop vs the
static twin (NCU) to attack the 2–3× residual; separately, the `hd128.dynM` K-chunk pick needs the flash TILE
enumeration to prefer the uniform-k form — but that can't be diagnosed until `eval analytic`/`variants` learn to
enumerate attention shapes (see Workflow notes).

Non-causal flash is *faster* than torch SDPA static (hd64/hd128 ~1.11–1.12×), but the `.dynM` masked flash is
0.52–0.60× (golden), 1.7–1.9× behind static. `hd128.dynM` greedy (59.49) is a further 1.96× behind its own
golden (30.43) — a prior shortfall on top of the masked-flash residual. Unlike the matmuls this is not a tier
fallback (the flash goldens already use MMA); it is genuine masked-streaming-flash work.

**Part A (the residual) — ADDRESSED, branch `feature/dynamic-flash-staging`.** The residual was *not* intrinsic
masked-flash cost — it was the missing K/V staging. `_resolve_twisted_stage` only staged a static, block-divisible
kv; TMA rides the runtime globalDim and zero-fills the box overhang, and the streaming drain already masks the tail
keys, so the zero-filled tail contributes nothing — bit-identical to gmem-direct. Admitting a symbolic kv under **TMA**
(cp.async stays static-only: no OOB zero-fill) + threading the symbolic `Dim` through `staged_kloop` makes `.dynM`
flash stageable. 5090 `-O3`, `hd64.dynM` snippet, same warp tile, gmem→+TMA: `w1x1` 19.2→16.9, `w2x1` 22.2→9.2,
**`w4x1` 26.3→9.1 µs** — best staged **0.89× cuBLAS** (torch SDPA 10.2), ~at the static hd64 8.6. Bit-identity
test-enforced at a divisible (64) and overhanging (100) seq.

**Part B — golden re-recorded + scalar fallback fixed; greedy DEPLOY still open.** Re-recorded `hd64.dynM` (9.1 µs, was
19.6; bare `w4x1/f1x8/k4` + `STAGE=d2/tma/ring`) and `hd128.dynM` (16.4 µs, was 30.8; `w2x1/f1x8/k8` + stage). Fixed the
flash-form-fork **scalar fallback**: a `STAGE` pin now keeps the warp rows alone (only the warp tier stages), so the
`--ab STAGE=…` / `emmy tune` staging probe and `EMMY_STAGE` no longer collapse to a ~100× scalar row (regression-tested
at H=8/seq=512). Result: **static** flash greedy-deploys the staged form (8.5 µs); **dynamic** flash greedy still
deploys the unstaged warp (19.6 µs) — the masked-tier LINEAR prior underprices the low-occupancy staged form, and the
analytic refit can't learn it because `build_cases`/`eval analytic` are matmul-only (attention isn't enumerated).
Boosting the dyn `D_stage_*` weights was inert (the flash gmem-vs-staged decision rides features the matmul-only eval
can't surface). The staged config is reachable (golden / pin / `--ab`), so a served model that tunes deploys it;
greedy-by-default staged `.dynM` flash is the remaining **CatBoost lever** (train over attention shapes). **Next:**
retrain the CatBoost prior over the goldens (incl. attention) so greedy deploys the staged flash + `g8k` split-K.

Small but reproducible: `softmax.k2048` (static + `.dynM`) golden is `REDUCE=b512` at 3.7 µs; the greedy prior deploys
`b256` at 3.8–3.9 µs (**~5%** slower, 3/3 re-benches: 3.9 / 3.8 / 3.9). This is a mild prior shortfall — the fold
ladder now *reaches* `b512` (the `rms_norm.k2048` golden uses it and the greedy deploys it there), but for the softmax
variant at K=2048 the prior scores `b256` first. Note the asymmetry: for `rms_norm.k4096`/`.dynM` the greedy *does*
find the wider `b512` and it's at parity with the recorded `b256` (left unchanged as noise). **Recommendation (low):**
this is within a whisker of the noise floor and torch-parity either way (0.95× cuBLAS); a patience bump on the
K=2048 coop-reduce fork would close it, but it is not worth an engineered feature.

## Notes on the near-misses left unchanged

- **`matmul.square.512.fp16`** — recorded `emmy_us=3.62`, but the golden re-benches at 4.3 µs (3.4 main + 0.9 reduce)
  and the greedy pick is 4.2 µs (single kernel, no split-K). Greedy ≈ live golden; the recorded 3.62 looks like a
  main-kernel-only number from before the split-reduce and is optimistic. Left as-is (matmul golden, no analytic refit
  wanted this sweep) but flagged — a re-record would land ~4.2.
- **`matmul.square.2048` / `qkv.h4096` / `qkv.h4096.dynM`** — greedy benched 2–4% *faster* than the recorded golden,
  but the live golden rows re-benched pathologically slow (`square.2048` golden 125 µs vs recorded 102 — the Step 4
  flatten caveat), so the "win" is the golden benching slow, not a real greedy gain. All within the 5% noise band vs
  the recorded number → no change.

## Workflow notes

Retrospective on this loop, for whoever maintains the CLI + skill. PR #317's report raised three tooling gaps; status
of each below, then this sweep's new friction.

- **`eval variants --kernel <shape>` never matches.** `eval variants --kernel square.2048` / `mlp_down.h4096` /
  `hd128` all return *"No measured variants matching"* even though the shape was just tuned — the `--kernel` filter
  matches the DB **kernel-hash** name (`k_matmul_631750`), not the golden shape name that `run --bench --golden NAME`
  and `eval golden`/`analytic`/`prior` all use. This made the reachability leg of every finding unusable; I fell back
  to `eval golden` + `eval analytic` + the A/B. *Fix:* let `--kernel` match the golden/shape name (or accept both), so
  the four `eval` views share one key.
- **`eval analytic` / `eval variants` are matmul-only.** Both return empty for attention/softmax/reduce/pointwise, so
  Findings 2 and 3 (attention, softmax) have no rank/reachability evidence — only the live A/B. *Fix:* extend the
  analytic enumeration + variants join to the non-matmul kernel kinds, or at least print a "not supported for kind K"
  line instead of a bare empty table (I had to probe to learn it was unsupported vs. a filter miss).
- **Golden e2e-vs-kernel-sum ambiguity (PR #317 gap, still open).** `run --bench --golden` prints golden rows as
  kernel-sum µs while the split-K shapes' real cost is e2e (`o_proj.h4096` kernel-sum 99.7 but e2e 102.3;
  `mlp_down.dynM` 374.3 vs 375.6). I recorded e2e for the split-K shapes and kernel-sum otherwise, by hand. The
  proposed `e2e` column per golden row would remove the judgment call — still worth doing.
- **Which prior produced the greedy pick (PR #317 gap, still open).** `eval prior` does print the loaded prior path
  now (`prior.json (loaded)`), which is progress, but `compile`/`run --bench` still don't say cold-analytic vs learned.
  Not a blocker this sweep (fresh `--clean` prior), but the ask stands.
- **Small-shape noise re-runs (new).** The reduce/softmax/small-matmul goldens swing ~10–13% run-to-run, so every win
  candidate needed 3 benches (Step 4). The reduce wins (10–23%) and `attention.hd64` (12%) reproduced cleanly; the
  matmul "wins" all collapsed into noise on the recorded number. A `--repeat N` flag on `run --bench --golden` that
  benches each golden row N times and prints min/median would fold Step 4 into Step 2 and cut hand re-running.
- **`! golden` rows carry a physics warning inline (new, minor).** The tiny `reduce.*.dynM` golden rows print with a
  `!` prefix and an `impossible: implies 649 TFLOP/s > 105 device peak` note (a sub-µs bench tripping the FLOP sanity
  check). Harmless here but it initially read like a golden failure — the warning should say "sub-µs bench, FLOP
  estimate unreliable" rather than "impossible".
