# Fused cold deploys: root cause, fixes, and evidence (2026-08-27)

Three controlled runs on one RTX 4090 (same 12 Qwen3-0.6B layer-0 targets, budget 12 / patience 4 / seed 0)
separate the failure into independently provable parts. Phase-1 = pre-#648 compiler, tune ranking at `-Xcicc -O1`
(the old default). Phase-3 = same compiler and targets, tune under `EMMY_NVCC_FLAGS=` (deployable regime — what
upstream #660 has since made the default). Local CPU replays of the phase-3 evidence against the current branch
close the loop.

## Findings

1. **-O1 tune evidence can never reach a deploy** (phase-1). All 866 perf rows keyed to the `-O1` context; the
   verified/reservoir/DB tiers each exclude non-deployable rows by design. Verify deploys were decided by the cold
   heuristic: `k_mean_20f978` knobless at 445µs (eager 141µs), both s512 SDPA fusions at 211ms. Upstream #660
   already retires this; phase-1 is its controlled demonstration.

2. **Deployable-regime tune measures everything the deploy needs** (phase-3). 810 `ok` rows in one -O3-keyed
   context, including the catastrophic realizations (bench truncation, commit 121af376, keeps the 860ms-per-launch
   kernels measurable under the 2s tune watchdog). `k_mean_20f978`'s best measured realization: 1.6µs (88× eager).

3. **The replay validity check rejected proposal-sourced winners** (phase-3 verify, all reps rc=2).
   `persist_tune_winner` keeps `source: proposal` when the searched winner lands on a measured proposal row;
   `run --golden` demanded `source == "tune"` and exited 2 for the whole file. Fixed in 9620d8e6 (reader accepts
   both; regression test).

4. **The old deploy stack still went cold with readable evidence beside it**: the structural (keep-fused vs cut)
   fork was priced model-vs-model — keep-fused predicted 3.79µs vs cut 3.87µs (a 2% model-noise margin) — so the
   raw fused kernel won and ran 447µs, a 118× prediction error. The measured 1.6µs cut fragment could not vouch at
   that fork. The current branch's direct measured descend closes this: replaying the same evidence locally, the
   deploy picks the fused kernel's measured `t8/coop` row at 12.1µs (11.7× faster than eager; the 447µs cold pick
   is gone). Residual: the direct pick short-circuits structural pricing, so the measured 12.1µs fused row wins
   without being compared against the ~3µs cut side — a bounded suboptimality, not a cold deploy.

5. **Post-#660 context keys orphan pre-#660 tune DBs.** `split_opt_level` respells the deployable regime's key
   (`("Context", cc, "")` → `("Context", cc, 3, "")`), so every row tuned before #660 is invisible to deploys after
   it. Re-tune, or rekey old rows, after upgrading across #660.

6. **On current main (#648 maximal fusion), per-op golden tuning is impossible**: a fresh layer trace collapses to
   ONE whole-layer kernel (47GB host RAM, >90min at -O3 — uncompilable), and pre-#648 working files fail replay
   with "provenance target no longer resolves after lowering" plus ambiguous `REDUCE@...` knob paths. Upstream
   report; blocks running this branch's fixes end-to-end until #648 is amended. It also breaks evidence
   portability for the complex fused forms: the rebased lowering emits different `S_*` structural features for the
   SDPA fusions, so pre-rebase tune rows match zero deploy candidates there (the simpler k_mean signature happens
   to survive, which is what made the finding-4 replay possible).

7. **The s512 SDPA fusion targets have no good fused realization**: 231 measured candidates, ranking calibration
   +0.96, and every realization keeping the fused SDPA kernel runs ~860ms/launch. The s1 (decode) SDPA variants of
   the same source fusion run 15–30µs — the pathology is specific to the prefill shape's fused form. Schedule /
   decomposition work (fix #3), not evidence work.

## Verified phase-3 deploy table (s512, 3 reps, sub-1% spread; eager / torch.compile / emmy µs)

| target | eager | tc | emmy | note |
|---|--:|--:|--:|---|
| k_mean_3cfe9d | 276.9 | — | 5.8 | tuned `t32/coop` pinned, 47.9× eager |
| k_linear_a44745 | 63.2 | 19.6 | 21.1 | mma winner pinned |
| k_linear_acf1a1 | 41.1 | 12.3 | 20.1 | mma winner pinned |
| k_linear_mean_reduce | 190.2 | 52.3 | 87.1 | mma winner pinned |
| k_mean_20f978 | 141.3 | 5.6 | 447.0 | finding 4 — cold on the old stack; 12.1µs on current branch replay |
| SDPA fusions ×2 | ~50 | ~42 | ~850k | finding 7 |
