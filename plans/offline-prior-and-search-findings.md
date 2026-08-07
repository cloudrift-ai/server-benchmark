# Offline prior and automated search — investigation findings

RTX 5090, 2026-08-05 → 2026-08-07. Branch `feature/prior-fit-deployed-score`.

## For a reader with no context

When emmy compiles a GPU kernel it must choose a *schedule*: tile sizes, how operands are staged through shared
memory, whether the reduction is split across thread blocks, and so on. Thousands to hundreds of thousands of legal
combinations exist per kernel, and they differ in speed by orders of magnitude. Two mechanisms choose:

- **The prior** — a model that predicts which schedule is fast, consulted when nothing has been measured. It has two
  halves. The **offline prior** is a fixed linear model shipped in the repo, fitted ahead of time on *goldens*
  (schedules verified fastest for a shape and recorded in YAML). The **online prior** is a CatBoost model trained
  during a tuning run from that run's own measurements.
- **The search** (`emmy tune`) — explores schedules by compiling and timing them, steered by the prior.

At deploy time a compile walks an **evidence hierarchy**: recorded goldens first, then measured data, and only if
none of that exists does the prior decide. So the prior governs exactly the shapes nobody has tuned yet.

The team had stopped using `emmy tune` to find new goldens and gone back to sweeping by hand. This investigation
asked why the prior ranks goldens poorly, fixed what it found, and then tested on real hardware whether automated
search can now replace the hand sweep.

## All experiments, side by side

### Golden ranking — every model tried, same candidate pools, no GPU

How far down its own candidate pool each model puts the known-best config. Pessimistic rank; lower is better. `top1`
is the only rank a greedy deploy actually gets. Pools are ~57k candidates (PRE) and ~119k (POST).

`PRE` rows are the pre-#465 dataset (280 pools, 242 static cases); `POST` rows are post-#465 (286 pools, 248 static).
**Ranks are not comparable across that line** — the tile-scheduler rebuild roughly doubled every candidate pool.

| model / fit | n | median | top1 | top10 | top100 | mean log2 |
| --- | --: | --: | --: | --: | --: | --: |
| PRE shipped weights (what ships today) | 242 | 228 | 5 | 66 | 98 | 7.13 |
| PRE shipped weights **+ the deploy gates** | 242 | 367 | 5 | 47 | 100 | 7.62 |
| PRE repo trainer, refit on this slice, as-is | 242 | 111 | 3 | 60 | 115 | 6.34 |
| PRE repo trainer, tier weighting neutralized | 241 | 57 | 12 | 85 | 160 | 4.91 |
| PRE repo trainer, seeded from zeros | 242 | 523 | 5 | 71 | 74 | 7.36 |
| PRE convex linear, in-sample | 242 | 6 | 2 | 131 | 193 | 4.01 |
| PRE convex linear, **held out** | 242 | 9 | 2 | 122 | 175 | 4.36 |
| PRE convex linear + `MMA_*`, in-sample | 242 | 5 | 2 | 134 | 208 | 3.57 |
| PRE convex linear + `MMA_*`, **held out** | 242 | 8 | 3 | 126 | 186 | 3.92 |
| PRE CatBoost, in-sample | 242 | 1 | 32 | 204 | 232 | 1.99 |
| PRE CatBoost, **held out** | 242 | 3 | 13 | 169 | 211 | 3.06 |
| PRE new trainer, all fixes applied | 242 | 45 | 57 | 99 | 165 | 4.52 |
| POST new trainer, before the step feature | 248 | 160 | 0 | 79 | 119 | 5.99 |
| **POST final scoped artifact** | 248 | **17** | **48** | 93 | 163 | 4.65 |

Reading it: the shipped weights get *worse* when the deploy gates are applied (228 → 367) — those numbers were never
what deployed. Seeding from zeros is worse than the incumbent (523), and the convex fit beats the repo's optimizer on
the repo's own objective (4.01 vs 4.91) — an optimizer problem, not a model-class one. Held-out rows are within a few
points of in-sample throughout, so nothing here is a generalization failure. CatBoost is the only genuine model-class
gain, and it is the last 3× of a 25× gap.

### Hardware — 10 Qwen3-1.7B shapes with no goldens, RTX 5090

Microseconds, all pinned rows at deployable flags so they are mutually comparable. `cold` is the offline prior's own
pick with no search and nothing measured. Lower is better; **bold** marks the best per row.

| shape | eager (cuBLAS) | cold prior | A: tune (`-O1` p50) | C: online off | D: tune `-O1` p15 | D: tune **`-O3`** p15 | B: hand sweep |
| --- | --: | --: | --: | --: | --: | --: | --: |
| qk_proj.m32 | 8.93 | 7.79 | 7.68 | 7.60 | 7.60 | 7.43 | **6.39** |
| v_proj.m32 | 6.15 | 3.67 | 3.75 | 3.75 | 3.75 | 3.57 | **3.13** |
| o_proj.m32 | 6.15 | 4.91 | 5.60 | 5.58 | 5.60 | 4.55 | **4.12** |
| gate_up.m32 | 26.64 | 18.05 | 15.76 | 15.27 | 21.65 | 15.75 | **14.77** |
| down.m32 | 12.29 | 8.90 | 10.40 | 10.39 | 10.39 | 9.15 | **7.83** |
| qk_proj.m512 | 49.21 | 51.59 | 41.29 | 41.26 | 50.60 | 44.99 | **39.38** |
| v_proj.m512 | 14.42 | 17.61 | 16.89 | 20.62 | 17.27 | 16.86 | **15.00** |
| o_proj.m512 | 26.64 | 38.10 | 30.52 | 28.76 | 28.77 | 28.08 | **26.02** |
| gate_up.m512 | 127.64 | 162.31 | 136.12 | 136.03 | 144.75 | 129.64 | **124.39** |
| down.m512 | 72.13 | 83.10 | 87.70 | 81.07 | 81.19 | 78.27 | **75.36** |
| **median vs hand sweep (B)** | — | — | 1.168 | — | 1.194 | **1.114** | 1.000 |
| **geomean vs eager — M=32** | 1.000 | 0.728 | 0.751 | 0.744 | — | — | **0.612** |
| **geomean vs eager — M=512** | 1.000 | 1.218 | 1.079 | 1.092 | — | — | **0.963** |
| **geomean vs eager — all ten** | 1.000 | 0.941 | 0.900 | 0.901 | — | — | **0.768** |
| **search time spent** | — | 0 s | 2185 s | 1017 s | 1153 s | 1374 s | 1110 s |

Reading it: the hand sweep wins **every shape**, in half the automated budget. The cold prior beats cuBLAS by 27% at
M=32 *with no search at all* and loses by 22% at M=512. Arms A and C land within 0.1% of each other in aggregate —
removing the online prior changed the speed of the search, not its outcome. Arm D's two columns are the controlled
flag comparison at equal patience: ranking at `-O3` wins on 10/10 shapes and, at patience 15, **beats the shipped
default on both speed and quality** (1374 s and 1.114× vs Arm A's 2185 s and 1.168×).

## What we found, in one paragraph

The prior's poor golden ranking was **not** a generalization failure and **not** mainly a model-class failure. It was
a weak fitter optimizing the wrong objective over a diluted dataset, plus two mechanical defects. Fixing those
improved golden rank on the RTX 5090 matmul slice from median **708 to 17** and top-1 from **0/248 to 48/248**. But
on hardware, against ten real shapes with no goldens, a hand-driven sweep still beat automated search on **all ten**,
and the reason is not the prior at all — it is that the search ranks candidates at a non-deployable compiler
optimization level. Separately we found a **73× misdeploy** that is a bug in the evidence hierarchy, not in any model.

---

## Part 1 — Why the prior ranked goldens badly

Method: rebuild every RTX 5090 matmul golden's candidate pool with the repo's own case builder, restricted to one
card and one kernel kind. 280 pools, 17.3 M candidates. The harness was validated by scoring those pools with the
shipped weights and reproducing `_tune/fits/20260730-l2-refit/metrics.json` **exactly on all 216 goldens the two runs
share**, so every number below is comparable with the repo's own tooling.

All ranks are the repo's pessimistic convention: ties count against the golden, because a greedy argmax breaks ties
by emission order.

### It is not generalization

Cross-validated by shape family — each held-out shape had no relative in training — a linear model on the existing
features reached **median rank 9** out of ~57,000 candidates, against **median 6** in-sample. The gap is negligible.
The shipped weights managed median 228 *on their own training data*.

### Model class matters, but it is the last 3× of a 25× gap

CatBoost on the same features and the same loss beat the best linear fit on held-out families: median **3 vs 9**,
top-10 **169 vs 122**, with a small train→holdout gap. Real headroom — but the linear form reaches 6 in-sample and 9
held out, so it is not what pinned the shipped prior at 228.

### The fitter's optimizer was the single biggest lever

`fit_weights` is random search plus coordinate descent on a non-smooth rank loss. On this slice:

- seeded from the shipped weights it reached mean log2 **4.91**; a convex listwise fit reached **3.99** — better *on
  the fitter's own objective*
- seeded from zeros it landed at **7.36**, worse than the weights it was meant to improve
- handed the convex solution it **could not move at all** — every single-coordinate step worsened the loss

### One weight vector was serving contradictory regimes

Same trainer, only the data narrowed to 5090 matmul: median **228 → 57**. The refit disagreed in *sign* with the
shipped weights on `D_reduce_transposed` (−50.98 → **+10.88**), the second-largest shipped term. One additive model
cannot hold both positions.

### Two mechanical defects

**The fit's feature view dropped the feature that decides the fast-math atom.** `MMA_acc_bits` (32 for
f32-accumulate, 16 for f16) was emitted by the featurizer but excluded by the fit's `D_*,MMA_tier` view. On a
fast-math pool, **93% of 118,206 candidates sat in two-row buckets** differing only in that feature, and **125 of 280
goldens had a feature-identical candidate ahead of them in emission order** — unrankable at top-1 by construction,
for any model.

**The deploy-time gates were not in the fit.** `OfflinePrior` subtracted two hand-set terms the fitter never saw.
Applying them: median **228 → 367**, top-10 **66 → 47**. So the numbers `emmy fit` reported were not the numbers
that deployed. The damage was entirely `splitk_roundtrip_weight = 0.25` — a feature the fit had *pruned as
worthless*, yet which deployed with a hand-set coefficient.

**The loss weighted tiers, not cases.** Each case counted `1/count(tier)`, so on the whole corpus one fp32 golden
outweighed 30 fp16 ones; on a single-tier slice one case carried half the loss (median 111 with it, 57 without).

---

## Part 2 — What changed in the code

Four commits on `feature/prior-fit-deployed-score`. All verified with `make test` (2421 passed) and `make lint`.

| commit | change |
| --- | --- |
| `25204cf1` | Fit the score that actually deploys; `MMA_acc_bits` into the view; drop the tier weighting |
| `3ce1c8db` | Merge main (#465 tile scheduler rebuild) |
| `38017fb9` | `MATMUL_FEATURES` — the 53 features that can move a matmul ranking |
| `c0af49e3` | `D_stage_prefetch` step feature; a scoped RTX 5090 matmul artifact |

**Fitting the deployed score.** `OfflinePrior.quality` and the fitter's `quality_rows` are now one function over one
shared definition, and the non-linear term's weight *and threshold* are descent coordinates — possible because the
optimizer is derivative-free. The two hand-set gates were plain linear terms on features the model already had, so
they folded into the weights; the shipped artifact was migrated so the deployed quality is algebraically unchanged
(verified: identical ranks on 400 random rows, both weight sets).

**`MATMUL_FEATURES`.** Of 71 emitted features, 19 are excluded — 12 that are constant within every pool (so the term
cancels out of any within-pool ranking) and 7 that are globally affine copies of a kept feature. Both classes are
expressiveness-neutral. Notably `D_pow2_threads` carries the shipped artifact's **largest** weight (+136.5) and
cannot change a matmul ranking at all.

**`D_stage_prefetch`.** #465 retired `D_stage_ring`, and top-1 on the matmul fit collapsed from 54/242 to 4. The flag
turned out to be *exactly* `D_stage_depth >= 2` (1.0000 agreement over 2,033,344 rows) — no information was lost. What
was lost was a **precomputed step**: a linear model cannot form an indicator from a feature it holds linearly.
Re-adding the step as a real feature recovers and exceeds the original (top-1 **65**), at less than half the weight.

**Result of the refit**, on the 286 RTX 5090 matmul pools: median golden rank **708 → 17**, top-1
**0/248 → 48/248**, top-10 **6 → 93**.

---

## Part 3 — Does automated search work on shapes with no goldens?

Golden shapes cannot answer this: on a shape that has a golden the evidence hierarchy resolves it at tier 1 and the
prior is never consulted. So we used **Qwen3-1.7B**, whose hidden 2048 / intermediate 6144 sit outside a golden
corpus that is 95/103 gemma-4-shaped. A live audit confirmed 29 of its 30 contraction forks are uncovered.

Ten shapes, fp16, `F.linear` layout, M ∈ {32, 512}, N/K from the model's projections. All measured as pinned rows at
deployable flags, so they are mutually comparable.

| | cold prior vs eager | tune vs cold | hand sweep vs cold |
| --- | ---: | ---: | ---: |
| **M=32** | **0.728** | 1.023 | 0.842 |
| **M=512** | 1.218 | **0.896** | 0.790 |
| all ten | 0.941 | 0.957 | 0.816 |

*(lower is faster; `cold prior vs eager` < 1 means it beats cuBLAS)*

**The cold prior has a sharp regime boundary.** With no search whatsoever it beats cuBLAS by 27% at M=32 — the thin
decode widths its goldens come from — and loses by 22% at M=512. It does not transfer to prefill widths.

**Tuning helps exactly where the prior is weak.** It improves M=512 by 10% and *degrades* M=32 by 2%. Better on 6 of
10 shapes, worse on 4.

**The hand sweep beat automated search on all ten shapes**, by 5–26%, using half the time budget (1110 s of 2185 s).
Its advantage is structural, not cleverness: `emmy tune` ranks candidates at `-Xcicc -O1`, which the source itself
labels a ranking signal that inverts against `-O3`. The hand sweep pins and measures at `-O3` throughout.

**Arm D measured this directly** rather than inferring it. At equal patience (15), ranking at `-O3` beat the `-O1`
proxy on **10 of 10 shapes** (median 0.924) for only **+19% wall time** — 1374 s vs 1153 s, 2.72 s vs 2.29 s per
variant. Nowhere near the multiples the `-O1` default exists to avoid, because the default is already not a pure
`-O1` pipeline: `O3_REBENCH_TOL = 2.0` re-benches every config within **3×** of the best `-O1`, so the `-O1` arm
already pays much of the `-O3` bill (13% of its node rows are `H_opt=3`, against 100% for the `-O3` arm).

The mechanism is visible in the chosen knobs: the `-O1` arm picked the narrowest fragment `f1x1` on **all five** M=32
shapes with one near-identical worker shape regardless of N/K, while the `-O3` arm picked wide fragments
(`f1x2`/`f1x4`/`f1x8`/`f2x4`) and differentiated per shape. `-O1` does not merely add noise — it **systematically
mis-ranks the wide register-tile mma family**, which is exactly the cicc unroll blowup the source comment describes.

But it only closes about 40% of the gap: `-O3` ranking lands 1.114× the hand sweep where `-O1` was 1.194×. **The
residual ~11% is the search strategy, not the proxy metric.**

**The online prior is not the cause.** With it removed from the search's selection signal, the tune still regressed
on 4 of 10 shapes — the same count, and three of the *same* shapes to within 0.4 percentage points. It did converge
2.15× faster (1017 s vs 2185 s) with more improvements from fewer measurements, but found a worse basin on one shape.

The hand sweep also produced transferable rules an automated search does not: staging dominates (same tile, no
staging 58.6 µs → `d2/cp` 15.4 µs); TMA only pays at deep `bk`; split-K is worth up to 3× where K is deep and N
narrow, neutral-to-negative at M=512, and its factor must *divide* K.

---

## Status of every issue — fixed, measured, or only reported

**Nothing here has shipped.** All code changes sit on the unmerged branch `feature/prior-fit-deployed-score`
(4 commits; `origin` is at the merge commit `3ce1c8db`, so `c0af49e3` is local only). And critically:

> **The shipped `offline_weights.json` has NOT been refit.** Its `provenance.fitted` is still `2026-07-30` and it
> still carries a `+38.28` weight on `D_stage_ring`, a feature the featurizer no longer emits. **Every fitter fix
> below is in code but absent from the artifact that actually deploys today.** Regenerating it is `emmy fit
> --artifact`, and it is the single step that converts this work into deployed behaviour.

### A — Fixed in code (committed, unmerged)

| # | issue | fix | commit |
| --- | --- | --- | --- |
| A1 | The fit's feature view dropped `MMA_acc_bits`, leaving 125/280 goldens unrankable at top-1 | added to `DEFAULT_FEATURES` | `25204cf1` |
| A2 | The fitter optimized a proxy: two hand-set gates applied at deploy were absent from the objective (median 228 reported vs 367 deployed) | `OfflinePrior.quality` and the fitter's `quality_rows` are one shared function; the interaction's weight *and* threshold are now fitted; the two linear gates folded into the weights, artifact migrated so deployed quality is algebraically unchanged | `25204cf1` |
| A3 | The loss weighted tiers not cases (one fp32 golden outweighed 30 fp16 ones) | removed; each case counts one | `25204cf1` |
| A4 | 19 of 71 features cannot move a matmul ranking (12 constant-in-pool, 7 affine duplicates) | `MATMUL_FEATURES`, a 53-feature spec. **Available, not adopted** — `emmy fit` still defaults to the full view | `38017fb9` |
| A5 | #465 retired `D_stage_ring`, collapsing matmul top-1 from 54/242 to 4 | `D_stage_prefetch` = `depth >= 2`, the step the flag was carrying | `c0af49e3` |

### B — Fix measured but not implemented

Evidence exists that each would help; no code change was made.

| # | issue | what was measured | why not done |
| --- | --- | --- | --- |
| B1 | `emmy tune` ranks at `-Xcicc -O1`, a proxy that mis-ranks the wide register-tile mma family | Arm D: `-O3` ranking wins **10/10 shapes**, median 0.924, for +19% wall; at patience 15 beats the shipped default on speed *and* quality | Changing the default needs validation on the kernel classes `-O1` exists to protect (big unrolled register-tile kernels). Today it is a flag: `--nvcc-flags ""` |
| B2 | The fitter's coordinate descent is a weak optimizer | A convex listwise fit beat it **on the fitter's own objective** (mean log2 4.01 vs 4.91); descent from zeros lands at 7.36; it cannot refine a good solution at all | Implemented only as a scratchpad probe |
| B3 | The linear model class leaves headroom | CatBoost on the same features/loss: held-out median **3 vs 9**, top-10 **169 vs 122**, small train→holdout gap | Scratchpad probe only; a shippable artifact needs the `catboost` trainer cell |
| B4 | The refit prior is better in deployed latency, not just rank | Cold picks **1.64× faster geomean** than the shipped prior across the 10 uncovered shapes (2.24× at M=32) | Measurement is sound but *uncontrolled* — the two columns come from different runs, and the shipped-prior side was not designed as a cold measurement. Needs one clean both-priors/fresh-DB run before quoting |
| B5 | A scoped matmul-only artifact | `offline_weights_matmul_rtx5090.json`, median golden rank 708 → 17 on its slice | Committed but **opt-in only** — nothing loads it unless `EMMY_OFFLINE_FILE` points at it, by design |

### C — Reported, not fixed

| # | issue | evidence | severity |
| --- | --- | --- | --- |
| C1 | **A 73× misdeploy in the evidence hierarchy.** `qk_proj.m32` picks a proper mma config cold (7.74 µs) but an empty knob map at **566.7 µs** once a DB and reservoir are populated | Deterministic: five observations, spread 0.09 µs, across loop positions and prior states; reproduced with the online prior disabled, so not search steering. Recurred in Arm D's `-O1` arm and not its `-O3` arm | **Highest.** Something in tiers 2–3 promotes a scalar tile over a warp-eligible mma config |
| C2 | **The offline prior's contribution to the search blend is inert.** `FallbackPrior.score` clamps the offline multiplier to `e**±8` | Over 261 goldens the exp-argument runs −18.6 / −16.6 / −2.7; **255 of 261 clamp to the identical constant**, contributing zero ranking signal. Same saturation class as the retired ±80 quality clip | High — a documented mechanism silently does nothing |
| C3 | **The search strategy itself.** With the proxy metric removed, the tuner still trails the hand sweep by ~11% | Arm D: `-O3` ranking closes only ~40% of the gap | Medium; this is where the remaining headroom is |
| C4 | **Golden coverage gaps on the 5090**: 11 fused warp contractions at gemma-4 serving widths, plus 15 non-contraction forks | Checked-in gap baseline; misdeploy cost on this class documented at 114–1014× | Medium-high — the fused ones are where the prior actually decides |
| C5 | **No fused golden has ever trained the prior.** The case builder handles matmul/reduce/pointwise only; `NormLinearGoldenConfig` + `MlpGeGluGoldenConfig` (80 entries on the 5090) are out of scope | `build_golden_groups` in `emmy/commands/fit.py` | Medium — the prior extrapolates on a structure it has never seen |
| C6 | **Train/deploy routing seam.** 22 case-buildable goldens (16 pointwise, 6 reduce) with a symbolic axis train under the *static* weight set but deploy under the *dynamic* one | `_base()` stamps `S_ext_n_symbolic_axis` only for dynamic matmul goldens, while `OfflinePrior` routes on that stamp | Medium — same class as the gate mismatch fixed in A2 |
| C7 | **Featurizer blind spot**: transposed warp fragment grids (`TILE=f2x4` vs `f4x2`) featurize identically even in the full vocabulary | 126 of 63,386 rows in a standard pool | Low — no golden in this slice is affected |
| C8 | **`ShapeKey` ignores operand layout**, so a canonical `torch.matmul` golden can join an `F.linear` fork | Observed on `matmul.square.2048` vs a Qwen3 o_proj; the documented aspect-blind shadow class | Low-medium |
| C9 | **Docs are wrong about the `-O3` re-bench window.** `config.py`'s docstring and the `tune-golden` skill imply a 15% default; the real constant is `O3_REBENCH_TOL = 2.0`, a 3× window | Read directly from `policy/mcts.py` | Low, but it misled this investigation |
| C10 | **Data-quality trap.** The tune log's `best: … us @ bench #N` sits under `[prior] global` — a statistic over the prior's whole stream, systematically optimistic | Both arms' extractors initially recorded it under a misleading field name | Low, but it silently corrupts any number sourced from it |

---

## Claims made during this work and later retracted

Recorded because each was believed on real evidence and overturned by better evidence.

| claim | correction |
| --- | --- |
| "#465 deleted a feature carrying real signal; a refit cannot recover it" | The feature was exactly `depth >= 2` — no information lost. The loss was a precomputed *step*, recoverable as a derived feature. |
| The retired flag was a *conjunction* (`depth>=2 AND async`) | It is a plain threshold. Both formulas matched only because `depth>=2` implies `async` in this candidate space. |
| "15 plain matmul shapes have no golden" | None are matmuls. 9 have no reduce axis at all (glue/pointwise); 6 are M=1 gemv. The `kind=""` field means "not a sweep kind", not "is a matmul" — the "plain matmul" label was invented, not read. |
| The prior causes a 64× misdeploy on `qk_proj.m32` | The prior picks a proper mma config cold. The misdeploy comes from the evidence hierarchy's measured tiers. |
| "Tuning made things worse than not tuning, on 8 of 10 shapes" | Artifact of comparing against a greedy measured *after* tuning. Against a cold baseline: better on 6, worse on 4. |
| The `-O3` re-bench window is 15% of the best `-O1` | `O3_REBENCH_TOL = 2.0` — the window is **3×**, very wide. The `0.15` in `config.py`'s docstring is an example of the env-override format, not the default. The narrow-window mechanism I proposed does not exist. |
| Qwen3-1.7B's kernels are 5 plain contractions | The serving-twins decomposition and the `--layer 0` trace differ. The real traced kernels fuse projections with norms, GeGLU and SDPA; q/k/v merge to one N=4096 kernel. |

## Recommended order of work

1. **Refit the shipped artifact (unblocks A1-A5).** Nothing in section A affects a real deploy until this runs. It
   still carries a weight on a feature that no longer exists. `emmy fit --artifact`, ideally `--folds both` so the
   cross-validation blocks compare against `_tune/fits/20260730-l2-refit`.
2. **Chase C1, the evidence-path misdeploy.** Largest measured cost (73×), affects every deploy on an untuned shape
   that has partial measurements, and is independent of everything else here.
3. **Fix or remove C2, the saturated offline multiplier.** A documented mechanism that silently does nothing.
4. **Decide B1** — make `-O3` ranking the tune default, after validating it on the kernel classes `-O1` protects.
   Measured as a strict win on these ten shapes, on both speed and quality.
5. **Run the controlled version of B4** before quoting the 1.64× cold-pick speedup anywhere.
6. **Then B2/B3** — a stronger optimizer or a model class with interactions. Both are measured wins; neither is the
   bottleneck today.

## Reproducing

Scoped artifact: `emmy/compiler/pipeline/search/prior/offline_weights_matmul_rtx5090.json`, selected with
`EMMY_OFFLINE_FILE` / `--offline-file`. It is fit on RTX 5090 matmul goldens alone and has no reason to beat the
shipped weights outside that slice; its `provenance.scope` says so.

Hardware runs used per-arm scratch (`EMMY_TUNE_DB`, `EMMY_ONLINE_FILE`, `EMMY_CUBIN_CACHE`) rather than `--clean`,
which would destroy the shared `node` table. Measurement rules that matter: compare pinned rows against each other
and never against a plain greedy row (~7% environment gap); a `pin_unmatched` or `bench_fail` row is not a
measurement; and replicate noise on these shapes measured ~1%, not the 10–13% that applies to the golden re-bench
path.
