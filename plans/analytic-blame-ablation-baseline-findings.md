# Per-feature blame + ablation — incumbent AnalyticPrior, before/after the reduce featurizer fix (4090 / 5090 / PRO 6000 Max-Q, 2026-07-07 → 08)

**Background, in one paragraph.** When emmy compiles an op to a CUDA kernel, an autotune search walks a tree of
scheduling decisions ("knobs": tile size, staging depth, split-K, …). Each branching point in that tree is a **fork**;
its children are the candidate configs for one decision. A model called the **prior** scores candidates so the search
(and a no-benchmark "cold" deploy) can pick fast configs without measuring everything. This report evaluates the
**analytic prior** — the hand-fit linear scoring model used when no learned data exists — against the **node store**,
a database of past search trees where every fork child carries a measured best-reachable latency. The two views here
answer "*why* does the prior mispick": **blame** decomposes each wrong pick into per-feature contributions, and
**ablation** re-runs every pick with one feature hidden to see if picks improve. This is Phase 2 of the analytic-prior
rework (`plans/analytic-prior-catboost-rework.md`); the run is repeated before and after PR #322, which fixed a bug
where reduction-related knobs were invisible to the feature encoding. The re-run doubles as **#322's acceptance
check**, defined in advance by the baseline: the "blind fork" counter must collapse and blame must move onto the new
reduction features. It did. These are diagnostic views, NOT promotion-gate metrics: when features are correlated,
splitting credit between them is inherently ambiguous, so nothing here should ever be thresholded on.

**How to read the numbers.**
*Regret* — for one fork, `latency(child the prior picks) / latency(best child)`; 1.00x means the prior steers into
the best subtree, 2.00x means following the prior costs 2× at that fork. Forks are grouped by the kind of decision
they make (their knob **family**: TILE = tile size, STAGE = data-staging pipeline, REDUCE = how the reduction axis is
split/accumulated, …).
*Blame* — per wrong pick, `term(picked) − term(best)` for every term of the prior's score (the linear weights plus
its few hardcoded interaction terms, shown as `gate:*`). Positive = this feature argued FOR the wrong child; negative
= it argued for the right child and was outvoted. Each fork's contribution is weighted by how much the miss costs
(`regret − 1`), then summed per family. For a linear model this decomposition is exact (terms sum to the score).
*BLIND* — a wrong pick where picked and best have identical values on every scored feature: the model could not have
told them apart no matter its weights. That's a feature-encoding gap, not a weight problem.
*Ablation Δ* — median regret of a family with one feature hidden from the model, minus with it: `< 0` means the
model picks BETTER without the feature (it is actively misleading); `> 0` means it is load-bearing. *Support* = the
number of forks where the feature actually differs between siblings — only those picks can change, so a Δ on tiny
support is noise.
Fork records are built per GPU; the tables pool all three cards (regret is a per-fork ratio, so it compares safely
across hardware).

**Data.** Before: 23,063 usable node rows / 330 multi-child forks (the Phase-1 collection sweeps,
`plans/analytic-prior-cold-baseline-findings.md`). After: 36,558 rows / 1,536 forks — the same store plus freshly
collected tuning runs from `_tune/` (a 4090 sweep, a 5090 sweep, and a 5090 run made specifically to validate #322),
merged with `scripts/merge_node_db.py`. The PRO 6000 got **no new data**: the store keeps raw knob values and encodes
features at read time, so #322 upgrades its old rows in place. Reproduce with
`emmy eval prior --dataset nodes --blame --ablate`.

## Before → after, per fork family (pooled; regret medians, blame regret-weight, blind count)

| family (decision kind) | forks | median regret | missed | regret-weight | BLIND | → |
|---|---|---|---|---|---|---|
| REDUCE (reduction split) | 43 → 280 | **34.12x → 1.09x** | 43 → 208 | **1670 → 84** | **42 → 0** | #1 |
| TILE (tile size) | 51 → 51 | 2.29x → 2.36x | 50 → 47 | 138 → **681** | 3 → 4 | #2 |
| STAGE (staging pipeline) | 224 → 1185 | 1.00x → 1.02x | 106 → 688 | 42 → 351 | 0 → 0 | #4 |
| PLACE+R+S+T (several at once, root) | 8 → 8 | 1.19x → 1.53x | 8 → 8 | 9 → 13 | 0 → 0 | #2 |
| …+WSPEC (root, sm_120 cards only) | 4 → 4 | 3.41x → 3.41x | 4 → 4 | 10.5 → 10.5 | 0 → 0 | #4 |
| FAST_EXP+…+VECTORIZE_LOADS (new pointwise forks) | — → 8 | — → 1.09x | — → 6 | — → 1.0 | — → 0 | |

Per-card REDUCE medians: 4090 **68.50x → 1.09x**, 5090 **30.08x → 1.13x**, PRO 6000 **34.13x → 1.00x** (the last
purely from re-encoding old rows — see Data). The leaf-level check agrees: e.g. the 5090's "prior's pick vs measured
best config" mean fell 11.97x → 1.30x.

## Blame after the fix (top rows per family)

| family | + pushed the wrong pick | − argued right, overruled | → |
|---|---|---|---|
| REDUCE | D_near_threads +313, D_l2_threads +255, D_ctas_ge_sm +31, **gate:splitk_roundtrip +26** | D_near_waves −66, D_finalize_kernel −24 | #1 #3 |
| TILE | D_near_waves +2746, D_square +2608, D_cells_cap +1800, D_cells +1349, D_log2_area +1250 | D_ctas_ge_sm −3964, D_near_tilen −1384 | #2 |
| STAGE | D_stage_reg_depth +3475, D_stage_depth +615 | D_stage_async −4778, D_stage_tma −3509, D_stage_ring −1682 | #4 |

(Feature names: `D_*` are the prior's engineered inputs — thread-count targets (`D_*_threads`), GPU-occupancy
closeness (`D_near_waves`, `D_ctas_ge_sm`), tile shape/area (`D_square`, `D_cells*`, `D_log2_area`), staging depth
and transport kind (`D_stage_*`). `gate:*` are the hardcoded interaction terms in `AnalyticPrior`.)

## Ablation Δ after the fix (rows worth acting on)

| feature | support | Δ (family) | reading | → |
|---|--:|---|---|---|
| D_l2_bm | 17 | **−0.55x** (TILE) | actively misleading — the new worst offender | #3 |
| D_tile_m | 71 | −0.07x (TILE) | mildly misleading | #3 |
| D_splitk_roundtrip | 57 | −0.06x (REDUCE) | the newly-activated interaction term leans wrong on reduce forks | #3 |
| D_l2_reuse | 63 | **+0.05x** (TILE) | was −0.56x — the 2026-07-07 weight refit fixed it (v1 finding 2's Next, done) | #3 |
| D_log2_area / D_near_waves / D_square / D_near_area / D_cells(_cap) / D_pow2_threads | 57–71 | +1.7..+2.5x (TILE) | the load-bearing (and mutually redundant) tile-geometry block | #2 |
| D_splitk / D_splitk_deficit | 211 / 221 | ≈0, real support | newly exercised by #322 — finally trainable | #5 |

## Findings

### 1 — the #322 acceptance check PASSED: the model can now see reduction decisions; the rest is a cheap weight problem

Before the fix, the feature encoder returned **identical vectors for all children of a reduction fork** — the split
and accumulation knobs never reached the features, so no model, however good, could rank those children. That showed
up as 42 of 43 wrong REDUCE picks being blind, at 30–68× cost. After the fix: blind 42 → 0, total miss cost
(regret-weight) 1670 → 84, per-card medians down to 1.00–1.13x — on 6.5× more reduction forks. The PRO 6000 isolates
the mechanism: it received zero new measurements, yet its REDUCE median fell from 34.13x to 1.00x just from
re-encoding stored knob values — confirming the rework plan's resilience bet that encoding fixes must never require a
data-collection campaign. The remaining 208 misses are genuine weight errors, and cheap ones (median 1.09x — cents,
not the old 30–90×): blame pins them on the two thread-count features (`D_near_threads` +313, `D_l2_threads` +255)
preferring the wrong cooperative-reduction widths. **Next:** nothing blocking — hand the thread-target terms to the
Phase-4 model fitter (or an earlier linear-weight refit, now that reduce forks produce training signal); re-run this
view afterwards.

### 2 — the worst mispredictions moved to tile-size picks on the 4090's new large-K matmul shapes

TILE's total miss cost jumped 138 → 681 with the same 51 forks, concentrated on matmul shapes recently added to the
golden set (the h4096 model-projection shapes, which have a very large reduction dimension K): on the 4090,
`free=4096 red=14336` prices at **215x**, `free=12288 red=4096` at 45x, `free=28672 red=4096` at 20x — while the 5090
prices the same shapes at 1.4–5.5x. Blame shows a tug-of-war: the tile-shape/occupancy features (`D_near_waves`
+2746, `D_square` +2608, `D_cells_cap` +1800) argue for the wrong tile, while `D_ctas_ge_sm` (−3964; "are there
enough thread blocks to fill the GPU") votes correctly and is outvoted. Ablation says the same block is strongly
load-bearing everywhere else (masking any one of it costs +1.7..+2.5x), so no single weight can be deleted — one set
of linear weights cannot price small-K and large-K tile geometry at once. That structural ceiling is the core reason
for this rework's move to a nonlinear model, now demonstrated on exactly the shapes the fleet cares about.
**Next:** make these three shapes a priority held-out test for the Phase-4 CatBoost fitter (train without the 4090,
verify it prices them); short-term, an A/B against the recorded golden configs will tell whether a cold deploy really
lands in the 215x subtree or the search's patience saves it in practice.

### 3 — ablation as a refit scorecard: one feature fixed, two new suspects

The previous run's worst offender, `D_l2_reuse` (masking it IMPROVED tile picks by 0.56x median), now reads a healthy
+0.05x — the 2026-07-07 weight refit on main fixed it, closing v1 finding 2. The new suspects: `D_l2_bm` (−0.55x on
TILE, 17-fork support), `D_tile_m` (−0.07x), and the `splitk_roundtrip` interaction term (−0.06x on REDUCE, +26
blame). The last one independently confirms a hesitation already written into `analytic.py` when that term was first
activated ("measured ~neutral on the golden gate; the split-K goldens sink within their pools") — the node store now
says the same thing with more data. **Next:** carry `D_l2_bm` and the roundtrip term into the next refit's
sign/zero-candidate list and Phase-4's monotone-constraint discussion; the term's weight is a constructor parameter,
so A/B-ing it at zero is a one-line experiment.

### 4 — staging-depth picks: wrong often, but each miss costs a few percent

STAGE — the choice of how deeply to pipeline operand staging and via which transport (synchronous copy, cp.async,
TMA) — now has 1,185 forks (5.3× more data) and stays at 1.02x median: the prior mispicks 688 of them, but each miss
costs a few percent, not multiples. The blame pattern from v1 holds at scale: the depth features (`D_stage_reg_depth`
+3475, `D_stage_depth` +615) argue for wrong depths while the transport features (async −4778, TMA −3509, ring −1682)
vote right and get outvoted; ablation still nets every stage feature load-bearing (+0.01..+0.12x). The 4-fork
`+WSPEC` root family is unchanged (no new data from the sm_120 cards this round). **Next:** unchanged — the
depth-vs-transport tension is a ready-made interaction case for the Phase-4 fitter; whatever changes, do not regress
the 1.0x median.

### 5 — coverage bookkeeping (and a correction to v1): 8 features are untestable, not 21

Correction: v1's finding 5 conflated two different lists. The "Δ = 0.00x everywhere" list (21 features then, 16 now)
is features that DO vary at some forks but never change a pick when masked — mostly ones carrying no weight (the
`MMA_a_bits` class) or never-decisive ones. The genuinely **untestable** set — features that never differ between any
fork's siblings, so ablation can say nothing about them — was and remains 8: `D_wspec_warps`,
`D_scalar_on_warp_eligible` (the input of the prior's largest hardcoded penalty, weight 40), `D_neg_masked_{m,n,k}`,
`D_bk_ge32`, `D_w_l2_bk`, `D_l2_bk`. Their weights ride on golden-set evidence alone. Meanwhile #322 also fixed
coverage: the whole split-K feature family went from ≤1-fork support to 211–221 forks — the encoding fix doubles as a
training-data fix. **Next:** the "design collection sweeps so these decisions actually fork" item stands, but only
for the 8; the warp-specialization features additionally need sweeps on the sm_120 cards that exercise that fork.

## Artifacts / workflow notes

- Store after merge: `~/.cache/emmy/autotune.db` — 36,558 usable rows, 1,536 multi-child forks (4090 13,233 / 5090
  15,597 / PRO 6000 7,766). Sources merged from `_tune/{golden-sweep-rtx4090-refit, golden-sweep-rtx5090-refit,
  reduce-featurizer-gate-5090}` via `scripts/merge_node_db.py --src <snapshot>`; the pre-merge local DB is backed up
  in the session scratchpad. The reduce-gate DB is a WAL snapshot that only opens with SQLite's `immutable=1` —
  copying it locally first is the easy path the script could adopt.
- Reproduce: `emmy eval prior --dataset nodes --blame --ablate` (cold prior; blame is exact for the linear model,
  unit-tested to sum to the scored quality, interaction terms included). Full console outputs of both runs are
  archived in the session scratchpad.
- The before/after comparison is a diff of two report runs — by design there is no separate comparison tool.
- Wall time: ~2.5 min for the full `--blame --ablate` pass over 36.5k rows / 1.5k forks — the per-view re-encoding
  noted in v1 is still fine at this size.
