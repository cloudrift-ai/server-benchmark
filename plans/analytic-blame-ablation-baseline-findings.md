# Per-feature blame + ablation — incumbent AnalyticPrior, before/after the reduce featurizer fix (4090 / 5090 / PRO 6000 Max-Q, 2026-07-07 → 08)

Phase-2 of the analytic-prior rework (`plans/analytic-prior-catboost-rework.md`): the per-feature regret attribution
views, run for the cold linear `AnalyticPrior` twice — the 2026-07-07 incumbent baseline (pre-fix code + the original
3-card sweep store), and the 2026-07-08 re-run after merging main's #320–#325 (the #322 TILE-less REDUCE
featurization, the `_W_A`/`_W_A_DYN` refit, the h4096 golden seeds) with the fresh refit-sweep data merged in. The
re-run doubles as **#322's acceptance check**, which the baseline defined in advance: the REDUCE blind counter must
collapse and blame must redistribute onto the new reduce features. It did. Diagnostic views, NOT gate metrics:
attribution among correlated features is non-unique, so nothing here thresholds.

**The numbers.** *Blame*: per missed fork, `term(picked) − term(best)` over the prior's exact per-term decomposition
(`explain_features`; linear terms + the `gate:*` interaction pseudo-terms), weighted by the fork's excess cost
`regret − 1`, summed per fork family. Positive = the feature argued FOR the wrong child; negative = it argued right
and was overruled. **BLIND** = a missed fork where no term separates pick from best (featurizer gap, not a weight
problem). *Ablation Δ*: each family's median regret with one feature masked, minus the full median; `< 0` = actively
misleading. *Support* = forks where the feature varies across siblings. Fork records build per card; tables pool all
three (regret is a within-fork ratio).

**Provenance.** Before: 23,063 ok nodes / 330 forks (the Phase-1 sweeps, `plans/analytic-prior-cold-baseline-findings.md`).
After: 36,558 ok nodes / 1,536 forks — the same store plus the merged `_tune` refit sweeps (4090 golden-sweep refit,
5090 golden-sweep refit, 5090 reduce-featurizer-gate run; `scripts/merge_node_db.py`, all `feat_ver=2`). The PRO 6000
got **no new data** — its rows simply re-featurize under #322 (raw-knob storage means the fix needs no recollection).
Reproduce: `emmy eval prior --dataset nodes --blame --ablate`.

## Before → after, per fork family (pooled; regret medians, blame regret-weight, blind count)

| family | forks | median regret | missed | regret-weight | BLIND | → |
|---|---|---|---|---|---|---|
| REDUCE | 43 → 280 | **34.12x → 1.09x** | 43 → 208 | **1670 → 84** | **42 → 0** | #1 |
| TILE | 51 → 51 | 2.29x → 2.36x | 50 → 47 | 138 → **681** | 3 → 4 | #2 |
| STAGE | 224 → 1185 | 1.00x → 1.02x | 106 → 688 | 42 → 351 | 0 → 0 | #4 |
| PLACE+R+S+T (root) | 8 → 8 | 1.19x → 1.53x | 8 → 8 | 9 → 13 | 0 → 0 | #2 |
| …+WSPEC (root, sm_120) | 4 → 4 | 3.41x → 3.41x | 4 → 4 | 10.5 → 10.5 | 0 → 0 | #4 |
| FAST_EXP+…+VECTORIZE_LOADS (new) | — → 8 | — → 1.09x | — → 6 | — → 1.0 | — → 0 | |

Per-card REDUCE medians: 4090 **68.50x → 1.09x**, 5090 **30.08x → 1.13x**, PRO 6000 **34.13x → 1.00x** (the last on
re-featurized old rows alone — see provenance). Per-card leaf reachability collapsed with it: 5090 mean 11.97x →
1.30x, PRO 6000 12.89x → 1.27x.

## Blame after the fix (top rows per family)

| family | + pushed the wrong pick | − argued right, overruled | → |
|---|---|---|---|
| REDUCE | D_near_threads +313, D_l2_threads +255, D_ctas_ge_sm +31, **gate:splitk_roundtrip +26** | D_near_waves −66, D_finalize_kernel −24 | #1 #3 |
| TILE | D_near_waves +2746, D_square +2608, D_cells_cap +1800, D_cells +1349, D_log2_area +1250 | D_ctas_ge_sm −3964, D_near_tilen −1384 | #2 |
| STAGE | D_stage_reg_depth +3475, D_stage_depth +615 | D_stage_async −4778, D_stage_tma −3509, D_stage_ring −1682 | #4 |

## Ablation Δ after the fix (rows worth acting on)

| feature | support | Δ (family) | reading | → |
|---|--:|---|---|---|
| D_l2_bm | 17 | **−0.55x** (TILE) | actively misleading — the new worst offender | #3 |
| D_tile_m | 71 | −0.07x (TILE) | mildly misleading | #3 |
| D_splitk_roundtrip | 57 | −0.06x (REDUCE) | the newly-activated gate leans wrong on reduce forks | #3 |
| D_l2_reuse | 63 | **+0.05x** (TILE) | was −0.56x — the refit rehabilitated it (v1 finding 2's Next, done) | #3 |
| D_log2_area / D_near_waves / D_square / D_near_area / D_cells(_cap) / D_pow2_threads | 57–71 | +1.7..+2.5x (TILE) | the load-bearing (and redundant) tile-geometry block | #2 |
| D_splitk / D_splitk_deficit | 211 / 221 | ≈0, real support | newly exercised by #322 — trainable at last | #5 |

## Findings

### 1 — #322 acceptance check PASSED: REDUCE blindness is gone; what remains is a (cheap) weight problem

Blind forks 42 → 0, REDUCE regret-weight 1670 → 84, per-card medians 34–68x → 1.00–1.13x, on 6.5× more REDUCE forks.
The PRO 6000 proves the mechanism: zero new rows, yet its REDUCE median fell 34.13x → 1.00x purely from re-featurizing
stored raw knobs — exactly the "additive encoding change, no version bump, no data campaign" property the rework plan's
resilience decisions demand. Blame now lands on real features: the remaining 208 misses (median 1.09x — cents, not
the old 30–90x) are driven by the thread-count pair `D_near_threads` +313 / `D_l2_threads` +255 arguing for the wrong
coop-fold widths. **Next:** nothing blocking — the thread-target terms are a ready-made input for the Phase-4 fitter
(and for a quick `_W_A` reduce-pool refit if one happens sooner); re-run this view after either.

### 2 — the worst class moved to TILE: the newly-seeded big-K 4090 shapes are catastrophically mispriced

TILE regret-weight 138 → 681 with the same 51 forks. The damage is concentrated on the 4090's newly-seeded golden
shapes: `free=4096 red=14336` TILE regret **215x**, `free=12288 red=4096` 45x, `free=28672 red=4096` 20x (5090 prices
the same shapes at 1.4–5.5x). Blame: the occupancy/area block (`D_near_waves` +2746, `D_square` +2608, `D_cells_cap`
+1800) argues for the wrong tile while `D_ctas_ge_sm` (−3964) votes correctly and is overruled — on big-K shapes the
2026-07-07 refit's band compromise breaks down for the cp.async-tier card. Ablation confirms the same block is
load-bearing elsewhere (+1.7..+2.5x masked), so this is not a delete-a-weight fix; it is the linear model's structural
ceiling showing at exactly the shapes the fleet now cares about (h4096 projections). **Next:** treat as a priority
test case for the Phase-4 CatBoost fitter (leave-one-card-out on the 4090); short-term, a tune-golden A/B on those
three shapes will tell whether cold-deploy actually hits the 215x subtree or patience saves it.

### 3 — refit scorecard via ablation: `D_l2_reuse` fixed; `D_l2_bm` and the splitk-roundtrip gate are the new shortlist

The v1 headline offender `D_l2_reuse` (−0.56x) now reads +0.05x — the 2026-07-07 refit resolved it, closing v1
finding 2. New shortlist: `D_l2_bm` −0.55x (TILE, 17-fork support), `D_tile_m` −0.07x, and `gate:splitk_roundtrip`
−0.06x on REDUCE with +26 blame — independent, node-store confirmation of main's own hesitation in `analytic.py`
("activation measured ~neutral on the golden gate; the g<n>k goldens sink within their pools"). **Next:** carry
`D_l2_bm` and the roundtrip gate into the next refit's sign/zero candidates and the Phase-4 monotone-constraint list;
the gate's weight is a constructor param, so a zero-default A/B is one-line.

### 4 — STAGE at scale: same depth-vs-transport compromise, still cheap, now statistically solid

1,185 STAGE forks (5.3× more) keep the family at 1.02x median. The pattern from v1 holds with much larger numbers:
`D_stage_reg_depth` +3475 / `D_stage_depth` +615 argue for wrong depths, the transport terms (async −4778, tma −3509,
ring −1682) vote right and get overruled; ablation nets every stage feature load-bearing (+0.01..+0.12x). The +WSPEC
root forks are unchanged (no new sm_120 data). **Next:** unchanged — a pair-term / interaction candidate for the
Phase-4 fitter; do not regress the 1.0x median.

### 5 — support bookkeeping (and a v1 correction): the untestable set is 8 features, not 21

v1's finding 5 overstated: the "Δ = 0.00x everywhere" list (21 → now 16 features) is features that VARY but never flip
a pick — mostly unweighted (`MMA_a_bits`-class) or never-decisive ones — not zero-support features. The truly
untestable set (never varying across any fork's siblings) was and remains the design-scan's 8: `D_wspec_warps`,
`D_scalar_on_warp_eligible` (the ±40 gate driver), `D_neg_masked_{m,n,k}`, `D_bk_ge32`, `D_w_l2_bk`, `D_l2_bk`.
Meanwhile #322 moved the whole `D_splitk*` family from ≤1-fork support to 211–221 forks — the reduce fix is also a
data-coverage fix. **Next:** the collection-sweep design item stands, but only for the 8; WSPEC additionally needs
sm_120 sweeps that actually fork it.

## Artifacts / workflow notes

- Store after merge: `~/.cache/emmy/autotune.db` — 36,558 ok nodes, 1,536 multi-child forks (4090 13,233 / 5090
  15,597 / PRO 6000 7,766 rows). Sources merged from `_tune/{golden-sweep-rtx4090-refit, golden-sweep-rtx5090-refit,
  reduce-featurizer-gate-5090}` via `scripts/merge_node_db.py --src <snapshot>`; pre-merge local DB backed up to the
  session scratchpad. The reduce-gate DB needed `immutable=1` (WAL snapshot) — copying it locally first is the easy
  path the script could adopt.
- Reproduce: `emmy eval prior --dataset nodes --blame --ablate` (cold `FallbackPrior` → `AnalyticPrior`; blame is
  exact, unit-tested to sum to the scored quality, gates included). Full console outputs (both runs) archived in the
  session scratchpad.
- The before/after comparison is two report diffs, exactly as the rework plan intended (no comparison runner).
- Wall time: ~2.5 min for the full `--blame --ablate` pass over 36.5k nodes / 1.5k forks — the per-view re-featurize
  noted in v1 is still fine at this size.
