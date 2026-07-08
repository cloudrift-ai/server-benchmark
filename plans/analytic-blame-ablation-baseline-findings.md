# Per-feature blame + ablation — incumbent AnalyticPrior attribution (4090 / 5090 / PRO 6000 Max-Q), 2026-07-07

Phase-2 of the analytic-prior rework (`plans/analytic-prior-catboost-rework.md`): the new per-feature regret
attribution views, run for the incumbent linear `AnalyticPrior` (cold — no learned checkpoint) over the same 3-card
node store as the Phase-1 baseline (`plans/analytic-prior-cold-baseline-findings.md`). Diagnostic, NOT gate metrics:
attribution among correlated features is non-unique (ablation double-counts redundancy), so nothing here thresholds.

**The numbers.** *Blame*: per missed fork, `term(picked) − term(best)` over the prior's exact per-term decomposition
(`explain_features`; linear terms + the three `gate:*` interaction pseudo-terms), weighted by the fork's excess cost
`regret − 1` and summed per fork family. Positive = the feature's term argued FOR the wrong child; negative = it
argued for the right one and was overruled. A missed fork where NO term separates pick from best is **BLIND** — the
prior structurally cannot distinguish the siblings. *Ablation Δ*: each family's median regret with one feature masked
before scoring, minus the full median; `< 0` = actively misleading, `> 0` = load-bearing. *Support* = forks where the
feature varies across siblings (only those can change their pick); `~` marks support < 5 (one-fork noise). Fork
records build per card, tables pool all three (regret is a within-fork ratio, so cards pool safely).

**Provenance.** Same store as Phase 1: 23,063 ok `feat_ver=2` nodes, 330 multi-child forks (4090 / 5090 /
PRO 6000 Max-Q, ε-greedy 0.25 sweeps of 2026-07-06/07). Pooled per-family full medians: REDUCE 34.12x (43/43
missed), TILE 2.29x (50/51), STAGE 1.00x (106/224), root PLACE+R+S+T 1.19x (8/8), +WSPEC 3.41x (4/4).
Reproduce: `emmy eval prior --dataset nodes --blame --ablate`.

## Blame — regret-weighted term diffs per fork family (top rows)

| family | forks | missed | regret-weight | top blame (+ pushed the wrong pick) | overruled right voices (−) | → |
|---|--:|--:|--:|---|---|---|
| REDUCE | 43 | 43 | **1670.33** | D_splitk_excess +2.27, D_near_waves +0.82 | — (**42/43 BLIND**) | #1 |
| TILE | 51 | 50 | 138.26 | D_near_cells +119.9, D_near_waves +105.9, D_l2_bm +77.6, D_log2_waves +48.5 | D_square −93.9, D_pow2_threads −58.5, D_ctas_ge_sm −52.7 | #3 |
| STAGE | 224 | 106 | 41.76 | D_stage_depth +24.2, D_stage_tma +2.0 | D_stage_async −127.5, D_stage_reg_depth −81.7, D_stage_ring −30.7 | #4 |
| +WSPEC (root) | 4 | 4 | 10.50 | D_stage_reg_depth +63.9, D_square +38.4 | D_stage_tma −166.9, D_stage_async −129.9 | #4 |
| PLACE+R+S+T (root) | 8 | 8 | 8.82 | D_ctas_ge_sm +55.8, D_near_cells +30.6 | D_near_waves −79.2, D_stage_reg_depth −75.5 | #3 |

## Ablation Δ — masked-feature change in median regret (rows with any |Δ| ≥ 0.01x)

| feature | support | TILE | STAGE | root | → |
|---|--:|--:|--:|--:|---|
| D_l2_reuse | 63 | **−0.56x** | · | +0.00x | #2 |
| D_tile_m | 63 | −0.04x | · | +0.00x | #2 |
| D_l2_bn / D_bn_band | 16 / 4~ | −0.04x | · | · | #2 |
| D_near_cells / D_near_intensity / D_log2_waves / D_l2_cells_occ / D_l2_bm / D_near_kchunks | 63…9 | −0.03x | · | ≤+0.05x | #2 |
| D_stage_depth | 219 | · | **+0.04x** | +0.00x | #4 |
| D_stage_async | 69 | · | +0.01x | +0.00x | #4 |
| D_cells / D_square / D_l2_threads / D_reuse / D_near_threads / D_tile_n / D_near_waves / D_near_area / D_log2_ctas | 63…57 | **+0.50x** each | · | ≤+0.06x | #3 |
| Δ = 0.00x everywhere (21 features) | — | incl. all D_splitk\*, D_stage_ring/tma/reg_depth, MMA_\*, D_pow2_threads | | | #5 |

## Findings

### 1 — blame independently confirms the REDUCE featurizer blindness, and sizes it: 89% of all misranking weight

42 of 43 missed REDUCE forks are BLIND (no term separates the prior's pick from the measured-best sibling), and the
family's regret-weight (1670) is ~8× every other family combined (~200). This is the Phase-1 finding-1 root cause
(`_reduce_decomp` only fires inside `_tile_features`) re-derived mechanically — what took a hand decomposition in the
baseline is now the blame table's degenerate case. The one non-blind REDUCE fork shows the incumbent's only reduce
signal: `D_splitk_excess` (+2.27) actively arguing for the wrong split. **Next:** the reduce featurizer fix
(baseline finding 1) stays the top-priority item; after it lands, re-run this view — the blind counter dropping to ~0
and blame redistributing onto the new `D_*` reduce features is the fix's acceptance check, and the blame table then
feeds the pending `_W_A` reduce refit directly.

### 2 — `D_l2_reuse` is actively misleading on TILE forks: masking it cuts median regret 2.29x → 1.73x

The single biggest actionable number in the ablation table: Δ −0.56x on 63-fork support. The smaller
band/L2 negatives (`D_tile_m`, `D_l2_bn`, `D_bn_band`, `D_near_cells`, `D_near_intensity` at −0.03..−0.04x) point the
same direction: the L2/band block as fit in the 2026-07-02 refit (matmul-golden-only objective, per the `_W_A`
comment) mis-generalizes to the node store's fork population. **Next:** feed this list to the pending analytic refit
(`scripts/golden_knob_heuristics.py`) as sign/zero candidates, and carry `D_l2_reuse` into the Phase-4 fitter's
monotone-constraint discussion — a constrained direction (or exclusion) is the durable home for "this feature must
not dominate tile picks".

### 3 — TILE misses are an occupancy-band tug-of-war between correlated features, not one bad weight

Blame: `D_near_cells` / `D_near_waves` / `D_l2_bm` push wrong picks (+120/+106/+78) while `D_square` /
`D_pow2_threads` / `D_ctas_ge_sm` argue right and get overruled. Ablation agrees but exposes the redundancy the plan
warned about: masking ANY of nine tile-geometry features (`D_cells`, `D_square`, `D_l2_threads`, `D_reuse`, …) costs
the same +0.50x — they encode one underlying geometry axis, so single-feature ablation double-counts and none of
these Δs are additive. This is the linear model's band-threshold compromise (the plan's "why" section) showing up as
data. **Next:** treat the tile-geometry block as a group in the Phase-4 fitter (grouped masking augmentation, and a
tree model's interactions replace the hand-tuned band thresholds); don't chase individual sign flips here.

### 4 — STAGE: the depth weight is a compromise that loses cheaply but often

106/224 STAGE forks miss, but the family's total regret-weight is only 41.8 (median regret 1.00x — misses cost a few
percent). Blame says `D_stage_depth` (+24.2) argues for wrong depths while the transport terms (`D_stage_async`,
`D_stage_reg_depth`, `D_stage_ring`) vote right and get overruled; net, ablation still prices `D_stage_depth`
load-bearing (+0.04x masked). Same story at the root +WSPEC forks (D_stage_reg_depth +63.9 vs D_stage_tma −166.9).
**Next:** nothing urgent (STAGE is near-optimal and the baseline gate says "do not regress") — but the depth-vs-
transport tension is a ready-made pair-term / interaction candidate for the Phase-4 fitter notes.

### 5 — a fifth of the weighted features are untestable on this store: zero fork support

21 features never vary across any fork's siblings — including every `D_splitk*` (support ≤ 1 fork), the
`D_stage_ring/tma/reg_depth` trio at the STAGE family level, `MMA_*`, and (from the gate features) the
scalar-on-warp driver that the qwen3-emb deploys proved matters. Ablation can say nothing about them; their weights
ride on golden-only evidence. **Next:** fold "does the sweep exercise this feature's fork?" into the collection-sweep
design (the `collect-node-data` ε-greedy pass currently never generates sibling variation for these decisions —
e.g. split-K forks collapse before enumeration on most shapes); revisit after the reduce featurizer fix, which
re-homes several of these.

## Artifacts / workflow notes

- Reproduce: `emmy eval prior --dataset nodes --blame --ablate` (append `--kernel <SUBSTR>` to scope); full console
  output archived in the session scratchpad only. Views: `diagnostics.attribution_report` over the shared
  `fork_records` (the same records the regret gate metric projects — pick semantics identical by construction).
- Exactness: the incumbent is the linear prior, so blame is exact (unit-tested: terms sum to the scored quality,
  gates included) and masking is exact term removal. For a learned CatBoost prior the report auto-prints an
  out-of-distribution caveat until the Phase-4 dropout-trained fitter lands.
- Wall time: the full --blame --ablate run over 23k nodes / 330 forks is ~1 min on the laptop (dominated by
  featurizing each parented node once per view — node_report and attribution_report each build their own records; a
  shared cache is a possible later cleanup, not worth it at this size).
- The ablation table's family columns get very wide with the root-fork family names
  (`PLACE+REDUCE+STAGE+TILE+WSPEC`); acceptable in a terminal, abbreviated by hand in this report's tables.
