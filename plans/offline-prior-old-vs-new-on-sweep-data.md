# Offline prior A/B: pre-refit (#363, fitted 07-09) vs promoted refit (#364, fitted 07-14) on the sweep data

- **Date / method**: 2026-07-15, GPU-free, local. Pure weights A/B: both artifacts evaluated through the SAME
  current code — the desaturated squash, tie-pessimistic golden rank, card-faithful contexts, and the
  golden-anchored descent section (branch `feature/golden-anchored-regret` @ `cd353f87`). Historical numbers for
  the old weights ("27/28 top-1", rank 0) came from the saturated/tie-optimistic stack and are NOT comparable;
  this report is the first honest old-vs-new measurement.
- **OLD** = `offline_weights.json` at `b75631af` (#363, the last pre-refit main), swapped in via
  `--offline-file` / `EMMY_OFFLINE_FILE`. **NEW** = the checked-in artifact on this branch (#364 refit). Both
  `feat_ver=2`, kind `linear`.
- **Data**: (a) golden rank over the live enumeration against the CURRENT golden YAMLs (incl. this sweep's 4
  replaces + 1 add — 122 scored rows); (b) fork-sibling regret / descent / reachability / -O3 endpoints over the
  07-14→15 RTX 4090 sweep store (16,700 nodes, scratch copy, read-only) and the pre-campaign local store
  (8,505 current-vocabulary 4090 rows) — node data NEITHER artifact trained on. Online checkpoint pinned to the
  sweep's `online.json` in every nodes run (constant across the A/B; only the offline half is compared).

## 1. Golden rank (the fit objective) — decisive NEW win outside the documented trade zone

| group | OLD median | NEW median | OLD top-10 | NEW top-10 | OLD top-100 | NEW top-100 |
| --- | --: | --: | --: | --: | --: | --: |
| gemma4_12b.* (41 rows) | 1422 | **90** | 3/41 | **11/41** | 3/41 | **24/41** |
| fp16 squares + sq512.dynM (25) | **120** | 238 | 1/25 | 2/25 | **11/25** | 10/25 |
| everything else (56) | 476 | **37** | 11/56 | **19/56** | 16/56 | **34/56** |
| ALL (122) | 461 | **79** | 15/122 | **32/122** | 30/122 | **68/122** |

Biggest improvements are exactly the gemma / K-heavy dynM family (kv_proj.dynM 8306→98, mlp_down.dynM 4430→6,
o_proj.dynM 4286→6). Worst regressions are exactly the promotion-time documented trade (square.512.dynM 52→6610
and 46→2110, square.512/1024.fp16 ~2–3x worse) plus qkv.h4096 (6444→9844, already deep for both).

## 2. Search-steering metrics on measured tree data — a wash, mixed directions within noise

Offline-half numbers; the store is held fixed within each comparison.

| metric (sweep store) | OLD | NEW |
| --- | --- | --- |
| TILE fork regret (median) | **1.88x** | 1.95x |
| PLACE+REDUCE+STAGE+TILE combined forks | 1.97x | **1.79x** |
| REDUCE / STAGE / RASTER | 1.02x / 1.00x / 1.05x | identical |
| leaf reachability mean (76 ops, -O1) | 1.41x | **1.30x** |
| leaf reachability worst | **3.46x** | 4.19x |
| golden-anchored descent kept | **8/30** | 7/30 |
| -O3 pick/golden endpoints mean (31) | 1.68x | **1.58x** |
| -O3 endpoints worst | **3.31x** | 4.92x (an fp16 square) |

Pre-campaign store: same picture (e.g. 4090 block TILE 1.82x OLD vs 1.98x NEW, reachability mean 1.32x OLD vs
1.19x NEW; the second per-card block flips a few cells the other way). No family moves beyond ~0.2x in either
direction on either store.

## Findings

1. **The refit is a large, real win on its objective and on the shapes that matter operationally** (gemma /
   K-heavy dynM): golden-rank median 461→79 overall, gemma 1422→90, with the improvements concentrated exactly
   where cold deploys were catastrophic pre-refit.
2. **The fp16-square regression is confirmed at the promotion-time magnitude and stays confined to that group**
   (median 120→238; sq512.dynM 52→6610 is the worst row). It also shows up as the NEW worst-case cells in the
   tree metrics (-O3 endpoint 4.92x, reachability worst 4.19x — both fp16 squares). This is the known
   global-linear-model trade; these rows remain the CatBoost rework's acceptance test.
3. **Fork-level steering barely moved** — regret/descent/reachability differences are small and mixed on both
   stores. The coordinate-descent refit reshaped the flattened-pool ranking (what greedy deploys cold) far more
   than the within-fork sibling ordering (what PUCT steers by). Consequence: tune efficiency should be roughly
   unchanged by the refit; the sweep's high benches-to-best on big matmuls is a property of both weight sets.
4. **Neither prior's weakness is deploy-critical anymore in the current stack**: warm deploys are owned by the
   online prior (TILE regret 1.34x, calibration +0.75 on this store), and cold deploys of seeded shapes are now
   decided by the golden evidence tier ahead of any model. The offline prior's remaining live surface is cold
   UNSEEDED shapes and cold-PUCT steering — where finding 1 says NEW is much better on golden-adjacent families
   and finding 3 says steering is unchanged. Keeping the refit is the right call; the fp16-square rows are the
   priority examples for the nonlinear rework.

## Repro

Outputs and scratch copies under the session scratchpad (`prior-ab/`): `{old,new}_golden.txt`,
`{old,new}_nodes_{sweep,pre}.txt`, `run.sh`. OLD artifact: `git show
b75631af:emmy/compiler/pipeline/search/prior/offline_weights.json`. Every eval:
`emmy eval offline [--offline-file OLD]` and `EMMY_TUNE_DB=<store> EMMY_ONLINE_FILE=<sweep online.json>
[EMMY_OFFLINE_FILE=OLD] emmy eval online --dataset nodes`.
