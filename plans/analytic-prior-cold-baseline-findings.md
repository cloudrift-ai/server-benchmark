# Analytic-prior cold baseline — golden rank + fork sibling regret (4090 / 5090 / PRO 6000 Max-Q), 2026-07-07

Phase-1 baseline for the analytic-prior rework (`plans/analytic-prior-catboost-rework.md`): the incumbent linear
`AnalyticPrior`, scored **cold** (no learned checkpoint — exactly the state a fresh clone / new card deploys from), on
the two gate metrics. These numbers are the bar the CatBoost prior (Phase 3+) must beat.

**The numbers.** *Golden rank*: the golden config's 0-based rank under the prior over its shape's full candidate
enumeration (ties count against — greedy breaks ties by emission order; rank 0 = a cold greedy deploy picks the
golden). *Fork sibling regret*: per multi-child fork of the node store, `value_us(predicted-best child) /
value_us(true-best child)` — 1.00x = the prior steers the search into the best-reachable subtree; ties price
pessimistically. Forks bucket by the knob FAMILY the fork decides; the per-family `ALL (median)` line is the gate
number. *Leaf reachability*: the prior's argmin over an op's fully-measured configs vs the measured best (an upper
bound on greedy quality). Regret exists only in the -O1 blocks: -O3 rows are parentless regime re-benches by design.

**Data provenance.** Three fresh ε-greedy sweeps (`collect-node-data`, `--explore-eps 0.25`, 39/39 golden shapes
each, 2026-07-06/07): RTX 4090 (8,030 rows), RTX 5090 (7,293), RTX PRO 6000 Blackwell **Max-Q** (7,766 — the plain
Workstation Edition was capacity-exhausted on CloudRift; the Max-Q clocks differently and is keyed as its own card).
All rows `feat_ver=2`; ~26 `bench_fail` pins (compile/bench-budget guards, mostly `attention.hd128.dynM`) recorded
and correctly excluded from metrics. The local DB was rebuilt from scratch — the pre-existing store was 100%
retired-vocabulary (pre-tile-IR-rebuild spellings) and was deleted rather than migrated.

## Golden rank (metric 1) — 42 golden entries, matmul-only (see finding 5)

`emmy eval analytic`; card attribution by golden-file block order (4080 and 4090 goldens are ranked from their
recorded YAML even though only 4090/5090/PRO 6000 have node data). Summary:

**median rank = 263, top1 = 4/42, top10 = 5/42, top25 = 9/42, top50 = 13/42, top100 = 16/42**

| card | kernel | rank | pool | → |
|---|---|--:|--:|---|
| RTX 4080 | matmul.square.512 | 106 | 2368 | |
| RTX 4080 | matmul.square.1024 | 342 | 2128 | |
| RTX 4080 | matmul.square.2048 | 15 | 1318 | |
| RTX 4080 | matmul.square.4096 | 29 | 538 | |
| RTX 4080 | matmul.square.512.fp16 | 1548 | 9884 | #2 |
| RTX 4080 | matmul.square.1024.fp16 | 1544 | 8890 | #2 |
| RTX 4080 | matmul.square.1024.fp16 | 296 | 8890 | |
| RTX 4080 | matmul.square.2048.fp16 | 1182 | 5692 | #2 |
| RTX 4080 | matmul.square.4096.fp16 | 319 | 3025 | |
| RTX 4090 | matmul.square.512 | 3 | 2368 | |
| RTX 4090 | matmul.square.1024 | 593 | 2128 | |
| RTX 4090 | matmul.square.2048 | 69 | 1318 | |
| RTX 4090 | matmul.square.4096 | 29 | 538 | |
| RTX 4090 | matmul.square.512.fp16 | 1548 | 9884 | #2 |
| RTX 4090 | matmul.square.1024.fp16 | 188 | 8890 | |
| RTX 4090 | matmul.square.2048.fp16 | 21 | 5692 | |
| RTX 4090 | matmul.square.4096.fp16 | 433 | 3025 | |
| RTX 4090 | matmul.square.512.dynM | 36 | 2368 | |
| RTX 5090 | matmul.square.512 | 24 | 4172 | |
| RTX 5090 | matmul.square.512.fp16 | 339 | 19081 | |
| RTX 5090 | matmul.square.1024 | 625 | 17300 | #3 |
| RTX 5090 | matmul.square.2048 | 0 | 11843 | |
| RTX 5090 | matmul.square.4096 | 497 | 7457 | #3 |
| RTX 5090 | matmul.square.512.dynM | 127 | 19081 | |
| RTX 5090 | matmul.qkv.h4096 | 400 | 9476 | #3 |
| RTX 5090 | matmul.o_proj.h4096 | 376 | 14801 | #3 |
| RTX 5090 | matmul.mlp_gate_up.h4096 | 271 | 7565 | #3 |
| RTX 5090 | matmul.mlp_down.h4096 | 376 | 14801 | #3 |
| RTX 5090 | matmul.qkv.h4096.dynM | 435 | 9476 | #3 |
| RTX 5090 | matmul.o_proj.h4096.dynM | 93 | 14801 | |
| RTX 5090 | matmul.mlp_gate_up.h4096.dynM | 0 | 7565 | |
| RTX 5090 | matmul.mlp_down.h4096.dynM | 0 | 14801 | |
| PRO 6000 | matmul.square.512 | 133 | 4172 | |
| PRO 6000 | matmul.square.1024 | 535 | 3740 | #3 |
| PRO 6000 | matmul.square.2048 | 345 | 2282 | #3 |
| PRO 6000 | matmul.square.4096 | 66 | 902 | |
| PRO 6000 | matmul.square.512.fp16 | 36 | 19081 | |
| PRO 6000 | matmul.square.512.fp16 | 255 | 19081 | |
| PRO 6000 | matmul.square.1024.fp16 | 0 | 17300 | |
| PRO 6000 | matmul.square.2048.fp16 | 661 | 11843 | #3 |
| PRO 6000 | matmul.square.4096.fp16 | 271 | 7457 | #3 |
| PRO 6000 | matmul.square.512.dynM | 16 | 4172 | |

## Fork sibling regret (metric 2) — per-family medians, -O1 blocks (the gate lines)

`emmy eval prior --dataset nodes` (cold `FallbackPrior` → `AnalyticPrior`):

| card (@ -O1) | PLACE+R+S+T | +WSPEC | REDUCE | STAGE | TILE | forks |
|---|--:|--:|--:|--:|--:|--:|
| RTX 4090 | 1.83x | — | **68.50x** | 1.05x | 2.33x | 119 |
| RTX 5090 | 1.32x | 3.84x | **30.08x** | 1.00x | 1.92x | 103 |
| PRO 6000 Max-Q | 1.30x | 3.41x | **34.13x** | 1.01x | 1.73x | 108 |

(`PLACE+R+S+T[+WSPEC]` are the root-level forks that decide several families at once — the flattened big fork; WSPEC
appears only on the sm_120 cards.) Per-kernel breakdown, worst rows: matmul REDUCE (split-K) forks 32.6–77.6x across
all three cards; reduce-op REDUCE forks 18.1–88.6x; `matmul free=4096 red=14336` STAGE 3.79x / TILE 4.45x on the
4090 is the worst non-REDUCE cell.

Leaf reachability corroborates and extends to the deploy regime:

| card | -O1 mean / median / worst | -O3 mean / median / worst | leaf ρ (-O1 / -O3) |
|---|---|---|---|
| RTX 4090 | 26.47x / 1.89x / 88.56x | 41.63x / 2.65x / 153.59x | +0.22 / +0.04 |
| RTX 5090 | 11.97x / 1.41x / 35.86x | 33.28x / 1.42x / 168.78x | +0.46 / +0.09 |
| PRO 6000 Max-Q | 12.89x / 1.65x / 38.41x | 39.31x / 1.38x / 174.40x | +0.51 / +0.22 |

Worst -O3 rows are all the same class: reduce shapes where the pick is a ~200–920 µs config against a 2–5.5 µs best
(e.g. PRO 6000 `reduce red=8192`: pick 921 µs vs best 5.3 µs, **174x**).

## Findings

### 1 — REDUCE-family forks are the catastrophic class — ROOT CAUSE: the featurizer is blind to the REDUCE codec
### on tile-less and partial rows

Median REDUCE regret 30–68x; every REDUCE fork (reduce ops AND matmul split-K) lands 18–89x. **This is not a model
problem.** Verified directly: all 8 children of a reduce fork (`REDUCE@a1 = '' / b4 / b8 / b16 / …`, values 17 µs to
759 µs) produce **byte-identical feature vectors** under `knob_features`. Mechanism: `features._tile_features`
returns `{}` when `_free_slots(knobs)` is `None` (no `TILE` codec), and `_reduce_decomp` — the ONLY reader of the
`REDUCE` codec — is called exclusively inside it. So the reduce partition contributes zero features (a) on every row
of a tile-less reduce kernel, leaves included, and (b) at every matmul REDUCE fork, where the tile isn't decided
yet. The docstring's "pointwise / non-tiled kernels are unaffected" intent silently swallowed the reduce tier. No
model — linear, CatBoost, anything — can rank siblings with identical inputs; the regret's pessimistic-tie rule then
prices the miss honestly (the 2026-07-07 training probe below confirms: a CatBoost trained on two cards reproduces
the cold prior's REDUCE picks exactly on the held-out third, while fixing every family the featurizer can see).
These rows are also label noise for training (identical features, 17-vs-759 µs labels).
**Next:** featurize the reduce decomposition unconditionally (a `_reduce_features` block firing without a tile —
`D_threads` from the coop fold, `D_splitk`(+`_le2`) from the cta split, the fold/finalize fields). Additive encoding
change: node rows store raw knobs and re-featurize at read time, so the 23k fresh rows upgrade in place — NO
`FEATURIZER_VERSION` bump, just a refit. CAUTION: the new features will also fire under the incumbent `_W_A` linear
weights (fit with no reduce rows — e.g. `D_pow2_threads` +56 would suddenly score reduce configs), so the fix must
be validated with the golden A/B before it ships in a deploy path.

### 2 — fp16 warp-tier goldens are unreachable on the sm_89 cards (golden-rank view)

`square.{512,1024,2048}.fp16` rank 1182–1548 in 5.7–9.9k pools on the 4080/4090 (cp.async tier). No realistic
patience reaches them; a cold tune on these cards effectively cannot find the recorded fp16 optima. The TMA-tier
cards are better but still mid-hundreds for half their entries. **Next:** a Phase-3 holdout check — leave-one-card-out
CV must show the fp16 warp tier generalizing across tiers, not just memorizing the TMA cards.

### 3 — mid-hundreds golden ranks are the norm for real model shapes

`qkv` / `o_proj` / `mlp_*` and the big squares sit at ranks 271–661 in 7.5–17k pools — past any patience budget, so
cold tunes burn benches walking to them. Median 263 overall. **Next:** the Phase-7 promotion threshold for golden
median rank should be set well under 100 (the top-100 bucket is where patience plausibly reaches).

### 4 — the deploy regime (-O3) is where the cold prior is weakest

-O3 leaf reachability is 33–42x mean (vs 12–26x at -O1) with calibration collapsing to +0.04..+0.22. Deploy decisions
run at -O3, so the regime that matters most has the least cold-prior signal. Regret can't see -O3 at all (parentless
rows, no sibling groups — by design). **Next:** Phase-3 training must keep `(pool, H_opt)` groups separate and
preferentially retain -O3 rows in the snapshot (already a plan decision); consider an -O3-specific reachability line
in the promotion gate.

### 5 — golden-rank coverage is matmul-only; regret fills the hole

The 5090 golden file holds 37 entries (rms_norm / softmax / reduce / pointwise / attention included), but only matmul
goldens enumerate through the live-fork capture, so metric 1 sees 42 matmul rows and nothing else. The regret metric
covers reduce ops from the same sweeps (they're in the node store) — the two metrics are complementary today, but
golden-rank stays blind to non-matmul regimes until the reduce/pointwise enumeration gap is fixed. **Next:** keep the
known-gap note in the rework plan; fixing `enumerate_graph` for the restored reduce fork would let goldens gate those
tiers too.

## Training-feasibility probe (2026-07-07): stock CatBoost on the node rows, leave-one-card-out

Can the collected data train a prior at all? A stock `CatBoostPrior` (unchanged hyperparams, RMSE log-latency — none
of the Phase-3 machinery) fed the node rows directly (`(features, value_us)` per ok/feat_ver-2 row), trained on the
4090 + PRO 6000 (15,776 rows), evaluated on the **held-out 5090**:

| 5090 @ -O1, family | cold baseline | LOCO-trained | verdict |
| --- | --: | --: | --- |
| PLACE+R+S+T (root) | 1.32x | **1.03x** | fixed |
| …+WSPEC | 3.84x | **1.06x** | fixed |
| TILE | 1.92x | **1.46x** | improved |
| STAGE | 1.00x | 1.00x | held |
| REDUCE | 30.08x | **30.08x — picks byte-identical to cold** | featurizer-blind (finding 1) |

In-sample reservoir calibration +0.73 (trustworthy gate passes). Verdict: the sweep data trains well — cross-card
generalization is real for every family the featurizer can distinguish — and the REDUCE row isolates finding 1's
root cause from model capacity. Probe artifacts in the session scratchpad only; nothing shipped to the live prior
path. (`train_probe.py`: ~20 lines — load node rows, `add_rows`, `fit`, checkpoint; evaluated via
`EMMY_PRIOR_FILE=<probe> emmy eval prior --dataset nodes`.)

## Baseline gate numbers (the bar for Phase 3+)

> **SUPERSEDED 2026-07-08** for the regret metric: after #322 (finding 1's featurizer fix) + the `_W_A` refit +
> the merged refit sweeps, REDUCE fell to **1.09x / 1.13x / 1.00x** and the worst class moved to the 4090's
> big-K TILE goldens — current numbers and per-feature attribution in
> `plans/analytic-blame-ablation-baseline-findings.md`. The rows below are the pre-fix bar, kept for the record.

- Golden rank: **median 263, top10 5/42, top100 16/42**.
- REDUCE regret: **68.50x / 30.08x / 34.13x** (4090 / 5090 / PRO 6000). TILE: 2.33x / 1.92x / 1.73x.
  STAGE: 1.05x / 1.00x / 1.01x (already near-optimal — do not regress).

## Artifacts / workflow notes

- Node store: `~/.cache/emmy/autotune.db` — 23,089 rows (23,063 ok), 3 cards, all `feat_ver=2`, 330 multi-child
  forks. Reproduce: `emmy eval prior --dataset nodes`; golden table: `emmy eval analytic`.
- Sweep wall time was ~3–3.75 h per card (vs the skill's 30–45 min estimate; `.dynM` fp16 + attention shapes at
  ~10 min each dominate) — the `collect-node-data` skill's estimate needs updating.
- CloudRift notes: one PRO 6000 VM died mid-setup (re-rented); the plain PRO 6000 edition was capacity-exhausted →
  Max-Q variant; agents' local driver processes were killed mid-run several times — the detached remote tunes were
  unaffected and the documented `scripts/merge_node_db.py` fallback handled every merge (no lock contention across
  three concurrent-ish merges into one fresh DB).
- Cosmetic nit: cold-prior eval emits scipy `ConstantInputWarning` from `diagnostics._calibration` (constant
  predictions on some leaf groups) — harmless, worth a suppress in a later cleanup.
