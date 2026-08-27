# Analytic prior rework: CatBoost + additive fallback, trained on a frozen local measurement snapshot

> Pruned by the plans/ cap on 2026-07-21 (commit 3c000f1f) while still active; restored and brought current
> 2026-07-23 — see the Update 2026-07-23 section and the STATUS stamps for what is done vs open.

## Goal

Replace the linear `AnalyticPrior` (`emmy/compiler/pipeline/search/prior/analytic.py`; its weight sets live in the
repo-checked `analytic_weights.json` since 2026-07-10, `_W_A` / `_W_A_DYN` in older notes below) with a nonlinear,
CatBoost-based cold-start prior that (a) handles feature interactions and per-regime structure
the linear form cannot, and (b) is **at least as resilient** to code / feature / search-space changes as the current
setup. "Analytic prior" here means the *shipped, cold-start* ranking used when the learned online `CatBoostPrior` is
absent, cold, or quarantined — it must work on a fresh clone with zero measurements.

Success criteria: on the k-fold held-out sets used during training (grouped leave-one-op-family-out /
leave-one-card-out folds, applied identically to BOTH data assemblies — goldens-only and goldens+freeze; goldens are
training data, NOT a reserved held-out set — corrected 2026-07-23, history in decision 5), the new prior beats the
linear baseline on golden rank, per-depth fork sibling-ranking, and golden path-descent; a real-hardware A/B (cold
greedy deploy + cold tune efficiency) confirms it; and a breaking featurizer change costs one GPU-free refit command,
not a data campaign.

## Why (the problem with the current linear prior)

- **Interactions get hardcoded back in as escape hatches.** `AnalyticPrior.__init__` carries
  `atomic_free_split_threshold` / `atomic_free_weight` with a comment saying a linear weight can't express the
  interaction. That's the old rule-based prior sneaking back, one special case at a time.
- **The nonlinearity lives in the featurizer.** Band/target features (`D_bm_band`, `D_near_waves`, `D_w_near_bk`)
  hand-encode thresholds someone must re-derive when hardware or the enumeration changes — the old rule prior's
  brittleness, relocated.
- **Two disjoint weight sets** (`_W_A` vs `_W_A_DYN`, selected on `S_ext_n_symbolic_axis`) is a manual regime split a
  tree model handles with one split.
- **Additive scoring is architecture-blind by construction — confirmed by the 2026-07-09 golden sweeps.** The
  featurizer already carries the arch identity (`Context.features()`: `H_cc`, `H_tc_gen`, `H_sm_count`, `H_smem_*`,
  `H_total_mem`), but those are constant across a pool's siblings, so on a linear model — or ANY additive one, a
  depth-1 stump included — they shift every candidate equally and never change the ranking. Consistently,
  `_W_A`/`_W_A_DYN` carry zero `H_*` weights, and the only hardware sensitivity in the `D_*` block is `H_sm_count`
  feeding the occupancy terms; every other target (warp-BK≈2, tile-area, BM/BN bands) is an arch-independent
  constant. The one weight set therefore prices warp tiles for whichever card dominated the fit: analytic TILE fork
  regret 3.62× (sm_89) vs **11.49×** (sm_120), golden TILE match 5/17 vs **0/17**, and the learned prior inherits
  the censoring (1.48× vs 7.20×) — see `plans/golden-sweep-rtx4090-findings.md` /
  `plans/golden-sweep-rtx5090-findings.md`. Arch-differentiation is an *interactions* problem (`H_* × D_*`), which
  trees express natively and additive tables must get via declared pair terms (decision 11).
- The codebase already made this argument once: `prior/catboost.py`'s docstring on why the linear `BayesianRidgePrior`
  was replaced (monotone-in-every-knob → corner-seeking) applies to the analytic prior too.

## Context for a fresh agent

Read `emmy/compiler/pipeline/ARCHITECTURE.md` (sections: "Forks and the one ranking path", "Learned prior",
"Featurizer vocabulary versioning", the `SearchDB` / node-table paragraphs) before touching anything. Key modules:

- `search/prior/analytic.py` — the linear prior being replaced (the hardcoded interaction gates + the artifact
  loader). Since 2026-07-10 the weights are NOT source literals: they live in the repo-checked, `feat_ver`-stamped
  `search/prior/analytic_weights.json` (params + provenance included), written by the fitter, overridable via
  `EMMY_ANALYTIC_FILE` / `eval --analytic-file` for A/Bs, hard-error on version mismatch — the artifact format +
  loading/quarantine path Phase 5's tier-1 artifact extends.
- `search/prior/base.py` — the `Prior` ABC: reservoir, refit cadence, `evidence_pick`, the **calibration gate**
  (`trustworthy`, `CALIBRATION_MIN`) added on branch `feature/prior-calibration-gate`. The quarantine pattern there
  (measure after fit → gate ownership of decisions → log the verdict) is the template this plan's promotion gate mirrors.
- `search/prior/catboost.py` — the learned online prior (NaN-fill semantics, `feat_ver`-stamped JSON checkpoint,
  discard-whole on version mismatch). Its training loop is NOT touched by this plan.
- `search/prior/fallback.py` — `FallbackPrior` composition; `score()` folds the analytic in as a dimensionless
  multiplier `analytic**W` centered at neutral 1.0 (`config.analytic_tilt`).
- `search/features.py` — `knob_features` (THE one featurization), `FEATURIZER_VERSION`, `tile_signature`.
- `search/db.py` — the `node` table (`NodeRow`): leaf vs branch semantics, per-kind upsert, label-quality columns,
  `merge_nodes` (cross-hardware accumulation, driven by the `collect-node-data` skill).
- `search/policy/mcts.py` — `_collect_node_records` (what writes node rows), PUCT `_select` (scores *partial* prefixes).
- `search/prior/diagnostics.py` — `node_report` (fork sibling-ranking eval), backing `emmy eval`.
- `search/analytic.py` — golden-eval glue: `enumerate_graph` (live-fork candidate capture), `evaluate_golden`.
- `search/golden.py` + per-GPU golden YAMLs — the curated verified-optimum configs.
- `scripts/golden_knob_heuristics.py` — the current offline fitter (random search + coordinate descent over linear
  weights, golden-rank objective). Superseded by the Phase-4 fitter.
- `search/data/` — `Sample` / `Dataset` / `ShapeKey`; `Dataset.fold_node_rows` (group-holdout folds by op_sig / gpu).

Vocabulary: **leaf** = complete knob config with a direct bench; **branch** = partial knob prefix, value-of-position
label (min over benched descendants); **pool** = one op's candidate enumeration on one card/regime; **goldens** =
verified-best configs (tuned + A/B'd with integrity gates), committed per-GPU.

## Decisions made, and why

1. **CatBoost (tier 1) + machine-fit additive table (tier 2) + option-0. No linear model; no hand-written rule prior.**
   The fallback chain is: repo-checked CatBoost artifact → additive table → option-0 emission order. The user explicitly
   rejected keeping the linear model. The tier-2 additive table replaces BOTH the linear weights and the earlier
   "hand-written physics rules" idea: a boosted **depth-1 (stump) ensemble is exactly additive** — `Σ_k shape_k(x_k)` —
   exported at fit time to a plain per-feature table (the `analytic_weights.json` artifact — since 2026-07-10 the
   linear weights already ship there rather than as source literals; CatBoost JSON dump → collapse stumps). Scoring is
   ~20 lines of `shape_k(feats[k]) if k in feats else 0`. Each shape function reads as a fitted rule
   ("`D_near_waves`: +8 in [1.5, 2.5]") — the old rule prior's legibility without its bit-rot, regenerated by one script.
2. **Resilience is a property of the fit pipeline, not the artifact.** The linear prior survives churn because (a) the
   fitter regenerates candidate pools from the live enumeration, (b) goldens store raw knobs + shape matched
   schema-agnostically (`tile_signature`), (c) missing features degrade gracefully (`feats.get(k, 0.0)`). Keep all three
   properties; swap the model class. Never persist feature matrices — always re-featurize raw knob dicts at fit time.
3. **Graceful degradation semantics per tier.** Tier 2 gets the linear model's locality *by construction* (additivity +
   skip-if-missing: a retired feature deletes one term, the rest stand). Tier 1 (depth 4–6, interactions → non-local
   damage possible) gets it *statistically*: per-**pool** masking augmentation (features NaN'd with p≈0.05–0.15, K
   replicas + one clean copy; `nan_mode="Min"` trains real absent-feature routing) plus `rsm < 1.0` for split-level
   redundancy. Caveats: masking must be per-pool, not per-row (a retired feature is absent for the WHOLE pool at
   inference; per-row masking teaches a ranking loss that missingness discriminates siblings — exactly wrong). And
   dropout does NOT cover semantic drift (same name, new meaning/scale) — distribution fingerprints do (decision 9).
4. **NaN vocabulary is preserved.** Absent feature = "not-yet-decided" (partial fork prefix); explicit OFF value =
   "decided unused" (the knob-stamp invariant). The masking augmentation therefore masks the structural side
   (`S_*`/`H_*`/`D_*`) at pool level only; knob-side missingness in training comes from REAL prefix rows (decision 6),
   so the not-yet-decided semantics stay honest.
5. **Training data = a frozen, LEAF-ONLY local extract of the fleet node DB; ~~goldens are the held-out acceptance
   set~~ goldens train too — held-out evaluation is the k-fold splits (corrected 2026-07-23; reversal history
   below).** RESCOPED 2026-07-09 — was "a committed, curated snapshot". With no second consumer of the dataset the
   commit bought nothing: row-level PR review of thousands of leaves was never real, and the reproducibility /
   verifiability properties survive as a digest pin. The data stays local; distribution (HF dataset repo, GCS,
   git LFS) is deferred until sharing matters.
   - Why frozen (not fit-from-DB): the DB is a live store — tunes and merges write into it — so fitting from it
     directly makes fits non-reproducible and fitter A/Bs (CV folds, loss choices, masking rates on identical data)
     incomparable. One command extracts → writes one local file → sha256s it; the fit is a pure function of
     (repo, pinned freeze), and the tier-1 artifact stamps the digest. CI keeps every data-free check (feat_ver,
     column vocabulary, fingerprints, golden gate) — none need the freeze present.
   - Why leaf-only: a leaf is a durable fact about (config, GPU, compiler version). A branch row is a run-artifact — a
     policy-dependent coverage bound (min over whatever that search benched), possibly stale/non-monotone vs its own
     re-measured leaves, and **path-structure-fragile** (its identity is a prefix in the *historical* fork-tree topology;
     level reorders / knob moves orphan it, and no spelling migration fixes that). Leaves are complete points in knob
     space — valid under any tree organization.
   - ~~Why goldens never train: they are the only *verified-optimum* labels (tune-golden A/B + integrity gates).
     Training on them makes the promotion gate measure memorization. Their value is spent as the eval set.~~
     **REVERSED 2026-07-15: goldens DO train, as a fit-time union.** The search data is censored exactly where the
     verified optimum sits (the -O1 lane inversion, the post-swizzle fm optima found only by manual sweeps): a golden
     the search never reached has NO nearby training rows in the freeze at all, so excluding goldens starves the fit
     in the one region we know the answer. The fitter unions `Dataset.from_golden()` (source-marked, card-faithful
     context, deployable-regime latency — must group with `H_opt=3` pools per decision 7, never the -O1 lane) with the
     freeze rows at data-assembly time; the freeze file itself stays a pure DB extract (goldens are repo-checked, so
     the fit's `(repo commit, freeze digest)` pin already covers them — baking copies in would only go stale). Gate
     integrity lives in the k-fold holdouts used during training (decision 13): grouped leave-one-op-family-out /
     leave-one-card-out folds, the same mechanism for the goldens-only and goldens+freeze assemblies — each golden
     is scored held-out by the fold model that never trained on it, and full-train vs held-out is the memorization
     split. *(Corrected 2026-07-23: an earlier version of this note also reserved "the next newly-onboarded model's
     goldens" as a standing unseen acceptance set — dropped; no golden set is held out of training.)*
   - Freeze contents: ok leaves + **fail leaves** (negative examples — "doesn't build/launch here" is durable) + all
     regime rows (-O1 and -O3 both; no size budget — keep EVERY leaf passing the sanity filter). Raw knob dicts +
     shape/context/gpu + bench stats + provenance (measured_at, run_id; the freeze header stamps the repo commit and a
     collection-policy note at freeze time — per-row commit/policy aren't recorded in the DB). No `parent_key`, no
     `visits`, no tree schema.
   - Durability: local-only means `~/.cache/emmy/autotune.db` is the SOLE copy of data that cost rented-GPU money (a
     `.bak-pre-wipe` neighbor shows cache wipes happen). After each `collect-node-data` merge, copy the DB (or the
     freeze) somewhere durable; re-collection is possible but costs rental hours per card. *(Partially automated
     2026-07-22: `remote_node_collect.py` backs up the local autotune DB after each merge; an off-laptop copy is
     still an open question below.)*
6. **Branch training rows are SYNTHESIZED at fit time, not stored.** Group the freeze's leaves by prefix under the
   **current** code's fork structure; label = min over the group (value-of-position); confidence weight = descendant
   count. Strictly better than stored branch rows: consistent with today's tree (what PUCT will actually score),
   consistent with the kept leaves (no stale bounds), and it doubles as the principled prefix-masking augmentation.
   Favorable error structure: shallow prefixes aggregate many leaves → tight labels exactly where a misranking costs
   the most; deep sparse prefixes get loose labels where errors cost least.
7. **Loss = grouped ranking (per fork sibling set, per level), not µs regression.** The analytic prior's contract is
   ordinal — µs calibration is the learned prior's job, and `FallbackPrior`'s tilt needs the dimensionless neutral-1.0
   multiplier. Emit one ranking group per fork node (its children with value-of-position labels), depth-weighted so
   shallow levels dominate; also group per `(pool, H_opt)` so -O1 ranking-lane rows and -O3 deployable rows never pool.
   Wrap the raw score in the monotone `exp(-scale·s)` squash to preserve the multiplier convention, with the special
   case: no `D_*` features → exactly 1.0 (no opinion).
8. **Monotone constraints are the durable home for physics rules.** Declared per feature name next to the feature list
   (e.g. `D_splitk_excess` can only hurt); survive refactors legibly; bound damage from degraded inputs; a renamed
   feature loudly fails the constraint mapping instead of silently misfiring.
9. **Promotion gate + CI, mirroring the calibration-gate pattern — TWO quality metrics only.** Metric minimalism is a
   deliberate decision: one metric per question the prior answers, each denominated so a threshold is meaningful.
   (a) **Flattened golden rank** (exists today) — "would a cold greedy deploy find the golden?"; (b) **fork sibling
   value regret**, bucketed by the knob FAMILY of the fork's delta (TILE / REDUCE / STAGE / structural), on held-out
   node data — "does the prior steer the search into fast subtrees?": median of `value_us(predicted-best child) /
   value_us(true-best child)` per fork group, gated hardest on the shallow families. Family bucketing, not raw
   `depth` — depth is rule-step distance (renumbers on pass changes; live store shows forks only at depths 4–8/11–15
   with a 75-child bulge at the tile fork), family is the stable semantic level. Rejected as gate metrics: golden
   path-descent (presumes the golden's branch must win every level — a sibling branch may hold a near-equal config;
   sibling regret over measured values answers the same question honestly), top-1/top-k rates and Spearman (proxies
   for regret). A **feature-ablation robustness check** (mask each feature/family, measure metric collapse) is part
   of the gate as a resilience verification, not a quality metric — built in Phase 2 (blame + ablation tooling) and consumed by the Phase-8 gate.
   Promote only if ≥ incumbent; otherwise keep the old artifact and log the
   quarantine. CI: artifact `feat_ver == FEATURIZER_VERSION`, model cols ⊆ live featurizer keys, golden median rank ≤
   threshold, per-feature **distribution fingerprints** (training quantiles vs live golden-enumeration featurization —
   catches same-name-new-meaning drift, the class dropout can't).
10. **Refit workflow (two clocks).** Steady state: no refit (deterministic fit, pinned artifact, CI green).
    *Code events* (frequent, GPU-free): featurizer/encoding change → measurements untouched, re-featurize the freeze,
    one-command refit; search-space growth → old configs remain valid at the new knob's OFF value, refit immediately as
    a bridge (model neutral on the new dimension via NaN), then schedule a collection sweep for coverage.
    *Data events* (rare, GPU-spend): codegen drift — detected by the golden gate degrading against freshly re-tuned
    goldens — or new hardware → collection sweep → merge → re-freeze → refit. The learned prior stays the
    *online* model (freshest local data); the analytic prior is a *release artifact* on the repo clock.
11. **Arch-differentiation is an interactions problem; every tier gets an explicit answer (added 2026-07-09, from
    the cross-card golden sweeps).** `H_*` features are constant within a sibling pool, so they carry zero marginal
    ranking signal — they act only through `H_* × knob` interactions. Tier 1: keep `H_*` in the training columns,
    depth ≥ 2 (never stumps-only), and gate on the leave-one-card-out fold so a learned arch split is verified, not
    assumed — the per-pool ranking groups mean per-arch *coverage* in the freeze, not row volume, decides whether
    the split is learnable (the 5090's golden-tile region is censored today; Finding 3's pollution filter is a
    prerequisite). Tier 2: additive ⇒ arch-blind unless arch×knob pair terms are declared — seed the whitelist with
    `H_tc_gen × warp-aspect`, `H_tc_gen × splitk-need`, and TMA/WSPEC staging × tile geometry (the 5090 sweep's
    exact miss directions: wide warps over-weighted, split-K under-weighted on sm_120). Featurizer: add the
    engineered features the sweeps showed missing on BOTH cards — masked warp aspect (`wNxM` under a symbolic axis;
    today no feature separates those siblings, the 4090's Finding 1) and TMA/WSPEC-conditioned tile pricing.
    Vocabulary additions are code events (decision 10): GPU-free refit, no new collection required.
    **STATUS (2026-07-09): the featurizer additions + analytic refit LANDED** (branch
    `feature/prior-arch-features`): `D_w_grid_m/n/aspect` (warp-grid arrangement over the canonical free slots —
    same-tile different-grid siblings were byte-identical before) and `D_tma_{grid_m,grid_n,aspect,log2_area,
    l2_splitk}` (geometry mirrored onto TMA-staged rows — the per-candidate arch key). Two findings along the way:
    (a) `S_masked_m/n/k` have NO producers left (`masked_axis_features` is orphaned — the `D_neg_masked_*` weights
    are dead everywhere), so the masked-warp-aspect axis rides the dyn weight-set split instead (the regime split IS
    the masking condition); (b) the fitter's random-restart stage overfits the golden objective while trashing
    node-store calibration (4090 Spearman +0.46 → +0.18) — the landed weights are coordinate-descent-from-seed only
    (`--samples 0`), which Phase 4's fitter should keep as a lesson (regularize toward the incumbent). Results:
    `eval analytic` top1 31 → 40/53, top50 41 → 47/53, every sweep `.dynM` miss to rank 0 (mlp_down 2358 → 0,
    square.512 1198 → 0, qkv 780 → 0, o_proj 663 → 0); 4090 TILE fork regret 3.62x → 2.31x, calibration +0.46 →
    +0.53; the 5090 node table is unadjudicable until the Phase-3 roofline floor lands (its regressed rows sit on
    physically-impossible baselines). The dyn set's divergence is exactly the predicted axis: `D_tma_grid_m/n`
    +4.75 vs static +2.49, `D_tma_l2_splitk` +3.15 vs −0.68 (the 5090's split-K-under-TMA wins). NEXT: re-run the
    golden sweep on a rented 4090/5090 to confirm the cold search now reaches the goldens (tune-golden skill).
12. **Naming: the two priors are named by role/clock, not model class — `OfflinePrior` / `OnlinePrior` (settled
    2026-07-11).** Once both priors are CatBoost-backed, implementation names stop differentiating them (and
    "analytic" becomes factually wrong — the shipped prior is machine-fit). `AnalyticPrior` → `OfflinePrior`
    (offline-fit release artifact, repo clock); learned `CatBoostPrior` → `OnlinePrior` (online-updated local
    model). The tier chain (CatBoost → additive table → option-0) hides behind the ONE public `OfflinePrior`
    class — its loader picks the best valid artifact via the existing `kind` field, so `FallbackPrior` stays a
    two-way composition (`online µs × offline**W`). Rename surface: `prior/analytic.py` → `prior/offline.py`,
    `prior/catboost.py` → `prior/online.py`, `search/analytic.py` (golden-eval glue — shares the doomed name),
    config vars (`EMMY_ANALYTIC_FILE` / `analytic_tilt` / `EMMY_PRIOR_FILE` → offline/online spellings; old env
    names stay as accepted aliases with a deprecation warning — they live in shell profiles), `eval analytic`,
    artifact filenames, docs/skills. Unchanged: the `Prior` ABC, `FallbackPrior`, DB schema, checkpoint keys.
    Lands as its OWN mechanical PR BEFORE the Phase-4 extraction so the fitter and its artifacts are born with
    the new vocabulary. (Prose elsewhere in this plan predates the rename and says "analytic" / "learned".)
    **STATUS: LANDED 2026-07-13 (#355)** — `prior/offline.py` / `prior/online.py`, `offline_weights.json`,
    `EMMY_OFFLINE_FILE` / `EMMY_ONLINE_FILE` (old spellings accepted as aliases), `eval offline` / `eval online`.
13. **Phase-4 fitter: one pipeline, two orthogonal switches — model class × training data (settled 2026-07-11).**
    "Current vs new training" differs on two independent axes, so the fitter exposes both:
    `--trainer {linear,catboost}` × `--data {golden,freeze:<path>}` (trainers declare which data kinds they
    support; catboost is freeze-only). Three cells get run and compared: linear×golden (the incumbent process,
    preserved), linear×freeze (isolates the data/objective effect), catboost×freeze (the target) — and for
    cell 3 vs cell 2 to isolate the model class, the linear freeze trainer optimizes the SAME grouped pairwise
    loss CatBoost does (the coordinate-descent optimizer with its objective parameterized), with everything else
    held fixed: one featurization, same eval folds, same seed policy. Preservation is verified, not promised:
    `golden_knob_heuristics.py` becomes a thin wrapper over the extracted fit module (`search/prior/fit/`),
    pinned by two regression tests — same seed + same incumbent → byte-identical artifact, and the incumbent
    artifact through the new eval path reproduces today's `eval analytic` numbers exactly. Entry point: an
    `emmy fit` subcommand (the command layer owns the snippet-tracing golden-case builder — `pipeline/` never
    imports `commands/`). Every fit run writes a per-run METRICS FILE (JSON, in a `_tune/fits/…` run dir, not
    repo-checked): a header (trainer, data kind + digest, feat_ver, seed, fold spec, repo commit, augmentation
    params — two runs are comparable iff their headers differ only in the axis under test), per-fold blocks keyed
    by held-out group with both gate metrics per card (never pooled), golden rank reported both ways (full-train
    vs leave-that-op-out — the memorization split), and per-card aggregates. Keys are stable identifiers (golden
    name, family, card), so comparison = joining two files; a side-by-side renderer can come later. Deterministic:
    same header inputs → identical metrics content, so an A/B diff contains only real differences. Fold-trained
    models plug into the eval as stub priors via the Phase-2 features seam, so the eval suite has zero
    per-trainer branches. Masking augmentation is trainer-owned, not a shared prepare stage (tier-1 needs it,
    linear would only gain noise).

14. **The freeze adopts the goldens' declarative row spelling (settled 2026-07-23).** The landed v1 freeze dumps
    `NodeRow`s: identity as digests (`node_key` / `context_key` / `op_sig`) and the persisted `features` dict —
    the featurizer's encoded `S_*`/`H_*` output — which is exactly what decision 2 forbids ("never persist feature
    matrices"): the shape is unrecoverable from a row, and every encoding-only featurizer change quarantines the
    file (`feat_ver`). Goldens got this right: declarative kind + shape fields + verbatim TUNABLE knobs, with
    `S_*`/`H_*`/pools re-derived at load/fit time. Settled shape of the rework:
    - **Rows are golden-entry-shaped**: kernel kind, declarative shape fields, `knobs` = verbatim tunable knobs
      only, measured µs, grouped per GPU — plus a measured-row extension block with no golden equivalent:
      `status` (`bench_fail` negatives), `variance` / `n_samples` (confidence weighting + quality-aware
      replacement), the nvcc opt lane (the -O1/-O3 twins), `run_id` / `measured_at`. No `cublas_us` — sweep
      points bench emmy-only; it is NOT the literal `GoldenConfig` schema, but the same spelling.
    - **Declarative identity is captured at COLLECTION time, not reverse-engineered at freeze time.** The
      three-slice sweep already holds (kind, shape fields, gpu, knob row, opt level) per benched point — its
      ledger is keyed `(gpu, shape, knob signature)` — so the bench-to-node recorder carries the identity
      (`ShapeKey` in `search/data/shape.py` is the carrier). Tune-written rows have no such identity today:
      either `ShapeKey` gets recorded at tune time too, or they are excluded from goldens-format freezes.
    - **The freeze's provenance contract survives unchanged**: sorted rows, digest over the row payload,
      provenance header, hard-error loading — orthogonal to row spelling.
    - **Payoff**: one data-assembly path in the fitter — the case builder snippet-traces each distinct shape
      once, enumerates the pool, and joins golden entries AND freeze rows into it by shape + knob spelling
      (`ShapeKey` / `tile_signature`), no `op_sig`-digest bridging; encoding-only featurizer changes stop
      quarantining freezes (re-derive at load — most of Phase 9's motivation dissolves); the k-fold group keys
      (op family, gpu) read directly off every row.
    **STATUS (2026-07-23): IMPLEMENTED on `feature/golden-neighbor-collect`** (CPU-tested; GPU smoke pending the
    next rented card). Landed: `NodeRow.shape_spec` + node column (COALESCE-kept both replacement directions),
    `run --bench --record-shape` with the per-leaf `ShapeKey.joins` gate in `bench_record.py` (`joins` tolerates
    the sweep kinds' snippet-unstable dtype-derived `is_warp` — found implementing this: an all-fp16 norm snippet
    flips it vs the goldens' recorded `False`), the sweep passing its group spec, `traced_s_features` (kind-generic
    multi-op `S_*` re-derivation), and freeze v2 (`FREEZE_VER=2`): per-GPU YAML dir + manifest, content-level
    digests, loader re-derivation, v1 JSONL refused with a re-freeze pointer, load-time `feat_ver` gate dropped.

## Constraints and requirements

- **ONE featurization**: everything (both tiers, synthesis, evals) reads `features.knob_features`. No private feature
  views (that's what killed the old rule prior).
- **ONE ranking path**: both tiers implement the `Prior` ABC and compose behind `FallbackPrior`. No policy special-cases.
- Analytic contract: lower-is-better latency *proxy*, ordinal only, neutral exactly 1.0 when it has no opinion, usable
  with zero measurements on a fresh clone, cheap enough for greedy's ~1k-row flattened batches (vectorize tier 1's
  predict like `CatBoostPrior.mean_scores`).
- Tier-1 artifact: repo-checked, stamped `feat_ver` + training-data digest + fingerprints, discarded WHOLE on version
  mismatch (the `CatBoostPrior.from_json` semantics), deterministic refit (seeded).
- Freeze: leaf-only, raw-knob spelling, digest-stamped, keep-everything (no row budgets — the file is local and never
  reviewed row-by-row; curation is one shared sanity filter: current `feat_ver`, degenerate-bench latency floor, fails
  kept as negatives).
- Train/inference representation identity: synthesized partial rows NaN out undecided knobs exactly as the live
  descent featurization does.
- Code-event refits must need no GPU. ~~Goldens must never enter training.~~ (Reversed 2026-07-15 — decision 5:
  goldens join training via the fit-time union; held-out integrity comes from the k-fold splits alone — corrected
  2026-07-23, the earlier "+ the next new model's goldens" reservation is dropped.)
- Repo conventions: knobs declared only in `search/space.py`; `pipeline/` never imports `backend/`; markdown wrapped
  ~120 chars; plans are ephemeral (this file gets deleted when the work lands — durable content goes to ARCHITECTURE.md).

## Do NOT change

- The `Prior` ABC surface (`score` / `mean_score` / `mean_scores` / `pick` / `trustworthy` / `evidence_pick`) and
  `FallbackPrior`'s blend semantics (learned µs × `analytic**W`, neutral 1.0; evidence-pick precedence).
- The learned `CatBoostPrior`'s online loop: reservoir sampling, `REFIT_SCHEDULE`, checkpoint format, calibration gate.
  Different clock, different job — this plan only touches the *analytic* half of the composition.
- The knob-stamp invariant and OFF-value / NaN semantics (`tests/compiler/passes/test_knob_stamp_invariant.py`).
- The `node` table schema semantics (per-kind upsert, newest-leaf-wins, `merge_nodes`) — the freeze step is a
  read-only consumer. ~~The `collect-node-data` flow keeps working unchanged (ε-greedy 0.25 collection stays; it's
  the provenance the freeze prefers).~~ *(Overtaken 2026-07-22: the ε-greedy tune phase was retired and
  `collect-node-data` is now the three-slice golden-anchored sweep — see the Update 2026-07-23 section. The table
  schema semantics above still hold.)*
- Golden dataset role/format and `tile_signature` matching; the golden A/B integrity gates.
- Greedy mechanics: `flatten_leaves` (complete-leaf scoring), validity blocklist retries, option-0 last resort.
- Existing `emmy eval` subcommands keep working (new evals are additive).

## Phases

Each phase is independently verifiable; 1–4 need no GPU; hardware spend concentrates in 7. Dependencies: 1 → {2, 3};
3 → 4 → {5, 6} → 7 → 8; 9 floats; the decision-12 rename PR precedes 4's extraction. Phase 3 shrank in the 2026-07-09 rescope to a thin freeze step — effectively step 0
of Phase 4. Phase 2 is diagnostic tooling — parallel with 3, consumed by 4's fitter loop and 8's gate.

1. **Evaluation harness + incumbent baseline.** Exactly the two gate metrics (decision 9): flattened golden rank
   (exists — `eval analytic`) and family-bucketed fork sibling value regret (rework `node_sibling_ranking`:
   regret instead of top-1/Spearman, bucket by the fork delta's knob family). NO comparison runner — comparing
   priors = diffing two report files (the golden-sweep findings already track progress across dates this way);
   the Phase-4 fitter emits the same report per CV fold. May add a `Prior`-internal score-on-features seam
   (both priors already featurize-then-predict) — needed by the Phase-2 attribution tooling and the Phase-4 fitter; no
   behavior change, equivalence-tested. **Prerequisite (SATISFIED 2026-07-09 — see assumptions): one fresh ε-greedy
   `collect-node-data` sweep on a rented card** — the 2026-07-06 audit found the local node store 100% retired-vocabulary (see assumptions), so no
   nodes-based metric has usable data until then; the sweep doubles as Phase-2/3 seed data. Deliverable: baseline
   findings report (plans/) following the house report conventions (title + card + date; context header defining
   the numbers; data tables FIRST with a per-row link to the numbered finding each bad row motivates; numbered
   findings each ending in a "Next:" action; artifacts/workflow notes last). Report layout: per-KERNEL rows ×
   per-FAMILY columns — golden table one row per golden (`name | rank/pool | → finding#`); regret table one row
   per op label (`kernel | TILE | REDUCE | STAGE | structural | n_forks`) with a per-family aggregate line at the
   bottom, which IS the gate number; the per-kernel rows are the diagnostic view feeding the findings. Verify:
   read-only vs live DBs; unit tests on synthetic `NodeRow` sets + stub priors; `make test` green; no behavior
   change to compile/tune.
   **STATUS: DONE.** The two-metric harness landed as #361; the metrics only became *honest* with #364
   (de-saturated squash + tie-pessimistic golden rank) and #363 (card-faithful eval contexts) — see the 07-15
   Update. Baseline reports exist (`plans/analytic-blame-ablation-baseline-findings.md`, the golden-sweep
   findings series); the family-bucketed regret is a projection of the shared `ForkRecord` builder (Phase 2).
2. **Per-feature regret attribution — blame decomposition + ablation Δ (diagnostic, NOT gate).** Two output modes
   over the per-fork records the regret metric already produces. (a) **Blame**: for each missed fork, the
   per-feature signed contribution to the pick-vs-best score gap — EXACT for the linear analytic prior
   (`Σ_k w_k · (f_k(picked) − f_k(best))` sums to the gap by construction), SHAP-difference for CatBoost —
   aggregated per family, regret-weighted: "which feature caused this misranking". (b) **Ablation Δ**: re-pick with
   one feature/family masked; `Δregret = regret_masked − regret_full` per family — a negative Δ flags an actively
   misleading feature. Motivation (2026-07-07): both headline baseline discoveries were manual instances of these
   views — the identical-vector REDUCE blindness is blame's degenerate case ("no feature varies across siblings"),
   and the post-#318 `_W_A` serial preference on reduce forks took a hand decomposition to find; the pending `_W_A`
   reduce refit consumes the blame table directly. Deliverables: the linear blame table first (exact, ~small),
   CatBoost SHAP mode + ablation Δ after; run for the incumbent on the fresh sweep data; and wire the new views
   into the tune-golden / tune-model skill report templates (their finding-evidence lists — the regret block was
   added there when it landed; blame replaces their hand-derived "per-knob misses" attribution step). STRICTLY diagnostic — the
   gate stays two numbers (decision 9): attribution among correlated features is non-unique (ablation double-counts
   redundancy, SHAP splits it arbitrarily), so never threshold on it. NaN-mask ablation is a fair proxy only once
   the fitter's dropout training (phase 4) makes masked queries in-distribution — flag that in the output until
   then. Verify: unit tests with stub priors (a planted weight's blame is recovered; the linear decomposition sums
   exactly to the score gap); read-only against the node store.
   **STATUS (2026-07-07): LANDED except the CatBoost SHAP mode** (deferred until a trained artifact exists to test
   against — decided with the user; the out-of-distribution caveat flag ships now and fires for any non-analytic
   prior). Shipped: `eval prior --dataset nodes --blame/--ablate` over a shared `ForkRecord` builder (the regret
   metric is now a projection of it), the `Prior` features seam (`mean_score[s]_features` + exact
   `explain_features`, the hardcoded interactions as `gate:*` pseudo-terms), the skill-template wiring, and the
   incumbent run — `plans/analytic-blame-ablation-baseline-findings.md` (headline: 42/43 REDUCE misses BLIND at
   ~89% of total regret-weight; `D_l2_reuse` actively misleading on TILE forks, Δ −0.56x).
   **Re-run 2026-07-08 post-#322 + refit sweeps (same report, before/after): the acceptance check passed** —
   REDUCE blind 42 → 0, regret-weight 1670 → 84, per-card medians 34–68x → 1.00–1.13x (PRO 6000 on re-featurized
   old rows alone); `D_l2_reuse` rehabilitated by the refit; the worst class moved to the 4090's big-K TILE
   goldens (215x — a Phase-4 priority test case), and `D_l2_bm` / the splitk-roundtrip gate are the new
   misleading shortlist.
3. **Local measurement freeze (RESCOPED 2026-07-09 — was "committed measurement snapshot"; effectively step 0 of
   Phase 4).** One command: read the node DB read-only → sanity filter (leaf-only, current `feat_ver`, the
   degenerate-bench latency floor, fails kept as negatives — a shared ~20-line predicate, not an exporter with
   policy) → write ONE local file (JSONL of leaf rows or a `VACUUM INTO` sqlite copy; no format ambition — it is
   regenerated from the DB) → sha256 + provenance header (repo commit, run_ids, policy note). The latency floor is
   **load-bearing for sm_120**: the 5090 sweep found physically-impossible leaves in the node store (9.17 µs on a
   ~60 GFLOP fp16 matmul ⇒ ~6700 TFLOP/s vs the card's ~210 peak; up to 24643× fake fork regret) — floor each leaf
   at a per-shape FLOP roofline from its `S_ext_*` extents and the card's peak (derivable from the `H_*` regime),
   and drop rows below it; without this the 5090 rows poison both training and the regret gate. Loader: freeze →
   leaf-only `NodeRow`s (`parent_key=None`, `feat_ver` from the header) so `Dataset.from_node_rows` /
   `fold_node_rows` and the Phase-1/2 evals work unchanged. Dropped with the commit: pool-grouped per-card repo
   files, row budgets, coverage-quota curation, in-repo manifest, CI data loading, the format doc. Deferred until a
   second consumer exists: HF dataset / GCS / git LFS distribution. Verify: freeze-twice determinism (same DB → same
   digest); round-trip vs `iter_nodes`; evals accept a freeze path interchangeably with the live DB.
   **STATUS (2026-07-15): LANDED** (branch `feature/measurement-freeze`; merged as #382, 2026-07-16, together with
   the item-5 bench-to-node recorder): `search/data/freeze.py`
   (`freeze_reason` / `write_freeze` / `load_freeze` / the sniffing `load_node_rows`) + `scripts/freeze_node_store.py`;
   JSONL, header line with both version axes reserved (`knob_ver`/`encoding_ver`), digest over sorted row payload
   only; `eval online --dataset nodes --db` takes a freeze via the sniff seam. Verified on the real store: 5,990
   leaves frozen (5,963 ok + 27 bench_fail; 10,787 branch rows excluded), freeze-twice digest identical, eval
   degrades to leaf metrics as designed. `load_node_rows` is the seam the Phase-4 `emmy fit --data freeze:<path>`
   consumes.
   **FORMAT REWORKED (decision 14; implemented 2026-07-23 on `feature/golden-neighbor-collect`):** the v1 row
   spelling (a `NodeRow` dump — digest identity + persisted featurization) is superseded by the goldens-format
   freeze v2 (per-GPU YAML dir + manifest, identity captured at collection time, features re-derived at load);
   v1 files are refused with a re-freeze pointer. The determinism/digest contract and the `load_node_rows` seam
   carried over. See decision 14's STATUS for the landed pieces.
4. **Training pipeline.** Prefix-row synthesis under the current fork structure, sibling-group ranking datasets,
   pool-level masking augmentation, group-holdout k-fold (leave-one-op-out AND leave-one-card-out via
   `Dataset.fold_node_rows`), candidate CatBoost rankers with monotone constraints; every fold reported through the
   Phase-1 suite. Deliverable: the fitter (successor to `golden_knob_heuristics.py`) + CV report vs the linear
   baseline on identical folds. **This is where old-vs-new comparison lives.** Offline only.
   Settled design in decision 13 (trainer × data switches, extract-and-wrap, per-run metrics files, `emmy fit`).
   Build order: (0) the decision-12 rename PR; (1) extraction + wrapper + the two preservation tests (pure
   refactor, no behavior change); (2) the Phase-3 freeze command + loader; (3) metrics file + fold harness → run
   linear×golden — the first held-out numbers the current process has ever had (golden-case folds:
   leave-one-op-family-out and leave-one-card-out; small-n, so the card axis is the meaningful one; node folds
   need no per-fold retraining for this cell — the incumbent trains on goldens, so any node fold is
   out-of-sample); (4) prefix synthesis + ranking groups → linear×freeze; (5) catboost×freeze — the open Phase-4
   questions below (loss, masking p/K, monotone list, depth budget) get decided empirically here, each candidate
   config just another metrics file. `--folds both` = the union of the two fold axes (two report sections), not
   nested op×gpu — nesting starves training at this dataset size.
   **STATUS (2026-08-13): steps 0–3 LANDED; the CatBoost trainer LANDED on the GOLDEN dataset (a cell this
   build order did not anticipate — it assumed catboost was freeze-only); step 4's freeze data is still the open
   frontier.** The `catboost` × `golden` cell exists because the model-class question and the training-data
   question turned out to be separable: holding the data and the pool-rank objective fixed and changing only the
   model class is the controlled comparison `plans/offline-prior-and-search-findings.md` B3 half-ran. What landed
   with it: `CatBoostModel` (base64 `cbm` in the artifact, `kind` dispatching the load), a `QuerySoftMax` trainer
   with hard-negative mining, the routing stamp demoted from a weight-set selector to an ordinary column, and
   `TREE_FEATURES` — the view minus every feature that exists only because a linear model cannot form it.
   Still open here: linear×freeze and catboost×freeze, both waiting on sweep volume.
   - (0) rename — #355 (07-13). (1) extraction + wrapper + preservation tests — #383 (07-20): fit core in
     `search/prior/fit/` (`linear.py`), `golden_knob_heuristics.py` is the thin legacy wrapper. (2) freeze
     command + loader — #382 (07-16, Phase 3 below). (3) metrics file + fold harness + **linear×golden** —
     #404 (07-20): `emmy fit` (`emmy/commands/fit.py` owns the snippet-tracing case builder; `fit/cv.py` owns
     the folds/metrics). **This cell is the CURRENT training pipeline, and it trains on goldens only** — no
     freeze/node rows enter any fit yet. Shape as landed: `--trainer {linear,catboost} × --data
     {golden,freeze:<path>}` with every cell except linear×golden rejected loudly; `--folds
     {op_family,gpu,both,none}`, each golden held out exactly once per axis, holdout vs train rank + per-card
     gap (overfit vs weak-model split), aggregates per card ONLY, fold models seeded from ZEROS (incumbent
     seeding would leak each golden into its own holdout model — the full-train shippable artifact keeps
     incumbent seeding, recorded in the header); writes `metrics.json` + `weights.json` (the shipped
     `offline_weights.json` format). Coverage caveat: attention / rms_norm / softmax goldens have no case
     builder — counted `out_of_scope` per card, so the goldens-only fit is also kind-limited.
   - (3b, 2026-07-30) modular-seams refactor LANDED (`feature/fit-pipeline-seams`): `search/prior/fit/` split
     into `group.py` (ndarray-backed `Group` dataset representation + `--features` view), `linear.py`
     (trainer+model, owner of the static/dyn split via `TwoStageFit.score_rows`), `rank.py` (rank metrics),
     `cv.py` (folds; trainer plugs in as a `fit_model` callable), `run.py` (pure `run_fit` harness) — the
     seams steps 4–5's freeze data / pairwise loss / CatBoost trainer plug into, zero behavior change
     (byte-identical gate per phase).
   - (4) prefix synthesis + ranking groups → linear×freeze and (5) catboost×freeze — NOT STARTED. Their
     training data is what the 07-2x collection rework (Update 2026-07-23) is gathering.
   Data assembly unions the freeze with
   `Dataset.from_golden()` rows (decision 5 as reversed 2026-07-15): source-marked so folds can exclude them for the
   memorization split, joined into pools by shape (`ShapeKey`/`tile_signature`), grouped with the deployable-regime
   (`H_opt=3`) lane — never the -O1 ranking lane.
   Arch generalization is a first-class output (decision 11): the leave-one-card-out fold directly tests the
   2026-07-09 cross-card failure (analytic TILE regret 3.62× sm_89 → 11.49× sm_120), so report both gate metrics
   **per card, never pooled**; verify tier 1 actually learned an `H_* × knob` interaction (interaction
   strength / SHAP on `H_tc_gen`/`H_cc`) rather than averaging the cards. Decision 11's featurizer additions
   (masked warp aspect, TMA/WSPEC-conditioned tile pricing) land here, each justified by its fold-level metric Δ.
5. **Tier-1 artifact.** Format (feat_ver, cols, digest, fingerprints), the new prior class (contract per constraints),
   loading/quarantine path, one-command refit make target. Not yet the default. Verify: Phase-1 suite; refit-twice
   determinism; version-mismatch quarantine.
6. **Tier-2 additive fallback.** Depth-1 fit stage + declared pair terms (atomic-free × split-width, plus decision
   11's arch×knob seeds: `H_tc_gen × warp-aspect`, `H_tc_gen × splitk-need`, TMA/WSPEC × tile geometry — without
   these the additive table is arch-blind by construction), export into the `analytic_weights.json` artifact
   (`kind` field distinguishes it from the linear weights it replaces), skip-if-missing scorer; drop the linear
   weight sets from the artifact once parity is shown. Verify: suite ≥ linear baseline on goldens **per card** (a
   pooled win that trades one arch against the other repeats today's failure) before removal; locality check (mask
   any feature → bounded, term-local degradation).
7. **Integration + real-hardware A/B.** Wire tier-1 → tier-2 → option-0 into `FallbackPrior` / `load_prior` (greedy +
   PUCT + structural pricing paths), quarantine rules. Verify on a rented card: cold greedy deploy A/B (old vs new
   analytic prior, golden A/B harness), cold-tune search-efficiency A/B (benches-to-best, wall time), full `make test`.
8. **Promotion gate + CI.** Formalize refit → eval → promote-or-keep; the CI checks from decision 9; ARCHITECTURE.md
   updates (the two-clock refit workflow). Verify: break it on purpose in a branch (rename a feature) → CI trips →
   refit target → green.
9. **(Optional, deferrable) Version split.** Split `FEATURIZER_VERSION` into knob-spelling vs feature-encoding axes so
   encoding-only changes stop quarantining node rows / freezes (most changes are encoding-only; raw knob dicts stay
   readable and a refit fully recovers). Touches `features.py`, the DB `feat_ver` stamp, checkpoint + freeze formats.
   The freeze header should reserve both axes from day one (currently equal) so this split needs no format migration.

## Update 2026-07-23 — collection strategy reworked (three-slice golden-anchored sweep); first real fitter, goldens only

Two things moved since 07-16: the data-collection strategy was replaced end to end, and the Phase-4 fitter now
exists — but only its goldens-only cell.

- **The ε-greedy `collect-node-data` tune phase is RETIRED; the ONE collection flow is now a budgeted
  golden-anchored enumeration sweep.** Base collector merged as #414 (07-23, from `feature/golden-neighbor-collect`);
  the three-slice rework + hardening ride the same branch (in review; skill rewritten to the single-phase flow,
  v0.3.0). Mechanism (`remote_node_collect.py` → `golden_neighbor_bench.py` on the box): every shape with a recorded
  golden — matmul AND the non-matmul kinds, which enumerate via the fit's snippet-trace path — builds its candidate
  pool on the live card; the pool splits into three slices sampled at configurable 60/25/15 budget shares:
  `own` (rows within `--max-dist` knob families of the live card's own golden anchors — dense label support where
  its deploy decisions land), `cross` (rows near OTHER cards' anchors that realize here — transfer signal and the
  arch-disagreement rows decision 11's `H_* × knob` interactions train on), `tail` (a deterministic hash-ordered
  subsample of the rest — landscape support so the fit also sees the bad tail). Within a slice the batch draw is
  kind-stratified (the 07-23 4090 run showed matmul, 87% of selectable points, starving attention / softmax /
  pointwise / linear_norm to ZERO benches in 4 h), then shape-proportional to remaining points. Every sampled point
  is benched at BOTH `-Xcicc -O1` and `-O3`, pinned via `emmy run --bench --ab` with the #382 default bench-to-node
  recording (integrity gates for free: `pin_unmatched` / flagged / failed rows go ledger-terminal, never recorded
  clean). A JSON ledger keyed by (gpu, shape, knob signature) makes runs resumable across boxes; the remote driver
  harvests the ledger + node rows even on timeout/failure and backs up the local DB after each merge. Why the
  replacement: search-driven collection over-samples the branches the incumbent prior already likes (coverage
  correlated with the thing being retrained — the 07-15 Update's complaint), and its wall time grew with the golden
  set; a budgeted enumeration sample gives clean leaf rows at fixed cost.
- **What the new collection changes in this plan's economics:** (a) deployable-regime labels by construction —
  every point gets a pinned -O3 measurement, so the item-1 `-O1` lane inversion no longer censors *collection*
  (the tuner-internal per-family re-bench floor stays open but drops in priority; the re-bench gate in
  `policy/mcts.py` is still the global `EMMY_O3_TOL` band today); (b) the item-4 offset dataset falls out for
  free — a point's -O1/-O3 twins share the knob set and join on `op_sig` + tunables; (c) coverage is golden-anchored
  by design, exactly the "one verified point, no neighborhood" censoring item 5 diagnosed, with the `cross` slice
  feeding decision 11's cross-card interaction data and the `tail` slice keeping the fit honest off-anchor;
  (d) the freeze (Phase 3) picks the rows up unchanged — same store, same `freeze_reason` filter.
- **Sweep data status:** first stratified 4090 run 2026-07-23 (it surfaced the kind-starvation fix); the 5090 side
  is still pending — as are the retirement of its poisoned checkpoint (07-13 item 2) and enough merged volume to
  assemble the first freeze-trained cells.
- **The current training pipeline is `emmy fit`, and it trains on GOLDENS ONLY.** #404 landed Phase-4 build steps
  0–3 (see the Phase-4 STATUS): the linear trainer fit on the golden dataset — the incumbent process, now with
  cross-validated held-out reporting (op_family / gpu axes, per-card-only aggregates, zero-seeded fold models) and
  a deterministic per-run metrics file. No measured node/freeze row enters any fit yet; the freeze-trained cells
  (linear×freeze, catboost×freeze — the point of this plan) are the open frontier, waiting on sweep volume.
- **Held-out evaluation clarified: goldens are NOT a held-out set.** Goldens are ordinary training data in every
  cell; held-out integrity is the k-fold mechanism during training — grouped folds (leave-one-op-family-out /
  leave-one-card-out), each golden scored held-out by the fold model that never trained on it — applied the same
  way to the goldens-only assembly (already live in `emmy fit`'s CV harness) and to the goldens+freeze assembly
  when the freeze cells land. The earlier ideas of reserving goldens (pre-07-15) or "the next new model's goldens"
  (07-15) as a standing never-trained acceptance set are both dropped; the affected passages above carry dated
  correction stamps.
- **Freeze format decision:** freeze rows adopt the goldens' declarative spelling — kind + shape fields + verbatim
  tunable knobs + µs with a measured-row extension block (status, variance/n_samples, opt lane, provenance),
  identity captured at collection time via `ShapeKey`, digest/header contract unchanged. Full rationale and scope
  in decision 14; Phase 3's landed v1 format is stamped superseded accordingly.
- **Deploy-surface context keeps narrowing** (extends the 07-15 "#368" note): #396 made deploy picks
  content-tie-deterministic, and #417's golden floor lets a realizable golden decide even prior-less resolves.
  The offline prior's live surface is ever more concentrated on cold UNSEEDED shapes + PUCT steering — weighting
  Phase 4's fold-based generalization gate over seeded-shape golden rank, as the 07-15 Update already argued.

## Update 2026-07-15 — what the saturation arc changed (status of the 07-13 items below, and new inputs to Phase 4+)

A week of landed work (#361 harness, #363 card-faithful eval contexts, #364 de-saturation + tie-pessimistic golden
rank + a linear refit, #368 golden evidence tier at deploy, #369 golden-anchored descent diagnostics + a full 4090
golden sweep) changes this plan's context materially:

- **The historical eval numbers adjudicating refits were wrong twice over.** The offline prior's exp-squash clipped
  quality at ±80, collapsing the whole good-tile region into a tie at `exp(-8)`: greedy fell through to emission
  order (the 12–29x gemma cold misdeploys) while the strictly-greater rank metric reported 0 for every tied row
  ("27/28 top-1" was a plateau artifact). Separately, `eval offline`/`eval online` built golden contexts from the
  host GPU, not the golden's card. Both fixed (#364, #363); every pre-07-14 golden-rank number in this plan's
  history is unreliable. The honest baseline: OLD weights median rank 461/top-100 30 of 122; the #364 refit 79/68.
- **The refit's honest profile, measured on fresh sweep data** (`plans/offline-prior-old-vs-new-on-sweep-data.md`):
  decisive on gemma (median 1422→90) and "everything else" (476→37); the fp16-square regression is real, exactly
  the promotion-time size, and confined — those rows stay this plan's tier-1 acceptance cases. Critically,
  **fork-level steering was a wash** (TILE regret 1.88x vs 1.95x): the linear refit reshaped the flattened-pool
  ranking greedy uses, not the within-fork ordering PUCT uses. Phase 4's grouped per-fork ranking loss is therefore
  not redundant with the refit — it targets the axis the refit provably did not move.
- **The offline prior's deployment surface narrowed (#368).** Recorded goldens now decide cold deploys directly
  (all golden kinds, verified on a 4090 incl. the attention hang-class shapes at 37–44 µs vs 143–158 prior picks),
  and warm deploys are owned by measured evidence. The offline prior's remaining live surface is cold UNSEEDED
  shapes and PUCT steering — which weights Phase 4's fold-based generalization gate even more heavily, and golden
  rank on seeded shapes less.
- **The gemma goldens are SPENT as the held-out unseen-shape set** — the #364 refit trained on them (item 3 below,
  as amended). ~~The Phase-4 gate needs a replacement: the next newly-onboarded model's goldens (record them BEFORE
  any refit sees them), plus the leave-one-op-family-out fold as the standing proxy.~~ *(Corrected 2026-07-23: no
  replacement unseen golden set — the gate is the k-fold holdout during training, on both data assemblies; goldens
  are not a held-out set at all.)* Report memorization vs generalization splits both ways, as decision 13 already
  requires.
- **New diagnostics exist for the gate** (#369): the golden-anchored descent section makes reachability explicit
  (matched fork levels per golden, loud NO TREE DATA absence, per-family divergence, -O3 endpoints) — the
  store-conditioned blind spot that hid the saturation bug is now a rendered metric. Phase 1's metric set should
  treat divergence between the store-conditioned view and the enumeration-wide view as itself reportable.
- **Data status**: one fresh 4090 golden-tune store exists locally (16.7k current-vocabulary rows + online
  checkpoint at `_tune/golden-tune-4090-2026-07-14/`, NOT yet merged into the canonical DB) — but it was a plain
  tune, not ε-greedy, so its coverage correlates with the incumbent priors; the ε-greedy collection sweeps (item 2)
  and the -O3 re-bench family floor (item 1) remain open prerequisites for training data. The 5090 side has no
  post-swizzle data at all and its poisoned checkpoint still needs retiring. Sweep-observed search behavior
  reinforces item 1: matmul bests were found at bench #112–159 of 233–255 — exploration is paying for ranking the
  -O1 lane censors.

## Added 2026-07-13 — data prerequisites from the golden/gemma sweeps, and the -O1→-O3 offset model

Motivated by the 07-12/07-13 sweep findings (the manual 5090/4090 golden sweeps, the post-swizzle 4090 refresh, the
gemma-4 golden seeding): the immediate goal is making the tune-golden skill trustworthy again, and the sweeps showed
the training data feeding this plan is censored and stale in ways the plan didn't yet account for. Ordered items:

1. **-O3 re-bench floor per tile family in the tuner (prerequisite for every retraining step in this plan).**
   *(STATUS 2026-07-23: NOT LANDED in the tuner — the re-bench gate is still the global `EMMY_O3_TOL` band in
   `policy/mcts.py` — but DEMOTED from prerequisite to improvement: the three-slice collection sweep benches every
   sampled point at both opt levels, so training data no longer routes through the tuner's -O1 band at all.)* The
   `-Xcicc -O1` ranking lane systematically inverts the big-register-tile f16-accumulate family ~5× (measured
   directly on the 5090: 2008 vs 392 µs at -O1 for configs ~32% *faster* at -O3), so those configs never land in
   the `EMMY_O3_TOL` band, the node store holds zero measurements in the winning region, and every fit inherits the
   censoring — the fm-lane optima were only found by manual sweeps. Change: grant the deployable -O3 re-bench to
   the top-K per (atom, tile-size family) bucket, not only the global -O1 top band; raise the tuner's per-kernel
   compile budget for the -O3 lane (the big tiles already trip the 12 s cap at -O1). Cost with coarse buckets
   (accumulator type × tile-area band) and K=1–2 is roughly +10–30% collection wall time — acceptable on rented
   collection boxes; before implementing, size K and the bucket definition by replaying the rule against the
   existing autotune DB's -O1 rows (read-only, no GPU) to count the extra re-benches it would have triggered.
2. **Fresh post-swizzle collection sweeps; retire the poisoned checkpoints.** *(STATUS 2026-07-23: the ε-greedy
   sweep as specified here is OBSOLETE — collection is now the three-slice golden-anchored sweep (Update
   2026-07-23), which replaces both this item's mechanism and its bias complaint. First stratified 4090 run done
   07-23; the 5090 sweep and the poisoned-checkpoint retirement remain open.)* *(STATUS 2026-07-15: partial — one
   plain-tune 4090 store exists locally (see the Update section); the ε-greedy sweeps and the 5090 side remain.)*
   The node store has NO post-swizzle
   measurements and the cp.async slab swizzle moved the fm optima (07-12 4090 refresh: "a region no prior trained
   on old data would revisit"); the 5090 sweep checkpoint (`_tune/golden-sweep-5090/prior.json`) is trained on the
   pre-purge fake rewards and must not be reused (already flagged in the assumptions above). After item 1 lands:
   one ε-greedy `collect-node-data` sweep per card (4090 + 5090 at minimum), then discard or predicate-filter the
   poisoned checkpoint. These sweeps produce both the Phase-3 freeze contents and the -O1/-O3 measurement pairs
   item 4 trains on.
3. **Phase-4 gate addition: the gemma-4 goldens are the held-out unseen-shape acceptance set.** *(SUPERSEDED
   2026-07-15: the #364 refit trained on them — spent; ~~the next new model's goldens replace them~~. Corrected
   2026-07-23: no unseen golden acceptance set exists — the gate is the k-fold holdout during training, see the
   07-15 Update bullet as corrected.)* They were recorded
   manually (07-13, 5090 + 4090 files), never trained on, and are the first goldens off the h4096 shape family —
   "rank the gemma entries well from a fit that never saw gemma shapes" is exactly the generalization test whose
   absence let the incumbent process score 27/28 top-1 while cold greedy misdeployed every unseeded gemma shape
   (kv_proj ~770× off, two shapes picking hangs). Do NOT re-run the incumbent `golden_knob_heuristics` refit as a
   stopgap — it was attempted and rejected on 07-12 (top-1 27→20, one shape to rank 3449); the Phase-4 fitter with
   held-out folds is the only sanctioned refit path.
4. **New artifact: the -O1→-O3 offset model.** *(STATUS 2026-07-23: the model is still untouched, but its
   training data now arrives by construction — the collection sweep benches each point at both opt levels with a
   shared knob set, joinable on `op_sig` + tunables, and the pairs are NOT band-censored since the sweep never
   routes through the tuner's -O1 band.)* *(STATUS 2026-07-15: untouched; the 4090 sweep store adds fresh
   -O1/-O3 pairs for the retrodiction test, still band-censored until item 1 lands.)* A small model (same
   featurization, same freeze/fit pipeline — one
   more cell in the Phase-4 fitter matrix) trained on same-config measurement pairs, label = log(-O3/-O1) latency
   ratio. The ratio is dimensionless, so it transfers across shapes and cards, and most absolute-latency variance
   cancels out of it. Go/no-go before any integration work: retrodiction — trained on pre-07-12 pairs only, it
   must place the big-tile f16-accumulate family into the re-bench band; if it fails, run one more item-2 sweep
   and refit before proceeding. Integration points, in payoff order:
   - **The -O3 re-bench gate**: rank candidates for the deployable re-bench by offset-corrected -O1 (measured -O1 ×
     predicted ratio), not raw -O1 — this is where the inversion actually gets fixed. Item 1's unconditional
     per-family floor stays regardless, so the offset never gates its own training data.
   - **PUCT selection**: a multiplicative term on the prior score via the `FallbackPrior` dimensionless-multiplier
     convention (neutral 1.0) — steering only, so a wrong offset costs wasted benches, never wrong data.
   - **Training-label translation**: convert the -O1-rich node labels into estimated deploy-regime labels for
     offline-prior ranking groups, weighted down as model-corrected rather than measured.
   Hard rules: never fold the offset into observed rewards / Q-values — measurements stay ground truth; and the
   offset is the SOLE owner of cross-regime translation (keep the priors regime-conditional via per-`(pool, H_opt)`
   grouping, decision 7) so the learned `H_opt` feature and the offset never double-correct.

5. **Golden-neighborhood sweep collector (added 2026-07-15).** The 07-12 manual `--ab` sweeps found the fm-lane
   optima, but NONE of those ~120 pinned benches reached the node store: the tune engine (`two_level.py` →
   `record_nodes`) is the table's only writer — `run --bench --ab/--golden` results go to the printed table and
   `--json` only. So the region around each golden stays censored in the training data even after the goldens
   themselves join training (decision 5 as reversed) — one verified point, no neighborhood. Fix: a repeatable
   collector that benches each golden's knob neighborhood and records it. Preferred mechanism (investigated
   2026-07-15): a script that (1) enumerates the shape's offer space via `golden_eval.enumerate_graph` (gate pinned
   for fm goldens), (2) filters to rows differing from the golden's `stamp_schedule_families` spelling in ≤ k
   families (k≥2 — the fm headline was a joint atom × geometry move a k=1 star misses), (3) benches pinned at -O3
   via the `_bench_golden_variants` harness (pin-match + intensity-floor + wrong-answer flags for free), and
   (4) writes leaf `NodeRow`s through `record_nodes` with a dedicated `run_id` (freeze picks them up unchanged;
   -O3 labels dodge the item-1 lane inversion by construction). Zero-code alternative for a first campaign:
   per-family pinned-subspace tunes (`EMMY_KNOBS="<all but F pinned>" emmy tune --golden NAME --explore-eps 0.25`
   with `EMMY_NVCC_FLAGS=""`), at higher GPU cost. ~1k benches ≈ a day on one rented card for the full golden set.
   **SETTLED 2026-07-15 — the recorder is `run --bench` itself, DEFAULT-ON behind a quality bar.** Whenever a
   bench meets the tuner's measurement standard (warmup/iters at the tune bench level; an opt-out flag covers the
   rest), `run --bench` records into the canonical node store: every clean pinned golden/`--ab` row as an `ok`
   leaf; a realized config's compile/launch failure as a `bench_fail` negative; and the greedy pick itself via its
   `greedy (isolated)` re-bench (branch `feature/greedy-isolated-rebench` — re-benches the greedy graph emmy-only
   through the same pinned-row worker, so the number is pinned-comparable; the documented ~7% skew was measurement
   position, not config), which makes every benched pool self-anchoring: the prior's argmax gets a measured value
   (the bench-the-argmax anchor for unconditional regret). Never recorded: `pin_unmatched` rows (the claimed
   config never ran; "not offered" is not "doesn't launch"), wrong-answer- or intensity-floor-flagged rows, and
   the whole `--ir` path (serialization drops `op.knobs` — no honest feature dict). Rows are parentless deployable
   -O3 leaves keyed with the tune's own recipes (same pools), `run_id` `bench-<UTC ts>`. Write protection beyond
   the plausibility gate: `record_nodes` gains quality-aware leaf replacement — a materially lower-quality
   measurement (fewer `n_samples`, higher variance) never replaces a stored leaf; newest-wins among comparable
   quality (keep-min stays wrong for leaves: min over re-measurements noise-mines, and a fake-fast row would be
   unrepairable). With this, the neighborhood collector reduces to spec steps (1)–(3) driving
   `run --bench --golden NAME --ab … --json`; step (4) is the default recorder. **LANDED 2026-07-15** (this
   branch; the isolated re-bench merged as #376): `search/bench_record.py` (offer-site pool keying via
   `source_chain` — validated byte-identical against the store's tune-written `op_sig`s; whole-variant leaf per
   site), the `run --bench` default-on wiring (`--no-record-nodes` opts out), and `record_nodes`' quality-aware
   leaf replacement. **GPU-verified 2026-07-16 on a rented CloudRift 4090**, which caught one real bug (fixed +
   regression-tested): the mma tile-lowering preserves no `LoopOp` in `.source`, so the loop-only offer-site
   predicate silently dropped every tensor-core kernel — a golden sweep recorded zero rows; the tile-dialect
   fallback digests to the identical tune `op_sig`. Quality guard, opt-out/quality-bar, freeze pickup, and the
   leaf-only eval degrade all passed on-device. A second on-device pass caught and fixed two more: (a) the split-K
   COMBINE kernel carries no `S_*` provenance anywhere in its chain, so it was silently dropped and the leaf held
   a partial-only value (~14% fast-biased vs the tune's whole-slice leaves in the same pool) — orphan kernels now
   attribute to their nearest sited producer through the graph edges, verified on the 4090 (leaf 8.71 → 10.18 µs
   = the kernel table's sum); (b) the record confirmation was `logger.info`, invisible at `emmy run`'s default
   WARNING verbosity — a default-on DB write must announce itself, so the record-nodes notices print like the
   rest of the bench output. Known deviation: bench rows key under the probe context, which is a DIFFERENT
   `context_key` than the tune's -O3 re-bench context (flag-spelling difference; same `H_opt=3.0`) — grouping per
   `(pool, H_opt)` is unaffected, but a bench row never dedups against its tune -O3 twin; fix would be
   compile-flag canonicalization in `Context.structural_key`, deferred.
   **STATUS 2026-07-23: the collector spec above SHIPPED and then GREW into the whole collection strategy.**
   Steps (1)–(3) landed as `scripts/golden_neighbor_bench.py` driving `run --bench --golden/--ab --json` (base
   collector merged #414); step (4) is the default recorder as settled. It then absorbed the rest of collection:
   the own-neighborhood star became one of three slices (own / cross-card exchange / uniform tail), sampling went
   kind-stratified, runs became ledger-resumable across boxes, and the ε-greedy tune phase it was meant to
   *supplement* was retired outright — full detail in the Update 2026-07-23 section.

The golden A/B harness fixes decided alongside these (bench survives a greedy-row bench_fail; a pin matching no
offered row fails loudly; recorder-side schedule-family stamping; the dynM FLOP-floor overcount) are NOT part of
this plan — they land independently on `feature/golden-ab-harness-fixes`.

## Open questions and assumptions

Assumptions (verify early, Phase 3–4):
- ~~The merged fleet node DB has enough leaf volume/coverage at the current `feat_ver`~~ **VERIFIED FALSE
  (2026-07-06 audit)**: the local store (`~/.cache/deplodock/autotune.db`) holds 23,971 nodes / ~536 multi-child
  forks across RTX 4090 + RTX 5090 — structurally enough — but 100% in the retired pre-tile-IR-rebuild knob
  vocabulary (`SPLIT@a0`, old `REDUCE` codec; pre-enrichment schema → migrates to `feat_ver=1`, fully quarantined
  at `FEATURIZER_VERSION=2`). No `prior.json` checkpoint exists locally either. Consequence: fresh ε-greedy
  collection sweeps are a hard prerequisite for every nodes-based step (Phase 1 baseline regret, Phase 3 freeze,
  Phase 4 training); golden-based metrics are unaffected (live enumeration + committed goldens).
  **RESOLVED 2026-07-09: two ε-greedy `collect-node-data` sweeps landed** — 16,958 `feat_ver=2` rows / 6,036 leaves
  (6,009 ok + 27 `bench_fail`, 39% at `H_opt=3`) across RTX 4090 + RTX 5090, 42–43 op_sigs, 4 context regimes.
  Nodes-based phases are unblocked for these two cards; the old "-O3 rows are scarce" worry is moot (the deployable
  re-bench lane produces plenty). ~~CAVEAT: the 5090 rows include degenerate-fast leaves~~ **ROOT-CAUSED AND FIXED
  (2026-07-10)**: the impossible leaves were pre-#330 split-K variants whose over-budget staged main kernel was
  rejected at materialize — the bench saw only the shared combine kernel and cache-hit its ~9 µs row as the whole
  matmul (`ok`), min-propagated up the ancestries. #330 killed the mechanism; `SearchDB.record_nodes` now gates
  every write on physical plausibility (`implausible_value_reason` — work certified by the stamp identity
  `S_loop_depth == n_free + n_reduce + n_symbolic`, which exactly separates contraction/reduce kinds from
  norm/softmax/attention where free×red overcounts), and `scripts/purge_node_store.py` purged the local store
  (44 leaves + 129 dead branches deleted, 19 branch bounds repaired). Post-purge the node metrics are trustworthy
  on BOTH cards: 5090 TILE fork regret 11.49× → 2.70×, reachability mean 10.96× → 1.73×, calibration +0.59
  (with the 2026-07-09 arch-features refit in place); 4090 TILE 2.24×, reachability mean 1.31×. Phase 3's freeze
  sanity filter can now simply reuse `implausible_value_reason`. A residual small-shape class needed a SECOND
  predicate (`impossible_kernel_reason`): on `square.512.dynM` the combine-only ~2 µs implied a *legal* 133 TFLOP/s,
  so only kernel validity catches it — a cp.async-staged slab over the card's dynamic-smem cap cannot have launched
  (8 more rows purged; the 512² regret denominators are honest now, learned reachability there 11.76× → 2.89×).
  **Follow-ups from the 2026-07-10 clean-data regret run (analytic + both sweep-trained checkpoints):**
  - **The 5090 sweep checkpoint (`_tune/golden-sweep-5090/prior.json`) is itself poisoned** — trained on the fake
    rewards, it still steers into the purged region: TILE fork regret 379.9× (qkv.dynM) / 524.4× (mlp_down.dynM)
    on an otherwise-healthy 1.89× median. Do NOT reuse it; retrain from clean data (next sweep, or filter its
    reservoir with the shared predicates). The 4090 checkpoint shows no spikes (TILE worst 2.73×).
  - **The learned prior's reservoir feed (`record_bench` / `add_rows`) bypasses the node-store gate** — the bug
    mechanism is dead (#330), but the training path has no plausibility defense-in-depth; add one when touching
    the online loop (needs the card identity, which the reservoir rows don't carry today).
  - **The now-real analytic targets, by size:** structural PLACE on small matmuls (5090 128²/64² at 5.5×/4.2× —
    the keep-vs-cut pricing the learned model gets right at 1.03×), WSPEC 2.91×, the TILE/REDUCE K-heavy residuals
    (mlp_down family 4.4–5.3× TILE with 2.5× REDUCE co-misses), and the 5090 512² TILE (7.1×).
  - **Learned cross-card transfer quantified**: own-card TILE 1.48×/1.89× degrades to ~2.0× judging the other card,
    calibration +0.90/+0.77 → ~+0.5 — the decision-11 `H_* × knob` interaction gap, now with clean numbers.
- Known gap inherited from today: the reduce/pointwise tiers enumerate zero rows through the live-fork capture
  (`analytic.py`'s own comment) — path-descent/golden eval coverage for those tiers is limited until that's fixed.
- CatBoost's JSON model dump is stable enough to mechanically collapse a stump ensemble into the tier-2 table.

Open questions (each owned by its phase's detail discussion):
- Phase 3 (settled by the 2026-07-09 rescope): format is trivial and regenerable, no size caps / per-regime budgets
  (keep everything), no staleness policy v1 (the DB keeps newest-per-leaf, re-freezing IS the refresh; codegen drift
  stays on the data-event clock).
- Phase 4: exact ranking loss (YetiRank vs QueryRMSE vs PairLogit — whichever wins, the linear freeze trainer mirrors
  the same pairwise formulation, decision 13), depth-weighting scheme, masking rate p and replica
  count K, monotone-constraint list (features + directions), tier-1 depth/regularization budget. Also golden-shape
  leakage: `collect-node-data` tunes the golden dataset, so the freeze's pools ARE the golden shapes' pools (the
  golden config itself may sit there as an ordinary leaf) and the gate's golden rank partly measures memorization —
  arguably fine for a cold-start prior, but decide deliberately; report golden rank both ways (full-train artifact vs
  the leave-that-op-out fold artifact) so memorization vs generalization stays visible.
- Phase 5: artifact location/format in-repo (base64 JSON vs `.cbm` binary — diff noise vs size), the exp-squash scale /
  normalization for the neutral-1.0 contract, how tier-1 quarantine composes with the learned prior's `trustworthy`
  (three-model precedence in `FallbackPrior` — needs a small design note).
- Phase 6: pair-term whitelist beyond the decision-11 seeds (atomic-free × split-width + the arch×knob terms); bin
  cap per shape function (legibility budget); whether tier 2 should instead select among per-`H_tc_gen` tables (the
  5090 report's "per-arch weight set" recommendation) — equivalent expressiveness, different legibility trade.
- Phase 8: promotion thresholds (golden median rank, per-family sibling regret — hardest on shallow families,
  ablation-collapse tolerance); where the gate runs (CI-only vs also at artifact load).
- Phase 9: do it as part of this effort or defer; low-risk either way now — the freeze header reserves both version
  axes from day one (see phase 9), so deferring costs no format migration.

Needs further discussion / planning (beyond per-phase details):
- Freeze refresh + DB backup cadence and ownership — what triggers a golden re-tune + sweep (tie to the tune-golden
  skill?); where the durable off-laptop copy of the node DB lives (decision 5's durability note).
- Whether the tier-1 artifact should ALSO warm-start the learned prior's reservoir on fresh machines (out of scope
  here, but the freeze makes it possible — flag when designing Phase 4).
- Golden set growth: the acceptance gate is only as good as golden coverage (flash forks, WSPEC, structural splits
  have no goldens today). Decide whether golden expansion is a prerequisite for trusting the gate or a parallel track.
