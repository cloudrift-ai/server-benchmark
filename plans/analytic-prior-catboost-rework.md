# Analytic prior rework: CatBoost + additive fallback, trained on a frozen local measurement snapshot

## Goal

Replace the linear `AnalyticPrior` (`emmy/compiler/pipeline/search/prior/analytic.py`; its weight sets live in the
repo-checked `analytic_weights.json` since 2026-07-10, `_W_A` / `_W_A_DYN` in older notes below) with a nonlinear,
CatBoost-based cold-start prior that (a) handles feature interactions and per-regime structure
the linear form cannot, and (b) is **at least as resilient** to code / feature / search-space changes as the current
setup. "Analytic prior" here means the *shipped, cold-start* ranking used when the learned online `CatBoostPrior` is
absent, cold, or quarantined — it must work on a fresh clone with zero measurements.

Success criteria: on held-out folds and on the (never-trained-on) goldens, the new prior beats the linear baseline on
golden rank, per-depth fork sibling-ranking, and golden path-descent; a real-hardware A/B (cold greedy deploy + cold
tune efficiency) confirms it; and a breaking featurizer change costs one GPU-free refit command, not a data campaign.

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
5. **Training data = a frozen, LEAF-ONLY local extract of the fleet node DB; goldens are the held-out acceptance
   set.** RESCOPED 2026-07-09 — was "a committed, curated snapshot". With no second consumer of the dataset the
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
   - Why goldens never train: they are the only *verified-optimum* labels (tune-golden A/B + integrity gates). Training
     on them makes the promotion gate measure memorization. Their value is spent as the eval set.
   - Freeze contents: ok leaves + **fail leaves** (negative examples — "doesn't build/launch here" is durable) + all
     regime rows (-O1 and -O3 both; no size budget — keep EVERY leaf passing the sanity filter). Raw knob dicts +
     shape/context/gpu + bench stats + provenance (measured_at, run_id; the freeze header stamps the repo commit and a
     collection-policy note at freeze time — per-row commit/policy aren't recorded in the DB). No `parent_key`, no
     `visits`, no tree schema.
   - Durability: local-only means `~/.cache/emmy/autotune.db` is the SOLE copy of data that cost rented-GPU money (a
     `.bak-pre-wipe` neighbor shows cache wipes happen). After each `collect-node-data` merge, copy the DB (or the
     freeze) somewhere durable; re-collection is possible but costs rental hours per card.
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
- Code-event refits must need no GPU. Goldens must never enter training.
- Repo conventions: knobs declared only in `search/space.py`; `pipeline/` never imports `backend/`; markdown wrapped
  ~120 chars; plans are ephemeral (this file gets deleted when the work lands — durable content goes to ARCHITECTURE.md).

## Do NOT change

- The `Prior` ABC surface (`score` / `mean_score` / `mean_scores` / `pick` / `trustworthy` / `evidence_pick`) and
  `FallbackPrior`'s blend semantics (learned µs × `analytic**W`, neutral 1.0; evidence-pick precedence).
- The learned `CatBoostPrior`'s online loop: reservoir sampling, `REFIT_SCHEDULE`, checkpoint format, calibration gate.
  Different clock, different job — this plan only touches the *analytic* half of the composition.
- The knob-stamp invariant and OFF-value / NaN semantics (`tests/compiler/passes/test_knob_stamp_invariant.py`).
- The `node` table schema semantics (per-kind upsert, newest-leaf-wins, `merge_nodes`) — the freeze step is a
  read-only consumer. The `collect-node-data` flow keeps working unchanged (ε-greedy 0.25 collection stays; it's the
  provenance the freeze prefers).
- Golden dataset role/format and `tile_signature` matching; the golden A/B integrity gates.
- Greedy mechanics: `flatten_leaves` (complete-leaf scoring), validity blocklist retries, option-0 last resort.
- Existing `emmy eval` subcommands keep working (new evals are additive).

## Phases

Each phase is independently verifiable; 1–4 need no GPU; hardware spend concentrates in 7. Dependencies: 1 → {2, 3};
3 → 4 → {5, 6} → 7 → 8; 9 floats. Phase 3 shrank in the 2026-07-09 rescope to a thin freeze step — effectively step 0
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
4. **Training pipeline.** Prefix-row synthesis under the current fork structure, sibling-group ranking datasets,
   pool-level masking augmentation, group-holdout k-fold (leave-one-op-out AND leave-one-card-out via
   `Dataset.fold_node_rows`), candidate CatBoost rankers with monotone constraints; every fold reported through the
   Phase-1 suite. Deliverable: the fitter (successor to `golden_knob_heuristics.py`) + CV report vs the linear
   baseline on identical folds. **This is where old-vs-new comparison lives.** Offline only.
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
- Phase 4: exact ranking loss (YetiRank vs QueryRMSE vs PairLogit), depth-weighting scheme, masking rate p and replica
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
