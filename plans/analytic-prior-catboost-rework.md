# Analytic prior rework: CatBoost + additive fallback, trained on a committed measurement snapshot

## Goal

Replace the linear `AnalyticPrior` (`emmy/compiler/pipeline/search/prior/analytic.py`, the `_W_A` / `_W_A_DYN` weight
dicts) with a nonlinear, CatBoost-based cold-start prior that (a) handles feature interactions and per-regime structure
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
- The codebase already made this argument once: `prior/catboost.py`'s docstring on why the linear `BayesianRidgePrior`
  was replaced (monotone-in-every-knob → corner-seeking) applies to the analytic prior too.

## Context for a fresh agent

Read `emmy/compiler/pipeline/ARCHITECTURE.md` (sections: "Forks and the one ranking path", "Learned prior",
"Featurizer vocabulary versioning", the `SearchDB` / node-table paragraphs) before touching anything. Key modules:

- `search/prior/analytic.py` — the linear prior being replaced (weights + the hardcoded interaction).
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
   exported at fit time to a plain per-feature table (a source literal like `_W_A` today; CatBoost JSON dump → collapse
   stumps). Scoring is ~20 lines of `shape_k(feats[k]) if k in feats else 0`. Each shape function reads as a fitted rule
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
5. **Training data = a committed, curated, LEAF-ONLY measurement snapshot; goldens are the held-out acceptance set.**
   - Why committed: the fit must be a pure function of the repo (reproducible on a clone, verifiable in CI, refittable
     without a GPU fleet). The private `~/.cache` node DB gives none of that.
   - Why leaf-only: a leaf is a durable fact about (config, GPU, compiler version). A branch row is a run-artifact — a
     policy-dependent coverage bound (min over whatever that search benched), possibly stale/non-monotone vs its own
     re-measured leaves, and **path-structure-fragile** (its identity is a prefix in the *historical* fork-tree topology;
     level reorders / knob moves orphan it, and no spelling migration fixes that). Leaves are complete points in knob
     space — valid under any tree organization.
   - Why goldens never train: they are the only *verified-optimum* labels (tune-golden A/B + integrity gates). Training
     on them makes the promotion gate measure memorization. Their value is spent as the eval set.
   - Snapshot contents: ok leaves + **fail leaves** (negative examples — "doesn't build/launch here" is durable) +
     **-O3 regime rows** (scarce, most deploy-relevant; keep all). Raw knob dicts + shape/context/gpu + bench stats +
     provenance (measured_at, run_id, compiler commit, collection policy). No `parent_key`, no `visits`, no tree schema.
6. **Branch training rows are SYNTHESIZED at fit time, not stored.** Group snapshot leaves by prefix under the
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
    *Code events* (frequent, GPU-free): featurizer/encoding change → measurements untouched, re-featurize snapshot,
    one-command refit; search-space growth → old configs remain valid at the new knob's OFF value, refit immediately as
    a bridge (model neutral on the new dimension via NaN), then schedule a collection sweep for coverage.
    *Data events* (rare, GPU-spend): codegen drift — detected by the golden gate degrading against freshly re-tuned
    goldens — or new hardware → collection sweep → curation → snapshot refresh PR → refit. The learned prior stays the
    *online* model (freshest local data); the analytic prior is a *release artifact* on the repo clock.

## Constraints and requirements

- **ONE featurization**: everything (both tiers, synthesis, evals) reads `features.knob_features`. No private feature
  views (that's what killed the old rule prior).
- **ONE ranking path**: both tiers implement the `Prior` ABC and compose behind `FallbackPrior`. No policy special-cases.
- Analytic contract: lower-is-better latency *proxy*, ordinal only, neutral exactly 1.0 when it has no opinion, usable
  with zero measurements on a fresh clone, cheap enough for greedy's ~1k-row flattened batches (vectorize tier 1's
  predict like `CatBoostPrior.mean_scores`).
- Tier-1 artifact: repo-checked, stamped `feat_ver` + training-data digest + fingerprints, discarded WHOLE on version
  mismatch (the `CatBoostPrior.from_json` semantics), deterministic refit (seeded).
- Snapshot: leaf-only, raw-knob spelling, per-card files, provenance headers, coverage-first curation (every sibling
  group at depths 1–2 keeps ≥ m measured descendants incl. slow ones; stratified sample below), size kept PR-reviewable.
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
- The `node` table schema semantics (per-kind upsert, newest-leaf-wins, `merge_nodes`) — the snapshot exporter is a
  read-only consumer. The `collect-node-data` flow keeps working unchanged (ε-greedy 0.25 collection stays; it's the
  provenance the snapshot prefers).
- Golden dataset role/format and `tile_signature` matching; the golden A/B integrity gates.
- Greedy mechanics: `flatten_leaves` (complete-leaf scoring), validity blocklist retries, option-0 last resort.
- Existing `emmy eval` subcommands keep working (new evals are additive).

## Phases

Each phase is independently verifiable; 1–4 need no GPU (except the optional collection leg in 3); hardware spend
concentrates in 7. Dependencies: 1 → {2, 3}; 3 → 4 → {5, 6} → 7 → 8; 9 floats. Phase 2 is diagnostic tooling —
parallel with 3, consumed by 4's fitter loop and 8's gate.

1. **Evaluation harness + incumbent baseline.** Exactly the two gate metrics (decision 9): flattened golden rank
   (exists — `eval analytic`) and family-bucketed fork sibling value regret (rework `node_sibling_ranking`:
   regret instead of top-1/Spearman, bucket by the fork delta's knob family). NO comparison runner — comparing
   priors = diffing two report files (the golden-sweep findings already track progress across dates this way);
   the Phase-4 fitter emits the same report per CV fold. May add a `Prior`-internal score-on-features seam
   (both priors already featurize-then-predict) — needed by the Phase-2 attribution tooling and the Phase-4 fitter; no
   behavior change, equivalence-tested. **Prerequisite: one fresh ε-greedy `collect-node-data` sweep on a rented
   card** — the 2026-07-06 audit found the local node store 100% retired-vocabulary (see assumptions), so no
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
   ~89% of total regret-weight; `D_l2_reuse` actively misleading on TILE forks, Δ −0.56x; 21 weighted features
   have zero fork support in the store).
3. **Committed measurement snapshot.** Exporter (fleet node DB → repo files, leaf-only, curated, provenance), loader
   (re-featurizes through live code), format doc. Verify: coverage report; round-trip test vs the live DB path; loads
   in CI with no DB; optionally one fresh `collect-node-data` → merge → re-export to prove the refresh flow.
4. **Training pipeline.** Prefix-row synthesis under the current fork structure, sibling-group ranking datasets,
   pool-level masking augmentation, group-holdout k-fold (leave-one-op-out AND leave-one-card-out via
   `Dataset.fold_node_rows`), candidate CatBoost rankers with monotone constraints; every fold reported through the
   Phase-1 suite. Deliverable: the fitter (successor to `golden_knob_heuristics.py`) + CV report vs the linear
   baseline on identical folds. **This is where old-vs-new comparison lives.** Offline only.
5. **Tier-1 artifact.** Format (feat_ver, cols, digest, fingerprints), the new prior class (contract per constraints),
   loading/quarantine path, one-command refit make target. Not yet the default. Verify: Phase-1 suite; refit-twice
   determinism; version-mismatch quarantine.
6. **Tier-2 additive fallback.** Depth-1 fit stage + declared pair terms (atomic-free × split-width first), export to
   source-literal table, skip-if-missing scorer; delete `_W_A` / `_W_A_DYN` once parity is shown. Verify: suite ≥
   linear baseline on goldens before removal; locality check (mask any feature → bounded, term-local degradation).
7. **Integration + real-hardware A/B.** Wire tier-1 → tier-2 → option-0 into `FallbackPrior` / `load_prior` (greedy +
   PUCT + structural pricing paths), quarantine rules. Verify on a rented card: cold greedy deploy A/B (old vs new
   analytic prior, golden A/B harness), cold-tune search-efficiency A/B (benches-to-best, wall time), full `make test`.
8. **Promotion gate + CI.** Formalize refit → eval → promote-or-keep; the CI checks from decision 9; ARCHITECTURE.md
   updates (the two-clock refit workflow). Verify: break it on purpose in a branch (rename a feature) → CI trips →
   refit target → green.
9. **(Optional, deferrable) Version split.** Split `FEATURIZER_VERSION` into knob-spelling vs feature-encoding axes so
   encoding-only changes stop quarantining node rows / snapshots (most changes are encoding-only; raw knob dicts stay
   readable and a refit fully recovers). Touches `features.py`, the DB `feat_ver` stamp, checkpoint + snapshot formats.

## Open questions and assumptions

Assumptions (verify early, Phase 3–4):
- ~~The merged fleet node DB has enough leaf volume/coverage at the current `feat_ver`~~ **VERIFIED FALSE
  (2026-07-06 audit)**: the local store (`~/.cache/deplodock/autotune.db`) holds 23,971 nodes / ~536 multi-child
  forks across RTX 4090 + RTX 5090 — structurally enough — but 100% in the retired pre-tile-IR-rebuild knob
  vocabulary (`SPLIT@a0`, old `REDUCE` codec; pre-enrichment schema → migrates to `feat_ver=1`, fully quarantined
  at `FEATURIZER_VERSION=2`). No `prior.json` checkpoint exists locally either. Consequence: fresh ε-greedy
  collection sweeps are a hard prerequisite for every nodes-based step (Phase 1 baseline regret, Phase 3 snapshot,
  Phase 4 training); golden-based metrics are unaffected (live enumeration + committed goldens).
- Known gap inherited from today: the reduce/pointwise tiers enumerate zero rows through the live-fork capture
  (`analytic.py`'s own comment) — path-descent/golden eval coverage for those tiers is limited until that's fixed.
- CatBoost's JSON model dump is stable enough to mechanically collapse a stump ensemble into the tier-2 table.

Open questions (each owned by its phase's detail discussion):
- Phase 3: snapshot file format (YAML like goldens vs JSONL), per-card file layout, size cap / row budget, staleness
  policy (age-out vs re-measure), whether -O1 rows get a row budget separate from -O3.
- Phase 4: exact ranking loss (YetiRank vs QueryRMSE vs PairLogit), depth-weighting scheme, masking rate p and replica
  count K, monotone-constraint list (features + directions), tier-1 depth/regularization budget.
- Phase 5: artifact location/format in-repo (base64 JSON vs `.cbm` binary — diff noise vs size), the exp-squash scale /
  normalization for the neutral-1.0 contract, how tier-1 quarantine composes with the learned prior's `trustworthy`
  (three-model precedence in `FallbackPrior` — needs a small design note).
- Phase 6: pair-term whitelist beyond atomic-free × split-width; bin cap per shape function (legibility budget).
- Phase 8: promotion thresholds (golden median rank, per-family sibling regret — hardest on shallow families,
  ablation-collapse tolerance); where the gate runs (CI-only vs also at artifact load).
- Phase 9: do it as part of this effort or defer; interaction with the snapshot format if deferred.

Needs further discussion / planning (beyond per-phase details):
- Snapshot refresh cadence and ownership — what triggers a re-tune of goldens + sweep (tie to the tune-golden skill?).
- Whether the tier-1 artifact should ALSO warm-start the learned prior's reservoir on fresh machines (out of scope
  here, but the snapshot makes it possible — flag when designing Phase 4).
- Golden set growth: the acceptance gate is only as good as golden coverage (flash forks, WSPEC, structural splits
  have no goldens today). Decide whether golden expansion is a prerequisite for trusting the gate or a parallel track.
