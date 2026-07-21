# Offline-prior evaluation metrics — proposals deferred from the Phase-4 fitter discussion (2026-07-17)

Status note: the Phase-4 training pipeline ships with a deliberately **minimal** metric set first — dual golden rank
only (see "Adopted now"). Everything else here is a worked-out proposal, deferred until the golden-trained pipeline
cell is running. Context: the fitter's per-run metrics JSON (plan `analytic-prior-catboost-rework.md`, decision 13).

## Adopted now (v1 metrics file)

- **Dual golden rank**, per golden over the full gated enumeration:
  - `rank` (tie-pessimistic, the #364 semantics) — deploy-faithful: greedy's argmin breaks score ties by emission
    order, so a tied-but-earlier row is what a cold deploy ships. This is the gate number.
  - `rank_optimistic` (strictly-greater count, the pre-#364 semantics) — model-quality view: fair when several
    configs are genuinely interchangeable and the prior scores them equal.
  - The **gap** between the two is the tie-plateau width at the golden's score — a score-saturation canary. The
    exp-squash saturation bug hid inside exactly this gap ("27/28 top-1" while cold deploys were 12–29x off);
    reporting both makes that failure mode one glance. Aggregates (median, top-k) reported both ways, per card,
    never pooled; unrankable goldens (signature mismatch) counted loudly, never dropped.

## Deferred proposals

### 1. Golden path rank (enumeration-wide, the PUCT view)

Per golden: compute its path through the CURRENT fork tree; at each fork level enumerate ALL siblings from the live
fork structure (not the store), score them as PUCT scores partial prefixes, record the golden branch's rank (both
tie semantics) + the level's knob family. Scalar summary: sum of ranks = wrong sibling subtrees a best-first descent
prefers ahead of the golden's path (fork-granularity patience budget).

Why: store-independent (no censored choice set, no incumbent bias — fair cross-prior A/B by construction), GPU-free,
measures the descent axis the 07-14 linear refit provably did not move while flattened rank improved. It is the
enumeration-wide sibling of the #369 anchored-descent walk. Limitations: direction only, no cost (a rank-2 level may
cost 0.5% or 500x); only exists where goldens exist (flash/WSPEC/structural have none).

Possible gate restructure (deliberate decision-9 revision, decide when adopting): both gate metrics become
enumeration-anchored — flattened rank (greedy view) + path rank (PUCT view) — and store regret demotes to a reported
diagnostic. Kills the incumbent-favoring bias of a store-based gate structurally; cost: the gate's denominator
becomes golden coverage (sharpens the "golden set growth" open question).

### 2. Fork regret repairs (store-conditioned steering, made honest)

Two documented bias mechanisms in today's `sibling_regret`, both favoring the data-generating prior:
(a) **censored choice set** — fork groups are stored `parent_key` children, so a challenger prior's true preference
may not be an option; the metric silently substitutes its best pick among recorded children, no flag; (b) **loose
value bounds** — a branch's `value_us` is min over benched descendants, so incumbent-neglected branches carry
overstated costs. Repairs:

- Score the fork's FULL enumerable child set; when the prior's unconstrained preference has no measured row, count
  it per family as an **unpriceable pick** (rendered `n_pick_unmeasured`) instead of silently substituting. The
  unpriceable list doubles as the next sweep's bench work-list.
- Confidence-annotate priced forks via `visits` (benched-descendant count) — `n_low_confidence` per family.
- Build eval fork groups by **prefix-synthesizing from freeze leaves** (the training-synthesis machinery) rather
  than stored `parent_key` edges: consistent with the current tree topology (stored edges are path-structure
  fragile — decision 6's own argument), and it picks up the #382 parentless bench leaves the fork diagnostics skip
  today. Goldens enter as one more measured leaf each ("golden-augmented regret" is the special case). Regime
  filter is load-bearing: golden/-O3 values only enter `H_opt=3` groups (the -O1 lane inverts ~5x).
- Serialize the #369 anchored-descent coverage per card next to regret (goldens with no tree data — count + names,
  branch-explored counts, per-family divergence): every way the metric's view is narrower than reality gets a
  rendered number. Silent exclusion of any kind (stale feat_ver, bench_fail, regime) gets a rendered count.

### 3. Neighborhood pricing (measurement-anchored miss cost)

Post-#382, manual golden/`--ab` sweep benches are parentless `ok`/`bench_fail` leaves in the node store (`bench-…`
run_ids) — sweep-designed coverage, NOT incumbent-policy-driven. Join the enumeration against these by
`tile_signature` (-O3 only) and classify every row the prior ranks at-or-above the golden: measured-equal
(harmless plateau) / measured-slower (real miss, priced in µs) / unmeasured (counted). Headline: **predicted deploy
cost** = measured(top pick) / measured(golden) where the pick is measured — the GPU-free stand-in for the Phase-7
hardware A/B. Leakage needs no new mechanism: the op-holdout fold removes an op's neighborhood rows from training,
so the metric reports full-train vs held-out like golden rank already does (sweep rows stay trainable — de-censoring
them was #382's point). Caveats: partial coverage (render measured/unmeasured fractions), cross-run toolchain drift
(prefer same-run pairs, list contributing run_ids).

### 4. Supporting diagnostics in the metrics file

- **Calibration** (median per-op Spearman, non-gate): the guard that caught the 07-09 random-restart overfit
  (golden objective up, calibration +0.46 → +0.18). Cheap canary against golden-objective overfitting.
- Excluded from the file (stay in text reports): leaf reachability (redundant with rank + regret), blame/ablation
  (attribution non-unique — settled in Phase 2), `eval golden` real-compile check (integration test, Phase 7's job;
  post-#368 trivially satisfied on seeded shapes), knobs/variants/failures views.

### Framing: three evidence tiers

Enumeration-anchored (dual ranks, path rank — always available, direction only) → measurement-anchored
(neighborhood pricing, synthesized-group regret — where data exists, cost-denominated) → hardware (Phase-7 A/B —
the final arbiter). Each claim in a findings report should name its tier; each tier's coverage is a rendered number.
