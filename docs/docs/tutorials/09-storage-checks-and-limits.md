---
sidebar_position: 9
title: "9. Storage, Checks and Limits"
description: How measurements are keyed and stored, how the prior is evaluated when it chooses badly, and where the whole design falls short.
keywords: [Emmy, tuning database, structural key, eval, diagnostics, limitations, prior]
---

# 9. Storage, Checks and Limits

The last page. Three subjects, in increasing order of usefulness to somebody deciding whether to trust any of this:
how measurements are stored, how the prior is examined when it chooses badly, and what the design does not do.

## Two identities, and only two

Everything Emmy stores or replays is keyed by one of two identities. When adding a cache or a table, pick one of them
rather than inventing a third.

**Variant identity — the compile context plus the knob values.** Used by anything that *predicts* or *replays*. This
works as a complete identity because of the stamping pass: the structural facts about the operation — the counts in
its body, its loop extents, its data types — are already part of the row, so the merged set of values fully describes
what is being predicted. That is what lets the prior be a pure function of it, with no need to look at a graph.

**Measurement identity — the context plus the kernel's structural key.** Ground truth about kernels that were actually
built: their measured times, the inventory of operations encountered, and the deduplication that collapses 24
identical RMSNorm kernels into one unit of work.

The database holds an inventory of every operation seen along any lowering path, one row per rewrite step, the
measurements table, and the search-tree table. The last one is worth a closer look, because it is the richest and the
most misunderstood.

## The search-tree table

One row per position in a tuning search — every partly decided branch and every complete configuration. Each row
carries the full feature row the prior sees, a time for that position, a pointer to its parent, and the GPU it was
measured on.

Branch rows and leaf rows are updated by different rules, and the asymmetry is deliberate:

- **A branch keeps the minimum.** Its time is a bound over everything below it, and a faster descendant genuinely
  tightens that bound.
- **A leaf takes the newest measurement.** A leaf is a re-measurement of one single configuration, and taking the
  minimum of several noisy medians would drift steadily toward the noise floor rather than toward the truth.

The GPU's name is part of the key. Compute capability alone cannot separate two cards built on the same die — an H100
and an H200 share it, and their SM counts — so without the name their rows would merge and one card's data would
silently overwrite the other's.

Rows also record how much to trust their own label: how many measured configurations the label rests on, whether it is
a real measurement or a bound, the measurement's own spread and sample count, and whether the configuration failed to
run at all. **Failures are kept**, with the watchdog's placeholder time, because a search model needs negative
examples. A working row is never downgraded by a later failure.

**Data measured elsewhere can be merged in.** Another machine's database can be read and re-inserted through the same
update rules. The result does not depend on which database is merged into which: a stale leaf never comes back to
life, and the trust counts add up when two rows share a key. This is how measurements collected on a rented card fold
into one canonical local database.

**A frozen snapshot makes a fit reproducible.** The search-tree table is a live store — tuning runs and merges keep
writing into it — so a model fitted straight from it cannot be reproduced later. A freeze is a snapshot written as a
directory of YAML files laid out like the golden configuration files, beside a manifest of content digests. Nothing is
stored in feature form: the hardware description is rebuilt for the recorded card, and the structural counts are
rebuilt by re-tracing the shape's own program. So a change to how features are encoded never invalidates a freeze.
Loading is strict — a missing file, a foreign manifest or a digest mismatch is an error — and freezing the same
database twice produces byte-identical digests.

**Hand-run measurements are recorded too.** A `run --bench` that measured configurations with knob values forced by
hand records each clean result as a row, so that manually found optima are not lost when the session ends. It
is guarded: a newer measurement that is unambiguously worse — fewer samples *and* more spread — never displaces a
stored one, so a casual measurement cannot overwrite tuning-grade data, while an honest re-measurement still repairs a
stale row. Rows that were flagged by any of the integrity checks are never recorded at all.

## The version stamp, and what raising it costs

Every stored training artifact carries the version of the feature encoding it was written under. Raising that version
is the correct response to any incompatible change in how knobs are named or features are encoded — old artifacts age
out instead of poisoning the model with rows whose names no longer mean anything.

It is worth being explicit about how far the consequence reaches, because it is not obvious:

- **The prior's checkpoint from another version is discarded whole** — the model *and* the stored rows.
- Those stored rows are the reservoir. Discarding them therefore also deletes [tier 2 of the evidence
  hierarchy](./06-deploy-evidence-hierarchy.md), and disables the structural cost estimate, which needs a trusted online
  model.
- The machine's deployments silently drop to golden configurations, then database rows (usually ranking-setting ones),
  then the offline prior — **with no warning at compile time**. It behaves like a machine that has never been tuned,
  until it is tuned again.
- The measurements table survives, because it is keyed by content rather than by feature names.

A related rule protects the same tier from a much smaller change. Matching a candidate to measured rows deliberately
tolerates a changed feature set: a candidate's stamped features may include things the stored rows predate, and a join
demanding exact equality of the whole set would let a single added feature switch off the entire evidence tier against
every existing database at once. That happened, which is why the rule is what it is.

## Finding out where the prior is wrong

`emmy eval` exists for the question "the prior chose badly — where?". Four views matter.

**Where a golden ranks, with ties counted against it.** For each recorded golden configuration, the view reports how
many candidates the prior scored better. A tie counts as a loss, because when scores are equal the compile takes
whichever came first, which is not the golden. Counting only strictly-better candidates would report a perfect rank
for every configuration sitting inside a plateau of equal scores — and that is precisely how a saturated model scored
top-of-the-list on the goldens while real cold deployments missed by 12 to 29 times. Both counts are reported
side by side, and the gap between them is the width of the tie plateau: an early warning that the scores are
saturating.

These evaluations rebuild each golden's compile context for **the GPU the golden was recorded on**, never the machine
running the evaluation. Building them for the host makes the ranks machine-dependent, since the geometry features
would then be describing tiles for the wrong card.

**Per-fork comparison of the choice against the truth.** Using the search-tree table, this groups rows by their
parent and, for each fork, divides the measured time of the child the prior would pick by the measured time of the
child that actually measured best. A ratio of 1.00 means the prior steers into the best subtree available from that
point. Ties count against the prior, for the same reason as above. Results are grouped per GPU and split by
optimization level — two cards off the same die share a structure signature but not their latencies, and the two
compiler settings must never be pooled — and bucketed by which knob family the fork decides, which is the stable way
to name a level of the tree.

The whole block is rendered **once per half of the prior, each labeled**. The composite would answer with whichever
half is currently active, and the two halves' results point at different fixes — the shipped weights versus the local
training data — so an unlabeled number would destroy the diagnostic.

**What the per-fork view structurally cannot see.** It only speaks about forks the search actually measured. A golden
sitting in a subtree the search never built, or a shape with no search data at all, is silence that reads as health —
which is exactly how the saturation failure above hid. So each GPU's report ends with one row per golden recorded for
that card: how far the golden's path is covered by the explored tree, whether the prior's choice stays inside the
golden's subtree at each fork, and the loud absences. Coverage is always printed with a denominator, and where the
golden's own branch was never built the total is marked as an estimate rather than passed off as exact.

**Which feature is to blame.** Two views work at the level of individual features. The first breaks the score
difference between the chosen candidate and the one that measured best into one term per feature. A fork the prior got
wrong where *no* term separates the two is reported as `BLIND` — that is a gap in the features, not a problem with the
weights, and no amount of refitting will fix it. The second hides one feature at a time and re-picks every fork, and
reports how the results change.

Both are **diagnostics, never gate metrics**. Attributing an effect among correlated features has no unique answer:
hiding any one of a redundant block of geometry features costs about the same.

## Limitations

Gathered in one place, honestly.

1. **The calibration check is lenient on a small tuning run.** A model can be fitted on as few as 50 rows, and if all
   its operation groups are too small to compute a correlation, calibration cannot be measured — and an unmeasurable
   calibration passes. Such a model owns deployments and structural decisions on very little data.
2. **Ranking-setting measurements are known to invert against the deployable setting.** The hierarchy is careful to
   prefer deployable-setting evidence, but on a machine tuned only in the ordinary way, tier 3 is made of
   ranking-setting rows, and they can be wrong about the ordering they are being used for.
3. **Raising the feature version silently removes an evidence tier.** As described above: no warning at compile time,
   and the only symptom is that deployments get worse.
4. **A fresh machine deploys mostly on prediction.** Only golden configurations travel with a clone. Where no golden
   covers a shape, a rented box is choosing from a model that has never seen that card's measurements.
5. **A cold compile never changes which kernels exist.** Structural choices need a trusted online model to be costed,
   so on the offline prior the current kernel set is kept — even where splitting would be much faster. The
   1.8-times-faster split on [the goldens page](./07-golden-configurations.md) is only deployable because somebody
   recorded it.
6. **A recording that no longer realizes falls through**, to tiers that can be far slower than the number the
   recording advertises. The audits catch it, but only when they are run — and the isolated audit cannot see the
   pin-only case that appears solely inside a real model.
7. **There is no per-fork report of which tier decided.** Answering "which tier answered this fork, and did I expect
   that one?" means correlating warnings, the resolution record and the audits.
8. **The richest measurements are diagnostic-only.** The search-tree table is never consulted when deploying. Fitting
   the offline prior on a frozen snapshot of it is a planned path, not a current one — today `emmy fit` trains on the
   golden configurations only.
9. **Feature attribution has no unique answer** among correlated features, so the blame and ablation views can point
   at a plausible feature rather than the responsible one.

## See it yourself

The per-fork views need a tuning database, or a frozen snapshot in its place:

```bash
emmy eval online --dataset nodes
emmy eval online --dataset nodes --blame
emmy eval online --dataset nodes --ablate
```

And the two halves can be compared against candidate artifacts without touching the installed ones, which is how two
fits are judged against each other:

```bash
emmy eval offline --offline-file /tmp/candidate-weights.json
emmy eval online --online-file /tmp/candidate-checkpoint.json
```

## Where to go next

You have now seen the whole path: a model becomes a graph, the graph is rewritten pass by pass, a handful of those
rewrites offer several correct answers, and each of those is settled by the best evidence available — a reviewed
measurement if one exists, a local measurement if one was taken, a prediction otherwise, and a safe default when there
is nothing at all.

For the level of detail below this series, the reference documents live beside the code: `ARCHITECTURE.md` in
`emmy/compiler/pipeline/` for the pipeline itself, its `passes/` sibling for the rewrite rules, and `HISTORY.md` in
the same directory for the full stories behind the incidents this series mentions in passing. The vocabulary is
defined in `GLOSSARY.md` at the root of the repository.
