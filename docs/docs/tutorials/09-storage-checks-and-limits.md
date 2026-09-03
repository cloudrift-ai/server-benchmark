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
- Those stored rows are the reservoir. Discarding them therefore also deletes the reservoir's share of the
  [measured evidence](./06-deploy-evidence-hierarchy.md), and disables the structural cost estimate, which needs a
  trusted online model.
- The machine's deployments silently drop to the golden and database rows, then the offline prior — **with no
  warning at compile time** unless `--strict-evidence` is on. It behaves like a machine that has never been tuned,
  until it is tuned again.
- The measurements table survives, because it is keyed by content rather than by feature names.

A related rule protects the same evidence from a much smaller change. Matching a candidate to measured rows deliberately
tolerates a changed feature set: a candidate's stamped features may include things the stored rows predate, and a join
demanding exact equality of the whole set would let a single added feature switch off the entire evidence tier against
every existing database at once. That happened, which is why the rule is what it is.

## Finding out where the prior is wrong

`emmy eval` exists for the question "the prior chose badly — where?". Two views matter.

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

**What a wrong choice cost.** Over configurations that were all actually measured, this reports two things per card
and compiler setting: how closely the model's ordering follows the hardware's, and how much slower its best guess is
than the fastest configuration in the set. A ratio of 1.00 means the model's pick *is* the best available. This is
the view that tracks deployed speed, and it is why the ranking view above is only a screen — a rank says where a good
configuration landed in the ordering, never what missing it costs.

Both views are rendered **once per half of the prior, each labeled**. The composite would answer with whichever half
is currently active, and the two halves' results point at different fixes — the shipped weights versus the local
training data — so an unlabeled number would destroy the diagnostic.

Every figure carries the count of comparison sets behind it. The sets have different minimum sizes — a correlation
needs more members than a ratio does — and the ones too small for a given figure are excluded rather than averaged
in, so the count is what tells you how much of the data a number actually covers.

## Limitations

Gathered in one place, honestly.

1. **The calibration check is lenient on a small tuning run.** A model can be fitted on as few as 50 rows, and if all
   its operation groups are too small to compute a correlation, calibration cannot be measured — and an unmeasurable
   calibration passes. Such a model owns deployments and structural decisions on very little data.
2. **Ranking-setting measurements are known to invert against the deployable setting.** The evidence index is
   keyed by the compile's regime, so a sweep at the ranking setting is never read by a deploy — but such a machine
   deploys on the prior for those kernels, and the prior can be wrong about the ordering it is used for.
3. **Raising the feature version silently removes the reservoir's evidence.** As described above: no warning at
   compile time, and the only symptom is that deployments get worse.
4. **A fresh machine deploys mostly on prediction.** Only golden configurations travel with a clone. Where no golden
   covers a shape, a rented box is choosing from a model that has never seen that card's measurements.
5. **A cold compile changes which kernels exist only on evidence.** Structural choices need a trusted online model
   to be costed, so on the offline prior the current kernel set is kept — even where splitting would be much faster.
   The 1.8-times-faster split on [the goldens page](./07-golden-configurations.md) is only deployable because
   somebody recorded it, and that measured route row is what deploys it.
6. **A recording that no longer realizes is simply not evidence**, and the kernel falls to the prior, which can be
   far slower than the number the recording advertises. `--strict-evidence` makes that fall-through an error, and
   the release gate compiles the serving matrix under it; a plain deploy without the flag falls through silently.
7. **There is no per-fork report of which row decided.** Answering "which evidence answered this fork, and did I
   expect that one?" means correlating warnings, the resolution record and the release gate.
8. **The richest measurements are diagnostic-only.** The search-tree table is never consulted when deploying. Fitting
   the offline prior on a frozen snapshot of it is a planned path, not a current one — today `emmy fit` trains on the
   golden configurations only.
9. **Nothing evaluates a fork the search never descended into.** Both views score configurations that were built
   and offered as candidates. A search decides one fork at a time, and a fork it never took leaves no row
   anywhere — a good configuration sitting past one is silence that reads as health.

## See it yourself

The measured view needs a tuning database, or a frozen snapshot in its place:

```bash
emmy eval prior --dataset nodes
```

And the two halves can be compared against candidate artifacts without touching the installed ones, which is how two
fits are judged against each other, with `--json` writing the report in the same shape a fit records:

```bash
emmy eval prior --offline-file /tmp/candidate-weights.json --json /tmp/candidate.json
emmy eval prior --online-file /tmp/candidate-checkpoint.json --json /tmp/incumbent.json
```

## Where to go next

You have now seen the whole path: a model becomes a graph, the graph is rewritten pass by pass, a handful of those
rewrites offer several correct answers, and each of those is settled by the best evidence available — a reviewed
measurement if one exists, a local measurement if one was taken, a prediction otherwise, and a safe default when there
is nothing at all.

For the level of detail below this series, the reference documents live beside the code: `ARCHITECTURE.md` in
`emmy/compiler/pipeline/` for the pipeline itself, and its `passes/` sibling for the rewrite rules. The vocabulary is
defined in `GLOSSARY.md` at the root of the repository.
