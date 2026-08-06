---
sidebar_position: 6
title: "6. The Deploy Evidence Hierarchy"
description: The fixed order an ordinary compile works down to answer a fork — reviewed measurements, then local measurements, then prediction, then the rule's own first option.
keywords: [Emmy, deploy evidence hierarchy, evidence, greedy selection, golden configuration, prior, fork]
---

# 6. The Deploy Evidence Hierarchy

This is the page the series has been building towards. An ordinary compile answers every fork immediately, without
running anything, by working down a fixed order — **best evidence first**. That order has a name, the **deploy
evidence hierarchy**, and each step in it is called a **tier**.

## The order

1. **Golden configurations recorded for this GPU.** Reviewed measurements that ship with the repository. The compile
   takes the first offered option that agrees with the fastest recorded entry for this kernel's shape.
2. **Measured evidence from the reservoir.** The option that agrees with the fastest measurement of this same kernel
   that was itself taken at the deployable setting.
3. **Measured rows from the tuning database.** Measurements of this exact kernel. Within the tier, a row measured at
   the deployable setting decides outright; a row measured at the ranking setting decides only when no option has a
   deployable-setting measurement.
4. **The prior's prediction.** Only when no option has any measurement behind it at all: the option with the lowest
   predicted latency.
5. **The rule's own first option**, if there is no prior to ask. Rules order their options so that the first one is
   always safe.

The shape of that list is the whole design in miniature. A reviewed measurement beats a local one; any measurement of
this exact kernel beats a prediction; a prediction beats an arbitrary choice. Nothing further down ever overrules
something further up.

## How one fork is actually decided

Take the example kernel — the fused RMSNorm and query/key/value linear layer — on a machine with a tuned checkpoint.
The tile-lowering rule matches it and returns its options.

1. The engine hands the fork to the greedy chooser.
2. **The fork is flattened to complete leaves.** As [the forks page](./03-forks-and-knobs.md) explained, the tree is
   built level by level for the benefit of a search; greedy selection instead expands it to complete configurations
   first, because a half-decided option has no area and no memory footprint to score. Flattening produces knob values
   only — still no kernel is built.
3. **Each leaf becomes one row**: the hardware and regime this compile is running under, the summary of the kernel's
   body and extents that the stamping pass wrote onto it, and the leaf's complete knob values.
4. **Tier 1.** The kernel's shape is looked up among the golden configurations recorded for this GPU. If a shape
   matches, the fastest of its entries is taken and the first leaf agreeing with it wins.
5. **Tier 2.** Otherwise, the fastest reservoir measurement of this same kernel that was taken at the deployable
   setting, and the leaf that agrees with it.
6. **Tier 3.** Otherwise, the measurements table, deployable-setting rows preferred over ranking-setting ones.
7. **Tier 4.** Otherwise, all the leaves are scored by the prior in one batch and the lowest prediction wins.
8. **Only now is the winning leaf built for real**, and the compile moves to the next fork.

With no measurements and no prior at all, step 4 still runs — the golden tier needs no model — and every fork it does
not answer falls to the rule's first option.

## What "agrees with" means

A single rule serves all three measured tiers. **A measured row counts as evidence for an option when every knob the
option has already decided has the same value in that row.** Knobs the option has not decided yet are free; a later
pass will decide them.

That is what lets one fully decided measurement settle a fork whose options are only partly decided. Suppose the
reservoir holds a measurement of this kernel with `WORK = w2x2`, `TILE = f2x8`, `STAGE = depth 2`, and the fork on the
table is choosing only the worker arrangement:

```
option A:  WORK = w2x2     ← agrees: the one knob it decides matches the measured row
option B:  WORK = w4x1     ← does not agree
option C:  WORK = t32x8    ← does not agree
```

Option A wins on measured evidence, even though it has said nothing yet about tiles or staging. The remaining knobs
are decided at later forks, where the same measurement is consulted again — by then the option has more decided knobs,
so the agreement test is stricter, and the same row keeps steering the compile toward the configuration it measured.

## Which tiers apply under which settings

Regime gating is not symmetric between the tiers, and the asymmetry is deliberate.

- **Golden configurations and reservoir evidence apply only to a compile at the deployable setting.** Their numbers
  are true of that setting and no other. This is why the test suite, which compiles at the fast ranking setting for
  speed, never consults golden configurations at all.
- **The measurements table applies under any setting**, keeping its internal preference for deployable-setting rows.
  A measurement of this exact kernel is real information even under a different setting, and it still beats the
  model's extrapolation.
- A compile whose settings name no optimization level — which is the default for `emmy compile` and `emmy run` —
  counts as deployable. So an ordinary deployment always gets the full hierarchy.

## Ties are broken by content, never by order

The prior can score several options identically, and one measured row can agree with several offered options. Every
tier therefore breaks its ties the same way: by the option's knob values, rendered in sorted order. Never by the order
the rule happened to return its options in.

This is not fussiness. The order options are generated in can shift between processes, so breaking a tie by position
is a coin flip on every boot — and it once shipped a release image that compiled itself into a different set of
kernels depending on which boot you looked at. Determinism here is pinned by tests that re-run each tier under
shuffled option orders and require the same answer, plus a check that two separate processes select the same kernels.

## When there is no prior

The prior can be missing: a corrupted checkpoint, or weights that fail to load. In that case tiers 2, 3 and 4 are all
gone at once — the reservoir travels inside the prior's checkpoint, and the database tier is only consulted on the
path where a prior exists.

What survives is tier 1. The golden configurations still decide every fork they match, and the compile logs loudly
when a golden overrides what would otherwise have been the rule's first option. A broken checkpoint can never
silently cost a fork its reviewed measurement.

## Changing which kernels exist

Everything above decides *how* a kernel computes. Two mechanisms can change *which kernels exist* during an ordinary
compile, and both are deliberately harder to trigger.

**A recorded cut.** Before the schedule is chosen, a separate decision splits — or does not split — the recognized
work into kernels. Keeping it in one kernel is the default and is what the absence of a recording means. A golden
entry can instead record a cut at a named place inside the kernel, and that entry is consulted at recognition time,
matched against this GPU and restricted to the deployable setting like any other golden. Each resulting piece is then
recognized afresh and works down the same hierarchy for its own schedule.

**A structural fork.** The prior is never asked to rank structural options against each other, because it would be
comparing predictions across different kinds of kernel, where its errors do not cancel the way they do among siblings
of one fork. Instead the compile *costs* each side: for every kernel each side would produce, it runs a small nested
resolution through the same hierarchy above and takes the cost of that kernel's chosen configuration. The cheaper
total wins. Because the nested resolution consults the whole hierarchy, one side's total can mix a golden's recorded
time, local measurements and model predictions across its kernels.

That costing is only allowed when the online prior is trained and trusted. Without it — on a machine with no
measurements, or where one side cannot be costed at all — the structural option is dropped and the default kernel set
is kept. **A cold compile never changes which kernels exist**, unless a recorded cut tells it to.

## When the chosen option does not fit

The prior ranks by predicted latency, so it can rank first a tile that fails validation — one that needs more shared
memory or more threads than the card has. A tuning run would build it, watch it fail, and move on. An ordinary compile
builds nothing, so it has to notice differently.

It notices at the end: a node that never became a kernel. The compile then adds the offending tile to a block list
and resolves the whole graph again. Every other choice replays identically, because each is decided from the same
evidence as before, so backtracking is cheap and needs no saved snapshots. Retries are bounded.

If the budget runs out with the node still un-lowered — a trained prior can rank many oversized tiles above the first
one that fits — there is a last resort: resolve once more taking each rule's first option, which is emitted to be
budget-safe. Two things still hold during that last pass. Golden configurations are still consulted, so one oversized
kernel cannot cost every *other* kernel its reviewed measurement; and the block list still applies, so a tile that
already failed cannot be chosen again. Only if even the first option does not fit does the compile stop with a clear
error naming the node.

## How to tell which tier answered

Honestly: there is no single switch that reports, per fork, which tier decided it. What exists is three things you
correlate.

- **The warnings.** A golden shape whose entries match nothing on offer, measured evidence that overlaps none of the
  offered options, and a golden overriding the first option on a compile with no prior are all logged loudly.
- **The record of the resolution.** Each decided fork records what was chosen and the time of whichever row decided
  it — a measured time when a golden or a measurement decided, a predicted one otherwise.
- **The audits.** `emmy eval golden` re-runs the golden-tier consultations and reports what the compiler produces
  against what was recorded. That is the subject of the next page.

## See it yourself

The audit below re-compiles each recorded shape with nothing pinned, prints the knob values the compiler chose beside
the recorded ones, and then reports whether each recorded configuration is still among the options the compiler
offers at all:

```bash
emmy eval golden
emmy eval golden --kernel matmul.square.512
```

Neither needs a GPU: once the shape and the target card are known, the set of offered options is fixed, so nothing has
to be run. The next page explains how to read the result.

Next: [7. Golden configurations](./07-golden-configurations.md).
