---
sidebar_position: 6
title: "6. The Deploy Evidence Hierarchy"
description: How a compile answers a fork — measured evidence first, then prediction, then the rule's first option.
keywords: [Emmy, deploy evidence hierarchy, evidence, greedy selection, golden configuration, prior, fork]
---

# 6. The Deploy Evidence Hierarchy

This is the page the series has been building towards. An ordinary compile answers every fork immediately, without
running anything, by working down a fixed order — **best evidence first**. That order has a name, the **deploy
evidence hierarchy**, and each step in it is called a **tier**.

## The order

1. **Measured evidence.** Every measured row whose structural signature is this kernel's, from three stores read as
   one index: the reservoir (measurements taken at the deployable setting), the tuning database's rows for this
   compile's regime, and the golden rows in scope — the golden configurations recorded for this GPU that ship with
   the repository, or the file `--golden PATH` names instead. The option that agrees with the fastest such row
   decides. A golden is a preference among measured rows, never a forced pin: a recorded row that is slower than a
   local measurement loses, and a row nothing measured yet (a proposal) is not evidence until `run --golden PATH
   --bench` has measured it.
2. **The prior's prediction.** Only when no option has any measurement behind it at all: the option with the lowest
   predicted latency. `--strict-evidence` turns this step into an error naming the kernel, for a deploy that must
   run on measurements alone.
3. **The rule's own first option**, if there is no prior to ask. Rules order their options so that the first one is
   always safe.

The shape of that list is the whole design in miniature. Any measurement of this exact kernel beats a prediction; a
prediction beats an arbitrary choice. Nothing further down ever overrules something further up, and there is one
mechanism, not one per store: a golden row and a tune's row are the same kind of thing to the pick.

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
4. **Measured evidence.** The fastest reservoir, database or golden row of this same kernel, and the leaf that
   agrees with it — measured rows first, whichever store they came from.
5. **The prior.** Otherwise, all the leaves are scored by the prior in one batch and the lowest prediction wins.
6. **Only now is the winning leaf built for real**, and the compile moves to the next fork.

With no measurements and no prior at all, every fork falls to the rule's first option.

## What "agrees with" means

A single rule serves every measured store. **A measured row counts as evidence for an option when every knob the
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

## Which evidence applies under which settings

- **Reservoir and golden rows apply only to a compile at the deployable setting.** Their numbers are true of that
  setting and no other.
- **The measurements table applies under the setting it was measured in**: the index is keyed by the compile's
  regime, so a sweep at another optimization level is never read by an ordinary deploy.
- A compile whose settings name no optimization level — which is the default for `emmy compile` and `emmy run` —
  counts as deployable. So an ordinary deployment always gets the full hierarchy.

## Ties are broken by content, never by order

The prior can score several options identically, and one measured row can agree with several offered options. Every
step therefore breaks its ties the same way: by the option's knob values, rendered in sorted order. Never by the order
the rule happened to return its options in.

This is not fussiness. The order options are generated in can shift between processes, so breaking a tie by position
is a coin flip on every boot — and it once shipped a release image that compiled itself into a different set of
kernels depending on which boot you looked at. Determinism here is pinned by tests that re-run each step under
shuffled option orders and require the same answer, plus a check that two separate processes select the same kernels.

## When there is no prior

The prior can be missing: a corrupted checkpoint, or weights that fail to load. The reservoir travels inside the
prior's checkpoint, so its rows are gone with it, and with no prior every fork it would have ranked falls to the
rule's first option. The database and golden rows are read on the path where a prior exists, so a broken checkpoint
costs a deploy its measured evidence too; `--strict-evidence` makes that loud instead of silent. Pinned knobs still
apply: a pinned family never reaches a fork at all.

## Changing which kernels exist

Everything above decides *how* a kernel computes. Two mechanisms can change *which kernels exist* during an ordinary
compile, and both are deliberately harder to trigger.

**A placement cut.** Before the schedule is chosen, a separate decision splits — or does not split — the recognized
work into kernels. An explicit placement pin decides it outright; unpinned, the cut is offered as an ordinary
structural fork (the fused form first, one fragment per legal seam), so a tuning run can discover a profitable
split and a chosen cut records as a row whose keys spell the route. Such a row is measured evidence for the kernel
it was recorded on: at the placement fork it is the measured price of that cut, it outranks any arm the prior would
have to price, and taking it means taking the pass's own cut arm. Each resulting piece is a brand-new kernel,
recognized afresh, that works down the same hierarchy for its own schedule from rows of its own.

**Splitting a reduction across blocks.** Dividing a long reduction among several blocks turns one kernel into a
kernel that computes partial results plus one that combines them (or, on the cheaper arm, a single kernel that adds
its partials straight into the output). The pieces are treated exactly like a cut's: each is a **brand-new kernel**
that inherits nothing from the kernel it replaced and works down this whole hierarchy for its own schedule. They
are differently shaped kernels doing different work, so there is no reason for them to end up configured the same,
and nothing makes them. Each also records its own measurements under its own identity, so a stored time always
describes the kernel that earned it.

Because a hand pin is a statement about how kernels run, it reaches those new kernels too, which raises an obvious
question: does a split kernel then split again? No. Dividing a reduction that is *already* one block's share of a
larger one is not a further choice, it is the same choice applied twice, so the compiler does not offer it. What
remains of the pin — how each piece folds its own share within a block, say — still applies. A measured row is
different: it says nothing to the pieces, whose own rows say what they measured.

**A structural fork nothing measured.** The prior is never asked to rank structural options against each other,
because it would be comparing predictions across different kinds of kernel, where its errors do not cancel the way
they do among siblings of one fork. Instead the compile *costs* each side: for every kernel each side would produce,
it runs a small nested resolution through the same hierarchy above and takes the cost of that kernel's chosen
configuration. The cheaper total wins. Because the nested resolution consults the whole hierarchy, one side's total
can mix a golden's recorded time, local measurements and model predictions across its kernels — which is why a
measured route row, when one exists, outranks that costing outright.

That costing runs with whichever prior is loaded — the online model when it is trusted, the offline half otherwise —
so on a machine with no measurements it is a comparison of predictions. When one side cannot be costed at all, nothing
is withheld: every option, the structural ones included, goes to the ordinary ranking. A measured route row is the one
thing that settles such a fork without a prediction, and a placement pin removes the fork altogether.

## When the chosen option does not fit

The prior ranks by predicted latency, so it can rank first a tile that fails validation — one that needs more shared
memory or more threads than the card has. A tuning run would build it, watch it fail, and move on. An ordinary compile
builds nothing, so it has to notice differently.

It notices at the end: a node that never became a kernel. The compile then adds the offending tile to a block list
and resolves the whole graph again. Every other choice replays identically, because each is decided from the same
evidence as before, so backtracking is cheap and needs no saved snapshots. Retries are bounded.

If the budget runs out with the node still un-lowered — a trained prior can rank many oversized tiles above the first
one that fits — there is a last resort: resolve once more taking each rule's first option, which is emitted to be
budget-safe. The block list still applies during that last pass, so a tile that already failed cannot be chosen
again. Only if even the first option does not fit does the compile stop with a clear error naming the node.

## How to tell whether evidence answered

Honestly: there is no single switch that reports, per fork, which row decided it. What exists is three things you
correlate.

- **The warnings.** Measured evidence for a kernel that overlaps none of the offered options is logged loudly.
- **The record of the resolution.** Each decided fork records what was chosen and the time of whichever row decided
  it — a measured time when a measurement decided, a predicted one otherwise.
- **The audits.** `emmy eval golden --golden GOLDEN_YAML --serving-config PATH` re-runs the file's own-program and
  exact serving-matrix compiles on the pinned GPU, one verdict per schedule fork. That is the subject of the next
  page.
- **Strict evidence.** `--strict-evidence` on `run`, `compile` or `serve` refuses to let the prior decide at all, so
  a compile that finishes under it was decided by measurements alone.

## See it yourself

The audit below validates one canonical file against the release configuration and live GPU, then checks its own
programs and freshly traced serving matrix:

```bash
emmy eval golden --golden <canonical-golden.yaml> --serving-config <models/slug.env>
```

The next page explains how to read the result.

Next: [7. Golden configurations](./07-golden-configurations.md).
