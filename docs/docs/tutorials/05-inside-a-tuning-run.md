---
sidebar_position: 5
title: "5. Inside a Tuning Run"
description: What emmy tune actually does — a search over forks, split into two levels, with every kernel tuned on its own.
keywords: [Emmy, autotuning, MCTS, tune, benchmark, kernel, search]
---

# 5. Inside a Tuning Run

`emmy tune` is the only part of Emmy that creates knowledge rather than using it. This page describes what it does,
because the rest of the series is about consuming what it leaves behind.

```bash
emmy tune Qwen/Qwen3-Embedding-0.6B --layer 0
```

That command traces the layer, runs the pipeline over and over with different answers at each fork, builds and times
the resulting kernels, and writes what it measured into the tuning database and the online prior's checkpoint.

## The search, in plain words

The forks in a compile form a tree. At the root, the first fork's options are the branches; inside each branch, the
next fork's options branch again; a path from the root down to a fully decided configuration ends at a candidate that
can actually be built and timed.

Emmy walks that tree with **Monte Carlo tree search**. One step of the loop looks like this:

1. **Descend** from the root, at each fork choosing the child with the best combination of *how good the measurements
   below it have been* and *how little it has been tried*. A branch nothing is known about is attractive because it is
   untried; a branch whose measurements were poor becomes less attractive as evidence accumulates.
2. When the path reaches a fully decided candidate, **build it and time it** on the GPU.
3. **Send the result back up** the path, so every fork on the way down now knows a little more about what lies
   beneath it.

Two details of Emmy's version are worth knowing. The first is that the estimate of how promising an untried branch is
comes from **the prior** — the model that predicts latency from a configuration's knob values, which the [prior
page](./08-inside-the-prior.md) covers in full. It is the only signal used for that; the search never forces itself to
try every untested sibling once. A branch the prior is confident is slow simply sinks down the priority order, and the
budget goes elsewhere.

The second is when it stops. The search counts how many fully decided candidates it has benchmarked since the last
time it found a new best. When that count passes `--patience` (50 by default), that part of the search is finished.

## Two levels, not one

Running one search over the whole graph does not work. The pipeline applies rules one after another, so choices about
*which kernels exist* and choices about *settings inside one kernel* end up nested inside each other, multiplying out
under a single budget — and the kernels deepest in the graph get whatever is left, which is nothing.

So `emmy tune` splits the search in two.

**The outer search decides which kernels exist.** It drives the passes that can change the graph, and it stops as soon
as every structural fork has been answered. Each of its endpoints is one candidate grouping of operations into
kernels. Its score is one divided by the total time of that grouping — the sum, over its kernels, of the best time the
inner search could find for each.

**The inner search tunes one kernel at a time.** For each kernel in the grouping, Emmy cuts it out of the graph into a
slice of its own — the kernel plus its inputs, with everything else replaced by a stand-in input — and runs a plain
search over the lowering passes on that slice alone. The search sees one kernel's forks and spends the whole patience
budget on them.

The saving is large. Tuning `n` kernels each with `k` configurations costs `n × k` benchmarks in this arrangement,
where a single combined search would face `k^n` paths.

**Why cutting kernels apart is legitimate:** the choices the inner search makes are all in-place ones. They change how
a kernel computes, never what the graph looks like, so the whole graph's time is the sum of its kernels' times.
Emmy checks that assumption rather than assuming it: once the best grouping is picked, the assembled graph is
benchmarked once, for real, and compared against the sum of the isolated measurements. A gap would reveal cache or
launch effects that isolated benchmarks cannot see. In practice, on small graphs, it stays under 2 percent.

!!

There is one deliberate exception. When the fork being tuned is an attention fusion, the slice also carries the kernel
that produces the scores the fusion consumes. Replacing that producer with a stand-in input would make the fused
option impossible to build inside the slice, and every path would silently fall back to the split form — tuning
kernels that a real deployment would never use, and never measuring the fused one at all.

!!

## Identical kernels are tuned once

A transformer has many layers, and their kernels are frequently identical — the same shapes, the same operations, the
same body. Emmy identifies each kernel by its **structural key**, a fingerprint of what it computes rather than what
it is called, so 24 identical RMSNorm kernels across 24 layers collapse into one unit of work. The total time still
counts all 24; only the tuning is done once.

This is why a whole-model tuning run reports progress against a much smaller number than the kernel count. On
`Qwen/Qwen3-Embedding-0.6B`, 337 kernels reduce to about 14 distinct ones.

The same key makes results transferable. A kernel tuned inside its slice is the same kernel in the assembled graph,
and two candidate groupings that happen to contain the same kernel share its tuning — the second one finds the
measurement already in the database and does not run anything.

## Re-running is cheap, and worth it

The inner search runs for every kernel on every invocation. It is never skipped because a kernel was tuned before.
That sounds wasteful and is not: every configuration already measured is served from the database with no GPU work at
all. Re-running an identical tuning run walks the same path, hits the cache every time, and finishes having launched
nothing.

The reason to re-run anyway is that the prior keeps changing as it learns from other kernels and other runs. The same
patience budget, steered by a better prior, descends into a different part of the tree — so a re-run reaches genuinely
new configurations and replays the rest for free. An earlier version of Emmy skipped kernels that had already been
tuned, which suppressed exactly this, and it was removed.

## Finishing at the deployable setting

The sweep compiles at the fast ranking setting, which as [the previous page](./04-measuring-and-recalling.md) explained
is good for ordering configurations but is not what a deployment runs. So whenever a measurement lands within 15
percent of the best result so far — a band wide enough to include near-ties, not just the winner — that configuration
is measured a second time at the deployable setting.

Those second measurements are tagged with the regime they were taken under, and they are the numbers a deployment
should be choosing from. They are written to the reservoir inside the prior's checkpoint and to the search-tree table,
and — this is the asymmetry from the previous page — never to the measurements table.

Every measurement, in both passes, is taken with CUDA graph capture by default, so what is recorded is GPU time
without the launch overhead of the host loop.

## Using more than one GPU

Because kernels are tuned independently, the per-kernel loop fans out across devices:

```bash
emmy tune Qwen/Qwen3-Embedding-0.6B --gpus 4
emmy tune Qwen/Qwen3-Embedding-0.6B --devices 0,2
```

Each device gets one kernel at a time. The per-kernel results and the total are identical no matter how many devices
are used — only the order rows arrive in the prior's checkpoint differs, since that follows completion order. The
default is a single device, which is exactly the sequential loop.

## See it yourself

Tune one shape rather than a whole model — a golden configuration's own small program is the quickest thing to run:

```bash
emmy tune --golden matmul.square.512 --patience 10 -v
```

`-v` prints one line per step of the search instead of the live progress bar, so you can watch configurations being
tried and the best time falling. This one needs a GPU.

Afterwards, the same database can be read back with no GPU at all:

```bash
emmy eval variants --kernel matmul
emmy eval failures
```

The first lists what was measured, best first. The second groups the configurations that failed to build or run, with
the knob values every failing row shares — which is usually how a systematically bad corner of the search space gets
noticed.

Next: [6. The deploy evidence hierarchy](./06-deploy-evidence-hierarchy.md).
