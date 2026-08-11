---
sidebar_position: 8
title: "8. Inside the Prior"
description: The model that ranks schedules before they are measured — its features, its two halves, how it is trained, and the check that decides whether it is trusted.
keywords: [Emmy, prior, features, CatBoost, calibration, quarantine, offline prior, online prior]
---

# 8. Inside the Prior

When no measurement answers a fork, something still has to choose. That something is the **prior**: a model that
predicts how fast a configuration will be, from the configuration alone. This page is about how it works, and it goes
slower than the pages before it, because this is where the interesting failures live.

Two properties frame everything else.

**There is one ranking path.** Whatever is choosing — a tuning search deciding where to explore, or a compile deciding
what to deploy — asks the same object the same way. Forks carry no score of their own, and nothing builds a kernel in
order to rank it. The knob values go straight into numbers the model consumes.

**The prior has two halves.** The **offline prior** is fitted ahead of time and ships in the repository; it is what
answers on a machine that has never measured anything. The **online prior** is trained from local measurements as
tuning runs accumulate. A composite object holds both and decides which one answers.

## Features: what the model actually sees

A **feature** is one number describing a candidate, computable without building it. Every prediction is a function of
a row of features, and the row has three groups.

| Group | What it describes | Where it comes from |
| --- | --- | --- |
| Hardware and regime | which GPU, its memory, the optimization level this compile is running at | probed from the machine, or named with `--target` |
| Structure of the operation | counts of the statements and operations in the kernel's body, the loop extents, the data types of the inputs | stamped onto the operation by the stamping pass, before any fork is reached |
| The candidate itself | its knob values, encoded by type; a named tensor core instruction expands into the properties of that instruction | the fork option |

The offline half additionally computes hand-designed descriptions of a tile's geometry from those knob values — its
area, its shape, how much shared memory it needs, how many groups of threads could be resident at once.

### The subtlety that shapes the feature set

The hardware group has the same value for every candidate competing at one fork. Every option is being compiled for
the same GPU. **So no weight on a hardware feature can change the ranking within that set** — it shifts every
candidate's score by the same amount and cancels out.

That has a counter-intuitive consequence: what tells GPU generations apart has to be a *per-candidate* feature, one
that only takes a value where the hardware offers the thing it describes. Two families of feature exist for exactly
this reason:

- Features that mirror the tile's geometry onto candidates that stage their data through the newer hardware
  transport. On a card that has it, those candidates get a distinguishable description; on a card that does not, they
  never appear. One weight set can then score newer tiles differently from older ones.
- Features that separate candidates with the same tile but a different arrangement of workers. Without them, two
  genuinely different candidates produced identical feature rows, and no model could have told them apart.

If you take one thing from this page for future feature work: a feature that is constant across a fork's candidates
cannot influence that fork.

## The offline half

The offline prior scores a candidate with a linear formula over the geometry features, fitted ahead of time and stored
in the repository as a small artifact carrying its own version and a record of where it came from. It never falls back
on the order options were emitted in.

**Loading it is strict.** A missing artifact, or one whose feature version does not match the running code, is a hard
error — refit it, never silently continue with something else. (A compile treats the load as best-effort so a bad
artifact does not abort a deployment; what it gets instead is the no-prior behaviour from [the hierarchy
page](./06-deploy-evidence-hierarchy.md), where golden configurations still decide and the rest falls to the rule's first
option.)

**It is fitted on the golden configurations**, by `emmy fit`. For each recorded golden, the fitter reconstructs the
set of candidates that golden competed against — by tracing the shape's own small program and enumerating the fork —
and trains the weights to rank the recorded configuration well inside that set. The loss has two parts:

- an objective pushing each golden's rank up within its own candidate set, with the kinds of case weighted so that no
  one kind dominates the fit;
- a penalty on the weights, expressed in raw feature units.

**The penalty is there to make the fit well-determined, not to shrink the weights**, and the difference matters. The
ranking objective barely moves when you scale a weight on a feature that hardly varies across the golden candidate
sets. An unpenalized fit is therefore free to pick an arbitrarily large weight there, and nothing in the golden-rank
metrics will show it. It becomes catastrophic at a fork, where a not-yet-decided knob makes that feature zero: the
enormous weight now dominates the score of every candidate that has not yet decided it. The penalty has to be in raw
units, because after rescaling the inflated weight looks like an ordinary one.

**The score is turned into a stand-in for latency by an exponential curve**, and there is a rule about that curve
which is easy to violate: it must never flatten out over the range of scores that actually occur. A curve that
saturates inside the live range collapses good candidates onto one identical value; the choice among them then falls
back to the order the options were emitted in, which is arbitrary. That is not hypothetical — it is how cold
deployments once shipped kernels 12 to 29 times slower than the recorded configuration for the same shape, while the
golden-rank metrics reported the model was choosing correctly. (Why the metrics were fooled is on [the next
page](./09-storage-checks-and-limits.md).)

Two more details are worth knowing about the offline half. A separate weight set ranks kernels whose tiles are masked
because an axis is symbolic, selected on the stamped structure. And two feature interactions sit outside the linear
weights, expressing a preference for the tensor core path — they stop a kernel that could use the tensor cores from
deploying a plain-arithmetic configuration instead.

## The online half

The online prior is a **CatBoost** model trained on the measurements a tuning run produces. There is exactly one of
them across every kernel, every GPU and every compiler setting — hardware and structure are *features in every row*,
not a key that partitions separate models.

**Why a tree model rather than a linear one.** The model's best guess must not run off to a degenerate extreme. A
linear model moves in one direction with each knob, so its optimum always sits at a corner of the box of candidate
values — the largest tile, the deepest staging — which shipped real blow-ups before the switch. Any tree ensemble is
bounded: an untested extreme simply inherits the value of the nearest measured region, so it stays sane outside the
data. Among the bounded models, CatBoost also generalized best to an operation that had never been tuned, which is
the case that matters for a deployment.

**How a partly decided option gets a label.** Real measurements exist only at complete configurations, but the model
has to rank half-decided options at every level of the fork tree. So the label for any position in the search tree is
the best measured latency anywhere below it. A branch is judged by the best thing reachable from it, which is exactly
what a search wants to know when deciding where to descend.

**The training data lives inside the checkpoint.** Rows stream in as kernels are tuned, into a bounded random sample
capped at 100,000 rows that is maintained across runs. The model refits on a schedule that starts frequent and
coarsens as the data grows, and checkpoints itself to a JSON file holding both the model and its data. That data is
the **reservoir** — the same one that is [tier 2 of the evidence hierarchy](./06-deploy-evidence-hierarchy.md). The
checkpoint therefore carries deployment evidence, not just model state, which is why losing it costs more than a
retrain.

The model is deliberately **not** refit during a single kernel's own search. Within one run it is a fixed model, so
the search is not chasing a moving target.

## Calibration: the check that decides whether it is trusted

A trained model is not automatically believed. After every fit, Emmy measures how well the model ranks **the very rows
it trained on**: predictions against its own labels, as a rank correlation, computed per operation and taken as the
median across operations. Groups with fewer than 8 rows are skipped.

- A genuinely trained model scores around **+0.85**.
- The failure this catches scores around **0** — the collapse where the model and its stored rows no longer share
  feature names, so predictions are effectively constant and ranking is worse than chance.
- Below **0.5** the model is **quarantined**: it keeps training and keeps checkpointing, but the deployment ranking,
  the search's steering signal and the structural cost estimate all fall back to the offline half, and the verdict is
  logged.

Measured evidence from the reservoir stays live under quarantine. A measurement needs no trusted model to be true.

Three things to understand about this gate:

- **It is an alarm for measured failure, not a demand for proof of quality.** A calibration that could not be measured
  at all — no operation group large enough, or the statistics library missing — passes.
- **It is known to be lenient in one case.** A small tuning run needs only 50 rows to fit. If all its operation groups
  stay under 8 rows, calibration cannot be computed, so it passes, and the model is trusted to own deployments and
  structural decisions on very little data.
- **It deliberately does not catch the subtler failures**: a model that fits the operation families that were tuned
  and generalizes badly, or one that ranks well but is wrong about absolute times on operations it never saw — which
  matters because the structural cost estimate compares sums of absolute predictions. Those are what the diagnostics
  on the next page exist to surface.

The gate exists because being fitted was once the only requirement, and a mis-calibrated model owned deployments
silently.

## How the two halves combine

**When deploying, one half answers, never both.** If the online model is fitted and passes calibration, prediction
calls go to it alone and the offline half is out of the deployment path entirely. Otherwise the offline half answers.
There is no blending of predicted times, because a deployment's choice should not be a compromise between a trained
model and a hand-fitted heuristic.

**When steering a search, they blend.** The signal used to decide where to explore next is the online prediction
multiplied by the offline score raised to a small power (0.3 by default, adjustable; zero gives a pure online search).
The offline factor is clamped, only its ordering is meaningful, and its no-opinion value is exactly 1.0 — so a
configuration the offline heuristic has no view on leaves the online prediction untouched.

The point of the blend is to keep exploring regions the cold heuristic rates well but a data-poor online model has
buried, while making sure the offline factor's arbitrary magnitude never touches the times a deployment sees.

## See it yourself

Evaluate each half against the golden configurations — where each recorded configuration ranks among the candidates it
competed against:

```bash
emmy eval offline
emmy eval online
```

Print the feature row the model actually sees for each golden:

```bash
emmy eval online --features
```

And refit the offline half from the golden configurations, with cross-validation, no GPU required:

```bash
emmy fit
```

That writes a metrics file and a weights file to a fresh directory under `_tune/fits/`, so two fits can be compared by
diffing their metrics rather than by argument.

Next: [9. Storage, checks and limits](./09-storage-checks-and-limits.md).
