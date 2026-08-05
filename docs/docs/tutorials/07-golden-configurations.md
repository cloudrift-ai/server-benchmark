---
sidebar_position: 7
title: "7. Golden Configurations"
description: The reviewed, per-GPU measurements that ship with the repository — how they are recorded, what makes one unusable, and the audits that catch it.
keywords: [Emmy, golden configuration, evidence, audit, benchmark, pin, realize]
---

# 7. Golden Configurations

A **golden configuration** is a reviewed measurement of a standard problem shape on a specific GPU: the schedule that
was fastest, and how fast it was. They matter more than their modest description suggests, because as [the stores
page](./04-measuring-and-recalling.md) noted, **they are the only measured data that travels with the repository**. On a
freshly rented machine they are the difference between deploying on evidence and deploying on a guess.

They do three jobs at once:

1. **The first tier** of the deploy evidence hierarchy.
2. **The training data** for the offline prior, which is fitted on them.
3. **A regression reference** — if today's compiler produces something slower than the recording, that is a defect
   with a number attached.

## What one looks like

Golden configurations live in one file per GPU. A file names the card it was measured on and then lists entries:

```yaml
gpu_name: NVIDIA GeForce RTX 5090
compute_cap: [12, 0]
configs:
  - kernel: norm_linear
    name: gemma4_12b.norm_q_proj.m32
    M: 32
    H: 3840
    N: 4096
    dtype: 'fp16'
    knobs: {WORK: 'w1x16', TILE: 'mma_m16n8k16_f16_f32/f2x2/k2', REDUCE: 'g8k', RASTER: '', STAGE: 'd2/sync'}
    emmy_us: 26.7
    cublas_us: 19.8
```

Everything above the `knobs` line identifies the shape; the `knobs` line is the schedule, in the exact spelling from
[the forks page](./03-forks-and-knobs.md); the last two lines are what it measured, beside a reference implementation.

`kernel:` says which standard shape this is. The kinds are matrix multiplication, attention, RMSNorm, softmax,
reduction, elementwise, the fused RMSNorm-and-linear kernel, and its multi-output sibling in the feed-forward block.
Two more kinds — rotary position embedding, and embedding lookup — are recorded as memory-bound reference points only:
their lowering has no fork in it, so there is nothing for a compile to decide and they serve purely as regression
checks.

Each kind comes with a small PyTorch program that reproduces its shape. That is what makes a golden runnable: the
recorded shape can be traced and compiled on its own, without the model it came from.

Names repeat across files — every GPU has its own `matmul.square.512` — with different shapes, different data types
and different measured times. So a compile only ever consults the file for the GPU it is targeting. Pooling them would
mean deploying one card's configuration on another.

## Recording one

A golden is recorded from a side-by-side comparison run:

```bash
emmy run --golden matmul.square.512 --bench --ab "WORK=w2x2,TILE=f2x8,STAGE=d2/cp"
```

That compiles the shape the way the compiler would on its own, then compiles it again with the given knob values
pinned, and prints both. Two rules about which number to copy:

- **Record from a pinned row, never from the ordinary comparison row.** The ordinary row is measured interleaved with
  the PyTorch baselines, so the allocator state and cache contents of another framework are resident while it runs. It
  is the right number to compare against PyTorch and the wrong number to compare against a pinned row — the gap
  observed in practice is around 7 percent. Whenever pinned rows are measured, the ordinary configuration is
  additionally re-measured on its own, and *that* is the baseline the pinned rows are compared against.
- **Copy the whole knob map, including the families that are off.** Each row prints the knob values the compile
  actually produced, with every schedule family written out explicitly. An entry that omits a family leaves that
  family to whatever the compiler fills in when the entry is replayed, which shifts as the compiler evolves — a
  recurring source of regressions that look real and are not.

Three checks guard the measurement before it is believed:

1. **The pin must have been honored.** The knob values the compile produced are compared against the pinned ones, and
   a mismatched row fails without being measured at all. A structurally invalid pin silently falls back to the
   compiler's own choice, so measuring it would compare the compiler against itself and report a flattering result
   under the pin's name.
2. **The arithmetic must be plausible.** A row implying more arithmetic per second than the card can physically
   perform is flagged as a bad measurement rather than celebrated as a fast kernel.
3. **The answer must be right.** Each pinned configuration is executed once on the same inputs as the ordinary run and
   its outputs compared. A kernel that skips a step it should not have produces plausible-looking garbage very
   quickly.

Every row is measured in a separate worker process that can be killed, so one configuration that hangs takes down its
own process, is reported as a failure, and the remaining rows continue.

## The failure that matters most

A recorded configuration is only useful if the compiler still offers it. The word for that is **realize**: a recording
realizes at a fork when the options the compiler offers there include one that matches it.

**A golden that realizes nowhere is worse than no golden at all.** When a shape matches but none of its entries
matches anything on offer, the compile warns loudly and falls through to the tiers below — and those tiers, for a
shape somebody bothered to record, can be hundreds of times slower than the entry claims. The recording promises a
number the deployment cannot produce.

Two ways to record one by accident, both worth knowing:

- **A recorded cut and a tile choice cannot go in the same entry.** A cut is decided before schedules are chosen, so
  no single offered option carries both, and an entry containing both matches nothing. Record one or the other. Here
  is a real cut entry, recorded for the same shape as the fused one above:

  ```yaml
  - kernel: norm_linear
    name: gemma4_12b.norm_q_proj.m32.cut
    M: 32
    H: 3840
    N: 4096
    dtype: 'fp16'
    knobs: {PLACE: 'cut'}
    emmy_us: 16.0
    cublas_us: 19.0
  ```

  It stores the split and nothing else. Each resulting piece is recognized on its own afterwards and finds its own
  schedule through the hierarchy. In this case the split is 1.8 times faster than keeping the work fused, which is
  exactly the kind of finding that has to be recorded to be usable — a cold compile keeps work fused unless told
  otherwise.

- **A row must be verified to deploy, not merely to reproduce.** Pinning reproduces configurations the compiler would
  never offer on its own, so a recording that only works when pinned still looks healthy in an isolated check. Only
  the in-model audit catches it.

## The two audits

**Against the shape's own program** — `emmy eval golden`. For every entry that decides a fork, the shape's own small
program is re-compiled with nothing pinned. The report has two halves: a knob-by-knob comparison of what the compiler
chose against what was recorded, and then the check that matters here — whether the recorded values are among the
options the compiler offers at all. Neither half runs a kernel, because once the shape and the target card are known
the set of offered options is fixed, so the whole audit works with no GPU present. An
entry that only a hand-forced pin can produce is reported as `PIN-ONLY`. That is legal — a pin is a documented lever —
as long as some other entry for the same shape still gives a deployment something good. A shape whose entries are
*all* pin-only is reported as `FALL-THROUGH`, and the audit exits with a failure, because a deployment there will log
"nothing on offer matches" and fall past the golden tier with nothing to catch it.

**Inside the real model** — `emmy eval golden --in-model`. An entry may record which model its shape came from. Those
entries are audited by rebuilding the model as a weight-free stand-in from its configuration file alone — no
checkpoint download, since tracing never reads a weight value — and compiling it with the golden tier as the only
evidence available. Every consultation yields one of:

| Verdict | Meaning |
| --- | --- |
| `MATCH` | a recorded configuration matched an option the compiler offered |
| `DRIFT` | the shape matched but none of its entries did — always a defect, since the recording claims a time the deployment can no longer produce |
| `GAP` | there is no recording for this shape |

The two views genuinely differ, which is why both exist. Isolated checks have passed 68 out of 68 while the same
configurations drifted inside the model, where the kernel is fused with its neighbours and the offered options are not
the same set.

Coverage is gated in continuous integration so that it can only improve: each GPU's set of gaps is pinned exactly, a
new gap fails until a configuration is recorded, and a gap that has been closed fails until its line is removed from
the baseline. The stand-in models deliberately follow the installed modeling library, so an upgrade that changes the
forward pass changes the audit exactly as it changes serving.

## Two smaller rules

**Fast math never loses.** Entries recorded with faster, less precise arithmetic are only kept when they are faster
than the best ordinary sibling. A slower one could never be used anyway — the ordinary entry is picked first, whether
fast math is enabled or not — so such rows are dropped, and a missing one simply means a fast-math deployment uses the
ordinary configuration there.

**The two halves of the prior treat goldens differently.** The online prior never trains on them: a recorded
configuration enters no training data anywhere, which leaves the goldens as a clean acceptance set — data the model is
judged against but never learns from. The offline prior *is* fitted on them. That distinction is the subject of the
next page.

## See it yourself

Read a real file — they are plain YAML, and the comments in them are the record of why each entry is what it is:

```bash
ls emmy/compiler/pipeline/search/goldens/
grep -n -A9 "kernel: norm_linear" emmy/compiler/pipeline/search/goldens/rtx5090_sm120_gemma4.yaml | head -30
```

Then run the audits. The first needs no GPU at all; the second needs the modeling library installed, but no
checkpoint download:

```bash
emmy eval golden --kernel matmul.square.512
emmy eval golden --in-model
```

Next: [8. Inside the prior](./08-inside-the-prior.md).
