---
sidebar_position: 3
title: "3. Forks and Knobs"
description: How a compiler choice is represented before anything is built — knobs, fork trees, schedule keys, and pinning.
keywords: [Emmy, fork, knob, schedule, tile, pin, autotuning]
---

# 3. Forks and Knobs

A **fork** is a rewrite rule saying "these options are all correct; you pick". This page is about how those options
are represented — what identifies one, how the set is built, and how you force a particular one by hand. How an option
gets *chosen* is the next page onwards.

## A knob names one choice

Every fork option is identified by the values it fixes for one or more **knobs**. A knob is a named tuning dimension:
`TILE` is a knob, `STAGE` is a knob. Nothing else identifies an option — not the order it was returned in, not an
index, not the kernel it would produce.

Take the example kernel from page one, the fused RMSNorm and query/key/value linear layer. When the tile-lowering pass
reaches it, one of the questions it faces is how large a piece of the output each group of cooperating threads should
compute. Suppose it offers four sizes:

```
TILE = 64x64      TILE = 128x64     TILE = 64x128     TILE = 128x128
```

Four options, one fork. Nothing has been built. No CUDA source exists, no kernel has been compiled, nothing has been
measured. All that separates the four is a knob value — and that is deliberate, because it means the machinery
deciding between them can look at all four without paying to construct any of them.

The knobs a schedule is made of:

| Knob | What it decides |
| --- | --- |
| `WORK` | how many workers the kernel runs and how they are arranged — a grid of warps for the tensor core path, a rectangle of threads for the plain path |
| `TILE` | which fragment of the output one worker computes, and which hardware instruction the arithmetic uses |
| `REDUCE` | how the summed axis is divided: kept serial, folded across cooperating threads, spread across registers, or split across blocks |
| `STAGE` | how inputs are moved into faster memory before use, and how deep the pipeline of in-flight copies is |
| `RASTER` | the order blocks are launched in, which changes how much of a shared input stays resident in cache |

There are a few more, including gates that trade numerical precision for speed. None of those is ever enabled
silently; each is off unless asked for.

### What the real values look like

This page is the one place the series shows the exact spellings. Values are short text with a small grammar of their
own:

```
WORK   = w2x2                              a 2x2 grid of warps
TILE   = f2x8                              each worker computes a 2 by 8 fragment of the output
STAGE  = d2/cp                             two blocks in flight, copied asynchronously into shared memory
REDUCE = g2k                               the summed axis split across 2 blocks, combined by a second kernel
RASTER = ''                                the default launch order
```

That is what a golden configuration file contains, what the tuning database stores, and what you type when you force a
choice by hand. Everywhere else in this series, values are written in a simplified form — `TILE = 64x64`, `STAGE =
depth 2` — because the grammar is not the point there. When you go looking at real files, expect the form above.

## The set of options is built lazily

A fork is rarely flat. Choosing a schedule means choosing several things whose sensible values depend on each other,
so the options form a tree: a first level fixes one group of knob values, and each branch expands into a next level
that fixes the rest. A complete option is a leaf of that tree.

The tree is built only where someone looks. A branch knows its own knob values without being expanded, and expanding
it — building the next level of children — happens when the search actually descends into it. A fork with hundreds of
leaves therefore costs almost nothing if only a few branches are ever visited.

This matters when reading later pages, because the two ways of deciding treat the tree differently. A search walks it
level by level, which is exactly what it is shaped for. An ordinary compile does not: it flattens the fork to its
complete leaves first and ranks those, because a half-decided branch — a tile whose width is fixed but not its
height — has no area and no memory footprint yet, so a model asked to score it would be guessing.

## Every finished option answers every knob

A complete option carries an explicit value for **every** declared knob, including the ones it does not use. Each knob
has a value meaning "unused here", and at the end of each pass the pipeline fills that in for any of that pass's knobs
the option left unset — whether the pass acted, declined, was skipped, or produced nothing.

The reason is worth stating, because it is the kind of rule that looks like bookkeeping and is not. The models that
rank options treat a missing value as "unknown". If a finished option could leave a knob unset, "unknown" would mean
two incompatible things: *this option does not use that knob*, and *this option has not decided it yet*. Filling in an
explicit "unused" leaves exactly one meaning — not yet decided — which is precisely the state of a half-decided branch
part-way down the tree.

## Naming a choice inside a kernel

One kernel can contain more than one step that takes the same kind of choice. Attention is the standard example: it
schedules two matrix multiplications, one producing the scores and one applying them to the values. Writing just
`TILE` would not say which was meant.

So a knob key can carry a suffix naming the step it applies to. Together, a knob name and all its suffixed forms are
called a **knob family**:

| Written | What it names |
| --- | --- |
| `TILE` | the kernel's main step — what you write whenever a kernel has only one |
| `TILE@dd` | the tile choice for the step that produces the attention scores |
| `TILE@pj` | the tile choice for the step that applies them to the values |

The **shortest unambiguous form is the canonical one**, and it is exactly what gets stored. Kernels with a single such
step store a plain `TILE`; an attention kernel stores both suffixed keys. `WORK` and `RASTER` never take a suffix at
all, because they apply to the whole kernel — one kernel has one worker arrangement and one launch order.

A stored key that becomes ambiguous, or that no longer resolves because the kernel's structure changed, is an error
the compiler reports loudly rather than a value it quietly ignores.

## Forcing a choice by hand

Any knob can be **pinned** from the environment, which makes the rule emit that one value instead of forking:

```bash
EMMY_STAGE=d2/cp emmy compile Qwen/Qwen3-Embedding-0.6B --layer 0 --target sm_89

EMMY_KNOBS="WORK=w2x2,TILE=f2x8,STAGE=d2/cp" emmy run --golden matmul.square.512 --bench
```

Two properties of pinning are easy to get wrong:

- **A pin is authoritative.** A value outside the list of options the rule would have offered is honored, not
  discarded. That is the point — it is how you explore a configuration the compiler would never reach on its own.
- **A pin does not bypass the structural checks.** Divisibility requirements, the limit on threads per block, and the
  eligibility rules for a given staging transport all still apply. A pin that violates one of them leaves the rule
  with no options at all, and whatever fallback that place in the compiler has takes over.

The second property has a sharp edge. If a pinned compile quietly falls back, you are no longer measuring the pinned
configuration — you are measuring whatever the compiler picked instead, under the pin's name. Emmy's replay paths
therefore compare the knob values the compile actually produced against the ones that were pinned, on every pinned
measurement, and fail the row rather than benchmark the fallback. That check exists because a run once reported a
flattering result that was, in fact, the compiler benchmarked against itself.

## Two kinds of fork

The forks described so far choose settings *within* one kernel. There is a second kind:

- An **ordinary fork** picks a schedule for a kernel that exists either way. Tile size, staging depth, how the
  reduction is divided. The graph is the same whichever option wins.
- A **structural fork** changes *which kernels exist*. Keeping the RMSNorm fused with the linear layer versus
  splitting them into two kernels is a structural fork; so is splitting a reduction across blocks, which produces a
  main kernel plus a second kernel that combines the partial results.

The engine tells them apart by what the option did: an option that rebinds one operation in place is an ordinary
variant, while an option that splices a new fragment into the graph is structural. They are then treated very
differently — the ranking model is never asked to compare structural options directly, for reasons the [evidence
hierarchy page](./06-deploy-evidence-hierarchy.md) explains.

## See it yourself

Print every knob the compiler knows about, with its type and the values it considers:

```bash
emmy eval knobs
```

No GPU and no measurements are needed for that — it is the schema, not data. Then compile one small matrix
multiplication and look at the schedule the compiler picked:

```bash
emmy compile --code "F.linear(torch.randn(512, 512, dtype=torch.float16), torch.randn(512, 512, dtype=torch.float16))" \
  --target sm_89 --ir tile
```

```
=== 0: k_linear_f3a56f ===
    Contraction [Σ a2] x0 @ x1 trans -> acc0 (scalar)
        for a2 in 0..256
            ...
    store: linear[a0, a1] += acc0  (atomic)
```

One kernel. The summed axis is 512 long but the loop runs to 256, and the store accumulates atomically — the
compiler split the reduction across two blocks and had them add their partial results straight into the output. Now
pin the other way of combining partial results, where each block writes its own and a second kernel adds them up:

```bash
EMMY_REDUCE=g2k emmy compile --code "F.linear(torch.randn(512, 512, dtype=torch.float16), torch.randn(512, 512, dtype=torch.float16))" \
  --target sm_89 --ir tile
```

```
=== 0: k_linear_f3a56f__partial ===
    ...
    store: linear__partial[a2_ks, a0, a1] = acc0

=== 1: k_linear_f3a56f ===
    Init(f32 acc0 = 0.0)
    for a2_ks in 0..2
        acc0__p = load linear__partial[a2_ks, a0, a1]
        ...
    store: linear[a0, a1] = acc0
```

Two kernels now, where there was one. Both compute the same matrix product. That is a structural fork, decided by one
knob value — and it is the last thing this series shows before turning to how such a value gets chosen without anyone
typing it.

Next: [4. Measuring and recalling](./04-measuring-and-recalling.md).
