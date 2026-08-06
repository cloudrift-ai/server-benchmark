---
sidebar_position: 2
title: "2. Passes and Rewrite Rules"
description: How one compiler rewrite happens — pattern, match, replacement — and what the pipeline is made of.
keywords: [Emmy, compiler, rewrite rule, pass, pipeline, graph, fusion]
---

# 2. Passes and Rewrite Rules

The previous page described the pipeline from a distance: a model becomes a graph, the graph is rewritten stage by
stage, and CUDA source comes out. This page zooms in on one rewrite. It is the mechanical layer — no tuning happens
here — and it is worth understanding first, because a fork is nothing more than one particular kind of rewrite
result.

## A rule is a pattern plus a replacement

Every rewrite rule is a small file with two things in it: a **pattern** saying what it applies to, and a function
saying what to put there instead.

The pattern names an operation type, optionally with constraints on its fields. A pattern can also be a short chain —
"a reduction whose only consumer is a division" — which matches only when each operation in the middle has exactly one
consumer, since otherwise rewriting the chain would strand the other consumers.

The engine collects every place in the graph where the pattern matches. Each place is a **match**: the specific nodes
that were found, which of them are to be removed, and which node's outgoing edges the replacement inherits. The rule's
function then receives the match and returns the replacement.

A rule may also decline. If it looks at a match and finds it cannot handle this case — an unsupported shape, a data
type it does not implement — it says so, the engine records the reason, and moves on to the next candidate rewrite.
Declining is normal and cheap.

## Three kinds of result

What the rule returns decides what the engine does with it.

**A graph fragment.** The rule builds a small graph and returns it, and the engine splices it in place of the matched
node: the fragment's inputs are wired to the existing nodes it referenced, its output takes over every consumer of the
node being replaced, and the consumed nodes are removed. This is how decomposition works. The rule for a linear layer
returns a fragment containing a matrix multiplication and an optional bias add; the linear operation itself
disappears.

**An operation, rebound in place.** The rule returns a single operation and the engine assigns it to the node it
matched, keeping the node's identity, its input list and its output tensors. Lowering works this way: the same node
becomes a tiled operation, then a kernel, then CUDA source, without ever changing its name. That matters more than it
sounds — the generated kernel's output buffer is named after the node, so a new identity would break the binding
between the kernel and the memory it writes to.

**A list of alternatives.** The rule returns several results instead of one, meaning "all of these are correct; the
machinery around me should pick". This is the **fork**. The engine turns each alternative into its own candidate and
hands the whole set to whatever is deciding — a search when tuning, a single immediate choice when compiling. Rules
never rank their own options; a rule that returns a list has finished its job. The next page is entirely about this
case.

A rule returning exactly one result — one fragment, one operation, or a list of length one — is the deterministic
case. Nothing is decided and nothing is recorded.

## Two rules the engine enforces

**Rules must be idempotent.** After a rewrite, the engine re-runs the whole pipeline over the resulting candidate from
the first pass onwards. So a rule whose own output still matches its own pattern would loop forever. Most rules avoid
this naturally, because they change the operation type — a loop becomes a tiled operation, and the pattern that
matched loops no longer fires. The ones that would otherwise re-match carry an explicit guard that declines a match
that has already been rewritten.

**A rule always sees its inputs as the graph currently holds them.** Within one round of rewriting, an earlier rewrite
can replace a node's operation with a freshly built one, and a match built before that happened would be holding stale
information. The engine therefore refreshes the operation's inputs and outputs both when the match is built and again
when it is applied. This is not a theoretical concern: reading data types off a stale operation once made a rule
believe it was looking at 32-bit values in an all-16-bit graph, so it declined the tensor core path, and a kernel
shipped at 16 times the latency it should have had.

## What the pipeline is made of

The passes run in this order. Each one is a directory of rules; the rules within a pass are applied in a fixed order
so that a compile is reproducible.

| Pass | What its rules do |
| --- | --- |
| Decomposition | Rewrite model operations — linear layers, matrix multiplications, attention, RMSNorm, softmax, layout changes — into a small set of primitives. Sibling linear layers that share one input, such as query, key and value, are merged into one wider linear layer first. |
| Optimization | Collapse chains of layout-only operations into one, so that a transpose followed by a reshape does not sit between two operations that could otherwise fuse. |
| Lifting | Wrap each surviving primitive in a loop of its own. |
| Fusion | Merge neighbouring loops into one wherever it is legal, so several primitives share a kernel and intermediate values stay in registers. Folding constant broadcasts into their consumers alone takes the example model from 394 kernels down to 337. |
| Naming and stamping | Name each remaining loop after the operations it implements (`k_rms_norm`, `k_sdpa_reduce`) and stamp onto it a summary of its body and its loop extents. That summary matters later: it is how a measurement taken elsewhere is recognized as being about *this* kind of work. |
| Tile lowering | Turn each fused loop into a tiled schedule — which worker computes which piece of the output, how the reduction is divided, which hardware instruction does the arithmetic. Most forks live here. |
| Kernel lowering | Turn the schedule into a kernel body: staging into shared memory, vectorized loads and stores, synchronization. |
| CUDA lowering | Render the body to CUDA source. |

The stamping pass deliberately runs *after* fusion, so that what gets stamped describes the final kernel rather than
one of the pieces it was built from.

## When an option does not fit

A rewrite can produce something the hardware cannot run — a tile whose shared memory requirement exceeds what the card
offers, or one that would need more threads per block than exist. Every operation is validated before it is accepted,
and one that fails validation is dropped.

Dropping is the right response during a tuning search, where the same fork has sibling options carrying other tile
sizes and the search simply continues with those. It is the wrong response during an ordinary compile, where there are
no siblings: dropping the only rewrite leaves the node un-lowered, and the failure would surface much later as a
confusing type error inside the backend. So an ordinary compile records every drop with its reason and, once the graph
settles, raises a clear error naming the node that never became a kernel. It also retries first — that mechanism is
described on [the deploy evidence hierarchy page](./06-deploy-evidence-hierarchy.md).

## See it yourself

Ask the compiler to show every rewrite it applies:

```bash
emmy compile Qwen/Qwen3-Embedding-0.6B --layer 0 --target sm_89 -vv
```

Each application prints as a difference between the matched piece of the graph and its replacement, bracketed by
markers naming the pass and the rule:

```
>>> f:020_merge_loop_ops
<<< f:020_merge_loop_ops
```

The letter is the pass: `d` decomposition, `o` optimization, `l` lifting, `f` fusion, `s` stamping, `t` tile
lowering, `k` kernel lowering, `c` CUDA lowering. Because the markers bracket each block, one pass or one rule can be
sliced out of the output:

```bash
emmy compile Qwen/Qwen3-Embedding-0.6B --layer 0 --target sm_89 -vv | awk '/^>>> f:/,/^<<< f:/'
```

A single `-v` is quieter: pass timings and how many times each rule applied, without the differences. And `--passes`
runs a prefix of the pipeline, which is the quickest way to see what a stage receives as input:

```bash
emmy compile Qwen/Qwen3-Embedding-0.6B --layer 0 --target sm_89 --passes dolf --ir loop
```

Next: [3. Forks and knobs](./03-forks-and-knobs.md).
