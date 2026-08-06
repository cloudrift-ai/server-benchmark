---
sidebar_position: 1
title: "1. How a Model Becomes GPU Code"
description: The path from a traced PyTorch model to CUDA kernels, and where the compiler has to choose between correct answers.
keywords: [Emmy, compiler, CUDA, kernel, GPU, PyTorch, lowering, pass]
---

# 1. How a Model Becomes GPU Code

This is the first page of a series about Emmy's compiler. It starts here, with what the compiler does at all, and
ends several pages later inside the model that predicts which GPU code will be fast.

You do not need to have read any Emmy code to follow it. You do need to be comfortable with the idea of a model as a
chain of tensor operations, and with a few GPU words — kernel, thread, block, shared memory. The vocabulary used
throughout is defined in the repository's `GLOSSARY.md`; anything beyond it is explained where it first appears.

## The path a model takes

Emmy takes a PyTorch model and produces CUDA source it can launch. It does that in three stages, each one a bit
closer to the hardware than the last.

```
PyTorch model
   │   tracing — run the model on example inputs and record what it did
   ▼
a graph of tensor operations              ← matrix multiplication, RMSNorm, softmax, add …
   │   decomposition, fusion
   ▼
a graph of fused loops, one per kernel    ← "these operations run together, over these axes"
   │   tile lowering, kernel lowering
   ▼
a graph of CUDA kernels                   ← actual source, compiled and launched
   ▼
GPU
```

Each arrow is a **pass**: one phase of the compiler that looks for a pattern and rewrites it into something closer to
the target. A pass is made of **rewrite rules**, and a rule is small — it recognizes one thing and replaces it. The
ordered list of passes is the **pipeline**, which is what this series is named after.

Two properties of the middle stage are worth holding on to, because everything later depends on them:

- **Operations are decomposed, then fused back together.** A model operation such as RMSNorm is first broken into
  primitives — multiply, sum, reciprocal square root, multiply again — and neighbouring primitives are then merged
  into one loop. Decomposing first and fusing after means the compiler is not limited to the groupings PyTorch
  happened to use.
- **One fused loop becomes one kernel.** After fusion, each remaining loop is lowered on its own into a kernel. So
  "how are operations grouped into loops" is the same question as "which kernels will exist".

## The example we will follow

One example runs through the whole series. It is a real one: inside a single layer of the embedding model
`Qwen/Qwen3-Embedding-0.6B`, an RMSNorm is followed by the linear layer that produces the query, key and value
tensors that attention consumes.

```
x ──▶ RMSNorm ──▶ linear (weights for query, key, value) ──▶ q, k, v
```

By the time the compiler is done with it, this is one kernel. Getting there involves, in order:

1. decomposing RMSNorm and the linear layer into primitives;
2. fusing them into one loop, so the normalized values never travel to global memory and back;
3. deciding *how* that one kernel computes — which piece of the output each worker owns, how the inputs are staged
   into faster memory, how the sum inside the linear layer is divided up;
4. deciding whether keeping them in one kernel was a good idea at all, or whether two kernels would be faster.

Steps 1 and 2 have one right answer each. Steps 3 and 4 do not, and that is what the rest of this series is about.

## Where the freedom is

Most rewrites are **deterministic**: there is one correct result and the rule returns it. Turning a linear layer into
a matrix multiplication over primitives is like that. So is naming a kernel after the operations it implements.

A handful of decisions are different. Consider step 3 above. The kernel has to compute a large output, and a GPU
computes it in **tiles** — small rectangular pieces, each handled by one group of cooperating threads. How big should
a tile be? A larger tile does more arithmetic per value it loads, which is good, but it needs more registers and more
shared memory, which limits how many groups can run at once. There is no answer that is right everywhere. It depends
on the shape of the matrices, on how much shared memory the card has, on how fast its memory is relative to its
arithmetic units, and on what the rest of the kernel is doing.

The same is true of the other decisions at that level:

| Decision | The trade-off |
| --- | --- |
| Tile size | arithmetic per loaded value, against registers and shared memory used |
| Staging | copying inputs into shared memory first costs instructions and memory, and saves repeated global reads |
| Splitting a reduction | more blocks working in parallel, against the cost of combining their partial results |
| Which workers cooperate | threads, warps, or the tensor cores — each wants a different arrangement of the data |

All of these compute exactly the same numbers. They are not correctness choices, they are **schedule** choices: the
plan for running correct mathematics on hardware. And the gap between a good schedule and a bad one is not a few
percent. On a shape where the compiler once chose badly, the deployed kernel was measured at **29 times** the latency
of the configuration that should have been chosen.

So the compiler cannot simply pick. It has to have a way of knowing.

## The question the rest of the series answers

When a rewrite rule has several correct answers, it returns **all** of them. That return is called a **fork**, and
deciding forks is what most of Emmy's compiler machinery exists to do. There are two very different situations:

- **`emmy tune` has a GPU and time to spend.** It can build kernels, run them, measure them, and remember what it
  learned.
- **`emmy compile` and `emmy run` measure nothing.** An ordinary compile has to answer every fork immediately, from
  what was recorded earlier — and it may be running on a machine that has never measured anything at all.

The pages that follow work through that, in order:

| Page | What it covers |
| --- | --- |
| [2. Passes and rewrite rules](./02-passes-and-rewrite-rules.md) | how one rewrite happens, and what the pipeline is made of |
| [3. Forks and knobs](./03-forks-and-knobs.md) | how a choice is represented before anything is built |
| [4. Measuring and recalling](./04-measuring-and-recalling.md) | the two ways a fork gets answered, and where knowledge is kept |
| [5. Inside a tuning run](./05-inside-a-tuning-run.md) | what `emmy tune` actually does |
| [6. The deploy evidence hierarchy](./06-deploy-evidence-hierarchy.md) | the fixed order an ordinary compile works down |
| [7. Golden configurations](./07-golden-configurations.md) | the reviewed measurements that ship with the repository |
| [8. Inside the prior](./08-inside-the-prior.md) | the model that ranks options when nothing was measured |
| [9. Storage, checks and limits](./09-storage-checks-and-limits.md) | how it is all stored, how it is checked, where it falls short |

## See it yourself

Every stage in the diagram above can be written to disk. This compiles one layer of the example model and saves the
graph after each pass:

```bash
EMMY_DUMP_DIR=/tmp/emmy-dump emmy compile Qwen/Qwen3-Embedding-0.6B --layer 0
```

The files are numbered in pass order, so the graph before and after each phase sits side by side, and the generated
CUDA source is in there too. Nothing has to run on a GPU for this — you are reading the compiler's output, not
executing it. If the machine has no CUDA device at all, name the card you want the compiler to target so that passes
gated on hardware features take the same path they would there:

```bash
emmy compile Qwen/Qwen3-Embedding-0.6B --layer 0 --target sm_89 --ir tile
```

`--ir` prints a single stage instead of saving everything. The stages are `torch`, `tensor`, `loop`, `tile`, `kernel`
and `cuda` — the same sequence as the diagram, at increasing levels of detail.

Next: [2. Passes and rewrite rules](./02-passes-and-rewrite-rules.md).
