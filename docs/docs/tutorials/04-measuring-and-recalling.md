---
sidebar_position: 4
title: "4. Measuring and Recalling"
description: The two ways a fork is answered — measure it now, or recall what was measured before — and the four places knowledge is kept.
keywords: [Emmy, autotuning, greedy selection, evidence, reservoir, tuning database, regime]
---

# 4. Measuring and Recalling

A fork offers several correct options. Something has to pick one. There are exactly two situations in which that
happens, and they could hardly be more different.

## Two situations

**`emmy tune` has a GPU and time to spend.** It can take a fork, build a kernel for one of its options, run that
kernel, time it, and keep the number. It explores, it compares, and it writes down everything it learns. A tuning run
takes minutes to hours.

**`emmy compile` and `emmy run` measure nothing.** An ordinary compile has to produce a program now. It cannot build
four kernels to find out which is faster; it picks one, in a fraction of a second, and moves on to the next fork.
Choosing the option that currently looks best without exploring alternatives is called **greedy selection**, and it is
what every deployment does.

The interesting problem is the second one. An ordinary compile can only *use* knowledge; it can never create any. So
everything depends on what was recorded earlier, and on where.

## The four stores

Four stores hold everything a compile can know. Telling them apart is the single most useful thing to learn early,
because they have different writers, different readers and different lifetimes.

| Store | Where it lives | Written by | Read by |
| --- | --- | --- | --- |
| **Golden configurations** | model files under `recipes/<model>/golden/`, one per exact GPU; model-agnostic files under compiler search | promoted from measured comparisons | an ordinary compile, first of all; also the training data for the offline prior |
| **Reservoir** | inside the online prior's checkpoint, `~/.cache/emmy/online.json` | `emmy tune`, every training row | the online prior's own training; and an ordinary compile, for the rows measured at deployable settings |
| **Measurements table** | the tuning database, `~/.cache/emmy/autotune.db` | `emmy tune`, one row per benchmarked kernel | an ordinary compile, after the two above; and as a cache, so a configuration already measured is never re-run |
| **Search-tree table** | the same database | `emmy tune`, one row per point in its search; also `emmy run --bench` for hand-forced measurements | the `emmy eval` diagnostics only — **never** consulted when compiling |

The last row surprises people. The search-tree table is the richest data Emmy has — it records not just the winners
but every position the search visited, including the failures — and it is deliberately not consulted when deciding
what to deploy. It exists to answer questions about the search itself, which is what the [last
page](./09-storage-checks-and-limits.md) is about.

```
WRITERS                                     STORES                                READERS

emmy tune ─┬─ each benchmark ─────────────▶ measurements table ────────────────▶ ordinary compile
           ├─ each training row ──────────▶ reservoir  ────────────────────────▶ ordinary compile, online prior
           └─ each search position ───────▶ search-tree table ─────────────────▶ emmy eval only

emmy run --bench, hand-forced rows ───────▶ search-tree table

recorded by hand from those rows ─────────▶ golden configuration files ────────▶ ordinary compile
                                                        └── emmy fit ─────────▶ offline prior weights ──▶ ordinary compile
```

## Only one of them travels

Of the four, **only the golden configurations are in the repository**. The reservoir and the tuning database are
caches under `~/.cache/emmy` on whichever machine ran the tuning.

That has a consequence worth pausing on. A freshly rented GPU box has: the golden configuration files, and the
weights of the offline prior that also ship with the repository. It has no measurements of its own, and nothing local
to fall back on. Every fork on that machine is answered either by a recorded golden configuration or by a model's
prediction. This is the normal case, not an edge case — it is what happens every time somebody rents a machine to
serve a model. It is also why the golden files matter as much as they do, and why they get [a page of their
own](./07-golden-configurations.md).

## Measurements are not interchangeable

One more thing has to be introduced here, because everything after this page depends on it: a measurement is only
true of the settings it was taken under. Those settings are called the **regime**, and the part of it that matters
most is the optimization level the CUDA compiler ran at.

`-O3` is the **deployable** setting. It is what `emmy compile` and `emmy run` use, so it is what a served model
actually runs — and it is what a tuning sweep measures at too. **Emmy tunes in the regime it deploys into**, so a
tuned latency is the latency you get.

That sounds too obvious to state, so it is worth saying why it needs stating. A sweep benches thousands of
configurations, and a cheaper compiler setting (`-Xcicc -O1`) was once used to make that affordable, on the
assumption that it would still *rank* correctly even if the absolute numbers were off. It did not. The cheap
setting's error was not random noise but a systematic bias along tile size: it made big register tiles look slow,
which is exactly the family it was most important to get right. Ranking in a regime you do not deploy in means the
winner of the search need not be the winner in production.

The general lesson outlives the specific setting: **a measurement is only evidence about the conditions it was taken
under.** Emmy therefore keeps the regime on every stored measurement and gates on it, so a number taken under some
other setting is never silently read as if it applied here. If you deliberately pin a different optimization level
with `--nvcc-flags`, the sweep still runs and still records — but under that regime's own identity, where no ordinary
compile will read it. You will get a warning saying so.

## Where this is going

The next page follows a tuning run and shows what it produces. The page after that is the one that answers the
question the series opened with — given all of this, in what order does an ordinary compile consult it?

## See it yourself

Look at what a tuned machine actually has:

```bash
ls -la ~/.cache/emmy/
```

Model golden configuration files live beside their recipes, one file per exact GPU model and compute capability:

```bash
find recipes -path '*/golden/*.yaml' -print
```

The central `emmy/compiler/pipeline/search/goldens/` directory contains only model-agnostic hardware goldens.

If a tuning database exists, the measured configurations for each kernel can be listed as a table, best first, with
the one an ordinary compile would choose marked:

```bash
emmy eval variants
```

That command reads the database only — it runs no kernels and needs no GPU.

Next: [5. Inside a tuning run](./05-inside-a-tuning-run.md).
