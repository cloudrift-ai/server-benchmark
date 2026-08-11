# Pipeline Architecture

The pipeline is the part of the compiler that turns a traced graph into finished CUDA kernels, one rewrite at a time.
This document explains it end to end for someone new to the code. It assumes you know the shared vocabulary in
[`GLOSSARY.md`](../../../GLOSSARY.md) — fork, knob, candidate, prior, evidence, golden configuration — but nothing
about the internals of this package. Words that carry a special meaning inside the pipeline are explained in plain
language where they first appear; the few that also turn up in neighboring documents are in the glossary.

Four companion documents cover what this one doesn't:

- The rewrite rules themselves and their authoring invariants → [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md).
- What each IR dialect looks like → `ir/ARCHITECTURE.md`.
- The shipped bugs and retired designs behind the stricter rules below → [`HISTORY.md`](HISTORY.md). This file states
  each rule and its reason briefly; where the reason was a production incident, the full story lives there.
- A gentler, worked-example introduction to the same material, written for someone who has not read the code → the
  Tutorials section of the course site (`docs/docs/tutorials/`). It covers forks, the deploy evidence hierarchy, the
  prior and the goldens in nine short pages; this file remains the reference the tutorials defer to.

**How to read this.** "The big picture" below is the mental model everything else refines — read it first, in order.
After that the Parts are largely independent:

| Part | Covers | Read it when |
|------|--------|--------------|
| 1 | The rewrite engine: patterns, rule contract, splicing | you are writing or debugging a rule |
| 2 | Forks: how a rule's alternatives are represented | a fork's options look wrong or incomplete |
| 3 | The prior and **the deploy evidence hierarchy** | you need to know why a compile picked what it picked |
| 4 | The two drivers (`tune`'s search loop, the greedy compile) | you are tracing control flow through a compile |
| 5 | The tuning workflow (`emmy tune`) | you are running or extending autotune |
| 6 | Persistence: the two identities, the tables, the freeze | you are adding a cache, a table, or a column |
| 7 | Golden configs and the A/B integrity gates | you are recording or auditing goldens |
| 8 | Evaluating the prior (`emmy eval`) | the prior is picking badly and you want to know where |
| 9 | Tile lowering, at the pipeline level | you want the pipeline-side view of `lowering/tile/` |

The reference sections after Part 9 — "Tunable knobs", the pass table, the dump hooks — are lookup material, not
reading material.

## The big picture

The compiler lowers a graph by repeatedly applying small rewrite rules. Most rewrites are deterministic: there is one
right answer and the rule just returns it. Some choices are not. Tile sizes, staging depth and split-K all have many
valid answers, and which one is fastest depends on the GPU and the shapes. A rule facing such a choice returns *all*
its options. That return is a **fork**, and the machinery around the rules — never the rule itself — decides which
option wins.

Almost everything in this package exists to answer one question well: *at each fork, which option do we take?* There
are two fundamentally different situations.

**`emmy tune` has a GPU and time to spend.** A Monte-Carlo tree search (MCTS) explores the fork tree, benchmarks real
kernels, records every measurement in a SQLite database, and trains the **online prior** — a CatBoost model that
predicts kernel latency from a variant's features (Part 5).

**`emmy compile` / `emmy run` (a "greedy" compile) benchmarks nothing.** Every fork is decided on the spot from
knowledge recorded earlier, in a fixed order — best evidence first:

1. **Golden configurations** recorded for the GPU being compiled for — reviewed measurements that ship with the repo
   (Part 7).
2. **Measured evidence** — first the measurements stored inside the online prior's checkpoint (its **reservoir**) that
   were taken at deployable flags, then rows from the tune database (Part 3).
3. **The prior** — the online model when trained and calibrated, the offline model otherwise (Part 3).
4. **Option-0** — the first option in the order the rule emitted them. Rule authors order options so this is always
   safe.

That order has a name — the **deploy evidence hierarchy** — and each numbered step in it is called a **tier**. The
list above is only a summary. **Part 3's "The deploy evidence hierarchy" is the authoritative statement** of the exact
order, of what each tier holds, and of the rule that tiers 1 and 2 apply only to a compile at deployable `-O3` flags.

Structural forks — the ones that change which kernels exist — are stricter. The prior never ranks their options
directly, and without a trusted online prior the kernel set stays at its default (Part 4).

### The four stores

Four stores hold everything a compile can know. They have different writers, different readers and different
lifetimes, and telling them apart is the single most useful thing to learn early.

| Store | Where it lives | Written by | Consulted by |
|-------|----------------|------------|--------------|
| **Golden configs** | per-GPU YAML files under `search/goldens/`, checked into the repo | promoted by hand from deployable `run --bench` golden / `--ab` rows (Part 7) | greedy compile (tier 1); `emmy fit` trains the offline prior on them; `emmy eval golden` |
| **Reservoir** | inside the online prior checkpoint (`~/.cache/emmy/online.json`) — the sample of past measurements the model trains on | `emmy tune` — every training row, including the `-O3` re-benches | greedy compile (tier 2, its `H_opt=3` rows); the online prior's own refits |
| **`perf` table** | the tune DB (`~/.cache/emmy/autotune.db`) | `emmy tune` — one row per benched kernel, at whatever flags the sweep ran | greedy compile (tier 3); the per-variant replay cache |
| **`node` table** | the same tune DB | `emmy tune` (every search-tree node) and `run --bench` (rows benched with hand-forced knob values) | `emmy eval` diagnostics — **never** consulted at deploy |

Of the four, only the goldens travel with a clone: they are the only *measured* data a fresh machine has. The
reservoir and the tune DB are machine-local caches written by local tunes, so a freshly rented box starts with the
goldens plus the shipped offline prior artifact and nothing else.

```
WRITERS                                STORES                                READERS

emmy tune ─┬─ sweep benches ─────────▶ perf table   (autotune.db) ─────────▶ greedy compile, tier 3
           ├─ every training row ────▶ reservoir    (online.json) ─────────▶ greedy compile, tier 2 (H_opt=3 rows)
           ├─ -O3 re-benches ────────▶ reservoir + node table                online prior refits
           └─ every tree node ───────▶ node table   (autotune.db) ─────────▶ emmy eval only (never a deploy)
run --bench pinned/golden/--ab rows ─▶ node table   (autotune.db)
recorded by hand from those rows ────▶ search/goldens/*.yaml (repo) ───────▶ greedy compile, tier 1
                                                                  └─ emmy fit ─▶ offline_weights.json (repo)
                                       offline_weights.json ──────────────▶ greedy compile, tier 4 (cold)
                                       online prior model (online.json) ──▶ greedy compile, tier 4 (trusted)
```

One asymmetry trips people up: the `-O3` re-benches a tune runs never reach the `perf` table. On a machine tuned at
the default `-Xcicc -O1` flags, the only measurements taken at deployable `-O3` flags live in the reservoir (Part 3).

### How one fork gets decided, end to end

A worked example, to fix the vocabulary. Take `emmy compile` on a machine with a tuned checkpoint. A tile-lowering
rule matches a `LoopOp` and returns several tile options.

1. The engine turns the option list into a lazy fork tree and hands the fork point to `greedy_decide` (Parts 2, 4).
2. `greedy_decide` **flattens** the fork to its complete leaves — knob dicts only; no kernel is built yet (Part 4).
3. Each leaf becomes one row: the compile context's `H_*` features (which GPU, which nvcc flags), the `S_*` features
   an earlier pass wrote onto the op (a summary of its body and loop extents), and the leaf's complete knob values
   (Part 6).
4. **Tier 1, goldens.** The op is joined by `ShapeKey` against the goldens recorded for this GPU. The winner is the
   first leaf that agrees with the fastest recorded entry on every knob the leaf has decided. Deployable `-O3` flags
   only.
5. **Tier 2, reservoir.** Otherwise: the leaf that agrees the same way with the fastest reservoir row of the same op
   that was itself measured at `-O3` (`H_opt=3`). Deployable flags only.
6. **Tier 3, `perf` rows.** Otherwise: measured rows for this exact op — a row measured at deployable flags decides
   ahead of a row measured at the `-Xcicc -O1` flags a tune sweep uses.
7. **Tier 4, the prior.** Otherwise: the `mean_scores` argmin over all leaves, in one batched predict.
8. Ties at every tier break by `knob.canonical_row_key`, never by the order the rule emitted its options in.
9. The winning leaf is built for real. The µs of whichever row decided it is written onto the fork's
   `Decision.score`, and the resolve moves to the next fork.

With no evidence and no prior at all, step 4 still runs — the golden tier needs no prior — and every fork it does not
answer falls to option-0.

### Terms used throughout

Everything in this table recurs on nearly every page below. The rest of the document uses these words freely.

| Term | Meaning |
|------|---------|
| **rule** | One pattern + rewrite function in a `NNN_<name>.py` file under a pass directory. |
| **pass** | An ordered directory of rules; the pass layout is frozen in a `Pipeline`. |
| **candidate** | One in-flight compilation state (a graph snapshot part-way through the pipeline). |
| **fork** | A rule returning multiple alternatives; the engine turns each option into a child candidate. |
| **knob** | A named tuning dimension (e.g. `TILE`, `STAGE`). Every fork option is identified by the knob values it fixes. |
| **to pin a knob** | To force a knob's value by hand instead of letting the compiler choose — from the environment (`EMMY_STAGE=d2/cp`), or by reproducing a golden entry's recorded values. A *pinned row* is a benchmark of such a forced configuration. |
| **to stamp a value** | To write a value onto an op as metadata, where later passes and the prior can read it: the `S_*` shape/body features, knob values, scheduler facts. "The op's stamped `S_*` features" means the ones an earlier pass wrote onto it. |
| **to realize** | A recorded configuration *realizes* at a fork when the options the compiler actually offers there include one that matches it. A recording that realizes nowhere cannot be deployed, no matter how good its recorded µs. |
| **regime** | The compile settings a measurement was taken under, or that a compile is running under: mainly the nvcc optimization level (`H_opt`) — `-O3` is the **deployable** regime, `-Xcicc -O1` the fast-compiling one a tune sweep uses — plus whether fast math is on. |
| **prior** | The ranking model — the fit-offline **offline prior** when cold, the CatBoost **online prior** trained from local measurements once data exists. |
| **terminal** | A fully-lowered candidate (every fork on its path resolved) that can be benchmarked. |
| **golden record** | A reviewed program-backed schedule measurement, selected by frontend provenance and used as deploy evidence and an A/B reference. |
| **`Op.cache_key`** | A name-invariant digest of an op's body + knobs — the identity measurements are stored under. A `TileOp`'s structure digests as the α-invariant term hash (`Fold.structural_key`), never the lowered nest. |

## Module map

| Module | What lives there |
|--------|------------------|
| `pipeline.py` | Engine core: `Pattern` / `Match` / `Rule` / `Pass` / `Pipeline` (the frozen pass layout) plus `Run` — the per-run state and engine loop. |
| `fork.py` | The `Fork` interface (`OptionFork`, `ThunkFork`) and the reusable `Level` + `build_fork_tree`, which builds a tree of knob-value combinations lazily. |
| `knob.py` | The `Knob` descriptor system and the `EMMY_<KNOB>` env namespace (borrowing `config.knob_var` / `config.knob_raw`; `format_tuning_knobs` renders the real tuning knobs for `tune` output). Holds NO concrete knob declarations. |
| `search/space.py` | **The single home of the search space.** Every `Knob` instance is declared here and nowhere else — the schedule codecs (`WORK` / `TILE` / `REDUCE` / `STAGE` / `RASTER`), the kernel-lowering policy knobs (`VECTORIZE_LOADS` / `INTERLEAVE_LOADS`), and the enumeration value grids (`scalar_tile_moves` & co). A rule that decides a knob imports it from here; registration is construction (`Knob.__post_init__`), and `knob.registry()` imports `space.py` before answering, so the registry is complete in any process. |
| `search/domain.py` | The candidate domain as a **constrained integer set** — `Dimension` (a name + its finite integer values), `Bound` (`coeff · ∏ dims` `<=` / `==` / `divides` a limit) and `Space` (enumerate the legal points, or ask whether a recorded one is still a member). The constraints that bound a schedule family are products of the unknowns, so the feasible set is not convex and no coordinate change makes both the products and the budgets affine at once; the answer is to keep integer coordinates and enumerate, pruning each prefix the moment a running product overruns its bound. Generation machinery only — it holds no schedule family today (`space.py`'s grids are still curated), and categorical legality stays with the scheduler. |
| `search/features.py` | The featurizers (`knob_features`, `tile_signature`, the `D_*` / `MMA_*` encodings) — kept beside `space.py` so the whole space (dimensions × values × encoding) is analyzable in one package. |
| `search/db.py` | `SearchDB`, the persistent SQLite store (Part 6). |
| `search/policy/mcts.py` | The in-memory MCTS (`SearchTree`) colocated with its only reader, `TuningSearch`. |
| `search/policy/greedy.py` | `greedy_decide` — the no-tree fork resolver used by `compile` / `run`. |
| `search/two_level.py` | The two-level tuner: outer structural MCTS, inner per-op reward. |
| `search/prior/` | The ONE ranking path: a `Prior` ABC with the cold `OfflinePrior` and the `OnlinePrior` composed behind `FallbackPrior` (`load_prior`). `diagnostics.py` here backs the `eval` reachability / calibration reports; `fit/` is the offline fitter, split by responsibility — `group.py` data representation, `linear.py` trainer+model, `rank.py` rank metrics, `cv.py` fold harness, `run.py` the pure `emmy fit` run harness. |
| `search/data/` | The harmonized read-view over the three data sources (golden records / DB `perf` rows / prior reservoir): `Sample`, `Dataset`, and the derived `ShapeKey` index. |
| `search/golden.py` | Generic program-backed records, repository indexing, stable-format validation, and lazy provenance-derived structural indexes (see Part 7). |
| `search/audit.py` | The golden drift audit: compile graphs with the golden tier as the only evidence, one MATCH / DRIFT / GAP verdict per consulted fork (via `greedy.golden_audit`, the supported sink; records also carry `unrealized`, the per-entry pin-only signal). Backs `emmy eval golden` (the pin-only offer audit), `--in-model`, and serving-image release qualification (see Part 7). |
| `slice.py` | Isolates one finalized kernel into a standalone graph (used by the inner tune and structural pricing). |
| `dump.py`, `rule_diff.py` | The dump and `-vv` presentation layers (see the end of this file). |
| `passes/{frontend,loop,lowering}/` | The rules themselves — documented in [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md); a per-pass overview table is near the end of this file. |

## Part 1: The rewrite engine

This Part covers the mechanics every rule author touches: how a pattern matches, what a rewrite may return, and how
the engine splices the result back. Nothing here involves tuning.

### Patterns and matching

A `Pattern(name, op_type, constraints={})` matches one node by op type plus optional `node.op` field equality. A *list*
of patterns matches a chain: the seed node matches `pattern[0]`, its sole consumer matches `pattern[1]`, and so on.
Multi-node patterns only fire when each intermediate node has exactly one consumer.

`match_pattern(graph, pattern) → list[Match]` walks every topo-ordered seed. Overlapping matches are allowed — the
rewriter exits after the first successful rewrite per iteration, so overlap is just candidate enumeration.
`Match.nodes` maps each pattern entry's name to the matched `Node`. `Match.consumed` and `Match.output` are
overridable by the rewrite function, to control which nodes the splicer removes and which node's edges get rewired.

### Writing a rule

Every file named `NNN_<name>.py` under a pass directory is a rule:

```python
PATTERN = [Pattern("root", SomeOp), ...]  # required


def rewrite(ctx: Context, graph: Graph, match: Match) -> Graph | Op | list[Graph | Op]: ...
```

- The dispatcher binds `rewrite`'s parameters **by name**. Reserved names: `graph`, `match`, `root`, `out`, `ctx`.
  Pattern names from `PATTERN` bind to their matched `Node` objects. Anything else binds positionally to
  `root.inputs[i]`. Take only what you need — `ctx` is optional.
- Files starting with `_` (e.g. `_broadcast.py`) are **not** loaded as rules — they're shared helpers.
- Raise `RuleSkipped(reason)` to decline a match; the engine logs the reason at DEBUG and moves on.

A rule always sees **graph-true operand Tensors**: op `inputs` / `outputs` are refreshed from the graph at match build
AND again at apply time (`Candidate.try_rewrite`). This matters because an earlier apply in the same batch may have
swapped a consumed node's op for a rebuilt instance still carrying its `(f32, ())` seeding placeholders — a change
`Match.is_alive`'s node-identity check cannot see. (This was the gemma o_proj misdeploy: a scalar tile shipped at 16x
the kernel's measured mma rows because the warp atom gate read placeholder dtypes off an all-f16 graph — HISTORY.md:
"Placeholder dtypes read off a rebuilt op".)

### The three kinds of rewrite result

The return type discriminates the rewrite flavor:

- **Functional** — returns a `Graph` fragment, spliced in place of `match.output` (defaults to `match.root_node_id`).
  Fragment `InputOp` nodes reference existing graph nodes by id; non-Input nodes get fresh ids.
- **In-place** — returns an `Op`. The engine assigns it to `root.op` directly, preserving the node id, inputs list,
  output Tensor and hints. The lowering rules use this because `KernelOp.arg_order` / `CudaOp.arg_order` embed the
  original node id as the output buffer name — a fresh id would break the generated kernel's buffer binding.
- **List = autotune fork.** A rule unsure which parameter to use returns the alternatives as a list, in any order. The
  engine spawns one `LazyCandidate` per option (sharing the parent's graph snapshot) and hands them ALL to a `Search`
  policy, which ranks them via a `Prior` (Part 3). A single-option return (or a bare `Graph` / `Op`) is the
  deterministic case — no fork.

### Rules must be idempotent

Every rule MUST be idempotent on its own output. The engine re-runs the entire pipeline on each popped candidate from
pass 0, so a rule whose output is already in the graph must `RuleSkipped` or have a pattern that no longer matches.
Most rules satisfy this implicitly via op-type changes (`LoopOp` → `TileOp`); the rest carry explicit
`raise RuleSkipped("already X")` guards.

### How fragments are spliced in (`engine._apply_replacement`)

1. Walk the fragment in topo order. `InputOp` nodes forward their id to the existing graph node (external reference);
   non-Input nodes are added with fresh ids.
2. `replace_node(match.output or match.root_node_id, new_output)` rewires all consumers (and `graph.outputs` slots)
   from the old output to the fragment's output id.
3. Merge hints from every consumed node into the new output.
4. Remove consumed nodes and run `_remove_orphans` to drop any now-dangling constants / inputs.

## Part 2: Forks — how choices are represented

A fork is how a rule says "these N options are all correct; you pick". This Part covers how the options are
represented, what identifies one, and what happens to an option that turns out to be invalid. Part 3 covers how one
gets chosen.

### Lazy hierarchical forks

A fork with many options would be expensive to build eagerly, and most options are never visited. So forks are lazy
trees. `Fork` (`fork.py`) is an interface with three members:

- `knobs` — the knob values this fork level fixes. Those values are the variant's identity: both the perf DB and the
  prior are keyed on them, and they can be read **without expanding** the fork.
- `is_leaf` — whether this is a concrete option or an inner branch.
- `expand()` — builds the next level of options.

The search loop pops a `LazyCandidate` waiting on a fork, calls `expand()` to build the children, pushes them back and
continues, so only the subtrees the search actually walks into ever get built. `OptionFork` is a concrete `Op` /
`Graph` leaf. `ThunkFork` is a generic flat fork: `expand_fn(knobs)` is a function of the fork's own knob values, so
all its siblings share one function.

A fork whose levels form a cartesian product of knob values reuses **`build_fork_tree`**. A rule supplies one `Level`
per level plus a `materialize=` callable, and gets back a lazy root `_Branch` whose `expand()` builds children on
demand, in grouping order. The algorithm — group the parameters by each level's knob keys, collapse a level with one
key, skip a level with no keys, and defer building a leaf until `expand()` — lives once in `fork.py`. A one-shot flat
fork stays inline as a `ThunkFork`.

### Every finished option carries a value for every knob

**Every emitted variant carries an explicit value for every declared knob** — no complete option leaves a knob absent.
This rule is known in the code as the knob-stamp invariant.

Each `Knob` declares an `off` value, meaning "unused here / this pass declined it". At the end of each pass
(`Cursor.advance` → `_off_fill_pass`, via `knob.apply_off_defaults`), the pipeline fills in that `off` value for any
of *that pass's* knobs the variant left unspecified. It covers a pass that acted, one that declined, one that was
skipped and one that returned no variants, all the same way. Filling only the just-finished pass's knobs is
deliberate: writing a later pass's knob early would trip that pass's idempotency guard.

Why it matters: the online prior fills any missing feature column with NaN. Because a finished option always writes an
explicit "off", NaN can mean exactly one thing — *not yet decided*, i.e. a partly-decided option seen part-way down
the fork tree — and never "decided: unused", which is the explicit off value on a complete option. A knob with no
`off` value (the `_UNSET` default — a knob its owning pass always writes itself) is never auto-filled. Which code path
a variant belongs to is always read off knob *values* (`knob.is_warp` / `knob.mma_atom`), never off a knob's presence.
Verified by `tests/compiler/passes/test_knob_stamp_invariant.py`.

### Invalid options: rewrites that get filtered, and rewrites that raise

A rewrite that *returns* an op failing `Op.validate(ctx)` — e.g. a `KernelOp` whose smem exceeds
`ctx.max_dynamic_smem` — is dropped by `Candidate.try_rewrite`. Dropping it is right during a search, where sibling
branches carry other tile shapes, but fatal in a single-path greedy compile, where it leaves the node un-lowered. So:

- `Pipeline.run` installs a `rejections` sink on the `Run`, recording each drop as `(node, pass, reason)`. After the
  terminal settles, `_raise_on_unlowered` raises a loud `LoweringError` naming any still-un-lowered node, instead of
  leaking a cryptic `non-CudaOp` `TypeError` to the backend.
- The sink is absent under `tune`, so dropping options during a search stays silent there.

A rewrite that *raises* mid-lowering — a deterministic pass hitting an un-representable shape — is the same dead end
expressed as an exception. Greedy `resolve` lets it propagate. Under `tune`, `Run.drive` catches it per-candidate,
drops that subtree and bumps `Run._dropped_candidates`. Without this, one search-only un-lowerable fork aborted the
whole tune.

## Part 3: The prior — how choices are ranked

This Part answers "how does a fork get decided when nothing may be benchmarked?". Its core is **the deploy
evidence hierarchy**: the fixed order of tiers a greedy compile walks, best evidence first. The sections before it
explain the machinery that order leans on; the two after it are the guards that keep the machinery honest.

### One ranking path

Ranking always happens in one place: the search policy asks a single `Prior`. Forks carry no score of their own, and
nothing builds or scores a `TileOp` merely to rank it — the `Prior` turns the row's knob values straight into features
(`features.knob_features`). Several older per-variant scoring mechanisms were removed in favor of this single path
(HISTORY.md: "Retired designs").

That one path has two halves: the `OfflinePrior` that ships with the repo, and the `OnlinePrior` that a local tune
trains. `FallbackPrior` composes them, and `load_prior` builds the composite. The offline half is called the *cold*
one because it is what answers on a machine that has no local measurements yet — a freshly rented box, say.

### The offline prior (the cold half)

`OfflinePrior` scores a candidate with a linear formula over the `D_*` features — hand-designed descriptions of a
tile's geometry and its occupancy — fitted ahead of time. It never falls back on the order the rule emitted its
options in. The complete scoring function lives in the repo-checked artifact `search/prior/offline_weights.json`:
both weight sets plus the scalar params, carrying a `feat_ver` version and a `provenance` block. The offline fitter
writes it (`search/prior/fit/`, driven by `emmy fit`). Building the training cases from the goldens lives in
`emmy/commands/fit.py`, because reconstructing the set of candidates a golden competed against needs the command
layer's tracer for the golden's little PyTorch snippet, which `pipeline/` never imports.

What a newcomer needs to know about the fit:

- **The loss has two parts**: an objective that pushes each recorded golden's rank up inside its own candidate set —
  with the case categories (`thread` / `warp` / `dyn` / …) weighted so no category dominates — plus an L2 penalty in
  raw feature units (`DEFAULT_L2`, CLI `--l2`). The penalty exists to make the fit **well-determined, not to shrink
  the weights**. The rank objective barely moves when you scale a feature that hardly varies across the golden
  candidate sets, so an unpenalized fit is free to pick an arbitrarily large weight there. That is invisible in
  golden-rank metrics and catastrophic when scoring a fork, where a not-yet-decided knob scores such a feature 0.0.
  The penalty must be in raw units (`w_z/sd`), because after de-standardizing, the inflated weight looks like an
  ordinary O(1) weight (HISTORY.md: "The `D_pow2_threads` cold-deploy pick").
- **Loading is strict.** A missing artifact, or one whose `feat_ver` does not match, is a hard error — refit it, never
  a silent fallback. The error comes from the artifact loader, and it surfaces in `tune` / `eval`, which load the
  prior directly. A greedy compile wraps `load_prior` best-effort, so there a bad artifact does not abort the compile:
  it produces the no-prior resolve described under the hierarchy below (goldens + option-0, with the DB tier lost
  along with the prior object). A weight key that is no longer used, inside an artifact of the current version, is
  simply ignored. `EMMY_OFFLINE_FILE` (or `emmy eval … --offline-file`) swaps in a candidate fit for an A/B.
- A separate `weights_dynamic` set ranks kernels whose tiles are masked because an axis is symbolic; it is selected on
  the stamped `S_ext_n_symbolic_axis`.
- Two hard-coded feature interactions sit outside the linear weights: the atomic-free split-K term, and the pair
  `D_scalar_on_warp_eligible` / `D_splitk_roundtrip`, which express a preference for the tensor-core path. The pair is
  driven by the per-kernel `S_warp_eligible` value the scheduler stamps, and it stops a contraction that could use the
  tensor cores from deploying an f16 scalar split tile instead.
- **The linear quality score is turned into a positive stand-in for latency by an exponential**
  (`exp(-scale·quality)`), whose argument is clipped only at the point where floats stop being safe (~±700). **That
  exponential must never flatten out over the range of quality scores that actually occur.** A clip inside the live
  range collapses good candidates onto one identical value, and the argmin then falls back on the order the options
  were emitted in (HISTORY.md: "The saturated-score plateau"). The one consumer that needs a bounded value —
  `FallbackPrior`'s offline multiplier — clamps to `e**±8` on its own side; the consumers that rank get the
  strictly-ordered version.

### The online prior (the online half)

`OnlinePrior` is trained from tune measurements (Part 5) and composed behind `FallbackPrior`. How it is trained,
checkpointed and bounded is Part 5's subject; what matters here is when it is allowed to decide, which is the
calibration gate below.

**A subtlety about features.** The `H_*` features (which GPU, which nvcc level) have the same value for every
candidate competing at one fork, so no weight on them can change the ranking within that set. What tells GPU
architectures apart is therefore a *per-candidate* feature — one that only takes a value where the hardware offers the
thing it describes. The `D_tma_*` features mirror the tile geometry onto rows that stage through TMA, which lets one
weight set score Hopper/Blackwell tiles differently from cp.async-era ones. The `D_w_grid_*` features separate
candidates with the same tile but a different warp grid, which used to produce byte-identical feature vectors.

### What a `Prior` offers its callers

The names below recur throughout this document; together they are the whole public API of a `Prior`:

| Member | Caller | What it is |
|--------|--------|------------|
| `score(knobs)` | MCTS selection (PUCT) only | Predicted latency, used to steer exploration. On the composite prior this is the one call that blends the two halves (see the calibration-gate section below). |
| `mean_score` / `mean_scores` | deploy + eval ranking | The model's latency prediction for one row / for a batch of candidates. `FallbackPrior` routes these to the online half when it is `trustworthy`, else to the offline half — no blending. |
| `evidence_pick(rows)` | deploy tier 2 | The pick made from measured reservoir rows (defined below). Returns `(index, measured_µs)` or `None`. Consulted whatever the calibration verdict says, because measured evidence needs no trusted model: a quarantined model — or a checkpoint whose reservoir has rows but no fitted model yet — still supplies this tier. |
| `pick(rows)` | deploy + eval | `evidence_pick` first; when no candidate has evidence, the `mean_scores` argmin with the canonical tie-break. Returns `(index, µs)` — a measured µs when evidence decided, a predicted one otherwise. This covers tiers 2 and 4 only: `greedy_decide` puts the golden tier above it and the DB tier between the two, so the `Prior` never owns the whole hierarchy. |
| `sig_groups` | both measured-evidence tiers | How a candidate is matched to measured rows by its `S_*` features. It still matches when the feature set has changed since those rows were written (Part 4) — one rule shared by the reservoir tier and the DB tier. |
| `trustworthy` | the check that lets the online model decide | `fitted` AND passing the calibration gate. |
| `mean_score_features` / `explain_features` | diagnostics only | Scoring / decomposing a row that is ALREADY in feature form (Part 8) — which is what lets the attribution views hide individual features that no knob value corresponds to. |

### The deploy evidence hierarchy

`TuningSearch` (`tune`) ranks the PUCT frontier with the prior's `score`. `greedy_decide` (`compile` / `run`, via
`Run.resolve`) never explores: at each fork it picks once, working down the list below from the top. **This list is
the authoritative order** — the summaries elsewhere in this file defer to it.

1. the **goldens** recorded for the GPU being compiled for (the verified-evidence tier — see below): the first
   candidate that agrees with the fastest recorded golden for the op's shape;
2. measured **reservoir** evidence (`Prior.evidence_pick`): the candidate that agrees with the fastest reservoir row
   of the same op that was measured at `-O3` (`H_opt=3`);
3. the tune DB's measured `perf` rows, with a preference order inside the tier: a row measured at **deployable flags**
   decides outright, and a row measured at the `-Xcicc -O1` flags a tune sweep uses decides only when no candidate has
   a deployable-flag measurement. An `-O1` median is a ranking signal that is known to invert against `-O3`, so it
   must never override a deployable-flag row — but it is still a real measurement of this exact op, so it beats the
   model's extrapolation;
4. the prior's `mean_scores` argmin — only when no candidate has any evidence at all. Score ties break by
   `knob.canonical_row_key`, never by the order options were emitted in.

Three definitions the list leans on:

- **What "agrees with" means** (`evidence_row_vouches` in the code; the same rule serves the golden, reservoir and DB
  tiers): a measured row counts as evidence for a candidate when every tuning knob the candidate has decided so far
  has the same value in that row. Knobs the candidate has not decided yet are free — a later pass will decide them.
  That is what lets one fully-decided measured row settle a fork whose candidates are still only partly decided.
- **The reservoir** is the online prior's own training dataset: a bounded uniform sample (Algorithm R, capped at
  `MAX_ROWS` = 100k) of every training row ever streamed in across runs, stored INSIDE the online checkpoint
  (`online.json`, Part 5). Its `H_opt=3` rows — the deployable re-benches of Part 5 — double as deploy evidence, so
  tier 2 is not a separate store. A tune writes its `-O3` re-benches to the reservoir and the `node` table ONLY, never
  to `perf`. So on a machine tuned at the default `-Xcicc -O1` flags, the measurements taken at deployable flags live
  in the reservoir, and the `perf` table gets deployable-flag rows only when a sweep itself ran at those flags. That
  asymmetry in who writes where is why the reservoir sits above the DB tier. One consequence: anything that discards
  the checkpoint — a `FEATURIZER_VERSION` bump discards it WHOLE, see "Featurizer versioning" — deletes this evidence
  tier along with the model, and the machine's deploys silently drop to goldens → DB rows (usually `-O1` ones) →
  offline prior. The SQLite `perf` rows (tier 3) survive such a bump: the DB is keyed by content, and the join that
  matches rows to candidates tolerates feature-set changes, so old rows stay usable.
- **Which compile flags each tier applies under**: the golden and reservoir tiers apply only to a compile at
  deployable `-O3` flags (`H_opt=3`). A golden's µs and the `-O3` evidence are true of the deployable settings only,
  and must never settle an `-Xcicc -O1` compile, so `make test`, which compiles at `-O1`, never consults goldens.
  `H_opt` is read from the `-O<n>` in the compile flags; flags with no `-O<n>` at all — the `compile` / `run`
  default — count as 3, so a default deploy is always deployable. The DB tier applies under any flags: its
  "deployable" half means any context key that is not the `-O1` one, so an `-O3` row decides outright even under an
  `-O1` compile. The two tests are deliberately not mirror images: only `-Xcicc -O1` counts as the ranking flags,
  while anything else counts as deployable for the DB tier. An explicit `-O2` pin therefore gets DB evidence but
  neither goldens nor reservoir — an accepted edge case.

**With no prior object at all, tiers 2–4 are ALL gone.** That happens when `load_prior` failed (a corrupt online
checkpoint, or the strict offline-artifact load raising; the loader is best-effort and swallows any failure), and on
`Pipeline.run`'s last-resort resolve that deliberately takes the rule's first option. The reservoir is carried by the
prior object, and the DB tier is only consulted on the path where a prior exists, so a corrupt checkpoint costs the
resolve its DB evidence too. The goldens still decide the forks they match — that tier needs no prior — and only the
rest falls to option-0.

**What is deliberately NOT in this hierarchy: the tune DB's `node` table** (Part 6). Node rows are never consulted at
deploy. They feed the `emmy eval` diagnostics (Part 8), and they are what the offline fitter would train on if it
trained on a frozen snapshot of that table — a planned path, not a current one (`emmy fit --data freeze:<path>` is not
yet supported; today `emmy fit` trains on goldens only).

**Whichever tier decides, the µs of the winning row is written onto the fork's trace entry** (`Decision.score`): a
measured µs when a golden or an evidence row decided, the model's predicted µs otherwise. That number is what the
structural cost estimate reads off the partition fork (Part 4), so the Σ compared there mixes measured and predicted
µs — measured wherever the tune benched that kernel, predicted only where nothing was.

**How to see which tier answered.** There is no flag that reports, per fork, which tier decided it; a live compile
does not print that. What exists today is: the loud warnings (a golden shape none of whose entries matches anything on
offer, measured evidence that overlaps none of the offered candidates, and the message logged when a golden overrides
option-0 on a resolve with no prior), the resolve trace (`Decision.score` carries the deciding row's µs), and the
audits — `emmy eval golden` re-runs the golden-tier consultations and prints a verdict per fork (MATCH / DRIFT / GAP,
Part 7). Answering "which tier decided this fork, and did I expect that one?" means correlating those three, not
flipping one switch.

**Where the kernel gets cut is settled before any of this.** Ahead of the schedule pick, a separate decision splits —
or does not split — the recognized work into kernels. A **routing** golden entry is how that decision is recorded: an
entry whose knobs are only `PLACE@<label>` values, where the label names an edge inside the recognized kernel.
`PLACE@cone = cut`, for instance, says "split at the edge labelled `cone`, so that sub-computation becomes its own
kernel". The loader rejects an entry that mixes `PLACE` with schedule knobs, and `_golden_evidence_index` skips
routing entries. They are consulted during recognition (`lowering/tile/_cut.py`, joined by `ShapeKey.joins` against
the live GPU, and restricted to `-O3` like the golden tier itself). Each resulting piece is then recognized afresh and
resolves its OWN `(kind, shape)` through the same hierarchy above; see the tile-lowering ARCHITECTURE's
placement-routing section. Keeping the work in one kernel is the default and is what an absent entry means; cutting
happens only from recorded evidence or a hand-forced pin. That makes routing one of exactly two ways a greedy compile
can change which kernels exist: a routing golden (or pin) cuts with no prior involved, while a structural fork
(Part 4) needs the trusted online prior to estimate its cost. Everything else — offline prior and option-0
included — keeps the default kernel set.

**Both file-backed inputs to that pick are built once per process.** The parsed online prior and the DB perf index are
memoized on the source file's `(path, mtime)` — the online file, and the DB file plus its `-wal` sidecar. A generative
serve boot compiles ~96 programs, and `structural_key` folds only cc + nvcc flags (never the op shape), so both inputs
are identical across every program; without the memo each compile re-parsed the 56 MB `online.json` and re-scanned the
whole perf table. The mtime key invalidates on any on-disk change, so a rewritten checkpoint or a fresh perf commit is
still picked up.

### Goldens are the first evidence tier of a greedy compile

**The per-GPU golden files are the only *measured* data that ships with a clone.** The reservoir and tune DB are
machine-local caches written by local tunes, so a fresh machine — every rented box — used to deploy on pure model
extrapolation, with misdeploys up to 29× (HISTORY.md: "The saturated-score plateau").

At a fork, `greedy_decide` joins the op by `ShapeKey` against the goldens recorded for the GPU being compiled for, and
picks the offered candidate that agrees with the fastest recorded entry. Every **kind** of golden takes part — an
entry's kind is which standard shape it describes: matmul, attention (flash), rms_norm, softmax, reduce, pointwise,
and norm_linear, the kernel that fuses an RMSNorm with the linear layer that follows it. That kernel computes its own
A operand instead of loading one, which is what "**computed-A**" means wherever this document says it. Keys and values
are compared with the same canonical matching the A/B pin check uses.

The join has some deliberately non-obvious mechanics:

- Static and dynamic entries never match each other. Which kind an op is gets decided from the counts an earlier pass
  stamped onto it, through the test `S_loop_depth < n_free + n_reduce + n_symbolic`. That test is what keeps a flash
  or norm op apart from a contraction that merely happens to have the same extents.
- Among the rsqrt-based kinds, a SECOND reduce axis (`S_ext_n_reduce_axis >= 2` — the contraction sitting beside the
  statistic's own reduce) marks the `"fused"` computed-A form, and `is_warp` is forced True for it. A computed-A
  contraction is a tensor-core (warp) kernel whose f32 statistic constants would otherwise make it look like a scalar
  one.
- At the deploy fork, a flash op is recognized differently: from the pair of tile keys its options carry, `TILE@dd`
  and `TILE@pj`. (A schedule key names the step inside the kernel that it applies to, by that step's reduce axis —
  `dd` is the query×key product, `pj` the probability×value product. "Naming a schedule choice inside a kernel", near
  the end of this file, is the full reference.) The tile pass restructures the flash op and gives it re-derived
  extents but no stamped counts, so the classifier above cannot fire there.
- The computed-A norm→linear kernel is likewise recognized at the deploy fork from what its options offer: a `STAGE`
  value of the form `d*/sync`. That is the *compute-fill* staging — the shared-memory block for A is filled by
  evaluating the producer per cell instead of copying values in — and only a computed-A contraction ever offers it.
  The detour is needed
  because before the split, this fork carries only one reduce axis, with the rsqrt still buried inside the sub-body
  that produces A, so the stamped counts read it as a plain scalar matmul. It is rebuilt under the fused key, which is
  what lets the norm→qkv kernels find their goldens on a cold deploy. A fused golden is required by the schema to
  record a `d*/sync` STAGE. `GoldenRecord.shape_key` replays that stored schedule prefix through the same fork-key
  classifier, so a Loop fallback persists the fused join without serializing a derived ShapeKey; its config can never
  be used for a plain matmul that loads A from global memory and happens to have the same extents.
- A golden key that names its axes (a static attention golden's `TILE@dd` plus `TILE@pj`) is all-or-nothing: both must
  match. A golden key written with no axis, for a kind that has several, behaves like a hand pin instead — one plan,
  satisfied by ANY option of the same knob family. That is how a dynamic attention golden's single unsuffixed `TILE`
  matches the axis-named leaves of the masked fork. A fast-math entry excludes itself when the mma instruction it
  names is not among the offered options — for instance whenever the fast-math gate is off.
- Picking the fastest entry first is what makes the **"fast math never loses" rule** work (checked statically in
  `test_golden_configs.py`): within one GPU's rows for a given name, a fast-math entry recorded SLOWER than the best
  standard sibling could never be used anyway, because the standard row matches first whether fast math is on or off.
  Such rows are therefore dropped, and a missing fast-math row simply means a fast-math deploy uses the standard
  config there.

**Whether goldens are training data differs between the two halves of the prior.** The **online** prior never trains
on them: a recorded golden row enters no reservoir and no checkpoint, so for the online model the goldens are an
untouched acceptance set — data it is judged against but never learns from. (Benchmarks of a golden *shape* during a
tune are ordinary measurements and do train it; it is the recorded configs and their µs that never become labels.)
The **offline** prior IS fitted on them: `emmy fit` reconstructs the set of candidates each golden competed against
and trains the weights to rank the recorded config well inside that set (the fit description above). An offline
golden-rank number is therefore measured on the very data the fit saw; the fit harness's cross-validation folds are
the view on data it did not. A golden's µs is true of the deployable flags only, and never settles a compile at other
flags.

**A golden entry the compiler can no longer produce is worse than no entry at all.** When a shape matches but none of
its entries matches any candidate on offer, the compile logs a loud warning that the option set has moved since the
recording, and falls through to the tiers below. That fall-through is a hazard, not graceful degradation: the tiers
below can land on a config hundreds of times slower than the entry claims (HISTORY.md: "The unrealizable-golden
fallthrough"). Two consequences when recording a fused row:

- Recording `PLACE@cone` **together with** a `TILE` can never match, because no single offered candidate carries both
  — so record one or the other. The schema check accepts a combination that the offered options then reject.
- A row must be verified to **deploy**, not merely to reproduce under a hand-forced pin. `--ab` reproduces configs the
  compiler would never offer on its own, so a row that only works when pinned still looks healthy in the isolated
  golden-reproduction check. Only the in-model audit (`eval golden --in-model`) catches it.

**The golden tier needs no prior.** A resolve with no prior at all — `load_prior` failed (a corrupt online checkpoint
or an erroring offline artifact; see the hierarchy section above), or `Pipeline.run` fell back to taking each rule's
first option — still consults the goldens and deploys one that matches, logging loudly when it overrides option-0. So
a broken checkpoint can never silently cost a fork its verified golden.

### `FallbackPrior` and the calibration gate

**`FallbackPrior` only lets the online half answer once it is `trustworthy`** — fitted AND passing the **calibration
gate** (`Prior.trustworthy`).

After every fit, `maybe_refit` measures how well the model ranks the very rows it trained on: the median, across ops,
of the Spearman correlation between its predictions and its own reservoir labels (`_reservoir_calibration` — ops
grouped by their `S_*` signature, groups of fewer than 8 rows skipped, the verdict stored in the checkpoint). Below
`CALIBRATION_MIN` (0.5 — a genuinely trained model scores ~+0.85, while the collapse where the model and its rows no
longer share feature names scores ~0) the model is **quarantined**: it keeps training and checkpointing, but the
deploy ranking calls, PUCT, and the structural cost estimate (`greedy._pick_structural`) all fall back to the offline
half, and the verdict is logged. The reservoir evidence tier stays live under quarantine, because measured evidence
needs no trusted model.

A calibration that could not be measured at all (`None` — e.g. scipy is missing, or no op group is big enough) passes.
The gate is an alarm for measured failure, not a demand for proof of quality. It is known to be lenient in one case: a
small tune (the fit needs only `min_rows` = 50 dataset rows) whose op groups all stay under 8 rows ends up fitted with
calibration `None` — trustworthy, and therefore owning deploys AND the structural cost estimate, on very little data.

Why the gate exists: `fitted` alone once let a mis-calibrated model own deploys silently (HISTORY.md: "The
mis-calibrated online model"). Correlating predictions against the training rows catches one failure specifically: the
collapse where model and rows no longer share feature names (constant predictions, worse-than-random ranking). It
deliberately does NOT catch subtler failures — overfitting to the op families that were tuned, or being wrong about
absolute µs on ops that were not. Those are what the Part 8 diagnostics exist to surface.

**When the online half is trusted, `mean_score` / `mean_scores` answer with the online model alone**, and `pick` is
the reservoir evidence first, then the online argmin — the offline half is out of the deploy path entirely. `score`,
the signal MCTS uses to decide what to explore next, is the one call that still blends the two:
`online_µs · offline**W`. The offline factor is `exp(-scale·quality)` (with the artifact's fitted scale, ~0.1) clamped
to `e**±8`. Only its ordering is meaningful, and its no-opinion value is exactly 1.0, so a config the offline
heuristic has no view on leaves the online prediction untouched. `W` is `config.offline_tilt` (`EMMY_OFFLINE_TILT`,
default 0.3; `W=0` gives pure-online selection). The point of the blend: PUCT still explores regions the cold
heuristic rates well but the data-poor online model buries, while the offline factor's arbitrary magnitude never
touches the µs scale a deploy sees.

### Featurizer versioning

`features.FEATURIZER_VERSION` is written onto every stored training artifact:

- **The prior checkpoint** (`to_json`): `from_json` discards a checkpoint from another version WHOLE — model and
  reservoir rows alike. Rows recorded under a retired version's feature names produce meaningless feature vectors, and
  a refit on them collapses to constant predictions. Note how far that reaches: discarding the checkpoint also deletes
  the reservoir evidence tier (Part 3) and disables the structural cost estimate (there is no trusted online prior any
  more) until the machine re-tunes. A version bump therefore changes deploy behavior — the machine drops to
  goldens → DB `perf` rows → offline prior, with no warning at deploy time.
- **The autotune DB's `node` rows** (a `feat_ver` column, added without rewriting old rows):
  `diagnostics.node_report` excludes rows from another version and prints how many it dropped. Rows written before the
  column existed default to version 1 (the retired feature names) and are excluded, which errs on the safe side.

Bump the constant on any incompatible change to knob naming or feature encoding; artifacts from the old version then
age out instead of poisoning the model.

## Part 4: The drivers — two ways to run the pipeline

`Pipeline.build(passes)` wraps a pass list; the result exposes the compile entry points, each driving one of the `Run`
engine loops — `drive` for exploration, `resolve` for deterministic resolution. Both loops share one rule-batch body;
they differ only in what happens at a fork.

### `Run` — the per-run state

`Run` bundles everything scoped to one compilation: `pipeline` + `ctx` + `search` + `db` + `backend` + `dump` +
`rejections`. `Pipeline` stays a frozen, shareable pass layout, while everything that collects output for one run
lives on the `Run` and is reached through the candidate (`cand.run.dump`, `cand.ctx`).

### `Run.drive` — the exploration loop (`tune`)

`Run.drive(graph) -> Iterator[(token, Candidate)]` seeds the root candidate. Per iteration it pops a `LazyCandidate`,
resolves it, runs one rule batch (`Run._step`, shared with `resolve`), and pushes successors under the pop's token.
Selection is `TuningSearch`'s job (PUCT over the online prior); the perf DB still *records* every bench as training
data.

Each fork's children are classified by their effect at the moment they are pushed, which is where the raw option list
is concrete: an option that splices a `Graph` (and so changes which kernels exist) marks the push `structural=True`;
an option that only rebinds an `Op` is a variant of one kernel (`False`).

### `Run.resolve` — deterministic resolution

`Run.resolve(graph, decide) -> (Graph, list[Decision])` is the deterministic counterpart. Both entry points share one
rule-batch body (`Run._step`), but `resolve` walks the graph once instead of searching: ONE live graph is mutated in
place, with no sibling snapshots and no per-fork copies, so the terminal IS the graph it started from. At each
undecided fork a `decide` callback gets a `ForkPoint` (the `Match`, the raw options as the rule emitted them, the op
as it was before the decision, `ctx`) and returns the option to apply.

The returned trace — one `Decision(rule_name, node_id, chosen_kind, knob_delta, score, n_options)` per decided fork —
is the resolution's only process-state output. Questions like "did this compile take a structural pick" or "what did
the partition fork predict for this kernel" are trace queries, never accumulated policy attributes.

### `Pipeline.run` — the greedy compile

`Pipeline.run(graph, *, backend=None, db=None) -> Graph` is a single-shot greedy compile: a deterministic resolution
(`Run.resolve`) with the greedy pick (`greedy_decide`) — NOT a search. No frontier, no tree, no benching. The graph is
copied once per attempt and resolved in place — no per-fork copies.

**Greedy flattens forks before ranking.** The lazy fork tree is an MCTS structure — it stages knob choices across
levels (`BR` → `BM/BN` → `FM/FN`) so MCTS pays one node per pop. Greedy must NOT walk it level-by-level: a branch
carries only a *partial* tile, and `features.knob_features` can't compute its area / occupancy until `FM/FN` are
pinned, so the prior would be blind at the `BM/BN` choice. Instead `greedy_decide` flattens each fork point to its
complete leaves (`fork.flatten_leaves` expands branches depth-first; only knob dicts — materialization stays deferred
to the chosen leaf) and picks the lowest `Prior.mean_scores` over the full `{H_*, S_*, complete-knob-row}` vector in
one batched `predict`, invariant to the tree's level order. Cold, the `OfflinePrior` ranks (including a positive
`MMA_tier` warp preference); if `load_prior` returns nothing entirely the recorded goldens still decide the forks
they match and only the golden-less forks fall to option-0 (first leaf, emission order).
Greedy benches nothing, so it can only *use* a prior, never train one.

**And it flattens each decision once.** A decision is a conclusion over evidence, so it is memoized GREEDY-SIDE (one
factory call — one compile attempt; never the shared `SessionCache`, which would hand MCTS cached picks): the memo
keys on the schedule `pool_key` (the dtype / hint / pin discriminators op identity excludes) plus the node's
blocklist content, so N same-shape kernels flatten-and-score once and the rest replay by descending the lazy tree's
level keys to the one matching leaf (`_find_decided_leaf` — the O(path) descent `build_fork_tree` was built for),
while a validate-retry with a blocked tile is a different key and re-decides. A GOLDEN-covered schedule fork never
flattens at all: the pool's rows already sit on the tree's root branch, so the golden tier probes them raw
(`_pool_group` → `_golden_pick`) and a MATCH descends straight to its leaf — DRIFT and GAP record their verdicts
there once and fall through to the scored path unchanged.

**Every deploy pick breaks ties by candidate content, never enumeration order.** The model can score many
same-featurized siblings identically (the offline `D_*` geometry doesn't separate an `f2x4` from an `f4x2` fragment or
the `bk` variants — 8 exact ties at the gemma-4 m16 mlp_down/o_proj forks), and one measured row / one golden prefix
can match several offered candidates. Every tier therefore resolves its ties through `knob.canonical_row_key` (the
sorted tuning-knob rendering): the model argmin (`Prior.pick` and the greedy fallback), the reservoir and DB
measured-evidence argmins, and the golden realization pick. An order-broken tie is a per-boot coin flip — leaf order
can shift across processes — and shipped the 2026-07 RTX 5090 gemma-4 image with a bimodal boot-time cubin set
(HISTORY.md: "The bimodal boot-time kernel set").
Pinned by `tests/compiler/pipeline/search/test_deploy_pick_determinism.py` (tier-level permutation invariance plus a
cross-subprocess selected-kernel-set pin, the resolution counterpart of `test_source_determinism.py`).

**Structural options are priced, never raw-scored.** With the trained prior loaded, `greedy_decide`'s
`_pick_structural` prices each side of a structural fork: a nested `resolve` per kernel over a `lowering/tile`-only
pipeline, the price being the `score` of the slice-resolve's partition-fork `Decision`, memoized per `Op.cache_key`.
The cheaper kernel set wins, so an unpinned compile deploys the splits `tune` measured best. The nested resolve carries
the deploy's `db`, so each kernel's price follows the same evidence hierarchy as a knob pick (reservoir -O3, then the
tune DB's -O1 ranking rows, model prediction only where unmeasured) — a pure sum-of-predictions comparison would be
exposed to the model's absolute-µs error, which doesn't cancel across different kernel families. Cold, or when a side is
unpriceable, the structural leaf is filtered — a cold compile never changes kernel sets.

**Evidence joins are drift-tolerant.** `Prior.sig_groups` is one contract for both the reservoir -O3 tier and the DB
tier: a candidate's fork-time `S_*` base may carry scheduler stamps the persisted perf rows predate (#311's
`S_warp_eligible` is on no row recorded before it), and a strict-equality signature join would let one added feature
silently disable the whole evidence tier against every existing DB — the ninth-4090-sweep `mlp_gate_up` misdeploy (the
model's `g2k` pick beating the measured-faster fused config it was never allowed to see; HISTORY.md: "The evidence
tier silently disabled by one new feature"). The index spans three
context keys (the deploy's own flags, and the same key with `-Xcicc -O1` and with `-Xcicc -O3` — where the tune's
deployable re-benches land), and the pick is two-tier: a row measured at deployable flags decides outright; `-O1`
rows decide only when no candidate has deployable evidence, because an -O1 median is a ranking signal with -O3
inversions and must not override a well-trained model on its own.

**Retries are decide-wrappers over a deterministic re-resolve** — every other choice replays identically (cheap
non-chronological backtracking, no snapshots). A structural pick that leaves a fragment kernel un-lowered retires
structural picks wholesale and re-resolves the keep-fused branch before falling back to tile blocklisting.

**Greedy validity fallback.** The prior ranks by predicted latency, which can rank a tile that fails `validate(ctx)`
(smem / thread budget) first — `tune` benches-and-skips it, but greedy benches nothing. So when a deterministic compile
leaves a node un-lowered, `Pipeline.run` blocklists that tile's `tile_identity` (its planner knobs) and re-resolves:
`greedy_decide(blocked=…)` drops the matching leaf and picks the next-best. This is bounded by `_MAX_GREEDY_RETRIES`.
When the retry budget exhausts with the node still un-lowered (an *online* prior can rank many over-budget tiles above
the first in-budget one), `Pipeline.run` takes one last **option-0 (emission-order) resolve**
(`greedy_decide(blocked=…, prior=None)`): the planner emits a budget-safe tile first, so it lowers whenever any
in-budget tile exists. The goldens still decide the forks they match on this resolve — one over-budget node must not
cost every *other* kernel its verified golden — and the blocklist rides along so this last resolve can never re-pick
a tile that already failed `validate(ctx)`. Only when even option-0 overflows does `_raise_on_unlowered` fire the
loud `LoweringError`.

### `Pipeline.tune_async` — the autotune sweep

`async Pipeline.tune_async(graph, *, search, backend=None, db=None)` is the (async-only) autotune sweep. Pass a
`TuningSearch(patience=, ucb_c=)`; the async generator yields one terminal `Candidate` per fully-explored rollout, and
`tune_async` benches each via `await _bench_terminal_async` (writes per-kernel `perf` / `lowering` / inventory rows,
returns the aggregate `PerfStats`), then calls `search.observe(stats, status)`.

- With `backend=None` the bench is stubbed to `latency_us=1.0` and nothing is persisted, so a backend-less sweep never
  overwrites tuned rows.
- A terminal that still holds an un-lowered kernel-bearing node (because its rewrite was dropped by validation) is
  marked `bench_fail` **before** any bench or cache lookup happens. The bench only sums `CudaOp`s, so without this
  guard the un-lowered kernel would count as zero and the µs of whatever cached kernels remained could stand in for
  the whole graph as an `ok` measurement (HISTORY.md: "The zero-priced un-lowered kernel").

## Part 5: The tuning workflow (`emmy tune`)

The autotune loop selects one tile-lowering variant per CudaOp by repeatedly running the lowering pipeline with
different knob choices at each fork point, benching the produced kernels, and steering subsequent rollouts toward the
lowest measured latency.

### Two-level search: outer structural MCTS + inner per-op tuning

`emmy tune` does **not** run one MCTS over the whole graph. The pipeline applies rules one after another, so the two
kinds of fork would nest and multiply out under a single patience budget, starving the ops deepest in the graph. The
two kinds are **op-variant** forks, which choose tile / pad / stage settings inside one kernel, and **structural**
forks, which change which kernels exist — how ops are grouped into kernels, and the split taken by a demoted matmul
(one whose shape makes the compiler handle it as a plain reduction rather than as a contraction). Because the two have
opposite structure, `two_level.py` separates them by the fork's *effect*, reusing the `Op`-rebind vs `Graph`-splice
classification made where the children are pushed.

**Outer search** (`run_two_level_tune`) drives the graph-changing passes: `frontend` + `loop`, plus the part of
`lowering/tile` that runs before `partition_loops`. A **terminal** here is the state in which the cursor reaches
`partition_loops` with every structural fork resolved. Each terminal is one candidate grouping of ops into kernels;
its **reward** is `1 / Σ best-per-op time` from the inner search, backpropagated by the reused `TuningSearch`.

Within one trajectory, structurally identical fork points all take the same side: `Run.drive` replays the first
decision, read off the trajectory's own graph (`_replay_structural_decision`), so the outer tree grows with the number
of *unique* kernels rather than as `2^n` in the number of such points. Fusion itself is still deterministic (no rule
offers a multi-option fusion fork), so a graph with no structural forks yields exactly one terminal and the whole
thing reduces to "tune each op once, sum, assemble". The global prior drives the outer PUCT too: each terminal emits
one combined Σ row per structural decision it took (features `{ctx, op knobs before the decision, the decision's knob
delta}`, label = the Σ of that side's per-kernel bests), so a re-tune on a warm machine descends into the kernel set
predicted to be cheaper first.

**Inner search** (`_inner_reward_async`) tunes each finalized kernel **independently** in its own single-node slice
(`single_node_graph`, `slice.py`) with a plain `TuningSearch` over the lowering passes only (`tile → kernel → cuda`):

- The slice keeps the root kernel + its leaf-op closure and turns every other kernel-input into a synthetic `InputOp`.
  The root op is shared **by reference**, so its body — and thus `Op.cache_key` — is byte-for-byte the full-graph op's.
- One fold-aware exception: a flash fold offer site's slice CARRIES the score producer its fusion consumes
  (`_flash.fused_producer_ids` → `single_node_graph(absorb=…)`), and the absorbed producer loses its own slice. A
  synthetic-input boundary would make `try_flash` unfusable in-slice, silently degrading every tune trajectory to the
  cut (benching fragment kernels greedy deploy never picks) and leaving the fused flash fork unreachable under tune.
- Because the inner tree holds one op, MCTS explores only that op's forks with `patience` as the op's own budget —
  `Σ_k n_k` benches total, never the product.
- **Leaves are deduped by `Op.cache_key`**: 24 RMSNorm LoopOps across 24 layers collapse to one work unit, and the
  outer `total_us` accumulates `best * multiplicity` so the reward stays multiplicity-weighted. The progress
  denominator is the deduped count, so Qwen3-Embedding-0.6B's ~14 unique kernels show as 14/14, not 14/337.

**Separability + the structural handoff.** Op-variant forks are separable: every multi-option fork is an in-place `Op`
rebind that leaves the graph unchanged, so whole-graph time is `Σ_k t_k`. Results key structurally (`Op.cache_key` =
name-invariant body+knobs digest), so a kernel tuned in its slice transfers to the assembled graph unchanged **and**
is shared across outer terminals — two fusion candidates sharing an identical op reuse its tuning (a DB hit). After
the best fusion is picked, the assembled `Graph[CudaOp]` is benched **once** for the real in-context whole-graph
latency; comparing it to the `Σ` estimate is the **separability check** — a gap exposes L2 / clock / launch coupling
the isolated benches can't see (in practice <2% for small graphs).

**Always re-run, replay from the cache.** The inner search runs for **every** op on every pass — it is never skipped
on prior effort. Replay is cheap, not gated: each benched terminal hits the per-variant `perf` cache, so an
already-measured variant is served from the DB with no GPU bench. An identical re-run (same prior) re-walks the same
deterministic trajectory → every terminal is a cache hit → zero benches and the same total. But the global online
prior keeps changing (it refits across ops and runs), so the same patience can steer the MCTS down a *different*
trajectory; re-running lets it reach and bench the genuinely-new variants the improved prior surfaces, replaying the
rest for free. (An earlier "skip already-tuned ops" gate suppressed exactly that re-exploration and was removed —
HISTORY.md: "Retired designs".)

### Per-kernel and working-target GPU parallelism (`--gpus N` / `--devices 0,1,2`)

Because the inner search tunes each unique kernel independently, the per-op loop fans out across GPUs. The whole tuner
is async-only: `run_two_level_tune` `await`s `_inner_reward_async` per outer terminal, which runs one coroutine per
unique kernel over an `asyncio.Queue` of `len(pool)` device-pinned `CudaBackend`s — each pops a backend, drives its
op's whole inner search via `Pipeline.tune_async`, then returns the backend. So `len(pool)` benches run at once, one
per GPU.

A `--golden-file` sweep adds a second safe unit of parallelism: independent targets are driven concurrently over the
same backend queue after their process-global proposal pins have been measured in file order. All target tasks share
one DB and one prior instance on the same event-loop thread; they do not create independent checkpoints. This keeps
multiple GPUs busy even when every traced reproducer contains only one post-fusion kernel. A requested dump root gets
one indexed target subdirectory so concurrent compiler artifacts never overwrite each other.

- **True single-thread asyncio**: every Python statement (lowering, DB writes, prior `add_rows` / `maybe_refit` /
  `checkpoint`) runs on the one event-loop thread and yields only at the bench `await`, so the shared `db` / `prior`
  need no locks.
- Each op seeds its `TuningSearch` by `seed + op_idx` and the reward is a commutative `Σ`, so the per-op DB bests and
  `total_us` are byte-identical regardless of slot count; only the online prior checkpoint varies run-to-run (rows
  arrive in completion order).
- The **default single-GPU** path is a one-slot pool whose coroutines acquire the lone worker in `op_idx` order —
  strictly sequential, identical to the old serial loop.
- A backend pins its async worker to a physical GPU via the child spawn env (`CUDA_VISIBLE_DEVICES`, plus a per-device
  `EMMY_GPU_LOCK` suffix), never mutating the parent `os.environ`.
- Parallelism is bounded by the unique-kernel count; devices must be homogeneous.
- Multi-GPU tune initializes isolated workers two at a time before measuring candidates. CUDA-context startup uses a
  separate readiness budget, so host import pressure cannot become a false 16-second candidate failure on larger pools.

### Working-golden proposals and measured-candidate budgets

`search/working_golden.py` owns the mutable working-file lifecycle: fold-aware trace inventory generation, safe
sidecar writes, target/candidate reconstruction, exact proposal measurement, and atomic ranking/winner persistence.
`search/pins.py` owns the scoped knob environment and realized-pin validation shared with `run`; neither reusable
service belongs to the CLI layer.

For `tune --golden-file`, every entry with an explicit `knobs` mapping (including `{}` for a forkless anchor) is
compiled with scoped authoritative pins and measured through
the normal isolated benchmark and DB persistence path before the unpinned MCTS. A realized-vs-requested check marks an
unoffered proposal `pin_unmatched` instead of attributing the planner's fallback to the proposed row. Successful seed
rows feed the shared prior immediately, so the following MCTS can use their evidence. Ranking feedback is flushed to
the working file as soon as proposal measurement finishes, before MCTS. A multi-CudaOp result records realized knobs
only when their union is conflict-free; otherwise the ranking is explicitly ambiguous.

`--max-candidates N` is a hard per-kernel budget. Each supplied proposal reserves one slot even if its measurement is
already cached, which makes hybrid-vs-MCTS comparisons charge LLM proposals consistently. MCTS receives the remaining
slots and counts only terminals that reached a live backend; cached replay observations update the tree without
spending the live-measurement budget. Ranking feedback is written under the entry's working-only `ranking` mapping,
and the final tune winner is annotated or appended as another proposal only when one directly searched observation
provides both its knob row and cost. A later greedy deploy replay can select different golden/DB evidence and is never
paired with that search reward. These `-O1` ranking numbers never populate
`emmy_us` / `cublas_us`; promotion still requires the separate repeated, correct, deployable `-O3` A/B gate.

Hybrid-vs-MCTS baselines start from identical inventory-only working files: verified rows are not copied into either
proposal set. Canonical repository goldens remain the common implicit deploy context for both runs.

### Search dynamics (the MCTS itself)

Each level reuses the **same** SP-MCTS (`policy/mcts.py`) — outer over structural forks, inner over one op's forks —
with max-Q normalized UCB1:

- **Selection** is PUCT (`_select`): `score(c) = Q(c) + c · P(c) · √(N_parent+1)/(1+N_c)`, where
  `Q = best_reward/global_best` (0 if unvisited), `reward = 1/median_us`, and `P` is the prior's predicted reward on
  the same scale (the prior predicts latency `û(c)`, which `_select` converts to `1/û` and normalizes by the same
  `global_best` — no softmax; `c = --ucb-c`). The prior is the SOLE selection signal — there is no greedy tiebreak,
  no static score, and no force-bench of unvisited children. A confidently-slow sibling (large `û` → small `P`) is
  deprioritized instead of force-benched.
- **Expansion** is implicit (one rule batch per pop, one child per alternative).
- **Simulation** is the actual `await backend.benchmark_async(...)` on the terminal.
- **Backprop** walks the popped candidate's parent chain updating `visits` and `best_reward`.
- **Patience** counts terminals since the last new global best; when it exceeds `--patience N` (default 50), the level
  exits.

### Re-benching the near-best configs at deployable `-O3` flags

The sweep compiles at `-Xcicc -O1`. That is fast, but it is only good enough to *rank*: it gives equal times to
configs that would differ at `-O3` — an ILP fold in `REDUCE`, say, or a warp tile's dedicated group of producer warps.
So whenever a bench lands **within `EMMY_O3_TOL` (default 15%, `config.o3_tol`) of the best `-O1` result so far** — a
wider band than "strictly better", so near-tied contenders qualify too — the engine re-benches that config at
`-Xcicc -O3` (`_rebench_o3`). `observe_o3` then records an extra row, carrying the same knob values the config
actually ended up with and tagged `H_opt=3` (the deployable regime), into the reservoir AND the `node` table — never
the `perf` table (Part 3's reservoir definition explains what that means at deploy time). In the `node` table it lands
as a leaf row with no parent, under its own `-O3` `context_key`. Each config is re-benched at most once.

The `H_*` features are what let the broad `-O1` rows and the near-best `-O3` rows live in one dataset. `compile` /
`run` run at `-O3` (`H_opt=3`), so a greedy compile ranks by the deployable rows and reaches the true optimum — for as
long as the checkpoint holding them survives (see the consequence spelled out under "Featurizer versioning" in
Part 3). The `nvcc_flags` override travels with the bench request to the worker, so only the winners pay for the `-O3`
recompile, and the compiled cubin is cached under those flags.

All tune/bench timings are **CUDA-graph-captured** by default (pure GPU time); each `perf` row records its mode in the
`captured` column, and on write a captured measurement supersedes a wall-semantics one for the same key (never the
reverse), so old rows upgrade in place.

### Training the online prior

There is ONE global `OnlinePrior` across every kernel, GPU, and nvcc setting — not per-op, not partitioned by
regime. Op structure (`S_*`) and the host/hardware regime (`H_*` — GPU compute capability + nvcc opt level, from
`Context.features`) are **features in every row**, not a cache key.

**A partly-decided config is labeled with the best result reachable from it.** Real benches exist only at leaves, but
the prior ranks partly-decided siblings at every fork level, so the label for any node is the best (minimum) median
latency in µs over its
benched descendants (`1/best_reward`) — the prior regresses on **latency**, and the `1/û` conversion lives in the MCTS
`_select` loop, not the stored data. `TuningSearch._collect_rows` walks the live tree and emits `(knobs, label)` for
every node with a benched descendant:

- A **leaf** that was benched directly uses its `realized_knobs` — the FULL configuration read off the resolved
  graph's op in `observe`. That way the knobs written at deterministic, non-forking lowering steps (`FK` / `BK` /
  `SPLITK` / `STAGE`) are captured too, not only the knobs the fork itself decided.
- A **branch** falls back to `_node_knobs`: the partly-decided `fork.knobs` it carries, on top of the op's `S_*` /
  `H_*` base, labeled with the best latency among the descendants that were benched.

**Why CatBoost** (chosen by `scripts/prior_bakeoff.py`): the model's greedy pick must not run off to a degenerate
extreme. A linear model is monotone in every knob, so its optimum always sits at a corner of the box of candidate
values — which shipped real blow-ups before the switch (HISTORY.md: "The linear prior's corner pick"). Any tree
ensemble is **bounded**, so it stays sane outside the region the data covers: an un-benched extreme simply inherits
the value of the nearest leaf. Among the bounded models, CatBoost also generalizes to an op that was never tuned
almost perfectly (leave-one-op-out pick ratio ~1.0, against 1.18 for xgb/lgbm and 1.31 for rf). One global CatBoost
prior is therefore good enough on a new op that it is **not refit during an op's own search** — within a run it is a
fixed model.

**Dataset and checkpoint.** The dataset is bounded + batched (`base.Prior`): each tuned op's training rows stream into
a reservoir-sampled dataset capped at `MAX_ROWS` (100k, Algorithm R across runs), and the model refits (`maybe_refit`)
on a dataset-size-tiered cadence (`REFIT_SCHEDULE` — frequently while data-poor, coarsening as it grows), then
checkpoints. End-of-run does a `maybe_refit(force=True)` so even a small tune ends with a fitted model. The checkpoint
is a JSON file (`config.online_path()`, `~/.cache/emmy/online.json`) holding the CatBoost `cbm` blob (base64) + the
dataset; `tune` writes it, `compile` / `run` read it. This reservoir-sampled dataset IS the reservoir of the deploy
hierarchy's tier 2 (Part 3): its `H_opt=3` rows are the measured evidence a greedy compile consults, so the
checkpoint carries deploy evidence, not just model state.

### Driving the loop

`emmy tune <model_or_ir | --code EXPR>` probes a `Context`, opens the tuning database (`EMMY_TUNE_DB` or
`~/.cache/emmy/autotune.db`), and calls `run_two_level_tune(...)`. The DB accumulates rows across runs; re-running
resumes from the cached state. On default verbosity (and a tty) a `TuneProgress` draws a live single-line bar
(completed/total tuned op leaves plus a `<kernel> <current us> (best <best us>) <knobs>` tail), threaded as an optional
`progress=` through `run_two_level_tune` (duck-typed, so the search package keeps no `commands/` dependency); `-v`
shows the per-`[tune]` INFO lines instead, `-q` is quiet. `--bench` re-benches the tuned winner at -O3 (deployable,
not the -O1 ranking pass): the full model against the real torch module and each kernel via its in-memory frontend
provenance slice, vs eager / `torch.compile` / Emmy.

## Part 6: Persistence and keys

This Part is about what survives a process: which identity a row is keyed by, which table it lands in, and how a live
store becomes a reproducible snapshot. Read the keying map before adding any cache or column.

### The keying map: two identities

Everything the search stores or replays is keyed by one of TWO identities — when adding a cache or table, pick one;
don't invent a third:

- **Variant identity = `(context, knobs)`** — anything *predictive or replayable*. The `S_*` structural features
  (`loop/stamp` stamps a stmt/op histogram + loop extents + operand dtypes) make the merged knob dict a COMPLETE
  identity, so a prior is a pure function of it. The online prior is exactly `score(features(ctx, knobs))`: the
  structural facts are already in the knob dict, so `features.knob_features` turns it straight into the model feature
  vector (the `S_*` knobs pass through; tuning knobs encode by type, `MMA` expands to atom props).
- **Measurement identity = `(ctx.structural_key, Op.cache_key)`** — ground truth about *materialized leaves*: `perf`
  rows (the per-variant replay cache), op inventory (`loop_op` / `tile_op` / `kernel_op` / `cuda_op`), and two-level
  dedup. The structural `child_key` on `lowering` rows is measurement linkage (it joins the inventory), NOT a replay
  key.

### Search persistence: on-disk inventory vs in-memory MCTS

**`SearchDB`** (`db.py`) is a SQLite store partitioned into:

- **Four op-inventory tables** — one row per op encountered along any lowering chain, keyed by `Op.cache_key`.
- **A `lowering` edge table** — one row per rewrite hop carrying the knob delta plus a best-median upsert
  (`best_per_op_time` walks the chain to resolve a pre-final op's measured cost; loop→loop source hops are skipped as
  structural/decision hops).
- **A backend-partitioned `perf` table** — full stats + `backend` + `status` + `knobs` + `captured`.
- **A `node` table** — one row per **search-tree node**, meaning every partly-decided branch and every leaf of a
  per-kernel search. It is keyed by `digest(context_key, gpu, op_sig, tunable-knob set)` and carries the full feature
  dict the prior sees, a latency for that position in the tree, a `parent_key` pointer, a `gpu` column and depth
  bookkeeping (all written by `record_nodes`). Branch rows and leaf rows are updated by different rules: a branch row
  keeps the minimum, because its latency is a bound over the subtree that a faster descendant genuinely tightens,
  while a leaf row takes the **newest** measurement, because a leaf is a re-measurement of one single config and
  taking the min of K noisy medians would drift toward the noise floor. The `node` table is never consulted at deploy
  (Part 3's hierarchy reads only goldens / reservoir / `perf`); its consumers are the `emmy eval` diagnostics (Part 8)
  and the offline fitter's planned training path over a frozen snapshot.

Each `node` row also carries **label-quality columns** (additive migration; old rows degrade to unknowns):

- `visits` — how many benched descendants the row's label rests on, i.e. how much to trust it; SUMmed across writes
  and merges (unlike `n_updates`, which counts writes within one batch).
- `is_leaf` — whether this is a real measurement or a minimum over explored descendants.
- `variance` / `n_samples` — the leaf's own bench statistics.
- `status` — `ok` / `bench_fail`. Failed leaves ARE recorded, with the watchdog's placeholder value as `value_us`;
  they are the negative examples a search prior needs. An `ok` row is never downgraded by a later failure.
- `run_id` / `measured_at` — the tune session (one id per CLI invocation) and the time that produced the CURRENT
  `value_us`; both are replaced only when that value is.
- `feat_ver` — the `features.FEATURIZER_VERSION` the row's feature dict was written under (Part 3). Rows written
  before the column existed default to the retired version 1 and are excluded from prior evaluation.

The `gpu` identity (`Context.hardware_id`, the PCIe product name) is folded into the node key so that rows from
different hardware never collide. `context_key` (compute capability + optimization level) cannot separate two SKUs off
the same die — H100 and H200 share both compute capability and SM count — so without `gpu` their rows would merge and
the upsert would silently drop one GPU's data. (The `H_total_mem` VRAM feature is what then lets the prior model the
difference between them.) `node` and `perf` are keyed by content, independently of any parent tree, and survive a
`_SCHEMA_VERSION` bump; only `lowering`, which is keyed by the graph's topology, is dropped on a mismatch.

**Merging data measured on another GPU.** `SearchDB.merge_nodes(src_path)` is how data accumulates: it reads another
autotune DB's `node` rows read-only and re-inserts them through the same per-kind update rules. The result does not
depend on which DB is merged into which — a stale leaf snapshot never comes back to life, and `visits` sums when two
rows share a key — so node data measured on a rented GPU (with no local CUDA) folds into one canonical DB without
different GPUs' rows colliding.

Cross-machine DBs are combined by `scripts/merge_node_db.py`. The retired class-specific golden-neighborhood and
remote orchestration scripts are no longer part of persistence; measurement collection uses the ordinary tune/run
paths and stable DB operation identity.

**Measurement freeze** (`data/freeze.py`, driven by `scripts/freeze_node_store.py`). The node DB is a live store —
tunes and merges keep writing into it — so a model fit read straight from it is not reproducible. A *freeze* (v3) is a
snapshot of it, written into a local DIRECTORY laid out like `goldens/`: one YAML file per `(gpu, compute_cap)`
(a `gpu_name`/`compute_cap` header plus a `configs` list), beside a `manifest.json` holding the provenance header and
the content digests.

- **Each row records DB `op_sig`, its measured `S_*` structural row, tunable knobs, and measurement metadata.** This
  is a regenerable measurement snapshot, not the stable golden format. Device `H_*` facts are derived faithfully from
  the GPU header and `opt` at load time.
- **Only current-vocabulary leaves freeze**, as filtered by `freeze_reason`: `feat_ver` must have been current when
  the row was written, and the row must pass the two physical-plausibility checks the DB shares
  (`implausible_value_reason` / `impossible_kernel_reason`). `bench_fail` leaves are kept, as negative examples.
  Branch rows are never frozen and no tree structure is stored — the partly-decided rows are rebuilt at fit time under
  whatever fork structure is current then.
- **Freezing the same DB twice yields the same digests.** Every row serializes to one canonical JSON line, rows sort
  by that line, the per-file sha256 covers exactly those lines (content-level — immune to YAML style), the manifest's
  top sha256 folds the sorted per-file digests, and `created_at` enters none of them.
- A loaded row retains the DB's canonical `op_sig`, so the `-O1` and `-O3` measurements of one operation group
  together without a second shape schema.
- **Loading is strict.** `load_freeze` hard-errors on a missing/foreign/corrupt manifest, a `freeze_ver` mismatch, a
  listed file missing, a per-file digest mismatch, or an un-instantiable row — never a silent fallback.
  `load_node_rows` sniffs a path (directory = freeze, sqlite file = DB, a v1 JSONL freeze is refused with a re-freeze
  pointer) and yields `NodeRow`s from either, which is what lets every nodes consumer
  (`eval online --dataset nodes --db`, `Dataset.from_node_rows` / `fold_node_rows`) take a freeze interchangeably
  with the live DB.
- Rows loaded from a freeze have no parent and `depth=0`, which is how the diagnostics recognize that no tree
  structure is available. The fork-regret view skips them, and the golden-anchored descent prints its loud "no
  fork-tree data" row, so a freeze is evaluated through the leaf-level metrics without inventing fork groups.
- Handing a freeze to something that expects the perf table (the `--dataset db` paths) fails at `open_readonly`, with
  a message that spells out the difference between a freeze and the nodes DB.

**How node rows get written.** The same finished tree that feeds the reservoir is also walked once by
`_collect_node_records` and stored via `record_nodes`. Where the reservoir is an unkeyed random sample, this is the
keyed, deduplicated, parent-linked version of the same data. The walk fills in the columns that say how good each
label is (`SearchNode.visits`, the leaf's `bench_stats` / `bench_status` that `observe` stashed, and `is_leaf` from
`realized_knobs`). It also writes:

- **`bench_fail` leaves** — leaves only. Their value never contributes to any branch's minimum, so a branch's value
  comes from its working descendants instead, and a branch all of whose leaves failed is not recorded at all.
- **`-O3` rows** for any leaf the tune re-benched at the deployable `-Xcicc -O3` (`observe_o3` stashes
  `SearchNode.o3_us`). They are keyed under the tune's context with `O3_NVCC_FLAGS` substituted, so they can never
  collide with the `-O1` row of the same config, their features carry `H_opt=3.0` (the same convention the reservoir
  uses), and they have no parent: this is a re-measurement under different flags, not part of the tree, and never one
  of a fork's siblings.

**Recording benches as node rows** (`search/bench_record.py`) is the node table's second writer. A `run --bench` that
benched rows with hand-forced knob values (golden or `--ab` rows) records each clean measurement — plus the greedy
pick, through its comparable `greedy (isolated)` re-bench — as leaf rows with no parent and `depth=0`. This is on by
default, behind the same quality bar the tuner applies to its own pinned benches; `--no-record-nodes` turns it off. It
is what stops measurements from a manual sweep evaporating (HISTORY.md: "Hand-found optima that never reached the
store").

- **The row must be keyed to the same set of candidates the tuner used.** That means recovering, for each kernel, the
  fork point it descended from, via `source_chain`: descent writes further `S_*` values onto the op, so keying off the
  final op's own stamps would key the row to the wrong candidate set. The recorder takes the deepest ancestor in the
  loop dialect that carries `S_*` features, and falls back to the deepest one in the tile dialect. The mma
  tile-lowering keeps no `LoopOp` in `.source`, so without that fallback every tensor-core kernel was silently
  unrecordable. Both paths digest to the same `op_sig` a tune would write, verified on an RTX 4090.
- The kernels of one variant (a split-K main kernel plus its combine kernel) are grouped under one fork point and
  recorded as ONE leaf for the whole variant. If every kernel in a graph loses its fork point, the recorder warns
  loudly rather than silently recording nothing. Rows that were flagged (a pin that did not match, a wrong answer, an
  implausible arithmetic intensity) and anything from the `--ir` path are never recorded.
- `record_nodes` protects the leaf update by **comparing measurement quality**: a newer measurement that is
  unambiguously worse (fewer `n_samples` AND higher `variance`) never displaces a stored leaf, so a casual bench
  cannot overwrite tune-grade data. When quality is comparable or unknown, newest simply wins, so an honest
  re-measurement still repairs a stale row.

Within one batch, a deterministic step that changes no knob can give a child exactly its parent's knob set, and hence
the same `node_key`. Such duplicates collapse into one row (keeping the leaf's stats, and the max — not the sum — of
their visits), so the SUM accumulation in `record_nodes` never double-counts a single run. The store is ready to be
split into held-out groups for cross-validation (`Dataset.fold_node_rows`, by `op_sig` / `gpu`): an op's `-O1` tree,
its `-O3` rows and its failed leaves all move to the same side together, and no parent edge ever crosses a fold
boundary. (`run_id` records where the surviving deduplicated value came from and is deliberately NOT used to split
folds.)

**`SearchTree`** (`policy/mcts.py`) is pure-Python in-memory MCTS state, colocated with `TuningSearch` because MCTS is
the only policy that reads it. Each tree node wraps a `LazyCandidate` and carries `visits`, `best_reward` (the maximum
reward over the subtree's measured leaves), and a `live` counter that filters out exhausted subtrees. Parentage is
tracked by TOKEN, never by the order calls happen to arrive in: `pop()` returns `(token, candidate)` (the token IS the
`SearchNode`), the engine pushes children with `parent=token` and observes the terminal with the same token, so the
tree stays correct however the engine interleaves pops, pushes and observes. It is rebuilt fresh in each process;
cached `perf` rows ensure no re-bench on warm starts. Greedy compiles build no tree at all — they never go through a
`Search`.

**`_bench_terminal_async`** is the only path that knows about all four parts (graph, DB, tree-through-`search.observe`,
backend). It short-circuits when every `CudaOp` in the graph already has a `perf` row for the current `(context_key,
backend)`. Otherwise it does one `await backend.benchmark_async(...)`, walks `Op.source` once to record op inventory +
lowering edges + the `perf` row per kernel, and returns the aggregate `PerfStats` for the search to score.

## Part 7: Golden records and the A/B integrity gates

A golden record is a reviewed, per-GPU measurement of a frontend program target. It serves three purposes: deploy evidence
(Part 3, tier 1), training data for the offline prior, and a regression reference. This Part covers what the kinds
are, how a golden's layout has to match the fork it is meant to decide, the two audits that catch a golden that no
longer deploys, and the checks that keep the A/B honest.

`golden.py` holds one generic `GoldenRecord`. Each record references a stable frontend Torch IR program by its
document-local list index. The preferred target selector is a non-empty, unique set of frontend provenance origins.
When lowering produces a kernel without such a selector, the record points into the document's optional `loops` pool,
which stores that standalone post-fusion Loop IR slice. Current lowering derives the `S_*` histogram, `ShapeKey`, dtype
classification, dynamic status, and operation kind lazily; none is serialized. Trace inventories retain the complete
frontend program so provenance selectors re-lower in their original fusion context, while Loop IR fallbacks load
directly. There are no kernel-kind classes or snippet generators.

**Repository goldens are the entire compatibility boundary.** The embedded Torch IR has no independent version field.
The golden document has no format version either. When the YAML schema or its Torch IR encoding changes, regenerate
every golden under `search/goldens/` in the same change. The loader does not carry migrations or legacy decoders for
working files outside the repository; keeping the checked-in corpus loadable is the compatibility gate. Programs are
a plain list and configs refer to them by integer index; no program digest or persistent identifier is stored. Loop IR
fallbacks are implementation-level rather than a compatibility promise and follow the same regenerate-the-corpus
invariant. Frontend graph nodes omit empty `attrs` / `inputs`, store tensors as `[name, dtype, shape]`, and encode static
dimensions as integers to keep the persistence surface small.

**One YAML format serves working candidates and reviewed goldens, but the trust boundaries differ.** A working file
may contain an inventory entry (no knobs or timings), a proposed candidate (knobs but no timings), or a verified
candidate (knobs plus paired positive Emmy/reference measurements). `load_golden_file` and `dump_golden_file` validate
this format without mutating the parsed entries; dumping refuses to replace an existing file unless its caller opts
in explicitly. A traced inventory embeds stable frontend Torch IR and selects targets by origin IDs or a Loop IR
fallback. Repository promotion is stricter: every entry must carry an explicit knobs mapping
(possibly empty for a verified forkless anchor) plus both positive finite timings. Missing, one-sided, zero, NaN,
and infinite measurements are rejected before they can become trusted deploy evidence. A working entry may also
carry an opaque `ranking` mapping
for fast-compile feedback (`status`, `latency_us`, compile flags, and measured knobs); it does not change the entry's
state and is rejected by repository validation because only
deployable-regime timings belong in trusted goldens.
An axis-scoped schedule family (`REDUCE@a1`, for example) and its bare spelling must not coexist in one promoted
entry. Bare pins fan out across eligible axes, so storing both spellings can make an otherwise offered row
self-contradictory during the all-of offer check. Promotion rejects that ambiguity before the offer audit.

The preferred reference is the runnable Torch slice (`torch-eager`) or the applicable library kernel (`cublas`). A
Loop IR fallback has no frontend callable by construction; an origin slice can also have synthetic boundaries whose
post-fusion output geometry is not independently comparable to its Torch slice. Such a target may use a separately
compiled, repeated O3 `emmy-greedy` row as its positive reference only when the candidate and reference execute on
identical deterministic inputs, their outputs pass the normal accuracy policy, and the model report discloses that
this checks compiler-configuration parity rather than independent framework correctness. The original frontend
program remains embedded for provenance, while the selected standalone target is what both configurations execute.

The three historical RTX 4080 rows without measurements were dropped during migration; repository validation has no
provisional exception.

**A matmul golden's layout must match the fork it is meant to decide.** The embedded Torch IR spells the
serving Linear layout — B given `(N, K)`, contracted as `x @ w.T`. The traced contraction
carries `b_trans`. The warp tier stages it like any canonical matmul (cp.async and TMA fill an N-MAJOR B slab —
`tile_n × bk`, K stride-1 in gmem and smem alike — drained by the plain no-`.trans` ldmatrix; historically the
transports declined transposed B and the `.lin` forks ran gmem-direct only — HISTORY.md: "The transposed-B staging
gap"), so the
same STAGE spellings realize on both layouts — but the measured µs still differ per layout (different slab geometry
and gmem walk), which is why a golden meant to decide a served model's linear fork must still be TUNED on the
`F.linear` snippet. The two layouts share one ShapeKey on purpose: at a fork the shared bucket sorts by µs, so a
canonical entry (the harness/eval truth) and a `trans_b` entry (the serving truth) coexist under one shape — keep
BOTH current, since with staging realizable on either layout a stale twin's config now deploys cross-layout with its
foreign µs (the layout signal in the stamped `S_*` features / ShapeKey still does not exist). The same rule applies
to fused computed-A programs: their stored `torch.linear` edge is the served layout, and the sync compute-fill stages
every B fold channel via cp.async on either layout.

**Provenance and the in-model drift audit.** A golden file (or entry) may carry an optional `model:` header — the HF
model id whose serving graph the targets came from (`GoldenRecord.model`; pure provenance, never part of any join key).
Model-tagged goldens opt into the **in-model drift audit** (`emmy eval golden --in-model`, library `search/audit.py`):
the model's serving twins are re-traced **weight-free** (`emmy/serving/twins.py` builds a trimmed random-init skeleton
from `config.json` alone — a trace never reads a weight value) and each tagged card's twins are compiled with the
golden tier as the only evidence (no tune DB, online file pointed at a nonexistent path, deployable nvcc regime
forced — under `-Xcicc -O1` the `H_opt` guard would silently skip golden consultation — and the card targeted via
`Context.from_target`, so verdicts are machine-independent). Each golden-tier consultation yields MATCH (a recorded
golden realized), DRIFT (shape keyed but nothing realizes — always a defect: the recording claims a µs the deploy can
no longer produce), or GAP (no golden for the shape). This is the in-model half of the reproduction check: the
isolated snippet A/B reproduced 68/68 while the in-model deploys drifted (the cast-splice class), which is exactly the
blind spot the audit closes. `major_gap_keys` isolates uncovered warp-contraction forks, the misdeploy/hang hazard
class. Serving-image release qualification runs `scripts/check_serving_goldens.py` with `--strict-major-gaps` (the
`make serve-goldens` gate) for the pinned model, revision, card, and configured widths; a
drift, compile failure, or major gap fails the release. The default correctness suite tests the reusable audit
mechanism with synthetic verdicts rather than retracing and compiling a model/card matrix. The twins track the
installed `transformers` modeling code by design: a transformers bump that changes the forward changes the twins
exactly as it changes serving, and release qualification goes loudly red.
`scripts/diagnostics/audit_golden_match.py` is the same audit over explicit graph JSONs on a live box.

**The pin-only offer audit** (`emmy eval golden`, same `search/audit` seam) is the record-time complement: for every
forking golden entry it re-compiles the shape's OWN snippet un-pinned (deployable regime, the golden file's own card —
the enumeration is static given shape+context, so no GPU bench) and checks the recorded knobs against the offered
candidates. An entry only a pin can realize (`EMMY_KNOBS` / working-file proposal measurement benches it, the
enumeration never offers
it) reports **PIN-ONLY** — legal as a documented lever while an OFFERED sibling floors the shape (the 4090
`attention.hd512.s4096` split-KV row beside its serial deploy-floor sibling); a shape whose entries are ALL pin-only
reports **FALL-THROUGH** and exits 1: a deploy logs "no offered candidate realizes any of them" and falls past the
golden tier — the missing-floor pathology that deployed a 111 ms 0.03x `mlp_down.m4096` kernel and NaN-poisoned the
downstream accuracy check before the floor-sibling discipline. Fast-math entries audit under the pinned
`F16_MMA_F32_ACC` gate (their own deploy regime). The own-snippet and in-model views genuinely differ: the 5090
`mlp_down.m4096` split-K row realizes standalone but not on the serving twin's epilogue-fused down — the offer audit
passes it and `--in-model` is the authority there, while the s4096 split-KV row fails even standalone, which is what
this audit catches at record time.

**Live-GPU scoping.** `run` / `compile --golden NAME` prefer the **live** card's goldens
(`goldens_for_live_gpu`) — names repeat across per-GPU golden files with diverging shapes/dtypes, so a flat union can
select another card's spelling. They keep the union fallback on an uncovered card (the seed / transfer flow — the
pinned config re-benches live), and off-GPU the full union is returned (pure-logic tests). Tuning instead consumes an
explicit working file whose GPU header is checked against the selected tune device.

**The A/B carries three integrity gates:**

1. **Realized-vs-pinned knob check — a miss FAILS the row before it benches.** A structurally invalid pin silently
   falls back to the planner's own pick, so benching it would compare greedy against itself and report a fake 1.00×
   under the pin's name (HISTORY.md: "The pin that benched greedy against itself"). The check runs right after the
   pinned compile. A pin that matches none of the knob values the compile actually produced marks the row
   `pin_unmatched` / `unreproducible pin … NOT benched` (a loud error log; the row is kept in the table and in
   `--json`, and no GPU time is spent), and the remaining rows still run. Matching is aware of knob families — a
   golden written as plain `TILE: …` matches the `TILE@dd` the compile produced — and values are compared through the
   registered knob's canonical `Knob.parse`, so alternative ways of writing the same value, like `FAST_EXP=1`, do not
   raise a false alarm. A pin satisfied by ANY kernel counts as honored, which is what makes split main+finalize pairs
   work, but it does mean that a pin dropped on its intended kernel goes undetected if a sibling kernel happens to
   match it.
2. **Arithmetic-intensity check.** A row whose FLOP/s, implied by its shape, exceeds the peak recorded in the live
   GPU's `GpuSpec` is flagged as a bad measurement rather than a fast kernel.
3. **Wrong-answer check.** Each pinned config is executed once on the greedy run's inputs and its outputs are
   compared, which catches kernels that are silently wrong (a skipped finalize produces plausible-looking garbage).

**Every `run --bench` row is measured in a bench worker process that can be SIGKILLed — the parent never launches a
kernel.** The greedy comparison (eager / torch.compile / emmy, with the torch side rebuilt inside the child — the same
mechanism `tune --bench` uses) and every pinned golden / `--ab` row run as jobs on ONE persistent worker per run
session. That makes the A/B survive any failed row by construction: a hung kernel dies with the SIGKILLed child, the
parent's CUDA context stays clean, the row is reported `bench_fail` with its reason, and the next row's job starts a
fresh child — no escalation modes, no `os._exit`.

- NOTE: the *process* is the same for every row, but the measurement *environment* is not. The greedy row is benched
  interleaved with the live torch closures, so torch's allocator state and cuBLAS's L2 carve-outs are resident, while
  a pinned row is benched emmy-only in a job that never touches torch. A greedy-row µs and a pinned-row µs for the
  same config are therefore NOT directly comparable (the gap observed in practice is ~7% on split-K pairs).
- One number cannot be both comparable to torch and comparable to the pinned rows. So whenever pinned rows are
  benched, the greedy graph is ALSO re-benched emmy-only through the same pinned path (one extra worker job, no
  recompile). That produces the `greedy (isolated)` row printed beneath each greedy kernel in the table, and the
  `greedy.isolated` block in `--json`. Those are the baseline the pinned rows' speedups are measured against.
  **Record goldens from `--ab` / golden rows only, never from the greedy row's number.**
- The greedy pick hanging, or blowing the bench budget, is itself a *finding* — precisely the hazard a golden exists
  to prevent — so the pinned rows are still benched afterwards. Pinned rows that fail to compile or to bench are kept
  as `bench_fail` rows, never dropped, and the run exits non-zero if any row failed.
- The greedy job also carries the accuracy check: the emmy program runs on the rebuilt module's real inputs in-child,
  and a numeric failure aborts the run, because a latency table for a miscompiling program is meaningless. That run's
  `(inputs, outputs)` become the pinned rows' wrong-answer reference.
- Only the no-`--bench` accuracy probe still runs in-process (it hosts the `--debug` per-launch dumps and the ncu
  child's profiled launches), so with `--bench` those two want a separate plain `run`.

Plus `--json PATH` — a machine-readable record of the whole comparison (backends / greedy kernels / pinned rows with
their flags and a `status` field: `ok` / `pin_unmatched` / `bench_fail`; a failed greedy block carries
`status: bench_fail` and an `error`, with null timings), so a sweep's judgments can be traced to flagged fields
instead of to parsed terminal text. Each kernel row also carries **`record_knobs`**: the tuning knobs the compile
actually produced, with every schedule knob family (`knob.SCHEDULE_FAMILIES`: WORK / TILE / REDUCE / STAGE / RASTER)
written out explicitly, including the ones that are off (`knob.stamp_schedule_families`). That is the map to copy
verbatim into a golden YAML `knobs:` entry. An entry that omits a family leaves that family to whatever the planner
fills in at replay time, which shifts as the planner evolves — the recurring source of regressions that look real but
come from an unpinned `REDUCE`. Golden rows attach to the run's SHAPE rather than to a kernel node, so a pinned row
whose shape matches no greedy kernel — because greedy deployed a split partial+finalize pair — still prints and still
lands in the record.

## Part 8: Evaluating the prior (`emmy eval`)

`emmy eval` is how you find out whether the prior is any good and, when it isn't, where it goes wrong. The views
below run over the goldens, the tune DB's `node` table, or a measurement freeze.

**A golden's rank counts ties against it** (`eval offline` / `eval online`, via `golden_eval.evaluate_record`). The
golden's rank counts every candidate scoring strictly better PLUS every candidate that ties with it and was emitted
earlier. A tie is counted as a loss because greedy's argmin, faced with equal scores, takes whichever came first.
Counting only strictly-better candidates would report rank 0 for every row inside a plateau of equal scores, which
once let a saturated prior score "top-1" on goldens that real cold deploys missed by 12–29× (HISTORY.md: "The
saturated-score plateau"). Both counts come from ONE computation (`prior/fit/rank.dual_rank`): the pessimistic rank is
the one that gates, and the strictly-better **optimistic** rank is reported beside it in `emmy fit`'s metrics file.
The gap between them is the width of the tie plateau at the golden's score, and thus an early warning that the scores
are saturating.

**Golden evaluations build their features for the golden's own GPU.** `eval offline` / `eval online` rebuild each
golden's compile context as `Context.from_target(compute_cap, gpu_name=…)`, using the GPU recorded in the golden file
along with its known SM count and smem specs — never the host's. Building them for the host's context makes golden
ranks machine-dependent, because the occupancy features then describe tiles for a GPU that is not the one the row came
from (HISTORY.md: "Machine-dependent golden evals"). The offline fitter's case builder always did this correctly; the
eval now matches it.

**Fork-sibling regret** (`eval online --dataset nodes`, via `iter_nodes` → `diagnostics.node_report`) measures, **per
GPU**, what following the prior's choice at each fork costs. It groups nodes by `parent_key` and computes
`value_us(the child the prior predicted best) / value_us(the truly best child)`; 1.00x means the prior steers into the
best subtree reachable from there, and ties in predicted score count against the prior, since greedy breaks them by
the order the options were emitted in. This is the search-faithful evaluation that no view of leaves alone can give.
Each fork is bucketed by which knob FAMILY its children decide (`TILE` / `REDUCE` / `STAGE` / …, read off the
difference between child and parent knobs). That is the stable way to name a level of the tree, because the raw
`depth` counts rule steps and renumbers whenever passes change. The result is rendered as a per-kernel × per-family
regret table, with an aggregate line per family. `node_report` drops `bench_fail` rows up front — their `value_us` is
the watchdog's placeholder, not a measurement — and splits each GPU's block by `H_opt`, so `-O1` and `-O3` latencies
are never pooled. The per-GPU
grouping matters for a cross-hardware dataset: two SKUs off the same die (H100/H200) share an `S_*` op signature but
not their latencies, so mixing them would corrupt both metrics; the `gpu` key keeps their rows apart. `--db` also
accepts a measurement freeze (Part 6) in place of the live DB — its rows are leaves with no parents, so the report
falls back to the leaf-level metrics.

That block is rendered once per **half** of the prior, offline and online, each labeled. The composite would answer
with whichever half is currently active, and the two halves' regrets point at different fixes — the cold-start weights
versus the training data — so an unlabeled "prior" number would destroy the diagnostic.

**The report section labeled "golden-anchored descent"** covers what the regret view structurally cannot see. Regret
only speaks about forks the search actually measured, so a golden sitting in a subtree the search never built — or a
shape with no node data at all — was silence that read as health. That is how a past saturation bug hid from regret
while the then-broken golden rank claimed top-1. Each GPU's block therefore ends with one row per golden recorded FOR
that GPU (a golden is only ever matched against rows measured on its own GPU), reporting: how far its path is covered
by the explored tree (branches are matched with the same family-aware, registry-canonical rule the A/B pin check
uses), whether the prior's tie-pessimistic pick stays inside the golden's subtree at each fork (with the measured gap,
at matching flags, where it does not), and the loud absences — `NO TREE DATA` for a golden whose path is nowhere in
the tree, a count per GPU, and a closing line for GPUs that have recorded goldens but no node rows at all.

Coverage is always printed with a denominator. A fully followed path is exact (`followed 6/6 fork levels to a measured
leaf`), while a partial match's total is an ESTIMATE, marked `~` (`followed 2 of ~7 fork levels`), taken from the
deepest chain of siblings below the fork where the paths diverged — the golden's own branch was never built, so those
siblings' depth is the only evidence of how much tree is left. Keeping the flags straight is essential: the golden's
recorded µs is a deployable `-O3` number and never enters the `-O1` walk or its gaps, because the two systematically
invert. It appears only in the `-O3 pick/golden` endpoint, computed over the op's `H_opt=3` rows with the fast-math
setting matched (the `golden_deploy_perf` convention). This is a diagnostic, not a gate: losing a fork whose measured
sibling is near-equal is fine, and the gap column is what tells you so.

Both halves accept a candidate artifact for A/Bs: `--online-file` (legacy `--prior`) swaps the online checkpoint
(`EMMY_ONLINE_FILE`), and `--offline-file` (on `eval offline` / `eval online`; env `EMMY_OFFLINE_FILE`) swaps the
offline weights artifact — comparing two fits is running the same eval against two files and diffing the reports.

**Which feature is to blame** (`eval online --dataset nodes --blame / --ablate`). Both views consume one shared
per-fork record (`diagnostics.fork_records`: the siblings, their rows in feature form, their scores, the
tie-pessimistic pick and the measured best), so all three views agree on what "the pick" means by construction. They
score through the `Prior`'s feature-level entry points — `mean_score_features` / `mean_scores_features` take a row
that is already in feature form, and are contractually identical to `mean_score` on the raw knob dict. That is what
lets the diagnostics hide individual `D_*` features that no knob value corresponds to.

- **Blame** diffs `Prior.explain_features` — a signed breakdown of the quality score into one term per feature, exact
  for the linear offline prior, with its hard-coded interactions included as `gate:*` pseudo-terms, and unit-tested to
  sum back to the scored quality — between the pick and the sibling that measured best, weighted by regret and grouped
  by fork family. A fork the prior missed where no term separates the two is reported **BLIND**: a gap in the
  featurizer, not a problem with the weights.
- **Ablation Δ** re-picks every fork with one feature hidden, using each model's own notion of an absent feature (a
  `0.0` term for the linear prior, which removes it exactly; `NaN` routing for CatBoost, which is flagged as
  out-of-distribution until a model trained with feature dropout exists) and reports the change in median regret per
  family, along with how many forks that feature had any say in.

Both are **diagnostics, never gate metrics**: attributing an effect among correlated features has no unique answer
(hiding any one of a redundant block of geometry features costs the same Δ). Unlike the per-GPU regret tables,
attribution POOLS GPUs and flag settings, which is safe because regret is a ratio computed within one fork.

## Part 9: Tile lowering at the pipeline level

`lowering/tile/` lowers each fused `LoopOp` to a kernel-ready `TileOp` over the block-DAG Tile IR (`ir/tile/ir.py`):
`010_recognize` (lift `LoopOp` → `TileOp`, recognize the flash / softmax streaming forms, annotate each reduce
`Loop` with its `AxisRole` — the only loop annotation; the algebra is the body — and **atomize**: resolve the
algebra→hardware-atom binding structurally onto the node, so an unbindable atom never becomes one; `_atomize.py`) →
`030_split_reduce` (cross-CTA split-K as a graph rewrite). It **never dispatches on a named
shape** — every decision is gated on the derived role of the stored fold (`PLANAR` / `CONTRACTION` / `TWISTED`; flash
attention is the `TWISTED` fold on the streaming schedule, a twisted monoid is a monoid, selected structurally), not
on a matmul / pointwise / attention archetype. The full design lives in
[`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md).

The step BETWEEN those two — schedule enumeration: mapping the free axes onto the grid and forking the per-node
`TILE` / `REDUCE` / `STAGE` / `WORK` / `RASTER` families — is ONE recursive row enumerator over the term's own site
tree (the `020_schedule` rule). It covers every single-site term, the COMPUTED `a` edge (the fused norm→linear /
gate⊗up cone) and the flash streaming pair — the two-site families are why it recurses. A term it cannot schedule
enumerates NO rows and stays unmapped rather than being guessed at — the guardrail contract, with any coverage gap
riding `tests/xfail_registry.py`. See the leading section of [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md) for
the design.

## Tunable knobs

A **`Knob`** (`knob.py`) is the canonical schema for one tuning dimension: name, type (`INT` / `BOOL` / `BINMASK` /
`STR`), candidate `hints` (advisory — the rule still validates structural fit), and a help string. Rules stamp values
into `TileOp.knobs` dicts; the autotuner reads those back as the per-hop knob delta in the `lowering` table. Every
knob is declared **in `search/space.py`** — the single home for the whole tunable surface — and imported by the rule
that resolves it (for the schedule codecs, the tile scheduler's row enumerator). Declaring a `Knob` IS
registering it (`Knob.__post_init__`); `knob.registry()` imports `space.py` before answering, so the set is complete
in any process — no module scanning, no manual registration. `knob.py`
also owns the `EMMY_<KNOB>` env namespace (decode per `Knob` type; `config.py` remains the sole owner of
`os.environ`).

### Pinning knobs from the environment

Two equivalent forms:

- **Per-knob:** `EMMY_<NAME>=<value>` (e.g. `EMMY_STAGE=d2/cp`). Read by the rule that owns the knob via
  `Knob.narrow`. The env-var key is built by `config.knob_var` and read via `config.knob_raw` / `config.int_env`.
- **Aggregate:** `EMMY_KNOBS="K1=V1,K2=V2,..."` (e.g.
  `EMMY_KNOBS="WORK=w2x2,TILE=mma_m16n8k16_f16_f32/f2x2/k2,STAGE=d2/cp"` — the worker widths ride `WORK`, so a
  `TILE` / `REDUCE` pin that embeds its own raises). Parsed once at `knob.py` import via
  `apply_knobs_env()`, which splats each entry into the corresponding `EMMY_<K>` var
  (`config.set_knob(..., overwrite=False)`). An explicit per-knob var wins over the aggregate.

Pinning replaces tuner choice (the rule emits exactly that variant instead of forking) and is **authoritative** — an
env value outside the knob's hint tuple is honored, not silently dropped (`Knob.narrow` returns `(pinned,)` regardless
of hint membership). Downstream structural gates (divisibility, threads-per-CTA budget, TMA eligibility) still apply,
so a structurally invalid pin yields an empty enumeration and the per-call-site fallback takes over. This lets a tile
shape the planner wouldn't reach on its own be explored manually. The replay paths (`run --bench --golden` / `--ab`)
can't accept that silent fallback — it would substitute the planner's own pick and turn the A/B into greedy-vs-greedy
— so they verify realized-vs-pinned knobs on every pinned row right after the pinned compile and FAIL a mismatched
row (`unreproducible pin … NOT benched`) instead of benching the fallback (see the integrity gates in Part 7).

A few pins are rejected outright (a clear `ValueError`) rather than silently degraded — they would otherwise lower to
a wrong or un-launchable kernel:

- A codec width must be `≥ 1` (a degenerate `b0` / `f0` / `n0` no longer parses to a silently-dropped level).
- A warp `TILE` pin needs its **static** contraction K to be a multiple of the inner mma K-step (`atom_k·bk`), since
  the warp K-loop has no static-K tail masking (a **symbolic** K is fine — it reaches the masked zero-filled tier).
- A warp `TILE` atom must belong to the target's selected MMA family. On SM70, newer `m16n8k16` atoms and `cp.async`
  or TMA `STAGE` pins fail explicitly; the Volta m8n8k4 atom accepts global-memory-direct or `d<n>/sync` staging.
- A scalar `TILE` parallel block (`par_n·par_m`) is capped at the 1024-thread/CTA hardware limit.
- A `BOOL` knob rejects an unrecognized value instead of coercing a typo (`ture`) to `False`.

### Registered knobs

All declared in `search/space.py`; see [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md) for the per-rule mechanics.
The "owning rule" for the schedule codecs is the tile scheduler (the `020_schedule` rule), whose recursive row
enumerator spells each family exactly once, site-local, where a row becomes stored state.

**`WORK`** (STR codec, stamped by `seal_workers` at option assembly) — the kernel-global **worker inventory**,
spelled exactly once per row (step 7): `w<M>x<N>[+p<np>]` (warps — the mma tier; `+p<np>` the dedicated producer
band the retired per-row `WSPEC` key spelled) / `t<N>x<M>` (the scalar thread tile, native n-then-m) / `t<N>` (the
1-D cooperative width). Empty = a 1-thread register strip whose launch geometry stays derived. The tier
discriminator IS the worker kind — never a per-`TILE` spelling — and `seal_workers` derives the inventory from the
resolved site slices, failing loudly on cross-site disagreement (one kernel, one inventory).

**`TILE`** (STR codec, the tile schedule) — the **output-fragment** codec, site-local since step 7. A
contraction's output tile is *either* the **scalar** register sub-tile `f<fn>[x<fm>]` *or* the **warp** tensor-core
mma tile `<atom>/f<FM>x<FN>[/k<bk>]` (atom + register sub-tile + K-chunk) — no worker tokens; the worker halves live
in `WORK`, and `resolve_site_tile` disambiguates an empty site `TILE` beside a thread `WORK` from the coop tier.
Empty = per-cell. The retired embedded-worker spellings (`n<N>[x<M>]/f…`, `a:<atom>/w<WM>x<WN>/f…/k<bk>`) RAISE —
the worker widths have exactly one home, so a value carrying its own cannot decode into a second, self-contained
reading. The `a:scalar` / `a:none` aliases stay pin-only vocabulary for the scalar tier (stripped at parse, never
stored).

**`REDUCE`** (STR codec, the tile schedule) — the reduce-axis partition codec, site-local since step 7:
`[g<n>[a|k]][/coop[-t]][/r<n>]` — `g` cross-CTA split-K (+ finalize letter), `coop` the cooperative-thread fold
(its WIDTH lives in `WORK`; `-t` the transposed lane map), `r` ILP register fold. Empty = serial (the
per-thread remainder is derived, never spelled); the retired `b<n>` coop-width spelling raises. The
cross-CTA split is the `g<n>` field (GRID stage), and the
**finalize** is that field's trailing letter — `g<n>a` = in-place `atomicAdd` (one kernel, additive single-fold
carriers only; both tiers — an mma partial's C fragment rides `RegStore.atomic`, the packed f16x2/bf16x2 red, at the
cost of one output-dtype rounding per partition), `g<n>k` = deferred `__partial` workspace + a sibling combine kernel
(any carrier; the only legal arm for the twisted flash `(m, l, O)` split-KV and for a multi-channel ⊗-combine). Pin
via `EMMY_REDUCE=g2k` (one flat knob — no per-axis `EMMY_REDUCE_<axis>`, no `EMMY_FINALIZE`). The split is consumed by `lowering/tile/030_split_reduce` as a graph rewrite (partial + finalize); the
letter round-trips through `ReducePlan.parse`/`spell` and reads back as `ReducePlan.finalize`. The atomic finalize
applies the kernel's projection epilogue **per partition** before the `atomicAdd`, so it is only correct when that
projection *distributes* over the add (`Σ φ(xₛ) = φ(Σ xₛ)`): a constant scale like `mean`'s `×1/N` distributes and
rides the atomic; a non-distributive epilogue (`l2`'s `sqrt`, a fused bias/activation) is refused
(`NotImplementedError` → pin `g<n>k`, which projects once after the combine). The check is
`030_split_reduce._projection_distributes`.

Two deploy-only dominance/default rules live beside the generic schedule enumeration. A coalesced wide-K `F.linear`
MATVEC (the M=1 contraction-demotion tier) always uses a `b32` single-warp fold unless `REDUCE` is explicitly pinned;
the serial sibling walks all of K in one thread and measured 4–16× slower on DiT conditioning projections. That
dominance rule is generic (any card, any coalesced wide-K matvec), which is why it stays here.

The SKU-exact `facebook/DiT-XL-2-256` deploy overrides that used to sit beside it are GONE — a hardcoded contraction
table, a flash-winner matcher and a `b128` LayerNorm `REDUCE` narrowing, all string-matched on `NVIDIA GeForce RTX
4080`. A recorded winner belongs in the golden corpus, which the deploy evidence tier already consults ahead of the
prior, and which is versioned, auditable through `emmy eval golden`, and covered by the drift gate. What could be
expressed as a golden moved to `goldens/rtx4080_sm89.yaml` (`dit_xl_2.*`, the three exact provisional entries
described in Part 7): the two plain-A projections as `matmul` entries and the block's SDPA as an `attention` entry.

What could NOT be expressed was deleted rather than mis-filed, both times for the same reason: **no golden kind
describes a LayerNorm**. The DiT prologue is AdaLayerNorm-Zero, while every fused kind (`norm_linear` / `mlp_geglu`)
is RMSNorm by construction — its `snippet()` builds `F.rms_norm` — and the reduce kinds are `torch.sum` /
`torch.nn.RMSNorm`. An entry filed under those would join correctly at deploy (the ShapeKey matches numerically) and
hand every re-tune / `eval golden` / drift-gate consumer the wrong kernel to rebuild. So the two LayerNorm→linear
contractions and the LayerNorm statistic reduce now deploy off the prior; adding a LayerNorm-cone kind is what would
let them be recorded.

**`STAGE`** (STR codec, the tile schedule → `lowering/kernel/010_materialize`) — the operand-staging codec
`d<depth>/sync|cp|tma[/split][/p<reg_depth>]` on the typed `Stage` schedule struct (composes with both fragments
of the `TILE` knob): `d<depth>` the gmem→smem ring depth, `sync`/`cp.async`/TMA transport, `p<reg_depth>` the
smem→register double-buffer. `stage=None` (unset / unparseable) = gmem-direct. A `STAGE` value names only what the
schedule CHOOSES — rotation and refill discipline derive at materialization from the depth alone (which is why the
retired `ring` flag compiled byte-identically with and without it), and `smem` / `bk_elems` are resolver outputs,
never spelled. `split` is the transport GROUP GRANULARITY: off = ONE transport over all the fold's staged edges
(a contraction's single multiply consumes both, so there is one group to cut), on = one transport PER edge. It
therefore rides the warp-flash TWISTED stream (`STAGE@<kv>` — the K/V slabs of one streaming block; `reg_depth`
clamps to 1), where `d1/tma/split` / `d1/cp/split` gives each operand its own slab (TMA: its own mbarrier;
cp.async: its own commit group, a uniform `wait_group(1)` completing the older sibling) and each refill lands at
its operand's kill point by the liveness-scheduled skeleton (derived from the segment live ranges, not
hand-assembled), Q staged through smem — the wide (64-key) streaming block's staging. Eligibility is structural:
≥ 2 staged operand edges consumed at DISTINCT positions of the derived evaluation, which is why the matmul
resolvers decline it. See `lowering/kernel/ARCHITECTURE.md`.

**`WSPEC`** (STR codec, RETIRED) — the warp-specialization producer band `p<np>` is INVENTORY: realized rows spell
it as `WORK`'s `+p<np>` suffix, `SCHEDULE_FAMILIES` no longer lists it, no shipped golden carries the key, and the
enumeration neither reads the `EMMY_WSPEC` pin nor offers a `WSPEC` level — pin `EMMY_WORK=w4x2+p2` instead. A stray
`WSPEC` key on a stored row is no longer stripped before matching; it simply names a family no row decides, which
the "family not decided at this fork" rule already reads as free. The `Knob` declaration is gone, and so is the
codec that served it — what survives is one integer, `WarpSpec.producer_warps`, which the materializer reads off
`TileOp.workers`. A band is
legal on a warp `TILE` over a resolved **TMA** `STAGE` within the thread budget (`block_threads + 32·p ≤ 1024`,
`32·p ≤ block_threads`) with no cross-CTA split; an inventory whose band nothing can drive enumerates no row at
all, rather than silently degrading to uniform. Empty = uniform SIMT. Materialized as the staged K-loop's
producer/compute band split (`_stage._producer_band_kloop`).

**`RASTER`** (STR codec, the tile schedule → `lowering/kernel/010_materialize`) — the CTA launch-order
codec (bare/root-global; the fifth schedule-fork level): `gm<G>` iterates `G` M block-tiles fastest per
launch stripe so consecutive CTAs share the streamed B slab (L2 reuse — the flat order streams B from DRAM once per
M-row: `A + C + B×2` measured on the 4090's `mlp_gate_up`, 503.6 vs cuBLAS's 365.8 MB); `gn<G>` is the transpose
(A streamed); empty = the flat N-fastest row-major order (option-0, byte-identical to historical codegen). Changes
no per-CTA work, layout, or schedule — only the block-id decode (`ir/kernel` `Tile.render`, `Tile.raster_axes` the
`grid_tile` eligibility). Enumerated `('', 'gm8')` on 2-D contraction rows; wall-time effect is small and
shape-dependent (±2–4% measured), so the search/goldens arbitrate per shape.

**`S_*`** (FLOAT, `loop/stamp/020_stamp_structural_features`) — the LoopOp's structural features (stmt/op histogram +
loop extents + operand dtypes). Not tunable — identity facts that make a knob dict a complete variant identity (the
online prior's feature vector). Skipped by `format_tuning_knobs`.

**`FAST_MATH` / `F16_MMA_F32_ACC` / `FAST_EXP`** (BOOL, pin-only, the f16-accumulate enumeration gate /
`lowering/kernel/085_fast_exp`) — the **precision-trading family**, never silently on. Precedence per knob: its own
pin > the `FAST_MATH` umbrella > off (`space.precision_pin`). `FAST_EXP` swaps libm `expf` for `__expf`;
`F16_MMA_F32_ACC` offers the f16-accumulate mma atom forks (`a:mma_m16n8k16_f16_f16` — chunked f32 register promote;
its own pin offers on any target, the umbrella only on the consumer dies where f32-accumulate is half rate).
`FAST_MATH` is a meta gate over the others — `unfeatured`, never stamped/enumerated/featurized (the realized fork is
identified by what it enables: `FAST_EXP`'s stamped BOOL, the `TILE` atom token).

### Tree-path schedule keys (the phase-2/3 codec)

A per-node schedule key addresses the node it decorates by POSITION in the recognized tile tree —
`FAMILY@<node-path>[.<axis>][<n>]`, resolved by the ONE walker/resolver in `ir/tile/path.py` (`sites` / `resolve` /
`spell` — total over the sugar levels, idempotent, loud on ambiguity and on a stored short key a structural change
broke). **Short paths are canonical**: the stampers spell the SHORTEST key unique for the kernel's tree, which is
exactly the stored golden/DB spelling — bare `TILE`/`REDUCE`/`STAGE` on today's single-primary trees, `TILE@dd` /
`TILE@pj` on flash (the axis is the real discriminator), `REDUCE@<stat axis>` for the fused kernel's cone statistic
(the path form — `REDUCE@a.fold.k` — when the axis name collides; edge labels `a`/`b` are view-role sugar off the
bilinear parse). Bare-family sugar resolves to the PRIMARY (root-most schedule-bearing) node, so bare `REDUCE` on
norm_linear/geglu still means the contraction's K fold; `WORK` / `RASTER` stay root-global (bare). Since step 7 the
VALUES are site-local too: the worker inventory is spelled once in `WORK` (`w<M>x<N>[+p<np>]` / `t<N>[x<M>]` — the
`+p` band absorbing the retired per-row `WSPEC` key), `TILE` values drop their worker tokens
(`<atom>/f<FM>x<FN>[/k<bk>]` | `f<fn>[x<fm>]`) and `REDUCE` its coop width (`[g<n>[a|k]][/coop[-t]][/r<n>]` — the
finalize letter kept: a MODE, not an axis token); the retired embedded-token spellings raise, and the
golden corpus itself was re-spelled mechanically (715 rows, replay digest-identical; the one-shot script is gone
with the grammar it read).
The reserved
graph-level placement grammar (`in.<operand>` path prefix, leading-`=` value pins) is rejected, never reused. The
golden-spelling tripwire (`tests/.../test_golden_spelling_canonical.py`) resolves every stored knob dict against its
kind's tree and proves every spelling canonical; the
one documented exception is
the dynamic-attention bare `TILE` (its PV plan, matched any-of by the golden layer — a symbolic trace resolves no
stable axis key, so the bare spelling is LIVE corpus semantics, not legacy debt). The tune DB / reservoir /
online prior are REGENERATED after a re-key, never migrated — no reader special-cases pre-phase-3 axis-suffixed
spellings, and `tuning_knob_items` renders keys AS STORED (the old `@<axis>`→bare display collapse is gone). What
remains is the live bare-golden contract: `family_value(knobs, family)` / `pin_key_matches`' bare↔explicit any-of
(how a dynamic attention golden's bare `TILE` matches the masked fork's axis-keyed leaves); it survives the step-7
re-spell deliberately and retires only when symbolic-trace keyed resolution exists.

### Odds and ends

- `BINMASK` parsing accepts a binary string (`"101"` = bits 0 and 2), the keywords `"all"` / `"none"`, or a decimal /
  `0x`-hex int clamped to the candidate width.
- `format_tuning_knobs` leaves `BOOL` knobs out of the rendered `knobs=` line — they are treated as markers saying
  that a pass ran.
- `HOIST_COMPUTE` and `PAD_SMEM` are BOOL autotune forks emitted in a fixed order, with the greedy default first
  (inline-fuse and pad-on respectively); both honor their `EMMY_*` pin.
- The alignment padding for a masked-K MMA block is **not** a fork. It is written onto the `Source` at staging as an
  intrinsic property, because it is almost always a win, and a greedy compile deploys it without needing a re-tune.

## Pass directories

Pass files are numerically prefixed so `sorted()` picks them up deterministically. Pick a fresh prefix when adding a
rule; the loader ignores the prefix itself — it only makes the ordering readable. Per-pass authoring invariants are in
[`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md); the tile passes (`010_recognize` → `030_split_reduce`) and the set
of algebraic rewrites they may apply are documented there too.

| Pass                      | What rules do                                                                                |
|---------------------------|----------------------------------------------------------------------------------------------|
| `frontend/decomposition/` | Rewrite frontend ops (`LinearOp`, `MatmulOp`, `SdpaOp`, layout ops, fused `rms_norm` / `layer_norm` / `softmax`) into tensor-IR primitives + layout-only `IndexMapOp`s, broadcast-explicit via `_broadcast.broadcast_to`. Before `LinearOp` decomposes, `merge_sibling_linears` folds ALL sibling linears sharing one activation (q/k/v, gate/up) into ONE linear over load-time N-concat weights and optional biases (`ConstantOp.source_parts` — the loader concatenates before the `load_ops` chain, zero runtime cost) with `SliceOp` views re-deriving each original output; one launch (one split-K partial+finalize) replaces the per-projection set, and the merged result is a plain matmul that every downstream code path already handles. Guards: pristine exclusively-owned parameters with uniform bias presence, and no sibling whose output reaches a graph output through layout ops alone (the view would demote to a copy kernel at the capture ABI). The concat order is graph-insertion order — canonical regardless of match enumeration, because the buffer layout is ABI for goldens and packs. |
| `frontend/optimization/`  | `compose_indexmaps`: collapse chains of single-source / single-consumer `IndexMapOp` into one coord_map, so trivial layout kernels don't block fusion. |
| `loop/lifting/`           | `lift_*` rules wrap each surviving tensor primitive in a trivial one-op `LoopOp`.            |
| `loop/fusion/`            | `split_shared_indexmap` dissolves a fan-out pure-indexmap into separate consumers when its branches do not reconverge; `merge_loop_ops` uses the same N-way splicer for adjacent pairs and closed reconvergent producer DAGs, preserving shared SSA definitions instead of treeifying them; `dedup_loads` drops identical `(input, index)` Loads; `fold_output_reshape` retargets a producer's `Write` through a graph-output memcpy-identity flatten (verified exactly over the finite domain; clean affine re-decomposition onto the output strides) — the copy kernel the splicer cannot take (a producer that carries a reduce, read through a div/mod index map). Folding scalar-constant broadcasts into consumers cuts Qwen3-Embedding-0.6B from 394 → 337 kernels. |
| `loop/recognize/`         | Empty (retired) — flash / online-softmax recognition moved into `lowering/tile/010_recognize` (the `_flash` / `_softmax` helpers), so the loop dialect carries no pattern recognizers. |
| `loop/stamp/`             | `stamp_loop_names` (`provenance.name_for`, e.g. `k_rms_norm_3f2a1b`) + `stamp_structural_features` (the `S_*` dict). Runs last in the loop dialect — after fusion and recognition — so every kernel is named / stamped against its final body. |
| `lowering/tile/`          | `LoopOp → TileOp` over the block-DAG Tile IR: `010_recognize` (structural — reads the algebra off the `LoopOp` body and emits an UNMAPPED `TileOp`) → the schedule step (REMOVED — see Part 9) → `030_split_reduce`. Dispatch is on the fold's derived role (`Fold.role` — `FREE` / `PLANAR` / `CONTRACTION` / `TWISTED`), never a named shape. |
| `lowering/kernel/`        | `010_materialize` is a `TileOp → KernelOp` tier dispatcher (scalar / `_reduce`). A tiled `CONTRACTION` arrives as a `Fold` already **built recognize-side** in the bilinear shape (`is_contraction` is the reading, not a kind) (`lowering/tile/010_recognize._nodify_contraction` — one flat node splitting the algebra params (axes / operands / acc / epilogue) from the schedule, which the fork places onto the grid), so materialize only synthesizes its bare grid-`Write` and **expands** it through the one atom-generic `_factor.factorize` over the shared tiling layer (in `_factor.py`) (the geometry is derived on the PLACED `TilePlan` slice, the algebra on the node; `_atom.reduce_codegen` emits the shared K-loop and a swappable `store` sink, dispatched off the atom). Then the Kernel-IR peepholes: `030_stamp_types` (+ `040_demote_to_write_dtype`) resolve dtypes, `050_vectorize_loads` / `080_vectorize_stores` / `095_interleave_loads` pack/reorder memory ops, `110_drop_redundant_syncs`. See [`passes/lowering/kernel/ARCHITECTURE.md`](passes/lowering/kernel/ARCHITECTURE.md). |
| `lowering/cuda/`          | `delegate_zero_init` (first) moves an atomic accumulator's per-launch zero-init off the runtime memset and into a dataflow-predecessor kernel as a `ZeroPrologue` stmt (CTA 0 writes zero words; stream order guarantees happen-before) — one CUDA-graph MEMSET node saved per site; the capture's first launch, symbolic-shaped accumulators, and accumulators past the one-CTA break-even cap (`_MAX_DELEGATED_WORDS`, 64 KB — CTA 0 zeroes serially, so a large buffer costs more than the MEMSET node it replaces) keep their memset, and the slab planner starts the buffer's live interval at the delegating launch (`CudaOp.zero_prologues`). `lower_kernelop` then renders the `KernelOp` body to a `__global__` source string (`ir/kernel/render.py::render_kernelop`) and mutates the node's op to `CudaOp` in place. |

## Dump hooks (`dump.py`)

`CompilerDump.on_pass(idx, pass_name, graph)` dumps the post-pass graph uniformly for every pass:
`NN_<pass_name>.{json,txt,dot}` (+ `NN_<pass_name>.kernels.txt` if any node has a non-empty `pretty_body()`). Slashes
in the pass name flatten to underscores. The pre-pipeline input graph is dumped separately as `00_input.*` via
`dump.dump_input_graph(graph)`. The uniform strategy means adding a pass automatically gets dumped — no registration.

Per compute kernel, `_dump_per_kernel` writes `<prefix>.kernels/<kname>.json` — a standalone lowered sub-graph
(kernel + its `InputOp` / `ConstantOp` producers) loadable via `emmy run --ir`. Original frontend slices selected by
provenance stay in memory for tune benchmarking and are never written as trace artifacts.

## Per-rule diff output (`rule_diff.py`)

At `compile -vv` (DEBUG) the engine emits one block per rule application: a unified diff between the matched subgraph
and the rewritten fragment, bracketed by `>>> <pass>:NNN_rulename` and `<<< <pass>:NNN_rulename` markers. The `<pass>`
prefix is the single-letter shorthand from `PASS_SHORTHAND` (`d` / `o` / `l` / `f` / `t` / `k` / `c`) — the same
letters the CLI accepts in `--passes dolft` (`commands/compile.py` imports `PASS_SHORTHAND` so the flag and the marker
prefix can't drift). Skipped rules collapse to a one-liner. The bracketing makes per-rule / per-pass slicing trivial
via `awk`; ANSI color is applied only inside the diff body so the markers stay plain ASCII. Color follows
`compile --color`. Body-carrying ops render through their own `pretty_body` (the in-flight `TileGraphOp` pretty-prints
its block-DAG), so a tile-pass diff reads as a readable block-DAG delta. The structured `.rules.json` dump is
unaffected — the diff is purely presentation.

## Invariants

- A rule module must not reach into the engine's internals; its interface is `PATTERN` + `rewrite(graph, match)`.
- `pipeline/` imports from `ir/` but never from `backend/`. Lowering rules produce IR; executing that IR is the
  backend's job.
