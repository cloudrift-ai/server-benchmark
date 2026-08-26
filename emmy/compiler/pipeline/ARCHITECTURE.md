# Pipeline Architecture

The pipeline is the part of the compiler that turns a traced graph into finished CUDA kernels, one rewrite at a time.
This document explains it end to end for someone new to the code. It assumes you know the shared vocabulary in
[`GLOSSARY.md`](../../../GLOSSARY.md) — fork, knob, candidate, prior, evidence, golden configuration — but nothing
about the internals of this package. Words that carry a special meaning inside the pipeline are explained in plain
language where they first appear; the few that also turn up in neighboring documents are in the glossary.

Four companion documents cover what this one doesn't:

- The rewrite rules themselves and their authoring invariants → [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md).
- What each IR dialect looks like → `ir/ARCHITECTURE.md`.
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
| 7 | Golden configs and the A/B integrity gates | you are recording or replaying goldens |
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

1. **The verified goldens** recorded for this GPU — reviewed measurements that ship with the repository, joined by
   STRICT structural identity (the recognized term's algebra digest + dtype fingerprint + axis-extent fingerprint,
   derived record-side from the record's own persisted program) and decoded by EXACT spelled-row equality (Part 7).
   Fail-closed: a record that matches the identity but equals no enumerated row is drift — a loud warning, never a
   fuzzy acceptance.
2. **Measured evidence** — first the measurements stored inside the online prior's checkpoint (its **reservoir**)
   that were taken at deployable flags, then rows from the tune database (Part 3).
3. **The prior** — the online model when trained and calibrated, the offline model otherwise (Part 3).
4. **Option-0** — the first option in the order the rule emitted them. Rule authors order options so this is always
   safe.

That order has a name — the **deploy evidence hierarchy** — and each numbered step in it is called a **tier**. The
list above is only a summary. **Part 3's "The deploy evidence hierarchy" is the authoritative statement** of the exact
order, of what each tier holds, and of the rule that the reservoir tier applies only to a compile at deployable `-O3`
flags.

Structural forks — the ones that change which kernels exist — are stricter. The prior never ranks their options
directly; the compiler compares whole-kernel-set costs, priced by measurements first and any loaded prior — the
offline model on a cold machine — for the remainder (Part 4).

### The four stores

Four stores hold everything a compile can know. They have different writers, different readers and different
lifetimes, and telling them apart is the single most useful thing to learn early.

| Store | Where it lives | Written by | Consulted by |
|-------|----------------|------------|--------------|
| **Golden configs** | model YAML under `recipes/<model>/golden/`; model-agnostic YAML under `search/goldens/` | promoted from deployable `run --bench` golden / `--ab` rows (Part 7) | greedy compile (tier 1, the verified tier); pinned replay (`run --golden NAME`); `emmy fit` trains the offline prior on them; `emmy eval` datasets |
| **Reservoir** | inside the online prior checkpoint (`~/.cache/emmy/online.json`) — the sample of past measurements the model trains on | `emmy tune` — every deployable-regime training row | greedy compile (tier 2); the online prior's own refits |
| **`perf` table** | the tune DB (`~/.cache/emmy/autotune.db`) | `emmy tune` — terminal kernel measurements plus validated whole-slice structural routes, at the sweep's flags | greedy compile (tier 3); the per-variant replay cache |
| **`node` table** | the same tune DB | `emmy tune` (every search-tree node) and `run --bench` (rows benched with hand-forced knob values) | `emmy eval` diagnostics — **never** consulted at deploy |

Of the four, only the goldens travel with a clone: they are the only *measured* data a fresh machine has. The
reservoir and the tune DB are machine-local caches written by local tunes, so a freshly rented box starts with the
goldens plus the shipped offline prior artifact and nothing else.

```
WRITERS                                STORES                                READERS

emmy tune ─┬─ sweep benches ─────────▶ perf table   (autotune.db) ─────────▶ greedy compile, tier 3
           ├─ every training row ────▶ reservoir    (online.json) ─────────▶ greedy compile, tier 2 (H_opt=3 rows)
           └─ every tree node ───────▶ node table   (autotune.db) ─────────▶ emmy eval only (never a deploy)
run --bench pinned/golden/--ab rows ─▶ node table   (autotune.db)
recorded from those rows ────────────▶ recipe-local / hardware golden YAML ─▶ greedy compile, tier 1 + pinned replay
                                                                  └─ emmy fit ─▶ offline_weights.json (repo)
                                       offline_weights.json ──────────────▶ greedy compile, tier 4 (cold)
                                       online prior model (online.json) ──▶ greedy compile, tier 4 (trusted)
```

Everything above is measured in ONE regime: the deployable one a compile runs in. A sweep benches at the flags a
deploy compiles with, so a tuned latency is the deployed latency and no store needs a per-regime lane (Part 3).

### How one fork gets decided, end to end

A worked example, to fix the vocabulary. Take `emmy compile` on a machine with a tuned checkpoint. A tile-lowering
rule matches a `LoopOp` and returns several tile options.

1. The engine turns the option list into a lazy fork tree and hands the fork point to `greedy_decide` (Parts 2, 4).
2. `greedy_decide` **flattens** the fork to its complete leaves — knob dicts only; no kernel is built yet (Part 4).
3. Each leaf becomes one row: the compile context's `H_*` features (which GPU, which nvcc flags), the `S_*` features
   an earlier pass wrote onto the op (a summary of its body and loop extents), and the leaf's complete knob values
   (Part 6).
4. **The reservoir tier.** The leaf that agrees with the fastest reservoir row of the same op — agreement means
   every knob the leaf has decided has the same value in the row. (The example starts here because this card records
   no golden for the op; the **verified** tier would otherwise decide first. Part 3 numbers the full list.)
5. **The `perf` tier.** Otherwise: measured rows for this exact op, under this compile's own context key.
6. **The prior.** Otherwise: the `mean_scores` argmin over all leaves, in one batched predict.
7. Ties at every tier break by `knob.canonical_row_key`, never by the order the rule emitted its options in.
8. The winning leaf is built for real. The µs of whichever row decided it is written onto the fork's
   `Decision.score`, and the resolve moves to the next fork.

With no evidence and no prior at all, every fork falls to the first emitted leaf — not a chosen default, just what
is left when there is nothing to rank with (env pins still apply — a pinned family never reaches a decide).

### Terms used throughout

Everything in this table recurs on nearly every page below. The rest of the document uses these words freely.

| Term | Meaning |
|------|---------|
| **rule** | One pattern + rewrite function in a `NNN_<name>.py` file under a pass directory. |
| **pass** | An ordered directory of rules; the pass layout is frozen in a `Pipeline`. |
| **candidate** | One in-flight compilation state (a graph snapshot part-way through the pipeline). |
| **fork** | A rule returning multiple alternatives; the engine turns each option into a child candidate. |
| **knob** | A named tuning dimension (e.g. `TILE`, `STAGE`). Every fork option is identified by the knob values it fixes. |
| **to pin a knob** | To force a knob's value by hand instead of letting the compiler choose — from the environment (`EMMY_STAGE=d2/smem-async`), or by reproducing a golden entry's recorded values. A *pinned row* is a benchmark of such a forced configuration. |
| **to stamp a value** | To write a value onto an op as metadata, where later passes and the prior can read it: the `S_*` shape/body features, knob values, scheduler facts. "The op's stamped `S_*` features" means the ones an earlier pass wrote onto it. |
| **to realize** | A recorded configuration *realizes* at a fork when the options the compiler actually offers there include one that matches it. A recording that realizes nowhere cannot be deployed, no matter how good its recorded µs. |
| **regime** | The compile settings a measurement was taken under, or that a compile is running under: mainly the nvcc optimization level (`H_opt`) — `-O3` is the **deployable** one, and the only one anything is measured in — plus whether fast math is on. |
| **prior** | The ranking model — the fit-offline **offline prior** when cold, the CatBoost **online prior** trained from local measurements once data exists. |
| **terminal** | A fully-lowered candidate (every fork on its path resolved) that can be benchmarked. |
| **golden record** | A reviewed program-backed schedule measurement, selected by frontend provenance and used as deploy evidence and an A/B reference. |
| **`Op.cache_key`** | A name-invariant digest of an op's body + knobs — the identity measurements are stored under. A `TileOp`'s structure digests as the α-invariant term hash (`Fold.structural_key`), never the lowered nest. |

## Module map

| Module | What lives there |
|--------|------------------|
| `pipeline.py` | Engine core: `Pattern` / `Match` / `Rule` / `Pass` / `Pipeline` (the frozen pass layout) plus `Run` — the per-run state and engine loop. |
| `fork.py` | The `Fork` interface (`OptionFork`) and the reusable `Level` + `build_fork_tree`, which builds a tree of knob-value combinations lazily. |
| `knob.py` | The `Knob` descriptor system and the `EMMY_<KNOB>` env namespace (borrowing `config.knob_var` / `config.knob_raw`; `format_tuning_knobs` renders the real tuning knobs for `tune` output). Holds NO concrete knob declarations. |
| `search/space.py` | **The single home of the search space.** Every `Knob` instance is declared here and nowhere else — the schedule codecs (`WORK` / `TILE` / `REDUCE` / `STAGE` / `RASTER`), the kernel-lowering policy knobs (`VECTORIZE_LOADS` / `INTERLEAVE_LOADS`), and the enumeration value grids (`scalar_tile_moves` & co). A rule that decides a knob imports it from here; registration is construction (`Knob.__post_init__`), and `knob.registry()` imports `space.py` before answering, so the registry is complete in any process. |
| `search/domain.py` | The candidate domain as a **constrained integer set** — `Dimension` (a name + its finite integer values), `Bound` (`coeff · ∏ dims` `<=` / `==` / `divides` a limit) and `Space` (enumerate the legal points, or ask whether a recorded one is still a member). The constraints that bound a schedule family are products of the unknowns, so the feasible set is not convex and no coordinate change makes both the products and the budgets affine at once; the answer is to keep integer coordinates and enumerate, pruning each prefix the moment a running product overruns its bound. Generation machinery only — it holds no schedule family today (`space.py`'s grids are still curated), and categorical legality stays with the scheduler. |
| `search/features.py` | The featurizers (`knob_features`, `tile_signature`, the `D_*` / `MMA_*` encodings) — kept beside `space.py` so the whole space (dimensions × values × encoding) is analyzable in one package. |
| `search/db.py` | `SearchDB`, the persistent SQLite store (Part 6). |
| `search/policy/mcts.py` | The in-memory MCTS (`SearchTree`) colocated with its only reader, `TuningSearch`. |
| `search/policy/greedy.py` | `greedy_decide` — the no-tree fork resolver used by `compile` / `run`. |
| `search/strategy/` | The search shapes: `base.SearchStrategy`, `greedy.GreedyStrategy`, `two_level.TwoLevelStrategy`. |
| `search/prior/` | The ONE ranking path: a `Prior` ABC with the cold `OfflinePrior` and the `OnlinePrior` composed behind `FallbackPrior` (`load_prior`). `linear_model.py` holds `LinearModel`, the offline prior's scoring function as a value object — the one definition the fitter optimizes and the deploy path ranks by. `diagnostics.py` backs the `eval` reachability / calibration reports; `fit/` is the offline fitter, split by responsibility — `linear.py` trainer, `cv.py` fold harness, `tables.py` the rank-table rendering, `run.py` the pure `emmy fit` run harness. The candidate pool it all trains over is `search/data/group.Group`, one layer down: a pool is data, not a fitter detail. |
| `search/metrics.py` | What a scored candidate pool is worth, as pure functions over numbers: golden ranks and their tie conventions, `topk_pick` / `topk_regret` against measured latencies, and Spearman ρ. No model, no I/O, no strings, so the callers cannot each hold a slightly different definition — the rank metrics, the three calibration paths and the reachability ratio all resolve here. Rendering lives with the caller (`prior/fit/tables.py` for the fit's rank tables; the other top-k summaries have not been unified yet). |
| `search/data/` | The harmonized read-view over the three data sources (golden records / DB `perf` rows / prior reservoir): `Sample`, `Dataset`, the derived `ShapeKey` index, and `group.py`'s `Group` — one candidate pool packed as a matrix plus one label per row. The base says nothing about what the labels mean, which is all a ranking metric needs; `GoldenGroup` is the subclass whose labels MARK rows (`golden_ids`) rather than measure them, and only it can be asked which rows are the answer. `group_measured` builds base groups from benched node rows, labelled with measured µs. Nothing here imports `search/prior/`: a group carries every column it was given, and each model class narrows to the ones it wants when it asks for the matrix — `TREE_FEATURES`, the view argued entirely from what a tree can re-derive, lives with the CatBoost trainer for the same reason. |
| `search/golden.py` | Generic program-backed records, a repository corpus loaded on first evidence access, stable-format validation, and lazy provenance-derived structural indexes (see Part 7). |
| `search/audit.py` | The verified-tier drift audit: one MATCH / DRIFT / GAP verdict per consultation, collected off a whole card's graphs under isolated evidence. Backs the `emmy eval golden --serving-config` release gate. |
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
Matches retain the watched node objects themselves, so `Match.is_alive()` rejects removal followed by a different
node at the same graph id even when the Python allocator would otherwise recycle an integer object address.

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

### Strategies — engine events for cross-cutting concerns

The engine is IR-dialect-agnostic: it emits a small fixed set of EVENTS — frozen records of an engine moment
(handlers act on the compilation state an event references, never on the event) — and never branches on pass
names, dialects, or per-concern flags. Three prefixed protocols share the "strategy" vocabulary, each with its own
ABC, told apart by what the loop is doing when they act:

- **`PipelineStrategy`** (`pipeline/strategy.py`) — reacts to what the loop DID (events below): the cross-cutting
  concerns (provenance, identity, the kernel inventory). Never steers the search; the loop's trajectory is
  identical without them.
- **`Search`** (`search/policy/base.py`) — answers what ONE loop ASKS during exploration: frontier ranking
  (`push`/`pop`) and terminal valuation (`evaluate`); `TuningSearch` is the realization, and `greedy_decide`'s
  decide callback is the deterministic sibling for `Run.resolve`.
- **`SearchStrategy`** (`search/strategy/base.py`) — the search SHAPES above the loop: how many loops, over which pass
  lists, with which policy inside, and what the results mean together (`GreedyStrategy`, `TwoLevelStrategy`; each
  implements `run(graph, ctx)`).

The composition chain for a tune: a `SearchStrategy` constructs runs with a `Search` policy inside; the loop emits
events that `PipelineStrategy` instances act on. Each layer only knows the one below it. Every cross-cutting
concern is a `PipelineStrategy` implementing the event methods it cares about; extension is a new strategy over
the existing events (or a new event field), never a new engine parameter. The events, each a payload object: `RunStartEvent` (a loop starts driving a graph),
`SpliceEvent` (before a `Graph` fragment splices in — op identities stable; strategies may mutate fragment OPS,
never the graph or cursor), `SplicedEvent` (after the splice, carrying its `SpliceReceipt` — `Graph.splice` is pure
surgery and hands back what it did), and `PassEndEvent` (a named pass completed a quiescent scan).

Two binding scopes share the protocol:

- **Discovered** (build-scoped): strategy modules are plain `.py` files at the top level of `passes/`
  (`passes/provenance.py`, `passes/identity.py`); `Pipeline.build` collects every `PipelineStrategy` subclass they
  define into shared instances (`strategy.discovered_strategies`), class-name-sorted. Dispatch order MUST NOT be
  load-bearing — no strategy may depend on another having handled an event first — which is why identity is a
  computed function, not a stamp others wait on. Build-scoped instances are shared across runs and candidates:
  immutable config plus content-keyed caches only, never trajectory state.
- **Composed per run** (`Pipeline.with_strategies`): instances with per-run state — e.g. the two-level tuner's
  minted-kernel watcher — composed into the run's own pipeline instance after the discovered set. A pipeline
  composed with stateful strategies serves one run; sharing across runs is only safe when every strategy is
  stateless.

The two discovered strategies: **`ProvenanceStrategy`** owns op provenance end to end (`seed` at run start,
`propagate` from the splice receipt, mint for `frontend/decomposition`'s fragments, aggregate for everything else)
and keeps the replaced result's ultimate `Op.source` object on its rewrite fragments. The pattern root may be an
upstream producer while `Match.output` names the consumer result that the fragment replaces. A fragment may consume
inputs from other origins without losing that result identity; those producer edges retain their own sources. The
source identity lets semantic rewrites distinguish a frontend operation's private decomposition edges from tensor
boundaries between operations. A pipeline built without the strategy has no provenance anywhere, and `graph.py`
imports none of it.
**`IdentityStrategy`** owns the `S_*` structural identity: computed (`structure_features`) and materialized into
`op.knobs` exactly once per kernel, at birth — fusion-born kernels at the loop dialect's end (`PassEndEvent` of
`loop/stamp`), minted pieces (a cut's fragments, a split's pieces) at the splice event, before the fragment even
enters the graph, so no rule at any cursor position can observe an unstamped kernel. The stamped row rides the
engine's rebind knob-merge into every later dialect (which is what keeps a terminal `CudaOp`'s cache key, DB rows,
and prior feature columns carrying the loop-birth identity). It also threads decomposition attribution
(`Op.source`) and serves the read API (`signature` / `op_sig` — knobs-first, compute-fallback) every identity
consumer goes through. The search shapes (`SearchStrategy` subclasses) are the same idea one level up — they own
loop composition and terminal aggregation — but are constructed by their entry points, not discovered.

A rule always sees **graph-true operand Tensors**: op `inputs` / `outputs` are refreshed from the graph at match build
AND again at apply time (`Candidate.try_rewrite`). This matters because an earlier apply in the same batch may have
swapped a consumed node's op for a rebuilt instance still carrying its `(f32, ())` seeding placeholders — a change
`Match.is_alive`'s node-identity check cannot see. (This was the gemma o_proj misdeploy: a scalar tile shipped at 16x
the kernel's measured mma rows because the warp atom gate read placeholder dtypes off an all-f16 graph.)

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
`Graph` leaf.

A fork whose levels form a cartesian product of knob values reuses **`build_fork_tree`**. A rule supplies one `Level`
per level plus a `materialize=` callable, and gets back a lazy root `_Branch` whose `expand()` builds children on
demand, in grouping order. The algorithm — group the parameters by each level's knob keys, collapse a level with one
key, skip a level with no keys, and defer building a leaf until `expand()` — lives once in `fork.py`.

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
and the design it retired.

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

`offline_weights.json` is the one artifact anything loads by default. A sibling file in that directory is a **scoped
experiment**, not a second default: `offline_weights_matmul_rtx5090.json` is fit on RTX 5090 matmul goldens alone and
is reached only by pointing `EMMY_OFFLINE_FILE` (or `--offline-file`) at it. Each such file says so in its
`provenance.scope`; read that before drawing conclusions from one, because a scoped artifact has no reason to beat
the shipped weights outside the slice it was fit on.

What a newcomer needs to know about the fit:

- **The fit optimizes the deployed score itself, not a linear stand-in for it.** Both sides go through one
  `LinearModel`, which offers the same arithmetic in two access shapes: a per-dict entry for scoring a live
  candidate, and a matrix entry for scoring a whole candidate pool (one fp16 golden enumerates ~78k rows, so the
  fitter cannot use the dict path). The non-linear term's weight and threshold are fitted alongside the feature
  weights — the optimizer is derivative-free, so a threshold costs it nothing. A scoring constant the fit cannot see
  is a constant the fit optimizes *around*: while two hand-set gates sat outside the objective, the reported golden
  ranks were not the deployed ones (on the RTX 5090 matmul goldens, median rank 228 reported against 367 deployed).
- **The trainer is an object, and fitting is pure.** `LinearTrainer` carries the hyperparameters — feature names, the
  incumbent to chain from, sample count, L2 strength, seed, warm start, and the ranking loss — and `fit(groups)`
  returns a `LinearFit` without touching the trainer. One instance therefore serves every cross-validation fold with
  no copying, and a fit is a function of its inputs alone. The two seeding policies are data rather than code: the
  full-train fit warm-starts from the incumbent, and the fold trainer is the same object under
  `replace(trainer, warm_start=False)`, because the incumbent's weights were fit on every golden and warm-starting a
  fold from them would leak each held-out golden into the model meant never to have seen it. Both are recorded in
  the metrics header, along with the loss — two fits are only comparable under the same one.
- **A group is a candidate pool, and it may have more than one right answer.** `GoldenGroup.golden_ids` is the
  set of rows in that pool a golden verified: usually one, several when the builder matched
  several goldens onto one pool (the same shape recorded under two names, or one name recorded twice). Which
  goldens share a pool is settled before any group is built, so a group's labels are final at construction.
  The per-group term is then the BEST rank over that set (`search/metrics.best_rank`), because deploy ships one
  config: any acceptable one ranked first is the win, and a mean would spend weights pushing up the runner-up.
  At one positive it is the single-golden rank exactly, so the
  supervision generalized without moving any fitted artifact. The sibling positives also stop being drawn as the
  tree fit's negatives, which had been teaching it that a measured-good config was bad.
- **A pool may be a SAMPLE of itself.** `emmy fit --pool-sample N` draws its candidates during enumeration
  (`search/pool.py`), so `Group` carries both the drawn rows and `total`, the true pool size. The linear
  trainer's z-scoring is over the FULL pools' moments, now estimated rather than counted: each group's rows
  carry weight `total / len(feats)` in the two streaming passes, so a 5-row pool and a 325k one do not weigh
  the same under fixed-size sampling — which would otherwise change the standardization and with it the
  raw-space L2 the artifact ships. Unsampled every weight is exactly 1.0 and the arithmetic is bit-identical,
  so a full-pool refit reproduces byte for byte.
- **The loss has two parts**: an objective that pushes each recorded golden's rank up inside its own candidate set —
  each group counting once — plus an L2 penalty in
  raw feature units (`DEFAULT_L2`, CLI `--l2`). The penalty exists to make the fit **well-determined, not to shrink
  the weights**. The rank objective barely moves when you scale a feature that hardly varies across the golden
  candidate sets, so an unpenalized fit is free to pick an arbitrarily large weight there. That is invisible in
  golden-rank metrics and catastrophic when scoring a fork, where a not-yet-decided knob scores such a feature 0.0.
  The penalty must be in raw units (`w_z/sd`), because after de-standardizing, the inflated weight looks like an
  ordinary O(1) weight.
- **Loading is strict.** A missing artifact, or one whose `feat_ver` does not match, is a hard error — refit it, never
  a silent fallback. The error comes from the artifact loader, and it surfaces in `tune` / `eval`, which load the
  prior directly. A greedy compile wraps `load_prior` best-effort, so there a bad artifact does not abort the compile:
  it produces the no-prior resolve described under the hierarchy below (first leaf, with the DB tier lost along with
  the prior object). A weight key that is no longer used, inside an artifact of the current version, is
  simply ignored. `EMMY_OFFLINE_FILE` (or `emmy eval … --offline-file`) swaps in a candidate fit for an A/B.
- A separate `weights_dynamic` set ranks kernels whose tiles are masked because an axis is symbolic; it is selected on
  the stamped `S_ext_n_symbolic_axis`. That stamp **routes and never carries a weight**: the dataset packs it like
  any other column, and the linear fit narrows it out of its own descent coordinates (`descent_cols`) while a tree
  splits on it to price both regimes in one model. The reason is identifiability: the stamp is constant
  across a candidate pool, so a linear term on it shifts every candidate equally and cancels out of the within-pool
  ranking. The rank objective cannot see such a term at all, which makes whatever value a descent lands on there
  noise rather than a fitted quantity.
- One feature interaction sits outside the linear weights, because it cannot be written as one: the atomic-free
  split-K term, which rewards the deferred combine kernel above a split-count threshold and penalizes it below.
  Its weight and its threshold are both fitted. `D_scalar_on_warp_eligible` and `D_splitk_roundtrip` — which express
  a preference for the tensor-core path, driven by the per-kernel `S_warp_eligible` value the scheduler stamps — used
  to carry hand-set coefficients here as well. They are plain linear terms on features the weight vector already
  holds, so they were double-counting constants the fit could not see, and the fitted weights now carry them alone.
- **The linear quality score is turned into a positive stand-in for latency by an exponential**
  (`exp(-scale·quality)`), whose argument is clipped only at the point where floats stop being safe (~±700). **That
  exponential must never flatten out over the range of quality scores that actually occur.** A clip inside the live
  range collapses good candidates onto one identical value, and the argmin then falls back on the order the options
  were emitted in. The one consumer that needs a bounded value —
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
| `policy(knobs_list)` | MCTS selection (PUCT) only | How much the model prefers each sibling in ONE fork's set, normalized within it. The one call that may combine the two halves (see the calibration-gate section below). |
| `mean_score` / `mean_scores` | deploy + eval ranking | The model's latency prediction for one row / for a batch of candidates. `FallbackPrior` routes these to the online half when it is `trustworthy`, else to the offline half — no blending. |
| `evidence_pick(rows)` | deploy tier 2 | The pick made from measured reservoir rows (defined below). Returns `(index, measured_µs)` or `None`. Consulted whatever the calibration verdict says, because measured evidence needs no trusted model: a quarantined model — or a checkpoint whose reservoir has rows but no fitted model yet — still supplies this tier. |
| `pick(rows)` | deploy + eval | `evidence_pick` first; when no candidate has evidence, the `mean_scores` argmin with the canonical tie-break. Returns `(index, µs)` — a measured µs when evidence decided, a predicted one otherwise. This covers tiers 2 and 4 only: `greedy_decide` puts the verified tier above it and the DB tier between the two, so the `Prior` never owns the whole hierarchy. |
| `sig_groups` | both measured-evidence tiers | How a candidate is matched to measured rows by its `S_*` features. It still matches when the feature set has changed since those rows were written (Part 4) — one rule shared by the reservoir tier and the DB tier. |
| `trustworthy` | the check that lets the online model decide | `fitted` AND passing the calibration gate. |
| `mean_score_features` / `mean_scores_features` | the model classes' own seam | Scoring a row that is ALREADY in feature form. `mean_score` / `mean_scores` featurize and delegate here, so a model class implements the featurized half only. Not a pool-scoring surface — that is `score_rows`, which projects a packed matrix and is what the fitter and the evaluation report use. |

### The deploy evidence hierarchy

`TuningSearch` (`tune`) ranks the PUCT frontier with the prior's `policy`. `greedy_decide` (`compile` / `run`, via
`Run.resolve`) never explores: at each fork it picks once, working down the list below from the top. **This list is
the authoritative order** — the summaries elsewhere in this file defer to it.

**These four tiers are the whole ranking mechanism.** Three of them are recordings of something that ran and the
fourth is a model fitted to such recordings; there is no fifth, hand-written tier anywhere below them. The passes
that produced the candidates ordered nothing, defaulted to nothing and withheld nothing (see
[`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md)), so when all four tiers are silent the pick falls out of the
enumeration's emission order, which carries no meaning. Such a pick can be far off the best kernel in the space,
and that is an accepted outcome of the design — the fix is a measurement, a recorded golden, or a better-fitted
prior, never a preference written into a pass or into this policy.

1. the **verified goldens** recorded for this GPU (`greedy._verified_index` / `_verified_pick`): the record whose
   `deploy_identity` — the recognized term's α/buffer-invariant algebra digest folded with the operand/output dtype
   fingerprint and the axis-extent fingerprint (static sizes and symbolic markers, never hints — the α-invariant
   digest canonicalizes sizes away, and without extents every same-algebra cone on a card would share one key),
   derived record-side through the shared recognition core (`_lift.recognized_tile`) — equals the
   fork's, and whose spelled row (`knob.schedule_row_key`, the recording canonicalizer restricted to the schedule
   families) equals EXACTLY one enumerated leaf. Fastest matching record first; a record that matches the identity
   but equals no leaf is DRIFT — a loud warning and nothing else (fail-closed). A ROUTING record (`PLACE`-only
   knobs) decides the placement fork by picking the cut fragment whose parent piece stamps its keys, and a fused
   schedule record holds the fused side against the prior. The tier needs no prior, applies only at deployable
   `-O3` flags, and scopes records to the live card and the exact live pin regime;
2. measured **reservoir** evidence (`Prior.evidence_pick`): the candidate that agrees with the fastest reservoir row
   of the same op that was measured at `-O3` (`H_opt=3`);
3. the tune DB's measured `perf` rows for this compile's own context key — one lane, because a sweep measures in the
   regime a deploy compiles in. A real measurement of this exact op beats the model's extrapolation. Rows from a
   deliberately non-deployable `--nvcc-flags` run key elsewhere and are simply never consulted;
4. the prior's `mean_scores` argmin — only when no candidate has any evidence at all. Score ties break by
   `knob.canonical_row_key`, never by the order options were emitted in.

Pins sit ABOVE the whole list: replaying a record's pins + knobs (`run --golden NAME`, `--ab`, `EMMY_KNOBS`)
settles the pinned families before any fork reaches a decide.

**Auditing tier 1.** Whether the recorded goldens still decide is a question about the tier itself, so `_verified_pick`
carries one supported seam for it: `greedy.golden_audit(records)` installs a verdict sink that every SCHEDULE
consultation appends to — `MATCH` (a record carrying the fork's identity had a row equal to exactly one enumerated
leaf), `DRIFT` (records carry the identity, no offered leaf equals any of their rows — the fail-closed branch), `GAP`
(no record carries it). Unset, the sink is one identity test per consulted fork and nothing else. `search/audit.py`
drives it over a whole card's graphs with the machine-local evidence removed (`config.online_file_override` at a
nonexistent path, `nvcc_flags_override("")` for the deployable regime, `golden.records_override` scoping the corpus to
one file or precision lane), so the verdicts are the same on a GPU-less box and the recording host. `eval golden
--serving-config` is the consumer, and it also ratchets `consultation_counts` — the count is the one thing the
verdicts cannot report, because a kernel whose lowering stops enumerating candidates deploys single-option, consults
nothing, and loses its recorded MATCHes with zero DRIFT.

Three definitions the list leans on:

- **What "agrees with" means** (`evidence_row_vouches` in the code; the same rule serves the reservoir and DB
  tiers): a measured row counts as evidence for a candidate when every tuning knob the candidate has decided so far
  has the same value in that row. Knobs the candidate has not decided yet are free — a later pass will decide them.
  That is what lets one fully-decided measured row settle a fork whose candidates are still only partly decided. At
  a placement fork, `PLACE` is the exception: each candidate's complete `PLACE` subset must exactly equal the measured
  row's subset, including the fused candidate's empty subset. A cut measurement therefore vouches only for that cut,
  while the same prefix rule continues to apply to every non-`PLACE` knob.
- **The reservoir** is the online prior's own training dataset: a bounded uniform sample (Algorithm R, capped at
  `MAX_ROWS` = 100k) of every training row ever streamed in across runs, stored INSIDE the online checkpoint
  (`online.json`, Part 5). Its rows are all `H_opt=3` — `Prior.add_rows` admits no other regime — and they double as
  deploy evidence, so tier 2 is not a separate store. The reservoir sits above the DB tier because it is the online
  prior's own view of what it has seen, and because it carries the value-of-position labels the DB's per-kernel rows
  do not. One consequence: anything that discards the checkpoint — a `FEATURIZER_VERSION` bump discards it WHOLE, see
  "Featurizer versioning" — deletes this evidence tier along with the model, and the machine's deploys drop to DB
  rows → the offline prior. The SQLite `perf` rows (tier 3) survive such a bump: the DB is keyed by content, and
  the join that matches rows to candidates tolerates feature-set changes, so old rows stay usable.
- **Which compile flags each tier applies under**: all of them apply to the deployable regime, and that is the only
  regime anything is measured in. `H_opt` is read from the `-O<n>` in the compile flags; flags with no `-O<n>` at all
  — the default everywhere — count as 3, so an ordinary compile is always deployable. The identity a measurement is
  *stored* under agrees with that reading: `Context.structural_key` folds the flags **split** into an opt level plus
  the other flags (`context.split_opt_level`), never the raw string, so `""` and an explicit `-Xcicc -O3` are one key
  for the one regime they physically are. Keyed on the raw string they were two, and a row written under an explicit
  `-O3` pin was declared deployable by `H_opt` and then unreadable at a default deploy. A compile deliberately pinned
  to another opt level reads no measured tier at all: the reservoir gate rejects it on `H_opt`, and its own context
  key holds only whatever was measured under that same pin. That is the intended outcome — a non-deployable
  measurement is not evidence about a deploy.

**With no prior object at all, every tier is gone.** That happens when `load_prior` failed (a corrupt online
checkpoint, or the strict offline-artifact load raising; the loader is best-effort and swallows any failure), and on
`Pipeline.run`'s last-resort resolve that deliberately takes the rule's first option. The reservoir is carried by the
prior object, and the DB tier is only consulted on the path where a prior exists, so a corrupt checkpoint costs the
resolve its DB evidence too — every fork falls to the first emitted leaf. That leaf is not a chosen default and
nothing arranges the enumeration to make it a good one; it is simply what is left when there is nothing to rank
with. Env pins still apply (they never reach a decide).

**What is deliberately NOT in this hierarchy: the tune DB's `node` table** (Part 6). Node rows are never consulted at
deploy. They feed the `emmy eval` diagnostics (Part 8), and they are what the offline fitter would train on if it
trained on a frozen snapshot of that table — a planned path, not a current one (`emmy fit --data freeze:<path>` is not
yet supported; today `emmy fit` trains on goldens only).

**Whichever tier decides, the µs of the winning row is written onto the fork's trace entry** (`Decision.score`): a
measured µs when an evidence row decided, the model's predicted µs otherwise. That number is what the
structural cost estimate reads off the partition fork (Part 4), so the Σ compared there mixes measured and predicted
µs — measured wherever the tune benched that kernel, predicted only where nothing was.

**How to see which tier answered.** There is no flag that reports, per fork, which tier decided it; a live compile
does not print that. What exists today is: the loud warning for measured evidence that overlaps none of the offered
candidates, and the resolve trace (`Decision.score` carries the deciding row's µs). Answering "which tier decided
this fork, and did I expect that one?" means correlating those, not flipping one switch.

**Where the kernel gets cut is settled before any of this.** Ahead of the schedule pick, a separate decision splits —
or does not split — the recognized work into kernels (`lowering/tile/_cut.py`): `PLACE@<label> = cut` says "split at
the edge labelled `label`, so that sub-computation becomes its own kernel". A `PLACE` pin is authoritative — it cuts
(or keeps fused) with no prior involved. UNPINNED, placement is an ordinary **structural fork**: recognition offers
the fused form plus one cut fragment per legal seam, so `tune` discovers cuts and a compile
prices them through the same kernel-set costing as any structural option (Part 4) — measurements first, then
whichever prior is loaded. That costing exists because a `Graph` leaf carries no knob row the ordinary ranking
could score, not to defend the fused side: when some leaf cannot be priced the pricing decides nothing and every
leaf, cuts included, goes on to the ordinary ranking. A measured whole-route row is direct evidence for the complete
fused or cut candidate, so the reservoir and DB tiers consult it before estimating a route from its child kernels;
only a placement fork without exact aggregate evidence reaches that structural cost estimate. Exactly two computed
seams whose runnable normalized Loop bodies are alpha-equivalent collapse to one cut fragment: one workspace producer
replaces both contextual uses. Different external buffers or operations, and equivalence classes larger than two,
remain independent seams. A chosen cut's parent piece carries `PLACE@<seam>: cut` in its op knobs, so a measured cut
records and replays as the exact pin. A
**routing** golden entry (knobs that are only `PLACE@<label>` values; the loader rejects a mix of `PLACE` with
schedule knobs) is the recorded form of that pin — replayed like any other pinned row, never consulted by an
unpinned compile.

**Both file-backed inputs to that pick are built once per process.** The parsed online prior and the DB perf index are
memoized on the source file's `(path, mtime)` — the online file, and the DB file plus its `-wal` sidecar. A generative
serve boot compiles ~96 programs, and `structural_key` folds only cc + nvcc flags (never the op shape), so both inputs
are identical across every program; without the memo each compile re-parsed the 56 MB `online.json` and re-scanned the
whole perf table. The mtime key invalidates on any on-disk change, so a rewritten checkpoint or a fresh perf commit is
still picked up.

### The verified tier: strict structural identity, exact row decode

**The per-GPU golden files are the only *measured* data that ships with a clone.** A golden record is a named,
reviewed, pinned measurement: the input-pin regime it was measured under, the knob row that was selected inside
that regime, and the paired Emmy/reference timings. Its uses are the verified deploy tier (tier 1 above), exact
replay (`run --golden NAME`, `--ab`), training data for the offline prior (`emmy fit`), the `emmy eval` datasets,
and regression reference points.

The tier's join is exact by construction. The record side lowers the record's OWN persisted program through the
loop passes, selects its one target kernel, and recognizes it through the SAME core the live pass uses
(`_lift.recognized_tile`) — so record-side and fork-side identity cannot drift, and there is no classified shape
anywhere (the old `ShapeKey.kind` classifier and its offer-signal special cases are gone for good). The row decode
is exact equality of the schedule-family view after the one recording canonicalizer; a record that stops equaling
any enumerated row warns and decides nothing at deploy. The nightly `onboard-model` workflow strictly decodes and
replays the selected recipe's golden on its exact GPU. Per-commit tests do not load checked-in goldens because proving
a row enumerates its whole fork and costs record count times fork size. A target that lowers to more than one kernel
cannot carry a
row (a row decorates exactly one kernel) and must be re-seeded as a per-kernel Loop IR target.

**Whether goldens are training data differs between the two halves of the prior.** The **online** prior never trains
on them: a recorded golden row enters no reservoir and no checkpoint. (Benchmarks of a golden *shape* during a tune
are ordinary measurements and do train it; it is the recorded configs and their µs that never become labels.) The
**offline** prior IS fitted on them: `emmy fit` reconstructs the set of candidates each golden competed against and
trains the weights to rank the recorded config well inside that set.

### `FallbackPrior` and the calibration gate

**`FallbackPrior` only lets the online half answer once it is `trustworthy`** — fitted AND passing the **calibration
gate** (`Prior.trustworthy`).

After every fit, `maybe_refit` measures how well the model ranks the very rows it trained on: the median, across ops,
of the Spearman correlation between its predictions and its own reservoir labels (`_reservoir_calibration` — ops
grouped by their `S_*` signature, groups of fewer than 8 rows skipped, the verdict stored in the checkpoint). Below
`CALIBRATION_MIN` (0.5 — a genuinely trained model scores ~+0.85, while the collapse where the model and its rows no
longer share feature names scores ~0) the model is **quarantined**: it keeps training and checkpointing, but the
deploy ranking calls, PUCT, and the structural cost estimate (`greedy._priced_pick`) all fall back to the offline
half, and the verdict is logged. The reservoir evidence tier stays live under quarantine, because measured evidence
needs no trusted model.

A calibration that could not be measured at all (`None` — e.g. scipy is missing, or no op group is big enough) passes.
The gate is an alarm for measured failure, not a demand for proof of quality. It is known to be lenient in one case: a
small tune (the fit needs only `min_rows` = 50 dataset rows) whose op groups all stay under 8 rows ends up fitted with
calibration `None` — trustworthy, and therefore owning deploys, on very little data.

Why the gate exists: `fitted` alone once let a mis-calibrated model own deploys silently. Correlating predictions
against the training rows catches one failure specifically: the
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
  the reservoir evidence tier (Part 3) and hands the structural cost estimate to the offline half (there is no trusted online prior any
  more) until the machine re-tunes. A version bump therefore changes deploy behavior — the machine drops to
  goldens → DB `perf` rows → offline prior, with no warning at deploy time.
- **The autotune DB's `node` rows** (a `feat_ver` column, added without rewriting old rows):
  `data/group.group_measured` excludes rows from another version and counts how many it dropped. Rows written before
  the column existed default to version 1 (the retired feature names) and are excluded, which errs on the safe side.

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

`emmy run --golden PATH --strict` consumes a working golden rather than changing search. It visits every distinct
target sequentially in the current process, or one target selected with `--target NAME`. A valid directly measured
tune winner is an automatic exact pin; verified rows remain automatic pins as before. The ordinary strict run accepts
only captured whole-forward timing with direct eager correctness at `rtol=atol=1e-3`. Process isolation and repeated
observations come from independent command invocations, not a second orchestration layer inside `run`.

**Greedy flattens forks before ranking.** The lazy fork tree is an MCTS structure — it stages knob choices across
levels (`BR` → `BM/BN` → `FM/FN`) so MCTS pays one node per pop. Greedy must NOT walk it level-by-level: a branch
carries only a *partial* tile, and `features.knob_features` can't compute its area / occupancy until `FM/FN` are
pinned, so the prior would be blind at the `BM/BN` choice. Instead `greedy_decide` flattens each fork point to its
complete leaves (`fork.flatten_leaves` expands branches depth-first; only knob dicts — materialization stays deferred
to the chosen leaf) and picks the lowest `Prior.mean_scores` over the full `{H_*, S_*, complete-knob-row}` vector in
one batched `predict`, invariant to the tree's level order. With no online prior the `OfflinePrior` ranks (including a
positive `MMA_tier` warp preference — a fitted weight, not a hand-written rule); if `load_prior` returns nothing
entirely every fork falls to the first leaf in emission order, which is meaningless and may be slow.
Greedy benches nothing, so it can only *use* a prior, never train one.

**And it flattens each decision once.** A decision is a conclusion over evidence, so it is memoized GREEDY-SIDE (one
factory call — one compile attempt; never the shared `SessionCache`, which would hand MCTS cached picks): the memo
keys on the schedule `pool_key` (the dtype / hint / pin discriminators op identity excludes) plus the node's
blocklist content, so N same-shape kernels flatten-and-score once and the rest replay by descending the lazy tree's
level keys to the one matching leaf (`_find_decided_leaf` — the O(path) descent `build_fork_tree` was built for),
while a validate-retry with a blocked tile is a different key and re-decides.

**Every deploy pick breaks ties by candidate content, never enumeration order.** The model can score many
same-featurized siblings identically (the offline `D_*` geometry doesn't separate an `f2x4` from an `f4x2` fragment or
the `bk` variants — 8 exact ties at the gemma-4 m16 mlp_down/o_proj forks), and one measured row / one golden prefix
can match several offered candidates. Every tier therefore resolves its ties through `knob.canonical_row_key` (the
sorted tuning-knob rendering): the model argmin (`Prior.pick` and the greedy fallback), the reservoir and DB
measured-evidence argmins, and the golden realization pick. An order-broken tie is a per-boot coin flip — leaf order
can shift across processes — and shipped the 2026-07 RTX 5090 gemma-4 image with a bimodal boot-time cubin set
across boots.
Pinned by `tests/compiler/pipeline/search/policy/test_deploy_pick_determinism.py` at every evidence tier; rendered
bytes are independently pinned across fresh interpreters by `test_source_determinism.py`.

**Structural options are priced, never raw-scored.** A `Graph` leaf carries no knob row, so the per-op prior cannot
score it; `greedy_decide`'s `_priced_pick` asks the same evidence a different way instead. It prices EVERY leaf of a
structural fork — the cut fragments and the keep-fused side alike — by a nested `resolve` per kernel over a
`lowering/tile`-only pipeline, the price being the `score` of the slice-resolve's partition-fork `Decision`, memoized
per `Op.cache_key`, and takes the argmin. So an unpinned compile deploys the splits `tune` measured best. The nested
resolve carries the deploy's `db`, so each kernel's price follows the same evidence hierarchy as a knob pick
(the reservoir, then the tune DB's measured rows, model prediction only where unmeasured) — a pure
sum-of-predictions comparison would be exposed to the model's absolute-µs error, which doesn't cancel across
different kernel families, and that is a fitting requirement on the prior. When some leaf cannot be priced at all,
the pricing decides nothing and every leaf — cuts included — goes on to the ordinary leaf ranking. **No leaf is
withheld to keep a kernel set unchanged.** The one thing that does withdraw the splices is `price_structural=False`,
which is not about speed: it is how `GreedyStrategy` retires a structural pick whose fragment kernel failed to LOWER
(the splice minted fresh node ids, so it cannot be blocklisted at the fork site), and how a nested price probe
avoids re-splitting the slice it is pricing.

**Evidence joins are drift-tolerant.** `Prior.sig_groups` is one contract for both the reservoir -O3 tier and the DB
tier: a candidate's fork-time `S_*` base may carry scheduler stamps the persisted perf rows predate (#311's
`S_warp_eligible` is on no row recorded before it), and a strict-equality signature join would let one added feature
silently disable the whole evidence tier against every existing DB — the ninth-4090-sweep `mlp_gate_up` misdeploy (the
model's `g2k` pick beating the measured-faster fused config it was never allowed to see). The index spans three
the deploy's own context key — one regime, one key, one lane — and the pick is the plain argmin over the
matching measured rows.

**Retries are decide-wrappers over a deterministic re-resolve** — every other choice replays identically (cheap
non-chronological backtracking, no snapshots). A structural pick that leaves a fragment kernel un-lowered retires
structural picks wholesale and re-resolves the keep-fused branch before falling back to tile blocklisting.

**Greedy validity fallback.** The whole greedy retry orchestration is search policy, owned by
`policy/greedy.GreedyStrategy` — `Pipeline.run` is a thin entry point delegating to it. The prior ranks by
predicted latency, which can rank a tile that fails `validate(ctx)` (smem / thread budget) first — `tune`
benches-and-skips it, but greedy benches nothing. So when a deterministic compile leaves a node un-lowered, the
strategy blocklists that tile's `tile_identity` (its planner knobs) and re-resolves: `greedy_decide(blocked=…)`
drops the matching leaf and picks the next-best. This is bounded by `_MAX_GREEDY_RETRIES`.
When the retry budget exhausts with the node still un-lowered (an *online* prior can rank many over-budget tiles above
the first in-budget one), the strategy takes one last **emission-order resolve**
(`greedy_decide(blocked=…, prior=None)`): its point is that it ignores the prior whose extrapolation caused the
overflow, and the blocklist rides along so this last resolve can never re-pick a tile that already
failed `validate(ctx)`. It is a validity fallback, not a quality one — it makes no claim about the speed of what it
lands on, and the enumeration promises it no particular leaf. When that leaf overflows too, `_raise_on_unlowered`
fires the loud `LoweringError`.

### `Pipeline.tune_async` — the autotune sweep

`async Pipeline.tune_async(graph, *, search, backend=None, db=None)` is the (async-only) autotune
sweep. Pass a `TuningSearch(patience=, ucb_c=)`; the async generator yields one terminal `Candidate` per
fully-explored rollout and awaits `search.evaluate(token, cand, backend=, db=)` — terminal VALUATION is search
policy, not engine mechanics: the bench (or cache/stub short-circuit), the per-kernel `perf` / `lowering` /
inventory rows and the observe protocol live on the policy
(`search/policy/terminal_bench.py` + `TuningSearch.evaluate`). Per-run engine-event strategies are composed into
the pipeline itself (`Pipeline.with_strategies` — see the strategies section of the rule contract above).

- With `backend=None` the bench is stubbed to `latency_us=1.0` and nothing is persisted, so a backend-less sweep never
  overwrites tuned rows.
- A terminal that still holds an un-lowered kernel-bearing node (because its rewrite was dropped by validation) is
  marked `bench_fail` **before** any bench or cache lookup happens. The bench only sums `CudaOp`s, so without this
  guard the un-lowered kernel would count as zero and the µs of whatever cached kernels remained could stand in for
  the whole graph as an `ok` measurement.

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

The whole design is ONE class — `TwoLevelStrategy` — composing the engine's loop; the engine knows nothing
two-level-shaped. **Outer**: drive the graph-changing passes (`TwoLevelStrategy.OUTER_PASSES` = `frontend` +
`loop`, the strategy's OWN boundary config, never an engine parameter). The outer never ventures into Tile IR; a
**terminal** is the fused graph of finalized `LoopOp`s. Each terminal is one candidate grouping of ops into
kernels; its **reward** is `1 / Σ best-per-op time` from the strategy's separable scoring, backpropagated by the
reused `TuningSearch`. Tile-dialect structural forks (a `PLACE` cut, a cross-CTA split) stay INNER — they are part
of a kernel's independent measurement, and a slice whose kernel set changed benches as the Σ over the pieces it
minted.

Within one trajectory, structurally identical fork points all take the same side: `Run.drive` replays the first
decision, read off the trajectory's own graph (`_replay_structural_decision`), so the outer tree grows with the number
of *unique* kernels rather than as `2^n` in the number of such points. Fusion itself is still deterministic (no rule
offers a multi-option fusion fork), so a graph with no structural forks yields exactly one terminal and the whole
thing reduces to "tune each op once, sum, assemble". The global prior drives the outer PUCT too: each terminal emits
one combined Σ row per structural decision it took (features `{ctx, op knobs before the decision, the decision's knob
delta}`, label = the Σ of that side's per-kernel bests), so a re-tune on a warm machine descends into the kernel set
predicted to be cheaper first.

**Separable scoring** (`TwoLevelStrategy._evaluate_terminal`) tunes each finalized kernel **independently** in
its own single-node slice (`single_node_graph`, `slice.py`) with a plain `TuningSearch` over the lowering passes
only (`tile → kernel → cuda`), and returns the Σ once ALL Loop kernels are measured:

- The slice keeps the root kernel + its leaf-op closure and turns every other kernel-input into a synthetic `InputOp`.
  The root op is shared **by reference**, so its body — and thus `Op.cache_key` — is byte-for-byte the full-graph op's.
  It filters the graph's canonical topological order rather than iterating its set-backed ancestor closure, so slice
  inputs and persisted Loop programs stay byte-identical across fresh Python processes. Every retained `InputOp` is
  registered as a slice input even when a minting fragment did not list the boundary in its own `Graph.inputs`.
- Because the inner tree holds one op, MCTS explores only that op's forks with `patience` as the op's own budget —
  `Σ_k n_k` benches total, never the product.
- **Leaves are deduped by `Op.cache_key`**: 24 RMSNorm LoopOps across 24 layers collapse to one work unit, and the
  outer `total_us` accumulates `best * multiplicity` so the reward stays multiplicity-weighted. The progress
  denominator is the deduped count, so Qwen3-Embedding-0.6B's ~14 unique kernels show as 14/14, not 14/337.
- **Minted kernels are enrolled as first-class targets.** The strategy's private splice watcher (`_KernelInventory`, in `two_level.py`)
  rides every inner run and reports each genuinely NEW kernel a splice mints (a cut's fragments, a split's
  pieces), deduped by structural identity across the whole session, outer kernels included. Each reported kernel
  is enrolled in a wave after the current wave completes: tuned in its own slice cut from the minting fragment,
  its `perf` rows keyed under its own `cache_key`, its node rows under its own `op_sig` — and its own inner run
  may mint further pieces, which the same inventory catches for the next wave (waves terminate: cut/split trees
  strictly shrink and the seen-set dedups). Enrolled kernels are evidence, never reward terms — the parent
  slice's Σ already priced them, so they stay out of `per_op` / `total_us` and out of `searched_winner()`.

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
rest for free. (An earlier "skip already-tuned ops" gate suppressed exactly that re-exploration and was removed.)

### Per-kernel and working-target GPU parallelism (`--gpus N` / `--devices 0,1,2`)

Because the inner search tunes each unique kernel independently, the per-op loop fans out across GPUs. The whole tuner
is async-only: `TwoLevelStrategy.run` `await`s its separable scoring per outer terminal, which runs one coroutine per
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

For `tune --golden-file`, each realization specializes its parent symbolic program with its `bindings` and applies
its registered input `pins` before enumeration. Every realization with an explicit `knobs` mapping (including `{}`
for a forkless anchor) is compiled with scoped authoritative pins and measured through
the normal isolated benchmark and DB persistence path before MCTS continues in the same input pin regime. A
realized-vs-requested check marks an
unoffered proposal `pin_unmatched` instead of attributing the planner's fallback to the proposed row. Successful seed
rows feed the shared prior immediately, so the following MCTS can use their evidence. The measured pipeline captures
the exact finalized single Loop's stamped structural identity, even when the working file starts from stable Torch IR.
At the kernel-set-changing splice it also captures the consumed parent carrying the complete scheduler feature row and
exact structural route. The proposal's measured whole-slice latency persists under that route-specific parent cache
key and context. This captured perf row is deploy evidence: the measured DB index can select the route again after a
cold reload while the unpinned Loop key remains the two-level search's cost bookkeeping and the split kernels retain
their independent terminal measurements. Parent-linked node rows preserve the same proposal for diagnostics and
training; they do not drive deploy selection. The direct row is written only when the authoritative pins pass
realized-pin validation, search retains the exact structural route, one consumed parent realizes it, and the terminal
measurement succeeds. Ranking feedback is flushed to the working file as soon as proposal measurement finishes,
before MCTS. A multi-CudaOp result records realized knobs only when their union is conflict-free, or when search
retained that exact structural replay row; otherwise the ranking is explicitly ambiguous.

Each of those per-target persists rewrites the whole file, so its cost is the size of the inventory rather than of
the entry that changed, and a whole-model sweep pays it once per target. They are written **incrementally**: the
program and loop pools are nearly all of such a file by size and no persist mutates them, so an incremental write
keeps the text they were already serialized to and reserializes only the configurations. The document is still
validated in full and the bytes written are the ones a full dump writes; canonical and promotion dumps simply
reserialize everything. On the 279-target DeepSeek V4 Flash V100 inventory a persist is 2.5 s before and 0.24 s
after — the sweep's write path was most of its wall time.

`--max-candidates N` is a hard per-kernel budget. Each supplied proposal reserves one slot even if its measurement is
already cached, which makes hybrid-vs-MCTS comparisons charge LLM proposals consistently. MCTS receives the remaining
slots and counts only terminals that reached a live backend; cached replay observations update the tree without
spending the live-measurement budget. Ranking feedback is written under the entry's working-only `ranking` mapping,
and the final tune winner is annotated or appended as another proposal only when one directly searched observation
provides both its knob row and cost. When the fastest searched terminal changes the kernel set, the winner is its first
exact structural replay row: a `PLACE`-only routing row for a placement cut, or the complete pre-split schedule row for
a cross-CTA reduction. The pieces remain independent tuning targets; promotion never fabricates their heterogeneous
schedules into one row or falls back to a slower monolithic sibling. A cross-CTA parent becomes a tune winner only
when its ordinary schedule pins reproduce the decisions on every directly measured child kernel; a parent whose pins
name a different independently tuned child is left unpromoted. `PLACE`-only rows remain routing receipts and do not
claim the child schedules. A later greedy deploy replay can select different golden/DB evidence and is never paired
with that search reward. A search number never populates `emmy_us` / `cublas_us`; promotion still requires the
separate repeated, correct, deployable A/B gate.

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

### One measurement regime

The sweep benches in the **deployable regime** — the same nvcc flags `compile` / `run` use — so a terminal earns
exactly one measurement and a tuned latency is the deployed latency. Nothing is re-benched, nothing is translated
between lanes, and no store carries a per-regime column.

It was not always so: the sweep used to rank at `-Xcicc -O1` and re-bench near-best configs at `-O3`, which put a
proxy in charge of the search and the confirmation in charge of nothing but training rows. Two measurements retired
it. The proxy's error is *biased along tile area* — the axis being tuned — so it systematically priced wide tiles as
slow (paired over 1,818 configs: p90 regret 1.68×, and the `-O1` argmin was the `-O3` argmin on only 44.5% of pools).
And the compile time it was buying no longer exists: over 4,888 nvcc compiles, `-O3` compiled at a median 0.96× of
`-O1`. The rationale had been measured against WMMA codegen deleted days later.

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
values — which shipped real blow-ups before the switch. Any tree
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
`~/.cache/emmy/autotune.db`), and drives `TwoLevelStrategy(...).run(graph, ctx)`. The DB accumulates rows across runs; re-running
resumes from the cached state. On default verbosity (and a tty) a `TuneProgress` draws a live single-line bar
(completed/total tuned op leaves plus a `<kernel> <current us> (best <best us>) <knobs>` tail), threaded as an optional
`progress=` through `TwoLevelStrategy` (duck-typed, so the search package keeps no `commands/` dependency); `-v`
shows the per-`[tune]` INFO lines instead, `-q` is quiet.

The final greedy assembly (`result.assembled`, what `--output` writes and `--bench` measures) is the one greedy
compile that **holds the verified-golden tier out** (`golden.records_override([])`). That tier is a deploy
statement — "this config is known good on this card" — and a tune must assemble what it measured: the recorded
winner (`persist_tune_winner`) names the searched config, so letting a golden win the replay would report a benched
number for a config the tune did not choose. Every other tier still applies.

`--bench` benches the tuned winner end to end:
the full model against the real torch module and each kernel via its in-memory frontend
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
  they are the negative examples a search prior needs. An `ok` row is never downgraded by a later failure. A config
  whose **compile** ran past its budget is not one of these and is not recorded at all: nothing about its speed was
  measured, and a stored row would make it a permanent cache hit that is never re-benched (see the two bench budgets
  in `backend/cuda/ARCHITECTURE.md`).
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
snapshot written into a local directory: one YAML file per `(gpu, compute_cap)` (a `gpu_name`/`compute_cap` header plus
a `configs` list), beside a `manifest.json` holding the provenance header and the content digests.

- **Each row records DB `op_sig`, its measured `S_*` structural row, tunable knobs, and measurement metadata.** This
  is a regenerable measurement snapshot, not the stable golden format. Device `H_*` facts are derived faithfully from
  the GPU header and `opt` at load time.
- **Only current-vocabulary, deployable-regime leaves freeze**, as filtered by `freeze_reason`: `feat_ver` must have
  been current when the row was written, `H_opt` must be the deployable level, and the row must pass the two
  physical-plausibility checks the DB shares (`implausible_value_reason` / `impossible_kernel_reason`). `bench_fail`
  leaves are kept, as negative examples. The regime gate is what keeps a freeze a fair yardstick: a freeze is the
  corpus a reported prior number is computed over, so rows from a regime nothing deploys in would put half a card's
  pools in a lane no one runs. `group_measured` inherits the same filter, so an analysis over a live DB agrees with
  one over a freeze.
  Branch rows are never frozen and no tree structure is stored — the partly-decided rows are rebuilt at fit time under
  whatever fork structure is current then.
- **Freezing the same DB twice yields the same digests.** Every row serializes to one canonical JSON line, rows sort
  by that line, the per-file sha256 covers exactly those lines (content-level — immune to YAML style), the manifest's
  top sha256 folds the sorted per-file digests, and `created_at` enters none of them.
- A loaded row retains the DB's canonical `op_sig`, so a store's older cross-regime measurements of one operation group
  together without a second shape schema.
- **Loading is strict.** `load_freeze` hard-errors on a missing/foreign/corrupt manifest, a `freeze_ver` mismatch, a
  listed file missing, a per-file digest mismatch, or an un-instantiable row — never a silent fallback.
  `load_node_rows` sniffs a path (directory = freeze, sqlite file = DB, a v1 JSONL freeze is refused with a re-freeze
  pointer) and yields `NodeRow`s from either, which is what lets every nodes consumer
  (`eval prior --dataset nodes --db`, `Dataset.fold_node_rows`) take a freeze interchangeably with the live DB.
- Rows loaded from a freeze have no parent and `depth=0`. That costs the consumers nothing: they read benched
  leaves, and a freeze is leaf-only by construction.
- Handing a freeze to something that expects the perf table (the `--dataset db` paths) fails at `open_readonly`, with
  a message that spells out the difference between a freeze and the nodes DB.

**How node rows get written.** The same finished tree that feeds the reservoir is also walked once by
`_collect_node_records` and stored via `record_nodes`. Where the reservoir is an unkeyed random sample, this is the
keyed, deduplicated, parent-linked version of the same data. The walk fills in the columns that say how good each
label is (`SearchNode.visits`, the leaf's `bench_stats` / `bench_status` that `observe` stashed, and `is_leaf` from
`realized_knobs`). It also writes:

- **`bench_fail` leaves** — leaves only. Their value never contributes to any branch's minimum, so a branch's value
  comes from its working descendants instead, and a branch all of whose leaves failed is not recorded at all.

**Recording benches as node rows** (`search/bench_record.py`) is the node table's second writer. A `run --bench` that
benched rows with hand-forced knob values (golden or `--ab` rows) records each clean measurement — plus the greedy
pick, through its comparable `greedy (isolated)` re-bench — as leaf rows with no parent and `depth=0`. This is on by
default, behind the same quality bar the tuner applies to its own pinned benches; `--no-record-nodes` turns it off. It
is what stops measurements from a manual sweep evaporating.

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
split into held-out groups for cross-validation (`Dataset.fold_node_rows`, by `op_sig` / `gpu`): an op's tree and
its failed leaves all move to the same side together, and no parent edge ever crosses a fold
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

**`terminal_bench.bench_terminal_async`** is the only path that knows about all four parts (graph, DB, tree-through-`search.observe`,
backend). It short-circuits when every `CudaOp` in the graph already has a `perf` row for the current `(context_key,
backend)`. Otherwise it does one `await backend.benchmark_async(...)`, walks `Op.source` once to record op inventory +
lowering edges + the `perf` row per kernel, and returns the aggregate `PerfStats` for the search to score.
Tune terminals request one nominal warmup; the CUDA benchmark's existing clock-ramp floor extends that warmup until
it covers 10 ms of GPU time. A slow candidate therefore spends one iteration warming instead of exhausting the
run-stage budget on discarded repeats. Pinned and deployable comparisons retain their caller-selected warmup count.

## Part 7: Golden records and the A/B integrity gates

A golden record is a reviewed, per-GPU measurement of a frontend program target. It serves three purposes: exact
pinned replay (`run --golden NAME`, `--ab` — never consulted by an unpinned compile; see Part 3), training data for
the offline prior, and a regression reference. This Part covers the record format, its layout obligations, and the
checks that keep the A/B honest.

`golden.py` holds one generic `GoldenRecord` per realization. A structural config references a stable frontend Torch
IR program by its document-local list index. The preferred target selector is a non-empty, unique set of frontend
provenance origins.
When lowering produces a kernel without such a selector, the record points into the document's optional `loops` pool,
which stores that standalone post-fusion Loop IR slice. Current lowering derives the `S_*` histogram, `ShapeKey`, dtype
classification, dynamic status, and operation kind lazily; none is serialized. Trace inventories retain the complete
frontend program so provenance selectors re-lower in their original fusion context, while Loop IR fallbacks load
directly. There are no kernel-kind classes or snippet generators.

**Repository goldens are the entire compatibility boundary.** The embedded Torch IR has no independent version field.
The golden document has no format version either. When the YAML schema or its Torch IR encoding changes, regenerate
every recipe-local and model-agnostic repository golden in the same change. The loader does not carry migrations or
legacy decoders for working files outside the repository; keeping the checked-in corpus loadable is the compatibility
gate. Programs are
a plain list and structural configs refer to them by integer index; no program digest or persistent identifier is
stored. Loop IR
fallbacks are implementation-level rather than a compatibility promise and follow the same regenerate-the-corpus
invariant. Frontend graph nodes omit empty `attrs` / `inputs`, store tensors as `[name, dtype, shape]`, and encode static
dimensions as integers to keep the persistence surface small.

**One YAML format serves working candidates and reviewed goldens, but the trust boundaries differ.** Each structural
config contains only `model`, `program`, `target`, and a non-empty `realizations` array. A realization contains its
name, positive named dimension `bindings`, and explicit registered input `pins`, plus optional `knobs`, `measurements`,
and working-only `ranking`. `pins` defines the enumeration regime; `knobs` records the configuration selected and
measured inside that regime. Empty bindings retain the symbolic program; non-empty bindings specialize it before
lowering. A working realization may be inventory-only, a proposal, or verified. Repository promotion requires an
explicit knob mapping (possibly empty for a forkless anchor) and paired positive finite Emmy/reference timings on
every realization. Missing, one-sided, zero, NaN, infinite measurements, and ranking metadata are rejected before
they become trusted deploy evidence. `load_golden_file` and `dump_golden_file` validate this format without mutating
the parsed entries, and dumping refuses replacement unless its caller opts in explicitly.
An axis-scoped schedule family (`REDUCE@a1`, for example) may coexist with a non-OFF bare spelling of the same
family in one promoted entry — that IS the canonical stamped spelling (the bare key is the primary node's decision,
the scoped keys are the other tree sites' decisions, a `''` scoped value recording a site that declined). The one
rejected shape is a bare OFF beside scoped keys of the same family: a bare OFF pin fans out across eligible axes on
replay and contradicts the scoped decisions, so `stamp_schedule_families` drops it when stamping and promotion
rejects any that remain.

The preferred reference is the runnable Torch slice (`torch-eager`) or the applicable library kernel (`cublas`). A
Loop IR fallback has no frontend callable by construction; an origin slice can also have synthetic boundaries whose
post-fusion output geometry is not independently comparable to its Torch slice. Such a target may use a separately
compiled, repeated O3 `same-input-greedy` row as its positive reference only when the candidate and reference execute on
identical deterministic inputs, their outputs pass the normal accuracy policy, and the model report discloses that
this checks compiler-configuration parity rather than independent framework correctness. The original frontend
program remains embedded for provenance, while the selected standalone target is what both configurations execute.

The three historical RTX 4080 rows without measurements were dropped during migration; repository validation has no
provisional exception.

**A matmul golden's layout is part of what it measures.** The embedded Torch IR spells the serving Linear layout — B
given `(N, K)`, contracted as `x @ w.T`; the traced contraction carries `b_trans`. The warp tier stages it like any
canonical matmul (cp.async and TMA fill an N-MAJOR B slab — `tile_n × bk`, K stride-1 in gmem and smem alike —
drained by the plain no-`.trans` ldmatrix), so the same STAGE spellings realize on both layouts — but the measured µs
still differ per layout (different slab geometry and gmem walk), which is why a record meant for a served model's
linear fork must be TUNED on the `F.linear` snippet, and why a canonical entry (the harness/eval truth) and a
`trans_b` entry (the serving truth) both stay current. The same rule applies to fused computed-A programs: their
stored `torch.linear` edge is the served layout, and the smem compute fill stages every B fold channel via cp.async
on either layout.

**Provenance validation.** `emmy eval golden GOLDEN_YAML --serving-config PATH` derives model, revision, GPU,
canonical file, precision regimes, and reachable static/symbolic widths from one pinned env, requires that exact file
and live GPU, and validates that every structural target contains every expected realization. It is a corpus/schema
validation only — with no consulting deploy tier there is nothing to audit at deploy time; a recorded row's health is
its exact pinned replay (`--ab` / `run --golden NAME`, gated by the A/B integrity checks below).

**Live-GPU scoping.** `run` / `compile --golden NAME` prefer the **live** card's goldens
(`goldens_for_live_gpu`) — names repeat across per-GPU golden files with diverging shapes/dtypes, so a flat union can
select another card's spelling. They keep the union fallback on an uncovered card (the seed / transfer flow — the
pinned config re-benches live), and off-GPU the full union is returned (pure-logic tests). Tuning instead consumes an
explicit working file whose GPU header is checked against the selected tune device.

**The A/B carries three integrity gates:**

1. **Realized-vs-pinned knob check — a miss FAILS the row before it benches.** A structurally invalid pin silently
   falls back to the planner's own pick, so benching it would compare greedy against itself and report a fake 1.00×
   under the pin's name. The check runs right after the
   pinned compile. A pin that matches none of the knob values the compile actually produced marks the row
   `pin_unmatched` / `unreproducible pin … NOT benched` (a loud error log; the row is kept in the table and in
   `--json`, and no GPU time is spent), and the remaining rows still run. Matching is aware of knob families — a
   golden written as plain `TILE: …` matches the `TILE@dd` the compile produced — and values are compared through the
   registered knob's canonical `Knob.parse`, so alternative ways of writing the same value, like `FAST_EXP=1`, do not
   raise a false alarm. A pin satisfied by ANY kernel counts as honored, which is what makes split main+finalize pairs
   work, but it does mean that a pin dropped on its intended kernel goes undetected if a sibling kernel happens to
   match it. Two realizations are **structural** and cannot be read off a knob stamp at all, so the check skips them:
   a `PLACE` cut, and the `g<n>` cross-CTA stage of a `REDUCE` value. A split replaces the kernel it splits, and
   `knob.consume_kernel_row` strips the schedule row from the pieces it mints — no piece may carry the `g<n>` it came
   from — so the receipt is the piece's sliced reduce axis, not a stamp. Only that stage is exempt: the rest of the
   value (`coop` / `r<n>`) is decided by the piece on its own body and stays gated. The cost of the exemption is that a
   `g<n>` pin which genuinely never split cannot be told apart from one that did.
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
their flags and a `status` field: `ok` / `pin_unmatched` / `bench_fail` / `compile_timeout` (the config's compile ran
past its budget, so nothing about it was measured and the row is reported but never recorded); a failed greedy block carries
`status: bench_fail` and an `error`, with null timings), so a sweep's judgments can be traced to flagged fields
instead of to parsed terminal text. Each kernel row also carries **`record_knobs`**: the tuning knobs the compile
actually produced, with every schedule knob family (`knob.SCHEDULE_FAMILIES`: WORK / TILE / REDUCE / STAGE / RASTER)
written out explicitly, including the ones that are off (`knob.stamp_schedule_families`). That is the map to copy
verbatim into a golden YAML `knobs:` entry. An entry that omits a family leaves that family to whatever the planner
fills in at replay time, which shifts as the planner evolves — the recurring source of regressions that look real but
come from an unpinned `REDUCE`. Golden rows attach to the run's SHAPE rather than to a kernel node, so a pinned row
whose shape matches no greedy kernel — because greedy deployed a split partial+finalize pair — still prints and still
lands in the record.

## Part 8: Evaluating the prior (`emmy eval prior`)

`emmy eval prior` is how you find out whether the prior is any good and, when it isn't, where it goes wrong. It runs
over the goldens, the tune DB's `node` table, or a measurement freeze, and it reports BOTH halves of the composite
prior, each labelled — they fail for different reasons, so an unlabelled "prior" number destroys the diagnostic.

**Two datasets, two questions, one report.** `search/prior/report.py` assembles both into one serialisable schema
(`--json`), so comparing two models is a `diff`. `emmy fit` writes the same summaries into its `metrics.json`, through
the same `report.rank_metrics`. The report computes nothing itself:
`search/metrics.py` owns every metric's definition, and `Prior.score_rows(group)` — the pool-shaped scoring surface
both halves answer, projecting the packed matrix onto each model's own columns with its own absent-value fill — is
where a score comes from.

- A MEASURED pool (`--dataset nodes`: freeze or `node`-table rows, every candidate benched, grouped by
  `(gpu, kernel signature, H_opt)`) can answer what a wrong pick COST — Spearman over the pool, and regret at k=1
  (the deploy question: the pick ships, so its latency IS the cost) and k=10 (the tuning question: bench the top ten,
  keep the measured best). This is the half that tracks deployed speed.
- A GOLDEN pool (`--dataset golden`: an enumeration with the verified-optimum row marked) can only answer WHERE the
  known-good row landed, and is reported as a SCREEN. A rank is blind to the latency gap behind it, and the corpus
  aggregate is dominated by pools small enough to rank by accident, so golden summaries are stratified by pool size.

**Every summary publishes what it covered.** Summaries carry the axes they were keyed on as a dict — measured: `gpu` ×
`H_opt`; golden: `gpu` × `tier` × pool-size bucket; both plus `half` — along with how many pools keyed into them, how
many the model could not score at all, and — where a metric has a size minimum and so covers fewer pools than the
summary holds — that metric's own count. The minimums differ (regret needs two rows, Spearman five, regret@10 eleven),
so on the v3 freeze's 336 pools those counts are 297, 216 and 90. An aggregate that averaged the excluded pools in
would be reporting mostly arithmetic.

**A measured pool is keyed on the KERNEL, not on the site that offered it.** The key digests the row's own `S_*`
stamps — the same digest `Identity.op_sig` computes for an op, asked of the kernel that ran. Two kernels of one
structure on one card are ONE tuning problem whatever produced them, which is already how the deploy path joins
evidence: `Prior.evidence_pick` and `policy/greedy._db_measured_pick` both index on the `S_*` signature. It is safe
because the identity strategy stamps a kernel **at birth**, in recognition, before `020_schedule` offers the first
fork — so nothing a schedule fork decides can move an `S_*` value, and sibling schedules cannot be split apart.

Keying on the recorded `op_sig` column gets it wrong in both directions, and the RTX 5090 freeze shows both. It
**over-merges**, because `op_sig` digests the *pre-descent offer op*: nine pools paired a fused `rms_norm`→linear
megakernel with a row for just one kernel of the same op's unfused realization — a 5.9 µs norm kernel filed as a
rival of a 131 ms whole-op row, where the unfused pair actually costs 24–191 µs. And it **fragments**: 73 structures
were searched in two separate pools, the losing pool's best landing a median 1.46× behind the winning pool's (p90
3.89×, worst 14×) — the same kernel tuned twice because a placement cut minted one copy of it. Against `op_sig` the
kernel key gives 336 pools rather than 401, but more rows sitting beside a rival (3778 of 3817 against 3760) and a
median pool of 7 rather than 5; the pool count falls because merging is the point.

**`--dataset golden` also runs the deploy-faithful check the rank is only a screen for**: the greedy tile-pipeline
pick vs the recorded golden, per shape, with the deployable `-O3` latency of the prior's pick beside it
(`golden_deploy_perf`, read from the reservoir with no re-bench).

**`--dataset db` is rejected.** Those rows are fully-decided leaves with no op identity or compile regime to group a
comparison set by; the same DB read as `--dataset nodes` has both.

**A golden's rank counts ties against it** (via `search/metrics.dual_rank`). The
golden's rank counts every candidate scoring strictly better PLUS every candidate that ties with it and was emitted
earlier. A tie is counted as a loss because greedy's argmin, faced with equal scores, takes whichever came first.
Counting only strictly-better candidates would report rank 0 for every row inside a plateau of equal scores, which
once let a saturated prior score "top-1" on goldens that real cold deploys missed by 12–29×. Both counts come from
ONE computation (`search/metrics.dual_rank`): the pessimistic rank is
the one that gates, and the strictly-better **optimistic** rank is reported beside it in `emmy fit`'s metrics file.
The gap between them is the width of the tie plateau at the golden's score, and thus an early warning that the scores
are saturating.

**Golden evaluations build their features for the golden's own GPU.** They go through ONE golden group builder —
`emmy fit`'s `build_golden_groups`, which `eval prior --dataset golden` calls — so the eval and the fit see the
same corpus, the same sampling draw and the same rows. Each golden's compile context is rebuilt as
`Context.from_target(compute_cap, gpu_name=…)`, using the GPU recorded in the golden file along with its known SM
count and smem specs — never the host's. Building them for the host's context makes golden ranks machine-dependent,
because the occupancy features then describe tiles for a GPU that is not the one the row came from.

The eval builds its pools over the FULL featurization while a fit trains under its trainer's feature view. The view is
a property of the model being fitted, and the eval scores two model classes: the linear half reads only its own weight
names, so its ranks are identical either way, while the online half regresses on the `S_*` / `H_*` columns a narrow
view drops and would otherwise be asked about a kernel with no shape.

**The per-fork view is retired.** Until 2026-08 this part also documented three node-tree diagnostics: fork-sibling
regret (what following the prior's pick at each fork cost, bucketed by knob family), a golden-anchored descent (how
far a golden's path was covered by the explored tree), and per-feature blame / ablation Δ. They are gone, with
`Dataset.from_node_rows` and the `Prior.masking_exact` chain that existed only to caveat the ablation numbers.

Two reasons, and the second is why nothing replaced them in kind. They answered questions about a SEARCH TREE, and
the stores no longer hold one: every row in the current node table and in the v3 freeze is a parentless `depth=0`
bench leaf, so the fork metrics had nothing to group by `parent_key` and degraded to leaf-level numbers that
`eval prior`'s summaries now compute directly. And the ablation half rested on hiding one feature at a time, which
attributes an effect among correlated features with no unique answer — hiding any one of a redundant block of
geometry features costs the same Δ.

What a tree could tell you that a pool of benched leaves cannot is real and unaddressed: a search is a sequence of
partial decisions, most of whose subtrees are never explored, and a golden sitting in an unexplored one is silence
that reads as health. The writer for such a store is not hypothetical — `policy/mcts.py`'s `_collect_node_records`
walks the finished tree and emits parent-linked rows, and `working_golden.py` calls it into the same `node` table.
What changed is the collection path: the budgeted leaf sweep the `collect-node-data` skill drives records benched
leaves through `bench_record`, bypassing the tree writer. Per-fork evaluation returning is a question about which
collection flow runs, not about building a new store.

Both prior halves accept a candidate artifact for A/Bs: `--online-file` (legacy `--prior`) swaps the online
checkpoint (`EMMY_ONLINE_FILE`), and `--offline-file` (env `EMMY_OFFLINE_FILE`) swaps the offline weights artifact —
comparing two fits is running the same eval against two files and diffing the reports.

## Part 9: Tile lowering at the pipeline level

`lowering/tile/` lowers each fused `LoopOp` to a kernel-ready `TileOp` over the block-DAG Tile IR (`ir/tile/ir.py`):
`010_recognize` (lift `LoopOp` → `TileOp`, recognize the online-softmax streaming form, annotate each reduce
`Loop` with its `AxisRole` — the only loop annotation; the algebra is the body — and **atomize**: resolve the
algebra→hardware-atom binding structurally onto the node, so an unbindable atom never becomes one; `_classify.py`) →
`030_split_reduce` (cross-CTA split-K as a graph rewrite). It **never dispatches on a named
shape** — every decision is gated on the derived role of the stored fold (`PLANAR` / `CONTRACTION` / `TWISTED`; the
online-softmax reduce is the `TWISTED` fold, a twisted monoid is a monoid, selected structurally), not
on a matmul / pointwise / attention archetype. The full design lives in
[`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md).

The step BETWEEN those two — schedule enumeration: mapping the free axes onto the grid and forking the per-node
`TILE` / `REDUCE` / `STAGE` / `WORK` / `RASTER` families — is ONE recursive row enumerator over the term's own site
tree (the `020_schedule` rule). It covers every single-site term and the COMPUTED `a` edge (the fused norm→linear /
gate⊗up cone), whose nested statistic site is why it recurses. A term it cannot schedule
enumerates NO rows and stays unmapped rather than being guessed at — the guardrail
contract. See the leading section of [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md) for
the design.

**The enumerator ranks nothing.** It narrows by legality alone — this node's dtypes, extents, divisibilities and
smem budget — and hands back a SET whose emission order is an artefact of the recursion. No family leads with a
default, no shape's stored layout promotes a spelling, and no candidate is dropped for having measured slow
somewhere. Everything above (the deploy evidence hierarchy) is what turns that set into one kernel, and a compile
with no evidence taking a poor row from it is the accepted cost of keeping the scheduler free of judgment.

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

- **Per-knob:** `EMMY_<NAME>=<value>` (e.g. `EMMY_STAGE=d2/smem-async`). Read by the rule that owns the knob via
  `Knob.narrow`. The env-var key is built by `config.knob_var` and read via `config.knob_raw` / `config.int_env`.
- **Aggregate:** `EMMY_KNOBS="K1=V1,K2=V2,..."` (e.g.
  `EMMY_KNOBS="WORK=w2x2,TILE=mma_m16n8k16_f16_f32/f2x2/k2,STAGE=d2/smem-async"` — the worker widths ride `WORK`, so a
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
- A warp `TILE` pin on an **fp8** atom needs a static contraction K that the inner mma K-step (`atom_k·bk`) tiles —
  the byte-gather fragment loaders have no masked-K zero-fill family. Every other atom takes any K: the warp K-loop
  zero-fills the fragment halves past K on its final partial step, static and symbolic alike.
- A warp `TILE` atom must belong to the target's selected MMA family. On SM70, newer `m16n8k16` atoms and `cp.async`
  or TMA `STAGE` pins fail explicitly; the Volta m8n8k4 atom accepts global-memory-direct or `d<n>/smem` staging.
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
carriers only, with no f16/bf16 destination; both tiers — an mma partial's C fragment rides `RegStore.atomic`),
`g<n>k` = deferred f32 `__partial` workspace + a sibling combine kernel (any carrier; the only legal arm for a
low-precision output, a multi-component twisted carrier, and a multi-channel ⊗-combine). A direct atomic low-precision
destination would round once per partition and can cross the strict correctness boundary; the deferred arm combines
carrier state in f32 and rounds once. Pin
via `EMMY_REDUCE=g2k` (one flat knob — no per-axis `EMMY_REDUCE_<axis>`, no `EMMY_FINALIZE`). The split is realized by
`lowering/tile/030_split_reduce` as a graph rewrite whose pieces are **brand-new kernels** — unmapped, knob-free,
re-stamped, each scheduled at its own fork; a split node is priced as the Σ of its pieces' bests, and the split is
CONSUMED by the kernel that realizes it (the sliced axis is a `Window` of its parent, so nothing partitions it
twice). See [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md) for the invariant. The
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
4080`. A recorded winner belongs in the golden corpus as a versioned, exactly-replayable pinned row — never in code.

What could NOT be expressed was deleted rather than mis-filed, for one reason: **no recorded program in the corpus
describes a LayerNorm**. The DiT prologue is AdaLayerNorm-Zero, while every fused recorded program (`norm_linear` /
`mlp_geglu`) is RMSNorm by construction, and the reduce entries are `torch.sum` / `torch.nn.RMSNorm`. An entry filed
under those would hand every re-tune consumer the wrong kernel to rebuild. So the two LayerNorm→linear
contractions and the LayerNorm statistic reduce now deploy off the prior; adding a LayerNorm-cone kind is what would
let them be recorded.

**`STAGE`** (STR codec, the tile schedule → `lowering/kernel/010_materialize`) — the operand-staging codec
`d<depth>/sync|cp|tma[/p<reg_depth>]` on the typed `Stage` schedule struct (composes with both fragments
of the `TILE` knob): `d<depth>` the gmem→smem ring depth, `sync`/`cp.async`/TMA transport, `p<reg_depth>` the
smem→register double-buffer. `stage=None` (unset / unparseable) = gmem-direct. A `STAGE` value names only what the
schedule CHOOSES — rotation and refill discipline derive at materialization from the depth alone (which is why the
retired `ring` flag compiled byte-identically with and without it), and `smem` / `bk_elems` are resolver outputs,
never spelled. See `lowering/kernel/ARCHITECTURE.md`.

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
(A streamed); empty = the flat N-fastest row-major order. Changes
no per-CTA work, layout, or schedule — only the block-id decode (`ir/kernel` `Tile.render`, `Tile.raster_axes` the
`grid_tile` eligibility). Enumerated `('', 'gm8')` on 2-D contraction rows; wall-time effect is small and
shape-dependent (±2–4% measured), so golden evidence arbitrates per shape.

**`S_*`** (FLOAT, the `IdentityStrategy` — `passes/identity.py`) — the LoopOp's structural features (stmt/op histogram +
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
exactly the stored golden/DB spelling — bare `TILE`/`REDUCE`/`STAGE` on today's single-primary trees,
`REDUCE@<stat axis>` for the fused kernel's cone statistic
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
kind's tree and proves every spelling canonical. The tune DB / reservoir /
online prior are REGENERATED after a re-key, never migrated — no reader special-cases pre-phase-3 axis-suffixed
spellings, and `tuning_knob_items` renders keys AS STORED (the old `@<axis>`→bare display collapse is gone). What
remains is the live bare-golden contract: `family_value(knobs, family)` / `pin_key_matches`' bare↔explicit any-of
(a bare golden key matches an axis-keyed realization of the same family); it survives the step-7
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
| `frontend/decomposition/` | Rewrite frontend ops (`LinearOp`, `MatmulOp`, `SdpaOp`, layout ops, fused `rms_norm` / `layer_norm` / `softmax`) into tensor-IR primitives + layout-only `IndexMapOp`s, broadcast-explicit via `_broadcast.broadcast_to`. |
| `frontend/optimization/`  | `compose_indexmaps`: collapse chains of single-source / single-consumer `IndexMapOp` into one coord_map, so trivial layout kernels don't block fusion. |
| `loop/lifting/`           | `lift_*` rules wrap each surviving tensor primitive in a trivial one-op `LoopOp`; an additive scan writes its accumulator after every ordered scan-axis update. |
| `loop/prefusion/`         | `dissolve_narrowing` runs the SAME splice as `loop/fusion` (both call `_merge.merge_region`) over the subset of regions whose sink is no wider than the producer. A merge makes the sink the region's output, so those can only shrink what gets written; draining them to fixpoint first means every contraction has CLOSED before anything can splice into its open product and force the outer product to gmem. It refuses nothing — a widening merge is deferred, and `loop/fusion` offers it afterwards. A separate PASS because rule batches interleave WITHIN a pass but a pass is left only once quiescent. |
| `loop/fusion/`            | `split_shared_indexmap` dissolves a fan-out pure-indexmap into separate consumers when its branches do not reconverge; `merge_loop_ops` uses the same N-way splicer for adjacent pairs and closed reconvergent producer DAGs, preserving shared SSA definitions instead of treeifying them. A merged nested-reduce or multi-statistic cell stays fail-closed unless tile recognition proves one exact grouped placement inverse; that witness also replaces duplicated raw-loop work with the child-once + parent count for the boundedness gate, while fused and materialized forms remain priced siblings. `dedup_loads` drops identical `(input, index)` Loads; `fold_output_reshape` retargets a producer's `Write` through a graph-output memcpy-identity flatten (verified exactly over the finite domain; clean affine re-decomposition onto the output strides) — the copy kernel the splicer cannot take (a producer that carries a reduce, read through a div/mod index map). Folding scalar-constant broadcasts into consumers cuts Qwen3-Embedding-0.6B from 394 → 337 kernels. |
| `loop/canonicalize/`      | `fuse_split_free_axes` re-fuses an adjacent free-axis pair a fused reshape split (`p → f/Q, q → f%Q`, kept only when every access folds clean — composites collapse to the bare fused axis, a split store's row-major flatten folds back to an affine address), so split and unsplit spellings of one contraction converge to one canonical nest, one kernel identity, one shape key. Runs after fusion's fixpoint (the splicer composes through the very indices it re-spells) and before `loop/stamp`. See the passes `ARCHITECTURE.md` for why it is not a `normalize_body` pass. |
| `loop/recognize/`         | Empty (retired) — recognition is classification of the lifted Fold tree (`lowering/tile/_classify`), so the loop dialect carries no pattern recognizers. |
| `loop/stamp/`             | `stamp_loop_names` (`provenance.name_for`, e.g. `k_rms_norm_3f2a1b`) + `stamp_structural_features` (the `S_*` dict). Runs last in the loop dialect — after fusion and recognition — so every kernel is named / stamped against its final body. |
| `lowering/tile/`          | `LoopOp → TileOp` over the block-DAG Tile IR: `010_recognize` (structural — reads the algebra off the `LoopOp` body and emits an UNMAPPED `TileOp`) → the schedule step (REMOVED — see Part 9) → `030_split_reduce`. Dispatch is on the fold's derived role (`Fold.role` — `FREE` / `PLANAR` / `CONTRACTION` / `TWISTED`), never a named shape. |
| `lowering/kernel/`        | `010_materialize` is a `TileOp → KernelOp` tier dispatcher (scalar / `_reduce`). A tiled `CONTRACTION` arrives as a `Fold` already **built recognize-side** in the bilinear shape (`is_contraction` is the reading, not a kind) (`lowering/tile/_classify.bind_bilinear` — one flat node splitting the algebra params (axes / operands / acc / epilogue) from the schedule, which the fork places onto the grid), so materialize only synthesizes its bare grid-`Write` and **expands** it through the one atom-generic `_factor.factorize` over the shared tiling layer (in `_factor.py`) (the geometry is derived on the PLACED `TilePlan` slice, the algebra on the node; `_atom.reduce_codegen` emits the shared K-loop and a swappable `store` sink, dispatched off the atom). Then the Kernel-IR peepholes: `030_stamp_types` resolves dtypes, `050_vectorize_loads` / `080_vectorize_stores` / `095_interleave_loads` pack/reorder memory ops, `110_drop_redundant_syncs`. See [`passes/lowering/kernel/ARCHITECTURE.md`](passes/lowering/kernel/ARCHITECTURE.md). |
| `lowering/cuda/`          | `delegate_zero_init` (first) moves an atomic accumulator's per-launch zero-init off the runtime memset and into a dataflow-predecessor kernel as a `ZeroPrologue` stmt (CTA 0 writes zero words; stream order guarantees happen-before) — one CUDA-graph MEMSET node saved per site; the capture's first launch and symbolic-shaped accumulators keep their memset, and the slab planner starts the buffer's live interval at the delegating launch (`CudaOp.zero_prologues`). `lower_kernelop` then renders the `KernelOp` body to a `__global__` source string (`ir/kernel/render.py::render_kernelop`) and mutates the node's op to `CudaOp` in place. |

SiLU decomposition follows PyTorch opmath precision: f16 and bf16 inputs widen once to f32, the primitive
negative/exp/denominator/reciprocal chain and final product compute in f32, and the result converts once to the
declared output dtype. F32 and f64 inputs retain their dtype, so decomposition never demotes a wider input.

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
