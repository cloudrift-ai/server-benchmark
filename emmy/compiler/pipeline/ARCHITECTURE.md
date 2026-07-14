# Pipeline Architecture

The pipeline is the part of the compiler that turns a traced graph into finished CUDA kernels, one rewrite at a time.
This file explains three things:

1. **The rewrite engine** — how pattern-matching rules transform the graph, and how to write a rule.
2. **Forks and knobs** — how a rule says "there are several valid ways to do this", and how each choice is named so it
   can be measured, stored, and replayed.
3. **The autotune search** — how `emmy tune` measures those choices, how the online prior ranks them, and how a plain
   `emmy compile` / `emmy run` reuses that knowledge without benchmarking anything.

Two companion documents cover what this one doesn't: the rules themselves and their authoring invariants live in
[`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md), and what each IR dialect looks like is in `ir/ARCHITECTURE.md`.

## The big picture

The compiler lowers a graph by repeatedly applying small rewrite rules. Most rewrites are deterministic — there is one
right answer. But some (tile sizes, staging depth, split-K) have many valid answers whose relative speed depends on the
GPU and the shapes. Those rules return *all* their options as a **fork**, and something else decides which option wins:

- Under **`emmy tune`**, a Monte-Carlo tree search (MCTS) explores the fork tree, benchmarks real kernels, records every
  measurement in a SQLite database, and trains a machine-online model (the **prior**) that predicts kernel latency
  from a config's features.
- Under **`emmy compile` / `emmy run`** (a "greedy" compile), nothing is benchmarked. Each fork is resolved on the spot
  by asking the prior — measured evidence first, model prediction otherwise.

Terms used throughout:

| Term | Meaning |
|------|---------|
| **rule** | One pattern + rewrite function in a `NNN_<name>.py` file under a pass directory. |
| **pass** | An ordered directory of rules; the pass layout is frozen in a `Pipeline`. |
| **candidate** | One in-flight compilation state (a graph snapshot part-way through the pipeline). |
| **fork** | A rule returning multiple alternatives; the engine turns each option into a child candidate. |
| **knob** | A named tuning dimension (e.g. `TILE`, `STAGE`). Every fork option is identified by the knob values it pins. |
| **prior** | The ranking model — a fit-offline model when cold (the **offline prior**), a CatBoost model trained online from local measurements (the **online prior**) once data exists. |
| **terminal** | A fully-lowered candidate (every fork on its path resolved) that can be benchmarked. |
| **golden config** | A hand-recorded known-good config for a benchmark shape, used as ground truth and for A/B checks. |
| **`op_cache_key`** | A name-invariant digest of an op's body + knobs — the identity measurements are stored under. |

## Module map

| Module | What lives there |
|--------|------------------|
| `pipeline.py` | Engine core: `Pattern` / `Match` / `Rule` / `Pass` / `Pipeline` (the frozen pass layout) plus `Run` — the per-run state and engine loop. |
| `fork.py` | The `Fork` interface (`OptionFork`, `ThunkFork`) and the reusable `Level` + `build_fork_tree` lazy knob-cartesian tree builder. |
| `knob.py` | The `Knob` descriptor system and the `EMMY_<KNOB>` env namespace (borrowing `config.knob_var` / `config.knob_raw`; `format_tuning_knobs` renders the real tuning knobs for `tune` output). Holds NO concrete knob declarations. |
| `search/space.py` | **The single home of the search space.** Every `Knob` instance is declared here and nowhere else — the schedule codecs (`REDUCE` / `TILE` / `STAGE` / `WSPEC` / `RASTER`), the pin-only structural `PLACE`, the kernel-lowering policy knobs (`VECTORIZE_LOADS` / `INTERLEAVE_LOADS`), and the enumeration value grids (`scalar_tile_moves` & co). A rule that decides a knob imports it from here; `knob.registry()` still discovers knobs by walking loaded modules (`space.py` loads at pipeline startup via those rules). |
| `search/features.py` | The featurizers (`knob_features`, `tile_signature`, the `D_*` / `MMA_*` encodings) — kept beside `space.py` so the whole space (dimensions × values × encoding) is analyzable in one package. |
| `search/db.py` | `SearchDB`, the persistent SQLite store (see Part 6, "Search persistence"). |
| `search/policy/mcts.py` | The in-memory MCTS (`SearchTree`) colocated with its only reader, `TuningSearch`. |
| `search/policy/greedy.py` | `greedy_decide` — the no-tree fork resolver used by `compile` / `run`. |
| `search/two_level.py` | The two-level tuner: outer structural MCTS, inner per-op reward. |
| `search/prior/` | The ONE ranking path: a `Prior` ABC with the cold `OfflinePrior` and the `OnlinePrior` composed behind `FallbackPrior` (`load_prior`). `diagnostics.py` here backs the `eval` reachability / calibration reports. |
| `search/data/` | The harmonized read-view over the three data sources (golden configs / DB `perf` rows / prior reservoir): `Sample`, `Dataset`, and `ShapeKey` (the single golden↔measured join key). |
| `search/golden.py` | `GoldenConfig` and its subclasses (see Part 7, "Golden configs and the A/B integrity gates"). |
| `keys.py` | `op_cache_key` / `dialect_of` / `source_chain`. |
| `slice.py` | Isolates one finalized kernel into a standalone graph (used by the inner tune and structural pricing). |
| `dump.py`, `rule_diff.py` | The dump and `-vv` presentation layers (see the end of this file). |
| `passes/{frontend,loop,lowering}/` | The rules themselves — documented in [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md); a per-pass overview table is near the end of this file. |

## Part 1: The rewrite engine

### Patterns and matching

A `Pattern(name, op_type, constraints={})` matches one node by op type plus optional `node.op` field equality. A *list*
of patterns matches a chain: the seed node matches `pattern[0]`, its sole consumer matches `pattern[1]`, and so on.
Multi-node patterns only fire when each intermediate node has exactly one consumer.

`match_pattern(graph, pattern) → list[Match]` walks every topo-ordered seed. Overlapping matches are allowed — the
rewriter exits after the first successful rewrite per iteration, so overlap is just candidate enumeration.
`Match.nodes` maps each pattern entry's name to the matched `Node`; `Match.consumed` and `Match.output` are overridable
by the rewrite function to control which nodes the splicer removes and which node's edges get rewired.

### Writing a rule

Every file named `NNN_<name>.py` under a pass directory is a rule:

```python
PATTERN = [Pattern("root", SomeOp), ...]   # required
def rewrite(ctx: Context, graph: Graph, match: Match) -> Graph | Op | list[Graph | Op]:
    ...
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
the kernel's measured mma rows because `_warp_atoms` read placeholder dtypes off an all-f16 graph.)

### The three kinds of rewrite result

The return type discriminates the rewrite flavor:

- **Functional** — returns a `Graph` fragment, spliced in place of `match.output` (defaults to `match.root_node_id`).
  Fragment `InputOp` nodes reference existing graph nodes by id; non-Input nodes get fresh ids.
- **In-place** — returns an `Op`. The engine assigns it to `root.op` directly, preserving the node id, inputs list,
  output Tensor, and hints. The lowering rules use this because `KernelOp.arg_order` / `CudaOp.arg_order` embed the
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

### Lazy hierarchical forks

`Fork` (`fork.py`) is an interface with three members:

- `knobs` — the knob delta this fork pins. This is the variant's identity: the perf DB and the prior key on it, and it
  is readable **without expanding** the fork.
- `is_leaf` — whether this is a concrete option or an inner branch.
- `expand()` — materializes the next level of options.

The search loop pops a Fork-pending `LazyCandidate`, invokes `expand()` to materialize the children, pushes them back,
and continues — only the subtrees the search actually walks into get materialized. `OptionFork` is a concrete `Op` /
`Graph` leaf; `ThunkFork` is a generic flat fork (`expand_fn(knobs)` is a function of the fork's own delta, so siblings
share one function).

Multi-level knob-cartesian forks reuse **`build_fork_tree`**: a rule supplies per-level `Level`s plus a `materialize=`
callable and gets back a lazy root `_Branch` whose `expand()` builds children on demand in grouping order. The
algorithm (group params by per-level knob keys, collapse single-key levels, skip empty-key levels, defer leaf
materialization to `expand()`) lives once in `fork.py`; one-shot flat forks stay inline as `ThunkFork`s.

### The knob-stamp invariant

Every emitted variant carries an explicit value for **every declared knob** — no realized leaf has an absent knob.

Each `Knob` declares an `off` value (its "unused / declined" sentinel), and the pipeline fills any of a pass's declared
knobs the variant left unspecified at the **pass boundary** (`Cursor.advance` → `_off_fill_pass`, via
`knob.apply_off_defaults`). This covers a pass that acted, declined, was skipped, or returned no variants alike.
Scoping the fill to the just-finished pass avoids prematurely stamping a later pass's knob (which would trip that
pass's idempotency guard).

Why it matters: the online prior NaN-fills absent feature columns. With explicit OFF values, NaN means *only*
"not-yet-decided" (a partial fork prefix during descent), distinct from "decided: unused" (an OFF value on a complete
leaf). A knob with no `off` (the `_UNSET` default — a knob its owning pass always stamps itself) is never auto-filled.
Tier discrimination is value-based throughout (`knob.is_warp` / `knob.mma_atom`). Verified by
`tests/compiler/passes/test_knob_stamp_invariant.py`.

### Invalid options: validation filtering and raised rewrites

A rewrite that *returns* an op failing `Op.validate(ctx)` (e.g. a `KernelOp` whose smem exceeds
`ctx.max_dynamic_smem`) is filtered by `Candidate.try_rewrite`. That is correct as fork pruning — sibling branches
carry other tile shapes — but fatal in a single-path greedy compile, where it leaves the node un-lowered. So:

- `Pipeline.run` installs a `rejections` sink on the `Run`, recording each drop as `(node, pass, reason)`. After the
  terminal settles, `_raise_on_unlowered` raises a loud `LoweringError` naming any still-un-lowered node, instead of
  leaking a cryptic `non-CudaOp` `TypeError` to the backend.
- The sink is absent under `tune`, so the fork-pruning path stays silent there.

A rewrite that *raises* mid-lowering (a deterministic pass hitting an un-representable shape) is the same dead end
expressed as an exception: greedy `resolve` lets it propagate; under `tune`, `Run.drive` catches it per-candidate,
drops that subtree, and bumps `Run._dropped_candidates`. Without this, one search-only un-lowerable fork aborted the
whole tune.

## Part 3: The prior — how choices are ranked

### One ranking path

Ranking is always the policy's job over a single `Prior`. Forks carry NO score, and nothing materializes or scores a
`TileOp` just to rank it — the per-variant `lazy_score` / `score_tile_geometry` formulas, the `Fork.score` /
`Search.score_of` plumbing, the DB-best `_best_fork` replay, and the `_priority_*` enumeration sorts are all gone. The
`Prior` featurizes the row knobs directly (`features.knob_features`).

The two halves of the one path:

- **`OfflinePrior`** (cold) — a fit-offline linear *score* over the engineered `D_*` geometry / occupancy features,
  not emission order. The complete scoring function (both weight sets + the scalar params, `feat_ver`-stamped, with a
  `provenance` block) lives in the repo-checked artifact `search/prior/offline_weights.json`, written by
  `scripts/golden_knob_heuristics.py`; `EMMY_OFFLINE_FILE` (or `emmy eval … --offline-file`) swaps in a candidate
  fit for an A/B. Loading is strict: a missing or `feat_ver`-mismatched artifact is a hard error (refit it), never a
  silent fallback — a retired weight key inside a current-version artifact is merely a dead term. A separate
  `weights_dynamic` set ranks symbolic-axis masked-tile kernels, selected on the stamped `S_ext_n_symbolic_axis`. Two
  hard-coded interaction gates ride outside the linear weights: the atomic-free split-K term, and the tensor-core
  preference pair `D_scalar_on_warp_eligible` / `D_splitk_roundtrip` driven by the scheduler's per-kernel
  `S_warp_eligible` row stamp — which stops a warp-eligible f16 contraction deploying a scalar split tile.
- **`OnlinePrior`** (online) — trained from tune measurements (Part 5), composed behind `FallbackPrior`.

A subtlety about features: the `H_*` regime features (GPU / nvcc level) are constant across a pool's siblings, so no
additive weight on them can change a within-pool ranking. Architecture differentiation instead rides *per-candidate*
features that only exist where the hardware offers them: the TMA-conditioned geometry interactions (`D_tma_*`,
mirroring the tile geometry on TMA-staged rows) let one weight set price Hopper/Blackwell tiles separately from
cp.async-era ones, and the warp-grid features (`D_w_grid_*`) separate same-tile different-grid siblings that were
previously byte-identical (the 2026-07-09 4090/5090 golden-sweep TILE findings).

Who consumes the ranking: `TuningSearch` (`tune`) ranks the PUCT frontier; `greedy_decide` (`compile` / `run`, via
`Run.resolve`) picks via `Prior.pick` — measured -O3 reservoir evidence first (`evidence_pick`: the candidate
prefix-consistent with the fastest `H_opt=3` row of the same op), the `mean_score` argmin otherwise.

### `FallbackPrior` and the calibration gate

`FallbackPrior` hands surfaces to the online half only once it is **trustworthy** — fitted AND passing the
**calibration gate** (`Prior.trustworthy`). After every fit, `maybe_refit` measures the median per-op in-sample
Spearman correlation between the model's predictions and its own reservoir labels (`_reservoir_calibration`, persisted
in the checkpoint). Below `CALIBRATION_MIN` the model is quarantined: it keeps training and checkpointing, but deploys,
PUCT, and structural pricing (`greedy._pick_structural`) stay offline, and the verdict is logged.

Why: `fitted` alone let a mis-calibrated model own deploys silently (the 2026-07 RTX 5090 sweeps). In-sample Spearman
is a lenient tripwire that specifically catches the model-and-rows-don't-share-a-feature-vocabulary collapse (constant
predictions, worse-than-random ranking).

When trusted: `mean_score` / `mean_scores` / `pick` (deploy + eval) are pure-online + evidence. But `score` — the MCTS
*selection* signal — tilts the online µs by the offline prior's dimensionless ranking multiplier
(`online · offline**W`, `W = config.offline_tilt`, neutral 1.0), so PUCT still explores regions the cold heuristic
prices well but the data-poor online model buries.

### Featurizer vocabulary versioning

`features.FEATURIZER_VERSION` stamps every persisted training artifact:

- **The prior checkpoint** (`to_json`): `from_json` discards a checkpoint from another version WHOLE — model and
  reservoir rows alike — since rows spelled in a retired knob vocabulary featurize to garbage and a refit on them
  collapses to constant predictions.
- **The autotune DB's `node` rows** (a `feat_ver` column, additively migrated): `diagnostics.node_report` excludes
  rows from another version with a printed count. Pre-stamp rows default to version 1 (the retired pre-rebuild
  vocabulary) and quarantine conservatively.

Bump the constant on any incompatible knob-spelling or feature-encoding change; artifacts from the old vocabulary then
age out instead of poisoning the model.

## Part 4: The drivers — two ways to run the pipeline

`Pipeline.build(passes)` wraps a pass list; the result exposes the compile entry points, each driving one of the `Run`
engine loops (`drive` for exploration, `resolve` for deterministic resolution).

### `Run` — the per-run state

`Run` bundles everything scoped to one compilation: `pipeline` + `ctx` + `search` + `db` + `backend` + `dump` +
`rejections`. `Pipeline` stays a frozen, shareable pass layout while every run-scoped sink lives on the Run, reached
through the candidate (`cand.run.dump`, `cand.ctx`).

### `Run.drive` — the exploration loop (`tune`)

`Run.drive(graph) -> Iterator[(token, Candidate)]` seeds the root candidate, then per iteration pops a
`LazyCandidate`, resolves it, runs one rule batch (`Run._step`, shared with `resolve`), and pushes successors under the
pop's token. Selection is `TuningSearch`'s job (PUCT over the online prior); the perf DB still *records* every bench
as training data. Each fork push is classified by effect at the spawn site (where the raw option list is concrete): any
`Graph`-splicing option (a kernel-set change) marks the push `structural=True`; an `Op` rebind is op-variant (`False`).

### `Run.resolve` — deterministic resolution

`Run.resolve(graph, decide) -> (Graph, list[Decision])` is the deterministic counterpart. Both entry points share one
rule-batch body (`Run._step`), but `resolve` is a fold, not a search: ONE live graph mutated in place (no sibling
snapshots, no per-fork copies — the terminal IS the seeded graph). At each undecided fork a `decide` callback gets a
`ForkPoint` (the `Match`, the raw options as the rule emitted them, the pre-decision op, `ctx`) and returns the option
to apply.

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
`MMA_tier` warp preference); option-0 (first leaf, emission order) only if `load_prior` returns nothing entirely.
Greedy benches nothing, so it can only *use* a prior, never train one.

**Structural options are priced, never raw-scored.** With the trained prior loaded, `greedy_decide`'s
`_pick_structural` prices each side of a structural fork: a nested `resolve` per kernel over a `lowering/tile`-only
pipeline, the price being the `score` of the slice-resolve's partition-fork `Decision`, memoized per `op_cache_key`.
The cheaper kernel set wins, so an unpinned compile deploys the splits `tune` measured best. The nested resolve carries
the deploy's `db`, so each kernel's price follows the same evidence hierarchy as a knob pick (reservoir -O3, then the
-O1 ranking lane, model prediction only where unmeasured) — a pure sum-of-predictions comparison would be exposed to
the model's absolute-µs error, which doesn't cancel across different kernel families. Cold, or when a side is
unpriceable, the structural leaf is filtered — a cold compile never changes kernel sets.

**Evidence joins are drift-tolerant.** `Prior.sig_groups` is one contract for both the reservoir -O3 tier and the DB
tier: a candidate's fork-time `S_*` base may carry scheduler stamps the persisted perf rows predate (#311's
`S_warp_eligible` is on no row recorded before it), and a strict-equality signature join would let one added feature
silently disable the whole evidence lane against every existing DB — the ninth-4090-sweep `mlp_gate_up` misdeploy (the
model's `g2k` pick beating the measured-faster fused config it was never allowed to see). The index spans three
context twins (the deploy's own flags, the `-Xcicc -O1` ranking lane, the `-Xcicc -O3` lane where the tune's deployable
re-benches land), and the pick is two-tier: a deployable-lane row decides outright; `-O1` rows decide only when no
candidate has deployable evidence, because an -O1 median is a ranking signal with -O3 inversions and must not override
a well-trained model on its own.

**Retries are decide-wrappers over a deterministic re-resolve** — every other choice replays identically (cheap
non-chronological backtracking, no snapshots). A structural pick that leaves a fragment kernel un-lowered retires
structural picks wholesale and re-resolves the keep-fused branch before falling back to tile blocklisting.

**Greedy validity fallback.** The prior ranks by predicted latency, which can rank a tile that fails `validate(ctx)`
(smem / thread budget) first — `tune` benches-and-skips it, but greedy benches nothing. So when a deterministic compile
leaves a node un-lowered, `Pipeline.run` blocklists that tile's `tile_identity` (its planner knobs) and re-resolves:
`greedy_decide(blocked=…)` drops the matching leaf and picks the next-best. This is bounded by `_MAX_GREEDY_RETRIES`.
When the retry budget exhausts with the node still un-lowered (an *online* prior can rank many over-budget tiles above
the first in-budget one), `Pipeline.run` takes one last **option-0 (emission-order) resolve**
(`greedy_decide(prior=None)`): the planner emits a budget-safe tile first, so it lowers whenever any in-budget tile
exists. Only when even option-0 overflows does `_raise_on_unlowered` fire the loud `LoweringError`.

### `Pipeline.tune_async` — the autotune sweep

`async Pipeline.tune_async(graph, *, search, backend=None, db=None)` is the (async-only) autotune sweep. Pass a
`TuningSearch(patience=, ucb_c=)`; the async generator yields one terminal `Candidate` per fully-explored rollout, and
`tune_async` benches each via `await _bench_terminal_async` (writes per-kernel `perf` / `lowering` / inventory rows,
returns the aggregate `PerfStats`), then calls `search.observe(stats, status)`.

- With `backend=None` the bench is stubbed to `latency_us=1.0` and nothing is persisted, so a backend-less sweep never
  overwrites tuned rows.
- A terminal still carrying an un-lowered kernel-bearing node (a validation-filtered rewrite) is a `bench_fail`
  decided **before** any bench or cache lookup. The bench sums `CudaOp`s only, so without this guard the un-lowered
  kernel priced at zero and a cached residual kernel's µs could stand in for the whole graph as an `ok` measurement
  (issue #327).

## Part 5: The tuning workflow (`emmy tune`)

The autotune loop selects one tile-lowering variant per CudaOp by repeatedly running the lowering pipeline with
different knob choices at each fork point, benching the produced kernels, and steering subsequent rollouts toward the
lowest measured latency.

### Two-level search: outer structural MCTS + inner per-op tuning

`emmy tune` does **not** run one MCTS over the whole graph. The pipeline applies rules sequentially, so two kinds of
fork — **op-variant** forks (tile / pad / stage choices for one kernel) and **structural** forks (which kernels exist:
fusion grouping, the demoted-matmul split) — would nest and cross-product under one global patience, starving deep
ops. The two kinds have opposite structure, so `two_level.py` splits them on the fork's *effect* (the spawn-site
`Op`-rebind vs `Graph`-splice classification):

**Outer search** (`run_two_level_tune`) drives the graph-changing passes — `frontend` + `loop` plus the pre-partition
head of `lowering/tile` (`010_split_demoted`'s keep-vs-split offer followed by the non-forking post-split re-fusion
aliases). A **terminal** is the state where the cursor reaches `partition_loops` with every structural fork resolved.
Each terminal is a candidate fused graph; its **reward** is `1 / Σ best-per-op time` from the inner search,
backpropagated by the reused `TuningSearch`. Structurally identical offer sites within one trajectory take the same
side: `Run.drive` replays the first decision read off the trajectory's own graph (`_replay_structural_decision`), so
the outer tree stays linear in *unique* kernels instead of `2^sites`. Fusion itself is still deterministic (no rule
emits a multi-option fusion fork), so a graph with no structural offers yields one terminal and this reduces to "tune
each op once, sum, assemble". The global prior also drives the outer PUCT: each terminal emits one composed Σ row per
structural decision it realized (features `{ctx, pre-decision op knobs, decision delta}`, label = the Σ of that side's
per-kernel bests), so a warm re-tune descends the predicted-cheaper kernel set first.

**Inner search** (`_inner_reward_async`) tunes each finalized kernel **independently** in its own single-node slice
(`single_node_graph`, `slice.py`) with a plain `TuningSearch` over the lowering passes only (`tile → kernel → cuda`):

- The slice keeps the root kernel + its leaf-op closure and turns every other kernel-input into a synthetic `InputOp`.
  The root op is shared **by reference**, so its body — and thus `op_cache_key` — is byte-for-byte the full-graph op's.
- One fold-aware exception: a flash fold offer site's slice CARRIES the score producer its fusion consumes
  (`_flash.fused_producer_ids` → `single_node_graph(absorb=…)`), and the absorbed producer loses its own slice. A
  synthetic-input boundary would make `try_flash` unfusable in-slice, silently degrading every tune trajectory to the
  cut (benching fragment kernels greedy deploy never picks) and leaving the fused flash fork unreachable under tune.
- Because the inner tree holds one op, MCTS explores only that op's forks with `patience` as the op's own budget —
  `Σ_k n_k` benches total, never the product.
- **Leaves are deduped by `op_cache_key`**: 24 RMSNorm LoopOps across 24 layers collapse to one work unit, and the
  outer `total_us` accumulates `best * multiplicity` so the reward stays multiplicity-weighted. The progress
  denominator is the deduped count, so Qwen3-Embedding-0.6B's ~14 unique kernels show as 14/14, not 14/337.

**Separability + the structural handoff.** Op-variant forks are separable: every multi-option fork is an in-place `Op`
rebind that leaves the graph unchanged, so whole-graph time is `Σ_k t_k`. Results key structurally (`op_cache_key` =
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
rest for free. (The old `op_effort` "skip already-tuned" gate is gone — it suppressed exactly that prior-driven
re-exploration.)

### Per-kernel GPU parallelism (`--gpus N` / `--devices 0,1,2`)

Because the inner search tunes each unique kernel independently, the per-op loop fans out across GPUs. The whole tuner
is async-only: `run_two_level_tune` `await`s `_inner_reward_async` per outer terminal, which runs one coroutine per
unique kernel over an `asyncio.Queue` of `len(pool)` device-pinned `CudaBackend`s — each pops a backend, drives its
op's whole inner search via `Pipeline.tune_async`, then returns the backend. So `len(pool)` benches run at once, one
per GPU.

- **True single-thread asyncio**: every Python statement (lowering, DB writes, prior `add_rows` / `maybe_refit` /
  `checkpoint`) runs on the one event-loop thread and yields only at the bench `await`, so the shared `db` / `prior`
  need no locks.
- Each op seeds its `TuningSearch` by `seed + op_idx` and the reward is a commutative `Σ`, so the per-op DB bests and
  `total_us` are byte-identical regardless of slot count; only the online prior checkpoint varies run-to-run (rows arrive
  in completion order).
- The **default single-GPU** path is a one-slot pool whose coroutines acquire the lone worker in `op_idx` order —
  strictly sequential, identical to the old serial loop.
- A backend pins its async worker to a physical GPU via the child spawn env (`CUDA_VISIBLE_DEVICES`, plus a per-device
  `EMMY_GPU_LOCK` suffix), never mutating the parent `os.environ`.
- Parallelism is bounded by the unique-kernel count; devices must be homogeneous.

### Search dynamics (the MCTS itself)

Each level reuses the **same** SP-MCTS (`policy/mcts.py`) — outer over structural forks, inner over one op's forks —
with max-Q normalized UCB1:

- **Selection** is PUCT (`_select`): `score(c) = Q(c) + c · P(c) · √(N_parent+1)/(1+N_c)`, where
  `Q = best_reward/global_best` (0 if unvisited), `reward = 1/median_us`, and `P` is the prior's predicted reward on
  the same scale (the prior predicts latency `û(c)`, which `_select` converts to `1/û` and normalizes by the same
  `global_best` — no softmax; `c = --ucb-c`). The prior is the SOLE signal — the greedy tiebreak, the static
  `TileOp.score` tiebreak, and the `+∞`-unvisited UCB rule are all gone. A confidently-slow sibling (large `û` → small
  `P`) is deprioritized instead of force-benched.
- **Expansion** is implicit (one rule batch per pop, one child per alternative).
- **Simulation** is the actual `await backend.benchmark_async(...)` on the terminal.
- **Backprop** walks the popped candidate's parent chain updating `visits` and `best_reward`.
- **Patience** counts terminals since the last new global best; when it exceeds `--patience N` (default 50), the level
  exits.

### -O3 deployable samples

The sweep compiles at `-Xcicc -O1` — fast, but a *ranking* signal: it ties configs that differ at -O3 (e.g. a `REDUCE`
ILP fold or a warp tile's `WSPEC`). So whenever a bench lands **within `EMMY_O3_TOL` (default 15%, `config.o3_tol`) of
the best -O1 so far** — a band wider than a strict new best, so near-tied contenders all qualify — the engine
re-benches it at `-Xcicc -O3` (`_rebench_o3`), and `observe_o3` records an extra row with the same realized knobs
tagged `H_opt=3` (the deployable regime) — into the reservoir AND the `node` table, where it lands as a parentless
leaf row under its own -O3 `context_key`. Each config is re-benched at most once. The `H_*` feature lets the -O1
(broad) and -O3 (near-best) rows coexist; `compile` / `run` run at -O3 (`H_opt=3`), so greedy ranks by the deployable
rows and reaches the true optimum. The `nvcc_flags` override rides the bench request to the worker, so only winners
pay the -O3 recompile and the cubin cache keys on the flags.

All tune/bench timings are **CUDA-graph-captured** by default (pure GPU time); each `perf` row records its mode in the
`captured` column, and on write a captured measurement supersedes a wall-semantics one for the same key (never the
reverse), so old rows upgrade in place.

### Training the online prior

There is ONE global `OnlinePrior` across every kernel, GPU, and nvcc setting — not per-op, not partitioned by
regime. Op structure (`S_*`) and the host/hardware regime (`H_*` — GPU compute capability + nvcc opt level, from
`Context.features`) are **features in every row**, not a cache key.

**Value-of-position labels.** Real benches exist only at leaves, but the prior ranks partial-knob siblings at every
fork level, so the label for any node is the best (min) median latency in µs over its benched descendants
(`1/best_reward`) — the prior regresses on **latency**, and the `1/û` conversion lives in the MCTS `_select` loop, not
the stored data. `TuningSearch._collect_rows` walks the live tree and emits `(knobs, label)` for every node with a
benched descendant:

- A directly-benched **leaf** uses its `realized_knobs` — the FULL config read off the resolved graph's op in
  `observe`, so knobs stamped at deterministic non-forking lowering steps (`FK` / `BK` / `SPLITK` / `STAGE`) are
  captured, not just the fork knobs.
- A **branch** falls back to `_node_knobs` (its partial `fork.knobs` prefix under the op's `S_*` / `H_*` base),
  carrying the value-of-position label.

**Why CatBoost** (chosen by `scripts/prior_bakeoff.py`): the model's greedy pick must not run off to a degenerate
corner. A linear model (the former `BayesianRidgePrior`) is monotone in every knob, so its optimum is always a corner
of the candidate box — the `BR=1` blow-up (4µs → 232µs / invalid kernels). Any **bounded** tree ensemble is
off-manifold-safe (an un-benched extreme inherits the nearest leaf's value), and among them CatBoost also generalizes
to an *untuned* op near-perfectly (leave-one-op-out pick ratio ~1.0 vs xgb/lgbm 1.18, rf 1.31). So one global CatBoost
prior is good enough on a new op that it is **not refit within an op's own search** — it is a fixed model per run.

**Dataset and checkpoint.** The dataset is bounded + batched (`base.Prior`): each tuned op's value-of-position rows
stream into a reservoir-sampled dataset capped at `MAX_ROWS` (100k, Algorithm R across runs), and the model refits
(`maybe_refit`) on a dataset-size-tiered cadence (`REFIT_SCHEDULE` — frequently while data-poor, coarsening as it
grows), then checkpoints. End-of-run does a `maybe_refit(force=True)` so even a small tune ends with a fitted model.
The checkpoint is a JSON file (`config.online_path()`, `~/.cache/emmy/online.json`) holding the CatBoost `cbm` blob
(base64) + the dataset; `tune` writes it, `compile` / `run` read it.

### Driving the loop

`emmy tune <model_or_ir | --code EXPR>` probes a `Context`, opens the tuning database (`EMMY_TUNE_DB` or
`~/.cache/emmy/autotune.db`), and calls `run_two_level_tune(...)`. The DB accumulates rows across runs; re-running
resumes from the cached state. On default verbosity (and a tty) a `TuneProgress` draws a live single-line bar
(completed/total tuned op leaves plus a `<kernel> <current us> (best <best us>) <knobs>` tail), threaded as an optional
`progress=` through `run_two_level_tune` (duck-typed, so the search package keeps no `commands/` dependency); `-v`
shows the per-`[tune]` INFO lines instead, `-q` is quiet. `--bench` re-benches the tuned winner at -O3 (deployable,
not the -O1 ranking pass): the full model against the real torch module and each kernel via its `.torch.json`
provenance reproducer, vs eager / `torch.compile` / Emmy.

## Part 6: Persistence and keys

### The keying map: two identities

Everything the search stores or replays is keyed by one of TWO identities — when adding a cache or table, pick one;
don't invent a third:

- **Variant identity = `(context, knobs)`** — anything *predictive or replayable*. The `S_*` structural features
  (`loop/stamp` stamps a stmt/op histogram + loop extents + operand dtypes) make the merged knob dict a COMPLETE
  identity, so a prior is a pure function of it. The online prior is exactly `score(features(ctx, knobs))`: the
  structural facts are already in the knob dict, so `features.knob_features` turns it straight into the model feature
  vector (the `S_*` knobs pass through; tuning knobs encode by type, `MMA` expands to atom props).
- **Measurement identity = `(ctx.structural_key, op_cache_key)`** — ground truth about *materialized leaves*: `perf`
  rows (the per-variant replay cache), op inventory (`loop_op` / `tile_op` / `kernel_op` / `cuda_op`), and two-level
  dedup. The structural `child_key` on `lowering` rows is measurement linkage (it joins the inventory), NOT a replay
  key.

### Search persistence: on-disk inventory vs in-memory MCTS

**`SearchDB`** (`db.py`) is a SQLite store partitioned into:

- **Four op-inventory tables** — one row per op encountered along any lowering chain, keyed by `op_cache_key`.
- **A `lowering` edge table** — one row per rewrite hop carrying the knob delta plus a best-median upsert
  (`best_per_op_time` walks the chain to resolve a pre-final op's measured cost; loop→loop source hops are skipped as
  structural/decision hops).
- **A backend-partitioned `perf` table** — full stats + `backend` + `status` + `knobs` + `captured`.
- **A `node` table** — one row per **search-tree node** (every partial branch + leaf of a per-kernel search), keyed by
  `digest(context_key, gpu, op_sig, tunable-knob set)`, carrying the full feature dict the prior sees, a
  value-of-position latency with a **per-kind upsert** (branch rows keep-min — a coverage bound a faster descendant
  genuinely tightens; leaf rows take the **newest measurement**, since a leaf is a re-measurement of one config and
  min-of-K noisy medians drifts to the noise floor), a `parent_key` pointer, a `gpu` column, and depth bookkeeping
  (written by `record_nodes`).

Each `node` row also carries **label-quality columns** (additive migration; old rows degrade to unknowns):

- `visits` — benched-descendant count, the label's confidence weight; SUM-accumulated across writes and merges (unlike
  the write-batch `n_updates`).
- `is_leaf` — a real measurement vs a min over explored descendants.
- `variance` / `n_samples` — the leaf's own bench stats.
- `status` — `ok` / `bench_fail`. Failed leaves ARE recorded, with the watchdog sentinel as `value_us` — the negative
  examples a search prior needs; an `ok` row is never downgraded by a later fail.
- `run_id` / `measured_at` — the tune session (one id per CLI invocation) and time that produced the CURRENT
  `value_us`, replaced only when the value is.
- `feat_ver` — the `features.FEATURIZER_VERSION` the row's feature dict is spelled in (see Part 3); pre-stamp rows
  default to the retired version 1 and are excluded from prior evaluation.

The `gpu` identity (`Context.hardware_id`, the PCIe product name) is folded into the node key so a cross-hardware
dataset never collides: `context_key` (cc + opt) can't separate same-die SKUs (H100 vs H200 share cc + SM count), so
without `gpu` their rows would merge and the upsert would silently drop one card's data (the `H_total_mem` VRAM
feature is what then lets the prior model the difference). `node` and `perf` are content-keyed
(parent-tree-independent) and survive a `_SCHEMA_VERSION` bump; only the topology-keyed `lowering` table is dropped on
mismatch.

**Cross-hardware merge.** `SearchDB.merge_nodes(src_path)` is the accumulation entry point: it reads another autotune
DB's `node` rows read-only and re-upserts them through the same per-kind path (direction-independent — a stale leaf
snapshot never resurrects; `visits` SUMs on a shared key), so a card's node data measured on a rented GPU (no local
CUDA) folds into one canonical DB without cross-card collision. Driven by `scripts/merge_node_db.py` / the
`collect-node-data` skill, whose sweeps run ε-greedy (`remote_node_tune.py` launches the remote tune with
`--explore-eps 0.25` by default) so the collected labels and fork coverage de-correlate from the incumbent prior.

**How node rows get written.** Alongside the reservoir feed, the same finished tree is walked once by
`_collect_node_records` and persisted via `record_nodes` — the keyed, deduplicated, parent-linked counterpart to the
unkeyed/sampled reservoir. The walk fills the label-quality columns (`SearchNode.visits`, the leaf's `bench_stats` /
`bench_status` stashed by `observe`, `is_leaf` from `realized_knobs`). It also emits:

- **`bench_fail` leaves** — leaf-only; never value anchors (no monotone participation, children anchor past them, an
  all-failed branch stays unrecorded).
- **-O3-regime rows** for a leaf the tune re-benched at the deployable `-Xcicc -O3` (`observe_o3` stashes
  `SearchNode.o3_us`): keyed under the tune context with `O3_NVCC_FLAGS` substituted (never colliding with the -O1
  twin), features stamped `H_opt=3.0` (the reservoir convention), parentless — a regime re-measurement, not part of
  the tree, never in a fork sibling group.

Within one batch, a deterministic no-knob-delta step can give a child its parent's exact knob set (same `node_key`);
duplicates collapse to one row (leaf's stats, max — not sum — of their visits) so `record_nodes`'s SUM accumulation
never double-counts a run. The store is **group-holdout-fold ready** (`Dataset.fold_node_rows`, by `op_sig` / `gpu`):
an op's -O1 tree, -O3 rows, and fail leaves move to one side atomically and parent edges never cross a fold (`run_id`
is provenance of the surviving deduped value, deliberately NOT a fold axis).

**`SearchTree`** (`policy/mcts.py`) is pure-Python in-memory MCTS state, colocated with `TuningSearch` because MCTS is
the only policy that reads it. Each tree node wraps a `LazyCandidate` and carries `visits`, `best_reward` (max reward
over the subtree's measured leaves), and a `live` counter that filters out drained subtrees. Lineage is
TOKEN-THREADED, not call-order-dependent: `pop()` returns `(token, candidate)` (the token IS the `SearchNode`), the
engine pushes children with `parent=token` and observes the terminal with the same token, so the tree stays correct
however the engine interleaves pops / pushes / observes. It is rebuilt fresh each process; cached `perf` rows ensure
no re-bench on warm starts. Greedy compiles build no tree (they don't go through a `Search`).

**`_bench_terminal_async`** is the only path that knows about all four parts (graph, DB, tree-through-`search.observe`,
backend). It short-circuits when every `CudaOp` in the graph already has a `perf` row for the current `(context_key,
backend)`. Otherwise it does one `await backend.benchmark_async(...)`, walks `Op.source` once to record op inventory +
lowering edges + the `perf` row per kernel, and returns the aggregate `PerfStats` for the search to score.

## Part 7: Golden configs and the A/B integrity gates

`golden.py` holds `GoldenConfig` and its matmul / attention / softmax / reduce / rms_norm / pointwise subclasses — the
`OfflinePrior`'s ground truth. Every kind carries `shape_key()` / `snippet()` / `dtype`, so `tune --dataset golden`
and the `run --bench --golden` A/B cover the reduce / pointwise entries too, not just matmul.

**Live-GPU scoping.** `tune --dataset golden` (and `--golden NAME` resolution) scopes to the **live** card's goldens
(`goldens_for_live_gpu`) — names repeat across per-GPU golden files with diverging shapes/dtypes, so a flat union
would tune another card's config under the live card's name. For `tune` the scoping is strict: a live card with no
recorded goldens exits with an error instead of inheriting the union fallback — golden tuning targets the live card's
own recordings only, so an uncovered card is fixed by recording goldens for it, not papered over
(`live_recorded_goldens` is the no-fallback probe that tells an uncovered card from an off-GPU run). `run` / `compile`
`--golden` keep the union fallback on an uncovered card (the seed / transfer flow — the pinned config re-benches
live), and off-GPU the full union is returned (pure-logic tests).

**The A/B carries three integrity gates:**

1. **Realized-vs-pinned knob check.** A structurally invalid pin silently falls back to the planner's own pick, so the
   row would compare greedy to itself and report a fake 1.00×; the flag marks the row `unreproducible pin` with what
   actually ran. Matching is family-aware — a bare golden spelling like `PLACE: fuse` matches its axis-stamped
   `PLACE@fold` realization — and values compare through the registered knob's canonical `Knob.parse`, so alias
   spellings like `FAST_EXP=1` don't false-flag. A pin satisfied by ANY kernel counts as honored, which tolerates
   split main+finalize pairs but means a pin dropped on its target kernel that a sibling coincidentally matches passes
   undetected.
2. **Arithmetic-intensity floor.** A row whose shape-implied FLOP/s exceeds the live card's recorded `GpuSpec` peak is
   flagged as a wrong bench, not a fast kernel.
3. **Wrong-answer check.** Each pinned config executes once on the greedy run's inputs and its outputs are compared —
   catching the silently-wrong `g2a` skipped-finalize class.

Plus `--json PATH` — a machine-readable record of the whole comparison (backends / greedy kernels / pinned rows with
their flags), so sweep judgments trace to flagged fields instead of parsed terminal text. Golden rows attach to the
run's SHAPE, not a kernel node: a pinned row whose shape matches no greedy kernel (greedy deployed a split
partial+finalize pair) still prints and lands in the record.

## Part 8: Evaluating the prior (`emmy eval`)

**Fork-sibling regret** (`eval online --dataset nodes`, via `iter_nodes` → `diagnostics.node_report`): **per card**, it
groups nodes by `parent_key` and prices what following the prior's per-fork pick costs —
`value_us(predicted-best child) / value_us(true best)` (1.00x = the prior steers into the best-reachable subtree;
predicted-score ties price pessimistically, since greedy breaks ties by emission order) — the search-faithful
evaluation no leaf-only view can give. Each fork buckets by the knob FAMILY its children decide (`TILE` / `REDUCE` /
`STAGE` / …, from the child-vs-parent knob delta) — the stable notion of tree level (raw `depth` is rule-step distance
and renumbers as passes change) — rendered as a per-kernel × per-family regret table with a per-family aggregate line.
`node_report` drops `bench_fail` rows up front (their `value_us` is the watchdog sentinel, not a measurement) and
splits a card's block per `H_opt` regime so -O1 and -O3 latencies never pool in leaf reachability. The per-card
grouping matters for a cross-hardware dataset: same-die SKUs (H100/H200) share an `S_*` op signature but not their
latencies, so mixing them would corrupt both metrics — the `gpu` key keeps their rows distinct.

The regret/reachability block renders once per prior **half** (offline vs online, labeled) — the composite would
answer with whichever half is active, and the two halves' regrets point at different fixes (cold-start weights vs
training data), so an unlabeled "prior" number destroys the diagnostic.

Both halves accept a candidate artifact for A/Bs: `--online-file` (legacy `--prior`) swaps the online checkpoint
(`EMMY_ONLINE_FILE`), and
`--offline-file` (on `eval offline` / `eval online`; env `EMMY_OFFLINE_FILE`) swaps the offline weights artifact —
comparing two fits is running the same eval against two files and diffing the reports.

**Per-feature attribution** (`eval online --dataset nodes --blame / --ablate`): both views consume one shared per-fork
record (`diagnostics.fork_records`: siblings, featurized rows, scores, the pessimistic pick, the measured best), so
the three views agree on pick semantics by construction. They score through the `Prior` **features seam** —
`mean_score_features` / `mean_scores_features` take an already-featurized row (contract: identical to `mean_score` on
the raw knob dict), which is what lets the diagnostics mask individual `D_*` features that have no knob-level
spelling.

- **Blame** diffs `Prior.explain_features` (a signed per-term quality decomposition; exact for the linear offline
  prior, its hardcoded interactions included as `gate:*` pseudo-terms — the terms sum to the scored quality,
  unit-tested) between the pick and the measured-best sibling, regret-weighted per fork family. A missed fork no term
  separates is **BLIND** — a featurizer gap, not a weight problem.
- **Ablation Δ** re-picks every fork with one feature masked (each model's own absent semantics: `0.0` term for the
  linear prior = exact removal, `NaN` routing for CatBoost — flagged out-of-distribution until a dropout-trained model
  exists) and reports the per-family median-regret change with the feature's fork support.

Both are **diagnostic only, never gate metrics**: attribution among correlated features is non-unique (masking any one
of a redundant geometry block costs the same Δ). Unlike the per-card regret/reachability tables, attribution POOLS
cards and regimes — regret is a within-fork ratio, so it compares safely.

## Part 9: Tile lowering at the pipeline level

`lowering/tile/` lowers each fused `LoopOp` to a kernel-ready `TileOp` over the block-DAG Tile IR (`ir/tile/ir.py`):
`010_recognize` (lift `LoopOp` → `TileOp`, recognize flash / softmax carriers, annotate each reduce `Loop` with its
`AxisRole` + `Carrier`, then schedule inline via the `_schedule` helper — no separate `020` pass: map free axes to the
grid, pick the reduce partition + output `TILE` fragment, and **atomize** — resolve the algebra→hardware-atom binding
structurally onto the schedule as each warp / cooperative option is built, so an unbindable atom is rejected at fork
construction; `_atomize.py`) → `030_split` (cross-CTA split-K as a graph rewrite). It **never dispatches on a named
shape** — every decision is gated on the reduce axes' carrier algebra read off the body (`MAP` / `SEMIRING` /
`MONOID`; flash attention is the `MONOID` algebra on the streaming schedule, a twisted monoid is a monoid, selected
structurally), not on a matmul / pointwise / attention archetype. The full design lives in
[`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md). One interaction reaches up to the pipeline level:

**The demoted-matmul split (`CUT`) as an outer structural fork.** `split/010_split_demoted` may un-fuse a demoted
matmul (a multiply operand reading a computed / K-folded cone that keeps the matmul off the warp tier) into an
`xn`-producer + clean-gemm-consumer `Graph` fragment — a kernel-set change. The two-level tuner owns the offer as an
outer structural fork: keep-vs-split branches the outer tree, each side's kernels are tuned in first-class per-op
slices, and the Σ-per-op terminal rewards compare the kernel sets; greedy deploys the split only via the *trained*
prior's structural pricing, never cold. The `op.knobs` `CUT` stamp is the considered-vs-declined idiom (`keys.py`):
simultaneously the rule's idempotence guard, the online prior's training signal (absent = never offered →
NaN-filled; `"0"` / `"1"` = the decision), and the `op_cache_key` separation that keeps each decision state distinct
in the search tree. The stamp is deterministic per offer site, so identical kernels across graphs stamp identically
and keep sharing perf rows.

## Tunable knobs

A **`Knob`** (`knob.py`) is the canonical schema for one tuning dimension: name, type (`INT` / `BOOL` / `BINMASK` /
`STR`), candidate `hints` (advisory — the rule still validates structural fit), and a help string. Rules stamp values
into `TileOp.knobs` dicts; the autotuner reads those back as the per-hop knob delta in the `lowering` table. Every
knob is declared **in `search/space.py`** — the single home for the whole tunable surface — and imported by the rule
(`lowering/tile/_schedule`, the scheduling helper inside `010_recognize`) that resolves it. The registry
(`knob.registry()`) auto-collects every `Knob` instance in every loaded module — no manual registration. `knob.py`
also owns the `EMMY_<KNOB>` env namespace (decode per `Knob` type; `config.py` remains the sole owner of
`os.environ`).

### Pinning knobs from the environment

Two equivalent forms:

- **Per-knob:** `EMMY_<NAME>=<value>` (e.g. `EMMY_STAGE=d2/cp`). Read by the rule that owns the knob via
  `Knob.narrow`. The env-var key is built by `config.knob_var` and read via `config.knob_raw` / `config.int_env`.
- **Aggregate:** `EMMY_KNOBS="K1=V1,K2=V2,..."` (e.g.
  `EMMY_KNOBS="TILE=a:mma_m16n8k16_f16_f32/w2x2/f2x2/k2,STAGE=d2/cp"`). Parsed once at `knob.py` import via
  `apply_knobs_env()`, which splats each entry into the corresponding `EMMY_<K>` var
  (`config.set_knob(..., overwrite=False)`). An explicit per-knob var wins over the aggregate.

Pinning replaces tuner choice (the rule emits exactly that variant instead of forking) and is **authoritative** — an
env value outside the knob's hint tuple is honored, not silently dropped (`Knob.narrow` returns `(pinned,)` regardless
of hint membership). Downstream structural gates (divisibility, threads-per-CTA budget, TMA eligibility) still apply,
so a structurally invalid pin yields an empty enumeration and the per-call-site fallback takes over. This lets a tile
shape the planner wouldn't reach on its own be explored manually. The replay paths (`run --bench --golden` / `--ab`)
can't accept that silent fallback — it would substitute the planner's own pick and turn the A/B into greedy-vs-greedy
— so they verify realized-vs-pinned knobs on every pinned row and flag mismatches `unreproducible pin` (see the
integrity gates in Part 7).

A few pins are rejected outright (a clear `ValueError`) rather than silently degraded — they would otherwise lower to
a wrong or un-launchable kernel:

- A codec width must be `≥ 1` (a degenerate `b0` / `f0` / `n0` no longer parses to a silently-dropped level).
- A warp `TILE` pin needs its **static** contraction K to be a multiple of the inner mma K-step (`atom_k·bk`), since
  the warp K-loop has no static-K tail masking (a **symbolic** K is fine — it reaches the masked zero-filled tier).
- A scalar `TILE` parallel block (`par_n·par_m`) is capped at the 1024-thread/CTA hardware limit.
- A `BOOL` knob rejects an unrecognized value instead of coercing a typo (`ture`) to `False`.

### Registered knobs

All declared in `search/space.py`; see [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md) for the per-rule mechanics.
The "owning rule" for the schedule codecs is the `_schedule` helper inside `lowering/tile/010_recognize` — there is no
separate `020_schedule` pass.

**`TILE`** (STR codec, `010_recognize` / `_schedule`) — the **unified output-fragment** codec. A contraction's output
tile is *either* the **scalar** register sub-tile `n<N>[x<M>]/f<fn>[x<fm>]` (parallel thread-tile `n`/`m`, register
sub-tile `f`) *or* the **warp** tensor-core mma tile `a:<atom>/w<WM>x<WN>/f<FM>x<FN>/k<bk>` (atom + warps + register
sub-tile + K-chunk), never both. The value self-discriminates: an `a:<atom>` token selects the warp form (a
tensor-core-atom `TilePlan`; `schedule.is_warp_codec`), otherwise the scalar `TilePlan`. Empty = per-cell. A pin may
spell the scalar tier explicitly as `a:scalar` (alias `a:none`) — the one `a:`-prefixed value that is *not* warp;
pin-input vocabulary only, normalized back to the canonical scalar spelling (`""` / `n../f..`) so it never rides a
stored knob (mirrors `PLACE`'s `auto`).

**`REDUCE`** (STR codec, `010_recognize` / `_schedule`) — the reduce-axis partition codec `g<n>[a|k]/b<n>/r<n>`: `g`
cross-CTA split-K (+ finalize letter), `b` cooperative-thread fold, `r` ILP register fold. Empty = serial (the
per-thread remainder is derived, never spelled). The cross-CTA split is the `g<n>` field (GRID stage), and the
**finalize** is that field's trailing letter — `g<n>a` = in-place `atomicAdd` (one kernel, additive carriers only),
`g<n>k` = deferred `__partial` workspace + a sibling combine kernel (any carrier; the only legal arm for the twisted
flash `(m, l, O)` split-KV). Pin via `EMMY_REDUCE=g2k` (one flat knob — no per-axis `EMMY_REDUCE_<axis>`, no
`EMMY_FINALIZE`). The split is consumed by `lowering/tile/030_split` as a graph rewrite (partial + finalize); the
letter round-trips through `ReducePlan.parse`/`spell` and reads back as `ReducePlan.finalize`. The atomic finalize
applies the kernel's projection epilogue **per partition** before the `atomicAdd`, so it is only correct when that
projection *distributes* over the add (`Σ φ(xₛ) = φ(Σ xₛ)`): a constant scale like `mean`'s `×1/N` distributes and
rides the atomic; a non-distributive epilogue (`l2`'s `sqrt`, a fused bias/activation) is refused
(`NotImplementedError` → pin `g<n>k`, which projects once after the combine). The check is
`030_split._projection_distributes`.

**`STAGE`** (STR codec, `010_recognize` / `_schedule` → `lowering/kernel/010_materialize`) — the operand-staging codec
`d<depth>/sync|cp|tma[/ring][/p<reg_depth>]` on the typed `Stage` schedule struct (composes with both fragments of the
`TILE` knob): `d<depth>` the gmem→smem ring depth, `sync`/`cp.async`/TMA transport, `p<reg_depth>` the smem→register
double-buffer. `stage=None` (unset / unparseable) = gmem-direct. Also rides the warp-flash TWISTED stream
(`STAGE@<kv>` — the K/V slabs of one streaming block; cp.async only, `reg_depth` clamps to 1). See
`lowering/kernel/ARCHITECTURE.md`.

**`WSPEC`** (STR codec, `010_recognize` / `_schedule` → `lowering/kernel/010_materialize`) — the warp-specialization
codec `p<np>`: a producer warp band split over the fixed pipeline (bare/root-global; the fourth schedule-fork level).
Legal on a warp `TILE` over a resolved **TMA** `STAGE` within the thread budget
(`block_threads + 32·aux ≤ 1024`, `32·aux ≤ block_threads`); anything else — including the reserved producer `q`
param — degrades to uniform. Empty = uniform SIMT. Materialized as the staged K-loop's producer/compute band split
(`_stage._wspec_kloop`).

**`RASTER`** (STR codec, `010_recognize` / `_schedule` → `lowering/kernel/010_materialize`) — the CTA launch-order
codec (bare/root-global like `WSPEC`; the fifth schedule-fork level): `gm<G>` iterates `G` M block-tiles fastest per
launch stripe so consecutive CTAs share the streamed B slab (L2 reuse — the flat order streams B from DRAM once per
M-row: `A + C + B×2` measured on the 4090's `mlp_gate_up`, 503.6 vs cuBLAS's 365.8 MB); `gn<G>` is the transpose
(A streamed); empty = the flat N-fastest row-major order (option-0, byte-identical to historical codegen). Changes
no per-CTA work, layout, or schedule — only the block-id decode (`ir/kernel` `Tile.render`, `Tile.raster_axes` the
`grid_tile` eligibility). Enumerated `('', 'gm8')` on 2-D contraction rows; wall-time effect is small and
shape-dependent (±2–4% measured), so the search/goldens arbitrate per shape.

**`PLACE`** (STR, pin-only, `010_recognize`) — structural placement of an intermediate edge: `auto` | `fuse` | `cut`,
per edge-class element — `PLACE@cone` (producer-cone inlining), `PLACE@fold` (flash vs separate softmax + P@V
kernels), `PLACE@tuple` (online softmax vs two-pass stats). Precedence `PLACE@<element>` > bare `PLACE` > built-in
`auto` (today: fuse everywhere). `auto` is pin vocabulary only — the stamped value is the *resolved* `fuse`/`cut`,
stamped for `fold`/`cone` only (`tuple` is dominance — never stamped, never enumerated). A forced `fuse` on an
uncertifiable kernel (RoPE'd QK) degrades to `cut` with a log line. Since `@` is not a valid shell var character,
per-element pins ride `EMMY_KNOBS` (e.g. `EMMY_KNOBS="PLACE@fold=cut"`); bare `EMMY_PLACE` pins every eligible edge.
Never enumerated — the `auto` seam is the future search hook for `fold`/`cone`.

**`S_*`** (FLOAT, `loop/stamp/020_stamp_structural_features`) — the LoopOp's structural features (stmt/op histogram +
loop extents + operand dtypes). Not tunable — identity facts that make a knob dict a complete variant identity (the
online prior's feature vector). Skipped by `format_tuning_knobs`.

**`FAST_MATH` / `F16_MMA_F32_ACC` / `FAST_EXP`** (BOOL, pin-only, `_schedule._f16acc_allowed` /
`lowering/kernel/085_fast_exp`) — the **precision-trading family**, never silently on. Precedence per knob: its own
pin > the `FAST_MATH` umbrella > off (`space.precision_pin`). `FAST_EXP` swaps libm `expf` for `__expf`;
`F16_MMA_F32_ACC` offers the f16-accumulate mma atom forks (`a:mma_m16n8k16_f16_f16` — chunked f32 register promote;
its own pin offers on any target, the umbrella only on the consumer dies where f32-accumulate is half rate).
`FAST_MATH` is a meta gate over the others — `unfeatured`, never stamped/enumerated/featurized (the realized fork is
identified by what it enables: `FAST_EXP`'s stamped BOOL, the `TILE` atom token).

### Axis-named schedule keys

A per-node schedule codec is stored keyed `FAMILY@<axis>` — `TILE@<k_axis>` / `STAGE@<axis>` / `REDUCE@<axis>`, the
reduce/contraction axis the node schedules — so a multi-node kernel (flash: `TILE@d` QK + `TILE@sk` PV) can address
each schedule-bearing node; `WSPEC` / `PLACE` stay root-global (bare). The **bare** form is first-class:
`resolve_axis(family, key, eligible)` maps a bare `TILE` to the unique eligible axis (a hand pin on a two-node kernel
raises naming the candidates; a family with no eligible axis drops). Readers go through `family_value(knobs, family)`
so a bare and a suffixed key parse / featurize / golden-match identically — the schema is **invisible on one-node
kernels** (the display collapses `TILE@d` / `REDUCE@d` back to bare when there is a single eligible axis for the
family, so those tables read as before and match the bare golden YAML). The schedule reduce partition IS the
axis-named reduce family — there is no separate native `REDUCE@` decision to collide with (the reduce/split-K
partition is the one reduce family), so `REDUCE` joins `TILE`/`STAGE` on the `@<axis>` keying. The op cache key
re-keys onto the axis-named identity (expected — the transfer handle for the prior, not a regression).

### Odds and ends

- `BINMASK` parsing accepts a binary string (`"101"` = bits 0 and 2), the keywords `"all"` / `"none"`, or a decimal /
  `0x`-hex int clamped to the candidate width.
- `format_tuning_knobs` drops `BOOL` knobs from the rendered `knobs=` line — they're treated as pass-presence markers.
- `HOIST_COMPUTE` and `PAD_SMEM` are BOOL autotune forks emitted in a fixed order (the greedy default first —
  inline-fuse / pad-on respectively); both honor their `EMMY_*` pin.
- The masked-K MMA slab alignment pad is **not** a fork — it's stamped intrinsically on the `Source` at staging (a
  near-strict win greedy deploys without a re-tune).

## Pass directories

Pass files are numerically prefixed so `sorted()` pickup is deterministic. Pick a fresh prefix when adding a rule; the
loader ignores the prefix itself — it's only for ordering readability. Per-pass authoring invariants are in
[`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md); the tile passes (`010_recognize` → `030_split`) and the algebraic
moveset are also documented there.

| Pass                      | What rules do                                                                                |
|---------------------------|----------------------------------------------------------------------------------------------|
| `frontend/decomposition/` | Rewrite frontend ops (`LinearOp`, `MatmulOp`, `SdpaOp`, layout ops, fused `rms_norm` / `layer_norm` / `softmax`) into tensor-IR primitives + layout-only `IndexMapOp`s, broadcast-explicit via `_broadcast.broadcast_to`. |
| `frontend/optimization/`  | `compose_indexmaps`: collapse chains of single-source / single-consumer `IndexMapOp` into one coord_map, so trivial layout kernels don't block fusion. |
| `loop/lifting/`           | `lift_*` rules wrap each surviving tensor primitive in a trivial one-op `LoopOp`.            |
| `loop/fusion/`            | `split_shared_indexmap` (first) fuses a fan-out pure-indexmap `LoopOp` into all its consumers in one rewrite; `merge_loop_ops` then splices adjacent single-consumer `LoopOp` pairs; `dedup_loads` drops identical `(input, index)` Loads. Folding scalar-constant broadcasts into consumers cuts Qwen3-Embedding-0.6B from 394 → 337 kernels. |
| `loop/recognize/`         | Empty (retired) — flash / online-softmax recognition moved into `lowering/tile/010_recognize` (the `_flash` / `_softmax` helpers), so the loop dialect carries no pattern recognizers. |
| `loop/stamp/`             | `stamp_loop_names` (`provenance.name_for`, e.g. `k_rms_norm_3f2a1b`) + `stamp_structural_features` (the `S_*` dict). Runs last in the loop dialect — after fusion and recognition — so every kernel is named / stamped against its final body. |
| `lowering/tile/`          | `LoopOp → TileOp` over the block-DAG Tile IR: `010_recognize` (recognition + inline scheduling via the `_schedule` helper — maps the grid, picks the reduce/output fragment, and **atomizes** — resolves the algebra→atom binding onto the schedule via `_atomize.py` when each option is built, rejecting an unbindable atom at fork construction) → `030_split`. Dispatch is on the carrier algebra (`MAP` / `SEMIRING` / `MONOID`), never a named shape. |
| `lowering/kernel/`        | `010_materialize` is a `TileOp → KernelOp` tier dispatcher (scalar / `_reduce`). A tiled `CONTRACTION` arrives as a high-level `Contraction` node already **built recognize-side** (`lowering/tile/_schedule._contraction_node` at fork-emit — one flat node splitting the algebra params (axes / operands / acc / epilogue) from the schedule (a `tile: TilePlan`); seam #1), so materialize only synthesizes its bare grid-`Write` and **expands** it through the one atom-generic `_factor.factorize` over the shared tiling layer (in `_factor.py`) (the geometry is derived on the `Contraction` node; `_atom.reduce_codegen` emits the shared K-loop and a swappable `store` sink, dispatched off the atom). Then the Kernel-IR peepholes: `030_stamp_types` (+ `040_demote_to_write_dtype`) resolve dtypes, `050_vectorize_loads` / `080_vectorize_stores` / `095_interleave_loads` pack/reorder memory ops, `110_drop_redundant_syncs`. See [`passes/lowering/kernel/ARCHITECTURE.md`](passes/lowering/kernel/ARCHITECTURE.md). |
| `lowering/cuda/`          | `lower_kernelop` renders the `KernelOp` body to a `__global__` source string (`ir/kernel/render.py::render_kernelop`) and mutates the node's op to `CudaOp` in place. |

## Dump hooks (`dump.py`)

`CompilerDump.on_pass(idx, pass_name, graph)` dumps the post-pass graph uniformly for every pass:
`NN_<pass_name>.{json,txt,dot}` (+ `NN_<pass_name>.kernels.txt` if any node has a non-empty `pretty_body()`). Slashes
in the pass name flatten to underscores. The pre-pipeline input graph is dumped separately as `00_input.*` via
`dump.dump_input_graph(graph)`. The uniform strategy means adding a pass automatically gets dumped — no registration.

Per compute kernel, `_dump_per_kernel` writes `<prefix>.kernels/<kname>.json` — a standalone sub-graph (kernel + its
`InputOp` / `ConstantOp` producers) loadable via `emmy run --ir`. When op provenance is present (see
`compiler/provenance.py`), it also writes `<kname>.torch.json` + `.torch.txt`: the **original Torch ops** that kernel
implements, sliced from the pristine pre-decomposition graph by origin id (so the slice is always whole Torch ops),
with an `i/N` coverage header — runnable via `emmy run --ir <kname>.torch.json --bench` to reproduce accuracy /
latency vs torch.

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
