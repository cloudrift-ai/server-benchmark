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
| **`op_cache_key`** | A name-invariant digest of an op's body + knobs — the identity measurements are stored under. A `TileOp`'s structure digests as the α-invariant term hash (`ops.term_key`), never the lowered nest. |

## Module map

| Module | What lives there |
|--------|------------------|
| `pipeline.py` | Engine core: `Pattern` / `Match` / `Rule` / `Pass` / `Pipeline` (the frozen pass layout) plus `Run` — the per-run state and engine loop. |
| `fork.py` | The `Fork` interface (`OptionFork`, `ThunkFork`) and the reusable `Level` + `build_fork_tree` lazy knob-cartesian tree builder. |
| `knob.py` | The `Knob` descriptor system and the `EMMY_<KNOB>` env namespace (borrowing `config.knob_var` / `config.knob_raw`; `format_tuning_knobs` renders the real tuning knobs for `tune` output). Holds NO concrete knob declarations. |
| `search/space.py` | **The single home of the search space.** Every `Knob` instance is declared here and nowhere else — the schedule codecs (`WORK` / `TILE` / `REDUCE` / `STAGE` / `RASTER`, plus the `WSPEC` env-pin alias), the kernel-lowering policy knobs (`VECTORIZE_LOADS` / `INTERLEAVE_LOADS`), and the enumeration value grids (`scalar_tile_moves` & co). A rule that decides a knob imports it from here; registration is construction (`Knob.__post_init__`), and `knob.registry()` imports `space.py` before answering, so the registry is complete in any process. |
| `search/features.py` | The featurizers (`knob_features`, `tile_signature`, the `D_*` / `MMA_*` encodings) — kept beside `space.py` so the whole space (dimensions × values × encoding) is analyzable in one package. |
| `search/db.py` | `SearchDB`, the persistent SQLite store (see Part 6, "Search persistence"). |
| `search/policy/mcts.py` | The in-memory MCTS (`SearchTree`) colocated with its only reader, `TuningSearch`. |
| `search/policy/greedy.py` | `greedy_decide` — the no-tree fork resolver used by `compile` / `run`. |
| `search/two_level.py` | The two-level tuner: outer structural MCTS, inner per-op reward. |
| `search/prior/` | The ONE ranking path: a `Prior` ABC with the cold `OfflinePrior` and the `OnlinePrior` composed behind `FallbackPrior` (`load_prior`). `diagnostics.py` here backs the `eval` reachability / calibration reports; `fit/` is the offline fitter, split by responsibility — `group.py` data representation, `linear.py` trainer+model, `rank.py` rank metrics, `cv.py` fold harness, `run.py` the pure `emmy fit` run harness. |
| `search/data/` | The harmonized read-view over the three data sources (golden configs / DB `perf` rows / prior reservoir): `Sample`, `Dataset`, and `ShapeKey` (the single golden↔measured join key). |
| `search/golden.py` | `GoldenConfig` and its subclasses (see Part 7, "Golden configs and the A/B integrity gates"). |
| `search/audit.py` | The golden drift audit: compile graphs with the golden tier as the only evidence, one MATCH / DRIFT / GAP verdict per consulted fork (via `greedy.golden_audit`, the supported sink; records also carry `unrealized`, the per-entry pin-only signal). Backs `emmy eval golden` (the pin-only offer audit), `--in-model`, and the CI gate (see Part 7). |
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
the kernel's measured mma rows because the warp atom gate read placeholder dtypes off an all-f16 graph.)

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
  `provenance` block) lives in the repo-checked artifact `search/prior/offline_weights.json`, written by the offline
  fitter — library code in `search/prior/fit/`, split along trainer / dataset / harness boundaries: `group.py` (the `Group`
  dataset representation — one ndarray-backed candidate pool + its labels — and the `--features` view parser),
  `linear.py` (the linear trainer + fitted model, owner of the loss — the tier-weighted golden-rank objective plus a
  raw-space L2 penalty (`DEFAULT_L2`, CLI `--l2`) whose job is identifiability, not shrinkage: the rank objective is
  flat in the magnitude of a feature with tiny golden-pool variance, and an unpenalized fit picks arbitrarily there —
  invisible to golden rank, catastrophic at fork scoring, where an undecided prefix scores the feature 0.0 (the
  D_pow2_threads 686 cold-deploy incident); the penalty must be raw-space (`w_z/sd`) because the inflated raw weight
  is an ordinary O(1) z-space weight — and of the static/dyn weight-set split), `rank.py`
  (model-agnostic rank metrics), `cv.py` (fold axes, pooled holdout/train tables, the metrics dict), `run.py` (the
  pure run harness `run_fit` — trainers plug in as callables). Driven by `emmy fit` (which also writes the per-run
  metrics file, and with `--artifact` regenerates the repo-checked artifact in place); the golden case building
  (`build_golden_groups`) lives in `emmy/commands/fit.py` (reconstructing each golden's candidate pool needs the
  command layer's snippet tracer, which `pipeline/` never imports);
  `EMMY_OFFLINE_FILE` (or `emmy eval … --offline-file`) swaps in a candidate
  fit for an A/B. Loading is strict: a missing or `feat_ver`-mismatched artifact is a hard error (refit it), never a
  silent fallback — a retired weight key inside a current-version artifact is merely a dead term. A separate
  `weights_dynamic` set ranks symbolic-axis masked-tile kernels, selected on the stamped `S_ext_n_symbolic_axis`. Two
  hard-coded interaction gates ride outside the linear weights: the atomic-free split-K term, and the tensor-core
  preference pair `D_scalar_on_warp_eligible` / `D_splitk_roundtrip` driven by the scheduler's per-kernel
  `S_warp_eligible` row stamp — which stops a warp-eligible f16 contraction deploying a scalar split tile.
  The linear quality is mapped through an exponential into a positive latency proxy (`exp(-scale·quality)`) whose
  argument clips only at the float-safety bound (~±700) — **the exponential must never saturate over live quality
  scores**. A former ±80 *quality* clip sat inside the live range: every good warp tile collapsed onto one `exp(-8)`
  plateau, greedy's argmin fell through to emission order (the degenerate-`w1x1` gemma misdeploys, 12–29× off the
  golden on a real 4090) while the
  golden-rank eval reported 0 for every tied row. The one consumer needing a bounded value — `FallbackPrior`'s
  offline multiplier — clamps to `e**±8` on its own side; ranking consumers get the strictly-ordered proxy.
- **`OnlinePrior`** (online) — trained from tune measurements (Part 5), composed behind `FallbackPrior`.

A subtlety about features: the `H_*` regime features (GPU / nvcc level) are constant across a pool's siblings, so no
additive weight on them can change a within-pool ranking. Architecture differentiation instead rides *per-candidate*
features that only exist where the hardware offers them: the TMA-conditioned geometry interactions (`D_tma_*`,
mirroring the tile geometry on TMA-staged rows) let one weight set price Hopper/Blackwell tiles separately from
cp.async-era ones, and the warp-grid features (`D_w_grid_*`) separate same-tile different-grid siblings that were
previously byte-identical (the 2026-07-09 4090/5090 golden-sweep TILE findings).

Who consumes the ranking: `TuningSearch` (`tune`) ranks the PUCT frontier; `greedy_decide` (`compile` / `run`, via
`Run.resolve`) picks through the deploy evidence hierarchy, top first: (1) the card's recorded **goldens** — the
verified evidence tier below — then (2) measured -O3 reservoir evidence (`evidence_pick`: the candidate
prefix-consistent with the fastest `H_opt=3` row of the same op), (3) the tune DB's measured rows, and (4) the
`mean_score` argmin only when no candidate has evidence. PLACEMENT resolves BEFORE this schedule pick entirely:
a ROUTING golden entry (a kind entry whose knobs are `PLACE@<seam>` cuts only — the loader rejects an entry mixing
`PLACE` with schedule keys, and `_golden_evidence_index` skips routing entries) is consulted at recognition
(`lowering/tile/_cut.py`, `ShapeKey.joins` on the live card, -O3-gated like the tier itself), each resulting piece
re-recognizing and resolving its OWN `(kind, shape)` through this same hierarchy — see the tile-lowering
ARCHITECTURE's placement-routing section. Fuse is the default by absence; cut is evidence/pin-only.

The two file-backed inputs to that pick — the parsed online prior and the DB perf index — are built **once per
process**, memoized on the source file's `(path, mtime)` (the online file; the DB file plus its `-wal` sidecar). A
generative serve boot compiles ~96 programs, and `structural_key` folds only cc + nvcc flags (never the op shape), so
both inputs are identical across every program: without the memo each compile re-`json.loads`'d the 56 MB
`online.json` and re-scanned the whole perf table. The mtime key invalidates on any on-disk change, so a rewritten
checkpoint or a fresh perf commit is still picked up.

**Goldens are the first evidence tier of a greedy compile.** The per-GPU golden files are the only *measured* data
that ships with a clone — the reservoir and tune DB are machine-local caches written by local tunes, so a fresh
machine (every rented box) previously deployed on pure model extrapolation (the gemma 12–29× cold misdeploys). At a
fork, `greedy_decide` joins the op against the deploy card's recorded goldens — every kind: matmul, attention
(flash), rms_norm, softmax, reduce, pointwise, norm_linear (the fused RMSNorm→linear computed-A megakernel) — by
`ShapeKey` (static and dynamic entries never cross; the key's `kind` discriminator, classified off the stamped
histogram via the sweep identity `S_loop_depth < n_free + n_reduce + n_symbolic`, keeps a flash/norm op apart from
an extent-coincident contraction — and within the rsqrt family a SECOND reduce axis (`S_ext_n_reduce_axis >= 2`,
the contraction beside the statistic reduce) marks the `"fused"` computed-A form, `is_warp` forced True since a
computed-A contraction is a warp mma whose f32 statistic constants would otherwise read scalar;
at the DEPLOY fork the flash op is recognized from its offer's `TILE@dd` + `TILE@pj` pair instead — the tile pass's
restructured twisted op carries re-derived extents only, no histogram, so the stamp classifier cannot fire there; and
the computed-A norm→linear cone is likewise recognized at the deploy fork from its OFFER — a `d*/sync` STAGE row,
the mandatory compute-fill only a computed-A contraction enumerates (the catalog offers only gmem-direct/cp/tma) —
since its PRE-SPLIT fork carries only one reduce axis with the rsqrt still buried in the A sub-body, so the histogram
misreads it as a plain scalar matmul; it is rebuilt to the fused key so the norm→qkv cones join their goldens at cold
deploy, and a fused golden is schema-required to record a `d*/sync` STAGE so its config can
never realize on a plain gmem-A matmul fork of coincident extents)
and picks the offered candidate prefix-consistent with the fastest recorded entry — keys and values compare through
the A/B pin gate's canonical matching. An axis-keyed golden key (a static attention golden's `TILE@dd` + `TILE@pj`)
is all-or-nothing; a bare golden key on a multi-axis family carries the pin-resolution semantics — one plan,
satisfied by ANY same-family realization (how a dynamic attention golden's single bare `TILE` matches the masked
fork's axis-keyed leaves) — and a fast-math entry self-excludes when its atom isn't offered (gate off). The
fastest-first pick also grounds the **fm-never-loses invariant** (statically gated in `test_golden_configs.py`):
within one card's rows of a name, a fast-math entry recording ABOVE the best standard sibling can never realize
(the std row matches first whether fast-math is on or off), so such rows are dropped — an absent fm row just means a
fast-math deploy uses the std config there. Goldens are
**consulted, never trained on**: they enter no reservoir, no checkpoint, no dataset (they are the held-out
acceptance set). Golden µs is deployable-regime truth and never arbitrates a non--O3 compile. A shape match none of
whose entries realizes against the offer logs a loud enumeration-drift warning and falls through to the tiers below.
**That fallthrough is a hazard, not a graceful degradation**, which makes an unrealizable entry worse than an absent
one: on the gemma-4 m64 gate/up cone the tiers below land on a per-cell `fuse`/`b128` scalar config and the edge
measures 54.6 ms against the realizing entry's 169 µs — 323×. Two consequences when recording a fused row. Pinning
`PLACE@cone` **together with** a `TILE` never realizes — no single offered candidate carries both — so record one or
the other; the schema gate accepts a combination the offer then rejects. And a row must be verified to **deploy**,
not merely to pin: `--ab` reproduces configs the enumeration would never choose, so a pin-only row looks healthy in
the isolated golden-reproduction check and only the in-model drift audit (`eval golden --in-model`) catches it.
The tier depends on **no prior**: a resolve with no prior at all — a failed `load_prior` (corrupt/unreadable online
checkpoint) or `Pipeline.run`'s last-resort emission-order resolve — still consults the goldens and deploys a
realizable one (logged loudly when it overrides option-0), so a broken checkpoint can never
silently cost a fork its verified golden.

### `FallbackPrior` and the calibration gate

`FallbackPrior` hands surfaces to the online half only once it is **trustworthy** — fitted AND passing the
**calibration gate** (`Prior.trustworthy`). After every fit, `maybe_refit` measures the median per-op in-sample
Spearman correlation between the model's predictions and its own reservoir labels (`_reservoir_calibration`, persisted
in the checkpoint). Below `CALIBRATION_MIN` the model is quarantined: it keeps training and checkpointing, but deploys,
PUCT, and structural pricing (`greedy._pick_structural`) stay offline, and the verdict is logged.

Why: `fitted` alone let a mis-calibrated model own deploys silently (the 2026-07 RTX 5090 sweeps). In-sample Spearman
is a lenient tripwire that specifically catches the collapse where the model and its rows no longer share feature
names (constant predictions, worse-than-random ranking).

When trusted: `mean_score` / `mean_scores` / `pick` (deploy + eval) are pure-online + evidence. But `score` — the MCTS
*selection* signal — nudges the online µs by the offline prior's dimensionless ranking multiplier
(`online · offline**W`, `W = config.offline_tilt`, neutral 1.0), so PUCT still explores regions the cold heuristic
prices well but the data-poor online model buries.

### Featurizer versioning

`features.FEATURIZER_VERSION` stamps every persisted training artifact:

- **The prior checkpoint** (`to_json`): `from_json` discards a checkpoint from another version WHOLE — model and
  reservoir rows alike — since rows recorded under a retired version's feature names featurize to garbage and a refit
  on them collapses to constant predictions.
- **The autotune DB's `node` rows** (a `feat_ver` column, additively migrated): `diagnostics.node_report` excludes
  rows from another version with a printed count. Pre-stamp rows default to version 1 (the retired pre-rebuild
  feature names) and quarantine conservatively.

Bump the constant on any incompatible change to knob naming or feature encoding; artifacts from the old version then
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
`MMA_tier` warp preference); if `load_prior` returns nothing entirely the recorded goldens still decide the forks
they match and only the golden-less forks fall to option-0 (first leaf, emission order).
Greedy benches nothing, so it can only *use* a prior, never train one.

**Every deploy pick breaks ties by candidate content, never enumeration order.** The model can score many
same-featurized siblings identically (the offline `D_*` geometry doesn't separate an `f2x4` from an `f4x2` fragment or
the `bk` variants — 8 exact ties at the gemma-4 m16 mlp_down/o_proj forks), and one measured row / one golden prefix
can match several offered candidates. Every tier therefore resolves its ties through `knob.canonical_row_key` (the
sorted tuning-knob rendering): the model argmin (`Prior.pick` and the greedy fallback), the reservoir and DB
measured-evidence argmins, and the golden realization pick. An order-broken tie is a per-boot coin flip — leaf order
can shift across processes — and shipped the 2026-07 RTX 5090 gemma-4 image with a bimodal boot-time cubin set.
Pinned by `tests/compiler/pipeline/search/test_deploy_pick_determinism.py` (tier-level permutation invariance plus a
cross-subprocess selected-kernel-set pin, the resolution counterpart of `test_source_determinism.py`).

**Structural options are priced, never raw-scored.** With the trained prior loaded, `greedy_decide`'s
`_pick_structural` prices each side of a structural fork: a nested `resolve` per kernel over a `lowering/tile`-only
pipeline, the price being the `score` of the slice-resolve's partition-fork `Decision`, memoized per `op_cache_key`.
The cheaper kernel set wins, so an unpinned compile deploys the splits `tune` measured best. The nested resolve carries
the deploy's `db`, so each kernel's price follows the same evidence hierarchy as a knob pick (reservoir -O3, then the
tune DB's -O1 ranking rows, model prediction only where unmeasured) — a pure sum-of-predictions comparison would be
exposed to the model's absolute-µs error, which doesn't cancel across different kernel families. Cold, or when a side is
unpriceable, the structural leaf is filtered — a cold compile never changes kernel sets.

**Evidence joins are drift-tolerant.** `Prior.sig_groups` is one contract for both the reservoir -O3 tier and the DB
tier: a candidate's fork-time `S_*` base may carry scheduler stamps the persisted perf rows predate (#311's
`S_warp_eligible` is on no row recorded before it), and a strict-equality signature join would let one added feature
silently disable the whole evidence tier against every existing DB — the ninth-4090-sweep `mlp_gate_up` misdeploy (the
model's `g2k` pick beating the measured-faster fused config it was never allowed to see). The index spans three
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
head of `lowering/tile`. A **terminal** is the state where the cursor reaches `partition_loops` with every structural fork resolved.
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

**Labels credit a partial config with its best completion.** Real benches exist only at leaves, but the prior ranks
partial-knob siblings at every fork level, so the label for any node is the best (min) median latency in µs over its
benched descendants (`1/best_reward`) — the prior regresses on **latency**, and the `1/û` conversion lives in the
MCTS `_select` loop, not
the stored data. `TuningSearch._collect_rows` walks the live tree and emits `(knobs, label)` for every node with a
benched descendant:

- A directly-benched **leaf** uses its `realized_knobs` — the FULL config read off the resolved graph's op in
  `observe`, so knobs stamped at deterministic non-forking lowering steps (`FK` / `BK` / `SPLITK` / `STAGE`) are
  captured, not just the fork knobs.
- A **branch** falls back to `_node_knobs` (its partial `fork.knobs` prefix under the op's `S_*` / `H_*` base),
  labeled with the best latency among its benched descendants.

**Why CatBoost** (chosen by `scripts/prior_bakeoff.py`): the model's greedy pick must not run off to a degenerate
corner. A linear model (the former `BayesianRidgePrior`) is monotone in every knob, so its optimum is always a corner
of the candidate box — the `BR=1` blow-up (4µs → 232µs / invalid kernels). Any **bounded** tree ensemble is
off-manifold-safe (an un-benched extreme inherits the nearest leaf's value), and among them CatBoost also generalizes
to an *untuned* op near-perfectly (leave-one-op-out pick ratio ~1.0 vs xgb/lgbm 1.18, rf 1.31). So one global CatBoost
prior is good enough on a new op that it is **not refit within an op's own search** — it is a fixed model per run.

**Dataset and checkpoint.** The dataset is bounded + batched (`base.Prior`): each tuned op's training rows
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
`collect-node-data` skill: `remote_node_collect.py` drives `scripts/golden_neighbor_bench.py` on the box — a
wall-clock-budgeted sweep of every golden kind's candidate pool, paired -O1/-O3 pinned benches (`run --bench
--ab`), the pool sliced by distance to the recorded golden anchors (the live card's own neighborhood / other
cards' golden vicinities that realize here / a capped uniform tail) and sampled by configurable budget shares,
each batch proportional to the slice's remaining pool (a time-truncated run is a near-uniform sample of it) and
resumed across runs via a ledger the orchestrator pushes/fetches. The budgeted sweep replaced the earlier
ε-greedy `emmy tune --dataset golden` collection mode: search-driven collection over-sampled the branches the
incumbent prior preferred and its wall time grew with the golden set, while the sweep's selection is independent
of the incumbent and fixed-cost (`tune --explore-eps` survives for interactive tuning).

**Measurement freeze** (`data/freeze.py`, driven by `scripts/freeze_node_store.py`). The node DB is a live store —
tunes and merges keep writing into it — so a model fit read directly from it is not reproducible. A *freeze* (v2)
snapshots it into a local DIRECTORY mirroring the `goldens/` layout: one per-`(gpu, compute_cap)` YAML file
(`gpu_name`/`compute_cap` header + a `configs` list) beside a `manifest.json` carrying the provenance header and the
content digests. Each row is **golden-spelled**: the declarative `shape_spec` captured at collection time (`kernel` +
shape fields + `dynamic` — see the bench-to-node recorder below), the verbatim TUNABLE knobs, and the measurement
extension block (`value_us`, `opt` — the nvcc cicc lane, `status`, `variance`/`n_samples`, `run_id`/`measured_at`).
Nothing featurized persists: the loader re-derives `H_*` card-faithfully from the gpu registry
(`Context.from_target`, `H_opt` overridden from the row's `opt`) and the full `S_*` histogram by re-tracing the
kind's snippet once per distinct shape (`data/sample.py::traced_s_features`, the kind-generic sibling of
`compiled_s_features` that selects the traced op keying to the golden's `ShapeKey`; arithmetic fallback with a
warning) — so an encoding-only featurizer change never quarantines a freeze, and there is deliberately no load-time
`feat_ver` gate (the stored `feat_ver`/`knob_ver`/`encoding_ver` are provenance). Only identity-carrying **leaves**
freeze, filtered by `freeze_reason`: a `shape_spec` present (identity-less legacy tune rows stay in the DB but never
freeze), current `feat_ver` at write time, the two plausibility predicates above, `bench_fail` leaves kept as
negatives; branch rows never freeze and no tree schema is stored — prefix rows are re-synthesized at fit time under
the current fork structure. Freezing the same DB twice yields the same digests: every row serializes to one canonical
JSON line, rows sort by that line, the per-file sha256 covers exactly those lines (content-level — immune to YAML
style), the manifest's top sha256 folds the sorted per-file digests, and `created_at` enters none of them. A loaded
row's `op_sig` is the canonical JSON of its `shape_spec` (a stable declarative fold-by-op key — a shape's -O1/-O3
twins share it). `load_freeze` hard-errors on a missing/foreign/corrupt manifest, `freeze_ver` mismatch, a listed
file missing, a per-file digest mismatch, or an un-instantiable row — never a silent fallback; `load_node_rows`
sniffs a path (directory = freeze, sqlite file = DB, a v1 JSONL freeze is refused with a re-freeze pointer) and
yields `NodeRow`s from either, which is what lets every nodes consumer (`eval online --dataset nodes --db`,
`Dataset.from_node_rows` / `fold_node_rows`) take a freeze interchangeably with the live DB. Loaded freeze rows are
parentless with `depth=0` — the marker the diagnostics read as "no tree schema": the fork-regret view skips them and
the golden-anchored descent renders its loud "no fork-tree data" absence row, so a freeze evaluates through the
leaf-level metrics without inventing fork groups. Handing a freeze to a perf-table consumer (`--dataset db` paths)
fails at `open_readonly` with a message naming the freeze/nodes distinction.

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

**Bench-to-node recording** (`search/bench_record.py`) is the node table's second writer: a `run --bench` that
benched pinned golden/`--ab` rows records each clean measurement — and the greedy pick, via its pinned-comparable
`greedy (isolated)` re-bench — as parentless `depth=0` leaf rows, default-on behind a quality bar (the tuner's own
pinned-bench standard; `--no-record-nodes` opts out). This is what keeps manual sweep measurements from evaporating
(the fm-lane optima were found by hand and never reached the store). Pool fidelity comes from recovering each
kernel's pre-descent offer site via `source_chain` (descent stamps further `S_*` deltas, so the terminal op's own
stamps would mis-key the pool): the deepest loop-dialect `S_*`-carrying ancestor, falling back to the deepest
tile-dialect one — the mma tile-lowering preserves no `LoopOp` in `.source`, and without the fallback every
tensor-core kernel was silently unrecordable (both paths digest to the same tune-written `op_sig`, verified on an
RTX 4090). A variant's kernels (split-K main + combine) group under one site and record ONE whole-variant leaf; a
graph whose every kernel loses its site warns loudly instead of recording nothing in silence. Flagged rows (pin mismatch, wrong answer, intensity floor) and the `--ir` path never record.
The caller may pass a **declarative identity** via `run --bench --record-shape '<json>'` (a golden YAML entry minus
knobs/latencies — validated by instantiating its golden kind class; the collection sweep passes its group's spec):
it lands as the node row's `shape_spec` on exactly the leaves whose stamped extents key to the spec's `ShapeKey`
(`ShapeKey.joins` — tolerant of the sweep kinds' snippet-unstable dtype-derived `is_warp`, strict on contraction
twins), so a multi-op snippet's secondary ops (a `linear_norm`'s producer matmul) stay identity-less; a spec
matching zero leaves warns loudly. `shape_spec` is what makes a row freezable in the goldens-format measurement
freeze, and it is identity-intrinsic in the store: `record_nodes` COALESCE-keeps it in both replacement directions,
so a later identity-less re-measurement never erases freezability.
`record_nodes` guards the leaf upsert with **quality-aware replacement**: a newer measurement of unambiguously lower
quality (fewer `n_samples` AND higher `variance`) never displaces a stored leaf, so a drive-by bench can't overwrite
tune-grade data, while comparable or unknown quality keeps plain newest-wins (honest re-measurement still heals
stale rows).

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

`golden.py` holds `GoldenConfig` and its matmul / attention / softmax / reduce / rms_norm / norm_linear / mlp_geglu /
rope / embedding / pointwise subclasses — the `OfflinePrior`'s ground truth. The `rope` / `embedding` kinds are
fork-nothing memory-bound anchors: they record empty knobs and can only serve as `eval golden` regression checks (no
fork means nothing to warm at deploy). Every kind carries `shape_key()` / `snippet()` / `dtype`, so
`tune --dataset golden` and the `run --bench --golden` A/B cover the reduce / pointwise entries too, not just matmul.
`NormLinearGoldenConfig` (the fused `rms_norm(x)·nw @ W` computed-A megakernel) and `MlpGeGluGoldenConfig` (its
multi-channel gate⊗up→GeGLU sibling) are the snippet-reproducible computed-A kinds — both trace to the single fused
mma kernel and share `kind="fused"`; the gate⊗up snippet binds its shared RMSNorm output via a lambda
(`(lambda r: gelu(r@Wg)*(r@Wu))(rms_norm(x))`) since a torch expression cannot otherwise share it.

**Latencies are recorded in pairs, or not at all.** A MEASURED entry carries `emmy_us` and `cublas_us` (both > 0)
— the ordinary case, and the only one `ratio` / `golden` / the A/B gates below mean anything for. An UNMEASURED entry
carries both as exactly `0.0`: a verified-deployable SCHEDULE with no timing, for a shape whose winner is known but
whose µs were never recorded. The deploy tier accepts it and ranks it LAST (`_golden_evidence_index` sorts on
`emmy_us or inf`), so it decides a shape no measured entry covers and yields the moment one is recorded. One-sided is
a recording bug and the well-formedness gate rejects it (`ratio` would silently read 0 or divide by a missing
baseline). Prefer measuring; an unmeasured entry is how a hardcoded deploy default gets out of the scheduler and into
the corpus without inventing numbers for it.

**A matmul golden's layout must match the fork it is meant to decide.** `MatmulGoldenConfig.trans_b` spells the
serving Linear layout — B given `(N, K)`, contracted as `x @ w.T` via an `F.linear` snippet. The traced contraction
carries `b_trans`. The warp tier stages it like any canonical matmul (cp.async and TMA fill an N-MAJOR B slab —
`tile_n × bk`, K stride-1 in gmem and smem alike — drained by the plain no-`.trans` ldmatrix; historically the
transports declined transposed B and the `.lin` forks ran gmem-direct only, the 1.3–2.75× serving gap class), so the
same STAGE spellings realize on both layouts — but the measured µs still differ per layout (different slab geometry
and gmem walk), which is why a golden meant to decide a served model's linear fork must still be TUNED on the
`F.linear` snippet. The two layouts share one ShapeKey on purpose: at a fork the shared bucket sorts by µs, so a
canonical entry (the harness/eval truth) and a `trans_b` entry (the serving truth) coexist under one shape — keep
BOTH current, since with staging realizable on either layout a stale twin's config now deploys cross-layout with its
foreign µs (the layout signal in the stamped `S_*` features / ShapeKey still does not exist). The fused computed-A
kinds (`NormLinearGoldenConfig` / `MlpGeGluGoldenConfig`) carry the same `trans_b` field — their `F.linear` snippets
are the fused edges a SERVED model deploys (`.lin` fused twins; the sync compute-fill stages every B fold channel
via cp.async on either layout, so the same `d*/sync` spellings realize on both).

**Provenance and the in-model drift audit.** A golden file (or entry) may carry an optional `model:` header — the HF
model id whose serving graph the shapes came from (`GoldenConfig.model`; pure provenance, never part of any join key).
Model-tagged goldens opt into the **in-model drift audit** (`emmy eval golden --in-model`, library `search/audit.py`):
the model's serving twins are re-traced **weight-free** (`emmy/serving/twins.py` builds a trimmed random-init skeleton
from `config.json` alone — a trace never reads a weight value) and each tagged card's twins are compiled with the
golden tier as the only evidence (no tune DB, online file pointed at a nonexistent path, deployable nvcc regime
forced — under `-Xcicc -O1` the `H_opt` guard would silently skip golden consultation — and the card targeted via
`Context.from_target`, so verdicts are machine-independent). Each golden-tier consultation yields MATCH (a recorded
golden realized), DRIFT (shape keyed but nothing realizes — always a defect: the recording claims a µs the deploy can
no longer produce), or GAP (no golden for the shape). This is the in-model half of the reproduction check: the
isolated snippet A/B reproduced 68/68 while the in-model deploys drifted (the cast-splice class), which is exactly the
blind spot the audit closes. Coverage is gated as a **ratchet over every GAP key** — contractions, rms_norm/reduce
sweeps, and pointwise forks alike; `major_gap_keys` (uncovered warp-contraction forks, the misdeploy/hang hazard
class) is the close-these-first emphasis view. The CI gate (`tests/compiler/test_golden_drift_gate.py`, offline via
a checked-in `config.json` fixture) pins the per-card gap set exactly — a new gap fails until a golden is recorded
or the baseline is deliberately extended, a closed one fails until its baseline line is deleted, and an emptied
baseline means full model coverage is thereafter enforced (only fork-free deterministic lowerings — rope/embedding
gathers — sit outside the gate, having no fork for the golden tier to decide). The twins track the installed `transformers` modeling code by design: a transformers
bump that changes the forward changes the twins exactly as it changes serving, and the gate goes loudly red.
`scripts/diagnostics/audit_golden_match.py` is the same audit over explicit graph JSONs on a live box.

**The pin-only offer audit** (`emmy eval golden`, same `search/audit` seam) is the record-time complement: for every
forking golden entry it re-compiles the shape's OWN snippet un-pinned (deployable regime, the golden file's own card —
the enumeration is static given shape+context, so no GPU bench) and checks the recorded knobs against the offered
candidates. An entry only a pin can realize (`EMMY_KNOBS` / `tune --golden` benches it, the enumeration never offers
it) reports **PIN-ONLY** — legal as a documented lever while an OFFERED sibling floors the shape (the 4090
`attention.hd512.s4096` split-KV row beside its serial deploy-floor sibling); a shape whose entries are ALL pin-only
reports **FALL-THROUGH** and exits 1: a deploy logs "no offered candidate realizes any of them" and falls past the
golden tier — the missing-floor pathology that deployed a 111 ms 0.03x `mlp_down.m4096` kernel and NaN-poisoned the
downstream accuracy check before the floor-sibling discipline. Fast-math entries audit under the pinned
`F16_MMA_F32_ACC` gate (their own deploy regime). The own-snippet and in-model views genuinely differ: the 5090
`mlp_down.m4096` split-K row realizes standalone but not on the serving twin's epilogue-fused down — the offer audit
passes it and `--in-model` is the authority there, while the s4096 split-KV row fails even standalone, which is what
this audit catches at record time.

**Live-GPU scoping.** `tune --dataset golden` (and `--golden NAME` resolution) scopes to the **live** card's goldens
(`goldens_for_live_gpu`) — names repeat across per-GPU golden files with diverging shapes/dtypes, so a flat union
would tune another card's config under the live card's name. For `tune` the scoping is strict: a live card with no
recorded goldens exits with an error instead of inheriting the union fallback — golden tuning targets the live card's
own recordings only, so an uncovered card is fixed by recording goldens for it, not papered over
(`live_recorded_goldens` is the no-fallback probe that tells an uncovered card from an off-GPU run). `run` / `compile`
`--golden` keep the union fallback on an uncovered card (the seed / transfer flow — the pinned config re-benches
live), and off-GPU the full union is returned (pure-logic tests).

**The A/B carries three integrity gates:**

1. **Realized-vs-pinned knob check — a miss FAILS the row before it benches.** A structurally invalid pin silently
   falls back to the planner's own pick, so benching it would compare greedy to itself and report a fake 1.00× under
   the pin's name (a misspelled hd256 flash pin was read as a form refusal this way). The check runs right after the
   pinned compile: a pin matching no realized knob marks the row `pin_unmatched` / `unreproducible pin … NOT benched`
   (loud error log, row kept in the table and `--json`, zero GPU time spent), and the remaining rows still run.
   Matching is family-aware — a bare golden spelling like `TILE: …` matches its axis-stamped
   `TILE@dd` realization — and values compare through the registered knob's canonical `Knob.parse`, so alias
   spellings like `FAST_EXP=1` don't false-flag. A pin satisfied by ANY kernel counts as honored, which tolerates
   split main+finalize pairs but means a pin dropped on its target kernel that a sibling coincidentally matches passes
   undetected.
2. **Arithmetic-intensity floor.** A row whose shape-implied FLOP/s exceeds the live card's recorded `GpuSpec` peak is
   flagged as a wrong bench, not a fast kernel.
3. **Wrong-answer check.** Each pinned config executes once on the greedy run's inputs and its outputs are compared —
   catching the silently-wrong `g2a` skipped-finalize class.

**Every `run --bench` row measures in the SIGKILL-able bench worker — the parent never launches a kernel.** The
greedy comparison (eager / torch.compile / emmy, the torch side rebuilt in-child — the same transport `tune --bench`
uses) and every pinned golden / `--ab` row run as jobs on ONE persistent worker per run session. That makes the A/B
survive any failed row by construction: a hung kernel dies with the SIGKILL'd child, the parent's CUDA context stays
clean, the row is reported `bench_fail` (with the reason), and the next row's job respawns a fresh child — no
escalation modes, no `os._exit`. NOTE: process placement is unified, the measurement *environment* is not — the
greedy row benches interleaved with the live torch closures (torch allocator state, cuBLAS L2 carveouts resident),
while a pinned row benches emmy-only in a job that never touches torch, so a greedy-row µs and a pinned-row µs for
the same config are NOT directly comparable (the field-observed gap is ~7% on split-K pairs). One number can't be
both torch-comparable and pinned-comparable, so when pinned rows bench, the greedy graph is ALSO re-benched
emmy-only through the same pinned path (one extra worker job, no recompile): the `greedy (isolated)` twin row
beneath each greedy kernel in the table, the `greedy.isolated` block in `--json` — the baseline pinned-row
speedups read against. Record goldens from `--ab`/golden rows only, never from the greedy row's number. The greedy
pick hanging or blowing the bench budget is a *finding* — the exact hazard a golden exists to pin — so pinned rows
still bench after it; failed pinned rows (compile / bench) are kept as `bench_fail` rows, never dropped, and the run
exits non-zero when any row failed. The greedy job also carries the accuracy check (the emmy program runs on the
rebuilt module's real inputs in-child; a numeric failure aborts the run — a latency table for a miscompiling program
is meaningless) and returns that run's `(inputs, outputs)` as the pinned rows' wrong-answer reference. Only the
no-`--bench` accuracy probe still runs in-process (it hosts the `--debug` per-launch dumps and the ncu child's
profiled launches), so with `--bench` those two want a separate plain `run`.

Plus `--json PATH` — a machine-readable record of the whole comparison (backends / greedy kernels / pinned rows with
their flags and a `status` field: `ok` / `pin_unmatched` / `bench_fail`; a failed greedy block carries
`status: bench_fail` + `error` with null timings), so sweep judgments trace to flagged fields instead of parsed
terminal text. Each kernel row also carries **`record_knobs`** — the realized tuning knobs with every schedule codec
family (`knob.SCHEDULE_FAMILIES`: WORK / TILE / REDUCE / STAGE / RASTER) explicitly stamped, OFF spelling included
(`knob.stamp_schedule_families`). That is the map to copy verbatim into a golden YAML `knobs:` entry: an entry that
omits a family leaves it to the planner's replay-time fill, which drifts as the planner evolves (the recurring
unpinned-`REDUCE` phantom-regression class). Golden rows attach to the
run's SHAPE, not a kernel node: a pinned row whose shape matches no greedy kernel (greedy deployed a split
partial+finalize pair) still prints and lands in the record.

## Part 8: Evaluating the prior (`emmy eval`)

**Golden rank is tie-pessimistic** (`eval offline` / `eval online`, via `golden_eval.evaluate_golden`): the golden's
rank counts every row scoring strictly better PLUS every tied row emitted earlier — because greedy's argmin breaks
score ties by emission order, a tie is a loss, not a win. The former strictly-greater count reported rank 0 for every
row inside a tie plateau, which let a saturated prior score "top-1" on goldens that real cold deploys missed by
12–29× (the same convention the fork-regret metric already used: predicted-score ties price pessimistically). Both
flavors come from ONE computation (`prior/fit/rank.dual_rank`): the pessimistic rank gates, and the strictly-greater
**optimistic** rank is reported beside it in `emmy fit`'s metrics file — their gap is the tie-plateau width at the
golden's score, the score-saturation canary that would have flagged the clipped-squash bug at a glance.

**Golden evals featurize under the golden's own card.** `eval offline` / `eval online` rebuild each golden's compile
context as `Context.from_target(compute_cap, gpu_name=…)` — the card recorded in the golden file, with its memorized
SM count / smem specs — never the live host's. The host-context version silently made golden ranks
machine-dependent (a 4090 golden scored on a 5090 host featurized as "sm_89 with 170 SMs"; on a GPU-less host, with
the default SM count) — the occupancy features then priced tiles for a card that doesn't exist, reporting rank 0 on
shapes the real card misdeployed 12–29×. The offline fitter's case builder always did this correctly; the eval
gate now matches it.

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
latencies, so mixing them would corrupt both metrics — the `gpu` key keeps their rows distinct. `--db` also accepts a
measurement freeze (Part 6) in place of the live DB; its rows are leaf-only and parentless, so the report degrades to
the leaf metrics.

The regret/reachability block renders once per prior **half** (offline vs online, labeled) — the composite would
answer with whichever half is active, and the two halves' regrets point at different fixes (cold-start weights vs
training data), so an unlabeled "prior" number destroys the diagnostic.

**Golden-anchored descent** closes the regret view's structural blind spot: regret conditions on forks the search
measured, so a golden in a subtree the search never built — or a shape with zero node data — was silence that read
as health (how the 2026-07 prior-saturation bug hid from regret while the then-broken golden rank said top-1).
Each card block ends with one row per golden recorded FOR that card (goldens never anchor against another card's
rows): the coverage of the golden's path through the explored tree (branch matching is family-aware and
registry-canonical, the A/B pin gate's rule), whether the prior's tie-pessimistic pick keeps the golden's subtree
at each fork (with the measured same-regime gap when lost), and the loud absences — `NO TREE DATA` per unanchorable
golden, a per-card count, and a closing line for cards with recorded goldens but no node rows at all. Coverage
always renders with a denominator: a fully-followed path is exact (`followed 6/6 fork levels to a measured leaf`),
while a partial match's total is an ESTIMATE marked `~` (`followed 2 of ~7 fork levels`) taken from the deepest
sibling chain below the divergence fork — the golden's own branch topology was never materialized, so the
siblings' depth is the only witness of how much tree remains. Regime
discipline is hard: the golden's recorded µs is a deployable (-O3) number and never enters the -O1 walk or its
gaps (the regimes systematically invert); it appears only in the `-O3 pick/golden` endpoint, computed over the
op's `H_opt=3` regime rows with the fast-math regime matched (the `golden_deploy_perf` convention). Diagnostic,
not a gate: a lost fork with a near-equal measured sibling is fine — the gap column is what says so.

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
`010_recognize` (lift `LoopOp` → `TileOp`, recognize the flash / softmax streaming forms, annotate each reduce
`Loop` with its `AxisRole` — the only loop annotation; the algebra is the body — and **atomize**: resolve the
algebra→hardware-atom binding structurally onto the node, so an unbindable atom never becomes one; `_atomize.py`) →
`030_split_reduce` (cross-CTA split-K as a graph rewrite). It **never dispatches on a named
shape** — every decision is gated on the derived role of the stored fold (`PLANAR` / `CONTRACTION` / `TWISTED`; flash
attention is the `TWISTED` fold on the streaming schedule, a twisted monoid is a monoid, selected structurally), not
on a matmul / pointwise / attention archetype. The full design lives in
[`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md).

The step BETWEEN those two — schedule enumeration: mapping the free axes onto the grid and forking the per-node
`TILE` / `REDUCE` / `STAGE` / `WORK` / `RASTER` families — has been REMOVED pending the generic recursive
enumerator. Recognition, the codec, the move catalog and the materializer are untouched; nothing currently maps a
`TileOp`, so every compile that reaches scheduling fails and rides `tests/xfail_registry.py`. See the leading
section of [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md) for the contract the replacement must meet.

## Tunable knobs

A **`Knob`** (`knob.py`) is the canonical schema for one tuning dimension: name, type (`INT` / `BOOL` / `BINMASK` /
`STR`), candidate `hints` (advisory — the rule still validates structural fit), and a help string. Rules stamp values
into `TileOp.knobs` dicts; the autotuner reads those back as the per-hop knob delta in the `lowering` table. Every
knob is declared **in `search/space.py`** — the single home for the whole tunable surface — and imported by the rule
that resolves it (for the schedule codecs, the absent tile scheduler). Declaring a `Knob` IS
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
- A scalar `TILE` parallel block (`par_n·par_m`) is capped at the 1024-thread/CTA hardware limit.
- A `BOOL` knob rejects an unrecognized value instead of coercing a typo (`ture`) to `False`.

### Registered knobs

All declared in `search/space.py`; see [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md) for the per-rule mechanics.
The "owning rule" for the schedule codecs is the tile scheduler (`lowering/tile/_schedule.py`, driven by
`020_schedule`), whose row enumerator spells each family exactly once, site-local, where a row becomes stored state.

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
expressed as a golden moved to `goldens/rtx4080_sm89.yaml` (`dit_xl_2.*`, schedule-only — see the unmeasured-entry
convention under GOLDEN below): the two plain-A projections as `matmul` entries and the block's SDPA as an
`attention` entry.

What could NOT be expressed was deleted rather than mis-filed, both times for the same reason: **no golden kind
describes a LayerNorm**. The DiT prologue is AdaLayerNorm-Zero, while every fused kind (`norm_linear` / `mlp_geglu`)
is RMSNorm by construction — its `snippet()` builds `F.rms_norm` — and the reduce kinds are `torch.sum` /
`torch.nn.RMSNorm`. An entry filed under those would join correctly at deploy (the ShapeKey matches numerically) and
hand every re-tune / `eval golden` / drift-gate consumer the wrong kernel to rebuild. So the two LayerNorm→linear
contractions and the LayerNorm statistic reduce now deploy off the prior; adding a LayerNorm-cone kind is what would
let them be recorded.

**`STAGE`** (STR codec, the tile schedule → `lowering/kernel/010_materialize`) — the operand-staging codec
`d<depth>/sync|cp|tma[/ring][/alt][/p<reg_depth>]` on the typed `Stage` schedule struct (composes with both fragments
of the `TILE` knob): `d<depth>` the gmem→smem ring depth, `sync`/`cp.async`/TMA transport, `p<reg_depth>` the
smem→register double-buffer. `stage=None` (unset / unparseable) = gmem-direct. Also rides the warp-flash TWISTED
stream (`STAGE@<kv>` — the K/V slabs of one streaming block; `reg_depth` clamps to 1), where `d1/tma/alt` /
`d1/cp/alt` is the **alternating single-slab pipeline**: one slab per operand (TMA: its own mbarrier; cp.async: its
own commit group, a uniform `wait_group(1)` completing the older sibling), each refill placed at its operand's kill
point by the liveness-scheduled skeleton (derived from the segment live ranges, not hand-assembled), Q staged through
smem — the wide (64-key) streaming block's staging (flash stream only; the matmul resolvers decline it). See
`lowering/kernel/ARCHITECTURE.md`.

**`WSPEC`** (STR codec, env-pin alias only since step 7) — the warp-specialization producer band `p<np>`, RETIRED as
a stamped row family: realized rows spell the band as `WORK`'s `+p<np>` suffix (the absorb — producer/aux warps are
inventory), and `SCHEDULE_FAMILIES` no longer lists it; `ingest_row` strips a stray `WSPEC` key off a stored /
pinned row before matching (no realized row carries one). The `EMMY_WSPEC` pin is still accepted and the fork level
still enumerates. Legal on a warp `TILE` over a resolved **TMA** `STAGE` within the thread budget
(`block_threads + 32·aux ≤ 1024`, `32·aux ≤ block_threads`); anything else — including the reserved producer `q`
param — degrades to uniform. Empty = uniform SIMT. Materialized as the staged K-loop's producer/compute band split
(`_stage._wspec_kloop`).

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
- `format_tuning_knobs` drops `BOOL` knobs from the rendered `knobs=` line — they're treated as pass-presence markers.
- `HOIST_COMPUTE` and `PAD_SMEM` are BOOL autotune forks emitted in a fixed order (the greedy default first —
  inline-fuse / pad-on respectively); both honor their `EMMY_*` pin.
- The masked-K MMA slab alignment pad is **not** a fork — it's stamped intrinsically on the `Source` at staging (a
  near-strict win greedy deploys without a re-tune).

## Pass directories

Pass files are numerically prefixed so `sorted()` pickup is deterministic. Pick a fresh prefix when adding a rule; the
loader ignores the prefix itself — it's only for ordering readability. Per-pass authoring invariants are in
[`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md); the tile passes (`010_recognize` → `030_split_reduce`) and the algebraic
moveset are also documented there.

| Pass                      | What rules do                                                                                |
|---------------------------|----------------------------------------------------------------------------------------------|
| `frontend/decomposition/` | Rewrite frontend ops (`LinearOp`, `MatmulOp`, `SdpaOp`, layout ops, fused `rms_norm` / `layer_norm` / `softmax`) into tensor-IR primitives + layout-only `IndexMapOp`s, broadcast-explicit via `_broadcast.broadcast_to`. Before `LinearOp` decomposes, `merge_sibling_linears` folds ALL sibling linears sharing one activation (q/k/v, gate/up) into ONE linear over load-time N-concat weights and optional biases (`ConstantOp.source_parts` — the loader concatenates before the `load_ops` chain, zero runtime cost) with `SliceOp` views re-deriving each original output; one launch (one split-K partial+finalize) replaces the per-projection set, and the merged edge is a plain matmul every downstream tier handles. Guards: pristine exclusively-owned parameters with uniform bias presence, and no sibling whose output reaches a graph output through layout ops alone (the view would demote to a copy kernel at the capture ABI). The concat order is graph-insertion order — canonical regardless of match enumeration, because the buffer layout is ABI for goldens and packs. |
| `frontend/optimization/`  | `compose_indexmaps`: collapse chains of single-source / single-consumer `IndexMapOp` into one coord_map, so trivial layout kernels don't block fusion. |
| `loop/lifting/`           | `lift_*` rules wrap each surviving tensor primitive in a trivial one-op `LoopOp`.            |
| `loop/fusion/`            | `split_shared_indexmap` (first) fuses a fan-out pure-indexmap `LoopOp` into all its consumers in one rewrite; `merge_loop_ops` then splices adjacent single-consumer `LoopOp` pairs; `dedup_loads` drops identical `(input, index)` Loads; `fold_output_reshape` retargets a producer's `Write` through a graph-output memcpy-identity flatten (verified exactly over the finite domain; clean affine re-decomposition onto the output strides) — the copy kernel the splicer can't take (reduce-bearing producer × div/mod reader σ). Folding scalar-constant broadcasts into consumers cuts Qwen3-Embedding-0.6B from 394 → 337 kernels. |
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
