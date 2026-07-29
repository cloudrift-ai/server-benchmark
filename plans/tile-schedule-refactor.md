# Tile IR + schedule refactor: generalized nodes, tree-path knobs, PLACE as a per-seam edge property

## Context

Branch `feature/remove-place-knob` deleted the old placement machinery: the `020_cut_edge` /
`025_sink_row_reduce` / `032_fuse_finalize` realizer passes and `_sink.py` are gone, `PLACE` is scrubbed
from the search space, and the gemma golden YAMLs keep their old PLACE keys only as `# retired knob
dropped:` comments. Nothing in the compiler consumes placement seams right now. That is deliberate: the IR
and the knob codec are redesigned first, on the smaller surface, and the placement functionality is then
restored on top of the new vocabulary instead of being ported.

## North star

**The recognized algebra tree is the single semantic object. Everything else is a decoration on it or a
derived view of it.**

A kernel's math is the `project ∘ reduce ∘ map` composition stored as structural nodes (`Map` /
`Reduction` / `Contraction`, in `ir/tile`). Today that tree already separates the *combine* (the algebra)
from the *schedule* (tile plans, reduce partitions, staging). This refactor extends the same separation to
two things the current design still conflates or scatters:

1. **Operand sharing vs fusion.** "These two matmuls read the same A" is a structural FACT; "lower them as
   one loop with one A fragment" is a scheduling DECISION. `Contraction.folds` bakes both into one field.
   After the refactor, the fact is a shared reference in the tree and the decision is schedule/placement
   state — so the compiler can *see* reuse and *choose* what to do about it.
2. **Kernel boundaries.** Which parts of the tree end up in which CUDA launch is a placement DECISION —
   a `cut | fuse` bit on each parent↔child seam of the tree. Kernels are a derived view: the cut set
   partitions the tree, each partition materializes as one launch, each cut seam materializes its value to
   a buffer. The old design had this backwards: four hand-named PLACE sites (`@fold` / `@fin` / `@cone` /
   `@stat`), each with its own bespoke realizer pass, each invented when a new seam was needed.

From this one idea everything else follows:

- **Knobs are paths into the tree.** A schedule key addresses the node (or edge) it decorates by its
  position: `TILE@map.reduce.contraction.dd`. No parallel namespace of hand-invented site names.
- **Keys stamp against the pre-placement tree.** Since kernels are derived, a cut never re-keys the
  decisions inside either half. The tree is the stable coordinate system; launches are not.
- **Derive, never store.** Loop nests, carriers, tile geometry, the multi-channel split state — all
  synthesized on demand from the params (`Reduction.loop` already works this way). Stored state is only
  the params + decisions; anything derivable is a `@property` or a helper.
- **Defaults are the recognized form.** An unseeded shape deploys exactly what recognition produced —
  every rewrite away from it (a cut, a split, an exotic tier) is evidence- or pin-driven. "An unseeded
  site never pays" is the deployment-safety invariant the whole golden system rests on.

## The tree vocabulary (worked examples)

Agents should read the `ir/tile/ir.py` module docstring first; these examples fix the shapes this plan
talks about.

RMSNorm — projection over a fold (`project ∘ reduce`):

```
Map                               ← body: rsqrt(acc/N+ε) statistic + the per-element sweep + Write
└─ source: Reduction(axis=k)      ← carrier: sum monoid; partial: load, square, accumulate
```

Fused norm→linear / gate⊗up after phase 1 — sharing is a binding, channels are siblings:

```
bindings: x̂ = Map(body=scale, source=Reduction(stat over k))     # the cone, defined ONCE, a real node tree
op:       Map(body=swiglu(acc_g, acc_u) + Write,
              sources=(Contraction(a=Ref(x̂), b=Wg → acc_g),
                       Contraction(a=Ref(x̂), b=Wu → acc_u)))
```

Flash — a twisted streaming reduce whose per-step partial composes two contractions:

```
Map
└─ source: Reduction(axis=kv, TWISTED)
   └─ partial: [Contraction(QK, k=dd), …softmax step…, Contraction(PV, k=pj)]
```

Path examples read off these trees: `REDUCE@map.reduce.k` (RMSNorm's fold partition),
`TILE@map.reduce.contraction.dd` (flash QK), `REDUCE@map.contraction.a.reduce.k` (the cone's stat fold —
note it reduces the SAME `k` the contraction folds over; only the path distinguishes them, which is the
single strongest motivation for path addressing).

## Target design

### Phase-1 IR generalization

- **Let-bound sharing.** `TileOp.bindings` (name → node tree) + a `Ref(name)` leaf wherever a bound subtree
  is consumed. Deliberately a let-tree, NOT an implicit DAG: a shared subtree has exactly one home (its
  binding), so paths stay unique, `structural_key` stays a tree fold, and every existing walk gains one
  `Ref` case instead of needing visited-set DAG traversal. The rewrite machinery renames binding names the
  way it renames SSA names, so two refs to one binding canonicalize differently from two copies — sharing
  is part of the structural identity.
- **Sibling contractions replace fold channels.** `Contraction` drops `folds` and holds one `b_load` / one
  `acc`. A fused multi-fold edge (gate⊗up) is N sibling contractions under `Map.sources` (a tuple now;
  `source` remains as the len-≤1 compat property) sharing an A `Ref`. Consequences, all deliberate:
  - A fused sibling group is SCHEDULED AS ONE UNIT — one shared TilePlan/Stage/ReducePlan row for the
    group. Rationale: the fused lowering requires the channels to agree anyway (today the shared `tile`
    field enforced this implicitly), and it means the knob codec needs no sibling ordinals — one
    `TILE@…contraction.k` key per group. Siblings only schedule independently after a cut, when they are
    separate kernels (separate trees) anyway.
  - The fused group loop and the N-component product-monoid carrier (what the cross-CTA split tier folds)
    are DERIVED from the group at lower/split time — the same derive-never-store rule as `Reduction.loop`.
    The derived loop must be byte-compatible with what the `folds` path synthesizes today, because
    `op_cache_key` and kernel identity hang off it.
  - `b_trans`-must-agree stops being a node assert and becomes a group-formation gate: channels with
    disagreeing B layouts simply never group (they were never legally fusable).
- **The cone becomes a real node tree.** `a_operand: Load | Body` → `a: Load | Ref`; the computed-A cone is
  a bound `Map(body=per-cell normalize, source=Reduction(stat))`. The stat reduce is thereby addressable
  and cuttable; `stat_prologue()`'s ad-hoc body-splitting at the K seam becomes a read of the node
  boundary (Reduction = the statistic, Map body = the per-cell cone).
- **One home for projections.** Retire `Contraction.epilogue`. Today a projection can ride either a
  wrapping `Map.body` or the contraction's own epilogue — two spellings of the same algebra, which would
  force the future cut realizer to handle two seam shapes. After this, EVERY projection is a `Map` wrapper;
  a bare contraction/reduction's grid `Write` remains materializer glue (never part of the tree).
- **No root-residue schedule fields.** Retire `TileOp.tier` / `TileOp.stage`: with everything nodified,
  each schedule slice rides the node it decorates (split partials carry theirs on their own node). `TileOp`
  shrinks to `op + bindings + place + workers + knobs` — the root keeps only what is genuinely
  root-global (the free-axis grid binding, the warp split, the knob row).
- **Stretch (do only if the `Ref` work makes it nearly free): one composition rule for nested reduces.**
  `Reduction.source` (split-K's `Reduction ⊃ Contraction`, spliced ahead of the partial) and
  node-in-`partial` (flash) are two mechanisms for the same thing. Folding `source` into "node at the head
  of `partial`" leaves one rule for `_flatten_nodes` / `.loop` / seam walkers. Otherwise defer.

### Knob codec (phases 2–3)

Grammar: `FAMILY@<node-path>[.<axis>][<n>] = value`.

- **Families keep their names** — full backwards compatibility of the outer key shape. `TILE` / `REDUCE` /
  `STAGE` select which property of the addressed node (TilePlan / ReducePlan / Stage); `PLACE` is the edge
  property; `RASTER` / `WSPEC` / `LOOPIFY` stay root-global and bare. The family prefix is what lets the
  prior's featurizers, `_FAMILY_ORDER` grouping, and every stored evidence row parse unchanged.
- **Path** = lowercase node-kind segments from the tree root (`map.reduce.contraction`), plus field-edge
  labels only where kind alone is ambiguous under one parent (`a` for the A-operand edge; a binding's name
  for its subtree). **Axis** = the schedule-bearing axis, the leaf discriminator for TILE/REDUCE/STAGE;
  absent for PLACE (whose path names the seam's CHILD node) and for a `Map` body tile (`TILE@map = f2`).
- **Short paths are canonical.** Stampers and ALL stored evidence (goldens, tune DB, online prior) use the
  SHORTEST spelling that is unique for the kernel's tree: bare family where one node is eligible,
  `FAMILY@<axis>` where the axis disambiguates, longer suffixes only on real collisions. Every live golden
  spelling is already canonical under this rule → **zero migration** of goldens/DB/prior.
- **Resolution**: any unique path-suffix is accepted at pin/parse time; an ambiguous one raises, naming the
  candidates. If a FUTURE structural change makes a stored short key ambiguous for its kernel, resolution
  fails loudly and that entry alone is re-spelled by hand — caught by the compat test, never silently
  re-keyed. The ordinal `<n>` (canonicalized traversal order) exists only for true same-path
  same-kind same-axis collisions (LayerNorm's mean/var); current kernels never need it.
- **Bare-resolution guard**: bare-family sugar resolves only over nodes the scheduler ENUMERATES FORKS for.
  The cone's stat reduce stays out of that set (spell it explicitly: `REDUCE@a.reduce.k`), so the ~46
  stored bare `REDUCE` keys on norm_linear/geglu goldens keep meaning the contraction's K fold after the
  cone is nodified. General principle: nodifying something must never change what existing spellings mean.
- **Cut/fuse invariance**: keys stamp against the pre-placement tree; a cut child re-recognizes as its own
  tree and its keys re-root (`map.contraction.a.reduce.k` → `reduce.k`). Axis names are preserved through
  cuts, so a suffix key names the same node on BOTH sides of a placement decision — one pin string stays
  valid across a cut/fuse A/B, and a child kernel's evidence is shape-transferable (a cut-out stat kernel
  at (M, K) is the same kernel whatever parent it was cut from). The parent-path spelling of a child
  decision must resolve to the same evidence as the child-tree anchor.
- **Shared bindings**: canonical spelling from the binding root (its name); a single-reference binding may
  be spelled through the referencing edge as sugar.
- **RESERVED grammar (implement nothing, reject cleanly, never reuse the tokens):** graph-level placement,
  when it earns its way back in, spells as value-centric placement in this same namespace. Every seam IS a
  value (an in-tree seam materializes the child node's bound output; a graph edge is a named tensor), so
  the vocabulary generalizes from `cut | fuse` to WHO COMPUTES THE VALUE: `own` (its own kernel — in-tree
  cut), `consumer` (inline where read — in-tree fuse / old `fin=fuse`), `producer` (computed upstream —
  old `stat=sink`). Canonical stored spellings stay KERNEL-ANCHORED so evidence rides a ShapeKey and stays
  shape-transferable: the path prefix `in.<operand>` addresses the graph edge feeding an operand
  (`PLACE@in.a.stat = producer`, `PLACE@in.ws = consumer`). Absolute SSA / tensor names (`PLACE@=acc0`,
  `PLACE@=xhat_17`) are accepted ONLY as pin-time sugar resolved against the live compile — site-specific
  names are never stored as evidence. Phase 2 reserves: the `in.` path prefix, the leading-`=` value-name
  pin form, and the three tokens — the parser recognizes and rejects them with "reserved for graph-level
  placement", so no future spelling migration.

Spellings on the live gemma goldens (~580 entries — all unchanged; resolution targets shown):

| Kind | Stored (today = after) | Resolves to |
| --- | --- | --- |
| matmul | bare `TILE` / `REDUCE` / `STAGE` | `contraction.k` |
| norm_linear, mlp_geglu | bare | `map.contraction.k` (one shared row per fused group) |
| flash | `TILE@dd` / `TILE@pj` / bare `REDUCE` / `STAGE` | `map.reduce.contraction.dd` / `…pj` / `map.reduce.kv` |
| rms_norm | bare `REDUCE` | `map.reduce.k` |
| bare reduce | bare `REDUCE` | `reduce.k` |
| pointwise | bare `TILE` | `map` |
| cone stat (NEW) | `REDUCE@a.reduce.k` | `map.contraction.a.reduce.k` |

### Placement (phases 4–5)

`PLACE@<child-path> = cut | fuse` on every in-tree parent↔child seam, replacing the ad-hoc site names.

- **Semantics of `cut`**: split the tree at the seam. The child subtree becomes its own graph node
  (re-entering recognition as a fresh tree); the seam value is materialized to a buffer (f32 state for
  reduce seams, mirroring the split-reduce workspace rule); the parent consumes a plain `Load` of that
  buffer where the child used to be. A cut child that is SHARED (a multi-reference binding) becomes a
  multi-output/single producer via the MIMO foundation (#433) — materialize once, read N times, never
  recompute per consumer.
- **`fuse` is the default on every seam; `cut` is evidence/pin-only.** The recognized tree IS the fused
  form, so the default literally means "no rewrite". This kills the old per-site default zoo and preserves
  the safety invariant: an unseeded site deploys the recognized kernel and never pays for a cut it has no
  evidence for.
- Old → new mapping: flash's `PLACE: fuse` → `PLACE@map = fuse` (projection seam in-kernel);
  `PLACE@cone: cut` → `PLACE@map.contraction.a = cut` (cone → stat kernel + scale kernel + plain matmul —
  the plain matmul is the payoff, it unlocks the standard matmul tiers/goldens the fused computed-A form
  cannot legally use). NEW, previously inexpressible: the 3-kernel split reduce —
  `REDUCE@map.reduce.k = g<n>k` + `PLACE@map = cut` = partial kernels → combine kernel → separate
  elementwise projection kernel. Placement decisions COMPOSE with schedule decisions instead of needing
  new vocabulary per combination.
- **Honest evidence accounting**: the cut-vs-fuse decision is judged on the parent row (N-kernel total vs
  the fused kernel — the pair/triple economics), while each child's schedule evidence lives on its own
  child-tree anchor (shape-transferable). Both spellings of a child decision resolve to the same evidence.
- **Graph-level placement stays out of scope**: inlining a finalize into its CONSUMERS (old
  `PLACE@fin=fuse`, refuted e2e once) and producer-side stat taps (old `PLACE@stat=sink`) cross graph
  edges, not tree seams — this plan does not restore them; the codec merely RESERVES their grammar (the
  `in.<operand>` prefix + `own | consumer | producer` value vocabulary above) so restoring them later
  needs no spelling migration. One forward constraint to honor anyway: if the
  stat-tap plan (`stat-tap-loop-fusion.md`) lands later, its tap seam joins this namespace (old
  `sink`→`fuse`, old `fuse`→`cut`) but keeps `cut` as ITS default — measured anti-wins (qknorm / post_ff /
  m64) make evidence-only taps the safe resting state; that will be the one exception to fuse-default.

### Completeness (proof sketch)

Why the seam/path schema loses no schedules relative to "cut along any SSA variable", and why the fork
space stays enumerable. Not a formal proof — the invariants an implementing agent should be able to defend.

**Claim 1 — legal cut points = node outputs.** A cut along value `v` is legal iff `v` is a complete,
materializable value: every element of `v` is fully computed before any consumer in the other kernel reads
it. SSA names inside a fold (`acc` mid-reduction) are incomplete until the loop closes; per-cell names
inside a sweep have no identity outside their loop instance. The values that ARE complete at a program
point are exactly the outputs of finished algebraic operators — i.e. the bound outputs of structural nodes
(`Reduction.out` at fold close, `Contraction.acc` at contraction close, `Map.out` at sweep close). So the
tree's seam set is the legal-cut set; enumerating SSA names over-generates candidates whose legality check
would reject them, and the tree is that legality check, precomputed.

**Claim 2 — the only quantization loss is mid-pointwise cuts, and it is schedule-trivial.** A cut between
two statements INSIDE one `Map.body` (after `v4`, before the sweep) is not a seam. Such a cut never
changes schedule class: a pointwise chain has one parallelization regime (per-cell), no reuse structure,
and no partition decision — an interior cut only inserts a gmem round-trip between memory-bound
statements. Any case where an interior split ever mattered is expressible by re-associating the tree
(`Map ∘ Map`), i.e. by changing the ALGEBRA, not by extending the codec. Empirical check: every cut the
old system actually deployed lands on a node seam in the new trees — the cone cut's two distinct values
(the stat and `x̂`) are the binding's internal `map.reduce` seam and the `contraction.a` seam respectively.

**Claim 3 — the fork space is a finite, generically enumerable product.** One shared walker yields the
finite site sets; the schedule space is:

```
Π over (path, node, axis) sites:  family vocab(node kind)      # TILE / REDUCE / STAGE rows — as today
× Π over (path, edge) seams:      { fuse, cut } gated by seam legality
× root-globals                                                  # RASTER / WSPEC / LOOPIFY
[ × Π over boundary operands:     { own, consumer, producer }   # reserved, graph-level, also finite ]
```

Seam legality is structural and decidable per seam (carrier materializability — a twisted multi-component
state needs the kernel-finalize arm; f32 workspace for reduce seams; no graph-output crossing) — no
liveness analysis. The old design required hand-REGISTERING each site (`@fold`/`@fin`/`@cone`/`@stat`,
one realizer each); here the site registry is DERIVED, and only the per-family value vocabularies and the
legality gates remain hand-written.

**Boundedness caveats** (enumerable ≠ cheap): cut sets compose (2^seams per tree), so enumeration stays
prior-ranked, never exhaustive — the same combinatorics discipline the fork tree already applies to knob
values; and evidence-only-`cut` is what keeps the enumerated space from being paid for cold.

## Phases

Each phase is independently landable and keeps `make test` green. Parity with pre-refactor deployments is
settled ONCE, at the end (phase 5) — intermediate phases carry only unit-level verification, so agents
should not burn time on golden/dump diffing mid-stream.

### Phase 1 — IR generalization

Goal: the target IR above, with recognition/scheduling/materialization producing byte-compatible kernels.

Sub-steps, in landing order:

1a. **Vocabulary first, no behavior change**: introduce `Ref` + `TileOp.bindings` + `Map.sources` and
    thread `Ref` resolution through the shared walkers (`lower`, `_flatten_nodes`, `pretty`, role/carrier
    readers, `rewrite`/`structural_key`). Nothing produces bindings yet; all tests must pass untouched.
1b. **Recognize-side flip**: the contraction nodifier emits binding + sibling group instead of stacking
    fold channels; the cone is nodified (with the bare-resolution guard from the codec design — decide the
    guard's mechanism here even though the codec lands later, so spellings never shift twice); the
    scheduler stamps one shared row per fused group; `b_trans` becomes a group-formation gate. Retire
    `Contraction.epilogue` and `TileOp.tier`/`stage` in this step — they are recognize/schedule-side
    concepts and moving them later would mean touching the same call sites twice.
1c. **Materialize/split**: factorization reads the group off `Map.sources` + `Ref` identity (one A
    fragment, N mma chains, one C fragment per channel — unchanged emission); the split pass derives the
    N-component carrier from the group. Re-run the #389 multichannel-split A/B — it was correct-but-null
    under the bespoke encoding and may flip once the split is structural.
1d. **Stretch**: `Reduction.source` → head-of-partial, only if 1a's plumbing already did the work.

Done when: `make test` green; unit tests cover Ref round-trip through rewrite/structural-key (two refs to
one binding ≢ two copies), group formation + fallback (disagreeing layouts recognize separately), cone
nodification; accuracy on the geglu / norm_linear golden snippets passes vs eager.

### Phase 2 — codec core

Goal: the grammar above as a self-contained addressing layer, decoupled from any consumer.

Design notes for the implementing agent:

- Build ONE tree walker that enumerates `(path, node, schedule-bearing axis)` triples off a `TileOp` —
  it is the single source of truth the resolver, the stampers (phase 3), and the seam enumerator (phase 4)
  all share. Resist per-consumer walks; divergence between them is the classic failure mode here.
- The resolver generalizes the existing bare-key contract (`resolve_axis`) one level: given a family and a
  possibly-partial key, return the canonical shortest-unique spelling or raise with candidates. It must be
  idempotent (canonical in → canonical out) and total over the sugar forms (bare, `@axis`, any unique
  suffix, full path).
- Family-level reads (`family_of` / `axis_of` / pooled `family_value`) keep their signatures — downstream
  featurizers and ordering must not notice the change.

Done when: unit tests cover round-trip, every sugar level, ambiguity errors, ordinal emission, and the
reserved graph-level forms (`in.` prefix, `=`-value pins, `own`/`consumer`/`producer` tokens) rejecting
with the reserved-grammar error; a compat test resolves every knob dict in ALL golden YAML files (not just
gemma) against its kernel kind's tree and asserts the stored spelling is already canonical.

### Phase 3 — stamp sites

Goal: the scheduler's fork rows and stamped knob dicts spell keys via the phase-2 resolver.

The stamped spellings must come out byte-identical to today's on every current kernel shape (that is what
"short paths are canonical" buys) — DB rows, online-prior features, and golden matching all keep working
with zero translation. Any spelling change on an existing shape is a bug in the resolver or the walker,
not something to migrate around.

Done when: compiling one golden per kind (`--golden`) deploys the recorded config on both cards' golden
sets; the pin-only offer audit stays green; `eval` tooling shows unchanged knob rows.

### Phase 4 — PLACE realizer

Goal: restore placement as ONE generic edge-cut pass + fork enumeration, per the placement design above.

Design notes:

- The realizer takes a stamped `PLACE@<path> = cut`, walks to the seam via the shared walker, and performs
  the split: materialize the seam value, emit the child as its own graph node (MIMO if the binding has
  other references), re-enter recognition for both halves. It should know NOTHING about which seam it is —
  seam-specific knowledge (what buffer dtype, what the child recognizes as) must fall out of the node
  kinds, or it has been factored wrong. Thanks to phase 1 there are exactly two seam shapes: a `Map`
  projection seam and an operand-binding seam.
- Enumeration: recognition offers PLACE rows per seam with option-0 = `fuse` (the no-rewrite row); `cut`
  rows are enumerated only where evidence or a pin exists. Pin precedence: exact path > suffix > bare.
- Composition order with the split-reduce rewrite matters: the split consumes the GRID stage and produces
  partial+finalize; a `PLACE@map = cut` on the same kernel then cuts projection off the finalize. Decide
  and document the pass ordering once (split first, then cuts, is the expected order — cuts operate on the
  post-split trees whose keys are still the pre-placement spellings).

Done when: the cone cut reproduces the recorded pair economics (the `cut_cone_stat` + `cut_cone_scale`
child anchors deploy, ~3.8 µs pair vs 6.0 µs fused on the 5090 per the YAML comments); rms_norm deploys
unchanged under default fuse; the 3-kernel split-reduce form compiles and passes accuracy.

### Phase 5 — PLACE golden re-seeding + the consolidated parity pass

No evidence migration anywhere (short spellings are canonical). Re-seed the retired PLACE goldens by
hand-pinned `--ab` sweeps (the manual method — the tuner is not used for golden work): flash `PLACE@map`,
cone cuts on norm_linear/geglu at the recorded shapes, both cards. The commented-out PLACE entries in the
YAMLs are re-keyed to the new spellings and re-enabled ONLY behind a fresh `--ab` each — pre-wipe µs are
not evidence.

This phase carries THE parity gate for the whole refactor (there are no per-phase parity gates — earlier
phases deliberately deferred it): `emmy eval golden --in-model` on both cards with MATCH across the board;
any DRIFT/GAP is triaged here (golden µs re-verified by `--ab` where the deployed config legitimately
moved); pin-only offer audit green; serving twins deploy from tier; decode TPOT / TTFT within noise of the
YAML-comment baselines.

## Risks

- **Fused-lowering drift (phase 1)** re-keys kernel caches and can invalidate golden µs. Parity is settled
  once at phase 5 — budget triage time there; if the eval-golden pass surfaces broad drift, bisect back to
  the phase-1 commits rather than patching goldens forward.
- Cone nodification changes `--ir tile` dumps and structural test assertions — sweep the structural tests
  early in 1b, they are the likeliest silent-assumption breakage.
- Stored-short-key ambiguity from future nodifications — the resolver fails loudly by design; the phase-2
  compat test is the tripwire. Never "fix" it by silently re-keying evidence.
- Dump/kname churn: kernel names derive from realized ops; verify the per-kernel torch-reproducer slicing
  still attributes the cone's ops correctly once it lives in a binding.

## Cleanup

Docs at the end of each landed phase: the pipeline ARCHITECTURE (knob/fork system), the tile-lowering
ARCHITECTURE (bindings/Ref, sibling groups, PLACE as edge property), and CLAUDE.md's tile-lowering blurb
(node vocabulary changes in phase 1). Delete this plan when phase 5 lands.
