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
three things the current design still conflates or scatters:

1. **Operand sharing vs fusion.** "These two matmuls read the same A" is a structural FACT; "lower them as
   one loop with one A fragment" is a scheduling DECISION. `Contraction.folds` bakes both into one field.
   After the refactor, the fact is a shared reference in the tree and the decision is schedule/placement
   state — so the compiler can *see* reuse and *choose* what to do about it.
2. **Kernel boundaries.** Which parts of the tree end up in which CUDA launch is a placement DECISION —
   a `cut | fuse` bit on each parent↔child seam of the tree. Kernels are a derived view: the cut set
   partitions the tree, each partition materializes as one launch, each cut seam materializes its value to
   a buffer. The old design had this backwards: four hand-named PLACE sites (`@fold` / `@fin` / `@cone` /
   `@stat`), each with its own bespoke realizer pass, each invented when a new seam was needed.
3. **How a node names its inputs.** Every node consumes values, and today every node kind spells that
   differently: `Contraction.a` was `Load | Body | str`, `Contraction.b_load` was `Load`-only, `Map.sources`
   is node-only, and a `Reduction` composes by placing a node *inside* `partial` at a load-bearing
   position. Four vocabularies for one relation, and the differences are not algebraic — they are the
   accidents of which case was built first. After the refactor there is ONE: the **operand edge**
   (see *The operand edge* below), whose three inhabitants are the three things an input can be —
   materialized, computed, or shared.

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
- **All fusion happens in loop IR; tile IR only cuts.** Loop-level fusion (`merge_loop_ops` and friends)
  decides what lives together; recognition STRUCTURES that output into trees, never merges more; the only
  placement rewrite at tile level is the cut. This is a hard direction invariant — do not add a tile-level
  fusion rewrite, however convenient. Anything that wants more fusion (a tap, an inline) is realized by
  fusing at loop level with a tile-level cut escape (the stat-tap architecture), never by a tile-level
  merge.

## The tree vocabulary (worked examples)

Agents should read the `ir/tile/ir.py` module docstring first; these examples fix the shapes this plan
talks about.

Every input below is an **operand edge** (`Load` = materialized, a node = computed, `str` = shared) — see
*The operand edge*. Read `└─` as an edge, never as a stmt position.

RMSNorm — projection over a fold (`project ∘ reduce`):

```
Map                               ← body: rsqrt(acc/N+ε) statistic + the per-element sweep + Write
└─ sources: Reduction(axis=k)     ← carrier: sum monoid; partial: square, accumulate
   └─ sources: Load(x[k])         ← the materialized edge — already-cut, no placement decision
```

Fused norm→linear / gate⊗up after phase 1 — sharing is a binding, channels are siblings:

```
bindings: { "x̂": Map(body=scale, sources=(Reduction(stat over k),)) }  # the cone, defined ONCE, keyed by its out
op:       Map(body=swiglu(acc_g, acc_u) + Write,
              sources=(Contraction(a="x̂", b=Load(Wg) → acc_g),
                       Contraction(a="x̂", b=Load(Wu) → acc_u)))
```

Flash — a twisted streaming reduce whose per-step `partial` is a SEQUENCE composing two contractions:

```
Map
└─ sources: Reduction(axis=kv, TWISTED)
   └─ partial: [ Contraction(QK, k=dd),      ← a=Load(Q), b=Load(K)
                 …online-softmax merge: m' = max(m, s); α = exp(m − m'); P = exp(s − m')…
                 Contraction(PV, k=pj) ]     ← a="P" (bound cone), b=Load(V)
```

Position here is **semantic**, not incidental: PV's A operand is the `P` the merge stmts of that same loop
step produce, so the step cannot be hoisted ahead of them. That is why a reduce composes through its
sequence while a contraction composes through labelled edges — see the correction under *The operand
edge*, which records the hoisted-`sources`-tuple design this example once showed and why it is refuted.

Path examples read off these trees: `REDUCE@map.reduce.k` (RMSNorm's fold partition),
`TILE@map.reduce.contraction.dd` (flash QK), `REDUCE@map.contraction.a.reduce.k` (the cone's stat fold —
note it reduces the SAME `k` the contraction folds over; only the path distinguishes them, which is the
single strongest motivation for path addressing).

## Target design

### The operand edge

```python
Operand = Load | Map | Reduction | Contraction | str
```

ONE type for ONE relation — *how a node names a value it consumes*. Its three inhabitants are the three
things an input can be, and there is no fourth:

- **`Load` — materialized.** The value is already in a buffer. This is NOT a degenerate node to be wrapped
  as `Map(body=(Load,))`: it is the **terminal of the cut lattice**. Cut semantics (phases 4–5) say the
  parent consumes a plain `Load` of the child's buffer, so `Load` is exactly what an edge decays to when
  its seam is cut. It carries no schedule slice and admits no placement decision — its seam is trivially
  already-cut. Wrapping it would add a tree level to the overwhelmingly common case and hide the
  `.input` / `.index` leaf reads the whole staging / TMA / dtype machinery is built on.
- **a node (`Map` / `Reduction` / `Contraction`) — computed.** The value is produced by a subtree.
- **`str` — shared.** A name into `TileOp.bindings`.

Invariants:

- **Every edge admits `Load`.** This is the universal part, and it is what cut needs: no edge may be
  node-only. `Map.sources` is node-only today, so a cut at a projection seam cannot yet be spelled in the
  parent at all.
- **A labelled `Contraction` operand stores `Load | str` — a computed one is ALWAYS a binding**, shared or
  not (`_atomize.bind_cone` is already the one binder, and every construction site already produces a
  `Load` or a name). These are precisely the edges where sharing is the point — N siblings over one A is
  the reason the let table exists — so admitting an inline node beside the name would make "shared" and
  "not shared" differ in spelling, the defect this refactor already retired twice (`Reduction.source` vs
  node-in-`partial`; `Contraction.epilogue` vs `Map.body`).
- **A `Map` / `Reduction` source stores the node inline**, and takes the `str` arm only if a source ever
  becomes shared. The discriminator is arity, not taste: *bind iff the value can have more than one
  consumer.* A contraction operand can (the fused edge). A source, by tree ownership, has exactly one
  consumer — its parent. Cut is what creates a second consumer, and a cut child that is shared goes through
  the MIMO foundation (#433) and reaches its parents as a `Load` of the materialized buffer anyway.
- The node arm on a `Contraction` operand field is therefore the *resolved* form only.
- **`Body` is not an operand form.** It survives today only because `ops.resolve` inlines with
  `Body(tuple(lower(bound, bindings)))` — it *lowers* mid-resolve. Resolve is a tree operation and must
  splice the bound NODE (`replace(op, a=bindings[nm])`). That drops the third spelling, makes `tree_nodes`
  uniform (it currently special-cases an operand `Body` and iterates its stmts), and makes `ops.cone_seam`
  readable on a resolved tree instead of pre-resolve only.
- **Closure is a property, checked where it is required — not a universal invariant.** A subtree is
  *closed* when it reads no VALUE name from its enclosing body (iteration variables excluded — they are
  bound by the loop nest, not by any value tree). Closure is what makes a seam liftable, so it is
  *required* of a SHARED binding (one home, N reading sites: a captured name would have to be in scope at
  all of them) and it is the precondition a placement cut asks before lifting anything into its own
  kernel. It is NOT required in general: flash's `P = exp(s − m)` reads the online-softmax carrier's
  running max, updated by the merge stmts of the very loop step that consumes it. That seam is simply not
  cuttable. `ir.captured_values` is the predicate; `validate_bindings` enforces it on shared bindings.
- **Contraction operands are edges; a `Reduction`'s composed steps are a SEQUENCE.** See the correction
  below — `partial` position is semantic, not an accident.

| node | inputs | body |
| --- | --- | --- |
| `Contraction` | `a`, `b` — labelled operand edges (M-resident / streamed) | none; the projection rides the wrapping `Map` |
| `Reduction` | composed nodes in `partial`, positionally | `partial` — the lift, the composed steps, the carrier update |
| `Map` | `sources: tuple[Operand, ...]` | `body` — the per-cell sweep / projection |

**Correction (measured, 2026-07-29): a `Reduction`'s composed nodes cannot become a hoisted `sources`
tuple.** An earlier draft of this section had `Reduction.sources: tuple[Operand, ...]` with `partial` going
stmt-only, on the argument that only an edge can be cut and addressed. Flash refutes it. Online softmax is
`m' = max(m, s); α = exp(m − m'); P = exp(s − m'); O = O·α + P·V`: the PV contraction's A operand (`P`)
depends on `m'`, which the merge stmts of that same loop step compute. Hoisting the PV node ahead of the
merge would read a value that does not exist yet — a correctness break, not a digest change. Measured
directly: `captured_values` over the whole compiler suite reports flash's `P` binding capturing exactly one
value name (the running max) while every other binding's free names are iteration variables only. So a
reduce's per-step body is a **sequence**, composed nodes are steps in it, and their position carries
dataflow meaning. That is the algebra, not a defect — and it is why 1d's node-in-`partial` survivor was
right after all.

What DID survive from that draft is the type honesty, and it resolved in the opposite direction to the one
first sketched. `Body` is `tuple[Stmt]`, and `Contraction` was the only node kind that subclassed `Stmt` —
yet `_schedule` legitimately puts a `Map` (the fused sibling group) into a split reduce's `partial`
(measured across the suite: 4214 `Contraction` occurrences inside a `Body`, 2 `Map`). The answer is not to
widen the sequence to `tuple[Stmt | Node, ...]` but to make the data honest: **every structural node IS a
`Stmt`** (1f, landed). A composed step genuinely occupies a statement position and lowers to a loop nest —
that is what `Contraction` already demonstrated — so uniform `Stmt`-hood removes the special case rather
than encoding it, and generic body walks reach a composed node's children through the same `nested()` they
use for any block stmt. Cut at such a step is then a substitution in place (`partial[i] = Load(buf)`),
well-defined precisely because `Load` is a `Stmt` too.

Consequences:

- **`a` and `b` are symmetric** — `b_load: Load` → `b: Operand` (landed, 1h), `b_trans` is `Load`-only.
  The A/B asymmetry that IS real — A is M-resident and can be compute-filled, B is streamed and staged —
  is a **schedule** fact, and encoding it in the structural type is the same conflation as `folds` baking
  sharing-plus-fusion into one field. After symmetry the schedule gates *state* it
  (`isinstance(c.b, Load)` as an explicit eligibility precondition in `_demoted_atoms`,
  `_tma_operand_rank_ok`, `_sync_operands`) rather than inheriting it from an unstated type guarantee.
  `_demoted_atoms` already writes that check — dead under today's type, and a hint the surrounding code
  already thinks in the wider vocabulary.
- **Cut becomes a pure edge substitution** — `replace(node, <edge>=Load(buf))`. Same edge, same path,
  flippable back, and the parent's keys never re-root. This is what makes `PLACE@<path> = cut | fuse` a
  *bit* rather than a restructuring, and it is the reason the edge must admit `Load`. A composed input
  sitting at position *i* of a `Body` has no stable path to address, and cutting it means deleting a node
  and inserting a stmt at the right index.
- **A composed reduce step is addressed by kind + axis, and cut in place.** Flash's two contractions under
  one `Reduction` are already spelled `TILE@dd` / `TILE@pj` on the live goldens — kind + axis, never an
  index — so a positional sequence costs the grammar nothing. Cut substitutes a `Load` stmt at that
  position; the child re-roots exactly as the plan's cut/fuse-invariance rule already says.
- **The path grammar gains `b` as a peer label to `a`**, and `in.b` beside `in.a` in the reserved
  graph-placement prefix. Reserve both NOW even though no producer needs `b` yet: the grammar and the cut
  model are being designed in phases 2–5, and retrofitting an operand label after evidence is stored
  re-keys entries, against this plan's zero-migration goal. Flash's two `Contraction` sources under one
  `Reduction` stay disambiguated by axis (`TILE@dd` / `TILE@pj`), exactly as the live goldens spell them.

**Sequencing.** The `b: Operand` widening should land *with* its first producer, not ahead of it. Nothing
can construct a non-`Load` B today (`_atomize.semiring_binding` binds cones into `a` only; `_flash` builds
both K and V as `Load`s), so a widened type with no producer is untestable dead vocabulary whose every new
branch reads "decline the optimization". The first real producer is a B-side fused prologue — qk-norm or
RoPE folded into flash's QK, where K is the B operand indexed `[j, dd]` and gemma's qk-norm is a per-`j`
statistic over the contraction axis, structurally the mirror of the norm→linear cone; on-the-fly weight
dequant is the other. The machinery is already shape-symmetric — `SyncOperand(tag="a", value=a_value)` plus
`sync_stat_fill` for the row-invariant prologue is the right primitive, just written A-side. That work is
`_sync_operands` compute-filling a B slab, a column-invariant `cone_seam`, and lifting
`assert len(ops.channels) == 1` on the non-sync path; the type line is the last 5% of it.

What SHOULD land ahead of any producer, because the placement phases depend on it: `resolve` splicing nodes
instead of lowering them, the closure invariant, and the `b` / `in.b` grammar reservations.

### Phase-1 IR generalization

- **Let-bound sharing, referenced by SSA name — no new reference primitive.** `TileOp.bindings` is a table
  of bound subtrees keyed by the name each tree's root defines (its `out`); a consumer references the bound
  value by that plain SSA name in an operand field (`Contraction.a = "x̂"`). There is deliberately NO `Ref`
  node kind: SSA names are already the IR's one reference mechanism (`deps`/`defines`, the rewrite rename
  maps, `structural_key` canonicalization all speak names), and a wrapper node would add vocabulary without
  information. The invariant that replaces it: **tree-wide name uniqueness** — every name defined anywhere
  in `op` + `bindings` is defined exactly once per `TileOp`, validated in `__post_init__` (which also
  rejects a string operand that resolves to no binding — dangling references fail at construction, not at
  lowering). The rewrite machinery renames binding keys through the same rename map as SSA names, so
  references and definitions stay in lockstep and two references to one binding canonicalize differently
  from two copies — sharing is part of the structural identity. Still a let-tree, NOT an implicit DAG: a
  shared subtree has exactly one home (its binding), so paths stay unique and `structural_key` stays a
  tree fold; walks consult `bindings` at name-operand fields instead of needing visited-set DAG traversal.
  (If a shared subtree ever needs to sit in STATEMENT position inside a `Body` — where a bare string
  cannot stand — add a tiny Ref-stmt then; no current form needs it.)
- **Sibling contractions replace fold channels.** `Contraction` drops `folds` and holds one `b` / one
  `acc`. A fused multi-fold edge (gate⊗up) is N sibling contractions under `Map.sources` (a tuple now;
  `source` remains as the len-≤1 compat property) referencing one shared A name. Consequences, all
  deliberate:
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
- **The cone becomes a real node tree.** `a_operand: Load | Body` → `a: Operand` stored as `Load | str` (a
  binding name); the computed-A cone is
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
- **One composition rule for nested reduces (required — the no-duplication invariant).**
  `Reduction.source` (split-K's `Reduction ⊃ Contraction`, spliced ahead of the partial) and
  node-in-`partial` (flash) are two mechanisms for the same thing. Fold `source` into "node at the head
  of `partial`" so `_flatten_nodes` / `.loop` / seam walkers handle ONE shape.
  **Confirmed correct.** A later draft proposed hoisting composed nodes onto a `Reduction.sources` tuple
  instead, on the argument that only an edge is cuttable; flash's `P`-depends-on-the-just-updated-running-max
  refutes it (see the correction under *The operand edge*). Node-in-`partial` stays.
- **One windowing representation (required — same invariant).** A sliced reduce stream currently spells
  its window across two overlapping vocabularies: `Reduction.offset` / `bound` (absolute base/end of a
  cross-CTA slice) and `Axis.source_axis` / `real_extent`. Unify into a single slice/window concept read
  by the realizer and the mask machinery. Orthogonal to the sharing work — may land as its own late
  phase-1 commit.

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
- **Bare-resolution guard**: bare-family sugar resolves to the PRIMARY node for that family — the
  root-most schedule-bearing node in the tree — NOT to "whichever node is unique". So after the cone is
  nodified, bare `REDUCE` on norm_linear/geglu still means the contraction's K fold (the primary), and the
  ~46 stored bare keys keep their meaning; the cone's stat reduce is addressed explicitly
  (`REDUCE@a.reduce.k`). Crucially this is a RESOLUTION rule, not an enumeration limit: the walker
  enumerates forks for EVERY schedule-bearing node under explicit spellings — non-primary nodes are fully
  part of the fork space (full enumerability), they just never claim the bare spelling. General principle:
  nodifying something must never change what existing spellings mean.
- **Only the pre-placement tree receives keys.** Post-rewrite artifacts of derived kernels — the split
  pass's sliced partials, synthesized finalize kernels — are never key targets; the stamper asserts it.
  (A cut CHILD that re-enters recognition is a fresh pre-placement tree and keys normally — the assert is
  about rewrite debris, not re-rooted children.)
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
  placement", so no future spelling migration. Realization constraint for whoever restores them: the
  fusion-direction tokens (`consumer` / `producer`) must be realized as LOOP-LEVEL fusion with a
  tile-level cut escape (the stat-tap architecture) — never as a tile-level merge rewrite; the
  all-fusion-in-loop-IR invariant (north star) applies to them too.

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
would reject them, and the tree is that legality check, precomputed. Once every input is an operand edge,
the seam set is literally the edge set — one `Operand`-typed field per seam, each cut by substituting a
`Load` for the node it holds.

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

**Status: 1a–1d LANDED** on `feature/remove-place-knob`, through `c8755dc2` "Phase 1 close-out:
group-formation gate tests + docs". Sub-steps, in landing order:

1a. **Vocabulary first, no behavior change** — LANDED. `TileOp.bindings` (out-name-keyed) +
    `Map.sources` + construction-time validation, and binding-name resolution threaded through the
    shared walkers (`ops.resolve` inlines every name operand ahead of `lower` / `pretty` /
    `reduce_loop` / `axis_role`; the rewrite rename map covers references). NOTE: the uniqueness
    check is scoped to the BINDING NAMES, not every SSA name — loop IR legitimately binds one name
    from several stmts (a fold's `Init` seed plus its `Accum` steps; split-K's outer reduce folding
    its inner contraction's accumulator), and what a reference needs is only that its own name be
    defined in exactly one place.
1b. **Recognize-side flip** — LANDED. The cone is a real node tree bound in the let table: its
    SOURCE is the row-invariant prologue (the per-row statistic — a projected reduce over the stat
    `Reduction` — plus any k-invariant cone prefix) and its `body` the per-cell normalize, so the K
    seam IS the node boundary (`ops.cone_seam` reads it; `Contraction.stat_prologue`'s stmt scan is
    retired). Every computed A goes through the one binder (`_atomize.bind_cone`): the fused edge,
    the stat-free cone, the mixed-dtype demotion, the pin-driven demoted warp tier, flash's `P`.
    The contraction nodifier emits N sibling `Contraction`s over that one reference instead of
    stacking fold channels; the scheduler stamps one shared row per group; `b_trans` is a
    group-formation gate. `Contraction.epilogue`, `TileOp.tier` and `TileOp.stage` are all retired —
    every schedule slice rides the node it decorates.
1c. **Materialize/split** — LANDED. `_factor` binds the group as one unit (`_AtomOps.siblings`: one
    A fragment, N mma chains, one C fragment per channel) and `030_split_reduce` splits the group,
    deriving the N-component carrier from `ops.group_loop`. STILL OPEN: re-run the #389
    multichannel-split A/B — correct-but-null under the bespoke encoding, it may flip now that the
    split is structural (needs GPU bench time on the gemma gate-up shapes).
1d. **Composition + windowing unification** — LANDED. `Reduction.source` is retired (a composed
    reduce puts the node in `partial`, and "is this a bare statistic reduce?" is the structural
    negation), and `Reduction.offset`/`bound` + `Axis.source_axis`/`real_extent` collapse into one
    `Axis.window` (`parent` + the slice's absolute `base`/`bound`); `real_extent` was dead and is
    dropped.

Byte-compatibility was checked directly, not just by construction: kernel-source digests for
norm_linear, gate-up, matmul, rms_norm and sdpa across gmem-direct / cp.async ring / warp mma /
pinned split-K matched the pre-change commit on every step that claimed to be emission-neutral.

Done when: `make test` green; unit tests cover binding-reference round-trip through rewrite/structural-key
(name uniqueness violations + dangling references rejected at construction; two references to
one binding ≢ two copies), group formation + fallback (disagreeing layouts recognize separately), cone
nodification; accuracy on the geglu / norm_linear golden snippets passes vs eager.

#### Current state of the codebase

What a phase-2 agent actually finds today, and where it sits against *The operand edge*:

| stored field | today | target |
| --- | --- | --- |
| `Contraction.a` | `Load \| str`; node when resolved (1e) | reached |
| `Contraction.b` | same union as `a` (1h) | reached |
| `Reduction.partial` | `Body`, carries composed nodes positionally; all node kinds are `Stmt`s (1f) | reached |
| `Map.sources` | `tuple[Reduction \| Contraction \| Map, ...]` | `tuple[Operand, ...]` (must admit `Load` for cut) |
| `TileOp` | `op + name + place + workers + bindings` (+ the inherited `Op` knob metadata) | unchanged |

Everything else the design section calls for is in place: every schedule slice rides its node
(`Contraction.tile` / `.stage`, `Reduction.reduce` / `.stage`); `Contraction.epilogue`, `TileOp.tier`,
`TileOp.stage`, `Reduction.source`, `Reduction.offset` / `bound` and `Axis.source_axis` / `real_extent` are
all gone; `Axis.window` is the one windowing vocabulary. The shared walkers are `ops.resolve` / `lower` /
`reduce_loop` / `axis_role` / `pretty` / `cone_seam` / `is_group` / `group_loop` / `nodify_reduce`, with
`ir.tree_nodes` the one node walk and `ir._flatten_nodes` the one flatten-a-node-sitting-in-a-Body helper.
Tests: `tests/compiler/ir/tile/test_bindings.py` (binding validation, resolve, reference-vs-copy identity,
rename lockstep, pretty), `test_structural_reduction.py`, and the group-formation gates in
`tests/compiler/passes/test_recognize_boundary_rules.py`. `make test` green at HEAD.

#### Residuals — the delta to *The operand edge*

1e, 1f and 1g have LANDED (below); phase 1 is closed. Nothing remaining blocks phase 2's codec work —
paths and families are well-defined on the landed tree. 1i is a phase-4 prerequisite that belongs in the
realizer's own commit; 1h waits on a producer.

1e. **`resolve` splices instead of lowering; the `Body` arm retires** — LANDED. `ops.resolve` now returns
    `replace(op, a=resolve(bound, bindings))`, so the operand edge keeps its NODE. `Body` was never a
    *stored* spelling (every construction site produces a `Load` or a name), and the resolved arm was
    provably unreached by the canonicalizer: instrumenting `_rewrite(Contraction)` across the whole
    compiler suite recorded 1 hit, on the `str` arm, 0 on the resolved arm — so that branch is now a loud
    assertion. Payoff banked in the same step: `_factor` no longer precomputes
    `seams = {name: cone_seam(node) …}` ahead of `resolve` and threads it through `Ctx.seams`; it reads
    `cone_seam(c.a)` at the point of use, and the side table is gone. `tree_nodes` walks a spliced operand
    like any other node instead of special-casing a stmt `Body`. Verified emission-neutral by
    kernel-source digest A/B over 11 kernels (matmul f32/f16/thin-M, norm_linear f32/f16, mlp_geglu
    f32/f16, softmax, sdpa, sdpa causal, pointwise): all 11 digests identical across the change.
1f. **Every structural node is a `Stmt`** — LANDED. The original plan here (hoist composed nodes onto
    `Reduction.sources`) is refuted; see the correction under *The operand edge*. The residual was that
    `Reduction.partial` is a `Body` (= `tuple[Stmt]`) legitimately carrying structural nodes, which worked
    only because `Contraction` subclassed `Stmt` while `Map` / `Reduction` did not — so any generic stmt
    walk crashed on the `Map` group `_schedule` puts inside a split reduce's partial (hit exactly that
    while building 1g's `axis_names`). `Map` and `Reduction` now subclass `Stmt` with `nested()` /
    `with_bodies()` / a raising `render()`; `defines()` stays the block-stmt default, since a fold's names
    are bound by the `Accum`s inside `partial` exactly as for a plain reduce `Loop`. `_stmt_axis_names`
    drops its node special case. `Map.sources` are deliberately NOT `nested()`: they are node edges,
    reached by the node-aware walk (`tree_nodes`), the same way a `Contraction`'s operand is.
    Emission-neutral by the same 11-kernel digest A/B.
1g. **Closure — LANDED as a predicate plus a shared-binding check.** `ir.captured_values(root, axes)`
    returns the VALUE names a subtree reads but does not define (iteration variables excluded via
    `ir.axis_names`); `validate_bindings` rejects a binding with two or more references that captures
    anything. Measured before enforcing: across the compiler suite, every multi-reference binding is closed
    and exactly one binding captures a value — flash's `P`, reading the carrier's running max, at one
    reference. So the check is enforced where it is a correctness requirement and available as a predicate
    everywhere else; phase 4's seam-legality gate calls it before lifting any subtree. Note this *reverses*
    the earlier framing of flash's `P` as a defect to re-derive: it is legal, it is simply not cuttable.
1h. **`b_load: Load` → `b: Operand`** — LANDED (at the author's direction, ahead of the producer the
    *Sequencing* note argued for). Both edges now carry the same union and read through one set of
    accessors (`_operand_ref` / `_operand_body` / `_operand_name` behind `a_*` / `b_*`); `resolve` splices
    either edge; `tree_nodes` walks both; `validate_bindings` counts references from both, so the closure
    rule applies to a shared B exactly as to a shared A. `b_trans` became `Load`-only (a gmem LAYOUT
    question) and answers `False` for a computed B.

    It did NOT turn out to be dead vocabulary, which is the part the sequencing note got wrong: the
    gmem-direct scalar tier genuinely lowers a computed B through the same `contraction_loop` builder a
    computed A rides — `_ScalarOps.read_col` now reads `c.b_body` symmetrically with `read_row`'s
    `c.a_body` — so the widening has executable semantics and end-to-end tests, not just a type. What the
    staged tiers do is DECLINE, and each now says so explicitly: `_resolve_scalar_stage` and
    `_resolve_warp_stage`'s TMA rank gate require `isinstance(c.b, Load)`, and `_splitk_option` refuses a
    computed B (it would have to slice B's producing subtree, the mirror of the cone's
    redundant-statistic split — nothing builds that yet). That is the schedule stating the asymmetry the
    structural type deliberately does not.

    Still true from the note: the first real B-side producer is a fused per-column prologue (qk-norm or
    RoPE folded into flash's QK, on-the-fly weight dequant), and the work there is `_sync_operands`
    compute-filling a B slab plus a column-invariant `cone_seam` — not the type. Phase 2 must still
    reserve the `b` edge label and the `in.b` prefix in the grammar.
1i. **`Map.sources` must admit `Load`** — OPEN, and correctly deferred to phase 4. It is node-only today,
    so a cut at a projection seam cannot be spelled in the parent at all. The widening is one line; it
    belongs in the commit that teaches the realizer to perform the substitution, not ahead of it.

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
- **Non-closed subtrees are legal but uncuttable (residual 1g, landed).** `ir.captured_values` is the
  predicate; phase 4 must call it before lifting a seam, or flash's `P` — which reads the running max its
  own loop step updates — will be silently cut into a kernel that cannot compute it. Shared bindings are
  already rejected at construction; single-reference ones are the phase-4 caller's responsibility.

## Cleanup

Docs at the end of each landed phase: the pipeline ARCHITECTURE (knob/fork system), the tile-lowering
ARCHITECTURE (bindings + name references, sibling groups, PLACE as edge property), and CLAUDE.md's
tile-lowering blurb
(node vocabulary changes in phase 1). Delete this plan when phase 5 lands.
