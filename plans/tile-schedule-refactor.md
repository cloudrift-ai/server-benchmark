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

1. **Operand sharing is edge reuse in ONE general fold.** "These two matmuls read the same A" is one
   `Fold` whose step reads one operand edge twice — the tuple lift `(a·b, a·c)` into an N-component
   product carrier. The stored tree is fully symmetric: no privileged operand slot, no channel pairing,
   no name-reference mechanism to police. The factored reading `a·(b, c)` — the bilinear normal form
   tensor cores require — is a DERIVED view (`extract_contraction`), computed at fork-emit where a
   consumer exists, never stored; the fusion decision stays where the north star puts it — loop IR
   fused these components before recognition ever saw them, and recognition only STRUCTURES the fact.
   (Two landed intermediates preceded this: sibling contractions over a let binding (1a/1b), then the
   product-carrier `Contraction` — one `a` edge + `(b_i, acc_i)` channels — as 1j. Each retired the
   previous one's naming machinery; 1k retires the stored kind split itself. See *The pure Fold IR and
   the contraction view*.)
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
   (see *The operand edge* below), whose two inhabitants are the two things an input can be —
   materialized or computed. Sharing is not a third thing an input can be — it is edge reuse inside one
   fold's step (point 1), so no reference arm exists. WHERE an input attaches is decided by one rule,
   **edge iff closed**: a subtree that captures no value from its enclosing step is an operand edge; a
   state-capturing composition (flash's PV) sits in the step sequence at its semantic position. Edges
   are exactly the cuttable seams.

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

Agents should read the `ir/tile/ir.py` module docstring for the LANDED (1j) vocabulary; the examples
below spell the 1k target (*The pure Fold IR and the contraction view*), which the landed tree migrates
to.

Every input below is an **operand edge** (`Load` = materialized, a node = computed) — see *The operand
edge*. Read `└─` as an edge, never as a stmt position.

RMSNorm — projection over a fold (`project ∘ reduce`):

```
Map                               ← body: rsqrt(acc/N+ε) statistic + the per-element sweep + Write
└─ sources: Fold(axis=k, sum)     ← operands: (Load(x[m, k]),) — the materialized edge, already-cut
                                    step: square, accumulate
```

Fused norm→linear / gate⊗up — sharing is edge reuse, ONE fold with a product carrier:

```
Map                                  ← body: swiglu(acc_g, acc_u) + Write
└─ sources: Fold(axis=k, product("acc_g", "acc_u"))
   ├─ operands: (cone, Load(Wg[k, n]), Load(Wu[k, n]))   ← cone = Map(normalize, sources=(Fold(stat),)), inline
   └─ step: vg = xn·wg_e; acc_g += vg; vu = xn·wu_e; acc_u += vu   ← xn read twice — sharing, no channels
```

Flash — a twisted streaming fold; the score fold is CLOSED (an operand edge), PV captures P (a step
element):

```
Map
└─ sources: Fold(axis=kv, exp("m", "l", "O"))
   ├─ operands: (Fold(axis=dd, sum, (Load(Q), Load(K))),  Load(V[kv, d]))   ← QK: closed → an edge
   └─ step: [ …merge: m' = max(m, s·scale); α = exp(m − m'); P = exp(s' − m')…,
              Fold(axis=pj, sum, P·V),       ← captures P → stays in the sequence, at its position
              O = O·α + O_blk; … ]
```

Position here is **semantic**, not incidental: PV's multiplicand is the `P` the merge stmts of that same
loop step produce, so PV cannot be hoisted ahead of them — while QK, which captures nothing, hoists to an
operand edge. That is the **edge-iff-closed** rule, and it is the principled form of the measured
correction under *The operand edge* (which refuted hoisting composed nodes *unconditionally*): closure
decides edge vs step, and the same predicate is cut legality — every operand edge is a legal seam by
construction.

Path examples read off these trees: `REDUCE@map.fold.k` (RMSNorm's fold partition), `TILE@…dd` (flash
QK). With ONE node kind the kind segment carries little — the axis is the real discriminator, and an
operand-edge label (positional, or view-role sugar like `a`) disambiguates the cone's stat fold from the
outer fold contracting the SAME `k` — still the single strongest motivation for path addressing. Phase 2
designs the segment grammar on the one-kind tree; stored short spellings (bare / `@axis`) are unaffected.

## Target design

### The operand edge

**Status (1k revision):** this section is being revised a second time. The LANDED vocabulary (through
1j) stores `Contraction` (one `a` edge + `(b_i, acc_i)` channels) and `Reduction` as separate kinds; the
1k target merges them into ONE `Fold` with a symmetric `operands` tuple and moves the factored bilinear
reading into a derived `ContractionView` — see *The pure Fold IR and the contraction view* below.
Bullets speaking in `Contraction`/channel terms describe the landed intermediate the migration starts
from.

```python
Operand = Load | Map | Reduction | Contraction
```

ONE type for ONE relation — *how a node names a value it consumes*. Its two inhabitants are the two
things an input can be, and there is no third:

- **`Load` — materialized.** The value is already in a buffer. This is NOT a degenerate node to be wrapped
  as `Map(body=(Load,))`: it is the **terminal of the cut lattice**. Cut semantics (phases 4–5) say the
  parent consumes a plain `Load` of the child's buffer, so `Load` is exactly what an edge decays to when
  its seam is cut. It carries no schedule slice and admits no placement decision — its seam is trivially
  already-cut. Wrapping it would add a tree level to the overwhelmingly common case and hide the
  `.input` / `.index` leaf reads the whole staging / TMA / dtype machinery is built on.
- **a node (`Map` / `Reduction` / `Contraction`) — computed.** The value is produced by a subtree, stored
  inline on the edge. Tree ownership gives it exactly one consumer — its parent — because the one in-tree
  relation that HAD several consumers (N matmuls over one A) is now a single edge: the product
  contraction's `a`. So there is no shared arm and no reference machinery: `TileOp.bindings`,
  `validate_bindings`, `ops.resolve` and the rename-lockstep all retire (1j); stored trees are already
  resolved. A computed operand may capture enclosing loop state (flash's `P` reads the running max its
  own loop step updates) — legal, since its one home is in scope; just uncuttable (`ir.captured_values`).

Invariants:

- **Every edge admits `Load`.** This is the universal part, and it is what cut needs: no edge may be
  node-only. `Map.sources` is node-only today, so a cut at a projection seam cannot yet be spelled in the
  parent at all.
- **A `Contraction` is the product node: one `a`, channels `(b_i, acc_i)`, one schedule row.** The
  gate⊗up edge is arity 2; the plain matmul is arity 1 (the overwhelmingly common case — keep its reads
  cheap through len-1 accessors). The product carrier and the fused group loop stay DERIVED (the
  `ops.group_loop` derivation, read off the channel tuple), byte-compatible with the sibling-group loop,
  because `op_cache_key` and kernel identity hang off it. Product FORMATION is a recognize-time gate, not
  a fusion decision: the channels arrive already loop-fused, and channels whose B layouts disagree
  (`b_trans`) simply never product — the group-formation gate moves, unchanged.
- **Channel separation is deliberately forfeited at tile level.** Siblings-under-`Map.sources` could in
  principle exile one channel by a source cut; a product node cannot be split per channel by any edge
  substitution. This is the all-fusion-in-loop-IR invariant applied honestly: de-fusing channels is loop
  IR's decision (don't merge them), never a tile rewrite — and the measured evidence agrees (#389
  multichannel split: correct-but-null). A forfeit, not an oversight.
- **Every computed input stores the node inline** — sources and contraction operands alike; there is no
  stored/resolved distinction any more. If some future form ever needs one value consumed from two places
  neither arity nor a product can absorb, that is a graph-level materialization (the MIMO foundation,
  #433) — the consumers read a `Load` of the buffer — never a new in-tree reference mechanism.
- **`Body` is not an operand form** (retired with 1e), and neither is a name: with inline nodes the
  `rewrite` / `structural_key` canonicalizer must walk a stored node arm — the branch that is a loud
  assertion today — and `tree_nodes` / `ops.cone_seam` read every tree the same way, there being no
  pre-/post-resolve distinction left.
- **Closure is a property, checked where it is required — not a universal invariant.** A subtree is
  *closed* when it reads no VALUE name from its enclosing body (iteration variables excluded — they are
  bound by the loop nest, not by any value tree). Closure is what makes a seam liftable, so it is the
  precondition a placement cut asks before lifting anything into its own kernel — and nothing else
  requires it: flash's `P = exp(s − m)` reads the online-softmax carrier's running max, updated by the
  merge stmts of the very loop step that consumes it — legal (its one home is in scope), just uncuttable.
  `ir.captured_values` is the predicate; with the let table gone there is no construction-time
  enforcement site left, and none is needed — an inline operand's single consumer always sits inside the
  scope it captures from.
- **Contraction operands are edges; a `Reduction`'s composed steps are a SEQUENCE.** See the correction
  below — `partial` position is semantic, not an accident.

| node | inputs | body |
| --- | --- | --- |
| `Contraction` | `a` (M-resident) + channels `(b_i, acc_i)` — labelled operand edges, `b_i` streamed | none; the projection rides the wrapping `Map` |
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

**Refinement (1k):** the refutation is of hoisting composed nodes *unconditionally*. The principled
boundary is **edge iff closed** (`ir.captured_values` empty): flash's QK score captures nothing and may
hoist to an operand edge; PV captures the running max and must stay a step element. The same predicate
is a placement cut's legality, so on the 1k tree every operand edge is a legal seam by construction, and
step-position stays semantic exactly where the algebra demands it.

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
  already thinks in the wider vocabulary. Under the product node each channel carries its own `b_i` edge;
  the gates quantify over all of them (one schedule row ⇒ one verdict for the whole node).
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
  re-keys entries, against this plan's zero-migration goal. Channels share the `b` label — a per-channel
  key (none exists today; the node schedules as ONE row) would disambiguate with the ordinal `<n>`.
  Flash's two `Contraction` steps under one `Reduction` stay disambiguated by axis (`TILE@dd` /
  `TILE@pj`), exactly as the live goldens spell them.

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

What SHOULD land ahead of any producer, because the placement phases depend on it: the `b` / `in.b`
grammar reservations. (The other two items once listed here — resolve splicing nodes, the closure
invariant — landed as 1e / 1g; 1j then retires `resolve` outright, there being no names left to splice.)

### The pure Fold IR and the contraction view (the 1k target)

The purest form of the vocabulary above, designed ahead of the scheduling rework that will consume it:
`Reduction` and `Contraction` merge into ONE fold node, the stored tree goes fully symmetric, and the
bilinear reading tensor cores require becomes a derived, hardware-gated view. Identity keys off the
LOWERED loop nest (`structural_key` lowers first — proved by the 1j digest A/B), so the stored node
shapes are free to change; the invariant to preserve is `Fold.loop`'s output, and the digest harness is
the safety net for every step.

```python
Operand = Load | Fold | Map        # materialized | computed (inline node, one consumer by tree ownership)

@dataclass(frozen=True)
class Fold(Stmt):                  # absorbs today's Reduction AND Contraction
    axis: Axis                     # the reduced iteration space
    carrier: Carrier               # the ⊕ monoid on an N-component product state
                                   #   family "id"  → componentwise ⊕ (sum, max, the gate⊗up product)
                                   #   family "exp" → the twisted action (online softmax / flash)
    operands: tuple[Operand, ...]  # every CLOSED input, one edge each; sharing = repeated reads in step
    step: Body                     # the per-step lift + carrier update, reading operand values by their
                                   # out-names. A SEQUENCE on purpose: a twisted lift may read state its
                                   # own step updates (flash's P) — algebra, not an emitter limitation
    # schedule slices (decorations; composition supplies multi-slice forms — split-K is an outer
    # reduce-partitioned Fold whose step composes the inner tiled one):
    tile: TilePlan | None = None
    reduce: ReducePlan | None = None
    stage: Stage | None = None

@dataclass(frozen=True)
class Map(Stmt):
    body: Body                     # projection / pointwise sweep
    sources: tuple[Operand, ...]   # project ∘ fold; admits Load — the cut terminal
```

- **Edge iff closed.** An input attaches as an operand edge exactly when it captures nothing from the
  enclosing step (`ir.captured_values`, iteration variables excluded); a state-capturing composition
  sits in `step` at its semantic position. ONE rule decides attachment, canonical form (maximal
  hoisting of closed subtrees), and cut legality at once: the seam set is the edge set, legal by
  construction.
- **Sharing is edge reuse.** The gate⊗up fold reads its cone edge once per component; `Channel`
  retires. Two reads of one edge ≢ two edges holding equal subtrees — the structural-identity
  distinction the let table and the product node each preserved in their own vocabulary survives here
  as edge multiplicity.
- **The contraction view — derived, never stored:**

```python
@dataclass(frozen=True)
class ContractionView:             # the bilinear reading of a Fold, computed at fork-emit
    fold: Fold
    m_axis: Axis                   # the tiled output axes, off the placement's trailing grid
    n_axis: Axis
    shared: Operand                # the common multiplicand — the A role (resident, compute-fillable)
    cofactors: tuple[tuple[Operand, str], ...]   # per-component (B-role edge, acc) — the streamed side
    # b_trans / dtypes / K geometry stay consumer-side gates, read off the cofactor Loads

def extract_contraction(fold, place) -> ContractionView | None:
    ...  # carrier family "id", every component additive, and the step factors: exactly one multiply
         # per component, all sharing a common factor (recognition's value-tree test, moved here)
```

  A VIEW and not a rewrite, for three reasons that are design-level, not scheduling-level: the same
  fold must keep offering its scalar / coop rows as fork siblings beside the warp rows (a destructive
  rewrite loses the unrefined reading); one relation keeps one spelling (a stored refined kind beside
  the general `Fold` is the two-spellings defect again); and knob keys stamp against the
  pre-specialization tree (the cut/fuse-invariance rule, applied to refinement). The factored
  `a·(b₁…bₙ)` form does not vanish — it moves into the view, where a normal form belongs: computed by
  the consumer that requires it (tensor cores genuinely need the pure bilinear shape), absent from
  storage. The A/B asymmetry is therefore fully evicted from the type: `shared` / `cofactors` are the
  view's role names, and the resident-vs-streamed schedule facts hang off them.
- **Pipeline:**

```
Loop IR ──recognize──▶ pure Fold/Map tree              (structure only; no hardware knowledge)
        ──schedule───▶ scalar / coop rows read the Fold directly;
                       tensor cores present? view = extract_contraction(fold, place)
                       view ─▶ warp / staged rows: operand roles from the view, TilePlan on the fold
        ──split / materialize──▶ as today: slices off the nodes, roles off the view
```

- **Knob-codec impact (phase 2):** with one node kind, `TILE` / `REDUCE` / `STAGE` select which slice
  decorates the addressed fold (the family IS the slice); kind path-segments carry little and the axis
  stays the leaf discriminator. Operand-edge labels (`a` / `b`, `in.a` / `in.b`) become view-role sugar
  resolved through `extract_contraction`; positional edge addressing is the general fallback. Stored
  short spellings (bare / `@axis`) are untouched.

### Phase-1 IR generalization

**Status note:** the bullets below are the record of the LANDED phase-1 vocabulary (through 1j — the
product-carrier contraction). The 1k revision above supersedes the node split they assume; they stay as
the description of the landed intermediate the 1k migration starts from.

- **No sharing mechanism at all.** An earlier revision of this plan spelled sharing as let-bound subtrees
  (`TileOp.bindings`, SSA-name-referenced, landed as 1a/1b — no `Ref` node, name-uniqueness validated at
  construction, rename-lockstep through the rewrite maps). The product contraction removes the only
  in-tree multi-consumer relation, so the whole apparatus retires with it (1j): `bindings`,
  `validate_bindings`, `ops.resolve`, the rename-lockstep and the uniqueness rules. Computed operands —
  the cone, flash's `P` — store inline on their edge; the tree is a plain tree, walked without a table
  lookup, and `structural_key` stays a tree fold trivially.
- **The product contraction replaces sibling contractions** (which replaced fold channels, 1b).
  `Contraction` holds one `a` and N channels `(b_i, acc_i)`; gate⊗up is arity 2, the plain matmul arity 1,
  and `Map.sources` drops back to ≤1 source in every current form. Kept from the sibling design, all
  deliberate:
  - The node is SCHEDULED AS ONE UNIT — one TilePlan/Stage/ReducePlan row, so the knob codec needs no
    channel ordinals: one `TILE@…contraction.k` key per node, exactly the stored spelling today.
  - The fused group loop and the N-component product-monoid carrier (what the cross-CTA split tier folds)
    stay DERIVED at lower/split time — the same derive-never-store rule as `Reduction.loop` — and must
    stay byte-compatible with what the sibling path synthesizes, because `op_cache_key` and kernel
    identity hang off it.
  - `b_trans`-must-agree stays a FORMATION gate, now at product-formation: channels with disagreeing B
    layouts simply never product (they were never legally fusable).
- **The cone becomes a real node tree.** The computed-A cone is a
  `Map(body=per-cell normalize, source=Reduction(stat))` stored inline on the `a` edge. The stat reduce is
  thereby addressable and cuttable; `stat_prologue()`'s ad-hoc body-splitting at the K seam becomes a read
  of the node boundary (Reduction = the statistic, Map body = the per-cell cone).
- **One home for projections.** Retire `Contraction.epilogue`. Today a projection can ride either a
  wrapping `Map.body` or the contraction's own epilogue — two spellings of the same algebra, which would
  force the future cut realizer to handle two seam shapes. After this, EVERY projection is a `Map` wrapper;
  a bare contraction/reduction's grid `Write` remains materializer glue (never part of the tree).
- **No root-residue schedule fields.** Retire `TileOp.tier` / `TileOp.stage`: with everything nodified,
  each schedule slice rides the node it decorates (split partials carry theirs on their own node). `TileOp`
  shrinks to `op + place + workers + knobs` — the root keeps only what is genuinely
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
  labels only where kind alone is ambiguous under one parent (`a` for the A-operand edge, `b` for a
  channel's). After 1k the kind vocabulary collapses to `map` / `fold` and edge labels resolve through
  the contraction view — see the 1k section's codec note. **Axis** = the schedule-bearing axis, the leaf
  discriminator for TILE/REDUCE/STAGE;
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
| norm_linear, mlp_geglu | bare | `map.contraction.k` (one row — the product-carrier contraction) |
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
  buffer where the child used to be. An in-tree cut child always has exactly ONE consumer — inline
  operands are single-consumer by tree ownership, and the product node's shared A is one edge — so the
  realizer needs no MIMO case (#433 stays a graph-level foundation).
- **`fuse` is the default on every seam; `cut` is evidence/pin-only.** The recognized tree IS the fused
  form, so the default literally means "no rewrite". This kills the old per-site default zoo and preserves
  the safety invariant: an unseeded site deploys the recognized kernel and never pays for a cut it has no
  evidence for.
- Old → new mapping: flash's `PLACE: fuse` → `PLACE@map = fuse` (projection seam in-kernel);
  `PLACE@cone: cut` → `PLACE@map.contraction.a = cut` (cone → stat kernel + scale kernel + plain matmul —
  the plain matmul is the payoff, it unlocks the standard matmul tiers/goldens the fused computed-A form
  cannot legally use; at arity N the same cut yields the N-channel `Load`-A product node — channels stay
  together, the forfeit under *The operand edge*). NEW, previously inexpressible: the 3-kernel split reduce —
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
(the stat and `x̂`) are the cone subtree's internal `map.reduce` seam and the `contraction.a` seam
respectively.

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

What a phase-2 agent actually finds today, and where it sits against the 1k target:

| stored field | today | target |
| --- | --- | --- |
| `Contraction.a` | `Load \| node`, stored inline — no `str` (1j) | absorbed into `Fold.operands` (1k) |
| `Contraction.channels` | `tuple[Channel, ...]` — `(b_i, acc_i)`, each `b_i` an operand edge (1j) | `Fold.operands` + `step`; `Channel` retires (1k) |
| `Reduction.partial` | `Body`, carries composed nodes positionally; all node kinds are `Stmt`s (1f) | becomes `Fold.step`; closed compositions hoist to edges (1k) |
| `Map.sources` | `tuple[Reduction \| Contraction \| Map, ...]`, ≤1 source in every current form (1j) | `tuple[Operand, ...]`; must admit `Load` for cut (1i) |
| `TileOp` | `op + name + place + workers` (+ the inherited `Op` knob metadata) | unchanged |

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

1e–1h and 1j have LANDED (below). 1k — the pure-Fold revision (*The pure Fold IR and the contraction
view*) — REOPENS phase 1 a second time and should land ahead of the phase-2 codec (one node kind is a
smaller grammar than two, and the operand-label question resolves through the view). 1i is a phase-4
prerequisite that belongs in the realizer's own commit.

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
1j. **The product contraction — `str` retires, the let table with it** — LANDED; the north-star revision
    (supersedes the sibling/binding encoding 1a/1b landed). `Contraction` holds one `a` and N
    `Channel`\ s `(b_i, acc_i)`; the fused gate⊗up edge is arity 2 and the wrapping `Map` carries a
    single source. Every computed operand stores inline (flash's `P` moved out of `bindings` onto PV's
    `a` edge); `TileOp.bindings` / `validate_bindings` / `ops.resolve` / the rename-lockstep and
    name-uniqueness rules are deleted, and `TileOp` is `op + place + workers` (+ the inherited knob
    metadata). The `rewrite` canonicalizer walks stored node arms (Map / Reduction handlers registered;
    the carrier renames via `Carrier.rename`, the same rule the `Loop` handler applies). `is_group` /
    `group_loop` went further than planned and RETIRED outright: `Contraction.loop` derives the
    N-channel product fold itself, so the generic `Map`-source paths in `lower` / `reduce_loop` /
    `_factor._emit` cover the fused edge with no group special case, and `_atom`'s channel tuple reads
    `c.channels` directly (the `siblings` threading through `_factor` deleted). `030_split_reduce`'s
    structural branch reads the bare product node from `partial` (no `Map`-of-siblings wrapper).
    Verified: kernel-source digest A/B over 17 golden kernels (matmul f32/f16/warp, split-K, matvec
    coop-t, norm_linear, norm_gate_up `.lin`, mlp_geglu, lm_head, rms/softmax/reduce/pointwise, flash
    hd128/hd256, rms+attention dynM) — ALL digests identical to the pre-1j baseline; the binding tests
    re-targeted to arity (`test_operand_edges.py`: a 2-channel node ≢ two separate contractions;
    formation gates unchanged in `test_recognize_boundary_rules.py`).
1k. **The one-Fold IR + the contraction view** — OPEN; the pure-IR revision (*The pure Fold IR and the
    contraction view*). Merge `Reduction` and `Contraction` into `Fold` (axis + carrier + symmetric
    `operands` tuple + `step` sequence; `Channel` retires; the `tile` / `reduce` / `stage` slices ride
    the one node); hoist every CLOSED input to an operand edge (edge-iff-closed — flash's QK score
    becomes an edge, PV stays a step element); port every `isinstance(…, Contraction)` consumer to
    `extract_contraction` (the derived bilinear view: shared multiplicand + per-component cofactors,
    computed at fork-emit, gated by the tensor-core tiers that need it — never stored). The scheduling
    rework then consumes the view instead of shaping the IR. Verification bar: `Fold.loop` byte-identical
    to today's synthesized nests (the same 17-kernel digest A/B), and the view reproducing today's
    operand→role binding on every golden kind.
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
  the split: materialize the seam value, emit the child as its own graph node (always single-consumer —
  no MIMO case in-tree), re-enter recognition for both halves. It should know NOTHING about which seam it
  is — seam-specific knowledge (what buffer dtype, what the child recognizes as) must fall out of the node
  kinds, or it has been factored wrong. Thanks to phase 1 + 1j there are exactly two seam shapes: a `Map`
  projection seam and a fold operand edge — and after 1k every operand edge is closed by construction
  (edge-iff-closed), so seam legality is structural rather than a checked predicate.
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
- **1j identity churn**: the channel-tuple conversion and flash's `P` moving from `bindings` to an inline
  edge must leave kernel-source digests and `structural_key` unchanged — prove it by the same 11-kernel
  digest A/B 1e/1f used, before trusting any golden µs. (Landed clean: 17/17 digests identical.)
- **1k identity churn**: the `Fold` merge must keep `Fold.loop` byte-identical to today's synthesized
  nests, and `extract_contraction` must reproduce the recognize-time operand→role binding it replaces on
  every golden kind — the same digest A/B gates every step; a divergence is a bug in the view, never
  something to migrate around. Watch the QK hoist specifically: moving the score fold from step to edge
  must not perturb the lowered nest's stmt order (the flatten walk re-splices it at the head).
- Cone nodification changes `--ir tile` dumps and structural test assertions — sweep the structural tests
  early in 1b, they are the likeliest silent-assumption breakage.
- Stored-short-key ambiguity from future nodifications — the resolver fails loudly by design; the phase-2
  compat test is the tripwire. Never "fix" it by silently re-keying evidence.
- Dump/kname churn: kernel names derive from realized ops; verify the per-kernel torch-reproducer slicing
  still attributes the cone's ops correctly once it lives inline on the operand edge.
- **Non-closed subtrees are legal but uncuttable (residual 1g, landed).** `ir.captured_values` is the
  predicate; phase 4 must call it before lifting a seam, or flash's `P` — which reads the running max its
  own loop step updates — will be silently cut into a kernel that cannot compute it. After 1j there is no
  construction-time closure check left (no shared bindings exist to require one) — every operand seam is
  the phase-4 caller's responsibility.

## Cleanup

Docs at the end of each landed phase: the pipeline ARCHITECTURE (knob/fork system), the tile-lowering
ARCHITECTURE (the product-carrier contraction + inline computed operands after 1j, PLACE as edge
property), and CLAUDE.md's tile-lowering blurb
(node vocabulary changes in phase 1). Delete this plan when phase 5 lands.
