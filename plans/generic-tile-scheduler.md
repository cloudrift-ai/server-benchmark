# Generic tile scheduler — every role emits rows

## The claim

The tile IR is compositional — ONE `Fold` node kind, nested (see the next section). The schedule output is
compositional — `TileOp.schedule`, per-node slices keyed by the tree-path codec. **The enumeration between them must
be compositional too, or the two ends do not meet.**

Stated as the success criterion: **every role emits rows through ONE recursive walk of the site tree; no role builds
`TileOp`s directly, and no term shape gets its own path.** A row is a joint assignment across every site of a term;
the tree that generates it is the term's own.

## The IR beneath it — the unified `Fold`

There is ONE stored node kind. The collapse LANDED (`7ba7468c` … `08c3e524`), so this section records what the term
IS — the site walk, the role derivation and the emitter dispatch below all read it — not work this plan does.

```
Fold(
    axis:     Axis | None,             # None = the zero-axis node (what `Map` was)
    operands: tuple[Load | Fold, ...], # the operand EDGES, bound POSITIONALLY to the lift params
    lift:     Lambda,                  # λ(k, v₁…vₙ) → S — the ONE joint stored per-element function
    init:     tuple,                   # the ⊕ seeds; () at zero axes
    combine:  Lambda | None,           # S × S → S — THE ⊕; None at zero axes
)
```

`Map` and `Contraction` are not types: `Fold.projection(...)` / `Fold.contraction(...)` build them and `axis is
None` / `is_contraction(x)` read them back. The predicate form matters — `isinstance` answered `False` for the
`Load` edges and plain `Assign`s that a step-stream scan walks, and a bare `x.role is …` raises on them.

**There is no `Edge` type**, and the question of adding one dissolved rather than being decided: the only consumer
that wanted a uniform access on a COMPUTED edge was the A/B derivation, and A/B rides operand order instead (below).
An operand edge stays the union it always was — MATERIALIZED (a gmem `Load`, whose `input` + `index` ARE the
source/access pair) or COMPUTED (the node itself, stored inline, its one consumer its parent). Two reasons to keep
it that way if the question is reopened: every structural node is a `Stmt`, which is what lets an inline node occupy
a statement position (`_splice_operands` splices an edge's producing stmts straight into a `Body`, `_flatten_nodes`
lowers it in place), and `term_key` is `repr()`, so a wrapper is a new level in every stored term's key for no
consumer. Reopen only if a pass appears that genuinely needs a computed edge's index space; today none does.

**The projection is not a field.** `Map(fn, sources=(Fold,))` is a zero-axis `Fold` whose one operand is the
reducing fold; the projection IS the outer node's `lift`. Composition supplies what the `fn` field did, so RMSNorm's
normalize, softmax's, the relu epilogue and flash's `divide(O, l)` are the same shape at the same depth. No node
carries both a per-element and a per-cell lambda.

**The A/B split rides the operand ORDER — not the accesses.** Node-locally `x[a0, a2]` and `w[a2, a1]` are
*symmetric* — each carries the K axis plus one free axis (the dump in example 4 shows it) — so telling M from N
requires the PLACEMENT, a caller fact on the `TileOp` and deliberately absent from the node. The accesses cannot
decide it, so the stored order `(b₀, a, b₁…)` carries the split: it is the spelling `Fold.contraction(...)`
generates, it is what the byte-identity gate pins at both arities, and it keeps `a` / `b` answerable node-locally,
which the path codec requires (`PLACE@a`). The one genuinely access-derived fact stays access-derived: `b_trans`
reads `Load.index` off the B edge once the edge is known. No stored `a` field, no `shared` index, no bilinear parse
— but no access-based derivation either.

**The role derives from arity.** Nothing stores a role (`Fold.role`):

| shape | axis | operands | lift / combine | derived role |
| --- | --- | --- | --- | --- |
| pure pointwise | none | 0 | — | FREE |
| projection over a node | none | 1 | — | FREE |
| bare reduce / stat | one | 0–1 | any / componentwise | PLANAR |
| contraction | one | 2+ | `multiply` over distinct params / `add` | CONTRACTION |
| split-K outer reduce | one | 1 (a contraction) | identity / `add` | CONTRACTION |
| streaming (flash, online softmax) | one | 1+ | any / exp-family | TWISTED |
| unbindable matvec | one | 0 (loads inline) | any / `add` | PLANAR |

The last two rows are the load-bearing ones and both hold: the matvec demotion is a FORMATION fact (recognition
keeps its loads inline in the lift, so there are no edges to parse and the row falls through to PLANAR), and the
collapse reading below is exactly "move the edges inline", which flips a contraction to a plain fold by the same
table. The reading mechanism and the role derivation are not two mechanisms.

**`Fold.demoted()` is back** (phase 2, ~15 lines): move each operand edge into the lift body before the first read of
its bound name, ties in operand order — a materialized `Load` verbatim, a computed edge as the structural NODE, which
`_flatten_nodes` flattens at lowering so the derived loop is byte-identical to the hoisted spelling's. Its one caller
is the COLLAPSE term reading (`_schedule._readings`), and `ir/ARCHITECTURE.md` names it again.

### What it means for this plan

**`type(node)` dispatch is not an option, and neither is role dispatch.** With one node kind there is no type key to
reach for. The role is not the replacement either (see "Dispatch is two predicates on the node"): the table above is
how the IR ANNOTATES a loop and how the materializer reads it back, not how the enumeration selects catalogs. Note
what the table itself shows — TWISTED is derived by matching the combine's operation family, and the split-K row is a
structural probe for one composition. Neither is a fact about arity, and neither belongs in a scheduling decision.

**`family_sites` restates without losing a site.** `REDUCE`/`STAGE` take every fold WITH an axis; `TILE` takes the
contraction folds (`is_contraction` — the same question `path._walk` asks; today `family_sites` asks the ROLE
instead, and the two diverge on exactly the split-K wrapper) plus the root zero-axis fold with no operands that is
not a raw-loop escape; `PLACE` every non-root, non-derived site. `Site.axis` is already `None` for the pointwise
node, so the zero-axis fold needs no new case.

**Keys survive — under one constraint, which holds today.** `_SEGMENT_TOKENS` is `{"map", "fold", "a", "b"}` and
spelling is short-path-canonical (bare → `FAMILY@<axis>` → anchored path subsequence). Checked against the stored
corpus: the only path-spelled keys anywhere in the golden YAMLs are `TILE@pj` (45), `TILE@dd` (42), `PLACE@cone`
(60), `PLACE@stat` (4), `PLACE@fin` (1) — all AXIS names, segment-independent — and **`PLACE@a` (9 rows), which is
the A-edge SEGMENT**. Nothing stored spells `map` or `fold`. So:

> The frozen-key invariant holds **iff `path._walk` keeps emitting `a` / `b` edge labels for contraction
> folds** — which it does: `_walk` tests the bilinear reading AHEAD of the generic operand walk and labels off the
> derived `node.a` / `node.channels`, so `PLACE@a` means what it meant.

That is the one constraint the one-kind term imposes on `path.py`, it is checkable in isolation, and `PLACE@a`'s
nine golden rows are its test.

**Per-operand lifts are a DERIVED view, never storage.** "Different lift per operand, one combine" is the right
reading of a contraction — but the per-operand part is a canonical partition of the existing `lift.body` (a stmt
reading one operand's bound name belongs to that operand; a stmt reading several is the joint step), not new fields.
It must stay derived because kernel identity is the α-invariant `repr()` of the stored term (`ops.term_key`, now in
`ir/tile/_key.py`): a stored partition
means two recognitions that split the same algebra differently key apart, splitting cache and evidence for identical
kernels. Derived, it costs nothing and unlocks the fused per-operand prologue `Fold.contraction`'s docstring already
anticipates (qk-norm / RoPE folded into a score, on-the-fly dequant) — which is a STAGE-family concern. Phase 2 did
NOT expose `operand_lift(i)`: nothing reads it, and the computed-`a` fill it would serve reads the cone off the edge
(`ops.cone_seam`) instead. Add it when a per-operand prologue emitter actually exists.

**The collapse did not re-spell terms.** Hoisting a load to an operand edge stays recognition's decision — a bare
sum keeps its load INLINE in the lift, a contraction hoists both operands because the schedule needs their
addresses. The collapse renamed the node kind and moved the projection down one level; it hoisted nothing. That is
what bounded its blast radius, and it is why examples 1–3 and 7 below are byte-identical to their pre-collapse dumps
but for the header word.

**The evidence orphaning is already paid.** `ops.term_key` is the `repr()` of the canonically-renumbered term, so the
changed class name and the projection's changed nesting changed the key for every term — and with it `op_cache_key`
and `Graph.structural_key`'s op field. **The cubin cache and every recorded DB / reservoir measurement keyed on
`(ctx.structural_key, op_cache_key)` are orphaned**; the golden corpus survives untouched because it keys on knob
spellings. Measurement therefore starts from an empty reservoir — a cost already sunk, not a risk
still to manage.

## Design

**One inventory, then a product over sites.** A row is a joint assignment across every scheduling SITE of a term.
The kernel's ONE worker inventory is chosen FIRST, at the root; the sites are then a product under that fixed
context:

```
def enumerate(tile, ctx) -> list[Row]:
    return uniform_keys([r for term in readings(tile.op, ctx)   # collapse / mixed-A — the two tree rewrites
                           for work in inventories(term, ctx)   # w<M>x<N>[+p<n>] | t<N>[x<M>] | "" — chosen, not folded
                           for raster in raster_values(term)    # kernel-global, like work
                           for r in rows(root_site(term), work, ctx)])

def rows(site, work, ctx) -> list[Row]:                  # Row = {canonical key -> spelled value}
    out = []
    for v in values(site, work, ctx):                    # RESOLVE against the fixed inventory; drop on ValueError
        for combo in product(*(rows(c, work, ctx) for c in children(site))):
            row = merge(site, v, combo)                  # spells each slice at ITS canonical path (ops.Sched.key)
            if legal(site, v, combo, ctx):               # the sibling/parent equalities — see below
                out.append(row)
    return out
```

**`WORK` leads because the codec says so.** `TilePlan.parse(spec, work)` and `ReducePlan.parse(spec, work)` both take
the inventory as an INPUT: a `TILE` value's unit widths and a `REDUCE` value's coop width are READ OFF the inventory,
never carried in the value. So the dependency runs work → slice. An earlier revision of this plan had it backwards
("`WORK` is derived and folds UP out of the slices"), which is not merely inelegant — it is CIRCULAR for a
cooperative site, where `ReducePlan.parse` cannot parse the candidate without the very inventory that candidate
determines. Choosing the inventory at the root removes the cycle, makes `derive_inventory` a validation rather than a
derivation, and aligns the enumeration order with the fork level order (`[WORK, *site keys, RASTER]`), which was
already the design.

**What the fixed inventory buys.** Three of the coupling rules between a parent and its children stop being rules at
all — they become "the child resolves against the same `work`, and a value that cannot spell against it raises, so
the combination is never built". That is exactly what `parse` and `derive_inventory` already do, so no new mechanism
carries them:

| coupling | under a fixed inventory |
| --- | --- |
| the cone statistic's `coop` must equal the parent's worker count, thread kind | `ReducePlan.parse(spec, work)` raises — the candidate is simply not in `values(site, work)` |
| flash's PV shares the QK child's warp map | both resolve against the same `work`; there is nothing to propagate |
| the chain / per-cell forms stage nothing | `work` is `""`, so no staged value resolves |

Two couplings are genuinely between a node and a candidate, and they are ordinary predicates in `_legality.py` over
`(term, candidate)` — no recursion state:

- a data-dependent gather in the epilogue refuses the warp tier (the fragment epilogue must be realizable);
- a COMPUTED `a` edge pins the parent's transport to the `sync` compute fill.

**One coupling is a genuine sibling equality**: flash's score tile must cover the parent's streaming key block
(`wn·fn·atom_n == bn`). It is stated as a `Bound(op="==")` in the twisted `Space` (see "The candidate domain"), which
is the form `domain.py` already supports — not as recursion state.

Two things stay OUTSIDE the product, deliberately:

- **Term readings** (collapse, mixed-A promotion) rewrite the tree, so they sit above it as a union of whole
  enumerations, with the uniform-key-set / decided-empty reconciliation at the top.
- **The fork LEVEL order.** The product decides the row SET; `build_fork_tree` decides the evidence hierarchy.
  Conflating them would tie the prior's prefix structure to the term's tree depth. Levels stay
  `[WORK, *site keys, RASTER]` — which the enumeration order now matches, but does not define.

The wire format does not move: rows stay flat dicts of canonical codec keys, so goldens, the tune DB,
`canonical_row_key` and the prior's key space are untouched by the whole rebuild.

### Why a flat product cannot do this

A flat product over ONE node's families covers exactly the terms with one site. Every role scheduled today is such a
term — which is why a single-site builder could serve them, and why that is not evidence the shape generalizes.

The two families beyond it are precisely the ones where **two sites must agree**: the fused cone carries a
statistic fold inside its A edge (LANDED, phase 2), and flash carries a hoisted QK operand edge plus a derived PV
contraction. Neither is a product over one node's families. Under a flat builder each needs its own bespoke emitter,
and the "one generic enumerator" claim becomes four hand-written products — the exact defect the rebuild exists to
remove. The cone half is the evidence that it does not: its rows are a `_site_values` entry, three legality predicates
and a reading, with no emitter of its own.

### The pieces

- **`path.family_sites(family, path.sites(term))`** — already written and tested. `REDUCE`/`STAGE` take every fold
  with an axis; `TILE` takes the contractions plus the root operandless zero-axis fold (the strip tier); `PLACE`
  every non-root, non-derived node. This plan does NOT invent a site walker. Two repairs it does owe: `family_sites`
  and `path._walk` must ask ONE contraction question (`_walk` tests `is_contraction`, `family_sites` tests the role —
  they diverge on exactly the split-K wrapper), and the `TILE` arm's three-clause shape match for the strip site must
  become the structural predicate that also separates the raw-loop escape (below).
- **`values(site, work, ctx) -> list[TilePlan | ReducePlan | Stage | Raster]`** — TYPED slices only, no `str` arm
  (`stage_moves` must return `Stage`, not a spelling, so nothing downstream sniffs a codec string). The catalogs in
  `search/space.py` already hand out typed slices in this currency for `TILE` and `REDUCE`.
- **`merge`** — spell each slice at its canonical path (`ops.Sched.key` already spells ANY site, so no new keys and
  no new codec) and union the child rows. This is what the deleted builder could not do: it took a fixed three-key
  tuple instead of merging what the walk produced.
- **`readings`** — the two tree rewrites; see below.

`build_fork_tree` then groups rows with levels `[WORK, *site keys, RASTER]`, which `fork.Level` supports unchanged.

### Dispatch is two predicates on the node — NOT `AxisRole`

`values(site, work, ctx)` selects its catalogs from two stored-param questions the node already answers:

```
node.axis is None    -> the register-strip values      (only at a depth-1 operandless root)
is_contraction(node) -> tile x stage x reduce values
otherwise            -> the reduce-partition values
```

**`AxisRole` does not appear in scheduling.** It looks like the natural key and is not, for three reasons the
enumeration makes plain:

- **`TWISTED` never selected anything.** The deleted dispatch table had two entries — `FREE` and `CONTRACTION` —
  with everything else falling through to the reduce emitter. This plan's own softmax example says why: the twist
  changes what the combine COSTS, not any variable's domain. The twist's one real consequence, `wn = 1` on a
  streaming site, is a legality predicate at the one site that needs it.
- **`TWISTED` is derived by matching an operation** (the combine's twist family). A scheduling decision keyed on it
  is an operation match wearing an algebraic name, which is exactly what "purely algebraic moveset — no shape
  specializations" excludes.
- **`PLANAR` is the residue** — "neither of the above", which is what `else` spells.

`AxisRole` stays what it is elsewhere: a LOOP annotation written by recognition, and a materializer read. It must
not appear in `path.py` or in any phase-1 emitter.

**The raw-loop escapes need a structural predicate, not a role fallback.** For an un-recognized loop-IR cell,
`030`'s finalize and the coop fused-tail sibling, the old code reached them through `ops.axis_role`'s `Loop.role`
fallback. That is not a role question either: `family_sites` currently admits ANY depth-1 operandless zero-axis fold
as a strip site, so it already calls the escape a strip site while `axis_role` calls it a reduce. One predicate, in
`family_sites`: a depth-1 operandless zero-axis fold whose body contains a `Loop` stamped with a reduce role is the
escape, not a strip site. Phase 1 asserts it directly.

### Term readings — the one mechanism above the product

("Reading", not "variant": `variant` is the recipe vocabulary — one concrete combination of recipe settings, the noun
behind `emmy eval variants` — and reusing it for a term rewrite collides with a live CLI concept.)

Five moves rewrite the term rather than decorating it. **The criterion that separates them is whether the rewrite
changes the SITE SET**, because that is what the product cannot absorb:

| move | changes the site set? | so it is |
| --- | --- | --- |
| strip — unroll the zero-axis fold's `lift` x r, α-rename SSA, fan out `TileOp.stores`, divide the inner extent | no — the root stays one site | a `TILE` VALUE (`r` IS the spelled `f<r>`) |
| split-K — wrap the sliced fold in an identity-lift `Fold(axis=ksplit)`, σ-reindex operands | no, ONCE the wrapper stops being a site (below) | a `REDUCE` VALUE (`cta` IS the spelled `g<cta>`) |
| the MONOID-producer composition — bind the map form's statistic + column loop as a contraction over a computed cone | yes, it ADDS two | a READING |
| collapse — splice a computed edge inline, REMOVING its schedule site | yes | a READING |
| mixed-A promotion — turn a MATERIALIZED f32 A edge into a computed cone so the sync compute-fill can convert it | yes, it ADDS one | a READING |

So the three READINGS are the site-set rewrites, and the two moves that stay values are values because their rewrite
happens at materialization from the slice the row already carries. Treating `r` and `cta` as readings would make
`_readings` a product and reintroduce the combinatorics this design removes.

The three are MUTUALLY EXCLUSIVE by shape — the composition applies to a map form whose head is a PLANAR statistic
fold, the collapse to a contraction whose `a` edge is already computed, the promotion to one whose `a` is a
materialized f32 `Load` — so a term has the base reading plus AT MOST ONE sibling, and `_readings` returns a list of
one or two. The union's key namespace is the REFERENCE reading's tree (the composition's, when it applies), consulted
before a reading's own; a rewrite keeps the site's tree POSITION, so for the other two the two spellings coincide and
the fallback is exact.

**Split-K is on the values side only after one IR repair.** `Fold.role` returns `CONTRACTION` when `composed is not
None` — an arm whose sole purpose is to classify the split-K wrapper — and `family_sites` admits any `CONTRACTION`
fold as a `TILE` site. The wrapper sits at depth 1 with an empty lift and tiles nothing, so today it becomes the
PRIMARY site and steals the bare `TILE` / `REDUCE` keys from the real contraction beneath it. Phase 1 removes the
`composed` arm and excludes composed folds from `family_sites`; the wrapper is then a rewrite with no site, which is
what makes `cta` a value.

**Three moves spell `REDUCE=g<cta>` and are NOT one move.** matmul split-K is a reassociation (legal iff the
projection distributes); flash split-KV needs an LSE combine that `finalize ∈ {kernel, atomic}` cannot spell; the
cone's redundant-statistic split RECOMPUTES the k-invariant prologue per partition. Three different rewrites under
one codec value means the prior averages three structurally different kernels under one feature row — the hazard
this section's third obligation exists to catch. Each gets its OWN predicate in `_legality.py`
(`projection_distributes` already exists; add the LSE-combinable and stat-recomputable ones), and if the row key
cannot distinguish them the fix is an `S_*` stamp, never a new knob key.

Three obligations the union carries, all of which the deleted code enforced and none of which are free:

- **Uniform key sets.** Every leaf of one fork must spell the SAME family keys, with `""` as a DECIDED empty.
  The evidence pick's prefix-consistency depends on it: an absent key reads as "free" and would let a gmem-direct
  leaf inherit a staged row's measurement. The collapsed reading lacks `TILE@dd`, so the union must stamp the
  union of both readings' keys, decided-empty where a reading lacks the site.
- **No cross-reading suppression.** A reading's rows may not depend on whether the SIBLING reading produced any —
  that has no home under a union. Where suppression is genuinely wanted it becomes a local predicate on the base
  term (*no warp tile when A's dtype ≠ the atom's a-dtype and the transport cannot convert*), evaluable without
  sibling knowledge.
- **Reading identity must survive into the prior's key space.** `build_fork_tree` keys leaves on the knob dict
  ALONE; measurement identity is `(ctx.structural_key, op_cache_key)` and distinguishes readings, but row identity is
  `(context, knobs)` and may not. Two structurally different kernels averaging under one feature row is a real
  hazard. **Check it as each reading lands** (`canonical_row_key(a) != canonical_row_key(b)` across pairs on each
  corpus shape); if they collide, the fix is an `S_*` stamp — like the existing `S_warp_eligible`, whose absence on
  materialized ops once cost a 330x fp16 misdeploy — never a new knob key. The same check is owed for the STRIP
  value, which ships today: a strip row and a contraction sub-tile row both spell `TILE=f<n>` and are different
  kernels with different cache keys.

### `WORK` is CHOSEN at the root, then validated

The inventory is the kernel-global context every site resolves against: `TilePlan.parse(spec, work)` and
`ReducePlan.parse(spec, work)` read the unit widths and the coop width OFF it, so a value never carries them. It is
therefore an INPUT to enumeration, not an output of it. `derive_inventory` remains — as the VALIDATION that the
chosen inventory is the one the resolved slices imply — and `ops.seal_workers` still stamps it, so nothing about the
stored spelling changes.

Three facts this must carry:

- **`""` is a first-class inventory** (`Workers.parse("")` is `None` — the per-cell / chain / pure-reduce tiers).
- **The tier couples the families, and the fixed inventory is what expresses the coupling.** `ReducePlan.parse`
  binds `coop` to `work.count` and RAISES unless `work.kind == "thread"`, so a tiled scalar site and a coop reduce
  are co-representable only when `par_m == 1 and par_n == coop`. Under a fixed `work` this is not a rule to enforce:
  the illegal candidate is simply not in `values(site, work)`. Consequence to state: the fused-cone case can express
  the inner stat fold's coop reduce only at a width equal to the whole inventory — a codec-imposed expressiveness
  limit, unchanged by this plan.
- **The producer band is part of the inventory and is CHOSEN, not derived.** `producer ∈ {0, 1, 2}` is a dimension of
  the root-level inventory (`w<M>x<N>+p<np>`), filtered by `_legality.producer_band`: a warp inventory, a resolved
  TMA transport at the contraction site, no cross-CTA split, `block_threads + 32·producer ≤ 1024` and
  `32·producer ≤ block_threads`. It has ONE name — `producer` — everywhere: the dimension, the `Workers` field, the
  `+p<np>` token. The retired `WSPEC` key is an env-pin ALIAS only; its dead move catalog is deleted, and phase 1
  should retire the alias too in favour of pinning `WORK=w4x2+p2` directly, which `Knob.narrow` already handles.

### The candidate domain — generated where the constraints couple, listed where they do not

**Settled, and the earlier revision of this section was half wrong.** It rejected generating the
domain from constraints on three grounds; only one survives contact with the code.

The rejected proposal was a linear system in **exponent coordinates**, and its fatal objection —
"the geometry is not a power-of-two lattice" (`f2x9`, `f4x26`, `bk = 5`) — is an objection to *that
encoding*, not to generation. Prime-exponent coordinates hold 9 = 3² and 26 = 2·13 exactly; the
corpus's prime support is `{2, 3, 5, 7, 13}`. What no coordinate change buys is **both** the
products and the budgets at once: the exponent map is a monoid iso, so `≤` becomes divisibility (a
partial order), and real logs linearize both but leave the feasible points off any lattice. ℕ⁺ is
free abelian of infinite rank under multiplication and rank 1 under addition-and-order; there is no
simultaneous linearization. So: **keep integer coordinates, keep the products multiplicative, and
enumerate** — `search/domain.py`'s `Dimension` / `Bound` / `Space`, with prefix pruning (every value
is ≥ 1, so a final product is a multiple of any partial one).

Two families have genuine multiplicative coupling and are now GENERATED from their bounds:

| family | bounds | was → is |
| --- | --- | --- |
| scalar tile | `par_n·par_m ≤ 1024` | 71 → 163 moves |
| warp tile | `32·WM·WN ≤ 1024`, `FM·FN ≤ 32` | 468 → 1140 per atom |

Everything else stays a LIST, and that is not a compromise: stage spellings, split widths, the coop
partitions, the raster orders and the strip ladder have no products to couple, so a `Space` over
them would be ceremony.

The empirical case for the old section was also thinner than it read. Of ~700 golden `TILE`
register spellings only ten are non-powers-of-two, and its headline example (`f2x9`,
`rtx4080_sm89.yaml:132`) is a flash `TILE@pj` site that
`test_golden_knobs_are_members_of_the_move_catalog` **skips** (`MatmulGoldenConfig` only) and that
the scheduler does not enumerate at all. The permanence gate covers scalar/warp matmul `TILE` and
reduce `REDUCE` — not the shapes the argument leaned on.

What DOES survive from the old section, and is now the operative constraint: the curated lists were
strict subsets of their own boxes **with no stated rule for the omissions**, which is exactly how
the sixth sweep's `f2x14`/`f4x8`/`f4x10`/`f4x26` orphaning (1.29–1.49× reachability loss) happened
unnoticed. Generation removes the failure mode; the value ladders keep the measured points as
`Dimension` values, where they are visible.

**Cost, paid once and stated plainly:** `commands/fit.py::build_golden_groups` re-enumerates to fit
`prior/offline_weights.json`, so a widened pool changes the fit's training data and makes recorded
`emmy eval offline` rank/pool columns incomparable. An `emmy fit --artifact` refit plus golden
rank / top-1/10/25/50 re-verification is owed before any GPU sweep is trusted.

### The constraint table — documentation and assertions, not a space

Legality is still worth writing down in one place: it is the checklist each emitter implements, the tier-0 assertion
list, and the review artifact that makes a dropped predicate visible. It is NOT the definition of the candidate
space (see above), and nothing solves or projects it. Instantiated for flash's two contraction sites with the
`m16n8k16` family (`atom_n = 8`, `atom_m = atom_k = 16`) and `d = 64`, f16, `depth = 2`:

| # | constraint | form | where it is checked |
| --- | --- | --- | --- |
| 1 | threads: `WM · WN · 32 ≤ max_threads_per_cta` | `≤ 1024` | emitter filter + `validate` — **after P1 restores it** |
| 2 | PV covers `d`: `WN · FN_pv · atom_n = d` | equality | construction (the PV site's factorization) |
| 3 | PV rows match QK rows | `fm_pv = fm_qk` | `assemble` — verified to hold on all 40 `TILE@pj` golden rows |
| 4 | QK stream block: `WN · FN_qk · atom_n = bn` | equality | construction (defines the kv-block) |
| 5 | `bk · atom_k` divides `dd` | divisibility | emitter filter |
| 6 | m-block covers the extent | `ceil_div` + mask | emitter filter — **NOT `≤ extent`**; 35 goldens over-cover |
| 7 | split-KV composes: `g<n>k` × `bn` within the kv extent | divisibility | `Fold` split legality |
| 8 | PV k-block coupling: `bk_pv = max(1, nt · atom_n / atom_k)` | derived from QK | `assemble` |
| 9 | smem: the staged slabs fit the cap | exact, via the stage resolvers | `_resolve_*_stage` declines; `validate` backstops |
| 10 | registers | **unmodeled** | nothing below tier 3 |
| 11 | P→A repack: `atom.c_to_a_repack`, i.e. `shape == (16, 8, 16)` | categorical | emitter filter (the QK/PV atom pair) |
| 12 | dtype gate (f32 A ⇒ no warp atom) | categorical | a `_legality` predicate, or the mixed-A reading |
| 13 | `producer` band: `wm·wn·32 + 32·producer ≤ cap` and `32·producer ≤ block_threads` | arithmetic | `_legality.producer_band` |
| 14 | `producer` eligibility: warp inventory ∧ resolved TMA ∧ no `cta` split | categorical | `_legality.producer_band` |
| 15 | **TWISTED sites are m-only: `WN = 1`** | categorical | emitter filter — the materializer has no cross-warp merge of the twisted carrier (`units=(um, 1)` is hardcoded on both flash sites) |

Row 15 is new and was missing from every previous revision. Without it a widened flash inventory generates rows that
cannot be materialized at all.

**Rows 9 and 10 are where the honesty is.** Row 9 is enforced by the STAGE RESOLVERS, which return the largest legal
`Stage` under the cap or decline — per-site, cheap, and already pinned by
`test_move_catalog.py::test_warp_staged_rows_fit_the_smem_budget`. There is no cross-node predicted-smem sum in this
plan: `KernelOp.validate` already IS `pack_smem(...) > ctx.max_dynamic_smem`, enumeration is lazy (`build_fork_tree`
defers materialization to the chosen leaf), and `Pipeline.run`'s blocklist + re-resolve retry already handles a row
that only fails when picked. Row 10 has no model anywhere in scheduling (`gpu.py` has `regs_per_block` but nothing in
enumeration or `validate` reads it), and this plan does not add one.

### Predicates: one home, one severity

`lowering/tile/_legality.py`: one function per rule, each returning the refusal REASON or `None`,
with `enforce(reason, pinned=…)` choosing the severity — an env pin raises it, the unpinned
enumeration drops the candidate. The duplicated raise/drop pairs the old scheduler carried
(`_check_warp_static_k` vs `_warp_move_ok`; `_fragment_epilogue_ok` checked once as a silent `()`
and once as a raise) are gone, and with them the "the pin says yes and the enumeration says no" bug
class.

Two corrections the rebuild forced:

- **The rules are ordinary arithmetic, not `domain.Bound`s.** An early cut routed every scalar
  divisibility test through a constructed `Bound(("_",), …)` "to state them in the same currency as
  the domain". The currency was shared in name only — `Bound` is consumed by `Space` and no legality
  rule is ever installed into one — and it cost a second file open to read `k % step == 0`, at 13
  sites, some inside a per-candidate search loop. `domain.py` is for GENERATION; `_legality` is a
  checker over term facts a `Space` cannot see.
- **The stage resolvers stay resolvers.** `resolve_warp_stage` / `resolve_scalar_stage` return the
  largest legal `Stage` or decline: the legal answer is a SIZE, not a yes/no, and this is the one
  enforcement point for the smem budget (table row 9). Merging them with the two boolean transport
  predicates would fuse a predicate with a search.

Term-reading analyses (the matvec B k-stride, the shared-row buffer, the fragment-epilogue check) stay in the
CHOICE layer: they read the TERM, not a candidate. `has_contraction_tail` and `projection_distributes` are the
exception — pure statement-shape predicates, so they live in `ir/stmt/passes.py`.

### Placement

`Placement` is a `TileOp` field, not a schedule slice, and each family constructs it at materialization from the
row's own slices — the flash warp shrink (`ceil_div(um·fm·atom_m)`, value axis dropped, `Window(parent=...)`), the
chain grid truncation, the strip's re-derived `free`. **Split-K adds no placement** — it reuses `place` verbatim,
puts `ksplit` on the fold's axis, and `030_split_reduce` owns the grid.

Grid RANK varies with the row on the chain and warp-flash families. That is fine when placement is built at
materialization. State the obligation: every placement construction is a closed-form function of (row, term).

**What a tier reads is already bound.** `Sched` carries the kernel's `Placement` and `tile_of` returns a slice with
its `(m, n)` output axes ALREADY set, so no reader states a placement rule of its own — the three hand-written
`TilePlan.at(...)` calls the materializer used to carry are gone. The pair is a function of the SITE:

| site | m | n |
| --- | --- | --- |
| root contraction | `grid[-2]` | `grid[-1]` |
| derived (flash PV, `TILE@pj`) | `free[-2]` | `free[-1]` |
| nested edge (flash QK, `TILE@dd`) | `free[-2]` | the PARENT fold's axis, through its window parent |

`TilePlan.axes` stays `compare=False` / `repr=False`, so binding cannot reach `spell()`, a stamped row, a golden or
a prior key. The depth-1 rule has ONE home — `Placement.root_mn`, read by both `TilePlan.placed_on` (binding at option
assembly)
and `Sched._mn_for` (binding for a reader that takes the slice off the tree).

### Order — the prior ranks; position is a safety net

`build_fork_tree`: *"Siblings are emitted in grouping order; ranking is the search policy's job (the online prior),
not the tree's."* Greedy resolves through goldens → reservoir → DB → prior. Ties break on `canonical_row_key`, NOT
enumeration order — `prior/base.py` is explicit: *"candidate content, never enumeration order — an order-broken tie
flips the deployed kernel per boot"*. So `flatten_leaves`' "option-0 first" docstring is not the authority.

But position DOES deploy a kernel on three live paths, and they matter more as the space widens: `greedy.py:757`
(no prior, no golden), `greedy.py:800` (every leaf blocklisted), and `pipeline.py:556` — the validate-retry budget
exhausted, re-resolving prior-free *because "the planner emits a budget-safe tile first"*.

So: **enumeration produces a SET; the only ordering obligation is that each family's FIRST value is its conservative
default.** `space.py` already states this as an invariant, and it is PER-FAMILY, not global — the reduce tier
deliberately leads with its cooperative pick, not a serial one. Mechanize it: `stamp_schedule_families(rows[0])` is
all-`off`, with the reduce exception encoded explicitly.

## Where the tree is now

**Phase 1 has LANDED: the CHOICE layer is the recursion.** `_schedule.py` is `_inventories` → `_rows_at` over the
site tree → `_merge`, with `_site_values` dispatching on the two node predicates. All three layers are live:

| layer | owns | where | state |
| --- | --- | --- | --- |
| DOMAIN | which candidate values exist at all | `search/space.py` (+ `search/domain.py` for the two coupled families) | KEPT — `stage_moves` now hands out typed `Stage`s |
| LEGALITY | what this term's K / N / dtype / smem cap refuses | `lowering/tile/_legality.py`, raise-vs-drop by `pinned` | the recursion's downward filter; `producer_transport` added |
| CHOICE | which families a site offers, each option-0, row → `TileOp` | `lowering/tile/_schedule.py` | **REBUILT recursively** |

Phase 1's acceptance set — every SINGLE-SITE term, all restored:

| role | covered |
| --- | --- |
| `FREE` | pointwise cell + the register-strip TERM VARIANT (`TILE=f<r>`) |
| `PLANAR` / `TWISTED` | the reduce partition (heuristic option-0 + coop / ILP catalog + the matvec layout gate) |
| `CONTRACTION`, materialized edges | tile × stage × reduce × raster, scalar + warp tiers, the producer band as inventory, split-K through the `Fold ⊃ Fold` composition |

**Phase 2 has LANDED too: the COMPUTED operand edge, and with it the term-reading union.** `_readings(tile, ctx)` is
the one mechanism above the product — at most two readings per term, the second whichever rewrite applies (the
MONOID-producer composition, the COLLAPSE, the mixed-A PROMOTION; mutually exclusive by shape). The union's key
namespace is the REFERENCE reading's tree (`_Term.key` — the reference first, the reading's own as the fallback, which
is exact because a rewrite keeps the site's tree POSITION), its keys are the union of both readings' with `""` as the
decided empty (`_union_keys`), and reading identity is keyed on `canonical_row_key` with a collision RAISING. The
computed-A site itself is `_contraction_values`' second inhabitant: warp tiles only (the scalar tier has no compute
fill; the reduce tiers ride the COLLAPSE reading instead), the mandatory `sync` fill resolved at `d1`/`d2`
(`_legality.resolve_sync_stage`), `computed_a_cover` for the exact-N/static-K geometry, and the redundant-statistic
split-K (`_sliced_a` σ-reindexes the cone's per-cell BODY, never its row-invariant prologue).

The one that remains — the flash streaming pair — is why the enumeration is recursive: a `_site_values` entry plus
legality, not a new emitter.

**Measured at the phase-1 commit**: `digest_kernels.py --check` green — all 24 cases byte-identical to the baseline
and all 14 schedulable cases landing their pins (from 0), the 10 `UNSCHEDULED` still dark; `make test` green with 50
registry ids deleted; the enumerated row SET identical to the pre-demolition builder's on every corpus term, option-0
included, with ONE accepted change (below). The widest term measures 132 246 rows against a 400 000 budget.

**Measured at the phase-2 commit**: `digest_kernels.py --check` green with 5 cases moved OUT of `UNSCHEDULED`
(`norm_linear`, `.lin`, `.splitk`, `.dynm`, `mlp_geglu` — all landing their pins, `.splitk` on its `__partial`) and the
other 19 cases byte-identical, including `norm_linear_coop`, whose kernel is unchanged even though its row re-keys to
`REDUCE@<stat>`; `make test` + `make lint` green with 25 registry ids deleted (20 `test_fused_edge`, 4
recognize-boundary, the golden-spelling canonical gate). Three predicates were added
(`resolve_sync_stage`, `computed_a_cover`, `warp_operand_dtype`) and one per-value inventory filter became a per-ROW
one (`_work_holds`) — a nested serial fold claims no workers, so it composes with any parent inventory, which is what
lets the cone's statistic sit under a warp parent at all.

**Of the three predicted row-key changes only the first fired.** (1) The deploy override is gone: `matvec_coopt`'s
wide-K coalesced matvec enumerated ONE row under `ctx.validate_pins` and now enumerates the full 12-row catalog with
the same `coop=32` pick leading, so the row set is a function of the term alone and the prior-free deploy is
unchanged. (2) `_row_stage` is now a `STAGE` value (`d1/sync`, a resolver — offered exactly when the shape carries a
shared row), but no corpus shape fires it, so no stored key moved. (3) `family_sites` vs `head(op)` cannot differ
yet: every corpus term is single-site, which is also why phase 1 ships the two-site FIXTURE
(`test_two_site_term_merges_both_sites_under_one_inventory`) — a contraction over a computed B edge, asserting the
merge, the non-primary key spelling (`REDUCE@j` / `STAGE@j`), the shared inventory and the per-site row-count
equation.

### The gates that hold the rebuild

| gate | what it pins | where |
| --- | --- | --- |
| kernel-source digests | every case's rendered kernel, byte for byte, against a committed baseline | `scripts/kernel_digests.txt` (27 rendered kernels over 24 cases); `digest_kernels.py --check` reports drifted / missing / unexpected separately |
| pin LIVENESS | that a case's pins actually REACHED a kernel — all of them, on one emitted op (the `__partial` under split-K) | same harness; a digest alone cannot tell a covered path from an unmapped one, since both render and both hash stably |
| the xfail registry | every scheduler casualty as an exact node id, `strict=True` — an id that starts passing FAILS until deleted | `tests/xfail_registry.py`; the file shrinking to empty IS the completion gate |

The liveness half is what makes the digest baseline mean something during a rebuild: at the demolition it reports
**0 of 23** cases landing their pins, and each role that returns moves that number. The cases that stay dark
are recorded in a STRICT `UNSCHEDULED` set, so each phase reports its own completion.

**A Hopper run is owed** before the merge: verification so far is sm_89, where every `d*/tma*` row declines, leaving
the staged-TMA tiers unexercised.

### What already exists — do not rebuild it

| the plan does NOT need to build | it already exists |
| --- | --- |
| a row-set oracle | `test_move_catalog.py::test_schedule_leaf_set_equals_catalog` — per-family set equality **plus a row-count equation** |
| a golden reachability test | `test_golden_configs.py::test_golden_knobs_are_members_of_the_move_catalog` — **passes today** (matmul + reduce goldens only; flash `TILE@dd`/`TILE@pj` are NOT covered) |
| a byte-identity gate | `scripts/digest_kernels.py --check` against the committed baseline |
| an enumeration dump | `golden_eval.enumerate_graph(graph, ctx, family=)` |
| a site walker | `path.family_sites` / `path.sites` / `ops.Sched` |
| worker sealing / inventory derivation | `ops.seal_workers`, `ir/schedule.derive_inventory`, `derive_workers`, `plan_workers` |
| ~~a row → typed slices resolver~~ | **`ir/schedule.resolve_row` / `RowSlices` do NOT survive**: one `TILE`, one `STAGE`, one `REDUCE` is the fixed-arity tuple this rebuild indicts, and flash needs two `TILE` slices. Phase 1 deletes them; a row resolves per key through `Sched` against one `Workers`, and the env pins take the same path. `resolve_site_tile` survives alone |
| the scheduled-`TileOp` constructor | `ops.scheduled` (construct + key slices through `Sched` + seal) |
| the `""`-TILE ambiguity resolver | `schedule.resolve_site_tile` |
| smem footprint | `pack_smem`, `KernelOp.smem_bytes()` |
| split-K carrier legality | `ir/stmt/passes.py::projection_distributes` + `030_split_reduce`'s own gates |
| pin narrowing / no-match-keeps-full-list | `Knob.narrow`, `pin_key_matches`, `family_value` |
| row dedup / tie-break | `knob.canonical_row_key` |
| the recording view | `knob.stamp_schedule_families` |
| the loop → term parser | `passes/lowering/tile/_fromloop.py::fold_from_loop` / `nodify_reduce` |
| the `(m, n)` binding | `ops.Sched.tile_of` returns a PLACED slice |

## Gates

| gate | catches | does not catch |
| --- | --- | --- |
| `test_schedule_leaf_set_equals_catalog` + its 3 siblings (restore from xfail) | a family's value domain or row count changing | cross-family composition on non-matmul shapes |
| `test_golden_knobs_are_members_of_the_move_catalog` (live today) | a golden becoming unreachable | non-golden losses |
| per-key value-domain snapshot (new, ~80 lines) | which family lost which value, localized | ranking quality |
| `xfail_registry` shrink (107 ids) | a restored behavior regressing | the 605 CPU skips; GPU-only; silent XPASS under `strict=False` |
| `digest_kernels.py` vs a committed baseline (+ its per-case pin liveness) | a pinned/golden row materializing differently; a case whose pins stop reaching a kernel, or an `UNSCHEDULED` one that starts | the 5 remaining `UNSCHEDULED` (flash) cases' materialization, until phase 3 lands and they leave the set |
| eval-golden MATCH sweep (GPU) | the deployed pick drifting | non-golden shapes |
| option-0 assertion (`stamp_schedule_families(rows[0])` all-`off`, per-family) | the safety-net pick becoming non-degenerate | — |

**The gate is set equality plus a conservative option-0**, not ordered-row equality — ranking is the prior's job and
no ordered baseline survives. Row order is re-derived per family and documented, not preserved.

**Snapshot instead of a system dump.** For each corpus shape check in `{codec key → sorted set of spelled values}`
plus the row count per key. It is in the STORED spelling (so it joins goldens and DB rows directly), it is what
`test_move_catalog` already asserts by hand, and it doubles as the row-count oracle. Copy `search/data/freeze.py`'s
checked-in-YAML + manifest + drift-detection pattern; there is no other snapshot infrastructure in `tests/`.
Completeness then upgrades from "the golden row appears somewhere" to **"for each golden key, its value is a member
of that key's domain"** — ~3000 per-key assertions instead of 747 set-membership ones, each localizing the loss.

## Soundness tests

Two directions. SOUNDNESS: every emitted row is materializable and legal — and it matters more here than usual
because `validate` checks smem ONLY (the thread and CTA checks are documented but "pending rebuild" — they walked
the demolished tile-flavor wrappers; `ctx.max_threads_per_cta` still exists as a field, read by nothing).
COMPLETENESS: every row that should exist is emitted — covered by
the two catalog tests above.

Costs are MEASURED, not guessed:

| tier | what | measured cost | when |
| --- | --- | --- | --- |
| 0 | codec round-trip through `resolve_site_tile`, `assemble` invariants, table rows 1–8/11–15 as assertions, row counts | **~0.7 s per shape** for the round-trip alone | every `make test` |
| 1 | materialize → `validate` → render, seeded stratified sample | **~65 ms/row** non-flash, **~1.3 s/row** flash | every `make test`, N tuned to budget |
| 2 | tier 1 exhaustively, small shapes only | thousands of rows × 65 ms = 1–3 min/shape | nightly / `-m slow`; **exclude flash or cap it** |
| 3 | `nvcc` + execute, stratified ~30 rows/kernel | seconds/row + GPU | pre-merge, manual |

Tier 1 rules that make it a real gate: seed derived from the run id, PRINTED and re-injectable (do not let
`pytest-randomly` own it); STRATIFIED by tier with the extremes force-included deterministically (min/max tile
widths, max depth, `+p` rows, `g8k/coop-t` composites, both term readings); and **failures promoted** to a frozen
regression list that runs exhaustively thereafter, so the gate strengthens monotonically.

Rows 9–10 must NOT be asserted in tier 0 — row 9 is enforced by the resolvers and row 10 by nothing.

## Migration — the clean reimplementation

`_schedule.py` was DELETED and rebuilt recursively; phases 1 and 2 have landed. A term the enumeration cannot schedule
still yields no rows and stays unmapped — the guardrail contract, so kernels compile on the materializer's per-cell path
and what fails is coverage, never a compile. The progress meter: `digest_kernels.py`'s **live-pin count** (0 of 24 at
the demolition, 14 after phase 1, 19 of 24 now — the remaining 5 are the flash cases phase 3 owns) and the **registry
size** (241 registered xfails at the demolition, 198 after phase 1, 173 now — 25 deleted by phase 2).

| phase | scope | gate |
| --- | --- | --- |
| ~~**1**~~ | **LANDED — the enumerator, the target design, built once.** `_inventories(term)` at the root, then `_rows_at(site, work)` as a product over the site tree; `_merge` spelling every slice through `ops.Sched.key`; `_site_values(site, work)` returning TYPED slices from the `search/space.py` catalogs; `_legality.py`'s predicates as the `(term, candidate)` filter. Dispatch is the two node predicates. The four IR repairs landed with it: one contraction predicate shared by `path._walk` and `family_sites`; the `composed` arm out of `Fold.role`; the raw-loop escape separated structurally in `family_sites`; `stage_moves` returning `Stage`. `resolve_row` / `RowSlices` deleted and the `WSPEC` pin alias retired | **MET**: byte-identical kernels on all 24 digest cases, 14 of 14 schedulable cases live, `make test` + `make lint` green, 50 registry ids deleted, the row set identical but for the one accepted change above |
| ~~**2**~~ | **LANDED — the COMPUTED operand edge, plus the term-reading union it needed.** `_readings` above the product (the monoid composition / the collapse / the mixed-A promotion, ≤ 2 per term, one reference key namespace, identity keyed on `canonical_row_key`); `_contraction_values`' computed-`a` inhabitant (warp-only, the mandatory `resolve_sync_stage` at `d1`/`d2`, `computed_a_cover`, the redundant-statistic split through `_sliced_a`); `_fill_realized` for the one nested site the parent form already partitions; the per-value inventory filter became the per-ROW `_work_holds`. `Fold.demoted()` returned; `warp_operand_dtype` made the copy transports' dtype rule a checked predicate | **MET**: 5 digest cases (`norm_linear` ×4 + `mlp_geglu`) left `UNSCHEDULED` landing their pins, the other 19 byte-identical, `make test` + `make lint` green, 25 registry ids deleted (20 fused-edge + 4 recognize-boundary + the golden-spelling gate) |
| 3 | TWISTED — the flash streaming pair: streaming / chain / per-cell / split-KV over a hoisted QK operand edge and a derived PV contraction. Two sites that must agree, which is what the recursion is for; adds `twisted_warp_moves`' geometry as a `values` entry | 40 attention-pin ids + 4 attention-coverage ids; the 5 flash digest cases leave `UNSCHEDULED` |
| 4 | `schedule()`'s own dispatch and flash-form selection once 2–3 exist; `kernel/_twist.py`'s `qk.acc` (reads a field `TilePlan` does not have — unreachable until the flash warp realizer runs, so it has no obtainable baseline until then) | registry EMPTY + `make test` + `make lint` |

**Phase 1 must land with no new BEHAVIOUR — but its row KEYS provably change in three places, and pretending
otherwise would get the gate argued away at merge.** The three are known and small; each is either accepted with its
corpus shape named, or fixed:

1. **The deploy override.** The old reduce catalog returned a SINGLE hardcoded row (`coop=32`) when
   `ctx.validate_pins` was set and the shape matched a wide-K coalesced matvec — so the row SET was a function of a
   context flag, which no `values()` signature should express. It is a RANKING decision; move it to the prior-free
   fallback and the enumeration becomes flag-independent.
2. **`_row_stage`.** The shared-row slab was a `Stage` slice with NO knob — fired implicitly whenever a cooperative
   partition met a contraction tail. Promoting it to a `STAGE` value (which the design requires: every decision is a
   row) changes the stored key set on the shapes where it fires. Name them: `lm_head`'s coop-cap lift.
3. **`family_sites` vs `head(op)`.** `REDUCE`/`STAGE` sites are every fold with an axis; the old key spelling took
   the primary node alone. On a term with a nested fold the walk emits keys the flat builder never did, and the
   uniform-key obligation then forces a decided-empty into every sibling row.

Everything else is identity: same values, same spellings, same kernels. Any row phase 1 emits beyond the three above
is a bug, not a bonus; any row it drops is a regression the row-count equation catches. Phases 2 and 3 then add
`values` entries and legality predicates to a structure that is already proven, instead of adding a third and fourth
hand-written product.

**#2 and #3 both fired in phase 2, on the fused shapes, and #2 cost a deploy.** The fused term is where the nested-fold
walk is real: every reading of a norm→linear / geglu term spells `REDUCE@<stat>` + `STAGE@<stat>` beside the bare
contraction keys, so the map reading's coop row moved off the bare `REDUCE` it used to carry (kernels unchanged —
`norm_linear_coop`'s digest is byte-identical). The promoted `STAGE@<stat>` then changed what the cold pick ranks: on
`test_fused_prologue_compiles_in_budget[rmsnorm_linear_n4096]` the shared-row lift is enumerated and materializes (pin
the row and it passes) but no longer DEPLOYS, because the cold prior ties and a tie breaks on `canonical_row_key`,
where the serial row's empty values sort ahead of `coop`. That case is xfailed strict with the cause named; the fix is
evidence — the owed `emmy fit --artifact` refit — not an enumeration change.

**Phase 1 also ships a two-site FIXTURE.** The single-site corpus cannot exercise the product, so the structure would
otherwise land untested and only be discovered wrong in phase 2. A synthetic two-site term with asserted rows is the
cheapest way to prove the merge, the key spelling at a non-primary site, and the shared-inventory resolution before
a production term depends on them. It paid: phase 2's production two-site term needed no structural change to the
walk, and the fixture's own row-count equation is what named the one thing that did move — the per-value inventory
filter became a per-ROW one, so the equation is now the pairs that PASS `_work_holds`, times the rasters.

The term-reading union this plan specifies landed with phase 2 (`_readings` — the monoid composition, the collapse and
the mixed-A promotion), and all three obligations are mechanized rather than checked by hand: `_union_keys` stamps the
union of both readings' keys with `""` as the DECIDED empty; no gate reads a sibling reading (the dtype rule that used
to suppress the base's warp rows is now `_legality.warp_operand_dtype`, a local predicate on the term's own `a` edge);
and reading identity is the `canonical_row_key` the row → reading map is keyed on, so a collision RAISES instead of
averaging two kernels under one feature row. Phase 3 inherits it.

**Watch the row count — with a number, not a hope.** The product across sites is now GENERATED where the flat
builder structurally avoided it. The live catalogs are already large on ONE site (163 scalar tiles, 1140 warp tiles
per atom, 10 stages, 19 coop reduces, 6 split-Ks, 2 rasters), so a two-site term multiplies rather than adds, and the
fixed inventory prunes by a smaller factor than the second site multiplies by.

Two obligations, both in phase 1: a **row budget per kernel** — a stated number, enforced in the enumerator with a
LOUD failure, not a silent truncation — and the row-count equation extended to a PER-SITE form and asserted against
the two-site fixture. `test_move_catalog` today runs one f32 static matmul, so it never sees a warp tile, a second
site, or a term reading; extending it "before there is anything multi-site to count" is not a gate, which is why the
fixture ships in the same phase.

**Before any GPU sweep is trusted: refit the offline prior.** `commands/fit.py::build_golden_groups`
reconstructs each golden's candidate pool by RE-ENUMERATING (`enumerate_graph`) and fits
`prior/offline_weights.json` against it, so a changed enumeration silently changes the fit's
training data and `emmy eval offline`'s rank/pool columns are measured against the new pool — a rank
"improvement" can be an artifact of a shrunken pool. **This is now OWED, not pending**: generating
the two tile domains widened them (71 → 163 scalar, 468 → 1140 warp per atom). Run `emmy fit
--artifact` and re-verify golden rank + top-1/10/25/50.

Merge gate (GPU): `make bench-kernels`, a flash/attention compile + tune probe, the eval-golden MATCH
sweep. A **Hopper** run is specifically owed — the paired verification ran on sm_89, where every
`d*/tma*` row declines, so the staged-TMA tiers are unexercised.

### Deferred to phase 1, deliberately — no gate exists for them yet

Two repairs the panel review identified are correct and are NOT being done during the demolition window, because the
digest baseline records the SCHEDULED kernels and the tree currently renders unscheduled ones. There is no live
byte-identity gate, and a coordinate-ordering change without one is how a silent m/n swap ships:

- **`TilePlan.units` / `regs` store `(m, n)` in the warp tier and `(n, m)` in the scalar tier**, with four
  accessors existing only to undo the flip. Store `(m, n)` always and reorder inside `spell()`; the stored spellings
  are unchanged, so the digests are the gate. Then the "codec reminder" about reading orders deletes itself — it is
  documentation of a trap, not a design.
- **The matvec layout gate is stated twice** — an index-shape classifier on the reduce path and `node.b_trans` on
  the contraction path, one fact with two predicates and opposite answers available. One predicate in `_legality.py`
  over `(node, plan)`.

## Invariants

- **Every role emits rows; no role builds `TileOp`s directly.** This is the success criterion.
- **ONE stored node kind.** Role, the A/B edge labels and the per-operand lifts are all DERIVED; nothing that a
  reading can produce gets a field. A new stored field on `Fold` is a design regression, not an optimization.
- **Knob KEYS and VALUE spellings are frozen** — which means `path._walk` must keep emitting the `a` / `b` edge
  labels. Term keys are NOT frozen; the collapse already changed them once, by design.
- **Uniform key sets per fork**, `""` as a decided empty — prefix-consistency for the evidence pick.
- **Bare `REDUCE` is the contraction's K fold, never the cone's stat** — both readings of a fused term spell against
  the contraction tree. (The earlier proposal to let each reading spell its own keys is dropped: it would change
  what stored bare keys mean.)
- **Ranking is the prior's; enumeration produces a SET** plus a per-family conservative first value.
- **Terms are never mutated in place** — readings are explicit, each with its own `term_key`, site set and
  `op_cache_key`, and their identity must survive into the prior's key space.
- **`WORK` is derived from the slices**, and leads the fork levels; `RASTER` closes and stays CONTRACTION-SCOPED
  (`test_raster_fork_offers_both_orders` and `test_raster_symbolic_grid_stays_flat` already pin this).
- **The bare-`TILE` dynamic-attention pin any-of** stays until symbolic keyed resolution exists. Note the 20 dynamic
  attention rows record a DIFFERENT key set (`STAGE`, `TILE`, `WORK` — no `TILE@dd`, no `TILE@pj`), so the
  completeness gate is subset matching for them, not key-set equality.
- **A predicate has one home and one severity**, raise-vs-drop chosen by `pinned`.

## Interfaces this plan commits to

Previously unstated and each otherwise discovered mid-phase:

- `lowering/tile/_schedule.py::schedule(tile, name, knobs, ctx, reduce_key) -> Fork | list[TileOp] | TileOp` — the
  pre-deletion signature; `020_schedule` had no other caller.
- Levels are `fork.Level`s, `[WORK, *site keys, RASTER]`; a `Level.key` returning `()` is HOW readings with
  different key sets interleave as siblings. That mechanism is already implemented.
- **Empty enumeration returns `[]`, never raises** — "the guardrail contract". Tier 0 asserts every corpus term
  yields ≥ 1 row except the documented computed-A-no-legal-warp case.
- **Split-K re-entry**: `030_split_reduce` produces a `__partial` kernel that re-enters as its own `TileOp` and must
  itself be enumerable, without further splitting. Its `__partial` kernels ARE in the digest baseline (the matmul
  and norm_linear split cases each render one).

## Worked examples

The acceptance corpus: for each shape the new walk must enumerate the SAME rows, in the same stored spellings.

**Every term dump below is REAL** — produced from `scripts/digest_kernels.py`'s case list run through
`Pipeline.build(CUDA_PASSES[:index("lowering/kernel")])` and printed with `TileOp.pretty_body()`, at HEAD. They are
therefore *pre-schedule*: `place` reads `unmapped`, there is no `work` line and no schedule slices,
because the scheduler is deleted. The `←` annotations are the axis → grid/warp/thread mapping the pre-deletion
dumps carried — i.e. what the emitters must produce, not what the tree emits today.

Each example leads with the **loop IR** the tile term is recognized FROM — the same case builders run through
`Pipeline.build(LOOP_PASSES)` (i.e. one stage earlier, after `loop/stamp` and before `lowering/tile`) and printed
with `LoopOp.pretty_body()`. That view is the scheduler-free ground truth: a plain affine nest with `acc <- ⊕(…)`
accumulator statements and gmem `load`/store, no axis roles, no operand edges, no monoid. Reading the pair is what
makes the collapse legible — recognition's whole job is to turn that nest into ONE node kind, and every `←` mapping
on the tile view is a decision the loop form does not yet contain.

Codec reminders for reading the annotations: thread `WORK` is `t<N>x<M>` and scalar `TILE` is `f<fn>[x<fm>]` — both
**n-then-m**; warp `WORK` `w<M>x<N>` and warp `TILE` `<atom>/f<FM>x<FN>` are m-then-n. `operand[a]` / `operand[b]`
are the bilinear reading's edge labels — the `a` / `b` path segments `PLACE@a` is keyed against.

**Each example carries a SITE TABLE** — the recursion's job for that shape, stated as
`site → variables → constraints`. Read it as the contract phase 1 must satisfy: one row per site the walk visits,
the free VARIABLES that site offers, and what bounds them (`↓` = inherited from the parent, `↑` = folded up to the
kernel, otherwise site-local legality).

The variables are named as they are registered, so a table row reads directly against the code:

| variables | what they geometrize | registered as |
| --- | --- | --- |
| `par_n`, `par_m`, `reg_n`, `reg_m` | the scalar output tile | `_SCALAR_TILE_SPACE` — a `Space` in `search/space.py`, generated from `Bound(("par_n","par_m"), 1024)` |
| `wm`, `wn`, `fm`, `fn`, `bk` | the warp output tile | `_WARP_TILE_SPACE` — generated from `Bound(("wm","wn"), 1024, coeff=32)` and `Bound(("fm","fn"), 32)` |
| `atom` | the mma atom kind | CATEGORICAL — `ATOM_REGISTRY`, filtered by operand dtype. Not a `Space` dimension: `domain.py` knows integers and products only |
| `depth`, `transport`, `ring`, `reg_depth`, `alt` | the operand pipeline | the `Stage` fields, spelled `d<depth>/<transport>[/ring][/alt][/p<reg_depth>]`; LISTED (`stage_moves`) — nothing multiplicative couples them |
| `cta`, `coop`, `reg`, `finalize`, `coop_transposed` | the reduce partition | the `ReducePlan.of` fields, spelled `g<cta>[a\|k]` / `coop[-t]` / `r<reg>` — the WIDTH lives in `WORK`, and `ReducePlan.parse` RAISES on the retired `b<n>` grammar; LISTED (`splitk_moves`, `coop_reduce_moves`) |
| `group`, `orient` | the launch order | the `Raster` fields, spelled `g<orient><group>`; LISTED (`raster_moves`) |
| the flash geometry | — | NOT its own vocabulary: `twisted_warp_moves`' `(warps_m, key_atoms, q_tiles)` ARE `(wm, fn, fm)`. Phase 3 declares a second `Space` over the same dimension names with the twisted bounds |
| `units`, `producer` | the `WORK` inventory | ENUMERATED FIRST, at the root — every site resolves against it (`parse(spec, work)`); `derive_inventory` then VALIDATES |

A LISTED family is not a lesser one: `domain.py`'s scope is integer dimensions coupled by PRODUCTS, so a family whose
values are a hand-kept ladder stays a list until a real multiplicative bound appears. Phase 1 registers nothing new.

The rule the tables make concrete, and the one reason the walk must recurse:

- **A MATERIALIZED operand is not a site.** It is a gmem `Load`, so there is nothing below to schedule and its
  transport is enumerated AT THE PARENT — the parent's `STAGE` family (`d<depth>` / `sync` | `cp` | `tma` / `ring`)
  covers every operand it stages.
- **A COMPUTED operand IS a site.** It is a node, so it enumerates its OWN families (its reduce partition, its
  register geometry), and the parent's `STAGE` collapses to the compute-fill it can actually use. Transport for
  whatever *that* subtree reads is then enumerated one level further down, at ITS sites.

### 1. Bare reduction — one axis, no edges, PLANAR

```python
torch.randn(N, 4096).sum(dim=-1)
```

Loop IR:

```
    for a0 in 0..64                                   ← the free axis is just an outer loop here
        for a1 in 0..4096
            in0 = load x[a0, a1]
            acc0 <- add(acc0, in0)                    ← the accumulator stmt IS the whole algebra: init, lift and
        y[a0, 0] = acc0                                  combine are all implicit in `acc0 <- add(acc0, in0)`
```

Tile IR:

```
    place  free=(a0)  unmapped                        ← a0 → blockIdx: one CTA per output cell
    Fold  free                                        ← the ZERO-AXIS node (what `Map` was): its lift IS
    ├─ operand[0]: Fold[a1 in 0..4096] planar            the projection — here the identity
    │  ├─ init: (0)                                   ← a1 → lane ⊗ serial under REDUCE=coop, then the
    │  ├─ lift: λ(a1) -> (in0)                           cross-lane tree combine
    │  │    in0 = load x[a0, a1]                      ← INLINE, not an edge: the collapse hoists nothing
    │  └─ combine: λ(acc0, acc0__o) -> (acc0)
    │       acc0 = add(acc0, acc0__o)                 ← the SAME λ is the cross-lane merge: carrier-generic
    └─ lift: λ(acc0) -> (acc0)
    stores
    └─ y[a0, 0] = acc0
```

No operands, so the arity table lands the inner node on PLANAR: serial option-0 (past the free-cell cap the
heuristic stays scalar), then the coop catalog — each `t<n>` row's worker demand IS its `WORK` inventory, nothing to
unify against — then guarded `g<n>` / `r<n>`.

Note the bare reduce is ALREADY wrapped by an identity zero-axis node. That is not new: it is what the projection
wrapper always did, and it means the FREE and PLANAR emitters compose at depth 0/1 on even the simplest shape.

| site | enumerate | constraints |
| --- | --- | --- |
| root `Fold free` (depth 0) | — (it has an operand, so no strip `TILE`); owns the free-axis → grid mapping | — |
| `Fold[a1] planar` — bare `REDUCE` | `coop ∈ {4,8,…,512}`, `reg ∈ {2,4}`, `cta ∈ {2,4,8}`, `finalize ∈ {kernel, atomic}`, `coop_transposed ∈ {F,T}` — serial (all 1) is option-0 | `coop` a power of two ≤ the CTA cap; `cta` needs a STATIC extent and divides it; `finalize=atomic` only when the projection distributes; `coop_transposed` needs a k-major B and a 32-divisible inner free axis. `↑` `coop` IS the kernel's `WORK` inventory |
| same fold — bare `STAGE` | nothing to stage: the operand is INLINE in the lift, not an edge | decided empty |
| `RASTER` | — | CONTRACTION-scoped; a pure reduce offers no rows |


### 2. RMSNorm — the same two depths, a real projection

```python
torch.nn.functional.rms_norm(x, (4096,), wn)
```

Loop IR:

```
    in0 = load y_mean_count[0]                        ← the row-invariant prologue, hoisted above the nest
    v0 = reciprocal(in0)
    in1 = load y_eps[0]
    for a0 in 0..64
        for a1 in 0..4096
            in2 = load x[a0, a1]
            v1 = multiply(in2, in2)
            acc0 <- add(acc0, v1)
        v2 = multiply(acc0, v0)                       ← the between-loops straight-line block is what becomes the
        v3 = add(in1, v2)                                OUTER lift; recognition splices the prologue into it
        v4 = rsqrt(v3)
        for a2 in 0..4096                             ← the SECOND a-axis loop over the same extent: a sweep, not
            in3 = load x[a0, a2]                         a reduce, so it becomes `stores`' sweep, not a fold
            v5 = multiply(in3, v4)
            in4 = load wn[a2]
            v6 = multiply(in4, v5)
            y[a0, a2] = v6
```

Tile IR:

```
    place  free=(a0)  unmapped                        ← one CTA per row
    Fold  free
    ├─ operand[0]: Fold[a1 in 0..4096] planar   ‹computed›
    │  ├─ init: (0)                                   ← a1 → 128 lanes × 32 serial + tree combine
    │  ├─ lift: λ(a1) -> (v1)
    │  │    in2 = load x[a0, a1]                      ← coalesced band read of the row
    │  │    v1 = multiply(in2, in2)
    │  └─ combine: λ(acc0, acc0__o) -> (acc0)
    │       acc0 = add(acc0, acc0__o)
    └─ lift: λ(acc0) -> (v6)                          ← runs once per OUTPUT cell of the sweep, on its owning lane
         in0 = load y_mean_count[0]
         v0 = reciprocal(in0)
         in1 = load y_eps[0]
         v2 = multiply(acc0, v0)
         v3 = add(in1, v2)
         v4 = rsqrt(v3)
         in3 = load x[a0, a2]                         ← the TWICE-READ edge (also read in the inner lift): the
         v5 = multiply(in3, v4)                          shared-row stage move's site — its benefit gate (a
         in4 = load wn[a2]                               contraction tail) declines here, so it stays gmem-direct
         v6 = multiply(in4, v5)
    stores
    └─ sweep(a2) y[a0, a2] = v6                       ← a2 → the SAME lanes: each writes 4096/128 = 32 cells
```

Structurally identical to example 1 — only the outer lift has a body. The two nodes are the same kind at two
depths, dispatched by `ops.axis_role` (FREE outer, PLANAR inner) with no wrapper type in between.

| site | enumerate | constraints |
| --- | --- | --- |
| root `Fold free` | — (has an operand); its lift is the per-cell normalize, its `stores` the sweep | the sweep's lane mapping follows the inner fold's `↑` inventory — one CTA, one row |
| `Fold[a1] planar` — bare `REDUCE` | `coop`, `reg`, `cta`, `finalize`, `coop_transposed` — as example 1 | as example 1 |
| same fold — bare `STAGE` | the SHARED-ROW slab: the input read in BOTH the inner lift and the outer sweep | offered only over a cooperative partition AND a contraction tail (`has_contraction_tail`); declines here, so decided empty |


### 3. Softmax — the same shape again, twisted combine

```python
torch.nn.functional.softmax(x, dim=-1)
```

Loop IR:

```
    for a0 in 0..64
        for a1 in 0..4096                             ← TWO sequential reduce loops over the same axis: the max
            in0 = load x[a0, a1]                         pass…
            acc0 <- maximum(acc0, in0)
        for a1 in 0..4096                             ← …and the sum pass that DEPENDS on it. `Fold.from_loop`'s
            in1 = load x[a0, a1]                         twisted merge regenerates the fused (m, l) combine from
            v0 = subtract(in1, acc0)                     these two and byte-compares — that merge, not loop
            v1 = exp(v0)                                 fusion, is what makes the single node below
            acc1 <- add(acc1, v1)
        v2 = reciprocal(acc1)
        for a2 in 0..4096
            in2 = load x[a0, a2]
            v3 = subtract(in2, acc0)
            v4 = exp(v3)
            v5 = multiply(v2, v4)
            y[a0, a2] = v5
```

Tile IR:

```
    place  free=(a0)  unmapped
    Fold  free
    ├─ operand[0]: Fold[a1 in 0..4096] twisted   ‹computed›
    │  ├─ init: (-inf, 0)                             ← each lane's running state is the PAIR (m, l) — the
    │  ├─ lift: λ(a1) -> (acc0__osin, 1)                 monoid ARITY is the only difference from example 1
    │  │    acc0__osin = load x[a0, a1]
    │  └─ combine: λ(acc0, acc1, acc0__o, acc1__o) -> (acc0, acc1)
    │       acc0__o__t0 = maximum(acc0, acc0__o)      ← the cross-lane tree merge runs THIS λ on (m, l) pairs —
    │       acc0__o__t1 = subtract(acc0, acc0__o__t0)    the twisted rescale, same code as the serial step
    │       …
    │       acc1 = add(acc0__o__t3, acc0__o__t6)
    │       acc0 = copy(acc0__o__t0)
    └─ lift: λ(acc0, acc1) -> (v5)                    ← BOTH components bind as params (positional binding over
         v2 = reciprocal(acc1)                           the edge's result components)
         …
    stores
    └─ sweep(a2) y[a0, a2] = v5
```

Twisted changes what the combine COSTS, never which fold moves are legal, so no new path exists for softmax at all.
Post-collapse the claim "same node kind as example 1, only the monoid arity differs" is literally true of the
storage, not just of the algebra.

| site | enumerate | constraints |
| --- | --- | --- |
| root `Fold free` | — | — |
| `Fold[a1] twisted` — bare `REDUCE` | `coop`, `reg`, `cta`, `finalize`, `coop_transposed` — the SAME variables as examples 1–2 | identical legality; the twist changes the combine's COST, not any variable's domain — there is no twisted-only dimension |
| same fold — bare `STAGE` | shared-row slab | declines (no contraction tail) |


### 4. Matmul — one axis, two edges, CONTRACTION

```python
torch.randn(512, 512) @ torch.randn(512, 512)
```

Loop IR:

```
    for a0 in 0..512                                  ← nothing here says a0 is M and a1 is N — they are two
        for a1 in 0..512                                 free loops, symmetric until the PLACEMENT names them
            for a2 in 0..512
                in0 = load w[a2, a1]                  ← the two loads become operand[b] / operand[a], in THIS
                in1 = load x[a0, a2]                     stored order — the order is all the A/B label is
                v0 = multiply(in0, in1)
                acc0 <- add(acc0, v0)                 ← multiply-then-accumulate: the bilinear reading recognition
            y[a0, a1] = acc0                             matches to build the contraction
```

Tile IR:

```
    place  free=(a0, a1)  unmapped          ← block-tiled: a0 (m) rows × a1 (n) cols per CTA
    Fold[a2 in 0..512] contraction          ← a2 (K) → SERIAL per thread, chunked by the staging ring
    ├─ operand[a]: in1 = load x[a0, a2]   ‹materialized›   ← A: M-resident, staged slab → smem per ring step
    ├─ operand[b]: in0 = load w[a2, a1]   ‹materialized›   ← B: the streamed K×N slab, double-buffered
    ├─ init: (0)
    ├─ lift: λ(a2, in0, in1) -> (acc0__v)   ← params bind the operands POSITIONALLY, in stored order (b, a)
    │    acc0__v = multiply(in0, in1)       ← the JOINT step: a function of BOTH operands, so it is neither
    └─ combine: λ(acc0, acc0__o) -> (acc0)     operand's per-operand lift. This is why `lift` stays ONE lambda
         acc0 = add(acc0, acc0__o)
```

The role is read off the shape — two operands, a `multiply` lift over distinct params, an `add` combine — and the
A/B labels off the stored operand ORDER `(b, a)`. **Note what the dump makes plain:** A/B is *not* derived from the
accesses. Node-locally `x[a0, a2]` and `w[a2, a1]` are symmetric — each
carries the K axis plus one free axis — and telling M from N needs the PLACEMENT, which is a caller fact on the
`TileOp` and deliberately absent here. Order is what `Fold.contraction(...)` generates and what the gate pins.

Depth-0 is the tile × stage × reduce × raster product. The warp sibling family maps the same axes differently:
`WORK=w<M>x<N>` puts
32-lane warps on an (m, n) warp grid, `TILE=<atom>/f<FM>x<FN>[/k<bk>]` gives each warp an
(FM·atom_m) × (FN·atom_n) fragment tile, and a2 advances in `atom_k`-element mma steps, `bk` per smem stage.

| site | enumerate | constraints |
| --- | --- | --- |
| root `Fold[a2] contraction` — bare `TILE` | SCALAR `par_n ∈ {16,32,64}`, `par_m ∈ {8,16}`, `reg_n ∈ {1,2,4}`, `reg_m ∈ {1,2,4,6,8,10,12,14,26}` — OR WARP `atom`, `wm`, `wn ∈ {1,2,4,8,16}`, `fm`, `fn ∈ {1,2,4,8}`, `bk ∈ {1,2,4,8}` | `par_n·par_m ≤ 1024`; `32·wm·wn ≤ 1024`; `fm·fn ≤ 32`; `atom`'s operand dtype must match A/B; `bk·atom.atom_k` divides a static K. `↑` `(par_n, par_m)` / `(wm, wn)` ARE the `WORK` inventory |
| same node — bare `STAGE` | `depth ∈ {1,2,3,4}`, `transport ∈ {sync, cp, tma}`, `ring ∈ {F,T}`, `reg_depth ∈ {1,2}` — **the transport for BOTH materialized operands**, since neither is a site | the resolver returns the largest slab fitting `ctx.max_dynamic_smem`, or declines to gmem-direct; `transport=tma` needs `ctx.has_tma`; `reg_depth=2` is `↓` warp-only |
| same node — bare `REDUCE` | `cta ∈ {2,4,8}`, `finalize ∈ {kernel, atomic}` (serial = option-0) | `cta` divides K; `finalize=atomic` only when the projection distributes; `↑` no `coop` beside a warp inventory |
| kernel-global `WORK` | `units`, `producer` — DERIVED, never chosen | `↑` folded from the slices; the combination drops if they disagree |
| root-global `RASTER` | `group ∈ {8}`, `orient = m` (flat = option-0; `orient = n` is pin-only until a shape wants it) | 2-D-tiled static grid only |
| the producer band (`WORK`'s `+p<np>`) | `producer ∈ {1, 2}` | warp `TILE` + a RESOLVED `transport=tma`, no `cta` split, `block_threads + 32·producer ≤ 1024`, `32·producer ≤ block_threads` |
| `operand[a]`, `operand[b]` | **NOT SITES** — gmem `Load`s | their transport is the parent's `STAGE`, above |


### 5. Epilogue fusion — a zero-axis fold over example 4

```python
torch.relu(x @ w + bias)
```

Loop IR:

```
    for a0 in 0..512
        for a1 in 0..512
            in0 = load bias[a1]
            for a2 in 0..512                          ← example 4's nest VERBATIM, one level deeper
                in1 = load w[a2, a1]
                in2 = load x[a0, a2]
                v0 = multiply(in1, in2)
                acc0 <- add(acc0, v0)
            v1 = add(acc0, in0)                       ← the post-reduce straight-line tail: the zero-axis fold's
            v2 = relu(v1)                                lift, exactly as in example 2
            y[a0, a1] = v2
```

Tile IR: no dump case for this shape exists in the digest battery — the term is

Example 4's node verbatim, as the one operand of a zero-axis fold whose lift is `add`-then-`relu` and whose store is
the root `Write` (structurally examples 2 and 4 composed; no separate dump case exists in the digest battery).

The enumeration is example 4's product plus one local filter at the outer node: the fragment-epilogue gather check
(a warp row folds the outer lift into the per-fragment `RegEpilogue`, so a data-dependent gather index refuses the
warp tier). The operand subtree is byte-identical to example 4's, so the candidate enumeration is identical **by
construction** — and post-collapse that means the same emitter ran, not that two emitters agreed.

| site | enumerate | constraints |
| --- | --- | --- |
| root `Fold free` (the epilogue) | — (has an operand) | its lift becomes the per-fragment `RegEpilogue`; a data-dependent gather in it `↓` REFUSES the child's warp tier — the one cross-level constraint this shape adds |
| child `Fold[a2] contraction` | example 4's variables verbatim — `par_*`/`reg_*` or `atom`,`wm`,`wn`,`fm`,`fn`,`bk`; `depth`,`transport`,`ring`,`reg_depth`; `cta`,`finalize`; `group`,`orient` | example 4's constraints, plus the `↓` epilogue filter above |


### 6. SwiGLU — the fused gate⊗up edge, where sharing IS arity

```python
F.silu(self.gate(x)) * self.up(x)  # gate/up: Linear(1024, 3072, bias=False)
```

Loop IR (the `mlp_geglu` pipeline case — a norm-fed SwiGLU, so the shared cone is the RMSNorm row):

```
    in0 = load xn_mean_count[0]
    v0 = reciprocal(in0)
    in1 = load xn_eps[0]
    in2 = load sg_one[0]
    for a0 in 0..32
        for a1 in 0..1024                             ← the statistic fold: recognition's cone SOURCE, the
            in3 = load x[0, a0, a1]                      row-invariant prologue of the shared A edge
            v1 = multiply(in3, in3)
            acc0 <- add(acc0, v1)
        v2 = multiply(acc0, v0)
        v3 = add(in1, v2)
        v4 = rsqrt(v3)
        for a2 in 0..3072
            for a3 in 0..1024                         ← ONE k loop, TWO accumulators — the arity-2 form is right
                in4 = load x[0, a0, a3]                  here in the loop IR, as two `acc <- …` stmts sharing a
                v5 = multiply(in4, v4)                   loop, not as a fusion the scheduler must discover
                in5 = load wn[a3]
                v6 = multiply(in5, v5)
                in6 = load wg[a2, a3]
                v7 = multiply(in6, v6)
                acc1 <- add(acc1, v7)
                v8 = multiply(in4, v4)                ← the cone body is spelled TWICE in the nest (v5/v6 and
                v9 = multiply(in5, v8)                   v8/v9 are the same term); one edge read by both
                in7 = load wu[a2, a3]                    channels is exactly what the node's arity expresses
                v10 = multiply(in7, v9)
                acc2 <- add(acc2, v10)
            v11 = negative(acc1)                      ← the silu/multiply tail: the wrapping zero-axis fold
            v12 = exp(v11)
            v13 = add(in2, v12)
            v14 = reciprocal(v13)
            v15 = multiply(acc1, v14)
            v16 = multiply(acc2, v15)
            o[0, a0, a2] = v16
```

Tile IR — the arity-2 form, dumped from the operand-edge unit fixture (the `mlp_geglu` pipeline case does not reach
the fused node pre-scheduler — its cone/contraction recognition lives in `_atomize`, inside the deleted arm):

```
    Fold[k in 0..256] contraction
    ├─ operand[a]: Fold  free  ‹pointwise›   ‹computed›   ← the shared A cone: the normalized/projected row,
    │  └─ lift: λ() -> (xhat)                                compute-filled per k-block into smem, then consumed
    │       xhat_e = load x[m, k]                            by BOTH channels' mma/scalar steps
    │       xhat_s = load w[k]
    │       xhat = multiply(xhat_e, xhat_s)
    ├─ operand[b0] -> acc_g: acc_g_b = load Wg[k, n]   ‹materialized›
    ├─ operand[b1] -> acc_u: acc_u_b = load Wu[k, n]   ‹materialized›
    ├─ init: (0, 0)
    ├─ lift: λ(k, acc_g_b, xhat, acc_u_b) -> (acc_g__v, acc_u__v)
    │    acc_g__v = multiply(acc_g_b, xhat)      ← ONE edge, TWO terms. No privileged operand slot, no let
    │    acc_u__v = multiply(xhat, acc_u_b)         table — the shared edge appearing in BOTH terms is
    └─ combine: λ(acc_g, acc_u, acc_g__o, acc_u__o) -> (acc_g, acc_u)   the arity (`Channel` survives only as
         acc_g = add(acc_g, acc_g__o)                                     the builder's argument pair)
         acc_u = add(acc_u, acc_u__o)
```

The cone's inner stat fold answers with its own `REDUCE@<axis>` slice, from the same recursion that answers the
outer node — no hand-threaded prologue keys. The two term readings of one loop reduce to "recurse on each reading,
union the rows". `PLACE@cone` and `PLACE@a` are the segment spellings the collapse must preserve.

| site | enumerate | constraints |
| --- | --- | --- |
| root `Fold free` (silu ⊗ multiply) | — (has an operand) | the ⊗-combine defers into the finalize under split-K |
| `Fold[k] contraction` — bare `TILE` | `atom`, `wm`, `wn`, `fm`, `fn`, `bk` — the WARP variables ONLY | `par_*` / `reg_*` are not offered: the fill is a compute fill, which only the warp realizer has. Same bounds as example 4; `↑` `(wm, wn)` is the inventory BOTH channels share |
| same node — bare `STAGE` | `depth ∈ {1, 2}` only — `transport` is not free, `ring` / `reg_depth` / `alt` unoffered | `↓` a COMPUTED `a` edge PINS `transport = sync` (the compute fill); `depth=2` is the asymmetric B-only prefetch ring; multi-channel (gate⊗up) forbids `cp` / `tma` outright; `producer` unoffered — the band assumes a COPYING producer |
| same node — bare `REDUCE` | `cta ∈ {2,4,8}`, `finalize = kernel` — the redundant-statistic split | `finalize=atomic` unoffered; single-channel for the classic arm, multi-channel splits with per-channel raw `C` partials; the k-invariant prologue is RECOMPUTED per partition, so only small-free decode shapes admit it |
| `operand[a]` — the cone, `PLACE@a` | **IS a site**: the seam is real, so cut legality is spelled here | edge iff closed — structural, by construction |
| the cone's statistic fold — `REDUCE@<axis>` / `STAGE@<axis>` | its OWN `coop`, `reg`, `cta` — plus the prologue's placement (hoisted above the K loop, published by a CTA barrier) | `↓` `coop` binds to the parent's inventory: `ReducePlan.parse` requires a THREAD kind and `coop == work.count`, so beside a WARP parent the statistic's coop band is unspellable — the codec-imposed limit, unchanged by this plan |
| `operand[b0]`, `operand[b1]` | **NOT SITES** — gmem `Load`s, one per channel | plain-copy fills issued BEFORE the compute fill, under the parent's `STAGE` |
| root-global `RASTER` | `group`, `orient` | load-bearing here: B re-streams per M-tile row, so the grouped order's L2 reuse is real (`orient=n, group=8` measured −8% on the gemma gate_up edge, 5090) |


### 7. Pure pointwise — a zero-axis fold with no operands

```python
torch.relu(x)
```

Loop IR:

```
    for a0 in 0..64                                   ← no accumulator stmt anywhere: this is what "zero-axis"
        for a1 in 0..4096                                means, and both loops are free
            in0 = load x[a0, a1]
            v0 = relu(in0)
            y[a0, a1] = v0
```

Tile IR:

```
    place  free=(a0, a1)  unmapped          ← EVERY axis → grid; one THREAD per grid cell (per-cell tier —
    Fold  free  ‹pointwise›                    no work line, launch geometry derived by the materializer)
    └─ lift: λ() -> (v0)
         in0 = load x[a0, a1]               ← the xr strip value unrolls THIS body, divides the inner grid
         v0 = relu(in0)                        extent and fans out the stores; `r` is the spelled TILE value
    stores
    └─ y[a0, a1] = v0
```

**This is the shape that now collides with the loop-IR escapes** — `030`'s finalize and the un-recognized flat cell
are also zero-axis folds with no operands. They are separated only by `ops.axis_role`'s `Loop.role` fallback and
`family_sites`' root-only rule. Phase 1 must assert it directly.

| site | enumerate | constraints |
| --- | --- | --- |
| root `Fold free` (no operands, depth 1) — bare `TILE` | `reg_n ∈ {2, 4}` — the register STRIP ratio, spelled `f<reg_n>` (per-cell `""` is option-0). The one case a zero-axis fold is a `TILE` site | `reg_n` divides the inner free extent STATICALLY; the cell body must be stateless (no sweep, no carried state); `atom` / `wm` / `wn` are illegal on a pointwise cell. The ladder stops at 4 — `f8` regressed both pointwise goldens on register pressure |
| everything else | — | no axis ⇒ no `REDUCE`/`STAGE`; no operands ⇒ no child sites; `RASTER` is CONTRACTION-scoped |


### 8. Causal SDPA — one node kind at three depths

```python
F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Loop IR — TWO loop nodes, the score materialized to a whole S×S buffer between them (this is the `flash_chain`
case, hd = 64, causal mask absent to match the tile dump below; in the second node the score buffer's name is
elided to `…_scaled` for width, everything else is verbatim):

```
    in0 = load scaled_dot_product_attention_scale[0]
    for a0 in 0..4
        for a1 in 0..128
            for a2 in 0..128
                for a3 in 0..64                       ← the QK contraction, its own nest writing gmem…
                    in1 = load x1[0, a0, a2, a3]
                    in2 = load x0[0, a0, a1, a3]
                    v0 = multiply(in1, in2)
                    acc0 <- add(acc0, v0)
                v1 = multiply(acc0, in0)
                scaled_dot_product_attention_scaled[0, a0, a1, a2] = v1
```

```
    for a0 in 0..4
        for a1 in 0..128
            for a2 in 0..128                          ← …re-read THREE times by the softmax passes and the PV
                in0 = load …_scaled[0, a0, a1, a2]       loop. The S×S traffic flash exists to delete is the
                acc0 <- maximum(acc0, in0)               plain reading of this loop IR
            for a2 in 0..128
                in1 = load …_scaled[0, a0, a1, a2]
                v0 = subtract(in1, acc0)
                v1 = exp(v0)
                acc1 <- add(acc1, v1)
            v2 = reciprocal(acc1)
            for a3 in 0..64
                for a4 in 0..128                      ← the PV contraction, on the SAME axis the twisted stream
                    in2 = load …_scaled[0, a0, a1, a4]   runs on, and its P factor (v5) reads the reduction's own
                    v3 = subtract(in2, acc0)             result — which is why the fused form keeps PV BELOW the
                    v4 = exp(v3)                         seam as derived material rather than an operand edge
                    v5 = multiply(v2, v4)
                    in3 = load x2[0, a0, a4, a3]
                    v6 = multiply(in3, v5)
                    acc2 <- add(acc2, v6)
                scaled_dot_product_attention[0, a0, a1, a3] = acc2
```

Everything flash *is* — one kv stream, the score never materialized, the (m, l, O) state register-resident — is
absent from that loop IR. It is produced by recognition folding the score node back in as a hoisted operand edge
and twisted-merging the two softmax passes; the whole distance between these two views is why the twisted emitter
must be generic rather than a flash special case.

Tile IR:

```
    place  free=(b0, b1, m, d)  unmapped              ← b0·b1 (batch · heads) → blockIdx on EVERY family; how m
    Fold  free                                           and d map is what the fork decides (per family below)
    ├─ operand[0]: Fold[kv in 0..128] twisted   ‹computed›
    │  │                                          ← kv NEVER maps to the grid un-split: it is the STREAM —
    │  ├─ operand[0]: Fold[dd in 0..64] contraction   ‹computed›      serial bn-blocks per CTA; g<n>k splits it
    │  │  ├─ operand[a]: q_e = load x0[b0, b1, m, dd]   ‹materialized›   ← Q tile loads ONCE, resident across
    │  │  ├─ operand[b]: k_e = load x1[b0, b1, kv, dd]  ‹materialized›      the stream; K slab staged per
    │  │  ├─ init: (0)                                                     kv-block; dd → mma k-steps
    │  │  ├─ lift: λ(dd, k_e, q_e) -> (sacc__v)
    │  │  │    sacc__v = multiply(k_e, q_e)
    │  │  └─ combine: λ(sacc, sacc__o) -> (sacc)
    │  │       sacc = add(sacc, sacc__o)
    │  ├─ operand[1]: v_e = load x2[b0, b1, kv, d]   ‹materialized›   ← V slab staged per kv-block, consumed by
    │  ├─ init: (-inf, 0, 0)                                             the DERIVED PV contraction below the seam
    │  ├─ lift: λ(kv, sacc, v_e) -> (s, 1, v_e)   ← (m, l, O) state stays register/fragment-resident across the
    │  │    scale_c = load _flash_scale[]            whole stream — the demand the fold puts on ITSELF
    │  │    s = multiply(sacc, scale_c)
    │  └─ combine: λ(m_i, l_i, O_i, m_i__o, l_i__o, O_i__o) -> (m_i, l_i, O_i)
    │       …                                     ← the per-block rescale; on the coop family the SAME λ is the
    │                                                cross-lane merge of (m, l, O) triples
    └─ lift: λ(m_i, l_i, O_i) -> (O_i__proj)
         O_i__proj = divide(O_i, l_i)             ← once per (m, d) output cell, at stream end
    stores
    └─ scaled_dot_product_attention[0, b1, m, d] = O_i__proj
```

Flash is where the collapse pays: **zero-axis projection → twisted stream → contraction, all one node kind**, and
the derived PV is a fourth instance of the same kind rather than a synthesized member of a different class. (This
dump is the `flash_chain` case, whose causal mask is absent; the masked form adds the `s_causal = s when (kv <= m)`
select to the twisted lift.)

**There is no FORM family.** The four flash schedules below are not values of any knob — `REDUCE`'s codec is
`g<cta>[a|k] / coop[-t] / r<reg>` and the frozen-key invariant forbids adding a token. The corpus already records the
form IMPLICITLY, as a READING of the `(WORK, TILE@dd, TILE@pj)` tuple, and that is what the enumeration must produce:

| the form | is the reading of |
| --- | --- |
| warp streaming | warp `WORK` + a warp `TILE@dd` + a warp `TILE@pj` |
| chain | `WORK=""` + a scalar `TILE@pj=f<d>` register vector |
| coop | thread `WORK` + `REDUCE@<kv>=coop` + no `TILE` |
| per-cell | the COLLAPSE reading (the QK edge moves inline) + everything decided-empty |

So three of the four fall out of `inventories(term) x values(site, work)` with no flash-specific code, and the fourth
is one of the two term readings this plan already has. **Phase 3's gate is exactly that**: the same `values()` that
produces warp streaming must produce the tuple that spells chain. If it cannot, phase 3 is four hand-written emitters
and must be labelled as such rather than shipped as an enumeration.

The four forms and their axis mappings:

- **WARP streaming** (dtype-gated): grid = (b0·b1, m/bm). m → bm query rows held as
  warp mma fragments (bm = WM·FM·atom_m from `WORK` + `TILE@dd`); kv → serial stream, bn keys per step
  (bn = WN·FN·atom_n — the kv-block ↔ score-tile coupling IS the downward constraint on the QK child); dd → mma
  k-steps; d → PV fragment columns via `TILE@pj`. ONE `WORK=w<M>x<N>` shared by the
  QK child and the derived PV. Constraint-table row 15 applies: `WN = 1`.
- **CHAIN** (FA-2 scalar): grid = (b0, b1, m) — one THREAD per query row; d → a per-thread
  register vector (`TILE@pj=f64`, legal since d = 64 ≤ the register budget); kv → serial per thread, the score
  computed ONCE per key and shared across all 64 columns. `WORK=""`.
- **Per-cell**: grid = (b0, b1, m, d) — one thread per output cell; kv serial; the QK edge collapses inline (the
  collapse READING — the edges move into the lift, so the reduce-partition values apply with no rewrite anywhere),
  so the score recomputes per d — the redundant form the chain exists to beat.
- **Coop**: grid = (b0, b1, m, d), a `t<n>` band splits kv across lanes within the CTA; the cross-lane merge of
  (m, l, O) triples is the fold's own combine λ — carrier-generic, same machinery as examples 1–3.
- **Split-KV** (`g<n>k`): composes with the warp rows — kv → n CTAs × serial stream; each partial keeps fragment
  residency; `030_split_reduce` realizes the partial + LSE-combine finalize. The same `g<n>` move as matmul
  split-K's `g<cta>` VALUE, under its own legality (kv divides, slices block-whole, the LSE combine is realizable).

Pins (`TILE@dd` / `TILE@pj` / `REDUCE@<kv>`) narrow at their paths like every other kernel — no per-form routing
block, no flash-specific pin escape.

| site | enumerate | constraints |
| --- | --- | --- |
| root `Fold free` (the `divide(O, l)` projection) | — (has an operand) | — |
| kernel-global `WORK` | `units`, `producer` — CHOSEN first, as everywhere | the form is a READING of what this and the two `TILE` sites resolve to: `""` gives chain / per-cell, thread gives coop, warp gives streaming |
| `Fold[kv] twisted` — `REDUCE@<kv>` | `cta ∈ {2,4,8}`, `coop`, `finalize = kernel` | `cta` divides kv and slices block-whole; `finalize=atomic` is ILLEGAL — the twisted `e^{Δm}` rescale cannot be an atomic, and the LSE combine split-KV needs is not spellable by `finalize`, so it carries its own `_legality` predicate |
| same fold — `STAGE` | `depth ∈ {1,2,3,4}`, `transport ∈ {cp, tma}`, `ring`, `alt` for the K/V stream | slab fits smem; a `""` inventory resolves no staged value, which is what makes the chain / per-cell forms unstaged — not a form check |
| `operand` — the hoisted QK score, `TILE@dd` | **IS a site**: `atom`, `wm`, `wn`, `fm`, `fn`, `bk` of the TWISTED `Space` | `wn·fn·atom.atom_n == bn`, the parent's streaming key block — a `Bound(op="==")` in that `Space`, NOT recursion state; `wn = 1`; the fragment budget is the twisted one, not `fm·fn ≤ 32` (the corpus records `f1x64`) |
| the DERIVED PV contraction, `TILE@pj` | its own `atom` (including the f16-acc sibling), `fm`, `fn`, `bk` | `wm` / `wn` are not free — both sites resolve against the SAME chosen `work`, so there is nothing to propagate; `wn·fn·atom_n == d`; `Site.derived` ⇒ EXCLUDED from `PLACE` (it lies below the seam, so it is not a cut) |
| `Q`, `K`, `V` loads | **NOT SITES** | transport is the parent fold's `STAGE` |



### Obligations the eight examples do not cover

Not covered above, and not to be dropped: the `PLACE=cut` re-recognized pieces (18 golden rows, each its own
kernel), `rope` / `embedding` (which record `{WORK: ''}` and nothing else — a zero-choice enumeration, benign but it
must not crash), `lm_head`'s coop-cap lift, `qknorm.k256/k512`, `mlp_ch`, `mlp_geglu`, `mlp_gate_up_split`, and
every `.dynM` twin. The completeness test must also FILTER the 57 golden rows carrying non-scheduler keys
(`FAST_EXP`, `INTERLEAVE_LOADS`, `VECTORIZE_LOADS`, `LOOPIFY`, `PLACE`) — those come from `lowering/kernel` passes
and recognition, not from this enumeration.

## Risks

- **The measured evidence is orphaned**: the collapse changed `term_key`, so no DB / reservoir row and no cubin
  keyed on `(ctx.structural_key, op_cache_key)` still matches. Measurement restarts from empty; goldens are
  unaffected. Sunk, not pending — but every "regression vs recorded evidence" reading is void until refilled.
- **A path segment can be silently lost.** `_SEGMENT_TOKENS` is `{"map", "fold", "a", "b"}` and the stored corpus
  depends on `a` (`PLACE@a`, 9 rows) while depending on `map` / `fold` nowhere. If `path._walk`'s derived labelling
  regresses, those nine rows do not fail loudly — they re-spell to a path form and read as a different site. Assert
  them by name, not by set membership.
- **Zero-axis folds are ambiguous by shape.** The pointwise cell, `030`'s finalize and the un-recognized escape are
  all operandless zero-axis folds; only `ops.axis_role`'s `Loop.role` fallback and `family_sites`' root-only rule
  separate them. Pre-collapse the type distinguished nothing either, but the failure was loud; now it is a wrong
  recursion. Phase 1 asserts it directly.
- **The remaining reconstruction is the risk**, and it is now only phase 3's: the flash paths, for which NOTHING in
  the registry re-asserts the deleted predicates. `_legality.py` and the committed digest baseline are the mitigations.
  Phase 2's own version of this risk was real and caught by the gates: `_demote_mixed_a` came back as `_promoted` +
  `warp_operand_dtype`, and the missing dtype predicate surfaced as a `cuTensorMapEncodeTiled` failure on an f32-A
  matmul under a warp pin — a row the pre-demolition builder also emitted, refused now.
- **`validate` enforces smem only.** Row 1 is enforced by the emitters and by nothing else until P1 restores the
  thread/CTA checks; registers are unenforced at every tier below 3.
- **Variant identity may collide in the prior's key space** (see Design). Cheap to check, expensive if real.
- **`_resolve_scalar_stage` invents `bk_elems`** — a real schedule decision no codec spells, found by a search loop
  with depth step-down. It is why the resolvers are projections, not filters, and why row dedup on the RESOLVED
  spelling is mandatory (`d2` clamping to `d1` is one row, not two).
- **Some flash goldens may not be enumerator-reachable**, and the gate cannot currently tell. `dit_xl_2.attn.s256`'s
  `k5` cannot come from `_FLASH_KEY_ATOMS = (2,4,8,16)`; it may be a hand pin. This is invisible today because
  `test_golden_knobs_are_members_of_the_move_catalog` skips every non-`MatmulGoldenConfig`, so no flash `TILE@dd` /
  `TILE@pj` value is checked against any domain at all. Phase 3 must extend the gate to the flash sites AND classify
  pin-only goldens explicitly, rather than leaving them silently unchecked.
- **The prior is downstream.** See the refit obligation; skipping it makes every post-refactor rank number suspect.
- **Order accidents will surface.** Resolve each explicitly — semantic (encode it) or not (document the diff);
  never let a harness pass by sorting.

## Out of scope

- **Materialization**, except the `qk.acc` repair — a fix, not a change. `030_split_reduce` keeps consuming the
  same slices.
- **`Stage.alt` → `top`/`late`** — its own plan, when a second inhabitant exists.
- **Widening any candidate domain.** The catalog is the domain; widening is a separate, benchmarked change that
  moves the deploy pick on the prior-free paths and requires a refit.
- **A register model.**
- **Multi-output nodes**, which make an edge's requirement per-slot.
- **Choosing `PLACE = cut|fuse` by schedule** — resolved at recognize time, before any fork exists.
