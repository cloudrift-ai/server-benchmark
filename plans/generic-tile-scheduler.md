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
collapse variant below is exactly "move the edges inline", which flips CONTRACTION to PLANAR by the same table. The
variant mechanism and the role derivation are not two mechanisms.

**`Fold.demoted()` does not exist** — it was dead once the scheduler was deleted (zero callers) and went in the
minimization pass. It is ~15 lines and fully specified here: move each operand edge into the lift body before the
first read of its bound name, ties in operand order. The collapse variant re-adds it (phase 2).

### What it means for this plan

**`type(node)` dispatch is not an option.** The next section argued emitters must key on `AxisRole` rather than node
type; with one node kind there is no type key to reach for and the argument is free. The three-arm asymmetry named in
"The actual defect" loses its last structural excuse: CONTRACTION, TWISTED and FREE are three readings of one node,
so an arm that builds `TileOp`s eagerly is visibly a special case rather than a different kind.

**`family_sites` restates without losing a site.** `REDUCE`/`STAGE` take every fold WITH an axis; `TILE` takes the
`role is CONTRACTION` folds plus the root zero-axis fold with no operands (the register-strip tier — a non-root
operandless fold, e.g. a one-load demoted cone, is still not a strip target); `PLACE` every non-root, non-derived
site. `Site.axis` is already `None` for the pointwise node, so the zero-axis fold needs no new case.

**Keys survive — under one constraint, which holds today.** `_SEGMENT_TOKENS` is `{"map", "fold", "a", "b"}` and
spelling is short-path-canonical (bare → `FAMILY@<axis>` → anchored path subsequence). Checked against the stored
corpus: the only path-spelled keys anywhere in the golden YAMLs are `TILE@pj` (45), `TILE@dd` (42), `PLACE@cone`
(60), `PLACE@stat` (4), `PLACE@fin` (1) — all AXIS names, segment-independent — and **`PLACE@a` (9 rows), which is
the A-edge SEGMENT**. Nothing stored spells `map` or `fold`. So:

> The frozen-key invariant holds **iff `path._walk` keeps emitting `a` / `b` edge labels for contraction-role
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
anticipates (qk-norm / RoPE folded into a score, on-the-fly dequant) — which is a STAGE-family concern, so phase 2
should expose `operand_lift(i)` even though no emitter reads it yet.

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

## Design — the north star

**The enumeration is a RECURSION over the site tree.** A row is a joint assignment across every site of the term;
the tree that generates it is the term's own. Two things thread through the recursion in opposite directions, and
they are the whole design:

```
def rows(site, inherited, ctx) -> list[Row]:            # Row = {canonical key -> spelled value}, multi-site
    local = [v for v in values(site.role, site, ctx)    # the per-family catalogs — search/space.py, unchanged
             if legal(v, inherited)]                    # DOWNWARD: the parent's tier constrains the child
    out = []
    for v in local:
        for combo in product(*(rows(c, ctx_of(site, v), ctx) for c in children(site))):
            row = merge(site, v, combo)                 # spells each slice at ITS canonical path (ops.Sched.key)
            if derive_inventory(slices(row)) is not CONFLICT:   # UPWARD: WORK folds up out of all the slices
                out.append(row)
    return out

def enumerate(tile, ctx) -> list[Row]:
    return uniform_keys([r for term in variants(tile.op, ctx)   # collapse / mixed-A — a finite set, unioned
                           for r in rows(root_site(term), TOP, ctx)])
```

- **Downward context** turns generate-and-drop into generate-only-legal. A child's candidate set is a function of
  what the parent already chose: the cone's inner statistic `REDUCE` is bounded by the parent's worker inventory
  (`ReducePlan.parse` binds `coop` to `work.count` and refuses a non-thread kind), and a warp parent fixes the
  child edge's fill kind. Under a flat product those are late `_assemble` drops — or, for the cone, nothing at all.
- **Upward derivation** is `WORK`. It is kernel-global and derived from the union of ALL sites' slices, so
  `derive_inventory` folds on the way up and a disagreeing combination is never built. A `TILE@dd` / `TILE@pj` pair
  that cannot share one inventory stops being a droppable row and becomes an unrepresentable one.

Two things stay OUTSIDE the recursion, deliberately:

- **Term variants** (collapse, mixed-A promotion) rewrite the tree, so they sit above the fold as a union of whole
  enumerations, with the uniform-key-set / decided-empty reconciliation at the top. Recursion does not absorb them.
- **The fork LEVEL order.** The recursion decides the row SET; `build_fork_tree` decides the evidence hierarchy.
  Conflating them would tie the prior's prefix structure to the term's tree depth. Levels stay
  `[WORK, *site keys, RASTER]`, derived from the site list — a flat order over a recursively generated set.

The wire format does not move: rows stay flat dicts of canonical codec keys, so goldens, the tune DB,
`canonical_row_key` and the prior's key space are untouched by the whole rebuild.

### Why a flat product cannot do this

A flat product over one node's families covers exactly the terms with one site. Every role that is scheduled today is
such a term — which is why a single-site builder could serve them, and why it is not evidence the shape generalizes.

The two families that remain are precisely the ones where **two sites must agree**: the fused cone carries a statistic
fold inside its A edge, and flash carries a hoisted QK operand edge plus a derived PV contraction. Neither is a product
over one node's families. Under a flat builder each needs its own bespoke emitter, and the "one generic enumerator"
claim becomes four hand-written products — the exact defect the rebuild exists to remove.

### The pieces — three of four already exist

- **`path.family_sites(family, path.sites(term))`** — already written and tested. `TILE` eligibility is already keyed
  on `role is AxisRole.CONTRACTION` (plus the root operandless zero-axis fold for the strip tier); `REDUCE`/`STAGE` on
  every fold/contraction; `Site.derived` already marks flash's `TILE@pj`. This plan does NOT invent a site walker.
- **`values(family, site, ctx) -> list[TilePlan | ReducePlan | Stage | str]`** — four implementations, keyed on the
  site's `AxisRole` (see below). The surviving catalogs in `search/space.py` already hand out typed slices in exactly
  this currency.
- **`merge` / `derive_inventory`** — spell each slice at its canonical path (`ops.Sched.key` already spells ANY
  site, so no new keys and no new codec), fold the inventory out of all of them, drop the combination when they
  disagree. This is the piece the deleted builder got wrong: it took a fixed three-key tuple instead of merging
  what the recursion produced.
- **`variants`** — the one genuinely new idea; see below.

`build_fork_tree` then groups rows into the fork tree with levels `[WORK, *site keys, RASTER]` — the pre-deletion
order, which `fork.Level` supports unchanged.

### Emitters are keyed on `AxisRole`, not `type(node)`

The IR already decided this: *"Detection stamps each loop with its role so scheduling dispatches on the axis's job,
never on a node type"* (`ir/axis.py`), *"The schedule is flat and kind-free"* (`ir/schedule.py`). Under the unified
`Fold` there is no type key left to reach for, so this section is a statement of what the emitters key on rather than
an argument against an alternative: **`ops.axis_role`, derived per the arity table above.** Half the dispatch is free
anyway, since `family_sites` already keys `TILE` on the CONTRACTION role.

The escapes depend on this. For an un-recognized loop-IR cell, `030`'s finalize and the coop fused-tail sibling,
`head()` is `None` and `ops.axis_role` falls back to the stamped `Loop.role`, landing them on the reduce entry with a
single scalar row — as today. This is the case the collapse makes MORE important, not less: pre-collapse they were
all `Map`, and a type key would have sent them to the strip fork (a silently wrong entry, worse than an omission);
post-collapse they are zero-axis folds indistinguishable from a real pointwise cell by shape alone, so the
`Loop.role` fallback is the only thing separating them. Phase 1 must assert it directly.

### Term variants — the one new mechanism

Four moves rewrite the term rather than decorating it, and each therefore yields a different `term_key`, a different
`op_cache_key`, and a different SITE SET:

| variant | what it does |
| --- | --- |
| strip | unrolls the zero-axis fold's `lift` ×r, α-renames SSA, fans out `TileOp.stores`, divides the inner extent |
| split-K | wraps the sliced contraction fold in an identity-lift `Fold(axis=ksplit)`, σ-reindexes operands |
| collapse | splices a computed edge inline (`Fold.demoted()`), removing its schedule site |
| mixed-A promotion | turns a MATERIALIZED f32 A edge into a computed cone so the sync compute-fill can convert it |

**The first two are functions of the ROW, not members of a pre-enumerated set.** This is the correction that
matters: `r` (strip ratio) IS the spelled `TILE=f<r>`, and `w` (split width) IS the spelled `REDUCE=g<w>`. They are
candidate VALUES, and the rewrite happens at materialization, from the plan the row already carries. Only collapse and
the mixed-A
promotion are a genuine finite set: two extra readings of one term, unioned. Treating `r` and `w` as variants would
make `variants()` a product and reintroduce the combinatorics this design removes.

Three obligations the union carries, all of which the deleted code enforced and none of which are free:

- **Uniform key sets.** Every leaf of one fork must spell the SAME family keys, with `""` as a DECIDED empty.
  The evidence pick's prefix-consistency depends on it: an absent key reads as "free" and would let a gmem-direct
  leaf inherit a staged row's measurement. A collapsed variant lacks `TILE@dd`,
  so the union must stamp the union of all variants' keys, decided-empty where a variant lacks the site.
- **No cross-variant suppression.** A variant's rows may not depend on whether a SIBLING variant produced any —
  that has no home under a union. Where suppression is genuinely wanted it becomes a local predicate on the base
  term (*no warp tile when A's dtype ≠ the atom's a-dtype and the transport cannot convert*), evaluable without
  sibling knowledge.
- **Variant identity must survive into the prior's key space.** `build_fork_tree` keys leaves on the knob dict
  ALONE; measurement identity is `(ctx.structural_key, op_cache_key)` and distinguishes variants, but variant
  identity is `(context, knobs)` and may not. Two structurally different kernels averaging under one feature row is a
  real hazard. **Check it as each variant lands** (`canonical_row_key(a) != canonical_row_key(b)` across variant pairs
  on each
  corpus shape); if they collide, the fix is an `S_*` stamp — like the existing `S_warp_eligible`, whose absence on
  materialized ops once cost a 330× fp16 misdeploy — never a new knob key.

### `WORK` is DERIVED from the slices

Not hoisted, not a free variable, not a shared symbol. The codec's dependency runs slice → work: `plan_workers` →
`derive_workers` → `ops.seal_workers`, and the parse signatures encode it (`TilePlan.parse(spec, work)`,
`ReducePlan.parse(spec, work)`). `assemble` derives the inventory from the row's own slices and drops the row when
they disagree — which is also what makes `WORK` legal as the outermost fork level despite being derived.

Two facts this must carry:

- **`""` is a first-class inventory** (`Workers.parse("")` is `None` — the per-cell / chain / pure-reduce tiers).
- **The tier couples the families.** `ReducePlan.parse` binds `coop` to `work.count` and RAISES unless
  `work.kind == "thread"`, so a tiled scalar site and a coop reduce are co-representable only when
  `par_m == 1 and par_n == coop`. `assemble` must drop those rows, not raise. Consequence to state: the fused-cone
  case can express the inner stat fold's coop reduce only at a width equal to the whole inventory — a codec-imposed
  expressiveness limit, unchanged by this plan.

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
| 12 | dtype gate (f32 A ⇒ no warp atom) | categorical | emitter filter, or the mixed-A variant |
| 13 | `+p` band: `WM·WN·32 + 32·aux ≤ cap` and `32·aux ≤ block_threads` | arithmetic | `_wspec_workers` |
| 14 | `+p` eligibility: warp tier ∧ TMA ∧ no reduce split | categorical | `_wspec_candidates` |
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

**The CHOICE layer is DELETED and being rebuilt recursively.** `_schedule.py` is gone; `020_schedule` enumerates
nothing, so every term stays unmapped and lowers on the materializer's per-cell path. The other two layers survive
untouched and phase 1 reuses both:

| layer | owns | where | state |
| --- | --- | --- | --- |
| DOMAIN | which candidate values exist at all | `search/space.py` (+ `search/domain.py` for the two coupled families) | KEPT — no change planned |
| LEGALITY | what this term's K / N / dtype / smem cap refuses | `lowering/tile/_legality.py`, raise-vs-drop by `pinned` | KEPT, currently unimported — becomes the recursion's downward filter |
| CHOICE | which families a site offers, each option-0, row → `TileOp` | (was `lowering/tile/_schedule.py`) | **DELETED** — phase 1 rebuilds it |

Phase 1's acceptance set — the roles whose correct output is already committed as a digest baseline:

| role | covered |
| --- | --- |
| `FREE` | pointwise cell + the register-strip TERM VARIANT (`TILE=f<r>`) |
| `PLANAR` / `TWISTED` | the reduce partition (heuristic option-0 + coop / ILP catalog + the matvec layout gate) |
| `CONTRACTION`, materialized edges | tile × stage × reduce × wspec × raster, scalar + warp tiers, split-K through the `Fold ⊃ Fold` composition |

Every one of those is a SINGLE-SITE term. The two that remain — a COMPUTED operand edge (the fused cone) and the
flash streaming pair — are the multi-site ones, and are why the enumeration is recursive.

### The gates that hold the rebuild

| gate | what it pins | where |
| --- | --- | --- |
| kernel-source digests | every case's rendered kernel, byte for byte, against a committed baseline | `scripts/kernel_digests.txt` (23 cases); `digest_kernels.py --check` reports drifted / missing / unexpected separately |
| pin LIVENESS | that a case's pins actually REACHED a kernel — all of them, on one emitted op (the `__partial` under split-K) | same harness; a digest alone cannot tell a covered path from an unmapped one, since both render and both hash stably |
| the xfail registry | every scheduler casualty as an exact node id, `strict=True` — an id that starts passing FAILS until deleted | `tests/xfail_registry.py`; the file shrinking to empty IS the completion gate |

The liveness half is what makes the digest baseline mean something during a rebuild: at the demolition it reports
**0 of 23** cases landing their pins, and each role that returns moves that number. The 10 cases that stayed dark
under the previous builder are recorded in a STRICT `UNSCHEDULED` set, so phases 2 and 3 report their own completion.

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
| a row → typed slices resolver | `ir/schedule.resolve_row` (both the env pins and an enumerated row) — single-site today, generalizes in phase 1 |
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
| `digest_kernels.py` vs a committed baseline (+ its per-case pin liveness) | a pinned/golden row materializing differently; a case whose pins stop reaching a kernel, or an `UNSCHEDULED` one that starts | the 10 `UNSCHEDULED` cases' materialization, until phases 2–3 land and they leave the set |
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
widths, max depth, `+p` rows, `g8k/coop-t` composites, every term variant); and **failures promoted** to a frozen
regression list that runs exhaustively thereafter, so the gate strengthens monotonically.

Rows 9–10 must NOT be asserted in tier 0 — row 9 is enforced by the resolvers and row 10 by nothing.

## Migration — the clean reimplementation

`_schedule.py` is DELETED. `020_schedule` enumerates nothing and every term stays unmapped — the guardrail contract,
so kernels still compile on the materializer's per-cell path and what fails is coverage, never a compile. The tree's
measured state at the demolition commit: `make test` green with **241 registered xfails** (52 added), and
`digest_kernels.py` reporting **0 of 23 cases landing their pins** (from 13). Those two numbers are the progress
meter for everything below.

| phase | scope | gate |
| --- | --- | --- |
| **1** | **The recursive enumerator — the target design, built once.** `rows(site, inherited, ctx)` over the site tree with downward context and upward `WORK` derivation; `merge` spelling every slice through `ops.Sched.key`; `values(role, site, ctx)` reading the unchanged `search/space.py` catalogs; `_legality.py`'s predicates reused as the downward filter. Restores the four single-site roles ONLY — `FREE` + the strip variant, `PLANAR`/`TWISTED`'s reduce partition, `CONTRACTION`'s tile × stage × reduce × raster over the scalar and warp tiers with split-K | **row-set identity, three independent ways**: the 13 liveness cases land again AND their digests are byte-identical to the committed baseline; `test_move_catalog`'s per-family set equality **plus its row-count equation**; `test_golden_knobs_are_members_of_the_move_catalog`. 52 registry ids deleted |
| 2 | CONTRACTION with computed edges — the fused norm→linear / gate⊗up cone, whose A edge is an inline node rather than a gmem `Load`. Under the recursion this is a `values` entry for the cone's sites plus legality (warp-only, sync fill mandatory, multi-channel ⇒ no cp.async/TMA) — NOT a new emitter | 4 `norm_linear` + `mlp_geglu` digest cases leave `UNSCHEDULED` + 6 recognize-boundary ids |
| 3 | TWISTED — the flash streaming pair: streaming / chain / per-cell / split-KV over a hoisted QK operand edge and a derived PV contraction. Two sites that must agree, which is what the recursion is for; adds `twisted_warp_moves`' geometry as a `values` entry | 40 attention-pin ids + 4 attention-coverage ids; the 5 flash digest cases leave `UNSCHEDULED` |
| 4 | `schedule()`'s own dispatch and flash-form selection once 2–3 exist; `kernel/_twist.py`'s `qk.acc` (reads a field `TilePlan` does not have — unreachable until the flash warp realizer runs, so it has no obtainable baseline until then) | registry EMPTY + `make test` + `make lint` |

**Phase 1 is the whole bet.** It must land with no new behaviour whatsoever: same rows, same order-independent set,
same kernels byte-for-byte. That is what makes it verifiable without a GPU and without judgement — the roles it
rebuilds are exactly the roles whose correct output is already committed as a baseline. Any row the recursion emits
that the single-site builder did not is a phase-1 bug, not a bonus; any row it drops is a regression the row-count
equation catches. Phases 2 and 3 then add `values` entries and legality predicates to a structure that is already
proven, instead of adding a third and fourth hand-written product.

Phases 2 and 3 both need the term-variant union this plan specifies (`variants()` — the collapse and
mixed-A promotion readings) and its three obligations: uniform key sets with `""` as a DECIDED
empty, no cross-variant suppression, and variant identity surviving into the prior's key space
(check `canonical_row_key(a) != canonical_row_key(b)` across variant pairs; if they collide the fix
is an `S_*` stamp, never a new knob key).

**Watch the row count.** The recursion GENERATES the cross-site product that the flat builder structurally avoided.
The downward context is what keeps it bounded, and `test_move_catalog`'s row-count equation is what makes growth
visible instead of silent — extend it to a per-site equation in phase 1, before there is anything multi-site to
count.

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
- **Terms are never mutated in place** — variants are explicit, each with its own `term_key`, site set and
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
- Levels are `fork.Level`s, `[WORK, *site keys, RASTER]`; a `Level.key` returning `()` is HOW variants with
  different key sets interleave as siblings. That mechanism is already implemented.
- **Empty enumeration returns `[]`, never raises** — "the guardrail contract". Tier 0 asserts every corpus term
  yields ≥ 1 row except the documented computed-A-no-legal-warp case.
- **Split-K re-entry**: `030_split_reduce` produces a `__partial` kernel that re-enters as its own `TileOp` and must
  itself be enumerable, without further splitting. Its `__partial` kernels are absent from `digest_kernels` output
  today; the baseline must include them.

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
| `cta`, `coop`, `reg`, `finalize`, `coop_transposed` | the reduce partition | the `ReducePlan.of` fields, spelled `g<cta>[k\|a]` / `b<coop>[t]` / `r<reg>`; LISTED (`splitk_moves`, `coop_reduce_moves`) |
| `group`, `orient` | the launch order | the `Raster` fields, spelled `g<orient><group>`; LISTED (`raster_moves`) |
| `warps_m`, `key_atoms`, `q_tiles` | the flash geometry | LISTED (`twisted_warp_moves`) — a candidate for a `Space` once its bounds are stated |
| `units`, `producer` | the `WORK` inventory | NOT enumerated — derived (`derive_inventory`) and folded up |

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
         in0 = load x[a0, a1]               ← the ×r strip variant unrolls THIS body, divides the inner grid
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

The fold's move families and their axis mappings:

- **WARP streaming** (dtype-gated): grid = (b0·b1, m/bm). m → bm query rows held as
  warp mma fragments (bm = WM·FM·atom_m from `WORK` + `TILE@dd`); kv → serial stream, bn keys per step
  (bn = WN·FN·atom_n — the kv-block ↔ score-tile coupling IS the downward constraint on the QK child); dd → mma
  k-steps; d → PV fragment columns via `TILE@pj`. ONE `WORK=w<M>x<N>` shared by the
  QK child and the derived PV. Constraint-table row 15 applies: `WN = 1`.
- **CHAIN** (FA-2 scalar): grid = (b0, b1, m) — one THREAD per query row; d → a per-thread
  register vector (`TILE@pj=f64`, legal since d = 64 ≤ the register budget); kv → serial per thread, the score
  computed ONCE per key and shared across all 64 columns. `WORK=""`.
- **Per-cell**: grid = (b0, b1, m, d) — one thread per output cell; kv serial; the QK edge collapses inline (the
  collapse variant — the edges move into the lift, so the node derives PLANAR by the arity table with no role
  rewrite), so the score recomputes per d — the redundant form the chain exists to beat.
- **Coop**: grid = (b0, b1, m, d), a `t<n>` band splits kv across lanes within the CTA; the cross-lane merge of
  (m, l, O) triples is the fold's own combine λ — carrier-generic, same machinery as examples 1–3.
- **Split-KV** (`g<n>k`): composes with the warp rows — kv → n CTAs × serial stream; each partial keeps fragment
  residency; `030_split_reduce` realizes the partial + LSE-combine finalize. The same `g<n>` move as matmul
  split-K, under legality (kv divides, slices block-whole) plus the inventory fold.

Pins (`TILE@dd` / `TILE@pj` / `REDUCE@<kv>`) narrow at their paths like every other kernel — no per-form routing
block, no flash-specific pin escape.

| site | enumerate | constraints |
| --- | --- | --- |
| root `Fold free` (the `divide(O, l)` projection) | — (has an operand) | — |
| `Fold[kv] twisted` — `REDUCE@<kv>` | the FORM (warp streaming / chain / per-cell / coop), plus `cta ∈ {2,4,8}`, `coop`, `finalize = kernel` for split-KV | `cta` divides kv and slices block-whole; `finalize=atomic` is ILLEGAL (the twisted `e^{Δm}` rescale cannot be an atomic); `↑` ONE inventory shared by every site below |
| same fold — `STAGE` | `depth ∈ {1,2,3,4}`, `transport ∈ {cp, tma}`, `ring`, `alt` for the K/V stream | slab fits smem; `↓` warp forms only — the chain and per-cell forms stage nothing |
| `operand` — the hoisted QK score, `TILE@dd` | **IS a site**: `atom`, `wm` (the `warps_m` of the flash grid), `wn`, `fm` (`q_tiles`), `fn` (`key_atoms`), `bk` | `↓` `wn·fn·atom.atom_n` must EQUAL the parent's streaming key block — the kv-block ↔ score-tile coupling, the archetypal downward constraint, and a `Bound(op="==")` if this family is ever generated; `wn = 1` |
| the DERIVED PV contraction, `TILE@pj` | its own `atom` (including the f16-acc sibling), `fm`, `fn`, `bk` | `↓` `wm` / `wn` are NOT free here — it shares the QK child's warp map by construction; `Site.derived` ⇒ EXCLUDED from `PLACE` (it lies below the seam lattice, so it is not a cut) |
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
- **The remaining reconstruction is the risk**, and it is concentrated: phases 2-3 are the paths for which
  NOTHING in the registry re-asserts the deleted predicates (`_demote_mixed_a` above all). `_legality.py` and the
  committed digest baseline are the mitigations — but see the baseline's blind spot in Status: it does not reach the
  tiered/placed contraction path, so the flash phase leans on the liveness assertion for its gate.
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
