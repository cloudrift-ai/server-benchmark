# Generic tile scheduler — every role emits rows

## The actual defect

The tile IR is compositional (one `Fold` node kind — see the next section) and the schedule output is compositional
(`TileOp.schedule`, per-node slices keyed by the tree-path codec). The deleted `_schedule.py` was compositional in
ONE of its three arms and not in the other two, and that asymmetry — not "hand-written whole-tree paths" in general —
is where the complexity lived:

- **CONTRACTION was already tabular.** `_tile_rows` was a five-deep product over per-family candidate functions
  (`tiles × _stage_candidates × _reduce_candidates × _wspec_candidates × _raster_candidates`), each row assembled by
  `_site_row` and handed to `build_fork_tree` with `WORK` as the outermost level. That arm is the model to copy.
- **TWISTED built `TileOp`s eagerly.** `schedule()`'s tail constructed `_option` siblings directly — the warp move
  grid, the chain form, the reduce tiers — never producing rows. ~570 lines, all flash-specific.
- **FREE built `TileOp`s eagerly too**, via `_map_strip_fork`.

Because two arms never produced rows, the fused-term merge had to be a flat `[*maps, *con]` list of ops rather than a
row union, pins had to be narrowed per path, and every new composition needed another whole-tree path.

**The goal, stated precisely: every role emits `list[dict]` rows through ONE `build_fork_tree`; no role builds
`TileOp`s directly.** That is a smaller and more defensible claim than "replace the scheduler with a solver", and it
is the one the corpus supports.

**Read the Status section before costing anything.** The old scheduler is DELETED; this is a reconstruction.

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
first read of its bound name, ties in operand order. Phase 3 re-adds it as part of the collapse variant.

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
spellings. Phase 2's measurement work therefore starts from an empty reservoir — a cost already sunk, not a risk
still to manage.

## Design

```
def rows(tile, ctx) -> list[dict]:
    out = []
    for term in variants(tile.op, ctx):                 # collapse / mixed-A promotion — a finite set
        sites = {f: family_sites(f, path.sites(term)) for f in ("TILE", "REDUCE", "STAGE")}
        for combo in product(*(values(f, site, ctx) for f, site in flat(sites)), raster_values(term, ctx)):
            row = assemble(term, combo, ctx)            # = the old `_site_row`: derives WORK, returns None on conflict
            if row is not None:
                out.append(row)
    return out
```

Four pieces, three of which already exist:

- **`path.family_sites(family, path.sites(term))`** — already written and tested. `TILE` eligibility is already keyed
  on `role is AxisRole.CONTRACTION` (plus the root operandless zero-axis fold for the strip tier); `REDUCE`/`STAGE` on
  every fold/contraction; `Site.derived` already marks flash's `TILE@pj`. This plan does NOT invent a site walker.
- **`values(family, site, ctx) -> list[TilePlan | ReducePlan | Stage | str]`** — four implementations, keyed on the
  site's `AxisRole` (see below). The surviving catalogs in `search/space.py` already hand out typed slices in exactly
  this currency.
- **`assemble`** — the old `_site_row`: spell each slice at its canonical path, derive `WORK` from the slices, return
  `None` when they disagree. Eight lines. Its docstring already said "today's enumeration never builds one … so this
  is a guard, not a path".
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
`Loop.role` fallback is the only thing separating them. Phase 3 must assert it directly.

### Term variants — the one new mechanism

Four moves rewrite the term rather than decorating it, and each therefore yields a different `term_key`, a different
`op_cache_key`, and a different SITE SET:

| variant | what it does | who did it |
| --- | --- | --- |
| strip | unrolls the zero-axis fold's `lift` ×r, α-renames SSA, fans out `TileOp.stores`, divides the inner extent | `_map_strip_option` |
| split-K | wraps the sliced contraction fold in an identity-lift `Fold(axis=ksplit)`, σ-reindexes operands | `_splitk_option` + `_factor_k` |
| collapse | splices a computed edge inline (re-add `Fold.demoted()`), removing its schedule site | `_demote_planar` / `_demoted_warp_option` |
| mixed-A promotion | turns a MATERIALIZED f32 A edge into a computed cone so the sync compute-fill can convert it | `_demote_mixed_a` |

**The first two are functions of the ROW, not members of a pre-enumerated set.** This is the correction that
matters: `r` (strip ratio) IS the spelled `TILE=f<r>`, and `w` (split width) IS the spelled `REDUCE=g<w>`. They are
candidate VALUES, and the rewrite happens at materialization — exactly as the deleted code did it (`_map_strip_option`
and `_splitk_option` both received the plan and built the rewritten term from it). Only collapse and the mixed-A
promotion are a genuine finite set: two extra readings of one term, unioned. Treating `r` and `w` as variants would
make `variants()` a product and reintroduce the combinatorics this design removes.

Three obligations the union carries, all of which the deleted code enforced and none of which are free:

- **Uniform key sets.** Every leaf of one fork must spell the SAME family keys, with `""` as a DECIDED empty.
  `_tile_rows`' own comment says why: *"The evidence pick's prefix-consistency depends on it: an absent key reads as
  'free' and would let a gmem-direct leaf inherit a staged row's measurement."* A collapsed variant lacks `TILE@dd`,
  so the union must stamp the union of all variants' keys, decided-empty where a variant lacks the site.
- **No cross-variant suppression.** `_tile_rows` did `if demoted_rows: tiles = [p for p in tiles if not p.is_warp]`
  — the base term's rows depended on whether the mixed-A sibling produced any. That has no home under a union; it
  becomes a local predicate on the base term (*no warp tile when A's dtype ≠ the atom's a-dtype and the transport
  cannot convert*), evaluable without sibling knowledge.
- **Variant identity must survive into the prior's key space.** `build_fork_tree` keys leaves on the knob dict
  ALONE; measurement identity is `(ctx.structural_key, op_cache_key)` and distinguishes variants, but variant
  identity is `(context, knobs)` and may not. Two structurally different kernels averaging under one feature row is a
  real hazard. **Check it in phase 2** (`canonical_row_key(a) != canonical_row_key(b)` across variant pairs on each
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

### The candidate domain stays the catalog

`search/space.py`'s catalogs are the domain. They are not arbitrary: their comments record provenance —
`f2x14`/`f4x8`/`f4x10`/`f4x26` are golden winners a previous rebuild orphaned (1.29–1.49× reachability losses),
`(1,16)` is the thin-M decode geometry, `(2,8)` the lm_head.m64 winner (2392 → 1215 µs). A generated lattice
reproduces a box and discards the measurements that justify its points.

**Do not generate the domain from constraints.** An earlier revision of this plan proposed deriving the `(wm, wn)`
domain from a linear system in exponent coordinates, taking flash's `WORK` level from 4 options to 19. Three reasons
that is wrong, and the first is fatal:

1. **The geometry is not a power-of-two lattice.** `_codec_width` requires only `int >= 1`; the only power-of-two
   requirement anywhere is `ReduceStage.width` at `Level.BLOCK`. The corpus proves it: goldens carry `f2x6`, `f2x9`,
   `f4x6`, `f4x10`, `f4x12`, `f4x26`, six `d3/` rows, and — on a flash shape, on exactly the two sites an exponent
   table would model — `rtx4080_sm89.yaml:132` spells `TILE@dd: mma_m16n8k16_f16_f32/f2x8/k5` with
   `TILE@pj: .../f2x9/k4`. `bk = 5` and `fn_pv = 9` are not points in an exponent space. `_SCALAR_REG` in the
   SURVIVING catalog carries the same non-powers of two, commented "golden-informed".
2. **Tiles do not divide extents.** 35 golden rows over-cover (`q_proj.m16`: `M=16`, `tile_m=32` — masked partial
   tiles, which is what `Side.mask` exists for), and real extents are not products of tile factors
   (`qk_global_cat` has `N = 8704 = 2^9·17`). The governing relation is `ceil_div` + masking: neither an equality
   nor linear.
3. **It breaks the tree's only live completeness anchor.**
   `test_golden_configs.py::test_golden_knobs_are_members_of_the_move_catalog` — *"Permanence: every recorded golden
   knob set stays REACHABLE by the search"* — passes today, against the catalog. Generating the domain dissolves its
   referent.

Note `a602874a` deleted `map_tile_moves` and `wspec_moves` from `space.py` (they went unread the moment the
scheduler did), so the strip-tile and `+p` producer-band domains must be reconstructed with the rest. The reduce
catalog SURVIVES — `coop_reduce_moves` / `splitk_moves` — as do `scalar_tile_moves`, `warp_tile_moves`,
`twisted_warp_moves`, `stage_moves` and `raster_moves`.

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

The one genuine mess the deleted code carried is worth fixing, and it is cheaper than any of the above: several
predicates existed TWICE — once as a silent drop in the enumerator, once as a loud raise on the pin path
(`_check_warp_static_k`, `_fragment_epilogue_ok`). That is the bug class producing "the pin says yes and the
enumeration says no".

**One predicate module, one function per predicate, each returning `str | None` (the refusal reason), with the
caller choosing raise-vs-drop from a single `pinned: bool`.** The table's "where checked" column then cites a
function name, and a dropped constraint is a dead function visible in review.

The term-reading predicates live here too, and they are real symbolic analyses — not lookups. Budget them as such:
`_matvec_b_kstride` (enumerates loads, takes `free_vars()` of each index, calls `gmem_row_stride`; TRI-VALUED —
`none` means "no layout gate applies"), `_shared_row_buf` (index-tuple equality returning a buffer name),
`_fragment_epilogue_ok` (epilogue def/use dataflow), `_tma_operand_rank_ok`, `_warp_atoms`' dtype scan,
`_has_contraction_tail`. Precompute them once per kernel into a feature record; the emitters then read fields.

### Placement

`Placement` is a `TileOp` field, not a schedule slice, and each family constructs it at materialization from the
row's own slices — the flash warp shrink (`ceil_div(um·fm·atom_m)`, value axis dropped, `Window(parent=...)`), the
chain grid truncation, the strip's re-derived `free`. **Split-K adds no placement** — `_splitk_option` reuses
`place` verbatim, `_factor_k` puts `ksplit` on the fold's axis, and `030_split_reduce` owns the grid.

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
a prior key. **This path has no digest coverage** (see Status) — it is pinned by unit tests instead.

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

### `Stage.alt` — split out of this plan

Retiring `alt` for a derived `top`/`late` refill policy is removed entirely. Four reasons: it ADDS rows rather than
re-spelling them (`stage_moves()` does not offer `alt`; it reaches the compiler only via pins and 33 golden rows —
28 `d1/cp/alt`, 5 `d1/tma/alt`); `late` cannot be derived from def-use alone (`alt` also implies staging the A
operand, motivated by register pressure); `Stage` is one object per reduce loop with a single depth and a SET of
staged names, so a per-buffer derivation is unspellable, and on a single-phase step `late ≡ top` would emit
duplicate rows; and it reaches `features.py`'s `D_stage_alt` and `_stage_sig`, orphaning every recorded `d1/*/alt`
row of DB evidence. **Consequence: keys and value spellings are ALL frozen**, so no gate is forfeited and no
correctness budget is spent on churn.

## Status — what the tree actually holds

The old scheduler is DELETED at `e27d8fdc`: `_schedule.py` (2458 lines), `_view.py` (96), `020_schedule.py` (109).
Recognition, the codec, the materializer and `030_split_reduce` were untouched by that commit.

**Every helper this plan reconstructs returns zero greps at HEAD** — `_tile_rows`, `_reduce_specs`,
`_map_strip_fork`, `_computed_a_rows`, `_twisted_warp_options`, `_stamp_twisted_split`, `_narrow_flash_forms`,
`_demote_planar`, `_demote_mixed_a`, `_demoted_warp_option`, `_twisted_chain_option`, `prologue_knob_bases`,
`_shared_row_buf`, `_row_stage`, `_coop_carrier`, `_pick_coop`, `_warp_atoms`, `_f16acc_allowed`,
`_fragment_epilogue_ok`, `twisted_pair`, `contraction_view`, `_FREE_CAP`, `_has_contraction_tail`. So every "port"
is "read `git show e27d8fdc^:…/_schedule.py` and rewrite".

**But far more survives than an earlier revision of this plan claimed**, and it changes both the cost and the gates:

| the plan does NOT need to build | it already exists |
| --- | --- |
| a row-set oracle | `test_move_catalog.py::test_schedule_leaf_set_equals_catalog` — per-family set equality **plus a row-count equation** (`len(tiled) == len(stages) * n_reduces * len(raster_moves())`), xfailed at `xfail_registry.py:79` |
| a golden reachability test | `test_golden_configs.py::test_golden_knobs_are_members_of_the_move_catalog` — **passes today** |
| an enumeration dump | `golden_eval.enumerate_graph(graph, ctx, family=)` — returns every row keyed by canonical spelling; already the join point for `emmy fit` / `eval` / `golden_neighbor_bench` |
| a site walker | `path.family_sites` / `path.sites` / `ops.Sched` |
| worker sealing | `ops.seal_workers`, `derive_workers`, `plan_workers` |
| the `""`-TILE ambiguity resolver | `schedule.resolve_site_tile` — and round-trip assertions must go through IT, not `TilePlan.parse` |
| smem footprint | `pack_smem`, `KernelOp.smem_bytes()` |
| split-K carrier legality | `ir/stmt/passes.py::projection_distributes` + `030_split_reduce`'s own gates |
| pin narrowing / no-match-keeps-full-list | `Knob.narrow`, `pin_key_matches`, `family_value` |
| row dedup / tie-break | `knob.canonical_row_key` |
| the recording view | `knob.stamp_schedule_families` |
| the loop → term parser | `passes/lowering/tile/_fromloop.py::fold_from_loop` / `nodify_reduce` (moved out of the IR) |
| the `(m, n)` binding | `ops.Sched.tile_of` returns a PLACED slice; readers state no placement rule (see Placement) |

Three things the deleted scheduler leaned on do NOT exist and must be written, not ported:

- `Fold.demoted()` — the collapse variant. Fully specified above; ~15 lines (phase 3).
- `Fold.operand_lift(i)` — the per-operand prologue reading (derived; phase 2 exposes it for the fused prologue).
- any consumer that expected `Map` / `Contraction` as *types*: the readings are `axis is None` and
  `is_contraction(x)`.

Three tooling facts that decide what a gate can mean:

- **`tests/xfail_registry.py` holds 107 ids**, applied `strict=False` — a restored test passes UNNOTICED. On this
  box the suite reports 605 skips, concentrated in the scheduler-dependent modules (`test_attention_coverage.py`
  120 tests, `test_matmul_coverage.py` 142). "Registry empty" is not a completion criterion.
- **`scripts/digest_kernels.py` runs green and prints garbage.** Measured at HEAD: exit 0, 24 lines, no `<ERROR>`
  line, but only **15 distinct digests** — `matmul_scalar`/`warp_tma`/`splitk` collide, as do
  `warp_f16acc`/`raster`/`wspec`, `norm_linear`/`_splitk`/`_coop`, `flash_hd128`/`_cp`, `flash_hd256_alt`/`_fm`, and
  `flash_chain`/`flash_scalar`. The pins are ignored and the un-recognized escape renders. No baseline exists in the
  tree. Measured at `e27d8fdc^`: exit 1, 24 kernel lines all with DISTINCT digests, four of them `__partial`, plus
  four errored cases (next bullet). The script already exits nonzero when a case raises — what it lacks is a
  liveness check that the pins actually landed, which is the half of P0 that stays open.

  **What that means for the rebuild, measured rather than assumed:** instrumenting `Sched.tile_of` across a
  full digest run shows it reached on every contraction site (matmul `a2`; flash `dd` and `pj`) and returning
  `None` every time — there is no `TILE` slice to return, because nothing consumes the pins. So the digest gate
  pins RECOGNITION, term storage and the UN-SCHEDULED lowering path, and does **not** exercise the tiered/placed
  contraction path at all. Any change to `_factor`'s tiled arm or `_twist`'s warp realizer is invisible to it, and
  needs its own unit coverage until P0 restores the harness.
- **The materializer has a live regression on the flash warp path.** `kernel/_twist._realize_prologue` reads
  `qk.acc` off a parameter annotated `TilePlan`, which has no such field (`atom, units, regs, bk, axes`) — ONE site;
  its siblings take the node, which does have it. It is unreachable at HEAD (nothing schedules, so the warp realizer
  never runs), but at `e27d8fdc^` all four flash warp cases error with `AttributeError`, so the flash warp path has
  no obtainable baseline until it is fixed.

## Gates

| gate | catches | does not catch |
| --- | --- | --- |
| `test_schedule_leaf_set_equals_catalog` + its 3 siblings (restore from xfail) | a family's value domain or row count changing | cross-family composition on non-matmul shapes |
| `test_golden_knobs_are_members_of_the_move_catalog` (live today) | a golden becoming unreachable | non-golden losses |
| per-key value-domain snapshot (new, ~80 lines) | which family lost which value, localized | ranking quality |
| `xfail_registry` shrink (107 ids) | a restored behavior regressing | the 605 CPU skips; GPU-only; silent XPASS under `strict=False` |
| `digest_kernels.py` vs a committed baseline | a pinned/golden row materializing differently | the 4 broken flash-warp cases until P0 |
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
the demolished tile-flavor wrappers; `ctx.max_threads_per_cta` still exists as a field, read by nothing). COMPLETENESS: every row that should exist is emitted — covered by
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

## Migration

**Spike first. The generic machinery is not the risk; the reconstruction is.**

**P0 — make the harness honest (independent, land this week).** Fix `kernel/_twist.py`'s `qk.acc`. Add a liveness
assertion to `digest_kernels.py` — each case asserts its pinned knobs landed on the emitted `TileOp` (the nonzero
exit on a raising case already exists). Commit a baseline from `e27d8fdc^` (needs `PYTHONPATH=.` from a worktree,
and record which mode it runs in — the deploy/tune reduce narrowing differs). Flip the registry to `strict=True`.
Write the full-corpus golden-reachability loop over `GOLDEN_CONFIGS` asserting `evaluate_golden(...).rank is not
None` — ~10 lines on existing API; it xfails until phase 2 and then gates for free. **Nothing here depends on the
design.**

**SPIKE (1–2 days) — one shape end to end.** Restore `020_schedule` + `assemble` + the three CONTRACTION candidate
functions for a plain matmul only. Gate: `pytest tests/compiler/passes/test_move_catalog.py
tests/compiler/e2e/test_matmul_coverage.py`. This proves site enumeration → typed slices → `seal_workers` →
`build_fork_tree` → materialize → `validate` → digest against checked-in per-family set assertions and a row-count
equation, before any generic code exists. It also produces the `020` rule as a by-product, which is otherwise a
gateless prerequisite.

Then, one phase per EMITTER KEY (not per recursion depth — the walk is flat):

| phase | scope | reconstruct | gate |
| --- | --- | --- | --- |
| 1 | generalize the spike: the `values`/`assemble`/`variants` shape, predicate module, term-variant union + the key-collision check | ~290 | the spike's tests still green; snapshot checked in |
| 2 | CONTRACTION, no computed edges — incl. split-K and mixed-A variants, raster, `+p` | ~985 | 23 matmul-coverage ids + ~11 digest cases + `test_schedule_leaf_set_equals_catalog` |
| 3 | FREE + PLANAR — strip variant, coop catalog, **`coop-t`** + `_matvec_b_kstride`, deploy narrowing | ~332 | rms_norm / softmax / reduce / pointwise / matvec digest cases + the coop-catalog test |
| 4 | CONTRACTION with computed edges — the fused cone; the `020` merge becomes a row union under ONE spelling | ~97 | 4 norm_linear + mlp_geglu digest cases + 6 recognize-boundary ids |
| 5 | TWISTED — streaming / chain / per-cell / split-KV; retire the eager `_option` construction | ~570 | 40 attention-pin ids + 4 attention-coverage ids; digest only after P0 |
| 6 | `schedule()`'s own dispatch, pin narrowing, flash-form selection | ~190 | registry + `make test` + `make lint` |

Measured from the deleted source; total ≈ **2460 lines reconstructed**, plus new code. Budget it as "read 2458 lines
of history, write 2000–2500 lines in a different shape" — not as a port.

**After phase 6, before any GPU sweep is trusted: refit the offline prior.** `commands/fit.py::build_golden_groups`
reconstructs each golden's candidate pool by RE-ENUMERATING (`enumerate_graph`), pins the golden's index in it, and
fits `prior/offline_weights.json` against that. A changed enumeration silently changes the fit's training data, and
`emmy eval offline`'s rank/pool columns are measured against the new pool — so a rank "improvement" can be an
artifact of a shrunken pool. Run `emmy fit --artifact` and re-verify golden rank + top1/10/25/50.

Merge gate (GPU): `make bench-kernels`, a flash/attention compile + tune probe on the 5090, the eval-golden MATCH
sweep.

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
  today; P0's baseline must include them (they appear at `e27d8fdc^`).

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

No operands, so the arity table lands the inner node on PLANAR. The reduce emitter produces `_reduce_specs`' rows
verbatim: serial option-0 (past `_FREE_CAP` the heuristic stays scalar), then the coop catalog — each `t<n>` row's
worker demand IS its `WORK` inventory, nothing to unify against — then guarded `g<n>` / `r<n>`.

Note the bare reduce is ALREADY wrapped by an identity zero-axis node. That is not new: it is what the projection
wrapper always did, and it means the FREE and PLANAR emitters compose at depth 0/1 on even the simplest shape.

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
A/B labels off the stored operand ORDER `(b, a)`. **Note what the dump makes plain and my earlier draft of this plan
got wrong:** A/B is *not* derived from the accesses. Node-locally `x[a0, a2]` and `w[a2, a1]` are symmetric — each
carries the K axis plus one free axis — and telling M from N needs the PLACEMENT, which is a caller fact on the
`TileOp` and deliberately absent here. Order is what `Fold.contraction(...)` generates and what the gate pins.

Depth-0: the CONTRACTION emitter's tile × stage × reduce × raster product — `_tile_rows` almost unchanged, minus
the `contraction_view` shape-probe. The warp sibling family maps the same axes differently: `WORK=w<M>x<N>` puts
32-lane warps on an (m, n) warp grid, `TILE=<atom>/f<FM>x<FN>[/k<bk>]` gives each warp an
(FM·atom_m) × (FN·atom_n) fragment tile, and a2 advances in `atom_k`-element mma steps, `bk` per smem stage.

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

The cone's inner stat fold answers with its own `REDUCE@<axis>` slice from its own emitter (retiring
`prologue_knob_bases`' hand-threading), and the `020` MONOID-producer merge (two term readings of one loop) reduces
to "run `candidates` on each reading, union the rows". `PLACE@cone` and `PLACE@a` are exactly the segment spellings
the collapse had to preserve.

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
`family_sites`' root-only rule. Phase 3 must assert it directly.

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

- **WARP streaming** (`_twisted_warp_options` today; dtype-gated): grid = (b0·b1, m/bm). m → bm query rows held as
  warp mma fragments (bm = WM·FM·atom_m from `WORK` + `TILE@dd`); kv → serial stream, bn keys per step
  (bn = WN·FN·atom_n — the kv-block ↔ score-tile coupling `_stamp_twisted_split` hand-computes IS the unification
  check on the QK edge); dd → mma k-steps; d → PV fragment columns via `TILE@pj`. ONE `WORK=w<M>x<N>` shared by the
  QK child and the derived PV. Constraint-table row 15 applies: `WN = 1`.
- **CHAIN** (FA-2 scalar, `_twisted_chain_option`): grid = (b0, b1, m) — one THREAD per query row; d → a per-thread
  register vector (`TILE@pj=f64`, legal since d = 64 ≤ the register budget); kv → serial per thread, the score
  computed ONCE per key and shared across all 64 columns. `WORK=""`.
- **Per-cell**: grid = (b0, b1, m, d) — one thread per output cell; kv serial; the QK edge collapses inline (the
  collapse variant — the edges move into the lift, so the node derives PLANAR by the arity table with no role
  rewrite), so the score recomputes per d — the redundant form the chain exists to beat.
- **Coop**: grid = (b0, b1, m, d), a `t<n>` band splits kv across lanes within the CTA; the cross-lane merge of
  (m, l, O) triples is the fold's own combine λ — carrier-generic, same machinery as examples 1–3.
- **Split-KV** (`g<n>k`): composes with the warp rows — kv → n CTAs × serial stream; each partial keeps fragment
  residency; `030_split_reduce` realizes the partial + LSE-combine finalize. The same `g<n>` move as matmul
  split-K — legality (kv divides, slices block-whole) plus unification replace `_stamp_twisted_split`.

Pins (`TILE@dd` / `TILE@pj` / `REDUCE@<kv>`) narrow at their paths like every other kernel; `_narrow_flash_forms`
and the warp/stage-pin routing block at the top of the twisted branch retire.


### Summary — what each hand-written path becomes

| Today | Under the walk |
| --- | --- |
| `_tile_rows` (contraction product) | the CONTRACTION emitter, depth-0 |
| `_reduce_specs` + `_coop_carrier` | the PLANAR emitter, depth-0 |
| `_row_stage` / `_shared_row_buf` shape-match | stage move on a twice-read materialized edge (gate kept) |
| `_computed_a_rows` + `prologue_knob_bases` | depth-1 recursion; the child fold spells its own slices |
| `020`'s MONOID-producer merge | candidate-union of two term readings; decided-empty generic |
| `twisted_pair` + `_twisted_warp_options` | TWISTED emitter + child recursion + unification |
| `_twisted_chain_option` | a row in the TWISTED emitter |
| `_stamp_twisted_split` + matmul split-K | ONE `g<n>` fold move (atomic / kernel-finalize legality) |
| `_narrow_flash_forms` + per-path pin escapes | per-path narrowing of local candidate lists |
| `_demote_planar` / `_demoted_warp_option` | the collapse variant — edges inline, role falls to PLANAR |
| `Contraction.a` / `.channels` / `.b_trans` | derived readings of `operands` (A/B by stored order; `b_trans` off B's index) |

Not covered by these eight and not to be dropped: the `PLACE=cut` re-recognized pieces (18 golden rows, each its own
kernel), `rope` / `embedding` (which record `{WORK: ''}` and nothing else — a zero-choice enumeration, benign but it
must not crash), `lm_head`'s coop-cap lift, `qknorm.k256/k512`, `mlp_ch`, `mlp_geglu`, `mlp_gate_up_split`, and
every `.dynM` twin. The completeness test must also FILTER the 57 golden rows carrying non-scheduler keys
(`FAST_EXP`, `INTERLEAVE_LOADS`, `VECTORIZE_LOADS`, `LOOPIFY`, `PLACE`) — those come from `lowering/kernel` passes
and recognition, not from this enumeration.

## Risks

- **The measured evidence is already orphaned** (see "The IR beneath it"): the collapse changed `term_key`, so no
  DB / reservoir row and no cubin keyed on `(ctx.structural_key, op_cache_key)` still matches. Phase 2 measures from
  empty; goldens are unaffected. Sunk, not pending — but every "regression vs recorded evidence" reading is void.
- **A path segment can be silently lost.** `_SEGMENT_TOKENS` is `{"map", "fold", "a", "b"}` and the stored corpus
  depends on `a` (`PLACE@a`, 9 rows) while depending on `map` / `fold` nowhere. If `path._walk`'s derived labelling
  regresses, those nine rows do not fail loudly — they re-spell to a path form and read as a different site. Assert
  them by name, not by set membership.
- **Zero-axis folds are ambiguous by shape.** The pointwise cell, `030`'s finalize and the un-recognized escape are
  all operandless zero-axis folds; only `ops.axis_role`'s `Loop.role` fallback and `family_sites`' root-only rule
  separate them. Pre-collapse the type distinguished nothing either, but the failure was loud; now it is a wrong
  emitter. Phase 3 asserts it directly.
- **The reconstruction is the risk.** ~2460 lines exist only in `git show`, their tests were deleted with them, and
  for several predicates — `_demote_mixed_a` above all — NOTHING in the 107-id registry re-asserts them. The
  predicate module and the spike are the mitigations.
- **`validate` enforces smem only.** Row 1 is enforced by the emitters and by nothing else until P1 restores the
  thread/CTA checks; registers are unenforced at every tier below 3.
- **Variant identity may collide in the prior's key space** (see Design). Cheap to check, expensive if real.
- **`_resolve_scalar_stage` invents `bk_elems`** — a real schedule decision no codec spells, found by a search loop
  with depth step-down. It is why the resolvers are projections, not filters, and why row dedup on the RESOLVED
  spelling is mandatory (`d2` clamping to `d1` is one row, not two).
- **Non-power-of-two goldens may not be enumerator-reachable.** `dit_xl_2.attn.s256`'s `k5` cannot come from
  `_FLASH_KEY_ATOMS = (2,4,8,16)`; it may be a hand pin. Determine this in phase 5 — if it is pin-only, the
  reachability gate must classify it, not silently fail.
- **The prior is downstream.** See the refit obligation; skipping it makes every post-refactor rank number suspect.
- **Order accidents will surface.** Resolve each explicitly — semantic (encode it) or not (document the diff);
  never let a harness pass by sorting.

## Out of scope

- **Materialization**, except P0's `qk.acc` repair — a fix, not a change. `030_split_reduce` keeps consuming the
  same slices.
- **`Stage.alt` → `top`/`late`** — its own plan, when a second inhabitant exists.
- **Widening any candidate domain.** The catalog is the domain; widening is a separate, benchmarked change that
  moves the deploy pick on the prior-free paths and requires a refit.
- **A register model.**
- **Multi-output nodes**, which make an edge's requirement per-slot.
- **Choosing `PLACE = cut|fuse` by schedule** — resolved at recognize time, before any fork exists.
