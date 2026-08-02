# Generic tile scheduler — every role emits rows

## The actual defect

The tile IR is compositional (`Map` / `Fold` / `Contraction`) and the schedule output is compositional
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
  on `role is AxisRole.CONTRACTION` (plus the root sourceless `Map` for the strip tier); `REDUCE`/`STAGE` on every
  fold/contraction; `Site.derived` already marks flash's `TILE@pj`. This plan does NOT invent a site walker.
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
never on a node type"* (`ir/axis.py`), *"The schedule is flat and kind-free"* (`ir/schedule.py`). A type key is
strictly coarser — `Fold` spans PLANAR, TWISTED and CONTRACTION — so it would re-dispatch on `.role` internally.
Half the dispatch is free anyway, since `family_sites` already keys `TILE` on the CONTRACTION role.

The escapes depend on this. For an un-recognized loop-IR cell, `030`'s finalize and the coop fused-tail sibling,
`head()` is `None` and `ops.axis_role` falls back to the stamped `Loop.role`, landing them on the reduce entry with a
single scalar row — as today. Under a type key they would all be `Map` and take the strip fork: a silently WRONG
entry, which is worse than an omission.

### Term variants — the one new mechanism

Four moves rewrite the term rather than decorating it, and each therefore yields a different `term_key`, a different
`op_cache_key`, and a different SITE SET:

| variant | what it does | who did it |
| --- | --- | --- |
| strip | unrolls the `Map` body ×r, α-renames SSA, fans out `TileOp.stores`, divides the inner extent | `_map_strip_option` |
| split-K | builds `Fold(axis=ksplit, source=Contraction(k_axis=kslice))`, σ-reindexes operands | `_splitk_option` + `_factor_k` |
| collapse | splices a computed edge inline (`Fold.demoted()`), removing its schedule site | `_demote_planar` / `_demoted_warp_option` |
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

Note `a602874a` deleted `map_tile_moves` / `reduce_moves` / `wspec_moves` from `space.py`, so the map-tile and
reduce families have no surviving catalog and must be reconstructed with the rest.

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

`Placement` is a `TileOp` field, not a schedule slice, and `_factor.factorize` threads `place.grid` / `place.free`
into `Ctx`. Each family constructs it at materialization, from the row's own slices — the flash warp shrink
(`ceil_div(um·fm·atom_m)`, value axis dropped, `Window(parent=...)`), the chain grid truncation, the strip's
re-derived `free`. Correction to an earlier revision: **split-K adds no placement** — `_splitk_option` reuses `place`
verbatim, `_factor_k` puts `ksplit` on the fold's axis, and `030_split_reduce` owns the grid.

Grid RANK varies with the row on the chain and warp-flash families. That is fine when placement is built at
materialization; it is exactly what made "placement is a projection of a solved point" untenable in the previous
revision. State the obligation: every placement construction is a closed-form function of (row, term).

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

Three tooling facts that decide what a gate can mean:

- **`tests/xfail_registry.py` holds 107 ids**, applied `strict=False` — a restored test passes UNNOTICED. On this
  box the suite reports 605 skips, concentrated in the scheduler-dependent modules (`test_attention_coverage.py`
  120 tests, `test_matmul_coverage.py` 142). "Registry empty" is not a completion criterion.
- **`scripts/digest_kernels.py` runs green and prints garbage.** At HEAD: exit 0, 24 lines, but only **15 distinct
  digests** — `matmul_scalar`/`warp_tma`/`splitk` collide, as do `warp_f16acc`/`raster`/`wspec`,
  `norm_linear`/`_splitk`/`_coop`, `flash_hd128`/`_cp`, `flash_hd256_alt`/`_fm`, and `flash_chain`/`flash_scalar`.
  The pins are ignored and the un-recognized escape renders. No baseline exists in the tree. At `e27d8fdc^` all 22
  emitted digests are distinct and four `__partial` kernels appear.
- **The materializer has a live regression on the flash warp path.** `_twist.py:315` reads `qk.acc` where
  `qk: TilePlan` (fields: `atom, units, regs, bk, axes`). ONE site — 544/548 take a `Contraction`, which has the
  property. Four `digest_kernels` cases error with `AttributeError` **even at the pre-deletion commit**, so the flash
  warp path has no obtainable baseline until it is fixed.

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
because `validate` checks smem ONLY (the thread and CTA checks are stubbed "pending rebuild"; `max_threads_per_cta`
appears once in the tree, inside that docstring). COMPLETENESS: every row that should exist is emitted — covered by
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

**P0 — make the harness honest (independent, land this week).** Fix `_twist.py:315`. Add a liveness assertion to
`digest_kernels.py` (each case asserts its pinned knobs landed on the emitted `TileOp`) AND a nonzero exit on any
`<ERROR>` line — today it exits 0 with errors printed as data. Commit a baseline from `e27d8fdc^` (needs
`PYTHONPATH=.` from a worktree, and record which mode it runs in — the deploy/tune reduce narrowing differs). Flip
the registry to `strict=True`. Write the full-corpus golden-reachability loop over `GOLDEN_CONFIGS` asserting
`evaluate_golden(...).rank is not None` — ~10 lines on existing API; it xfails until phase 2 and then gates for
free. **Nothing here depends on the design.**

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
- **Knob KEYS and VALUE spellings are frozen.** No codec exemption is taken.
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

The acceptance corpus. Dumps live in the pre-deletion kernel dumps and `scripts/digest_kernels.py`'s case list.

| # | expression | term | what the emitters must produce |
| --- | --- | --- | --- |
| 1 | `randn(8,512,1024).sum(-1)` | root `Fold`, PLANAR | the reduce family: conservative COOP first (not serial — the per-family exception), then the catalog, then guarded `g<n>`/`r<n>` |
| 2 | `rms_norm(...)` | `Map(fn, sources=(Fold,))` + sweep `Store` | fold rows + the sweep; the shared-row `sync` stage is a stage value on a twice-read materialized edge, gated on a contraction tail, and is NOT a knob |
| 3 | `softmax(...)` | `Map` over TWISTED `Fold` | identical to 2 — twisted changes combine COST, never which values are legal |
| 4 | `A @ B` | root `Contraction` | the tile × stage × reduce × raster product — the arm that already worked |
| 5 | `relu(A@B + bias)` | `Map` over `Contraction` | example 4 plus one filter (the fragment-epilogue gather refusal) |
| 6 | SwiGLU gate⊗up | `Map(combine, sources=(Contraction,))`, arity 2 | the computed-A rows; the cone's stat fold spells against the CONTRACTION tree; the `020` merge is a row union |
| 7 | `silu(x) * y` | `Map(sources=())` + `Store`s | grid-map + the strip rewrite, where `r` is the spelled `TILE` value |
| 8 | causal SDPA | `Map` over TWISTED `Fold`, computed QK + derived PV | five fold families as ROWS: warp streaming (row 15: `WN = 1`), chain (`WORK=""`), per-cell (the collapse variant), coop, split-KV — the same `g<n>` move as matmul split-K |

Not covered by these eight and not to be dropped: the `PLACE=cut` re-recognized pieces (18 golden rows, each its own
kernel), `rope` / `embedding` (which record `{WORK: ''}` and nothing else — a zero-choice enumeration, benign but it
must not crash), `lm_head`'s coop-cap lift, `qknorm.k256/k512`, `mlp_ch`, `mlp_geglu`, `mlp_gate_up_split`, and
every `.dynM` twin. The completeness test must also FILTER the 57 golden rows carrying non-scheduler keys
(`FAST_EXP`, `INTERLEAVE_LOADS`, `VECTORIZE_LOADS`, `LOOPIFY`, `PLACE`) — those come from `lowering/kernel` passes
and recognition, not from this enumeration.

## Risks

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
