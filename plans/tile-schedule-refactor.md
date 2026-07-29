# Tile IR + schedule refactor: generalized nodes, tree-path knobs, PLACE as a per-seam edge property

Single master plan — supersedes `knob-tree-path-codec.md` and `contraction-shared-operand-refactor.md`
(both deleted). Context: branch `feature/remove-place-knob` deleted the old PLACE machinery
(`020_cut_edge` / `025_sink_row_reduce` / `032_fuse_finalize` / `_sink.py`; `PLACE` scrubbed from
`search/space.py`; gemma golden PLACE keys survive only as comments). Nothing consumes placement seams
right now, so the IR and codec change first, then the schedule functionality is restored on top.

## Target design

### IR (phase 1)

- **Let-bound sharing.** `TileOp.bindings` (name → node tree) + a `Ref(name)` leaf. Sharing is structural
  and discoverable; a let-tree, never an implicit DAG — every tree walk gains one `Ref` case, nothing needs
  DAG traversal. `rewrite` renames binding names like SSA so `structural_key` canonicalizes sharing.
- **Sibling contractions replace `folds` channels.** `Contraction` drops `folds` → scalar `b_load` / `acc`.
  The fused gate⊗up edge = N sibling contractions under `Map.sources` (now a tuple; `source` = len-1 compat
  property) sharing an A `Ref`. Fused sibling groups schedule as ONE unit (one shared TilePlan/Stage/
  ReducePlan row — channels agree by construction) → the codec needs no sibling ordinals. The group loop /
  N-component split carrier are DERIVED at lower/split time (derive-never-store), byte-identical to today.
  `b_trans`-must-agree moves from node assert to group-formation gate.
- **Cone nodified.** `a_operand: Load | Body` → `a: Load | Ref`; the computed cone becomes a bound
  `Map(body=scale, source=Reduction(stat))`. The stat reduce is now addressable; `stat_prologue()`'s
  body-split becomes a read of the binding's node boundary.
- **One home for projections.** Retire `Contraction.epilogue`: EVERY projection rides a wrapping `Map`
  (a bare contraction's grid `Write` stays materializer glue). One projection seam kind for the cut realizer.
- **No root-residue schedule fields.** Retire `TileOp.tier` / `TileOp.stage`: with everything nodified they
  ride the node they schedule (030's split partials carry theirs on their `Contraction`/`Reduction`).
  `TileOp` = `op + bindings + place + workers + knobs`.
- **Stretch (only if `Ref` makes it ~free): one composition rule for nested reduces** — fold
  `Reduction.source` (split-K `Reduction ⊃ Contraction`) into "node at the head of `partial`" (flash's
  form), so `_flatten_nodes` / `.loop` / seam walkers handle one shape. Otherwise leave for later.

### Knob codec (phases 2–3)

Grammar: `FAMILY@<node-path>[.<axis>][<n>] = value`.

- Families keep their names (`TILE` / `REDUCE` / `STAGE` = node properties; `PLACE` = edge property;
  `RASTER` / `WSPEC` / `LOOPIFY` root-global, bare). Path = lowercase node-kind segments + field-edge labels
  where kind alone is ambiguous (`a` for the A-operand edge; binding names for bound subtrees). Axis = leaf
  discriminator for TILE/REDUCE/STAGE; absent for PLACE (path names the seam's child) and Map body tiles.
- **Short paths are canonical**: stampers and ALL evidence (goldens/DB/prior) store the shortest-unique
  spelling — bare family where one node is eligible, `FAMILY@<axis>` where the axis disambiguates, longer
  suffixes only on real collisions. Every live gemma golden spelling is already canonical → **no migration**.
- Suffix resolution: any unique suffix accepted at pin/parse time; ambiguous suffix = `ValueError` naming
  candidates (extends `resolve_axis`). If a future nodification makes a STORED short key ambiguous, the
  resolver fails loudly and that entry alone is re-spelled by hand (caught by the compat test, never
  silently re-keyed). Ordinal `<n>` (canonicalized traversal order) only for same-path kind+axis collisions.
- **Bare-resolution guard**: bare-family sugar resolves only over nodes the scheduler enumerates forks for.
  The cone's stat reduce stays OUT of that set (explicit `REDUCE@a.reduce.k`), so the ~46 stored bare
  `REDUCE` keys on norm_linear/geglu keep meaning the contraction fold.
- Keys stamp against the pre-placement tree; kernels derive from the cut set. A cut child re-recognizes as
  its own tree (keys re-root); axis names survive cuts, so suffix keys are cut/fuse-invariant and the
  parent-path spelling of a child decision resolves to the same evidence as the child-tree anchor.
- Shared bindings: canonical spelling from the binding root (its name); single-reference bindings may spell
  through the referencing edge (sugar).

Spellings (all ~580 live gemma golden entries — unchanged; resolution targets shown):

| Kind | Stored (today = after) | Resolves to |
| --- | --- | --- |
| matmul | bare `TILE` / `REDUCE` / `STAGE` | `contraction.k` |
| norm_linear, mlp_geglu | bare | `map.contraction.k` (one shared row per fused group) |
| flash | `TILE@dd` / `TILE@pj` / bare `REDUCE` / `STAGE` | `map.reduce.contraction.dd` / `…pj` / `map.reduce.kv` |
| rms_norm | bare `REDUCE` | `map.reduce.k` |
| bare reduce | bare `REDUCE` | `reduce.k` |
| pointwise | bare `TILE` | `map` |
| cone stat (NEW) | `REDUCE@a.reduce.k` | `map.contraction.a.reduce.k` |

### PLACE (phases 4–5)

`PLACE@<child-path> = cut | fuse` on in-tree seams, replacing the ad-hoc `@fold`/`@fin`/`@cone`/`@stat` sites.

- **`fuse` is the default on every seam; `cut` is evidence/pin-only.** The recognized tree IS the fused form
  — default means "no rewrite"; an unseeded site deploys the recognized kernel. One rule, no per-site zoo.
- Old → new: flash `PLACE: fuse` → `PLACE@map = fuse`; `PLACE@cone: cut` → `PLACE@map.contraction.a = cut`
  (cone → stat + scale + plain matmul; the parent row must resolve to the same evidence as the
  `cut_cone_stat` / `cut_cone_scale` child anchors). NEW: 3-kernel split reduce = `REDUCE@map.reduce.k=g<n>k`
  + `PLACE@map = cut` (partial → combine → separate projection), previously inexpressible.
- A cut materializes the seam value (f32 state for reduce seams, like `030`'s workspace); a cut child
  sharing operands with siblings becomes a MIMO producer (#433) instead of recompute.
- Graph-level placement (`PLACE@fin=fuse` consumer-inline; `PLACE@stat=sink` producer tap) stays OUT —
  phase 6. When `stat-tap-loop-fusion.md` lands, its tap seam joins the path namespace (old `sink`→`fuse`,
  old `fuse`→`cut`) but keeps `cut` as ITS default (evidence-only taps — measured anti-wins at qknorm /
  post_ff / m64), the one documented exception to fuse-default; its cut-out realizer becomes a client of
  the generic edge-cut pass.

## Phases

### Phase 0 — baseline capture

Capture on THIS branch (not main): `emmy eval golden --in-model` MATCH/DRIFT/GAP counts both cards;
`op_cache_key`s for the fused shapes (geglu, norm_linear, lm_head fused, flash); `EMMY_DUMP_DIR` CUDA dumps
for gemma-4 layer-0. These are the parity gates every later phase diffs against.

### Phase 1 — IR generalization (`ir/tile/ir.py`, `ir/tile/ops.py`, recognize/schedule/factorize)

1a. `Ref` + `TileOp.bindings` + `Map.sources`; strip `folds`; `a: Load | Ref`; retire `Contraction.epilogue`
    and `TileOp.tier`/`stage` (moved onto nodes). Mechanical `Ref`-threading through `ops.*` in one
    no-behavior-change commit first.
1b. Recognize-side flip: `_contraction_node` emits binding + sibling group; cone nodified (with the
    bare-resolution guard); scheduler stamps one shared row per fused group; `b_trans` gate at group
    formation.
1c. Materialize/split: `factorize` reads the group off `Map.sources` + `Ref` identity; `030` derives the
    N-component carrier from the group; re-run the #389 multichannel-split A/B (null may flip).
1d. Stretch: `Reduction.source` → head-of-partial, only if `Ref` plumbing already did the work.

Verify: group-loop `op_cache_key` byte-parity vs phase-0 capture; `make test`; eval-golden counts identical;
CUDA dumps byte-identical (or name-churn-only); accuracy on geglu/norm_linear snippets + 5090 decode twin
TPOT within noise.

### Phase 2 — codec core (`knob.py`, `search/keys.py`, `search/space.py`)

Path parse/format; tree walker in `ir/tile/ops.py` enumerating `(path, node, axis)` (single source for
stampers + resolver); `resolve_path` (generalizes `resolve_axis`) with shortest-unique + ambiguity errors +
ordinals; `family_of` / `axis_of` / `family_value` / `_FAMILY_ORDER` read through paths.

Verify: unit round-trip / suffix / ordinal tests; compat test resolving every knob dict in ALL golden YAMLs
against its kernel kind's tree unchanged.

### Phase 3 — stamp sites (`_schedule.py`, `010_recognize.py`)

Stampers emit shortest-unique spellings (byte-identical to today's keys on every current shape — DB/prior
rows match with zero translation).

Verify: `emmy compile --golden <one per kind> --ir tile` deploys the recorded config per kind, both cards;
pin-only offer audit (#435) green.

### Phase 4 — PLACE realizer (new generic pass, replaces `020_cut_edge`)

One edge-cut pass: `PLACE@<path> = cut` splits the tree at the seam → producer/consumer kernels, seam value
materialized (MIMO when shared), children re-recognized. Seams in scope: `map` (projection off any
reduce/contraction — one shape now, thanks to phase 1), `contraction.a` (the cone). Composes with
`REDUCE=g<n>k` for the 3-kernel form. `010_recognize` enumerates PLACE rows (option-0 = fuse; cut rows
evidence-gated); greedy pin precedence exact path > suffix > bare, mirroring `narrow_at`.

Verify: restored cone cut reproduces the recorded pair economics (cut_cone_stat+scale ≈ 3.8 µs vs fused
6.0 µs, 5090 YAML comments); rms_norm goldens unchanged; 3-kernel split reduce compiles + passes accuracy
on `--golden rms_norm.k3840`.

### Phase 5 — PLACE golden re-seeding

No evidence migration (short spellings canonical). Re-seed retired PLACE goldens by hand-pinned `--ab`
(manual sweep method): flash `PLACE@map`, cone cuts on norm_linear/geglu at recorded shapes, both cards.
Commented-out PLACE entries re-keyed + re-enabled ONLY behind a fresh `--ab` each — pre-wipe µs are not
evidence.

Verify: eval-golden pin-only audit green; serving twins deploy from tier; decode TPOT / TTFT within noise
of the YAML-comment baselines.

### Phase 6 — deferred follow-ups (separate plans when picked up)

Graph-level placement (fin consumer-inline — refuted e2e once, low priority; stat tap per
`stat-tap-loop-fusion.md` rebased onto the generic realizer); re-expressing `035` merged-sibling concat and
MIMO producer cuts as reference-driven decisions; LayerNorm multi-stat (N bindings, falls out free);
axis-window unification (`Reduction.offset`/`bound` vs `Axis.source_axis`/`real_extent`).

## Risks

- **Fused-lowering byte-parity (phase 1) is THE load-bearing gate** — drift re-keys kernel caches and
  invalidates golden µs. The captured-key test lands before the recognize-side flip.
- Cone nodification changes `--ir tile` dumps and structural test assertions — sweep
  `tests/compiler/passes/` early (test_structural_features and friends).
- Stored-short-key ambiguity from future nodifications — resolver fails loudly; compat test is the tripwire.
- Dump/kname churn: verify `<kname>.torch.json` reproducer slicing with the cone in a binding.

## Cleanup

Docs at the end of each landed phase: `pipeline/ARCHITECTURE.md` (knob/fork), `passes/ARCHITECTURE.md`
(tile lowering — bindings/Ref, PLACE as edge property), CLAUDE.md tile-lowering blurb (node vocabulary
changes in phase 1). Delete this plan when phase 5 lands.
