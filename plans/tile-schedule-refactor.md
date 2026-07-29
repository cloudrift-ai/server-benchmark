# Tile IR + schedule refactor: the pure Fold IR, tree-path knobs, PLACE as a per-seam edge property

## Context

Branch `feature/remove-place-knob` deleted the old placement machinery: the `020_cut_edge` /
`025_sink_row_reduce` / `032_fuse_finalize` realizer passes and `_sink.py` are gone, `PLACE` is scrubbed
from the search space, and the gemma golden YAMLs keep their old PLACE keys only as `# retired knob
dropped:` comments. Nothing in the compiler consumes placement seams right now. That is deliberate: the IR
and the knob codec are redesigned first, on the smaller surface, and the placement functionality is then
restored on top of the new vocabulary instead of being ported.

The IR design went through three iterations (let-bound sharing → the product-carrier contraction → the
pure Fold IR); this document describes ONLY the final design and the landed state. The superseded
intermediates survive as one list under *Landed trail* — do not resurrect their vocabularies.

## North star

**The recognized algebra tree is the single semantic object. Everything else is a decoration on it or a
derived view of it.**

A kernel's math is the `project ∘ reduce ∘ map` composition stored as structural nodes in `ir/tile`. The
design rests on four ideas:

1. **ONE general fold node.** `Reduction` and `Contraction` are one kind: a `Fold` with an iteration
   axis, a carrier (the ⊕ monoid on an N-component product state), a symmetric tuple of **operand
   edges**, and a **step** (the per-step lift + carrier update, a sequence). A matmul, a bare sum,
   RMSNorm's statistic, the fused gate⊗up edge and flash are all this node at different carrier arities
   and roles. Sharing is **edge reuse**: "these two matmuls read the same A" is one fold whose step reads
   one operand edge twice — the tuple lift `(a·b, a·c)` into a product carrier. No privileged operand
   slot, no channel pairing, no name-reference mechanism.
2. **An operand edge has two inhabitants** — MATERIALIZED (a gmem `Load`) or COMPUTED (the node itself,
   stored inline; tree ownership gives it exactly one consumer). WHERE an input attaches is decided by
   one rule, **edge iff closed**: a subtree that captures no value from its enclosing step
   (`ir.captured_values`, iteration variables excluded) is an operand edge; a state-capturing
   composition (flash's PV, whose multiplicand `P` the same loop step's merge produces) sits in the step
   sequence at its semantic position. One rule decides attachment, the canonical direction (hoist
   closed subtrees), and cut legality at once: on the cut lattice — nodes reachable from the root
   through edges alone — every seam is closed by construction. A closed STEP element is equally
   legal and cuttable in place, and stays put where hoisting would perturb the lowered nest (flash's
   QK — see the worked example). (An IN-STEP node's own operand may capture from the step scope, as
   flash's `P` does; it sits below an uncuttable step anyway.)
3. **Hardware readings are DERIVED views, never stored.** The bilinear factored form `a·(b₁…bₙ)` that
   tensor cores require is `contraction_view` — computed at fork-emit where a consumer exists, absent
   from storage. Three design-level reasons it is a view and not a rewrite: the same fold must keep
   offering its scalar/coop rows as fork siblings beside the warp rows; one relation keeps one spelling
   (a stored refined kind beside the general fold is the two-spellings defect); and knob keys stamp
   against the pre-specialization tree.
4. **Kernel boundaries are a placement DECISION** — a `cut | fuse` bit on each parent↔child seam.
   Kernels are a derived view: the cut set partitions the tree, each partition materializes as one
   launch, each cut seam materializes its value to a buffer.

From this everything else follows:

- **Knobs are paths into the tree.** A schedule key addresses the node (or edge) it decorates by
  position; no parallel namespace of hand-invented site names.
- **Keys stamp against the pre-placement tree.** A cut never re-keys the decisions inside either half.
- **Derive, never store.** Loop nests, carriers, tile geometry, the contraction view — synthesized on
  demand from the params. Stored state is only the params + decisions.
- **Defaults are the recognized form.** An unseeded shape deploys exactly what recognition produced;
  every rewrite away from it is evidence- or pin-driven ("an unseeded site never pays").
- **All fusion happens in loop IR; tile IR only cuts.** Loop-level fusion decides what lives together;
  recognition STRUCTURES that output, never merges more. Hard direction invariant — anything wanting
  more fusion is realized at loop level with a tile-level cut escape, never a tile-level merge.
- **Identity keys off the LOWERED loop nest** (`structural_key` lowers first), so stored node shapes are
  free to change as long as the derived loop is byte-identical — the kernel-source digest harness
  (17 golden kernels: scalar/warp matmul, split-K, coop-t matvec, norm_linear, norm_gate_up `.lin`,
  mlp_geglu, lm_head, rms/softmax/reduce/pointwise, flash hd128/hd256, dynM forms) is the gate for
  every migration step.

## The IR

```python
Operand = Load | Fold | Map        # materialized | computed (inline node, one consumer by tree ownership)

@dataclass(frozen=True)
class Fold(Stmt):
    axis: Axis                     # the reduced iteration space
    carrier: Carrier               # the ⊕ monoid on an N-component product state
                                   #   family "id"  → componentwise ⊕ (sum, max, the gate⊗up product)
                                   #   family "exp" → the twisted action (online softmax / flash)
    # role is DERIVED (1l), never stored: TWISTED ⟺ non-id twist family; CONTRACTION ⟺ the
    # bilinear parse of (operands, step) succeeds, or the step composes the sliced split-K fold;
    # PLANAR otherwise — the matvec demotion is the PLANAR arm (unbindable ⇒ loads stay inline)
    step: Body                     # the lift + carrier update, reading operand values
                                   # by their out-names. A SEQUENCE on purpose: a twisted lift may read
                                   # state its own step updates (flash's P) — algebra, not an emitter
                                   # limitation
    operands: tuple[Operand, ...]  # the CLOSED inputs, one edge each; sharing = repeated reads in step
    # schedule slices (decorations; composition supplies multi-slice forms — split-K is an outer
    # reduce-partitioned fold whose step composes the inner tiled one):
    tile: TilePlan | None          # role=CONTRACTION only
    reduce: ReducePlan
    stage: Stage | None

@dataclass(frozen=True)
class Map(Stmt):
    body: Body                     # projection / pointwise sweep
    sources: tuple[Operand, ...]   # project ∘ fold; must admit Load — the cut terminal (phase 4)
```

**Lowering rule.** `Fold.loop` splices each operand edge's body immediately before its first use in the
step (ties in tuple order — `ir._splice_operands`), then flattens nested nodes in place. Deterministic,
so the derived loop — and with it `op_cache_key` / kernel identity — depends only on the stored params.

**The contraction view** — derived at fork-emit, never stored:

```python
def contraction_view(fold, m_axis, n_axis, lead_axes) -> ContractionView | None:
    ...  # role=CONTRACTION + the step factors as one multiply per component sharing a common factor
         # (`ir._parse_bilinear`); the OUTPUT axes are the CALLER's placement facts (trailing grid for
         # a root kernel), which is why they are parameters and not fold fields
```

The `ContractionView` dataclass IS the view: one shared `a` edge, `(b, acc)` channels, the `(m, n)` `Side`
geometry, `b_trans` — the whole reading the warp/staged tiers and `_atom`/`_factor` consume.
`ContractionView.as_fold()` is the storage direction; the round-trip `contraction_view(as_fold(v)) == v` and
`as_fold(v).loop == v.loop` (byte-identical) are unit-tested. `ir.shared_operand(fold)` is the
placement-free read for callers that need only the shared edge (the cone).

**Pipeline:**

```
Loop IR ──recognize──▶ pure Fold/Map tree              (structure only; no hardware knowledge)
        ──schedule───▶ scalar / coop rows read the fold directly;
                       tensor cores present? view = contraction_view(fold, place)
                       view ─▶ warp / staged rows: operand roles from the view, TilePlan on the fold
        ──split / materialize──▶ slices off the nodes, roles off the view (re-derived from ctx.grid)
```

Worked shapes:

```
RMSNorm:   Map(body=[rsqrt stat + sweep + Write],
               sources=(Fold(axis=k, sum, operands=(Load(x[m,k]),), step=[square, accumulate]),))

gate⊗up:   Map(body=[swiglu(acc_g, acc_u) + Write],
               sources=(Fold(axis=k, product("acc_g","acc_u"),
                             operands=(Load(Wg), cone, Load(Wu)),   ← cone read by BOTH lifts
                             step=[vg = wg·xn; acc_g += vg; vu = xn·wu; acc_u += vu]),))
           cone = Map(body=normalize, sources=(Fold(stat),)) — inline; the K seam is the node boundary

Flash:     Map(body=[O/l projection + Write],
               sources=(Fold(axis=kv, exp("m","l","O"),
                             step=[QK-fold; …merge; P = exp(s′−m′); PV-fold; O = O·α + O_blk…])))
           PV captures P → a step element at its semantic position. QK is CLOSED and therefore
           cuttable in place, but stays a step element too — hoisting it to an operand edge would
           REORDER the lowered nest (the scale Load precedes the score's first use, so the
           first-use splice lands QK after it), re-keying every flash kernel against zero-migration.
           Edge-hoisting closed steps is optional canonicalization, deferred to a re-keying window.
```

A projection has ONE home — the wrapping `Map.body`, never a node field; a bare fold's grid `Write` is
materializer glue. Every schedule slice rides the node it decorates; `TileOp` keeps only
`op + place + workers + knobs`; a sliced axis's window is the one `Axis.window` vocabulary.

## Landed trail (compressed history — the vocabularies below are RETIRED)

- **1a–1g** built the structural-node tree (typed reduce, nodified contractions, cone as a node tree,
  one composition rule, `Axis.window`, closure predicate, every node a `Stmt`, resolve-as-splice) via a
  let table (`TileOp.bindings`, name-referenced sharing). Retired by 1j.
- **1h** made both contraction operand edges the same union; the A/B asymmetry moved to schedule gates
  (`isinstance(c.b, Load)` eligibility preconditions) — that stance survives, now via the view's roles.
- **1j** replaced let-bound sharing with the product-carrier `Contraction` (one `a` edge + `(b, acc)`
  channels; sharing as arity) and deleted the whole reference apparatus (`bindings`,
  `validate_bindings`, `ops.resolve`, rename-lockstep, `is_group`/`group_loop`). 17/17 digests
  identical. The `Contraction` dataclass survives as the 1k VIEW; its `Stmt`-hood is a 1k-iii removal.
- **1k-i** (landed): `operands` + the first-use splice + `tile` slot on the fold node; walks and the
  rewrite canonicalizer cover the edges.
- **1k-ii** (landed): every ROOT contraction is stored as a `role=CONTRACTION` fold —
  recognition/`_schedule` build the view and store `view.as_fold()`; `_schedule._contraction_node`,
  `_factor._bind` and `030_split_reduce` re-derive the view from the fold + the placement's trailing
  grid axes (a split partial's lead axes fall out of its own grid — nothing restamped on the node);
  `nodify_reduce` drops `tile` when the coop/ILP K partition takes over. 17/17 digests identical;
  round-trip unit-tested (`tests/compiler/ir/tile`).
- **1k-iii** (landed — phase 1 CLOSED on the pure-Fold IR):
  - *Flash storage*: QK and PV store as in-step `role=CONTRACTION` folds (PV's `P` cone a capturing
    operand — legal below an uncuttable step; QK is closed and cuttable in place, but hoisting it to
    an operand edge would REORDER the lowered nest — the scale `Load` precedes the score's first use
    — so under zero-migration it stays a step element; edge-hoisting closed steps is deferred to a
    re-keying window). Consumers derive views: `_twisted_pair` takes `free` and returns the stored
    folds + the views (stamping targets folds by identity, reads go through views);
    `030._split_twisted_warp` reads `(m, d)` off `tile.place.free` and hands the split partial a
    placement whose `free` keeps the true `(m, d)` tail; `place.free` threads through `Ctx.free` so
    `realize_warp_twist` / `_realize_chain` derive views in the materializer (the score view's
    stream axis reads through a slice partial's `window.parent`). Verified by the 17-kernel digest
    A/B plus a stash A/B on the hd512.s2048 split-KV golden — all byte-identical.
  - *`Contraction` de-Stmt'd*: a plain frozen dataclass (the view); every stored-tree walk is
    `Fold`/`Map`-only; `ops.lower`/`reduce_loop` keep a view-convenience arm.
  - *The rename*: the schedule enum became `FoldMove`; `Reduction` → `Fold` and the `partial` field
    → `step` across the compiler, tests and docs; `Contraction` → `ContractionView` (the name now
    announces the lifecycle). Digests unaffected (identity keys off the lowered nest).
- **1l** (landed): `Fold.role` retired — the role is a DERIVED property (TWISTED off the carrier's
  twist family; CONTRACTION off the bilinear parse of the hoisted `(operands, step)` pair, the
  `_parse_bilinear` role gate dropped, or the composed split-K step; PLANAR otherwise). The
  matvec DEMOTION relocated out of recognition: an unbindable contraction just keeps its loads
  inline (no operand hoist, no role rewrite — `_nodify_contraction` / `_demote_planar` are plain
  `Fold.from_loop`), derives PLANAR and takes the reduce tiers at schedule dispatch. Verified by a
  transient stored-vs-derived assert across the suite (only hand-built test folds diverged) and
  the 17-kernel digest A/B — all byte-identical, `down_proj.m1.t` (the demoted-matvec row)
  included; the lowered `Loop` annotations reproduce exactly (the loop-level `AxisRole` stays
  stored — `contraction_loop` marks CONTRACTION, `semiring_binding` scans for it).

## Knob codec (phases 2–3)

Grammar: `FAMILY@<node-path>[.<axis>][<n>] = value`.

- **Families keep their names** — `TILE` / `REDUCE` / `STAGE` select which slice decorates the addressed
  fold (the family IS the slice); `PLACE` is the edge property; `RASTER` / `WSPEC` / `LOOPIFY` stay
  root-global and bare. Full backwards compatibility of the outer key shape.
- **Path** = lowercase node-kind segments from the root. With one fold kind the segments collapse to
  `map` / `fold` and carry little — the **axis** is the real discriminator (the leaf key for
  TILE/REDUCE/STAGE); operand-edge labels (`a` / `b`, and `in.a` / `in.b` in the reserved graph prefix)
  are VIEW-ROLE sugar resolved through `contraction_view`, positional edge addressing the general
  fallback. Phase 2 designs the segment grammar on the one-kind tree.
- **Short paths are canonical.** Stampers and ALL stored evidence use the SHORTEST spelling unique for
  the kernel's tree. Every live golden spelling is already canonical under this rule → **zero
  migration** of goldens/DB/prior. Any unique suffix is accepted at pin time; ambiguity raises naming
  candidates; a future structural change that breaks a stored short key fails loudly and that entry is
  re-spelled by hand (the compat test is the tripwire — never silently re-key). The ordinal `<n>`
  (canonicalized traversal order) exists only for true same-path collisions; current kernels never
  need it.
- **Bare-resolution guard**: bare-family sugar resolves to the PRIMARY (root-most schedule-bearing)
  node — so bare `REDUCE` on norm_linear/geglu still means the contraction's K fold and the ~46 stored
  bare keys keep their meaning; the cone's stat fold is addressed explicitly. A resolution rule, not an
  enumeration limit: non-primary nodes are fully part of the fork space under explicit spellings.
  Nodifying something must never change what existing spellings mean.
- **Only the pre-placement tree receives keys.** Post-rewrite artifacts (split partials, synthesized
  finalizes) are never key targets; a cut child re-recognizes as a fresh tree and keys normally.
- **Cut/fuse invariance**: axis names survive cuts, so a suffix key names the same node on both sides of
  a placement decision, and a child kernel's evidence is shape-transferable (a cut-out stat kernel at
  (M, K) is the same kernel whatever parent it was cut from).
- **RESERVED grammar (reject cleanly, never reuse):** graph-level placement spells as value-centric
  placement — `own | consumer | producer` over the `in.<operand>` path prefix and the leading-`=`
  value-name pin form. The parser recognizes and rejects them ("reserved for graph-level placement");
  absolute SSA/tensor names are pin-time sugar only, never stored. Whoever restores them must realize
  `consumer`/`producer` as LOOP-LEVEL fusion with a tile-level cut escape — the all-fusion-in-loop-IR
  invariant applies to them too.

Spellings on the live gemma goldens (~580 entries — all unchanged):

| Kind | Stored (today = after) | Resolves to |
| --- | --- | --- |
| matmul | bare `TILE` / `REDUCE` / `STAGE` | the fold's k |
| norm_linear, mlp_geglu | bare | the product fold's k (one row per node) |
| flash | `TILE@dd` / `TILE@pj` / bare `REDUCE` / `STAGE` | the QK edge / the PV step fold / the kv stream |
| rms_norm / bare reduce | bare `REDUCE` | the fold's k |
| pointwise | bare `TILE` | the map |
| cone stat (NEW) | `REDUCE@a.fold.k` | the cone's stat fold (the SAME `k` name the outer fold contracts — the path disambiguates; the strongest motivation for path addressing) |

**Phase 2 — codec core.** ONE tree walker enumerating `(path, node, schedule-bearing axis)` triples,
shared by the resolver, the stampers (phase 3) and the seam enumerator (phase 4); the resolver
generalizes `resolve_axis` (idempotent, total over the sugar forms). Done when unit tests cover
round-trip / every sugar level / ambiguity errors / ordinal emission / reserved-form rejection, and a
compat test resolves every knob dict in ALL golden YAML files against its kernel kind's tree, asserting
the stored spelling is already canonical.

**Phase 3 — stamp sites.** The scheduler's fork rows spell keys via the resolver, byte-identical to
today's spellings on every current shape (any spelling change is a resolver bug, not a migration). Done
when `--golden` deploys the recorded configs on both cards' golden sets, the pin-only offer audit stays
green, and `eval` tooling shows unchanged knob rows.

## Placement (phases 4–5)

`PLACE@<child-path> = cut | fuse` on every in-tree parent↔child seam.

- **`cut`**: split the tree at the seam — the child subtree becomes its own graph node (re-entering
  recognition as a fresh tree), the seam value materializes to a buffer (f32 for reduce seams,
  mirroring the split-reduce workspace rule), the parent consumes a plain `Load` where the child was
  (which is why every edge must admit `Load`; the `Map.sources` widening is 1i, deferred to the
  realizer's own commit). An in-tree cut child always has exactly ONE consumer (inline operands are
  single-consumer by tree ownership; the shared A is one edge), so the realizer needs no MIMO case —
  #433 stays a graph-level foundation.
- **`fuse` is the default on every seam; `cut` is evidence/pin-only.** The recognized tree IS the fused
  form, so the default means "no rewrite" — the deployment-safety invariant.
- There are exactly two seam shapes: a `Map` projection seam and a fold operand edge — and every edge on
  the cut lattice is closed by construction (edge-iff-closed), so seam legality is structural. The
  realizer should know NOTHING about which seam it is — seam-specific knowledge must fall out of the
  node kinds.
- Cut composes with the split rewrite (split first, then cuts — cuts operate on post-split trees whose
  keys are the pre-placement spellings). The cone cut's payoff is the stat kernel + scale kernel +
  plain matmul; at arity N the cut lands on the N-component `Load`-A fold — per-component separation is
  deliberately forfeited at tile level (de-fusing components is loop IR's decision; #389 measured null).
- Old → new: flash's `PLACE: fuse` → `PLACE@map = fuse`; `PLACE@cone: cut` → the `a`-edge cut. NEW,
  previously inexpressible: the 3-kernel split reduce (`REDUCE@…k = g<n>k` + `PLACE@map = cut`).
- **Honest evidence accounting**: cut-vs-fuse is judged on the parent row (N-kernel total vs the fused
  kernel), each child's schedule evidence lives on its own child-tree anchor (shape-transferable), and
  both spellings of a child decision resolve to the same evidence.
- **Graph-level placement stays out of scope** (old `fin=fuse` / `stat=sink` crossed graph edges); the
  codec merely reserves its grammar. If the stat-tap plan lands later, its seam joins this namespace but
  keeps `cut` as its default (measured anti-wins) — the one exception to fuse-default.

**Phase 4 done when**: the cone cut reproduces the recorded pair economics (~3.8 µs pair vs 6.0 µs fused
on the 5090 per the YAML comments); rms_norm deploys unchanged under default fuse; the 3-kernel
split-reduce form compiles and passes accuracy.

**Phase 5** carries THE consolidated parity gate for the whole refactor: re-seed the retired PLACE
goldens by hand-pinned `--ab` sweeps (the manual method — the tuner is not used for golden work; both
cards; pre-wipe µs are not evidence), then `emmy eval golden --in-model` MATCH across the board,
pin-only offer audit green, serving twins deploy from tier, decode TPOT / TTFT within noise of the
YAML-comment baselines.

## Completeness (proof sketch)

- **Legal cut points = node outputs = the edge set.** A cut along `v` is legal iff every element of `v`
  is complete before any consumer in the other kernel reads it; those are exactly the bound outputs of
  finished operators — the tree's edges. Enumerating SSA names over-generates candidates whose legality
  check is the tree, precomputed; with edge-iff-closed, legality is built into the edge set itself.
- **The only quantization loss is mid-pointwise cuts, and it is schedule-trivial**: an interior cut
  inserts a gmem round-trip between memory-bound stmts and never changes schedule class; anything that
  ever mattered is re-association (`Map ∘ Map`) — a change of algebra, not codec. Every cut the old
  system deployed lands on a node seam in the new trees.
- **The fork space is a finite, generically enumerable product**: Π schedule-family vocab per
  (path, node, axis) site × Π {fuse, cut} per seam (legality structural — carrier materializability,
  f32 workspace, no graph-output crossing) × root-globals, with the reserved graph-level triple also
  finite. The site registry is DERIVED from one walker; only per-family vocabularies and legality gates
  stay hand-written. Enumerable ≠ cheap: cut sets compose (2^seams), so enumeration stays prior-ranked,
  and evidence-only-`cut` keeps the space unpaid-for cold.

## Risks

- **1k-iii flash churn**: the twist realizer is the subtlest emitter; the QK step→edge hoist must not
  perturb lowered stmt order (the head splice is order-preserving there, but prove it by the flash
  digests, including split-KV), and view-side axis derivation must reproduce the stored axes exactly. A
  divergence is a bug in the view derivation, never something to migrate around.
- **Parity is settled once, at phase 5** — intermediate phases carry only unit-level verification plus
  the digest harness; if the eval-golden pass surfaces broad drift, bisect back to the phase-1 commits
  rather than patching goldens forward.
- Stored-short-key ambiguity from future structural changes — the resolver fails loudly by design; the
  phase-2 compat test is the tripwire. Never "fix" it by silently re-keying evidence.
- Dump/kname churn: kernel names derive from realized ops; verify the per-kernel torch-reproducer
  slicing still attributes the cone's ops correctly from the operand edge.

## Cleanup

Docs at the end of each landed phase: the pipeline ARCHITECTURE (knob/fork system), the tile-lowering +
kernel ARCHITECTURE files, and CLAUDE.md's tile-lowering blurb. Delete this plan when phase 5 lands.
