# Tile IR + schedule refactor: the λ-foldMap IR, tree-path knobs, PLACE as a per-seam edge property

## Context

Branch `feature/remove-place-knob` deleted the old placement machinery: the `020_cut_edge` /
`025_sink_row_reduce` / `032_fuse_finalize` realizer passes and `_sink.py` are gone, `PLACE` is scrubbed
from the search space, and the gemma golden YAMLs keep their old PLACE keys only as `# retired knob
dropped:` comments. Nothing in the compiler consumes placement seams right now. That is deliberate: the IR
and the knob codec are redesigned first, on the smaller surface, and the placement functionality is then
restored on top of the new vocabulary instead of being ported.

The IR design went through four iterations (let-bound sharing → the product-carrier contraction → the
pure Fold IR → the λ-foldMap IR: `Lambda` lifts + the carrier update derived from the monoid
presentation); this document describes ONLY the final design and the landed state. The superseded
intermediates survive as one list under *Landed trail* — do not resurrect their vocabularies.

## North star

**The recognized algebra tree is the single semantic object. Everything else is a decoration on it or a
derived view of it. A kernel IS `π ∘ foldMap`** — a projection over a monoidal fold of a pure
per-element lift: `⟦Fold⟧ = ⊕_{k ∈ axis} ι(lift(k))`, seeded at `e`. The design rests on five ideas:

1. **ONE general fold node — axis + monoid + lift.** A `Fold` stores the reduced `axis`, the ⊕
   `Monoid` presentation (the product `components` — the `ElementwiseImpl` ⊕ᵢ ops themselves, each
   eᵢ its `identity` trait; the singleton embedding `inject : A → S`; and the twist family `psi` —
   transport of structure: `⊕_ψ` is generated from the
   presentation, never spelled), a PURE `lift` `Lambda` (`λ(k, v₁…vₙ) → A` — it sees the iteration var
   and the operand values, nothing else), and one CLOSED operand edge per lift param. ALL state-reading
   lives inside ⊕: the carrier update — the stabilized streaming step, the cross-partition combine, the
   identity seeds — is DERIVED from the presentation at lowering, and nothing dissolved is ever stored,
   so the step/carrier duplication (and its rename-lockstep bug class) does not exist. A matmul, a bare
   sum, RMSNorm's statistic, gate⊗up and flash are all this node at different monoid arities and
   twists. Sharing is still **arity**: the tuple-valued lift into the product monoid; in-lambda sharing
   is ANF (every `Assign` is a let). Flash's `P = exp(s − m′)` reading the max its own step just
   updated is the derived INLINED evaluation of `s ⊕ ι(a)` — an operational form, not storage; the
   "legal but uncuttable state-capturing step element" quirk evaporates with it.
2. **`Lambda` = explicit binders over the REUSED stmt vocabulary — no second expression language.**
   `Lambda(params, body, results)`: `params` the binders, `body` a `Body` of PURE stmts only
   (`Load` / `Assign` — A-normal form ≙ a let-chain), `results` the returned defs (replacing every
   `out` / last-def convention). Purity is a FORMATION invariant, not a new type universe: no `Accum`,
   `Write` or `Loop` inside a lift; free names ⊆ params ∪ iteration vars; results defined. α-invariance
   is canonical renumbering (the existing rename machinery), not de Bruijn. **Primitives live in their
   home modules** — they are not tile-IR-private: `Lambda` beside `Body` in `ir/stmt` (NEVER in
   `tile/ir.py`); `Monoid` in `ir/stmt/algebra` (the finished form of `Carrier`/`Twist` — components
   are the ⊕ ops, `inject` an explicit `Lambda`, bound mode surviving for the one-shot `StateMerge`);
   scalar/index exprs stay `ir/expr`; `Fold`/`Map`/`TileOp` and the derived views stay `ir/tile`.
3. **Edge iff closed — by CONSTRUCTION.** An operand edge has two inhabitants — MATERIALIZED (a gmem
   `Load`) or COMPUTED (an inline node; tree ownership gives it one consumer) — and operands bind
   POSITIONALLY to lift params, so an operand cannot see the fold's state or its siblings: closure
   needs no scan (`captured_values` demotes to a validation assert), every seam on the cut lattice is
   closed structurally, and composition is nesting a term on an edge. Operands may read the ENCLOSING
   iteration var (flash's per-key score fold) — never state. Effects sit at the kernel boundary only:
   `results` + materializer glue synthesize every root store; no `Write` inside a term.
4. **Hardware readings are DERIVED views, never stored.** The bilinear factored form `a·(b₁…bₙ)` the
   tensor cores require is `contraction_view` — a factoring of the LIFT (`_parse_bilinear` reads the
   lambda body), computed at fork-emit where a consumer exists, absent from storage; the role (landed
   1l) is a predicate — ψ non-id / lift bilinear — not even a derived enum in the end state. Three
   design-level reasons a view and not a rewrite: the same fold must keep offering its scalar/coop rows
   as fork siblings beside the warp rows; one relation keeps one spelling (a stored refined kind beside
   the general fold is the two-spellings defect); and knob keys stamp against the pre-specialization
   tree.
5. **Kernel boundaries are a placement DECISION** — a `cut | fuse` bit on each parent↔child seam.
   Kernels are a derived view: the cut set partitions the tree, each partition materializes as one
   launch, each cut seam materializes its value to a buffer (a cut = let-materializing a closed term).

From this everything else follows:

- **Knobs are paths into the tree.** A schedule key addresses the node (or edge) it decorates by
  position; no parallel namespace of hand-invented site names. Explicit binders make paths stable by
  construction.
- **Keys stamp against the pre-placement tree.** A cut never re-keys the decisions inside either half.
- **Derive, never store.** Loop nests, the carrier update (the stabilized ⊕_ψ step), roles, tile
  geometry, the contraction view, cross-CTA splits (reassociation `fold_k = fold_{k₁} ∘ fold_{k₂}`,
  legal by ⊕'s stored associativity) — synthesized on demand from the params. Stored state is only the
  params + decisions.
- **Defaults are the recognized form.** An unseeded shape deploys exactly what recognition produced;
  every rewrite away from it is evidence- or pin-driven ("an unseeded site never pays").
- **All fusion happens in loop IR; tile IR only cuts.** Loop-level fusion decides what lives together;
  recognition STRUCTURES that output, never merges more. Hard direction invariant — anything wanting
  more fusion is realized at loop level with a tile-level cut escape, never a tile-level merge.
- **Identity keys off the LOWERED loop nest** (`structural_key` lowers first), so stored node shapes are
  free to change as long as the derived loop is byte-identical — the kernel-source digest harness
  (17 golden kernels: scalar/warp matmul, split-K, coop-t matvec, norm_linear, norm_gate_up `.lin`,
  mlp_geglu, lm_head, rms/softmax/reduce/pointwise, flash hd128/hd256, dynM forms) is the gate for
  every migration step. The END-STATE identity is the α-invariant hash of the canonically renumbered
  term itself; switching to it is a re-keying event and rides the same deferred re-keying window as the
  other order-canonicalizations — never a silent re-key.
- **The λ formulation is also the executable SPEC**: a ~20-line denotational `foldMap` evaluator plus
  an agreement test (`⟦recognized tree⟧ == lowered loop` on random inputs) pins the algebra — ⊕
  associativity, ψ-transport, the flash monoid — so every purification step refactors toward an oracle
  that already runs.

## The IR (target)

The primitives are NOT tile-IR-private — each lives in its home module, importable by any IR layer:

```python
# ── ir/stmt (beside Body) — the ONE binder kind, common to every IR level ──────────────────────────
@dataclass(frozen=True)
class Lambda:
    params: tuple[str, ...]          # explicit binders — closedness by construction
    body: Body                       # PURE stmts only (ANF ≙ a let-chain); __post_init__ VALIDATES the
                                     # LOCAL invariant: every stmt passes the `Stmt.pure` trait (declared
                                     # on the Stmt interface, conservative default False — Load/Assign
                                     # opt in; Accum/Write/Init/Loop never do; no isinstance whitelist)
                                     # and every result is defined. The CONTEXTUAL half — free names ⊆
                                     # params ∪ enclosing iteration vars — is the consuming node's check
                                     # (Fold/Map formation), since a bare Lambda can't know its scope
    results: tuple[str, ...]         # the returned defs — replaces every `out` / last-def convention

# ── ir/stmt/algebra — the finished form of Carrier/Twist ──────────────────────────────────────────
@dataclass(frozen=True)
class Monoid:
    components: tuple[ElementwiseImpl, ...]  # the ⊕ᵢ ops THEMSELVES; eᵢ = ⊕ᵢ.identity — the op trait
                                             # already carries the neutral element, so there is NO
                                             # Component wrapper duplicating it
    inject: Lambda | None = None     # ι : A → S (None = the identity embedding); subsumes Channel.term
    psi: str | None = None           # twist family ("exp"; None = degenerate) — Twist.family purified.
                                     # WHY it must exist: components underdetermine ⊕ — flash and the
                                     # plain (max,add,add) product share a components tuple but differ
                                     # in ⊕ (coupled rescale vs componentwise); psi is the coupling.
                                     # WHY a NAME and not a ψ Lambda: the stable form is NOT derivable
                                     # from the bijection (naive conjugation is the overflow the
                                     # representation avoids) — each family pairs a generator with a
                                     # hand-written stabilizer, keying ONE source for many derived
                                     # shapes: the streaming step, the blocked (warp) evaluation, the
                                     # LSE cross-partition combine, StateMerge, seeds, and the psi-keyed
                                     # gates (role TWISTED, no atomic finalize)
                                     # (subsumes Channel.lift — the expectation ⊗ is ψ's business)
    # the per-component ACCUMULATOR dtype (today Channel.dtype, None = lowering default) is
    # precision, not algebra — it survives only as an optional parallel tuple so lowered Accums stay
    # byte-identical; a precision decoration may absorb it later

# ── ir/tile/ir.py — the two node kinds only (plus TileOp and the derived views) ───────────────────
Operand = Load | Fold | Map          # materialized | computed (inline node, one consumer by tree ownership)

@dataclass(frozen=True)
class Fold(Stmt):
    axis: Axis                       # the reduced iteration space
    monoid: Monoid
    lift: Lambda                     # λ(k, v₁…vₙ) → A — PURE; state-reading lives ONLY inside ⊕
    operands: tuple[Operand, ...]    # one CLOSED term per lift param vᵢ — POSITIONAL binding
    # schedule slices (decorations; composition supplies multi-slice forms):
    tile: TilePlan | None
    reduce: ReducePlan
    stage: Stage | None

@dataclass(frozen=True)
class Map(Stmt):
    fn: Lambda                       # π : λ(s₁…sₙ) → out; sources bind positionally to params
    sources: tuple[Operand, ...]     # project ∘ fold; must admit Load — the cut terminal (phase 4)
```

**Lowering rule.** `Fold.loop` = the operand bodies (positional binding — no first-use scan, no tie
rule), the lift body, then the DERIVED carrier update: the stabilized `s ⊕ ι(lift(k))` unfolding of
`(psi, components)`. Deterministic, so the derived loop — and with it `op_cache_key` / kernel identity —
depends only on the stored params. Derived, never stored: the update sequence, the role predicates
(landed 1l), the contraction view, the loop nest, cross-CTA splits (reassociation), identity.

**The contraction view** — derived at fork-emit, never stored: `contraction_view(fold, m, n, lead)`
requires the LIFT to factor bilinearly — one multiply per component sharing a common factor
(`_parse_bilinear`, now a read of the lambda body); the OUTPUT axes are the CALLER's placement facts,
which is why they are parameters and not fold fields. The `ContractionView` dataclass IS the view: one
shared `a` edge, `(b, acc)` channels, the `(m, n)` `Side` geometry, `b_trans`. `as_fold()` is the
storage direction; the round-trip and loop byte-identity stay unit-tested; `ir.shared_operand` the
placement-free cone read.

**Pipeline:**

```
Loop IR ──recognize──▶ pure Fold/Map term tree         (structure only; no hardware knowledge)
        ──schedule───▶ scalar / coop rows read the fold directly;
                       tensor cores present? view = contraction_view(fold, place)
                       view ─▶ warp / staged rows: operand roles from the view, TilePlan on the fold
        ──split / materialize──▶ slices off the nodes, roles off the view (re-derived from ctx.grid)
```

Worked shapes (pure spellings; `M(…)` a Monoid):

```
sum:      Fold(k, M(add), λ(k,x). x, (Load(x[m,k]),))
matmul:   Fold(k, M(add), λ(k,a,b). a·b, (Load(A), Load(B)))
gate⊗up:  Map(swiglu, (Fold(k, M(add, add), λ(k,x̂,g,u). (x̂·g, x̂·u), (cone, Load(Wg), Load(Wu))),))
          — sharing is arity: ONE pure lambda, tuple-valued, into the product monoid; the cone is an
          ordinary inline operand (Map(fn=normalize, sources=(Fold(stat),)) — the K seam is the node
          boundary)
rmsnorm:  Map(λs. …rsqrt(s/K)…, (Fold(k, M(add), λ(k,x). x², (Load(x),)),))
softmax:  Map(π, (Fold(k, M(max, add; ι=λs.(s,1); ψ=exp), λ(k,x). x, (Load(x),)),))
flash:    Map(λ(m,l,O). O/l, (Fold(kv, M(max, add, add; ι=λ(s,v).(s,1,v); ψ=exp),
                                   λ(j,s,v). (s,v), (score, Load(V))),))
          score = Fold(dd, M(add), λ(d,q,kk). q·kk, (Load(Q), Load(K))) — an ordinary CLOSED operand
          (operands may read the enclosing iteration var, never state). Today's interleaved in-step
          spelling — merge stmts, P = exp(s′−m′), the PV fold — is the DERIVED blocked evaluation of
          ⊕_ψ, generated at lowering, byte-identical to what is stored today (the 1p gate).
```

A projection has ONE home — the wrapping `Map.fn`, never a node field; every root store is materializer
glue synthesized from `results` + the graph output (the bare-fold `Write` glue, generalized — no `Write`
inside a term). Every schedule slice rides the node it decorates; `TileOp` keeps only
`op + place + workers + knobs`; a sliced axis's window is the one `Axis.window` vocabulary.

**Landed today vs target** (the deltas the phase-1 continuation closes): `Fold.step` still stores the
dissolved lift+update sequence and splices operands by first use (→ 1o/1p); `Map.body` has no binder and
`out` is a last-def convention (→ 1n); projection `Write`s still ride `Map` bodies (→ 1q); `Lambda` /
`Monoid` / the `Stmt.pure` trait don't exist yet (→ 1m); identity keys off the lowered nest (kept
through migration; the α-invariant term hash is a re-keying-window event).

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
- **1k-iii** (landed — the pure-Fold milestone; phase 1 reopened by the λ-foldMap target, see 1m–1q):
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

## Phase 1 continuation (open): 1m–1q — reaching the λ-foldMap target

Ordered so every byte-neutral step lands first; each carries the digest gate. 1p/1q are
re-keying-window-gated and do NOT block phases 2–3 (the codec resolves by axis on either spelling, and
identity keys off the lowered nest throughout).

- **1m — the primitives, in their home modules.** `Lambda` lands in `ir/stmt` beside `Body`:
  `__post_init__` validates the LOCAL formation invariant — every body stmt passes the new **`Stmt.pure`
  trait** (declared on the `Stmt` interface with a conservative `False` default; `Load` / `Assign` opt
  in, `Accum` / `Write` / `Init` / `Loop` never do — no isinstance whitelist, and a new stmt kind is
  excluded from lambdas until it declares itself) and every result is defined; the contextual half
  (free names ⊆ params ∪ enclosing iteration vars) is validated at Fold/Map formation, where the scope
  exists — plus canonical renumbering for α-invariant equality/hash. `Carrier`/`Twist` refit into the
  `Monoid` presentation in `ir/stmt/algebra`: components are the `ElementwiseImpl` ⊕ ops themselves
  (eᵢ read off the op's `identity` trait — no `Component` wrapper; `Channel.term`/`lift` subsumed by
  `inject` and the derived ⊕_ψ; the accumulator dtype survives as a precision side-tuple only for
  byte-identical Accums); bound mode survives for the one-shot `StateMerge`. Ships the executable SPEC too:
  the denotational `foldMap` evaluator + the ⟦tree⟧ == lowered-loop agreement test. Pure additions —
  no storage change, no digest impact.
- **1n — `Map` grows its binder.** `Map.body` → `fn: Lambda`: sources bind positionally to params,
  `results` replace the `out` last-def convention (the `source` len-≤1 compat read retires with it).
  Lowering splices identically — byte-neutral, digest-gated.
- **1o — the degenerate lift.** For id-family folds, `step` → `lift: Lambda` and the update is
  DERIVED: the fold `Accum`s are generated from the monoid at lowering and spliced at the step tail,
  exactly where they sit today. Operands bind positionally (the first-use splice + tuple-order tie rule
  retire for folds); `_parse_bilinear` becomes a plain read of the lift body. Byte-identical lowering
  on every `as_fold` / `from_loop` shape is the gate.
- **1p — the twisted derivation.** Online softmax / flash store `(psi, components, inject)` + the pure
  block lift; the stabilized interleaved sequence — the merge stmts, `P`, the PV contraction, the
  rescales — is DERIVED as the blocked evaluation of ⊕_ψ. The generator must reproduce today's stored
  stmt sequence byte-exactly through `030`'s σ-slicing and the twist realizer's reads — the flash
  digests including split-KV are the gate. If exact reproduction proves unreachable, this step moves
  into the deferred RE-KEYING WINDOW already reserved for the QK edge-hoist (one window, all
  order-canonicalizations at once). Knob spellings must resolve unchanged either way: `TILE@dd`
  addresses the score operand edge, `TILE@pj` the derived PV site by its axis name (the phase-2 walker
  enumerates derived-⊕ sites for twisted folds).
- **1q — effects to the boundary + the identity switch.** Projection `Write`s leave `Map.fn` (`results`
  + materializer glue synthesize every root store — the bare-fold glue generalized to `030`'s partials
  and flash's layout-aware store); `captured_values` demotes to a validation assert. Then, inside the
  same re-keying window as 1p's fallback: `Body` order inside lambdas canonicalizes and kernel identity
  switches from lowered-nest bytes to the α-invariant term hash — a re-keying event by definition,
  never a silent one.

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

- **1p is the riskiest remaining step**: the twist realizer is the subtlest emitter, and the derived
  ⊕_ψ unfolding must reproduce today's dissolved stmt sequence byte-exactly through `030`'s σ-slicing
  and the realizer reads (flash digests including split-KV are the gate). A divergence is a bug in the
  generator, never something to migrate around; the sanctioned escape is the re-keying window, not a
  silent re-key. The QK edge-hoist constraint survives inside 1p: hoisting must not perturb lowered
  stmt order outside that window.
- **Purity erosion in lambdas**: the whole design rests on lifts never touching state — `Accum` (or any
  effectful stmt) creeping back into a `Lambda` body re-creates the step/carrier duplication. The
  `Stmt.pure` trait + `Lambda.__post_init__` formation validation (conservative default: a new stmt
  kind is impure until declared) is the guard; never bypass it with a pre-built `Body`.
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
