# Tile IR + schedule refactor: the λ-foldMap IR, tree-path knobs, PLACE as a per-seam edge property

## Context

Branch `feature/remove-place-knob` deleted the old placement machinery: the `020_cut_edge` /
`025_sink_row_reduce` / `032_fuse_finalize` realizer passes and `_sink.py` are gone, `PLACE` is scrubbed
from the search space, and the gemma golden YAMLs keep their old PLACE keys only as `# retired knob
dropped:` comments. Nothing in the compiler consumes placement seams right now. That is deliberate: the IR
and the knob codec are redesigned first, on the smaller surface, and the placement functionality is then
restored on top of the new vocabulary instead of being ported.

The IR design went through four iterations (let-bound sharing → the product-carrier contraction → the
pure Fold IR → the λ-foldMap IR: singleton-state `Lambda` lifts + the true-monoid `(init, combine)`,
the serial step derived); this document describes ONLY the final design and the landed state. The
superseded intermediates survive as one list under *Landed trail* — do not resurrect their
vocabularies.

## North star

**The recognized algebra tree is the single semantic object. Everything else is a decoration on it or a
derived view of it. A kernel IS `π ∘ foldMap`** — a projection over a monoidal fold of a pure
per-element lift: `⟦Fold⟧ = ⊕_{k ∈ axis} ι(lift(k))`, seeded at `e`. The design rests on five ideas:

1. **ONE general fold node — axis + algebra + lift, all FLAT fields.** A `Fold` stores the reduced
   `axis` and the ⊕ as flat `(init, combine)` fields — NO `Monoid` wrapper class (dissolved into the
   node at 1r; `M(op…)`, the componentwise convenience constructor, survives as a free helper
   building the pair; recognition's pattern builders build the twisted ones — the family name is
   construction-time knowledge and dissolves there; DEGENERATE is a derived shape predicate on
   `combine`, not a storage arm). `combine : S × S → S` is a TRUE monoid's ⊕, ONE program,
   generated once at construction — a
   PURE `lift` `Lambda` (`λ(k, v₁…vₙ) → S` — it produces the element's SINGLETON STATE, seeing the
   iteration var and the operand values, nothing else; ι is spelled in the lift — flash's emits
   `(s, 1, v)`, and the degenerate lifts are already singleton-shaped), and one CLOSED operand edge
   per lift param. There is NO stored serial update: the streaming step IS
   `s′ = combine(s, lift(k))` — combine specialized at the singleton, simplified deterministically at
   lowering — so update-vs-combine consistency is correct BY CONSTRUCTION (no second program to keep
   coupled) and the step/carrier duplication (with its rename-lockstep bug class) cannot exist. ALL
   state-reading lives inside `combine`. A matmul, a bare sum, RMSNorm's statistic, gate⊗up and flash
   are all this node at different monoid arities and twists. Sharing is still **arity**: the
   tuple-valued lift into the product monoid; in-lambda sharing is ANF (every `Assign` is a let).
   Flash's `P = exp(s − m′)` is `combine`'s internals, derived — the "legal but uncuttable
   state-capturing step element" quirk evaporates with it.
2. **`Lambda` = explicit binders over the REUSED stmt vocabulary — no second expression language.**
   `Lambda(params, body, results)`: `params` the binders, `body` a `Body` of PURE stmts only
   (`Load` / `Assign` — A-normal form ≙ a let-chain), `results` the returned defs (replacing every
   `out` / last-def convention). Purity is a FORMATION invariant, not a new type universe: no `Accum`,
   `Write` or `Loop` inside a lift; free names ⊆ params ∪ iteration vars; results defined. α-invariance
   is canonical renumbering (the existing rename machinery), not de Bruijn. **Primitives live in their
   home modules** — they are not tile-IR-private: `Lambda` beside `Body` in `ir/stmt` (NEVER in
   `tile/ir.py`); the algebra HELPERS in `ir/stmt/algebra` — no `Monoid` class, the pair lives flat
   on `Fold`: the componentwise constructor (`M(op…)`), the `component_ops` shape-readers, the
   twisted-combine regeneration rule (in the `Fold` rewrite lockstep), `StateMerge` (riding the
   stored-combine rename unchanged) and the executable spec; `family` / `inject` / `Channel` dissolve
   at construction; scalar/index exprs stay `ir/expr`; `Fold`/`Map`/`TileOp` and the derived views
   stay `ir/tile`.
3. **Edge iff closed — by CONSTRUCTION.** An operand edge has two inhabitants — MATERIALIZED (a gmem
   `Load`) or COMPUTED (an inline node; tree ownership gives it one consumer) — and operands bind
   POSITIONALLY to lift params, so an operand cannot see the fold's state or its siblings: closure
   needs no scan (`captured_values` demotes to a validation assert), every seam on the cut lattice is
   closed structurally, and composition is nesting a term on an edge. Operands may read the ENCLOSING
   iteration var (flash's per-key score fold) — never state. Effects sit at the kernel boundary only:
   `results` + materializer glue synthesize every root store; no `Write` inside a term. The stored
   `combine` sits BELOW the seam lattice — never a cut target — so flash's `P` (its derived
   serial-step internals) needs no special legality case.
4. **Hardware readings are DERIVED views, never stored.** The bilinear factored form `a·(b₁…bₙ)` the
   tensor cores require is `contraction_view` — a factoring of the LIFT (`_parse_bilinear` reads the
   lambda body), computed at fork-emit where a consumer exists, absent from storage; the role (landed
   1l) is a predicate — non-degenerate `combine` / bilinear lift — not even a derived enum in the end
   state. Three
   design-level reasons a view and not a rewrite: the same fold must keep offering its scalar/coop rows
   as fork siblings beside the warp rows; one relation keeps one spelling (a stored refined kind beside
   the general fold is the two-spellings defect); and knob keys stamp against the pre-specialization
   tree.
5. **Kernel boundaries are a placement DECISION, resolved FIRST and recursively.** A `cut | fuse` bit
   on each parent↔child seam; kernels are a derived view: the cut set partitions the tree, each
   partition materializes as one launch, each cut seam materializes its value to a buffer (a cut =
   let-materializing a closed term). Resolution is TWO-LEVEL: structure (the cut set) resolves first,
   then EVERY resulting kernel — the cut children AND the residue — re-recognizes as a fresh root and
   resolves its OWN singular schedule independently, recursing through the same lookup (a piece's
   entry may itself cut). Golden storage factors the same way: ROUTING entries hold cuts only — never
   schedules — and schedule entries hold ONE kernel's schedule; a cut seam's shape is a recognition
   fact, so no schedule decision ever spans a kernel boundary.

From this everything else follows:

- **Knobs are paths into the tree.** A schedule key addresses the node (or edge) it decorates by
  position; no parallel namespace of hand-invented site names. Explicit binders make paths stable by
  construction.
- **Keys stamp against the pre-placement tree.** A cut never re-keys the decisions inside either half —
  and under the factored golden storage this holds trivially: a knob row only ever addresses ONE
  kernel's own recognized tree, so there is no cross-kernel row to protect.
- **Derive, never store — and store ONCE what cannot derive.** Loop nests, the serial step (`combine`
  at the singleton), roles, the degeneracy predicate and the op-trait legality reads (off a
  trivially-shaped `combine`), tile geometry, the contraction view, cross-CTA splits (reassociation
  `fold_k = fold_{k₁} ∘ fold_{k₂}`, certified by the associativity property test) — synthesized on
  demand from the params. Stored state is only the params + decisions; `(init, combine)` are
  themselves params, with exactly one home.
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
  an agreement test (`⟦recognized tree⟧ == lowered loop` on random inputs) and the ASSOCIATIVITY
  property test (`combine(a, combine(b, c)) == combine(combine(a, b), c)` on random states — the
  split/coop legality certificate; no update/combine coupling is left to test) pin the algebra, so
  every purification step refactors toward an oracle that already runs.

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

# ── ir/stmt/algebra — HELPERS only; NO Monoid class, the ⊕ lives FLAT on Fold ─────────────────────
# The serial streaming step is NOT stored: it is `s′ = combine(s, lift(k))` — combine specialized at
# the singleton state the lift produces, simplified deterministically at lowering (×1 folds,
# shared-exp CSE; emmy's OWN simplifier, never nvcc's) — so update-vs-combine consistency is correct
# BY CONSTRUCTION: there is no second program to keep coupled. The pair is generated ONCE at
# construction — `M(op…)` (the componentwise free constructor: one independent self-fold ⊕ᵢ per
# component, seeds the op identities, threading recognition's REAL accumulator names) for a plain
# fold, recognition's pattern builders (_softmax / _flash) for a twisted one; the family name
# (Twist.family / psi), `inject` and `Channel.term`/`lift` dissolve there (ι is spelled IN the
# lift — flash's emits `(s, 1, v)`), and downstream reads the program, never a name. DEGENERATE is a
# DERIVED predicate, not a storage arm: `component_ops(combine)` — a free shape-reader (every result
# `sᵢ″ = ⊕ᵢ(sᵢ, sᵢ′)`, independently) whose ⊕ᵢ handles the trait/legality queries consume. The
# twisted-combine REGENERATION rule rides the `Fold` rewrite lockstep: a generated program is
# regenerated over renamed names, asserted equal to its generator's output at formation.
# ASSOCIATIVITY is TEST-enforced (the certificate the split/coop tiers rest on):
# combine(a, combine(b, c)) == combine(combine(a, b), c) on random states — plus the
# ⟦tree⟧ == lowered-loop agreement test. `StateMerge` and the executable spec (eval_lambda /
# foldmap_eval) stay here; Carrier / Twist / State die at step 7 (see *Retirement ledger*).

# ── ir/tile/ir.py — the two node kinds only (plus TileOp and the derived views) ───────────────────
Operand = Load | Fold | Map          # materialized | computed (inline node, one consumer by tree ownership)

@dataclass(frozen=True)
class Fold(Stmt):
    axis: Axis                       # the reduced iteration space
    init: tuple[float, ...]          # the ⊕ seeds — op identities for a plain fold; (−inf, 0, 0) LSE
    combine: Lambda                  # S × S → S — THE ⊕: the serial tiers specialize it at the
                                     # singleton, split / coop / StateMerge apply it to states
                                     # (bound-mode rename, generalized), the blocked (warp) form is
                                     # its reassociated evaluation. State enters as params and
                                     # leaves as results — no Accum in any stored program.
    lift: Lambda                     # λ(k, v₁…vₙ) → S — PURE, produces the element's SINGLETON
                                     # state; state-reading lives ONLY inside combine
    operands: tuple[Operand, ...]    # one CLOSED term per lift param vᵢ — POSITIONAL binding
    dtypes: tuple = ()               # optional per-component accumulator dtype — precision, not
                                     # algebra; a precision decoration may absorb it later
    # NO schedule fields: the slices live on TileOp.schedule (1r) — the term is pure algebra
    # __post_init__ carries the S×S→S arity check (params 2n / results n vs init) — Monoid's old
    # formation invariant, relocated

@dataclass(frozen=True)
class Map(Stmt):
    fn: Lambda                       # π : λ(s₁…sₙ) → out; sources bind positionally to params
    sources: tuple[Operand, ...]     # project ∘ fold; must admit Load — the cut terminal (phase 4)

Slice = TilePlan | ReducePlan | Stage  # the EXISTING schedule value types (ir/schedule.py) — no new
                                       # class; at 1r the warp-form TilePlan sheds `units` into `work`

@dataclass
class TileOp(Op):
    op: Fold | Map                   # the pure term tree — IMMUTABLE across the whole schedule search
    place: Placement
    work: Workers                    # the ONE worker inventory (w4x1 / t16x8 / t512) — factored
                                     # in-memory at 1r; absorbs WarpSpec/WSPEC at step 7
    schedule: dict[Key, Slice]       # the slice decorations (1r), keyed by the phase-2 CODEC key —
                                     # `FAMILY@path` — NOT by path alone: one fold may carry all
                                     # three families at once, and the family selects the slice
                                     # kind, so key and value agree by construction. Values are the
                                     # RESOLVED slices, which makes the stamped knob row DERIVABLE
                                     # (spell() of the values + `work` + the root-globals) — honest
                                     # stamping by type. A fork is a DIFFERENT MAP, never a rebuilt
                                     # tree — "keys stamp against the pre-placement tree" holds by type
```

**Lowering rule.** `Fold.loop` = the operand bodies (positional binding — no first-use scan, no tie
rule), the lift body, then the DERIVED serial step: `combine` specialized at the singleton `lift(k)`,
simplified deterministically (×1 folds, shared-exp CSE — emmy's own simplifier, pre-emission) and
re-bound to the loop's accumulator names (a result of the form `sᵢ′ = ⊕(sᵢ, t)` lowers to the `Accum`
form) — landing exactly where the dissolved merge sits today. Deterministic, so the derived loop — and
with it `op_cache_key` / kernel identity — depends only on the stored params. Derived, never stored:
the serial step, the degeneracy predicate, the role predicates (landed 1l), the contraction view, the
loop nest, cross-CTA splits (reassociation), identity.

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
                       view ─▶ warp / staged rows: operand roles from the view, the slices into
                       TileOp.schedule
        ──split / materialize──▶ slices off the schedule dict, roles off the view (re-derived from ctx.grid)
```

Worked examples — the term in YAML-like spelling: the node kind is a `!Fold` / `!Map` tag, one field
per line; a nested node is hoisted to a named key (`stat:`, `product:`, `score:`) for READABILITY
ONLY — storage is inline, sharing is edge reuse, there is no let table. The lift's first param is the
iteration var, then one per operand edge, bound positionally; the ⊕ is the flat `(init, combine)`
pair — a componentwise combine is spelled by its ops (`add`, `(add, add)` — what the free `M(op…)`
constructor builds), a twisted one by its generator family (`lse`, `flash`; programs in the notes
below). Schedules are END-STATE codec entries (grammar under *Knob codec*; values
illustrative): one block = ONE kernel's golden entry; a labeled cascade = the recursive routing
resolution (*Placement*); "same term" = a fork sibling — a fork is a different map over the immutable
term:

| Operation | Tile IR (YAML-like) | Schedule entries (end-state codec) |
| --- | --- | --- |
| bare reduce (sum) | `!Fold`<br>`axis: k`<br>`init: [0]`<br>`combine: add`<br>`lift: λ(k,x)→x`<br>`operands: [Load x]` | `WORK: t512`<br>`REDUCE@fold.k: coop` |
| bare reduce, grid-split sibling | same term | `WORK: t256`<br>`REDUCE@fold.k: g4`<br>— the partials + synthesized finalize stay in-kernel artifacts, never key targets |
| matmul, warp tier | `!Fold`<br>`axis: k`<br>`init: [0]`<br>`combine: add`<br>`lift: λ(k,a,b)→a·b`<br>`operands: [Load A, Load B]` | `WORK: w1x8`<br>`TILE@fold.k: mma_m16n8k16_f16_f32/f4x1/k4`<br>`REDUCE@fold.k: g2`<br>`STAGE@fold.k: d3/tma/ring` |
| matmul, scalar-tier sibling | same term | `WORK: t16x8`<br>`TILE@fold.k: f4x8`<br>`REDUCE@fold.k: g2`<br>`STAGE@fold.k: d3/tma/ring` |
| pointwise cell | `!Map {fn: cell, sources: []}` | `WORK: t16x8`<br>`TILE@map: f4x8` |
| rms_norm | `!Map`<br>`fn: λ(s)→…rsqrt(s/K)…`<br>`sources: [stat]`<br>`stat: !Fold`<br>`  axis: k`<br>`  init: [0]`<br>`  combine: add`<br>`  lift: λ(k,x)→x²`<br>`  operands: [Load x]` | `WORK: t512`<br>`REDUCE@fold.k: coop` |
| softmax (online) | `!Map`<br>`fn: π`<br>`sources: [lse]`<br>`lse: !Fold`<br>`  axis: k`<br>`  init: [−inf, 0]`<br>`  combine: lse`<br>`  lift: λ(k,x)→(x,1)`<br>`  operands: [Load x]` | `WORK: t256`<br>`REDUCE@fold.k: coop` |
| norm_linear / gate⊗up, fused | `!Map`<br>`fn: swiglu`<br>`sources: [product]`<br>`product: !Fold`<br>`  axis: k`<br>`  init: [0, 0]`<br>`  combine: (add, add)`<br>`  lift: λ(k,x̂,g,u)→(x̂·g, x̂·u)`<br>`  operands: [cone, Load Wg, Load Wu]`<br>`cone: !Map`<br>`  fn: normalize`<br>`  sources: [stat]`<br>`stat: !Fold`<br>`  axis: k`<br>`  init: [0]`<br>`  combine: add`<br>`  lift: λ(k,x)→x²`<br>`  operands: [Load x]` | `WORK: w4x2`<br>`TILE@fold.k: mma_m16n8k16_f16_f32/f4x8/k4`<br>`STAGE@fold.k: d2/tma/ring`<br>`REDUCE@a.fold.k: serial`<br>— the cone's stat; the path disambiguates the same-named `k` |
| norm_linear, cut | same term — placement is a decision, never a rewrite of the term | `norm_linear:  # routing — no schedules`<br>`  PLACE@a: cut`<br>`rms_norm:  # the cone, re-recognized — routes again`<br>`  PLACE@map: cut`<br>`reduce:  # the stat`<br>`  WORK: t512`<br>`  REDUCE@fold.k: coop`<br>`pointwise:  # the scale`<br>`  WORK: t16x8`<br>`  TILE@map: f4x8`<br>`matmul:  # the residue — the warp-tier entry above; evidence reuse` |
| flash | `!Map`<br>`fn: λ(m,l,O)→O/l`<br>`sources: [stream]`<br>`stream: !Fold`<br>`  axis: kv`<br>`  init: [−inf, 0, 0]`<br>`  combine: flash`<br>`  lift: λ(j,s,v)→(s,1,v)`<br>`  operands: [score, Load V]`<br>`score: !Fold`<br>`  axis: dd`<br>`  init: [0]`<br>`  combine: add`<br>`  lift: λ(d,q,kk)→q·kk`<br>`  operands: [Load Q, Load K]` | `WORK: w4x1`<br>`TILE@fold.dd: mma_m16n8k16_f16_f32/f1x16/k4`<br>`TILE@fold.pj: mma_m16n8k16_f16_f16/f1x8/k8`<br>`STAGE@fold.kv: d2/cp/ring` |

Notes the examples rest on:

- `lse`: `combine = λ((m,l), (m′,l′)) → (m″, l·e^{m−m″} + l′·e^{m′−m″})`, seeds `(−inf, 0)`; the
  serial step is combine at the singleton (m′ = x, l′ = 1), simplified — m″ = max(m, x),
  l = l·e^{m−m″} + e^{x−m″} — today's dissolved merge, DERIVED.
- `flash`: the coupled-rescale (m, l, O) ⊕, seeds (−inf, 0, 0), built once by flash recognition.
  Today's interleaved in-step spelling — merge stmts, P = exp(s′−m′), the PV fold — is combine's
  singleton specialization (the 1p derivation gate); the blocked warp form is the same combine
  reassociated over key blocks. `score` is an ordinary CLOSED operand — operands may read the
  enclosing iteration var, never state.
- Sharing is arity / edge reuse: gate⊗up is ONE tuple-valued lift into the product monoid, and the
  cone an ordinary inline operand read as one edge — the K seam is the node boundary.

A projection has ONE home — the wrapping `Map.fn`, never a node field; every root store is materializer
glue synthesized from `results` + the graph output (the bare-fold `Write` glue, generalized — no `Write`
inside a term). The schedule slices live in `TileOp.schedule`, keyed by tree path (1r) — the term is
pure algebra, immutable across the whole schedule search — and a sliced axis's window is the one
`Axis.window` vocabulary.

**Landed today vs target** (after 1m–1p + steps 1–5 and 7): EVERY fold stores the λ spelling — no `step` field
exists (`lift` + flat `(init, combine)`, carrier/step/Accums DERIVED; the `Monoid` wrapper dissolved at 1r; the
composed evaluations — flash's blocked evaluation with the QK operand edge and the synthesized PV, split-K's
identity-lift embedding — derive at step 7); `Map` stores
`fn: Lambda` + positional `sources`; the root stores ride `TileOp.stores` and every recognized term's
`Map.fn` is a strict pure `Lambda` (1q — the raw-loop-IR kernels keep the one `_loop_ir_fn` arm); the schedule
slices live in `TileOp.schedule` keyed by the codec
(the term IMMUTABLE across the search) with the worker inventory sealed into `TileOp.work` AND spelled once as the
`WORK` row entry beside site-local `TILE`/`REDUCE` values (step 7; legacy spellings are validated pin aliases);
identity is the α-invariant term hash (`ops.term_key`). The `Carrier`/`Twist`/`State` apparatus is RETIRED
(2026-07-30): the loop annotation is the flat `Algebra(combine, terms, dtypes)` — the SAME combine `Lambda` the
`Fold` stores, family derived via `component_ops`, `combine_states` the stored body, `Fold.carrier` a derived
property (no stored field) — with `Channel`/`exp_channels`/`_carrier.py`/the `id_*` channel arms deleted and
`StateMerge.state` a plain name tuple (27-kernel digest byte-identical). The remaining deltas:
`TilePlan.units` remains a (validated, `work`-agreeing) value-object field; the dynamic-attention bare-`TILE`
contract stays a documented live exception (blocked on symbolic-trace keyed resolution).

## Landed trail (compressed history — the vocabularies below are RETIRED; every step was gated on the
17-kernel source-digest A/B, byte-identical)

- **1a–1g** built the structural-node tree (typed reduce, nodified contractions, cone as a node tree,
  one composition rule, `Axis.window`, closure predicate, every node a `Stmt`, resolve-as-splice) via a
  let table (`TileOp.bindings`, name-referenced sharing). Retired by 1j.
- **1h** made both contraction operand edges the same union; the A/B asymmetry moved to schedule gates
  (`isinstance(c.b, Load)` eligibility preconditions) — that stance survives, now via the view's roles.
- **1j** replaced let-bound sharing with the product-carrier `Contraction` (one `a` edge + `(b, acc)`
  channels; sharing as arity) and deleted the whole reference apparatus (`bindings`, `ops.resolve`,
  rename-lockstep, `is_group`/`group_loop`).
- **1k-i/ii**: `operands` + the first-use splice + the `tile` slot; every ROOT contraction stored as a
  `role=CONTRACTION` fold, consumers re-deriving the view from the fold + the caller's placement axes
  (`view.as_fold()` the storage direction; round-trip unit-tested).
- **1k-iii** (the pure-Fold milestone): flash's QK/PV store as in-step `role=CONTRACTION` folds (PV's
  `P` cone a capturing operand; QK closed, but hoisting it to an edge would REORDER the lowered nest —
  deferred to the step-7 window); consumers derive views (`_twisted_pair`; `place.free` threads
  through `Ctx.free` to the materializer). `Contraction` de-Stmt'd into the view; the renames:
  `Reduction`→`Fold`, `partial`→`step`, `Contraction`→`ContractionView`. Also verified by a stash A/B
  on the hd512.s2048 split-KV golden.
- **1l**: `Fold.role` retired — derived (TWISTED off the twist family; CONTRACTION off the bilinear
  parse or the composed split-K step; PLANAR otherwise); the matvec DEMOTION relocated out of
  recognition (an unbindable contraction keeps its loads inline, derives PLANAR, takes the reduce
  tiers at dispatch — `down_proj.m1.t` verified). The loop-level `AxisRole` stays stored.
- **1m**: the primitives in their home modules — `Lambda` beside `Body` (the `Stmt.pure` trait gate,
  α-invariance by canonical renumbering), `Monoid` in `ir/stmt/algebra` (dissolves into flat `Fold`
  fields at 1r), the executable SPEC (`eval_lambda` / `foldmap_eval` + the agreement and
  associativity property tests). Pure additions.
- **1n**: `Map.body` → `fn: Lambda` — positional `sources`, `results` replaced the `out` last-def
  convention. INTERIM: projection effects ride the one sanctioned `effectful_lambda` constructor —
  deleted at 1q.
- **1o**: every DEGENERATE fold stores `lift` + the ⊕ pair; serial step / `Accum` forms / carrier
  annotation DERIVED (combine at the singleton); `Fold.from_loop` keeps the λ spelling ONLY on
  construction-time byte-identity of the derived loop; `_parse_bilinear` reads the lift body.
- **1p** (the non-composed twisted shape — online softmax): lift `(x, 1)` + the true `(init, combine)`
  over recognition's real state names; the serial step derives through the exp/LSE generator
  (`exp_merge`); the carrier reconstructs structurally (the stored combine must BE the generator's
  program — asserted at formation); state-component ROLES are SHAPE-DERIVED off the singleton
  (pivot = comp 0, literal-1 = denominator, value name = expectation — no annotation). RESIDUAL:
  flash's kv fold keeps the composed `step`; its dissolution rides the phase-2 walker + the step-7
  window. The hd512.s2048 split-KV digest matches the pre-refactor base byte-exactly.
- **Step 1 / phase 2 (LANDED)** — the codec core: `ir/tile/path.py` (ONE walker `sites` +
  resolver `resolve` + canonical speller `spell`; anchored path subsequences, edge-label
  preference, primary-bare guard, ordinal emission, reserved-form rejection) + the golden compat
  tripwire (`test_golden_key_compat.py` — every stored spelling proven canonical against per-kind
  recognized trees; the dynamic-attention bare-`TILE` any-of contract the one asserted exception).
- **Step 2 / 1r (LANDED, three digest-gated commits, 27-kernel harness byte-identical)** —
  (i) `Monoid` dissolved into flat `Fold.init/combine/dtypes` (`M(op…)` / `component_ops` /
  `degenerate` / `rename_combine` free helpers; arity check in `Fold.__post_init__`;
  `foldmap_eval(init, combine, lift, …)`); (ii) `tile`/`reduce`/`stage` left the nodes for
  `TileOp.schedule` `{codec key → resolved slice}` via the `ops.Sched` accessor
  (`contraction_view` takes caller slices, `warp_source`/`chain_source`/`Ctx.sched` read it,
  `nodify_reduce` returns the fold to key on, `030` re-keys onto the partial's own tree, graph
  JSON round-trips the dict values) — the term is immutable across the schedule search (the flash
  variants share one op verbatim); (iii) `TileOp.work` — the ONE worker inventory
  (`ir.schedule.Workers` + `derive_workers`, sealed per option, LOUD on cross-site disagreement).
  RESIDUAL (deliberate, step-7 wire item): `TilePlan.units` stays a field of the value object —
  the slot is authoritative and validated, the ~150-site consumer migration rides the
  value-grammar split that re-spells the wire anyway.
- **Step 4 / 1q (LANDED, four digest-gated commits, 27-kernel harness byte-identical)** — effects
  to the boundary: (i) the `Store` vocabulary — a kernel-boundary decoration holding the root
  `Write` whole (+ the output-sweep axis/unroll), `TileOp.stores`, the ONE reconstitution rule
  `effect_tail` with its conversion-side inverse `split_effects` (round-trip byte-identity gate,
  the 1o pattern), `Select.pure`, and the params flattening (one `Map.fn` param per source RESULT
  COMPONENT — the geglu `acc_u` free-name fix); (ii) recognition's root stores converted (the
  projected reduces incl. the rms/softmax sweep restructuring, matmul epilogues, the fused
  norm→linear/geglu `Write`, flash's layout-aware store) with every reader reconstituting — the
  scheduler's tail gates (`ops.projection_tail`), the materializer's peel/flat-root arms, `030`'s
  projection/cell reads (keyed off the TileOp: an epi-only-`Write` projection leaves a BARE fold);
  (iii) `030`'s split partials nodify their sliced annotated `Loop` into a `Fold` source + carry
  workspace/atomic stores as boundary `Store`s, the finalizes and the register strip / pointwise
  cells convert; (iv) `effectful_lambda` DELETED — strict `Lambda` formation everywhere, ONE
  Map-private raw-loop-IR arm (`_loop_ir_fn`) for the kernels that are loop IR, not algebra: the
  un-recognized escape cell, `030`'s finalize (`Init` seeds + un-annotated merge `Loop`), the
  prologue'd partial, and the coop norm→linear/geglu sibling's composed contraction tail (these
  carry no root `Write` in-term either — only iteration/seed structure); the graph rehydrator
  repoints to the same arm, `captured_values` demotes to the validation reading. `op_cache_key` /
  `Graph.structural_key` shift for the converted kernels (sanctioned — evidence regenerated);
  CUDA sources byte-identical.
- **Step 3 / phase 3 (LANDED)** — the stampers spell knob keys via the resolver
  (`_schedule._family_key`; `_at` deleted, dead `knob.resolve_axis` deleted): stamped rows now ARE
  the stored/golden spellings — bare on single-primary trees, `TILE@dd`/`TILE@pj` + bare
  `REDUCE`/`STAGE` on flash, the cone stat's explicit `REDUCE@<stat axis>` shared across the
  merged prologue fork (`prologue_knob_bases` spells against the con tree; the map form keys its
  own reduce spec on the stat key). The `tuning_knob_items` bare-collapse is DELETED — keys render
  as stored — and pre-phase-3 DB/prior evidence is REGENERATED, not migrated (decided 2026-07-29);
  `_node_axes/_node_slice` grew the
  bare-remainder group so mixed flash rows keep their stage/reduce geometry in the sum-pool;
  `enumerate_graph` keeps rows by family, not `@`-presence. Kernel sources digest-identical;
  golden drift gate + offer-compat tests green on the 5090 goldens.
- **Step 7 (LANDED, seven digest-gated commits — 27-kernel harness byte-identical, under legacy pins for the wire
  flip; the identity + wire re-keys sanctioned as the window's evidence regeneration)** —
  (i) flash's composed step DISSOLVED: the kv fold λ-spells (lift `λ(kv, sacc, v_e) → (score, 1, v_e)`, QK a
  HOISTED inline-node operand edge reading the enclosing kv var — never state — and the flat exp-family ⊕ over the
  real `(m_i, l_i, O_i)` names); the serial step derives as the blocked evaluation (`ir._twisted_derived_step`,
  the PV contraction synthesized via `_split_expect`/`_derived_expect_fold` — byte-identical to the retired
  `_split_pv` material, MEMOIZED so the PV keeps ONE identity per stored fold); `Fold.step_stmts()` is the public
  read every former `.step` consumer goes through; the walker enumerates derived-combine sites (`TILE@pj` keeps its
  spelling; `Site.derived` marks combine material below the seam lattice, excluded from PLACE — the QK edge stays
  a PLACE site); `flash_combine`/`_split_pv` deleted.
  (ii) split-K DISSOLVED: an IDENTITY lift over its ONE inline sliced-fold operand (one lift param per operand
  RESULT component), combine at that singleton embedding the operand verbatim — no outer Accums, the nest
  byte-identical to the retired `step=(inner,)`; `ir.composed_contraction` the ONE read of the composition
  (`030`'s structural arm + the derived role share it).
  (iii) `Fold.step` DELETED — `Fold = (axis, operands, lift, init, combine, dtypes)`, formation asserts λ;
  `from_loop → Fold | None` (the raw-loop escape — `010`'s flat-Map arms, `nodify_reduce → (op, None)`,
  `030._residual`), the byte gate comparing body/axis/unroll ONLY (role/carrier excluded — the 1l matvec demotion
  now a formation fact); `_extract_lift` accepts any pure prefix; `ir.demote_operands` un-hoists the demoted cone;
  `030._slice_loop` keeps the sliced Accum's ORIGINAL axes (the `(_ksplit, a1)` expansion silently dropped the
  partial's coop tiling).
  (iv) kernel identity = the α-invariant TERM HASH (`ops.term_key`: canonical renumbering in first-appearance walk
  order through the ONE `_rewrite` registry, buffers positional `B<i>`, plus HASH-TIME ANF body-order
  canonicalization — Kahn over def/use edges, name-independent ready tokens; the stored term is never reordered,
  the lowered nest keeps storage order); consumed by `op_cache_key`'s TileOp arm + `Graph.structural_key`'s op
  field; the lowered-nest identity retired.
  (v-a..d) the value-grammar split: `WORK` (`w<M>x<N>[+p<np>]` — the `+p` band absorbing the retired `WSPEC` row
  family — / `t<N>x<M>` / `t<N>`), site-local `TILE` (`<atom>/f<FM>x<FN>[/k<bk>]` | `f<fn>[x<fm>]`; the tier
  discriminator IS the worker kind) and `REDUCE` (`[g<n>[a|k]][/coop[-t]][/r<n>]` — DELIBERATE DEVIATION from the
  re-spell table under *Knob codec*: the GRID finalize letter is KEPT, the atomic-vs-deferred finalize MODE, not
  an axis token — `g4a` and `g2k` are different live rows); `seal_workers` the ONE stamp chokepoint (derives
  `TileOp.work` + `knobs['WORK']` off the resolved slices, LOUD on cross-site disagreement; a 1-thread register
  strip stays `None`); `_site_row`/`_site_knobs` convert the legacy-speaking internal enumeration at row assembly;
  `resolve_site_tile` disambiguates an empty site `TILE` beside a thread `WORK` from the coop tier;
  `SCHEDULE_FAMILIES = (WORK, TILE, REDUCE, STAGE, RASTER)`; legacy embedded-token spellings survive as
  loudly-validated pin ALIASES that must agree with the inventory
  (`ingest_legacy_row`/`canon_family_value`/`values_equal`); the golden corpus re-spelled mechanically
  (`scripts/respell_goldens.py` — 715 rows, 6 YAMLs, text-level editing never a YAML dump, replay byte-identical
  BY MEASUREMENT; golden test set 163 passed). Residual: `_composes` SURVIVES as the derived composed-reduce
  predicate (it reads `step_stmts()` + the operand edges — no stored step), and stale step-era docstring passages
  remain in `ir/tile/ir.py` / `_flash.py`.

## Execution order (remaining work)

Steps 1–3 (the phase-2 codec core, 1r, the phase-3 stamp sites), 4 (1q), 5 (phase 4) and now 7 (the re-keying
window — flash/split-K dissolution, `Fold.step` deletion, the term-hash identity, the value-grammar split + golden
re-spell) are LANDED — see the trail above. What remains:

- **The evidence re-keying / re-seeding** (the step-7 aftermath + the phase-5 consolidated parity gate): tune DB /
  reservoir / online prior REGENERATION (the identity + wire re-keys discarded the old rows by design) and the
  hand-pinned golden `--ab` verification — including re-seeding the retired PLACE goldens as routing + child
  schedule entries — on BOTH cards. Dmitry executes this separately (the manual golden method; the tuner is not
  used for golden work).
- ~~**`Carrier` / `Twist` / `State` retirement**~~ — DONE (2026-07-30). The design pass resolved `Loop.carrier`
  to the flat `Algebra(combine, terms, dtypes)` (the field name survives; the classes and the SPEC/BOUND
  dual-mode apparatus do not — BOUND was already construction-dead). Two deviations: `projection_distributes`
  moved to `ir/stmt/passes.py`, not "030's home" (a digit-leading module name cannot be an import target, and
  `_schedule` consumes it too); `Channel` was deleted rather than surviving (the `terms` tuple subsumed its one
  irreplaceable field — `exp_merge` now takes `(state, terms)`).
- **`TilePlan.units` field deletion** (~150 consumer sites): deliberately NOT taken by the value-grammar split —
  the site wire no longer spells workers but the value object keeps its validated, `work`-agreeing field; one
  dedicated consumer-migration commit.
- **The F3 flash pin plumbing deletions — RESOLVED AS BLOCKED** (see the deferred list below): the
  dynamic-attention bare-`TILE` contract and the bare↔explicit any-of are LIVE corpus semantics, not legacy debt.

The digest gate is in-repo:
`scripts/digest_kernels.py` (27 kernels, off-GPU) — every storage change A/Bs against it.

**Evidence stance (decided 2026-07-29): the tune DB / reservoir / online prior are REGENERATED, not
migrated.** Pre-phase-3 evidence rows (axis-suffixed `TILE@k`-era spellings) are simply discarded
with the DB; no compat layer bridges them — the display bare-collapse in `tuning_knob_items` is
DELETED (keys render as stored), and no reader special-cases old spellings. What SURVIVES is the
bare-golden matching contract, which is a live semantic of the hand-curated YAMLs, not DB compat:
`pin_key_matches` / `family_value`'s bare↔explicit any-of is how a stored bare `REDUCE` matches a
row that also carries the cone stat's explicit `REDUCE@<axis>` key — and the step-7 re-spell KEPT
it deliberately (values re-spelled, keys untouched): the dynamic-attention bare-`TILE` rows make
it live corpus semantics, blocked on symbolic-trace keyed resolution (see the deferred list).

**Deferred work (deliberate residuals):**

- `TilePlan.units` field deletion (~150 consumer sites across the materializer): `TileOp.work` is
  the authoritative slot and `derive_workers` fails loudly on disagreement. The step-7 wire split
  removed the worker tokens from the VALUES without deleting the field — the value object keeps
  its validated, `work`-agreeing `units` until a dedicated consumer-migration commit.
- **BLOCKED — flash's special-cased pin plumbing** (the greedy all-or-nothing `TILE@dd`+`TILE@pj`
  contract, golden.py's dynamic-attention bare-`TILE` schema arm, `_narrow_flash_forms`'
  keyed-only arm + the masked-flash bare-`TILE` fallback) **and the bare↔explicit any-of**
  (`pin_key_matches` / `family_value`): the step-7 re-spell did NOT retire them, because they are
  LIVE corpus semantics, not legacy debt — the dynamic-attention golden entries (attention
  `.dynM` rows on every card) record the PV plan on a bare `TILE` precisely because a symbolic
  trace resolves no stable axis key, and the bare row must match the masked fork's axis-keyed
  leaves any-of. They die only when symbolic-trace KEYED resolution exists (a codec that spells
  a stable site key off a symbolic-axis tree); until then the exceptions stay documented and
  tested, never silently widened.
- Newly RECORDED fused entries spell the cone stat's `REDUCE@<stat axis>` key explicitly (no
  collapse); the axis name is trace-deterministic but ugly. The step-7 re-spell was VALUE-level
  only (keys untouched), so the move to the path form (`REDUCE@a.fold.k`) remains an open
  cosmetic item for whenever such entries are recorded.

**Step 4 — 1q: effects to the boundary — LANDED** (see the trail). Two deviations from the
original sketch, both deliberate and documented in the code: (a) the stores live as an explicit
`TileOp.stores` DECORATION (`Store` — the `Write` held whole plus the sweep spelling) rather than
being re-derived from `results` + graph output alone — flash's layout-aware index and `030`'s
workspace layouts are recognition-time facts the materializer cannot rediscover; the glue is the
reconstitution rule `effect_tail`, and byte-identity is enforced by `split_effects`' round-trip
gate at every conversion site. (b) strict `Lambda` formation is total over recognized ALGEBRA, but
the kernels that are raw loop IR (the un-recognized escape cell, `030`'s finalize, the prologue'd
partial, the coop fused-tail sibling) keep an impure fn through the one Map-private `_loop_ir_fn`
arm — the plan's original text did not inventory these; they carry no root `Write` in-term (only
iteration/seed structure) and they die as recognition grows toward totality, not at a re-keying
window. The graph-scope rehydrator therefore survives REPOINTED at that arm (escape dumps must
round-trip), no longer lenient for algebra terms.

**Step 5 — Phase 4, the placement realizer — LANDED** (`lowering/tile/_cut.py` + the `PLACE` path
family + the routing-entry loader split; tests in `test_placement_routing.py`, the two restored
020-era cone-cut e2e tests back from the xfail registry). All three done-gates verified on the
5090: the cone cut cascades recursively to statistic + scale + plain matmul (accuracy PASS; the
greedy-cold pair benches 4.9 µs vs torch's 6.0 at the gemma M=256/K=3840 shape), rms_norm
deploys unchanged under default fuse (byte-identical digests with routing off), and the 3-kernel
split-reduce form compiles + passes accuracy (the cut-out stat piece's own schedule g-splits).
Deviations, deliberate: (a) v1 realizes ONE cut per routing entry (the recursion covers every
cascade the examples spell; multi-cut entries raise loudly); (b) cut legality adds two structural
gates the plan text implied — single-component closed children only, and the pure-copy projection
seam of an empty-body root refused (the first cascade run found the non-terminating degenerate);
(c) the 020-era `cut_cone_*` schedule entries stamp the OLD piece shapes' keys (the eps epilogue
sat in 020's stat kernel; the new realizer cuts at the fold output), so the pieces resolve
existing KINDS but not those stale entries — phase 5 re-seeds them fresh, per the already-decided
"pre-wipe µs are not evidence" stance; (d) the loop-fusion brake on `__cut_` workspaces returns
(tune-mode slicing re-enters fusion with the pieces as ordinary pairs).

**Step 6 — Phase 5, THE consolidated parity gate** (detail under *Placement* below; NOT yet run — Dmitry
executes it with the evidence regeneration): re-seed the
retired PLACE goldens as routing + child schedule entries by hand-pinned `--ab` sweeps on both cards,
eval-golden MATCH across the board, twins from tier, TPOT/TTFT within noise.

**Step 7 — the ONE re-keying window: LANDED** (see the trail entry for the full mechanics). Deviations from the
original window sketch, all deliberate: (a) the `REDUCE` GRID finalize letter is KEPT in the site value (`g<n>a`
vs `g<n>k` is the finalize MODE, not an axis token — the original re-spell table mis-classified it); (b) `Body`
order canonicalizes at HASH time only (`_canon_order` inside `term_key`) — the stored term is never reordered,
since the lowered nest depends on storage order; (c) `from_loop`'s gate did not become an assert — it became the
raw-loop-escape discriminator (`from_loop → Fold | None`, byte gate on body/axis/unroll only), which is what
deleted the fallback-to-step-spelling; (d) `_composes` survives as the derived composed-reduce predicate rather
than dying with the step field; (e) the value re-spell touched VALUES only — keys, including the documented
dynamic-attention bare-`TILE` exception, are unchanged (see the BLOCKED deferred item above).

## Knob codec (phases 2–3)

Grammar: `FAMILY@<node-path>[.<axis>][<n>] = value`.

- **Families keep their names** — `TILE` / `REDUCE` / `STAGE` select which slice decorates the addressed
  fold (the family IS the slice); `PLACE` is the edge property; `RASTER` / `WSPEC` / `LOOPIFY` stay
  root-global and bare (`WSPEC` folds into `WORK` at the step-7 value-grammar split; until then
  unchanged). Full backwards compatibility of the outer key shape.
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
| flash | `TILE@dd` / `TILE@pj` / bare `REDUCE` / `STAGE` | the QK / PV in-step folds (QK an operand edge after step 7) / the kv stream |
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

**End state — the factored value grammar (a step-7 window item; phases 2–3 change KEYS only, never
values).** Today's VALUE spellings denormalize two kernel-global facts into per-site strings: the
worker inventory (`TILE`'s `w4x1` warp split and the scalar form's `n16x8` thread prefix, `REDUCE`'s
coop width `b512`) and the axis name (`REDUCE`'s `g2k`). The end-state codec stores each fact once —
the key names the site and axis, the value holds only site-local facts, the worker inventory is
spelled exactly once:

```
WORK                 = <worker inventory>          # kernel-global, exactly one: w4x1 (warps — the
                                                   # mma tier) / t16x8, t512 (threads — the
                                                   # scalar/coop tiers); `+p<n>` absorbs WSPEC
                                                   # (producer/aux warps are inventory too; live
                                                   # value today: 'p1'). The tier discriminator IS
                                                   # the worker kind — never a per-TILE spelling.
TILE@<path>.<axis>   = <atom>/f<FM>x<FN>/k<bk>     # atom (incl. accumulator dtype) + register
                                                   # repeat + serial-K; NO `w`/`n` worker token
REDUCE@<path>.<axis> = [g<n>][/coop[-t]][/r<n>]    # the STAGE PIPELINE (ReducePlan, coarse→fine:
                                                   # GRID split / BLOCK coop / REG), minus the axis
                                                   # token (in the key) and the coop WIDTH (= WORK);
                                                   # `-t` = the transposed-coop lane map (today's
                                                   # `b256t`), empty = serial. Live `g16k/b256t`
                                                   # composes — the value stays a pipeline, never a
                                                   # 3-way enum
STAGE@<path>.<axis>  = d<n>/<transport>/<pattern>  # already site-local — unchanged
PLACE@<path>         = cut | fuse                  # ROUTING entries only (see *Placement*) — a
                                                   # cutting entry stores no schedules
```

Full worked rows (term + schedule per kernel kind, including the cut/routing forms) live in the
examples table under *The IR (target)*. The step-7 re-speller's mechanical mapping, per stored value:

| Today | End state |
| --- | --- |
| `TILE: a:mma…f16_f32/w1x8/f4x1/k4` | `WORK: w1x8` + `TILE@fold.k: mma…f16_f32/f4x1/k4` |
| `TILE: n16x8/f4x8` (scalar) | `WORK: t16x8` + `TILE@fold.k: f4x8` |
| flash `TILE@dd: …/w4x1/f1x16/k4` + `TILE@pj: …/w4x1/f1x8/k8` | `WORK: w4x1` once + `TILE@fold.dd: …/f1x16/k4` + `TILE@fold.pj: …/f1x8/k8` |
| `REDUCE: g2k` | `REDUCE@fold.k: g2` |
| `REDUCE: b512` (rms coop) | `WORK: t512` + `REDUCE@fold.k: coop` |
| `REDUCE: g16k/b256t` (split-K + transposed coop) | `WORK: t256` + `REDUCE@fold.k: g16/coop-t` |

Unrepresentable by construction: disagreeing warp geometry across sites (flash's PV `w` token is DEAD
today — `realize_warp_twist` reads `um`/`fm` off the stream head only, so an inconsistent pin is
silently half-ignored, the worst failure mode); key/value axis mismatch; scalar/warp tier conflation
in one kernel. One cross-check stays a loud validation, not a type: a warp atom requires `WORK: w…`.
And `WORK`'s kernel-global-ness encodes today's TRUE invariant — fragment-resident dataflow between
composed folds shares the warp map — not a law of nature: a future fused form whose seam goes through
smem (barrier + re-read) could legally re-partition workers between phases; if that kernel ever
exists, relax the validation, don't pre-build per-site worker geometry.

This split re-spells every stored `TILE`/`REDUCE` value (~580 golden entries + DB + prior, both
cards) — by definition a re-keying event, so the wire format ships ONLY inside the step-7 window, via
a mechanical re-speller with the phase-2 compat test as the tripwire.

## Placement (phases 4–5)

`PLACE@<child-path> = cut | fuse` on every in-tree parent↔child seam — but resolution and storage are
TWO-LEVEL and RECURSIVE, never one compound row:

- **Two entry roles in the goldens.** A ROUTING entry stores the cut set only — NO schedules. A
  SCHEDULE entry stores ONE kernel's singular schedule (one `WORK` + its site slices under the
  end-state codec). The golden LOADER enforces the split: an entry mixing `PLACE` keys with schedule
  keys is rejected at load, not silently accepted. PLACE is wiped on this branch and the retired keys
  survive only as comments, so the factored format has ZERO migration cost — it is phase 4's design,
  not a later re-format.
- **Recursive resolution.** `(kind, shape)` → entry; a routing entry applies its cuts, then EVERY
  resulting piece (the cut children AND the residue) re-recognizes as a fresh root and resolves its
  own `(kind, shape)` through the full deploy hierarchy (goldens → DB/reservoir → prior), recursing —
  a piece's entry may itself cut. Terminates: trees strictly shrink. NO routing entry = fuse = the
  recognized form — the deployment-safety invariant holds by absence. A cutting entry stores no
  schedules at all.
- **Residue evidence reuse is the payoff.** The cone cut's pieces — stat kernel, scale kernel, plain
  matmul — are all EXISTING golden kinds: the cut arm inherits their whole evidence corpus, and
  re-tuning the matmul golden automatically improves every parent that cuts into it. The
  dual-spelling machinery is DELETED: a child decision has exactly ONE spelling, on its own tree.
- **`cut` mechanics**: split the tree at the seam — the child subtree becomes its own graph node
  (re-entering recognition as a fresh tree), the seam value materializes to a buffer (f32 for reduce
  seams, mirroring the split-reduce workspace rule), the parent consumes a plain `Load` where the
  child was (which is why every edge must admit `Load`; the `Map.sources` widening is 1i, deferred to
  the realizer's own commit). An in-tree cut child always has exactly ONE consumer (inline operands
  are single-consumer by tree ownership; the shared A is one edge), so the realizer needs no MIMO
  case — #433 stays a graph-level foundation.
- **`fuse` is the default on every seam; `cut` is evidence/pin-only.** The recognized tree IS the
  fused form, so the default means "no rewrite" — spelled as the ABSENCE of a routing entry.
- There are exactly two seam shapes: a `Map` projection seam and a fold operand edge — and every edge on
  the cut lattice is closed by construction (edge-iff-closed), so seam legality is structural. The
  realizer should know NOTHING about which seam it is — seam-specific knowledge must fall out of the
  node kinds.
- **Cuts resolve first; splits are per-kernel schedule.** A cut seam is a node output whose shape is
  a recognition fact — never a function of any schedule decision — which is what makes the two-level
  order sound. The `g<n>` split stays a schedule decision on whichever piece retains the reduce (its
  partials / synthesized finalize are post-rewrite artifacts inside that kernel's own compilation,
  never key targets). The cone cut's payoff is the stat kernel + scale kernel + plain matmul; at
  arity N the cut lands on the N-component `Load`-A fold — per-component separation is deliberately
  forfeited at tile level (de-fusing components is loop IR's decision; #389 measured null).
- Old → new: flash's `PLACE: fuse` → no routing entry; `PLACE@cone: cut` → a routing entry with the
  `a`-edge cut. The 3-kernel split reduce — previously inexpressible — needs no compound row either:
  `PLACE@map = cut` routes, and the cut-out stat kernel's OWN entry carries `REDUCE: g<n>` + its
  schedule.
- **Honest evidence accounting**: cut-vs-fuse is judged on the routing entry's recorded total
  (N-kernel pipeline vs the fused kernel) — now a claim about a CONFIGURATION of other entries at
  seed time, so the children's resolved schedules are recorded as provenance comments and the
  eval-golden audit is what catches drift when a child golden re-tunes.
- **Context-blindness is accepted deliberately.** A piece's own golden was tuned standalone (cold
  gmem inputs); post-cut its input may be L2-hot, so the standalone optimum can be mildly wrong in
  context. Safe by construction — a handicapped cut arm loses the A/B to fuse rather than deploying
  badly; a pin-form per-child override in the routing entry is the escape, built only if phase 5
  measures a real case, never speculatively.
- **Graph-level placement stays out of scope** (old `fin=fuse` / `stat=sink` crossed graph edges); the
  codec merely reserves its grammar. If the stat-tap plan lands later, its seam joins this namespace but
  keeps `cut` as its default (measured anti-wins) — the one exception to fuse-default.

**Phase 4 done when**: the recursive routing resolve is implemented; the cone cut reproduces the
recorded pair economics (~3.8 µs pair vs 6.0 µs fused on the 5090 per the YAML comments) with the
pieces resolving EXISTING golden entries; rms_norm deploys unchanged under default fuse (no routing
entry); the 3-kernel split-reduce form compiles and passes accuracy.

**Phase 5** carries THE consolidated parity gate for the whole refactor: re-seed the retired PLACE
goldens as ROUTING + child schedule entries (never compound rows) by hand-pinned `--ab` sweeps (the
manual method — the tuner is not used for golden work; both cards; pre-wipe µs are not evidence),
then `emmy eval golden --in-model` MATCH across the board, pin-only offer audit green, serving twins
deploy from tier, decode TPOT / TTFT within noise of the YAML-comment baselines.

## Completeness (proof sketch)

- **Legal cut points = node outputs = the edge set.** A cut along `v` is legal iff every element of `v`
  is complete before any consumer in the other kernel reads it; those are exactly the bound outputs of
  finished operators — the tree's edges. Enumerating SSA names over-generates candidates whose legality
  check is the tree, precomputed; with edge-iff-closed, legality is built into the edge set itself.
- **The only quantization loss is mid-pointwise cuts, and it is schedule-trivial**: an interior cut
  inserts a gmem round-trip between memory-bound stmts and never changes schedule class; anything that
  ever mattered is re-association (`Map ∘ Map`) — a change of algebra, not codec. Every cut the old
  system deployed lands on a node seam in the new trees.
- **The fork space is a finite, generically enumerable product — and it FACTORIZES.** Level 1:
  Π {fuse, cut} per seam (legality structural — state materializability, f32 workspace, no
  graph-output crossing). Level 2, per resulting kernel: Π schedule-family vocab per
  (path, node, axis) site × root-globals — searched INDEPENDENTLY per piece, so a piece's evidence is
  shared across every parent and shape that cuts to it; the reserved graph-level triple is also
  finite. The site registry is DERIVED from one walker; only per-family vocabularies and legality gates
  stay hand-written. Enumerable ≠ cheap: cut sets compose (2^seams), so level 1 stays prior-ranked,
  and evidence-only-`cut` keeps the space unpaid-for cold.

## Retirement ledger (everything still standing that is GONE at end state)

The deletion contract, re-audited after step 7 (2026-07-30, grep-verified). When a step lands, grep for its
retirees — a stated deletion that still answers a grep is not done (the refactor-invariant rule).
The already-dead (`020`/`025`/`032` + `_sink.py`, `TileOp.bindings` + `ops.resolve`, `Fold.role`,
`Map.body`/`out`, `_best_fork`, `Fold.tile/reduce/stage` + the node-slice
stampers, the `Monoid` class, `knob.resolve_axis`, `_schedule._at`, the `tuning_knob_items`
bare-collapse, and — with step 7 — `Fold.step`, `flash_combine`/`_split_pv`, the `from_loop` step-spelling
fallback, `WSPEC` as a stamped row family + the denormalized value spellings, and the lowered-nest identity)
live in the *Landed trail*; this table is only what remains. Two step-7 rows RESOLVED AS BLOCKED (live corpus
semantics, not legacy debt — see the deferred list). The
placement dual-spelling apparatus is cancelled before construction (factored golden storage) —
nothing to delete. Pre-phase-3 DB/reservoir/online evidence is REGENERATED, never migrated.

| Retiree | Home | Replaced by | Dies at |
| --- | --- | --- | --- |
| `TilePlan.units` (the FIELD) — the `w…`/`n…` worker tokens LEFT the wire at step 7(v) | `ir/schedule.py` | the ONE `Workers` slot — `TileOp.work` / `derive_workers` authoritative and loudly validated; the field itself survived the wire split deliberately (~150 consumer sites) | a dedicated consumer-migration commit |
| ~~`effectful_lambda`~~ — DONE at 1q (deleted; the rehydrator survives repointed at the Map-private raw-loop-IR arm `_loop_ir_fn`, which itself dies with recognizer totality) | `ir/stmt/body`, `graph.py` | strict `Lambda` formation; `TileOp.stores` + the `effect_tail` reconstitution carry every root store | ~~1q~~ landed |
| ~~`captured_values` as the attachment/legality decider~~ — DONE at 1q (demoted to the validation reading) | `ir/tile/ir.py` | edge-iff-closed by construction | ~~1q~~ landed |
| **BLOCKED** — the bare-golden matching arm: `pin_key_matches`' bare↔explicit any-of + `family_value`'s pooled read + `axis_of`'s featurizer grouping | `pipeline/knob.py`, `search/features.py` | the step-7 re-spell rewrote VALUES only and deliberately kept this: the dynamic-attention `.dynM` corpus records the PV plan on bare `TILE` (a symbolic trace resolves no stable axis key) and must match axis-keyed leaves any-of — LIVE corpus semantics | symbolic-trace keyed resolution |
| **BLOCKED** — flash's special-cased pin plumbing remnants: the all-or-nothing `TILE@dd`+`TILE@pj` contract (`greedy.py`), the dynamic-attention bare-`TILE` schema arm (`golden.py`), `_narrow_flash_forms`' keyed-only arm + the masked-flash bare-`TILE` fallback (`_schedule.py`) | `search/`, `_schedule.py` | the live mechanics of the same bare-`TILE` contract (the row above) — generic codec keys can replace them only when a symbolic-axis tree spells a stable site key | symbolic-trace keyed resolution |
| ~~`Fold.step` (the composed step sequence) + the step-splice arm of `Fold.loop`~~ — DONE at step 7(i–iii) (the derived blocked evaluation; QK a hoisted operand edge; `_composes` deliberately SURVIVES as the derived composed-reduce predicate reading `step_stmts()`/operand edges; residual stale docstrings in `ir/tile/ir.py`/`_flash.py`) | `ir/tile/ir.py`, `_schedule.py` | the derived blocked evaluation of `combine` | ~~step 7~~ landed |
| ~~`Carrier` / `Twist` / `State` + every reader~~ — DONE 2026-07-30 (the flat `Algebra(combine, terms, dtypes)` annotation; `Fold.carrier` a derived property; realizer/030/graph arms read `names`/`terms`/`ops`/`combine_states` off it) | `ir/stmt/algebra` + tile/kernel lowering | structural derivation off the stored `combine` | ~~follow-up window~~ landed |
| ~~`_carrier.py` — the Carrier-assembly layer~~ — DONE (deleted; `Algebra.exp_family` is the constructor; `projection_distributes` moved to `ir/stmt/passes.py` — a digit-named pass module can't be imported from) | `lowering/tile/_carrier.py` | recognition builds `(init, combine)` directly via the `ir/stmt/carrier` generators | ~~with the retirement~~ landed |
| ~~`ir/stmt/carrier.py`'s Carrier-facing surface~~ — DONE (`id_*` deleted, emission inlined into `Algebra`; `Channel` deleted too — a DEVIATION from the survive-plan: `terms` subsumed it, `exp_merge` takes `(state, terms)`; `exp_combine_states` + the generator/stabilizer/certificate survive) | `ir/stmt/carrier.py` | the stored `combine` + singleton specialization | ~~with the retirement~~ landed |
| ~~the `from_loop` byte-identity FALLBACK + the `_lambda_of_*` reader tolerance~~ — DONE at step 7(iii), with a deviation: the gate became the raw-loop-escape discriminator (`from_loop → Fold \| None`; byte gate on body/axis/unroll only), not an assert — asserting would forbid the legitimate raw-loop-IR kernels | `ir/tile/ir.py` | every recognized fold λ-spelled by construction; non-λ shapes keep the `Map` raw-loop escape | ~~step 7~~ landed |
| ~~`WSPEC` as a root knob family; the denormalized value spellings (`REDUCE`'s coop width `b512`, `TILE`'s worker tokens)~~ — DONE at step 7(v) (the `WSPEC` Knob + legacy spellings survive as loudly-validated env-pin ALIASES; the `g<n>a\|k` finalize letter KEPT — a mode, not an axis token) | goldens/DB/prior, `knob.py` codecs | `WORK` + site-local values (mechanical re-speller, 715 rows) | ~~step 7~~ landed |
| ~~lowered-nest kernel identity (`structural_key` lowers first)~~ — DONE at step 7(iv) | `ir/tile/ops.py` / `search/keys.py` | the α-invariant hash of the canonically renumbered term (`ops.term_key`) | ~~step 7~~ landed |

`_schedule.py` splits across three fates, which is the cleanup the codec buys: the
candidate vocabularies and option/row builders (`_stage_candidates`, `_reduce_candidates`,
`_warp_option`, `_splitk_option`, …) SURVIVE as the per-family vocab the walker-derived site
registry consumes; the key-spelling layer DIED into the resolver (steps 1–3: `_family_key` is the
one speller, `_at` is gone) and the node-slice stamping re-keyed to the `TileOp.schedule` dict
(1r). The flash pin-narrowing remnant (`_narrow_flash_forms`) survives past step 7 — BLOCKED with the
bare-`TILE` contract on symbolic-trace keyed resolution. Anything in it
spelling a knob key by hand is a bug.

## Risks

- **The flash dissolution (step 7) stays the subtlest move**: the derived blocked evaluation must
  reproduce today's step material through the twist realizer's reads (flash digests including
  split-KV are the gate; the shape-derived role story landed with 1p and must survive the PV fold
  becoming derived); keep the associativity + agreement tests green before touching the emitter.
  It lives INSIDE the one re-keying window by design — residual spelling drift re-keys there,
  never silently outside it.
- **Purity erosion in lambdas**: the whole design rests on state entering `combine` as params and
  leaving as results — `Accum` (or any effectful stmt) creeping into ANY stored `Lambda` (lift,
  combine) re-creates dissolved-state storage. The `Stmt.pure` trait + `Lambda.__post_init__`
  formation validation (conservative default: a new stmt kind is impure until declared; the
  structural nodes `Fold`/`Map` declare pure — a term is a value) is the guard; never bypass it with
  a pre-built `Body`.
- **Parity is settled once, at phase 5** — intermediate phases carry only unit-level verification plus
  the digest harness; if the eval-golden pass surfaces broad drift, bisect back to the phase-1 commits
  rather than patching goldens forward.
- Stored-short-key ambiguity from future structural changes — the resolver fails loudly by design; the
  phase-2 compat test is the tripwire. Never "fix" it by silently re-keying evidence. (The tune
  DB/prior are regenerable and carry no key contract; the GOLDEN YAMLs are the hand-curated corpus
  the tripwire protects.)
- **The factored placement storage couples entries**: a routing entry's recorded total is only
  reproducible against the child goldens as of seed time — a child re-tune shifts it silently until
  the eval-golden audit runs. The provenance comments + the phase-5 audit are the guard; never "fix"
  drift by re-benching only one side of the cut A/B.
- Dump/kname churn: kernel names derive from realized ops; verify the per-kernel torch-reproducer
  slicing still attributes the cone's ops correctly from the operand edge.

## Cleanup

Docs at the end of each landed phase: the pipeline ARCHITECTURE (knob/fork system), the tile-lowering +
kernel ARCHITECTURE files, and CLAUDE.md's tile-lowering blurb. Delete this plan when step 7 (the
re-keying window) lands — it is the last item, after phase 5.
