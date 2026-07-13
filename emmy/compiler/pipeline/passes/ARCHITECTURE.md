# Pass-authoring invariants

Rules that apply to EVERY pass in this tree (`frontend/`, `loop/`, `lowering/`). Per-dialect details live in
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) (pass order, knob table, fork semantics). The **tile-lowering** phase
(`lowering/tile/`) is the canonical instance of the invariant below — a **purely algebraic moveset, no
specializations**: it dispatches on the carrier algebra (`MAP` / `SEMIRING` / `MONOID`), never on a named shape
(matmul / pointwise / attention) — flash attention is the `MONOID` algebra on the streaming schedule (a twisted
monoid is a monoid), selected structurally, not a distinct kind.

## No shape-specific pattern matching

A pass must not dispatch on enumerated shapes ("if this is the gated-MLP body do X, if it is the QK^T body do Y").
Each named shape that needs handling is evidence of a missing GENERAL rule; find the per-element formulation that
makes the old and new shapes degenerate cases of one code path. Shape dispatch compounds: every new model
architecture would add a sibling branch to every pass it touches — the combinatorial explosion of compiler
complexity this invariant exists to prevent. It also breeds divergent incidental behavior (per-branch dtype or
layout rules that drift apart) and silently narrows coverage to the shapes someone happened to name.

How to comply:

- **Write the rule per element, not per shape.** Example: `lowering/tile/_atomize.map_cone` /
  `bind_prologue_contraction` classify each ⊗-fold operand independently (plain `Load` stays put; a computed cone is
  bound as the shared A value by value-tree equality, however many fold channels read it). Norm→linear, gate/up +
  SwiGLU, scale→matmul, SDPA P@V, and rotary QK^T are *instances* of that one rule, not branches — and a shape
  nobody designed for (a weight-side dequant cone) is covered for free.
- **Gate in the negative.** Enumerating admissible shapes is shape matching by another name. Walk the body and
  report the first thing the transform *fundamentally cannot do*, like `lowering/_predicates.classify_fragment_epilogue`
  (the epilogue folds unless it has an ineligible op/dependency) — the eligible set then grows with the renderer
  instead of with a hand-maintained list.
- **Bail conservatively on well-formedness, never on shape identity.** `return None` / `RuleSkipped` for a body
  the rule doesn't fully understand is fine; the conditions must be structural properties (escaping values,
  symbolic extents, mixed dtypes), not "is this the X kernel".
- **Phrase dataflow conditions over cones, don't hand-roll the walk.** `Body.backward_cone` / `forward_cone` /
  `defs_die_at` (`ir/stmt/body.py`) are the shared slicing substrate: a rule asks for a cone and judges its
  *properties* (members, external reads, escapes) — construction never fails, so every bail stays a rule-side
  condition. See the dependence-cones section of `compiler/ir/ARCHITECTURE.md`.
- **When generalizing an existing rule, normalize its incidental divergences** (one dtype rule, one index rule)
  and name the behavioral deltas explicitly in the commit — don't preserve two behaviors behind one entry point.

## Resolve the hardware-atom binding once, structurally, at the tile level

The same invariant applies *across* the tile→kernel boundary: the kernel materializer must not re-recognize structure
the tile IR already holds. The **atomize** step (`lowering/tile/_atomize.py`, called from the `_schedule` helper inside `010_recognize` when it builds
the warp / register-tiled option — *not* a standalone pass) resolves the algebra→hardware-atom binding once at fork-emit
and feeds it into the `Contraction` structural node (`_schedule._contraction_node`), so materialize reads the operands /
`acc` / epilogue off the node and only `factorize`s. Resolving it at option-build time means an atom that **cannot** be
bound (e.g. a non-`Load` operand — a computed-cone / demoted matmul) is rejected at fork construction, alongside
`_check_warp_static_k`, instead of failing several passes later:

- a `CONTRACTION` contraction → the `(a_load, b_load, acc, epilogue)` operand→role facts
  (`_atomize.bind_contraction`): the A/B operands bound to roles by which output grid axis each operand's OWN leaf `Load`
  index carries (structural — read off the annotated loop, not a flattened-loop scan), plus the fold accumulator and the
  projection epilogue. The binding now happens ONCE at **recognize time** (`010_recognize._nodify_contraction` — every
  recognized contraction, per-cell scalar included, is a `Contraction` node with a deferred `TilePlan()`; an unbindable
  one — a 1-D matvec-shaped output — demotes to `PLANAR` and folds as an ordinary `Reduction`); the schedule fork only
  swaps the node's `tile` field (`_schedule._contraction_node`), and `_factor.factorize` reads the facts off the node
  instead of `lower()`-ing the contraction and pattern-matching the result. A `STAGE` pin follows the same rule: the
  option builders resolve it against the built node ONCE (`_resolve_warp_stage` / `_resolve_scalar_stage` — transport
  eligibility, the slab K-chunk `bk_elems`, the depth clamps) and stamp the resolved `Stage` (or `None`, gmem-direct)
  on the `TileOp`, so the materializer's one staged driver applies it verbatim, deciding nothing. Fitting the smem
  budget is part of resolving: a tile whose single depth-1 slot already exceeds it declines to gmem-direct (the warp
  slab is codec-sized and cannot shrink; the scalar resolver steps `bk_elems` / depth down first), so every offered
  stage row materializes within budget — a resolved-but-unfittable row would only die at `validate(ctx)`, leaving an
  un-lowered `TileOp` in the tune's terminal (issue #327). One staging fact
  is derived at materialization rather than resolved here because it is layout, not eligibility: a slab feeding an
  mma drain is **swizzled** (`_stage.pick_swizzle_atom` picks B32/B64/B128 per operand from the slab's inner row span;
  TMA permutes the 16 B chunks in hardware during the box copy, a cp.async fill applies the identical XOR in software
  on its destination index, each staged `LdmatrixLoad` XORs the address back, and the kernel stays bit-identical to
  its unswizzled sibling — swizzle relocates smem bytes only, which is what keeps the ldmatrix drain free of
  shared-memory bank conflicts; the unswizzled cp path was 4-way/8-way conflict-bound on sm_89, the measured residual
  to cuBLAS there. The fused-edge **sync compute-fill** slabs swizzle the same way — the compute/copy
  fill's slab `Write` carries the mode (the identical flattened-index XOR, one VECTOR store per 16 B run; V scalar
  2 B stores at 16 B thread stride were 8-way store-conflicted), the canonical-B cp.async fill its `Operand`, and
  the drain reads each slab back through its own mode. Unswizzled, the gemma-shape fused edge drained 294.9 M ld
  conflicts / 82.5 M LSU inst on the 5090; the swizzle + vector fill store recovered -29% (std) / -41% (fm).
  Scalar-`Load`-drained slabs stay plain row-major).
- the **MONOID-producer composition** — the fused norm→linear edge and its N-channel form, the gate/up MLP edge —
  binds at recognize time too (`_atomize.bind_prologue_contraction`, structure-only): a projecting `Map` over a
  per-row `PLANAR` statistic whose tail is one or more ⊗-folds of one shared A value nodifies to
  `Map(body=projection, source=Contraction)` — the computed-A `Contraction` carrying the statistic prologue in its A
  cone and the `(B, acc)` channels on `folds` (a product-monoid fold: channels never interact per step; the combine
  — SwiGLU — is projection, riding the wrapping `Map.body`). `010_recognize` schedules it as a fork SIBLING of the
  cooperative reduce form (option-0 stays the coop row; the warp mma rows ride the mandatory `sync` compute-fill;
  dtype / geometry legality stays schedule-side in `_computed_a_rows`). This retired the pin-only
  `_prologue_warp_option` rescue.
- a cooperative / ILP reduce (`PLANAR` / `TWISTED`, or a non-output-tiled `CONTRACTION`) needs **no** binding here — its
  accumulator dtype + the shuffle/tree fold mechanism are **derived** at materialize time (`emit_combine` off the carrier
  + `ReduceStage.combine`), never stored. Its one schedule-time staging decision follows the same
  resolve-once-structurally rule: `_schedule._row_stage` detects the fused norm→linear shared row when the cooperative
  partition is chosen and stamps a `sync` `Stage` naming it (`smem`) on the `TileOp` — a derived schedule field, not a
  knob — so `_factor._tile_reduce_axis` only applies it, never re-detects.

The atom spec is subtyped by kind (`ir/atom.py`: `AtomKind` is the fixed mma cell selected by name; `ScalarAtom`
is the plain scalar fma cell). The contraction binder (`bind_contraction`) is loop-addressable so warp-flash can later
reuse it on flash's nested QK^T / PV; flash's inner score IS now a structural `Contraction` **node** (per-cell
`TilePlan()` today, `source` of the streaming `Reduction` — the `Reduction ⊃ Contraction` composition), so warp-flash is
just that node gaining a warp `TilePlan` — no new path.

**The f16-accumulate atom sibling** (`mma_m16n8k16_f16_f16`, C→f16 — atom names follow
`mma_<shape>_<ab_dtype>_<acc_dtype>`, the compressed PTX/CUTLASS D.A.B.C order; the historical acc-unspecified
spellings stay as parse aliases for the f32-accumulate atoms): on the consumer GeForce dies (sm_86/89/120)
f32-accumulate HMMA runs at HALF the f16-accumulate rate, so this atom keeps the whole mma chain on the full-rate f16
accumulator and the lowering promote-folds the packed f16 partials into f32 shadow fragments per K chunk
(`FragmentPromote` — the staged bk slab is the cadence; gmem-direct promotes every `_atom._F16ACC_STEPS` steps plus a
final fold; flash promotes the P@V accumulator per streaming KV block, folded in at the `O·α` rescale point, while the
score node ALWAYS stays f32-accumulate). Precision-gated enumeration, off by default: `_schedule._f16acc_allowed` —
the precise `EMMY_F16_MMA_F32_ACC` pin is authoritative on any target, else the `EMMY_FAST_MATH` umbrella offers it on
the consumer-die ccs only (`_F16ACC_CCS`); a `TILE` pin naming the atom (or the flash golden's axis-keyed
`TILE@<pv_k>` spelling) bypasses the gate — pins are authoritative. The realized fork is identified by the `TILE`
codec's atom token and priced by the `MMA_acc_bits` feature; f16 only (mma.sync has no bf16-accumulate form).

**The move catalog** (`search/space.py`) is the permitted-move enumeration the schedule emit forks over, keyed on
`AxisRole`: `scalar_tile_moves()` is the legality-guarded scalar register-tile product (`par × reg`, `block_threads ≤
1024`) with per-cell `""` as the conservative option-0, crossed with the warp / reduce / stage move families by
`_schedule._tile_rows` for an unpinned contraction so `compile` / `tune` explores the space (each row → a structural
`Contraction`-node leaf keyed `TILE@<k_axis>` in a hierarchical `build_fork_tree`; an env pin wins via `Knob.narrow`).
`wspec_moves()` is the fourth level (bare `WSPEC`, option-0 `""` = uniform SIMT) — offered only on a warp row over a
resolved **TMA** stage without a cross-CTA split, and resolved/thread-budget-gated at materialization
(`_wspec_workers`; an ineligible spec degrades to uniform). A computed-A (fused-cone) contraction enumerates its own
warp-only rows (`_schedule._computed_a_rows` — the mandatory resolved `sync` compute-fill stage, no scalar /
gmem-direct / split-K / WSPEC rows; the compute-producer role for the fused edge is the anticipated `RoleKind`
extension). The **flash-form fork**: a `TWISTED` streaming contraction pair (the flash tree) offers its
structurally-different schedules as ONE prior-ranked fork — the warp (fragment-resident) rows over
`twisted_warp_moves()`'s `(warps-per-CTA × key-atoms-per-block × query-tiles-per-warp)` geometry grid (option-0 = the
conservative one-warp / `2·atom_n` block / one tile; the third dimension is the `TILE` codec's `f<FM>x<FN>` reg_m —
each warp streams `fm` independent `(m, l, O)` chains against shared K/V fragments, FA-2's in-flight ILP; the Q@K /
P@V mma `TilePlan`s are derived per point, `_schedule._twisted_warp_options`), the scalar
register-vector CHAIN (the FA-2 shared-score form), then the cooperative / per-cell reduce-partition escapes — every
leaf row spelling the same `TILE@<qk_k>` / `TILE@<pv_k>` / `REDUCE@<kv>` key set (decided-empty where a form doesn't
tile). A non-empty `REDUCE` pin remains the scalar escape; a **warp** `TILE` pin keeps the mma rows alone (loud on a
divisibility violation, declining with a log line when the pin doesn't fit the flash form — a bare warp pin may target
another kernel), while a non-warp `TILE` pin narrows the flash rows by their stamped per-node spellings
(`_schedule._narrow_flash_forms`, codec-canonicalized so `a:scalar` ≡ `""` and `f64x1` ≡ `f64`): `TILE=a:scalar` keeps
the per-cell tier, `TILE=a:scalar,TILE@<pv_k>=f<d>` pins the CHAIN row deterministically, and an unmatched pin keeps
the full prior-ranked fork. Each warp geometry row crosses with its **K/V operand-stage** candidates (`STAGE@<kv>` —
`_schedule._twisted_stage_candidates`: gmem-direct option-0, then the resolver-gated cp.async AND TMA ring depths — the
batched K/V operands encode as rank-N TMA boxes with leading extent-1 dims, the load's own batch/head index exprs
riding as origin coords; cp.async slabs take the +16 B row pad, TMA slabs stay dense under the hardware swizzle; the
resolved `Stage` rides the `TileOp` and the streaming step becomes the `staged_kloop` drain, K/V slabs kept in each
operand's own layout so staging stays bit-identical to gmem-direct). **Both transports also stage a symbolic
(dynamic-`seq_len`) kv**: TMA rides the runtime globalDim and zero-fills the box overhang past the last key; cp.async
(which has no OOB zero-fill) clamp-reads the tail chunk's key rows to the last valid key. Either way the streaming
drain's tail masks (the same clamp the gmem-direct symbolic path makes) zero those keys' P columns exactly, so the
masked-flash `.dynM` kernel stages at bit-identity to gmem-direct on any sm (the `staged_kloop` ring allocates the
full depth and the last-chunk clamp / loop bound ride the symbolic `Dim`; WSPEC over a symbolic kv is not built). A resolved TMA row additionally offers the `WSPEC` producer-band splits (the matmul tier's
legality, `32·aux ≤ 32·um`; measured occupancy-negative at flash's CTA scale — offered, honest, not the default). The
chain / coop / serial escapes stamp the decided-empty `STAGE@<kv>: ""`. The causal tile-skip is the remaining flash
follow-up.
Two catalog invariants hold: every recorded golden's `TILE`/`STAGE`/`REDUCE` stays a **member** of the enumerated
grids (the permanence test in `tests/compiler/test_golden_configs.py` — a space edit can never silently orphan a
golden into unreachability again, the sixth sweep's `.s512` regression class; the scalar reg grid carries the
golden-informed deep-FM points `f2x6..f2x14`, `f4x6..f4x26` for exactly this reason), and a cross-CTA split deploy
(`030_split`) stamps the decided knob row onto its **partial** kernel — the engine merges knobs forward on 1:1
rebinds only, so without the explicit stamp the graph splice dropped them and the deployed split recorded no
schedule identity (the A/B table then couldn't say what greedy deployed).
