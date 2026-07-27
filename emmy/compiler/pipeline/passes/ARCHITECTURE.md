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

Loop fusion's profitability guard follows the same structural rule. `loop/fusion/010_merge_loop_ops` counts aggregate
arithmetic and reads, and separately counts transcendental executions. The separate count prevents an expensive
pointwise producer such as GELU's `tanh`, originally evaluated once per `(M,K)` element, from moving under a
contraction's `N` loop merely because the contraction's much larger cheap-FMA count hides that duplication. Flash
attention is the structural exception: a merged softmax-then-P@V offer deliberately streams `exp(score)` without
materializing the probability matrix, so the tile recognizer owns that composite and the generic transcendental
brake does not split it. The automatic brake is also limited to unary/single-source activation cones. A multi-source
gated activation such as GeGLU/SwiGLU remains an evidence-controlled placement choice, preserving its GPU golden
coverage until that entire cone has been measured.

## Resolve the hardware-atom binding once, structurally, at the tile level

The same invariant applies *across* the tile→kernel boundary: the kernel materializer must not re-recognize structure
the tile IR already holds. The **atomize** step (`lowering/tile/_atomize.py`, called from the `_schedule` helper inside `010_recognize` when it builds
the warp / register-tiled option — *not* a standalone pass) resolves the algebra→hardware-atom binding once at fork-emit
and feeds it into the `Contraction` structural node (`_schedule._contraction_node`), so materialize reads the operands /
`acc` / epilogue off the node and only `factorize`s. Resolving it at option-build time means an atom that **cannot** be
bound (e.g. a non-`Load` operand — a computed-cone / demoted matmul) is rejected at fork construction, alongside
`_check_warp_static_k`, instead of failing several passes later:

- a `CONTRACTION` contraction → the `(a_operand, b_load, acc, epilogue)` operand→role facts
  (`_atomize.bind_contraction`): the operands are named by the ⊗ **lift** (the `Assign` the fold accumulates) — B is its
  (n, k)-indexed `Load`, A is the lift's other argument, either a plain `Load` (clean gmem-direct) or, when loop fusion
  has inlined an operand cone, the cone itself as a `Body` (a STAT-FREE computed A, which rides the `sync` compute-fill
  like the norm→linear cone but carries no statistic prologue) — plus the fold accumulator and the projection epilogue.
  Binding off the lift rather than off "the first (m, k)-indexed `Load`" is load-bearing: a cone-INTERNAL load is
  (m, k)-indexed too, so the positional rule bound gemma's GeGLU combine as `gate @ W` and silently dropped the gelu and
  the up projection. Refusing to bind a stat-free cone at all is equally wrong — it demotes the cell to a PLANAR
  scalar fold, which cost the gemma-4 M=256 post twin 144 ms against 4.3 ms bound. The binding now happens ONCE at **recognize time** (`010_recognize._nodify_contraction` — every
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
  un-lowered `TileOp` in the tune's terminal (issue #327). TMA's box rank follows the flash convention on the matmul
  tiers too: the box's data plane is the operand's trailing 2 gmem dims, and extra LEADING dims ride as extent-1 box
  dims whose origin coordinates are the operand's own index exprs — eligible when those exprs don't move with the
  tile or the K loop (`_tma_operand_rank_ok`), so a model's `[1, seq, K]` unit-batch view stages exactly like the
  rank-2 snippet twin (the gemma in-model matmuls' TMA lockout). A **transposed B** (the serving `F.linear` layout —
  B given `(N, K)`, K gmem-contiguous, `Contraction.b_trans`) stages on the warp tier through an **N-major slab**:
  the B slot takes A's geometry (`tile_n × bk`, K the inner dim — stride-1 in gmem and smem alike, so cp.async chunks
  and the TMA box stay contiguous; `Operand.trans` stamps the layout) and the drain is the plain no-`.trans`
  ldmatrix (`LdmatrixLoad(b_trans=True)` — the same staged path flash's K slab rides). Both operands' inner span is
  then the K chunk, so the eligibility alignment gates on K alone (`_can_stage_warp` / `_can_stage_warp_tma`) and
  the B swizzle mode derives from `bk_elems` like A's. Historically the transports declined transposed B and the
  serving `.lin` forks ran gmem-direct only — the 1.3–2.75× gap class to cuBLAS on the 5090 goldens. The scalar
  tier still declines it (its plain-`Load` drain has no transposed variant; pin-only tier). The **sync compute-fill**
  (the fused computed-A edge) stages its transposed B the same way: every B fold channel — canonical K-major or
  transposed N-major — rides a vectorized `cp.async` `Operand` that flies UNDER the compute fill (the per-cell
  strided copy fill it replaced was the serving fused edges' weight-stream deficit), and the asymmetric B-only `d2`
  prefetch ring is enumerable on transposed-B fused edges too. One staging fact
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
  `_prologue_warp_option` rescue. The **degenerate M=1** composition (per-token decode: the unit row axis elided,
  `free = ()`) binds too — a synthesized unit free axis keeps the column grid and the cut anchor
  (`020_cut_edge` accepts `free ≤ 1`); without it the fused kernel schedules at grid 1, ~300× off the memory
  floor. The M=1 cut consumer is also why annotated-loop rewrites map the `Carrier` through SSA renames
  (`Carrier.rename` — a verbatim carrier left the cooperative combine reading a state name the renamed body no
  longer defined).
- a cooperative / ILP reduce (`PLANAR` / `TWISTED`, or a non-output-tiled `CONTRACTION`) needs **no** binding here — its
  accumulator dtype + the shuffle/tree fold mechanism are **derived** at materialize time (`emit_combine` off the carrier
  + `ReduceStage.combine`), never stored. Its one schedule-time staging decision follows the same
  resolve-once-structurally rule: `_schedule._row_stage` detects the fused norm→linear shared row when the cooperative
  partition is chosen and stamps a `sync` `Stage` naming it (`smem`) on the `TileOp` — a derived schedule field, not a
  knob — so `_factor._tile_reduce_axis` only applies it, never re-detects.

## The divide rules: `cut` a dataflow edge, `sink` a row statistic, `split` an iteration axis

`lowering/tile` carries three one-kernel→graph-fragment rules whose names sound alike but divide along different
dimensions, with their own re-entry contracts — keep them apart:

- **`020_cut_edge`** cuts the **dataflow** (the PLACE codec's memory edge, `PLACE@cone=cut` today): two different
  computations that loop fusion welded together — the producer cone and its matmul — become separate kernels again,
  the intermediate materialized to a workspace. Its pieces are emitted as plain **un-mapped `LoopOp`s** precisely so
  the pass-scan restart hands them back to `010_recognize`: the fused kernel's schedule is meaningless for the
  halves, and each must earn its own recognition, schedule and fork (the gmem-A matmul then reaches the staged
  cp.async/TMA tiers a computed A can never ride). The statistic materializes as its OWN kernel — a nested
  `for m: [stat…, for k: …]` producer only lifts `m` onto the grid, serializing the K sweep per thread. A
  **multi-fold** cone (the gate/up ⊗-folds — `swiglu(x̂@Wg, x̂@Wu)`) can't cut into one consumer (a multi-fold plain
  matmul has no mma tier), so it splits into **N single-channel matmuls** (each rides `d2/tma/ring` and beats cuBLAS)
  writing per-channel workspaces + a downstream pointwise **combine** kernel carrying the ⊗-combine (GeGLU) tail — the
  form that beats cuBLAS at prefill M (~270 vs ~317 µs eager on the 5090), which the fused computed-A megakernel can't.

  Like `030_split_reduce`, this rule **decides nothing** — it realizes a decision already in the schedule.
  `PLACE@cone` is a knob `010_recognize` enumerates as a fork sibling (a `cut`-stamped row beside the fused warp-mma
  and coop-reduce rows) and stamps on the chosen op; `020_cut_edge` reads that stamp, exactly as `030_split_reduce`
  reads the `GRID` reduce stage. That is what makes the placement selectable by the ordinary **evidence pick**: a
  golden records `PLACE@cone: cut` against the cut's measured TOTAL (stat + cone + channels + combine), the same
  whole-cost convention a split-K golden uses for partial+finalize, and `_golden_pick` compares it to the fused rows
  like any other knob. No Σ-of-predictions structural pricing is involved.

  **The cut row is evidence-only.** `greedy_decide` withholds it from the model fallback, so it can win only where it
  was actually measured. This is a safety property, not a preference: the cut is catastrophic where its consumers have
  no golden — the gemma-4 gate/up edge measures 35,558 / 71,160 / 142,702 µs at M=32/64/128 (the channel matmuls fall
  to a scalar tile) against 368.7 µs at the seeded M=256. An unseeded shape simply never matches and stays fused, so
  the choice is never made geometrically.
- **`025_sink_row_reduce`** migrates a **row statistic** across the dataflow edge in the OPPOSITE direction from the
  cut (the PLACE codec's `stat` element, `PLACE@stat=sink`): a fused norm kernel B —
  `Map(body=[π…, sweep Loop], source=Reduction(PLANAR))` whose reduce folds a pure per-cell term (`Σ x²`) of ONE
  tensor its in-graph producer A just computed — drops its `Reduction` to a `Load` of a fresh f32 `T__sq[row]` aux
  buffer, and A's complete-value store site (a split-K FINALIZE or pointwise sweep — never an atomic partial, whose
  per-partition values are incomplete) gains the term re-evaluated on each just-stored value plus a `RowAccum`
  (a hierarchical warp-shfl + smem block fold + one `atomicAdd` per block, zero-init'd per launch through the
  ordinary `zero_outputs` memset machinery). `T__sq` is an `AuxOutputOp` graph node (no launch of its own — the
  producer's launch writes it); B re-emits as an un-mapped `LoopOp` so `010_recognize` peels BOTH the row and sweep
  axes free — the whole point: the grid-per-row latency-bound stat kernel becomes a wide 2-D pointwise sweep. The
  structural probe (`_sink.bind_sinkable_stat`) gates in the negative: the reduce's flat address in T must be affine
  with reduce coefficient 1 and mixed-radix row coefficients (so `row = flat / N` is a bijection — the per-head
  qknorm case falls out for free), the projection must be `[pure prefix…, one sweep Loop]`. Like the cut it is a
  pure realizer of a recognizer-offered stamp, its rows are **evidence-only** (withheld from the model fallback),
  and its sink siblings mirror EVERY local row so an unrealizable site (atomic producer — the pass waits via
  `RuleSkipped` for `010`/`030` to settle the producer, then refuses permanently) degrades to the picked row's own
  local-stat schedule. It runs BEFORE `030` so a sink-stamped row is realized before any grid split of the norm
  could be.
- **`030_split_reduce`** splits the **reduce axis** (the REDUCE codec's `g<w>` cross-CTA shard): the SAME
  computation, its K partitioned across CTAs into a partial + finalize. It runs AFTER its decision — the `g` row was
  chosen FOR the split form — so the partial carries the decided knob row verbatim and the finalize is deliberately
  `_mapped`: both **opt out** of re-recognition, because re-entering would discard the very decision being realized.
- **`032_fuse_finalize`** realizes `PLACE@fin=fuse` — the deferred `g<w>k` finalize inlines into its CONSUMERS'
  read sites instead of launching (the M=1 decode twins' split-chain closer: each finalize is ~2-3 µs of serialized
  launch + sync around a fold that is a handful of f32 adds per cell). Every consumer `Load` of the finalize output
  is replaced by the finalize's own body σ-substituted to that load's index (per-site `__fin<n>` rename; redundant
  per-read recompute, no barrier — each thread folds exactly the cells it reads). Anchored on the consumer; the
  finalize node dissolves via the splice's orphan removal once its last reader is rewired, and the firing that
  removes the last EDGE consumer also inlines the remaining body-level readers in place (a `030` finalize's sweep
  operands are body refs without edges). Gates: not a graph output, every reader inlinable (flat `Map` / plain
  `Reduction` shapes; a `Contraction` reader bails), single-component workspace (flash's `(m,l,O)` stays a kernel).
  Default `cut` (evidence-only: golden rows / `PLACE@fin` pin opt sites in); `030` threads the stamp onto the
  finalize tile via `_fin_knobs`.

Same fragment idiom, inverted re-entry semantics, different decision altitude (placement: "should this be one
kernel at all" vs schedule: "how does this kernel run"). The shared fixpoint is what lets them compose without
knowing about each other: a cut consumer re-enters recognition, may pick a `g<w>k` row, and `030_split_reduce`
then shards it — cut-then-split on what was originally one fused kernel.

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
warp-only rows (`_schedule._computed_a_rows` — the mandatory resolved `sync` compute-fill stage at BOTH depths
(`d1` + the asymmetric B-only prefetch ring `d2` as fork siblings — the M=512 occupancy loss inverts at decode M,
so the depth is measured per shape), crossed with the shared `RASTER` launch-order candidates (its B stripes
re-stream per M-tile row, exactly the grouped order's L2 reuse — `gn8` measured −8% on the gemma gate_up fused
edge, 5090) and — single-channel nodes only — the **redundant-statistic split-K** rows: the contraction K slices
across CTAs while the k-invariant stat prologue stays full-row in every partition (each recomputes it, cheap
exactly on the small-free decode shapes the `_SPLITK_MAX_CTAS` gate admits), the per-cell cone σ-reindexed to
absolute k and the `Map`-wrapper projection folded into the deferred finalize (`_splitk_option`'s computed-A arm
→ `030_split_reduce`'s structural path). Multi-channel (gate/up) nodes split too: the synthesized fold loop
carries the true N-component identity-family carrier (one additive state per channel), the partial stores each
channel's raw C fragment to its `ws[comp, ksplit, *cell]` slice (the per-acc `RegStore` arm — no ⊗-combine in
the partial), and the deferred finalize folds every component before applying the combine projection once.
Still no scalar / gmem-direct / WSPEC rows; the compute-producer role for the fused edge is the anticipated
`RoleKind` extension. The **flash-form fork**: a `TWISTED` streaming contraction pair (the flash tree) offers its
structurally-different schedules as ONE prior-ranked fork — the warp (fragment-resident) rows over
`twisted_warp_moves()`'s `(warps-per-CTA × key-atoms-per-block × query-tiles-per-warp)` geometry grid (option-0 = the
conservative one-warp / `2·atom_n` block / one tile; the third dimension is the `TILE` codec's `f<FM>x<FN>` reg_m —
each warp streams `fm` independent `(m, l, O)` chains against shared K/V fragments, FA-2's in-flight ILP; the Q@K /
P@V mma `TilePlan`s are derived per point, `_schedule._twisted_warp_options`), the scalar
register-vector CHAIN (the FA-2 shared-score form), then the cooperative / per-cell reduce-partition escapes — every
leaf row spelling the same `TILE@<qk_k>` / `TILE@<pv_k>` / `REDUCE@<kv>` key set (decided-empty where a form doesn't
tile). A cross-CTA `REDUCE=g<n>k` pin selects the **flash split-KV** warp rows instead (pin-driven): the plan stamps
onto each row's `Reduction` node and `030_split_reduce` realizes it as a fragment-resident partial (the kv stream windowed to
the CTA's slice, its absolute base on `Reduction.offset`; raw `(m, l, O)` state to an f32 `__partial` workspace) plus
an LSE-combine finalize — kernel finalize only (the twisted `e^{Δm}` rescale can't be an atomic). A static kv must be
block-divisible; a **symbolic kv splits too**: the slice width is the bn-aligned runtime `ceil(S/(cta·bn))·bn` (a
composite `Dim`) and each slice stops/masks at its absolute end `min((s+1)·B, S)` (`Reduction.bound` — a mid-tensor
slice end reads VALID next-slice keys the extent-only tail masks would keep), an empty last slice contributing the
exact carrier identities; the split partial guards every state write with the symbolic-M `m_guard` (the tail CTA's
clamp-read overhanging query rows would otherwise write into the next head's workspace rows). It pays where the
un-split grid starves the SMs (few heads / short query axis: the 2-head hd256 seq-512 shape runs 33.6 → 11.3 µs
under `g8k`, parity with torch SDPA's internally-split flash; the symbolic hd512 dynM stream 135.2 → 116.7 µs
under `g2k` on the 5090). Any
other non-empty `REDUCE` pin remains the scalar escape; a **warp** `TILE` pin keeps the mma rows alone (loud on a
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
chain / coop / serial escapes stamp the decided-empty `STAGE@<kv>: ""`. Staging additionally requires the K/V (and,
for `alt`'s staged Q, the A) BUFFER dtypes to match the atom's operand dtypes — the slab fills byte-copy and cannot
convert, so a wide traced intermediate feeding the stream would deposit garbage; gmem-direct fragment loads convert
per element and keep the warp tier either way. To keep that gate from silently disabling staging on real models,
traced dtype CASTS are first-class: a dtype-changing view splits into a source-shaped elementwise `copy` + a pure
map at the frontend (`optimization/005_split_cast_from_indexmap`), and loop fusion's plumbing exemption admits only
dtype-PRESERVING copies (`merge_loop_ops._is_castfree_indexmap`), so the cast stays a materialized buffer at flash
offer sites and the stream sees an atom-dtype operand it can stage (the gemma V-norm's f32 `mul` → f16 SDPA edge, the
layer-0 findings' biggest lockout). That the cast is *usually free* — a fan-out-1 pointwise producer absorbing it and
simply writing the narrow dtype — is not something loop fusion can be relied on to arrange: fusion may merge either
way, and on gemma-4 it consistently spliced the cheap cast into its CONSUMERS instead, leaving the wide producer
buffer alive. `optimization/007_sink_narrowing_cast` makes it deterministic, retyping the producer's OUTPUT and
dropping the copy whenever the producer's SOLE consumer is the cast. It is a retype, not a numeric change (an
elementwise op computes in its inputs' promoted precision and rounds on store), and it is what keeps a norm→matmul
edge on the plain mma tier: a mixed-dtype A has no copy transport, so without it `_demote_mixed_a` diverts the
projection onto the `sync` compute-fill, which has no weight-prefetch ring — measured on gemma-4's gate/up as
1.12 TB/s against the 1.61 TB/s a clean-f16-A `d2/tma/ring` sibling reached on the same 118 MB
weight. **A causal stream tile-skips**: when the score
prologue carries the triangular `Select` (`kv ≤ m` — detected structurally off the predicate, never a kernel identity),
the realizer bounds the stream at the CTA's last query row (`kv_end = min(seq, (grid_m + 1) · um·fm·atom_m)`, hoisted
into the `StridedLoop`'s for-init `end` override; the staged prefetch clamp re-pins onto the last needed chunk). The
bound is CTA-uniform (barriers stay legal) and every skipped step is the carrier's exact identity (`α = 1`,
`P = expf(−1e30 − m_i) = 0`), so the early stop is bit-identical — it halves the streamed keys/mma work on average,
paying wall-clock wherever the grid oversubscribes the SMs (1.67× on hd256 seq-2048) and re-opening the small-CTA flash
forms that previously paid double K/V re-streaming. **A banded stream additionally starts late**: a trace-time
`SdpaOp.sliding_window` stamp (the HF wrapper knows `config.sliding_window` + `layer_types`; the trace itself erases
the window) decomposes to a second coordinate `Select` (`kv > m − W`) beside the causal one — unless the band is
STATICALLY VACUOUS (static seq, `W ≥ S`), which the decomposition drops instead of emitting: a vacuous Select's
predicate constant-folds, the +0 mask term hoists out of the reduce loops, and the mask-chain walk below can no
longer resolve it — the fuse then silently degraded to cut and the softmax·P@V fell to the unstructured sequential
lowering (the gemma-4 layer-0 `seq 512 < window 1024` trace deployed a grid-1 kernel). Flash classification
reads the whole mask CHAIN off the rowmax feed (coord Selects and the explicit additive bias compose; the bias stays
loaded, it may mask more, e.g. padding), re-synthesizes each canonically, and the realizer derives the stream START
off the band predicate exactly as it derives the causal end (`kv_start = ⌊max(0, first_row − W + 1)/bn⌋·bn`, the
kloops' `k_first`). Fusion keeps every mask add ON the softmax consumer (mask epilogues are exempt from the
score-producer deferral, the QK contraction is barred from chasing them, and `_reduce_heavy` discounts mask adds in
rowmax-bearing bodies so a multi-mask softmax still assembles onto its P@V offer site); a mask that lands on the
score producer anyway declines the fuse rather than being silently dropped. At seq ≫ W the sliding layers' stream is
O(seq·W), not O(seq²) — 40 of gemma-4's 48 layers at real context lengths.
Two catalog invariants hold: every recorded golden's `TILE`/`STAGE`/`REDUCE` stays a **member** of the enumerated
grids (the permanence test in `tests/compiler/test_golden_configs.py` — a space edit can never silently orphan a
golden into unreachability again, the sixth sweep's `.s512` regression class; the scalar reg grid carries the
golden-informed deep-FM points `f2x6..f2x14`, `f4x6..f4x26` for exactly this reason), and a cross-CTA split deploy
(`030_split_reduce`) stamps the decided knob row onto its **partial** kernel — the engine merges knobs forward on 1:1
rebinds only, so without the explicit stamp the graph splice dropped them and the deployed split recorded no
schedule identity (the A/B table then couldn't say what greedy deployed).
