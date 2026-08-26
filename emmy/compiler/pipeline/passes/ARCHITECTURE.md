# Pass-authoring invariants

Rules that apply to EVERY pass in this tree (`frontend/`, `loop/`, `lowering/`). Per-dialect details live in
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) (pass order, knob table, fork semantics). The **tile-lowering** phase
(`lowering/tile/`) is the canonical instance of the invariant below — a **purely algebraic moveset, no
specializations**: it dispatches on stored params of the fold (`axis is None` / `is_contraction` for the schedule
walk; the derived `Fold.role` for the loop annotation the materializer reads), never on a named shape
(matmul / pointwise / attention) —
SDPA is plain contractions plus the online-softmax `TWISTED` fold (a twisted monoid is a monoid), selected
structurally, not a distinct kind.

## Performance is never a pass decision

**Passes enumerate under legality. Ranking belongs to evidence and to the fitted prior — and to nothing else.**
Every semantically legal alternative is exposed as a fork option, an enumerated row, or a knob value, and the
choice among them is made in exactly two places, neither of which is a pass:

- a **deployed model** answers every choice from measured evidence — the box-local reservoir / tune-DB rows a
  local `emmy tune` produced, or a recorded golden row replayed exactly through its pins;
- everything else answers through the deploy evidence hierarchy, whose last learned tier is the fitted prior — on
  a fresh machine, the offline model. Unmeasured-deploy quality is the prior's responsibility, and the way to
  improve it is to fit it better or to measure the shape.

Legality is the ONLY thing a pass may narrow by. A candidate is dropped because this term cannot realize it — the
dtype the atom binds, a K-step that must divide a static extent, an smem budget, an epilogue the emitter cannot
express — and for no other reason. Everything a hardware profitability argument would say belongs on the evidence
side of that line.

Concretely:

- An enumeration never drops, caps, or truncates legal rows because some were measured slow somewhere. A ladder in
  `search/space.py` is a domain, not a preference history; a row set is a function of the term and its legality
  alone, never of a hardware profitability fact.
- **No hand-written ordering, default, or filter may exist to improve an unmeasured pick.** Enumeration produces a
  SET, and the position a row lands in is an implementation detail of whatever loop built it — never a policy,
  never an assertion, never a thing to preserve across a refactor. There is no "conservative option-0", no family
  whose leading value is its safe default, and no leaf withheld because nothing could price it.
- **A bad unmeasured pick is an accepted outcome.** With no golden, no measurement and no useful prior, a compile
  takes whatever the walk emitted first; that kernel may be far off the best one the space contains. It is not a
  regression and not a reason to reintroduce a rule. The path back to a good kernel is the pinned one: a golden
  replays exactly, and a `tune` turns it into evidence the hierarchy can use.
- A pass refusal states a semantic reason (correctness, SSA/region ownership, a resource impossibility) or a
  boundedness reason. Schedule spaces are recursive and lazy, so a large legal Cartesian product is not itself a
  refusal reason. "Measured slower",
  "occupancy", "register pressure", and "profitability" are never refusal reasons; they are the tuner's and the
  prior's vocabulary, and a fix for a slow configuration is new evidence — a re-tune and a refreshed golden file —
  never a new compile-time condition.
- A rewrite motivated by performance is a fork with the un-rewritten form as a sibling, so evidence can decide.
  Hiding a sibling is forbidden, and so is arranging the siblings so one of them wins by default.
- A misdeploy is debugged as an evidence problem: a missing or stale golden row, an unfitted prior, or a gap in the
  feature/key vocabulary that makes two different candidates indistinguishable. Fix the evidence path; do not fence
  the compiler.

This boundary is a review judgment, not a scripted check: when a change wants to refuse, cap, reorder, or default
for speed, move the decision to a fork plus evidence instead. The same rule governs the PROSE — a rationale for a
deleted gate is a template for its return, so it is deleted with the code rather than left behind as history.

## Quantization is not a concept past the decomposition band

A quantized checkpoint is spelled as generic in-graph algebra at BIRTH (`loader.quant`, immediately post-trace).
The scheme may choose different generic algebra, but it may not mint a scheme-specific op. The generic
`frontend/decomposition/032_fold_constant_subgraphs` rule collapses static computation cones into a bind-time
`ConstantOp` (`source_graph`) before Loop IR. It deliberately leaves storage-decode cones expanded so compressed
device storage is preserved, and leaves layout-only cones to the target-specific `050`/`060` load-layout policy.

The boundary is structural, not a naming guideline. Lowering, shared statement and tile dialects, backends, and search
may contain canonical dtypes, generic ops, and graph algebra. They may NEVER contain a checkpoint format's custom op,
statement, helper, pass branch, schedule feature, environment gate, comment, or name. A new format belongs in the
loader and its birth-time speller must emit only generic algebra. `tests/architecture/test_layering.py` scans every
post-decomposition Python source file for known format names.

## The tile scheduler: one stored tree

`015_twisted` first applies the general exp-family Fold rewrite described at the boundary below. `020_schedule` then
maps the free axes and exposes a recursive, addressable product over every stored Fold site. It chooses the kernel's
worker inventory, then composes legal `TILE`, `REDUCE`, `STAGE`, and `RASTER` values through generic worker and physical
axis interfaces. Keys use the tree-path codec, and every resolved slice lives beside the immutable Fold tree in
`TileOp.schedule`. Candidate dictionaries are created only at addressed leaves; there is no eager row list or row cap.

The scheduler does not classify, pair, bind, fuse, demote, or otherwise derive an alternate compute tree. If no row
can realize the stored shape, it leaves the tile unmapped for the scalar materialization path. Reintroducing faster
rows is recovery over Fold-tree structure, not another recognition layer. Child sites join when their tile widths and
worker units agree on every shared physical axis. A derived contraction with a unit marker axis presents its enclosing
Fold's sweep as its scheduling reduction domain. Neither rule inspects the operation family.

## No shape-specific pattern matching

A pass must not dispatch on enumerated shapes ("if this is the gated-MLP body do X, if it is the QK^T body do Y").
Each named shape that needs handling is evidence of a missing GENERAL rule; find the per-element formulation that
makes the old and new shapes degenerate cases of one code path. Shape dispatch compounds: every new model
architecture would add a sibling branch to every pass it touches — the combinatorial explosion of compiler
complexity this invariant exists to prevent. It also breeds divergent incidental behavior (per-branch dtype or
layout rules that drift apart) and silently narrows coverage to the shapes someone happened to name.

How to comply:

- **Write the rule per element, not per shape.** Total lift converts every reduction from its own `Accum` statements;
  nested reductions remain nested `Fold` statements regardless of which frontend operation produced them.
- **Gate in the negative.** Enumerating admissible shapes is shape matching by another name. Walk the body and
  report the first thing the transform *fundamentally cannot do*, like `ir/pure/algebra.classify_fragment_epilogue`
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

Multi-source `IndexMapOp` lifting preserves predicate dtype: an explicit source predicate is substituted unchanged,
and an unconditional fallback is a boolean `Literal(True, "bool")`. The rendered CUDA condition remains `1`, while
Loop IR persistence and vectorized reference evaluation retain the boolean type required by `Select`.

Loop fusion is maximal and schedule-blind: every structurally legal merge is taken to fixpoint before lowering
considers a kernel boundary. Fusion never asks whether the merged body is recognized, schedulable by an optimized
tier, or faster than its parts. Nested reductions and multi-statistic compounds are therefore not fusion gates. A
downstream failure to recognize or schedule a maximally fused body is a lowering coverage gap; it must not be
repaired by retaining an early graph boundary.

Only semantic splice boundaries remain: internal nodes must be owned, every escape must be an explicit live output,
and the splicer must preserve semantics. Fusion does not estimate arithmetic work and has no lowering-dependent
exception.

There is one fusion pass and one fixpoint. One rewrite takes the maximal downstream Loop region: non-reconvergent
consumers become output ports of one multi-output `LoopOp`, and all terminal Writes seed one splicer worklist. The
worklist's shared binding table emits an equal upstream demand once across every port, so fusion order cannot duplicate
a shared producer or change the recognized computation. Merge order may temporarily place a contraction inside
another reduction; the later legal merge is still taken. That maximal result is final: no later placement rule cuts
it apart.

Shape-only graph outputs participate through output equivalence clusters, not a separate fusion rule. A cluster is a
single-owner chain of same-dtype copies with one terminal live output and an exact proof that source and destination
coordinates are related by reshape and axis permutation. The proof matches source coordinates to mixed-radix digits
of the destination's dense flat address; equal element count without that bijection does not qualify. The splicer
composes each inverse layout into the computed source's `Write`, preserving the producer's loop structure through a
terminal flatten or transpose instead of reconstructing the reduction at div/mod-indexed copy loads. The proof cost
scales with tensor rank and expression size rather than output size.

**Split axes re-fuse after fusion is quiescent (`loop/canonicalize`).** A reshape fused into a contraction
splits one of the kernel's output axes into a nest of two (an attention projection's
`view(batch, seq, heads, head_dim)` carves N into heads × head_dim), leaving the operand loads addressing the
original axis through a composite index (`wt[k, h*D + d]`). Downstream that split is an eligibility lockout,
not a slowdown: contraction binding reads the trailing free-axis pair as `(m, n)`, so the split kernel binds
the wrong row, the weight load carries a third grid axis, and the warp/mma tiers are never enumerated — no
search budget can reach a schedule family that does not exist. The canonicalization re-fuses a free pair via
the bijective reindexing `p → f/Q, q → f%Q` (semantics-preserving unconditionally) and keeps the result only
when every access folds clean: operand composites collapse to the bare fused axis (the `(f/Q)·Q + f%Q → f`
recomposition fold in `Expr.simplify`), and a store spelling the pair as separate buffer dims keeps the honest
split-store spelling — `[…, f/Q, f%Q]` when the buffer's row-major flatten folds it back to an affine address,
or the permuted `[…, f/Q, …, f%Q]` of a transposed output. The pair need not be adjacent: free loops are
parallel, so the perfectly-nested free loops between them (the `transpose(1, 2)` every attention projection
fuses after its view puts `seq` between `heads` and `head_dim`) interchange outward and the fused axis takes
the inner loop's place. Any surviving residue elsewhere (an axis addressed alone, a predicate over the pair)
declines the pair and the nest stands. A store that reverses the quotient/remainder order also stays split because
output-storage order is canonical. Split and unsplit spellings of one contraction thereby converge to ONE canonical
nest — one kernel identity, one shape key, one golden family.

It runs as its own pass between `loop/fusion` and `loop/stamp`, not inside `normalize_body` and not as a
fusion rule. `normalize_body` is a pure body→body transform with no buffer shapes (the store-side stride
check needs them) and fires on every Op construction — including scheduled Tile-IR bodies and cross-CTA split pieces
minted at splice time, where re-fusing axes would fight the scheduler. Canonicalizing a producer that still awaits a
merge could re-spell the very indices the splicer composes through, so it waits for fusion's fixpoint; running before
`loop/stamp` means kernel identity and everything downstream see only the canonical spelling.

The consumers that had assumed "one output axis per buffer dim" were generalized with it, all on one
reading — an axis's unit step moves its INNERMOST carrying dim (the `%` dim of a split pair): the lift's
output-ordering positions an axis at that dim (under the permuted store the quotient dim sits outside another
axis entirely, and positioning there would make the fused axis the row and the stride-`Q` axis the column);
the mma `RegStore`'s auto row stride derives from the store template's innermost M-carrying dim (`row_dim`)
instead of assuming the inner extent; an epilogue load's per-dim role (`_warp_roles`) moves only that dim;
and `080_vectorize_stores` re-reads a run its per-dim matching declines by the row-major flat address when a
div/mod residue is present, so a row-major split store keeps its vectorized transactions (the permuted one
stores scalar on the scalar tiers — exact, unvectorized). The warp tier's fragment store evaluates the cell
base once per atom and adds `col` / `row · ldm` across it, so a split pair is mma-addressable only when the
row-major flatten recomposes it, or when the `%` dim is the innermost carrier (contiguous for `n`) with `Q` a
multiple of the atom extent — an aligned atom never straddles a `Q` boundary. That is the `warp_split_store`
legality predicate: dropped by the unpinned catalog, raised on a pin, like every other tile gate; the scalar
tiers, which evaluate every element's index, are always exact.

## Total lift at the Loop IR → Tile IR boundary

Loop fusion reaches its maximal fixed point before Tile IR. It consults neither recognizers nor schedule support and
has no placement-workspace fence.

The Tile IR boundary is one structural operation:

1. peel the outer parallel loop chain into the unmapped placement;
2. recursively replace every remaining reduction `Loop` with a `Fold`, in the same statement position;
3. move the root `Write` or output sweep to `TileOp.stores`;
4. reject any raw inner loop that remains;
5. rely on each `Lambda.__post_init__` to canonicalize its local pure body;
6. let `TileOp.__post_init__` factor maximal pure product-operand cones into canonical contractions, orient each
   contraction's shared argument, merge overlapping cones into multi-result edges, and apply the closed-child rules
   over the complete tree.

`_fromloop.fold_from_loop` reads each componentwise monoid directly from the loop's `Accum` statements. It does not
classify a shape, extract a contraction, pair softmax statistics, hoist a nested reduction, or validate a reconstructed
loop. Nested reductions are ordinary `Fold` statements in the parent lambda, so source order and SSA scope survive
without a placement or value-cut analysis.

`015_twisted` is a separate algebraic rewrite over the canonical tree. It clusters equivalent score lambdas and joins
a maximum with additive exp-weighted components into the one `(maximum, denominator, expectations…)` twisted monoid.
It reads both equivalent canonical spellings: sibling planar folds, and the contraction composition produced when
canonicalization factors a normalized exponential into a computed operand. Softmax, SDPA, and causal SDPA differ only
in carrier arity and score/value lambdas; there is no operation-family matcher. `020_schedule` enumerates the complete
rewritten tree. Direct contraction children and independent roots use the same physical-axis compatibility join, even
when roots reverse their algebraic M/N readings. A derived contraction uses the enclosing Fold domain through the same
parent/child interface. Materialization binds selected sites from their schedule slices; unsupported forms remain
unmapped.

## The divide rule: `split` an iteration axis

`lowering/tile` carries one one-kernel→graph-fragment rule:

- **`030_split_reduce`** splits the **reduce axis** (the REDUCE codec's `g<w>` cross-CTA shard): the SAME
  computation, its K partitioned across CTAs into a partial + finalize (or, on the atomic arm, one kernel that
  accumulates in place). Direct atomic finalization is legal only when it does not write each partial into f16/bf16
  output storage; low-precision output takes the deferred f32 workspace and rounds once after the combine. It runs
  AFTER its decision — the `g` row was chosen FOR the split form.

**Every split piece is a new kernel.** The rewrite consumes the scheduled kernel and returns fresh unmapped Tile IR
for the partial and, when required, the finalize. The partial keeps the same `Fold(init, combine)` over an axis
slice; the finalize identity-lifts stored state tuples through that same monoid. This is one carrier-independent
path for additive and exp-family folds. Each piece receives a fresh structural identity and chooses its own schedule.

A selected cross-CTA split is recorded structurally by an axis `Window`. The scheduler refuses to repartition an
axis that is already a slice, including partition axes nested inside the complete Fold tree. The one-kernel atomic arm
also splices a graph so it restarts the ordinary pass scan with the consumed schedule removed.

Tile lowering creates no other kernel boundaries. In particular there is no placement cut, routing fork, or
`__cut_` workspace path after maximal Loop IR fusion.

The atom spec is subtyped by kind (`ir/atom.py`: `AtomKind` is the fixed mma cell selected by name; `ScalarAtom`
is the plain scalar fma cell). The contraction binder (`bind_bilinear`) reads any lifted fold, so a nested
contraction reached through a composed edge binds through the same path (`_bound_producer`) — a tier is a node
gaining a `TilePlan`, never a new path.

An atom's logical cell and PTX instruction shape are separate. The Volta `mma_m8n8k4_f16_f32` atom is one logical
16×16×4 warp cell because one instruction performs four independent 8×8×4 operations; its fragment layout maps those
groups onto four output quadrants and carries 2/2/8 A/B/C registers per lane. SM70 has no `ldmatrix`, so the same
cooperative m8n8k4 lane map gathers fragments from shared memory, and every staged operand — a copied one and the
materialized peers of a compute fill alike — moves through the blocking vector copy (`sync_copy_staging`: ordinary
vector global loads and shared stores fill the existing slab ring). The atom takes a COMPUTED edge like any other:
the compute fill writes the plain row-major slab the cooperative gather already reads, so a fused producer cone —
and a materialized `a` whose dtype the atom cannot bind, through the converting fill — reaches the tier here exactly
as it does on the newer families. The generic staged-loop scheduler still owns `d<n>` slot rotation and `/p<n>`
register-fragment pipelining; blocking copies make deeper shared rings correct but do not promise copy/compute
overlap. C-to-A repacking still declines this atom. Target capability predicates select this family below SM80 and
the established `m16n8k16` families on SM80 and newer; an incompatible atom or copy-transport pin fails instead of
lowering through instructions the target cannot execute.

**The f16-accumulate atom sibling** (`mma_m16n8k16_f16_f16`, C→f16 — atom names follow
`mma_<shape>_<ab_dtype>_<acc_dtype>`, the compressed PTX/CUTLASS D.A.B.C order; the historical acc-unspecified
spellings stay as parse aliases for the f32-accumulate atoms): on the consumer GeForce dies (sm_86/89/120)
f32-accumulate HMMA runs at HALF the f16-accumulate rate, so this atom keeps the whole mma chain on the full-rate f16
accumulator and the lowering promote-folds the packed f16 partials into f32 shadow fragments per K chunk
(`FragmentPromote` — the staged bk slab is the cadence; gmem-direct promotes every `_atom._F16ACC_STEPS` steps plus a
final fold). Precision-gated enumeration, off by default —
the precise `EMMY_F16_MMA_F32_ACC` pin is authoritative on any target, else the `EMMY_FAST_MATH` umbrella offers it on
the consumer-die ccs only (`_F16ACC_CCS`); a `TILE` pin naming the atom bypasses the gate — pins are authoritative.
The realized fork is identified by the `TILE`
codec's atom token and priced by the `MMA_acc_bits` feature; f16 only (mma.sync has no bf16-accumulate form).

**The move catalog** (`search/space.py`) is the permitted-move enumeration the schedule emit forks over, keyed on
`AxisRole`: `scalar_tile_moves()` is the legality-guarded scalar register-tile product (`par × reg`, `block_threads ≤
1024`) with the per-cell `""` tile as one more member, crossed with the warp / reduce / stage move families
for an unpinned contraction so `compile` / `tune` explores the space (each row → a structural
contraction-fold leaf keyed `TILE@<axis>` in a hierarchical `build_fork_tree`; an env pin wins via `Knob.narrow`).
The producer band is the fourth level (`""` = uniform SIMT — since step 7 a resolved band
is spelled in `WORK`'s `+p<n>` suffix, never a per-row `WSPEC` key) — offered only on a warp row over a
resolved **TMA** stage without a cross-CTA split, and resolved/thread-budget-gated at materialization
(an ineligible spec degrades to uniform). A single-channel computed-A (fused-cone) contraction enumerates scalar
register-tile rows with staging off: the scalar atom evaluates the cone once per operand row or column and reuses it
across the sibling register cells. It also enumerates its warp rows with the mandatory resolved `sync` compute-fill
stage at BOTH depths
(`d1` + the asymmetric B-only prefetch ring `d2` as fork siblings — the M=512 occupancy loss inverts at decode M,
so the depth is measured per shape), crossed with the shared `RASTER` launch-order candidates (its B stripes
re-stream per M-tile row, exactly the grouped order's L2 reuse — `gn8` measured −8% on the gemma gate_up fused
edge, 5090) and — single-channel nodes only — the **redundant-statistic split-K** rows: the contraction K slices
across CTAs while the k-invariant stat prologue stays full-row in every partition (each recomputes it, which is
cheap on the small-free decode shapes and is left to evidence to price elsewhere), the per-cell cone σ-reindexed to
absolute k and the wrapping zero-axis fold's projection folded into the deferred finalize (the split-K option's
computed-A arm
→ `030_split_reduce`'s structural path). Multi-channel (gate/up) nodes split too: the synthesized fold loop
carries the true N-component identity-family carrier (one additive state per channel), the partial stores each
channel's raw C fragment to its `ws[comp, ksplit, *cell]` slice (the per-acc `RegStore` arm — no ⊗-combine in
the partial), and the deferred finalize folds every component before applying the combine projection once.
Multi-channel products still have no scalar / gmem-direct / WSPEC rows; the compute-producer role for the fused edge
is the anticipated
`RoleKind` extension. `TILE` pins narrow by MATCHING each site's own catalog, codec-canonicalized so `a:scalar` ≡
`""` and `f64x1` ≡ `f64`: an explicit `TILE@<axis>` pin names one site and is authoritative there, while a BARE pin
fans out to every eligible site and cannot say which it meant — so it narrows where it matches and leaves a site it
names nothing at alone (`Knob.narrow`'s no-match-keeps-full-list). A bare pin that named nothing at a site decides
nothing there, precision gates included. Staging additionally
requires the staged BUFFER dtypes to match the atom's operand dtypes — a slab fill byte-copies and cannot
convert; gmem-direct fragment loads convert
per element and keep the warp tier either way. To keep that gate from silently disabling staging on real models,
traced dtype CASTS are first-class: a dtype-changing view splits into a source-shaped elementwise `copy` + a pure
map at the frontend (`optimization/005_split_cast_from_indexmap`). Which
side of a cast a fused region lands on is fusion's ordinary outcome; a mixed-dtype A that misses a wanted tier is a
schedule-domain coverage question, answered by extending the domain and re-tuning, never by a pass pre-deciding the
fusion direction.
Two catalog invariants hold: every recorded golden's `WORK`/`TILE`/`STAGE`/`REDUCE` stays a **member** of the
enumerated grids (the permanence test in `tests/compiler/test_golden_configs.py`, site-aware since the step-7
re-spell — membership means the replayed pin resolves to a slice the catalog hands out; a space edit can never
silently orphan a golden into unreachability again, the sixth sweep's `.s512` regression class; the scalar reg grid
carries the golden-informed deep-FM points `f2x6..f2x14`, `f4x6..f4x26` for exactly this reason), and a cross-CTA
split deploy records a schedule identity per **piece**: each is a new kernel that reached its own fork, so each
carries the row it chose. (The engine merges knobs forward on 1:1 rebinds only. The split's earlier fix was to
stamp the pre-split row onto the partial by hand, which made the A/B table readable at the cost of attributing one
kernel's decision to another; minting the pieces unmapped removes both the stamp and the misattribution.)
