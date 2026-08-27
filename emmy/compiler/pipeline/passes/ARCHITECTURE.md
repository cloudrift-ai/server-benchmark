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

`020_twisted` first applies the general exp-family Fold rewrite described at the boundary below. `030_cut` then
offers the maximal fused tree beside every semantically closed stored Fold-edge cut whose workspace dtypes are
determined (an undeterminable seam is not offered — the offer and the realization must agree), and
`035_split_reduce` offers the unsplit tree beside every cross-CTA reduce split the head fold admits — both
STRUCTURAL forks whose chosen side replaces the kernel with fresh unmapped pieces. A bare `PLACE=cut` pin names the
placement decision, not a site, so it resolves among the CUTTABLE seams (the root-most one) rather than through the
codec's primary rule over every PLACE site (which can land on an edge no cut realizes — an unclosed cone, a seam
whose workspace dtypes stay undetermined).
`040_schedule` maps the free axes and enumerates the schedule. Keys
use the tree-path codec, and every resolved slice lives beside the immutable Fold tree in `TileOp.schedule`.

**The cross-CTA split is a kernel-set decision, not a schedule row.** A split kernel does not run — its cost is the
Σ over the partial and finalize it produces — so `035_split_reduce` stands beside the cut, BEFORE scheduling: the
rewrite consumes only the stored Fold algebra (a contraction slices through σ-reindexed operand edges, its cone's
row-invariant statistic staying full-row in every partition; any other fold slices through the generic
`Fold.rewrite`), and each piece re-enters the scan as a fresh kernel that decides its own row. The chain form keeps
its head fold as a BODY member of the projection wrapper, so the realization slices it in place: the sliced fold
carries the prologue cone it still captures (a per-cell scalar scale) into the partial, and the finalize's epilogue
drops the fold whose states now arrive from the workspace. The split is CONSUMED
by the kernel that realizes it — the sliced axis's partition `Window` is the receipt, kernel-scoped — so the pieces
skip the fork, and the walk's pin path strips a `REDUCE` pin's `g<n>[a|k]` half on a kernel that carries the
receipt (`g2k/coop` on a piece is `coop`); a realized split's independent projection SIBLING has no sliced axis, so
it carries the consumed-split receipt as the `split_consumed` flag instead — one pinned split is one split, however
the regions partition. The atomic arm's refusals (one additive state, a distributive
projection, an output that rounds once) sit beside the offer in `tile/_split.py`; the walk's own catalog carries no
`GRID` stage at all.

**A one-option fork is still a fork.** Every pass returns its decision as a fork even when nothing competes — the
lone unsplit arm, a pinned `PLACE` fuse, a fully forced schedule walk — and the engine records a one-option fork as
a decision, so a fully pinned kernel's row is keyed into the trace and the evidence exactly like a contested one.

**The enumeration is a recursive walk of the stored Fold tree.** A Fold offers its own options; each option extends a
context of what the kernel has already agreed; the subtree below is walked under that extended context, and siblings
thread left to right, so a choice anywhere restricts everything enumerated after it:

    S(node, ctx) = for each option o of node under ctx:  o x S(children(node), ctx + o)

There is no product over a flat site list and no join afterwards. The reasons two sites are not one kernel — one
worker inventory, agreeing tile geometry on a shared physical axis, one decision per Fold however many paths reach it,
and a compatible fragment seam across a producer/consumer edge — are stated once, in `Ctx.extend`, and applied while
descending, so an illegal combination is never built. That
matters because the join is where nearly all the pruning happens: on flash attention the unconstrained product is
8.9e6 against 13,280 legal rows, and on an EXL3 coded linear 5.3e12 against 19,407,312.

**Legality is not a separate layer.** A candidate a node cannot realize is one its option list does not contain.
Constraints that are a function of the MOVE live in the catalogs that generate it (the scalar tile space is generated
under the CTA thread budget, so no member can exceed it; the stage catalog filters its copy transports through
`Stage.available_on`, so a target without TMA never sees a TMA move — the atom registry's own shape); constraints that
are a function of the NODE live beside the moves they filter — the warp tier's eligible atoms are read once per node
from its algebra, its operand dtypes and the gmem addressing its fragment loaders and fragment store must reach, and
the fragment seam's refusals (the paired register bound included) sit beside the option builder in `_schedule`.
`tile/_staging.py` is not a legality layer: it holds the three stage RESOLVERS (whose legal answer is a size) plus the
compute fill's own node refusals, which live there because the fill is the move they filter. Nothing may narrow for
SPEED — a slow candidate is a fork the evidence decides, never a row withheld. One acknowledged exception, labeled as
a bound and not a legality: the reduce catalog drops a band wider than the axis has work for (idle lanes realize
fine — a pin still gets such a band; the drop only keeps a short axis from enumerating the whole band catalog to no
effect).

**A pin is authoritative over the VALUE, not over the catalog.** A site's spelling carries no worker widths — they are
read off `WORK` — so one `TILE` pin names a different plan under each inventory, and it may well name a plan no ladder
generates: fixing widths no catalog predicts is what a pin is for. A pin therefore REPLACES the site's candidates with
what it names at each inventory the site can spell it against (the pinned one, when `WORK` is pinned too). An option
the pin does not name is not offered, and its refusals are TWO-LAYERED: a pin whose named tier the node's algebra and
operand dtypes do not select drops onto the guardrail (a graph-wide pin fans out to siblings it cannot mean; `REDUCE`
has no such layer because it has no choice of tier), while a pin whose tier is selected but whose named plan cannot
realize — an atom these fragments cannot bind, an inventory over the CTA thread budget, a band the fold has no
geometry for — raises the recorded refusal rather than silently emptying the enumeration. `WORK` is different again:
it is kernel-global and cannot be narrowed at any one site, so it lives as a parsed fact beside the walk — an option
claiming a different inventory is refused where it is offered, and a walk that never claimed the pinned inventory is
refused at the leaf, which leaves the term unmapped.

**The walk IS the fork tree.** A branch holds the nodes still to decide, the context they must honour, and the row
prefix decided so far; nothing below it exists until it expands. A level with one option is collapsed, so the fork
tree carries choices only. Traversal order is the fork order — there is no separate level vocabulary to keep in step
with the walk. `WORK` leads because the root owns the free axes it is read off, and it is stamped the moment an
option claims an inventory, which `Ctx.extend` then refuses to change.

**Operand staging is resolved at option construction.** A contraction's option carries its already-SIZED `Stage`: the
resolvers (`tile/_staging.py`) answer with a size, not a yes/no — the resolved `bk_elems` slab chunk and the deepest
ring the per-site smem budget (`ctx.max_dynamic_smem`) affords — so an over-budget row is never offered rather than
failing at materialization, and the row's spelling is the RESOLVED one. Three transport families: the copy transports
(the synchronous copy on atoms that stage that way, cp.async, TMA — gmem-direct `None` is their ever-present sibling),
the fp8 byte slabs (a 1-byte operand staged as raw bytes and converted at the drain — the same `d<n>` fork family, no
new knob), and the smem compute fill, which is MANDATORY for a computed operand, a multi-channel product, or a
materialized A the atom cannot bind (only the fill's typed slab store converts — byte transports move raw bits), so it
has no gmem-direct sibling and a `STAGE` pin can only choose its depth. A NESTED-reduce B edge (the streamed
computed-B decode cone) rides the same mandatory multi-channel fill — the fill evaluates every non-materialized B
channel into its slab, nested reduce included — while a nested A, or a nested B on a single-channel node, keeps the
refusal: no transport realizes a nested scheduling site without a fill mandated to evaluate it. The fp8 (k32) gmem-direct tier rides the same
two-layer policy as the f16-accumulate family: precision-gated for the catalog (`FP8_MMA` / the `FAST_MATH` umbrella),
bindable by a pin regardless; its sm_89 hardware floor lives in the atom registry's target filter, which no pin
overrides.

**The fragment seam is a `Ctx` decision.** A fragment edge joins a consumer contraction to the one contraction
producing its computed fragment operand — nested in its A cone, or a sibling in the enclosing fold's derived step
(flash's PV reading the score). The walk decides the two at different steps, so each endpoint's option stakes a seam
entry the context reconciles whichever side arrives second: an untiled producer composes with anything (it is
evaluated elementwise into the consumer's synchronous slab); a TILED producer produces fragments, so it composes only
with a warp consumer over an smem compute fill whose atom family matches and whose slab chunk the producer's
single-unit N tile fills exactly. The paired producer/consumer register bound is NOT cross-site — the producer's
fragment block is a function of the consumer's own stage — so it filters at option construction. Derived sites (the
synthesized PV) join the one walk in `tile/_tree.py`; a derived unit-marker contraction inherits its enclosing fold's
reduction domain, a prescan fact, never a rewritten tree.

**The producer band is inventory a stage can drive.** `+p<n>` rides `WORK`: an option whose resolved stage is TMA also
offers band variants (the band arms the box-copy mbarrier ring; cp.async's wait-group is issuing-thread-scoped and a
compute fill has no async load half), budgeted against the CTA thread limit. A `+p` pin that no option can drive
composes with nothing and the term stays unmapped — the band is part of the inventory, not a row key.

**The per-cell contraction tier partitions its K like a plain fold.** A contraction is a monoid with a ⊗ lift, so the
untiled tile candidate composes with the same `coop_reduce_moves` catalog the plain folds offer — the cooperative
bands, the ILP register chains, the transposed band — under a static K (the scalar contraction emitters carry no
masked-K band), while a tiled output contracts K serially per register cell and takes only the serial fold. A
cooperative band claims the kernel's inventory as the `t<coop>` thread inventory (`derive_inventory`), which is how
`Ctx.extend` reconciles it with every other site, and a cooperative / ILP `REDUCE` pin reaches only the per-cell
tier — a tiled plan offers nothing under it, the same per-family fan-out that lets a serial pin keep every plan.

**The pointwise register strip is a `TILE` value materialized as a term variant.** The pure pointwise ROOT cell is the
one zero-axis `TILE` site (`path.family_sites`); the `map_tile_moves` ladder offers `f<r>` beside the flat per-cell
tile wherever `r` divides the static inner free extent (a masked overhang is refused because the slid last cell is no
longer a provably aligned affine base, which defeats the load/store vectorizers the strip exists to feed — measured,
see `_strip_refusal`), and a row whose root `TILE` names a width unrolls the cell into `r` grouped loads · computes ·
writes at materialization — a different term, hence a different structural identity and `Op.cache_key`.

**`RASTER` leads the walk as its own fork level.** The CTA launch-order codec is kernel-global with nothing for
`Ctx` to reconcile, so it is decided once per kernel, ahead of the sites: each candidate value is one branch whose
row prefix carries it and whose subtree is the whole site walk, and a kernel offering one value collapses the level
exactly as any other one-option level does — the honest parallel to `WORK` *leading* the walk. Contraction-scoped
and static-grid only — a symbolic (masked-tile) grid renders through the dynamic decode path, which does not carry
the swizzle, so the flat `""` is the one honest value there and a live pin drops with the other choice-layer drops.
The row spells the codec value; the kernel materializer's grid_tile seal applies it where the 2-D `(m, n)` block
grid exists.

Because options are a function of the node and the live pins alone, a node that offers nothing offers it under every
context: one pass over the tree says whether the term has a schedule at all. Past that check every node still has an
option that composes with anything (the per-cell tile, the serial fold), so no branch can expand to nothing and promise
leaves it does not have — a site pin that would empty a selected tier's offer raises there instead. A `WORK` pin can
only be answered once the walk reaches a leaf, and a bare site pin can still be emptied by a sibling site's geometry
(the fragment seam). A term with no schedule leaves the tile unmapped for the scalar materialization path.

A row is the kernel's WHOLE identity, so a family the walk decided nowhere is spelled at its declared OFF rather than
left absent — otherwise two rows of one kernel would carry different family vocabularies and the evidence hierarchy
would not join them. A schedule row also ALWAYS spells the kernel-global `WORK` (the leaf writes it unconditionally,
empty when nothing claimed an inventory), and a structural arm's knob delta — a cut, the cross-CTA split's `g`-half
or its unsplit receipt — never does: that is the one stated marker consumers use to tell a complete schedule row
from a kernel-set decision (`search/golden_eval` filters on it). The same reasoning puts the structural
`S_warp_eligible` stamp on the row prefix: it is read off
the sites' own atoms, not off the rows, so a pin naming the scalar tier cannot erase "tensor cores were on offer here"
from the rows it does enumerate.

**The pool memo and the sampled draw.** The prescan — each node's option list — is memoized in the Context's
session cache, keyed by the scheduler's own `pool_key` (the term and its knobs plus every enumeration input the
term omits: operand/output dtypes, per-axis extents, buffer shapes, stores, symbolic hints) folded with the live
env-pin fingerprint, two facts the walk consumes directly and therefore keys explicitly — the split receipt
(`carries_partition`, which strips a `REDUCE` pin's `g`-half where a receipt-free twin must raise) and the spelled
key vocabulary (the decided-empty OFF map the rows decode under) — so that soundness never rides on how the term
digest happens to serialize the `compare=False` `Axis.window` or the recognition-canonical axis names, which do
cover both today — and, when sampling, the sample's identity; target facts need no key
part because the cache lives ON the Context and one instance never spans two fact sets. What is shared is
immutable and op-independent — frozen options over read-only knob mappings; options are a function of the node
and the live pins — so a hit replays the walk over the memo, and every leaf row is a fresh dict the CURRENT
kernel's own materialization decodes. The memo sits below the search policies — greedy and MCTS hit it alike —
and holds no ranking and consults no evidence; a prescan that raises (a pin naming nothing) is never memoized.
Offline sampling (`emmy fit`, via `ctx.pool_sample`) is the one path that does not return the lazy fork: the
walk's leaf stream is sampled by single-pass reservoir sampling (`search/pool.py` — nothing proportional to the
pool is ever retained, and the exact pool size is known when the stream ends), and the drawn complete rows ride
the memo beside that exact total, keyed apart by the sample's identity, so a sampled Context sharing a session
cache with a live one can never serve it a draw.

**Cost is per kernel; a kernel SET is a sum.** A schedule fork picks one alternative and its cost is that
alternative's latency. A cut's — and a cross-CTA split's — cost is the minimum sum over the kernels it produces,
which is why each is a separate structural decision with a separate scoring rule (`policy/greedy._resolved_price`,
memoized per `Op.cache_key` so a piece appearing in several partitions is solved once) rather than something the
per-row prior can rank.

The scheduler does not classify, pair, bind, fuse, demote, or otherwise derive an alternate compute tree.

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

**Measured evidence this rule is spending.** Dropping the work-growth cap and the merge-ordering pass is a deliberate
trade: both existed because of a measurement, and neither measurement has been retaken. A merge that splices a compute
producer into a still-open contraction product lands that product in gmem; on a 1-layer Qwen3-0.6B trunk at seq 512 the
losing order planned a **6.006 GiB** scratch slab against **0.026 GiB** for the other, and at batch 4 it pushed a buffer
past 2^31 elements. Separately, removing fusion's memory bound outright produced a **194x** scratch blowup and a device
fault. Maximal fusion is claimed to make both unreachable because the scratch is never materialized at a kernel
boundary the placement fork can no longer introduce — that claim is what the scratch-plan assertions must hold, and it
is the first thing to check when a trunk compile reports an implausible workspace.

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
multiple of the atom extent — an aligned atom never straddles a `Q` boundary. That is `split_addressable`, the
per-axis address predicate beside `split_pair` in the shared addressing module, asked over the projection tail by the
scheduler's `_split_store_refusal`: a node whose store fails it binds no warp atom, and a pin naming one raises the
refusal; the scalar tiers, which evaluate every element's index, are always exact.

## Total lift at the Loop IR → Tile IR boundary

Loop fusion reaches its maximal fixed point before Tile IR. It consults neither recognizers nor schedule support and
has no placement-workspace fence.

The Tile IR boundary is one structural operation:

1. peel the outer parallel loop chain into the unmapped placement;
2. recursively replace every remaining reduction `Loop` with a `Fold`, in the same statement position;
3. move every `Write` to `TileOp.output_specs`, representing sibling output loops as pure `ProjectionRegion` terms;
4. reject any raw inner loop that remains;
5. rely on each `Lambda.__post_init__` to canonicalize its local pure body;
6. let `TileOp.__post_init__` factor maximal pure product-operand cones into canonical contractions, orient each
   contraction's shared argument, merge overlapping cones into multi-result edges, and apply the closed-child rules
   over the complete tree.

When every output specification proves that an otherwise planar one-free-axis reduction writes `[0, n]`, post-init
restores the elided extent-one row axis before applying the same contraction rule. This is output-boundary evidence,
not a schedule view or shape matcher; without the common proof the Fold remains planar.

`_fromloop.fold_from_loop` reads each componentwise monoid directly from the loop's `Accum` statements. It does not
classify a shape, extract a contraction, pair softmax statistics, hoist a nested reduction, or validate a reconstructed
loop. Nested reductions are ordinary `Fold` statements in the parent lambda, so source order and SSA scope survive
without a placement or value-cut analysis.

`020_twisted` is a separate algebraic rewrite over the canonical tree. It clusters equivalent score lambdas and joins
a maximum with additive exp-weighted components into the one `(maximum, denominator, expectations…)` twisted monoid.
It reads both equivalent canonical spellings: sibling planar folds, and the contraction composition produced when
canonicalization factors a normalized exponential into a computed operand. Softmax, SDPA, and causal SDPA differ only
in carrier arity and score/value lambdas; there is no operation-family matcher. `040_schedule` enumerates the complete
rewritten tree. Direct contraction children and independent roots use the same physical-axis compatibility join, even
when roots reverse their algebraic M/N readings. A derived contraction uses the enclosing Fold domain through the same
parent/child interface. Materialization binds selected sites from their schedule slices; unsupported forms remain
unmapped.

## Kernel boundaries after maximal fusion

Maximal Loop fusion remains canonical. Tile lowering may expose two kinds of graph-fragment siblings without changing
that canonical input:

- **`030_cut`** offers the maximal fused Fold tree and every closed stored child-Fold seam. A cut writes one workspace
  per state component and replaces all occurrences of the same canonically shared Fold object with workspace loads.
  Closure and replaceability are semantic gates; operation family, expected speed, row order, and search-space size
  are not. A contraction's operand edges are seams of the same class: cutting one materializes the cone feeding that
  operand into its own kernel and the contraction reads it back as an ordinary load. Such a seam's workspace dtype is
  decided EXPLICITLY — the dtype the consuming contraction's output is stored at (traced through any epilogue to the
  output it feeds, so a sibling output at another width cannot mis-type it), which is the element the fused slab
  would have stored — never the carrier the cone computed in: only the `a` edge has a converting fill, so an f32
  workspace on a `b` edge could feed no warp atom. Every seam's per-component dtypes are decided at offer time and
  ride the seam into realization, so the two cannot disagree. The new producer and consumer are fresh unmapped
  TileOps, so further legal cuts and schedules use the same ordinary passes.

- **The cross-CTA reduce split is not currently realized.** Splitting the reduce axis across CTAs into a partial +
  finalize is a *structural* alternative — it changes which kernels exist — but it used to be decided as a `REDUCE`
  spelling inside the schedule and realized by a pass downstream of that decision. Carrying a structural decision at
  a schedule position is what made it need a consumed-partition check, its own pin-refusal plumbing, and a separate
  realizer. When it returns it belongs beside `030_cut`, offered before any schedule knob is spelled — unless the
  partial genuinely needs the pre-split tile, which is the one thing to establish first.

**Every split piece is a new kernel.** The rewrite consumes the scheduled kernel and returns fresh unmapped Tile IR
for the partial and, when required, the finalize. The partial keeps the same `Fold(init, combine)` over an axis
slice; the finalize identity-lifts stored state tuples through that same monoid. This is one carrier-independent
path for additive and exp-family folds. Each piece receives a fresh structural identity and chooses its own schedule.

A selected cross-CTA split is recorded structurally by an axis `Window`. The scheduler refuses to repartition an
axis that is already a slice, including partition axes nested inside the complete Fold tree. The one-kernel atomic arm
also splices a graph so it restarts the ordinary pass scan with the consumed schedule removed.

Both forms keep their uncut or unsplit sibling addressable. The tuner chooses among them; neither greedy policy nor
schedule enumeration may hide a legal kernel set.

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
edge, 5090). The **redundant-statistic split-K** form is no longer a schedule row: the structural
`035_split_reduce` fork slices the contraction across CTAs BEFORE scheduling, σ-reindexing the per-cell cone to
absolute k while the k-invariant stat prologue stays full-row in every partition (each recomputes it, which is
cheap on the small-free decode shapes and is left to evidence to price), and the wrapping zero-axis fold's
projection folds into the deferred finalize. Multi-channel (gate/up) nodes split too: the sliced contraction
carries the true N-component identity-family carrier (one additive state per channel), the partial stores each
channel's raw state to its `ws[comp, ksplit, *cell]` slice — no ⊗-combine in
the partial — and the deferred finalize folds every component before applying the combine projection once.
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
