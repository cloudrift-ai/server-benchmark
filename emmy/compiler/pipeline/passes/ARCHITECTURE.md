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
  `ir/schedule/catalog.py` is a domain, not a preference history; a row set is a function of the term and its legality
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

`020_twisted` first applies the general exp-family Fold rewrite described at the boundary below. The single `030_cut`
pass runs to a fixpoint over two ordered domains. It first offers the maximal fused tree beside every semantically
closed stored Fold-edge cut whose workspace dtypes are determined (an undeterminable seam is not offered — the offer
and realization must agree). Once placement is consumed, it offers the unsplit tree beside every cross-CTA reduce
split the head Fold admits. A selected cut or split replaces the kernel with fresh unmapped pieces. A bare
`PLACE=cut` pin
names the
placement decision, not a site, so it resolves among the CUTTABLE seams (the root-most one) rather than through the
codec's primary rule over every PLACE site (which can land on an edge no cut realizes — an unclosed cone, a seam
whose workspace dtypes stay undetermined).
A contraction-operand seam stands for a VALUE, not only an object: closed cones that are alpha-equivalent up to
their captured axis names (attention's normalized K cone, once per score contraction) fold into one seam, each
duplicate carried as a sibling with its capture correspondence, and the cut replaces every one with workspace loads
spelled through its own axes. Any stored fold capturing enclosing-scope names — operand edge or body member — is
cuttable through PROVIDER CLOSURE. Each occurrence resolves captures outward through its lexical environment,
nearest scope first; every occurrence must resolve to equal straight-line providers and the same Fold producers. A
pure `Load`/`Assign` chain joins the produced piece verbatim, while a Fold producer makes the seam DEPENDENT: its piece
reads the name through that producer's workspace, so cutting it composes the producer's cut in. Dtypes for capturing
seams come from inference rooted at the Tile tree, where the enclosing bindings are visible. This is what lets row
statistics materialize once per query row instead of being copied into a contraction's evaluation domain.
Provider-closed and dependent seams join the UNPINNED fork as principal closures: each seam is offered with its
transitively required producers as ONE composed structural arm, built by the same composition walk the pin path
uses, so the evidence-driven route through a dependent seam is on the ballot (DeepSeek-V4 post4096's only working
placement was a dependent seam's closure — the previous plain-only ballot could never elect it, however the
evidence ranked). Two seams whose closures coincide are one arm. Bare `PLACE=cut` still resolves among the PLAIN
seams only: it names one deterministic pinned decision and is consumed on the fresh pieces. Unpinned recursive
placement over composed arms converges in practice — pieces collapse onto shared identities as the tree shrinks —
rather than being cut off by any count or depth guard.
Scoped `PLACE@path=cut` pins are authoritative and COMPOSE: every pin that resolves on
one kernel joins a single realization — one producer per seam, one consumer, a producer reading another seam's
workspace when its value nests inside (attention's statistics cone contains the score dots whose operand cones are
cut beside it) — and all pieces set `placement_decided` and proceed to scheduling. A bare pinned cut consumes its one
root-most cut the same way and may join scoped cuts in that single decision. A scoped pin whose site path does
not exist on a kernel addresses another kernel of the graph; a kernel none of the pins address fuses, deterministic,
so the unpinned placement fork never returns under a pin-driven compile. A pin that resolves to an edge no cut
realizes is an addressing error. Only unpinned cuts leave the pieces undecided, so search can explore their smaller
seams before scheduling.
`040_schedule` is the classic assignment boundary. The model under `ir/schedule` projects direct, plain-reduction,
scalar-contraction, precision-gated tensor-core, materialized-operand copy, computed-operand and multi-channel smem
compute-fill, and kernel-global raster domains. `ClassicScheduleContext` alone composes their compatibility. The pass
reads pins, mints the search-pool identity, and adapts accepted typed assignments to generic lazy Forks. A cross-CTA
split
piece with several contraction schedule sites uses the same boundary. A split piece's partition receipt consumes the
GRID stage
before `c` is built, so its immutable schedule restriction contains only the remaining `REDUCE` stages; neither the
domain projection nor the Algorithm 1 traversal reads that structural choice. A plain reduction projects serial,
cooperative and ILP choices independently from its node, while the kernel domain projects the union of their worker
inventories; the compatibility relation is the only join between them. A scalar contraction projects its complete
output-tile catalog as one node factor, materializes placed geometry only after selection, and uses physical-axis
claims to make independently projected sites agree. A tensor-core node domain is projected from the contraction's
semiring, typed operands, target atom availability, fragment addressing, and the same output-tile catalog; no selected
edge or kernel choice participates in that projection. The
kernel raster domain is projected separately from static grid facts. The compatibility relation admits a grouped
choice only beside a tiled contraction; symbolic grids expose only the direct choice. Static 2-D grids project direct,
`gm8`, `gn4`, and `gn8`; the schedule restriction excludes the transposed values unless an exact parameter selects one.
The
stage domain is projected once per operand edge from target-filtered transport choices. After `c` has selected one
node and its incident edge values, the context derives their local support without putting slab sizes into either
public factor; compatibility therefore rejects mixed transport assignments, and selected non-direct edges are
resolved again only during materialization. The production traversal follows compatible prefixes. When `c + p + t`
can prove that a prefix has no completion, the context may reject it without constructing later support. Bounded tests
compare the complete set against the literal node × edge × kernel product.
The fixed completion contract is that structural rewrites finish before site construction, every leaf is a complete
typed `Schedule`, only the search boundary encodes exact node and consumer scopes, and only materialization
derives placed geometry and resolved transport facts. Schedule parameters restrict Algorithm 1 without changing any
factor or overriding target, shape, addressing, or compatibility constraints. A partial row containing scoped keys is
inert on a kernel where all of those keys are foreign; its bare kernel values travel with it instead of accidentally
restricting every cut piece.

**The cross-CTA split is a kernel-set decision, not a schedule row.** A split kernel does not run — its cost is the
Σ over the partial and finalize it produces — so the cross-CTA domain is the second slice of `030_cut`, BEFORE
assignment composition. The
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

**Enumeration is Algorithm 1 from the introduction: the restricted compatible subset of independent domains.**
Kernel, node, and edge domains are projected from static problem facts without reading parameters or another selected
choice. For one immutable schedule restriction `c`, unscheduled Fold program `p`, and target `t`, there is exactly one
candidate set and one result:

    D(p, t) = K(p, t) × ∏ N(p, t, node) × ∏ E(p, t, edge)
    Algorithm 1(c, p, t) = {a ∈ D(p, t) | c.accepts(a) ∧ accepts(p, t, a)}

There is no production-specific product and no second notion of membership. The generic visitor carries `c + p + t`
intact and never unpacks its restriction or imports classic scheduling. The context may reject a prefix only when its
combined state proves that no completion can satisfy the same relation; it must still enumerate exactly
Algorithm 1(c, p, t), so traversal order can change only evaluation cost. Every leaf crosses the complete
compatibility relation once at the strict codec boundary, then carries that accepted typed assignment and canonical
row through search and materialization. Downstream reads never repeat the compatibility walk. Bounded spaces are
exhaustively compared with the literal Cartesian reference. That compatibility pruning matters because on
flash attention the unconstrained product is 8.9e6 against 13,280 compatible rows, and on an EXL3 coded linear 5.3e12
against 19,407,312.

The cut phase is the outer enumeration. `030_cut` reaches a fixpoint over fused/cut placement choices and then
unsplit/split reduction choices and emits those pass-native structural forks directly. `040_schedule` follows and
supplies `ClassicScheduleContext` to the generic driver for Algorithm 1(c, p, t). A structural realization creates
ordinary fresh kernels, so any later placement or split decision is discovered by the same pass rather than by a
classic-context refusal.

**Legality is not a separate layer.** A candidate a node cannot realize is one its option list does not contain.
Constraints that are a function of the MOVE live in the catalogs that generate it (the scalar tile space is generated
under the CTA thread budget, so no member can exceed it; the stage catalog filters its copy transports through
`Stage.available_on`, so a target without TMA never sees a TMA move — the atom registry's own shape); constraints that
are a function of the NODE are facts the context reads while composing a pick — the warp tier's eligible atoms are
read once per node from its algebra, operand dtypes, and the gmem addressing its fragment loaders and fragment store
must reach. Fragment-seam and paired-register refusals live only in `ClassicScheduleContext`.
`ir/schedule/staging.py` holds the three stage RESOLVERS (whose legal answer is a size) plus the compute fill's source
facts. The context alone decides whether the resolved fact composes. Nothing may narrow for
SPEED — a slow candidate is a fork the evidence decides, never a row withheld. A cooperative band wider than its axis
remains in the independent domain because idle lanes are legal; a restriction may select it without constructing a
new value.

**A schedule pin is a restriction, never a domain constructor.** `WORK`, `TILE`, `REDUCE`, `STAGE`, and `RASTER` pins
compare their exact canonical values with the applicable factors in Algorithm 1. They do not replace a factor or add a
value the static catalog did not project. Precision gates are restrictions of the same enumeration: their atom choices
remain in the fixed node domain, and the immutable `c` excludes them when it evaluates a complete assignment. A
malformed or unavailable exact value therefore names no member of `D(p, t)` and is refused; pinning cannot manufacture
a worker inventory, tile, transport, or raster value. A bare pin must be supported by at least one factor in a strict
kernel and restricts every factor that supports its value; non-supporting sibling sites are not silently given another
meaning. Under union probing, a kernel that supports the value nowhere ignores the bare pin so a sibling kernel may
carry it. Once the parameter set contains scoped schedule pins, its bare kernel values travel with that partial row:
they restrict a kernel where at least one scoped key resolves and are inert on kernels where every scoped key is
foreign.

**The context IS the fork tree state.** The generic schedule-fork adapter holds the immutable context and row prefix;
nothing below a branch exists until it expands. Classic contributes only row encoding and leaf materialization. Direct
iteration uses the same generic recursive `schedule(context)` traversal. `WORK` is stamped when a site claims an
inventory, which the next context refuses to change; traversal order never changes the Cartesian reference set.

**Operand staging is resolved at option construction.** A contraction's option selects an axis- and size-free `Stage`
choice. The resolvers (`ir/schedule/staging.py`) answer with a separate `ResolvedStage` containing the slab names, resolved
`bk_elems` chunk, and deepest ring the per-site smem budget (`ctx.max_dynamic_smem`) affords. An over-budget row is
never offered rather than failing at materialization, while row identity reads only `ResolvedStage.choice`. Three
transport families: the copy transports
(the synchronous copy on atoms that stage that way, cp.async, TMA — gmem-direct `None` is their ever-present sibling),
the fp8 byte slabs (a 1-byte operand staged as raw bytes and converted at the drain — the same `d<n>` fork family, no
new knob), and the smem compute fill, which is MANDATORY for a computed operand, a multi-channel product, or a
materialized A the atom cannot bind (only the fill's typed slab store converts — byte transports move raw bits), so it
has no gmem-direct sibling and a `STAGE` pin can only choose its depth. This requirement belongs to warp choices; the
scalar contraction tier evaluates every channel serially and keeps direct edges. A NESTED-reduce B edge (the streamed
computed-B decode cone) rides the same mandatory multi-channel fill — the fill evaluates every non-materialized B
channel into its slab, nested reduce included — while a nested A, or a nested B on a single-channel node, keeps the
refusal: no transport realizes a nested scheduling site without a fill mandated to evaluate it. ONE computed operand
has byte-transport siblings beside the fill: a packed-pair (NVFP4) weight cone, whose packed 4-bit values copy
verbatim as a raw byte slab while only its block scales are compute-filled, so `resolve_warp_stage` answers for it
and the cp.async and TMA rows sit beside the fill's depths as fork siblings. Which reading applies is a fact about
the NODE, not about the transport a pin names — a multi-channel product carrying a cp.async or TMA pin still RAISES,
since the single-sided byte-transport emitters carry one channel — and a shape the byte slab declines keeps the
generic reading, which computes the same values through the fill. Where BOTH operands are packed over one block
extent the native fp4 mma cell multiplies those values as stored and applies their raw block scales itself: that
node's atom is read off the pair rather than off the A edge's leaf dtype, and its stored slabs copy verbatim — only
an activation whose values this kernel computes takes a fill underneath them. That reading takes ANY channel arity:
the shared A stages once and each product channel adds its own codes and block-scale slab (`2 + 2N` in all), so a
fused gate⊗up MLP edge is the two-channel case of the same cell rather than a shape it declines. The fp8 (k32)
gmem-direct tier rides the same two-layer policy as the f16-accumulate family: precision-gated for the catalog
(`FP8_MMA` / the `FAST_MATH` umbrella), bindable by a pin regardless; its sm_89 hardware floor lives in the atom
registry's target filter, which no pin overrides.

The fill's `d1/smem` and `d2/smem` choices are projected independently onto every operand edge. Direct transport may
still occur in those public edge factors because a scalar node choice supports it; compatibility alone rejects a
direct or mixed edge assignment beside a warp node choice that needs the fill. Fill sizing uses the contraction's
effective reduction axis. A derived unit-marker contraction therefore inherits its enclosing Fold's K axis and
carried-state seam as immutable node facts; the synthetic unit axis never controls the K chunk or mask. An untiled
choice claims neither physical-axis geometry nor a fill, so it composes with a tiled sibling without inventing an
axis disagreement.

**The fragment seam is a compatibility decision.** A fragment edge joins a consumer contraction to the one contraction
producing its computed fragment operand — nested in its A cone, or a sibling in the enclosing fold's derived step
(flash's PV reading the score). The walk decides the two at different steps, so each endpoint's option stakes a seam
entry the context reconciles whichever side arrives second: an untiled nested producer can be evaluated elementwise
into the consumer's synchronous slab, but an untiled sibling-step result cannot be replayed outside its carrier stream.
A TILED producer produces fragments, so it composes only with a warp consumer over an smem compute fill whose atom
family matches and whose slab chunk the producer's single-unit N tile fills exactly. The paired producer/consumer
register bound is NOT cross-site — the producer's
fragment block is a function of the consumer's own stage — so it filters at option construction. Derived sites (the
synthesized PV) join the one walk in `ir/pure/tree.py`; a derived unit-marker contraction inherits its enclosing fold's
reduction domain, a prescan fact, never a rewritten tree.

**The producer band is inventory a stage can drive.** `+p<n>` rides `WORK`: an option whose resolved stage is TMA also
offers band variants (the band arms the box-copy mbarrier ring; cp.async's wait-group is issuing-thread-scoped and a
compute fill has no async load half), budgeted against the CTA thread limit. A `+p` pin that no option can drive
composes with nothing and the term stays unmapped — the band is part of the inventory, not a row key.

The band splits the CTA into warps that only fetch and warps that only compute, which is why the NVFP4 byte slab
above refuses it even under TMA, and refuses it as a LEGALITY: that slab's own TMA lowering returns the plain staged
K-loop rather than the band-splitting one, so it takes no warp inventory and the two halves are never separated —
while the CTA still widens to hold a band, because the thread budget is set from the inventory alone. The extra warps
then reach the compute body, where the box copy elects its arming thread on a wrapping linear thread id: thread 0 and
the band's first thread both match, so one mbarrier takes two arrivals against an arrival count of one and its phase
parity desynchronizes. That is a hang, not a slow kernel. The native fp4 mma cell needs no rule of its own — its
stage resolver takes cp.async only, so it never reaches a TMA stage for the question to be asked about.

**The per-cell contraction tier partitions its K like a plain fold.** A contraction is a monoid with a ⊗ lift, so the
untiled tile candidate composes with the same `coop_reduce_moves` catalog the plain folds offer — the cooperative
bands, the ILP register chains, the transposed band — under a static K (the scalar contraction emitters carry no
masked-K band), while a tiled output contracts K serially per register cell and takes only the serial fold. A
cooperative band claims the kernel's inventory as the `t<coop>` thread inventory (`derive_inventory`), which is how
the private partial assignment reconciles it with every other site. A cooperative / ILP `REDUCE` pin reaches only
the exact per-cell node site; a tiled plan offers nothing under it, while a serial choice remains valid for every tile.

**The pointwise register strip is a `TILE` value materialized as a term variant.** The pure pointwise ROOT cell is the
one zero-axis `TILE` site (`path.family_sites`); the `map_tile_moves` ladder offers `f<r>` beside the flat per-cell
tile wherever `r` divides the static inner free extent (a masked overhang is refused because the slid last cell is no
longer a provably aligned affine base, which defeats the load/store vectorizers the strip exists to feed — measured,
see `_strip_refusal`), and a row whose root `TILE` names a width unrolls the cell into `r` grouped loads · computes ·
writes at materialization — a different term, hence a different structural identity and the variant key
(`identity_key(with_io=True, with_knobs=True)`).

**`RASTER` leads the walk as its own fork level.** The CTA launch-order codec is kernel-global with nothing for
the partial assignment to reconcile, so it is decided once per kernel, ahead of the sites: each candidate value is one
branch whose row prefix carries it and whose subtree is the whole site walk, and a kernel offering one value collapses
the level
exactly as any other one-option level does — the honest parallel to `WORK` *leading* the walk. Contraction-scoped
and static-grid only — a symbolic (masked-tile) grid renders through the dynamic decode path, which does not carry
the swizzle, so the flat `""` is the one honest value there and a live pin drops with the other choice-layer drops.
The row spells the codec value; the kernel materializer's grid_tile seal applies it where the 2-D `(m, n)` block
grid exists.

Because options are a function of static problem facts alone, a node that projects an empty factor does so under every
restriction and context. Otherwise Algorithm 1 owns emptiness: a schedule parameter may exclude every member, and
compatibility may reject every cross-factor combination. Both outcomes produce no schedule leaf; neither one changes
a factor or falls back to an unrelated row. A term with no schedule remains unmapped for the scalar materialization
path.

A row is the kernel's WHOLE identity, so a family the walk decided nowhere is spelled at its declared OFF rather than
left absent — otherwise two rows of one kernel would carry different family vocabularies and the evidence hierarchy
would not join them. A schedule row also ALWAYS spells the kernel-global `WORK` (the leaf writes it unconditionally,
empty when nothing claimed an inventory), and a structural arm's knob delta — a cut, the cross-CTA split's `g`-half
or its unsplit receipt — never does: that is the one stated marker consumers use to tell a complete schedule row
from a kernel-set decision (`search/golden_eval` filters on it). The same reasoning puts the structural
`S_warp_eligible` stamp on the row prefix: it is read off
the sites' own atoms, not off the rows, so a pin naming the scalar tier cannot erase "tensor cores were on offer here"
from the rows it does enumerate.

**The session kernel cache.** Greedy lowering of one fused kernel is a function — Loop-IR program in,
lowered `KernelOp` out — and `pipeline/kernel_cache.py` memoizes it at its boundary: `lowering/tile/005`
fetches a finished lowering (io rebound through `Stmt.rename_buffers`) before the lift, and
`lowering/cuda/001` harvests every single-kernel lowering just before the per-graph negotiations
(zero-init delegation, rendering) that deliberately sit below the boundary. Caller-owned on
`Context.kernel_cache` (nothing installs it by default), greedy-only (tune strips it, pricing probes
strip it), multi-kernel origins poison their key. A twin program compiles ~750x faster than cold.

**The domains and pool identity.** Each `schedule()` call projects fixed independent domains from `p` and `t`, builds
one immutable `c`, then evaluates Algorithm 1(c, p, t). `Fork.pool_id` stamps the deploy identity, target, ordered
free-axis extents, exact codec vocabulary, schedule-parameter fingerprint, and split receipt; it keys the greedy
decision memo without weakening any enumeration input. Sampled lazy enumeration remains behind the explicit classic
reconstruction boundary.

**Cost is per kernel; a kernel SET is a sum.** A schedule fork picks one alternative and its cost is that
alternative's latency. A cut's — and a cross-CTA split's — cost is the minimum sum over the kernels it produces,
which is why each is a separate structural decision with a separate scoring rule (`policy/greedy._resolved_price`,
memoized per the variant key (`identity_key(with_io=True, with_knobs=True)`) so a piece appearing in several
partitions is solved once) rather than something the per-row prior can rank.

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

Before fusion, `090_spell_store_rounding` turns a public narrowing reduction store into a typed `copy` value. A
decomposition may place one transient, shape-only buffer between the accumulator and its public pass-through store;
that direct private copy inherits the same accumulator dtype. Actual computation over private reduction state remains
untyped, so normalization and softmax keep their f32 state until their own public result store. Fusion and placement
then preserve the typed `copy` as an ordinary statement rather than reconstructing a boundary from graph topology.

Loop fusion is maximal and schedule-blind: every structurally legal merge is taken to fixpoint before lowering
considers a kernel boundary. Fusion never asks whether the merged body is recognized, schedulable by an optimized
tier, or faster than its parts. Nested reductions and multi-statistic compounds are therefore not fusion gates. A
downstream failure to recognize or schedule a maximally fused body is a lowering coverage gap; it must not be
repaired by retaining an early graph boundary.

Only semantic splice boundaries remain: internal nodes must be owned, every escape must be an explicit live output,
the splicer must preserve semantics, and a region stops at a PACKED buffer it computes. Fusion does not estimate
arithmetic work and has no lowering-dependent exception.

Packed here means the storage sense — `logical_elems > 1`, two e2m1 codes to the byte — not a concatenated
projection. Such a dtype states that a tensor's stored last-axis extent is half its logical one, and only a tensor
carries that relation. The splice deletes the tensor. The codes then survive as an `Assign` at the packed dtype, a
value with no extent, and a consumer's index names half a byte rather than one whole element. The merged body answers
that by carrying the graph's own pack arithmetic into the consumer, deriving the whole byte at every logical index and
reading one half of it.

The splice also goes ONE WAY, and that is what makes this a refusal rather than a maximal merge evidence may cut
back. Fusion may go maximal because `030_cut` offers each closed operand seam back. **Kernel boundaries after maximal
fusion** below says what such a seam materializes: the dtype the consuming contraction stores, which is the decoded
element rather than the codes. The storage-frontier refinement described there would hold the raw storage bits
instead, but it recognizes a decode through the `ElementwiseImpl.decodes` trait, and only the fp8 casts carry that
trait — an e2m1 code decodes through a value-table gather. Nothing offers the codes back. The merged form is not the
widest of several siblings; it is the only one left.

Consumer count decides nothing. One consumer or three reach the same shape: the quantize is a kernel of its own and
the codes sit in memory. The buffer's readers leave the region together with everything downstream of them, so the
remainder keeps no holes — a hole would make the merged node depend on a node that depends on it.

Whatever is left feeding only what departed leaves with it. A cut materializes every buffer crossing it, so a survivor
whose entire readership is on the far side buys nothing: its value is stored once and read once, and it is stored at
whatever shape it happens to have. The quantized activation's scale is the case that shows why this matters. Its
per-consumer reconstruction multiplies the block scale by the per-tensor one, rounds the product to f16 and broadcasts
it across the block; left on the producer's side, that broadcast is what the boundary stores — one value per logical
element where its source held one per block. Released, the reconstruction sits beside its matmul and the boundary falls
on the raw block scales instead, which is the narrower buffer and the one a consumer can index block-wise. Two nodes
never leave this way: one writing the packed buffer itself, since that buffer is the boundary the refusal exists to
place, and one read from outside the region, whose value has to be stored for those readers regardless.

That is a boundary-placement rule, not a lowering-driven exception. It asks only which side a value's readers are on,
and it is what lets a contraction see a packed operand's two scale levels — the raw per-block byte and the k-invariant
per-tensor factor — as separate loads, the shape `ir/schedule/packing.py` reads and the block-scaled tensor-core cell
requires. Materializing the fused product instead does not merely cost bytes; it erases the block structure from the
consumer's index, and a reading that cannot prove k-block invariance declines.

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
3. move every `Write` to `TileOp.output_specs`, as a sweep spec over its output loop, the loop's per-cell projection
   lifting as a zero-axis term declaring that axis;
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
parent/child interface. Materialization binds accepted choices to placed geometry and resolved transport facts;
unsupported forms remain unmapped.

## Kernel boundaries after maximal fusion

Maximal Loop fusion remains canonical. Tile lowering may expose two kinds of graph-fragment siblings without changing
that canonical input:

- **`030_cut`** offers the maximal fused Fold tree and every closed stored child-Fold seam — body-member folds
  closed at offer time by provider closure, and dependent seams offered as principal closures (see the placement
  discussion above). A cut writes one workspace
  per state component and replaces all occurrences of the same canonically shared Fold object with workspace loads.
  Closure and replaceability are semantic gates; operation family, expected speed, row order, and search-space size
  are not. Closure reads the complete lowered statement stream through `Body`'s scope-aware dependence analysis:
  an axis bound by one loop does not scope its siblings, and dead-but-still-emitted statements retain their free axes
  until a lowering pass removes them. A contraction's operand edges are seams of the same class: cutting one
  materializes the cone feeding that
  operand into its own kernel and the contraction reads it back as an ordinary load. Such a seam's workspace dtype is
  decided EXPLICITLY — the dtype the consuming contraction's output is stored at (traced through any epilogue to the
  output it feeds, so a sibling output at another width cannot mis-type it), which is the element the fused slab
  would have stored — never the carrier the cone computed in: only the `a` edge has a converting fill, so an f32
  workspace on a `b` edge could feed no warp atom. One refinement overrides that rule: an operand cone that passes
  through a STORAGE FRONTIER — a decode (the `ElementwiseImpl.decodes` trait) of a value the cone itself computes —
  cuts at the frontier instead (`_cut.storage_frontier`): the producer piece is the encode prefix, the workspace holds
  the raw storage bits (exact — the element the graph's own quantize produced), and the consumer keeps the
  decode-plus-factors residue, which normalization then re-binds as a raw storage-dtype load with the factors hoisted
  onto the accumulator epilogue (W8A8's route to the fp8 mma tier). The frontier REPLACES the fed-store realization at
  that seam rather than joining the offer: the raw bits dominate the fed-store workspace on both precision (exact vs
  re-rounded) and footprint (storage width vs store width), so there is no trade for the evidence to decide. Every
  seam's per-component dtypes are decided at offer time and ride the seam into realization, so the two cannot
  disagree. A cut workspace retains captured axes plus static unit axes: unit extents add no storage, while preserving
  them keeps later schedule and split axes in their original geometric roles. The new producer and consumer are fresh
  unmapped TileOps, so further legal cuts and schedules use the same ordinary passes. An unpinned cut may expose more
  cut choices; any pinned cut consumes its restriction on every piece. If the parent already carries a cross-CTA
  split receipt, every placement piece inherits it, so a later cut cannot make the same split pending again. A piece
  minted by a structural apply stays in the ordinary pass sequence; no schedule-specific visitor discovers or
  realizes another placement decision.

- **The cross-CTA reduce split is structural.** Splitting the reduce axis across CTAs into a partial and finalize
  changes which kernels exist, so `030_cut` offers it after stored-edge placement and before any assignment
  is enumerated.
  Each fresh piece then enters the ordinary schedule pass with the partition receipt described below.

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
gaining a `Tile`, never a new path.

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
`mma_<shape>_<ab_dtype>_<acc_dtype>`, the compressed PTX/CUTLASS D.A.B.C order, with no acc-unspecified alias): on
the consumer GeForce dies (sm_86/89/120)
f32-accumulate HMMA runs at HALF the f16-accumulate rate, so this atom keeps the whole mma chain on the full-rate f16
accumulator and the lowering promote-folds the packed f16 partials into f32 shadow fragments per K chunk
(`FragmentPromote` — the staged bk slab is the cadence; gmem-direct promotes every `_atom._F16ACC_STEPS` steps plus a
final fold). Precision-gated enumeration, off by default — the precise `EMMY_F16_MMA_F32_ACC` parameter admits it on
any target where the atom is statically available, while the `EMMY_FAST_MATH` umbrella admits it on the consumer-die
ccs only (`_F16ACC_CCS`). A `TILE` restriction must still compose with that precision restriction.
The realized fork is identified by the `TILE`
codec's atom token and priced by the `MMA_acc_bits` feature; f16 only (mma.sync has no bf16-accumulate form).

**The move catalog** (`ir/schedule/catalog.py`) supplies the static choices projected into the classic domains.
`scalar_tile_moves()` is the union of three fixed scalar tile products: pure register tiles, a one-dimensional N-thread
ladder, and two-dimensional thread tiles × per-thread register tiles (`block_threads ≤ 1024`), with the per-cell
`""` tile as one more member. The normal cooperative-reduction catalog is likewise the fixed cooperative-width × ILP
product. The scheduler projects these alongside the warp and transport catalogs. Every accepted leaf is a complete
`Schedule` with exact integer node ids and `(consumer, operand)` edge tuples. Schedule parameters restrict
complete assignments without
changing those exact domains.
The producer band is a fixed kernel-domain factor (`""`, `+p1`, `+p2`; since step 7 a resolved band is spelled in
`WORK`, never a per-row `WSPEC` key). Compatibility accepts a nonzero member only on a warp row over resolved **TMA**
transport without a cross-CTA split and within the thread budget; a parameter can restrict this factor but cannot add
another width. A single-channel computed-A (fused-cone) contraction enumerates scalar
register-tile rows with staging off: the scalar atom evaluates the cone once per operand row or column and reuses it
across the sibling register cells. It also enumerates its warp rows with the mandatory resolved `sync` compute-fill
stage at BOTH depths
(`d1` + the asymmetric B-only prefetch ring `d2` as fork siblings — the M=512 occupancy loss inverts at decode M,
so the depth is measured per shape), crossed with the shared `RASTER` launch-order candidates (its B stripes
re-stream per M-tile row, exactly the grouped order's L2 reuse — `gn8` measured −8% on the gemma gate_up fused
edge, 5090). The **redundant-statistic split-K** form is no longer an assignment row: the structural
`030_cut` pass
slices the contraction across CTAs BEFORE assignment composition, σ-reindexing the per-cell cone to
absolute k while the k-invariant stat prologue stays full-row in every partition (each recomputes it, which is
cheap on the small-free decode shapes and is left to evidence to price), and the wrapping zero-axis fold's
projection folds into the deferred finalize. Multi-channel (gate/up) nodes split too: the sliced contraction
carries the true N-component identity-family carrier (one additive state per channel), the partial stores each
channel's raw state to its `ws[comp, ksplit, *cell]` slice — no ⊗-combine in
the partial — and the deferred finalize folds every component before applying the combine projection once.
Multi-channel products still have no scalar / gmem-direct / WSPEC rows; the compute-producer role for the fused edge
is the anticipated
`RoleKind` extension. `TILE` parameters match each site's own catalog through the exact codec spelling: an explicit
`TILE@n<ordinal>` restricts one site when supporting sites need different values, while the canonical bare spelling
restricts every site that supports its value. A value absent from every applicable factor leaves no assignment rather
than changing a factor. Staging additionally
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
