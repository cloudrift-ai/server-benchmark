# Tile IR — a complete Fold tree

`LoopOp → TileOp` first performs a structural lift. The boundary peels the outer parallel loop chain into
`Placement.free`, then converts every remaining reduction loop into a `Fold`. `TileOp.__post_init__` subsequently
canonicalizes the complete Fold tree before the separate algebraic rewrite and scheduling passes.

The invariant is simple: **a lifted Tile IR kernel contains no raw inner `Loop`**. A reduction nested in another
reduction occupies the same statement position in the parent fold's `lift.body`, so source order and SSA scope are
preserved. A non-reduction loop whose local values feed writes becomes a pure `ProjectionRegion`; each write becomes
an `OutputSpec` owned by the `TileOp`. Any other surviving loop is a formation error.

## Fold storage

`Fold` stores `reduce(⊕) ∘ map(f)` directly:

| field | meaning |
| --- | --- |
| `axis` | reduction axis; `None` for the root pointwise projection |
| `lift` | pure per-element `Lambda`; nested reductions occupy their original structural position here |
| `init` / `combine` | componentwise monoid read mechanically from the loop's `Accum` statements |
| `operands` | explicit materialized or computed inputs used by later transforms |
| `observe` | optional per-step observer — the scan spelling: a pure `λ(k, state…)` evaluated after each combine |

`Fold.defines()` exposes the fold results to its containing lambda. This is what lets later statements in an SDPA
cell read the maximum, denominator, or nested QK result without extracting or relocating any subtree.

`Fold.loop` is the inverse spelling used by materialization. The term itself carries no placement or schedule.

Every stored `combine` is claimed by a registered **monoid family** (`ir/pure/algebra.py` — componentwise, or a
twisted entry such as exp/LSE; membership is generator-output equality, never an annotation), and the claiming family
answers the family-shaped reads: the `TWISTED` role, the cross-partition merge realization, the rename regeneration,
and the legality properties (`commutative`, `observable`).

A fold with an `observe` is a **scan**: the observer binds `(axis, *state)` positionally, its results are fresh names
only kernel-boundary `OutputSpec` writes consume, and the streamed store reconstitutes inside the reduce loop after
the observer stmts (`observed_result_names` + the `observed=` reconstitution arm). An observed fold makes the stream
order-visible, so the schedule offers exactly the serial reduce plan and the cross-CTA split fork declines it.

## Total lift

`pipeline/passes/lowering/tile/_fromloop.py` implements the only loop conversion:

1. recursively lift nested reductions in place;
2. remove the current loop's `Accum` statements from its step body;
3. build the `lift`, `init`, and `combine` directly from those accumulators;
4. a per-step `Write` over the carried state (the `025_lift_scan` shape) peels into an observer — the fold gains
   `observe` with fresh `<state>__obs` results, and the store, rewritten to read the observed name, rides the stream
   position after the node, where output-spec extraction claims it as an ordinary boundary write.

There is no SDPA matching, byte-identity recognition gate, softmax pairing, fused view, or raw-loop fallback at this
boundary. Unsupported non-canonical Loop IR fails loudly. Kernel placement is a later fork over this complete tree.

## Canonicalization

`Lambda.__post_init__` owns context-independent construction normalization through `ir/pure/normalize.py`: every
pure body receives a dependency-safe order and commutative `Assign` arguments are sorted before it reaches a Fold.
Structural identity therefore reads the stored order directly. Contraction canonicalization first orders product
arguments by geometry, then places the one argument shared by every product in the Fold's shared operand slot.
For a broadcast-batched product whose batch axis occurs in only one operand, the placement's trailing output pair
still supplies that geometry. If its geometric first operand reads the reduction axis non-contiguously and the other
materialized operand reads it contiguously, the commutative product puts the contiguous operand in the shared A slot;
placement then derives the corresponding physical M/N orientation from the operand axes. Physical M/N orientation
remains a placement fact rather than part of the Fold algebra.

`normalize.py` owns only the idempotent, bottom-up rules that need Tile context: scoped lambda alpha-equivalence and
clustering, semiring contraction canonicalization, and closed child-Fold extraction from a root projection. The
contraction rule keeps the distributive product in the outer reduction and factors each maximal pure product-operand
cone into a zero-axis Fold edge. Alpha-equivalent product arguments coalesce to one shared result even when their
source cones overlap; other overlapping cones become one multi-result operand edge so shared computation remains
single. A semiring without one shared product argument remains a general planar Fold.

Closing runs at both scope kinds. A zero-axis root moves its body dependencies onto captured contraction operands;
a REDUCING fold does the same for a chain that depends on its own iteration axis and so lives in its lift body —
attention's per-key statistic and rsqrt ahead of the score dot's computed B cone. The reduce-body move is gated on
exclusive consumption (every moved definition dies into the closed edges), so the step's work is repackaged, never
duplicated. Both rules measure an edge's captures with `Fold.deps` — scope-aware, so a name a sibling operand binds
inside the edge is not a capture and an already-closed edge never re-fires the rewrite. A cone closed at its axes is
what the placement fork can offer as a workspace seam, which is how a computed operand (the RMSNorm'd, RoPE'd K
vector) becomes materializable once per key instead of recomputed per query row.

An iteration never crosses into a new evaluation domain. Attaching an iteration-bearing provider to a contraction
operand would evaluate it once per step of every intervening binder; normalization therefore leaves that provider at
its defining scope. Straight-line chains still close normally, and a reducing root's own axis counts as its existing
domain. A stored fold left capturing by this rule remains placeable: the placement fork resolves its captures outward
through the occurrence's lexical environment at offer time, without moving the stored tree.

Three walks compute a capture's provider cone, at three stages, and each is allowed to move something different.
Normalization's closing rules (`normalize.py`, above) are the only walk that REWRITES the stored tree, and only for
straight-line providers within one evaluation domain. The placement fork's provider closure
(`lowering/tile/_cut.py`) moves nothing: it proves every occurrence resolves to equal sources and offers the closed
value as a seam, recording fold producers as the seam's requirements. Kernel lowering's per-cell closure
(`lowering/kernel/_factor.py`) moves sibling providers into a computed operand's compute fill of the one kernel being
emitted — a codegen fact that exists only inside that realization. A new capture-resolution need belongs in one of
these three, not a fourth walk.

An identity pass-through — a projection that only re-exposes its single operand's results — dissolves wherever a
projection is formed or revisited. That is not cosmetic: a pass-through is what makes two occurrences of the same
computation compare unequal, and the placement fork's value clustering (`lowering/tile/_cut.py`) relies on
alpha-equivalent cones converging to one canonical shape.

Normalization ends by restoring OBJECT SHARING: same-value cones — alpha-equal with identical captures and exposed
result names, so a copy differing only in internal binder spelling still qualifies — collapse onto one Fold object
(`_share_common_cones`), so a value fusion inlined into several consumption sites — attention's softmax
statistics, read by the weight cone and by the epilogue — is one node again. This is an invariant, not an
optimization: seam grouping and cut realization key on object identity, so severed sharing silently turns one value
into per-site recompute that no schedule can undo (the class PR #679 measured at three orders of magnitude). A
recompute observed across kernels is therefore ALWAYS a Tile-level sharing or seam-offer defect — fix it here or in
the placement fork; a Loop IR fusion or emission workaround sees one kernel at a time and is the wrong altitude by
construction. Copies that differ in captured axis names cannot share an object and stay with value clustering; copies
that differ in exposed result names stay distinct because unifying them would rename their consumers.

An output sweep used by any nested contraction operand is promoted into the Tile's free-axis placement. Promotion
expands the enclosing-axis context, so construction normalizes the Fold tree once more under that final scope; one
construction and a reconstruction therefore expose the same closed operand edges and placement seams.
The invariant also applies when a schedule row constructs or reloads an already-mapped Tile: promotion extends the
grid in lockstep with the free axes, so per-cell replication never mistakes the swept coordinate for an SSA name.

**Storage-decode factors hoist to the epilogue.** A product operand whose cone is a STORAGE DECODE
(`ElementwiseImpl.decodes` — the trait, never an op-name list) times factors constant along the fold
axis is not left as a computed cone. The decode is absorbed by the raw load's storage dtype, since every
consumer converts a bits-carrier element by dtype, and the invariant factors commute out onto the
accumulator: `Sum_k a*(s*w) = s*Sum_k a*w`, the same reassociation category as split-K. The rule is
side-generic, so a W8A8 cell binds BOTH operands raw and composes the two scales into one epilogue
chain; with several channels a shared operand's factor is applied to each accumulator. A contraction
reached with no projection to host the factors gets one; nested contractions use their parent's.

This is what makes quantized weights reach the tensor cores at storage width. Without it the residue is
a computed cone, and a computed cone can feed no native fp8 atom at all — those atoms require a
materialized f8 `a` — so the whole `mma_m16n8k32_*` family becomes unreachable and a W8A16 weight
routes through the smem compute fill instead of the gmem-direct converting fragment load. The decode
gate is deliberate: an ordinary floating-point factor chain keeps its cone, so this never reassociates
arithmetic that was not already a dequantization.

**What the planar fallback costs.** Declining to factor a stat-free cone is not a neutral fallback — it demotes the
cell to a scalar fold, which cost the gemma-4 M=256 post twin **144 ms against a 4.3 ms bound**. A cone-internal load
is `(m, k)`-indexed like the shared operand, so a positional reading rather than a geometric one once bound gemma's
GeGLU combine as `gate @ W` and silently dropped both the gelu and the up projection; on the B side the same positional
reading dropped the fp8 decode cone's scale out of the kernel entirely. Both are silent — the numerics stay plausible
and only the measurement moves. Treat a contraction that canonicalizes to PLANAR as a coverage bug to investigate, not
as a supported slow path.

Canonicalization runs entirely in `TileOp.__post_init__`, including the legacy output-sweep-to-free-axis adjustment
whenever a contraction operand reads the sweep axis. The contraction may be the root compute node or a later site in
the Fold tree; in either case the coordinate belongs in kernel placement rather than a post-compute output loop.
Multiple output specifications owned by one projection region reconstitute one loop. The extraction round-trip gate
compares the reconstruction against the stream **as reconstitution will spell it**: a `ProjectionRegion` stores its
body in a `Lambda`, whose construction canonicalizes statement order and commutative arguments, so a raw stream that
normalization reorders is still representable — the gate normalizes both sides the same way. A stream already in
canonical form is gated on byte-identity exactly as before. A write whose stored value is captured from the enclosing
scope unchanged (`o[j] = acc`, broadcasting an already-reduced accumulator) has no body def to name, and is declined:
the fusion lift-preflight turns that into "leave the region unfused".

A sweep axis is bound only by the per-cell output `Loop` reconstitution wraps around the projection body — never at
kernel scope. Canonicalization therefore keeps a non-contraction fold that reads a sweep axis a projection BODY
member: hoisting it onto an operand edge would lower it outside the sweep loop, rendering the axis as an undefined
identifier (DeepSeek-V4 post16's per-column sum was the live case). A contraction is exempt because post-init
promotes a sweep its operands read into a real free axis right after normalization.

A fold FED by the body — one whose subtree captures a name a plain body member defines — is likewise never hoisted,
no matter what kind: a projection evaluates its operands before its scalar body, so the capture would read a value
that does not exist yet. The `closed` gate reads only the fold's own lift and cannot see a nested capture; the
composed placement cut builds exactly this shape (the consumer piece's workspace loads and rsqrt chain feed the
retained reduce — DeepSeek-V4 post4096's two-cut piece was the live case, every capture an undefined identifier at
nvcc). The classic reduce domain mirrors the fact for depth: a fold reached deeper than a chain-form root's direct body
members still offers only the serial fold, since the body recursion emits it serially per cell regardless of
partition. A DIRECT member is not so limited — the retained reduce above is one — and offers the full non-transposed
reduce catalog instead (absent a swept or streamed boundary store, neither of which the chain arm's lane-distributed
close can realize), bound through the chain arm ahead of the strided loop.

A matrix row that Loop IR elided because its static extent is one remains algebraic information when every output
specification starts with one or more literal-zero coordinates followed by the dense `n` coordinate, directly or
split into row-major quotient/remainder coordinates by a pure reshape. The `n` coordinate may already be free or may
still be the one shared output sweep. Post-init restores that proven unit free axis before contraction canonicalization,
even when a sibling reduction is the root-most Fold and the contraction is nested. A zero after `n` or a strided `n`
does not prove a unit matrix row. The rule is boundary-derived and general: it does not recognize a model or operation
family, and it does not alter a term whose output specifications disagree about the missing coordinate.

Factoring preserves the pure cone's statement order. If a scalar projection between two nested Folds feeds the later
Fold, the earlier Fold and scalar become a nested source projection; both Folds are never flattened ahead of that
scalar. The stored Fold tree therefore lowers to the same dependency order as the canonical Loop IR input.

Scoped lambda equivalence uses that normalized order. It therefore ignores SSA spelling and harmless interleaving
without weakening buffer or axis identity. The emit-side same-score legality query uses this same mechanism rather
than maintaining a second cone canonicalizer.

`pipeline/passes/lowering/tile/_fromloop.py` exposes the total-lift entry used by the pass and golden replay. It peels
the outer free axes, invokes the conversion, separates output specifications, checks the no-inner-loop invariant, and
creates one zero-axis root `Fold` over the lifted cell.

## Algebraic rewrite

`pipeline/passes/lowering/tile/020_twisted.py` runs after construction canonicalization and before scheduling. It
clusters sibling Folds by scoped lambda equivalence and rewrites a maximum plus additive exp-weighted components into
one exp-family twisted carrier `(maximum, denominator, expectations…)`. Pure softmax is the arity-two case; SDPA adds
expectation components, and a causal mask is simply part of the shared score lambda. The pass has no operation-family
matcher. Ordinary `copy` aliases of the carried maximum are followed before the exp-weighted components are compared.

The rewrite consumes the canonical Fold tree. It reuses the registered monoid generator, invariant-factor splitting,
and scoped score equivalence both for sibling maximum/additive folds and for the equivalent canonical composition in
which contraction normalization has placed those statistics inside a computed normalized-exponential operand.
Normalization factors remain in the projection epilogue, while a directly loaded expectation value becomes a Fold
operand; the generic twisted Fold derivation then exposes the corresponding contraction to scheduling.

## Kernel identity

Every "are these two kernels the same?" question is answered by ONE function — `Op.identity_key`
(`ir/base.py`), a lattice over the canonical Loop-IR body with one flag per additional fact
(`structural` cluster-collapse, `with_io`, `with_knobs`). `TileOp`'s contribution is `loop_body` —
the complete schedule-free Loop-IR body the kernel executes, derived from the term (the free grid
axes wrapped back as plain loops around `lower_with_output_specs`, so the extents, the store
program and a cut child's typed seam `Load` are all in the body) — and the private
`_body_identity` override that digests it. There is no separate term hasher: `Fold.structural_key`
is the exact-flavor digest of the term's own lowered body (the term is pure algebra; its body is
its normal form). The named lattice points are spelled at call sites: the deploy identity
(`with_io=True` — the durable join key) and the variant key (`with_io=True, with_knobs=True` —
the search tree and measurement stores). There is no schedule-space key on
the interface: the enumeration's `pool_id` stamp is minted at its one site in
`lowering/tile/040_schedule` (the variant key + hints + pins + sample identity) — a stamp for the
greedy decision memo and the budgeted descent seed, not a cache key: nothing stores pools.

Identity has two flavors: the default `structural=True` is schedule-equivalent (compute-unit op
clusters collapse — `relu` and `tanh` epilogues share a key because their schedule evidence
transfers, which is what golden records join on), while `structural=False` names the exact kernel.

The design lesson the interface encodes: a fact a schedule reads must be in the body or the io
fingerprint, never re-derived beside a caller. The pool digest once shipped without per-axis extents,
so two matmuls with transposed M/N — equal terms, equal `S_ext_*` summaries — shared one pool
entry over spaces of 57442 and 8280 candidates; buffer shapes and the output specifications were
once missing too, so a `(128, 128)` output and a `(4, 32, 128)` one over the same iteration space
collided — the split form spells its coordinate as a dim pair the fragment store can address only
under a divisibility rule, and a golden measured on the flat kernel joined a kernel that could not
realize its row. Both facts now live in the completed loop body and the io fingerprint.

Static extent products used as structural features saturate at the largest finite float. Feature extraction therefore
stays bounded even for a deeply nested symbolic-model fixture whose exact integer product is too large to convert,
while retaining exact values throughout the ordinary extent range.

## TileOp and scheduling

`TileOp` owns facts deliberately excluded from the Fold tree: placement, an accepted `Schedule`, its separate
classic materialization, knobs, and output specifications. The semantic assignment contains choices only;
site-indexed placed tile geometry and resolved transport sizes are lowering facts and cannot enter a row identity.
`ops.Sched` is a read-only lowering view over those typed fields. There is no keyed slice map, per-node schedule
field, compatibility adapter, alias codec, or dual reader.

A scheduled `TileOp` must carry both its schedule and materialization, and the materialization validates itself
against the schedule. A classic materialization contains exactly the contraction sites whose accepted tiles require
placed geometry and exactly the edges whose accepted transport is non-direct. Every placed tile must equal the
geometry derived from the structural placement and its axis-free choice; every resolved stage must retain its edge's
choice. Construction rejects missing, extra, mismatched, or partly attached facts.

`ir/schedule/classic.py` owns the semantic contract for the ordinary grid/CTA/warp/thread/register schedule:

- `ClassicScheduleContext` composes the unscheduled `TileOp` and its target. The `TileOp` itself is the site index:
  one stable integer id per Fold identity, one distinct `(consumer id, operand position)` tuple per edge (including
  multiple uses of one producer), each site's view, and each contraction's `ContractionFacts`.
- Reusable schedule views classify one Fold without target input. A contraction records consumer-relative operand
  positions; it does not mint alternate nodes or edge identities. The derivations memoize on the Fold ROOT, so every
  `TileOp` over one term shares them; the `TileOp` properties are accessors, not a second cache.
- `path.sites` is a reading of that same walk, adding only what the codec needs: the per-site ordinal among sites
  sharing a `(segments, axis)`. It owns spelling, resolution and ambiguity — not traversal.
- `KernelSchedule`, `ProjectionSchedule` / `ReductionSchedule`, and `EdgeSchedule` contain choices only. They do not
  cache paths, classifications, shapes, placed geometry, resolved shared-memory sizes, or codec spellings.
- `ClassicScheduleContext` derives local support after selecting a node and its incident edges. `extend` composes it
  through worker inventory, physical-axis and fragment agreements, then composes the kernel factor through target
  limits, raster eligibility, and restriction. It returns a new immutable context or raises `ScheduleRefused`; a
  terminal context contains the generic typed `Schedule`.
- The production walk keeps its catalogs private. Every leaf is accepted by the context before search can observe it;
  an encoded dictionary is never a semantic leaf.
- Kernel, node, and edge domains are projected independently. Enumeration is the compatible subset of their Cartesian
  product, so changing traversal order may change work but can never change membership.
- `ClassicScheduleCodec` is the sole wire boundary. Kernel keys are bare `WORK` / `RASTER`. A node family is bare
  when it has one applicable site and uses `@n<N>` only when ambiguous. `STAGE` is one value per consumer node and
  follows the same rule. Decode requires the full key set and rejects aliases, missing direct values, unknown keys,
  and semantically refused assignments.

Structural choices are deliberately outside this algebra. A cut or split changes the kernel set first; every fresh
kernel then constructs a fresh problem and fresh sites. Search ranks encoded accepted leaves and materialization
consumes the typed assignment, so neither layer defines schedule membership.

The single `lowering/tile/030_cut` pass reaches a fixpoint over kernel-set alternatives before scheduling: placement
first, then cross-CTA reduction splitting. `PLACE` uses the same tree-path codec to address a
stored non-root Fold edge. The fused sibling preserves the maximal Fold tree; each semantically closed cut sibling
writes the child Fold's complete state tuple to workspaces and replaces every canonically shared occurrence with
ordinary `Load` edges. Both producer and consumer are fresh unmapped `TileOp`s. Unpinned cuts re-enter placement
before scheduling; any pinned cut carries the consumed placement decision on both pieces and proceeds to reduction
splitting. Synthesized evaluation nodes are not cut sites, and the rule neither recognizes operation families nor
filters legal cuts by profitability.

A computed edge injected into a twisted expectation is already the operand of the derived contraction that appears
when placement materializes it. Its workspace therefore uses the consumer's public store dtype, not the producer's
f32 reduction-carrier dtype; otherwise the materialized B slab would make every f16 tensor-core atom ineligible.

Scheduling sees only the rewritten stored Fold tree. Every Fold has one node site; the scheduler does not suppress a
child because its parent may realize it. A derived unit-axis
contraction inherits its enclosing Fold's reduction domain through the parent/child scheduling interface, while its
output tile remains at the child site. Local catalogs compose lazily through one worker inventory and equal geometry
on any shared physical axes. The same rule combines independent roots, including roots whose algebraic M/N readings
reverse the same physical axes. A shape with no legal row remains unmapped, and scheduling never replaces or
annotates the Fold tree.

A consumer holding a node learns its own address through `Sched.site_of` — one identity-keyed lookup against the
problem's site index. A family outside that node's schedule sum reads as "family doesn't apply"; a node outside the
problem raises `UnknownSiteError`. An identity miss is never the silent direct path.
