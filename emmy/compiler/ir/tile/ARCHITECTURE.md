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

`Fold.defines()` exposes the fold results to its containing lambda. This is what lets later statements in an SDPA
cell read the maximum, denominator, or nested QK result without extracting or relocating any subtree.

`Fold.loop` is the inverse spelling used by materialization. The term itself carries no placement or schedule.

## Total lift

`pipeline/passes/lowering/tile/_fromloop.py` implements the only loop conversion:

1. recursively lift nested reductions in place;
2. remove the current loop's `Accum` statements from its step body;
3. build the `lift`, `init`, and `combine` directly from those accumulators.

There is no SDPA matching, byte-identity recognition gate, softmax pairing, fused view, or raw-loop fallback at this
boundary. Unsupported non-canonical Loop IR fails loudly. Kernel placement is a later fork over this complete tree.

## Canonicalization

`Lambda.__post_init__` owns context-independent construction normalization through `ir/pure/normalize.py`: every
pure body receives a dependency-safe order and commutative `Assign` arguments are sorted before it reaches a Fold.
Structural identity therefore reads the stored order directly. Contraction canonicalization first orders product
arguments by geometry, then places the one argument shared by every product in the Fold's shared operand slot.
Physical M/N orientation remains a placement fact rather than part of the Fold algebra.

`normalize.py` owns only the idempotent, bottom-up rules that need Tile context: scoped lambda alpha-equivalence and
clustering, semiring contraction canonicalization, and closed child-Fold extraction from a root projection. The
contraction rule keeps the distributive product in the outer reduction and factors each maximal pure product-operand
cone into a zero-axis Fold edge. Alpha-equivalent product arguments coalesce to one shared result even when their
source cones overlap; other overlapping cones become one multi-result operand edge so shared computation remains
single. A semiring without one shared product argument remains a general planar Fold.

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
exposed when factoring makes a contraction the root compute node. Multiple output specifications owned by one
projection region reconstitute one loop.

A matrix row that Loop IR elided because its static extent is one remains algebraic information when every output
specification writes `[0, n]`. Post-init restores that proven unit free axis before contraction canonicalization. The
rule is boundary-derived and general: it does not recognize a model or operation family, and it does not alter a term
whose output specifications disagree about the missing coordinate.

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

`identity.py` is the home for every "are these two kernels the same?" question, and the index over
the ones answered elsewhere. The term digest (`_key.py`) canonicalizes α-renaming, buffer spelling
**and sizes** away — right for the algebra, wrong for everything downstream — so each coarser
identity folds the excluded facts back in through a named fingerprint rather than deriving them at
the call site. `deploy_identity` is the verified-tier join key; `pool_key` is the schedule-space
key and takes the live pin fingerprint as a required argument, which is what keeps the module a
pure function of a `TileOp` and below the pipeline layer.

A fact that changes what a reader produces, and that the term does not carry, belongs in a
fingerprint here. Omitting one is silent, and both known omissions cost the same way. `pool_key`
shipped without per-axis extents, so two matmuls with transposed M/N — equal terms, equal
`S_ext_*` summaries — shared one pool entry over spaces of 57442 and 8280 candidates. Buffer
shapes and the output specifications were missing from both identities, so a `(128, 128)` output and a
`(4, 32, 128)` one over the same iteration space collided: the split form spells its coordinate as
a dim pair the fragment store can address only under a divisibility rule, and a golden measured on
the flat kernel joined a kernel that could not realize its row.

Static extent products used as structural features saturate at the largest finite float. Feature extraction therefore
stays bounded even for a deeply nested symbolic-model fixture whose exact integer product is too large to convert,
while retaining exact values throughout the ordinary extent range.

## TileOp and scheduling

`TileOp` owns facts deliberately excluded from the Fold tree: placement, workers, schedule slices, knobs, and output
specifications. Schedule slices remain keyed by `path.py` and read through `ops.Sched`.

`lowering/tile/030_cut` offers kernel placement before scheduling. `PLACE` uses the same tree-path codec to address a
stored non-root Fold edge. The fused sibling preserves the maximal Fold tree; each semantically closed cut sibling
writes the child Fold's complete state tuple to workspaces and replaces every canonically shared occurrence with
ordinary `Load` edges. Both producer and consumer are fresh unmapped `TileOp`s, so they re-enter the same placement
and scheduling rules. Synthesized evaluation nodes are not cut sites, and the rule neither recognizes operation
families nor filters legal cuts by profitability.

Scheduling sees only the rewritten stored Fold tree. Every Fold is an addressable schedule site; the scheduler does
not derive alternate classified views or suppress a child because its parent may realize it. A derived unit-axis
contraction inherits its enclosing Fold's reduction domain through the parent/child scheduling interface, while its
output tile remains at the child site. Local catalogs compose lazily through one worker inventory and equal geometry
on any shared physical axes. The same rule combines independent roots, including roots whose algebraic M/N readings
reverse the same physical axes. A shape with no legal row remains unmapped, and scheduling never replaces or
annotates the Fold tree.
