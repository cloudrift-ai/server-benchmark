# Tile IR — a complete Fold tree

`LoopOp → TileOp` first performs a structural lift. The boundary peels the outer parallel loop chain into
`Placement.free`, then converts every remaining reduction loop into a `Fold`. `TileOp.__post_init__` subsequently
canonicalizes the complete Fold tree before the separate algebraic rewrite and scheduling passes.

The invariant is simple: **a lifted Tile IR kernel contains no raw inner `Loop`**. A reduction nested in another
reduction occupies the same statement position in the parent fold's `lift.body`, so source order and SSA scope are
preserved. A non-reduction output sweep becomes a boundary `Store.sweep`; any other surviving loop is a formation
error.

## Fold storage

`Fold` stores `reduce(⊕) ∘ map(f)` directly:

| field | meaning |
| --- | --- |
| `axis` | reduction axis; `None` for the root pointwise projection |
| `lift` | pure per-element `Lambda`; nested reductions are nested `Fold` statements here |
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

There is no SDPA matching, byte-identity recognition gate, softmax pairing, fused view, placement cut, or raw-loop
fallback at this boundary. Unsupported non-canonical Loop IR fails loudly.

## Canonicalization

`Lambda.__post_init__` owns context-independent construction normalization through `ir/pure/normalize.py`: every
pure body receives a dependency-safe order before it reaches a Fold. Structural identity therefore reads the stored
order directly. Commutative argument sorting remains part of α-equivalence rather than stored normalization because
contraction operand position carries the A/B role.

`normalize.py` owns only the idempotent, bottom-up rules that need Tile context: scoped lambda alpha-equivalence and
clustering, semiring contraction canonicalization, and closed child-Fold extraction from a root projection. The
contraction rule keeps the distributive product in the outer reduction and factors each maximal pure product-operand
cone into a zero-axis Fold edge. Enclosing free-axis order determines A/B position; overlapping cones or an unproved
registered semiring leave the Fold planar. Canonicalization runs entirely in `TileOp.__post_init__`, including the
output-sweep-to-free-axis adjustment exposed when factoring makes a contraction the root compute node.

Scoped lambda equivalence uses the same dependency-respecting body order as structural identity and canonicalizes
commutative argument order. It therefore ignores SSA spelling and harmless interleaving without weakening buffer or
axis identity. The emit-side same-score legality query uses this same mechanism rather than maintaining a second cone
canonicalizer.

`pipeline/passes/lowering/tile/_lift.py` peels the outer free axes, invokes that conversion, separates root stores,
checks the no-inner-loop invariant, and creates one zero-axis root `Fold` over the lifted cell.

## Algebraic rewrite

`pipeline/passes/lowering/tile/015_twisted.py` runs after construction canonicalization and before scheduling. It
clusters sibling Folds by scoped lambda equivalence and rewrites a maximum plus additive exp-weighted components into
one exp-family twisted carrier `(maximum, denominator, expectations…)`. Pure softmax is the arity-two case; SDPA adds
expectation components, and a causal mask is simply part of the shared score lambda. The pass has no operation-family
matcher. Ordinary `copy` aliases of the carried maximum are followed before the exp-weighted components are compared.

The rewrite consumes the canonical Fold tree. It reuses the registered monoid generator, invariant-factor splitting,
and scoped score equivalence both for sibling maximum/additive folds and for the equivalent canonical composition in
which contraction normalization has placed those statistics inside a computed normalized-exponential operand.
Normalization factors remain in the projection epilogue, while a directly loaded expectation value becomes a Fold
operand; the generic twisted Fold derivation then exposes the corresponding contraction to scheduling.

## TileOp and scheduling

`TileOp` owns facts deliberately excluded from the Fold tree: placement, workers, schedule slices, knobs, and boundary
stores. Schedule slices remain keyed by `path.py` and read through `ops.Sched`.

Scheduling sees only the rewritten stored Fold tree. It does not derive alternate classified views. A shape for which
the current scheduler has no row remains unmapped; that is an expected red-state limitation while recovery proceeds
by fold-tree shape.
