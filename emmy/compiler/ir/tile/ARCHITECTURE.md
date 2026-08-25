# Tile IR — a complete Fold tree

`LoopOp → TileOp` first performs a structural lift. The boundary peels the outer parallel loop chain into
`Placement.free`, then converts every remaining reduction loop into a `Fold`. `TileOp.__post_init__` subsequently
canonicalizes the complete Fold tree before scheduling.

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

`normalize.py` owns the idempotent, bottom-up Fold-tree normalization invoked by `TileOp.__post_init__`. It contains
only reusable local rules: scoped lambda alpha-canonicalization and clustering, semiring contraction
canonicalization, and closed child-Fold extraction from a root projection. The contraction rule requires direct
indexed operands and the registered distributive product/commutative-monoid laws; a Fold that does not prove those
conditions remains planar.

Whole-tree patterns such as SDPA are intentionally absent. They consume this canonical tree in a later pass.

`pipeline/passes/lowering/tile/_lift.py` peels the outer free axes, invokes that conversion, separates root stores,
checks the no-inner-loop invariant, and creates one zero-axis root `Fold` over the lifted cell.

## TileOp and scheduling

`TileOp` owns facts deliberately excluded from the Fold tree: placement, workers, schedule slices, knobs, and boundary
stores. Schedule slices remain keyed by `path.py` and read through `ops.Sched`.

Scheduling sees only the stored Fold tree. It does not derive alternate classified views. A shape for which the
current scheduler has no row remains unmapped; that is an expected red-state limitation while recovery proceeds by
fold-tree shape.
