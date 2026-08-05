# Tile IR — one stored node kind, and everything else derived

`LoopOp → TileOp`: a map/reduce kernel with its *schedule* made explicit. The layer's rule is that the moveset is
**purely algebraic — no shape specializations**, and the way that is enforced is storage: there is exactly one node
kind, and every question a pass wants to ask is answered by deriving from it rather than by adding a field.

For the schedule VALUE types (`TilePlan` / `ReducePlan` / `Stage` / `Workers` / `Placement` / `WarpSpec`) and their
codecs see [`../ARCHITECTURE.md`](../ARCHITECTURE.md); for the row enumerator that fills them, see
[`../../pipeline/passes/ARCHITECTURE.md`](../../pipeline/passes/ARCHITECTURE.md).

## The one kind

`ir.py`'s **`Fold`** is `reduce(⊕) ∘ map(f)` in the λ-foldMap spelling:

| field | what |
| --- | --- |
| `axis` | the OPTIONAL iteration axis — `None` is the zero-axis node |
| `lift` | a pure `Lambda` `λ(k, v₁…vₙ) → S`: the element's SINGLETON state (ι is spelled here; softmax's is `(x, 1)`) |
| `init` / `combine` | the TRUE monoid, flat — ONE program whose results are the fold's real accumulator names (helpers in `ir/stmt/algebra`) |
| `operands` | a symmetric tuple of CLOSED inputs, each an edge, bound POSITIONALLY to the lift params |

**`Map` and `Contraction` are DERIVED READINGS, not stored kinds.** Each is a constructor returning a `Fold`, plus a
PREDICATE that answers the reading — `axis is None` for the projection, `is_contraction(x)` for the bilinear one. A
predicate cannot be constructed, subclassed or annotated, which is the point: there is no type to dispatch on and no
second place for the fact to live.

- A **ZERO-AXIS** fold is what `Map` was: no monoid, its `lift` IS the per-cell projection. So softmax's normalize,
  RMSNorm's, and flash's `divide(O, l)` are one kind composed at two depths.
- The **BILINEAR** shape — operands `(b₀, a, b₁…)` under a `multiply` lift with a componentwise-additive combine — is
  what `Contraction` was, exposing `a` / `channels` / `b_trans` off `operands`.

**Sharing is arity.** "These two matmuls read the same A" is ONE node whose output is a tuple: one `a` edge plus N
`Channel`s `(bᵢ, accᵢ)` folding the componentwise `(+, ×)`. The fused gate⊗up edge is that at N=2. No privileged
operand slot, no let table, no reference arm — the one in-tree relation that had several consumers is a single edge,
so a shared subtree has exactly one home by construction.

## Roles derive from arity

`Fold.role` is computed, never stored: `FREE` with no axis; `TWISTED` off the combine's twist family; `CONTRACTION`
off the bilinear reading alone; `PLANAR` otherwise.

Two consequences worth stating, because both used to be rewrites:

- **Split-K's outer reduce is not a contraction.** It tiles nothing and has no operand pair, so it derives `PLANAR`
  like any other additive fold. `Fold.composed` recognizes the reassociation — a structural probe, not a role.
- **The PLANAR demotion is a FORMATION fact.** An unbindable matvec-shaped contraction has its loads kept inline in
  the lift by recognition, so there are no edges for the bilinear reading to bind, and the derivation answers `PLANAR`
  by itself. Moving edges inline (`Fold.demoted()`, the schedule's COLLAPSE reading) is the whole operation; no role
  is rewritten anywhere.

Loops carry NO algebra — `Loop` / `StridedLoop` hold only their `AxisRole`. The lowering-side reads of the retired
`Algebra` bundle live in `passes/lowering/_reduction.Reduction` (the materializer's and `030_split_reduce`'s view).

## Nothing composed is stored

There is no `step` sequence. The composed evaluations DERIVE:

- flash's kv stream λ-spells with its QK score a HOISTED operand edge and its PV synthesized and memoized inside the
  derived blocked evaluation — `Fold.step_stmts()` is the one consumer read, and `ops.stream_pair` the one reading of
  the `(score, expectation)` pair that walk produces;
- split-K's outer reduce is the identity-lift composition — `Fold.composed` is the one read.

`Fold.from_loop` reconstructs the algebra from the loop BODY alone (degenerate facts off its `Accum`s; a twisted merge
regenerated-and-byte-compared, or extracted against a `like` fold for a split partial). It returns `None` for a
non-λ-representable loop, and callers keep the raw-loop-IR escape.

`Fold.loop` splices each operand's body before the first read of its bound param and flattens nested nodes in place,
so the derived loop depends only on the stored params. That is what makes kernel identity the α-INVARIANT **term hash**
(`ops.term_key`: canonical renumbering plus hash-time ANF body-order canonicalization), consumed by `op_cache_key`'s
TileOp arm and `Graph.structural_key`'s op field — never the lowered nest.

## An operand edge has two inhabitants

MATERIALIZED (a gmem `Load`) or COMPUTED (the node itself, stored INLINE; the cone built by `_atomize.make_cone`).

**Edge iff closed** holds BY CONSTRUCTION — operands bind positionally, so a subtree that reads a name from its
enclosing body cannot be one. The closure SCAN survives only as the validation reading, living with its one consumer
in `passes/lowering/tile/_cut.py`, where it decides cut legality: closed subtrees may hoist to edges, while combine's
derived material — flash's PV, whose `P` reads the running state — sits BELOW the seam lattice as a derived schedule
site excluded from PLACE (`Site.derived`). Flash's QK operand edge IS a PLACE site.

A cone's SOURCE is the row-invariant prologue (the per-row statistic) and its body the per-cell normalize, so the K
seam is the node boundary (`ops.cone_seam`). The A/B asymmetry that is real — A M-resident and compute-fillable, B
streamed — is a SCHEDULE fact read off the node (`isinstance(c.b, Load)` eligibility gates), not a storage fact.

## The node carries no placement and no schedule

The node is pure algebra, so its identity (`==` / `hash` / `term_key`) is its algebra alone and the term is IMMUTABLE
across the whole schedule search. The placed reading the tensor-core and staged tiers need — the `(m, n)` `Side`
geometry — belongs to the SCHEDULE SLICE: `TilePlan.at(m, n)` binds the caller's placement axes onto it, and the slice
then derives `mn` / `m` / `n` / `launch_threads`. `axes` is `compare=False`, so placement never reaches `spell()`, a
stamped knob row, a golden or a prior key.

The A/B split rides the stored operand ORDER, not the accesses: node-locally `A[m,k]` and `B[k,n]` are symmetric (each
carries K plus one free axis), so telling M from N needs the PLACEMENT — a caller fact. Node and slice travel as a
`(node, tile)` PAIR; there is no fused view type.

The slice reads the TILED CELL. What lies OUTSIDE it — the kernel's leading batch / ksplit grid axes, the per-cell
rename's shared coordinates — is the grid's fact, threaded to `kernel/_atom` by `_factor`.

A projection has ONE home, the wrapping zero-axis fold's `lift` — never a node field. Since 1q that lift is a STRICT
pure `Lambda`: the root-store `Write`s (and the rms/softmax output-sweep `Loop` around them) ride `TileOp.stores` as
`Store` decorations at the kernel boundary, reconstituted on demand by `effect_tail`. The raw-loop-IR kernels that are
not recognized algebra — the un-recognized escape cell, `030`'s finalize, the coop fused-tail sibling — keep an impure
lift through the one `_loop_ir_fn` arm.

## What each frontend shape stores

| shape | term |
| --- | --- |
| a bare reduce | a root `Fold` |
| softmax / RMSNorm | `Fold.projection(body=<per-cell normalize>, operands=(<the stat fold>,))` + a sweep `Store` |
| fused norm→linear / gate⊗up | a zero-axis fold over the product contraction (a fork sibling of its coop-reduce form) |
| a pure pointwise cell | `Fold.projection(body=…)` with no operands + its root `Store`s |
| flash | the `TWISTED` fold on the streaming schedule — QK a hoisted operand edge, PV the derived evaluation's synthesized node |

A twisted monoid is a monoid, selected structurally rather than as a distinct kind.

## `TileOp` and the tree-path codec

`TileOp` keeps `op` + `place` + `work` + `workers` + `knobs` + `schedule` + `stores`. `work` is the ONE worker
inventory, derived loudly from the TILE slices (`ops.seal_workers` — a cross-site disagreement raises).

Every schedule slice lives in **`TileOp.schedule`**, a dict keyed by the tree-path codec's canonical key. `path.py` is
ONE walker plus one resolver, short-path-canonical: bare for the primary node, `TILE@dd` / `TILE@pj` on flash. Read
and written through `ops.Sched`, which is also the one home of the `(m, n)` binding rule (`Sched.placed` /
`Sched.tile_of`). The path codec spells `map` / `fold` / `a` / `b` segments off the derived readings — `PLACE@a`'s
golden rows depend on it. A sliced axis's window is the one `Axis.window`.

Since step 7 the wire forms are SITE-LOCAL: the worker inventory is spelled once in `WORK`
(`w<M>x<N>[+p<np>]` / `t<N>[x<M>]`, the producer band riding `+p`), and `TILE` / `REDUCE` values shed their worker
tokens, so the stamped row IS the stored and golden spelling. The retired embedded-worker spellings raise.

Dispatch reads the role and algebra off the node: `ops.head` reaches it through the projection wrapper, and every
scheduling FACT (`Fold.role`, the reduce `Axis`, the operand edges) is a stored param on what it returns — so no
scheduling decision has to synthesize a nest. `reduce_loop` and `Fold.lower` are for callers that consume a body, and
both flatten any node back to the same loop nest. There is no stored `Monoid` / `Semiring` kind.
