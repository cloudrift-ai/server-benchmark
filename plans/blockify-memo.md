# Blocking a reduce stream: what the scheduler now owns, and the one gap left

Working note for `feature/blockify-twisted-carrier`; never reference it from durable docs or code. Rewritten
2026-09-04 after the width became a schedule decision. The earlier drafts' subject — three block loops the sibling
merge could not reach — is settled and no longer the point; see "What was dropped, and why".

## Where the tiers actually stand

Measured on an RTX 5090, sm_120, through the in-process pipeline. Three shapes: SDPA f16 `(1,4,128,32)`, a
`256×1024×512` f16 linear, a `64×4096` f32 row sum.

| | offered | assigned | emitted |
| --- | --- | --- | --- |
| FA `P·V` (blocked twisted channel) | 232 warp choices | yes, under a pin | **no `mma.sync`** |
| FA score (blocked pivot's cone) | 2282 warp choices | yes, under a pin | **no `mma.sync`** |
| matmul, blocked | warp choices at the inner site | yes | **no `mma.sync`** |
| matmul, unblocked | — | yes | `mma.sync` ×4, `ldmatrix` ×24 |
| reduction, blocked | — | `REDUCE@reduce=coop`, and `g<n>` splits into partial + finalize | as assigned |

Accuracy passes on all three shapes at every width tried (`emmy run`, 9/9: SDPA b64/b32, matmul b64/b128, reduce
b512/b128, plus the three declined arms).

So the schedule half is done and the realization half is not. Everything below the table is about that split.

## What the width is

One `Var` per blocked axis, named `__blk_<axis>`, in three places: the outer axis's ceil trip count, the inner
axis's extent, and the σ that reconstructs the absolute coordinate. `block_tree` mints it, `block_widths` reads the
domain off the blocked tree, and one arm of `025_block`'s fork substitutes it. The symbol never reaches Kernel IR.

Two things had to be true for that to work at all, and neither was obvious:

- **A σ image binds its free names.** `k_o·blk + k_i` closes over three names, so `Lambda.closing` bound the width
  as a coordinate param of every edge it reindexed, and `Fold.lower` then asked the axis table for an extent no axis
  has. Substituting a coordinate by a value stops it BEING a coordinate — `_drop_coordinates` is that half of the
  binding, and it is why `bind_widths` rebuilds lambdas rather than only rewriting expressions.
- **`Dim` does not simplify what it is handed.** `Dim(expr)` stores the expression; only the arithmetic operators
  fold. So the substituted ceil count `(1024 + 63) / 64` stayed a `BinaryExpr` and read back as NON-static, which
  silently withheld split-K, the coop bands and raster from every blocked outer fold. `simplify_extent` (the module
  helper, now public) is called at the substitution. Worth remembering: this failure mode is invisible — nothing
  errors, the schedule space is just quietly smaller.

The old prototype's `EMMY_BLOCK=64` int read is gone; `BLOCK` is a declared knob, so `EMMY_BLOCK=b64` pins it and
`EMMY_BLOCK=` pins the declined arm, both authoritatively.

## Why the emitted kernel has no mma, and it is not the schedule

`_bind` (`lowering/kernel/_factor.py`) binds ONE node — the kernel's root leaf — and lowers everything under it
through `op.lower()`, i.e. as plain nested loops. A contraction reaches its tile only when it IS that node. Both the
blocked matmul (`Fold[k_o] add (Fold[k_i] contraction)`) and blocked attention (a twisted carrier over three inner
folds) put the contraction one level DOWN, as an operand, so the accepted `TILE=mma_…` is recorded on the row and
then ignored by the emitter. Verified by pinning an enumerated row: the `TileOp` reaches `lowering/kernel` with
`place.is_mapped`, `materialization.tiles` populated and warp tiles at two sites, and the emitted CUDA is
`_gid`-indexed scalar code.

`_atom.scheduled_fold_contraction` is the mechanism meant for exactly this — "a tiled contraction result consumed by
an `Accum` into one of the enclosing carrier names" — but it scans `fold.lift.body` for a nested `Fold`, and a
child term has lived on `Fold.operands` since the node collapse; its `b_trans` still does `isinstance(self.b, Load)`
against what is now always a `Fold`. It cannot fire. Reviving it against the operand reading is the next piece of
work and it is not small: the O fragments have to stay in registers across blocks while the carrier's α/β rescale
is applied to them each block.

**Do not re-diagnose this from the schedule side.** The domains, the placement and the row are all correct; three
separate probes said so.

## What was dropped, and why

`025_block` no longer round-trips the blocked kernel through a `LoopOp`. That splice existed to reach
`normalize_body`'s sibling-reduce merge, which collapsed attention's three block loops to two. The merge fuses the
`P·V` channel with the denominator into one two-state fold, and `as_contraction` requires every lift result to be a
product against `operands[0]` — the denominator's is a `copy` of the shared weight. So the merge and the bilinear
reading are mutually exclusive, and the tier is worth more than the loop. Measured both ways: with the round trip
`map.1/twist.2` reads as a `reduce`; without it, as an `inner`.

The three defects that splice exposed are all still fixed and still wanted — `Const` as a binding site in
`loop/ir._validate`, `hoist_loop_invariants` pinning a consumer under the loop that defines what it reads, and
`scan_from_loop` reading back a `base`-`Accum`. They fire on the unblocked SDPA tree too.

## Open

- **The realization above.** Everything else here is downstream of it.
- **`Q·K` runs 3×** — once in the pivot's pass and once inside each channel's weight cone. The cones bind different
  coordinates, so they are different values; sharing them needs the score to be a tile over the block axis that both
  consumers read, which is a schedule fact, not a normalization.
- **Cold compile cost.** A planar stream's fork adds four or five arms to every reducible kernel. The unblocked
  matmul went 15 s → 46 s through the greedy walk. The declined value leads for planar streams so the descent order
  is unchanged, but the frontier is wider.
- **Not run:** the realization corpus, `make test-goldens`, `scripts/digest_kernels.py`, the perf lane. `BLOCK` is a
  new key on every row, so recorded rows are expected to need it before this goes near main.
