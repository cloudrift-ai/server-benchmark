# Blocking a twisted carrier: where it stands, and the six things left

Working note for `feature/blockify-twisted-carrier`; never reference it from durable docs or code. Rewritten
2026-09-04 after blocking became a canonical form. Earlier drafts described a `BLOCK` knob, a structural fork and a
`025_block` pass — all three are gone, and the reasoning that removed them is the useful part of this note.

## What blocking is now

`TileOp.__post_init__` runs it beside `normalize_fold_tree` (`ir/tile/block.py`). A twisted carrier's reduce axis is
stored split into `k_o × k_i` and the fold re-associated over the two levels, so the channel whose contribution is a
product of two distinct cones reads as a contraction. That is FlashAttention-2's shape, derived: the per-step rescale
coefficient is read out of the stored combine, and the value it multiplies is the fold's own lift result.

**The width is not in the term.** The outer axis walks the stream's own extent with `Axis.step`, each inner binder's
extent IS the width symbol, and the σ that reads the absolute coordinate is plain `k_o + k_i`. `Fold.lower` renders a
strided axis as the `StridedLoop` that already existed for this shape. Everything else follows from that one fact:

- the rewrite is parameter-free and idempotent, so it can be a normalization rather than a decision;
- every block form is the SAME term, so one kernel identity, one pool, one price;
- nothing in a row says "blocked". The width is bound in `materialize_classic` from the `TILE` the row put at the site
  blocking created — a blocked site's inner axis IS its K, so the block is exactly that tile's mma K-step. `bk` ranges
  freely because it DEFINES the block (`_blocked_kstep`, which used to narrow it, is deleted), and a row that took the
  scalar tier everywhere leaves the stream in one trip, which is the unblocked kernel.

**Only a twisted carrier is blocked.** A contraction's block is already spelled by `bk` and a plain reduction's
partition by `REDUCE`, with the cross-CTA split already factoring its axis; splitting either term would restate
another family's decision as a shape. A twisted fold is different in kind — its ⊕ is a rescaling program, so
`as_contraction` refuses at its first gate and NO site inside one is bilinear until the two monoids are separated.

## What is measured

RTX 5090, sm_120, `F.scaled_dot_product_attention` f16 `(1,4,128,32)`.

- The blocked form reads as a twist over a block pivot, a `P·V` channel `as_contraction` accepts, and a denominator.
- `map.1/twist.2/inner` (the `P·V`) is offered warp choices and can be assigned an mma tile.
- `emmy run` accuracy passes.
- 70 CPU-only tests across `tests/compiler/passes/{test_block_forks,test_classic_schedule_domains,test_move_catalog,
  test_placement_routing}.py` and `tests/compiler/ir/tile/{test_path_codec,test_identity_key}.py`. `make lint` clean.
- **`grep -c mma.sync` on the emitted CUDA is still 0.** See step 1.

## The six things left

**1. Make the emitter realize a nested contraction's tile.** The only thing between here and FA on tensor cores.
`_bind` (`lowering/kernel/_factor.py`) binds ONE node — the kernel's root leaf — and lowers everything under it
through `op.lower()`, so a contraction reaches its tile only when it IS that node; blocking puts `P·V` one level down
as an operand. Proven by pinning an enumerated row: the `TileOp` arrives at `lowering/kernel` with `place.is_mapped`,
populated `materialization.tiles` and warp tiles at two sites, and the CUDA is `_gid`-indexed scalar code.
`_atom.scheduled_fold_contraction` was built for exactly this relation but reads the pre-collapse tree — it scans
`fold.lift.body` for a nested `Fold` when child terms have lived on `Fold.operands` since the node collapse, and its
`b_trans` does `isinstance(self.b, Load)` against what is now always a `Fold`. Rework it against the operand reading,
then keep the O fragments in registers across blocks and apply the carrier's α/β rescale to them each block.
**Do not re-diagnose this from the schedule side** — the domains, the placement and the row were each checked.

**2. Teach the partition consumers to read `Axis.step`.** A blocked stream currently folds serially, by a guard in
`_reduction_domain`. It is there because the coop band, the serial remainder and the cross-CTA width all size
themselves against the axis extent, and a strided axis's trip count is `ceil(extent / step)` — without the guard a
blocked stream under `REDUCE=coop` hands every lane the wrong share, and `emmy run` was numerically wrong
(`mean_diff` 0.127, `max_diff` under the printed tolerance, so it does NOT surface as an obvious blow-up). Lifting it
gives the outer carrier back `coop` / `r<n>` / split-K.

**3. Resolve the symbolic K at the blocked site during projection.** Staging sizes shared memory against `Var(blk)`,
so its refusals there are conservative and the site only gets direct transport — which caps perf even once step 1
lands. The width is knowable locally: it is the mma K-step of the tile being considered at that same site, so
`_local_support` can resolve it before calling `_resolve_stage` / `_plan_node_refusal`.

**4. Re-record what the canonical form invalidated.** Blocking changed the stored form of every twisted kernel, so
their golden rows go stale: `make test-goldens`, the realization corpus, `scripts/digest_kernels.py`. NONE of these
has been run on this branch at all — that is standing verification debt, not work step 3 creates.

**5. Measure.** Only after 1–3 does "FA reaches the mma tier" mean anything about performance. Bench the blocked arm
against the two-kernel path and torch, per shape.

**6. Then the score.** `Q·K` still runs 3× — the pivot's pass and each channel's weight cone recompute it. The cones
bind different coordinates, so they are different values; sharing them needs the score to be a tile over the block
axis that both consumers read. That is the chained-score realization, a schedule fact rather than a normalization.

## Two things that cost a round each; keep them

- **`Dim(expr)` does not simplify.** Only the arithmetic operators fold. A substituted ceil count therefore read back
  as NON-static and silently withheld split-K, the coop bands and raster from the stream — nothing errors, the
  schedule space is just quietly smaller. `simplify_extent` (dim.py's own helper, now public) is the fix, and the
  failure mode is invisible, so suspect it whenever a derived extent stops being enumerable.
- **A `Lambda` may not read a name it does not bind**, and a σ image binds its free names. That pair is why the width
  could not simply stay a `Var` inside the term: `Lambda.closing` bound it as a coordinate param of every reindexed
  edge, and `Fold.lower` then asked the kernel's axis table for an extent no axis has. Dropping the param is refused
  by the `Lambda` constructor; registering an axis for it makes `lower` open a loop over it. The way out was to take
  the width out of the index arithmetic entirely — which is what the strided outer axis does.

## What the removals were about

- **A `BLOCK` codec family is unnecessary.** A blocked site's inner axis is its K, so the width is exactly the `k<bk>`
  half of that site's `TILE`. Two spellings of one quantity is what the family would have been.
- **Blocking must not create options.** It is not a decision — it does not change what the kernel computes — so
  enumerating one problem per width made every form a different kernel and split the pool. Binding at materialization
  is what removes both.
- **Blocking is not Fold-local**, despite reading only the fold's own algebra: a `Fold` names its axis but the extent
  lives in the kernel's axis table, and the split needs it. `TileOp` normalization is the nearest home that has both.
- `plans/fa-single-pass-residual.md` was deleted earlier on this branch: it planned against `_softmax.py`, `_merge.py`,
  `chain_stream_fill` and `split_invariant_factors`, none of which exist in the tree any more.
