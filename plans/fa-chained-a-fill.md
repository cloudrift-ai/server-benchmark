# FA restoration — the chained A fill

Continues the FA-through-the-codec campaign (Stage 1 N-channel pairing, Stage 2 TWISTED computed-A bind, Stage 3's
lowering half). This plan covers what is left after the COMPOSED STEP landed: the realization that makes the fused
single-kernel form worth deploying.

## What landed with this plan's first commit

Recognition of the single-kernel form is complete, end to end from a real SDPA trace:

- **The composed step reads.** `lowering/tile/_fromloop._hoist_step_nodes` lifts a nested reduce `Loop` in a step's
  prefix to an operand EDGE (recursively through the same parser); `_same_program` compares role-blind, because the
  `AxisRole` is a derived read the raw pre-annotation body does not carry. `Fold._twisted_derived_step` PLACES its
  inline-node edges at first use instead of prepending, so a loop-invariant prologue ahead of the producer survives
  the byte-identity gate.
- **The pairing matches composed passes.** `_softmax._score_cone` accepts one SOURCE that may be a producer node;
  `_cone_canon` α-renames every bound name and each nested loop's iteration var in walk order, so two
  separately-traced copies of one score compare equal. The fused stream carries ONE copy of the producer.
- **The cone binds over a computed score.** `_atomize.map_cone` nodifies a producer the cone reads;
  `make_cone` hangs it off the cone's OPERANDS (never its body) and the K seam treats a stmt reading a k-varying
  producer as k-varying; `bind_prologue_contraction`'s bare-statistic gate now refuses only the identity-lift
  (split-K) composition.
- **Split-K reindexes it.** `_schedule._sliced_edge` σ-rewrites the cone's k-varying producer edges — without it every
  partition recomputed partition 0's scores (wrong results, not slow ones).

Verified: with `loop/fusion/_merge`'s reduce-in-reduce and multi-statistic refusals lifted, SDPA lowers to ONE kernel
and matches torch at `(1,1,32,16)` and `(1,2,64,16)` f16 on a 5090 (`D = 128` stops at a PARTIAL merge — see the
blowup bound below — and also matches). `scripts/digest_kernels.py` is byte-identical on all 30 cases (the corpus does
not reach the new arm), and `tests/` is green but for the quant e2e failure that reproduces on `main`.

## Why fusion still refuses

`_merge._nests_reduce` stays as it was, and the criterion is REVERSIBILITY. `Q·Kᵀ` has two consumers in the merged
cell (the streaming statistic and the weight cone); splicing duplicates it at each demand site and no `PLACE` cut puts
it back — a cut fragment serves ONE consumer, so cutting both seams mints two score kernels, never the one shared
producer the unmerged graph had. Measured on the 5090 at `(1,8,512,128)` f16 with both refusals lifted: 1360 µs — a
1352 µs statistic kernel whose score is a scalar dot per element, plus a 7.7 µs PV kernel — against 21.6 µs for the
two-kernel graph the refusal preserves. Lift the refusal only when the fill below makes the fused form win.

## The OTHER refusal: `_total_work`'s blowup bound

`_merge`'s cost metric counts arithmetic leaves times their enclosing iteration product, and in LOOP IR the merged
score sits inside the output-column sweep — `for n { for kv { for d } } }` — so it counts `D` times over. The realized
fill computes it once per `(m, kv)` slab cell and shares it across the column tile, but the pass cannot see that.
Measured: `D = 16` merges (ratio at the `_BLOWUP_FACTOR = 8` bound), `D = 128` is refused outright. Any plan that lifts
`_nests_reduce` has to answer this too — either the metric learns that a k-invariant-in-`n` producer does not replay
per column, or the merge is offered at the seam rather than judged by leaf count.

## LANDED: the chained fill and the chained statistic

Both halves of the composed score now reach the tensor core, through `_atom_ops` on the score node — no second mma
emitter, no attention vocabulary:

- **`_score_block`** — one block of the composed score as C fragments, built from the same atom strategy every tiled
  contraction dispatches through, namespaced by `_AtomOps.frag_ns` so a nested emission never shadows the enclosing
  drain's accumulators.
- **`chain_a_fill`** — the A slab from that block, the cone folded into its fragment store (`RegStore` + `RegEpilogue`,
  statistics read at each element's own row). NONE-swizzle: a fragment store applies no address XOR.
- **`chain_stat_fill`** — the per-row statistic swept at fragment residence: `FragmentRowReduce` off the score
  fragments, then `exp_merge` at a BLOCK singleton `(rowmax, rowsum)` — the same generator, the same certificate.
- Recognition stores BOTH score sites bound as contractions (`_atomize.bound_producer`), which is what lets the
  schedule see a contraction instead of a scalar fold.

Measured on a 5090, f16 `(1, 8, 512, D)`, cold greedy, fused arm vs the two-kernel graph vs torch:

| shape | torch | two-kernel | fused, per-cell | fused, chained |
| --- | --- | --- | --- | --- |
| S=512 D=16 | 8 µs | 38.3 µs | 219 µs | **14.0 µs** |
| S=512 D=32 | 8 µs | 24.9 µs | — | **13.8 µs** |
| S=512 D=64 | 10 µs | 28.7 µs | — | 29.7 µs |
| S=512 D=128 | 19 µs | 21.6 µs | — | 69.2 µs |
| S=1024 D=128 | 45 µs | 72.1 µs | — | 370.6 µs |
| S=2048 D=128 | 145 µs | 179.5 µs | — | 1039.3 µs |

So the chained fused form WINS at small head dims (2.7× / 1.8× at D ≤ 32) and loses badly as either D or S grows. Two
distinct causes, and the second is the bigger one:

1. It computes `Q·Kᵀ` TWICE (statistic, then weight) — a fixed 1.5× flop penalty that grows in absolute terms with D.
2. **The nested score is gmem-direct.** `_score_block` passes `stage=None`, so both the statistic sweep and the weight
   fill read Q and K straight from gmem with no slab and no reuse: per query tile the statistic re-reads the WHOLE KV
   row of K. At S=2048 D=128 that is 512 KB of K per query tile × 1024 query tiles ≈ 512 MB of reads, against the
   two-kernel `Q·Kᵀ` kernel's staged, tiled, TMA-fed slabs. That is why the gap WIDENS with S rather than closing.

That is why fusion still refuses: real models are D = 64 / 128 at long sequence.

## What makes it win at every D: the SINGLE-PASS sweep

FA-2 computes the score once. Two things stand between here and that, and the machinery for both now exists:

1. **Hoist the `1/d` out of the cone.** `Σ_kv (exp(s−m)/d)·V = (1/d)·Σ_kv exp(s−m)·V` — the invariant-factor split
   Stage 1 already built (`_softmax.split_invariant_factors`), applied to the cone rather than to a sibling fold. The
   weight the A slab carries must not name a `d` that is not final yet.
2. **Carry the statistic in the weight loop.** Per KV block: score fragments → rowmax → merge into the running
   `(m, d)` → `P = exp(s − M)` → the A slab → rescale the OUTER C fragments by `α = exp(m_old − M)`
   (`FragmentApply(in_place=True)` over `_c{i}_{j}` — they are in scope inside the K-loop) → the PV drain. The final
   `O / d` lands in the output store's epilogue.

That removes the second `Q·Kᵀ` and the statistic prologue entirely. Then re-measure the table above and, if the fused
arm wins at D = 64 / 128, lift `_merge._nests_reduce` (and the blowup bound below) so evidence prices it.

## Still open, in the order that pays

- **Stage the nested score's operands.** The biggest lever, and the reason the gap widens with S: `_score_block` runs
  gmem-direct. Its K needs a slab (and its Q a fragment held across the KV sweep — today Q is re-loaded per KV chunk
  per D-chunk). Note the nesting: `_staged` inside a fill puts a `__syncthreads()` inside the outer K-loop body, which
  is legal only because every thread reaches the fill — check that before relying on it.
- **The single-pass sweep** (above) — removes the second `Q·Kᵀ` and, with it, the second pass over K.
- **A schedule site for the score.** `1a47f4fc` made an inline node carry no slice family (`Site.inline`), which was
  right while the fill could only evaluate it per cell. The chained fill derives the score's tile from the enclosing
  one instead, so nothing is pinnable there yet: its `bk`, its own staging, and the warp split of its rows are all
  implied. A `TILE` / `STAGE` slice on the producer edge is what makes them searchable.
- **`units_n > 1` duplicates the fill.** Warps that differ only in their N unit own the same A-slab rows and each
  compute them — same values, wasted work. Guard it, or require `units_n == 1` in legality.
- **`_schedule._row_stream_tiles` / `_warp_stream_place`** are still live orphans from the flash era (the `(score,
  P@V)` streaming pair's materialization half and its grid). The single-pass sweep is where they come back.

### Rejected, do not redo

`Fold.hoisted()` (the inverse of `demoted()`) — `_derived_expect_fold` hardcodes `Axis("pj", Dim(1))` so
`legal.warp_k_step` can never pass, `_has_computed_operand` closes the scalar-tile tier, and `_schedule._keeps_children`
prunes the site. Measured and reverted once already.

### Open question

`_derived_expect_fold`'s `pj` singleton is what makes the twisted fold's own PV a rank-1 update. Re-axing it to the KV
block is step 3's other half; the flash era did it with `Fold.with_axis`, which no longer exists.
