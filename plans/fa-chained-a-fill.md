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

## The work: score mma → cone epilogue on the C fragments → A slab

The fill EVALUATES the cone's k-varying producer per slab cell (`_atom._sync_operands.a_value` splices the lowered
loop into each cell), so the score is a scalar dot per cell — and it is computed twice, once in the row statistic and
once in the weight. Both halves are the same defect: a score contraction realized at scalar residence.

1. **Give the producer edge a schedule site.** `1a47f4fc` made an inline node carry no slice family (`Site.inline`) —
   correct while the fill can only evaluate it per cell, wrong once the fill can TILE it. The edge needs a `TILE`
   slice (its own `(m, kv)` geometry over the slab) plus a `STAGE` for its own Q / K operands.
2. **Emit the chain.** In the `smem` fill: `ldmatrix` Q once per query tile, mma the score into C fragments over the
   head-dim, apply the cone body on those fragments, store into the A slab the existing `_MmaOps.staged_drain` reads.
   The Kernel IR for the fragment half survived the flash deletion and is still dead code kept for this:
   `FragmentApply`, `FragmentRowReduce`, `FragmentMask`, `FragmentBiasAdd`, `FragmentRepack`.
3. **Block the statistic.** The row statistic cannot come from the fragments while it spans the whole KV row, so the
   twisted fold blocks: `exp_merge` at a BLOCK singleton `(m_b, d_b, o_b)` — the SAME generator, `_certify` still
   passes — with the per-block quantities reduced off the score fragments (`FragmentRowReduce`) and the outer step
   rescaling the O fragments by α. That is FA-2, and `tests/compiler/e2e/test_attention_coverage.py`'s hand-written
   `fa2` kernel is its executable spec (lane layouts, the C→A handoff, the 4-lane butterfly).
4. **Schedule + legality.** `_schedule._row_stream_tiles` and `_warp_stream_place` are live orphans from the flash era
   — the materialization half of the `(score, P@V)` streaming pair and its grid (query axis shrunk to its CTA-block
   count, value axis folded into the P@V fragment, KV walked serially per CTA). The enumeration half (`_stream_tiles`)
   was deleted and has to come back. Legality: the score tile must fit in C fragments.
5. **Then** lift `_merge._nests_reduce`'s refusal and let evidence price fused against cut.

### Rejected, do not redo

`Fold.hoisted()` (the inverse of `demoted()`) — `_derived_expect_fold` hardcodes `Axis("pj", Dim(1))` so
`legal.warp_k_step` can never pass, `_has_computed_operand` closes the scalar-tile tier, and `_schedule._keeps_children`
prunes the site. Measured and reverted once already.

### Open question

`_derived_expect_fold`'s `pj` singleton is what makes the twisted fold's own PV a rank-1 update. Re-axing it to the KV
block is step 3's other half; the flash era did it with `Fold.with_axis`, which no longer exists.
