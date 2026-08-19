# FA restoration — what is left after the single-pass sweep

The FA-through-the-codec campaign (Stage 1 N-channel pairing, Stage 2 TWISTED computed-A bind, Stage 3's lowering
half, the chained A fill + chained statistic, and now the SINGLE-PASS sweep) has taken the fused single-kernel form
past the two-kernel graph at every `D ≤ 64` and past torch at `D ≤ 32`. Loop fusion still refuses the merge, because
`D = 128` still loses. This plan is the residual.

Measurements, gates, and the mechanism live in `tests/compiler/e2e/test_attention_coverage.py`'s module docstring and
`emmy/compiler/pipeline/passes/lowering/kernel/ARCHITECTURE.md` — read those first; nothing here repeats them.

## The one wall: the nested score's operands are gmem-direct

At `(1, 8, 2048, 128)` f16 on a 5090 the fused kernel runs at **96% L1/TEX throughput against 19% compute**, 255
registers per thread, 15% achieved occupancy. Every score block reads Q and K straight from gmem through the
per-lane fragment loaders (`emmy_mma_load_a_gmem` / `emmy_mma_load_b_gmem_trans` — four scalar 2-byte loads per lane
per fragment), and each of the CTA's warps reads the WHOLE K chunk that way, so the chunk crosses L1 once per warp.

Two levers, in the order that pays:

1. **Stage the score's K.** One slab per KV chunk, drained by `ldmatrix` — ~4x fewer memory instructions and the
   chunk crosses L1 once per CTA instead of once per warp. `_score_block` passes `stage=None`; giving it a resolved
   `Stage` routes it through the same `_staged` skeleton, which needs (a) a slab NAMESPACE beside the existing
   fragment one (`_AtomOps.frag_ns`), or the nested `_a_smem` / `_b_smem` collide with the enclosing transport's,
   and (b) the scheduler's smem budget to account for the nested slabs. The barrier a nested `_staged` puts inside
   the outer chunk loop is legal — every thread reaches the fill — but it has never been exercised.
   The alternative (ride the enclosing `SyncTransport`'s `copy_operands`) does NOT work as-is: the compute fill runs
   BEFORE the transport's wait, so a slab filled in the same call is not yet readable.
2. **Hold Q across the KV sweep.** Q is invariant in the chunk loop and is re-read per chunk per D-step. Hoisting its
   fragments out costs `m.reg × D/atom_k` register fragments and removes all Q traffic. It needs the score's own
   K-loop unrolled over D with the A reads lifted — a different loop structure than `_contract_kloop`'s shared spine.

## Schedule forks the cold pick lands on badly

Both are enumerated, both are evidence's call, and both cost most of the fused arm's win when greedy guesses wrong:

- **split-K** on the contraction forces the two-pass pair (the single pass needs the sweep and the contraction to
  cover the same keys). 25.9 vs 14.3 µs at S=512 D=64.
- **`units_n > 1`** makes every `n` warp recompute the whole score for the same rows. 498 vs 273 µs at S=2048 D=128.
  Sharing one warp band's statistic with the rest needs the ψ-rescale published through an smem row and applied
  AFTER the transport's barrier — `SyncTransport` has no hook there today.

Seeding a golden per shape is the cheap answer to both; a featurization that sees "the A cone composes a score over
this axis" is the durable one.

## Rejected, do not redo

- `Fold.hoisted()` (the inverse of `demoted()`) — `_derived_expect_fold` hardcodes `Axis("pj", Dim(1))` so
  `legal.warp_k_step` can never pass, `_has_computed_operand` closes the scalar-tile tier, and
  `_schedule._keeps_children` prunes the site. Measured and reverted once already.
- **Normalizing the running output** (rescaling by `d_old/d_new` so the weight can keep naming a final denominator)
  — the invariant-factor split is strictly cheaper: it makes the ψ-rescale the carrier's own `exp(pivot − pivot')`
  with no denominator ratio, and the factors multiply back onto the output fragments once, after the loop.
