# FA restoration — what is left after the staged single-pass sweep

The FA-through-the-codec campaign (Stage 1 N-channel pairing, Stage 2 TWISTED computed-A bind, Stage 3's lowering
half, the chained A fill + chained statistic, the SINGLE-PASS sweep, and now the nested score's own STAGED operands)
has taken the fused single-kernel form past the two-kernel graph at every `D ≤ 64` and past torch at every `D` at
`S = 512`. Loop fusion still refuses the merge, because `D = 128` still loses. This plan is the residual.

Measurements, gates, and the mechanism live in `tests/compiler/e2e/test_attention_coverage.py`'s module docstring and
`emmy/compiler/pipeline/passes/lowering/kernel/ARCHITECTURE.md` — read those first; nothing here repeats them.

## The wall at `D = 128` is now the TILE, not the memory path

Staging the score took L1/TEX from 96% to 89% and the gmem fragment loaders out of the kernel entirely. What is left
at `(1, 8, 2048, 128)` f16 on a 5090 is the tile's own footprint: **254 registers per thread and four slabs
(A, V, score-K, score-Q) cap the kernel at ONE block per SM**, with compute utilization at 16%.

Levers, in the order that pays:

1. **Hold the score's C fragments in fewer registers.** At `bk = 128` the score block alone is `bk/atom_n = 16`
   C-fragments (64 registers) on top of the output tile's 16 (another 64), and `_contract_kloop` reads ALL register-col
   B fragments before contracting, which is 32 more. Halving `bk` halves the first two but measured WORSE (209.5 vs
   190.8 at `f1x16/k4` vs `k8`) — the chunk count and its barriers cost more than the registers save. The real fix is
   a drain that interleaves the reads with the mma instead of hoisting them all.
2. **Reuse ONE slab for the score's keys and the drain's values** where the two operands are the same buffer. They are
   not in attention (K and V differ), so this is a no-op there — noted only so it is not re-derived.
3. **`units_n > 1`** still makes every `n` warp recompute the whole score for the same rows (323 cold vs 190.8 pinned
   at S=2048 D=128). Sharing one warp band's statistic with the rest needs the ψ-rescale published through an smem row
   and applied AFTER the transport's barrier — the `LeadSegment` seam now exists, so this is a second segment away.

## Schedule forks the cold pick lands on badly

Both are enumerated, both are evidence's call:

- **split-K** on the contraction forces the two-pass pair (the single pass needs the sweep and the contraction to
  cover the same keys). 25.9 vs 14.3 µs at S=512 D=64.
- **`units_n > 1`**, above.

Seeding a golden per shape is the cheap answer to both; a featurization that sees "the A cone composes a score over
this axis" is the durable one.

## Sharp edge worth fixing separately

A `TILE` / `WORK` pin whose `Stage` cannot resolve (now including the score's reserved slabs) drops the tile slice
SILENTLY and the term falls to the per-cell scalar tier — at attention shapes that is minutes per launch, and it
surfaces as a bench watchdog "hang" rather than a decline. `EMMY_STAGE` pins decline loudly; the other two should too.

## Rejected, do not redo

- `Fold.hoisted()` (the inverse of `demoted()`) — `_derived_expect_fold` hardcodes `Axis("pj", Dim(1))` so
  `legal.warp_k_step` can never pass, `_has_computed_operand` closes the scalar-tile tier, and
  `_schedule._keeps_children` prunes the site. Measured and reverted once already.
- **Normalizing the running output** (rescaling by `d_old/d_new` so the weight can keep naming a final denominator)
  — the invariant-factor split is strictly cheaper: it makes the ψ-rescale the carrier's own `exp(pivot − pivot')`
  with no denominator ratio, and the factors multiply back onto the output fragments once, after the loop.
- **Riding the enclosing `SyncTransport`'s `copy_operands` for the score's keys** — the compute fill runs BEFORE the
  transport's wait, so a slab filled in the same call is not yet readable. The `LeadSegment` split is what makes the
  live ranges nest instead.
