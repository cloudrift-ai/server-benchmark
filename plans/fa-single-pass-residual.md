# FA restoration — what is left after the staged single-pass sweep

The FA-through-the-codec campaign (Stage 1 N-channel pairing, Stage 2 TWISTED computed-A bind, Stage 3's lowering
half, the chained A fill + chained statistic, the SINGLE-PASS sweep, and now the nested score's own STAGED operands)
has taken the fused single-kernel form past the two-kernel graph at every `D ≤ 64`, and the slab swizzles have taken
it past torch at `D = 128` too on an A100 at `S ≤ 512`. Loop fusion still refuses the merge — pricing fused against
cut needs the CUT arm's own schedule evidence, which nothing has measured yet. This plan is the residual.

Measurements, gates, and the mechanism live in `tests/compiler/e2e/test_attention_coverage.py`'s module docstring and
`emmy/compiler/pipeline/passes/lowering/kernel/ARCHITECTURE.md` — read those first; nothing here repeats them.

## The wall at `D = 128` was the SHARED path, and it is now cleared

The register reading was wrong. On an A100 the `D = 128` kernel was **shared-memory bank-conflict bound** — 78% of its
shared wavefronts were conflict replays, L1/TEX at 80% against 7% compute — and registers never bound it (ncu's block
limit from registers was 2, with negligible local traffic). Three slabs drained plain row-major, and the swizzle the
fourth used under-spread a row wider than its atom. All three are fixed; the measurements and the mechanism are in the
docstring and `ARCHITECTURE.md`. The A100 fused arm now beats torch at `S ≤ 512` and is at parity at `S = 1024`.

What is left, in the order that pays:

1. **The two long shapes.** `S = 1024` is at parity and `S = 2048` is ~1.35x behind, and the residual grows with `S` —
   so it is not a fixed overhead. Profile the long shape on its own before assuming it is the same term as the short
   one; L1/TEX is down to 61% and compute up to 24%, so neither is obviously the bound any more.
2. **The P round trip itself.** The weight tile still goes C fragments → smem → `ldmatrix` → A fragments, costing
   two `__syncthreads` and a slab per chunk. FA-2 recasts the C fragments into A fragments IN REGISTERS: for
   `m16n8k16`, C tiles `j = 2k, 2k+1` hold exactly the A fragment for k-block `k`, so the recast is an f32→f16 pack
   with no lane shuffle. That frees the slab (occupancy) and both barriers. Bigger than anything left below.
3. **`units_n > 1`** still makes every `n` warp recompute the whole score for the same rows (323 cold vs 190.8 pinned
   at S=2048 D=128 on the 5090). Sharing one warp band's statistic with the rest needs the ψ-rescale published through
   an smem row and applied AFTER the transport's barrier — the `LeadSegment` seam now exists, so this is a second
   segment away.
4. **Reuse ONE slab for the score's keys and the drain's values** where the two operands are the same buffer. They are
   not in attention (K and V differ), so this is a no-op there — noted only so it is not re-derived.

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
