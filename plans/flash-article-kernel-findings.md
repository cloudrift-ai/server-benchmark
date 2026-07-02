# Warp-flash K/V staging: sweep findings + Part-2 article handoff (RTX 5090, 2026-07-02)

Implementation landed on `feature/flash-kv-staging` (K/V cp.async staging on the warp-flash stream, `STAGE@<kv>`,
plus the staged transposed-B `ldmatrix.x2` drain and the handoff-aware smem budget clamp). This file records the
manual pinned-knob sweep and everything the article needs; delete once the article ships.

## The article's pinned config ($KNOBS)

Winner on the perf shape `(1, 8, 4096, 64)` f16, non-causal, RTX 5090 (sm_120), `emmy run --bench` (100 iters):

```bash
EMMY_TILE="a:mma_m16n8k16_f16/w2x1/f1x16/k4"   # 2 warps/CTA, 128-key streaming block
EMMY_STAGE="d1/cp"                              # single-buffer cp.async K/V slabs
```

## The full sweep table (µs; geometry × stage; eager torch SDPA flash = 205–208 µs)

| geom (w,nt) | gmem | d1/cp | d2/cp/ring | d3/cp/ring |
|-------------|------|-------|------------|------------|
| w1 n2       | 800  | 730   | 751        | 771        |
| w1 n4       | 649  | 582   | 599        | 790        |
| w1 n8       | 742  | 554   | 822        | 1417       |
| w1 n16      | 990  | 751   | 1172       | (clamps→d2)|
| w2 n2       | 867  | 697   | 695        | 689        |
| w2 n4       | 687  | 572   | 587        | 618        |
| w2 n8       | 700  | 517   | 559        | 755        |
| **w2 n16**  | 612  | **511** | 639      | (clamps→d2)|
| w4 n2       | 990  | 791   | 779        | 782        |
| w4 n4       | 806  | 622   | 630        | 647        |
| w4 n8       | 751  | 579   | 579        | 667        |
| w4 n16      | 701  | 529   | 569        | (clamps→d2)|

## The per-move latency ladder (perf shape)

| rung | config | µs | vs previous |
|------|--------|-----|-------------|
| Move 1 — scalar streaming chain (`EMMY_TILE=""`) | FA-2 scalar form | 24943 | — |
| Move 2 — mma, gmem-direct (best geom w2n16) | fragment loads from gmem | 612 | 41× |
| Move 4 — + K/V smem staging (`d1/cp`) | cp.async fill, ldmatrix drain | **511** | 1.20× |
| Move 5 — + prefetch ring (`d2/cp/ring`) | 639 | **regresses** (see below) |
| torch eager SDPA (FA-2 backend) | reference | 206 | emmy best = 0.40× |

Two honest findings the article should keep:

1. **The ring doesn't pay on this kernel yet.** `d1` beats `d2+/ring` at almost every geometry: the ring doubles
   the slab footprint, and at 32–64 threads/CTA the occupancy loss outweighs the copy/math overlap (the d2 config
   runs at 2–8% occupancy). The Move-5 machinery is real and lands (the listing shows the primed ring + clamped
   prefetch), but its payoff waits on the occupancy work below — say so rather than claiming a speedup.
2. **NCU attribution of the 2.5× gap to FA-2** (winner, 100-iter kernel: 8.3% occ, 13.7% SM, 2.4% FMA pipe):
   **132 M shared-memory load bank conflicts + 14.7 M store conflicts** vs FA-2's ~0 — the plain row-major K/V
   slabs and the `flash_pv_smem` C→A handoff have 128 B row strides, so every ldmatrix's 8 row reads land on one
   bank group; LSU instructions are 4× FA-2's (17.4 M vs 4.4 M — the handoff round-trip + the per-step Q reload).
   This is exactly the bank-conflict problem the article's swizzle/padding sections teach — Move 4 lands and
   immediately motivates them.

## Follow-ups (measured, not fixed — in expected-payoff order)

1. Swizzle/pad the K/V slabs + the C→A handoff slab (kills the 132 M conflicts; PAD_SMEM is matmul-tier only today).
2. Register-shuffle C→A handoff (removes the per-step `__syncthreads` + smem round-trip; also unblocks ring payoff).
3. Hoist the loop-invariant Q fragments out of the stream (bk gmem-direct A loads re-issued every KV block).
4. Causal tile-skip (unchanged follow-up; the example is non-causal).

## Accuracy table (listing shape (1,4,128,64), vs an fp64 torch reference — the Numerics placeholder)

| kernel | max abs err | mean abs err |
|--------|-------------|--------------|
| scalar fp32 flash | 4.352e-07 | 3.951e-08 |
| f16-mma flash (gmem-direct) | 7.045e-04 | 4.888e-05 |
| f16-mma flash (staged d2/cp/ring) | 7.045e-04 | 4.888e-05 |

The staged and gmem-direct rows are IDENTICAL — bit-identity (staging is a pure perf transform) made visible; the
fp32 carrier keeps the f16-matmul error at input-rounding scale, per Part 1's perturbation bound.

## Article stale-claim fixes (Part 2, cloudrift-landing)

1. "the static-shape and fp32 paths still lower to the Move-1 scalar kernel" — **static is stale** post-#300:
   block-divisible static shapes atomize (only ragged static tails stay scalar). fp32-stays-scalar remains true.
2. The cited test path `tests/compiler/e2e/test_flash_tensorcore_*` is now `tests/compiler/e2e/test_attention_coverage.py`.
3. "gated behind a knob today (`DEPLODOCK_FLASH=1`)" — stale: `PLACE`'s built-in `auto` resolves to fuse, so greedy
   ships the fused flash kernel by default; `PLACE@fold=cut` is the multi-kernel escape.
4. Moves 4–5 can now show real listings: `--ir cuda` with the pins above emits the cp.async fills / commit / wait
   (`d1/cp`) and the primed ring with the clamped prefetch (`d2/cp/ring`); keyed pins ride
   `EMMY_KNOBS="TILE@dd=…,STAGE@kv=…"` or the bare `EMMY_TILE`/`EMMY_STAGE` env vars on a single-kernel graph.
