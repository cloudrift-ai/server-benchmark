# Warp-flash K/V staging + slab padding: findings + Part-2 article handoff (RTX 5090, 2026-07-02)

Implementation landed on `feature/flash-kv-staging` (PR #302): K/V cp.async staging on the warp-flash stream
(`STAGE@<kv>`), the staged transposed-B `ldmatrix.x2` drain, the handoff-aware smem budget clamp, and the
**+16 B slab-row padding** on the cp.async K/V slabs + the `flash_pv_smem` C→A handoff (the cp.async-path
counterpart of the TMA slab swizzle — a near-strict win applied intrinsically, not a fork). This file records
the manual pinned-knob sweeps and everything the article needs; delete once the article ships.

## The article's pinned config ($KNOBS)

Winner on the perf shape `(1, 8, 4096, 64)` f16, non-causal, RTX 5090 (sm_120), `emmy run --bench` (100 iters):

```bash
EMMY_TILE="a:mma_m16n8k16_f16/w4x1/f2x8/k4"    # 4 warps/CTA, 2 query tiles/warp (reg_m), 64-key block
EMMY_STAGE="d2/tma/ring"                        # 2-slot TMA prefetch ring (rank-4 boxes, hw swizzle)
```

(**206 µs — parity with eager FA-2**; see the ladder below.)

## The padded sweep (µs; padding on — the shipped configuration; eager torch SDPA flash = 205–208 µs)

| geom (w,nt) | gmem | d1/cp |
|-------------|------|-------|
| w1 n2       | 794  | 601   |
| w1 n4       | 633  | 578   |
| w1 n8       | 734  | 564   |
| w1 n16      | 964  | 726   |
| w2 n2       | 859  | 514   |
| w2 n4       | 660  | 330   |
| w2 n8       | 635  | 334   |
| w2 n16      | 549  | 411   |
| w4 n2       | 984  | 539   |
| w4 n4       | 775  | 336   |
| **w4 n8**   | 679  | **315** |
| w4 n16      | 628  | 375   |

Ring depths at the winner: `d2/cp/ring` = 529 (w2n16; still regresses vs d1 — the smem doubling costs
occupancy and the in-step handoff `__syncthreads` caps the overlap; the ring's payoff waits on the
register-shuffle handoff).

## Pre-padding sweep (µs; the unpadded 48-config grid — the article's "Move 4 before the padding fix" data)

| geom (w,nt) | gmem | d1/cp | d2/cp/ring | d3/cp/ring |
|-------------|------|-------|------------|------------|
| w1 n2       | 800  | 730   | 751        | 771        |
| w1 n4       | 649  | 582   | 599        | 790        |
| w1 n8       | 742  | 554   | 822        | 1417       |
| w1 n16      | 990  | 751   | 1172       | (clamps→d2)|
| w2 n2       | 867  | 697   | 695        | 689        |
| w2 n4       | 687  | 572   | 587        | 618        |
| w2 n8       | 700  | 517   | 559        | 755        |
| w2 n16      | 612  | 511   | 639        | (clamps→d2)|
| w4 n2       | 990  | 791   | 779        | 782        |
| w4 n4       | 806  | 622   | 630        | 647        |
| w4 n8       | 751  | 579   | 579        | 667        |
| w4 n16      | 701  | 529   | 569        | (clamps→d2)|

## The per-move latency ladder (perf shape — the article's arc)

| rung | config | µs | vs previous |
|------|--------|-----|-------------|
| Move 1 — scalar streaming chain (`EMMY_TILE=""`) | FA-2 scalar form | 24943 | — |
| Move 2 — mma, gmem-direct (unpadded handoff, best geom) | fragment loads from gmem | 612 | 41× |
| Move 4 — + K/V smem staging (`d1/cp`, unpadded) | cp.async fill, ldmatrix drain | 511 | 1.20× |
| + slab padding (the bank-conflict fix; new best geom w4n8) | +16 B rows on slabs + handoff | 315 | 1.62× |
| + C→A register repack (the handoff eliminated) | `FragmentRepack`, no smem round-trip | 271 | 1.16× |
| + Q-fragment hoist (loop-invariant loads out of the stream) | resident `qa` fragments, d1 | 233 | 1.16× |
| Move 5 arrives — the `d2/cp/ring` prefetch NOW wins | copy/math overlap, finally exposed | 222 | 1.05× |
| + `096_pair_ldmatrix_loads` (x2→x4 drain pairing pass) | halves the staged-drain LSU count | 219 | 1.01× |
| + reg_m query tiles (`f2x8` — the register-tile move, on flash) | 2 independent (m,l,O) chains/warp | 212 | 1.03× |
| + TMA transport (`d2/tma/ring` — rank-4 box copies, hw swizzle) | single-thread fills, dense slabs | **206** | 1.03× |
| torch eager SDPA (FA-2 backend) | reference | 206 | emmy best = **1.00× — PARITY** |

The reg_m rung is the article's closing symmetry: the SAME register-tiling move that opened the matmul story
(Move 0 of the GPU-matmul article, 5.2×) closes the flash gap — `f<FM>x<FN>` on the flash geometry grid gives
each warp FM independent softmax chains against shared K/V fragments. The NCU convergence is the money shot:
**232 regs vs FA-2's 255, 11.6% occupancy vs 11.8%, LSU 5.5M vs 4.4M, SM 31.8% vs 33.3%** — the generated
kernel now has FlashAttention-2's structure, arrived at through generic prior-ranked moves.

The Move-5 story is the article's best sequence: the ring lost at every step (occupancy cost, nothing to
overlap) until the repack + Q-hoist thinned the streaming step enough to EXPOSE the copy latency — then the
same `d2/cp/ring` knob that regressed all session flipped to the win. Software pipelining pays only when the
loop body stops being its own latency blanket. NCU at the winner: SM 32.6% vs FA-2's 33.2% — the kernel is
as busy as FlashAttention-2; the residual 1.08× is LSU count (14.1M vs 4.4M: x2-vs-x4 drains) + the `expf` path.

The `d2+` prefetch ring (Move 5's software pipelining) is built and emits correctly (primed ring, clamped
prefetch) but REGRESSES at every geometry — the honest finding: the smem doubling costs occupancy and the
handoff barrier caps the copy/math overlap. Say so; its payoff is gated on the register handoff below.

## NCU attribution — the padding experiment's before/after (winner config, 100-iter kernel)

| counter | unpadded (w2n16 d1) | padded (w4n8 d1) | FA-2 ref |
|---------|--------------------:|-----------------:|---------:|
| smem load conflicts | 132,123,183 | **10,069** | 21,096 |
| smem store conflicts | 14,728,970 | 274,919 | 0 |
| occupancy | 8.3% | 21.0% | 11.8% |
| SM util | 13.7% | 27.1% | 32.8% |
| LSU instructions | 17.4M | 20.6M | 4.4M |
| registers | 199 | 131 | 255 |

The bank-conflict hypothesis is CONFIRMED: +16 B row padding removed 13,000× of load conflicts (now below
FA-2's own count) and bought 1.62× end-to-end. The remaining 1.5× gap to FA-2 tracks the LSU instruction
count (4.7× FA-2's) — the `flash_pv_smem` C→A round-trip and the per-step loop-invariant Q reload.

## The barrier experiment (run, hypothesis REFUTED — the convoy is not the bottleneck)

The handoff `__syncthreads` was warp-scoped (`Sync(warp=True)`), and it moved nothing: d1 315→314,
d2 322→320, gmem-direct 679→673 (all noise). Post-padding, the staged loop's transport barriers re-converge
the warps anyway — the gap was the LSU **volume**, not the barrier around it. (The whole handoff — slab,
sync, and all — was then retired by the register repack below.)

## The register repack (DONE — the C→A handoff eliminated)

The m16n8k16 C fragment is lane-map ALIGNED with the A fragment's k-halves, so P converts straight into its
P@V A-operand fragments per lane (`FragmentRepack` → `dpl_c_to_a_{f16,bf16}`, four `cvt.rn.*x2.f32` packs —
same round-to-nearest-even as the retired `RegStore`, so bit-identical). The `flash_pv_smem` slab, its sync,
and its pad are gone; `AtomKind.c_to_a_repack` gates the tier at schedule time. Measured (perf shape):

| config | with smem handoff | with register repack |
|--------|------------------:|---------------------:|
| w4n8 d1/cp | 315 µs | **271 µs** |
| w4n8 gmem-direct | 673 µs | 586 µs |
| w4n16 d1/cp | 375 µs | 283 µs |
| w4n8 d2/cp/ring | 320 µs | 303 µs (still loses to d1) |

NCU (w4n8 d1): smem now SPOTLESS — 109 load conflicts, 0 store conflicts; LSU 20.6M → 18.0M; occupancy
20.6%, SM 27.8% (FA-2: 33.0%). emmy best = **271 µs vs FA-2 206 µs = 1.32×** (was 2.5× at the start of the
session). The remaining gap is pure fragment-load volume: the per-step loop-invariant Q reload (16 gmem
loads/warp/step) and the per-B-fragment `x2` ldmatrix drains (an `x4` loads two B fragments at once), plus
the `expf`-vs-`exp2f` ALU path.

## TMA + WSPEC (DONE — parity reached; WSPEC measured and declined)

The "2-D descriptor" gate was plumbing: `encode_tiled` does rank 1–5 and `TmaLoad` renders
`cp_async_bulk_tensor_<rank>d`, so the batched K/V encode as rank-4 boxes `(1, 1, bn, head_dim)` with the
load's own batch/head index exprs as origin coords (GQA's `h // group` included). TMA slabs stay dense
under the hardware swizzle (`pick_swizzle_atom`; the x4-pairing pass now fuses equal-swizzle pairs — the
drain XOR is per-lane address-based). **`w4x1/f2x8/k4` + `d2/tma/ring` = 206 µs — PARITY with FA-2**, and
the NCU profile now beats the reference on memory: LSU **3.31M vs FA-2's 4.36M** (single-thread box copies
replaced the per-thread fill loops), conflicts 4K vs 22K, 233 regs / 11.4% occ.

**WSPEC on flash: built, correct, measured SLOWER** (213 vs 206 at the winner — the fifth honest negative).
The producer-band split is now offered on resolved-TMA flash rows (the matmul legality; the transport's
elected fill thread rides the WRAPPED linear tid — the raw tid would elect a compute thread and never
fill), and it lowers + passes accuracy — but at flash's CTA scale the uniform TMA ring already keeps the
tensor cores fed, so a dedicated producer band only costs occupancy. This matches the article's own
framing: warp specialization is FA-3's Hopper/wgmma story (deep pipelines, big CTAs), not an sm_120 win —
and now that claim is measured, not assumed.

## The warp-private ring (built, measured, REVERTED — the sixth honest negative)

The beyond-FA-2 candidate: per-warp K/V ring slices + private mbarriers, each warp's lane 0 its elected
TMA producer, slab reuse ordered by `__syncwarp` alone — ZERO CTA barriers in the streaming loop, warps
free-running their streams (a `wp` scope flag on the `Stage` codec; ~150 lines across codec/resolver/
`_warp_ring_kloop`/realizer). It lowered correctly and passed accuracy — and measured **291µs vs the
CTA-shared ring's 206** (flat across d1/d2/d3). NCU + arithmetic explain it architecturally:

- flash's K/V are read by EVERY warp — the CTA-shared slab is exactly the reuse that justified Move 4.
  Warp-private slabs replicate the same data `um`×: at um=4 the ×4 smem replication blew the budget so
  every depth clamped to single-buffer (hence the flat 291s), occupancy fell to 8.4%, SM to 22.7%.
- the benefit side was ALREADY known to be zero: the barrier experiment (ec87aa70) measured the CTA
  convoy as costless — so the wp ring's only differential was the negative one.

The move is structurally dominated FOR FLASH (any kernel whose staged operands are CTA-shared). The code
was reverted per the no-dead-moves rule; the finding stays. It would only make sense where warps stream
DISJOINT operand data — e.g. a future within-CTA split-KV — noted for the move catalog, not built.

## Drain reg-depth p2 (built, measured FLAT — kept as a codec-uniformity fix)

The last lever: the flash drains now honor the ``STAGE`` codec's ``p<reg_depth>`` field (the two-slot
ldmatrix ping-pong — step ``s+1``'s B fragments load into the alternate slot while step ``s``'s mmas
consume; load placement only, bit-identical). Measured at the winner: `d2/tma/ring/p2` = **206 vs 207µs**
— flat, the seventh null: ptxas already schedules the straight-line unrolled drain effectively at this
occupancy. Kept (unlike the wp ring) because it is not dominated: it removes a silent pin clamp (a
``p2`` pin on flash now applies instead of degrading), costs ~40 lines, and measured best-equal.

## Side-by-side vs the FA-2 source (Dao-AILab v2.7.4, as shipped in torch 2.11)

The exact instantiation torch ran on the 5090 (from NCU):
`flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, /*Is_Q_in_regs=*/0, /*Share_Q_K_smem=*/0, half_t>>`
— Headdim 64, **kBlockM=128, kBlockN=128, 4 warps**. Against our winner
(`w4x1/f2x8/k4` + `d2/tma/ring`, M=128 = 4 warps × 2 reg tiles × 16 rows, bn=64):

| dimension | emmy (generated) | FA-2 (hand-written) |
|---|---|---|
| CTA query tile | 128 rows — 4 warps × fm=2 × m16 | 128 rows (kBlockM) — 4 warps × MMA_M=2 × m16 |
| per-warp ILP | 2 independent (m,l,O) chains | 2 m-atoms per warp — **identical** |
| KV block | 64 keys | 128 keys |
| Q residency | hoisted register fragments, loaded once | Q in SMEM, re-read per KV block (`Is_Q_in_regs=false` shipped default!) |
| K/V transport | rank-4 TMA boxes, hw swizzle, elected single-thread issue | cooperative cp.async fills, `Swizzle<3,3,3>` smem layout |
| pipelining | d2 ring (2 slots/operand), chunk i+1 under chunk i | phase-interleaved: V loads under the QK gemm, next-K under PV |
| barriers | 2 `__syncthreads` per 64 keys | 2 per 128 keys (measured equivalent — the convoy is costless both ways) |
| P → A operand | `FragmentRepack` (cvt.rn in registers) | `convert_layout_acc_Aregs` — **the same register trick** |
| exp path | libm `expf` (fast-exp measured flat) | `exp2f` + scale·log₂e fold (`softmax_scale_log2`) |
| causal | per-element fragment mask each step | peeled masking steps + block-skip via loop bounds (the tile-skip) |

Article-ready observations:

1. **Convergent geometry.** The autotuner's winning point matches Dao's hand-tuned constants — 128 rows,
   4 warps, 2 m-atoms per warp. Better: the FA-2 source COMMENTS read like our sweep table ("Using 8
   warps is 18% slower… block (256 x 64) is 85% slower, because of register spilling" — our f4 point
   spilled to 341 µs the same way). The hand-tuning notebook and the measured fork rows agree.
2. **The repack is the reference's own move**: `convert_layout_acc_Aregs` is exactly our
   `FragmentRepack` — the C→A lane-map reuse, independently required.
3. **Two places we are structurally AHEAD**: Q lives in registers for the whole stream (FA-2 ships
   `Is_Q_in_regs=false` and re-reads Q from smem every block — our Q-hoist is their disabled variant,
   and our measurements say it wins at bn=64), and the TMA ring (single-thread box copies vs their
   per-thread cp.async fill loops — the LSU gap, 3.31M vs 4.36M, is exactly this).
4. **One place the trick differs and doesn't matter**: they fold scale·log₂e into `exp2f`; our `expf`
   measured flat twice. Their exp hygiene isn't where their speed is either — the pipe is.
5. **KV block width** (their 128 vs our 64): each is right for its transport — their cooperative
   fills amortize over bigger blocks; our TMA ring measured best at 64 (bn=128 rows cost occupancy).
6. **Causal**: their peeled-steps + block-skip is the causal tile-skip still on our follow-up list —
   the one behavior they have that we haven't built (non-causal shapes unaffected).

## Beyond-FA-2: the honest verdict

At 206µs the kernel runs ~167 TFLOPS ≈ **80% of the 5090's f16/fp32-acc tensor roofline — the same point
FA-2 sits at**. Every instruction-shaving lever is now measured (exp path flat twice; LSU already below
FA-2's; conflicts ~0; wave-quantization probes negative), and the two synchronization levers both
declined (WSPEC, warp-private ring). What remains is the shared last-20%-of-roofline territory: drain
reg-depth `p2` (the ldmatrix ping-pong, clamped to 1 — plausible low single digits), and fp8 QK^T (FA-3's
lever — a different accuracy class and a different article chapter). Parity, reached from a generic
moveset with every attribution measured, is the story.

## Future enhancements (the campaign on THIS shape is closed at parity — these are the next chapters)

Everything below has a stated precondition; none pays on the article's large-S non-causal self-attention
shape, where both kernels sit at ~80% of the f16 tensor roofline and every remaining lever measured flat
or negative. Ordered by expected reader/product value:

1. **Causal tile-skip** (causal workloads — ~2× algorithmic). FA-2's form: peel the masked KV steps into a
   short masked loop, skip fully-masked blocks via the loop bounds, run the steady state unmasked. Ours
   would be a schedule-side split of the streaming extent (the causal `Select` gives the per-row bound
   structurally) — no realizer change, the masked steps keep the existing `FragmentMask`. Restores parity
   on causal shapes (FA-2 skips too); the article example is non-causal so it never showed.
2. **The decode package: split-KV + within-CTA coop-KV + warp-private rings** (decode shapes — 1 query row
   × long KV cache, grid starvation). Three findings compose: the `REDUCE` codec's `g<n>k` deferred
   finalize is already documented as the only legal arm for the twisted `(m,l,O)` carrier; the standalone
   decode-combine kernel is the missing piece (`030_split`'s carrier-general combine builder is the seam);
   and the REVERTED warp-private ring (9b8acf2a) becomes CORRECT here — coop-KV warps stream DISJOINT
   K/V, so per-warp TMA rings with zero CTA barriers no longer forfeit any sharing. File under
   Flash-Decoding / the inference chapter.
3. **fp8 QK^T** (throughput at a different accuracy class — FA-3's low-precision pillar; sm_120 HAS fp8
   tensor cores, the one FA-3 pillar this chip can take). Needs: an fp8 mma atom in `ATOM_REGISTRY`, the
   flash eligibility to accept it, per-tensor scales through the recognizer, and Part 1's
   dequantize-bijection story for the numerics section. A real (separate) article chapter.
4. **The base-2 twist family** (exp-dense scalar kernels / ALU-bound shapes — NOT this kernel, measured
   flat twice). The proper shape, recorded so nobody re-derives it: `TwistSpec` gains `base` as data
   (family stays "exp" — the algebra class is unchanged); the recognizer folds `log₂e` into the
   synthesized `_flash_scale` constant when base=2 (constant folding at the source — the per-element
   multiply already exists, so the domain shift is free and EXACT, no end correction); both realizers key
   the exp emission off the base; `exp2` is a one-line numpy-aligned `ElementwiseImpl`. WARNING preserved:
   bare "exp2 in the loop + correct at the end" without the score pre-scale is invalid algebra.
5. **`FAST_EXP` default-on evaluation** (the pin-only knob exists, `085_fast_exp`): flat on flash, but
   exp-dense scalar kernels (softmax/RMSNorm tails, small-head-dim attention) may collect the libm-vs-MUFU
   gap. Needs an accuracy-gated A/B across the golden set before any default flip.
6. **TMA rank-N for the matmul tier** (the flash rank-4 box machinery generalizes; the matmul resolver
   still gates on rank-2 operands — batched matmuls would stage through the same descriptors).
7. **WSPEC re-audit on future hardware** (built, correct, 213 vs 206 on sm_120): the producer/consumer
   split pays only with asynchronous MMA the issuing warp can walk away from (`wgmma`/`tcgen05`). If a
   consumer part ever ships async MMA, the flash WSPEC rows are already wired — re-measure, don't rebuild.
8. **Persistent-kernel / megakernel exploration** (whole-model shapes): out of scope for one-kernel
   articles; noted because the 256-CTA single-wave residency here is the friendly case, and multi-layer
   decode is not.

The article rewrite (2026-07-02) consumed this file's ladder, NCU tables, FA-2 comparison, and the
graveyard; the stale-claim fixes are applied in the published text. This file can be deleted once the
article ships and the enhancement list above moves to an issue tracker.
