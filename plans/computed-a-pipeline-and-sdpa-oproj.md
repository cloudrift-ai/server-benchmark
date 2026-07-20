# Plan: close the computed-A pipeline gap + cover sdpa→o_proj (gemma-4 underperformers)

## Status / context

The golden-coverage sweep (PR #393, branch `feature/fused-subgraph-goldens`) seeded every gemma-4 kernel that can
misdeploy and confirmed the rest fork nothing. What it did NOT do — by design, it is coverage not perf — is close the
gaps on the kernels that deploy but underperform. Those all trace to **two research-class levers**: the computed-A
fused pipeline, and the fused sdpa→o_proj. This plan scopes both.

Everything below is measured this session on the RTX 5090 (fp16, decode M=32 unless noted), vs torch's UNFUSED
decomposition (the honest reference — emmy runs one fused kernel, torch runs the pieces):

| kernel | deployed µs | eager µs | ratio | where the gap lives |
| --- | --: | --: | --: | --- |
| `mlp_geglu.m32` (gate⊗up→GeGLU) | 172 | 158 | 0.91× | computed-A pipeline (async-B ring loses to d1) |
| `mlp_geglu` @ M=256 (prefill chunk) | 435 | — | <1× vs vLLM | same, amplified at large M (the vLLM-losing shape) |
| `norm_q_proj.m32` (norm→q) | 24.1 | 16.4 | 0.68× | computed-A loses to the split at decode M |
| `norm_kv_proj.m32` (norm→k/v) | 20.8 | 12.5 | 0.60× | same |
| sdpa→o_proj (o_proj-as-consumer) | 289 (scalar!) / 31 (gmem-mma) | 14 | 0.05× / 0.46× | transposed flash-output A → gmem-direct only; drift→scalar |

The unifying root cause: **a computed-A contraction sync-compute-fills its A tile and cannot async-stage it the way a
plain matmul stages a gmem A** — so the weight (B) stream isn't hidden, and the fused form loses to the unfused split
(which rides `d2/tma/ring` on a clean gmem A). The async-B ring for computed-A already exists
(`kernel/_stage.py::SyncTransport.async_operands`, `depth >= 2` = the asymmetric B-only prefetch ring) but its own
docstring records it measured SLOWER than `d1` on the reference shape — "the extra B slot crosses the smem occupancy
quantization." So the lever is not "add async-B"; it is **make async-B actually pay**.

## Workstream 1 — the computed-A async-B pipeline (the primary perf lever)

The gate⊗up / norm→linear megakernels deploy on `d1/sync` (no async overlap) because the `d2` B-only ring loses to it.
Close that:

- **1a. Root-cause the ring's occupancy loss.** `SyncTransport` (`kernel/_stage.py`) rings only the `async_operands`
  (the cp.async B slabs) at `depth>=2`; `_resolve_sync_stage` (`tile/_schedule.py`) clamps depth to the smem budget.
  Profile (NCU) the gate⊗up `d1` vs `d2` at M=32 and M=256: is the loss occupancy (the extra B slot drops CTAs/SM
  below the wave quantum) or a real prefetch-distance shortfall? The `_computed_a_rows` note ("the tradeoff INVERTS at
  decode M") says d2 should already win at M≤32 — verify the seeded `mlp_geglu.m32` (d1) is actually optimal or a
  search miss.
- **1b. TMA-B on the computed-A cone.** `SyncTransport.async_operands` fills B via `cp.async` only; the plain matmul
  tier reaches `d2/tma/ring`. A TMA B-slab (bulk-tensor copy, mbarrier ring) under the sync A-fill would hide the
  dominant weight stream at higher occupancy than cp.async (no per-thread issue). Scope: extend the sync transport to
  emit a TMA-B arm (reuse `_can_stage_warp_tma`'s eligibility on the B operand only; the A stays sync). This is the
  most likely single win for the large-M shapes (gate⊗up 435→~311, the vLLM parity target).
- **1c. Re-tune + re-seed** the computed-A goldens (`mlp_geglu`, `norm_q/kv_proj`) once 1a/1b land, and add the
  `.m256` / `.dynM` / `.s2048` twins the matmul goldens carry (the prefill widths are where the gap is worst).

Success: `mlp_geglu` decode ≥ 1.0× eager and ≥ parity with vLLM's unfused MLP at M=256; `norm_*` computed-A ≥ the
split it currently loses to (or a principled decision to keep splitting them — see WS3).

## Workstream 2 — sdpa→o_proj (`k_linear_sdpa_reduce`)

Two independent sub-problems; do 2a first (bounded), 2b is research.

- **2a. The split o_proj-as-consumer falls to scalar (289 µs / 0.05×).** The o_proj reads the reshaped/transposed
  flash output as its A; that A is a non-contiguous spliced intermediate, so the stage resolver
  (`tile/_schedule.py::_resolve_warp_stage` / `_can_stage_warp`) offers ONLY gmem-direct — the `o_proj.m32` golden's
  `d2/tma/ring` doesn't realize, so it drift-warns and greedy ranks a scalar tile. Fix options:
  - **Materialize the transpose** to a contiguous intermediate (one extra pointwise kernel) so the o_proj A is a clean
    gmem operand that stages `d2/tma/ring` like a standalone o_proj (~11 µs). Costs one transpose kernel's bandwidth;
    likely still a net win over gmem-direct mma (31 µs) and a huge win over scalar (289 µs).
  - **OR strided cp.async on the transposed A**: `_can_stage_warp` gates on the B operand only; extend the A-fill
    closure to carry the transposed index (cp.async has no descriptor, so it CAN issue a strided A load). Then the
    o_proj stages without a materialized transpose.
  - Only after the o_proj reliably stages, record an ACCURATE staged o_proj golden. **Do not** ship a gmem-direct
    o_proj golden first: the memory documents a prior golden on this exact form regressing it 129→189 (a matched-but-
    problematic golden), and capping the o_proj at gmem-direct risks the same.
- **2b. The fused form (`k_linear_sdpa_reduce`, one kernel).** In the tuned model the flash output compute-fills the
  o_proj A (a computed-A contraction with a flash prologue), avoiding the gmem round-trip entirely. It is not cold-
  reproducible (structural pricing chooses fuse only with a trained prior; the split forms above are what a cold
  compile emits). Scope: make the sdpa→o_proj fusion reachable + well-scheduled (the flash O fragments feed the
  o_proj mma without a gmem hop), then add a golden kind for it (an attention→projection computed-A). Gated on WS1's
  async-B work (same pipeline machinery).

## Workstream 3 — coverage completeness (the partial / un-modeled forms)

- **3a. norm→q/k with the per-head q_norm/k_norm epilogue.** The deployed `k_mean_linear_reduce` for q/k applies a
  per-head RMSNorm on the projection output (a reshape + head_dim-256 norm) that `NormLinearGoldenConfig` does not
  model — so `norm_q_proj.m32` anchors the norm→matmul prologue but not the exact deployed op. Decide: extend the
  norm_linear snippet to include the per-head norm (a richer fused form), or a dedicated kind. Verify its shape_key
  vs the deployed op.
- **3b. linear→norm epilogue** (`o_proj + post_attn_norm`, `down_proj + post_ff_norm`). These currently SPLIT into a
  covered matmul + covered rms_norm at decode — confirm that holds at prefill / dynM (the layer-0 dump showed them
  fused). If they fuse at prefill and underperform, they need the WS1 pipeline too.

## Workstream 4 — verification

- Per-kernel: `emmy run --bench --json` before/after each change, 3× reproduced, vs eager AND (for the matmul-shaped
  parts) cuBLAS.
- E2e serving: re-run the 12B `serve --generate` decode TPOT / prefill TTFT A/B vs stock vLLM (the
  `_tune/decode-twin-readiness` harness) — the computed-A pipeline is the last lever between the current decode win
  and a clean end-to-end beat. WS1 is the item that turns per-kernel wins into TPOT.
- Re-seed goldens only after the pipeline change lands (record the improved config, not the current-best-realizable).

## Order of work

1. WS1a (root-cause the ring loss — NCU, cheap, decides everything downstream).
2. WS1b (TMA-B on the computed-A cone — the primary win) → WS1c re-seed.
3. WS2a (o_proj-as-consumer staging — bounded, fixes the 289 µs scalar).
4. WS3a (per-head norm coverage — bounded).
5. WS2b + WS3b (fused sdpa→o_proj + fused epilogues — research, gated on WS1).
6. WS4 e2e verification throughout.

## Findings — WS1a executed 2026-07-17 (5090, CUDA 13.0, fp16)

WS1a (the gate) ran and **redirects WS1b**. All numbers `emmy run --bench`, decode tile held, greedy isolated where noted.

- **The async-B ring (`d2/sync`) — and therefore a TMA-B arm — is contraindicated for the 2-channel gate⊗up.**
  `mlp_geglu` rings BOTH B channels (wg+wu), so `d2` nearly doubles smem and halves occupancy; it loses at every M
  tested (M=32: 170→175; M=256 k2: 466→514, occ 67→33%). A TMA-B arm allocates the SAME ring slabs → cannot dodge the
  cliff (TMA cuts issue cost, not smem footprint). The seeded `mlp_geglu.m32` (`d1`) is optimal, not a search miss.
  `norm_q/kv_proj` win on `d2` only because they ring ONE B channel. **Do not build WS1b as written.**

- **The real M=256 gap is a golden misdeploy, not the pipeline.** Greedy cold-ranks `w2x2/f4x8` = 885 µs (0.34× eager
  304). Two knob fixes, both correcting decode-golden inheritance: (1) **drop split-K** — `g4k` was inherited from the
  M=32 golden; at M=256 the grid is already 960 blocks so split-K only adds a 4× redundant RMSNorm statistic + a 63 MB
  fp16 partial round-trip + a finalize kernel (568→487); (2) **halve bk** k4→k2 (487→466, occ 33→67%, smem 36→18 K).
  **Best shippable f32 = `w1x8/f4x2/k2` no-split = 458 µs = 1.93× over the 885 misdeploy, 0.68× eager.** This is a
  genuine cold-start rescue → **seed an `mlp_geglu.m256` golden** (WS1c, independent of any pipeline change).

- **The residual gap to eager is structural, not a missing pipeline lever.** A diagnostic hand-edit (stub the 8 mma
  calls; harness at `scratchpad/harness/`, driver-API cubin loader that reproduces emmy's 466 µs to <1%) shows the mma
  is only ~24 µs — the kernel is **95% memory-pipeline-stall bound at ~528 GB/s** (30% of peak), floor 447 µs. The knob
  optimum (466) is within 4% of that floor. Every prefetch attempt to hide the `cp.async` B-wait loses to the
  occupancy/sync tension: `d2/k1` (67% occ, prefetch) = 509 (2× the chunks/syncs), `d3/k1` = 487, `d4/k1` = 688.
  cuBLAS reaches peak by running PURE matmuls with a deep pipeline the fused form's occupancy budget can't afford — so
  closing 466→304 needs a structurally different kernel (or splitting the fused edge back into cuBLAS-class matmuls at
  prefill M, the way the tuned decode splits norm→qkv). f16-accumulate gets to 414 µs (0.73×) but is the precision
  lever below, out of scope.

## Findings — "can a fused kernel beat the cuBLAS split?" (2026-07-17, harness diagnostics)

Clock-robust min-of-rounds phase ablation (`scratchpad/harness/multi_bench.cu`; `if(0)`-guarded phase stubs) on the
best f32 tile decomposes the 494 µs (this run's clock) as: **B cp.async load 164, compute/sync pipeline floor 330**
(RMSNorm statistic 62, mma only 47, A compute-fill only 21, remainder ≈ syncthreads+ldmatrix over 120 K-chunks). Two
walls: (1) the B-load is **serialized**, not overlapped (494 ≈ 330 + 164 — `wait<0>` drains before the mma); (2) the
compute/sync floor **alone (330) already exceeds eager (301)**. So even a *perfect* B-overlap only reaches ~330.

Overlap (`d2` prefetch) and occupancy trade off and neither combination beats eager: occupancy is **register-limited**
(59 regs → 4 CTAs → 67%), `d2` either halves occupancy (k2, smem doubles) or doubles the chunk/sync count (k1). f16acc
does NOT free registers (emmy's f16acc keeps an f32 shadow → 74 regs) so f16acc×d2 caps at 413. **Exhaustive ceiling:
f32 fused 458, f16acc fused 413, eager 301.** No knob / micro-edit closes it — A-recompute (the earlier suspicion) is
cheap (21 µs); the wall is the barrier-heavy single-role pipeline. Beating eager needs a ground-up **warp-specialized,
barrier-light, multi-stage** computed-A kernel (dedicated producer warps: TMA/cp.async B + the statistic/compute-fill A
prologue; consumer warps: mma via mbarrier, no CTA syncthreads; fused GeGLU epilogue) — CUTLASS-class, a major
hand-write, not a knob. The mma is only 47 µs, so the tensor headroom to fund it is there.

## VERDICT — a single fused kernel cannot beat the cuBLAS split at M=256 (proven, 5090)

Two independent walls, each alone sufficient:

1. **Multi-channel fusion forbids the fast transport.** `_atom.py`: "cp.async / TMA staging is single-fold — a multi-B
   node rides the sync compute-fill." gate⊗up is 2-channel (wg+wu share A), so it is STRUCTURALLY forced onto the
   `sync` compute-fill (d1/d2 only) and can never ride `d2/tma/ring`. That fast ring is exactly what lets a single
   matmul beat cuBLAS: emmy `w2x2/f4x8/k4 d2/tma/ring` f16acc = **133 µs vs cuBLAS 148** on one gate matmul (1.11×).
2. **The compute floor already ≈ eager.** Sync-isolation ablation (harness): removing the two K-loop syncthreads saves
   only 17 µs (325→308), so warp-specialization (its win is barrier removal + producer/consumer B overlap) tops out at
   the ~308–325 µs compute floor — which already ≈ eager's 301 µs TOTAL, before streaming one byte of B. Warp-spec is
   also independently refuted on the 5090 (wspec matmul 187–397 vs 133 uniform). The fusion's only savings (share A,
   fuse gelu) are negligible against the 236 MB B-stream both approaches pay.

**The way to beat cuBLAS is emmy's SPLIT, not fusion** — and it is now LANDED via `PLACE@cone=cut`: `020_cut_edge`
splits the multi-fold gate/up cone into the norm producer + **N single-channel matmuls** (each rides `d2/tma/ring`,
each beats cuBLAS ~1.1×) + a downstream GeGLU combine. Measured e2e on the 5090 at M=256: **270 µs = 1.17× the
unfused eager pair (317)** — the fused computed-A form provably cannot reach this (its compute floor ≈ eager). This is
the structure the tuned gemma-4 decode already uses for norm→qkv; the mlp edge now splits too at prefill M under the
`cut` pin.

Landed this round: (a) `020_cut_edge` multi-fold split (per-channel matmuls + combine) + `test_place_cone_cut_splits_multi_fold`;
(b) `mlp_geglu.m256` golden seeded in BOTH regimes (regular f32 458 µs / FAST_MATH f16acc 413 µs) — the cold-misdeploy
rescue (885→458/413) for the DEFAULT (fused) deploy. Remaining: make the greedy/prior CHOOSE `cut` for the multi-fold
edge at prefill M by default (today `PLACE@cone` defaults to `fuse`, so the cut needs the pin) and seed the
single-channel cut-consumer matmul goldens so the split deploys at its measured ~113 µs/channel; fuse the gelu·mul into
the up-matmul epilogue to drop the combine kernel.

## Explicitly out of scope

- Non-gemma models. Everything here is scoped to the gemma-4-12B kernel set on the 5090; the 4090 follows once the
  5090 pipeline lands.
- The fast-math (f16-accumulate) regime on the computed-A forms — a separate precision-trading lever, orthogonal to
  the staging pipeline.
