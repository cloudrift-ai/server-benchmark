# WSPEC over cp.async — producer fill band for the latency-starved sm_89 warp kernels (REFUTED)

> Status: **REFUTED by the phase-1 hand prototype (2026-07-12, rented 4090, 176.124.69.204) — do not
> implement.** Six variants measured, all bit-identical to the baseline, best split −29% vs baseline; the
> full result and the two hardware mechanisms the scoping math missed are in "Prototype result" at the
> bottom. Phases 2–3 are dead. The scoping content below is kept as written for the record.

## Why (the evidence chain that leads here)

The 2026-07-12 investigation (see `golden-manual-sweep-rtx5090-findings.md`, the 4090 sections) pinned the
sm_89 warp matmuls' binding constraint by elimination, each step measured:

1. **DRAM bandwidth — eliminated.** The `RASTER` codec halved gate_up's traffic to the theoretical floor
   (503.6 → 261.6 MB); wall time moved ±2–4%, shape-dependent.
2. **Address-computation overhead — eliminated.** A hand-hoisted fill-addressing prototype (pointer-stepping,
   per-chunk clamp) freed 3 registers and ran **3.3% slower** — ptxas already CSEs optimally; the hoist
   serialized the fill burst's ILP.
3. **What remains, per NCU scheduler stats: latency starvation.** 76.3% of issue cycles have zero eligible
   warps (0.27 eligible / 1.84 active per scheduler) at 2 CTAs/SM. The registers pinning occupancy are the
   tile's **fragments** (~224 of 250 on the winning `w4x1/f4x8/k2` fm gate_up kernel) — the working set
   itself, not overhead.

The structural idea: today every compute warp carries BOTH the mma pipeline (fragments) AND the cp.async fill
work (issue slots, fill-loop state, the commit/wait rhythm). A **producer warp band** owning the fills —
exactly what `WSPEC` already does for TMA — would (a) remove fill issue/state from the consumer warps,
(b) add warps to the scheduler pool that are cheap (producers hold no fragments — tens of registers), and
(c) decouple the fill cadence from the consumer's `wait<N>` stalls. This is the cuBLAS/CUTLASS-class kernel
architecture for pre-Hopper parts (their sm_80 pipelines are producer/consumer over cp.async).

## Current seam (what gates it today)

`WSPEC` is deliberately TMA-only; the gates to relax, in order:

- `search/space.py::WSPEC` help text + `_schedule._wspec_candidates` — offers `p<np>` only over a resolved
  **TMA** stage ("cp.async's wait-group is issuing-thread-scoped and a sync compute-fill has no async load
  half" — the recorded reason; the wait-group scoping is the real design problem, see Risks).
- `_schedule._wspec_workers` — resolves the split; TMA-only assertions.
- `lowering/kernel/_stage.py::_wspec_kloop` — the materialized producer/consumer K-loop; asserts
  `isinstance(transport, TmaTransport)`. The TMA producer is an **elected thread** driving box copies +
  mbarrier expect-tx; a cp.async producer band is a different shape: all 32·np producer threads issue
  per-vector `LDGSTS`, and the hand-off cannot use `cp.async.wait_group` from the consumer side (wait-groups
  are per-issuing-thread) — it needs the **mbarrier + `cp.async.mbarrier.arrive`** form (sm_80+: the
  `cp.async` completes into an mbarrier the consumers wait on, same rhythm as the TMA ring).

## What changes (sketch)

- `_wspec_candidates`: offer `p1`/`p2` over a resolved **cp** stage too (thread budgets unchanged).
- `_wspec_workers`: resolve for cp transports; producer band size vs fill parallelism (a `p1` band = 32
  threads issuing all A+B slab vectors per chunk — gate_up's chunk is 1024+256 elems / 8-elem vectors = 160
  LDGSTS per chunk; one warp covers it in 5 rounds).
- `_stage.py`: a `_cpasync_wspec_kloop` — producer loop: fill chunk k+1's slabs via `LDGSTS` +
  `cp.async.mbarrier.arrive` into the slot's mbarrier; consumer loop: `mbarrier.try_wait` (the existing
  suspend-hint helper) instead of `cp_async_wait<N>`; ring/phase bookkeeping shared with the TMA form.
- Consumer codegen drops the fill loops entirely (register + issue relief on the fragment-heavy warps).

## Expected effect — and the honest occupancy math

Per-CTA registers: today 128 threads × 250 = 32000. Split: 128 consumers × ~225 (fills gone) + 32 producers
× ~40 ≈ 30080 — **still 2 CTAs/SM** (3 CTAs need ≤ 21845). So this does NOT buy a third CTA either; the
claimed win is **scheduler-pool depth and decoupling**: 5 warps/scheduler-pool instead of 4 (producers cheap
and always-runnable), and consumers that never burn eligible slots on fill issue or `wait_group` stalls.
Whether that converts 76%-no-eligible into throughput is exactly what the prototype must measure. Plausible
range: 0–8% (the full residual). It also composes with `RASTER` and the fm atoms unchanged.

## Risks / open questions

- The mbarrier-completion `cp.async` form (`cp.async.mbarrier.arrive`) has its own latency semantics vs
  wait-groups; the ring depth interplay (`d2` today) may want re-tuning per shape.
- Producer/consumer `__syncthreads()` must go (it would re-couple the bands) — bar.sync with named barriers
  or pure mbarrier phases, as the TMA form already does.
- The `p<np>` thread-budget gates (`block_threads + 32·np ≤ 1024`) hold; smem unchanged.
- Search cost: WSPEC rows currently exist only on TMA stages; offering on cp multiplies the sm_89 fork —
  the -O1 ranking-lane censoring (the standing rebench-floor issue) applies to ±5% effects, so goldens/pins
  are the realistic arbiter until that lands.

## Phasing (prototype-gated, in order)

1. **Hand prototype on the dumped gate_up fm kernel** (the `_tune/codegen` harness from the CSE experiment
   reuses directly): hand-write the producer/consumer split of `fm.cu`, bench vs baseline ≥3 passes.
   **Gate: ≥3% reproducible win, or stop and record the refutation.**
2. If passed: implement `_cpasync_wspec_kloop` + the two gate relaxations, off-default (WSPEC rows are
   already opt-in via search/pins); coverage tests mirroring the TMA WSPEC ones.
3. Manual `--ab` A/Bs on the 4090 golden set (`WSPEC=p1/p2` over the fm winners); record `[fm]`+WSPEC golden
   entries where they win; findings update.

## Prototype result (2026-07-12, rented 4090 176.124.69.204) — REFUTED, gate failed decisively

Hand prototype built exactly as phased: the dumped gate_up fm winner (`k_matmul_a8ecf3`, `w4x1/f4x8/k2
d2/cp/ring`, 512×4096 @ 4096×28672) split into consumer warps + a producer band owning all `LDGSTS` fills,
handed off through per-slot mbarriers (`cp.async.mbarrier.arrive.noinc` producer-side, spin on
`mbarrier.test_wait.parity` consumer-side — no `try_wait` on sm_89 — consumers release each slot *before*
their register-only promotes, no `__syncthreads` in the K loop). Harness + kernels on the box under
`_tune/codegen/` (`wspec-proto.py`, `wspec_p1.cu`, `wspec_merged.cu`); every variant's output is
**bit-identical** to the baseline (same per-accumulator op order), so the handoff machinery itself is sound
and reusable. Three interleaved rounds each:

| variant | warps/SM | ptxas regs (spill B) | µs (3 rounds) | vs fm |
| --- | --- | --- | --- | --- |
| fm baseline, 2 CTA/SM | 8 | 250 (0) | 796 / 802 / 830 | — |
| fm forced 1 CTA/SM (41KB smem pad — iso-warp control) | 4 | 250 (0) | 1122 / 1070 / 1108 | +35% |
| p1: 128 consumers + 32 producers | 5 | 238 (0) | 1171 / 1160 / 1190 | +45% |
| p2: 128 + 64 | 6 | 238 (0) | 1034 / 1043 / 1060 | +29% |
| p1 forced 2 CTA/SM (`__launch_bounds__(160,2)`) | 10 | 168 (452) | 2209–2212 | 2.7× |
| merged super-CTA: 2 N-tiles sharing the A slab + 1 producer warp, 288 thr | 9 | 168 (436) | 1773–1831 | 2.2× |

Two hardware facts kill the idea on sm_89, both missed by the scoping math above:

1. **Registers are allocated uniformly per kernel** — there is no `setmaxnreg` before sm_90, so "32
   producers × ~40 regs" does not exist: every producer warp costs the full kernel regcount. The scoping
   doc's 30080-regs-still-2-CTAs arithmetic was unrealizable from the start.
2. **The 64K regfile is banked 16K per SM sub-partition.** At 250 regs the baseline sits at exactly 2
   warps/SMSP (16000/16384) — 8 warps/SM is a *hardware ceiling* for this tile family, and any 9th resident
   warp puts 3 warps on one SMSP, capping the whole kernel at 16384/3/32 → **168 regs**. ptxas confirms:
   both the 9-warp merged CTA and the 10-warp lb2 target compile to exactly 168 regs and spill ~60
   accumulator registers (436–452 B) → the 2.2–2.7× rows. The merged shape was tried twice (A fragments
   cached, then streamed per M-row to shave 12 regs) — ptxas lands on the same 168-reg plateau either way.

And the iso-warp-count control shows there was no latent win to unlock anyway: at 4 compute warps/SM,
self-filling (1100) vs producer-split (p1 1173, p2 1046) is **±6%** — decoupling fill issue from the
consumers is roughly perf-neutral, so the 76%-no-eligible latency starvation is *not* fill-issue coupling;
it is the warp count itself, which this family cannot raise. Producer-band warp specialization on sm_89
could only pay inside tile families that already fit ≥3 warps/SMSP (≤168 regs), and the manual sweeps
showed those geometries lose the fm winners' margin outright. WSPEC stays TMA-only (sm_90+, where
`setmaxnreg` also exists); the sm_89 residual ~8% to cuBLAS stands as structural at this tile size.
