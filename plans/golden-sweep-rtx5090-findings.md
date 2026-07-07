# Golden sweep — RTX 5090 (sm_120) open problems, 2026-07-06

Context: the dynamic-matmul scalar→MMA fix, the `_W_A_DYN` refit, and the golden retune landed (PR #317) — all
five `.dynM` matmuls now deploy an MMA tile and sit at 0.93–1.00 of cuBLAS. This file tracks only what is
**still unaddressed**. Numbers are `-O3` deployable, `ratio = cuBLAS / emmy` (≥ 0.95 = golden).

**Resolved (was #1):** the K-heavy `mlp_down` GEMM (M=512, N=4096, **K=14336**) trailed cuBLAS ~0.85× because a
128-CTA grid at 33% occupancy under-parallelized the long K axis (NCU: latency-bound on barrier + short-scoreboard
stalls, *not* memory or bank conflicts). Mimicking cuBLAS's low-occupancy / deep-pipeline recipe made emmy
*slower*; the fix is the opposite — **8-way split-K (`g8k`) on a smaller high-occupancy tile
(`w4x4/f2x2/k4`, 67% occ)** explodes the grid to 2048 CTAs. Both twins now deploy this and reach **0.97–0.98×**
(static 303.6 µs, dynamic 305.1 µs). The greedy prior missed split-K here — and the `.dynM` planner didn't even
auto-select it — so the standing lever is teaching the search to try split-K on K-heavy static GEMMs.

**Resolved (was #2):** `rms_norm` (0.25–0.37×) / `softmax` (0.32–0.44×) trailed torch's fused norms because the
goldens deployed `REDUCE=b32` — one warp reducing *and* normalizing an entire K-row (NCU: 9.3% of peak DRAM
throughput vs torch's 42.9%, ~6% occ; bandwidth-starved, not vectorization). Root cause was the search space:
`coop_reduce_moves()` capped the coop ladder at `b32`, so the tuner could never enumerate wider folds. Extended
the fork to `b512` (`search/space.py`; `_reduce_candidates` gates each on `coop <= K`) and re-recorded all ten
`rms_norm`/`softmax` goldens (static + `.dynM`) at the per-K optimum — **`b512` for K=2048, `b256` for K≥4096** —
now **0.92–1.41×** (8–16 warps/row, 100% occ); the tuner ranks `b256` #1 for `rms_norm.k8192`.

**Resolved (pointwise):** `pointwise.n16384` sat at 0.79× because a pure `Map` (source=None) had **no
register/vector fork** — it emitted scalar 1-element-per-thread code (NCU: 4.8× torch's LSU instructions; torch
runs `vectorized_elementwise_kernel<4>`). Added a register-strip fork on the `FREE` schedule
(`_schedule._map_strip_fork`) reusing the scalar `TILE` codec's `f<fn>` sub-tile — `f<r>` hands each thread `r`
contiguous elements (blocked, unrolled + regrouped) that `050_vectorize_loads` / `080_vectorize_stores` merge into
one `float<r>` access. Gated to flat elementwise bodies (sweeps/accumulators stay scalar). Ladder kept at `f2`/`f4`
(`f8` regressed — register pressure). All four pointwise goldens re-recorded at `TILE=f2`/`f4` → **1.13–1.24×**;
the tuner picks `f2` #1 for `n16384`.

## Full perf table (all 36 shapes)

`emmy µs` / `cuBLAS µs` are the recorded golden latencies; `ratio = cuBLAS / emmy`. Reference per kind: matmul
fp16 → cuBLAS HGEMM (fp32 `square.512` → SGEMM at default tf32, a *soft* ref); attention → torch SDPA; softmax
/ rms_norm → torch's **fused** eager; reduce → `torch.sum` (unfused, a **weak** baseline); pointwise → torch
`relu`. `open` flags the unaddressed item below (blank = at/above parity, nothing to chase).

| shape | S/D | emmy µs | cuBLAS µs | ratio | open |
|---|---|--:|--:|--:|---|
| matmul.square.512 (fp32) | sta | 10.24 | 12.28 | 1.20 | |
| matmul.square.512.fp16 | sta | 3.62 | 6.14 | 1.70 | |
| matmul.square.1024 | sta | 15.59 | 14.51 | 0.93 | |
| matmul.square.2048 | sta | 101.95 | 98.36 | 0.96 | |
| matmul.square.4096 | sta | 640.83 | 640.93 | 1.00 | |
| matmul.qkv.h4096 | sta | 258.47 | 250.54 | 0.97 | |
| matmul.o_proj.h4096 | sta | 101.54 | 95.22 | 0.94 | |
| matmul.mlp_gate_up.h4096 | sta | 595.79 | 557.94 | 0.94 | |
| matmul.mlp_down.h4096 | sta | 303.60 | 296.95 | 0.98 | |
| matmul.square.512.dynM | dyn | 6.14 | 6.14 | 1.00 | #1 (deployed greedy 0.75) |
| matmul.qkv.h4096.dynM | dyn | 258.46 | 249.82 | 0.97 | |
| matmul.o_proj.h4096.dynM | dyn | 100.93 | 95.08 | 0.94 | (= static twin) |
| matmul.mlp_gate_up.h4096.dynM | dyn | 596.30 | 553.70 | 0.93 | (= static twin) |
| matmul.mlp_down.h4096.dynM | dyn | 305.10 | 295.03 | 0.97 | |
| attention.hd64 | sta | 9.25 | 10.24 | 1.11 | |
| attention.hd128 | sta | 16.51 | 18.42 | 1.12 | |
| attention.hd64.dynM | dyn | 19.63 | 10.23 | 0.52 | #3 |
| attention.hd128.dynM | dyn | 30.77 | 18.40 | 0.60 | #3 |
| softmax.k2048 | sta | 3.7 | 4.1 | 1.11 | |
| softmax.k2048.dynM | dyn | 3.7 | 4.1 | 1.11 | |
| softmax.k8192 | sta | 10.3 | 14.29 | 1.39 | |
| softmax.k8192.dynM | dyn | 10.1 | 14.27 | 1.41 | |
| rms_norm.k2048 | sta | 3.8 | 4.1 | 1.08 | |
| rms_norm.k2048.dynM | dyn | 3.9 | 4.1 | 1.05 | |
| rms_norm.k4096 | sta | 6.7 | 6.14 | 0.92 | |
| rms_norm.k4096.dynM | dyn | 6.7 | 6.14 | 0.92 | |
| rms_norm.k8192 | sta | 10.2 | 10.24 | 1.00 | |
| rms_norm.k8192.dynM | dyn | 10.4 | 10.25 | 0.99 | |
| reduce.k2048 | sta | 1.67 | 16.38 | 9.81 | |
| reduce.k2048.dynM | dyn | 1.66 | 16.38 | 9.87 | |
| reduce.k8192 | sta | 4.29 | 16.39 | 3.82 | |
| reduce.k8192.dynM | dyn | 4.26 | 16.39 | 3.85 | |
| pointwise.n4096 | sta | 3.30 | 4.10 | 1.24 | |
| pointwise.n4096.dynM | dyn | 3.30 | 4.10 | 1.24 | |
| pointwise.n16384 | sta | 11.10 | 12.51 | 1.13 | |
| pointwise.n16384.dynM | dyn | 11.10 | 12.52 | 1.13 | |

## 1 — deployed greedy picks a suboptimal MMA tile for `square.512.dynM`

Greedy reaches the MMA tier but not the best tile: it deploys `w4x4/f2x2` (g8k, 67% occ) at **8.19 µs / 0.75×**,
while the golden `w2x4/f2x2/k4` (g2k, 17% occ) is at **6.14 µs / 1.00×**. `eval golden` shows the misses on
TILE and REDUCE; the golden ranks **127/19081** under the cold prior — better than the pre-refit 2337 but not
shallow enough for patience to reach reliably, and the learned prior prefers the higher-occupancy tile that
actually loses. This is an occupancy-vs-smem mispricing for small symbolic-M tiles. **Next:** a `D_*` feature
for the split-K/occupancy interaction, or a patience bump on the small `.dynM` squares.

## 2 — attention golden re-benches are pathological on the current build

Re-benching the recorded attention goldens hit `attention.hd128` (static, `WSPEC=p1`) at **9609 µs** and
`attention.hd64.dynM` at **3103 µs** — ~200–1000× their recorded values, from a fragile warp-specialized /
masked-flash compile path. The greedy attention picks are healthy (hd64 11.75, hd128 16.41, hd64.dynM 19.75),
so the pathology is in the pinned golden knobs on this build, not the kernel kind. Attention goldens were left
untouched. **Next:** a separate investigation into the `WSPEC` / masked-flash compile fragility on sm_120;
until then treat attention golden latencies as suspect and bench greedy directly.

## 3 — dynamic flash trails static (masked streaming flash residual)

Non-causal flash is *faster* than torch SDPA static (hd64/hd128 ~1.11–1.12×), but the `.dynM` masked flash is
0.52–0.60× (golden), 1.7–1.9× behind static. `hd128.dynM` greedy (59.49) is a further 1.96× behind its own
golden (30.43) — a prior shortfall on top of the masked-flash residual. Unlike the matmuls this is not a tier
fallback (the flash goldens already use MMA); it is genuine masked-streaming-flash work. **Next:** revisit after
the attention build fragility (#2) is understood.

## 4 — causal attention deadlocks the bench worker on sm_120

Tuning a **causal** flash shape hangs: causal-mask flash variants hit `nvcc compile failed`, the bench worker
subprocess dies, and the parent tune blocks forever in `ep_poll` — no HungKernel watchdog fires because the
worker *died* rather than a GPU kernel hanging. An emmy bench-worker robustness bug (a dead worker isn't
detected). The attention goldens are therefore non-causal. **Next:** detect worker death in the tune loop, then
add causal attention goldens.

## Workflow gaps (open tooling)

- `run --bench --golden` reports golden rows as **kernel-sum** µs while recorded `emmy_us` is **e2e** — for
  split-K shapes these differ and one hand-bench recorded `mlp_gate_up.dynM` at an anomalous 749 vs the live
  596. Add an `e2e` column per golden row so the recordable number is unambiguous.
- `compile`/`eval` don't say **which prior** produced the greedy pick (cold analytic vs learned `prior.json`);
  a stale learned prior overriding the refit cost a cold-vs-live bake-off to diagnose. Print the active prior.
- Golden rows that bench >5× their recorded `emmy_us` should be flagged `⚠ pathological re-bench` (#2), not
  silently tabulated.
