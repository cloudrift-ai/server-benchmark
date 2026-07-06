# Golden sweep findings — RTX 5090 (sm_120), 2026-07-06 (dynamic-matmul scalar→MMA fix + prior refit + retune)

- **Branch:** `feature/dynm-matmul-prior-refit` (off `feature/rmsnorm-golden`). This sweep chases the single
  highest-value lead from the prior golden-set rework: the `.dynM` matmuls ran **2–5× slow**, originally
  written up as an inherent masked-MMA cost. **It was not.** Every `.dynM` matmul golden had fallen back to a
  **scalar CUDA-core** tile (no tensor cores). This branch fixes it end-to-end: (1) the goldens, (2) the cold
  `AnalyticPrior` dynamic weights, (3) the learned `prior.json`.
- **What changed, in order:**
  1. **Goldens** — the 5 `.dynM` matmul goldens now carry their static counterpart's MMA/TMA knobs
     (`a:mma_m16n8k16_f16/…`), re-benched `-O3` on the live 5090.
  2. **`_W_A_DYN` refit** — `scripts/golden_knob_heuristics.py` re-fit the dynamic analytic weights over the
     corrected goldens (dynamic golden median rank **2337 → 17** in the fit's pool; **127/19081** on the full
     live enumeration for `square.512.dynM`). Pasted into `search/prior/analytic.py`.
  3. **Retune** — `emmy tune --dataset golden --clean` (39 shapes, ~96 min wall, `-O1` ranking lane) wiped the
     stale `prior.json` and retrained it. This is what fixes *deployed* greedy (see Finding 2).
- **Sweep A/B:** `emmy run --bench --golden NAME` per shape, `-O3` deployable, greedy-vs-golden live this run.
  `greedy` = greedy pick kernel-sum µs; `e2e` = `Emmy` backend end-to-end; `cuBLAS` = the `Eager PyTorch` row
  (HGEMM for fp16, SDPA for attention); `ratio = cuBLAS / emmy` (≥ 0.95 = golden).
- **Headline:** deployed greedy now picks an **MMA tile for all 5 `.dynM` matmuls** (was scalar on every one).
  The dynamic-matmul cohort moved from **0.20–0.50× → 0.85–1.00×** of cuBLAS. `square.512.dynM` and
  `qkv.dynM` are golden; `o_proj/mlp_gate_up/mlp_down.dynM` land 0.85–0.94 (a genuine masked-tile residual vs
  their static twins, not a tier fallback).
- **Category tally (matmul + attention focus set, 18 shapes):** 1 replaced (`o_proj.dynM`, a real 10.5% tile
  win), 1 number-corrected (`mlp_gate_up.dynM`, stale 749→596 µs), rest validated-unchanged or greedy-worse
  (golden kept). The memory-bound kinds (softmax / rms_norm / reduce / pointwise) were retuned but not re-A/B'd
  this pass — their goldens are unchanged (reduce-tier, `REDUCE=b32`, static ≈ dynamic; see Finding 4).

## Per-shape outcomes — matmul + attention (`-O3` A/B, fresh prior)

`greedy µs` / `best-golden µs` are the live kernel-sum A/B; `cuBLAS µs` is the same run's eager row; `vs cuBLAS`
is `greedy_e2e / cuBLAS` (>1 = emmy slower). Recorded `emmy_us` shown where the YAML was edited.

| shape | S/D | greedy µs | best-gold µs | greedy/gold | cuBLAS µs | vs cuBLAS | tier | category |
|---|---|--:|--:|--:|--:|--:|---|---|
| matmul.square.512 (fp32) | sta | 8.52 | 8.50 | 1.00 | 12.28 | 0.83 | scalar | same |
| matmul.square.512.fp16 | sta | 3.78 | 4.36 | 0.87 | 6.14 | 0.62 | MMA | same (noise) |
| matmul.square.1024 | sta | 15.61 | 15.62 | 1.00 | 14.41 | 1.08 | MMA | same |
| matmul.square.2048 | sta | 101.96 | 113.90 | 0.90 | 97.42 | 1.05 | MMA | greedy>gold (static, noted) |
| matmul.square.4096 | sta | 627.02 | 646.02 | 0.97 | 639.23 | 0.98 | MMA | same |
| matmul.qkv.h4096 | sta | 248.74 | 259.05 | 0.96 | 248.38 | 1.00 | MMA | same |
| matmul.o_proj.h4096 | sta | 101.75 | 101.93 | 1.00 | 94.83 | 1.07 | MMA | same |
| matmul.mlp_gate_up.h4096 | sta | 592.19 | 633.89 | 0.93 | 552.80 | 1.07 | MMA | same |
| matmul.mlp_down.h4096 | sta | 346.67 | 346.05 | 1.00 | 293.28 | 1.18 | MMA | same |
| **matmul.square.512.dynM** | dyn | 6.58 | **4.42** | 1.49 | 6.14 | 1.33 | **MMA** | greedy worse → gold kept (1.00) |
| **matmul.qkv.h4096.dynM** | dyn | 250.28 | 259.24 | 0.97 | 248.73 | 1.01 | **MMA** | validated (0.97) |
| **matmul.o_proj.h4096.dynM** | dyn | 100.93 | 112.70 | 0.90 | 95.08 | 1.06 | **MMA** | **replaced → w4x2 (0.94)** |
| **matmul.mlp_gate_up.h4096.dynM** | dyn | 594.75 | 596.30 | 1.00 | 553.70 | 1.07 | **MMA** | **number fixed (0.93)** |
| **matmul.mlp_down.h4096.dynM** | dyn | 350.09 | 361.75 | 0.97 | 294.79 | 1.19 | **MMA** | validated (0.85) |
| attention.hd64 | sta | 11.75 | 9.88 | 1.19 | 10.23 | 1.15 | flash | greedy worse → gold kept |
| attention.hd128 | sta | 16.41 | *9609* | — | 18.40 | 0.89 | flash | gold re-bench pathological (Finding 5) |
| attention.hd64.dynM | dyn | 19.75 | *3103* | — | 10.23 | 1.93 | flash | gold re-bench pathological (Finding 5) |
| attention.hd128.dynM | dyn | 59.49 | 30.43 | 1.96 | 18.34 | 3.24 | flash | greedy worse → gold kept |

*Dynamic-matmul before/after (recorded golden ratio vs cuBLAS): `square.512` 0.50→**1.00**, `qkv` 0.23→**0.97**,
`o_proj` 0.23→0.94, `mlp_down` 0.20→0.85, `mlp_gate_up` 0.22→0.93. All five were scalar; all five are now MMA.*

## Finding 1 — the `.dynM` collapse was a scalar-tier fallback, and it's fixed (goldens + priors now MMA)

Inspecting the recorded kernels killed the original "inherent masked-MMA cost" theory: every `.dynM` matmul
golden had a **scalar `nXxY` tile** — a CUDA-core FMA loop (`acc += __half2float(a) * __half2float(b)`) that
never issues `mma.sync`. The 2–5× gap was exactly fp16-tensor-core vs fp32-accumulate CUDA-core; it widened
with `N` (more FLOP-bound → idle tensor cores cost more). The masked warp-MMA path **exists and compiles**
for a symbolic M (ceil-div grid, `dpl_mma_load_a_gmem_mclamp` row-clamp, guarded `__half2` stores) — forcing
the static tile onto `--dynamic seq_len@x0:0` lowers to a correct masked-MMA kernel and closes the gap:

| shape | scalar (old) | MMA (new) | speedup | ratio vs cuBLAS |
|---|--:|--:|--:|--:|
| square.512.dynM | 12.26 µs | 6.14 µs | 2.0× | 0.50 → **1.00** |
| qkv.h4096.dynM | 1111.68 | 258.46 | 4.3× | 0.23 → **0.97** |
| o_proj.h4096.dynM | 418.62 | 100.93 | 4.1× | 0.23 → 0.94 |
| mlp_gate_up.h4096.dynM | 2580.13 | 596.30 | 4.3× | 0.22 → 0.93 |
| mlp_down.h4096.dynM | 1453.73 | 348.71 | 4.2× | 0.20 → 0.85 |

The three big `N`/`K`-heavy GEMMs land 0.85–0.94 — *now* a genuine masked-tile residual vs their static twins
(the boundary-guard + no-static-prologue tax), the remaining matmul lead. **Recommendation:** chase the
masked-tile residual on `mlp_down.dynM` (0.85, worst) — it's the split-K reduction + guard overhead, not a
tier problem.

## Finding 2 — refitting the analytic prior fixes the *cold* prior; only the retune fixes *deployed* greedy

The scalar pick had two layers, and the fix needed both:

- **Cold `AnalyticPrior`** carries two weight sets, selected per-config on `S_ext_n_symbolic_axis`: `_W_A`
  (static) and `_W_A_DYN` (symbolic). The old `_W_A_DYN` was fit to reproduce the `.dynM` goldens — which
  were *themselves* the scalar picks — so it learned to rank scalar tiles first (`MMA_tier` weight 1.75 vs the
  static set's 7.17). A circular, self-reinforcing bad fit. Re-fitting over the corrected (MMA) goldens fixed
  it: verified `EMMY_PRIOR_FILE=/nonexistent emmy compile --dynamic … --ir cuda` now emits `mma.sync` (cold
  greedy picks `a:mma_m16n8k16_f16/w2x4/f2x2` TMA, 7 µs / 0.93×, up from 12 µs / 0.50× scalar).
- **Learned `prior.json`** is what `compile`/`run` actually read (the cold prior is only the fallback). It was
  trained on the old scalar data, so **deployed greedy kept picking scalar even after the refit** — proven by
  compiling with the live prior vs the cold prior (scalar vs MMA). `emmy tune --dataset golden --clean` wiped
  and retrained it; deployed greedy now picks MMA for all 5 `.dynM` matmuls (the A/B `tier` column).

There is **no feature discrepancy to fix**. `MMA_tier` is computed identically for both tiers. The
static/dynamic feature paths differ deliberately (the free-dim product excludes the symbolic axis — `free = N`
not `M·N`; `D_neg_masked_{m,n,k}` fire only when masked), but those are correct — a masked tile prices
differently. The scalar pick was a training-target bug (both priors trained on scalar goldens), not a modeling
bug. **Recommendation:** none outstanding — the two-weight-set design stays; the refit + retune closed it.

## Finding 3 — the deployed greedy MMA tile is not always the golden tile (residual search shortfall)

Greedy now reaches the MMA *tier* but not always the best MMA *tile*:

- `square.512.dynM`: greedy `w4x4/f2x2` (g8k) benches **8.19 µs e2e** vs the golden `w2x4/f2x2/k4` (g2k) at
  **6.14** — greedy is 1.33× cuBLAS, the golden is at parity. `eval golden` shows greedy misses on TILE
  (`w4x4` vs `w2x4/…/k4`) and REDUCE (`g8k` vs `g2k`). The golden ranks 127/19081 under the cold prior — much
  better than the old 2337, but not top, so patience doesn't always reach it and the learned prior's
  higher-occupancy `w4x4` (67% occ) loses to the golden's smem-heavy `w2x4/…/k4` (17% occ). This is the one
  shape where greedy is clearly worse than its golden.
- `o_proj.dynM`: the reverse — greedy `w4x2/f2x4/k4` (100.9 µs, reproduced 2×) **beats** the hand-recorded
  `w8x2/f2x4/k4` (112.7) by 10.5%. Recorded the greedy tile as the new golden.

**Recommendation:** the `square.512.dynM` miss is an occupancy-vs-smem tradeoff the linear prior misprices for
small symbolic-M tiles (it rewards the high-occupancy `w4x4` but the low-occupancy smem-heavy tile wins). A
targeted `D_*` feature for the split-K-vs-occupancy interaction, or a patience bump on the small `.dynM`
squares, would let the search reach rank 127 → top. Lower priority than the masked-tile residual (Finding 1).

## Finding 4 — memory-bound kinds unchanged; the `.dynM` penalty is warp-tier-only

softmax / rms_norm / reduce / pointwise `.dynM` goldens already carry the same knobs as their static twins
(`REDUCE=b32`, reduce-tier) and bench within ~2% static-vs-dynamic — no scalar-fork bug, nothing to copy, and
the `_W_A_DYN` refit (which only moved matmul-tier weights) leaves them untouched. Their recorded ratios stand
from the prior sweep: `reduce` 3.8–9.9× (vs unfused `torch.sum`, a weak baseline); `rms_norm` 0.25–0.37× /
`softmax` 0.32–0.44× (vs torch's *fused* norms — memory-bound kernels not saturating bandwidth, the top
non-matmul lead); `pointwise` 0.78–0.95×. So "masked-tile is slow" is specifically a **warp-tier** story
(matmul + attention), never a general dynamic-shape story.

## Finding 5 — attention golden re-benches are pathological on the current build (deferred, do not touch)

The A/B `--golden` re-bench of the recorded attention goldens hit catastrophic latencies: `attention.hd128`
(static, `WSPEC=p1`) benched **9609 µs** and `attention.hd64.dynM` (`w2x1/f1x8/k8`) **3103 µs** — ~200–1000×
their recorded values, from a fragile warp-specialized / masked-flash compile path (same class as the causal
deadlock, Finding 6). The **greedy** attention picks are healthy (hd64 11.75, hd128 16.41, hd64.dynM 19.75) —
so the pathology is in the pinned golden knobs on this build, not the kernel kind. The dynamic flash still
trails its static twin where greedy is healthy (`hd128.dynM` greedy 59.49 vs golden 30.43, 1.96×). Attention
goldens were **left entirely untouched** — copying static knobs onto them regresses them (verified earlier:
hd128 static-onto-dynamic → 9.7 ms), and their masked-flash improvement is real streaming-flash work.
**Recommendation:** a separate investigation into the `WSPEC` / masked-flash compile fragility on sm_120;
until then treat attention golden latencies as suspect and bench greedy directly.

## Finding 6 — causal attention deadlocks the bench worker on sm_120 (deferred, unchanged)

Tuning a **causal** flash shape still hangs: causal-mask flash variants hit `nvcc compile failed`, the bench
worker subprocess dies, and the parent tune blocks forever in `ep_poll` (no HungKernel watchdog fires because
the worker *died* rather than a GPU kernel hanging). An emmy bench-worker robustness bug (a dead worker isn't
detected), not a golden-schema issue. The attention goldens are therefore **non-causal**; add causal ones
after fixing worker-death handling. (This sweep's `--dataset golden` tuned all 39 non-causal shapes cleanly,
no deadlock.)

## Workflow notes

- **The tune dominates wall time (~96 min / 39 shapes).** The big `N`/`K`-heavy GEMMs at `-O3` and the flash
  shapes dominate; the memory-bound kinds are seconds each. A `--kernel matmul` narrow (or skipping the
  already-golden static squares) would cut a focused re-tune to ~15 min. *Improvement:* a `--dataset golden
  --kernel-type matmul` filter, or a `--skip-golden` flag to skip shapes already at ratio ≥ 0.95.
- **Metric mismatch bit the recording.** The A/B kernel table reports **kernel-sum** µs; the recorded
  `emmy_us` convention is **e2e** (`Emmy` backend, comparable to the e2e `cublas_us`). For split-K shapes
  (2 kernels) these differ, and one earlier hand-bench recorded `mlp_gate_up.dynM` at 749 µs (anomalous) vs
  the live 596. *Improvement:* have `run --bench --golden` print an `e2e` column per golden row (not just
  greedy), so the recordable number is unambiguous and A/B-comparable without a second pinned bench.
- **Two priors, one confusing pick.** Diagnosing "refit did nothing" took a cold-vs-live prior compile
  bake-off to discover the learned `prior.json` overrides the refit `AnalyticPrior`. *Improvement:* `emmy
  compile`/`eval` should print *which* prior produced the greedy pick (cold analytic vs learned), so a stale
  learned prior is visible without an `EMMY_PRIOR_FILE=/nonexistent` A/B.
- **Attention golden re-bench pathology (Finding 5)** cost several confused minutes — a golden row showing
  9609 µs reads as a data error, not a build fragility. *Improvement:* flag golden rows that bench >5× their
  recorded `emmy_us` as `⚠ pathological re-bench` rather than silently tabulating them.
- **Noise floor.** The `-O3` re-bench swings ~10–13% on the small shapes; every "win" here (`o_proj.dynM`
  10.5%, `square.512.dynM` golden-better 1.49×) was reproduced ≥2× before recording. The `square.512.fp16`
  static (greedy 3.78 vs golden 4.36, 13%) sat exactly on the noise band and was left unchanged.
- **Data + logs:** `_tune/golden-sweep-rtx5090/tune.log` (tune), scratch A/B JSONs (gitignored). Refit weights
  from `scripts/golden_knob_heuristics.py --samples 20000`.
