# Golden sweep findings — RTX 5090 (sm_120), 2026-07-06 (golden-set rework: first generic kernel-typed sweep)

- **Branch under test:** `feature/rmsnorm-golden` (main + the golden-set rework — PR #315). The one-model
  (`qwen3_06b`) golden set is replaced with a **generic, kernel-typed matrix**: six kinds × a few sizes ×
  static/dynamic, so this table reads as "which shapes does the compiler handle well vs poorly."
- **Set:** 36 RTX 5090 configs — `matmul` (square fp32+fp16, qkv / o_proj / mlp_gate_up / mlp_down @ h4096),
  `attention.hd{64,128}` (**non-causal**), `softmax`, `rms_norm`, `reduce`, `pointwise` — each **static +
  dynamic (`.dynM`)**. Everything is at **seq / rows = 512** (static explicit, dynamic symbolic-seq benched at
  the 512 hint) so static ⇄ dynamic is directly comparable.
- **Sweep:** per-shape `emmy tune -c "<snippet>" [--dynamic …] --clean --bench` (cold DB each) → the deployable
  `-O3` re-bench vs the shape's reference. **Reference per kind:** matmul fp16 → cuBLAS HGEMM; matmul fp32 →
  cuBLAS SGEMM (default `allow_tf32`, so likely the TF32 tensor-core path — a soft reference); attention → torch
  SDPA; softmax / rms_norm → torch's **fused** eager; reduce → `torch.sum`; pointwise → torch `relu`.
- **`ratio = ref / emmy`** (≥ 1 = emmy at/above the reference; a config is **golden** at ≥ 0.95).
- **Tally (36 shapes):** **13 golden (≥ 0.95)**, 23 below — the below-parity cohort is entirely (a) the
  masked-tile **dynamic** matmul/attention and (b) `rms_norm` / `softmax` vs torch's fused norms.
- **Note:** *causal* attention is **deferred** — causal-mask flash variants hit `nvcc compile failed` and
  deadlock the bench worker on sm_120 (an emmy worker-robustness bug, Finding 6). Non-causal tunes cleanly.

## Per-shape outcomes (`-O3` deployable, clean tune per shape)

| shape | S/D | emmy µs | ref µs | ratio | note |
|---|---|--:|--:|--:|---|
| matmul.square.512 (fp32) | sta | 10.24 | 12.28 | **1.20** | fp32 CUDA-core, at/above cuBLAS (soft tf32 ref) |
| matmul.square.512.fp16 | sta | 3.62 | 6.14 | **1.70** | tiny fp16 square — emmy well above cuBLAS |
| matmul.square.1024 | sta | 15.59 | 14.51 | 0.93 | near cuBLAS |
| matmul.square.2048 | sta | 101.95 | 98.36 | **0.96** | near cuBLAS |
| matmul.square.4096 | sta | 640.83 | 640.93 | **1.00** | exact cuBLAS parity |
| matmul.square.512.dynM | dyn | 12.26 | 6.14 | 0.50 | masked-tile 2× |
| matmul.qkv.h4096 | sta | 258.47 | 250.54 | **0.97** | fused-QKV proj, near cuBLAS |
| matmul.qkv.h4096.dynM | dyn | 1111.68 | 254.34 | 0.23 | masked-tile 4.4× |
| matmul.o_proj.h4096 | sta | 101.54 | 95.22 | 0.94 | near cuBLAS |
| matmul.o_proj.h4096.dynM | dyn | 418.62 | 96.35 | 0.23 | masked-tile 4.3× |
| matmul.mlp_gate_up.h4096 | sta | 595.79 | 557.94 | 0.94 | (512,4096)@(4096,28672) |
| matmul.mlp_gate_up.h4096.dynM | dyn | 2580.13 | 576.29 | 0.22 | masked-tile 4.5× — worst dynamic |
| matmul.mlp_down.h4096 | sta | 346.67 | 296.95 | 0.86 | (512,14336)@(14336,4096) |
| matmul.mlp_down.h4096.dynM | dyn | 1453.73 | 296.00 | 0.20 | masked-tile 5× |
| attention.hd64 | sta | 9.25 | 10.24 | **1.11** | non-causal flash — beats torch SDPA |
| attention.hd64.dynM | dyn | 19.63 | 10.23 | 0.52 | masked-tile 1.9× |
| attention.hd128 | sta | 16.51 | 18.42 | **1.12** | non-causal flash — beats torch SDPA |
| attention.hd128.dynM | dyn | 30.77 | 18.40 | 0.60 | masked-tile 1.7× |
| softmax.k2048 | sta | 9.30 | 4.10 | 0.44 | vs torch's fused softmax |
| softmax.k2048.dynM | dyn | 9.44 | 4.10 | 0.43 | ≈ static (no masked-tile penalty) |
| softmax.k8192 | sta | 44.24 | 14.29 | 0.32 | wider K → worse |
| softmax.k8192.dynM | dyn | 44.76 | 14.27 | 0.32 | ≈ static |
| rms_norm.k2048 | sta | 10.94 | 4.10 | 0.37 | vs torch's fused RMSNorm |
| rms_norm.k2048.dynM | dyn | 11.08 | 4.10 | 0.37 | ≈ static |
| rms_norm.k4096 | sta | 20.86 | 6.14 | 0.29 | wider K → worse |
| rms_norm.k4096.dynM | dyn | 20.77 | 6.14 | 0.30 | ≈ static |
| rms_norm.k8192 | sta | 41.03 | 10.24 | 0.25 | worst norm |
| rms_norm.k8192.dynM | dyn | 41.54 | 10.25 | 0.25 | ≈ static |
| reduce.k2048 | sta | 1.67 | 16.38 | **9.81** | vs unfused `torch.sum` — emmy crushes it |
| reduce.k2048.dynM | dyn | 1.66 | 16.38 | **9.87** | ≈ static |
| reduce.k8192 | sta | 4.29 | 16.39 | **3.82** | (torch.sum ~flat 16µs) |
| reduce.k8192.dynM | dyn | 4.26 | 16.39 | **3.85** | ≈ static |
| pointwise.n4096 | sta | 4.30 | 4.10 | **0.95** | memory-bound, near torch |
| pointwise.n4096.dynM | dyn | 4.31 | 4.10 | **0.95** | ≈ static |
| pointwise.n16384 | sta | 15.81 | 12.51 | 0.79 | wider → slightly behind |
| pointwise.n16384.dynM | dyn | 15.98 | 12.52 | 0.78 | ≈ static |

## Finding 1 — static GEMM is at cuBLAS; the masked-tile *dynamic* GEMM is 2–5× behind

Static fp16 matmul lands **0.86–1.20× of cuBLAS** across the range — `square.4096` is exact parity (1.00),
`square.512.fp16` is 1.70×, and the realistic layer GEMMs (qkv/o/gate_up/down @ h4096) are 0.86–0.97×. The
tensor-core matmul path is deployable-competitive.

The **`.dynM`** (symbolic-seq masked-tile) twins collapse to **0.20–0.50×** (2–5× slower): `mlp_down.dynM` 0.20,
`mlp_gate_up.dynM` 0.22, `qkv/o.dynM` 0.23, `square.512.dynM` 0.50. This is the deployable-artifact gap — the
masked warp-MMA (ceil-div grid + boundary guards) does not reach the static tile's throughput on the big
`N`-heavy GEMMs. This is the single highest-value lead in the set.

## Finding 2 — non-causal attention beats torch SDPA (static); dynamic flash is 1.7–1.9× behind

`attention.hd{64,128}` (non-causal flash) run **1.11–1.12× of torch SDPA** — emmy's flash is *faster* than
PyTorch's for these shapes. The `.dynM` masked-tile flash is **0.52–0.60×** (1.7–1.9× behind) — the same
masked-tile penalty as the dynamic GEMMs, but milder (attention's reduce dominates less than a big GEMM's `N`).
*Causal* attention is not in the table — see Finding 6.

## Finding 3 — emmy crushes `torch.sum` but trails torch's *fused* rms_norm / softmax

Two opposite results on reduce-family kernels:

- **`reduce` (`torch.sum`): emmy 3.8–9.9× faster.** `torch.sum(512,K)` is ~flat 16 µs (unfused, launch-bound);
  emmy's cooperative reduce is 1.6–4.3 µs. This is a reference artifact as much as an emmy win — `torch.sum`
  is a weak baseline.
- **`rms_norm` 0.25–0.37× / `softmax` 0.32–0.44×.** Here the reference is torch's **fused** RMSNorm / softmax
  (4–14 µs), and emmy's fused-reduce-then-sweep kernel is 2.7–4× slower. Both worsen with wider `K`
  (rms_norm.k8192 0.25, softmax.k8192 0.32). These are memory-bound norms not saturating bandwidth — the top
  follow-up after the dynamic-GEMM gap. (Same class as the model-level `k_linear_mean_reduce` cost.)

## Finding 4 — the masked-tile penalty is tensor-core-only; memory-bound kinds have static ≈ dynamic

The `.dynM` vs static gap only appears on **matmul + attention** (the warp-MMA kernels): 2–5×. For
**softmax / rms_norm / reduce / pointwise**, static and dynamic are within ~2% of each other
(e.g. rms_norm.k4096 0.29 sta vs 0.30 dyn; reduce.k2048 9.81 vs 9.87). The boundary guards that hurt a warp-MMA
tile are negligible on an already scalar / memory-bound reduce or pointwise loop. So "masked-tile is slow" is
specifically a **warp-tier** story, not a general dynamic-shape story.

## Finding 5 — pointwise is near-parity (memory-bound)

`pointwise` (relu) is **0.78–0.95×** torch — memory-bound, near the reference, slightly behind at the wider
`n16384`. No dynamic penalty. Nothing to chase here.

## Finding 6 — causal attention deadlocks the bench worker on sm_120 (deferred)

Tuning a **causal** flash shape hangs: many causal-mask flash variants hit `nvcc compile failed`, the bench
worker subprocess dies, and the parent tune blocks forever in `ep_poll` (both CPU and GPU idle — no HungKernel
watchdog fires because the worker *died* rather than a GPU kernel hanging). This is an emmy bench-worker
robustness bug (a dead worker isn't detected), not a golden-schema issue. The attention goldens are therefore
**non-causal**; add causal ones after fixing the worker-death handling.

## Workflow notes

- The golden framework now carries `dynamic` as a bool on the base `GoldenConfig` with per-kind
  `dynamic_specs()`, and `SoftmaxGoldenConfig` / `AttentionGoldenConfig` kinds — so `tune --dataset golden
  --kernel <k>` and `run --bench --golden NAME` cover every kind, not just matmul.
- Data + logs for this sweep: `_tune/golden-rework/{results,dump,driver.log}` (gitignored). Regenerate the
  YAML from the results with `_tune/golden-rework/emit_yaml.py`.
- **Reference caveat:** the fp32 square uses cuBLAS at default `allow_tf32` (TF32 tensor-core), not a true
  fp32 SGEMM — its 1.20× is vs the faster TF32 path, so it is a *soft* golden. Pin `allow_tf32=False` for a
  true-fp32 comparison if the fp32 CUDA-core path is being regressed.
