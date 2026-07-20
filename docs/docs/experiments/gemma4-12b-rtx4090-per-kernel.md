---
title: "gemma-4-12B per-kernel on RTX 4090"
description: Per-kernel latency of emmy's greedy deploy picks vs PyTorch eager and torch.compile on an RTX 4090.
---

# gemma-4-12B — per-kernel latency on RTX 4090 (sm_89)

The full model does not fit on a 24 GB RTX 4090, so this is the per-kernel view (same format as
[part 3 of the compiler blog series](https://www.cloudrift.ai/blog/building-gpu-compiler-from-scratch-3)): each case is
one golden shape from the gemma-4-12B decoder layer — attention projections (sliding + global variants), attention
itself at both head sizes, the gated MLP and the norms — benched via `emmy run --golden <name> --bench` against PyTorch
eager and `torch.compile`. Emmy numbers are the **greedy deploy picks** (repo-shipped `rtx4090_sm89*` goldens, no
on-box tuning).

**Setup** — vast.ai RTX 4090, driver 580.65.06, CUDA 12.8 (nvcc V12.8), PyTorch 2.13.0+cu130, emmy @ `7d34e841`,
fp16, prefill shapes at seq 512 (base names) and seq 2048 (`.s2048`). Measured 2026-07-19.

<iframe src="/experiments/gemma4_12b_rtx4090_per_kernel.html" style={{width: "100%", maxWidth: "960px", height: "720px", border: "none", background: "transparent", display: "block", margin: "0 auto"}} loading="lazy" title="Per-kernel speedup: emmy greedy deploy pick and torch.compile against PyTorch eager; 23 gemma-4-12B golden shapes on RTX 4090, sorted by emmy ratio."></iframe>

| kernel | eager µs | torch.compile µs | emmy µs | vs eager | vs torch.compile |
|---|--:|--:|--:|--:|--:|
| rms_norm.k3840 | 7 | 6 | 7.0 | 1.00x | 0.86x |
| qknorm.k256 | 5 | 4 | 4.0 | 1.25x | 1.00x |
| qknorm.k512 | 11 | 16 | 10.0 | 1.10x | 1.60x |
| q_proj | 117 | 116 | 118.0 | 0.99x | 0.98x |
| q_proj.s2048 | 420 | 417 | 492.0 | 0.85x | 0.85x |
| q_proj_global | 220 | 220 | 250.0 | 0.88x | 0.88x |
| q_proj_global.s2048 | 854 | 848 | 913.0 | 0.94x | 0.93x |
| kv_proj | 51 | 51 | 60.0 | 0.85x | 0.85x |
| kv_proj.s2048 | 221 | 220 | 258.0 | 0.86x | 0.85x |
| k_proj_global | 19 | 19 | 23.0 | 0.83x | 0.83x |
| k_proj_global.s2048 | 56 | 56 | 66.0 | 0.85x | 0.85x |
| attention.hd256 | 42 | 42 | 39.0 | **1.08x** | **1.08x** |
| attention.hd256.s2048 | 356 | 348 | 403.0 | 0.88x | 0.86x |
| attention.hd512 | 103 | 99 | 315.0 | 0.33x | 0.31x |
| attention.hd512.s2048 | 1054 | 1048 | 3502.0 | 0.30x | 0.30x |
| o_proj | 118 | 117 | 112.0 | **1.05x** | **1.04x** |
| o_proj.s2048 | 472 | 468 | 522.0 | 0.90x | 0.90x |
| o_proj_global | 248 | 248 | 254.0 | 0.98x | 0.98x |
| o_proj_global.s2048 | 848 | 848 | 938.0 | 0.90x | 0.90x |
| mlp_gate_up | 879 | 873 | 846.0 | **1.04x** | **1.03x** |
| mlp_gate_up.s2048 | 3169 | 3164 | 4382.0 | 0.72x | 0.72x |
| mlp_down | 415 | 415 | 454.0 | 0.91x | 0.91x |
| mlp_down.s2048 | 1655 | 1654 | 1848.0 | 0.90x | 0.90x |

All names are `gemma4_12b.*` golden entries; raw per-case data in
[gemma4-12b-rtx4090-per-kernel.json](./gemma4-12b-rtx4090-per-kernel.json).

## Headline

| Bucket | torch.compile | Emmy (greedy deploy pick) |
|:---|:---|:---|
| Cases at or above eager (ratio ≥ 1) | 22 / 23 | 6 / 23 |
| Geomean ratio vs eager | 1.01x | 0.85x (0.93x excl. hd512) |
| Best ratio vs eager | — | 1.25x |
| 90th-percentile ratio vs eager | — | 1.07x |
| Worst ratio vs eager | — | 0.30x (attention.hd512.s2048) |

On these large fp16 matmuls `torch.compile` dispatches to the same cuBLAS kernels as eager, so its column tracks eager
within noise everywhere except the small norm kernels — "vs eager" and "vs torch.compile" are effectively the same
comparison for the projection/MLP rows.

## Reading the results

- **Wins at seq 512**: flash attention at hd256 (1.08x over torch SDPA, causal tile-skip), `o_proj` (1.05x) and the
  fused `mlp_gate_up` (1.04x) beat cuBLAS.
- **Known sm_89 residual**: `kv_proj` / `k_proj_global` at 0.83–0.86x are the latency-bound losses recorded in the
  goldens file header — nothing in the knob space closes them on this cap.
- **Open problems**: `attention.hd512` (0.30–0.33x; the d_v-fold 255-register O-accumulator ceiling) and
  `mlp_gate_up.s2048` (0.72x, where the base shape wins). The `.s2048` shapes are generally weaker than their
  seq-512 twins (0.72–0.94x) — the goldens were seeded at M=512 and the larger-M tiles have had less tuning attention.
- **2026-07-20 re-tune note**: a cold golden sweep on a rented 4090 recorded a fast-math `attention.hd512` config at
  104 µs (0.90x vs torch SDPA — up from 0.33x) and confirmed `mlp_gate_up.s2048`'s recorded golden pins at 2300 µs
  (1.25x vs cuBLAS). Neither reaches a default deploy yet: the golden deploy tier misses these shapes (loud
  enumeration-drift warning on hd512, silent miss on the matmul) — tracked in the golden-sweep findings report.
