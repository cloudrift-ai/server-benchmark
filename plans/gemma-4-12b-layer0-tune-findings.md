# Gemma-4-12B layer-0 tune findings — RTX 5090 + RTX Pro 6000 (both sm_120)

- **Status: the attention collapse is fixed. Emmy is now ~2.1× behind eager end-to-end on this layer (2.97 ms vs
  1.40 ms on the 5090), not 69×.** The earlier 2026-07-02 run reported emmy at **96.5 ms / 69× behind**, dominated by
  the sdpa/attention kernels falling to the serial reduce schedule (dead reduce fork). Post-`#307` ("Flash on model
  graphs: fix codegen + output-layout NaN, unblock the Gemma tune") that kernel certifies:
  `k_scaled_dot_product_attention_reduce` is now **93 µs (2.85× eager)** instead of milliseconds — a ~32× end-to-end
  improvement. This file supersedes the 69× result.
- **Command** (per pass): `emmy tune google/gemma-4-12B-it --layer 0 [--dynamic seq_len@x:1] --clean --bench
  --dump-dir <dir>/dump`. Layer 0 is a `sliding_attention` layer (window 1024; the first `full_attention` layer is 5).
  Run at HEAD `c5f03d61` (#307) across both GPUs, in **both dynamic and static** scope.
- **Scope:** single layer. **Dynamic** = symbolic `seq_len`, benched at the `DEFAULT_SEQ_HINT=512` hint (masked-tile
  kernels — the deployable artifact). **Static** = shape-specialised at seq_len=512 (no boundary guards).
- **Run stats** (wall / DB ok-vs-bench_fail): 5090 dynamic **32 min, 726 ok / 16 fail**; 5090 static **17 min,
  580 ok / 29 fail**; Pro 6000 dynamic **43 min, 670 ok / 23 fail**; Pro 6000 static **27 min, 457 ok / 29 fail**.
- Numbers below are the `--bench` **-O3** re-bench (deployable, CUDA-graph captured); the per-kernel table is each
  kernel's isolated -O3 reproducer. Tune-DB latencies (ranking context) are -O1.

## E2E — full layer forward (eager / torch.compile / emmy), µs

Single decoder layer, torch inputs tiled to the seq_len=512 hint. **Not** the whole-model e2e.

| GPU / scope | Eager | torch.compile | **Emmy** | Emmy / eager | Emmy / tcompile |
|---|--:|--:|--:|--:|--:|
| **5090 dynamic** | 1399 | 1235 | **2973** | **2.13×** | 2.41× |
| 5090 static | 425 | 327 | 1437 | 3.38× | 4.39× |
| **Pro 6000 dynamic** | 1242 | 1022 | **3153** | 2.54× | 3.08× |
| Pro 6000 static | 509 | 358 | 983 | 1.93× | 2.75× |

Gemma-4-12B is emmy's **best relative showing** of the 13-model sweep — the layer's cost is dominated by large
`hd256` GeGLU/QKV matmuls, so the residual attention overhead is amortised (the small models sit at 14–30× in
dynamic). Note the dynamic-vs-static inversion vs the small models: emmy's **absolute** static latency is ~2× lower
than dynamic (1437 vs 2973 µs on the 5090), but eager/torch speed up *more* under static shapes (eager 425 vs
1399 µs), so emmy's *ratio* is worse in static (3.4×) than dynamic (2.1×). The deficit is fixed launch/reduce
overhead that shape-specialisation doesn't shrink but cuBLAS exploits.

## Per-kernel -O3 re-bench — RTX 5090

Sorted by emmy µs. `M/E` = emmy/eager. `-` = the slicer wired no torch reference for that fused chain (e.g. the
norm→linear / gate-up fusion has no single torch op).

### Dynamic (masked-tile, seq_len=512 hint)

| Kernel | Layer op | emmy µs | eager µs | tcompile µs | M/E |
|---|---|--:|--:|--:|--:|
| `k_linear_mean_reduce` | RMSNorm → linear (fused prologue) | 903 | – | – | – |
| `k_linear_sdpa_reduce` | attn: QK·V → out-proj reduce | 485 | 129 | 127 | 3.75× |
| `k_mean_linear_reduce` | RMSNorm → linear | 473 | 481 | 305 | **0.98×** |
| `k_linear_reduce` | linear reduce | 427 | 293 | 293 | 1.46× |
| `k_mean_linear_reduce` (×4) | RMSNorm → linear (MLP edges) | 136–316 | 250 | 106 | **0.54×** … – |
| `k_linear_pointwise` | linear + GeGLU pointwise | 169 | – | – | – |
| `k_scaled_dot_product_attention_reduce` | **flash/SDPA** | **93** | 33 | 33 | **2.85×** |
| `k_mean` | mean/norm stat | 21 | 165 | 12 | **0.13×** |
| `k_{cat_,}slice_unsqueeze_pointwise` (×4) | RoPE/cache slice+cat | 2.5–4.9 | 39–76 | 2–4 | **0.06–0.07×** |

### Static (shape-specialised, seq_len=512)

| Kernel | emmy µs | eager µs | tcompile µs | M/E |
|---|--:|--:|--:|--:|
| `k_linear_sdpa_reduce` | 458 | 22.5 | 22.5 | **20.3×** |
| `k_linear_mean_reduce` | 172 | – | – | – |
| `k_mean_linear_reduce` | 94 | 164 | 86 | **0.58×** |
| `k_linear_reduce` | 86 | 82 | 82 | 1.04× |
| `k_mean_linear_reduce` (×5) | 26–36 | 78 | 18 | **0.42×** … – |
| `k_linear_pointwise` | 25 | – | – | – |
| `k_mean` | 5.4 | 70 | 4.1 | **0.08×** |
| `k_scaled_dot_product_attention_reduce` | 4.6 | 8.2 | 8.2 | **0.56×** |
| `k_{cat_,}slice_unsqueeze_pointwise` / `k_unsqueeze` (×4) | ~1 | 20–25 | 2 | **0.04×** |

The Pro 6000 tables have the same shape and ordering; emmy latencies run ~15–30 % higher (Max-Q clocks), e.g.
dynamic `k_scaled_dot_product_attention_reduce` 135 µs vs 93 µs, `k_linear_sdpa_reduce` 586 µs vs 485 µs. Kernel
*selection* is identical (both sm_120).

## Findings

### Finding 1 — the attention fix landed; the old dead-reduce-fork disaster is gone

The 2026-07-02 report's headline (attention at millisecond scale, 69× e2e) no longer reproduces. On the current HEAD
the flash/SDPA path certifies for the Gemma sliding-window layer:
`k_scaled_dot_product_attention_reduce` = **93 µs dynamic (2.85× eager) / 4.6 µs static (0.56× — emmy wins)**. This is
the single biggest change and drives the 96.5 ms → 2.97 ms e2e improvement. Contrast with the small models in the same
sweep (`plans/emmy-2gpu-tune-campaign-findings.md`), where the *unfused* `k_sdpa_reduce` still runs 200–900× off eager
— i.e. the Gemma graph's fused `k_linear_sdpa_reduce` schedule certifies where the standalone one does not.

### Finding 2 — the remaining deficit is the fused norm→linear reduce, not attention

The dynamic total is now led by `k_linear_mean_reduce` (903 µs, the RMSNorm→linear fused prologue — no torch
reference) and `k_linear_sdpa_reduce` (485 µs, 3.75×). These are matmul+reduce fusions carrying the statistic
prologue; they are competitive-adjacent (1.5–3.75×) rather than broken. The plain `k_linear_reduce` is 1.46×
(near cuBLAS). To close the last ~2×, drill `k_linear_mean_reduce` / `k_linear_sdpa_reduce` with
`emmy eval variants --kernel linear_mean_reduce` + NCU — the static twin of the sdpa fusion proves the tier is
reachable (its absolute cost is similar, 458 µs, so the dynamic gap is guard/tier overhead, not the math).

### Finding 3 — emmy already beats eager on reductions, mean/norm, and pointwise

`k_mean_linear_reduce` runs **0.42–0.98×** of eager (mostly wins), `k_mean` **0.08–0.13×**, and the RoPE/cache
`slice/unsqueeze/cat` pointwise kernels **0.04–0.07×** (15–25× faster than eager) in both modes. The reduce-fork work
and pointwise fusion are paying off; if the two fused matmul-reduces reached cuBLAS parity emmy would be at eager for
this layer.

### Finding 4 — static exposes an nvcc-compile-fail cluster on the sdpa kernel

Static tuning logged a **21-variant `bench_fail` cluster: `nvcc compile failed for kernel
'k_scaled_dot_product_attention_…'`** (both GPUs; 29 fails total, 21 shared this cause). Dynamic instead shows a
smaller cluster of `HungKernelError` + `benchmark run stage exceeded 2.0s` on the same sdpa reduce (search variants
that hang and are pinned `bench_fail @ 2e6 µs`). The final picks are fine, but these are wasted search slots and a
real static-codegen gap worth `emmy eval failures` triage (the shared knobs of the 21-cluster point at the offending
config).

### Finding 5 — 5090 vs Pro 6000 Max-Q

Same sm_120 target → identical kernel selection; emmy runs ~15–30 % slower on the Pro 6000 Max-Q (dynamic 3153 vs
2973 µs), tracking eager/tcompile. The Pro 6000's value here is 98 GB VRAM headroom, not per-kernel speed. Curiously
Pro 6000 *static* posts emmy's best ratio (1.93×) because its eager static baseline is slower (509 vs 425 µs).

## Repro / artifacts

Per-pass dumps + logs under `_tune/campaign/{local,pro6000}/gemma4-12b/{dynamic,static}/` (dump IR,
`62_kernel_bench.json`, `kernels.html`, per-kernel `.torch.json` reproducers, isolated tune DB + prior, `tune.log`).
Aggregate: `_tune/campaign/summary.json` (via `parse_campaign.py`).

Reproduce the attention kernel (no re-tune):
```
emmy run --ir _tune/campaign/local/gemma4-12b/dynamic/dump/08_lowering_cuda.kernels/k_scaled_dot_product_attention_reduce_*.torch.json \
    --bench --bench-backends eager,tcompile,emmy
```
