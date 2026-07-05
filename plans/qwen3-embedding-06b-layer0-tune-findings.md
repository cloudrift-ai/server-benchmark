# Qwen3-Embedding-0.6B layer-0 dynamic tune findings — RTX 4080, 2026-07-04

**Status:** single-layer (`--layer 0`) dynamic tune complete; clean cold tune + -O3 re-bench recorded. Full-layer -O3
e2e obtained (emmy **8880 µs** vs eager 342 / tcompile 244 — **0.04× eager**). Attention is the dominant cost — its
scalar softmax→P@V slice alone (6703 µs) exceeds every non-attention kernel combined — and it lowers **un-fused,
scalar-tier** (finding 1). Second is a non-MMA q_proj main slice (finding 6). This is a **single layer**, not the whole
model — the single-layer scope has no servable artifact, so the serving A/B (skill Step 2b) is skipped; the numbers
below are the layer's e2e, not the model's.

- **Command:** `emmy tune Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --clean --bench --dump-dir
  _tune/tune-model-qwen3-emb-l0-4080/dump` (caches isolated into the work dir via `EMMY_TUNE_DB` / `EMMY_PRIOR_FILE` /
  `EMMY_CUBIN_CACHE` so `--clean` did **not** touch the global 525 MB golden-sweep DB / prior / cubin).
- **Run stats:** tune wall **1400.7 s (~23.3 min)**, 1 fused terminal; **364 ok / 3 `bench_fail`** DB rows (the 3
  fails are one MLP-kernel variant cluster, finding 5 — wasted slots, not the deployed pick). Prior: 798 benches
  (warmup 119 / post 679), best 3.226 µs @ #292, **post-Spearman calibration +0.97**.
- **Dynamic run: symbolic `seq_len`, ALL numbers benched at the 512 hint** (`DEFAULT_SEQ_HINT`); torch closures tiled
  to the same hint. The masked-tile boundary guards are part of the measured cost.
- **-O3 vs -O1 disclaimer:** the `--bench` full-layer + per-kernel tables and the `eval variants` **`-O3 us`** column
  are **-O3** (deployable). Bare `us` columns in `eval variants` are the **-O1** tune-ranking signal (reduction/attn
  kernels run 1.5–3× slower at -O1) — quoted for ranking context only. Never compared across the two families here.
- **NCU unavailable:** `ncu` reports `ERR_NVGPUCTRPERM` (perf-counter permission gate) on this box, so occupancy /
  throughput / register counters are absent. Triage below uses `eval variants` / `eval failures` + the emitted CUDA
  instead. Re-run with counters enabled to quantify the occupancy claim in finding 2.

## Bench results

### Full layer-0 forward, -O3 (deployable, CUDA-graph captured; seq_len=512 symbolic hint)

| Backend       | Latency (µs) | vs Eager |
|---|---|---|
| Eager PyTorch |          342 |    1.00× |
| torch.compile |          244 |    1.40× |
| **Emmy**      |     **8880** | **0.04×** |

emmy is **26× behind eager** and **36× behind torch.compile** on the layer. All of that gap is attention (finding 1).

### Per-kernel -O3 (14 kernels). **`repro µs` = standalone `run --ir` re-bench; `in-layer µs` = the `eval variants` `-O3 us` pick.**

The reproducer re-lowers each kernel's `.torch.json` **from scratch**; for a **partial-coverage** kernel (a slice of a
larger op) that re-lowers the *whole* op standalone, so `repro µs` is inflated up to 65× vs what runs in the layer. **For
partial kernels, `in-layer µs` is the truth** (finding 3). Sorted by `repro µs` (the raw `--bench` order):

| kernel (hash)                         | layer op                                | cover | eager | tcomp | repro µs | in-layer µs |
|---|---|---|---|---|---|---|
| k_linear_sdpa_reduce_e23835           | o_proj + SDPA-tail(1/28) + residual     | part  |    66 |    54 | **46938** |    **74** |
| k_sdpa_reduce_6874a2                  | Q@Kᵀ scores (SDPA 11/28)                | part  |    27 |    24 | **41574** |  **2667** |
| k_sdpa_linear_reduce_deb3d9           | softmax + P@V + v_proj(1/8) (SDPA 16/28)| part  |    38 |    38 |  **7117** |  **6703** |
| k_mean_linear_reduce_efb40d           | q_norm → q_proj(1/8)                     | part  |   100 |    20 |      345 |       5.4 |
| k_linear_reduce_ff3c09                | q_proj main slice (7/8, non-MMA)        | part  |    27 |    27 |       70 |  **1463** |
| k_linear_reduce_f8597f                | v_proj main slice (7/8, split-K)        | part  |    15 |    15 |      340 |     ~383  |
| k_linear_mean_reduce_e57b41           | post-attn RMSNorm → MLP gate/up + silu  | full  |   172 |    92 |      253 |     249   |
| k_linear_fc9a1d                       | down_proj + residual                    | full  |    44 |    44 |      102 |    ~101   |
| k_mean_linear_reduce_39ba47           | k_norm → k_proj(1/8)                     | part  |   183 |    34 |       76 |       3.5 |
| k_cat_slice_transpose_… (×2)          | rotary layout                           | full  | 37–56 |   2   |     3–6 |     3–6   |
| k_mean_d65726                         | input RMSNorm stats                      | full  |    81 |    5   |       5 |       5.2 |
| k_slice_transpose_… (×2)              | rotary layout                           | full  | 34–53 |   2–3 |     3–5 |     3–5   |

**Dominating kernels (by in-layer -O3):** `k_sdpa_linear_reduce` (softmax+P@V, **6703 µs**), `k_sdpa_reduce` (Q@Kᵀ,
**2667 µs**), and the non-MMA `k_linear_reduce_ff3c09` (q_proj main, **1463 µs**, finding 6). Attention (6703 + 2667 +
74 = **9444 µs**) is ~80% of the summed per-kernel -O3 and its softmax→P@V slice alone exceeds all non-attention kernels
combined (~2.2 ms). Note the per-kernel -O3 benches sum to ~11.7 ms — more than the 8.88 ms graph-captured layer,
because each includes per-launch overhead the CUDA-graph amortizes — so treat the shares as approximate and the 8880 µs
full-layer number as the deployable truth. The MLP gate/up (249 µs) and down_proj (~101 µs) trail torch.compile
~2.7× / 2.3× but are small; the norms and rotary kernels beat eager.

## Finding 1 — attention is un-fused and scalar-tier: the softmax→P@V reduce is a per-cell scalar contraction (the layer's dominant cost)

**µs at stake: ~9.4 ms (the two big SDPA slices, ~80% of the summed per-kernel -O3; the softmax→P@V slice alone, 6703
µs, exceeds every non-attention kernel combined).** Qwen3 applies RoPE to Q/K, so by the time SDPA is reached
its Q/K are computed SSA, not plain loads. `_flash.py` recognizes flash **only** for a *clean* scaled-QK producer —
"A fused score producer whose Q/K are computed SSA — RoPE'd QK — is NOT recognized as flash (it falls back to its
un-fused tiers)" ([`_flash.py`](../emmy/compiler/pipeline/passes/lowering/tile/_flash.py) docstring). So attention takes
the **un-fused multi-kernel path**, which realizes the one `scaled_dot_product_attention` op as three kernels
(coverage 11/28 + 16/28 + 1/28 = 28/28):

1. **`k_sdpa_reduce` — Q@Kᵀ scores, materialized to gmem.** Uses tensor cores (`dpl_mma_m16n8k16_f16`, 3 mma calls,
   `__launch_bounds__(64)` = 2 warps). 2667 µs -O3. See finding 2.
2. **`k_sdpa_linear_reduce` — softmax + P@V, fully SCALAR.** `mma.sync count: 0`, `__launch_bounds__(256)`. **6703 µs
   -O3 — the single biggest cost.** The emitted body
   (`_tune/tune-model-qwen3-emb-l0-4080/dump/08_lowering_cuda.kernels/k_sdpa_linear_reduce_deb3d9.txt`):

   ```c
   float acc0 = -1e30f;                              // pass 1: row-max over kv
   for (a2 = 0; a2 < seq_len; a2++) acc0 = fmaxf(acc0, score[a2] + mask);
   float acc1 = 0.f;                                 // pass 2: softmax denom over kv
   for (a2 = 0; a2 < seq_len; a2++) acc1 += expf(score[a2] + mask - acc0);
   for (a3 = 0; a3 < 128; a3++) {                    // pass 3: P@V, per head-dim column
     float acc2 = 0.f;
     for (a4 = 0; a4 < seq_len; a4++)
       acc2 += expf(score[a4] + mask - acc0) * V[a4, a3];   // scalar FMA + expf RECOMPUTED per (a3,a4)
     out[…, a3] = acc2 / acc1;
   }
   ```

   Two defects, both **known and documented** as design limitations:
   - **P@V is a per-cell scalar contraction, not tensor-core.** `_flash.py:29`: "flash lowers **only** on the scalar
     tier below — there is no divergent flash codegen path" (a tensor-core flash tier that attaches an mma `TilePlan`
     to the Q@K / P@V `Contraction` nodes and routes through the one `_factor` path is a stated follow-up).
     `_atomize.py:21`: "**Flash contractions are not recursively atomized … lowered on the scalar tier (block=1)**."
   - **The softmax probability `expf(score − max)` is recomputed once per head-dim column** — 128× redundant exp over
     the kv axis. `_flash.py:51`: the scalar tier "runs one independent streaming softmax **per output element**
     `(…, m, d)` — a correct, if redundant, form."
3. **`k_linear_sdpa_reduce` — o_proj matmul + residual.** Tensor cores (27 MMA configs), 74 µs. Healthy.

**Root cause (not a bug — a missing tier):** RoPE'd QK disqualifies flash fusion, and even the flash path has no
tensor-core tier yet, so the softmax→P@V reduce emits scalar with a 128× redundant `expf`. Eager's fused flash does
this in ~0.1 ms; emmy takes ~6.7 ms → **~65× behind on the dominant slice.**

**Repro (compile-only, no GPU — inspect the scalar body):**
`EMMY_KNOBS="" emmy compile _tune/tune-model-qwen3-emb-l0-4080/dump/08_lowering_cuda.kernels/k_sdpa_linear_reduce_deb3d9.torch.json --ir cuda`
(note: `run --ir` on this reproducer re-lowers the whole SDPA standalone — see finding 3.)

**Suggested fix (highest priority):** land the tensor-core flash tier (mma `TilePlan` on the Q@K / P@V `Contraction`
nodes through `_factor`) **and** extend flash recognition to a RoPE'd (computed-SSA) Q/K producer so Qwen-style
attention fuses instead of falling back to the materialized-score un-fused path. Short of that, at minimum hoist the
per-column `expf` out of the P@V head-dim loop (compute P once over kv, reuse across the 128 columns) — a pure
redundant-compute win independent of the tier work.

## Finding 2 — Q@Kᵀ score kernel: tensor-core but 1 measured config, 2 warps, materialized scores

**µs at stake: 2667 µs/layer (~30%).** `k_sdpa_reduce` (Q@Kᵀ, 11/28) does use MMA, but `eval variants --kernel
k_sdpa_reduce` shows **exactly 1 measured config** — the search never explored warp-count / tiling alternatives — and
the emitted kernel is `__launch_bounds__(64)` (2 warps) holding 32 accumulator fragments per warp
(`_c0_0`…`_c3_7`), a register-heavy 2-warp launch that almost certainly runs at low occupancy. Because attention is
un-fused, this kernel also **materializes the full `[S,S]` score matrix to gmem** for `k_sdpa_linear_reduce` to re-read
— bandwidth the fused flash path would avoid.

**Diagnostic to settle codegen-quality (class 3) vs search-shortfall (class 1):** NCU occupancy + regs/thread (blocked
here by `ERR_NVGPUCTRPERM`). The single-config observation is itself strong class-1 evidence: the terminal search
spent its budget on the 27-config o_proj tail and the 93-config MLP while giving the two dominant attention reduces one
config each. **Fix:** enumerate warp-count / K-split forks for the un-fused attention score + reduce kernels (they are
the highest-µs kernels yet the least-searched), or subsume them into finding 1's fused flash tier.

## Finding 3 — the per-kernel reproducer table is unreliable for partial-coverage kernels (measurement finding)

**Not a kernel defect — a measurement-semantics trap that would misdirect every subsequent drill-down.** `run --ir
<k>.torch.json` re-lowers the kernel's torch graph **standalone**. For a **partial-coverage** kernel the `.torch.json`
holds the *entire* op it sliced, so standalone it re-lowers the whole op (losing the in-graph split/fusion), and the
reproducer latency reflects a kernel that **never runs in the layer.** Proven by the `eval variants` `-O3 us` (in-layer)
column:

| kernel | coverage | `repro µs` | in-layer `-O3 µs` | inflation |
|---|---|---|---|---|
| k_sdpa_reduce_6874a2        | SDPA 11/28  | 41574 | 2667 | **16×** |
| k_linear_sdpa_reduce_e23835 | SDPA 1/28   | 46938 |   74 | **634×** |
| k_mean_linear_reduce_39ba47 | linear 1/8  |    76 |  3.5 | **22×** |
| k_mean_linear_reduce_efb40d | linear 1/8  |   345 |  5.4 | **64×** |
| k_linear_mean_reduce_e57b41 | **full**    |   253 |  249 |  1.02× ✓ |
| k_mean_d65726               | **full**    |     5 |  5.2 |  1.0× ✓ |
| k_linear_fc9a1d             | **full**    |   102 | ~101 |  1.0× ✓ |

Full-coverage kernels match within noise; partial ones are inflated 16–634×. The cross-check that catches it: the
three SDPA reproducers sum to ~95 ms but the whole layer is 8.88 ms. **Rule: for partial-coverage kernels trust the
`eval variants` `-O3 us` column, not the `--bench` reproducer table or `62_kernel_bench.json`** (the latter records the
inflated reproducer numbers with no coverage/partial flag — see Workflow notes).

## Finding 4 — -O1 tune-ranking mis-ranks vs -O3; the prior's picks are actually -O3-good (false "misses best")

The `eval variants` "pick: rank R/N … misses best" warning is computed on the **-O1** `us` column, which mis-orders the
MMA kernels vs their deployable **-O3** cost:

- `k_linear_sdpa_reduce`: pick is **rank 26/27 at -O1** (344 µs, "1.74× of best") but its **-O3 is 74 µs — the fastest**
  of any shown row (rank-1's -O3 is 144 µs). The prior picked the -O3 winner; the -O1 rank is a false alarm.
- `k_linear_mean_reduce` (MLP gate/up): pick rank 3/93 at -O1 (518 µs) but **-O3 249 µs beats rank-1's -O3 304 µs.**

So the prior is picking well; the -O1 ranking signal is just noisy for register-tile MMA kernels (exactly the class the
-O1 `-Xcicc` unroll change perturbs most). **Recommendation:** rank the leaderboard on the `-O3 us` column when present
(or flag that "misses best" is an -O1 artifact), so the warning stops firing on -O3-optimal picks.

## Finding 5 — 3 `bench_fail` variants on the MLP gate/up kernel (wasted search slots, class 4)

`eval failures`: **3 bench_fail rows** all on `k_linear_mean_reduce_e57b41` (post-attn RMSNorm → MLP gate/up), error
`benchmark run stage exceeded 2.0s of GPU time — variant marked bench_fail`, shared knobs `FAST_EXP=False,
INTERLEAVE_LOADS=True, VECTORIZE_LOADS=True`. The kernel's **deployed pick is fine** (249 µs -O3, rank 3/93) — these are
3 wasted search slots on a variant that hangs, not a deployability problem. **Fix:** gate this knob combination out of
enumeration for this kernel shape (or lower the per-variant GPU-time cap so it fails faster and frees the slot). Low
priority — 3 slots of ~360.

## Finding 6 — the q_proj main slice deploys a non-MMA reduce at 1463 µs (second-largest kernel; the greedy pick skipped its own MMA companion)

**µs at stake: ~1.5 ms — the largest cost after the two attention slices.** q/v projection is split 7/8 (main) + 1/8
(fused into the q/k-norm kernels). `eval variants --kernel k_linear_reduce_ff3c09` (q_proj main) shows the **deployed
pick is `n16x8/f2x2` — a non-MMA vector tile — at 1463 µs -O3** (rank 6/23). Yet the same op has an MMA-tiled companion
piece `k_linear_reduce_ff3c09__partial` whose pick is `a:mma_m16n8k16_f16/w2x1/f4x4` at **70 µs -O3** (rank 1/53). So
q_proj realizes as a fast MMA partial (70 µs) **plus** a slow non-MMA main (1463 µs) — the split-K decomposition routed
the bulk of the q_proj matmul onto a vector/scalar reduce path instead of tensor cores. v_proj (`k_linear_reduce_f8597f`)
is the milder version: ~383 µs -O3, also `n…`/split-K. At ~5 TFLOP/s for a [512,1024]×[…] f16 matmul, the main slice is
far under the 4080 roofline.

This is also a second `run --ir` reproducer trap (finding 3) in the *opposite* direction: the `ff3c09` reproducer
benched at **70 µs** (it re-lowered to the fast MMA form), **hiding** the 1463 µs non-MMA kernel that actually deploys.
Only the `eval variants` `-O3 us` column surfaces it.

**Root-cause hypothesis (needs confirmation):** the `030_split` split-K decomposition of a reduce-epilogue linear emits
its main tile on the non-MMA `g2a`/`g2k` vector path while only the remainder gets an MMA offer — a tier/pick miss.
**Diagnostic:** compare the tile IR of `ff3c09` vs `ff3c09__partial`
(`_tune/tune-model-qwen3-emb-l0-4080/dump/08_lowering_cuda.kernels/k_linear_reduce_ff3c09.txt`) to confirm the main tile
has no MMA atom offered; then A/B via a **full-layer re-tune with an MMA fork pinned** (not the isolated reproducer,
which already picks the MMA form). **Priority: medium** — 1.5 ms is real, but it is dwarfed by finding 1's 9.4 ms.

## Repro / artifacts

- Tune log: `_tune/tune-model-qwen3-emb-l0-4080/tune.log`; dump: `_tune/tune-model-qwen3-emb-l0-4080/dump`
  (reproducers under `08_lowering_cuda.kernels/`, machine-readable per-kernel bench in `62_kernel_bench.json`, chart in
  `kernels.html`). Isolated caches: `_tune/tune-model-qwen3-emb-l0-4080/{autotune.db,prior.json,cubin}`.
- Re-run any command in the tune's environment by first sourcing `_tune/tune-model-qwen3-emb-l0-4080/env.sh` (sets the
  three isolated `EMMY_*` paths so `eval` / `run` read this run's DB + prior, not the global golden-sweep ones).
- Inspect the scalar softmax→P@V body (no GPU):
  `source _tune/tune-model-qwen3-emb-l0-4080/env.sh && EMMY_KNOBS="" emmy compile \
   _tune/tune-model-qwen3-emb-l0-4080/dump/08_lowering_cuda.kernels/k_sdpa_linear_reduce_deb3d9.torch.json --ir cuda`
- In-layer per-kernel truth (the numbers to trust for partial kernels):
  `source …/env.sh && emmy eval variants --kernel k_sdpa_reduce` (and `k_sdpa_linear_reduce`, `k_linear_sdpa_reduce`).
- Failure cluster: `source …/env.sh && emmy eval failures`.

## Workflow notes

Friction hit during this pass, for whoever maintains the CLI + skill:

- **`62_kernel_bench.json` and the `--bench` per-kernel table record only the standalone reproducer latency, with no
  coverage/partial flag** — the single biggest trap this run (finding 3). Two SDPA reproducers read 41–47 ms while the
  kernels run 74 µs / 2667 µs in the layer; without the `eval variants` `-O3 us` cross-check the report would have
  claimed a 95 ms layer that benches at 8.88 ms. **Proposed:** add the coverage fraction (already in `<k>.torch.txt`)
  **and** the in-layer `eval variants` `-O3 us` to `62_kernel_bench.json` and the `--bench` table, and warn/asterisk any
  row whose reproducer/in-layer ratio exceeds ~2× ("partial coverage — reproducer re-lowers the full op"). This is the
  same class of gap that motivated `eval variants` originally.
- **The standard Step-4 drill-down (`run --ir <reproducer> --bench --ab …` / re-tune the reproducer) does not work for a
  split/partial kernel** — it re-lowers the whole op, so the A/B and re-tune measure a kernel that never ships. The
  dominant kernel here is exactly such a case, so knob A/Bs were not actionable. **Proposed:** a `run`/`tune` mode that
  targets the in-layer kernel slice (by hash) inside the full-layer lowering, rather than the standalone reproducer —
  so A/Bs and patience re-tunes apply to what actually deploys. Until then the skill should say: for partial kernels,
  A/B via a **full-layer re-tune with the fork pinned**, not the isolated reproducer.
- **`eval variants` "misses best" fires on -O1 ranking even when the pick is -O3-optimal** (finding 4) — false alarms
  on every register-tile MMA kernel. **Proposed:** rank on `-O3 us` when the column is populated.
- **NCU is gated by `ERR_NVGPUCTRPERM` on this box** — the occupancy/register claim in finding 2 is inferred from
  `__launch_bounds__(64)` + fragment count in the emitted CUDA, not measured. Enabling the perf-counter permission
  (`nvidia-smi`/module param) would let `--profile` quantify it. Noted, not fought, per the skill.
- **Cache isolation worked cleanly** — pointing the three `EMMY_*` vars into `_tune/…` let `--clean` run a genuine cold
  tune without nuking the 525 MB global golden-sweep DB. Worth promoting to a documented pattern / a `tune --work-dir`
  convenience flag that sets all three at once (the skill's "fresh dir under `_tune/`" step currently only covers the
  dump dir, not the DB/prior/cubin).
- **No flakiness this run** — the tune's per-kernel GPU-time cap caught the 3 hanging MLP variants (finding 5) without
  wedging the device; single 23-min run, exit 0, no retries.
