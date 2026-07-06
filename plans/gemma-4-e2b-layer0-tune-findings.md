# Gemma-4-E2B layer-0 dynamic tune findings — RTX 4080, 2026-07-05

**Status:** single-layer (`--layer 0`) dynamic tune complete **after a trace-blocker fix** (finding 0 — emmy had no
Per-Layer-Embedding support; the single-layer wrapper was patched to supply a synthetic `per_layer_input`). The
**full-layer -O3 e2e is UNOBTAINABLE** — `k_linear_reduce_8f622a` (k_proj main) hung >1 s during CUDA-graph capture
(finding 1), so there is no deployable layer number; the per-kernel picture is below. Unlike the qwen3-embedding layer,
attention here is **fine** (one fused MMA-flash kernel, 81 µs); the costs are the **non-MMA projection / PLE linears**
(finding 2) and a **compile-budget search pathology on the flash attention** (finding 3). Single-layer scope, generative
model → no serving A/B (skill Step 2b) and no servable artifact.

- **Command:** `emmy tune google/gemma-4-E2B --layer 0 --dynamic seq_len@x:1 --clean --bench --dump-dir
  _tune/tune-model-gemma-4-e2b-l0-4080/dump` (caches isolated via `EMMY_TUNE_DB` / `EMMY_PRIOR_FILE` /
  `EMMY_CUBIN_CACHE` — the global 525 MB golden-sweep DB / prior / cubin were not touched). Required a one-line-ish
  patch to `emmy/compiler/trace/huggingface.py` (finding 0) — **uncommitted**.
- **Run stats:** tune wall **1699.0 s (~28.3 min)**, 1 fused terminal; **686 ok / 17 `bench_fail`** DB rows (11 of the
  17 are the flash-attention compile-budget cluster, finding 3). Prior: 1148 benches, **calibration Spearman +0.69**
  (notably worse than qwen3-emb's +0.97 — the bench_fail-heavy attention search degrades the fit).
- **Dynamic run: symbolic `seq_len`, all numbers at the 512 hint** (`DEFAULT_SEQ_HINT`).
- **-O3 vs -O1:** `eval variants` **`-O3 us`** column + the `--bench` reproducer numbers are **-O3** (deployable); bare
  `us` columns are **-O1** ranking signal only. Not mixed here.
- **NCU unavailable:** `ncu` → `ERR_NVGPUCTRPERM` (perf-counter permission gate) on this box. Triage uses `eval
  variants` / `eval failures` + emitted CUDA. Re-run with counters enabled to quantify occupancy claims.

## Finding 0 — emmy had no Per-Layer-Embedding (PLE) support; the single-layer trace was blocked (FIXED this run)

Gemma-4 E2B/E4B are MatFormer "nano" models with **Per-Layer Embeddings**: every decoder layer computes `hidden *
per_layer_input` ([modeling_gemma4.py:1449](../venv/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py#L1449)).
Emmy's single-layer wrapper ([huggingface.py:230](../emmy/compiler/trace/huggingface.py#L230)) passed only `x` +
`position_embeddings`, so `per_layer_input` was `None` → `TypeError: unsupported operand type(s) for *: 'FakeTensor'
and 'NoneType'`. `grep -rn per_layer_input|altup|laurel emmy/` is empty — the whole nano arch family was untraceable
single-layer. (The dense gemma-4-12B has no PLE, which is why `plans/gemma-4-12b-layer0-tune-findings.md` traced fine.)

**Fix (this run, uncommitted):** `build_layer_wrapper` now registers a synthetic `per_layer_input` buffer of shape
`[1, S, hidden_size_per_layer_input]` (randn, seed 0, non-uniform so the mul isn't folded to identity) and slices it
in-graph like cos/sin, passing it only when the block exposes `hidden_size_per_layer_input`. Non-PLE architectures take
the byte-identical original path. **Perf-representative** — the PLE gate/mul/projection/norm kernels depend on shape,
not values, and the accuracy check stays valid (emmy and torch see the same buffer). **Caveat carried through the whole
report: the PLE-path kernels (finding 2's `k_linear_reduce_b02c13`, the `k_linear_pointwise` mul) are traced against a
synthetic per-layer embedding — their SHAPES / latencies are real, their numerics are not the deployed model's.**
If this fix is kept, it needs the full contribution checklist (wrapper docstring, trace `ARCHITECTURE.md`, a test).

## Bench results

### Full layer-0 forward, -O3 — **UNOBTAINABLE**

`[tune] full-model bench failed (bench worker error: HungKernelError("kernel 'k_linear_reduce_8f622a' did not complete
within 1000 ms")); continuing to per-kernel`. k_proj's main slice hangs during the CUDA-graph capture, so there is **no
deployable layer e2e number** (finding 1). The reference `plans/qwen3-embedding-06b-tune-findings.md` hit the same class
of failure at whole-model scope; here it happens at single-layer scope.

### Per-kernel -O3 (18 kernels). **`repro µs` = standalone `run --ir`; `in-layer µs` = `eval variants` `-O3 us` pick.**

For **partial-coverage** kernels the reproducer re-lowers the whole op standalone and is wildly inflated (finding 5) —
**trust `in-layer µs`**. Sorted by `in-layer µs` (the deployable truth), largest first:

| kernel (hash)                          | layer op                                  | cover | in-layer µs | repro µs | tier        |
|---|---|---|---|---|---|
| k_linear_reduce_b02c13                 | PLE gate + projection (gelu, synthetic)   | part  |    **1059** |     1225 | non-MMA r4  |
| k_linear_reduce_2da798 (+__partial)    | q_proj main (7/8)                         | part  | **676**/695 |      738 | non-MMA g2a |
| k_linear_mean_reduce_2e7682            | pre-ffn RMSNorm → MLP gate/up (gelu)      | full  |     **669** |      667 | MMA (slow)  |
| k_linear_reduce_d80bdd                 | down_proj main (7/8)                       | part  |     **376** |      373 | MMA         |
| k_linear_reduce_8f622a (+__partial)    | k_proj main (7/8) — **hangs (finding 1)** | part  | **166**/147 |      165 | non-MMA g4a |
| k_linear_sdpa_reduce_464c19__partial   | o_proj (7/8) + SDPA-tail(1/17)            | part  |     **110** |      864 | MMA         |
| k_scaled_dot_product_attention_reduce  | fused flash attention (softmax + Q@K/P@V) | main  |      **81** |       81 | **MMA-flash** |
| k_mean_linear_reduce_{9aa814,6ca3b9,…} | q/k-norm→proj & post-ffn-norm slices (1/8)| part  |     1.3–7.6 | 41–38460 | —           |
| k_mean_04e695                          | input RMSNorm stats                        | full  |         7.5 |        7 | —           |
| k_linear_pointwise_b86e0c              | v_proj(1/8) + PLE mul (synthetic)         | part  |         1.0 |      164 | —           |
| k_{cat_,}slice_unsqueeze_pointwise ×4  | rotary layout                              | full  |         1–4 |      1–4 | —           |

**Dominating deployable kernels (in-layer -O3):** PLE gate/projection (**1059 µs**, non-MMA), q_proj main
(**676+695 µs**, non-MMA), MLP gate/up (**669 µs**, MMA-but-slow), down_proj (**376 µs**, MMA). Attention is **not** a
problem here (81 µs, 0.43× eager). The three biggest are the **non-MMA projection/PLE linears** (finding 2).

## Finding 1 — full-layer e2e unobtainable: k_proj main hangs >1 s in CUDA-graph capture

**Blocks the headline deployable number.** The `--bench` full-layer pass compiles the layer with the greedy/prior picks
and captures a CUDA graph; `k_linear_reduce_8f622a` (k_proj main, 7/8 slice) `did not complete within 1000 ms`, aborting
the capture. Its own `eval variants` pick is only **166 µs -O3** (rank 5/107, `n16x16/f4x4 g4a`), so a >1 s hang is a
lowering/variant defect, not the kernel's nominal cost — the full-layer lowering evidently selected a hanging variant
(the prior picks structurally and can land on a config the per-kernel search never validated). **Diagnostic:** re-lower
just the layer and bisect which k_proj variant hangs; `eval variants --kernel k_linear_reduce_8f622a --top 0` lists 107
configs — the hanging one is likely a masked-`seq_len` split-K (`g8a`/`g4a`) tile whose boundary guard loops
unboundedly at the 512 hint. **Priority: high** — without it there is no deployable layer number, and a hanging kernel
in a captured graph is a correctness-adjacent risk. Variance caveat: re-run once to confirm the hang reproduces (scalar
split-K picks swing run-to-run).

## Finding 2 — projection & PLE linears deploy non-MMA (the largest deployable costs)

**µs at stake: ~2.4 ms of the (unmeasurable) layer — the top three deployable kernels.** Several matmuls deploy on the
`n…` vector / split-reduce path instead of tensor cores (`eval variants` TILE column, no `a:mma`):

| kernel | layer op | in-layer -O3 | deployed tile | MMA? |
|---|---|---|---|---|
| k_linear_reduce_b02c13 | PLE gate + projection | 1059 µs | `REDUCE=r4` (split reduce) | no |
| k_linear_reduce_2da798 | q_proj main (7/8) | 676 µs | `n16x8/f4x8 g2a` | no |
| k_linear_reduce_8f622a | k_proj main (7/8) | 166 µs | `n16x16/f4x4 g4a` | no |
| k_linear_sdpa_reduce_464c19 (main) | o_proj main (7/8) | (-O3 —) | `n16x8/f4x4 g8a` | no |

Contrast: **down_proj** (`k_linear_reduce_d80bdd`, 376 µs) and the **MLP gate/up** (`k_linear_mean_reduce_2e7682`,
669 µs) DO deploy MMA (`a:mma_m16n8k16_f16`), and the o_proj `__partial` companion is MMA (110 µs) while the o_proj
**main** slice is non-MMA — so the split-K decomposition routes the *main* tile of these projections onto the vector
path while (sometimes) only the remainder gets tensor cores. Same tier-miss as
`plans/qwen3-embedding-06b-layer0-tune-findings.md` finding 6, broader here (q/k/o-proj + the PLE projection). At
~1 TFLOP/s-class efficiency for `[512,H]×[H,·]` f16, these are far under the 4080 roofline. **Root-cause hypothesis
(needs confirmation):** `030_split`'s split-K decomposition of a reduce-epilogue linear emits the main tile on the
non-MMA `g?a`/`g?k` path; find the offer in
[`030_split.py`](../emmy/compiler/pipeline/passes/lowering/tile/030_split.py) / `005_split_demoted.py` and confirm the
main tile has no MMA atom offered. **A/B via a full-layer re-tune with an MMA fork pinned** (the isolated reproducer
already picks a different form — finding 5). **Priority: high** — biggest deployable slice. The PLE row is on the
synthetic path (finding 0 caveat) but its shape is the deployed model's.

## Finding 3 — flash-attention MMA tiles bench_fail on the 4 s compile budget; deploy is saved by the prior, DB coverage is not

**Symptom: 11 of 21 attention configs `bench_fail` (the largest cluster), and `eval variants` reports the pick as a
169 ms scalar — but the deployed kernel is an 81 µs MMA-flash.** `k_scaled_dot_product_attention_reduce` is recognized
as ONE fused flash kernel (unlike qwen3-emb's un-fused scalar attention), and the search DOES enumerate tensor-core
flash tiles (`TILE@dd`/`TILE@pj` = `a:mma_m16n8k16_f16`, ranks 1–4 at ~530 µs -O1). But `eval failures` shows those MMA
variants fail with **`compile stage exceeded 4.0 s budget (4.03 / 5.78 / 4.11 s)`** (plus hangs / >2 s GPU-time), so
they never get a valid -O3 latency. The best *surviving* DB row is a scalar fallback at **169140 µs -O1** (rank 19/21),
which `eval variants` marks as the "pick — 319× of best".

That pick marker is a **DB artifact**. The actual deployed kernel — both the dump lowering and a fresh reproducer
re-lower — is **MMA-flash** (`mma.sync` ×3, `__launch_bounds__(32)`) at **80.8 µs** (0.43× eager 34.6, tcompile 34.7).
The prior picks the flash tile *structurally*; outside the tune's tight per-variant 4 s budget it compiles fine. So:

- **Deploy is OK** (81 µs MMA-flash) — attention is not a deployable problem here.
- **Search is not**: 11 wasted slots, the DB has no valid latency for the deployed config (only a 169 ms scalar
  survives), the prior calibration drops to +0.69, and `eval variants` mis-reports the pick. If the prior ever had to
  fall back to the DB, it would deploy the 169 ms disaster.

**Root cause:** the tune's per-variant budgets — [`tune.py:159`](../emmy/commands/tune.py#L159)
`CudaBackend(bench_compile_timeout_s=4.0, bench_run_timeout_s=2.0, bench_wall_timeout_s=8.0)`, raised at
[`program.py:701`](../emmy/compiler/backend/cuda/program.py#L701) — are too tight for the big MMA-flash kernels
(nvcc/cicc on the two-contraction flash body takes 4–6 s; the 2 s run cap and 1 s hang timeout account for the other
attention fails). **Fix (medium-high):** raise/relax the compile budget for flash-tier variants, or compile them once
and cache
(the reproducer proves they compile + run fast), so the tensor-core flash tiles survive the search and the DB reflects
the deployable truth. **Repro (compile-only, shows the MMA body):** `EMMY_KNOBS="" emmy compile
_tune/tune-model-gemma-4-e2b-l0-4080/dump/08_lowering_cuda.kernels/k_scaled_dot_product_attention_reduce.torch.json
--ir cuda | grep -c mma.sync` → 3.

## Finding 4 — MLP gate/up: MMA but ~6× behind torch.compile; -O1 mis-ranks the pick

`k_linear_mean_reduce_2e7682` (pre-ffn RMSNorm → gate/up + gelu, full coverage) deploys MMA at **669 µs -O3** vs eager
228 / tcompile 113 — **~6× behind tcompile**. It carries **6 `bench_fail`** variants (`>2 s GPU-time` / hang, shared
knobs `FAST_EXP=False, INTERLEAVE_LOADS=True, VECTORIZE_LOADS=True` — same signature as qwen3-emb finding 5). The `eval
variants` pick is **rank 80/115 at -O1** ("3.11× of best") but its **-O3 (669 µs) ≈ rank-1's -O3 (688 µs)** — so the
-O1 ranking mis-orders and the "misses best" warning is a false alarm; the prior picks -O3-well. Reproducer 667 µs ≈
in-layer 669 µs (full coverage → reproducer trustworthy). **Class 3 (codegen quality):** right tier, still behind
cuBLAS/tcompile. NCU compare (blocked here) would localize it; likely the fused norm-prologue + gate/up epilogue costs
occupancy. **Priority: medium** (669 µs, but attention-free layer so it's a real slice).

## Finding 5 — reproducer table unreliable for partial kernels; worst inflation ~30000× (measurement)

Same trap as `plans/qwen3-embedding-06b-layer0-tune-findings.md` finding 3, more extreme. `run --ir` re-lowers a
partial kernel's *whole* op standalone; for the norm→proj(1/8) slices the reproducer is inflated up to **~30000×** vs
the deployed in-layer cost:

| kernel | coverage | repro µs | in-layer -O3 | inflation |
|---|---|---|---|---|
| k_mean_linear_reduce_155c25 | linear 1/8 | 38460 |   1.3 | **~30000×** |
| k_mean_linear_reduce_9aa814 | linear 1/8 |   683 |   5.1 |   ~130× |
| k_linear_pointwise_b86e0c   | linear 1/8 |   164 |   1.0 |   ~160× |
| k_linear_sdpa_reduce_464c19 | SDPA 1/17  |   864 |   110 |    ~8× |
| k_linear_mean_reduce_2e7682 | **full**   |   667 |   669 |  1.0× ✓ |
| k_mean_04e695               | **full**   |     7 |   7.5 |  1.0× ✓ |

The `--bench` per-kernel table and `62_kernel_bench.json` record only the inflated reproducer number with **no coverage
flag** — the 38460 µs row reads as the layer's worst kernel when it is a 1.3 µs slice. **For partial kernels trust the
`eval variants` `-O3 us` column.** (Proposed CLI fix in Workflow notes — this is the second report to hit it.)

## Repro / artifacts

- Tune log: `_tune/tune-model-gemma-4-e2b-l0-4080/tune.log`; dump: `_tune/tune-model-gemma-4-e2b-l0-4080/dump`
  (reproducers under `08_lowering_cuda.kernels/`; `62_kernel_bench.json`; `kernels.html`). Isolated caches under the
  same work dir; source `env.sh` before any `eval`/`run`/`compile` so they read this run's DB + prior.
- Trace-blocker fix (uncommitted): `git diff emmy/compiler/trace/huggingface.py`.
- Attention MMA-flash proof (no GPU): `EMMY_KNOBS="" emmy compile
  _tune/tune-model-gemma-4-e2b-l0-4080/dump/08_lowering_cuda.kernels/k_scaled_dot_product_attention_reduce.torch.json
  --ir cuda | grep -c mma.sync` → 3.
- In-layer per-kernel truth: `source …/env.sh && emmy eval variants --kernel <substr>`; failure clusters: `emmy eval
  failures`.

## Workflow notes

- **The 4 s per-variant compile budget silently discards the tensor-core flash tier** (finding 3). The MMA-flash
  attention compiles + runs fast (81 µs) but takes 4–6 s to nvcc, so all its variants `bench_fail` and the DB keeps only
  a 169 ms scalar. This is invisible unless you cross-check the dump/reproducer against the `eval variants` pick — the
  pick marker actively lied. **Proposed:** (a) a per-variant compile-budget override / a longer budget for flash-tier
  kernels, and (b) `eval variants` should flag when the marked pick is a `bench_fail`-adjacent fallback (i.e. the prior's
  structural pick has no valid DB latency) rather than silently marking a surviving scalar row as "the pick".
- **Reproducer inflation for partial kernels bit again, worse (30000×)** — same as the qwen report's finding 3/workflow
  note; still unfixed. **Proposed (repeat):** add the coverage fraction + the in-layer `eval variants -O3` to
  `62_kernel_bench.json` and the `--bench` table, and asterisk any row whose reproducer/in-layer ratio > ~2×.
- **Full-layer `--bench` dies on the first hanging kernel** (finding 1) with no partial table — the qwen whole-model
  report flagged this too. A per-kernel skip-and-continue for the full-layer capture (like the tune's `bench_fail`
  pinning) would salvage a partial e2e. **Proposed:** capture the layer graph kernel-by-kernel with a per-kernel
  timeout, report the sum-minus-hung with the hung kernel flagged.
- **Arch-support gaps surface as a raw `TypeError` mid-trace** (finding 0), not a clean "unsupported arch feature X"
  message. `grep per_layer_input emmy/` being empty was the tell. **Proposed:** the layer wrapper could introspect the
  block's forward signature and either supply known auxiliary inputs (per_layer_input, and eventually AltUp/Laurel
  state) or raise a named "arch feature not supported: PLE" error pointing at the wrapper.
- **NCU still blocked** (`ERR_NVGPUCTRPERM`) — findings 2/4 occupancy claims are hypotheses from emitted CUDA, not
  measured. Enabling perf counters would let `--profile` settle class-2-vs-3 on the non-MMA projections.
- **No run-to-run flakiness beyond the k_proj hang** (finding 1, confirm it reproduces before acting) and the attention
  compile-budget cluster. Single 28-min run; the per-kernel GPU-time caps contained the hangs without wedging the device.
