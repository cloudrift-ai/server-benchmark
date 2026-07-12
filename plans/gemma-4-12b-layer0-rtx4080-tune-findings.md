# gemma-4-12B layer-0 tune findings — NVIDIA GeForce RTX 4080 (sm_89)

**Status:** emmy is **≥2.4× slower than eager** on this layer, dominated by ONE kernel — the fused
RMSNorm→gate_up MLP megakernel (~7.4 ms, ~65 % of the layer), which is **locked out of cp.async pipelining**
(`d1/sync`, A/B-proven). The headline e2e ratio and most of the per-kernel table are **unreliable on this card**
(bench artifact + degenerate-fast tune rows); the megakernel is the one number that survives cross-checking and it
is the finding that matters.

**Run command (cold, one invocation):**

```
EMMY_O3_TOL=0.10 emmy tune google/gemma-4-12B --layer 0 --dynamic seq_len@x:1 --clean --bench \
    --dump-dir _tune/gemma4-12b-layer0-4080/dump
```

**Date:** 2026-07-10  **GPU:** RTX 4080 16 GB (sm_89)  **Base:** latest `main` `f7f35a38` (#339 f16-accumulate mma
atom) — deliberately, since gemma-4-12B is fp16 and #339 changes the mma atom the projections ride.
**Scope:** single layer, **dynamic** (`--dynamic seq_len@x:1`), benched at **seq_len=512** (symbolic `Dim` hint).
Single-layer scope ⇒ no servable artifact (no vLLM serving A/B) — not applicable here.
**Run stats:** wall ~39 min (trace/front-end ~10, autotune 1565 s / 774 benches, -O3 rebench ~3); **7 bench_fail
clusters** — the fused megakernel (`k_linear_mean_reduce`, 8 rows: hang >1 s / >2 s GPU-time) and wide attention
(`k_scaled_dot_product_attention_reduce`, 3 rows), isolated in the bench-worker subprocess.

**-O1 vs -O3 disclaimer:** the `--bench` tables below are the **-O3** deployable re-bench; any tune-DB latency
quoted for ranking context is `-Xcicc -O1` and labeled as such. The two families are never compared directly.

## Bench results

**Layer end-to-end** (`--bench`, benched at seq_len=512 symbolic hint; torch inputs tiled to match):

| Backend | Latency (µs) | vs Eager |
| --- | --- | --- |
| Eager PyTorch | 3110 | 1.00× |
| torch.compile | 2827 | 1.10× |
| **Emmy** | **11657** | **0.27×** |

**The 0.27× is a lower bound polluted upward** — see Finding 3. The single trustworthy dominant term is the
megakernel at ~7.4 ms, which alone puts the layer at ≥2.4× eager.

**Per-kernel** (`--bench` -O3 reproducer table, sorted by emmy µs; **Layer op** from each `.torch.json` provenance;
**tune-DB -O3** = the same kernel's best -O3 row in the tune DB, for the cross-check in Finding 3):

| Kernel | Layer op | eager | tcompile | emmy (repro) | tune-DB -O3 | agree? |
| --- | --- | --- | --- | --- | --- | --- |
| `k_linear_mean_reduce` | **fused RMSNorm→gate_proj+up_proj→GeLU-gate** (~120 GFLOP) | — | — | **7582** | 7451 | ✅ real |
| `k_mean_linear_reduce` ×5 | post-FFN norm / scale + linear fragments | 1078 / 465 / — | 747 / 186 | 2189…338 | **3–7** | ❌ 100–600× |
| `k_linear_reduce` (f1b366) | **mlp.down_proj** (15360→3840) | 732 | 727 | 2165 | 2158 | ✅ real |
| `k_linear_reduce` (eb26a6) | self_attn.q_proj (→4096) | — | — | 515 | 461 | ✅ real |
| `k_linear_reduce` (cdc109) | self_attn.k_proj (→2048) | — | — | 335 | 311 | ✅ real |
| `k_linear_sdpa_reduce` | attention QK·V reduce | 223 | 219 | 458 | — | — |
| `k_scaled_dot_product_attention_reduce` | flash attention | 52 | 52 | 73 | — | 0.70× |
| `k_linear_pointwise`, `k_mean`, `k_*_slice_unsqueeze_pointwise` | rope / mean / slice epilogues | 54–262 | 3–20 | 3–8 | — | ✅ wins (12–44×) |

Dominating term: **`k_linear_mean_reduce` alone is ~65 %** of the emmy total. The small pointwise/mean kernels
already beat eager 12–44× (memory-bound epilogues emmy fuses well) — one line, no drill-down.

## Finding 1 — fused gate_up megakernel locked to `d1/sync`: cp.async pipelining is refused (~7.4 ms, 65 % of layer)

**Symptom.** `k_linear_mean_reduce_02ef41` benches **7454 µs standalone** (reproducer) and **7451 µs** in the tune
DB (-O3) — they agree, and 7.4 ms is physically consistent for a 120-GFLOP kernel running at ~16 TFLOP/s (~13 % of
the 4080's f16 tensor-core peak). So it is a **genuinely slow kernel**, not the bench artifact of Finding 3.

**What it is.** The `.torch.json` provenance shows the whole MLP up-path fused into one kernel: `add_7` (residual
[1,seq,3840] f16) → `pow²`→`mean`→`+eps`→`rsqrt`→`×`→`× pre_ffn_norm.weight` (**RMSNorm, computed in f32**) →
**`gate_proj` [3840→15360]** and **`up_proj` [3840→15360]** matmuls → `gelu_tanh` → `× up` gate → out
[1,seq,15360] f16. The normalized activations feeding the two matmuls are **f32 computed on-chip**, then cast to
f16 for the `mma_m16n8k16_f16_f32` atom (hence the "float != Half" torch-ref skip on every repro of this kernel).

**Root cause — a class-2 tier lockout, proven.** The `eval variants` leaderboard has **54 configs, every one
`STAGE d1/sync`** — synchronous, single-buffered, no cp.async ring. The pipelined `d2/cp/ring` tier (what the golden
squares use to hit cuBLAS parity) is **never enumerated** for this kernel. And it is not merely un-enumerated: an
explicit `--ab "STAGE=d2/cp/ring"` / `"STAGE=d2/cp"` pin **realizes `STAGE@a3=d1/sync` anyway** (flagged
"unreproducible pin") — the lowering *forces* sync for the matmul stage-group. The reason is correct-by-design: the
matmul's **A-operand is the on-chip-computed normalized row**, and you cannot `cp.async`-copy a value that does not
live in global memory. The fused-prologue staging lives in the `PLACE@cone=fuse` path
([`tile/010_recognize.py`](emmy/compiler/pipeline/passes/lowering/tile/010_recognize.py) `bind_prologue_contraction`
+ [`kernel/_factor.py:767`](emmy/compiler/pipeline/passes/lowering/kernel/_factor.py#L767) "shared-row staging").

**The over-broad part (the actionable gap).** For gate/up the A-operand (normalized activations [512,3840]) is
small and computed, but the **B-operand — the weights [3840,15360] — is a large plain-global load**, and it is the
bandwidth-dominant operand of a 3840→15360 projection. Forcing the *whole* kernel to `d1/sync` because the A-side
is computed throws away the B-side cp.async pipelining that would hide the weight-load latency. The pick
(`w2x1/f4x4/k2`, 163 regs, 25 % occ) is the fastest of the 54 sync configs (its -O3 7451 < rank-1's 7785 — the
search did find the sync-tier optimum), so **the ceiling is the tier gate, not the prior**.

**Fix (high priority — this is 65 % of the layer).** Two options, either recovers most of the ~4–7× headroom:
(a) **stage the plain-global B-operand via cp.async even when the A-operand is a computed prologue** — split the
transport decision per operand instead of one `d1/sync` for the whole stage-group; (b) offer a **de-fused
`PLACE@cone=split` variant** (RMSNorm as its own kernel; gate_up then reads plain-global activations and can
pipeline `d2/cp/ring`) and let the search A/B fuse-vs-split. This matches the prior gemma theme
([[gemma4-12b-rmsnorm-rtx4090-tuning]]: "f32 flash-tail view blocks warp tier") — the fused f32 prologue blocking
the pipelined tier is now confirmed as a *staging* refusal, not just an enumeration gap.

## Finding 2 — the `k_linear_reduce` projections land on a smem-LESS schedule (no tiling, no cp.async) — 3× off cuBLAS

**Symptom.** The standalone projections `down_proj` (f1b366, -O3 2158 µs, `w8x2/f2x8`), `q_proj` (eb26a6, 461),
`k_proj` (cdc109, 311) are all ~2–4× off their FLOP roofline (`down_proj` 60 GFLOP / 2158 µs ≈ 28 TFLOP/s vs
eager's 732 µs ≈ 82 TFLOP/s — **emmy `down_proj` is 3× slower than cuBLAS**).

**Root cause — a smem-less schedule, not `d1/sync` (A/B-refined).** I first read these as sync-tier; the A/B
corrected it. `--ab "STAGE=d2/cp/ring"` on `down_proj` realizes **`STAGE (off)`**, and the deployed kernel uses
**0.0 K shared memory** (grid 60 × 512 threads, 128 regs, 33 % occ). So the `k_linear_reduce` (matmul-as-K-reduction)
form deploys a **register-only, smem-less matmul** — operands stream global→register→mma with **no smem tiling**, so
there is no ring to stage and cp.async is moot (nothing to pipeline into). It reloads operands from global on every
mma step instead of reusing an smem tile — hence ~28 TFLOP/s. This is a *different* class-2 lockout from Finding 1:
Finding 1 is smem-tiled-but-`d1/sync`; Finding 2 is **not smem-tiled at all**. The smem-tiled + cp.async tier the
golden squares ride (`n32x16 … d2/cp/ring`, cuBLAS parity) is **not selected for the reduce-form projections**.

**Not a search shortfall / not a prior miss.** `f1b366`'s pick is `eval variants` rank 22/28 by **-O1** ("misses
best") but its **-O3 (2158) is the fastest of all its measured configs** (rank-1-by-O1's -O3 is 2273) — a
`-O1/-O3` inversion, so the deployed pick is the -O3 optimum *within the offered (smem-less) tier*. The prior
ranked it correctly at -O3; the ceiling is the schedule family, not the prior.

**Fix (high priority).** Find where the `k_linear_reduce` K-reduction form is routed to a smem-less schedule
instead of the smem-tiled `mma` tier (the recognition/scheduling in
[`tile/010_recognize.py`](emmy/compiler/pipeline/passes/lowering/tile/010_recognize.py) `Reduction`/`Contraction`
dispatch + [`kernel/_factor.py`](emmy/compiler/pipeline/passes/lowering/kernel/_factor.py) staging) and enable the
smem-tiled cp.async schedule for it. A 3× projection speedup is the second-largest win after Finding 1.

## Finding 3 — the RTX 4080 bench path is unreliable in BOTH directions; per-kernel attribution and the e2e ratio can't be trusted

**Symptom.** For `k_mean_linear_reduce` the **tune-DB -O3 is 3–7 µs** but the **`--bench` reproducer is
338–2189 µs** — a **100–600× disagreement** on the same kernels. Neither bound is credible: 3–7 µs for a
kernel that references `down_proj.weight` would be ~12 000 TFLOP/s (**physically impossible** — a degenerate-fast
tune row, the roofline-floor problem the golden reports flagged), while the 2189 µs reproducer carries the known
**4080 layer-bench artifact** ([[rtx4080-bench-anomaly]]: a near-constant multi-ms cost attaches to one
kernel launch, position-stable, absent in a clean `--code` bench). Because the e2e (11657 µs) is a single
whole-layer run, it inherits the same artifact — so the **0.27× headline is inflated** and the true emmy standing
is better than 0.27× (but still ≥2.4× eager from the megakernel alone, which is the one term that cross-checks
clean).

**Why it matters.** This blocks clean per-kernel attribution on this card: the `--bench` reproducer table and the
e2e total are both suspect, and the tune-DB has degenerate-fast rows at the other extreme. The only kernels I could
report with confidence are those where **tune-DB -O3 ≈ reproducer -O3** (the megakernel, the three real
projections) — everything with a large disagreement is unresolved.

**Fix (blocks trusting any gemma e2e on the 4080).** (a) reject bench results below a FLOP-roofline floor before
they enter the tune DB (kills the 3–7 µs degenerate rows); (b) root-cause the reproducer/e2e layer-bench artifact
(per the memory, cross-check the suspect kernels via `emmy run --code` at the same shape — a clean-context bench —
and diff against the reproducer path). Until then, treat single-layer gemma e2e numbers on this box as
directional, not deployable.

## Why the prior is not the culprit here

Both dominant findings are **tier/enumeration gates**, not prior mispricing: the megakernel pick is the -O3
optimum of its (sync-only) offered set, and `down_proj` is a `-O1/-O3` inversion whose -O3 pick is best. A
`eval prior --dataset nodes` fork-regret pass would measure ranking *within* the offered tier — but the tier itself
is the ceiling, so a prior refit changes nothing until the cp.async offer/refusal (Findings 1–2) is addressed. No
per-half regret table is included for that reason; if Finding 1(a) lands and `d2/cp/ring` becomes reachable, re-run
the tune and *then* a fork-regret pass is meaningful.

## Repro / artifacts

- Work dir: `_tune/gemma4-12b-layer0-4080/` (tune.log, `dump/`). Golden-sweep prior+DB backed up under
  `_tune/golden-sweep-rtx4080/2026-07-10/` before this run's `--clean`.
- Megakernel tier-lockout proof (no re-tune needed):
  ```
  K=_tune/gemma4-12b-layer0-4080/dump/08_lowering_cuda.kernels/k_linear_mean_reduce_02ef41.torch.json
  emmy run --ir $K --bench --ab "STAGE=d2/cp/ring"     # pin realizes d1/sync — flagged unreproducible
  emmy eval variants --kernel k_linear_mean_reduce      # 54 configs, all d1/sync
  ```
- Source-only (no GPU) inspection of the forced-sync staging:
  `EMMY_KNOBS="STAGE=d2/cp/ring" emmy compile $K --ir cuda` (emits the d1/sync kernel regardless).
- Down_proj offer-gap check: `emmy run --ir …/k_linear_reduce_f1b366.torch.json --bench --ab "STAGE=d2/cp/ring"`.

## Workflow notes

Retrospective for whoever maintains the emmy CLI + this skill. The prior gemma-4-12B report was on a 4090
([`plans/gemma-4-12b-layer0-rtx4090-tune-findings.md`](plans/gemma-4-12b-layer0-rtx4090-tune-findings.md)); its
finding-6 note ("the tune `--bench` per-kernel table misattributes split pairs; the `run --bench` launch-order
table is the antidote") is a *milder cousin* of Finding 3 here — on the 4080 the per-kernel table isn't just
mis-split, it's off by 100–600× from the tune DB.

- **The 4080 bench path is the dominant analysis hazard and it cost the most time.** Every "surprising" per-kernel
  number needed a tune-DB-vs-reproducer cross-check (and for the megakernel a standalone reproducer bench) before I
  could tell a real 7.4 ms kernel from a 2189 µs artifact from a 5 µs degenerate row. *Improvement:* a single
  `eval` view (or a `--bench` column) that prints **tune-DB -O3 next to the reproducer -O3 and flags a >N×
  disagreement** would have collapsed the whole of Finding 3's legwork into one glance — the same shape as the
  golden skill's node-store-vs-fresh cross-check, which this run reconfirms is needed per-card.
- **`--ab` realizing a *different* knob than pinned is the single most useful signal in this run** — it turned
  "cp.async isn't enumerated" (ambiguous: gap vs refusal) into "cp.async is *refused*" (a correctness gate) in one
  command. Keep the "unreproducible pin" flag prominent; it is the tier-lockout detector.
- **`eval variants --kernel` needs the kernel C-identifier, not the layer op** — I dumped provenance from each
  `.torch.json` by hand to label rows "down_proj / q_proj / …". The per-kernel `--bench` table has no Layer-op
  column either. *Improvement:* join the `.torch.json` provenance into both the `--bench` table and `eval variants`
  so kernels self-label by the torch op they realize.
- **Playwright PNG export failed** (`chrome-headless-shell` not installed) — `kernels.html` was written but the PNG
  skipped with a full install banner in the log. Minor, but noisy. *Improvement:* detect the missing browser and
  emit one line, not the framed banner; or drop the PNG step when the HTML suffices.
- **Front-end (trace) was ~10 min of the 39** with no progress logging — a 12B multimodal `from_pretrained` +
  `torch.export` of one layer. It looked stalled (GPU idle, log quiet at "weights loaded"). *Improvement:* a
  `[trace] …` heartbeat between weight-load and the first pipeline dump would distinguish "tracing" from "hung".
