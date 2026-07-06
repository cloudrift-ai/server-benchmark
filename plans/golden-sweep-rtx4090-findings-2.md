# Golden sweep — RTX 4090 (sm_89), 2026-07-06

Supersedes `golden-sweep-rtx4090-findings.md` (2026-06-19, pre-#293 Tile-IR revision on the archived
`feature/golden-sweep-rtx4090` branch); kept as a sibling for the older-revision history, not merged into this one.

**GPU:** NVIDIA GeForce RTX 4090 (sm_89, 128 SMs), driver 580.65.06, CUDA 12.9. **Code:** `main` @ `2a256f7a`
(previous 4090 goldens seeded/refreshed at `272153d3`, before the #293 Tile-IR rebuild). **Relevant commits since the
seed:** #293 (new `TILE/REDUCE/STAGE` knob codec), #305/#306/#307 (scalar-staging / TMA-gate / flash fixes), #308 (4080
sweep adopting `d2/cp/ring` + `g2k`).

**Sweep command:**
```
emmy tune --dataset golden --clean          # 34/34 shapes, 0 failures/drops; trains the learned prior
emmy run --bench --golden NAME              # per-shape greedy-vs-golden A/B at -O3 (29 recorded matmul shapes) ×2
emmy run --bench -c "<snippet>"             # seed bench for the 5 pointwise/reduce shapes (no 4090 golden yet)
```
**Wall time:** tune ~2 h (34 shapes incl. the heavy fp16 + large squares), A/B pass-1 ~11 min, confirmation pass-2 ~5
min (kernels warm in cache), seed ~1 min. All logs under the gitignored `_tune/golden-sweep-rtx4090-main/` (`tune.log`,
`ab-pass1.log`, `ab-pass2.log`, `seed.log`, `seed-dump/`).

**Category tally: 18 replaced · 2 added · 5 seeded (new pointwise/reduce) · 6 unchanged · 3 worse (left).**
- **18 replaced** (greedy >3% faster than the golden row, confirmed in *both* A/B passes): 6 squares
  (512/1024/2048/4096 fp32 + 1024/2048 fp16) and 12 qwen3_06b projections. Recorded greedy's knobs + `emmy_us`;
  each supersedes a slower `g2a`/atomic entry (deleted in place — one config per name).
- **2 added** (within 3%, different knobs — at-parity alternates): `kv_proj.s128`, `down_proj.s512` — greedy's
  `g2k`/`d2/cp/ring` config added alongside the recorded `g2a` one.
- **5 seeded**: `reduce.{2048x2048,1024x512,2048x128}` (all `REDUCE: b32`, 1.2–2.1× cuBLAS) and
  `pointwise.{2048x2048,512x4096}` (`TILE: n128x8/f1x8|f1x16`, cuBLAS parity). These had **no 4090 entry** — they were
  recorded only for the RTX 5090, which is why `run --bench --golden` refused them (see Workflow note 1).
- **6 unchanged**: `gate_up_proj.s512` + `q_proj.s512` (greedy reproduces the golden knobs), `square.4096.fp16`
  (same knobs, 5% delta is noise), `kv_proj.s32` + `o_proj.s32` (within noise, ~1.01×), `down_proj.s512.dynM`
  (flaky — 0.93×/0.98× across the two passes, did not clear 3% in both).
- **3 worse** (greedy slower — left untouched): `gate_up_proj.s32`, `square.512.fp16` (real prior shortfalls, Findings
  1–2) and `o_proj.s128` (noise-band, Finding 3).

## Per-shape outcome — slowest-vs-golden first

`greedy` / `golden` are both **live -O3 re-benches** (the pass-2 A/B). `ratio` = greedy÷golden-row. `cuBLAS` = recorded
`cublas_us` (config-independent; live Eager agreed within ~5%); `vsCB` = greedy÷cuBLAS (>1 = emmy slower than cuBLAS).
The two seeded families use the live `run --bench -c` numbers (integer-rounded Eager — Workflow note 3).

| shape | greedy µs | golden µs | ratio | cuBLAS µs | vsCB | category |
|---|---|---|---|---|---|---|
| qwen3_06b.gate_up_proj.s32 | 13.7 | 12.2 | 1.12 | 11.3 | 1.21 | **worse** |
| square.512.fp16 | 7.1 | 6.7 | 1.06 | 5.8 | 1.22 | **worse** |
| qwen3_06b.o_proj.s128 | 22.4 | 21.5 | 1.04 | 19.0 | 1.18 | **worse** (noise) |
| qwen3_06b.o_proj.s32 | 12.5 | 12.4 | 1.01 | 9.9 | 1.26 | same |
| qwen3_06b.kv_proj.s32 | 7.4 | 7.3 | 1.01 | 6.9 | 1.07 | same |
| qwen3_06b.gate_up_proj.s512 | 87.7 | 87.9 | 1.00 | 85.5 | 1.03 | same (same knobs) |
| qwen3_06b.q_proj.s512 | 55.3 | 55.5 | 1.00 | 53.3 | 1.04 | same (same knobs) |
| qwen3_06b.down_proj.s512 | 78.4 | 81.1 | 0.97 | 113.2 | 0.69 | **added** |
| qwen3_06b.kv_proj.s128 | 12.6 | 12.8 | 0.98 | 12.4 | 1.02 | **added** |
| square.4096.fp16 | 876.5 | 924.7 | 0.95 | 822.3 | 1.07 | same (same knobs) |
| qwen3_06b.q_proj.s32 | 8.4 | 8.8 | 0.95 | 9.9 | **0.85** | better |
| square.512 | 14.4 | 15.2 | 0.95 | 10.8 | 1.33 | better |
| square.2048.fp16 | 119.4 | 124.9 | 0.96 | 115.2 | 1.04 | better |
| square.4096 | 2845.7 | 3044.4 | 0.93 | 2458.6 | 1.16 | better |
| qwen3_06b.kv_proj.s512 | 34.3 | 36.9 | 0.93 | 38.9 | 0.88 | better |
| qwen3_06b.down_proj.s32 | 17.2 | 18.9 | 0.91 | 13.0 | 1.32 | better |
| qwen3_06b.gate_up_proj.s512.dynM | 92.4 | 101.1 | 0.91 | 85.6 | 1.08 | better |
| qwen3_06b.o_proj.s512 | 56.0 | 61.6 | 0.91 | 67.8 | **0.83** | better |
| qwen3_06b.q_proj.s512.dynM | 61.9 | 67.8 | 0.91 | 53.0 | 1.17 | better |
| qwen3_06b.gate_up_proj.s128 | 31.6 | 36.5 | 0.87 | 25.5 | 1.24 | better |
| square.2048 | 359.1 | 429.1 | 0.84 | 320.0 | 1.12 | better |
| qwen3_06b.q_proj.s128 | 16.8 | 21.1 | 0.80 | 20.2 | **0.83** | better |
| square.1024 | 60.1 | 76.5 | 0.79 | 45.4 | 1.32 | better |
| qwen3_06b.kv_proj.s512.dynM | 28.8 | 37.0 | 0.78 | 37.0 | **0.78** | better |
| square.512.dynM | 11.5 | 16.6 | 0.69 | 10.8 | 1.06 | better |
| qwen3_06b.down_proj.s128 | 30.9 | 43.2 | 0.72 | 24.5 | 1.26 | better |
| qwen3_06b.o_proj.s512.dynM | 51.2 | 77.1 | 0.66 | 67.1 | **0.76** | better |
| square.1024.fp16 | 22.2 | 34.2 | 0.65 | 18.1 | 1.23 | better |
| reduce.1024x512 | 2.0 | — | — | 4.0 | **0.50** | seeded (new) |
| reduce.2048x128 | 2.1 | — | — | 4.0 | **0.53** | seeded (new) |
| reduce.2048x2048 | 4.7 | — | — | 6.0 | **0.78** | seeded (new) |
| pointwise.512x4096 | 5.0 | — | — | 5.0 | 1.00 | seeded (new) |
| pointwise.2048x2048 | 8.9 | — | — | 9.0 | 0.99 | seeded (new) |

## Key result — the fresh prior deploys `g2k` split-K + `d2/cp/ring` over the goldens' `g2a` atomic

The recorded goldens were transferred/older configs built almost entirely on the **atomic** split-K reduce (`g2a`,
single fused kernel) at assorted stages (`d1/cp`, `d3`, `d4`). This sweep's warm-prior greedy pick moved nearly every
winning shape to **k-split** (`g2k`, a `__partial` matmul + a tiny epilogue reduction kernel) on the unified
`d2/cp/ring` stage. That one structural shift — split-K + `d2/cp/ring` — is what drives all 18 replacements; the fp32
squares additionally drop the goldens' oversized tiles (`n64x16`, `n16x8`) for `n32x{8,16}`. The two-kernel `g2k`
program's total (matmul + epilogue) is what the recorded `emmy_us` now reflects (whole-op, so it stays comparable to
the whole-op `cublas_us`). This matches the #308 RTX 4080 sweep, which adopted the same `d2/cp/ring` + `g2k` regime.

Against cuBLAS the picture is the usual sm_89 split: emmy **beats** cuBLAS on the memory-bound tall/skinny projections
(`kv/o/q_proj.s512(.dynM)` 0.76–0.88×, the reduces 0.50–0.78×) and sits at parity on pointwise, but still **trails**
cuBLAS SGEMM/HGEMM on the compute-bound squares (1.1–1.3×) and the small fused shapes.

## Finding 1 — `gate_up_proj.s32`: the learned prior ranks the golden *deeper* than the cold analytic (1.12×)

The M=32 gate/up shape is the one non-marginal fp32 loss. Greedy deploys `n16x16/f2x2` + default reduce + `d2/cp/ring`;
the golden is `n32x8/f2x4` + `g2a` + `d4/cp/ring` — **all three knobs wrong** (`eval golden` m/t = 0/3). The evidence
says this is a *ranking* miss, not a reachability or -O3-inversion miss:

- `eval analytic`: golden ranks **19 / 2428** — mispriced but shallow.
- `eval prior --dataset golden`: golden ranks **32 / 2428** — the learned prior ranks it *worse* than the cold
  analytic did. Training this sweep pushed the true golden further down while the greedy pick (rank 1) is 15% slower.

So the learned prior actively regressed on this shape. This is the same pattern as Finding 2 and is the sweep's main
structural finding (see below). **Recommendation:** the small-M fused shapes need exploration, not just a heuristic
tweak — an ε-greedy tune pass (`--explore-eps`, as the `collect-node-data` flow uses) would let the prior *measure* the
`g2a`/`d4` sibling instead of committing to its extrapolated `g2k`/`d2` pick. A secondary lever is an engineered
feature separating the atomic-vs-split-K payoff at small M (the prior can't currently see why `g2a` wins here).

## Finding 2 — `square.512.fp16`: prior ranks the golden in the bottom half of the pool (1.06–1.23×)

The only fp16 loss, and the starkest ranking failure of the sweep. Greedy deploys
`a:mma_m16n8k16_f16/w1x2/f2x4/k2` + `d2/cp/ring/p2`; the golden is `.../w2x1/f1x4/k2` + `d4/cp/ring`.

- `eval analytic`: golden ranks **833 / 9884** — the analytic weights badly misprice the small fp16 tile.
- `eval prior`: golden ranks **5536 / 9884** — *bottom half*. The learned prior is worse than a coin flip at
  surfacing this config; the `run --bench` A/B shows 1.06× but the prior's own bench of its pick vs golden is 1.23×.

The fp16-512 regime (tiny M/N/K, warp-MMA tier) is the pool where both priors are weakest. **Recommendation:** refit
the analytic weights over the recorded goldens (`scripts/golden_knob_heuristics.py` — it emits `_W_A` / `_W_A_DYN` to
paste into `search/prior/analytic.py`) so the cold rank for the small fp16 tier is not 800+, and pair it with the same
ε-greedy exploration as Finding 1. Note the analytic refit is a global (cross-GPU) change and should land as its own
validated PR, not bundled with this YAML update.

## Finding 3 — `o_proj.s128`: noise-band, not a real loss

Flagged "worse" by the A/B (1.04× in both passes) but this is inside the small-shape noise floor, not a shortfall: the
`eval prior` re-bench of the *same* greedy pick vs golden shows **0.98×** (greedy faster). The greedy config
(`n16x8/f2x8` + `g2k`) and the golden (`n16x16/f2x4` + `g2a`) are within measurement noise (~19–22 µs). Left the golden
untouched; recording either is defensible. Worth noting the analytic ranks this golden **300 / 2428** — the same
small-shape analytic mispricing as Findings 1–2, just without a real perf consequence here.

## Cross-cutting — the learned prior regressed vs the cold analytic on the small shapes

Findings 1–3 share one shape: on every small fused shape the **learned prior ranks the true golden deeper than the
cold `AnalyticPrior` does** (gate_up.s32: 32 vs 19; square.512.fp16: 5536 vs 833; o_proj.s128 is the one exception,
analytic 300 worse than prior 64). This is the signature of a greedy prior over-committing to its own extrapolated
regime (`g2k` + `d2/cp/ring`, which genuinely wins the other 18 shapes) and under-sampling the atomic/`d4` siblings that
win at tiny M. The fix is exploration during tune, not a bigger enumeration — the goldens are all in the pool, they
just rank too deep for greedy patience to reach.

## Workflow notes

- **`run --bench --golden` can't seed a shape with no live-card entry.** The 5 pointwise/reduce goldens exist only for
  the RTX 5090, so `--golden` (which scopes to the live card, `compile.py:resolve_golden_arg`) reported them "unknown"
  *while listing them as Available* (the fallback list is the un-scoped `GOLDEN_CONFIGS`). Confusing. **Fix:** when a
  name resolves in `GOLDEN_CONFIGS` but not for the live card, say so explicitly ("recorded for RTX 5090, not this
  card — use `run --bench -c` to seed") instead of the generic "unknown".
- **Pointwise greedy pick carries no `TILE` knob, but the golden schema requires one.** The deployed pointwise kernel
  uses a knob-less default tile (`block_threads=None`); `test_golden_configs` asserts `c.knobs` is non-empty. Forcing
  the 5090's `TILE=n128x8/f1x8|f1x16` reproduced the greedy pick *byte-for-byte* (identical grid/block/latency), so
  that's what got recorded — but it took a dump inspection + a forced-knob re-bench to discover the default's TILE
  string. **Fix:** surface the resolved `TILE` for pointwise in the `run --bench` kernel table (it has no knob columns
  today), or let a pointwise golden omit `knobs` when the default tile is the pick.
- **The `Eager PyTorch` row is integer-rounded**, which is coarse for the sub-10 µs reduce/pointwise kernels (a "4 µs"
  eager could be 3.5–4.5). Fine for the golden flag (all ratios clear the classification), but imprecise as a recorded
  `cublas_us`. **Fix:** print the eager latency to one decimal.
- **Categorization needed a corrected parser.** The naive A/B parse took `min()` across a stanza's two golden rows and
  picked the tiny split-K *epilogue* kernel (1.9–3.3 µs) as "the golden," making four wins look like impossibly-fast
  degenerate replays. The golden *program* total is the sum of its rows; comparing greedy `TOTAL` vs golden-sum is the
  correct A/B. **Fix:** have `run --bench --golden` print a single `golden TOTAL` line (as it does for greedy) so no
  downstream summing is needed.
- **Confirmation pass-2 was cheap (~5 min for 29 shapes)** because the tune left every kernel warm in the cubin cache —
  the noise-floor re-runs the skill mandates cost almost nothing here. Greedy picks were steady within ±1% across the
  two passes; only the *golden re-bench* rows swung (down_proj.s512.dynM 0.93↔0.98), exactly the small-shape noise the
  step-4 re-run is meant to catch. One shape (`down_proj.s512.dynM`) was demoted from "better" to "unchanged" on that
  basis.
