# Golden sweep findings — NVIDIA GeForce RTX 5090 (sm_120)

**Date:** 2026-07-09  **GPU:** NVIDIA GeForce RTX 5090 (`rtx5090_sm120.yaml`)  **Where:** rented CloudRift box
(`riftuser@176.124.69.202`), repo rsynced + `make setup`, tuned + A/B'd remotely, data harvested to
`_tune/golden-sweep-5090/`.

**Sweep command (cold, one invocation):**

```
emmy tune --dataset golden --explore-eps 0     # deterministic; fresh box ⇒ cold
```

**Wall time:** the `remote_node_tune` poller hit its 14400 s (4 h) cap before the tune finished, but the **on-box tune
ran to completion — `done: 36/36 shape(s)`** (the poller giving up doesn't kill the tune; verified live via `pgrep`
and the Monitor's completion marker). sm_120 (Blackwell) `nvcc` compiles are slower and the attention shapes are ~10 min
each (640 s for `hd64`), so the 5090 ran past 4 h despite having *fewer* shapes than the 4090 (36 vs 42). **Action item:
the driver `--timeout` must scale with shape count / arch.**

**A/B method:** identical to the 4090 sweep — 2 `--json` passes, live -O3, medianed. `spread%` is the 2-pass greedy
spread (all ≤3% here).

## Category tally (36 shapes)

| Category | N | Action |
| --- | --- | --- |
| same-diffknob (≤3%, different-but-equivalent knobs) | 22 | none (already at parity) |
| **GREEDY_FASTER** (sweep beat the recorded golden >3%) | 4 | **2 recorded, 2 deferred** |
| **GREEDY_SLOWER** (prior couldn't reach the golden >3%) | 10 | findings |

Note the large `same-diffknob` bucket (22): the 5090's greedy picks reproduce the golden *latency* within ≤1% but with
a slightly different knob spelling (e.g. `d2/tma/ring` vs a `FAST_EXP/VECTORIZE_LOADS`-decorated equivalent), so they
land as "different knobs, same speed" — healthy, no action.

**Recorded (2 replacements — beat both the live golden AND the recorded µs, clean mapping, ~0% spread):**

| shape | old TILE/STAGE → new | emmy_us | vs live golden |
| --- | --- | --- | --- |
| `matmul.square.2048` | `w4x1/f4x8 · d4/tma` → `w2x4/f2x2/k4 · d2/tma` | 101.95 → **88.5** | 0.72× |
| `matmul.o_proj.h4096` | `w8x2/f2x4/k4 · d2/tma` → `w4x2/f2x4/k2 + g4k · d2/tma` | 101.54 → **90.1** | 0.78× (split-K) |

**Deferred, NOT recorded:**

- `matmul.mlp_gate_up.h4096.dynM` (greedy 663.9 vs live golden 768.1 = 0.86×) — but the golden's *recorded* 596.3 <
  greedy 663.9; the same config now re-benches 768.1 (codegen drift), so recording greedy would raise emmy_us against a
  number the golden no longer hits. Left for a codegen look, like the 4090's `square.1024`.
- `matmul.square.512` (0.96×, 4.6% over live golden) — marginal, inside the noise band.

## Fork sibling regret (this sweep's own prior, `eval prior --dataset nodes`, -O1)

```
family:  PLACE+R+S+T 1.03×   +WSPEC 1.09×   REDUCE 1.00×   STAGE 1.01×   TILE 7.20×   (ALL median, 69 forks)
worst TILE forks:  free=4096 red=14336 → 24643×(*),  free=4096 red=4096 → 20.8×,  free=512 red=512 → 1.99×
leaf reachability: mean 9.08×(*)  median 1.01×  worst 71.92×(*)   |   leaf calibration (Spearman): (per-op, healthy)
(*) inflated by degenerate-fast benched variants — see Finding 3.
```

**One-line diagnosis:** the **TILE fork is catastrophically mispriced on sm_120 — median 7.20×** (vs the 4090's 1.48×),
and `eval golden` confirms the greedy matched the golden **TILE in 0/17** matmul shapes (STAGE 3/17, REDUCE 6/8). The
5090 adds a Blackwell-only `WSPEC` (warp-spec) fork (1.09×) and leans on TMA staging (`d*/tma/ring`), enlarging the
search space the cold prior must rank — and it ranks the warp tile essentially at random. REDUCE/PLACE stay ~optimal.

## Per-shape outcome table (live -O3 A/B, greedy vs best live golden)

| shape | cuBLAS µs | greedy µs | golden µs | greedy/golden | greedy/cuBLAS | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| matmul.mlp_down.h4096.dynM | 291 | 578.7 | 301.7 | **1.92** | 1.99 | SLOWER (finding) |
| matmul.o_proj.h4096.dynM | 96 | 175.0 | 94.4 | **1.85** | 1.82 | SLOWER (finding) |
| matmul.qkv.h4096.dynM | 247 | 459.4 | 263.1 | **1.75** | 1.86 | SLOWER (finding) |
| matmul.square.1024 | 15 | 22.2 | 15.9 | 1.39 | 1.52 | SLOWER |
| matmul.qkv.h4096 (static) | 253 | 346.7 | 262.8 | 1.32 | 1.37 | SLOWER |
| matmul.square.512.fp16 | 6 | 5.6 | 4.4 | 1.27 | 0.92 | SLOWER |
| attention.hd128.dynM | 18 | 20.2 | 15.9 | 1.27 | 1.09 | SLOWER |
| attention.hd64 (static) | 10 | 10.2 | 8.4 | 1.21 | 0.99 | SLOWER |
| matmul.mlp_down.h4096 (static) | 291 | 337.5 | 301.5 | 1.12 | 1.16 | SLOWER |
| attention.hd64.dynM | 10 | 9.0 | 8.7 | 1.04 | 0.87 | SLOWER |
| matmul.square.2048 | 98 | 88.5 | 123.6 | **0.72** | 0.89 | FASTER → **recorded** |
| matmul.o_proj.h4096 | 95 | 90.1 | 115.3 | **0.78** | 0.94 | FASTER → **recorded** |
| matmul.mlp_gate_up.h4096.dynM | 558 | 663.9 | 768.1 | 0.86 | 1.19 | FASTER (deferred: drift) |
| matmul.square.512 | 12 | 8.3 | 8.7 | 0.96 | 0.67 | FASTER (deferred: marginal) |
| (softmax/reduce/rms_norm/pointwise, static+dynM) | — | — | — | ~1.00 | 0.25–1.07 | same (healthy) |

Memory-bound kinds crush cuBLAS on the 5090 (`reduce.k2048` 0.25× cuBLAS, `softmax.k8192` 0.71×) and reproduce their
goldens to ≤1%. All shortfalls are matmul/attention.

## Finding 1 — TILE fork ranked at random on sm_120; `.dynM` matmuls up to 1.9× (worst: `mlp_down.h4096.dynM`)

The masked-tile (`.dynM`) matmuls are ~2× their goldens (`mlp_down.dynM` 1.92, `o_proj.dynM` 1.85, `qkv.dynM` 1.75) and
even the *static* `qkv.h4096` (1.32×) and `mlp_down.h4096` (1.12×) miss. `eval golden` TILE match **0/17** — the cold
prior never picks the golden warp tile on this card. Golden ranks under the learned prior are deep
(`mlp_down.h4096.dynM` 3376/7433, `square.2048.fp16` 4370/5696). The 5090's TMA-staged (`d*/tma/ring`) golden configs
plus the extra `WSPEC` fork make the space the analytic prior must price much larger than sm_89's, and its hand-coded
warp-tile weights are simply wrong for Blackwell.
**Recommendation (high priority):** refit the analytic weights on the sm_120 goldens specifically
(`scripts/golden_knob_heuristics.py`) — the sm_89 `_W_A`/`_W_A_DYN` fits do not transfer to sm_120 (different WSPEC/TMA
fork structure). Consider a per-arch weight set keyed on the stamped WSPEC/TMA features. This is a weight problem (TILE
is the sole regret carrier at 7.20×; REDUCE/STAGE/PLACE are ≤1.03×), not a patience problem.

## Finding 2 — split-K + TMA co-selection (the two recorded wins show the target region)

Both recorded wins came from the search finding a **split-K + TMA** combo the prior disfavored: `square.2048` →
`w2x4/f2x2/k4` (k4 split in the tile) @ 88.5 (0.72×); `o_proj` → `w4x2/f2x4/k2 + g4k` @ 90.1 (0.78×). The recorded
goldens used wider single warps (`w4x1/f4x8`, `w8x2/f2x4`) that lose to the narrower-warp + higher-split configs on
Blackwell. So the *analytic prior systematically over-weights wide warps and under-weights split-K on sm_120* — the same
axis as Finding 1, visible from the winning side.
**Recommendation:** fold these two into the `golden_knob_heuristics.py` refit as positive examples; re-run the whole
sweep after the refit to see whether Finding 1's `.dynM` losses recover once the warp-aspect/split weights move.

## Finding 3 — degenerate-fast benched variants corrupt the sm_120 node store

`eval prior --dataset nodes` reports leaf reachability **mean 9.08×, worst 71.92×** and a TILE fork regret of
**24643×** on `free=4096 red=14336`. These are **not real** — the "best" baselines are physically impossible
(`free=4096 red=14336` best 9.17 µs for a ~60 GFLOP fp16 matmul ⇒ ~6700 TFLOP/s; the 5090 peaks ~210). Some benched
kernel variants return early / mis-measure and clock absurdly fast, poisoning the node store's value-of-position
minimum. The A/B (greedy vs recorded golden) is unaffected — it's the ground truth used above — but the node-store
diagnostics are unusable as-is for the big matmuls on this card.
**Recommendation (medium):** reject variants that bench below a FLOP-roofline floor (or below `cublas_us × 0.4`) before
they enter the `node`/`perf` tables. This also removes the split-K reduction-kernel `1.5 µs`-style rows that trap a
naive `min(golden_us)` in the text A/B.

## Workflow notes

- **Driver timeout too short for sm_120.** 14400 s wasn't enough; the tune survived but the harvest needed a manual
  liveness check (`pgrep [e]mmy tune` + Monitor on the done marker). Fix: scale `remote_node_tune.py --timeout` with
  `len(targets)` and detected arch, or add `--no-wait` returning a poll handle. **This bit us — same as the 4090 note,
  worse here.**
- **`--json` A/B is the right recording path** (confirmed on both cards): `greedy` vs `pinned` with per-kernel knobs +
  `recorded_emmy_us` + `flags`, so the YAML dict is auto-buildable and the noise floor comes free. Recommend the skill
  document it as canonical.
- **Degenerate-fast bench pollution** (Finding 3) is far worse on sm_120 than sm_89 — 24643× vs 34.9× worst TILE
  regret. The node-store diagnostics need the roofline floor before they're trustworthy on Blackwell.
- **Big `same-diffknob` bucket (22/36)** means the parser's exact-dict same-knobs test is too strict — many are
  latency-identical with a cosmetic knob-spelling difference (`FAST_EXP`/`VECTORIZE_LOADS` decoration). Improvement:
  compare on the *canonicalized* search-knob subset, not the raw dict, so these classify as `same-repro` (no action)
  rather than `same-diffknob` (candidate-add).
- **Attention on sm_120** is a mixed bag: static `hd64` 1.21× slower but `hd64.dynM` 0.87× *faster* than cuBLAS — the
  attention prior is noisy on Blackwell; needs its own tier like sm_89's.
