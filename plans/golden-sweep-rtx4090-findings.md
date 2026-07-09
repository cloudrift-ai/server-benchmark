# Golden sweep findings — NVIDIA GeForce RTX 4090 (sm_89)

**Date:** 2026-07-09  **GPU:** NVIDIA GeForce RTX 4090 (`rtx4090_sm89.yaml`)  **Where:** rented CloudRift box
(`riftuser@211.21.50.85`), repo rsynced + `make setup`, tuned + A/B'd remotely, data harvested to
`_tune/golden-sweep-4090/`.

**Sweep command (cold, one invocation):**

```
emmy tune --dataset golden --explore-eps 0     # deterministic; fresh box ⇒ cold (no --clean needed)
```

**Wall time:** 12 318 s (~3.4 h, `remote_node_tune` wait-only) for 42 tune targets. `shapes: 42/42`,
`bench_fails: 17` (the expected big-shape / wide-attention bench guards, e.g. `square.4096`'s hung-kernel +
one `CUDA_ERROR_ILLEGAL_ADDRESS` variant isolated in the bench-worker subprocess).

**A/B method:** `emmy run --bench --golden NAME` per shape (1 text pass) + `--json` × 2 passes for exact knobs and a
noise floor. All µs below are the live **-O3** A/B (greedy = deploy pick vs the recorded golden, both benched this run),
medianed over passes; `spread%` is the 2-pass greedy spread.

## Category tally (42 shapes)

| Category | N | Action |
| --- | --- | --- |
| same-repro (greedy == golden knobs, ≤3%) | 12 | none |
| same-diffknob (≤3%, different knobs) | 8 | none (already at parity) |
| **GREEDY_FASTER** (sweep beat the recorded golden >3%) | 10 | **4 recorded, 6 deferred (see below)** |
| **GREEDY_SLOWER** (prior couldn't reach the golden >3%) | 12 | findings |

**Recorded (4 replacements, all emmy_us ↓, clean knob mapping, ~0% spread):**

| shape | old knobs → new | emmy_us | vs live golden |
| --- | --- | --- | --- |
| `rms_norm.k3840` | `REDUCE b32` → `b256` | 18.4 → **6.9** | 0.40× (golden was a bad config; greedy == cuBLAS) |
| `rms_norm.k3840.dynM` | `REDUCE b32` → `b256` | 17.7 → **6.9** | 0.40× (golden pin also ⚑ unreproducible) |
| `gemma4_12b.qknorm.k256` | `REDUCE b32` → `b64` | 4.5 → **3.7** | 0.84× |
| `matmul.o_proj.h4096` | `w2x2/f4x4` → `w1x4/f4x4 + g2k` | 160.8 → **119.9** | 0.79× (split-K win) |

**Deferred, NOT recorded (flagged for a maintainer decision):**

- `attention.hd64` (0.88×), `attention.hd128` (0.85×) — greedy reproducibly beats the *live* golden (11.2 vs 12.6;
  22.0 vs 25.8), but the recorded golden's `emmy_us` (10.6 / 19.9) is an **⚑ unreproducible pin** (the recorded
  `TILE@dd/@pj` realize *different* knobs on re-bench). So the "win" also raises emmy_us vs a number that can no longer
  be reproduced. Recommend re-pinning to the reproducible greedy config as part of the `fix/golden-sweep-tooling-bugs`
  pin-gate work — but that's a deliberate golden downgrade of the headline ratio, so left to the maintainer.
- `attention.hd64.dynM` (0.83×) — greedy uses split `TILE@dd`/`TILE@pj`, but the dynM attention YAML schema records a
  single unified `TILE`. Not representable without a schema/knob-grammar decision; deferred.
- `matmul.square.1024` (greedy 57.9 vs live golden 65.0 = 0.89×) — but the golden's *recorded* 55.4 < greedy 57.9;
  the same golden config now re-benches at 65.0, i.e. a **codegen drift** since it was recorded, not a clean greedy win.
  Left untouched pending a look at the `n16x8/f4x8` regression.
- `matmul.qkv.h4096` (0.95×), `matmul.mlp_down.h4096.dynM` (0.97×) — marginal (<7%), inside the noise band.

## Fork sibling regret (this sweep's own prior, `eval prior --dataset nodes`, -O1)

```
family:  PLACE 1.03×   REDUCE 1.00×   STAGE 1.00×   TILE 1.48×   (ALL median, 102 forks)
worst TILE forks:  free=4096 red=14336 → 34.9×,  free=12288 red=4096 → 13.8×,  free=28672 red=4096 → 11.1×
leaf reachability: mean 1.17×  median 1.07×  worst 1.79×   |   leaf calibration (Spearman): +0.90
```

**One-line diagnosis:** the **TILE fork is the mispriced decision family**. PLACE/REDUCE/STAGE are ~optimal (≤1.03×);
the prior ranks warp-tile siblings ~1.48× off (worst 34.9× on the big MLP matmuls). Within reached leaves the ranking
is fine (calibration +0.90) — the miss is the cold search **not reaching** the best TILE leaf, confirmed by the deep
golden ranks under the learned prior (below).

## Per-shape outcome table (live -O3 A/B, greedy vs best live golden)

| shape | cuBLAS µs | greedy µs | golden µs | greedy/golden | greedy/cuBLAS | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| matmul.mlp_gate_up.h4096.dynM | 705 | 1379.4 | 936.4 | 1.47 | 1.96 | SLOWER (finding) |
| attention.hd128.dynM | 18 | 34.2 | 27.2 | 1.26 | 1.86 | SLOWER (finding) |
| matmul.o_proj.h4096.dynM | 110 | 175.6 | 151.3 | 1.16 | 1.60 | SLOWER (finding) |
| matmul.square.512.fp16 | 5 | 7.0 | 6.2 | 1.12 | 1.32 | SLOWER |
| matmul.mlp_down.h4096 | 369 | 408.8 | 373.8 | 1.09 | 1.11 | SLOWER |
| pointwise.n16384(.dynM) | 18 | 17.8 | 16.3 | 1.09 | 0.98 | SLOWER |
| matmul.qkv.h4096.dynM | 326 | 409.5 | 378.0 | 1.08 | 1.25 | SLOWER |
| matmul.square.512.dynM | 10 | 11.0 | 10.6 | 1.04 | 1.06 | SLOWER |
| matmul.square.1024.fp16 | 16 | 23.2 | 22.3 | 1.04 | 1.44 | SLOWER |
| matmul.square.512 | 11 | 13.0 | 12.6 | 1.03 | 1.21 | SLOWER |
| rms_norm.k3840(.dynM) | 7 | 6.9 | 17.2 | **0.40** | 1.01 | FASTER → **recorded** |
| matmul.o_proj.h4096 | 119 | 119.9 | 151.4 | **0.79** | 1.01 | FASTER → **recorded** |
| attention.hd64.dynM | 10 | 17.1 | 20.6 | 0.83 | 1.73 | FASTER (deferred: schema) |
| gemma4_12b.qknorm.k256 | 5 | 3.7 | 4.4 | **0.84** | 0.77 | FASTER → **recorded** |
| attention.hd128 / hd64 | 18 / 10 | 22.0 / 11.2 | 25.8 / 12.6 | 0.85 / 0.88 | 1.20 / 1.13 | FASTER (deferred: ⚑ pin) |
| matmul.square.1024 | 43 | 57.9 | 65.0 | 0.89 | 1.33 | FASTER (deferred: codegen drift) |
| (softmax/reduce/rms_norm k2048–8192, static+dynM) | — | — | — | ~1.00 | 0.5–1.05 | same-repro (healthy) |

Memory-bound kinds (softmax/reduce/rms_norm/pointwise) reproduce their goldens cleanly and beat cuBLAS
(vs-cuBLAS 0.5–0.95×); the matmul/attention families carry all the shortfalls.

## Finding 1 — TILE fork mispriced on masked-tile (`.dynM`) matmuls (worst: `mlp_gate_up.h4096.dynM`, 1.47×)

The static↔dynamic gap is the dominant theme: every heavy `.dynM` matmul lands slower than its golden while its static
twin is at parity or faster.

| shape | static greedy/golden | dynM greedy/golden |
| --- | --- | --- |
| mlp_gate_up.h4096 | 0.97 (repro) | **1.47** |
| o_proj.h4096 | **0.79** (win) | 1.16 |
| qkv.h4096 | 0.95 | 1.08 |
| square.512 | 1.00 (repro) | 1.04 |

`eval golden` per-knob: TILE matched the golden in only **5/17** matmul shapes (STAGE 14/17, REDUCE 3/7). For
`mlp_gate_up.h4096.dynM` the greedy picked `w1x8/f4x8/k4` + `STAGE d1/cp/p2` vs golden `w4x1/f4x8/k2` + `d2/cp/ring`
(0/2 TILE, wrong warp aspect *and* wrong split). Golden rank under the learned prior: **3738/7569** (deep) — cold PUCT
+ default patience cannot reach it. The masked warp tile is where the static↔dynamic gap lives (as the skill notes) and
the prior has no feature that separates the masked-tile siblings.
**Recommendation (high priority):** refit the analytic TILE weights over the recorded dynamic goldens
(`scripts/golden_knob_heuristics.py`, the `_W_A_DYN` masked-tier set) and add a `D_*` engineered feature that
distinguishes warp aspect (`wNxM`) under masking — the blame table shows TILE as the sole regret carrier, so this is a
weight/feature problem, not a patience problem.

## Finding 2 — `attention.hd128.dynM` (1.26×) and the unreproducible attention pins

`attention.hd128.dynM` greedy 34.2 vs golden 27.2 (1.26×, 1.86× cuBLAS). Attention calibration on this sweep was poor
(Spearman +0.03 on `hd64`, 149/221 benched variants ≥2× best) — the attention search is wide and the prior barely
ranks it. Related: the *static* `hd64`/`hd128` goldens are **⚑ unreproducible pins** (recorded `TILE@dd/@pj` realize
different knobs on re-bench), so their recorded µs (10.6 / 19.9) can't be reproduced and the greedy reproducibly beats
the *live* golden.
**Recommendation:** (a) treat the unreproducible attention pins in the `fix/golden-sweep-tooling-bugs` pin-gate work —
re-pin to the reproducible greedy config; (b) attention needs its own tier weighting / more patience — the poor
Spearman means the current prior can't rank attention siblings at all, so a weight refit alone won't fix it.

## Finding 3 — split-K (`g2k`) under-selected on medium matmuls

`o_proj.h4096` (recorded win) shows the pattern: the golden was a single-kernel `w2x2/f4x4` @ 151 µs; the sweep found
a **`g2k` split-K** `w1x4/f4x4` @ 120 µs (0.79×). `eval golden` REDUCE match is only 3/7 — several matmuls
(`qkv.h4096`, `square.512.fp16`, `mlp_down.h4096`) have golden `g2k` the greedy dropped, or vice-versa. The REDUCE fork
median regret is 1.00× (so REDUCE *ranking* is fine once reached) — the miss is joint TILE×REDUCE: the right split only
pays off with the right warp tile, and the TILE mispricing steers the search away from the region before the split is
evaluated.
**Recommendation:** lower priority than Finding 1 — a TILE-weight refit should recover most of these transitively;
re-check REDUCE match after the refit.

## Workflow notes

- **Slowest step:** the tune sweep (~3.4 h) dominated; the attention shapes alone were ~10 min each (619 s for
  `hd64`, 221 benches). The per-shape A/B (text) + 2 `--json` passes were ~12 min each. Lever: the memory-bound `.dynM`
  twins (softmax/reduce/rms_norm/pointwise) all landed within ~2% of their static twins (12 same-repro) — skipping them
  would trim ~8 tune targets with no information loss, as the skill already suggests.
- **Driver timeout is a sweep-length trap:** `remote_node_tune.py --timeout` defaults to 7200 s; even 14400 s was too
  short for the *5090* (see its report). The tune process survives the poller giving up, but the harvest logic then
  needs a manual liveness check. Proposed fix: make the timeout scale with `len(targets)` or expose a `--no-wait` that
  hands back a poll handle.
- **`--json` is the right recording path.** `run --bench --golden NAME --json PATH` emits `greedy` vs `pinned` with
  per-kernel knobs, `total_us`, `recorded_emmy_us`, and `flags` — enough to auto-build the YAML knob dict and confirm
  the noise floor in one artifact. The text table's split-K reduction row (a knob-less `1.5 µs` line under `square.512`)
  is a min() trap that the JSON's per-config grouping avoids. Symptom: the text A/B is ambiguous for split-K goldens.
  Improvement: document `--json` as the canonical A/B-recording output in the skill.
- **Degenerate-fast benched configs pollute the node store.** Some benched variants clock physically-impossible µs
  (a `square.512` golden row at 1.5 µs vs cuBLAS 10 µs; on the 5090 far worse). They inflate `eval prior --dataset
  nodes` reachability regret. Symptom: reachability "worst" numbers are not trustworthy as-is. Improvement: reject
  variants that bench below a FLOP-roofline floor before they enter the node store.
- **Attention dynM knob schema.** The greedy pick carries split `TILE@dd`/`TILE@pj` that the dynM attention YAML can't
  represent (single `TILE`). This blocked recording a real 0.83× win. Improvement: unify the attention knob schema
  (accept dd/pj in dynM) or document the single-TILE constraint so the sweep prunes those forks for dynM attention.
