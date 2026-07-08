# Golden sweep — RTX 4090 (sm_89), full op-typed set, 2026-07-08

Fourth 4090 sweep report (siblings: `golden-sweep-rtx4090-findings-2.md` 2026-07-06, `golden-sweep-rtx4090-findings-3.md`
2026-07-07 — findings-3 covered the matmul family only). **First sweep under the refit analytic weights** — the
2026-07-07 reduce/pointwise-inclusive refit, branch `feature/reduce-featurizer-fix` @ `495962b3` — and the first to
cover the whole 39-shape golden dataset on this card (memory-bound kinds + attention were never A/B'd or recorded here
before; findings-3's recommendation).

**GPU:** NVIDIA GeForce RTX 4090 (sm_89), CUDA 12.9, driver 580.65.06 (CloudRift, single GPU). **Code:** local working
tree of `feature/reduce-featurizer-fix` @ `495962b3`, rsynced to the box.

**Sweep commands:**

```
emmy tune --dataset golden --clean                       # one cold invocation, 39/39 shapes, 2 h 49 min wall
emmy run --bench --golden <name>                         # 18 recorded names, live -O3 A/B; ×3 for 9 marginal shapes
emmy run --bench -c "<snippet>" [--dynamic …]            # 21 seed benches for the never-recorded kinds
emmy run --bench -c "<attention snippet>" --dynamic … --ab "TILE=…"   # bare-TILE probes for the dynM attention seeds
```

**Category tally: 0 replaced · 6 added · 21 seeded · 4 unchanged (same knobs) · 2 left (in-band loss) · 6 worse ·
1 pruned** (the dominated `square.512.dynM` `n32x8/g4k` entry, 9 % slower than the new add in 3/3 runs).

**Fork sibling regret** (`eval prior --dataset nodes`, this card's `-O1` block, 6728 nodes, 104 forks) — per-family
`ALL (median)`: **PLACE+REDUCE+STAGE+TILE 1.04x · REDUCE 1.00x · STAGE 1.00x · TILE 1.49x**. TILE is the one steering
gap: the h4096 family's TILE regret runs 18.06x/20.54x, and `free=4096 red=14336` (mlp_down) hits 10709.79x — a
sentinel-poisoned fork (a 2 000 000 µs bench_fail pin sits where the prior steered). REDUCE/STAGE forks are clean.

**The refit verdict in one line:** the cold `AnalyticPrior` is now essentially perfect on this card's golden set
(`eval analytic`: **median rank 0; top-1 for 14/19 matmul goldens** — squares 8/11, h4096 6/8; findings-3's perennial
`square.512.fp16` mispricing went 833 → **0**), and the fp16-square regressions of findings-2/3 are gone — but the
**within-sweep learned prior unlearns the analytic's h4096 pricing** (learned golden ranks: median 316, top-10 0/19)
and deploys a systematic `w1x8 + deep k-split + d1/cp` pattern that loses 1.23–1.66× on all six h4096 fp16 shapes the
prior itself seeded at parity yesterday.

## Per-shape outcome

Live -O3 program totals (split-K = partial + epilogue summed); medians over the 3 confirmation passes where run.
`ratio` = greedy ÷ best-golden same-run re-bench; `vs CB` = greedy ÷ recorded `cublas_us` (>1 = emmy slower than
cuBLAS/torch eager). Seeded rows have no golden to compare (`—`); their `vs CB` uses the same run's eager row.

| shape | greedy µs | best-golden µs | ratio | cuBLAS µs | vs CB | category |
|---|---|---|---|---|---|---|
| matmul.square.512 | 12.7 | 13.1 | 0.97 | 10.8 | 1.18 | added (`f4x6`/`d3`, no split) |
| matmul.square.1024 | 55.2 | 56.0 | 0.99 | 45.4 | 1.22 | unchanged (same knobs) |
| matmul.square.2048 | 354.0 | 356.3 | 0.99 | 320.0 | 1.11 | unchanged (same knobs) |
| matmul.square.4096 | 2819.1 | 2793.7 | 1.01 | 2458.6 | 1.15 | added (`d3/cp/ring` twin) |
| matmul.square.512.fp16 | 6.3 | 6.6 | 0.96 | 5.8 | 1.09 | added (F5 fixed) |
| matmul.square.1024.fp16 | 25.7 | 26.8 | 0.96 | 18.1 | 1.42 | added (F5 fixed) |
| matmul.square.2048.fp16 | 116.2 | 114.3 | 1.02 | 115.2 | 1.01 | left (in-band loss, 3/3) |
| matmul.square.4096.fp16 | 905.2 | 861.2 | 1.05 | 822.3 | 1.10 | left (golden faster 2/3) |
| matmul.square.512.dynM | 10.6 | 11.0 | 0.96 | 10.8 | 0.98 | added + pruned `n32x8/g4k` |
| matmul.qkv.h4096 | 380.3 | 378.9 | 1.00 | 328.0 | 1.16 | added (`w2x2/f4x8/k2` twin) |
| matmul.o_proj.h4096 | 215.8 | 152.6 | **1.41** | 119.0 | 1.81 | **worse** (F2) |
| matmul.mlp_gate_up.h4096 | 1392.8 | 966.5 | **1.44** | 721.0 | 1.93 | **worse** (F1) |
| matmul.mlp_down.h4096 | 620.7 | 374.9 | **1.66** | 388.0 | 1.60 | **worse** (F1) |
| matmul.qkv.h4096.dynM | 539.4 | 379.9 | **1.42** | 330.0 | 1.63 | **worse** (F1) |
| matmul.o_proj.h4096.dynM | 183.1 | 149.2 | **1.23** | 112.0 | 1.63 | **worse** (F1) |
| matmul.mlp_gate_up.h4096.dynM | 1333.1 | 968.5 | **1.38** | 742.0 | 1.80 | **worse** (F1) |
| matmul.mlp_down.h4096.dynM | 400.9 | 391.5 | 1.02 | 388.0 | 1.03 | unchanged (same knobs) |
| pointwise.n4096 | 4.6 | 4.6 | 1.00 | 5.0 | 0.92 | unchanged (same knobs) |
| softmax.k2048 | 3.5 | — | — | 6.0 | 0.58 | seeded |
| softmax.k8192 | 12.4 | — | — | 14.0 | 0.89 | seeded |
| softmax.k2048.dynM | 3.6 | — | — | 6.0 | 0.60 | seeded |
| softmax.k8192.dynM | 13.2 | — | — | 14.0 | 0.94 | seeded |
| reduce.k2048 | 2.1 | — | — | 4.0 | 0.53 | seeded |
| reduce.k8192 | 4.5 | — | — | 7.0 | 0.64 | seeded |
| reduce.k2048.dynM | 2.1 | — | — | 4.0 | 0.53 | seeded |
| reduce.k8192.dynM | 4.6 | — | — | 7.0 | 0.66 | seeded |
| rms_norm.k2048 | 4.1 | — | — | 4.0 | 1.03 | seeded |
| rms_norm.k4096 | 7.2 | — | — | 7.0 | 1.03 | seeded |
| rms_norm.k8192 | 13.8 | — | — | 13.0 | 1.06 | seeded |
| rms_norm.k2048.dynM | 4.1 | — | — | 4.0 | 1.03 | seeded |
| rms_norm.k4096.dynM | 7.2 | — | — | 7.0 | 1.03 | seeded |
| rms_norm.k8192.dynM | 13.8 | — | — | 13.0 | 1.06 | seeded |
| pointwise.n16384 | 17.4 | — | — | 18.0 | 0.97 | seeded |
| pointwise.n4096.dynM | 4.6 | — | — | 5.0 | 0.92 | seeded |
| pointwise.n16384.dynM | 16.5 | — | — | 19.0 | 0.87 | seeded (pinned `f4`, F4) |
| attention.hd64 | 10.6 | — | — | 10.0 | 1.06 | seeded |
| attention.hd128 | 19.9 | — | — | 18.0 | 1.11 | seeded |
| attention.hd64.dynM | 20.6 | — | — | 11.0 | **1.87** | seeded (bare-TILE pin, F3) |
| attention.hd128.dynM | 27.3 | — | — | 19.0 | **1.44** | seeded (bare-TILE pin, F3) |

Emmy beats torch eager on the memory-bound reduce/softmax/pointwise kinds (0.53–0.97×) and is at parity on rms_norm
and static attention; it trails cuBLAS on every matmul (1.09–1.22× squares, 1.6–1.9× on the h4096 misses) and trails
torch SDPA badly on masked (dynM) flash attention (1.44–1.87×, F3).

## Finding 1 — the h4096 family: the learned prior unlearns a correct analytic pricing (five shapes, 1.23–1.66×)

The sweep's dominant miss, and it is one systematic failure, not five: on `mlp_down.h4096` (1.66×),
`qkv.h4096.dynM` (1.42×), `mlp_gate_up.h4096` (1.44×, plus its dynM twin at 1.38×) and `o_proj.h4096.dynM` (1.23×),
greedy deploys the same signature — **`w1x8` warp shape, deep k-split (`k4`, or `f2x8` narrow fragment), single-buffer
`STAGE=d1/cp`** — where every recorded golden in the family wants `w2x2`/`w2x4`/`w4x1` + `d2/cp/ring`. Findings-3
seeded all eight h4096 entries from greedy at parity 24 h earlier, so these configs were *reachable then*; this
sweep's search walked away from them. Evidence:

- `eval analytic --kernel h4096`: golden ranks **0 / 0 / 0 / 0 / 0 / 6 / 32 / 0** (median 0, top-1 6/8). The refit
  analytic weights price this family essentially perfectly — the cold heuristic is exonerated.
- `eval prior --dataset golden`: learned ranks 87–1946 (`mlp_down` 1498, its dynM 1946; median over all 19 golden
  shapes 316, top-10 **0/19**). The within-sweep CatBoost drifts the family's pricing away from the analytic. Its
  `vs gold` column agrees with the live A/B (1.39–1.58× on the statics) — the misdeploy is visible in-view now.
- `eval variants` per kernel: the pick misses the *measured* best inside its own group everywhere —
  `cd8b18` (mlp_down fused) pick rank **47/75**, 1.69× the -O1 best (`w1x4/f4x4 d2/cp/ring`, 626.6 µs);
  `a8ecf3__partial` pick 4/6; `631750__partial` pick 6/10 (1.35×); `755f10__partial` similar. This is not a
  reachability gap — the better configs were benched and then out-scored by the prior at pick time.
- Structural under-exploration compounds it: `cd8b18__partial` (the split structure greedy deployed) has exactly
  **1 measured config**, vs 75 in the fused group it displaced.
- `eval prior --dataset nodes`: TILE fork regret 18.06x (`free=28672`), 20.54x (`free=12288`), 10709.79x
  (`free=4096 red=14336` — a bench_fail sentinel at the steered-to child). All 34 bench_fail pins of the sweep landed
  on this family (17 on `mlp_gate_up.h4096.dynM` alone: 2 s bench-wall, 1000 ms hung-kernel, plus 4
  `matmul__partial → TileOp` lowering TypeErrors, see F6) — the reservoir carries dense 2 000 000 µs sentinels on
  exactly the region the prior must rank.
- Per-shape post-warmup calibration (tune.log): `mlp_gate_up` **+0.24**, `mlp_down` **−0.41**, and the dynM four at
  +0.11…+0.28 — against +0.7…+0.97 everywhere else. The learned model is uncalibrated precisely on this family.

**Recommendation** (priority order): (1) guard the learned prior against overriding a top-ranked analytic config with
an *unmeasured or sentinel-adjacent* extrapolation — e.g. require the deployed config's fork path to carry at least
one clean measured row, or blend in the analytic rank as a floor when per-op calibration is this low (the calibration
gate of `b4695256` targets promotion; this is the same idea applied at pick time). (2) Give sentinel rows a bounded
loss (cap or quantile-clip the 2 000 000 µs pins in the reservoir) so 17 fails cannot capsize one shape family.
(3) Re-check after the fix with the one-command regressions `run --bench --golden matmul.mlp_down.h4096` (worst
offender) — the goldens are recorded and reproduce (374.9 µs replay this sweep).

## Finding 2 — `o_proj.h4096` static: pick rides the -O1/-O3 inversion into a 1.41× miss

Greedy deploys the *fused* `w2x2/f2x8/k2` (215.8 µs) over the golden's derived `g4k` split (147.3 + 5.3 = 152.6 µs).
`eval variants --kernel e24882`: the pick ranks **23/29 by -O1 latency (600.0 µs, 1.94× the -O1 best)** but its -O3
rebench came back 223.2 µs — the -O1/-O3 inversion made a mid-pack config look like a winner, and the golden's split
structure has no measured rows in this sweep to argue back. Same family as F1 (and the same `d1/cp`-adjacent drift on
the dynM twin), but the mechanism here is the inversion, not the sentinel drift.

**Recommendation:** when the -O3 rebench of the pick lands >1.5× away from its -O1 rank neighborhood, trigger -O3
rebenches for the top-k -O1 configs of the *other* structural fork before finalizing the deploy (one or two extra
compiles); `EMMY_O3_TOL` only widens the band within the already-chosen kernel group, so it cannot catch a
cross-structure inversion.

## Finding 3 — masked (dynM) flash attention: greedy is 1.44–1.87× behind torch SDPA, and the good configs are unrecordable

First 4090 attention entries. Static flash is healthy (hd64 10.6 µs vs SDPA 10; hd128 19.9 vs 18) and replays exactly
from the recorded `TILE@dd`/`TILE@pj` pins. The dynM (masked-tile) twins are not: the tune's own best is 19.3 µs
(hd64) / 29.1 µs (hd128) vs SDPA 11 / 19 — the masked-flash variant space itself tops out well behind SDPA on sm_89.
Recording was its own fight: greedy picks *different* per-axis tiles (`dd=w1x1/f1x16/k4`, `pj=w1x1/f1x8/k8`) which the
schema rightly refuses for dynamic goldens (axis-keyed pins don't resolve on the masked flash), the 5090's golden
bare-TILEs transfer badly (29.4 µs / 66.2 µs — the hd128 pin even resolved asymmetrically), and one probe
(`w1x1/f1x8/k8`) hit the **flatten pathology live**: a scalar `f64` fallback at 3377.9 µs, 164× the greedy pick.
Recorded: verified bare-TILE pins `w1x1/f1x16/k4` (hd64.dynM, 20.6 µs) and `w2x1/f1x16/k8` (hd128.dynM, 27.3 µs),
both replay-validated (21.2 / 27.3 µs).

**Recommendation:** the masked-flash tile space needs its own look on sm_89 (the 1.9× gap vs SDPA is the largest
"loser" in the table after the F1 shapes) — likely a masked-tier enumeration gap rather than a prior miss, since the
tune's best measured config is already 1.75× SDPA. Separately: a recorder helper that probes bare-TILE candidates
automatically (the manual `--ab` loop is three commands per shape and one wrong guess costs a 3 ms bench).

## Finding 4 — `pointwise.n16384.dynM`: greedy leaves the vectorized tile on the table (1.10×)

Greedy deploys the masked relu with a blank TILE (scalar f1, 18.2 µs); pinning `TILE=f4` — exactly what greedy picks
for the *static* twin — gives 16.5 µs (0.87× eager). Recorded the pinned config. Smallest finding, but it is the one
memory-bound miss, and the static/dynM knob asymmetry (`f4` vs nothing) suggests the masked-tile path drops the
vectorization fork rather than losing it on merit.

**Recommendation:** check that the pointwise masked-tile enumeration offers `f2`/`f4` at all (`priority_mode=
"pointwise"` under a symbolic axis); if it does, this is a two-config prior fix; if it doesn't, it's an eligibility
gate (cite: the masked-variant enumerate path in the pointwise planner).

## Finding 5 (resolved) — the fp16 squares: findings-2/3's perennial mispricing is fixed by the refit

`square.512.fp16` (worse in three consecutive sweeps, analytic rank 833 in findings-3) and `square.1024.fp16` (1.13×
in findings-3) both now **win** their A/Bs (0.955, consistent in 3/3 and 2/3 passes respectively) with configs the
refit analytic ranks at 0. `square.2048.fp16` remains a consistent in-band loss (1.01–1.02, 3/3 — same verdict as
findings-3: not recordable, not actionable). The greedy `square.512.fp16` split (`w1x2/f4x2/k2+g2k`, 6.3 µs) and
`square.1024.fp16` (`w4x1/f2x4/k2+d1/cp`, 25.7 µs) are recorded as parity adds. Notably `d1/cp` *wins* here at
M=N=1024 — the same STAGE the F1 shapes misuse at 4096–28672; the learned prior seems to have generalized "d1/cp is
good" from the small-fp16 regime to the big-N regime where it is ruinous. That cross-regime leakage is consistent
with F1's calibration collapse and worth a `D_*` feature separating occupancy-bound from bandwidth-bound regimes.

## Workflow notes

Fixes from findings-3's notes: **the compile-budget bump (4 s → 12 s) held** — zero compile-budget bench_fails this
sweep (all 34 fails are bench-wall/hung/lowering, a different population). Everything else on that list is still
open: the eager row is still integer-rounded (all 21 new `cublas_us` seeds are integers), `--golden` still can't seed
a shape recorded only for another card (the 21 seeds went through `run --bench -c` + `--dynamic` again), `eval`
program totals vs isolated-row µs still disagree visibly (the `vs gold` 1.08× on `square.512.fp16` vs the A/B's
0.955 program ratio), and `bench_fail` reservoir sentinels still have no purge path (now with F1-scale consequences).

New this sweep:

- **The sweep tail is fp16-square-shaped.** 3 of 39 shapes (`square.{512,1024,2048}.fp16`, 899/711/844 s) ate 24 %
  of the 2 h 49 m wall — 293/217/238 benches each, vs ~100 for the h4096 shapes whose picks actually regressed.
  Patience spent where the prior is already right, none to spare where it drifts. A per-shape budget weighted by
  (pool size × calibration) would rebalance this.
- **`tune.log`'s `best:` line is per-kernel, not per-program** — it prints the epilogue kernel's 13.7 µs as the
  "best" for split-K `mlp_down` (whose real program is ~380 µs). Misleading for exactly the shapes under
  investigation; the fix is to print the program total (the A/B table already computes it).
- **The `matmul__partial → TileOp` lowering TypeError** (4 occurrences, `qkv`/`o_proj` dynM): a split variant that
  passes enumeration but cannot lower is a bug independent of benching — it burns a bench slot and pins a sentinel.
  Deserves an issue even though it is only 4 of 34 fails.
- **Attention dynM recording needs a helper** (F3): three manual `--ab` probes per shape, one of which detonated the
  flatten pathology. The schema's bare-TILE constraint is correct; the tooling to find a valid bare TILE is missing.
- **`eval variants`' `-O3 us` column is populated only for the pick** — verifying F2's inversion for the golden's
  side required the A/B run instead. More -O3 rows (top-k per structure) would make inversion checks one command.
- **Noise floor re-runs:** 9 shapes × 2 extra passes, ~15 s each warm — cheap. The A/B swing stayed ≤5 % for
  everything except `square.4096.fp16`'s golden row (912/861/860 — a 6 % swing that flipped its category from "add"
  to "left"); the three-pass protocol was decisive there, keep it.
- **A/B totals were parsed from 18 tee'd logs by hand-written regex** (split-K golden entries span 2 table rows).
  A `--json` emitter on `run --bench` would have saved the parser and its one bug.

## Validation

- Schema: `tests/compiler/test_golden_configs.py` — 20 passed (local and on the box against the updated YAML).
- Replays on the box (updated YAML rsynced): all four attention entries reproduce exactly (10.5/20.0/21.2/27.3 µs,
  no flatten), `softmax.k2048` 3.5 µs exact, `pointwise.n16384.dynM` golden 16.5 µs (still beats greedy's 17.4),
  both `square.512.dynM` entries (11.0 / 10.6 µs) and both `qkv.h4096` entries (378.2 / 372.5 µs) replay at parity.
- 46 configs load for the 4090 (24 matmul, 6 rms_norm, 4 each softmax/reduce/pointwise/attention).

## Artifacts

`_tune/golden-sweep-rtx4090-refit/` (local): `tune.log` (39/39, 2 h 49 m), 18 `ab-*.log` + 18 `ab{2,3}-*.log`
noise-floor passes, 21 `seed-*.log`, 6 `abpin*-*.log` attention/pointwise probes, 8 `validate-*.log`,
14 `eval-*.log` views, plus the box's trained state: **`autotune.db` (490 MB; `node` table 7818 rows, all
`NVIDIA GeForce RTX 4090`, `MAX(feat_ver)=2`) and `prior.json` (21.8 MB; parses; dataset reservoir 25 037 rows,
100 feature cols)** — the Phase-3 snapshot inputs for the analytic-prior rework.
