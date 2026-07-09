# Golden sweep — RTX 4090 (sm_89), full 47-shape dataset, 2026-07-09

Fifth 4090 sweep report (immediate predecessor: `golden-sweep-rtx4090-findings-4.md`, 2026-07-08, the refit-weights
sweep on `feature/reduce-featurizer-fix` @ `495962b3` — deleted in this commit per the plans convention, as
findings-2/-3 were before it; recover via `git log --diff-filter=D`). First sweep over the complete op-typed set
*including* the gemma-4 h3840/mixed projections (PR #331) and the k3840/qknorm rms_norm seeds (#323), and the first
on the #331 + main (#332/#333) merged code.

- **GPU / box:** NVIDIA GeForce RTX 4090 (sm_89), CloudRift rental, nvcc 12.9 (`CUDA_HOME=/usr/local/cuda`),
  torch 2.13.0+cu130.
- **Code:** `feature/golden-sweep-rtx4090-jul08` = PR #331 head (`d3200a38`, mixed schema + projection goldens)
  + `main` (`1674d795`: #332 readable codegen, #333 blame/ablation diagnostics) merged.
- **Sweep:** `EMMY_O3_TOL=0.10 emmy tune --dataset golden --clean` — one invocation, cold, 47 shapes,
  **wall 3h47m** (01:02–04:49Z; per-shape times sum to 3.8 h — no dead time). The fp16/h4096/h3840 matmul family
  is ~¾ of the wall (700–1150 s/shape); memory-bound kinds are 18–60 s. The skill's ~2.5 h budget was stale — the
  set has grown; SKILL.md updated to ~4 h.
- **A/B:** `emmy run --bench --golden NAME` × 47, then 2 warm-cache re-runs for every shape in the marginal band
  (24 shapes × 3 total runs); categories below use per-shape medians.
- **Tally: 7 replaced · 3 added · 1 µs-lowered (same knobs) · 20 unchanged · 16 worse.**
- **Fork sibling regret aggregate** (`eval prior --dataset nodes`, this card's -O1 block, `ALL (median)` row):
  PLACE+REDUCE+STAGE+TILE **1.03x** · REDUCE **1.00x** · STAGE **1.00x** · TILE **1.62x** over 131 forks.
  TILE is the one broken family — and it is catastrophically broken exactly where the A/B losses are:
  per-shape TILE regret 10818x on `free=4096 red=14336` (mlp_down), 83x on `free=3840 red=15360`, 23x on
  `free=12288 red=4096`, 10x on `free=28672 red=4096`.
- **bench_fails: 20**, clustered on the same family — 7 × `mlp_gate_up.h4096.dynM`, 5 × `qkv.h4096.dynM`,
  2 × `o_proj.h4096.dynM`, 4 × `square.4096(.fp16)`, 1 × `mlp_down.h4096.dynM`, 1 × `mlp_down.h3840`. All are
  runtime fails (HungKernelError >1000 ms, "run stage exceeded 2.0s", one CUDA_ERROR_ILLEGAL_ADDRESS) — zero
  compile-budget fails, so #320's 12 s budget bump held. One is a genuine lowering bug:
  `CudaBackend: node 'matmul__partial' has non-CudaOp 'TileOp'` (2 rows on `k_matmul_631750`).

Artifacts: `_tune/golden-sweep-rtx4090/` (tune.log, ab/, ab2/, eval-*.txt, autotune.db, prior.json — box deleted).

## Per-shape outcome

Medians over the confirmation runs (marginal shapes ×3, clear calls ×1); split-K program totals (partial + epilogue
summed). `ratio` = greedy ÷ best-golden live re-bench; `vs cuBLAS` = greedy ÷ recorded `cublas_us` (>1 = emmy slower
than PyTorch/cuBLAS). Within-run **same-knob** greedy/golden row pairs disagree by up to ±6% (see Finding 5), so
single-digit ratios are read against that floor, not the naive 3% band.

| shape | greedy µs | best-golden µs | ratio | cuBLAS µs | vs cuBLAS | category |
|---|--:|--:|--:|--:|--:|---|
| attention.hd128.dynM | 33.9 | 28.3 | 1.20 | 19.0 | 1.78 | worse |
| attention.hd128 | 22.2 | 26.0 | 0.85* | 18.0 | 1.23 | worse vs recorded (coerced pins, F2) |
| attention.hd64.dynM | 17.2 | 20.5 | 0.84 | 11.0 | 1.56 | **replaced** |
| attention.hd64 | 15.0 | 13.8 | 1.09* | 10.0 | 1.50 | worse (coerced pins, F2) |
| gemma4_12b.qknorm.k256 | 3.7 | 4.4 | 0.84 | 5.0 | 0.74 | **replaced** (b32 → b64) |
| matmul.kv_proj.h3840.mixed | 134.6 | 134.7 | 1.00 | 57.0 | 2.36 | same (same knobs) |
| matmul.mlp_down.h3840 | 617.4 | 408.5 | 1.51 | 385.0 | 1.60 | worse (F1) |
| matmul.mlp_down.h4096.dynM | 616.6 | 393.0 | 1.57 | 388.0 | 1.59 | worse (F1) |
| matmul.mlp_down.h4096 | 393.3 | 375.9 | 1.05 | 388.0 | 1.01 | same (same knobs) |
| matmul.mlp_gate_up.h3840.mixed | 574.0 | 628.1 | 0.91 | 410.0 | 1.40 | **replaced** |
| matmul.mlp_gate_up.h4096.dynM | 1462.9 | 914.4 | 1.60 | 742.0 | 1.97 | worse (F1) |
| matmul.mlp_gate_up.h4096 | 909.5 | 968.7 | 0.94 | 721.0 | 1.26 | **µs lowered** (same knobs, codegen win) |
| matmul.o_proj.h3840 | 119.9 | 130.9 | 0.92 | 113.0 | 1.06 | **replaced** |
| matmul.o_proj.h4096.dynM | 176.0 | 150.3 | 1.17 | 112.0 | 1.57 | worse (F1) |
| matmul.o_proj.h4096 | 121.3 | 153.8 | 0.79 | 119.0 | 1.02 | **replaced** — now golden=True |
| matmul.q_proj.h3840.mixed | 175.6 | 165.5 | 1.06 | 109.0 | 1.61 | same (same knobs, F5 artifact) |
| matmul.qkv.h4096.dynM | 590.1 | 383.9 | 1.54 | 330.0 | 1.79 | worse (F1) |
| matmul.qkv.h4096 | 366.3 | 379.4 | 0.97 | 328.0 | 1.12 | **added** (fused alternate) |
| matmul.square.1024.fp16 | 23.2 | 22.3 | 1.04 | 18.1 | 1.28 | worse (perennial, findings-3 F3) |
| matmul.square.1024 | 61.4 | 62.3 | 0.99 | 45.4 | 1.35 | **added** (n16x16/f4x10 alternate) |
| matmul.square.2048.fp16 | 130.3 | 114.7 | 1.14 | 115.2 | 1.13 | worse |
| matmul.square.2048 | 386.4 | 360.4 | 1.07 | 320.0 | 1.21 | worse |
| matmul.square.4096.fp16 | 863.2 | 863.2 | 1.00 | 822.3 | 1.05 | **added** (w1x4/f4x4 alternate) |
| matmul.square.4096 | 2993.4 | 2821.4 | 1.06 | 2458.6 | 1.22 | worse |
| matmul.square.512.dynM | 10.6 | 10.7 | 0.99 | 10.8 | 0.98 | same (same knobs) |
| matmul.square.512.fp16 | 7.2 | 6.4 | 1.12 | 5.8 | 1.24 | worse (perennial, findings-3 F4) |
| matmul.square.512 | 13.8 | 12.8 | 1.08 | 10.8 | 1.28 | worse |
| pointwise.n16384.dynM | 17.5 | 16.6 | 1.05 | 19.0 | 0.92 | worse (f2 vs golden f4) |
| pointwise.n16384 | 17.5 | 16.5 | 1.06 | 18.0 | 0.97 | worse (f2 vs golden f4) |
| pointwise.n4096(.dynM) | 4.6 | 4.6 | 1.00 | 5.0 | 0.92 | same (same knobs) |
| reduce.k2048(.dynM) | 2.1 | 2.1 | 1.00 | 4.0 | 0.53 | same (same knobs) |
| reduce.k8192(.dynM) | 4.6 | 4.6 | 1.00 | 7.0 | 0.66 | same (same knobs) |
| rms_norm.k2048(.dynM) | 4.0 | 4.0 | 1.00 | 4.0 | 1.00 | same (same knobs) |
| rms_norm.k3840 | 6.9 | 17.3 | 0.40 | 7.0 | 0.99 | **replaced** (b32 → b256) — now golden=True |
| rms_norm.k3840.dynM | 7.0 | 17.3 | 0.40 | 7.0 | 1.00 | **replaced** (b32 → b256) |
| rms_norm.k4096(.dynM) | 7.1 | 7.1 | 1.00 | 7.0 | 1.01 | same (same knobs) |
| rms_norm.k8192(.dynM) | 13.8 | 13.8 | 1.00 | 13.0 | 1.06 | same (same knobs) |
| softmax.k2048(.dynM) | 3.5 | 3.5 | 1.00 | 6.0 | 0.58 | same (same knobs) |
| softmax.k8192 | 12.5 | 12.4 | 1.01 | 14.0 | 0.89 | same |
| softmax.k8192.dynM | 12.5 | 13.2 | 0.95 | 14.0 | 0.89 | same (5% is inside the F5 artifact band) |

\* attention statics: the "golden" row benched a silently coerced config, not the recorded one (Finding 2) — hd128's
0.85 is vs the coerced row; vs the *recorded* 19.9 µs greedy is 1.11. Both left untouched.

Wins worth naming: `o_proj.h4096` 160.8 → 121.3 µs (g2k split displaces the fused w2x2/f4x4; **1.02× cuBLAS,
golden=True**), `rms_norm.k3840(±dynM)` 17.7–18.4 → 6.9–7.0 µs (b256 displaces b32, **parity with eager** — the
2.5×-slower gap the seeding session recorded is gone; #322's REDUCE-codec featurization is the enabling change),
`attention.hd64.dynM` 20.6 → 17.2 µs, `qknorm.k256` 4.5 → 3.7 µs (b64, 0.74× cuBLAS — best ratio in the file).
Emmy still trails cuBLAS on every fp16 compute-bound matmul (1.02–1.40×) and the mixed projections (1.40–2.36×);
the wins live in memory-bound kinds and split-K K-heavy shapes.

## Finding 1 — the w1x8 + d1/cp misdeploy persists (findings-4 F1), now concentrated on the masked-tile shapes

The sweep's dominant miss, and the direct continuation of findings-4's F1: `qkv.h4096.dynM` (1.54),
`o_proj.h4096.dynM` (1.17), `mlp_gate_up.h4096.dynM` (1.60), `mlp_down.h4096.dynM` (1.57), `mlp_down.h3840`
(1.51, static — new shape from #331). Every greedy pick shares the same extreme-aspect warp tile + no-ring staging
signature findings-4 named (`w1x8/f2x8` or `w8x2/f4x4`, `d1/cp`); every golden is the canonical balanced
`w2x4-or-w2x2/f4x4 + g2k + d2/cp/ring` (`eval golden`: TILE matched on 5/22 matmul shapes, the dynM family 0–1/3
with exactly this diff).

The delta vs findings-4 matters: its three *static* h4096 misses all recovered this sweep — `o_proj.h4096`
1.41 → 0.79 (replaced, now golden=True), `mlp_gate_up.h4096` 1.44 → 0.94 (greedy reproduces the golden knobs),
`mlp_down.h4096` 1.66 → same-knobs parity. What remains broken is precisely the masked-tile (`.dynM`) variants
plus the K-heaviest static (`mlp_down.h3840`, K=15360) — the family where this sweep's bench_fails landed.

- **The search measured better configs and deploy ignored them.** `eval variants --kernel 462b55`
  (gate_up.h4096.dynM): pick sits at rank **25/35** (2132 µs -O1) while rank 1 (`w2x2/f4x4/k2 d2/cp/ring/p2`,
  1090.6 µs) was measured in the same tune. `--kernel 539ebf` (mlp_down.h3840): fused-kernel pick rank **36/36**
  (4.55× of best). `--kernel 631750` (qkv.dynM): the FUSED kernel's pick is rank 1 (493 µs) — but deploy took the
  *split* structure whose pick ranks 12/25 (618 µs -O3). The knob-level ranking is not the whole story; the
  structural fork (fused vs split, and which split) is again priced by extrapolation, not evidence — the same
  weak link as findings-3 F1/F2 and the PR #331 layer-tune regression.
- **The learned prior buries the golden region; the cold analytic prior nails it.** `eval prior --dataset golden`:
  the family's goldens rank 500–2974 of ~3–7.5k under the sweep's own freshly trained prior (median rank across
  all 26 evaluable shapes: **166**, top-10 **0/26**). `eval analytic`: median rank **0**, top-1 **19/26** (partly
  circular — the analytic weights are fit on these goldens — but it proves the region is reachable from cold).
  The two-level tune's learned layer actively un-learns the golden region on this family.
- **The mechanism is sentinel poisoning of the good region.** 16 of the 20 bench_fails landed on this family
  (`eval failures`), and they are hang/timeout fails on big kernels — e.g. the `w4x4/f4x8` and `g4k w1x8/f4x8`
  rows on 462b55 that "still pending after 0.20s". Each pins a 2 000 000 µs sentinel into the DB **and** the
  prior's training reservoir. Findings-3 flagged the same poisoning under the old compile-budget fails; the budget
  fix (#320) moved the fails to runtime but the sentinel path is unchanged. Greedy then deploys a *sibling of a
  failed config* (462b55: deployed `g4k w1x8/f4x8/k4 d1/cp/p2`; the failed row is `g4k w1x8/f4x8 d1/cp`) — the
  model has no signal that this exact region hangs, only isolated 2M µs spikes that its smoothing averages away.
- `eval prior --dataset nodes --blame` cannot attribute it: "this prior has no per-term decomposition — blame
  unavailable" (the CatBoost prior isn't blame-decomposable; see Workflow notes).

**Recommendation** (priority order, unchanged in substance from findings-4 F1 — now three sweeps of the same
verdict): (1) bench_fail rows need a real retry/quarantine path — a hung variant should mark its *region* (same
TILE fold family) suspect for deploy, not stream 2M µs point-labels into training (findings-3's `--retry-fails`
recommendation; findings-4 asked for a bounded/clipped sentinel loss — either form works, neither has landed);
(2) land structural-evidence greedy (Ivan's `feature/greedy-structural-evidence`) and extend it past fp16
`ShapeKey`s so a measured 1090 µs row beats an extrapolated 2132 µs pick — this exact handoff is posted in
PR #331; (3) when the learned prior's golden rank diverges ≫ the analytic prior's on a family (166 vs 0 median
here), fall back to analytic ordering for that family's deploy — findings-4 proposed the same analytic-floor gate;
the signal is computable at deploy time from the golden set itself.

## Finding 2 — attention static goldens: recorded pins silently coerce to different configs

`run --bench --golden attention.hd64/hd128` benches golden rows whose knobs are **not** the recorded ones, with no
warning: hd64's recorded `TILE@dd …f1x16/k4 / TILE@pj …f1x8/k8` benched as `f1x4/k4 / f1x8/k2` (STAGE kept), and
hd128's entire config moved (`w4x1→w2x1`, both folds, `d2/cp/ring→d1/cp`) — 27.7 µs vs the recorded 19.9. The
`.dynM` twins pin faithfully (their bare `TILE` reproduces exactly; verified post-edit at 17.2 µs). The regression
window is tight: findings-4 replay-validated these exact pins ("all four attention entries reproduce exactly,
10.5/20.0 µs") on 2026-07-08 against pre-#332 code, so the coercion arrived with #332's flash-form narrowing
(`test_flash_form_narrowing.py`) or the #331 merge — the recorded forms are no longer enumerable and the pin
machinery substitutes the nearest bindable sibling silently. Both statics therefore stay untouched — greedy trails
even the coerced rows (hd64 1.09) and the recorded values (hd128 1.11) — but the recorded configs may now be
unreachable aspirations.

**Recommendation:** the golden A/B must mark a row whose *resolved* knobs differ from the recorded pins (a
`~coerced` flag next to the `!` impossible flag), and `tests/compiler/test_golden_configs.py` should assert every
recorded pin still binds in the current moveset — that turns a silent moveset regression into a red test. Then
re-tune the two statics and re-seed or confirm 10.6/19.9 are dead.

## Finding 3 — mixed goldens are un-evaluable by every prior view

`eval prior --dataset golden` crashed outright on the mixed shapes (`ValueError: unknown dtype 'mixed'` in
`search/analytic.py::_matmul_graph`) — the #331 schema addition never reached the eval path. **Fixed in this PR**
(mixed → f32-A/f16-B graph, matching the traced snippet). Post-fix the crash is gone but all three mixed shapes
still report `SKIPPED: recorded knobs not in the enumeration — pin/dtype mismatch?`: the enumeration join
(`tile_signature`) doesn't recognize the demoted computed-A-cone form's knob spelling, so rank/blame diagnostics
silently exclude the family whose deploy ranking is (per PR #331's retune) the whole ballgame on sm_89.

**Recommendation:** extend `evaluate_golden`'s join to the demoting sync compute-fill signature so mixed rows rank
like any other; add the three mixed goldens to whatever regression harness watches golden ranks.

## Finding 4 — rms_norm k3840: the seeded b32 golden was a featurization artifact; b256 is 2.5× faster

The 2026-07-07 seeding session recorded `REDUCE: b32` at 17.7–18.4 µs with greedy reproducing it as rank 1/8 — this
sweep's greedy picks `b256` at 6.9–7.0 µs, dead even with eager (0.99–1.00× cuBLAS), and `qknorm.k256` moves
b32 → b64 (3.7 µs, 0.74× cuBLAS). Goldens replaced. The enabling change is #322 ("featurize the REDUCE codec on
TILE-less rows"): pre-#322 the prior couldn't distinguish b-widths on these rows, so the seed-era search never
ranked b256 above b32. Corroboration: findings-4's rms_norm seeds (k2048/k4096/k8192, recorded post-#322 under the
refit weights) all picked b256 from the start and are unchanged this sweep — only k3840 and the k256 qknorm
geometry, seeded pre-#322 by #323, were left in the blind window.

**Recommendation:** none for the code — this is the system working post-#322. Worth remembering that goldens seeded
in a featurizer's blind window go stale the moment the featurizer learns to see; the periodic full re-sweep (this
skill) is the mechanism that catches them.

## Finding 5 — same-knob golden rows disagree with the greedy row by up to ±6% within one run

Three shapes benched golden rows with knobs *identical* to the greedy pick, same run, and still differed:
`mlp_gate_up.h4096` greedy 6% **faster** (909.5 vs 968.7, stable ×3), `mlp_down.h4096` greedy 5% **slower**
(393.3 vs 375.9, stable ×3), `q_proj.h3840.mixed` 6% slower — direction varies by shape but is stable per shape.
This is a bench-context artifact (launch order / cache state between the greedy program and the golden re-bench
programs), and it defines the A/B's real noise floor: a consistent sub-6% "win" can be pure artifact even when it
reproduces. It is why `softmax.k8192.dynM`'s stable 5% win and `qkv.h4096`'s 3.5% fused win were categorized
"same/add", not "replace", and why the marginal square regressions (1.04–1.08) are noted but not treated as five
independent search failures.

**Recommendation:** bench the golden rows and the greedy program under the same context (interleave repetitions
rather than sequential programs), or at minimum have `run --bench --golden` re-bench the greedy program alongside
each golden row batch and report the same-knob delta as a calibration line in the table footer.

## Workflow notes

Fixes from findings-4's notes: the **compile budget** fix (#320) held again — zero compile-budget fails. The
`matmul__partial → TileOp` lowering TypeError findings-4 said "deserves an issue" **still fires** (2 rows on
`k_matmul_631750`), the sweep tail is **still fp16-square-shaped** (`square.1024.fp16` was this sweep's slowest
shape at 1144 s), the eager row is **still integer-rounded** (recorded `cublas_us` for small shapes remains
4.0/5.0/7.0-grade), split-K golden program totals **still need hand-summing** (this sweep's parser folds the
epilogue continuation rows — findings-4 asked for a `--json` emitter, which would have obsoleted the parser and its
two bugs), `eval variants --kernel` **still matches only the C identifier** (had to harvest `k_matmul_<hash>` names
from A/B tables first), and `bench_fail` rows still have **no retry/purge path** (F1 — now three sweeps overdue).
One improved: findings-4's "attention dynM recording needs a helper" was less painful this time — the winning bare
TILE could be read straight off the greedy row's `TILE@dd` (the pj fold derives), no `--ab` probe loop needed.

New this sweep:

- **Tune loop 3h47m** (47 shapes; nominal ~2.5 h was for the pre-gemma set — SKILL.md budget updated to ~4 h).
  A/B pass-1 ~1 h cold-compile-bound; the 24-shape × 2 warm confirmation pass ~35 min. Evals are minutes each.
- **`pgrep -f "emmy tune"` self-match burned ~3 h idle**: the wait-for-tune-exit chain's own command line contains
  the pattern, so it never fired (and a `pkill -f "while pgrep"` suicided its own ssh session). Any wait-loop for
  a remote process must use a self-match-proof pattern (`pgrep -f "[e]mmy tune"`) or a completion marker file —
  worth encoding in this skill next time.
- **The `! impossible` flagger false-positives on every memory-bound `.dynM` golden row** ("implies 262 TFLOP/s >
  83 device peak" on a 4.1 µs rms_norm) — the known dyn AI-floor issue; it computes FLOPs from the full symbolic
  range rather than the hint. Cosmetic here but it prefixes rows (`! golden …`) and broke this sweep's first log
  parse; fix the FLOP estimate for symbolic axes or suppress the flag on hint-benched rows.
- **`eval prior --dataset golden` crash on mixed** (F3) — fixed in this PR; the remaining join miss is open.
- **`--blame`/`--ablate` are unavailable for the shipped prior** ("this prior has no per-term decomposition") —
  #333's attribution decomposes only priors with per-term scores, not the CatBoost model actually deployed. The
  skill's instruction to "cite the blame row" is unfollowable today; either wire a surrogate decomposition or
  scope the skill text. Also `--dataset nodes --kernel` matches op labels only (`matmul`), not golden names or
  kernel hashes — per-shape blame is impossible even where blame works.
- **`eval variants`' -O3 column is mostly `—`**: only greedy-path contenders get -O3 rebenches, and on the F1
  family even those rows are missing because the -O3 rebench itself bench_failed. The skill's "read the -O3
  column before calling a search shortfall" is only intermittently possible; consider always -O3-rebenching the
  top-3 plus the golden config when one is recorded.
- **`eval golden` covers only matmul kinds** — the attention pin coercion (F2) had to be diagnosed by hand from
  A/B tables; extending the found/golden diff to attention/rms_norm/softmax would have made it a one-liner.
- **Dynamic attention goldens can't record per-contraction tiles** (schema: "the masked-flash pin doesn't resolve
  `TILE@<axis>`") — fine today because the pj fold derives deterministically from the bare TILE (verified: recorded
  `w2x1/f1x16/k4` reproduces `dd=f1x16/k4, pj=f1x8/k8` exactly), but the constraint is invisible until the schema
  throws; noted here so the next editor doesn't rediscover it.
