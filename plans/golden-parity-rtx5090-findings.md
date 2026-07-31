# Golden parity vs cuBLAS — RTX 5090, 2026-07-30 (branch feature/remove-place-knob)

Goal (per the optimizing-gemma-4-12b-rtx article): every golden kernel at parity or better vs cuBLAS/eager on the
local 5090; update schedules where a better config exists. Method: `scripts/bench_golden_set.py --filter gemma4_12b`
over all 245 card-local goldens in BOTH lanes (std, then `EMMY_FAST_MATH=1`), then manual pinned `--ab` exploration on
the losers (the manual golden method — no tuner sweep; the branch's step-7 re-key wiped the DB/prior by design).
Wall time: ~2 h per sweep lane + ~1.5 h of `--ab` rounds.

**Sweep result (live, before edits)**: std 156/244 at ≥0.98× eager, fm 204/245 — matching the article's 147/254 and
214/254 within measurement drift. **After edits**: the m192 tier, the m32 down-proj family and the fused/dynM cut
routes moved to parity or better; the remaining losers are the documented research-class residuals (below).

Measurement caveats: the box ran ~20–25% slower than main's #444 session (live eager 147 µs where #444 recorded
cublas 120 — emmy µs scaled identically), and the eager reference itself swings ~10% run-to-run (74–86 µs observed on
one shape). New entries record same-run (emmy, eager) pairs so ratios stay apples-to-apples; treat single-digit
percent ratios as noise.

## What was recorded (all reproduced ≥2 runs or far above the noise band)

| shape | old µs (recorded) | new µs (live) | eager (live) | was → now | change |
| --- | --- | --- | --- | --- | --- |
| mlp_down.m192.lin | 193.1 (0.62×) | 160.6 | 147 | 0.63× → 0.91× | replace std: `w1x8 f2x2/k4 g8a gm8 d2/tma/ring` |
| qkv_cat.m192.lin | 89.3 | 76.8 | 79 | 0.76× → 1.03× | replace std: `w2x4 f2x2/k4 g2a gm8`; add fm twin (71.0) |
| qk_global_cat.m192.lin | 89.8 | 79.5 | 79 | 0.76× → 1.00× | replace std: same family; add fm twin (75.8) |
| gate_up_cat.m192.lin | 274.8 | 290.4 | 290 | 0.89× → 1.00× | replace std: `w2x4 f2x2/k4` unsplit; add fm twin (263.3) |
| norm_qkv.m192.lin.cut | — | 81.1 | 86 | 0.62× → 1.06× | NEW routing entry (PLACE=cut) |
| norm_qk_global.m192.lin.cut | — | 83.1 | 85 | 0.61× → 1.02× | NEW routing entry |
| norm_gate_up.m192.lin.cut | — | 294.2 | 307 | 0.58× → 1.04× | NEW routing entry |
| mlp_down_fused.m192.lin.cut | — | 169.2 | 170 | 0.53× → 1.00× | NEW routing entry |
| norm_qk_global.dynM.lin.cut | — | 212.5 | 217 | 0.72× → 1.02× | NEW routing entry (at the hint) |
| norm_gate_up.dynM.lin.cut | — | 654.3 | 636 | 0.61× → 0.97× | NEW routing entry (bounded by the dynM matmul piece) |
| norm_qk_global.m32.lin.cut | — | 21.3 | 33 | 0.80× → 1.55× | NEW routing entry — the #445 cut win through the phase-4 realizer |
| mlp_down.m32 (std) | 74.1 (g8k/k2) | 73.7 | 77 | tie | replace: `k4/g4a` single-kernel, converges with the .lin twin (Finding 1) |
| mlp_down.m32.lin (std twin) | — (fm only) | 76.1 | 82 | 0.75× → 1.07× | NEW std entry `f2x2/k4 g4a` |
| mlp_down_fused.m32.lin.cut | — | 80.4 | 98 | 0.82× → 1.22× | NEW routing entry |

The m192 rule that fell out: the #444 `f4x1/k8` tile-M-64 records were never at parity — at M=192 the winning family
is **tile-M-32 (`f2x2/k4`) + a cross-CTA split sized to fill the card** (`w2x4`+`g2a` at short-K wide-N, `w1x8`+`g8a`
at long-K), the same under-fill logic as the m32 splits, one rung wider. The fused m192/m32/dynM forms all route
better through PLACE=cut once their matmul piece tier is right — the fused d1/sync computed-A form loses ~1.6–1.9× to
the cut at these widths.

## Finding 1 — layout-blind golden join deployed the wrong twin's config (silent-wrong-deploy class)

`mlp_down.m32.lin` (std) deployed the **non-.lin** twin's `k2/g8k` golden (107 µs realized on the transposed layout
vs 76 for the right config) even after a correct std `.lin` entry was recorded — the evidence join picked the
lower-µs same-shape row across layouts (the ShapeKey aspect-blind shadow class; the q_proj_global.m32 shadow was the
same bug). The fm lane was immune only because its `.lin` entry happened to be the cheapest. Data-level fix applied:
both layouts re-recorded onto the config that ties on both (`k4/g4a`, 73.7/76.1). **Recommendation**: the join should
key on layout (trans_b) — a compiler-side fix; until then any twin-layout re-record must converge both entries.

## Finding 2 — the fused gate⊗up m4096 cut cannot lower (residual, compiler-class)

`mlp_geglu.m4096(.lin)` sits at ~0.45× in BOTH lanes; the fm matmul tier (gate_up_cat.m4096.lin fm = 3269 vs eager
4557, 1.39×) would rescue it via PLACE=cut, but the cut fails to lower: `plan_from_graph: node 'mul' has non-CudaOp
'TileOp'` — the geglu combine piece never re-recognizes (the #389 multichannel-split class). This is the one shape
family where a known-good config exists and no data edit can route to it. Recommendation: make the cut's residual
combine piece re-enter recognition (or refuse the seam and fall back loudly).

## Finding 3 — one flaky accuracy NaN on mlp_geglu.m4096.lin (std)

The std sweep's `run --golden` hit `accuracy check failed vs eager: max_diff=64.0 mean_diff=nan` once; two re-runs
passed. Un-reproduced; worth a watch during the phase-5 gate (a nan in the fused geglu path would be serious).

## Remaining below-parity (documented residuals, left alone)

- **attention.hd512 family** (0.84–0.95×): the hd512 codegen-bound residual (memory: parity only at s2048).
- **m2048/m4096 std-lane wide matmuls** (0.61–0.95×): the consumer-Blackwell FP32-accumulate half-rate wall; the fm
  lane covers all of them at ≥1.0× (article section 2) — no std-lane config can cross a compute wall.
- **mlp_geglu.m4096** — Finding 2. **mlp_geglu.m32/m64** (~0.91–0.94×): fused floor entries; cut nulls at multichannel.
- **Tiny launch-bound rows** (rms_norm.k3840 7 vs 6 µs, kv_proj_global m32 12 vs 10 µs, qknorm): 1–2 µs from parity,
  launch overhead, below the noise floor.
- **m256/s2048 std-lane fused norms** (0.87–0.95×): fm lane is at 1.14–1.32× on the same shapes; std within 10%.

## Workflow notes

- `bench_golden_set.py` has no `--lane`-aware loser table; the std/fm cross-referencing (which lane covers a shape)
  was hand-assembled from two JSONs. A merged two-lane view with per-shape best-lane ratio would cut the analysis.
- The eager reference swing (~10% run-to-run, and ~20–25% vs the #444 session's records) makes recorded `cublas_us`
  cross-session comparisons unreliable — a clocks/power-state stamp next to recorded µs would disambiguate drift
  from regression.
- `--ab PLACE=cut` prints the pieces as separate rows but no total; summing by hand per run was the single most
  repeated manual step. A `total` line for multi-kernel A/B rows would help.
- The golden A/B path realizes pins the deploy enumeration never offers (g8a and gm8 at one M-tile benched fine but
  could not deploy) — recording from an `--ab` row therefore needs a follow-up un-pinned `run --bench` to confirm the
  entry actually matches an offered row. `eval golden`'s offer audit passed while the deploy still missed (Finding 1),
  so the audit alone is not sufficient proof of deployability.
