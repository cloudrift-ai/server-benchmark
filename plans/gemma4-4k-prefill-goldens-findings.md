# gemma-4-12B M=4096 prefill-chunk goldens + fm-never-loses — session findings (2026-07-21, RTX 5090)

Execution of `plans/gemma4-fm-universal-and-4k-goldens.md` (WS1 measure, WS2 static m4096 seeding, WS3
symbolic mid-width, WS4 invariant + serving matrix), branch `feature/gemma4-fm-universal-4k-goldens`.
Primary target: single-batch 4K-in/4K-out (rides ONLY the static M=4096 chunk + bucket-32 decode twins —
no symbolic steps at c=1).

## WS1 — cold twin baselines (the work list)

M=4096 prefill-chunk twins captured (`capture_gen_twins.py --prefill-bucket 4096 --decode-bucket 0
--no-symbolic`), cold-benched under the serving evidence env (twins.db + empty online, -O3, 20 iters):

| twin (cold) | std | fm |
| --- | --: | --: |
| pre4096 | 1532.3 | 1243.2 |
| post4096 | 11846.8 | 14560.6 |
| pre4096-global | 1677.4 | 1466.8 |
| post4096-global | 12307.5 | 14944.7 |

The fused RMSNorm→gate⊗up→GeGLU megakernel (`k_linear_mean_reduce`, PLACE=fuse d1/sync) cold-picks
~8.7-8.8 ms = 70-74% of each post twin, BOTH lanes — the m256 "fused computed-A cannot reach the cut"
verdict at 16x the M. fm is WORSE than std on the post twins cold (the fm fused tile is slower and the
prior extrapolates badly at 4K M) — the measured root of the 4K/4K fm net-loss.

## WS2 — the M=4096 golden set (seeded, 3x medians, spread <=1.1%)

Method: manual pinned `--ab` (NOT the tuner), anchors = the s2048 rows + the plan's w4x4/w8x2 M-warp
variants. Winners per shape (median µs; std | fm | eager-cuBLAS):

| shape (.lin) | std winner | fm winner | eager | fm vs eager |
| --- | --: | --: | --: | --: |
| q_proj (N4096) | 680.9 | **530.6** | 701.1 | 1.32x |
| kv_proj (N2048) | 372.8 | **261.3** | 361.6 | 1.38x |
| o_proj (K4096) | 675.3 | **421.3** | 673.5 | 1.60x |
| gate_up split half (N15360) | 2426.3 | **1665.5** | 2318.4 | 1.39x |
| mlp_down (K15360) | 2399.3 (g2k) | **1527.5** | 2321.9 | 1.52x |
| q_proj_global (N8192) | 1257.3 | **913.9** | 1302.8 | 1.43x |
| k_proj_global (N512) | 111.2 | **71.9** (w4x1) | 105.7 | 1.47x |
| o_proj_global (K8192) | 1301.9 | **855.3** | 1297.7 | 1.52x |

(canonical-layout twins within ~2% of .lin throughout, seeded beside them; fm = f16acc
`w4x2/f4x8/k4 d2/tma/ring` SERIAL everywhere except k_proj_global's `w4x1`.)

Key structure findings:

- **Split-K drops out entirely at M=4096** (vs the s2048 rows' g2k on fm deep-K): the grid no longer
  starves — fm serial beats fm g2k on down (1519 vs 1641) and o_proj (420 vs 517). std mlp_down keeps
  g2k (2399 vs serial 2459).
- **The plan's w4x4/w8x2 M-warp hypothesis is REFUTED**: neither realizes `d2/tma/ring` at this smem
  footprint (`pin_unmatched` — depth clamps to d1/tma / staging off). `w4x2` stands; no `_WARP_UNITS`
  addition needed. Narrow-N k_proj_global instead moves DOWN to `w4x1/f4x8/k4` (71.9 vs w4x2's 123).
- **The fused geglu megakernel is hopeless at M=4096, the CUT is the route**: cold fused (both lanes)
  ~10-11.2 ms; the pinned std w4x2/f4x8/k4 fused tile is a 191.7 ms catastrophe. `PLACE@cone=cut` with
  cold halves = 7.3 ms; with the seeded split-half goldens the cut reaches ~3.7 ms fm — beating the
  4.88 ms unfused eager. Seeded as `mlp_geglu.m4096.cut(.lin)` + `mlp_gate_up_split.m4096(.lin)`.
- Norms: rms m4096 → b128 (20.0); qknorm k256 r65536 → **b64** (31.3, the family's first b64 winner);
  k512 r4096 → b128 (5.2); k512 r65536 → b128 91.3 vs eager 71.1 = 0.78x (the one emmy-loses row —
  seeded as misdeploy guard; deeper/shallower blocks all lose).

## Verification (twin re-bench under the serving evidence env, deploy-from-tier proof)

| twin | cold std | seeded std | cold fm | seeded fm |
| --- | --: | --: | --: | --: |
| pre4096 | 1532 | **1313** | 1243 | **986** |
| post4096 | 11847 | **7407** | 14561 | **4653** (3.1x) |
| pre4096-global | 1677 | **1396** | 1467 | **1011** |
| post4096-global | 12308 | see hang note | 14945 | see hang note |

m64 decode twins unregressed (46.2 / 261.0); `eval golden --in-model`: **MATCH 101 / DRIFT 0 / GAP 0
both cards** with the m4096 block in.

⚠ **In-twin o_proj_global bk>1 TMA-ring hang (both lanes)**: the o_proj_global.m4096 golden tiles
(std `w4x2/f2x4/k2`, fm `w4x2/f4x8/k4` — both 3x-clean STANDALONE, and the same family works in-twin
on post4096's K=4096 o_proj and K=15360 down) HANG >1 s when they realize on the post4096-global
twin's o_proj (K=8192, the attn_width-8192 input form). Cold's bk1 `w4x2/f4x8` ran 1267 µs on the same
form. Not reproducible standalone (bare linear, or linear+residual). Worked around by re-seeding the
o_proj_global.m4096(.lin) rows at bk1 configs (benched standalone 3x); the bk>1-in-this-form hang is
a codegen bug left open for a dedicated session — the misdeploy class the "matched-but-unrealizable
golden" memory warns about, now with a twin-level reproducer (`_tune/prefill-4k/post4096-global.json`).

## WS3 — symbolic mid-width path

(to fill)

## WS4 — invariant + serving matrix

(to fill)
