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
| post4096-global | 12308 | **8791** | 14945 | **5650** (2.6x) |

(the global row benches via `--warmup 0` — see the hang note; its o_proj k2/k4 tiles realize FASTER
in-twin than standalone: std 1274, fm 768.) Seeded fm chunk-twin total per 4K prefill:
40×5.64 + 8×6.66 ≈ **279 ms** (was ~650 cold std).

m64 decode twins unregressed (46.2 / 261.0); `eval golden --in-model`: **MATCH 101 / DRIFT 0 / GAP 0
both cards** with the m4096 block in.

⚠ **post4096-global bench-harness hang, BISECTED (not a kernel bug)**: with the geglu cut deployed,
this twin's `--bench` hangs on its FIRST kernel (o_proj_global) — at EVERY o_proj tile tried
(bk1/k2/k4 × tma/cp × std/fm), while `PLACE@cone=fuse` pinned benches clean and the same cut program
benches clean on the non-global twin. The bisect: **`--warmup 0` (captured-graph measurement from the
first iter) benches clean; any warmup ≥ 1 (the per-kernel-event EAGER framing) hangs** — and the
plain path, compute-sanitizer memcheck (0 errors), and three serving runs of the same program (raw
`run_device` launches) are all clean. So the hazard is the bench harness's eager per-kernel
record/launch/wait framing × this specific 9-kernel program, NOT the tiles and NOT serving — the
initial bk1 de-rate bought nothing and was reverted (std k2 1288/1302, fm k4 826/855 restored; the
k4 realizes at 768 µs in-twin). Root cause of the harness interaction stays open (next probe:
extend the watchdog + cuda-gdb warp PCs during the hang); bench this twin with `--warmup 0`.
Separately, mlp_down.m4096 std gained a SERIAL f4x8/k4 sibling (2444/2464) because the g2k row
cannot realize on the twin's epilogue-fused down (no split-K offer → loud fall-through).

## WS3 — symbolic mid-width path: MOOT for these workloads (measured)

The step-width histogram (temp instrumentation, not landed) shows the c=8 4K/4K run schedules ONLY
pure T=4096 chunks + cudagraph-replayed decode — zero symbolic steps (vLLM never mixes decode into a
chunk step at this config: 4096 chunk + decode tokens > mnbt). c=1 rides static-4096 + bucket-32
only. The dynM minimax re-record (`_tune/prefill-4k/sweep5-dynm.sh`, written but not run) is optional
hygiene for OTHER workloads (mid-length prompts producing tail chunks), not a lever for these.

## The m32 CUT flip (decode TPOT, beyond the plan)

With the m64/m256/m4096 cut precedent, the M=32 gate⊗up was the last fused holdout. Seeding
`mlp_gate_up_split.m32(.lin)` (each half at the 76-78 µs weight-stream floor) makes the cut win at
decode M too: **158.0 (can) / 162.8 (.lin) vs the fused megakernel's 179/173 AND unfused eager
(159.7/166.6)** — 3x at <=0.2%. m32 post twin: 275 → 251 µs (std and fm). Serving A/B (fresh pack,
c=1 fm): TPOT 20.33 → **20.10 ms**, tok/s 48.93 → 49.42, TTFT unchanged 450 — the twin-level
~1.15 ms/step compresses to ~0.23 in serving (the cut's +4 kernels/layer add node overhead inside
vLLM's decode capture); real but modest. The
cut's three producer kernels (stat/scale/combine) seeded as regression anchors (the pointwise pair
records dtype fp32 to join the cut-emitted scalar `is_warp=False` fork keys); audit lands
**MATCH 111 / DRIFT 0 / GAP 0**.

## 4090 port (same-day, remote manual --ab)

The full m4096 tier ported to `rtx4090_sm89_gemma4.yaml` (d2/cp/ring): the fm f16acc `w4x2/f4x8/k4`
serial family transfers at 1.3-1.5x over eager on every shape; `w4x1` k_proj_global transfers; gm8
pays MORE on sm_89 (gate half .lin 2297→2103, 9%) and records on the fm halves. Divergences: rms
m4096 → b256, qknorm k256 → b64 (34.9 vs 43.9), k512 r65536 at eager parity (unlike the 5090's
0.78x); the std lane runs 0.83-0.94x eager at this M (guards; fm is the perf lane). Cut rows seeded
in round 2 (std-lane totals 8727/10937 vs fused guards 12131/16153; the fm lane deploys the same
rows over its gm8 halves at ~4.9 ms projected vs eager 6.5). The m32 cut flip and the serving
KV-reclaim benefits port automatically (code, not per-card data); a 4090 serving A/B was not run
(no serving venv on the rental box) — the twin-level evidence is the port's proof.

## WS4 — invariant + serving matrix

Static invariant landed (`test_fast_math_rows_never_record_slower_than_std_siblings`); the 4 violating
fm rows dropped (5090 attention.hd256 + .dynM siblings, 4090 attention.hd256 + mlp_down.m32.lin) —
behavior-neutral: the fastest-first picker could never realize them (a faster row matches first in
either lane).

Serving A/B (protocol: twins.db + empty online + per-lane packs, seed 0; SINGLE = c=1, 3 prompts;
LONG = c=8, 16 prompts; both 4K-in/4K-out, mml 8448, mnbt 4096, bucket 32):

| arm | out tok/s | TTFT mean/med (ms) | TPOT mean/med (ms) |
| --- | --: | --: | --: |
| stock single | 56.97 | 561 / 545 | 17.42 / 17.40 |
| emmy single (std) | 48.32 | 612 / 606 | 20.55 / 20.46 |
| emmy single FAST_MATH | 48.93 | **458 / 450** | 20.33 / 20.32 |
| stock long | 371.65 | 2449 / 2081 | 20.93 / 21.02 |
| emmy long (std) | 239.65 | 25118 / 2725 | 21.96 / 22.18 |
| emmy long FAST_MATH | 186.80 | 40320 / 2290 | 21.39 / 21.46 |

**Single-batch (the session's primary target): emmy fm TTFT 450 ms BEATS stock's 545 by 17%** (std
606 = 1.11x) — down from the pre-session multi-second class; the whole remaining single-batch loss is
the decode TPOT residual (20.3 vs 17.4 — the computed-A pipeline, out of golden scope). fm ≥ std on
both single-batch metrics: the fm-never-loses gate holds where the static tier is the deploy surface.

**The c=8 LONG workload was UNMOVED by the static m4096 tier** (std 236.7→239.7, fm 182.0→186.8) —
and the step-width histogram (temp instrumentation, one run) REFUTED the symbolic-path hypothesis:
the c=8 run schedules ONLY pure T=4096 prefill chunks (16, riding the static twins) + cudagraph-
replayed decode steps. ZERO symbolic steps → WS3's dynM re-record is irrelevant to both benchmark
workloads (kept as optional hygiene).

**The real c=8 gap was KV-cache admission queueing.** vLLM profiles KV after the emmy boot, and
emmy's non-KV footprint ran ~3.7 GiB over stock (KV 17,727 vs 22,810 tokens; fm even smaller at
15,480 — why fm LOST std on the long workload: less cache, more queueing). The split (memory probe):
~2.0 GiB = the tied embed/lm_head matrix held TWICE across frameworks (torch head + the runner's
folded cupy-side table), ~1.4 GiB arena (capacity-sized, legitimate), ~0.5 GiB allocator slack.

**Fix landed** (`adopt_embed_table`): on tied checkpoints `load_weights` hands the raw head tensor
to the runner (embed_scale re-applies at gather in fp32; the shared table stays raw — a folded scale
would retemper every logit) and releases freed blocks to the driver before vLLM's profiling.
KV 17,727 → **27,532 tokens** (now ABOVE stock's 22,810). Re-run:

| arm (post-reclaim) | out tok/s | TTFT mean/med (ms) | TPOT mean/med (ms) |
| --- | --: | --: | --: |
| emmy long (std) | 338.14 (was 239.7) | 2657 / 2409 (was 25118) | 23.01 / 23.05 |
| emmy long FAST_MATH | **345.55** (was 186.8) | **1985 / 1789** | 22.66 / 22.72 |

fm ≥ std restored on the long workload; emmy fm TTFT BEATS stock (1985/1789 vs 2449/2081); residual
throughput 345.6 vs 371.7 = 0.93x is entirely the decode TPOT class (22.7 vs 21.0 = 1.08x).
