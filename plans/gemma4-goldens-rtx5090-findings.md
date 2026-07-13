# Gemma-4 golden seeding — RTX 5090 (sm_120), manual A/B sweep findings

- **Date / method**: 2026-07-13, local RTX 5090, manual pinned `--ab` exploration (no tuner runs, no search-code
  changes). Per shape: one broad round of 4–6 pinned candidates from the card's known winner families via
  `emmy run --bench -c "<snippet>" [--dynamic seq_len@…] --ab …`, 1–2 refinement rounds, winners reproduced 3× (all
  at <1% spread). `cublas_us` = median live `Eager PyTorch` row across the shape's runs. Sweep wall time ~25 min
  (~40 `run --bench` invocations, ~15–25 s each).
- **Deliverable**: `goldens/rtx5090_sm120_gemma4.yaml` — a NEW themed per-GPU golden file (same
  `gpu_name`/`compute_cap` header as `rtx5090_sm120.yaml`; the loader globs `*.yaml` and the live-GPU scoping
  merges both). **27 entries / 15 names**: 5 matmul shapes × {static, dynM} × {standard, fm}, attention hd256
  × {static, dynM} × {standard, fm}, rms_norm k3840 (+dynM), qknorm k256.
- **Shapes** (Gemma-4-12B: hidden 3840, intermediate 15360, 16 Q / 8 KV heads, head_dim 256; M = 512 =
  `DEFAULT_SEQ_HINT`, fp16): q_proj 512×4096 K3840 · kv_proj 512×2048 K3840 · o_proj 512×3840 K4096 ·
  mlp_gate_up 512×30720 K3840 (fused 2×15360) · mlp_down 512×3840 K15360 · SDPA 16h/hd256/causal ·
  rms_norm 512×3840 · qknorm 4096×256.

## Headline: every unseeded projection shape cold-greedy-misdeploys; two shapes hang outright

- **kv_proj (static AND dynM): scalar `b256` tile at ~37 000 µs vs cuBLAS 48 — ~770× off.** mlp_down greedy is a
  ~280 ms/launch kernel (aborts the 10 s bench cap). These are the same misdeploy class the gemma tune reports
  found on sm_89 — now shown on sm_120 with fresh shapes.
- **mlp_gate_up.dynM greedy picks a kernel that never completes the 1 s accuracy probe; attention.hd256.dynM
  greedy exceeds the 10 s bench cap** — the "cold greedy on an unseeded shape picks a hang" hazard from the 4080
  attention seeding, reproduced on the 5090 for two more shapes.
- After seeding, the goldens give `tune --golden` / prior-refit ground truth for all of these; the pinned
  configs replay true through the golden plumbing (see Validation).

## Per-shape outcomes (all -O3 `run --bench` A/B rows, 3× reproduced; split-K rows are main + finalize totals)

### Standard lane (f32-accumulate)

| shape | config | emmy µs | cuBLAS µs | vs cuBLAS |
| --- | --- | --: | --: | --: |
| q_proj | `w4x2/f2x4/k2 g8k d2/tma/ring` | 84.2 | 97.6 | **1.16×** |
| q_proj.dynM | same | 84.8 | 98.7 | **1.16×** |
| kv_proj | `w4x2/f2x4/k2 g8k d2/tma/ring` | 45.3 | 48.8 | **1.08×** |
| kv_proj.dynM | same | 45.8 | 48.9 | **1.07×** |
| o_proj | `w4x2/f2x4/k2 g4k d2/tma/ring` | 82.0 | 103.3 | **1.26×** |
| o_proj.dynM | same | 82.1 | 99.8 | **1.22×** |
| mlp_gate_up | `w4x2/f4x8/k4 d2/tma/ring` | 561.7 | 573.3 | **1.02×** |
| mlp_gate_up.dynM | same | 562.4 | 561.7 | 1.00× |
| mlp_down | `w4x2/f2x4/k2 g4k d2/tma/ring` | 286.2 | 286.3 | 1.00× |
| mlp_down.dynM | same | 286.2 | 286.3 | 1.00× |
| attention.hd256 | `dd w2x1/f1x8/k16, pj w2x1/f1x32/k4, d1/cp` | 82.0 | 30.7 | **0.37×** |
| attention.hd256.dynM | bare `w4x1/f1x2/k16, d2/cp/ring` | 39.3 | 30.7 | **0.78×** |
| rms_norm.k3840 (+dynM) | `b256` | 6.3 | 6.1 | 0.97× |
| qknorm.k256 | `b64` | 3.6 | 4.1 | **1.14×** |

Every matmul shape is at or past cuBLAS in the deployable (gate-off) lane — the split-K `g4k`/`g8k` region is
where the small-N gemma projections live (their un-split grids underfill 170 SMs), and the fused gate_up width
(N=30720) rides the same `w4x2/f4x8/k4 d2/tma/ring` big-tile family as the h4096 shapes.

### Fast-math lane (f16-accumulate atom, kept beside the standard entries)

| shape | config | emmy µs | vs standard | vs cuBLAS |
| --- | --- | --: | --: | --: |
| q_proj (+dynM) | `f16acc w4x2/f4x8/k4 g2k d2/tma/ring` | 61.5 / 61.7 | 0.73× | **1.59×** |
| kv_proj (+dynM) | `f16acc w2x4/f2x4/k4 g2k d2/tma/ring` | 36.4 / 36.3 | 0.80× | **1.34×** |
| o_proj (+dynM) | `f16acc w4x2/f4x8/k4 g2k d2/tma/ring` | 64.1 / 63.9 | 0.78× | **1.61×** |
| mlp_gate_up (+dynM) | `f16acc w4x2/f4x8/k4 d2/tma/ring` | 362.4 | 0.65× | **1.58×** |
| mlp_down (+dynM) | `f16acc w4x2/f4x8/k4 g2k d2/tma/ring` | 214.2 | 0.75× | **1.34×** |
| attention.hd256 | fm P·V (`pj` f16acc) | 77.1 | 0.94× | 0.40× |
| attention.hd256.dynM | bare fm PV plan `w4x1/f1x32` | 36.1 | 0.92× | **0.85×** |

The big-tile f16acc family sweeps the gemma projections exactly as it did the h4096 set (PR #350) — every
matmul 1.3–1.6× past cuBLAS HGEMM. The masked (.dynM) tiles pay no penalty anywhere: dynM ≈ static within noise
on all five matmuls, matching the 07-12 sweep's observation.

## Finding 1 — static hd256 flash is locked out of the form its own masked twin runs (2.1×)

Every static `TILE@dd`/`TILE@pj` pin at the `w4x1` geometry is loudly rejected ("unreproducible pin": realized
`w2x1/f1x8/k16` + `w2x1/f1x32/k4`), and any `d2+` STAGE degrades to `d1` — so the static flash tops out at
82.0 µs (4% occupancy, 255 regs). The **masked dynM form takes `w4x1/f1x2/k16 + d2/cp/ring` happily and runs
39.3 µs** — 2.1× faster, at 17% occupancy, on the *same* card and problem. The static form narrowing is leaving
2× on the table for hd256; worth a look at whether the w4x1 static form is refused for a real correctness
reason or just never offered. (The pin-verification gate from #335 made this a 2-minute diagnosis — every
degrade was flagged, nothing silent.)

Torch SDPA is 30.7 µs here, so even the best emmy flash (36.1 fm dynM) is 0.85× — the residual is the known
hd256 register-pressure codegen gap (243–255 regs), not search.

## Finding 2 — cp.async beats TMA by ~16% on the hd256 flash K/V stream

`d1/cp` = 82.0 vs `d1/tma` = 97.5 µs (static), and the dynM winner is `d2/cp/ring` (39.3) vs `d1/tma` (44.5).
All the 5090 hd64/hd128 attention goldens ride `d2/tma/ring` — hd256 inverts the transport preference. Deeper
staging (`d3`) and `p2` are within noise of `d2/cp/ring`; `FAST_EXP` adds nothing (77.0 vs 77.1).

## Finding 3 — the fm bare-TILE spelling for masked flash must be the PV plan, and the dd spelling hangs

Pinning the dynM fm sibling as the dd geometry (`f16acc w4x1/f1x2/k16`) degrades to a **>10 s hang** — not just
the scalar-fallback slowdown the 4090 yaml note warned about. The spelling that reproduces is the realized PV
plan (`f16acc w4x1/f1x32`, read off the standard row's realized `TILE@pj`), matching the hd64/hd128 precedent.
The recorded entry carries a comment pinning this down.

## Finding 4 — `run --golden` still cannot validate the shapes whose greedy pick hangs

`run --bench --golden` on mlp_down(.dynM), mlp_gate_up.dynM and attention.hd256.dynM aborts on the greedy row's
bench_fail before benching a single golden row — the exact 4080-findings bug, still open, now hitting 4 of 15
seeded names. Workaround used here: pin the deploy via `EMMY_KNOBS=<recorded knobs>` so the greedy row is sane,
then `--ab` the candidates (this also mirrors what the golden replay would measure — the pinned rows were
reproduced 3× each). The other 11 names replay clean through the golden plumbing (values within noise of the
recorded `emmy_us`, no integrity flags).

## Finding 5 — false-positive integrity flag on the dynM rms_norm replay

`run --bench --golden gemma4_12b.rms_norm.k3840.dynM` flags the (correct, 6.5 µs) row
`impossible: implies 310 TFLOP/s > 105 device peak`. The static twin at the same latency doesn't flag — the
arithmetic-intensity floor check appears to mis-size the symbolic axis for reduce-tier dynM kernels (a ~50×
FLOP overcount). Diagnostics-only, but it poisons the one integrity signal the golden workflow leans on.

## Workflow notes

- **The `--ab` + `--json` flow held up end-to-end**: pinned rows with realized-knob integrity flags caught every
  form degrade loudly (the #335 gate paying off); zero silent no-ops this sweep.
- **`EMMY_KNOBS`-pinned deploy is the workaround for greedy bench_fail aborts** — same knob grammar as `--ab`,
  works for `-c`/`--dynamic` runs. Without it, 4 of 15 shapes can't be measured at all (greedy aborts the run
  before any pinned row benches). The 4080 report's ask stands: the golden/ab bench should survive a greedy-row
  bench_fail and still report the pinned rows.
- Greedy-vs-ab attribution: an `EMMY_KNOBS`-pinned config in the *greedy* row position benches ~7% slower than
  the identical config as an `--ab` row on split-K pairs (in-program per-launch attribution vs isolated
  re-bench) — record from `--ab` rows only, consistently.
- `g16k` on K=3840 is refused with a clear diagnostic (slice 240 not divisible by the mma K-step 32) — good
  loud-failure behavior, no wasted bench slot.
- Follow-up candidates: (1) re-run `scripts/golden_knob_heuristics.py` to refit the analytic weights over the
  new dynM rows (deliberately NOT done here — dataset-only change); (2) the static-hd256 form-narrowing lockout
  (finding 1); (3) the dynM FLOP-floor overcount (finding 5); (4) a gemma layer-0 re-tune on this card to see
  how much of the 2.13× e2e gap the seeded prior ground truth recovers.

## Repro / artifacts

Work dir `_tune/gemma4-goldens-5090/` (gitignored): per-shape `--json` A/B records incl. all reproduction
passes (`<shape>.<static|dynM>[.r2|.rep{1..3}].json`). Every entry's exact repro command is recoverable from the
yaml knobs, e.g.:

```bash
# any matmul entry (pinned replay, dodges the greedy abort):
EMMY_KNOBS="TILE=a:mma_m16n8k16_f16_f16/w4x2/f4x8/k4,REDUCE=g2k,RASTER=,STAGE=d2/tma/ring,WSPEC=" \
  venv/bin/emmy run --bench -c "torch.matmul(torch.randn(512,15360,dtype=torch.float16), torch.randn(15360,3840,dtype=torch.float16))"
# golden replay (works for 11 of 15 names):
venv/bin/emmy run --bench --golden gemma4_12b.q_proj
```
