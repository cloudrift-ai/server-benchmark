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
| attention.hd256 | `dd w4x1/f1x4/k16, pj w4x1/f1x32/k2, d2/tma/ring` (nt=4 alt) | 34.8 | 30.7 | **0.88×** |
| attention.hd256.dynM | bare `w4x1/f1x4/k16, d2/tma/ring` (nt=4 alt) | 35.5 | 30.7 | **0.86×** |
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
| attention.hd256 | fm P·V (`pj` f16acc), `d2/cp/ring` | 36.5 | 1.03× (loses) | 0.84× |
| attention.hd256.dynM | bare fm PV plan `w4x1/f1x32`, `d2/cp/ring` | 36.1 | 0.99× | **0.85×** |

The big-tile f16acc family sweeps the gemma projections exactly as it did the h4096 set (PR #350) — every
matmul 1.3–1.6× past cuBLAS HGEMM. The masked (.dynM) tiles pay no penalty anywhere: dynM ≈ static within noise
on all five matmuls, matching the 07-12 sweep's observation.

## Finding 1 — the hd256 flash gap was a greedy rank miss + a pin-spelling trap, NOT a form lockout

**Corrected 2026-07-13 (same day, follow-up commit).** The sweep's first reading — "the static form narrowing
refuses the `w4x1` geometry the masked twin runs" — was wrong. The `w4x1/f1x2/k16` rows sit in the static fork
(the `twisted_warp_moves()` grid enumerates `(um=4, nt=2)`, and the 512-seq divisibility gates pass); pinned
with the CORRECT spelling they realize cleanly and run **35.6 µs — 2.3× faster than the 82.0 µs the sweep first
recorded, and at parity with the masked twin (36.4)**. Two separate things had masked it:

1. **Greedy misranks the flash forms**: cold greedy picks `w2x1/f1x8/k16` (64 KB K/V slabs, 64-thread CTAs,
   97 µs) over the `w4x1` form (16 KB slabs, 35.6 µs) sitting in the same fork — a 2.7× rank miss. That's a
   prior problem, and exactly what the recorded golden now pins.
2. **The static pin contract is all-or-nothing across BOTH keyed pins**: `TILE@dd` and `TILE@pj` must together
   name one enumerated row. The PV plan's spelling is NOT the dd plan's — for hd256 w4x1 it is `w4x1/f1x32`
   (32 = head_dim/atom_n, the k suffix elided at 1). The sweep's first pins spelled pj like dd
   (`f1x2/k16` / `f1x4/k8`), matched nothing, and `_narrow_flash_forms` kept the full fork → greedy's w2x1 row
   benched under the pin's name, loudly flagged "unreproducible". The flags were read as a form refusal; they
   were a no-match. Proposal stands from the 4090 report: a pin matching no offered row should FAIL rather
   than fall back with a flag, and an `emmy eval offer` view would have shown the w4x1 rows immediately.

Torch SDPA is 30.7 µs; the best recorded entries now sit at 0.86–0.88× — the residual is the known hd256
register-pressure codegen gap (255 regs), not search. A follow-up exhaustion pass over the remaining flash
knob space confirmed the frontier: the **nt=4 streaming block** (32 keys/step) is a consistent zero-spread
~2.3% win over nt=2 on the std lane (static 34.8, dynM 35.5 — recorded as parity alternates beside the nt=2
entries per the 3% gate), while every other direction loses or breaks — nt=8 fattens to 60 µs, the f2
query-tile ILP forms spill (127–135 µs), the WSPEC producer bands **nvcc-fail** on this kernel (compile
error, not a degrade), and FAST_EXP / d3 / d4 / p2 are neutral. Attention is the file's one sub-parity
kernel and it is codegen-bound; every matmul entry is at or past cuBLAS and the norms sit at 0.97–1.14×.

## Finding 2 — the K/V transport preference is slab-size- and lane-dependent, not "cp beats tma"

**Corrected alongside finding 1.** On the fat `w2x1` slabs (64 KB/step) cp.async beat TMA 16% (82.0 vs 97.5).
At the winning `w4x1` geometry (16 KB slabs) it flips per lane, reproduced 3× in both directions and identical
on the static and masked forms: the **std (f32-acc PV) lane prefers `d2/tma/ring`** (35.6 vs cp 39.1 static;
36.4 vs 39.3 dynM), the **fm (f16-acc PV) lane prefers `d2/cp/ring`** (36.5 vs tma 38.8 static; 36.1 vs 38.1
dynM). Deeper staging (`d3`, `d4`) and `p2` are within noise; `FAST_EXP` adds nothing. The recorded entries
carry each lane's own transport.

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
